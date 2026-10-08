// SPDX-License-Identifier: Apache-2.0
//! The measurement. Value gate first, then one discarded warmup per arm,
//! then N interleaved repeats. N is 5, or 3 when a repeat exceeds 30 s.
//! A 1-minute load above the host's threshold (`load_void_above`) voids the attempt and it is run again; it is
//! not published hot.

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use omega_backend_stabrank::StabRankBackend;
use omega_core::circuit::CircuitType;
use omega_core::executor::{Backend, Observable};
use omega_core::params::ParameterBinding;
use omega_emu_compare::{
    Direction, GateReference, GitRev, LoadRecord, Quantity, Row, RowBody, RowWriter, ValueGate,
    VoidRow, GATE_REL_F64,
};
use serde_json::json;

use omega_emu_compare::{wait_quiet_external, CpuCensus, QuietProbe};

use crate::assemble::{arm, capability_row, map, speed_arm, speed_row, witness, workload};
use crate::gate::{detector_noise_refusal, discriminating, exact_gate, rel_gap, GateObs};
use crate::stim_client::{min_of, StimClient};
use crate::workload::{
    append_t_on_zero, floor_pauli_expectation, floor_pauli_sample, load_artifact,
    negate_stim_pauli, observable_to_stim_pauli, pauli_expectation, pauli_sample, qec_d5,
    sha256_file, statevector_expectation, stim_pauli_to_observable, time_pauli_expectation,
    time_pauli_sample, to_stim, with_terminal_measures, without_measures, Artifact, QecMemory,
    SEED, SHOTS,
};

struct Run {
    stim: StimClient,
    git: GitRev,
    recorded: String,
    writer: RowWriter,
    ours_floor_e: f64,
    stim_floor_e: f64,
    ours_floor_s: f64,
    stim_floor_s: f64,
    prev_finished: bool,
    prev_heavy: bool,
    last_cooldown: u32,
    last_before: f64,
    /// The pre-row quiet census, and the census running over the row itself.
    probe: Option<QuietProbe>,
    census: Option<CpuCensus>,
    /// §4.4's load threshold for this host (`omega_emu_compare::load_void_above`).
    void_above: f64,
    /// The launcher's `--host-bytes` declaration (`omega_emu_compare::hostgate_cap`).
    hostgate_cap: String,
}

/// A row the cap killed has no process left to write it. The wrapper calls
/// this with the last `BEGIN` line, and the sentence is the one §4.4 names.
pub fn append_void(out: &Path, row_id: &str, reason: &str) -> Result<(), String> {
    if !out.exists() {
        return Err(format!(
            "{} does not exist; nothing to append a void to",
            out.display()
        ));
    }
    let recorded = today()?;
    let void = VoidRow {
        row_id: row_id.to_string(),
        lane: omega_emu_compare::Lane::Stabilizer,
        phase: "E2".to_string(),
        recorded,
        reason: reason.to_string(),
        context: json!({
            "source": "hostgate --watch killed the process tree",
        }),
    };
    let mut writer = RowWriter::append(out).map_err(|e| e.to_string())?;
    writer.write_void(&void).map_err(|e| e.to_string())
}

pub fn measure(out: &Path) -> Result<(), String> {
    if out.exists() {
        return Err(format!(
            "{} exists; refusing to append over a previous lane file",
            out.display()
        ));
    }
    let git = git_snap()?;
    let recorded = today()?;
    let cores = omega_emu_compare::host_cores()?;
    let void_above = omega_emu_compare::load_void_above(cores);
    let hostgate_cap = omega_emu_compare::hostgate_cap()?;
    eprintln!(
        "measuring at {} rev {}; load void above {void_above} ({cores} cores)",
        recorded, git.worktree
    );
    let mut stim = StimClient::launch()?;
    let ours_floor_e = min_of(5, floor_pauli_expectation)?;
    let stim_floor_e = min_of(5, || stim.floor_expectation())?;
    let ours_floor_s = min_of(5, floor_pauli_sample)?;
    let stim_floor_s = min_of(5, || stim.floor_sample())?;
    eprintln!(
        "floors  expectation ours {ours_floor_e:.6e}s stim {stim_floor_e:.6e}s  sampling ours {ours_floor_s:.6e}s stim {stim_floor_s:.6e}s"
    );
    let writer = RowWriter::append(out).map_err(|e| e.to_string())?;
    let mut run = Run {
        stim,
        git,
        recorded,
        writer,
        ours_floor_e,
        stim_floor_e,
        ours_floor_s,
        stim_floor_s,
        prev_finished: false,
        prev_heavy: false,
        last_cooldown: 0,
        last_before: 0.0,
        probe: None,
        census: None,
        void_above,
        hostgate_cap,
    };
    let c24 = timed_load("clifford-24q-d100")?;
    let c100 = timed_load("clifford-100q-d100")?;
    let qec = qec_d5()?;
    // 100q × 10^4 shots is the long row. The others are written first so a
    // cap kill on that shape still leaves the rows that finished.
    run.brickwall(&c24, true)?;
    run.qec(&qec)?;
    run.clifford_t(&c24.art)?;
    run.d25()?;
    run.brickwall(&c100, false)?;
    eprintln!("wrote {}", out.display());
    Ok(())
}

struct TimedLoad {
    art: Artifact,
    lower_s: f64,
}

fn timed_load(id: &str) -> Result<TimedLoad, String> {
    let start = std::time::Instant::now();
    let art = load_artifact(id)?;
    Ok(TimedLoad {
        art,
        lower_s: start.elapsed().as_secs_f64(),
    })
}

/// One shot and one expectation of each shape, printed and not published.
/// Debug builds are not a measurement of this lane; run the release binary.
pub fn probe() -> Result<(), String> {
    let mut stim = StimClient::launch()?;
    let c24 = load_artifact("clifford-24q-d100")?;
    let c100 = load_artifact("clifford-100q-d100")?;
    let qec = qec_d5()?;
    probe_expectation(&mut stim, "c24", &c24.circuit, c24.qubits, "Z0")?;
    probe_sample(&mut stim, "c24s", &c24.circuit, false, 1)?;
    probe_sample(&mut stim, "c24s10", &c24.circuit, false, 10)?;
    probe_expectation(&mut stim, "c100", &c100.circuit, c100.qubits, "Z0")?;
    probe_sample(&mut stim, "c100s", &c100.circuit, false, 1)?;
    let bare = without_measures(&qec.circuit);
    probe_expectation(&mut stim, "qec", &bare, qec.qubits, &qec.observable)?;
    probe_sample(&mut stim, "qecs", &qec.circuit, true, 1)?;
    let start = std::time::Instant::now();
    let v = statevector_expectation(&c24.circuit, "Z0")?;
    eprintln!(
        "statevector 24q Z0 = {v} in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn probe_expectation(
    stim: &mut StimClient,
    name: &str,
    circuit: &omega_core::circuit::CircuitIR,
    n: u32,
    obs: &str,
) -> Result<(), String> {
    let text = to_stim(circuit)?;
    let pauli = observable_to_stim_pauli(obs, n)?;
    stim.load(name, &text)?;
    let start = std::time::Instant::now();
    let ours = pauli_expectation(circuit, obs)?;
    let ours_s = start.elapsed().as_secs_f64();
    let (theirs, stim_s) = stim.expectation(name, &pauli)?;
    eprintln!("probe {name} expectation ours {ours} in {ours_s:.6}s stim {theirs} in {stim_s:.6}s");
    // The gate's other two observables, printed so the degenerate-gate fix
    // (§4.3c) can be read off the probe without publishing a row.
    let (image, weight) = stim.heisenberg_image(name, &pauli)?;
    let image_obs = stim_pauli_to_observable(&image)?;
    let negated = negate_stim_pauli(&image)?;
    let negated_obs = stim_pauli_to_observable(&negated)?;
    let (img_them, _) = stim.expectation(name, &image)?;
    let (neg_them, _) = stim.expectation(name, &negated)?;
    let img_ours = pauli_expectation(circuit, &image_obs)?;
    let neg_ours = pauli_expectation(circuit, &negated_obs)?;
    eprintln!(
        "probe {name} gate vector: {obs} ours={ours} stim={theirs}; U P U+ (weight {weight}) ours={img_ours} stim={img_them} predicted=1; -U P U+ ours={neg_ours} stim={neg_them} predicted=-1"
    );
    Ok(())
}

fn probe_sample(
    stim: &mut StimClient,
    name: &str,
    circuit: &omega_core::circuit::CircuitIR,
    collapse: bool,
    shots: u32,
) -> Result<(), String> {
    let text = if collapse {
        to_stim(circuit)?
    } else {
        to_stim(&with_terminal_measures(circuit))?
    };
    stim.load(name, &text)?;
    let compile_s = stim.compile(name, SEED)?;
    let start = std::time::Instant::now();
    let (observed, width) = pauli_sample(circuit, shots, SEED, collapse)?;
    let ours_s = start.elapsed().as_secs_f64();
    let sampled = stim.sample(name, u64::from(shots))?;
    eprintln!(
        "probe {name} sample shots {shots} ours {observed} width {width} in {ours_s:.6}s stim {} meas {} in {:.6}s (compile {compile_s:.6}s)",
        sampled.shots_observed, sampled.num_measurements, sampled.seconds
    );
    Ok(())
}

impl Run {
    fn brickwall(&mut self, loaded: &TimedLoad, oracle: bool) -> Result<(), String> {
        let art = &loaded.art;
        let obs = "Z0";
        let pauli = observable_to_stim_pauli(obs, art.qubits)?;
        let exp_name = format!("{}-exp", art.id);
        let samp_name = format!("{}-samp", art.id);
        let exp_text = to_stim(&art.circuit)?;
        let samp_text = to_stim(&with_terminal_measures(&art.circuit))?;
        let exp_loaded = self.stim.load(&exp_name, &exp_text)?;
        let samp_loaded = self.stim.load(&samp_name, &samp_text)?;
        if exp_loaded.num_qubits != art.qubits || samp_loaded.num_measurements != art.qubits {
            return Err(format!(
                "{}: stim sees {} qubits and {} terminal measurements",
                art.id, exp_loaded.num_qubits, samp_loaded.num_measurements
            ));
        }
        let compile_s = self.stim.compile(&samp_name, SEED)?;
        let heavy = art.qubits >= 100;
        self.expectation_row(
            &format!("stab-{}-z0-expectation", art.id),
            &exp_name,
            &art.circuit,
            obs,
            &pauli,
            art,
            loaded.lower_s,
            exp_loaded.parse_s,
            self.ours_floor_e,
            self.stim_floor_e,
            oracle,
            heavy,
            &arm_vs_arm_note(art.qubits, "Z0"),
            vec![
                "timed region is PauliBackend::expectation against TableauSimulator.do + peek_observable_expectation. QASM lowering and stim.Circuit parse are fixed_cost, outside the repeats.".to_string(),
                "observable is Z0, the single Z §3 pins.".to_string(),
            ],
        )?;
        self.sample_row(
            &format!("stab-{}-z0-sample-10k", art.id),
            &samp_name,
            &art.circuit,
            false,
            art.qubits,
            obs,
            &pauli,
            art,
            loaded.lower_s,
            samp_loaded.parse_s + compile_s,
            oracle,
            heavy,
            &arm_vs_arm_note(art.qubits, "Z0"),
            vec![
                "timed region is PauliBackend::execute(shots) against CompiledMeasurementSampler.sample. compile_sampler (skip_reference_sample=false, the default, so the bits are measurement results and not flips) is fixed_cost.".to_string(),
                "our sampler has no compile step; the per-shot tableau rebuild is inside execute, which is the call a user makes, and it stays in the timed region.".to_string(),
                "Stim has no implicit final measure, so its circuit gains a terminal M on every qubit. Ours measures every qubit at the end because the brickwall contains none. The value gate is peek of Z0 on the unmeasured circuit, not an estimate from the shots.".to_string(),
                format!("bit_packed=true, seed={SEED}, shots={SHOTS}."),
            ],
        )?;
        Ok(())
    }

    fn qec(&mut self, qec: &QecMemory) -> Result<(), String> {
        let bare = without_measures(&qec.circuit);
        let pauli = observable_to_stim_pauli(&qec.observable, qec.qubits)?;
        let exp_text = to_stim(&bare)?;
        let samp_text = to_stim(&qec.circuit)?;
        let exp_loaded = self.stim.load("qec-exp", &exp_text)?;
        let samp_loaded = self.stim.load("qec-samp", &samp_text)?;
        if samp_loaded.num_measurements != qec.measures {
            return Err(format!(
                "d=5 stim measurements {} != {}",
                samp_loaded.num_measurements, qec.measures
            ));
        }
        let compile_s = self.stim.compile("qec-samp", SEED)?;
        let note = "49 qubits, past the ≤26q dense-oracle line (§4.3). The gate is exact integer equality of the two tableaus.";
        let art = Artifact {
            id: "surface-d5-syndrome".to_string(),
            file: "crates/aria-qec/src/ecc/codes.rs".to_string(),
            sha256: sha256_source_pin()?,
            generator: "SurfaceCode::new(5).syndrome_circuit(), lowered by aria_qec::ecc::to_omega_core_ir. This is the rotated surface code whose checks tools/qec_cross_check decodes; that tool has no circuit file, so the row pins the generator.".to_string(),
            qubits: qec.qubits,
            depth: 1,
            circuit: qec.circuit.clone(),
        };
        // The syndrome circuit is not a committed QASM file. The sha256 is
        // of codes.rs, which is the generator, and the row says so.
        self.expectation_row(
            "stab-surface-d5-memory-expectation",
            "qec-exp",
            &bare,
            &qec.observable,
            &pauli,
            &art,
            0.0,
            exp_loaded.parse_s,
            self.ours_floor_e,
            self.stim_floor_e,
            false,
            false,
            note,
            vec![
                "one round of rotated-surface syndrome extraction at d=5, not Stim's generated multi-round memory. Logical observable Z0Z5Z10Z15Z20, the leftmost column.".to_string(),
                "expectation is on the circuit with measurements removed, on both arms. Stim's do() would collapse them and our expectation() would defer them, and those are not the same computation.".to_string(),
            ],
        )?;
        self.sample_row(
            "stab-surface-d5-memory-sample-10k",
            "qec-samp",
            &qec.circuit,
            true,
            qec.measures,
            &qec.observable,
            &pauli,
            &art,
            0.0,
            samp_loaded.parse_s + compile_s,
            false,
            false,
            note,
            vec![
                "sampling uses Collapse, keyed on the syndrome creg, which is the path aria-qec already runs for this circuit. Stim's M record is the same 24 ancilla measures.".to_string(),
                "the value gate is still peek of the logical Z on the unmeasured unitary. The sampling witness is the shot count.".to_string(),
            ],
        )?;
        Ok(())
    }

    fn clifford_t(&mut self, art: &Artifact) -> Result<(), String> {
        eprintln!("BEGIN stabrank-clifford-t-vs-stim-refusal");
        self.begin(false)?;
        let with_t = append_t_on_zero(&art.circuit);
        let obs = Observable::parse("X0").map_err(|e| e.to_string())?;
        let backend = StabRankBackend::new();
        let (value, cert) = backend
            .expectation_with_certificate(&with_t, &ParameterBinding::new(), &obs)
            .map_err(|e| e.to_string())?;
        if !cert.is_exact()
            || !cert.is_informative()
            || cert.final_chi != 2
            || cert.expectation_error_bound != 0.0
        {
            return Err(format!(
                "stabrank certificate is not an exact χ=2 run: exact={} informative={} chi={} bound={}",
                cert.is_exact(),
                cert.is_informative(),
                cert.final_chi,
                cert.expectation_error_bound
            ));
        }
        let no_t = pauli_expectation(&art.circuit, "X0")?;
        let stim_text = format!("{}\nT 0", to_stim(&art.circuit)?);
        let refusal = self.stim.refuse_t(&stim_text)?;
        let evidence = format!(
            "stabrank expectation_with_certificate on the committed 24q depth-100 brickwall plus one T on qubit 0, observable X0: value={value:.16}, state_dropped_mass={}, expectation_error_bound={}, final_chi={}, peak_chi={}, is_exact={}, is_informative={}. PauliBackend on the same brickwall without the T gives X0={no_t}. Stim 1.16.0 raises on the same text with a T appended.",
            cert.state_dropped_mass,
            cert.expectation_error_bound,
            cert.final_chi,
            cert.peak_chi,
            cert.is_exact(),
            cert.is_informative()
        );
        let after = load1()?;
        let row = capability_row(
            "stabrank-clifford-t-vs-stim-refusal",
            &self.recorded,
            self.git.clone(),
            workload(
                &art.id,
                &art.file,
                &art.sha256,
                &format!("{}, plus GateKind::T on qubit 0 (appended in memory; the file bytes are unchanged)", art.generator),
                art.qubits,
                art.depth,
                None,
                Quantity::Expectation,
                Some("X0"),
            ),
            Direction::OursOnly,
            "stabrank executes one T on the Clifford brickwall and returns a derived certificate (exact at χ=2, bound 0). Stim cannot execute T at all.",
            &evidence,
            arm(
                "omega-backend-stabrank",
                map(&[("omega-backend-stabrank", "0.1.0")]),
                1,
                "none",
                map(&[
                    ("truncation", "none"),
                    ("observable", "X0"),
                    ("T", "qubit 0, appended after the brickwall"),
                ]),
                vec![
                    witness(
                        "certificate",
                        &format!(
                            "final_chi={} == 2, expectation_error_bound={} == 0, is_informative={}",
                            cert.final_chi, cert.expectation_error_bound, cert.is_informative()
                        ),
                    ),
                    witness("backend_name", backend.name()),
                ],
            ),
            self.stim_arm(
                vec![witness(
                    "t_refusal",
                    &format!("ValueError, asserted refused: {refusal}"),
                )],
                map(&[("circuit", "committed brickwall plus T 0")]),
            ),
            &refusal,
            vec![
                "X0 is the observable because T is a Z rotation and leaves Z0 unchanged; X0 is a Pauli it can move. final_chi == 2 is the witness that the T split happened.".to_string(),
            ],
            false,
            "not a precision comparison: Stim has no value on this circuit (§4.5)",
        )?;
        // Load was taken after the work on purpose for a capability row that
        // is not a timed race; before is whatever begin() waited down to.
        let _ = after;
        self.finish_capability(row)
    }

    fn d25(&mut self) -> Result<(), String> {
        eprintln!("BEGIN stim-d25-rotated-memory-vs-pauli-ceiling");
        self.begin(self.prev_heavy)?;
        let text = self.stim.d25_text()?;
        let generated = text["text"].as_str().ok_or("d25 text missing")?;
        let path = crate::workload::repo_root()
            .join("tools/emu_compare/stim/surface_d25_rotated_memory_z.stim");
        let committed =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if generated != committed {
            return Err(
                "stim 1.16.0 regenerated a d=25 circuit that does not match the committed bytes"
                    .into(),
            );
        }
        let sha = sha256_file(&path)?;
        let qubits = text["qubits"].as_u64().ok_or("d25 qubits")? as u32;
        let names = string_list(&text["instructions"])?;
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let refusal = detector_noise_refusal(&refs)?;
        if !refusal.contains("DEPOLARIZE1") || !refusal.contains("DETECTOR") {
            return Err(format!("refusal does not name the workload: {refusal}"));
        }
        let sample = self.stim.d25_sample(u64::from(SHOTS))?;
        let shots = sample["shots_observed"].as_u64().ok_or("d25 shots")?;
        let obs_shots = sample["obs_shots_observed"]
            .as_u64()
            .ok_or("d25 obs shots")?;
        let detectors = sample["detectors"].as_u64().ok_or("d25 detectors")?;
        let wide = omega_core::circuit::CircuitIR::new(qubits, CircuitType::GateBased);
        let wide_z = pauli_expectation(&wide, "Z0")?;
        exact_gate(wide_z, 1.0).map_err(|d| d.reason)?;
        let evidence = format!(
            "Stim 1.16.0 Circuit.generated(surface_code:rotated_memory_z, distance=25, rounds=25, after_clifford_depolarization=0.001, before_round_data_depolarization=0.001, before_measure_flip_probability=0.001, after_reset_flip_probability=0.001): {qubits} qubits, {} measurements, {detectors} detectors, {} observables. compile_detector_sampler seed=0 then sample({SHOTS}, separate_observables=true, bit_packed=true) returned {shots} detector shots and {obs_shots} observable shots in {:.6}s (compile {:.6}s). PauliBackend::expectation of Z0 on an empty register of the same width is {wide_z}, so the width is in range; the refusal is the instruction set, not an allocation failure.",
            sample["measurements"],
            sample["observables"],
            sample["sample_s"].as_f64().unwrap_or(f64::NAN),
            sample["compile_s"].as_f64().unwrap_or(f64::NAN),
        );
        let row = capability_row(
            "stim-d25-rotated-memory-vs-pauli-ceiling",
            &self.recorded,
            self.git.clone(),
            workload(
                "stim-surface-d25-rotated-memory-z",
                "tools/emu_compare/stim/surface_d25_rotated_memory_z.stim",
                &sha,
                "stim.Circuit.generated('surface_code:rotated_memory_z', distance=25, rounds=25, after_clifford_depolarization=0.001, before_round_data_depolarization=0.001, before_measure_flip_probability=0.001, after_reset_flip_probability=0.001)",
                qubits,
                25,
                Some(SHOTS),
                Quantity::Sampling,
                None,
            ),
            Direction::CompetitorOnly,
            "Stim samples a d=25 rotated memory, rounds=25, under its circuit-level noise, and returns detector and logical-observable records. The pauli backend has no detector error model and refuses the noise and detector instructions. The tableau can hold an empty register of this width; that is not this workload.",
            &evidence,
            arm(
                "omega-backend-pauli",
                map(&[("omega-backend-pauli", "0.1.0")]),
                1,
                "none",
                map(&[("refused", refusal.as_str())]),
                vec![
                    witness(
                        "wide_register",
                        &format!("{qubits}-qubit empty register <Z0>={wide_z} == 1"),
                    ),
                    witness("instruction_refusal", &refusal),
                ],
            ),
            self.stim_arm(
                vec![
                    witness(
                        "detector_shots",
                        &format!("requested {SHOTS} == observed {shots}; observable shots {obs_shots}"),
                    ),
                    witness(
                        "detectors",
                        &format!("{detectors} detectors, asserted > 0"),
                    ),
                ],
                map(&[
                    ("code_task", "surface_code:rotated_memory_z"),
                    ("distance", "25"),
                    ("rounds", "25"),
                    ("after_clifford_depolarization", "0.001"),
                    ("before_round_data_depolarization", "0.001"),
                    ("before_measure_flip_probability", "0.001"),
                    ("after_reset_flip_probability", "0.001"),
                    ("bit_packed", "true"),
                    ("seed", "0"),
                ]),
            ),
            &refusal,
            vec![
                "noise probabilities are the ones in Stim's generated() docstring example, every knob set. Zeroing one would sample an easier circuit (§8).".to_string(),
                "the noiseless 60000-CX skeleton was not timed. It is not this workload, and timing it would be a different row.".to_string(),
            ],
            false,
            "not a precision comparison: the pauli backend cannot express the workload (§4.5)",
        )?;
        if detectors == 0 {
            return Err("d=25 circuit reported 0 detectors".into());
        }
        self.finish_capability(row)
    }

    #[allow(clippy::too_many_arguments)]
    fn expectation_row(
        &mut self,
        row_id: &str,
        stim_name: &str,
        circuit: &omega_core::circuit::CircuitIR,
        obs: &str,
        pauli: &str,
        art: &Artifact,
        ours_fixed: f64,
        stim_fixed: f64,
        ours_floor: f64,
        stim_floor: f64,
        oracle: bool,
        heavy: bool,
        arm_note: &str,
        mut notes: Vec<String>,
    ) -> Result<(), String> {
        let gate_obs = self.gate_vector(stim_name, circuit, obs, pauli, oracle)?;
        let gate = match value_gate(&gate_obs, arm_note) {
            Ok(g) => g,
            Err(reason) => return self.void(row_id, reason),
        };
        let stim_v = gate_obs[0].stim;
        let witnessed = gate_witness(&gate_obs);
        notes.push(format!(
            "the value gate is three observables, not one: {witnessed}. The second is the \
             Heisenberg image U P U-dagger, whose expectation on U|0> is <0|P|0> = +1 by \
             construction, and the third is its negation. The gate therefore pins a vector \
             no constant passes (§4.3c). All three peeks are outside the timed region; the \
             timings below are the row's own observable only."
        ));
        if oracle {
            let os: Vec<String> = gate_obs
                .iter()
                .map(|g| format!("{}={}", g.label, g.oracle.unwrap_or(f64::NAN)))
                .collect();
            notes.push(format!(
                "dense oracle omega-backend-statevector f64: {}, §4.3, outside the timed region.",
                os.join(" ")
            ));
        }
        self.timed_speed(
            row_id,
            heavy,
            notes,
            |run| {
                let warm_ours = time_pauli_expectation(circuit, obs)?;
                let (warm_val, warm_stim) = run.stim.expectation(stim_name, pauli)?;
                if warm_val != stim_v {
                    return Err(format!("warmup peek {warm_val} != gated {stim_v}"));
                }
                let mut n = if warm_ours > 30.0 || warm_stim > 30.0 {
                    3
                } else {
                    5
                };
                eprintln!("{row_id} warmup ours {warm_ours:.6}s stim {warm_stim:.6}s N={n}");
                let (ours_t, stim_t) = interleaved(
                    &mut n,
                    || time_pauli_expectation(circuit, obs),
                    || {
                        let (v, s) = run.stim.expectation(stim_name, pauli)?;
                        if v != stim_v {
                            return Err(format!("repeat peek {v} != gated {stim_v}"));
                        }
                        Ok(s)
                    },
                )?;
                let ours_arm = speed_arm(
                    arm(
                        "omega-backend-pauli",
                        map(&[("omega-backend-pauli", "0.1.0")]),
                        1,
                        "none",
                        map(&[
                            ("api", "PauliBackend::expectation"),
                            ("threads", "1"),
                            ("blas", "none"),
                        ]),
                        vec![
                            witness("backend_name", "pauli"),
                            witness("peek_observable_expectation", &witnessed),
                        ],
                    ),
                    &ours_t,
                    ours_fixed,
                    ours_floor,
                )?;
                let stim_arm = speed_arm(
                    run.stim_arm(
                        vec![witness("peek_observable_expectation", &witnessed)],
                        map(&[
                            ("api", "TableauSimulator.do + peek_observable_expectation"),
                            ("threads", "1"),
                            ("blas", "none"),
                        ]),
                    ),
                    &stim_t,
                    stim_fixed,
                    stim_floor,
                )?;
                Ok((ours_arm, stim_arm, gate.clone()))
            },
            |art_wl| {
                workload(
                    &art_wl.id,
                    &art_wl.file,
                    &art_wl.sha256,
                    &art_wl.generator,
                    art_wl.qubits,
                    art_wl.depth,
                    None,
                    Quantity::Expectation,
                    Some(obs),
                )
            },
            art,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_row(
        &mut self,
        row_id: &str,
        stim_name: &str,
        circuit: &omega_core::circuit::CircuitIR,
        collapse: bool,
        expected_bits: u32,
        obs: &str,
        pauli: &str,
        art: &Artifact,
        ours_fixed: f64,
        stim_fixed: f64,
        oracle: bool,
        heavy: bool,
        arm_note: &str,
        notes: Vec<String>,
    ) -> Result<(), String> {
        // The value gate reads the unmeasured unitary even on a sampling row.
        let bare = without_measures(circuit);
        let exp_name = format!("{stim_name}-gate");
        let bare_text = to_stim(&bare)?;
        self.stim.load(&exp_name, &bare_text)?;
        let gate_obs = self.gate_vector(&exp_name, &bare, obs, pauli, oracle)?;
        let gate = match value_gate(&gate_obs, arm_note) {
            Ok(g) => g,
            Err(reason) => return self.void(row_id, reason),
        };
        let witnessed = gate_witness(&gate_obs);
        let mut notes = notes;
        notes.push(format!(
            "the value gate is three observables on the unmeasured unitary, not one: \
             {witnessed}. The second is the Heisenberg image U P U-dagger (+1 by \
             construction) and the third its negation, so the gate pins a vector no \
             constant passes (§4.3c)."
        ));
        self.timed_speed(
            row_id,
            heavy,
            notes,
            |run| {
                let warm_ours = time_pauli_sample(circuit, SHOTS, SEED, collapse)?;
                let warm_stim = run.stim.sample(stim_name, u64::from(SHOTS))?;
                if warm_stim.num_measurements != expected_bits {
                    return Err(format!(
                        "stim measurements {} != {expected_bits}",
                        warm_stim.num_measurements
                    ));
                }
                let mut n = if warm_ours > 30.0 || warm_stim.seconds > 30.0 {
                    3
                } else {
                    5
                };
                eprintln!(
                    "{row_id} warmup ours {warm_ours:.6}s stim {:.6}s N={n}",
                    warm_stim.seconds
                );
                let (ours_t, stim_t) = interleaved(
                    &mut n,
                    || time_pauli_sample(circuit, SHOTS, SEED, collapse),
                    || {
                        let s = run.stim.sample(stim_name, u64::from(SHOTS))?;
                        if s.num_measurements != expected_bits {
                            return Err(format!(
                                "stim measurements {} != {expected_bits}",
                                s.num_measurements
                            ));
                        }
                        Ok(s.seconds)
                    },
                )?;
                let mode = if collapse { "Collapse" } else { "Skip" };
                let ours_arm = speed_arm(
                    arm(
                        "omega-backend-pauli",
                        map(&[("omega-backend-pauli", "0.1.0")]),
                        1,
                        "none",
                        map(&[
                            ("api", "PauliBackend::execute"),
                            ("mid_circuit_mode", mode),
                            ("shots", &SHOTS.to_string()),
                            ("seed", &SEED.to_string()),
                            ("threads", "1"),
                            ("blas", "none"),
                        ]),
                        vec![
                            witness("backend_name", "pauli"),
                            witness(
                                "shot_count",
                                &format!("requested {SHOTS} == observed {SHOTS}"),
                            ),
                            witness(
                                "peek_observable_expectation",
                                &witnessed,
                            ),
                        ],
                    ),
                    &ours_t,
                    ours_fixed,
                    run.ours_floor_s,
                )?;
                let stim_arm = speed_arm(
                    run.stim_arm(
                        vec![
                            witness(
                                "shot_count",
                                &format!(
                                    "requested {SHOTS} == observed {}; num_measurements={expected_bits}",
                                    warm_stim.shots_observed
                                ),
                            ),
                            witness(
                                "peek_observable_expectation",
                                &witnessed,
                            ),
                        ],
                        map(&[
                            ("api", "Circuit.compile_sampler + CompiledMeasurementSampler.sample"),
                            ("skip_reference_sample", "false"),
                            ("bit_packed", "true"),
                            ("seed", &SEED.to_string()),
                            ("shots", &SHOTS.to_string()),
                            ("threads", "1"),
                            ("blas", "none"),
                        ]),
                    ),
                    &stim_t,
                    stim_fixed,
                    run.stim_floor_s,
                )?;
                Ok((ours_arm, stim_arm, gate.clone()))
            },
            |_| {
                workload(
                    &art.id,
                    &art.file,
                    &art.sha256,
                    &art.generator,
                    art.qubits,
                    art.depth,
                    Some(SHOTS),
                    Quantity::Sampling,
                    Some(obs),
                )
            },
            art,
        )
    }

    /// The gate's observables, both arms, and the oracle where affordable.
    ///
    /// Three of them, not one. The row's own observable is first and is the
    /// one the timed region measures — the timings stay a single-observable
    /// comparison, unchanged. The other two exist only to make the gate
    /// fail-able (§4.3c): `U P U-dagger` is `+1` by construction and its
    /// negation is `-1`, so the gate pins a vector no constant passes.
    ///
    /// The image is derived on the **stim** side. A discriminating observable
    /// handed to us by the backend under test would be circular; one derived
    /// by the competitor and then predicted analytically is not.
    fn gate_vector(
        &mut self,
        stim_name: &str,
        circuit: &omega_core::circuit::CircuitIR,
        obs: &str,
        pauli: &str,
        oracle: bool,
    ) -> Result<Vec<GateObs>, String> {
        let (image, weight) = self.stim.heisenberg_image(stim_name, pauli)?;
        if weight == 0 {
            return Err(format!(
                "{stim_name}: U {pauli} U-dagger has weight 0, so the circuit does not move                  the observable and the image adds nothing to the gate"
            ));
        }
        let negated = negate_stim_pauli(&image)?;
        let specs: [(String, String, Option<i8>); 3] = [
            (obs.to_string(), pauli.to_string(), None),
            (stim_pauli_to_observable(&image)?, image.clone(), Some(1)),
            (
                stim_pauli_to_observable(&negated)?,
                negated.clone(),
                Some(-1),
            ),
        ];
        let labels = [obs.to_string(), "U P U+".to_string(), "-U P U+".to_string()];
        let mut out = Vec::new();
        for (i, (o, p, predicted)) in specs.into_iter().enumerate() {
            let ours = pauli_expectation(circuit, &o)?;
            let (stim_v, _) = self.stim.expectation(stim_name, &p)?;
            let oracle_v = if oracle {
                Some(statevector_expectation(circuit, &o)?)
            } else {
                None
            };
            out.push(GateObs {
                label: labels[i].clone(),
                obs: o,
                pauli: p,
                predicted,
                ours,
                stim: stim_v,
                oracle: oracle_v,
            });
        }
        Ok(out)
    }

    fn timed_speed<F, W>(
        &mut self,
        row_id: &str,
        heavy: bool,
        notes: Vec<String>,
        mut time_both: F,
        workload_of: W,
        art: &Artifact,
    ) -> Result<(), String>
    where
        F: FnMut(
            &mut Run,
        ) -> Result<
            (
                omega_emu_compare::SpeedArm,
                omega_emu_compare::SpeedArm,
                ValueGate,
            ),
            String,
        >,
        W: Fn(&Artifact) -> omega_emu_compare::Workload,
    {
        for attempt in 0..2 {
            eprintln!("BEGIN {row_id}");
            let before = self.begin(heavy)?;
            let timed = time_both(self);
            let external = self
                .census
                .take()
                .ok_or("the row's census was never opened")?
                .finish()?;
            let probe = self.probe.take().ok_or("the row had no pre-row census")?;
            let external = external.with_pre_row(&probe);
            let after = load1()?;
            let (ours_arm, stim_arm, gate) = match timed {
                Ok(t) => t,
                // A value disagreement is a published finding. Anything else
                // is a broken harness, and voiding it would hide that.
                Err(e) if e.contains("disagreement finding") => return self.void(row_id, e),
                Err(e) => return Err(e),
            };
            if omega_emu_compare::void_at_load(external.external_cores_during_row, self.void_above)
            {
                eprintln!(
                    "{row_id} attempt {attempt}: {:.2} cores of work outside the lane during \
                     the row > {} (load1 after {after}), discarding",
                    external.external_cores_during_row, self.void_above
                );
                self.sleep(60, "external-CPU void, re-run");
                self.prev_finished = true;
                self.prev_heavy = true;
                continue;
            }
            let load = LoadRecord {
                load1_before: before,
                load1_after: after,
                void_above: self.void_above,
                cooldown_s: Some(self.last_cooldown),
                hostgate_cap: Some(self.hostgate_cap.clone()),
                gpu: None,
                external_cpu: Some(external),
            };
            let row = speed_row(
                row_id,
                &self.recorded,
                self.git.clone(),
                workload_of(art),
                ours_arm,
                stim_arm,
                gate,
                load,
                notes.clone(),
            )?;
            self.writer.write_row(&row).map_err(|e| e.to_string())?;
            announce(&row);
            self.prev_finished = true;
            self.prev_heavy = heavy || row_was_slow(&row);
            return Ok(());
        }
        self.void(
            row_id,
            format!(
                "load1 stayed above {} across two attempts; not published (§4.4)",
                self.void_above
            ),
        )
    }

    fn begin(&mut self, heavy: bool) -> Result<f64, String> {
        self.last_cooldown = 0;
        if self.prev_finished && (heavy || self.prev_heavy) {
            self.sleep(60, "between heavy rows");
            self.last_cooldown = 60;
        }
        // Wait on the quantity the row is judged by (§4.4a). `wait_quiet` used
        // to loop on the 1-minute load average, which on andromeda never
        // clears 2.0 with a desktop running even when the box is 85% idle —
        // the lane blocked for five minutes and then voided every row.
        let probe = wait_quiet_external(self.void_above, &self.own_pids(), load1)?;
        eprintln!(
            "outside the lane: {:.3} cores over {:.1}s (load1 {:.2}, context only)",
            probe.cores, probe.window_s, probe.load1
        );
        self.last_before = probe.load1;
        self.probe = Some(probe);
        self.census = Some(CpuCensus::start(&self.own_pids())?);
        Ok(probe.load1)
    }

    /// The lane's own tree: this process and the stim arm it launched. The
    /// arm's CPU must not read as somebody else's work, or every row with a
    /// busy competitor voids.
    fn own_pids(&self) -> Vec<u32> {
        let mut pids = vec![std::process::id()];
        if let Some(pid) = self.stim.pid() {
            pids.push(pid);
        }
        pids
    }

    fn sleep(&self, secs: u64, why: &str) {
        eprintln!("cooldown {secs}s ({why})");
        std::thread::sleep(Duration::from_secs(secs));
    }

    fn void(&mut self, row_id: &str, reason: String) -> Result<(), String> {
        eprintln!("VOID {row_id}: {reason}");
        let void = VoidRow {
            row_id: row_id.to_string(),
            lane: omega_emu_compare::Lane::Stabilizer,
            phase: "E2".to_string(),
            recorded: self.recorded.clone(),
            reason,
            context: json!({"git": self.git.worktree}),
        };
        self.writer.write_void(&void).map_err(|e| e.to_string())?;
        self.prev_finished = true;
        Ok(())
    }

    fn finish_capability(&mut self, mut row: omega_emu_compare::Row) -> Result<(), String> {
        // A capability row carries no `LoadRecord`, so its quiet-box check is
        // here and must be the same one the speed rows use. It was still
        // `load1`, which voided both capability rows on a box holding 1.2
        // cores of outside work — and neither of them is a timing claim.
        let external = self
            .census
            .take()
            .ok_or("the capability row's census was never opened")?
            .finish()?;
        let after = load1()?;
        // The same predicate the fermionic lane's capability rows use.
        if let Some(reason) =
            omega_emu_compare::external_void_reason(&row.row_id, &external, self.void_above)
        {
            return self.void(
                &row.row_id.clone(),
                format!("{reason} (§4.4a); load1 after {after}"),
            );
        }
        row.notes.push(format!(
            "outside the lane: {:.3} cores over {:.1}s during the row, {} before it, void \
             above {} (§4.4a); load1 before {:.2} after {after:.2}, context only; cooldown \
             {}s. {}",
            external.external_cores_during_row,
            external.window_s,
            external
                .external_cores_before
                .map_or("unmeasured".to_string(), |c| format!("{c:.3} cores")),
            self.void_above,
            self.last_before,
            self.last_cooldown,
            external.method
        ));
        self.writer.write_row(&row).map_err(|e| e.to_string())?;
        announce(&row);
        self.prev_finished = true;
        Ok(())
    }

    fn stim_arm(
        &self,
        mut witnesses: Vec<omega_emu_compare::PathWitness>,
        mut knobs: std::collections::BTreeMap<String, String>,
    ) -> omega_emu_compare::Arm {
        witnesses.insert(
            0,
            witness("stim_version", &format!("{} == 1.16.0", self.stim.version)),
        );
        witnesses.insert(
            1,
            witness(
                "peek_smoke",
                &format!("H|0> <X>={} <Z>={}", self.stim.smoke_x, self.stim.smoke_z),
            ),
        );
        witnesses.insert(2, witness("stim_linkage", &self.stim.linkage));
        knobs
            .entry("threads".to_string())
            .or_insert_with(|| "1".to_string());
        knobs
            .entry("blas".to_string())
            .or_insert_with(|| "none (wheel links libSystem and libc++ only)".to_string());
        arm(
            "stim-1.16.0",
            map(&[
                ("stim", &self.stim.version),
                ("python", &self.stim.python),
                ("numpy", &self.stim.numpy),
            ]),
            1,
            "none",
            knobs,
            witnesses,
        )
    }
}

/// Build the row's value gate from its observable vector.
///
/// Every observable must agree exactly across the arms, every predicted value
/// must be the value that came back, the oracle must agree where there is one,
/// and the resulting vector must discriminate (§4.3c). Any of those failing is
/// a finding — a void row — not a quietly narrowed gate.
fn value_gate(obs: &[GateObs], arm_note: &str) -> Result<ValueGate, String> {
    let mut integers: Vec<i8> = Vec::with_capacity(obs.len());
    for g in obs {
        let integer =
            exact_gate(g.ours, g.stim as f64).map_err(|d| format!("{}: {}", g.label, d.reason))?;
        if let Some(p) = g.predicted {
            if integer != p {
                return Err(format!(
                    "{}: the construction predicts {p} and both arms returned {integer} for                      {} — a conjugation that disagrees with its own theorem voids the row                      rather than publishing the number it produced",
                    g.label, g.pauli
                ));
            }
        }
        integers.push(integer);
    }
    discriminating(&integers)?;

    let ours_values: Vec<serde_json::Value> = obs.iter().map(|g| json!(g.ours)).collect();
    let stim_values: Vec<serde_json::Value> = obs.iter().map(|g| json!(g.stim)).collect();

    let oracles: Option<Vec<f64>> = obs.iter().map(|g| g.oracle).collect();
    let (reference, ours_gap, stim_gap, tol) = match oracles {
        Some(os) => {
            let mut og = 0.0_f64;
            let mut sg = 0.0_f64;
            for (g, o) in obs.iter().zip(&os) {
                let a = rel_gap(g.ours, *o);
                let b = rel_gap(g.stim as f64, *o);
                if a > GATE_REL_F64 || b > GATE_REL_F64 {
                    return Err(format!(
                        "disagreement finding: {} arms at ours={} stim={} but statevector oracle is {o}, gaps {a:.3e}/{b:.3e} over {GATE_REL_F64:.0e} (§4.3); the row is not dropped",
                        g.label, g.ours, g.stim
                    ));
                }
                og = og.max(a);
                sg = sg.max(b);
            }
            (
                GateReference::Oracle {
                    arm: "omega-backend-statevector".to_string(),
                    versions: map(&[("omega-backend-statevector", "0.1.0")]),
                    value: json!(os),
                },
                og,
                sg,
                GATE_REL_F64,
            )
        }
        None => (
            GateReference::ArmVsArm {
                note: arm_note.to_string(),
            },
            0.0,
            0.0,
            0.0,
        ),
    };
    Ok(ValueGate {
        tolerance_rel: tol,
        reference,
        ours_value: json!(ours_values),
        competitor_value: json!(stim_values),
        ours_gap,
        competitor_gap: stim_gap,
        truncation: None,
        passed: true,
    })
}

/// The gate, written for the row's `executed_path` witness.
fn gate_witness(obs: &[GateObs]) -> String {
    obs.iter()
        .map(|g| {
            let predicted = match g.predicted {
                Some(p) => format!(" predicted={p}"),
                None => String::new(),
            };
            format!(
                "{}[{}] ours={} stim={}{predicted}",
                g.label, g.pauli, g.ours, g.stim
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn arm_vs_arm_note(qubits: u32, obs: &str) -> String {
    format!(
        "{qubits} qubits is past the ≤26q line where a dense oracle is affordable (§4.3). The gate is exact integer equality of the two tableaus on {obs}, via peek_observable_expectation."
    )
}

fn interleaved(
    n: &mut usize,
    mut ours: impl FnMut() -> Result<f64, String>,
    mut stim: impl FnMut() -> Result<f64, String>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut a = Vec::new();
    let mut b = Vec::new();
    while a.len() < *n {
        let ta = ours()?;
        let tb = stim()?;
        eprintln!("  repeat {} ours {ta:.6}s stim {tb:.6}s", a.len() + 1);
        a.push(ta);
        b.push(tb);
        if a.len() >= 3 && (a.iter().any(|t| *t > 30.0) || b.iter().any(|t| *t > 30.0)) {
            *n = (*n).min(3);
        }
    }
    Ok((a, b))
}

fn row_was_slow(row: &Row) -> bool {
    let RowBody::Speed(s) = &row.body else {
        return false;
    };
    let slow = |t: &omega_emu_compare::Timing| t.min_s > 5.0;
    s.ours.timing.get().is_some_and(slow) || s.competitor.timing.get().is_some_and(slow)
}

fn announce(row: &Row) {
    match &row.body {
        RowBody::Speed(s) => {
            let o = s.ours.timing.get().expect("present");
            let t = s.competitor.timing.get().expect("present");
            let load = s.load.get().expect("present");
            eprintln!(
                "ROW {}\t{:?}\tratio={:.6}\tours_min={:.6}\tours_med={:.6}\tstim_min={:.6}\tstim_med={:.6}\tload {}→{}\toverhead={}\tfixed_ours={:.6}\tfixed_stim={:.6}",
                row.row_id,
                s.classification,
                s.ratio_competitor_over_ours,
                o.min_s,
                o.median_s,
                t.min_s,
                t.median_s,
                load.load1_before,
                load.load1_after,
                s.overhead_dominated,
                s.ours.fixed_cost_s.get().copied().unwrap_or(f64::NAN),
                s.competitor.fixed_cost_s.get().copied().unwrap_or(f64::NAN),
            );
        }
        RowBody::Capability(c) => {
            eprintln!("CAP {}\t{:?}\t{}", row.row_id, c.direction, c.claim);
        }
    }
}

fn string_list(v: &serde_json::Value) -> Result<Vec<String>, String> {
    v.as_array()
        .ok_or("instructions is not a list")?
        .iter()
        .map(|x| {
            x.as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| "instruction name is not a string".to_string())
        })
        .collect()
}

fn sha256_source_pin() -> Result<String, String> {
    sha256_file(&crate::workload::repo_root().join("crates/aria-qec/src/ecc/codes.rs"))
}

fn git_snap() -> Result<GitRev, String> {
    let compiled = option_env!("OMEGA_EMU_GIT_REV").unwrap_or("").to_string();
    let worktree = git(&["rev-parse", "HEAD"])?;
    let porcelain = git(&["status", "--porcelain"])?;
    let rev = GitRev {
        compiled_from: compiled,
        worktree: worktree.trim().to_string(),
        worktree_dirty: !porcelain.trim().is_empty(),
    };
    if let Some(why) = rev.void_reason() {
        return Err(why);
    }
    Ok(rev)
}

fn git(args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(crate::workload::repo_root())
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn today() -> Result<String, String> {
    let out = Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .map_err(|e| format!("date: {e}"))?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn load1() -> Result<f64, String> {
    let out = Command::new("uptime")
        .output()
        .map_err(|e| format!("uptime: {e}"))?;
    let s = String::from_utf8_lossy(&out.stdout).to_string();
    let rest = s
        .split("load averages:")
        .nth(1)
        .or_else(|| s.split("load average:").nth(1))
        .ok_or_else(|| format!("cannot parse uptime: {s}"))?;
    let n = rest
        .split_whitespace()
        .next()
        .ok_or("uptime has no load")?
        .trim_end_matches(',');
    n.parse().map_err(|e| format!("load {n}: {e}"))
}
