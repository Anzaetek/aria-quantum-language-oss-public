// SPDX-License-Identifier: Apache-2.0
//! Value gates first. The four small shapes are written as capability rows
//! with no times. LUCJ-16 is then one warmup per arm and five interleaved
//! repeats. A 1-minute load above the host's threshold (`load_void_above`) voids the attempt and it is run again.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use omega_emu_compare::{
    wait_quiet_external, CpuCensus, Direction, ExternalCpu, GitRev, LoadRecord, RowWriter, VoidRow,
    GATE_REL_F64,
};
use serde_json::json;

use crate::assemble::{
    self, arm, capability, f64_precision, map, speed, speed_arm, witness, workload,
};
use crate::ffsim_client::{self, Ffsim};
use crate::fqe;
use crate::gate::{
    admit, hubbard_analytic, kitaev_analytic, kitaev_sector_refusal, rel_gap, row_void_at_load,
    sign_flip_detected, spread_flagged,
};
use crate::workload::{self, threads_statevector};

const FFSIM_PIN: &[(&str, &str)] = &[
    ("ffsim", "0.0.84"),
    ("numpy", "2.5.3"),
    ("qiskit", "2.5.2"),
    ("scipy", "1.18.1"),
    ("pyscf", "2.14.0"),
];

struct Ctx {
    ffsim: Ffsim,
    versions: BTreeMap<String, String>,
    blas: String,
    ffsim_threads: u32,
    fqe: String,
    git: GitRev,
    recorded: String,
    writer: RowWriter,
    ours_floor: f64,
    ffsim_floor: f64,
    /// §4.4's load threshold for this host (`omega_emu_compare::load_void_above`).
    void_above: f64,
    /// The launcher's `--host-bytes` declaration (`omega_emu_compare::hostgate_cap`).
    hostgate_cap: String,
}

pub fn measure(out: &Path) -> Result<(), String> {
    if std::env::var("OMEGA_HOSTGATE_MODE").ok().as_deref() != Some("enforce") {
        return Err(
            "OMEGA_HOSTGATE_MODE is not enforce: a row from a naked process is not a row (§4.4)"
                .into(),
        );
    }
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
        "measuring at {recorded} rev {}; load void above {void_above} ({cores} cores)",
        git.worktree
    );
    let mut ffsim = Ffsim::launch()?;
    let versions = pin_ffsim(&ffsim)?;
    let blas = ffsim.version("blas")?;
    let ffsim_threads: u32 = ffsim
        .hello
        .get("cpu_count")
        .and_then(|v| v.as_u64())
        .ok_or("ffsim hello missing cpu_count")? as u32;
    let fqe = fqe::require_result(&fqe::probe())?;
    eprintln!("fqe: {fqe}");
    let _ = workload::floor_sector()?;
    let ours_floor = min_of(5, workload::floor_sector)?;
    let _ = ffsim.floor()?;
    let ffsim_floor = min_of(5, || ffsim.floor())?;
    eprintln!("floors  sector {ours_floor:.6e}s  ffsim {ffsim_floor:.6e}s");
    let writer = RowWriter::append(out).map_err(|e| e.to_string())?;
    let mut ctx = Ctx {
        ffsim,
        versions,
        blas,
        ffsim_threads,
        fqe,
        git,
        recorded,
        writer,
        ours_floor,
        ffsim_floor,
        void_above,
        hostgate_cap,
    };
    ctx.kitaev_small(2)?;
    ctx.kitaev_small(8)?;
    ctx.hubbard()?;
    ctx.h2()?;
    ctx.kitaev64()?;
    ctx.lucj()?;
    eprintln!("wrote {}", out.display());
    Ok(())
}

impl Ctx {
    fn kitaev_small(&mut self, n: u32) -> Result<(), String> {
        let row_id = format!("ferm-kitaev-n{n}");
        eprintln!("BEGIN {row_id}");
        let probe = wait_quiet_external(self.void_above, &self.own_pids(), load1)?;
        let census = CpuCensus::start(&self.own_pids())?;
        let before = probe.load1;
        let got = workload::kitaev_small(n, false)?;
        let ff = self.ffsim.kitaev(n)?;
        let theirs = ffsim_client::number(&ff, "value")?;
        let refused = ffsim_client::text(&ff, "sector_refused")?;
        kitaev_sector_refusal(&refused)?;
        if !got.exact || got.dropped_mass != 0.0 {
            return Err(format!(
                "{row_id}: majoranaprop dropped_mass {} exact {}",
                got.dropped_mass, got.exact
            ));
        }
        if got.final_terms != (n - 1) as usize {
            return Err(format!(
                "{row_id}: final_terms {}, want one monomial per bond",
                got.final_terms
            ));
        }
        let reference = kitaev_analytic(n);
        let (ours_gap, theirs_gap) =
            admit(got.statevector, theirs, reference).map_err(|d| d.reason)?;
        if rel_gap(got.majorana, reference) > GATE_REL_F64 {
            return Err(format!(
                "{row_id}: majoranaprop {} misses {}",
                got.majorana, reference
            ));
        }
        let flipped = workload::kitaev_small(n, true)?.statevector;
        let delta = sign_flip_detected(theirs, flipped)?;
        let after = load1()?;
        let external = census.finish()?.with_pre_row(&probe);
        self.void_if_hot(&row_id, &external)?;
        let (ext_b, ext_d) = (
            external.external_cores_before.unwrap_or(f64::NAN),
            external.external_cores_during_row,
        );
        let sha = workload::file_sha(&format!("examples/fermionic/kitaev_n{n}.qasm"))?;
        let claim = format!(
            "Correctness at {n} modes, not a speed row. Statevector via Jordan-Wigner and \
             majoranaprop both return {reference}, and ffsim's jordan_wigner does too. A \
             wall-clock here measures call overhead, so no time is published in either direction."
        );
        let evidence = format!(
            "statevector {:.16} majoranaprop {:.16} ffsim {:.16} reference {reference} \
             gaps {ours_gap:.3e}/{theirs_gap:.3e} ≤ {GATE_REL_F64:.0e}. \
             majoranaprop seed_basis={} dropped_mass={} final_terms={} exact. \
             Pairing sign flip |Δ|={delta:.3e} against unflipped ffsim. \
             load1 {before} → {after}; outside the lane {ext_b:.2} cores before, {ext_d:.2} during (§4.4a). {}",
            got.statevector,
            got.majorana,
            theirs,
            got.seed_basis,
            got.dropped_mass,
            got.final_terms,
            self.fqe
        );
        let ours = arm(
            "statevector+majoranaprop",
            our_versions(&self.git.worktree),
            threads_statevector(),
            "none",
            map(&[
                ("statevector", "F2 jordan_wigner then expectation"),
                ("majoranaprop", "F3 ladder seed, no truncation"),
                ("sector", "refused: H does not conserve particle number"),
            ]),
            vec![
                witness("majoranaprop_seed_basis", &got.seed_basis),
                witness("majoranaprop_dropped_mass", format!("{}", got.dropped_mass)),
                witness("majoranaprop_exact", "true"),
                witness("majoranaprop_final_terms", format!("{}", got.final_terms)),
                witness(
                    "value_gate",
                    format!("gaps {ours_gap:.3e}/{theirs_gap:.3e}"),
                ),
                witness("sign_flip_detected", format!("{delta:.6e}")),
                witness("ffsim_sector_refused", "does not conserve particle number"),
            ],
        );
        let competitor = self.ffsim_arm(
            vec![
                witness("ffsim_version", self.versions["ffsim"].clone()),
                witness("value_gate", format!("{theirs:.16}")),
                witness("sector_linear_operator_refused", refused),
            ],
            map(&[
                ("path", "jordan_wigner on the alpha register, beta in |0>"),
                ("qasm", "not read; the state is |+>^N"),
            ]),
        );
        let row = capability(
            &row_id,
            &self.recorded,
            self.git.clone(),
            workload(
                &format!("kitaev-n{n}"),
                &format!("examples/fermionic/kitaev_n{n}.qasm"),
                &sha,
                &format!(
                    "examples/fermionic/kitaev_n{n}.qasm + examples/fermionic/kitaev_op_n{n}.txt"
                ),
                None,
                n,
                1,
                &format!("examples/fermionic/kitaev_op_n{n}.txt"),
            ),
            Direction::OursOnly,
            &claim,
            &evidence,
            ours,
            competitor,
            "speed classification refused: at this width any wall-clock is binding-dominated \
             (item 15's 6x6 permanent). ffsim did return the energy; the refusal is of a ratio, \
             not of the value. ffsim's own linear_operator refuses the pairing operator.",
            f64_precision("both values are f64. The published fact is the value gate, not a ratio."),
            vec![
                "binding-dominated capability/correctness row".into(),
                format!("defect not caught: |->^N is also a ground state at energy {reference}, so the other product ground state passes"),
                self.fqe.clone(),
            ],
        )?;
        self.write(row)
    }

    fn hubbard(&mut self) -> Result<(), String> {
        let row_id = "ferm-hubbard2";
        eprintln!("BEGIN {row_id}");
        let probe = wait_quiet_external(self.void_above, &self.own_pids(), load1)?;
        let census = CpuCensus::start(&self.own_pids())?;
        let before = probe.load1;
        let ours = workload::hubbard(false)?;
        let ff = self.ffsim.hubbard()?;
        let theirs = ffsim_client::number(&ff, "value")?;
        let reference = hubbard_analytic();
        let (ours_gap, theirs_gap) = admit(ours, theirs, reference).map_err(|d| d.reason)?;
        let flipped = workload::hubbard(true)?;
        let delta = sign_flip_detected(theirs, flipped)?;
        let after = load1()?;
        let external = census.finish()?.with_pre_row(&probe);
        self.void_if_hot(row_id, &external)?;
        let (ext_b, ext_d) = (
            external.external_cores_before.unwrap_or(f64::NAN),
            external.external_cores_during_row,
        );
        let sha = workload::file_sha("examples/fermionic/hubbard2.qasm")?;
        let evidence = format!(
            "statevector {ours:.16} ffsim {theirs:.16} reference {reference:.16} \
             gaps {ours_gap:.3e}/{theirs_gap:.3e}. rbs sign flip |Δ|={delta:.3e} \
             against unflipped ffsim. load1 {before} → {after}; outside the lane {ext_b:.2} cores before, {ext_d:.2} during (§4.4a). {}",
            self.fqe
        );
        let row = capability(
            row_id,
            &self.recorded,
            self.git.clone(),
            workload(
                "hubbard2",
                "examples/fermionic/hubbard2.qasm",
                &sha,
                "examples/fermionic/hubbard2.qasm --qasm-dialect lenient + hubbard2_op.txt",
                None,
                4,
                1,
                "examples/fermionic/hubbard2_op.txt",
            ),
            Direction::OursOnly,
            "Correctness on the two-electron ground energy 2-2*sqrt(2). Both arms return it. \
             No time is published: 4 qubits is binding-dominated in either direction.",
            &evidence,
            arm(
                "omega-backend-statevector",
                our_versions(&self.git.worktree),
                threads_statevector(),
                "none",
                map(&[("path", "F2 jordan_wigner of hubbard2_op.txt, then expectation")]),
                vec![
                    witness("value_gate", format!("gap {ours_gap:.3e}")),
                    witness("sign_flip_detected", format!("{delta:.6e}")),
                    witness("ffsim_version", self.versions["ffsim"].clone()),
                ],
            ),
            self.ffsim_arm(
                vec![
                    witness("ffsim_version", self.versions["ffsim"].clone()),
                    witness("value_gate", format!("{theirs:.16}")),
                ],
                map(&[(
                    "path",
                    "occupation-basis linear_operator on hubbard2_op.txt, symmetric singlet",
                )]),
            ),
            "speed classification refused: 4 qubits, a wall-clock measures call overhead. \
             ffsim returned the energy in the (1,1) sector.",
            f64_precision("f64 on both arms. Two-sided relative gate at 1e-10."),
            vec![
                "binding-dominated capability/correctness row".into(),
                "defect not caught: the energy is stationary in the rbs angle, so a small angle error inside 1e-10 is invisible. ffsim contracts the analytic state and does not execute the H/CX ladder.".into(),
                self.fqe.clone(),
            ],
        )?;
        self.write(row)
    }

    fn h2(&mut self) -> Result<(), String> {
        let row_id = "ferm-h2-ground";
        eprintln!("BEGIN {row_id}");
        let probe = wait_quiet_external(self.void_above, &self.own_pids(), load1)?;
        let census = CpuCensus::start(&self.own_pids())?;
        let before = probe.load1;
        let ours = workload::h2(false)?;
        let ff = self.ffsim.h2()?;
        let theirs = ffsim_client::number(&ff, "value")?;
        let reference = workload::h2_reference()?;
        let (ours_gap, theirs_gap) = admit(ours, theirs, reference).map_err(|d| d.reason)?;
        let flipped = workload::h2(true)?;
        let delta = sign_flip_detected(theirs, flipped)?;
        let after = load1()?;
        let external = census.finish()?.with_pre_row(&probe);
        self.void_if_hot(row_id, &external)?;
        let (ext_b, ext_d) = (
            external.external_cores_before.unwrap_or(f64::NAN),
            external.external_cores_during_row,
        );
        let sha = workload::file_sha("examples/fermionic/h2_ground.qasm")?;
        let evidence = format!(
            "statevector {ours:.16} ffsim {theirs:.16} E_FCI {reference:.16} \
             gaps {ours_gap:.3e}/{theirs_gap:.3e}. rz sign flip |Δ|={delta:.3e} \
             against unflipped ffsim. load1 {before} → {after}; outside the lane {ext_b:.2} cores before, {ext_d:.2} during (§4.4a). {}",
            self.fqe
        );
        let row = capability(
            row_id,
            &self.recorded,
            self.git.clone(),
            workload(
                "h2-ground",
                "examples/fermionic/h2_ground.qasm",
                &sha,
                "examples/fermionic/h2_ground.qasm + h2_op.txt, theta from h2_reference.txt",
                None,
                4,
                1,
                "examples/fermionic/h2_op.txt",
            ),
            Direction::OursOnly,
            "Correctness on the H2/STO-3G FCI energy at R=0.7414 Angstrom. Both arms return \
             the OpenFermion file's E_FCI. No time is published: 4 qubits is binding-dominated \
             in either direction.",
            &evidence,
            arm(
                "omega-backend-statevector",
                our_versions(&self.git.worktree),
                threads_statevector(),
                "none",
                map(&[("path", "F2 jordan_wigner of h2_op.txt, then expectation")]),
                vec![
                    witness("value_gate", format!("gap {ours_gap:.3e}")),
                    witness("sign_flip_detected", format!("{delta:.6e}")),
                    witness("ffsim_version", self.versions["ffsim"].clone()),
                ],
            ),
            self.ffsim_arm(
                vec![
                    witness("ffsim_version", self.versions["ffsim"].clone()),
                    witness("value_gate", format!("{theirs:.16}")),
                ],
                map(&[(
                    "path",
                    "occupation-basis cos(theta)|HF>+sin(theta)|doubles>, not a QASM replay",
                )]),
            ),
            "speed classification refused: 4 qubits, a wall-clock measures call overhead. \
             ffsim returned the energy in the (1,1) sector.",
            f64_precision("f64 on both arms. Two-sided relative gate at 1e-10."),
            vec![
                "binding-dominated capability/correctness row".into(),
                "defect not caught: the energy is stationary at theta, and adding 2*pi to the rz leaves it unchanged. ffsim builds the occupation state and does not execute the Pauli rotation.".into(),
                self.fqe.clone(),
            ],
        )?;
        self.write(row)
    }

    fn kitaev64(&mut self) -> Result<(), String> {
        let row_id = "ferm-kitaev-n64";
        eprintln!("BEGIN {row_id}");
        let probe = wait_quiet_external(self.void_above, &self.own_pids(), load1)?;
        let census = CpuCensus::start(&self.own_pids())?;
        let before = probe.load1;
        let got = workload::kitaev64(false)?;
        if !got.exact || got.dropped_mass != 0.0 || !got.informative {
            return Err(format!(
                "n64 certificate dropped_mass {} exact {} informative {}",
                got.dropped_mass, got.exact, got.informative
            ));
        }
        if got.final_terms != 63 || got.seed_basis != "ladder" {
            return Err(format!(
                "n64 final_terms {} seed {}",
                got.final_terms, got.seed_basis
            ));
        }
        let reference = kitaev_analytic(64);
        if rel_gap(got.energy, reference) > GATE_REL_F64 {
            return Err(format!("n64 energy {} vs {reference}", got.energy));
        }
        let flipped = workload::kitaev64(true)?.energy;
        let delta = sign_flip_detected(reference, flipped)?;
        let refusal_body = self.ffsim.kitaev64()?;
        if refusal_body.get("ok").and_then(|v| v.as_bool()) != Some(false) {
            return Err(format!("ffsim did not refuse Kitaev n64: {refusal_body}"));
        }
        if refusal_body.get("allocated").and_then(|v| v.as_bool()) != Some(false) {
            return Err("ffsim n64 response does not say allocated=false".into());
        }
        let error = ffsim_client::text(&refusal_body, "error")?;
        kitaev_sector_refusal(&error)?;
        let dense = ffsim_client::text(&refusal_body, "dense_amplitudes")?;
        if dense != "18446744073709551616" {
            return Err(format!("dense amplitude count is {dense}, want 2^64"));
        }
        let after = load1()?;
        let external = census.finish()?.with_pre_row(&probe);
        self.void_if_hot(row_id, &external)?;
        let (ext_b, ext_d) = (
            external.external_cores_before.unwrap_or(f64::NAN),
            external.external_cores_during_row,
        );
        let sha = workload::file_sha("examples/fermionic/kitaev_n64.qasm")?;
        let evidence = format!(
            "majoranaprop energy {:.16} reference {reference} dropped_mass {} \
             final_terms {} peak_terms {} observable_range {} seed_basis {} exact {} \
             informative {}. Pairing sign flip |Δ|={delta:.3e} against the closed form, \
             not against a second propagation. load1 {before} → {after}; outside the lane {ext_b:.2} cores before, {ext_d:.2} during (§4.4a). {}",
            got.energy,
            got.dropped_mass,
            got.final_terms,
            got.peak_terms,
            got.observable_range,
            got.seed_basis,
            got.exact,
            got.informative,
            self.fqe
        );
        let refusal = format!(
            "ffsim.protocols.linear_operator refused before allocating: {error} \
             Dense competitors face {dense} amplitudes (2^64); this process did not allocate them. \
             FQE's sector restriction does not help, because the Kitaev pairing terms do not \
             conserve particle number, so there is no (n_alpha, n_beta) block that contains the state. \
             That is why this is a capability result and not a smaller sector in disguise."
        );
        let row = capability(
            row_id,
            &self.recorded,
            self.git.clone(),
            workload(
                "kitaev-n64",
                "examples/fermionic/kitaev_n64.qasm",
                &sha,
                "examples/fermionic/kitaev_n64.qasm + examples/fermionic/kitaev_op_n64.txt",
                None,
                64,
                1,
                "examples/fermionic/kitaev_op_n64.txt",
            ),
            Direction::OursOnly,
            "majoranaprop returns a certified energy at 64 modes. Dense statevectors face 2^64. \
             FQE's particle-number sector does not contain the state, because the pairing terms \
             do not conserve particle number.",
            &evidence,
            arm(
                "omega-backend-majoranaprop",
                our_versions(&self.git.worktree),
                1,
                "none",
                map(&[
                    ("seed", "ladder, F3"),
                    ("truncation", "none"),
                    ("statevector", "not called"),
                ]),
                vec![
                    witness("seed_basis", &got.seed_basis),
                    witness("dropped_mass", format!("{}", got.dropped_mass)),
                    witness("exact", format!("{}", got.exact)),
                    witness("informative", format!("{}", got.informative)),
                    witness("final_terms", format!("{}", got.final_terms)),
                    witness("energy", format!("{:.16}", got.energy)),
                    witness("sign_flip_detected", format!("{delta:.6e}")),
                ],
            ),
            self.ffsim_arm(
                vec![
                    witness("ffsim_version", self.versions["ffsim"].clone()),
                    witness("linear_operator_refused", error.clone()),
                    witness("allocated", "false"),
                ],
                map(&[("n", "64"), ("call", "linear_operator norb=64 nelec=1")]),
            ),
            &refusal,
            omega_emu_compare::Precision {
                ours: "f64".to_string(),
                competitor: "none".to_string(),
                matched: false,
                note: Some(
                    "ffsim refuses before any dtype is involved. There is no competitor \
                     value, so this cannot be quoted as a precision win (§4.5)."
                        .to_string(),
                ),
            },
            vec![
                "capability row, ours only".into(),
                "defect not caught: |->^64 has the same energy, and a different operator with the same <+/H/+> and 63 monomials would pass".into(),
                self.fqe.clone(),
            ],
        )?;
        self.write(row)
    }

    fn lucj(&mut self) -> Result<(), String> {
        let row_id = "ferm-lucj16";
        for attempt in 1..=3 {
            eprintln!("BEGIN {row_id} attempt {attempt}");
            let probe = wait_quiet_external(self.void_above, &self.own_pids(), load1)?;
            let census = CpuCensus::start(&self.own_pids())?;
            let before = probe.load1;
            let prepared = self.ffsim.lucj_prepare()?;
            let ffsim_fixed = ffsim_client::number(&prepared, "fixed_cost_s")?;
            let ffsim_dim = prepared
                .get("dim")
                .and_then(|v| v.as_u64())
                .ok_or("lucj prepare missing dim")?;
            if ffsim_dim != 70 * 70 {
                return Err(format!("ffsim LUCJ dim {ffsim_dim}, want C(8,4)^2 = 4900"));
            }
            let (circuit, our_fixed) = workload::lucj_lower_seconds()?;
            let checked = workload::lucj(false)?;
            if checked.sector_dim != 12870 {
                return Err(format!(
                    "sector dim {}, want C(16,8)=12870",
                    checked.sector_dim
                ));
            }
            if checked.electrons != 8 {
                return Err(format!("LUCJ electrons {}", checked.electrons));
            }
            let ff_once = self.ffsim.lucj()?;
            let theirs = ffsim_client::number(&ff_once, "value")?;
            let gate = assemble::lucj_gate(checked.sector, theirs);
            if !gate.passed {
                let reason = format!(
                    "disagreement finding: sector {} ffsim {} gap {}",
                    checked.sector, theirs, gate.ours_gap
                );
                eprintln!("{reason}");
                let void = VoidRow {
                    row_id: row_id.to_string(),
                    lane: omega_emu_compare::Lane::Fermionic,
                    phase: "E6".to_string(),
                    recorded: self.recorded.clone(),
                    reason,
                    context: json!({
                        "sector": checked.sector,
                        "statevector": checked.statevector,
                        "ffsim": theirs,
                    }),
                };
                self.writer.write_void(&void).map_err(|e| e.to_string())?;
                return Err("LUCJ value gate failed".into());
            }
            if rel_gap(checked.sector, checked.statevector) > GATE_REL_F64 {
                return Err(format!(
                    "sector {} and statevector {} disagree on the same QASM",
                    checked.sector, checked.statevector
                ));
            }
            let flipped = workload::lucj(true)?;
            let delta = sign_flip_detected(theirs, flipped.sector)?;
            // Warmup, discarded. Imports already happened. This pays the first call.
            let _ = workload::time_lucj_sector(&circuit)?;
            let _ = self.ffsim.lucj()?;
            let mut ours_s = Vec::with_capacity(5);
            let mut theirs_s = Vec::with_capacity(5);
            for _ in 0..5 {
                ours_s.push(workload::time_lucj_sector(&circuit)?);
                let sample = self.ffsim.lucj()?;
                let again = ffsim_client::number(&sample, "value")?;
                if (again - theirs).abs() > 1e-8 {
                    return Err(format!("ffsim LUCJ value moved from {theirs} to {again}"));
                }
                theirs_s.push(ffsim_client::number(&sample, "seconds")?);
            }
            let after = load1()?;
            let external = census.finish()?.with_pre_row(&probe);
            if row_void_at_load(external.external_cores_during_row, self.void_above) {
                eprintln!(
                    "{:.2} cores outside the lane during {row_id} (load1 {before} → {after}); cooldown 60s",
                    external.external_cores_during_row
                );
                std::thread::sleep(Duration::from_secs(60));
                continue;
            }
            let spread = spread_flagged(&ours_s)? || spread_flagged(&theirs_s)?;
            let sha =
                workload::file_sha("tools/emu_compare/qasm/lucj16_norb8_na4_r1_s0_j0.1.qasm")?;
            let ours_arm = speed_arm(
                arm(
                    "omega-backend-sector",
                    our_versions(&self.git.worktree),
                    1,
                    "none",
                    map(&[
                        ("path", "F2 jordan_wigner of 1 [0^ 0], sector expectation"),
                        ("qasm", "lenient; rbs kept native"),
                        ("reconstruction", "to_statevector is outside the timer"),
                    ]),
                    vec![
                        witness("sector_dim", format!("{}", checked.sector_dim)),
                        witness("electrons", format!("{}", checked.electrons)),
                        witness(
                            "sector_matches_statevector",
                            format!("{:.3e}", rel_gap(checked.sector, checked.statevector)),
                        ),
                        witness("sign_flip_detected", format!("{delta:.6e}")),
                        witness("value_gate", format!("gap {:.3e}", gate.ours_gap)),
                    ],
                ),
                &ours_s,
                our_fixed,
                self.ours_floor,
            )?;
            let their_arm = speed_arm(
                self.ffsim_arm(
                    vec![
                        witness("ffsim_version", self.versions["ffsim"].clone()),
                        witness("ffsim_dim", format!("{ffsim_dim}")),
                        witness("value_gate", format!("{theirs:.16}")),
                        witness(
                            "qasm_not_read",
                            "UCJ rebuilt from seed 0 reps 1 jastrow 0.1",
                        ),
                    ],
                    map(&[
                        ("path", "apply_unitary of UCJOpSpinBalanced, one fused op"),
                        (
                            "generator",
                            "lucj_gen.py 8 4 --seed 0 --reps 1 --jastrow-scale 0.1",
                        ),
                        ("threading", "Accelerate default, not pinned to 1"),
                    ]),
                ),
                &theirs_s,
                ffsim_fixed,
                self.ffsim_floor,
            )?;
            let mut notes = vec![
                "gate-fusion asymmetry: ffsim applies UCJOpSpinBalanced as one fused orbital rotation; we decompose to Givens (rbs) plus diagonal phases (rz, cp, p). The ratio is that asymmetry.".into(),
                format!(
                    "sector dimension C(16,8)={} ; ffsim spin-balanced dimension C(8,4)^2={ffsim_dim}. Ours does not split alpha from beta.",
                    checked.sector_dim
                ),
                "the pin is the generator invocation, not the QASM bytes: rbs is not in qelib1.inc, and a gate rbs definition would expand through h, which the sector backend refuses.".into(),
                format!(
                    "n_0 sector {:.16} statevector {:.16} ffsim {:.16}",
                    checked.sector, checked.statevector, theirs
                ),
                format!(
                    "sign flip of rbs(-0.9916747497747819) q[1],q[0] |Δ|={delta:.6e} against \
                     unflipped ffsim. The file's first rbs on that pair is the identity on \
                     two filled modes, and negating every rbs leaves n_0 unchanged, so \
                     neither of those is the pin"
                ),
                "defect not caught: the gate is one occupation, <n_0>. An error that preserves n_0 passes, and negating every rbs is one of those: it leaves n_0 unchanged. A global phase does too.".into(),
                self.fqe.clone(),
            ];
            if spread {
                let ours_ratio = max_min(&ours_s);
                let their_ratio = max_min(&theirs_s);
                notes.push(format!(
                    "FLAGGED: max/min across repeats is {ours_ratio:.3} (ours) and {their_ratio:.3} (ffsim), over 5 (§4.3a). The classification is not a stable engine comparison."
                ));
            } else {
                notes.push(format!(
                    "max/min ours {:.3} ffsim {:.3}, under 5",
                    max_min(&ours_s),
                    max_min(&theirs_s)
                ));
            }
            let load = LoadRecord {
                load1_before: before,
                load1_after: after,
                void_above: self.void_above,
                cooldown_s: if attempt > 1 { Some(60) } else { None },
                hostgate_cap: Some(self.hostgate_cap.clone()),
                gpu: None,
                external_cpu: Some(external.clone()),
            };
            let row = speed(
                row_id,
                &self.recorded,
                self.git.clone(),
                workload(
                    "lucj16",
                    "tools/emu_compare/qasm/lucj16_norb8_na4_r1_s0_j0.1.qasm",
                    &sha,
                    "lucj_gen.py 8 4 lucj16_norb8_na4_r1_s0_j0.1.qasm --seed 0 --reps 1 --jastrow-scale 0.1",
                    Some(0),
                    16,
                    22,
                    "1 [0^ 0]",
                ),
                ours_arm,
                their_arm,
                gate,
                load,
                notes,
            )?;
            self.write(row)?;
            return Ok(());
        }
        Err(format!("{row_id} stayed above load 2"))
    }

    fn ffsim_arm(
        &self,
        witnesses: Vec<omega_emu_compare::PathWitness>,
        knobs: BTreeMap<String, String>,
    ) -> omega_emu_compare::Arm {
        arm(
            "ffsim",
            self.versions.clone(),
            self.ffsim_threads,
            &self.blas,
            knobs,
            witnesses,
        )
    }

    /// §4.4a: a capability row is refused when work outside the lane held more
    /// than the threshold before or during it. load1 is recorded in the
    /// evidence for context and is not the test.
    fn void_if_hot(&self, row_id: &str, external: &ExternalCpu) -> Result<(), String> {
        match crate::gate::external_void_reason(row_id, external, self.void_above) {
            Some(reason) => Err(reason),
            None => Ok(()),
        }
    }

    /// §4.4a: the lane's own tree — this process and the ffsim process.
    fn own_pids(&self) -> Vec<u32> {
        vec![std::process::id(), self.ffsim.pid()]
    }

    fn write(&mut self, row: omega_emu_compare::Row) -> Result<(), String> {
        match &row.body {
            omega_emu_compare::RowBody::Capability(cap) => {
                eprintln!("CAP {}\t{:?}", row.row_id, cap.direction);
            }
            omega_emu_compare::RowBody::Speed(speed) => {
                eprintln!(
                    "ROW {}\t{:?}\tratio={}",
                    row.row_id, speed.classification, speed.ratio_competitor_over_ours
                );
            }
        }
        self.writer.write_row(&row).map_err(|e| e.to_string())
    }
}

fn our_versions(rev: &str) -> BTreeMap<String, String> {
    map(&[
        ("omega-backend-statevector", rev),
        ("omega-backend-sector", rev),
        ("omega-backend-majoranaprop", rev),
        ("git", rev),
    ])
}

fn max_min(samples: &[f64]) -> f64 {
    let min = samples.iter().copied().fold(f64::MAX, f64::min);
    let max = samples.iter().copied().fold(0.0_f64, f64::max);
    max / min
}

fn pin_ffsim(ffsim: &Ffsim) -> Result<BTreeMap<String, String>, String> {
    let mut versions = BTreeMap::new();
    for (name, want) in FFSIM_PIN {
        let got = ffsim.version(name)?;
        if &got != want {
            return Err(format!(
                "{name} is {got}, MANIFEST pins {want}; refusing the row"
            ));
        }
        versions.insert((*name).to_string(), got);
    }
    versions.insert("python".to_string(), ffsim.version("python")?);
    versions.insert("blas".to_string(), ffsim.version("blas")?);
    Ok(versions)
}

fn min_of(n: usize, mut sample: impl FnMut() -> Result<f64, String>) -> Result<f64, String> {
    let mut best = f64::MAX;
    for _ in 0..n {
        best = best.min(sample()?);
    }
    Ok(best)
}

pub fn append_void(out: &Path, row_id: &str, reason: &str) -> Result<(), String> {
    if !out.exists() {
        return Err(format!(
            "{} does not exist; nothing to append a void to",
            out.display()
        ));
    }
    let void = VoidRow {
        row_id: row_id.to_string(),
        lane: omega_emu_compare::Lane::Fermionic,
        phase: "E6".to_string(),
        recorded: today()?,
        reason: reason.to_string(),
        context: json!({"source": "hostgate --watch killed the process tree"}),
    };
    let mut writer = RowWriter::append(out).map_err(|e| e.to_string())?;
    writer.write_void(&void).map_err(|e| e.to_string())
}

fn git_snap() -> Result<GitRev, String> {
    let compiled = option_env!("OMEGA_EMU_GIT_REV").unwrap_or("").to_string();
    let worktree = git(&["rev-parse", "HEAD"])?;
    let porcelain = git(&["status", "--porcelain"])?;
    let rev = GitRev {
        compiled_from: compiled,
        worktree,
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
        .current_dir(workload::repo_root())
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
    let out = Command::new("uptime").output().map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out.stdout);
    let rest = text
        .split("load average")
        .nth(1)
        .ok_or("uptime output has no load average")?;
    let mut nums = Vec::new();
    for tok in rest.split(|c: char| c == ',' || c.is_whitespace() || c == ':') {
        if let Ok(v) = tok.trim().parse::<f64>() {
            nums.push(v);
        }
    }
    nums.first()
        .copied()
        .ok_or_else(|| "uptime load average had no number".into())
}
