// SPDX-License-Identifier: Apache-2.0
//! Decisions for the quimb comparison lane.
//!
//! The example `mps_quimb_compare` is the script that runs the rows. This
//! module is the part a test can execute: the bond witness, the tie rule,
//! the truncation-match rule, and the circuit the witness is aimed at.
//! The akilles 32-core replicate is not run from here. Depth ≥ 12 is where
//! the SVD is the bulk of the row, and that is where a 32-core OpenBLAS box
//! can invert an M4 result; depth 4 does not saturate the bond and would not
//! be the row that flips.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use omega_backend_mps::select::AUTO_EPS;
use omega_backend_mps::svd::SvdResultFlat;
use omega_backend_mps::{default_svd_flat_fn, MpsBackend, SvdFlatFn};
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

static SVD_CALLS: AtomicUsize = AtomicUsize::new(0);
static SVD_MAX_COLS: AtomicUsize = AtomicUsize::new(0);
static SVD_LOCK: Mutex<()> = Mutex::new(());

/// Shim over the production SVD. `with_svd_fn` is how the harness proves the
/// measured path ran: the certificate then says `custom` because the shim is
/// what was installed, and the shim's body is [`default_svd_flat_fn`], which
/// is the kernel a normal `MpsBackend::new` would have called.
pub fn witness_svd(
    a: &[num_complex::Complex64],
    rows: usize,
    cols: usize,
    lda: usize,
    max_rank: usize,
    threshold: f64,
) -> SvdResultFlat {
    let out = (default_svd_flat_fn())(a, rows, cols, lda, max_rank, threshold);
    SVD_CALLS.fetch_add(1, Ordering::Relaxed);
    SVD_MAX_COLS.fetch_max(cols, Ordering::Relaxed);
    out
}

/// The function pointer the backend stores. Named so a reader can see the
/// shim is what `with_svd_fn` receives.
pub fn witness_svd_fn() -> SvdFlatFn {
    witness_svd
}

#[derive(Clone, Debug)]
pub struct GateRec {
    pub name: &'static str,
    pub qubits: Vec<u32>,
    pub params: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Built {
    pub circuit: CircuitIR,
    pub gates: Vec<GateRec>,
    pub n: u32,
    pub depth: usize,
}

/// The committed HEA file, lowered by this repo's parser. The in-memory
/// builder stays for the ghost and the exact fixture; a published row names
/// this artifact.
#[derive(Clone, Debug)]
pub struct Artifact {
    pub built: Built,
    pub id: String,
    pub file: String,
    pub sha256: String,
    pub generator: String,
}

/// Gates the quimb arm can apply, taken from a lowered circuit so the file
/// and the Python job cannot be two different circuits.
pub fn gates_from_circuit(circuit: &CircuitIR) -> Result<Vec<GateRec>, String> {
    circuit
        .ops
        .iter()
        .map(|op| {
            let name = match &op.gate {
                GateKind::H => "h",
                GateKind::CX => "cx",
                GateKind::Rz => "rz",
                GateKind::Rx => "rx",
                GateKind::Ry => "ry",
                other => {
                    return Err(format!(
                        "gate {other:?} is not in the HEA export this lane times"
                    ))
                }
            };
            let mut params = Vec::with_capacity(op.params.len());
            for p in &op.params {
                match p {
                    ParamExpr::Concrete(v) => params.push(*v),
                    _ => {
                        return Err(
                            "symbolic parameter in a pinned artifact; the file is not concrete"
                                .into(),
                        )
                    }
                }
            }
            Ok(GateRec {
                name,
                qubits: op.qubits.iter().map(|q| q.0).collect(),
                params,
            })
        })
        .collect()
}

fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Load one committed HEA artifact and refuse it if the bytes are not the
/// manifest's hash, or if lowering it does not reproduce [`hea`].
pub fn load_hea(n: u32, depth: usize) -> Result<Artifact, String> {
    let id = format!("hea-{n}q-d{depth}");
    let root = workspace_root();
    let manifest_path = root.join("tools/emu_compare/MANIFEST.json");
    let raw = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("reading {}: {e}", manifest_path.display()))?;
    let doc: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("MANIFEST.json: {e}"))?;
    let art = doc
        .get("artifacts")
        .and_then(|a| a.get(&id))
        .ok_or_else(|| format!("MANIFEST.json has no artifact {id}"))?;
    let file = art
        .get("file")
        .and_then(|v| v.as_str())
        .ok_or("artifact file")?
        .to_string();
    let sha = art
        .get("sha256")
        .and_then(|v| v.as_str())
        .ok_or("artifact sha256")?
        .to_string();
    let generator = art
        .get("generator")
        .and_then(|v| v.as_str())
        .ok_or("artifact generator")?
        .to_string();
    let bytes = std::fs::read(root.join(&file)).map_err(|e| format!("reading {file}: {e}"))?;
    let got = sha256_hex(&bytes);
    if got != sha {
        return Err(format!(
            "{file} sha256 is {got}, manifest says {sha}; the row would not name its circuit"
        ));
    }
    let source = String::from_utf8(bytes).map_err(|e| format!("{file} is not utf-8: {e}"))?;
    let circuit = omega_parser::lower_to_ir(&source)?;
    if circuit.num_qubits != n {
        return Err(format!(
            "{file} lowered to {} qubits, artifact id says {n}",
            circuit.num_qubits
        ));
    }
    let gates = gates_from_circuit(&circuit)?;
    let reference = hea(n, depth);
    if gates.len() != reference.gates.len() {
        return Err(format!(
            "{file} lowered to {} gates, the brickwall builder has {}",
            gates.len(),
            reference.gates.len()
        ));
    }
    for (i, (got, want)) in gates.iter().zip(&reference.gates).enumerate() {
        if got.name != want.name || got.qubits != want.qubits || got.params != want.params {
            return Err(format!(
                "{file} gate {i} is {got:?}, the brickwall builder has {want:?}"
            ));
        }
    }
    Ok(Artifact {
        built: Built {
            circuit,
            gates,
            n,
            depth,
        },
        id,
        file,
        sha256: sha,
        generator,
    })
}

/// Two qubits, no gates. The per-call floor (§4.7): the smallest invocation
/// both arms can run, measured rather than invented.
pub fn empty_pair() -> Built {
    Built {
        circuit: CircuitIR::new(2, CircuitType::GateBased),
        gates: Vec::new(),
        n: 2,
        depth: 0,
    }
}

fn push(
    circuit: &mut CircuitIR,
    gates: &mut Vec<GateRec>,
    gate: GateKind,
    qs: Vec<u32>,
    ps: Vec<f64>,
    name: &'static str,
) {
    gates.push(GateRec {
        name,
        qubits: qs.clone(),
        params: ps.clone(),
    });
    circuit.ops.push(GateOp {
        gate,
        qubits: qs.into_iter().map(Qubit).collect(),
        params: ps.into_iter().map(ParamExpr::Concrete).collect(),
        classical_bit: None,
        condition: None,
    });
}

/// The brickwall `mps_stage_profile` times: H on every qubit, then per depth
/// a CX·Rz(0.25)·CX brick and an Rx(0.15) ring. One builder writes the
/// circuit and the gate list, so the two arms cannot drift apart.
pub fn hea(n: u32, depth: usize) -> Built {
    let mut circuit = CircuitIR::new(n, CircuitType::GateBased);
    let mut gates = Vec::new();
    for q in 0..n {
        push(&mut circuit, &mut gates, GateKind::H, vec![q], vec![], "h");
    }
    for d in 0..depth {
        let offset = d as u32 & 1;
        let mut q = offset;
        while q + 1 < n {
            push(
                &mut circuit,
                &mut gates,
                GateKind::CX,
                vec![q, q + 1],
                vec![],
                "cx",
            );
            push(
                &mut circuit,
                &mut gates,
                GateKind::Rz,
                vec![q + 1],
                vec![0.25],
                "rz",
            );
            push(
                &mut circuit,
                &mut gates,
                GateKind::CX,
                vec![q, q + 1],
                vec![],
                "cx",
            );
            q += 2;
        }
        for qubit in 0..n {
            push(
                &mut circuit,
                &mut gates,
                GateKind::Rx,
                vec![qubit],
                vec![0.15],
                "rx",
            );
        }
    }
    Built {
        circuit,
        gates,
        n,
        depth,
    }
}

/// Four qubits of the HEA brickwall, then `Ry(0.8)` on qubit 0.
///
/// The plain HEA at this size leaves ⟨Z₀⟩ at the noise floor, where a
/// relative 1e-10 gate is a comparison of rounding. The final rotation
/// moves ⟨Z₀⟩ to O(1) while χ = 8 still holds the state exactly (the
/// maximum bond of 4 qubits is 4).
pub fn exact_biased() -> Built {
    let mut built = hea(4, 2);
    push(
        &mut built.circuit,
        &mut built.gates,
        GateKind::Ry,
        vec![0],
        vec![0.8],
        "ry",
    );
    built
}

/// CNOT on |+⟩|+⟩. That state is a +1 eigenstate of CNOT, so the gate is an
/// identity and the bond stays 1. `contract_bench` warmed up on this circuit,
/// printed "0 of N pairs on GPU" beside "speedup 1.16x", and the bond had
/// never crossed the dispatch threshold. Kept here so a harness that times
/// it is timing nothing.
pub fn cnot_on_plus_plus() -> Built {
    let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
    let mut gates = Vec::new();
    push(&mut circuit, &mut gates, GateKind::H, vec![0], vec![], "h");
    push(&mut circuit, &mut gates, GateKind::H, vec![1], vec![], "h");
    push(
        &mut circuit,
        &mut gates,
        GateKind::CX,
        vec![0, 1],
        vec![],
        "cx",
    );
    Built {
        circuit,
        gates,
        n: 2,
        depth: 1,
    }
}

pub fn z0() -> Observable {
    Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    }
}

#[derive(Clone, Debug)]
pub struct OursRun {
    pub value: f64,
    pub discarded_weight: f64,
    pub fidelity_estimate: f64,
    pub max_bond: usize,
    pub svd_calls: usize,
    pub max_cols: usize,
    pub seconds: f64,
}

fn run_ours_with(
    circuit: &CircuitIR,
    chi: usize,
    adaptive: Option<f64>,
) -> Result<OursRun, String> {
    let _guard = SVD_LOCK.lock().map_err(|e| e.to_string())?;
    SVD_CALLS.store(0, Ordering::Relaxed);
    SVD_MAX_COLS.store(0, Ordering::Relaxed);
    let params = ParameterBinding::new();
    let mut backend = MpsBackend::new(chi)
        .with_svd_fn(witness_svd_fn())
        .with_max_discarded_weight(f64::INFINITY);
    if let Some(eps) = adaptive {
        backend = backend.with_adaptive(eps);
    }
    let t = Instant::now();
    let value = backend
        .expectation(circuit, &params, &z0())
        .map_err(|e| e.to_string())?;
    let seconds = t.elapsed().as_secs_f64();
    let st = backend.last_run_stats();
    Ok(OursRun {
        value,
        discarded_weight: st.discarded_weight,
        fidelity_estimate: st.fidelity_estimate,
        max_bond: st.max_bond_reached,
        svd_calls: SVD_CALLS.load(Ordering::Relaxed),
        max_cols: SVD_MAX_COLS.load(Ordering::Relaxed),
        seconds,
    })
}

/// Fixed-rank truncation at `chi`. This is what `mps:<chi>` runs.
pub fn run_ours(circuit: &CircuitIR, chi: usize) -> Result<OursRun, String> {
    run_ours_with(circuit, chi, None)
}

/// Adaptive truncation at [`AUTO_EPS`], ceiling `chi`. This is the
/// coeff_min-style knob, and it is not quimb's cutoff.
pub fn run_ours_adaptive(circuit: &CircuitIR, chi: usize) -> Result<OursRun, String> {
    run_ours_with(circuit, chi, Some(AUTO_EPS))
}

pub fn dense_z0(circuit: &CircuitIR) -> Result<f64, String> {
    StatevectorBackend::new()
        .expectation(circuit, &ParameterBinding::new(), &z0())
        .map_err(|e| e.to_string())
}

pub fn abs_gap(got: f64, oracle: f64) -> f64 {
    (got - oracle).abs()
}

/// Relative error against the dense oracle. The floor keeps a near-zero
/// expectation from dividing the comparison by noise.
pub fn relative_error(got: f64, oracle: f64) -> f64 {
    abs_gap(got, oracle) / oracle.abs().max(1e-12)
}

/// 1e-10 relative when the expectation is large enough for that ratio to sit
/// above rounding, and 1e-10 absolute when both numbers are already at the
/// noise floor. A relative test of two values near 1e-18 fails on ulps.
pub fn passes_exact_value_gate(got: f64, oracle: f64) -> bool {
    let mag = oracle.abs().max(got.abs());
    if mag < 1e-8 {
        return abs_gap(got, oracle) <= 1e-10;
    }
    relative_error(got, oracle) <= 1e-10
}

/// Two truncations are the same work when their dense gaps agree within 2×,
/// or when both are already below 1e-8 (both exact for this purpose).
pub fn truncation_matched(our_gap: f64, their_gap: f64) -> bool {
    let our = our_gap.abs();
    let their = their_gap.abs();
    if our <= 1e-8 && their <= 1e-8 {
        return true;
    }
    let lo = our.min(their);
    let hi = our.max(their);
    lo > 0.0 && hi / lo <= 2.0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutoffMode {
    /// quimb's default: discarded Σσ² < cutoff × total Σσ².
    Rsum2,
    /// Drop σ < cutoff × σ_max. This is our adaptive epsilon, not our chi.
    Rel,
    /// Drop σ < cutoff. Our fixed-rank path uses an absolute floor of 1e-14
    /// under the chi cap; that is this mode, not rsum2.
    Abs,
}

impl CutoffMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rsum2 => "rsum2",
            Self::Rel => "rel",
            Self::Abs => "abs",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Knob {
    pub cutoff: f64,
    pub mode: CutoffMode,
}

/// The sweep is fixed before any timing. Order is not a preference.
pub fn knob_sweep() -> Vec<Knob> {
    vec![
        Knob {
            cutoff: 1e-10,
            mode: CutoffMode::Rsum2,
        },
        Knob {
            cutoff: 0.0,
            mode: CutoffMode::Rsum2,
        },
        Knob {
            cutoff: 1e-14,
            mode: CutoffMode::Abs,
        },
        Knob {
            cutoff: 1e-10,
            mode: CutoffMode::Rel,
        },
        Knob {
            cutoff: 1e-12,
            mode: CutoffMode::Rsum2,
        },
        Knob {
            cutoff: 1e-8,
            mode: CutoffMode::Rsum2,
        },
        Knob {
            cutoff: 1e-8,
            mode: CutoffMode::Rel,
        },
    ]
}

#[derive(Clone, Debug)]
pub struct MapObs {
    pub knob: Knob,
    pub max_bond: usize,
    pub abs_gap: f64,
}

#[derive(Clone, Debug)]
pub struct MapCell {
    pub chi: usize,
    pub our_abs_gap: f64,
    pub obs: Vec<MapObs>,
}

/// Pick the quimb knob that, on every mapping cell, saturates `chi` and
/// matches our fixed-rank dense gap. When two knobs are within 10% on that
/// score, take the larger cutoff: same accuracy, less work, which is the
/// competitor's side of an ambiguous knob. A knob that misses the bond or
/// the gap on any cell is out.
pub fn select_knob(cells: &[MapCell]) -> Result<Knob, String> {
    if cells.is_empty() {
        return Err("truncation mapping has no cells".into());
    }
    let mut best: Option<(Knob, f64)> = None;
    for cand in &cells[0].obs {
        let knob = cand.knob;
        let mut score = 0.0;
        let mut ok = true;
        for cell in cells {
            let Some(obs) = cell.obs.iter().find(|o| o.knob == knob) else {
                ok = false;
                break;
            };
            if obs.max_bond != cell.chi || !truncation_matched(cell.our_abs_gap, obs.abs_gap) {
                ok = false;
                break;
            }
            score += (cell.our_abs_gap - obs.abs_gap).abs();
        }
        if !ok {
            continue;
        }
        best = Some(match best {
            None => (knob, score),
            Some((prev, prev_score)) => {
                if prefer_knob(knob, score, prev, prev_score) {
                    (knob, score)
                } else {
                    (prev, prev_score)
                }
            }
        });
    }
    best.map(|(k, _)| k).ok_or_else(|| {
        "no quimb cutoff saturated chi and matched our dense gap on every mapping cell".into()
    })
}

fn prefer_knob(new_k: Knob, new_s: f64, old_k: Knob, old_s: f64) -> bool {
    let scale = old_s.max(new_s).max(1e-12);
    let rel = (new_s - old_s).abs() / scale;
    if rel > 0.1 {
        return new_s < old_s;
    }
    if new_k.cutoff != old_k.cutoff {
        return new_k.cutoff > old_k.cutoff;
    }
    new_k.mode == CutoffMode::Rsum2 && old_k.mode != CutoffMode::Rsum2
}

/// Quimb's reported max bond must equal the pinned chi, and so must ours.
/// The SVD call count and the column count are the other half: a row whose
/// bond saturated but whose SVD never ran is not the path on the label.
///
/// The quimb check is first. A ghost whose bond stays 1 still executes one
/// SVD, so a call-count witness would accept it.
pub fn admit_saturating_row(
    pinned_chi: usize,
    quimb_max_bond: usize,
    our_max_bond: usize,
    svd_calls: usize,
    max_cols: usize,
) -> Result<(), String> {
    if quimb_max_bond != pinned_chi {
        return Err(format!(
            "bond witness refused the row: quimb max_bond {quimb_max_bond} != pinned chi {pinned_chi}. The bond never reached the cap, so a timing would not be the chi-capped workload on the label."
        ));
    }
    if our_max_bond != pinned_chi {
        return Err(format!(
            "bond witness refused the row: our max_bond {our_max_bond} != pinned chi {pinned_chi}."
        ));
    }
    if svd_calls == 0 {
        return Err(
            "SVD witness refused the row: call count is 0. The production SVD path did not run."
                .into(),
        );
    }
    let expect = 2 * pinned_chi;
    if max_cols != expect {
        return Err(format!(
            "SVD witness refused the row: max SVD columns {max_cols} != 2*chi {expect}."
        ));
    }
    Ok(())
}

pub fn requires_bond_saturation(depth: usize) -> bool {
    depth >= 12
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeedClass {
    Win,
    Loss,
    Tie,
}

/// `ratio` is quimb's min divided by ours. The class is a win or a loss only
/// when that ratio is at least 1.3 in one direction and the [min, median]
/// intervals do not overlap. 1.3 is the pinned threshold; a 1.16× gap is a
/// tie even when the intervals are disjoint.
pub fn classify_speed(
    ours_min: f64,
    ours_med: f64,
    quimb_min: f64,
    quimb_med: f64,
) -> (SpeedClass, f64) {
    let ratio = quimb_min / ours_min;
    let disjoint = ours_med < quimb_min || quimb_med < ours_min;
    let class = if !disjoint {
        SpeedClass::Tie
    } else if ratio >= 1.3 {
        SpeedClass::Win
    } else if ours_min / quimb_min >= 1.3 {
        SpeedClass::Loss
    } else {
        SpeedClass::Tie
    };
    (class, ratio)
}

/// Depth 4 is sub-millisecond on this circuit family and is never a headline,
/// whatever the ratio says.
pub fn speed_headline(depth: usize, class: SpeedClass) -> &'static str {
    if depth < 12 {
        "directional-only"
    } else {
        match class {
            SpeedClass::Win => "win",
            SpeedClass::Loss => "loss",
            SpeedClass::Tie => "tie",
        }
    }
}

pub fn min_median(samples: &[f64]) -> Option<(f64, f64)> {
    if samples.is_empty() {
        return None;
    }
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    Some((v[0], v[v.len() / 2]))
}

pub fn timed_repeats(warmup_seconds: f64) -> usize {
    if warmup_seconds > 30.0 {
        3
    } else {
        5
    }
}

/// 1-minute load strictly above `void_above` voids the row; the threshold
/// itself does not. `void_above` comes from
/// [`omega_emu_compare::load_void_above`] for the host's core count, so the
/// harness can run on the 32-core replicate host it is assigned to.
pub fn row_void_at_load(load_1min: f64, void_above: f64) -> bool {
    omega_emu_compare::void_at_load(load_1min, void_above)
}

pub fn parse_uptime_load1(text: &str) -> Result<f64, String> {
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

pub fn heavy_row(n: u32, depth: usize) -> bool {
    n >= 20 || depth >= 12
}

/// Capability, not a speed class. Our `fidelity_estimate` is an estimate —
/// this MPS is not canonical, and the crate says so. What we attach and
/// quimb's `CircuitMPS` does not is the accumulated discarded-weight
/// certificate the refusal ceiling is written in. quimb's
/// `fidelity_estimate` / `error_estimate` are estimates from the state norm,
/// by quimb's own docstring.
pub fn format_capability_row(
    our_discarded: f64,
    our_fidelity: f64,
    quimb_fidelity: f64,
    quimb_error: f64,
) -> String {
    format!(
        "kind=capability\n\
         not-a-speed-result\n\
         ours_discarded_weight={our_discarded:.6e}\n\
         ours_fidelity_estimate={our_fidelity:.6e}\n\
         ours_fidelity_is=estimate\n\
         quimb_fidelity_estimate={quimb_fidelity:.6e}\n\
         quimb_error_estimate={quimb_error:.6e}\n\
         quimb_fidelity_is=estimate\n\
         asymmetry=accumulated discarded-weight certificate versus a norm-based estimate"
    )
}

/// One defect the row's own witnesses do not catch. Stated per shape so a
/// green run cannot be read as "nothing else could be wrong".
pub fn blind_spot(n: u32, depth: usize, chi: usize) -> &'static str {
    match (n, depth, chi) {
        (14, 4, 32) => {
            "does not catch a shared per-call overhead; depth 4 is directional and is not a headline"
        }
        (14, 4, 128) => {
            "does not catch this shape being quoted as the MPS regime; 14q at chi >= 128 is outside it"
        }
        (14, 12, 32) => {
            "does not catch quimb max_bond describing a tensor network other than the one local_expectation contracted"
        }
        (14, 12, 128) => {
            "does not catch an exact (bond-complete) 14q result being read as a truncation comparison"
        }
        (14, 24, 32) => {
            "does not catch both arms sharing an error that still matches within the 2x truncation rule"
        }
        (14, 24, 128) => {
            "does not catch a thermal drift that stays under the load-average threshold of 2"
        }
        (20, 12, 32) => {
            "does not catch the truncation knob failing to transfer to an observable other than Z0"
        }
        (20, 12, 128) => {
            "does not catch a 32-core BLAS host inverting this ranking; the akilles replicate was not run"
        }
        (20, 24, 32) => "does not catch cache residency left by the interleaved other arm",
        (20, 24, 128) => {
            "does not catch a hostgate SIGKILL, which cannot write its own void row"
        }
        _ => "does not catch a witness asserted against a different process than the one timed",
    }
}

pub fn grid_shapes() -> &'static [(u32, usize)] {
    &[(14, 4), (14, 12), (14, 24), (20, 12), (20, 24)]
}

pub fn grid_chis() -> &'static [usize] {
    &[32, 128]
}

/// Mapping is established at 14q and 20q, depth 12, before any timed row.
/// Depth 24 is checked per row against the knob this sweep selects; it is
/// not part of the sweep.
pub fn mapping_shapes() -> &'static [(u32, usize)] {
    &[(14, 12), (20, 12)]
}
