// Per-stage profile of the MPS two-qubit path: how much of a run is the
// truncated SVD, and how much is everything else (θ-contraction, reshapes,
// the SWAP network, sampling).
//
// WHY THIS EXISTS. `mps_chi_profile.rs` measures WALLCLOCK across shapes and
// was the basis for deferring MPS-SVD-on-Metal (`f9b477e`, 2026-05-12): at
// 14q × depth-4 the time is flat in χ, so bonds never saturate and no SVD
// work would be saved. That is a statement about one shape. The deferral's
// own text concedes "where SVD does dominate (depth >= 12)" without ever
// putting a number on the share — and a GPU-SVD decision rests entirely on
// that number, because Amdahl bounds the win by it. This harness measures it
// directly instead of inferring it from total time.
//
// HOW. `MpsBackend::with_svd_fn` takes a plain `fn` pointer, so a timing shim
// that delegates to the real kernel measures the production path with no
// library change and no feature gate. Accumulators are atomics because the
// Jacobi kernel is itself rayon-parallel: the shim runs on whichever thread
// called it, and SVD_NANOS therefore sums WALLCLOCK SPENT INSIDE the kernel
// (including its internal parallelism), which is the quantity Amdahl wants.
//
// MEASURE EVOLUTION, NOT RECONSTRUCTION. `execute` with `shots: None` ends in
// `mps.to_statevector()`, a dense 2^n contraction that is not MPS evolution at
// all and that no SVD accelerator would touch. At 20 qubits it swamps
// everything: the same sweep run through `execute` reports the SVD at 3% while
// the evolution-only figure below is an order of magnitude higher. This
// harness therefore drives `expectation` (MPS-contracted, no dense vector) for
// the evolution figure and times `to_statevector` SEPARATELY so the gap is
// visible rather than folded into "total".
//
// `mps_chi_profile.rs` — the empirical basis cited for the deferral — uses
// `shots: None`, so its "wallclock is flat in chi" reading is taken through
// that same dense reconstruction. That does not make the deferral's
// conclusion wrong (the dispatch-overhead argument stands on its own), but it
// does mean the shape evidence was measuring something other than the SVD.
//
// Read the result as: an ideal SVD accelerator — infinitely fast, zero
// transfer — would cut EVOLUTION time by at most the printed share.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use num_complex::Complex64;
use omega_backend_mps::svd::{truncated_svd_flat, SvdResultFlat};
use omega_backend_mps::MpsBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, MidCircuitMode, Observable, PauliOp};
use omega_core::params::ParameterBinding;

static SVD_NANOS: AtomicU64 = AtomicU64::new(0);
static SVD_CALLS: AtomicUsize = AtomicUsize::new(0);
static SVD_MAX_COLS: AtomicUsize = AtomicUsize::new(0);
static SVD_SUM_COLS: AtomicUsize = AtomicUsize::new(0);

/// Signature-identical shim over the production kernel. Must stay a plain
/// `fn` (no capture) to satisfy `SvdFlatFn`.
fn timing_svd(
    a: &[Complex64],
    rows: usize,
    cols: usize,
    lda: usize,
    max_rank: usize,
    threshold: f64,
) -> SvdResultFlat {
    let t = Instant::now();
    let out = truncated_svd_flat(a, rows, cols, lda, max_rank, threshold);
    SVD_NANOS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    SVD_CALLS.fetch_add(1, Ordering::Relaxed);
    SVD_SUM_COLS.fetch_add(cols, Ordering::Relaxed);
    SVD_MAX_COLS.fetch_max(cols, Ordering::Relaxed);
    out
}

fn reset() {
    SVD_NANOS.store(0, Ordering::Relaxed);
    SVD_CALLS.store(0, Ordering::Relaxed);
    SVD_MAX_COLS.store(0, Ordering::Relaxed);
    SVD_SUM_COLS.store(0, Ordering::Relaxed);
}

/// The same brickwall `mps_chi_profile.rs` uses, so the two harnesses are
/// talking about one circuit family.
fn entangling_circuit(num_qubits: u32, depth: usize) -> CircuitIR {
    let mut circuit = CircuitIR::new(num_qubits, CircuitType::GateBased);
    let push = |c: &mut CircuitIR, gate: GateKind, qs: Vec<u32>, ps: Vec<f64>| {
        c.ops.push(GateOp {
            gate,
            qubits: qs.into_iter().map(Qubit).collect(),
            params: ps.into_iter().map(ParamExpr::Concrete).collect(),
            classical_bit: None,
            condition: None,
        });
    };
    for q in 0..num_qubits {
        push(&mut circuit, GateKind::H, vec![q], vec![]);
    }
    for d in 0..depth {
        let offset = d as u32 & 1;
        for q in (offset..num_qubits - 1).step_by(2) {
            push(&mut circuit, GateKind::CX, vec![q, q + 1], vec![]);
            push(&mut circuit, GateKind::Rz, vec![q + 1], vec![0.25]);
            push(&mut circuit, GateKind::CX, vec![q, q + 1], vec![]);
        }
        for q in 0..num_qubits {
            push(&mut circuit, GateKind::Rx, vec![q], vec![0.15]);
        }
    }
    circuit
}

/// The default sweep, and the row set every recorded table in this harness's
/// history was taken from.
const DEFAULT_SHAPES: &[(u32, usize)] =
    &[(14, 4), (14, 12), (14, 24), (20, 12), (20, 16), (24, 12)];
const DEFAULT_CHIS: &[usize] = &[32, 128, 256];

/// Row selection, because the far end of the default sweep is not always
/// affordable. The 24q rows spend about an hour each in the `execute` column
/// alone and the 20q×depth-16 rows ran 155-214 s per row on a 32-thread host;
/// a 16 GB laptop measuring the depth split does not need either. Parse
/// `OMEGA_PROFILE_SHAPES=14x4,14x12` / `OMEGA_PROFILE_CHIS=32,128`; unset means
/// the full default sweep, so a run with no environment is the recorded one.
fn shapes_from_env() -> Vec<(u32, usize)> {
    let Ok(spec) = std::env::var("OMEGA_PROFILE_SHAPES") else {
        return DEFAULT_SHAPES.to_vec();
    };
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            let (q, d) = s
                .split_once('x')
                .unwrap_or_else(|| panic!("OMEGA_PROFILE_SHAPES entry {s:?} is not QxDEPTH"));
            (
                q.parse()
                    .unwrap_or_else(|_| panic!("bad qubit count {q:?}")),
                d.parse().unwrap_or_else(|_| panic!("bad depth {d:?}")),
            )
        })
        .collect()
}

fn chis_from_env() -> Vec<usize> {
    let Ok(spec) = std::env::var("OMEGA_PROFILE_CHIS") else {
        return DEFAULT_CHIS.to_vec();
    };
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap_or_else(|_| panic!("bad chi {s:?}")))
        .collect()
}

fn main() {
    let params = ParameterBinding::new();
    let shapes = shapes_from_env();
    let chis = chis_from_env();

    println!(
        "threads={}  (SVD share bounds what ANY svd accelerator can save)",
        rayon::current_num_threads()
    );
    println!(
        "\n{:>4} {:>6} {:>5} {:>10} {:>10} {:>7} {:>10} {:>7} {:>6}",
        "q", "depth", "chi", "evolve ms", "svd ms", "svd %", "exec ms", "calls", "maxc"
    );

    for &(nq, depth) in &shapes {
        let circuit = entangling_circuit(nq, depth);
        for &chi in &chis {
            // A single-qubit Z: the observable contraction is O(n·chi^3) and
            // negligible beside the gate sweep, so "evolve" is the gate path.
            let obs = Observable {
                terms: vec![(1.0, vec![(0u32, PauliOp::Z)])],
            };
            // Warm: first touch pays allocator and rayon pool costs.
            MpsBackend::new(chi)
                .with_svd_fn(timing_svd)
                .expectation(&circuit, &params, &obs)
                .unwrap();

            let iters = if nq > 20 || depth > 16 { 3 } else { 5 };
            let mut best = f64::INFINITY;
            let mut best_svd = 0.0;
            let (mut calls, mut maxc) = (0, 0);
            for _ in 0..iters {
                reset();
                let t = Instant::now();
                MpsBackend::new(chi)
                    .with_svd_fn(timing_svd)
                    .expectation(&circuit, &params, &obs)
                    .unwrap();
                let total = t.elapsed().as_secs_f64() * 1000.0;
                // Keep the SVD figure from the SAME run as the total, not a
                // min over independent runs — mixing runs can print a share
                // above 100%.
                if total < best {
                    best = total;
                    best_svd = SVD_NANOS.load(Ordering::Relaxed) as f64 / 1e6;
                    calls = SVD_CALLS.load(Ordering::Relaxed);
                    maxc = SVD_MAX_COLS.load(Ordering::Relaxed);
                }
            }

            // The same circuit through `execute(shots: None)`, which ends in
            // the dense 2^n `to_statevector`. `exec - evolve` IS the
            // reconstruction: the gate sweep in front of it is identical.
            // (`evolve_for_analytic` is private, so this difference is the
            // only way to price it from an example.)
            let cfg = ExecConfig {
                shots: None,
                seed: Some(0),
                mid_circuit_mode: MidCircuitMode::Skip,
            };
            let t = Instant::now();
            MpsBackend::new(chi)
                .with_svd_fn(timing_svd)
                .execute(&circuit, &params, &cfg)
                .unwrap();
            let recon = t.elapsed().as_secs_f64() * 1000.0;

            println!(
                "{:>4} {:>6} {:>5} {:>10.2} {:>10.2} {:>6.1}% {:>10.2} {:>7} {:>6}",
                nq,
                depth,
                chi,
                best,
                best_svd,
                100.0 * best_svd / best,
                recon,
                calls,
                maxc
            );
        }
    }
}
