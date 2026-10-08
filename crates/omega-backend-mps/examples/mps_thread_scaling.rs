// How the truncated SVD scales with the rayon pool size — and whether the
// `cols >= 128` parallel gate in `svd.rs` is set where it should be.
//
// WHY. `mps_stage_profile.rs` establishes that the SVD is 90-97% of MPS
// evolution. It does NOT establish that the kernel is parallel while spending
// that time, and it is mostly not: the gate is `let par = cols >= 128`
// (svd.rs), and the theta blocks reach only 64 columns at chi=32 and 120 at
// chi=128 on several shapes. Below the gate `tasks.iter()` runs, not
// `tasks.par_iter()`. Worse, `maxc` is a MAXIMUM over calls and the comment at
// svd.rs:334-336 notes a deep circuit spends most of its gates at small chi
// while ramping up — so even rows that peak above the gate are serial for most
// CALLS. This harness measures the pool-size response directly.
//
// CERTIFICATE SAFETY. `discarded_weight` feeds the truncation certificate, and
// the certificate is a BOUND, not an estimate. So every run fingerprints it:
// DW_XOR is the XOR of the raw bits of every value the kernel returned. XOR
// rather than a sum because once `par_iter` is live the calls land in
// nondeterministic order and a sum would differ in its last bits from ordering
// alone — XOR is insensitive to order but flips if any individual value
// changes. The expectation value is printed at 17 significant digits beside it.
// A change in either across thread counts is a correctness regression, not a
// timing artefact.
//
// WITHIN-HOST ONLY — the easy trap with this instrument. The fingerprint
// compares one BINARY against another ON ONE MACHINE. It is not portable:
// x86-64 and arm64 differ in the last bits of these reductions, so the same
// circuit at the same chi produces different fingerprints on different hosts
// (measured: 81f6b5cec77b7fe3 here against 81f7816be3c5add5 on an M4, with
// expectation values -9.43e-16 and 5.04e-16 — both numerically zero). They
// LOOK like they should match, which is exactly why this warning is here and
// not only in the thread where it was found. Compare a gate change on one
// box; never compare two boxes.
//
// (There is deliberately no max-discarded-weight column. A bitwise `fetch_max`
// over f64 bits is wrong for negative zero — -0.0 has the sign bit set, so it
// wins every bitwise comparison — and the observed weights are 0.0/-0.0 at
// these shapes. `last_run_stats()` already reports the real maximum.)

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use num_complex::Complex64;
use omega_backend_mps::svd::{truncated_svd_flat, SvdResultFlat};
use omega_backend_mps::MpsBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

static SVD_NANOS: AtomicU64 = AtomicU64::new(0);
static SVD_CALLS: AtomicUsize = AtomicUsize::new(0);
static DW_XOR: AtomicU64 = AtomicU64::new(0);

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
    DW_XOR.fetch_xor(out.discarded_weight.to_bits(), Ordering::Relaxed);
    out
}

fn reset() {
    SVD_NANOS.store(0, Ordering::Relaxed);
    SVD_CALLS.store(0, Ordering::Relaxed);
    DW_XOR.store(0, Ordering::Relaxed);
}

/// Identical to `mps_stage_profile`'s circuit, so the two tables compare.
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

fn main() {
    let params = ParameterBinding::new();
    let chi: usize = std::env::var("CHI")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);

    println!("threads={}  chi={}", rayon::current_num_threads(), chi);
    println!(
        "{:>4} {:>6} {:>5} {:>10} {:>10} {:>7} {:>7} {:>18} {:>26}",
        "q", "depth", "chi", "evolve ms", "svd ms", "svd %", "calls", "dw_xor", "expectation"
    );

    for &(nq, depth) in &[(14u32, 12usize), (14, 24), (20, 12), (20, 16)] {
        let circuit = entangling_circuit(nq, depth);
        let obs = Observable {
            terms: vec![(1.0, vec![(0u32, PauliOp::Z)])],
        };

        MpsBackend::new(chi)
            .with_svd_fn(timing_svd)
            .expectation(&circuit, &params, &obs)
            .unwrap();

        let mut ev = 0.0f64;
        let mut dw_xor = 0u64;
        let iters = if depth > 16 { 3 } else { 5 };
        let mut best = f64::INFINITY;
        let mut best_svd = 0.0;
        let mut calls = 0;
        for _ in 0..iters {
            reset();
            let t = Instant::now();
            let v = MpsBackend::new(chi)
                .with_svd_fn(timing_svd)
                .expectation(&circuit, &params, &obs)
                .unwrap();
            let total = t.elapsed().as_secs_f64() * 1000.0;
            // Certificate telemetry is deterministic per run, so capture it
            // every iteration rather than only on the fastest one.
            ev = v;
            dw_xor = DW_XOR.load(Ordering::Relaxed);
            if total < best {
                best = total;
                best_svd = SVD_NANOS.load(Ordering::Relaxed) as f64 / 1e6;
                calls = SVD_CALLS.load(Ordering::Relaxed);
            }
        }

        println!(
            "{:>4} {:>6} {:>5} {:>10.2} {:>10.2} {:>6.1}% {:>7} {:>18x} {:>26.17e}",
            nq,
            depth,
            chi,
            best,
            best_svd,
            100.0 * best_svd / best,
            calls,
            dw_xor,
            ev
        );
    }
}
