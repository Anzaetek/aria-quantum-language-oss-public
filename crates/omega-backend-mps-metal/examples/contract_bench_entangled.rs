//! Corrected θ-contraction bench: CPU vs Metal on a state that is ACTUALLY
//! entangled.
//!
//! WHY THIS EXISTS. `contract_bench.rs` warms with H on every qubit followed by
//! CNOT brickwall layers — but `CNOT(|+⟩⊗|+⟩) = |+⟩⊗|+⟩` exactly, so that
//! warm-up is the identity and leaves a product state with `bond_right == 1`
//! everywhere. Measured by sweeping `MPS_METAL_MIN_BOND`: at 1 every pair
//! dispatches, at 2 none do, which pins the bond at exactly 1. So the GPU path
//! never runs at the production threshold of 32, and that bench's acceptance
//! criterion ("≥1.3× CPU at 14q × depth-12 × χ=128") has never actually been
//! evaluated against a GPU — while it printed "speedup 1.16x", a CPU-vs-CPU
//! number, directly beside "0 of N pairs on GPU". See `PLAN-OPEN` §3f A10.
//!
//! Fix: interleave single-qubit rotations between the CNOT layers. That breaks
//! the |+⟩ invariance so the CNOTs genuinely entangle and bonds grow past 32,
//! which is the only regime where the Metal dispatch could possibly amortise.
//!
//! This harness ASSERTS that the GPU ran rather than reporting it. That is the
//! whole lesson of A10: `contract_bench` already had the counter and already
//! printed it, and it still went unnoticed. A count that reaches only
//! `println!` is not a guard.

use std::time::Instant;

use num_complex::Complex64;

use omega_backend_mps::mps::Mps;
use omega_backend_mps_metal::contract::{apply_two_site_gate_metal, cpu_path};

fn cnot() -> [Complex64; 16] {
    let o = Complex64::new(0.0, 0.0);
    let i = Complex64::new(1.0, 0.0);
    [i, o, o, o, o, i, o, o, o, o, o, i, o, o, i, o]
}

fn rx(theta: f64) -> [Complex64; 4] {
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    [
        Complex64::new(c, 0.0),
        Complex64::new(0.0, -s),
        Complex64::new(0.0, -s),
        Complex64::new(c, 0.0),
    ]
}

fn rz(theta: f64) -> [Complex64; 4] {
    let o = Complex64::new(0.0, 0.0);
    [
        Complex64::from_polar(1.0, -theta / 2.0),
        o,
        o,
        Complex64::from_polar(1.0, theta / 2.0),
    ]
}

fn hadamard() -> [Complex64; 4] {
    let isq2 = 1.0 / 2.0_f64.sqrt();
    [
        Complex64::new(isq2, 0.0),
        Complex64::new(isq2, 0.0),
        Complex64::new(isq2, 0.0),
        Complex64::new(-isq2, 0.0),
    ]
}

/// Warm to a genuinely entangled state. Unlike the original, the single-qubit
/// rotations between CNOT layers stop the state sitting in the CNOT-invariant
/// |+⟩^n subspace.
///
/// The warm depths at the call site are HAND-TUNED per shape, not derived —
/// bond roughly doubles every two brickwall layers, so warm 8 only reached 16
/// while warm 12-16 reaches 64-128. They will need revisiting if the circuit
/// changes.
fn warm_entangled(num_qubits: usize, chi: usize, warm_depth: usize) -> Mps {
    let mut mps = Mps::zero_state(num_qubits, chi);
    let h = hadamard();
    for q in 0..num_qubits {
        mps.apply_1q(q, &h);
    }
    let gate = cnot();
    for d in 0..warm_depth {
        for q in 0..num_qubits {
            mps.apply_1q(q, &rx(0.15));
            mps.apply_1q(q, &rz(0.25));
        }
        let offset = d & 1;
        for q in (offset..num_qubits - 1).step_by(2) {
            mps.apply_2q(q, &gate);
        }
    }
    mps
}

fn bond_stats(mps: &Mps) -> (usize, usize, usize) {
    let bonds: Vec<usize> = mps.tensors.iter().map(|t| t.bond_right).collect();
    let max = *bonds.iter().max().unwrap_or(&0);
    let min = *bonds.iter().min().unwrap_or(&0);
    let ge32 = bonds.iter().filter(|&&b| b >= 32).count();
    (min, max, ge32)
}

fn run_cpu(mps: &mut Mps, depth: usize) {
    let gate = cnot();
    for d in 0..depth {
        let offset = d & 1;
        for q in (offset..mps.n - 1).step_by(2) {
            let max_bm = mps.max_bond_dim;
            let (nl, nr, _w) = cpu_path(&mps.tensors[q], &mps.tensors[q + 1], &gate, max_bm, 1e-14);
            mps.tensors[q] = nl;
            mps.tensors[q + 1] = nr;
        }
    }
}

fn run_metal(mps: &mut Mps, depth: usize) -> usize {
    let gate = cnot();
    let mut metal_pairs = 0usize;
    for d in 0..depth {
        let offset = d & 1;
        for q in (offset..mps.n - 1).step_by(2) {
            let (nl, nr, _w, used) = apply_two_site_gate_metal(
                &mps.tensors[q],
                &mps.tensors[q + 1],
                &gate,
                mps.max_bond_dim,
                1e-14,
            );
            mps.tensors[q] = nl;
            mps.tensors[q + 1] = nr;
            if used {
                metal_pairs += 1;
            }
        }
    }
    metal_pairs
}

fn time_pass<F: FnMut()>(mut f: F, iters: usize) -> (f64, f64) {
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[0], times[times.len() / 2])
}

fn main() {
    let shapes = [
        (14usize, 8usize, 128usize, 12usize),
        (14, 8, 256, 14),
        (14, 8, 256, 16),
        (12, 8, 256, 16),
    ];
    let iters = 5;

    for &(nq, depth, chi, warm) in &shapes {
        let probe = warm_entangled(nq, chi, warm);
        let (bmin, bmax, ge32) = bond_stats(&probe);
        println!(
            "\n=== {nq}q x depth {depth} x chi={chi} (warm {warm}) — bonds after warm: min {bmin}, max {bmax}, {ge32} of {} sites >= 32 ===",
            probe.tensors.len()
        );

        let (cpu_min, cpu_med) = time_pass(
            || {
                let mut mps = warm_entangled(nq, chi, warm);
                run_cpu(&mut mps, depth);
            },
            iters,
        );
        println!(
            "  CPU   :  min {:>9.2} ms   median {:>9.2} ms",
            cpu_min, cpu_med
        );

        let (m_min, m_med) = time_pass(
            || {
                let mut mps = warm_entangled(nq, chi, warm);
                let _ = run_metal(&mut mps, depth);
            },
            iters,
        );
        let dispatched = {
            let mut mps = warm_entangled(nq, chi, warm);
            run_metal(&mut mps, depth)
        };

        // A10: FAIL, do not report. If a future change shifts the bond
        // distribution back below the dispatch threshold, this bench would
        // otherwise decay into exactly the bench it was written to replace and
        // print a plausible 1.00x while timing CPU against CPU.
        assert!(
            dispatched > 0,
            "{nq}q chi={chi} warm={warm}: the GPU path never ran — every pair fell \
             back to the CPU, so any ratio below would be CPU-vs-CPU. Raise the warm \
             depth until bonds clear the dispatch threshold."
        );

        println!(
            "  Metal :  min {:>9.2} ms   median {:>9.2} ms  ({} pairs on GPU)",
            m_min, m_med, dispatched
        );
        if m_med > 0.0 {
            println!("  speedup (CPU/Metal) : {:.3}x", cpu_med / m_med);
        }
    }
}
