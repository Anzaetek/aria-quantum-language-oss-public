//! Per-pair θ-contraction bench: CPU vs Metal.
//!
//! Runs a synthetic depth-D brickwall of two-site CNOTs on an
//! N-qubit MPS with bond-dim cap χ, measuring wallclock for the
//! per-gate `apply_two_site_gate_metal` call. Two passes:
//!
//!   * `cpu_path` — bypasses the GPU dispatch entirely; same
//!     arithmetic the existing `Mps::apply_2q` would do.
//!   * `apply_two_site_gate_metal` — dispatches to Metal when the
//!     middle bond dim ≥ `MIN_BOND_DIM_FOR_METAL`.
//!
//! Acceptance per `NEXT_SESSION_PLAN.md` item G-Metal: total wallclock
//! ≥ 1.3× CPU at 14q × depth-12 × χ=128. Run with
//!     cargo run --release --example contract_bench --features metal
//! Output reports min and median ms across 5 iterations per shape.

use std::time::Instant;

use num_complex::Complex64;

use omega_backend_mps::mps::Mps;
use omega_backend_mps_metal::contract::{apply_two_site_gate_metal, cpu_path};

fn cnot() -> [Complex64; 16] {
    let o = Complex64::new(0.0, 0.0);
    let i = Complex64::new(1.0, 0.0);
    [
        i, o, o, o, // |00⟩
        o, i, o, o, // |01⟩
        o, o, o, i, // |10⟩ → |11⟩
        o, o, i, o, // |11⟩ → |10⟩
    ]
}

fn warm_to_bond_dim(num_qubits: usize, chi: usize) -> Mps {
    // Saturate the bond dimension by repeatedly applying CNOT layers
    // across a random-phase Hadamard-like superposition starter.
    let mut mps = Mps::zero_state(num_qubits, chi);
    let h = {
        let isq2 = 1.0 / 2.0_f64.sqrt();
        [
            Complex64::new(isq2, 0.0),
            Complex64::new(isq2, 0.0),
            Complex64::new(isq2, 0.0),
            Complex64::new(-isq2, 0.0),
        ]
    };
    for q in 0..num_qubits {
        mps.apply_1q(q, &h);
    }
    let gate = cnot();
    let warmup_depth = 6;
    for d in 0..warmup_depth {
        let offset = d & 1;
        for q in (offset..num_qubits - 1).step_by(2) {
            mps.apply_2q(q, &gate);
        }
    }
    mps
}

fn run_brickwall_cpu(mps: &mut Mps, depth: usize) {
    let gate = cnot();
    for d in 0..depth {
        let offset = d & 1;
        for q in (offset..mps.n - 1).step_by(2) {
            // Mirror the public CPU pipeline: contract + gate + SVD
            // via the same `cpu_path` the Metal entry uses for its
            // SVD/split tail.
            let max_bm = mps.max_bond_dim;
            let (nl, nr) = cpu_path(&mps.tensors[q], &mps.tensors[q + 1], &gate, max_bm, 1e-14);
            mps.tensors[q] = nl;
            mps.tensors[q + 1] = nr;
        }
    }
}

fn run_brickwall_metal(mps: &mut Mps, depth: usize) -> usize {
    let gate = cnot();
    let mut metal_pairs = 0usize;
    for d in 0..depth {
        let offset = d & 1;
        for q in (offset..mps.n - 1).step_by(2) {
            let (nl, nr, used) = apply_two_site_gate_metal(
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
    let shapes = [(14usize, 12usize, 128usize), (14, 4, 128), (10, 8, 64)];
    let iters = 5;
    for &(nq, depth, chi) in &shapes {
        println!("\n=== {nq}q × depth {depth} × χ={chi} ({iters} iters) ===",);
        // CPU pass.
        let (cpu_min, cpu_med) = time_pass(
            || {
                let mut mps = warm_to_bond_dim(nq, chi);
                run_brickwall_cpu(&mut mps, depth);
            },
            iters,
        );
        println!(
            "  CPU   :  min {:>8.2} ms   median {:>8.2} ms",
            cpu_min, cpu_med,
        );

        // Metal pass.
        let (m_min, m_med) = time_pass(
            || {
                let mut mps = warm_to_bond_dim(nq, chi);
                let _ = run_brickwall_metal(&mut mps, depth);
            },
            iters,
        );
        let dispatched = {
            let mut mps = warm_to_bond_dim(nq, chi);
            run_brickwall_metal(&mut mps, depth)
        };
        println!(
            "  Metal :  min {:>8.2} ms   median {:>8.2} ms  ({} of N pairs on GPU)",
            m_min, m_med, dispatched,
        );
        if cpu_med > 0.0 {
            println!(
                "  speedup (median CPU / median Metal) : {:.2}×",
                cpu_med / m_med
            );
        }
    }
}
