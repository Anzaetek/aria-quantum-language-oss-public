//! Per-split certificate divergence: CPU f64 θ-contraction vs Metal f32.
//!
//! `discarded_weight` is documented as a BOUND, not an estimate, so a GPU path
//! that perturbs it is a correctness question rather than a precision note.
//! The dangerous direction is the Metal value being SMALLER than the CPU one:
//! a certificate that under-reports discarded weight silently weakens every
//! guarantee derived from it.
//!
//! Method: at every two-site split, call BOTH `cpu_path` and
//! `apply_two_site_gate_metal` on the SAME input tensors, record the two
//! discarded weights, then propagate the CPU result. The CPU drives the walk,
//! so both paths see byte-identical inputs at every step and the comparison
//! isolates the per-split divergence with no accumulated drift.
//!
//! MEASURED ON AN M4 (see `STATUS.md` §5.3): max relative difference 2.8e-8,
//! median ~1e-8, and the Metal value is smaller than the CPU value on roughly a
//! quarter of dispatched splits. The χ=128 row shows ZERO divergence on all 52
//! splits — because at 14 qubits the maximum middle-cut bond is 2^7 = 128, so
//! χ≥128 does not truncate and `cpu_path` returns exactly 0.0 every time.
//! Divergence appears only where truncation happens. That is the same signature
//! CUDA `gesvdj` shows (§16) — two vendors, two precisions, agreement when
//! exact and divergence only when truncating — which points at doing the
//! truncation DECISION in different arithmetic rather than at either kernel.

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

fn warm_entangled(num_qubits: usize, chi: usize, warm_depth: usize) -> Mps {
    let mut mps = Mps::zero_state(num_qubits, chi);
    let isq2 = 1.0 / 2.0_f64.sqrt();
    let h = [
        Complex64::new(isq2, 0.0),
        Complex64::new(isq2, 0.0),
        Complex64::new(isq2, 0.0),
        Complex64::new(-isq2, 0.0),
    ];
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

fn main() {
    let gate = cnot();
    for &(nq, chi, warm, depth) in &[
        (14usize, 32usize, 12usize, 8usize),
        (14, 64, 14, 8),
        (12, 32, 12, 8),
        (14, 128, 14, 8),
    ] {
        let mut mps = warm_entangled(nq, chi, warm);

        let mut splits = 0usize;
        let mut gpu_splits = 0usize;
        let mut cpu_zero_gpu_nonzero = 0usize;
        let mut metal_smaller = 0usize;
        let mut metal_larger = 0usize;
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        let mut rels: Vec<f64> = Vec::new();
        let mut worst: (f64, f64) = (0.0, 0.0);

        for d in 0..depth {
            let offset = d & 1;
            for q in (offset..nq - 1).step_by(2) {
                let (l, r) = (&mps.tensors[q], &mps.tensors[q + 1]);
                let (nl, nr, w_cpu) = cpu_path(l, r, &gate, chi, 1e-14);
                let (_ml, _mr, w_met, used) = apply_two_site_gate_metal(l, r, &gate, chi, 1e-14);

                splits += 1;
                if used {
                    gpu_splits += 1;
                    let d_abs = (w_met - w_cpu).abs();
                    if d_abs > max_abs {
                        max_abs = d_abs;
                        worst = (w_cpu, w_met);
                    }
                    if w_cpu > 0.0 {
                        let rel = d_abs / w_cpu;
                        rels.push(rel);
                        if rel > max_rel {
                            max_rel = rel;
                        }
                    } else if w_met > 0.0 {
                        cpu_zero_gpu_nonzero += 1;
                    }
                    if w_met < w_cpu {
                        metal_smaller += 1;
                    } else if w_met > w_cpu {
                        metal_larger += 1;
                    }
                }

                // Propagate the CPU result so both paths keep seeing identical
                // inputs. Without this the two walks diverge and the numbers
                // below would be accumulated drift, not per-split difference.
                mps.tensors[q] = nl;
                mps.tensors[q + 1] = nr;
            }
        }

        // A10: the GPU path must have run, or every count below is zero for the
        // uninteresting reason and the row looks like a clean bill of health.
        assert!(
            gpu_splits > 0,
            "{nq}q chi={chi} warm={warm}: no split dispatched to the GPU, so this row \
             would report zero divergence because nothing was compared."
        );

        rels.sort_by(|a, b| a.partial_cmp(b).unwrap());

        println!("\n=== {nq}q chi={chi} warm={warm} depth={depth} ===");
        println!(
            "  splits {splits}, of which on GPU: {gpu_splits} ({:.0}%)",
            100.0 * gpu_splits as f64 / splits as f64
        );
        println!(
            "  metal discarded_weight SMALLER than cpu : {metal_smaller}   (the dangerous direction)"
        );
        println!("  metal LARGER                            : {metal_larger}");
        println!("  cpu exactly 0 but metal > 0             : {cpu_zero_gpu_nonzero}");
        println!(
            "  max |abs diff| : {max_abs:.6e}   at cpu={:.6e} metal={:.6e}",
            worst.0, worst.1
        );
        if rels.is_empty() {
            // Not a bug: no split truncated, so there is no relative difference
            // to take. At 14 qubits that is every chi >= 128.
            println!("  max  rel diff  : n/a (no split truncated — chi is exact here)");
            println!("  median rel diff: n/a (no split truncated — chi is exact here)");
        } else {
            println!("  max  rel diff  : {max_rel:.6e}");
            println!("  median rel diff: {:.6e}", rels[rels.len() / 2]);
        }
    }
}
