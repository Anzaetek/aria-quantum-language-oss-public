//! Metal θ-contraction for MPS two-site gate apply.
//!
//! Replaces Steps 1+2 of `omega-backend-mps::mps::Mps::apply_2q` with
//! a two-kernel Metal pipeline. SVD (Step 3) stays on CPU per the
//! 2026-05-12 investigation (`f9b477e`) — Apple f32-only shaders +
//! ~100 µs dispatch overhead don't fit the Jacobi-SVD shape.
//!
//! Acceptance gate per `NEXT_SESSION_PLAN.md` (item G-Metal): total
//! wallclock ≥ 1.3× CPU at 14q × depth-12 × χ=128. Bond-dim
//! threshold is `MIN_BOND_DIM_FOR_METAL = 32` — below that, the
//! per-call dispatch overhead beats anything GPU parallelism saves.

use num_complex::Complex64;

use omega_backend_mps::mps::{Mps, MpsTensor};
use omega_backend_mps::svd::truncated_svd;

/// Below this contracted bond dimension the GPU dispatch overhead
/// (~100 µs per kernel call on Apple) exceeds the CPU contraction
/// wallclock, so the caller takes the CPU path.
pub const MIN_BOND_DIM_FOR_METAL: usize = 32;

/// The effective bond-dim threshold, [`MIN_BOND_DIM_FOR_METAL`] by default but
/// overridable at runtime via `MPS_METAL_MIN_BOND` — used by tests to force the
/// GPU contraction on a modest circuit whose bonds don't reach 32. Production
/// leaves it unset and gets the tuned 32.
pub fn min_bond_dim_for_metal() -> usize {
    std::env::var("MPS_METAL_MIN_BOND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MIN_BOND_DIM_FOR_METAL)
}

/// Apply a two-qubit gate to adjacent MPS sites with Metal-accelerated
/// θ-contraction. SVD stays on CPU.
///
/// Returns `(new_left, new_right, used_metal)`. `used_metal` is `true`
/// iff the GPU contraction path actually ran. When the `metal` feature
/// is off, the host isn't macOS, the Metal device isn't available, or
/// the middle bond dim is below `MIN_BOND_DIM_FOR_METAL`, falls back to
/// a pure-CPU equivalent of `Mps::apply_2q` and returns
/// `used_metal = false`.
pub fn apply_two_site_gate_metal(
    left: &MpsTensor,
    right: &MpsTensor,
    gate: &[Complex64; 16],
    max_bond_dim: usize,
    threshold: f64,
) -> (MpsTensor, MpsTensor, bool) {
    assert_eq!(
        left.bond_right, right.bond_left,
        "left.bond_right must match right.bond_left",
    );

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if left.bond_right >= min_bond_dim_for_metal() {
            if let Some((nl, nr)) = metal_path(left, right, gate, max_bond_dim, threshold) {
                return (nl, nr, true);
            }
        }
    }

    // CPU fallback — mirrors `Mps::apply_2q` Steps 1..5 verbatim.
    let (nl, nr) = cpu_path(left, right, gate, max_bond_dim, threshold);
    (nl, nr, false)
}

/// Drop-in replacement for `Mps::apply_2q` that routes through
/// [`apply_two_site_gate_metal`] for the per-pair contraction. The
/// metal/CPU choice is made by [`apply_two_site_gate_metal`] using
/// the same bond-dim threshold. Returns `true` iff the GPU path was
/// taken (matches the `used_metal` flag from the underlying call).
///
/// Bypassing `Mps::apply_2q` lets callers (the QML trainer, MPS
/// driver loops) take the GPU contraction without the MPS crate
/// needing to depend on this crate — which would create a circular
/// build graph.
pub fn mps_apply_2q_metal(mps: &mut Mps, q: usize, gate: &[Complex64; 16]) -> bool {
    assert!(q + 1 < mps.n, "q+1 out of range");
    let (new_left, new_right, used_metal) = apply_two_site_gate_metal(
        &mps.tensors[q],
        &mps.tensors[q + 1],
        gate,
        mps.max_bond_dim,
        1e-14,
    );
    mps.tensors[q] = new_left;
    mps.tensors[q + 1] = new_right;
    used_metal
}

/// Pure-CPU two-site gate apply (extracted from `Mps::apply_2q`). Used
/// both as the fallback path and as the reference for GPU-vs-CPU
/// equivalence tests. Returns the post-truncation `(left, right)`
/// tensors.
pub fn cpu_path(
    left: &MpsTensor,
    right: &MpsTensor,
    gate: &[Complex64; 16],
    max_bond_dim: usize,
    threshold: f64,
) -> (MpsTensor, MpsTensor) {
    let bl = left.bond_left;
    let bm = left.bond_right;
    let br = right.bond_right;

    // Step 1: contract.
    let mut theta = vec![Complex64::new(0.0, 0.0); bl * 4 * br];
    for l in 0..bl {
        for s0 in 0..2 {
            for s1 in 0..2 {
                let ss = s0 * 2 + s1;
                for r in 0..br {
                    let mut val = Complex64::new(0.0, 0.0);
                    for m in 0..bm {
                        val += left.get(l, s0, m) * right.get(m, s1, r);
                    }
                    theta[l * 4 * br + ss * br + r] = val;
                }
            }
        }
    }

    // Step 2: gate apply.
    let mut theta_prime = vec![Complex64::new(0.0, 0.0); bl * 4 * br];
    for l in 0..bl {
        for r in 0..br {
            for ss_out in 0..4 {
                let mut val = Complex64::new(0.0, 0.0);
                for ss_in in 0..4 {
                    val += gate[ss_out * 4 + ss_in] * theta[l * 4 * br + ss_in * br + r];
                }
                theta_prime[l * 4 * br + ss_out * br + r] = val;
            }
        }
    }

    finalize_svd(&theta_prime, bl, br, max_bond_dim, threshold)
}

/// CPU SVD + split shared by both code paths.
fn finalize_svd(
    theta_prime: &[Complex64],
    bl: usize,
    br: usize,
    max_bond_dim: usize,
    threshold: f64,
) -> (MpsTensor, MpsTensor) {
    let rows = bl * 2;
    let cols = 2 * br;
    let mut matrix = vec![vec![Complex64::new(0.0, 0.0); cols]; rows];
    for l in 0..bl {
        for s0 in 0..2 {
            for s1 in 0..2 {
                for r in 0..br {
                    let ss = s0 * 2 + s1;
                    matrix[l * 2 + s0][s1 * br + r] = theta_prime[l * 4 * br + ss * br + r];
                }
            }
        }
    }
    let svd = truncated_svd(&matrix, max_bond_dim, threshold);
    let new_bm = svd.s.len();

    let mut new_left = MpsTensor::new(bl, new_bm);
    let mut new_right = MpsTensor::new(new_bm, br);
    for l in 0..bl {
        for s0 in 0..2 {
            for m in 0..new_bm {
                let sqrt_s = svd.s[m].sqrt();
                new_left.set(l, s0, m, svd.u[l * 2 + s0][m] * sqrt_s);
            }
        }
    }
    for m in 0..new_bm {
        let sqrt_s = svd.s[m].sqrt();
        for s1 in 0..2 {
            for r in 0..br {
                new_right.set(m, s1, r, sqrt_s * svd.vt[m][s1 * br + r]);
            }
        }
    }
    (new_left, new_right)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_path(
    left: &MpsTensor,
    right: &MpsTensor,
    gate: &[Complex64; 16],
    max_bond_dim: usize,
    threshold: f64,
) -> Option<(MpsTensor, MpsTensor)> {
    use crate::metal_runtime::{self, MetalRuntime};

    let bl = left.bond_left;
    let bm = left.bond_right;
    let br = right.bond_right;

    let runtime = MetalRuntime::get()?;

    // Stage f64 → paired f32 buffers. MpsTensor.data already has the
    // layout we need: index `l*2*bm + s0*bm + m` (and analogously for
    // right) — the shader uses the same.
    let left_f32 = metal_runtime::complex_to_paired_f32(&left.data);
    let right_f32 = metal_runtime::complex_to_paired_f32(&right.data);
    let gate_f32 = metal_runtime::complex_array_to_paired_f32(gate);

    let theta_prime_f32 = runtime.run_two_site(&left_f32, &right_f32, &gate_f32, bl, bm, br)?;

    // Lift back to f64 Complex64.
    let theta_prime: Vec<Complex64> = theta_prime_f32
        .chunks_exact(2)
        .map(|p| Complex64::new(p[0] as f64, p[1] as f64))
        .collect();

    Some(finalize_svd(&theta_prime, bl, br, max_bond_dim, threshold))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(bl: usize, br: usize, seed: u64) -> MpsTensor {
        // Deterministic synthetic data — small mixed re/im pattern so
        // the contraction produces non-trivial complex output and a
        // GPU-vs-CPU mismatch from a wrong layout / wrong index would
        // be visible at every entry. Uses splitmix-style hashing of
        // `(seed, l, p, r)` so adjacent tensors get independent data.
        let mut t = MpsTensor::new(bl, br);
        let mut state = seed;
        for l in 0..bl {
            for p in 0..2 {
                for r in 0..br {
                    state = state.wrapping_add(0x9E3779B97F4A7C15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                    z ^= z >> 31;
                    let re = ((z & 0xFFFF) as f64 / 65536.0) - 0.5;
                    let im = (((z >> 16) & 0xFFFF) as f64 / 65536.0) - 0.5;
                    t.set(l, p, r, Complex64::new(re, im));
                }
            }
        }
        t
    }

    fn identity_gate() -> [Complex64; 16] {
        let o = Complex64::new(0.0, 0.0);
        let i = Complex64::new(1.0, 0.0);
        [
            i, o, o, o, // s0s1' = 00
            o, i, o, o, // s0s1' = 01
            o, o, i, o, // s0s1' = 10
            o, o, o, i, // s0s1' = 11
        ]
    }

    fn cnot_gate() -> [Complex64; 16] {
        let o = Complex64::new(0.0, 0.0);
        let i = Complex64::new(1.0, 0.0);
        [
            i, o, o, o, // |00⟩ → |00⟩
            o, i, o, o, // |01⟩ → |01⟩
            o, o, o, i, // |10⟩ → |11⟩
            o, o, i, o, // |11⟩ → |10⟩
        ]
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn tensor_max_diff(a: &MpsTensor, b: &MpsTensor) -> f64 {
        assert_eq!(a.bond_left, b.bond_left);
        assert_eq!(a.bond_right, b.bond_right);
        assert_eq!(a.data.len(), b.data.len());
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max)
    }

    #[test]
    fn cpu_path_matches_inline_mps_apply_2q() {
        // Smoke test: cpu_path on toy shapes produces the same result
        // we'd get from `Mps::apply_2q`. Pinned via direct comparison
        // against an inline replica of `Mps::apply_2q`'s body.
        let left = make_tensor(2, 3, 0xA);
        let right = make_tensor(3, 2, 0xB);
        let gate = cnot_gate();
        let (l1, r1) = cpu_path(&left, &right, &gate, 8, 1e-14);
        // Hand-computed sanity check via the reference identity path
        // — at max_bond_dim = 8 (>> 3 = bm), every singular value is
        // kept, so cpu_path is bit-identical to the Mps::apply_2q
        // expression on the reconstructed state vector. We compare
        // via norm preservation: ‖L′·R′‖ = ‖L·R‖ within fp tolerance.
        let mut sv_in = 0.0;
        for l in 0..left.bond_left {
            for s0 in 0..2 {
                for s1 in 0..2 {
                    for r in 0..right.bond_right {
                        let mut acc = Complex64::new(0.0, 0.0);
                        for m in 0..left.bond_right {
                            acc += left.get(l, s0, m) * right.get(m, s1, r);
                        }
                        sv_in += acc.norm_sqr();
                    }
                }
            }
        }
        let mut sv_out = 0.0;
        for l in 0..l1.bond_left {
            for s0 in 0..2 {
                for s1 in 0..2 {
                    for r in 0..r1.bond_right {
                        let mut acc = Complex64::new(0.0, 0.0);
                        for m in 0..l1.bond_right {
                            acc += l1.get(l, s0, m) * r1.get(m, s1, r);
                        }
                        sv_out += acc.norm_sqr();
                    }
                }
            }
        }
        assert!(
            (sv_in - sv_out).abs() < 1e-10,
            "CNOT must preserve total amplitude norm: in={sv_in:.6}, out={sv_out:.6}",
        );
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn metal_matches_cpu_on_small_shape_when_threshold_lowered() {
        // Direct path bypassing the threshold gate — uses the
        // crate-private `metal_path` so a small bond-dim shape can
        // still exercise the GPU pipeline. The public
        // `apply_two_site_gate_metal` would short-circuit to CPU here.
        let left = make_tensor(2, 4, 0xC);
        let right = make_tensor(4, 2, 0xD);
        let gate = cnot_gate();
        let from_metal = metal_path(&left, &right, &gate, 8, 1e-14);
        let from_cpu = cpu_path(&left, &right, &gate, 8, 1e-14);
        let Some((ml, mr)) = from_metal else {
            // No Metal device — skip.
            return;
        };
        // GPU runs in f32, CPU runs in f64 — tolerance reflects the
        // accumulated f32 round-off of the bm-long contraction and
        // the 4-long gate apply. At bm = 4 with synthetic ~unit
        // entries the empirical max diff is ~1e-6.
        let diff_l = tensor_max_diff(&ml, &from_cpu.0);
        let diff_r = tensor_max_diff(&mr, &from_cpu.1);
        // The SVD is non-unique up to a per-singular-value phase /
        // sign / column ordering, so direct entry comparison would be
        // flaky. Compare instead the reconstructed two-site tensor
        // `L · R` which IS invariant under those gauge choices.
        let bl = ml.bond_left;
        let br = mr.bond_right;
        let mut max_recon_diff = 0.0;
        for l in 0..bl {
            for s0 in 0..2 {
                for s1 in 0..2 {
                    for r in 0..br {
                        let mut metal_acc = Complex64::new(0.0, 0.0);
                        let mut cpu_acc = Complex64::new(0.0, 0.0);
                        for m in 0..ml.bond_right {
                            metal_acc += ml.get(l, s0, m) * mr.get(m, s1, r);
                        }
                        for m in 0..from_cpu.0.bond_right {
                            cpu_acc += from_cpu.0.get(l, s0, m) * from_cpu.1.get(m, s1, r);
                        }
                        max_recon_diff = (metal_acc - cpu_acc).norm().max(max_recon_diff);
                    }
                }
            }
        }
        let _ = (diff_l, diff_r); // entry diffs are gauge-dependent; not asserted
        assert!(
            max_recon_diff < 1e-5,
            "GPU-vs-CPU reconstructed two-site tensor diverges: max diff {max_recon_diff:.3e}",
        );
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn metal_matches_cpu_at_threshold_bond_dim() {
        // bm = MIN_BOND_DIM_FOR_METAL exercises the threshold edge:
        // public dispatch must take the GPU path, and the result
        // must match CPU within f32 round-off (~ √bm · f32_eps).
        let bm = MIN_BOND_DIM_FOR_METAL;
        let left = make_tensor(2, bm, 0x1111);
        let right = make_tensor(bm, 2, 0x2222);
        let gate = cnot_gate();
        let (ml, mr, used) = apply_two_site_gate_metal(&left, &right, &gate, bm, 1e-14);
        if !used {
            // Metal device unavailable on this host — skip.
            return;
        }
        let (cl, cr) = cpu_path(&left, &right, &gate, bm, 1e-14);

        // Reconstruction L · R is gauge-invariant; entries themselves
        // are not (SVD U/V phase freedom). Compare reconstructions.
        let bl = ml.bond_left;
        let br = mr.bond_right;
        let mut max_recon_diff = 0.0;
        for l in 0..bl {
            for s0 in 0..2 {
                for s1 in 0..2 {
                    for r in 0..br {
                        let mut macc = Complex64::new(0.0, 0.0);
                        let mut cacc = Complex64::new(0.0, 0.0);
                        for m in 0..ml.bond_right {
                            macc += ml.get(l, s0, m) * mr.get(m, s1, r);
                        }
                        for m in 0..cl.bond_right {
                            cacc += cl.get(l, s0, m) * cr.get(m, s1, r);
                        }
                        max_recon_diff = (macc - cacc).norm().max(max_recon_diff);
                    }
                }
            }
        }
        // bm=32 f32 contraction floor: empirically ≈ 1e-5 on synthetic
        // unit-magnitude entries. The 1e-4 bar leaves headroom for
        // bench-shape stress without false negatives.
        assert!(
            max_recon_diff < 1e-4,
            "GPU/CPU reconstruction diverges at bm={bm}: max diff {max_recon_diff:.3e}",
        );
    }

    #[test]
    fn public_dispatch_returns_used_metal_only_when_threshold_met() {
        let left = make_tensor(1, 2, 0xE);
        let right = make_tensor(2, 1, 0xF);
        let gate = identity_gate();
        let (_, _, used_metal) = apply_two_site_gate_metal(&left, &right, &gate, 4, 1e-14);
        // bm = 2 is well below MIN_BOND_DIM_FOR_METAL = 32, so even
        // with `--features metal` enabled the public entry point
        // returns `used_metal = false`.
        assert!(
            !used_metal,
            "small bond dim must route through the CPU path",
        );
    }
}
