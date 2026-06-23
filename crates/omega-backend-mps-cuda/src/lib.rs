#![allow(clippy::needless_range_loop)]
//! CUDA-accelerated truncated SVD for MPS bond compression via cuSOLVER `gesvdj`.
//!
//! # Why CUDA can do what Metal can't
//!
//! See `NEXT_SESSION_PLAN.md` 1a. CUDA has three properties Apple GPUs
//! lack — native f64, sub-microsecond per-op dispatch overhead, and
//! CUDA Graphs amortisation — that change which SVD algorithms fit
//! the hardware. The Metal arm (sibling crate
//! `omega-backend-mps-metal`) had to defer SVD-on-GPU entirely; the
//! CUDA arm runs the SVD on-GPU end-to-end via `cusolverDnZgesvdj`
//! (Jacobi SVD, f64-complex throughout).
//!
//! # API
//!
//! [`cuda_truncated_svd`] is the single entry point. Signature mirrors
//! the CPU `truncated_svd` so callers can swap on a per-call basis
//! (e.g. dispatch to GPU only above some bond-dim threshold).
//!
//! # Acceptance gate
//!
//! ≥ 3× CPU at 14q × depth-12 × χ=128 on the Linux+NVIDIA host
//! (RTX PRO 6000 Blackwell). Higher than the Metal arm because the
//! SVD half runs on-GPU. CPU baseline rebenches on the CUDA host
//! first for an apples-to-apples comparison.
//!
//! # Build modes
//!
//! - Default features → entry point routes to the CPU Jacobi SVD.
//!   Mirrors `omega-backend-mps-metal` so the workspace builds on
//!   every platform with `cargo build --workspace`.
//! - `--features cuda` on Linux/Windows → calls cuSOLVER `Zgesvdj`.
//! - `--features cuda` on macOS → still compiles (cudarc is
//!   target-gated). The entry point falls back to CPU.
//! - `--features cuda` on Linux without a CUDA driver → compiles, but
//!   the context init returns `None` at runtime and we fall back.

use num_complex::Complex64;

pub use omega_backend_mps::mps::Mps;
pub use omega_backend_mps::svd::SvdResult;

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
mod gesvdj;

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
pub use gesvdj::CudaSvdContext;

/// Apply a two-qubit gate at site `q` of `mps`, routing the bond
/// truncation SVD through the GPU when available. When the `cuda`
/// feature is off, the host isn't Linux/Windows, or the CUDA driver
/// is unavailable, falls back to the CPU Jacobi SVD.
///
/// Returns `true` iff at least the SVD inside this call ran on the
/// GPU. Useful for `X of N truncations dispatched to GPU` telemetry.
///
/// For the hot path (many two-site gates), hold a [`CudaSvdContext`]
/// and call [`CudaSvdContext::apply_2q`] directly — that path
/// amortises the cuSOLVER handle init across the whole circuit.
pub fn apply_2q_cuda(mps: &mut Mps, q: usize, gate: &[Complex64; 16]) -> bool {
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    {
        if let Some(ctx) = gesvdj::CudaSvdContext::new() {
            ctx.apply_2q(mps, q, gate);
            return true;
        }
    }
    mps.apply_2q(q, gate);
    false
}

/// CUDA-accelerated truncated SVD. Falls back to the CPU Jacobi SVD
/// when the `cuda` feature is off, the host isn't Linux/Windows, or
/// the CUDA driver is unavailable. Signature-compatible with
/// `omega_backend_mps::svd::truncated_svd`.
///
/// Returns `(SvdResult, used_cuda)` where `used_cuda` is `true` iff
/// the GPU path actually ran — useful for QML telemetry that reports
/// "X of N truncations dispatched to GPU". CPU fallback returns
/// `used_cuda = false`.
pub fn cuda_truncated_svd(
    matrix: &[Vec<Complex64>],
    max_rank: usize,
    threshold: f64,
) -> (SvdResult, bool) {
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    {
        if let Some(result) = gesvdj::try_cuda_svd(matrix, max_rank, threshold) {
            return (result, true);
        }
    }
    let result = omega_backend_mps::svd::truncated_svd(matrix, max_rank, threshold);
    (result, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity(n: usize) -> Vec<Vec<Complex64>> {
        let mut m = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = Complex64::new(1.0, 0.0);
        }
        m
    }

    #[test]
    fn cuda_truncated_svd_falls_back_to_cpu_on_no_feature() {
        // Default features → no cuda → CPU fallback.
        let m = make_identity(4);
        let (result, used_cuda) = cuda_truncated_svd(&m, 4, 1e-10);
        #[cfg(not(all(any(target_os = "linux", target_os = "windows"), feature = "cuda")))]
        assert!(!used_cuda, "default features must take the CPU fallback");
        // Either path: result must agree with reference.
        let _ = used_cuda;
        assert_eq!(result.s.len(), 4, "identity has 4 unit singular values");
        for sv in &result.s {
            assert!(
                (sv - 1.0).abs() < 1e-8,
                "identity singular value should be 1.0, got {sv}"
            );
        }
    }

    #[test]
    fn cuda_truncated_svd_respects_max_rank() {
        let m = make_identity(4);
        let (result, _) = cuda_truncated_svd(&m, 2, 1e-10);
        assert_eq!(result.s.len(), 2, "max_rank=2 must cap the rank");
    }

    #[test]
    fn cuda_truncated_svd_reconstructs_general_2x2() {
        let m = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)],
            vec![Complex64::new(3.0, -1.0), Complex64::new(4.0, 0.0)],
        ];
        let (r, _) = cuda_truncated_svd(&m, 2, 1e-12);
        assert_recon(&m, &r, 1e-6);
    }

    #[test]
    fn cuda_truncated_svd_reconstructs_tall_4x3() {
        let m = vec![
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(-1.0, 0.5),
            ],
            vec![
                Complex64::new(0.0, 1.0),
                Complex64::new(3.0, -1.0),
                Complex64::new(2.0, 0.0),
            ],
            vec![
                Complex64::new(-2.0, 0.5),
                Complex64::new(1.0, 0.0),
                Complex64::new(4.0, -0.5),
            ],
            vec![
                Complex64::new(1.5, -1.0),
                Complex64::new(-0.5, 2.0),
                Complex64::new(0.5, 0.5),
            ],
        ];
        let (r, _) = cuda_truncated_svd(&m, 3, 1e-12);
        assert_recon(&m, &r, 1e-6);
    }

    #[test]
    fn cuda_truncated_svd_reconstructs_wide_2x5() {
        // Exercises the m < n → transpose path.
        let m = vec![
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(-1.0, 0.5),
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.5, 1.0),
            ],
            vec![
                Complex64::new(0.0, 1.0),
                Complex64::new(3.0, -1.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 0.5),
                Complex64::new(0.5, -0.5),
            ],
        ];
        let (r, _) = cuda_truncated_svd(&m, 5, 1e-12);
        assert_recon(&m, &r, 1e-6);
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn apply_2q_cuda_matches_cpu_on_bell_state() {
        if cudarc::driver::CudaContext::new(0).is_err() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        let isq2 = 1.0 / 2.0_f64.sqrt();
        let h = [
            Complex64::new(isq2, 0.0),
            Complex64::new(isq2, 0.0),
            Complex64::new(isq2, 0.0),
            Complex64::new(-isq2, 0.0),
        ];
        let o = Complex64::new(0.0, 0.0);
        let i = Complex64::new(1.0, 0.0);
        let cx = [i, o, o, o, o, i, o, o, o, o, o, i, o, o, i, o];

        let mut mps_gpu = Mps::zero_state(2, 16);
        mps_gpu.apply_1q(0, &h);
        let used_cuda = apply_2q_cuda(&mut mps_gpu, 0, &cx);
        assert!(used_cuda, "expected GPU dispatch on this host");

        let sv = mps_gpu.to_statevector();
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (sv[0].re - expected).abs() < 1e-8,
            "|00>: {} vs {expected}",
            sv[0]
        );
        assert!(sv[1].norm() < 1e-8, "|01>: {}", sv[1]);
        assert!(sv[2].norm() < 1e-8, "|10>: {}", sv[2]);
        assert!(
            (sv[3].re - expected).abs() < 1e-8,
            "|11>: {} vs {expected}",
            sv[3]
        );
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn cuda_truncated_svd_uses_gpu_when_available() {
        // Under --features cuda on Linux/Windows with a driver present
        // (the gate the CI host runs under), the GPU path must actually
        // be taken — otherwise the fallback is silently masking a
        // breakage in the cuSOLVER call.
        if cudarc::driver::CudaContext::new(0).is_err() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        let m = make_identity(8);
        let (result, used_cuda) = cuda_truncated_svd(&m, 8, 1e-10);
        assert!(used_cuda, "cuda feature on + driver present → GPU path");
        assert_eq!(result.s.len(), 8);
    }

    fn assert_recon(m: &[Vec<Complex64>], r: &SvdResult, tol: f64) {
        let rows = m.len();
        let cols = m[0].len();
        let mut recon = vec![vec![Complex64::new(0.0, 0.0); cols]; rows];
        for k in 0..r.s.len() {
            for i in 0..rows {
                for j in 0..cols {
                    recon[i][j] += r.u[i][k] * r.s[k] * r.vt[k][j];
                }
            }
        }
        for i in 0..rows {
            for j in 0..cols {
                let err = (m[i][j] - recon[i][j]).norm();
                assert!(err < tol, "recon error at [{i},{j}]: {err}");
            }
        }
    }
}
