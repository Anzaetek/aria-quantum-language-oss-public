//! Metal-accelerated truncated SVD for MPS bond compression.
//!
//! # Status (2026-05-13): **deferred placeholder**
//!
//! The investigation commit `f9b477e investigation(mps): defer
//! MPS-SVD-on-Metal — bench shape + algorithm fit` (2026-05-12) ruled
//! the planned Metal compute kernel out for two independent reasons,
//! both empirical:
//!
//! 1. **Bench shape doesn't exercise SVD.** The cited target
//!    (14q × depth-4 × χ=128) shows flat wallclock from χ ∈ \{4..256\}
//!    — bonds never saturate, so a perfect Metal port would not move
//!    the bench number. See `crates/omega-backend-mps/examples/mps_chi_profile.rs`.
//!
//! 2. **Algorithm fit on Apple GPUs.** Where SVD does dominate
//!    (depth ≥ 12), parallel Jacobi rotations need ~12 700
//!    sub-sweep dispatches per SVD at n=256. Apple's ~100 µs
//!    per-dispatch overhead alone is ~1.3 s vs ~30 ms CPU.
//!    Apple GPUs also lack native f64, so the 1e-10 forward
//!    tolerance the test contract demands is unreachable in
//!    single-pass f32 Jacobi.
//!
//! Re-open triggers (TODO.md): batched-SVD via MLX or Accelerate
//! (fits QML gradient sweeps, not single-execute MPS); randomized
//! SVD or block Lanczos (GPU-friendly matmul-shaped but breaks the
//! bit-for-bit Jacobi reference); re-scoping the GPU-MPS effort
//! entirely to the θ-contraction half (`mps.rs:100-128`), which IS
//! GPU-friendly.
//!
//! # What this crate ships today
//!
//! - The signature-compatible host-side entry point
//!   [`metal_truncated_svd`]. Existing call sites in
//!   `omega-backend-mps::mps` could swap to it without code changes
//!   if/when a real kernel lands.
//! - A `(SvdResult, used_metal)` return shape so QML telemetry can
//!   report "X of N truncations dispatched to GPU"; today
//!   `used_metal` is always `false` because the body always falls
//!   back to `omega-backend-mps::svd::truncated_svd`.
//! - A `cfg(feature = "metal")` MTLDevice probe — preserved so
//!   `cargo build --features metal` exercises the same dispatch
//!   machinery the eventual kernel would need, catching link / cfg
//!   regressions early.
//! - No Metal Shading Language source. Adding `src/*.metal` and an
//!   MTLLibrary load would be premature given the deferral; the
//!   re-open trigger commit lands the kernel and the build glue
//!   together.
//!
//! # Why a separate crate (not a feature on `omega-backend-mps`)
//!
//! Even as a placeholder, the split is load-bearing: the `metal` crate
//! dep is target-gated to macOS, and on Linux runners
//! `cargo build --workspace --features metal` would otherwise thread
//! cfg gates through the MPS CPU code. Mirroring the
//! `omega-backend-statevector` / `omega-backend-statevector-metal`
//! split keeps the CPU MPS crate free of Metal cfg noise and reserves
//! the spot a re-opened effort would slot into.
//!
//! # API
//!
//! [`metal_truncated_svd`] is the single entry point. Signature mirrors
//! the CPU `truncated_svd` so a future implementation can swap behind
//! a feature flag with zero code change at the call site.

use std::sync::atomic::{AtomicU64, Ordering};

use num_complex::Complex64;
use omega_backend_mps::mps::MpsTensor;

pub use omega_backend_mps::svd::SvdResult;

pub mod contract;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal_runtime;

pub use contract::{
    apply_two_site_gate_metal, min_bond_dim_for_metal, mps_apply_2q_metal, MIN_BOND_DIM_FOR_METAL,
};

/// Number of two-site gates whose θ-contraction actually ran on the Metal GPU
/// this process (telemetry, mirrors the pauliprop/CUDA arms' branch counters).
pub static METAL_CONTRACTIONS: AtomicU64 = AtomicU64::new(0);

/// Read [`METAL_CONTRACTIONS`] — "X of N two-qubit gates dispatched to the GPU".
pub fn metal_contraction_count() -> u64 {
    METAL_CONTRACTIONS.load(Ordering::Relaxed)
}

/// Two-site-gate accelerator in the exact calling convention of
/// `omega_backend_mps::Contract2qFn` — install with
/// `MpsBackend::with_contract_fn(metal_contract_2q)`. Returns `Some` only when
/// the Metal θ-contraction actually ran (device present and bond ≥
/// [`MIN_BOND_DIM_FOR_METAL`]); `None` tells the MPS backend to use its own
/// built-in exact-f64 path, so wiring this under a `metal` build never changes
/// results below the GPU threshold. Above it, the contraction runs in f32 (Apple
/// GPUs have no native f64) — the same precision stance as the Metal
/// statevector backend.
pub fn metal_contract_2q(
    left: &MpsTensor,
    right: &MpsTensor,
    gate: &[Complex64; 16],
    max_bond_dim: usize,
    threshold: f64,
) -> Option<(MpsTensor, MpsTensor, f64)> {
    // Below the GPU threshold, decline immediately so the MPS backend takes its
    // own exact-f64 path — no wasted CPU recompute inside the metal helper.
    if left.bond_right < min_bond_dim_for_metal() {
        return None;
    }
    let (nl, nr, rel_discarded, used) =
        apply_two_site_gate_metal(left, right, gate, max_bond_dim, threshold);
    if used {
        METAL_CONTRACTIONS.fetch_add(1, Ordering::Relaxed);
        Some((nl, nr, rel_discarded))
    } else {
        None
    }
}

/// Metal-accelerated truncated SVD. Falls back to the CPU Jacobi SVD
/// when the `metal` feature is off, the host isn't macOS, or the Metal
/// device is unavailable. Signature-compatible with
/// `omega_backend_mps::svd::truncated_svd` so callers can swap on a
/// per-call basis (e.g. dispatch to GPU only above some bond-dim
/// threshold).
///
/// Returns `(SvdResult, used_metal)` where `used_metal` is `true` iff
/// the GPU path actually ran — useful for the QML telemetry that
/// reports "X of N truncations dispatched to GPU". The CPU fallback
/// returns `used_metal = false`.
pub fn metal_truncated_svd(
    matrix: &[Vec<Complex64>],
    max_rank: usize,
    threshold: f64,
) -> (SvdResult, bool) {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Some(result) = try_metal_svd(matrix, max_rank, threshold) {
            return (result, true);
        }
    }
    let result = omega_backend_mps::svd::truncated_svd(matrix, max_rank, threshold);
    (result, false)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn try_metal_svd(
    _matrix: &[Vec<Complex64>],
    _max_rank: usize,
    _threshold: f64,
) -> Option<SvdResult> {
    // Deferred — see crate-level rustdoc. The MTLDevice probe stays
    // wired so `cargo build --features metal` keeps exercising the
    // metal crate's handshake (catches "metal feature on but device
    // unavailable" at build time). When the deferral is reopened —
    // batched SVD via MLX / Accelerate, randomized SVD, or block
    // Lanczos — drop the kernel dispatch in below the probe and have
    // it return `Some(SvdResult)` on success.
    let device = metal::Device::system_default()?;
    let _ = device.name();
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_4x4() -> Vec<Vec<Complex64>> {
        let mut m = vec![vec![Complex64::new(0.0, 0.0); 4]; 4];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = Complex64::new(1.0, 0.0);
        }
        m
    }

    #[test]
    fn metal_truncated_svd_falls_back_to_cpu_on_no_feature() {
        // Default features → no metal → CPU fallback.
        let m = make_identity_4x4();
        let (result, used_metal) = metal_truncated_svd(&m, 4, 1e-10);
        assert!(!used_metal, "default features must take the CPU fallback");
        assert_eq!(result.s.len(), 4, "identity has 4 unit singular values");
        for sv in &result.s {
            assert!(
                (sv - 1.0).abs() < 1e-10,
                "identity singular value should be 1.0, got {sv}"
            );
        }
    }

    #[test]
    fn metal_truncated_svd_respects_max_rank() {
        let m = make_identity_4x4();
        let (result, _) = metal_truncated_svd(&m, 2, 1e-10);
        assert_eq!(result.s.len(), 2, "max_rank=2 must cap the rank");
    }
}
