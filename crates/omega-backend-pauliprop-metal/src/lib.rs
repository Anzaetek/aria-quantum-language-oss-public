//! Apple Metal accelerator for the Pauli-propagation **branch** step.
//!
//! The non-Clifford branch `P → cosθ·P + i sinθ·(R·P)` is the hot loop of
//! `omega-backend-pauliprop`: for every term in the (potentially millions-strong)
//! Pauli sum it runs an O(n) symplectic anticommute test and, for anticommuting
//! terms, produces a second child. That per-term expansion is embarrassingly
//! parallel, so we offload it to the GPU — but only the *integer* half.
//!
//! # Why the split, and why it's exact
//!
//! The CUDA sibling (`omega-backend-pauliprop-cuda`) runs the whole branch,
//! coefficients included, on device in f64. Apple GPUs have no native double,
//! so a straight port would compute `cosθ·coeff` etc. in f32 and blow the
//! `≤1e-9` parity bar. Instead this crate offloads only the symplectic work —
//! the anticommute parity, the sign parity, and the child key `R·P`, all pure
//! `u64`/`popcount` integer ops — and does every floating-point operation on
//! the CPU in f64. The GPU never touches a coefficient, so the result is
//! bit-for-bit the CPU branch (same f64 arithmetic, same merge), not an
//! approximation of it.
//!
//! [`metal_branch`] has the exact signature of
//! `omega_backend_pauliprop::BranchHook`, so it drops into
//! `PauliPropBackend::with_branch_hook`. It returns `false` (leaving the sum
//! untouched, so the caller runs the identical CPU path) when the `metal`
//! feature is off, the host isn't macOS, no Metal device is present, or the
//! term count is below the GPU-worthwhile threshold — the same
//! transparent-fallback contract as the CUDA arm.

use num_complex::Complex64;
use omega_backend_pauliprop::PauliSum;

/// Term count below which a device round-trip isn't worth it — small sums stay
/// on the CPU. Overridable at runtime via `PAULIPROP_GPU_MIN` (mainly for tests
/// that want to force the GPU path on a small sum). Mirrors the CUDA arm so the
/// two accelerators share the same knob.
const DEFAULT_MIN_TERMS: usize = 256;

#[cfg(all(target_os = "macos", feature = "metal"))]
mod gpu;

/// Number of branch steps actually executed on the GPU this process (telemetry
/// for tests / "X of N branches dispatched to GPU" reporting).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_branch_count() -> u64 {
    gpu::GPU_BRANCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Telemetry stub on non-Metal builds: the GPU path never runs.
#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn gpu_branch_count() -> u64 {
    0
}

fn min_terms() -> usize {
    std::env::var("PAULIPROP_GPU_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MIN_TERMS)
}

/// GPU branch hook (see module docs). Signature matches
/// `omega_backend_pauliprop::BranchHook`; install with
/// `PauliPropBackend::with_branch_hook(metal_branch)`.
#[allow(clippy::too_many_arguments)]
pub fn metal_branch(
    sum: &mut PauliSum,
    rx: &[u64],
    rz: &[u64],
    factor: Complex64,
    cos: f64,
    sin: f64,
    max_freq: Option<u32>,
    n: usize,
) -> bool {
    if sum.terms.len() < min_terms() {
        return false; // too small — let the CPU path handle it
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        gpu::branch_on_gpu(sum, rx, rz, factor, cos, sin, max_freq, n)
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = (rx, rz, factor, cos, sin, max_freq, n);
        false
    }
}
