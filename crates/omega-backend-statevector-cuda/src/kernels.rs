//! NVRTC kernel library — compiles the embedded `.cu` sources to PTX
//! at backend construction time and caches one [`CudaFunction`] per
//! kernel.
//!
//! Mirrors `omega-backend-statevector-metal/src/kernels.rs`.
//!
//! The CUDA module is loaded into the device context exactly once;
//! every [`crate::imp::StateBuffer`] reuses the same set of
//! [`CudaFunction`] handles via the [`KernelLibrary`] held inside the
//! [`crate::imp::DeviceHandle`].

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::CudaError;

/// Precision prelude, prepended to every kernel source. The kernels are written
/// once in terms of `real`/`real2` and compiled per [`Precision`]; see
/// `kernels/prelude.cuh` for why that is sound (no precision-specific
/// intrinsics anywhere in the kernel math).
const PRELUDE: &str = include_str!("kernels/prelude.cuh");

/// Which floating-point precision a kernel module is compiled for.
///
/// f32 is the default and remains so: it is measurably sufficient for a forward
/// expectation and roughly 64x faster than f64 on consumer-class Blackwell
/// (GB10, compute 12.1), where f64 runs at 1/64 rate. f64 exists because f32
/// caps agreement with the f64 CPU statevector at ~5e-7, six orders looser than
/// the 1e-9 this project gates cross-checks at — so an f32-only GPU cannot enter
/// its own verification gates. On an H100 (compute 9.0, f64 at ~1/2 rate) the
/// cost of choosing f64 is small.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Precision {
    #[default]
    F32,
    F64,
}

impl Precision {
    /// NVRTC defines that select the scalar types in `prelude.cuh`.
    fn defines(self) -> &'static [&'static str] {
        match self {
            // The prelude's own defaults are f32, so nothing to override.
            Precision::F32 => &[],
            Precision::F64 => &[
                "-DOMEGA_REAL=double",
                "-DOMEGA_REAL2=double2",
                "-DOMEGA_MAKE_REAL2=make_double2",
            ],
        }
    }

    pub fn bytes_per_amplitude(self) -> usize {
        match self {
            // Interleaved (re, im).
            Precision::F32 => 8,
            Precision::F64 => 16,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F64 => "f64",
        }
    }
}

/// CUDA C source for the kernels we ship. Embedded at compile time;
/// no runtime filesystem access required.
const KERNEL_APPLY_DIAGONAL: &str = include_str!("kernels/apply_diagonal.cu");
const KERNEL_APPLY_DIAGONAL_PAULI_SUM: &str = include_str!("kernels/apply_diagonal_pauli_sum.cu");
const KERNEL_APPLY_DIAGONAL_PRODUCT: &str = include_str!("kernels/apply_diagonal_product.cu");
const KERNEL_APPLY_1Q: &str = include_str!("kernels/apply_1q.cu");
const KERNEL_APPLY_2Q: &str = include_str!("kernels/apply_2q.cu");
const KERNEL_INNER_PRODUCT: &str = include_str!("kernels/inner_product.cu");
const KERNEL_PAULI_EXPECTATION: &str = include_str!("kernels/pauli_expectation.cu");
// Pooled-param variants — same math, params come from a device
// pool indexed by slot. Used by the CUDA-graph capture+replay path
// so per-training-point gate matrices update via memcpy_htod between
// graph launches without re-recording the graph.
const KERNEL_APPLY_DIAGONAL_POOLED: &str = include_str!("kernels/apply_diagonal_pooled.cu");
const KERNEL_APPLY_1Q_POOLED: &str = include_str!("kernels/apply_1q_pooled.cu");
const KERNEL_APPLY_2Q_POOLED: &str = include_str!("kernels/apply_2q_pooled.cu");
// Shot-sampling helper: writes per-amplitude |state[i]|² so the
// host-side CDF builder pulls half the bytes (one f32 per amp vs
// two f32s for the interleaved float2 state).
const KERNEL_COMPUTE_PROBABILITIES: &str = include_str!("kernels/compute_probabilities.cu");
// On-device shot sampling pipeline. `cdf_scan.cu` does the two-pass
// inclusive scan of the per-amp `probs[]` into a CDF on device;
// `sample_from_cdf.cu` does per-shot binary search + atomicAdd into
// a dense `counts[]`. The pair eliminates the host CDF + per-shot
// loop entirely for n ≤ 20.
const KERNEL_CDF_SCAN: &str = include_str!("kernels/cdf_scan.cu");
const KERNEL_SAMPLE_FROM_CDF: &str = include_str!("kernels/sample_from_cdf.cu");
// Fused inner_product + atomic-accumulate kernel for the captured
// backward sweep. Reads chain factor + symbol slot from device-side
// pools; writes 2·Re⟨ν|temp⟩·chain into `grad_dev[sym]`. Eliminates
// the per-(op, sym) host roundtrip entirely so the entire training-pt
// loop can run inside one CudaGraph replay.
const KERNEL_INNER_PRODUCT_ACCUMULATE_POOLED: &str =
    include_str!("kernels/inner_product_accumulate_pooled.cu");
// Per-qubit ⟨Z⟩ → predictions[slot] for the captured forward
// stage. Two-stage block reduction with atomic-add into the
// destination slot. Lets the captured graph compute predictions
// in-graph; a subsequent cuLaunchHostFunc callback derives the
// gradient observable's residual coefficients.
const KERNEL_PAULI_Z_EXPECTATION_TO_SLOT: &str =
    include_str!("kernels/pauli_z_expectation_to_slot.cu");
// Stage F2: residual coefficient builder. Replaces the
// `cuLaunchHostFunc` host callback with a pure-device kernel that
// computes `coeffs[i] = 2·(predictions[i] - y_label[i])` directly
// from device-side buffers — eliminates the host-thread context
// switch and the surrounding predictions memcpy_dtoh / coeffs
// memcpy_htod nodes.
const KERNEL_COMPUTE_RESIDUAL_COEFFS: &str = include_str!("kernels/compute_residual_coeffs.cu");
// Stage G1: out-of-place pooled 1q gate. Reads `src` state, writes
// `dst = U · src` — fuses the captured backward sweep's
// `CopyPhiToTemp` + `Deriv1q` step into one kernel (~16 fewer graph
// nodes per replay on the n=14/16-param HEA shape).
const KERNEL_APPLY_1Q_FROM_TO_POOLED: &str = include_str!("kernels/apply_1q_from_to_pooled.cu");
// Stage G2: triple fusion. Computes ∂U·φ on the fly per pair-thread,
// contracts with ν, block-reduces, and atomic-adds chain·Re⟨ν|∂U·φ⟩
// into grad_dev[sym]. Skips materialising `temp` entirely — saves
// another ~16 graph nodes for the 1q-deriv path.
const KERNEL_APPLY_1Q_INNER_PRODUCT_ACCUMULATE_POOLED: &str =
    include_str!("kernels/apply_1q_inner_product_accumulate_pooled.cu");
// Stage G3: dual-state pooled 2q gate. Applies U from `pool[slot]`
// to both `state_a` (φ) and `state_b` (ν) in one launch — fuses
// `DaggerPhi2q` + `DaggerNu2q` for non-parameterised 2q ops on the
// captured backward sweep (~13 fewer graph nodes on the
// n=14/16-param HEA).
const KERNEL_APPLY_2Q_POOLED_DUAL: &str = include_str!("kernels/apply_2q_pooled_dual.cu");
// Stage G4: dual-state pooled 1q gate. Same fuse pattern as G3
// but for non-parameterised 1q ops (H, X, etc. — ~14 more nodes on
// the n=14/16-param HEA).
const KERNEL_APPLY_1Q_POOLED_DUAL: &str = include_str!("kernels/apply_1q_pooled_dual.cu");
// Stage G5: chain of K consecutive pooled-diagonal gates folded
// into one kernel. Replaces the captured forward sweep's run of
// K back-to-back diagonal ops with a single launch (K-1 fewer
// graph nodes — 7 on the Phase 4c HEA Layer 2's 8-RZ chain).
const KERNEL_APPLY_DIAGONAL_CHAIN_POOLED: &str =
    include_str!("kernels/apply_diagonal_chain_pooled.cu");
// Stage G6: commutation-based RZ-chain restructuring. Two new
// kernels — `pauli_z_chain_accumulate` collapses K Pauli-Z gradient
// accumulations into one launch (using the identity
// gradient[θ_k] = chain·Im⟨ν_n|Z_{q_k}|φ_n⟩ valid when Z_{q_k}
// commutes with all later daggers in the chain), and
// `apply_diagonal_chain_dual_pooled` applies the K daggers to both
// φ and ν in one launch. Replaces 3K backward nodes with 2.
const KERNEL_PAULI_Z_CHAIN_ACCUMULATE: &str = include_str!("kernels/pauli_z_chain_accumulate.cu");
const KERNEL_APPLY_DIAGONAL_CHAIN_DUAL_POOLED: &str =
    include_str!("kernels/apply_diagonal_chain_dual_pooled.cu");
// Stage G7: per-op Pauli-Y triple-fusion (DaggerPhi + Im⟨ν|Y|φ⟩
// accumulate + DaggerNu). Used for parameterised RY ops whose
// generator is Y. 3 graph nodes per RY → 1 (8 RY ansatz on Phase
// 4c → -16 nodes).
const KERNEL_PAULI_Y_ACCUMULATE_THEN_DAGGER_BOTH_POOLED: &str =
    include_str!("kernels/pauli_y_accumulate_then_dagger_both_pooled.cu");

/// Virtual arch target for the embedded NVRTC compile, detected from the
/// live device's compute capability (`compute_{major}{minor}`).
///
/// A fixed floor no longer works: CUDA 13 dropped Volta from NVRTC, so
/// `compute_70` yields `NVRTC_ERROR_INVALID_OPTION` on a CUDA-13 toolkit
/// (e.g. the DGX Spark / GB10, sm_121). Targeting the device's own arch is
/// correct on every toolkit — NVRTC always accepts the arch of a device the
/// same-version driver can run — and skips the PTX→SASS JIT at module load.
///
/// Cached in a process-global `OnceLock`: device 0's arch is stable for the
/// process, and every kernel compiles for the same device.
fn nvrtc_arch(ctx: &Arc<CudaContext>) -> Result<&'static str, CudaError> {
    static ARCH: OnceLock<String> = OnceLock::new();
    if let Some(a) = ARCH.get() {
        return Ok(a.as_str());
    }
    let (major, minor) = ctx
        .compute_capability()
        .map_err(|e| CudaError::KernelCompile {
            kernel: "<arch-detect>",
            reason: format!("compute_capability: {e}"),
        })?;
    Ok(ARCH
        .get_or_init(|| format!("compute_{major}{minor}"))
        .as_str())
}

/// Pre-compiled, loaded CUDA functions ready for `launch_builder`.
/// Cloning is cheap (refcounted Arcs).
#[derive(Clone)]
pub(crate) struct KernelLibrary {
    pub apply_diagonal: CudaFunction,
    pub apply_diagonal_pauli_sum: CudaFunction,
    pub apply_diagonal_product: CudaFunction,
    pub apply_1q: CudaFunction,
    pub apply_2q: CudaFunction,
    pub inner_product: CudaFunction,
    pub pauli_expectation: CudaFunction,
    // Pooled-param variants for the CUDA-graph capture path.
    // Suppressed dead-code: these are read by the upcoming
    // ForwardGraph wiring (next slice). NVRTC compile + module load
    // already runs at backend construction so we can fail fast on a
    // missing kernel rather than at first use.
    #[allow(dead_code)]
    pub apply_diagonal_pooled: CudaFunction,
    #[allow(dead_code)]
    pub apply_1q_pooled: CudaFunction,
    #[allow(dead_code)]
    pub apply_2q_pooled: CudaFunction,
    pub compute_probabilities: CudaFunction,
    /// CDF inclusive-scan pass 1: per-block scan in shared memory,
    /// writes per-block totals to a separate array.
    pub cdf_block_scan_pass1: CudaFunction,
    /// CDF inclusive-scan pass 2: add the prior block's inclusive
    /// total to each element. Composed with a recursive scan of the
    /// block-totals array (handled in Rust) so this pair scales to
    /// any dim — n_qubits ≤ ~30 in practice.
    pub cdf_block_scan_pass2_from_inclusive: CudaFunction,
    /// Per-shot binary search over the on-device CDF + atomicAdd
    /// into a dense counts buffer.
    pub sample_from_cdf: CudaFunction,
    pub inner_product_accumulate_pooled: CudaFunction,
    pub pauli_z_expectation_to_slot: CudaFunction,
    pub compute_residual_coeffs: CudaFunction,
    pub apply_1q_from_to_pooled: CudaFunction,
    pub apply_1q_inner_product_accumulate_pooled: CudaFunction,
    pub apply_2q_pooled_dual: CudaFunction,
    pub apply_1q_pooled_dual: CudaFunction,
    pub apply_diagonal_chain_pooled: CudaFunction,
    pub pauli_z_chain_accumulate: CudaFunction,
    pub apply_diagonal_chain_dual_pooled: CudaFunction,
    pub pauli_y_accumulate_then_dagger_both_pooled: CudaFunction,
    // Hold the modules alive — `CudaFunction` borrows from the module
    // refcount, so dropping the modules invalidates the functions.
    _modules: Vec<Arc<CudaModule>>,
}

impl KernelLibrary {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, CudaError> {
        let (apply_diagonal_module, apply_diagonal) =
            load_kernel(ctx, KERNEL_APPLY_DIAGONAL, "apply_diagonal")?;
        let (apply_diagonal_pauli_sum_module, apply_diagonal_pauli_sum) = load_kernel(
            ctx,
            KERNEL_APPLY_DIAGONAL_PAULI_SUM,
            "apply_diagonal_pauli_sum",
        )?;
        let (apply_diagonal_product_module, apply_diagonal_product) =
            load_kernel(ctx, KERNEL_APPLY_DIAGONAL_PRODUCT, "apply_diagonal_product")?;
        let (apply_1q_module, apply_1q) = load_kernel(ctx, KERNEL_APPLY_1Q, "apply_1q")?;
        let (apply_2q_module, apply_2q) = load_kernel(ctx, KERNEL_APPLY_2Q, "apply_2q")?;
        let (inner_product_module, inner_product) =
            load_kernel(ctx, KERNEL_INNER_PRODUCT, "inner_product")?;
        let (pauli_expectation_module, pauli_expectation) =
            load_kernel(ctx, KERNEL_PAULI_EXPECTATION, "pauli_expectation")?;
        let (apply_diagonal_pooled_module, apply_diagonal_pooled) =
            load_kernel(ctx, KERNEL_APPLY_DIAGONAL_POOLED, "apply_diagonal_pooled")?;
        let (apply_1q_pooled_module, apply_1q_pooled) =
            load_kernel(ctx, KERNEL_APPLY_1Q_POOLED, "apply_1q_pooled")?;
        let (apply_2q_pooled_module, apply_2q_pooled) =
            load_kernel(ctx, KERNEL_APPLY_2Q_POOLED, "apply_2q_pooled")?;
        let (compute_probabilities_module, compute_probabilities) =
            load_kernel(ctx, KERNEL_COMPUTE_PROBABILITIES, "compute_probabilities")?;
        let (cdf_scan_module, cdf_scan_fns) = load_kernel_multi(
            ctx,
            KERNEL_CDF_SCAN,
            "cdf_scan",
            &[
                "cdf_block_scan_pass1",
                "cdf_block_scan_pass2_from_inclusive",
            ],
        )?;
        let mut cdf_scan_fns = cdf_scan_fns.into_iter();
        let cdf_block_scan_pass1 = cdf_scan_fns.next().expect("pass1 loaded");
        let cdf_block_scan_pass2_from_inclusive =
            cdf_scan_fns.next().expect("pass2_from_inclusive loaded");
        let (sample_from_cdf_module, sample_from_cdf) =
            load_kernel(ctx, KERNEL_SAMPLE_FROM_CDF, "sample_from_cdf")?;
        let (inner_product_accumulate_pooled_module, inner_product_accumulate_pooled) =
            load_kernel(
                ctx,
                KERNEL_INNER_PRODUCT_ACCUMULATE_POOLED,
                "inner_product_accumulate_pooled",
            )?;
        let (pauli_z_expectation_to_slot_module, pauli_z_expectation_to_slot) = load_kernel(
            ctx,
            KERNEL_PAULI_Z_EXPECTATION_TO_SLOT,
            "pauli_z_expectation_to_slot",
        )?;
        let (compute_residual_coeffs_module, compute_residual_coeffs) = load_kernel(
            ctx,
            KERNEL_COMPUTE_RESIDUAL_COEFFS,
            "compute_residual_coeffs",
        )?;
        let (apply_1q_from_to_pooled_module, apply_1q_from_to_pooled) = load_kernel(
            ctx,
            KERNEL_APPLY_1Q_FROM_TO_POOLED,
            "apply_1q_from_to_pooled",
        )?;
        let (
            apply_1q_inner_product_accumulate_pooled_module,
            apply_1q_inner_product_accumulate_pooled,
        ) = load_kernel(
            ctx,
            KERNEL_APPLY_1Q_INNER_PRODUCT_ACCUMULATE_POOLED,
            "apply_1q_inner_product_accumulate_pooled",
        )?;
        let (apply_2q_pooled_dual_module, apply_2q_pooled_dual) =
            load_kernel(ctx, KERNEL_APPLY_2Q_POOLED_DUAL, "apply_2q_pooled_dual")?;
        let (apply_1q_pooled_dual_module, apply_1q_pooled_dual) =
            load_kernel(ctx, KERNEL_APPLY_1Q_POOLED_DUAL, "apply_1q_pooled_dual")?;
        let (apply_diagonal_chain_pooled_module, apply_diagonal_chain_pooled) = load_kernel(
            ctx,
            KERNEL_APPLY_DIAGONAL_CHAIN_POOLED,
            "apply_diagonal_chain_pooled",
        )?;
        let (pauli_z_chain_accumulate_module, pauli_z_chain_accumulate) = load_kernel(
            ctx,
            KERNEL_PAULI_Z_CHAIN_ACCUMULATE,
            "pauli_z_chain_accumulate",
        )?;
        let (apply_diagonal_chain_dual_pooled_module, apply_diagonal_chain_dual_pooled) =
            load_kernel(
                ctx,
                KERNEL_APPLY_DIAGONAL_CHAIN_DUAL_POOLED,
                "apply_diagonal_chain_dual_pooled",
            )?;
        let (
            pauli_y_accumulate_then_dagger_both_pooled_module,
            pauli_y_accumulate_then_dagger_both_pooled,
        ) = load_kernel(
            ctx,
            KERNEL_PAULI_Y_ACCUMULATE_THEN_DAGGER_BOTH_POOLED,
            "pauli_y_accumulate_then_dagger_both_pooled",
        )?;
        Ok(Self {
            apply_diagonal,
            apply_diagonal_pauli_sum,
            apply_diagonal_product,
            apply_1q,
            apply_2q,
            inner_product,
            pauli_expectation,
            apply_diagonal_pooled,
            apply_1q_pooled,
            apply_2q_pooled,
            compute_probabilities,
            cdf_block_scan_pass1,
            cdf_block_scan_pass2_from_inclusive,
            sample_from_cdf,
            inner_product_accumulate_pooled,
            pauli_z_expectation_to_slot,
            compute_residual_coeffs,
            apply_1q_from_to_pooled,
            apply_1q_inner_product_accumulate_pooled,
            apply_2q_pooled_dual,
            apply_1q_pooled_dual,
            apply_diagonal_chain_pooled,
            pauli_z_chain_accumulate,
            apply_diagonal_chain_dual_pooled,
            pauli_y_accumulate_then_dagger_both_pooled,
            _modules: vec![
                apply_diagonal_module,
                apply_diagonal_pauli_sum_module,
                apply_diagonal_product_module,
                apply_1q_module,
                apply_2q_module,
                inner_product_module,
                pauli_expectation_module,
                apply_diagonal_pooled_module,
                apply_1q_pooled_module,
                apply_2q_pooled_module,
                compute_probabilities_module,
                cdf_scan_module,
                sample_from_cdf_module,
                inner_product_accumulate_pooled_module,
                pauli_z_expectation_to_slot_module,
                compute_residual_coeffs_module,
                apply_1q_from_to_pooled_module,
                apply_1q_inner_product_accumulate_pooled_module,
                apply_2q_pooled_dual_module,
                apply_1q_pooled_dual_module,
                apply_diagonal_chain_pooled_module,
                pauli_z_chain_accumulate_module,
                apply_diagonal_chain_dual_pooled_module,
                pauli_y_accumulate_then_dagger_both_pooled_module,
            ],
        })
    }
}

/// Every embedded kernel source with its label, for the dual-precision compile
/// test. Kept next to the constants so a new kernel that is not listed here is
/// visible in review.
pub fn all_kernel_sources() -> Vec<(&'static str, &'static str)> {
    vec![
        ("apply_diagonal", KERNEL_APPLY_DIAGONAL),
        ("apply_diagonal_pauli_sum", KERNEL_APPLY_DIAGONAL_PAULI_SUM),
        ("apply_diagonal_product", KERNEL_APPLY_DIAGONAL_PRODUCT),
        ("apply_1q", KERNEL_APPLY_1Q),
        ("apply_2q", KERNEL_APPLY_2Q),
        ("inner_product", KERNEL_INNER_PRODUCT),
        ("pauli_expectation", KERNEL_PAULI_EXPECTATION),
        ("apply_diagonal_pooled", KERNEL_APPLY_DIAGONAL_POOLED),
        ("apply_1q_pooled", KERNEL_APPLY_1Q_POOLED),
        ("apply_2q_pooled", KERNEL_APPLY_2Q_POOLED),
        ("compute_probabilities", KERNEL_COMPUTE_PROBABILITIES),
        ("cdf_scan", KERNEL_CDF_SCAN),
        ("sample_from_cdf", KERNEL_SAMPLE_FROM_CDF),
        ("compute_residual_coeffs", KERNEL_COMPUTE_RESIDUAL_COEFFS),
        ("apply_1q_from_to_pooled", KERNEL_APPLY_1Q_FROM_TO_POOLED),
        ("apply_2q_pooled_dual", KERNEL_APPLY_2Q_POOLED_DUAL),
        ("apply_1q_pooled_dual", KERNEL_APPLY_1Q_POOLED_DUAL),
        ("pauli_z_chain_accumulate", KERNEL_PAULI_Z_CHAIN_ACCUMULATE),
    ]
}

/// Open a context on device 0, or `None` when there is no usable device — lets
/// a device test skip out loud instead of failing on a CPU-only box.
pub fn compile_probe() -> Option<Arc<CudaContext>> {
    CudaContext::new(0).ok()
}

/// The two kernel sources the f64 forward path needs.
pub fn source_apply_1q() -> &'static str {
    KERNEL_APPLY_1Q
}
pub fn source_apply_2q() -> &'static str {
    KERNEL_APPLY_2Q
}

/// Compile one source at one precision and hand back the module plus the named
/// function. The module must be kept alive by the caller.
pub fn try_compile_module(
    ctx: &Arc<CudaContext>,
    source: &str,
    function_name: &'static str,
    prec: Precision,
) -> Result<(Arc<CudaModule>, CudaFunction), CudaError> {
    load_kernel_prec(ctx, source, function_name, prec)
}

/// Compile one source at one precision, discarding the module. Used by the
/// dual-precision test.
pub fn try_compile(
    ctx: &Arc<CudaContext>,
    source: &str,
    label: &'static str,
    prec: Precision,
) -> Result<(), CudaError> {
    let source = prefix_prelude(source);
    let opts = CompileOptions {
        arch: Some(nvrtc_arch(ctx)?),
        name: Some(label.to_string()),
        options: prec.defines().iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    compile_ptx_with_opts(&source, opts)
        .map(|_| ())
        .map_err(|e| CudaError::KernelCompile {
            kernel: label,
            reason: format!("nvrtc ({}): {e:?}", prec.as_str()),
        })
}

/// Prepend the precision prelude. Done by concatenation rather than `#include`
/// because NVRTC has no filesystem to include from.
fn prefix_prelude(source: &str) -> String {
    format!("{PRELUDE}\n{source}")
}

fn load_kernel(
    ctx: &Arc<CudaContext>,
    source: &str,
    function_name: &'static str,
) -> Result<(Arc<CudaModule>, CudaFunction), CudaError> {
    load_kernel_prec(ctx, source, function_name, Precision::default())
}

fn load_kernel_prec(
    ctx: &Arc<CudaContext>,
    source: &str,
    function_name: &'static str,
    prec: Precision,
) -> Result<(Arc<CudaModule>, CudaFunction), CudaError> {
    let source = &prefix_prelude(source);
    let opts = CompileOptions {
        arch: Some(nvrtc_arch(ctx)?),
        name: Some(function_name.to_string()),
        options: prec.defines().iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(source, opts).map_err(|e| CudaError::KernelCompile {
        kernel: function_name,
        reason: format!("nvrtc: {e:?}"),
    })?;
    let module = ctx.load_module(ptx).map_err(|e| CudaError::KernelCompile {
        kernel: function_name,
        reason: format!("module load: {e}"),
    })?;
    let func = module
        .load_function(function_name)
        .map_err(|e| CudaError::KernelCompile {
            kernel: function_name,
            reason: format!("function lookup: {e}"),
        })?;
    Ok((module, func))
}

/// Compile a single source file once and load multiple functions from
/// it. Used for the shot-sampler's CDF scan, which ships three
/// kernels (pass1, exclusive scan of block totals, pass2) in one
/// `.cu` source so they share the NVRTC compile cost.
fn load_kernel_multi(
    ctx: &Arc<CudaContext>,
    source: &str,
    label: &'static str,
    function_names: &[&'static str],
) -> Result<(Arc<CudaModule>, Vec<CudaFunction>), CudaError> {
    load_kernel_multi_prec(ctx, source, label, function_names, Precision::default())
}

fn load_kernel_multi_prec(
    ctx: &Arc<CudaContext>,
    source: &str,
    label: &'static str,
    function_names: &[&'static str],
    prec: Precision,
) -> Result<(Arc<CudaModule>, Vec<CudaFunction>), CudaError> {
    let source = &prefix_prelude(source);
    let opts = CompileOptions {
        arch: Some(nvrtc_arch(ctx)?),
        name: Some(label.to_string()),
        options: prec.defines().iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(source, opts).map_err(|e| CudaError::KernelCompile {
        kernel: label,
        reason: format!("nvrtc: {e:?}"),
    })?;
    let module = ctx.load_module(ptx).map_err(|e| CudaError::KernelCompile {
        kernel: label,
        reason: format!("module load: {e}"),
    })?;
    let mut fns = Vec::with_capacity(function_names.len());
    for &fname in function_names {
        let f = module
            .load_function(fname)
            .map_err(|e| CudaError::KernelCompile {
                kernel: fname,
                reason: format!("function lookup: {e}"),
            })?;
        fns.push(f);
    }
    Ok((module, fns))
}
