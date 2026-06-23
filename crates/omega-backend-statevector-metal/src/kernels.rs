//! Kernel library — compiles the embedded MSL sources at backend
//! construction time and caches one [`metal::ComputePipelineState`]
//! per kernel.
//!
//! Step 3 ships only `apply_diagonal`. Steps 4+ extend this module with
//! `apply_1q`, `apply_2q`, `inner_product`, `pauli_expectation`.

use metal::{CompileOptions, ComputePipelineState, Device, Function, MTLLanguageVersion};

use crate::MetalError;

/// MSL sources for the kernels we ship. Embedded at compile time so
/// no runtime filesystem access is required.
const SHADER_APPLY_DIAGONAL: &str = include_str!("shaders/apply_diagonal.metal");
const SHADER_APPLY_DIAGONAL_2Q: &str = include_str!("shaders/apply_diagonal_2q.metal");
const SHADER_APPLY_DIAGONAL_INTO: &str = include_str!("shaders/apply_diagonal_into.metal");
const SHADER_APPLY_DIAGONAL_PAULI_SUM: &str =
    include_str!("shaders/apply_diagonal_pauli_sum.metal");
const SHADER_APPLY_DIAGONAL_PRODUCT: &str = include_str!("shaders/apply_diagonal_product.metal");
const SHADER_APPLY_1Q: &str = include_str!("shaders/apply_1q.metal");
const SHADER_APPLY_1Q_INTO: &str = include_str!("shaders/apply_1q_into.metal");
const SHADER_APPLY_2Q: &str = include_str!("shaders/apply_2q.metal");
const SHADER_INNER_PRODUCT: &str = include_str!("shaders/inner_product.metal");
const SHADER_PAULI_EXPECTATION: &str = include_str!("shaders/pauli_expectation.metal");
const SHADER_SHOT_SAMPLE: &str = include_str!("shaders/shot_sample.metal");

/// Pre-compiled compute pipeline states keyed by kernel name. Held by
/// each `DeviceHandle` so a single device instance pays the MSL compile
/// cost only once. Cloning is cheap — `ComputePipelineState` is a
/// refcounted ObjC handle.
#[derive(Clone)]
pub(crate) struct KernelLibrary {
    pub apply_diagonal: ComputePipelineState,
    pub apply_diagonal_2q: ComputePipelineState,
    pub apply_diagonal_into: ComputePipelineState,
    pub apply_diagonal_pauli_sum: ComputePipelineState,
    pub apply_diagonal_product: ComputePipelineState,
    pub apply_1q: ComputePipelineState,
    pub apply_1q_into: ComputePipelineState,
    pub apply_2q: ComputePipelineState,
    pub inner_product: ComputePipelineState,
    pub pauli_expectation: ComputePipelineState,
    pub shot_probs: ComputePipelineState,
    pub shot_scan_step: ComputePipelineState,
    pub shot_sample: ComputePipelineState,
}

impl KernelLibrary {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        let apply_diagonal = compile_kernel(device, SHADER_APPLY_DIAGONAL, "apply_diagonal")?;
        let apply_diagonal_2q =
            compile_kernel(device, SHADER_APPLY_DIAGONAL_2Q, "apply_diagonal_2q")?;
        let apply_diagonal_into =
            compile_kernel(device, SHADER_APPLY_DIAGONAL_INTO, "apply_diagonal_into")?;
        let apply_diagonal_pauli_sum = compile_kernel(
            device,
            SHADER_APPLY_DIAGONAL_PAULI_SUM,
            "apply_diagonal_pauli_sum",
        )?;
        let apply_diagonal_product = compile_kernel(
            device,
            SHADER_APPLY_DIAGONAL_PRODUCT,
            "apply_diagonal_product",
        )?;
        let apply_1q = compile_kernel(device, SHADER_APPLY_1Q, "apply_1q")?;
        let apply_1q_into = compile_kernel(device, SHADER_APPLY_1Q_INTO, "apply_1q_into")?;
        let apply_2q = compile_kernel(device, SHADER_APPLY_2Q, "apply_2q")?;
        let inner_product = compile_kernel(device, SHADER_INNER_PRODUCT, "inner_product")?;
        let pauli_expectation =
            compile_kernel(device, SHADER_PAULI_EXPECTATION, "pauli_expectation")?;
        let shot_probs = compile_kernel(device, SHADER_SHOT_SAMPLE, "shot_probs")?;
        let shot_scan_step = compile_kernel(device, SHADER_SHOT_SAMPLE, "shot_scan_step")?;
        let shot_sample = compile_kernel(device, SHADER_SHOT_SAMPLE, "shot_sample")?;
        Ok(Self {
            apply_diagonal,
            apply_diagonal_2q,
            apply_diagonal_into,
            apply_diagonal_pauli_sum,
            apply_diagonal_product,
            apply_1q,
            apply_1q_into,
            apply_2q,
            inner_product,
            pauli_expectation,
            shot_probs,
            shot_scan_step,
            shot_sample,
        })
    }
}

fn compile_kernel(
    device: &Device,
    source: &str,
    function_name: &'static str,
) -> Result<ComputePipelineState, MetalError> {
    let opts = CompileOptions::new();
    // MSL 2.0 is plenty for the kernels we ship; pinning avoids the
    // toolchain picking up a newer dialect that breaks older Apple
    // GPU families (none we currently support, but defensive).
    opts.set_language_version(MTLLanguageVersion::V2_0);

    let library =
        device
            .new_library_with_source(source, &opts)
            .map_err(|e| MetalError::KernelCompile {
                kernel: function_name,
                reason: format!("library compile: {e}"),
            })?;

    let function: Function =
        library
            .get_function(function_name, None)
            .map_err(|e| MetalError::KernelCompile {
                kernel: function_name,
                reason: format!("function lookup: {e}"),
            })?;

    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| MetalError::KernelCompile {
            kernel: function_name,
            reason: format!("pipeline state: {e}"),
        })
}
