//! Metal device handle + compiled kernel pipelines for the MPS
//! θ-contraction. Only compiled when `target_os = "macos"` and
//! `feature = "metal"` are set; the parent crate gates this module
//! behind the same `cfg`.
//!
//! The runtime is a process-wide `OnceLock` so all `apply_two_site_gate_metal`
//! callers share a single device handle and pre-compiled MSL pipeline.
//! Each call still allocates fresh staging buffers for `left`, `right`,
//! `theta`, and `theta_prime`; pooling those is a follow-up for the
//! QML hot path bench (this crate's plan B fallback ladder).

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    CompileOptions, ComputePipelineState, Device, Function, MTLLanguageVersion, MTLResourceOptions,
    MTLSize,
};

const SHADER_CONTRACT_TWO_SITE: &str = include_str!("shaders/contract_two_site.metal");

/// Compiled-once Metal handles. Shared across calls via `MetalRuntime::get`.
pub struct MetalRuntime {
    device: Device,
    queue: metal::CommandQueue,
    contract_pipeline: ComputePipelineState,
    gate_apply_pipeline: ComputePipelineState,
}

static RUNTIME: OnceLock<Option<MetalRuntime>> = OnceLock::new();

impl MetalRuntime {
    pub fn get() -> Option<&'static MetalRuntime> {
        RUNTIME.get_or_init(Self::try_new).as_ref()
    }

    fn try_new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        let contract_pipeline = compile_kernel(&device, "mps_contract_two_site").ok()?;
        let gate_apply_pipeline = compile_kernel(&device, "mps_apply_two_site_gate").ok()?;
        Some(Self {
            device,
            queue,
            contract_pipeline,
            gate_apply_pipeline,
        })
    }

    /// Run the two-kernel θ-contraction pipeline. Returns the
    /// `theta_prime` paired-f32 buffer ready for host-side SVD.
    /// `left_f32` / `right_f32` lengths match `bl·2·bm·2` / `bm·2·br·2`
    /// (the trailing 2 is re/im interleave); `gate_f32` is 32 floats
    /// (16 complex entries).
    pub fn run_two_site(
        &self,
        left_f32: &[f32],
        right_f32: &[f32],
        gate_f32: &[f32; 32],
        bl: usize,
        bm: usize,
        br: usize,
    ) -> Option<Vec<f32>> {
        let theta_elems = bl * 4 * br;
        let theta_bytes = (theta_elems * 2 * std::mem::size_of::<f32>()) as u64;

        let left_bytes = std::mem::size_of_val(left_f32) as u64;
        let right_bytes = std::mem::size_of_val(right_f32) as u64;
        let gate_bytes = std::mem::size_of_val(gate_f32) as u64;

        let left_buf = self.device.new_buffer_with_data(
            left_f32.as_ptr() as *const c_void,
            left_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let right_buf = self.device.new_buffer_with_data(
            right_f32.as_ptr() as *const c_void,
            right_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let gate_buf = self.device.new_buffer_with_data(
            gate_f32.as_ptr() as *const c_void,
            gate_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let theta_buf = self
            .device
            .new_buffer(theta_bytes, MTLResourceOptions::StorageModeShared);
        let theta_prime_buf = self
            .device
            .new_buffer(theta_bytes, MTLResourceOptions::StorageModeShared);

        let cmd_buf = self.queue.new_command_buffer();

        // Stage 1: contract.
        {
            #[repr(C)]
            struct ContractParams {
                bl: u32,
                bm: u32,
                br: u32,
            }
            let p = ContractParams {
                bl: bl as u32,
                bm: bm as u32,
                br: br as u32,
            };
            let pipeline = &self.contract_pipeline;
            let total = (bl * 4 * br) as u64;
            let max_tg = pipeline.max_total_threads_per_threadgroup();
            let tg_size = MTLSize {
                width: max_tg.min(total),
                height: 1,
                depth: 1,
            };
            let grid_size = MTLSize {
                width: total,
                height: 1,
                depth: 1,
            };
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&left_buf), 0);
            encoder.set_buffer(1, Some(&right_buf), 0);
            encoder.set_buffer(2, Some(&theta_buf), 0);
            encoder.set_bytes(
                3,
                std::mem::size_of::<ContractParams>() as u64,
                (&p as *const ContractParams) as *const c_void,
            );
            encoder.dispatch_threads(grid_size, tg_size);
            encoder.end_encoding();
        }

        // Stage 2: gate apply.
        {
            #[repr(C)]
            struct GateApplyParams {
                bl: u32,
                br: u32,
            }
            let p = GateApplyParams {
                bl: bl as u32,
                br: br as u32,
            };
            let pipeline = &self.gate_apply_pipeline;
            let total = (bl * 4 * br) as u64;
            let max_tg = pipeline.max_total_threads_per_threadgroup();
            let tg_size = MTLSize {
                width: max_tg.min(total),
                height: 1,
                depth: 1,
            };
            let grid_size = MTLSize {
                width: total,
                height: 1,
                depth: 1,
            };
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&theta_buf), 0);
            encoder.set_buffer(1, Some(&gate_buf), 0);
            encoder.set_buffer(2, Some(&theta_prime_buf), 0);
            encoder.set_bytes(
                3,
                std::mem::size_of::<GateApplyParams>() as u64,
                (&p as *const GateApplyParams) as *const c_void,
            );
            encoder.dispatch_threads(grid_size, tg_size);
            encoder.end_encoding();
        }

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read theta_prime back as paired f32. Shared-mode buffer; no
        // staging copy needed.
        let out_floats = theta_elems * 2;
        let mut out: Vec<f32> = vec![0.0; out_floats];
        let src = theta_prime_buf.contents() as *const f32;
        // Safety: shared-mode buffer of `theta_elems * 8` bytes,
        // written exclusively by the `mps_apply_two_site_gate` kernel
        // which has completed (cmd_buf.wait_until_completed).
        unsafe {
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out_floats);
        }
        Some(out)
    }
}

fn compile_kernel(
    device: &Device,
    function_name: &'static str,
) -> Result<ComputePipelineState, String> {
    let opts = CompileOptions::new();
    opts.set_language_version(MTLLanguageVersion::V2_0);
    let library = device
        .new_library_with_source(SHADER_CONTRACT_TWO_SITE, &opts)
        .map_err(|e| format!("library compile: {e}"))?;
    let function: Function = library
        .get_function(function_name, None)
        .map_err(|e| format!("function lookup `{function_name}`: {e}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("pipeline state `{function_name}`: {e}"))
}

/// Pack a `&[Complex64]` host buffer into paired-f32 (re, im interleave)
/// for the Metal shader's `float2` layout.
pub fn complex_to_paired_f32(src: &[num_complex::Complex64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(src.len() * 2);
    for c in src {
        out.push(c.re as f32);
        out.push(c.im as f32);
    }
    out
}

/// Same as `complex_to_paired_f32` but for fixed-size gate arrays
/// (16 entries, 32 floats).
pub fn complex_array_to_paired_f32(src: &[num_complex::Complex64; 16]) -> [f32; 32] {
    let mut out = [0.0_f32; 32];
    for (i, c) in src.iter().enumerate() {
        out[2 * i] = c.re as f32;
        out[2 * i + 1] = c.im as f32;
    }
    out
}
