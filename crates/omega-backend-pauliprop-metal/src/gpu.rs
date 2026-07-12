//! Metal implementation of the branch expansion (macOS + `metal` feature).
//!
//! The Metal compute kernel does only the integer symplectic work — the
//! anticommute parity, the sign parity, and the child key `R·P` — over packed
//! u64-word arrays. Everything with a coefficient in it (cosθ·P, the i·sinθ·
//! factor·sign child, the `max_freq` cap, and the term merge) runs here on the
//! CPU in f64, so the numbers are bit-for-bit the CPU branch. Apple GPUs have
//! no native double; keeping the floats off-device is what makes that exact.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use metal::{
    CompileOptions, ComputePipelineState, Device, Function, MTLLanguageVersion, MTLResourceOptions,
    MTLSize,
};
use num_complex::Complex64;
use omega_backend_pauliprop::{pack_bits, unpack_bits, PauliKey, PauliSum};

const SHADER: &str = include_str!("shaders/branch_symplectic.metal");

/// Number of branch steps run on the GPU this process (telemetry, mirrors the
/// CUDA arm's `GPU_BRANCHES`).
pub static GPU_BRANCHES: AtomicU64 = AtomicU64::new(0);

/// Compiled-once Metal handles, shared across every branch call.
struct MetalBranch {
    device: Device,
    queue: metal::CommandQueue,
    pipeline: ComputePipelineState,
}

static RUNTIME: OnceLock<Option<MetalBranch>> = OnceLock::new();

impl MetalBranch {
    fn get() -> Option<&'static MetalBranch> {
        RUNTIME.get_or_init(Self::try_new).as_ref()
    }

    fn try_new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        let opts = CompileOptions::new();
        opts.set_language_version(MTLLanguageVersion::V2_0);
        let library = device
            .new_library_with_source(SHADER, &opts)
            .map_err(|e| format!("library compile: {e}"))
            .ok()?;
        let function: Function = library.get_function("branch_symplectic", None).ok()?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .ok()?;
        Some(Self {
            device,
            queue,
            pipeline,
        })
    }
}

/// Run the branch expansion on the GPU. Returns `true` on success (with `sum`
/// replaced by the branched result), `false` on any failure — leaving `sum`
/// untouched so the caller's CPU path produces the identical result.
#[allow(clippy::too_many_arguments)]
pub fn branch_on_gpu(
    sum: &mut PauliSum,
    rx: &[u64],
    rz: &[u64],
    factor: Complex64,
    cos: f64,
    sin: f64,
    max_freq: Option<u32>,
    n: usize,
) -> bool {
    // Word count per symplectic string — the same width `pack_bits(&key.x)`
    // produces for every term. Bail to the CPU on the degenerate n == 0 /
    // empty-sum cases rather than fudge widths.
    let w = rx.len();
    let num = sum.terms.len();
    if num == 0 || w == 0 {
        return false;
    }
    let Some(rt) = MetalBranch::get() else {
        return false;
    };

    // Host SoA, in a fixed iteration order we reuse for the merge. Deterministic
    // term order isn't required (the merge is commutative), but the order the
    // GPU sees and the order we read parities back in must be the same.
    let mut keys: Vec<&PauliKey> = Vec::with_capacity(num);
    let mut coeffs: Vec<Complex64> = Vec::with_capacity(num);
    let mut freqs: Vec<u32> = Vec::with_capacity(num);
    let mut x_host = Vec::with_capacity(num * w);
    let mut z_host = Vec::with_capacity(num * w);
    for (key, wt) in &sum.terms {
        x_host.extend_from_slice(&pack_bits(&key.x));
        z_host.extend_from_slice(&pack_bits(&key.z));
        keys.push(key);
        coeffs.push(wt.coeff);
        freqs.push(wt.freq);
    }

    // --- Upload + dispatch. Any failure returns false → CPU fallback. ---
    let dev = &rt.device;
    let u64_buf = |v: &[u64]| -> metal::Buffer {
        dev.new_buffer_with_data(
            v.as_ptr() as *const c_void,
            std::mem::size_of_val(v) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let x_buf = u64_buf(&x_host);
    let z_buf = u64_buf(&z_host);
    let rx_buf = u64_buf(rx);
    let rz_buf = u64_buf(rz);

    let anti_buf = dev.new_buffer(num as u64, MTLResourceOptions::StorageModeShared);
    let sgn_buf = dev.new_buffer(num as u64, MTLResourceOptions::StorageModeShared);
    let key_bytes = (num * w * std::mem::size_of::<u64>()) as u64;
    let ox1_buf = dev.new_buffer(key_bytes, MTLResourceOptions::StorageModeShared);
    let oz1_buf = dev.new_buffer(key_bytes, MTLResourceOptions::StorageModeShared);

    #[repr(C)]
    struct Params {
        num: u32,
        words: u32,
    }
    // Each packed u64 word is two little-endian `uint` lanes on the GPU; the
    // u64 buffers' bytes reinterpret as this `uint` array with no re-layout.
    let params = Params {
        num: num as u32,
        words: (w * 2) as u32,
    };

    let cmd = rt.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&rt.pipeline);
    encoder.set_buffer(0, Some(&x_buf), 0);
    encoder.set_buffer(1, Some(&z_buf), 0);
    encoder.set_buffer(2, Some(&rx_buf), 0);
    encoder.set_buffer(3, Some(&rz_buf), 0);
    encoder.set_buffer(4, Some(&anti_buf), 0);
    encoder.set_buffer(5, Some(&sgn_buf), 0);
    encoder.set_buffer(6, Some(&ox1_buf), 0);
    encoder.set_buffer(7, Some(&oz1_buf), 0);
    encoder.set_bytes(
        8,
        std::mem::size_of::<Params>() as u64,
        (&params as *const Params) as *const c_void,
    );
    let max_tg = rt.pipeline.max_total_threads_per_threadgroup();
    let tg = MTLSize {
        width: max_tg.min(num as u64),
        height: 1,
        depth: 1,
    };
    let grid = MTLSize {
        width: num as u64,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // --- Read the integer results back from the shared buffers. ---
    // Safety: shared-mode buffers written exclusively by the kernel, which has
    // completed (wait_until_completed above); lengths match the allocations.
    let anti = unsafe { std::slice::from_raw_parts(anti_buf.contents() as *const u8, num) };
    let sgn = unsafe { std::slice::from_raw_parts(sgn_buf.contents() as *const u8, num) };
    let ox1 = unsafe { std::slice::from_raw_parts(ox1_buf.contents() as *const u64, num * w) };
    let oz1 = unsafe { std::slice::from_raw_parts(oz1_buf.contents() as *const u64, num * w) };

    // --- Merge on the CPU, in f64, identically to the CPU branch. ---
    let i_unit = Complex64::new(0.0, 1.0);
    let mut out = PauliSum::new();
    out.dropped_mass = sum.dropped_mass;
    for i in 0..num {
        let coeff = coeffs[i];
        let freq = freqs[i];
        if anti[i] == 0 {
            // Commutes with R → unchanged.
            out.add_weighted(keys[i].clone(), coeff, freq);
            continue;
        }
        // Anticommutes: cosθ·P keeps the frequency ...
        out.add_weighted(keys[i].clone(), coeff * cos, freq);
        // ... plus the sin child i·sinθ·factor·sign·(R·P).
        let sign = if sgn[i] != 0 { -1.0 } else { 1.0 };
        let sin_coeff = coeff * i_unit * sin * factor;
        let child_freq = freq + 1;
        if max_freq.is_some_and(|m| child_freq > m) {
            // Over the split-frequency budget: certify the discarded L1 mass.
            out.dropped_mass += sin_coeff.norm();
            continue;
        }
        let base = i * w;
        let key1 = PauliKey {
            x: unpack_bits(&ox1[base..base + w], n),
            z: unpack_bits(&oz1[base..base + w], n),
        };
        out.add_weighted(key1, sin_coeff * sign, child_freq);
    }

    *sum = out;
    GPU_BRANCHES.fetch_add(1, Ordering::Relaxed);
    true
}
