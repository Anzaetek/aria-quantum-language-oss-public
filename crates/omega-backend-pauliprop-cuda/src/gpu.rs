//! CUDA implementation of the branch expansion (Linux/Windows + `cuda` feature).

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use num_complex::Complex64;
use omega_backend_pauliprop::{pack_bits, unpack_bits, PauliKey, PauliSum};

/// Virtual arch target for the NVRTC compile, detected from the live device's
/// compute capability (`compute_{major}{minor}`), matching statevector-cuda.
///
/// A fixed `compute_70` floor breaks on a CUDA-13 toolkit (Volta was dropped
/// from NVRTC → `NVRTC_ERROR_INVALID_OPTION`), e.g. the DGX Spark / GB10
/// (sm_121). Previously that failure was swallowed by `Ctx::new`'s `.ok()?`,
/// silently routing every branch to the CPU fallback. Detecting the device's
/// own arch compiles correctly on every toolkit. Cached: device 0's arch is
/// stable per process.
fn nvrtc_arch(ctx: &Arc<CudaContext>) -> Option<&'static str> {
    static ARCH: OnceLock<String> = OnceLock::new();
    if let Some(a) = ARCH.get() {
        return Some(a.as_str());
    }
    let (major, minor) = ctx.compute_capability().ok()?;
    Some(
        ARCH.get_or_init(|| format!("compute_{major}{minor}"))
            .as_str(),
    )
}

/// Number of branch steps run on the GPU this process (telemetry).
pub static GPU_BRANCHES: AtomicU64 = AtomicU64::new(0);

/// The branch-expansion kernel. One thread per input term; emits up to two
/// children (slots `2i` and `2i+1`). All symplectic strings are `W` little-endian
/// u64 words. Semantics mirror the CPU `branch` exactly: commuting terms pass
/// through unchanged, anticommuting terms yield `cosθ·P` plus the sin-child
/// `i sinθ·factor·sign·(P⊕R)`; a sin-child over the `max_freq` cap is dropped and
/// its L1 mass certified into `odropped`.
const KERNEL: &str = r#"
extern "C" __global__ void branch_expand(
    const unsigned long long* x,
    const unsigned long long* z,
    const double* cre,
    const double* cim,
    const unsigned int* freq,
    const unsigned long long* rx,
    const unsigned long long* rz,
    unsigned long long* ox,
    unsigned long long* oz,
    double* ocre,
    double* ocim,
    unsigned int* ofreq,
    unsigned int* ovalid,
    double* odropped,
    double fre, double fim,
    double cosv, double sinv,
    int has_maxfreq, unsigned int maxfreq,
    int num, int W)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num) return;

    unsigned long long anti_acc = 0ULL, sgn_acc = 0ULL;
    for (int w = 0; w < W; ++w) {
        unsigned long long px = x[(long long)i * W + w];
        unsigned long long pz = z[(long long)i * W + w];
        anti_acc ^= (px & rz[w]) ^ (pz & rx[w]);
        sgn_acc  ^= (rz[w] & px);
    }
    int aparity = __popcll(anti_acc) & 1;
    int sparity = __popcll(sgn_acc) & 1;

    long long c0 = 2LL * i, c1 = 2LL * i + 1;
    for (int w = 0; w < W; ++w) {
        ox[c0 * W + w] = x[(long long)i * W + w];
        oz[c0 * W + w] = z[(long long)i * W + w];
    }
    ofreq[c0] = freq[i];
    ovalid[c0] = 1u;
    ovalid[c1] = 0u;
    odropped[i] = 0.0;

    if (!aparity) {
        // Commutes with R → unchanged (coefficient and frequency preserved).
        ocre[c0] = cre[i];
        ocim[c0] = cim[i];
        return;
    }

    // Anticommutes → cosθ·P ...
    ocre[c0] = cre[i] * cosv;
    ocim[c0] = cim[i] * cosv;

    // ... plus the sin child i·sinθ·factor·(R·P). g = i·sinθ·factor.
    double gre = -sinv * fim;
    double gim =  sinv * fre;
    double pre = cre[i] * gre - cim[i] * gim;
    double pim = cre[i] * gim + cim[i] * gre;
    double sgn = sparity ? -1.0 : 1.0;
    pre *= sgn;
    pim *= sgn;

    unsigned int nf = freq[i] + 1u;
    if (has_maxfreq && nf > maxfreq) {
        odropped[i] = sqrt(pre * pre + pim * pim);
        return;
    }
    for (int w = 0; w < W; ++w) {
        ox[c1 * W + w] = x[(long long)i * W + w] ^ rx[w];
        oz[c1 * W + w] = z[(long long)i * W + w] ^ rz[w];
    }
    ocre[c1] = pre;
    ocim[c1] = pim;
    ofreq[c1] = nf;
    ovalid[c1] = 1u;
}
"#;

struct Ctx {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    _module: Arc<CudaModule>,
    _ctx: Arc<CudaContext>,
}

impl Ctx {
    fn new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        let opts = CompileOptions {
            arch: Some(nvrtc_arch(&ctx)?),
            name: Some("branch_expand".to_string()),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL, opts).ok()?;
        let module = ctx.load_module(ptx).ok()?;
        let func = module.load_function("branch_expand").ok()?;
        Some(Ctx {
            stream,
            func,
            _module: module,
            _ctx: ctx,
        })
    }
}

thread_local! {
    // Outer Option = "did we try to init"; inner = "device present". `!Send`
    // (holds driver handles), so a thread-local is the right home — propagation
    // runs single-threaded per backend call.
    static CTX: RefCell<Option<Option<Ctx>>> = const { RefCell::new(None) };
}

/// Run the branch expansion on the GPU. Returns `true` on success (with `sum`
/// replaced by the branched result), `false` on any failure — in which case
/// `sum` is left untouched so the caller's CPU path produces the identical
/// result.
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
    CTX.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(Ctx::new());
        }
        let Some(ctx) = slot.as_ref().and_then(|c| c.as_ref()) else {
            return false;
        };
        run(ctx, sum, rx, rz, factor, cos, sin, max_freq, n)
    })
}

#[allow(clippy::too_many_arguments)]
fn run(
    ctx: &Ctx,
    sum: &mut PauliSum,
    rx: &[u64],
    rz: &[u64],
    factor: Complex64,
    cos: f64,
    sin: f64,
    max_freq: Option<u32>,
    n: usize,
) -> bool {
    use cudarc::driver::PushKernelArg;

    // Word count per symplectic string. `rx` was packed from the n-qubit
    // generator, so `w == n.div_ceil(64)` — the SAME width `pack_bits(&key.x)`
    // produces for every term, keeping the SoA layout consistent. Bail to the
    // CPU on the degenerate `n == 0` / empty-sum cases rather than fudge widths.
    let w = rx.len();
    let num = sum.terms.len();
    if num == 0 || w == 0 {
        return false;
    }

    // Host SoA. Deterministic term order isn't required (the merge is
    // commutative), so iterate the map directly.
    let mut x_host = Vec::with_capacity(num * w);
    let mut z_host = Vec::with_capacity(num * w);
    let mut cre = Vec::with_capacity(num);
    let mut cim = Vec::with_capacity(num);
    let mut freq = Vec::with_capacity(num);
    for (key, wt) in &sum.terms {
        x_host.extend_from_slice(&pack_bits(&key.x));
        z_host.extend_from_slice(&pack_bits(&key.z));
        cre.push(wt.coeff.re);
        cim.push(wt.coeff.im);
        freq.push(wt.freq);
    }

    let s = &ctx.stream;
    // Upload inputs.
    let (mut x_dev, mut z_dev) =
        match (s.alloc_zeros::<u64>(num * w), s.alloc_zeros::<u64>(num * w)) {
            (Ok(a), Ok(b)) => (a, b),
            _ => return false,
        };
    if s.memcpy_htod(&x_host, &mut x_dev).is_err() || s.memcpy_htod(&z_host, &mut z_dev).is_err() {
        return false;
    }
    let mk = |v: &[u64]| -> Option<cudarc::driver::CudaSlice<u64>> {
        let mut d = s.alloc_zeros::<u64>(v.len().max(1)).ok()?;
        s.memcpy_htod(v, &mut d).ok()?;
        Some(d)
    };
    let (Some(rx_dev), Some(rz_dev)) = (mk(rx), mk(rz)) else {
        return false;
    };
    let upload_f64 = |v: &[f64]| -> Option<cudarc::driver::CudaSlice<f64>> {
        let mut d = s.alloc_zeros::<f64>(v.len().max(1)).ok()?;
        s.memcpy_htod(v, &mut d).ok()?;
        Some(d)
    };
    let (Some(cre_dev), Some(cim_dev)) = (upload_f64(&cre), upload_f64(&cim)) else {
        return false;
    };
    let mut freq_dev = match s.alloc_zeros::<u32>(num) {
        Ok(d) => d,
        Err(_) => return false,
    };
    if s.memcpy_htod(&freq, &mut freq_dev).is_err() {
        return false;
    }

    // Output buffers (zeroed → invalid slots skip cleanly).
    let alloc_u64 = |len: usize| s.alloc_zeros::<u64>(len);
    let alloc_f64 = |len: usize| s.alloc_zeros::<f64>(len);
    let alloc_u32 = |len: usize| s.alloc_zeros::<u32>(len);
    let (Ok(mut ox), Ok(mut oz)) = (alloc_u64(2 * num * w), alloc_u64(2 * num * w)) else {
        return false;
    };
    let (Ok(mut ocre), Ok(mut ocim)) = (alloc_f64(2 * num), alloc_f64(2 * num)) else {
        return false;
    };
    let (Ok(mut ofreq), Ok(mut ovalid)) = (alloc_u32(2 * num), alloc_u32(2 * num)) else {
        return false;
    };
    let Ok(mut odropped) = alloc_f64(num) else {
        return false;
    };

    // Scalars.
    let fre = factor.re;
    let fim = factor.im;
    let has_mf: i32 = if max_freq.is_some() { 1 } else { 0 };
    let mf: u32 = max_freq.unwrap_or(0);
    let num_i = num as i32;
    let w_i = w as i32;

    let cfg = LaunchConfig {
        grid_dim: ((num as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = s.launch_builder(&ctx.func);
    b.arg(&x_dev)
        .arg(&z_dev)
        .arg(&cre_dev)
        .arg(&cim_dev)
        .arg(&freq_dev)
        .arg(&rx_dev)
        .arg(&rz_dev)
        .arg(&mut ox)
        .arg(&mut oz)
        .arg(&mut ocre)
        .arg(&mut ocim)
        .arg(&mut ofreq)
        .arg(&mut ovalid)
        .arg(&mut odropped)
        .arg(&fre)
        .arg(&fim)
        .arg(&cos)
        .arg(&sin)
        .arg(&has_mf)
        .arg(&mf)
        .arg(&num_i)
        .arg(&w_i);
    if unsafe { b.launch(cfg) }.is_err() {
        return false;
    }
    if s.synchronize().is_err() {
        return false;
    }

    // Download.
    let (Ok(ox_h), Ok(oz_h)) = (s.clone_dtoh(&ox), s.clone_dtoh(&oz)) else {
        return false;
    };
    let (Ok(ocre_h), Ok(ocim_h)) = (s.clone_dtoh(&ocre), s.clone_dtoh(&ocim)) else {
        return false;
    };
    let (Ok(ofreq_h), Ok(ovalid_h)) = (s.clone_dtoh(&ofreq), s.clone_dtoh(&ovalid)) else {
        return false;
    };
    let Ok(odropped_h) = s.clone_dtoh(&odropped) else {
        return false;
    };

    // Merge children back into a fresh sum (add_weighted: sum coeffs, min freq).
    let mut out = PauliSum::new();
    out.dropped_mass = sum.dropped_mass + odropped_h.iter().sum::<f64>();
    for slot in 0..(2 * num) {
        if ovalid_h[slot] == 0 {
            continue;
        }
        let base = slot * w;
        let key = PauliKey {
            x: unpack_bits(&ox_h[base..base + w], n),
            z: unpack_bits(&oz_h[base..base + w], n),
        };
        out.add_weighted(
            key,
            Complex64::new(ocre_h[slot], ocim_h[slot]),
            ofreq_h[slot],
        );
    }

    *sum = out;
    GPU_BRANCHES.fetch_add(1, Ordering::Relaxed);
    true
}
