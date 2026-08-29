//! CUDA implementation of the branch expansion (Linux/Windows + `cuda` feature).

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use num_complex::Complex64;
use omega_backend_pauliprop::{PauliKey, PauliSum};

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

/// Cumulative nanoseconds per phase of `run`, and how many times it ran.
///
/// **Why this exists.** The claim "the host merge dominates, so buffer reuse
/// will not help" was asserted in three places in this repo and measured in
/// none — there was no pauliprop bench and no per-phase timer, only whole-run
/// wall clock, which cannot attribute cost to a phase. Optimising against that
/// is guessing. Enabled by `PAULIPROP_GPU_PROFILE=1`; zero cost otherwise (one
/// `bool` read per call, resolved once).
#[derive(Default, Clone, Copy)]
pub struct PhaseTimes {
    pub calls: u64,
    pub soa_ns: u64,
    pub alloc_ns: u64,
    pub upload_ns: u64,
    pub launch_sync_ns: u64,
    pub download_ns: u64,
    pub merge_ns: u64,
}

impl PhaseTimes {
    pub fn total_ns(&self) -> u64 {
        self.soa_ns
            + self.alloc_ns
            + self.upload_ns
            + self.launch_sync_ns
            + self.download_ns
            + self.merge_ns
    }
}

thread_local! {
    static PHASES: RefCell<PhaseTimes> = const { RefCell::new(PhaseTimes {
        calls: 0, soa_ns: 0, alloc_ns: 0, upload_ns: 0,
        launch_sync_ns: 0, download_ns: 0, merge_ns: 0,
    }) };
}

fn profiling() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PAULIPROP_GPU_PROFILE").is_ok())
}

/// Per-phase timings accumulated on this thread since the last reset.
pub fn phase_times() -> PhaseTimes {
    PHASES.with(|p| *p.borrow())
}

/// Zero the counters — call between measured configurations.
pub fn reset_phase_times() {
    PHASES.with(|p| *p.borrow_mut() = PhaseTimes::default());
}

/// Time `f` into `field` when profiling is on, else just run it.
#[inline]
fn phase<R>(field: fn(&mut PhaseTimes) -> &mut u64, f: impl FnOnce() -> R) -> R {
    if !profiling() {
        return f();
    }
    let t0 = std::time::Instant::now();
    let r = f();
    let dt = t0.elapsed().as_nanos() as u64;
    PHASES.with(|p| *field(&mut p.borrow_mut()) += dt);
    r
}

/// Device buffers held across calls, grown on demand and never shrunk.
///
/// **Why.** `run` allocated 14 buffers per call — each a `cuMemAlloc` PLUS a
/// memset, so 28 driver operations per rotation gate. With the packed
/// `PauliKey` landing, that phase went from 1% of the branch step to
/// **15-17%** in the wide/shallow regime (many small branch calls), because
/// everything around it got faster. It is now the largest cheap win here.
///
/// **The trap, and it is silent.** Reused buffers are LONGER than the current
/// call needs, so any consumer that reads a whole buffer instead of its live
/// prefix picks up the previous call's tail. `odropped` is summed into
/// `dropped_mass` — the CERTIFIED truncation bound — so a stale tail would
/// inflate an error budget that downstream code trusts, with nothing to
/// notice. Every download below is sliced to the live length for that reason;
/// the `[..num]` on the dropped-mass sum landed earlier for the same reason.
#[derive(Default)]
struct Buffers {
    x: Option<CudaSlice<u64>>,
    z: Option<CudaSlice<u64>>,
    rx: Option<CudaSlice<u64>>,
    rz: Option<CudaSlice<u64>>,
    cre: Option<CudaSlice<f64>>,
    cim: Option<CudaSlice<f64>>,
    freq: Option<CudaSlice<u32>>,
    ox: Option<CudaSlice<u64>>,
    oz: Option<CudaSlice<u64>>,
    ocre: Option<CudaSlice<f64>>,
    ocim: Option<CudaSlice<f64>>,
    ofreq: Option<CudaSlice<u32>>,
    ovalid: Option<CudaSlice<u32>>,
    odropped: Option<CudaSlice<f64>>,
}

/// Ensure `slot` holds at least `len` elements, allocating only on growth.
///
/// Zeroed on (re)allocation only. That is sound because the kernel writes
/// `ovalid` and `odropped` for EVERY slot it processes (`ovalid[c0]`,
/// `ovalid[c1]`, `odropped[i]` are unconditional), so no consumer depends on
/// a zero it did not write — and the reads are prefix-bounded regardless.
fn ensure<T>(stream: &Arc<CudaStream>, slot: &mut Option<CudaSlice<T>>, len: usize) -> Option<()>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
{
    let len = len.max(1);
    let need = match slot {
        Some(b) => b.len() < len,
        None => true,
    };
    if need {
        *slot = Some(stream.alloc_zeros::<T>(len).ok()?);
    }
    Some(())
}

struct Ctx {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    bufs: Buffers,
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
            bufs: Buffers::default(),
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
        let Some(ctx) = slot.as_mut().and_then(|c| c.as_mut()) else {
            return false;
        };
        run(ctx, sum, rx, rz, factor, cos, sin, max_freq, n)
    })
}

#[allow(clippy::too_many_arguments)]
fn run(
    ctx: &mut Ctx,
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
    let num = sum.len();
    if num == 0 || w == 0 {
        return false;
    }

    // Host SoA. Deterministic term order isn't required (the merge is
    // commutative), so iterate the map directly.
    let (x_host, z_host, cre, cim, freq) = phase(
        |p| &mut p.soa_ns,
        || {
            let mut x_host = Vec::with_capacity(num * w);
            let mut z_host = Vec::with_capacity(num * w);
            let mut cre = Vec::with_capacity(num);
            let mut cim = Vec::with_capacity(num);
            let mut freq = Vec::with_capacity(num);
            for (key, wt) in sum.iter() {
                x_host.extend_from_slice(key.x_words());
                z_host.extend_from_slice(key.z_words());
                cre.push(wt.coeff.re);
                cim.push(wt.coeff.im);
                freq.push(wt.freq);
            }
            (x_host, z_host, cre, cim, freq)
        },
    );

    let s = ctx.stream.clone();
    let s = &s;
    // Upload inputs. Allocation and H2D are timed together per buffer group
    // below; the split that matters is alloc-vs-copy, so `alloc_ns` counts the
    // `alloc_zeros` calls (cuMemAlloc + memset) and `upload_ns` the memcpys.
    // Grow the pool once, then borrow. All sizes are known here.
    if phase(
        |p| &mut p.alloc_ns,
        || {
            let b = &mut ctx.bufs;
            ensure(s, &mut b.x, num * w)?;
            ensure(s, &mut b.z, num * w)?;
            ensure(s, &mut b.rx, rx.len())?;
            ensure(s, &mut b.rz, rz.len())?;
            ensure(s, &mut b.cre, num)?;
            ensure(s, &mut b.cim, num)?;
            ensure(s, &mut b.freq, num)?;
            ensure(s, &mut b.ox, 2 * num * w)?;
            ensure(s, &mut b.oz, 2 * num * w)?;
            ensure(s, &mut b.ocre, 2 * num)?;
            ensure(s, &mut b.ocim, 2 * num)?;
            ensure(s, &mut b.ofreq, 2 * num)?;
            ensure(s, &mut b.ovalid, 2 * num)?;
            ensure(s, &mut b.odropped, num)?;
            Some(())
        },
    )
    .is_none()
    {
        return false;
    }
    // Borrow the pool. Uploads write only the live prefix; the tail beyond it
    // is last call's data and is never read — see `Buffers`.
    let bufs = &mut ctx.bufs;
    let (x_dev, z_dev, rx_dev, rz_dev, cre_dev, cim_dev, freq_dev) = (
        bufs.x.as_mut().unwrap(),
        bufs.z.as_mut().unwrap(),
        bufs.rx.as_mut().unwrap(),
        bufs.rz.as_mut().unwrap(),
        bufs.cre.as_mut().unwrap(),
        bufs.cim.as_mut().unwrap(),
        bufs.freq.as_mut().unwrap(),
    );
    if phase(
        |p| &mut p.upload_ns,
        || {
            s.memcpy_htod(&x_host, &mut x_dev.slice_mut(..x_host.len()))
                .and_then(|_| s.memcpy_htod(&z_host, &mut z_dev.slice_mut(..z_host.len())))
                .and_then(|_| s.memcpy_htod(rx, &mut rx_dev.slice_mut(..rx.len())))
                .and_then(|_| s.memcpy_htod(rz, &mut rz_dev.slice_mut(..rz.len())))
                .and_then(|_| s.memcpy_htod(&cre, &mut cre_dev.slice_mut(..cre.len())))
                .and_then(|_| s.memcpy_htod(&cim, &mut cim_dev.slice_mut(..cim.len())))
                .and_then(|_| s.memcpy_htod(&freq, &mut freq_dev.slice_mut(..freq.len())))
                .is_err()
        },
    ) {
        return false;
    }

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
    // Input views, sliced to this call's extent for the same reason the outputs
    // are: a reused buffer is longer than the call needs, and the kernel's
    // bounds must agree with the host's.
    let x_in = x_dev.slice(..num * w);
    let z_in = z_dev.slice(..num * w);
    let rx_in = rx_dev.slice(..rx.len());
    let rz_in = rz_dev.slice(..rz.len());
    let cre_in = cre_dev.slice(..num);
    let cim_in = cim_dev.slice(..num);
    let freq_in = freq_dev.slice(..num);

    // Output views, SLICED to this call's extent. Passing the whole reused
    // buffer would let the kernel's bounds and the host's disagree.
    let (mut ox, mut oz, mut ocre, mut ocim, mut ofreq, mut ovalid, mut odropped) = (
        bufs.ox.as_mut().unwrap().slice_mut(..2 * num * w),
        bufs.oz.as_mut().unwrap().slice_mut(..2 * num * w),
        bufs.ocre.as_mut().unwrap().slice_mut(..2 * num),
        bufs.ocim.as_mut().unwrap().slice_mut(..2 * num),
        bufs.ofreq.as_mut().unwrap().slice_mut(..2 * num),
        bufs.ovalid.as_mut().unwrap().slice_mut(..2 * num),
        bufs.odropped.as_mut().unwrap().slice_mut(..num),
    );

    let func = ctx.func.clone();
    let mut b = s.launch_builder(&func);
    b.arg(&x_in)
        .arg(&z_in)
        .arg(&cre_in)
        .arg(&cim_in)
        .arg(&freq_in)
        .arg(&rx_in)
        .arg(&rz_in)
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
    let launched = phase(
        |p| &mut p.launch_sync_ns,
        || unsafe { b.launch(cfg) }.is_ok() && s.synchronize().is_ok(),
    );
    if !launched {
        return false;
    }

    // Download.
    let Some((ox_h, oz_h, ocre_h, ocim_h, ofreq_h, ovalid_h, odropped_h)) = phase(
        |p| &mut p.download_ns,
        || {
            // Every one of these reads a SLICE, so a reused buffer's tail
            // cannot leak into the merge. See `Buffers`.
            let (Ok(a), Ok(b)) = (s.clone_dtoh(&ox), s.clone_dtoh(&oz)) else {
                return None;
            };
            let (Ok(c), Ok(d)) = (s.clone_dtoh(&ocre), s.clone_dtoh(&ocim)) else {
                return None;
            };
            let (Ok(e), Ok(f)) = (s.clone_dtoh(&ofreq), s.clone_dtoh(&ovalid)) else {
                return None;
            };
            let Ok(g) = s.clone_dtoh(&odropped) else {
                return None;
            };
            Some((a, b, c, d, e, f, g))
        },
    ) else {
        return false;
    };

    // STRUCTURAL CHECK on the buffer pool, not an argument in a comment.
    //
    // Every download above reads a sliced VIEW, so each vector must come back at
    // exactly this call's extent. If a view is ever widened back to the whole
    // reused buffer — the mutation that produced a 47x inflated `dropped_mass`
    // in testing — these lengths change and this fires HERE, rather than a
    // previous call's tail silently folding into a certified bound.
    // Seven integer compares per gate.
    debug_assert_eq!(ox_h.len(), 2 * num * w, "ox download is not prefix-sized");
    debug_assert_eq!(oz_h.len(), 2 * num * w, "oz download is not prefix-sized");
    debug_assert_eq!(ocre_h.len(), 2 * num, "ocre download is not prefix-sized");
    debug_assert_eq!(ocim_h.len(), 2 * num, "ocim download is not prefix-sized");
    debug_assert_eq!(ofreq_h.len(), 2 * num, "ofreq download is not prefix-sized");
    debug_assert_eq!(
        ovalid_h.len(),
        2 * num,
        "ovalid download is not prefix-sized"
    );
    // NOT a debug_assert: this one guards a CERTIFIED bound, so it holds in
    // release too.
    assert_eq!(
        odropped_h.len(),
        num,
        "odropped download is not prefix-sized — a stale tail would inflate \
         dropped_mass, which callers treat as a certified bound"
    );

    // Merge children back into a fresh sum (add_weighted: sum coeffs, min freq).
    let out = phase(
        |p| &mut p.merge_ns,
        || {
            // Sized at `num`, not the 2n worst case — see the CPU `branch()`
            // for the measurement. 2n measured SLOWER than no pre-sizing at
            // all: an over-sized table is a sparser table, and the locality
            // costs more than the rehashes it saves.
            let mut out = PauliSum::with_capacity(num);
            // Slice to the LIVE length. `odropped` is `num` long today, but under
            // buffer reuse it would be longer, and summing a stale tail silently
            // inflates `dropped_mass` — the certified truncation bound.
            out.dropped_mass = sum.dropped_mass + odropped_h[..num].iter().sum::<f64>();
            for slot in 0..(2 * num) {
                if ovalid_h[slot] == 0 {
                    continue;
                }
                let base = slot * w;
                let key = PauliKey::from_words(&ox_h[base..base + w], &oz_h[base..base + w], n);
                out.add_weighted(
                    key,
                    Complex64::new(ocre_h[slot], ocim_h[slot]),
                    ofreq_h[slot],
                );
            }
            out
        },
    );

    if profiling() {
        PHASES.with(|p| p.borrow_mut().calls += 1);
    }
    *sum = out;
    GPU_BRANCHES.fetch_add(1, Ordering::Relaxed);
    true
}
