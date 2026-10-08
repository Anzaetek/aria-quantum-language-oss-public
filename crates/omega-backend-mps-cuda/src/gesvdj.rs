//! cuSOLVER complex SVD wrapper: the EXACT thin SVD `cusolverDnZgesvd`
//! (Householder bidiagonalisation + implicit-shift QR), computed in full on the
//! device and truncated on the host by the CPU kernel's own rule.
//!
//! The module keeps its historical name (`gesvdj`); it has not called the
//! Jacobi solver for a long time, and until 2026-09-30 it did not call an exact
//! solver at all.
//!
//! # Why not `Zgesvda` (the defect this replaces)
//!
//! This file used to call `cusolverDnZgesvdaStridedBatched` — cuSOLVER's
//! *approximate*, rank-limited SVD — asking it for only
//! `rank = min(max_rank, k)` singular triplets. Its rank-r output is not the
//! exact top-r SVD ("accuracy is bounded by the implicit randomized
//! projection", as the old comment here put it), so it agreed with the CPU
//! Jacobi kernel only when nothing was truncated and diverged whenever
//! something was. Measured (`STATUS.md` §5.16, 14 qubits, depth 24, χ = 32):
//! `⟨Z⟩` = −1.9921e-9 through gesvda against −4.1442e-15 on the CPU, with the
//! exact χ ≥ 128 rows agreeing at 1e-15; and the per-run `discarded_weight`
//! off by 8% relative at depth 12, χ = 32. The truncation certificate is a
//! bound and must be fed identically whatever SVD runs, so an approximate
//! solver is the wrong tool here regardless of its speed.
//!
//! # Truncation and the certificate (`discarded_weight`)
//!
//! gesvd returns all `k = min(m, n)` singular values, so the host applies
//! exactly `omega_backend_mps::svd::truncated_svd_flat`'s rule — order the σ
//! largest-first, keep at most `max_rank` strictly above `threshold`, never
//! fewer than 1 — and the certificate is that kernel's quantity too: Σσ² over
//! the σ it dropped. (The gesvda path had to use ‖A‖²_F − Σσ²_kept because
//! the tail was never fetched; that identity is now a `debug_assert!`, not the
//! formula. It was also noise-dominated when nothing was truncated: 1.26e-14
//! against the CPU's 1.07e-28 at χ = 128.)
//!
//! Because CUDA only swaps the *SVD* (via `MpsBackend::with_svd_fn`) and leaves
//! the split/accumulation in `Mps::apply_2q_with_svd_flat`, the per-run
//! accumulation onto `Mps::discarded_weight` works automatically once this
//! field is correct — there is NO CUDA analogue of the Metal
//! `mps_apply_2q_metal` bypass to patch.
//!
//! # Failure → CPU fallback, with a reason
//!
//! Every failure returns a [`SvdFallback`] reason instead of a bare `None`, so
//! `cuda_svd_flat` can count it (`crate::cuda_svd_dispatch`). `info != 0`
//! (non-convergence of the bidiagonal QR, or a bad argument) is a fallback,
//! not a result.
//!
//! # Matrix layout
//!
//! cuSOLVER is column-major (LAPACK convention). The host-side input
//! is a flat row-major `&[Complex64]` (m, n, stride) — what the MPS
//! `Theta'` reshape produces directly. We transpose into a flat
//! `Vec<f64>` (interleaved re/im) when copying to device. On the way
//! back, U (m×k, ld m) and Vᴴ (k×n, ld k) come out column-major; we
//! convert to row-major `Vec<Complex64>` for the [`SvdResultFlat`] output.
//!
//! # m vs n
//!
//! `gesvd` requires `m >= n`. When the host matrix has `m < n`, we SVD `A^H`
//! instead (which is m'=n, n'=m so m' >= n'), then swap the roles of U and V:
//!
//!   A = U Σ V^H  ⇒  A^H = V Σ U^H
//!
//! So if `gesvd(A^H) = U' Σ V'^H`, then `U_A = (V'^H)^H` and `V_A^H = U'^H`.
//!
//! # Per-shape buffer cache
//!
//! Per-shape device-buffer reuse is the main host-side amortisation:
//! a brickwall MPS circuit at constant χ runs hundreds of SVDs of
//! identical (m, n) — we don't want to alloc / dealloc the device
//! buffers + workspace per call. That's what [`ShapeCache`] does; it is keyed
//! on (m, n) alone because the full thin SVD does not depend on `max_rank`.

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::cusolver::{safe::DnHandle, sys as cusolver_sys};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtrMut};
use num_complex::Complex64;

use omega_backend_mps::mps::Mps;
use omega_backend_mps::svd::{SvdResult, SvdResultFlat};

/// Why a device SVD did not produce a result, so the CPU fallback that follows
/// can be counted and named (`crate::CudaSvdDispatch::last_fallback`).
pub(crate) type SvdFallback = &'static str;

/// A device allocation (matrix, factors, workspace or `info`) failed.
pub(crate) const FALLBACK_ALLOC: SvdFallback = "device allocation failed";
/// A host↔device copy, or the stream synchronize that completes it, failed.
pub(crate) const FALLBACK_MEMCPY: SvdFallback = "host/device memcpy failed";
/// `cusolverDnZgesvd` or its `_bufferSize` query returned a non-success status.
pub(crate) const FALLBACK_SOLVER_STATUS: SvdFallback = "cuSOLVER gesvd returned an error status";
/// The solve ran but reported `info != 0`: the bidiagonal QR did not converge
/// (`info > 0`) or an argument was rejected (`info < 0`).
pub(crate) const FALLBACK_INFO: SvdFallback = "cuSOLVER gesvd reported info != 0 (non-convergence)";

/// Reusable cuSOLVER + driver context for repeated SVD calls.
///
/// One-shot callers go through [`try_cuda_svd`] which builds and tears
/// this down per call. The hot-path MPS truncation loop reuses one
/// context across many truncations to amortise handle / stream init —
/// see [`CudaSvdContext::truncated_svd_flat`].
pub struct CudaSvdContext {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    handle: DnHandle,
    cache: RefCell<Option<ShapeCache>>,
}

/// Per-shape buffer pool, reused when the same solver shape (m, n) recurs,
/// which dominates a brickwall MPS circuit at constant χ. `k = min(m, n) = n`
/// singular triplets are always computed (the full thin SVD).
struct ShapeCache {
    m: usize,
    n: usize,
    k: usize,
    lwork: i32,
    a: CudaSlice<f64>,
    s: CudaSlice<f64>,
    u: CudaSlice<f64>,
    vt: CudaSlice<f64>,
    work: CudaSlice<f64>,
    rwork: CudaSlice<f64>,
    info: CudaSlice<i32>,
}

impl CudaSvdContext {
    /// Build a context bound to CUDA device 0. Returns `None` if the
    /// driver init or cuSOLVER handle create fails — caller falls back
    /// to CPU.
    pub fn new() -> Option<Self> {
        Self::try_new().ok()
    }

    /// Build the context, converting a **panic during symbol resolution into a
    /// typed value** (3b.3 R1).
    ///
    /// `DnHandle::new` is where `cudarc` resolves the cuSOLVER entry points it
    /// declares, and on CUDA 13 that `dlsym` fails for `cusolverDnGeqrf` —
    /// a symbol we never call, absent from the *binding table* rather than
    /// from our call graph. `cudarc` 0.19 `.expect()`s it, so the process
    /// aborted before `new()`'s `Option` could say anything.
    ///
    /// `catch_unwind` is the right tool and not a workaround: the panic is a
    /// Rust panic from `.expect()`, not a foreign exception, and the workspace
    /// does not set `panic = "abort"`. The alternative — probing every symbol
    /// `cudarc` might resolve, ahead of it — would have to track that crate's
    /// binding table version by version.
    ///
    /// The panic hook is silenced for the duration so a *handled* condition
    /// does not print a backtrace that reads like a crash.
    ///
    /// **Unverified here.** This arm needs a CUDA 13 host; the classification
    /// it feeds is tested on every platform (see `availability`).
    pub fn try_new() -> Result<Self, crate::CudaSvdUnavailable> {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let attempt = std::panic::catch_unwind(|| {
            let ctx = CudaContext::new(0).map_err(|e| format!("{e}"))?;
            let stream = ctx.default_stream();
            let handle = DnHandle::new(stream.clone()).map_err(|e| format!("{e}"))?;
            Ok::<_, String>((ctx, stream, handle))
        });
        std::panic::set_hook(previous);

        match attempt {
            Ok(Ok((ctx, stream, handle))) => Ok(Self {
                ctx,
                stream,
                handle,
                cache: RefCell::new(None),
            }),
            // Returned an error rather than panicking: no device, or a driver
            // that declined. Distinct from a missing symbol and reported so.
            Ok(Err(_e)) => Err(crate::CudaSvdUnavailable::NoDevice),
            // Panicked. The payload carries the diagnosis; `classify_panic`
            // decides whether it is the CUDA 13 shape or something unrelated,
            // and is deliberately conservative about claiming the former.
            Err(payload) => {
                let text = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "panicked with a non-string payload".to_string());
                Err(crate::classify_panic(&text))
            }
        }
    }

    /// Build (or reuse) the per-shape buffer + workspace cache for the solver
    /// shape `(m, n)`, `m >= n`. After this returns `Ok`, `self.cache` holds a
    /// `ShapeCache` ready for `Zgesvd`. On failure the previous cache is left
    /// untouched and the reason is returned.
    fn ensure_cache(&self, m: usize, n: usize) -> Result<(), SvdFallback> {
        debug_assert!(m >= n);
        let mut slot = self.cache.borrow_mut();
        if let Some(c) = slot.as_ref() {
            if c.m == m && c.n == n {
                return Ok(());
            }
        }
        // Different shape (or first call) — rebuild.
        let k = n; // min(m, n), since m >= n
        let stream = &self.stream;
        let alloc = |len: usize| stream.alloc_zeros::<f64>(len).map_err(|_| FALLBACK_ALLOC);
        let a = alloc(2 * m * n)?;
        let s = alloc(k)?;
        let u = alloc(2 * m * k)?;
        let vt = alloc(2 * k * n)?;
        // `rwork` holds the unconverged superdiagonal: min(m, n) − 1 reals.
        let rwork = alloc(k.max(2) - 1)?;
        let info = stream.alloc_zeros::<i32>(1).map_err(|_| FALLBACK_ALLOC)?;

        let mut lwork: i32 = 0;
        unsafe {
            cusolver_sys::cusolverDnZgesvd_bufferSize(
                self.handle.cu(),
                m as i32,
                n as i32,
                &mut lwork as *mut _,
            )
            .result()
            .map_err(|_| FALLBACK_SOLVER_STATUS)?;
        }
        // `lwork` is in cuDoubleComplex elements: two f64 each.
        let work = alloc(2 * lwork.max(1) as usize)?;

        *slot = Some(ShapeCache {
            m,
            n,
            k,
            lwork,
            a,
            s,
            u,
            vt,
            work,
            rwork,
            info,
        });
        Ok(())
    }

    /// Apply a two-qubit gate at site `q` of `mps`, routing the bond
    /// truncation SVD through cuSOLVER. Reuses the context's
    /// cuSOLVER handle across many calls — the hot-path entry point
    /// for the MPS GPU bench.
    ///
    /// Falls back to the CPU SVD when the GPU SVD fails (e.g. an OOM /
    /// convergence-failed solve). Each split is counted in
    /// [`crate::cuda_svd_dispatch`] — on the GPU, or as a fallback with its
    /// reason — so the fallback is transparent to the result but not
    /// invisible.
    pub fn apply_2q(&self, mps: &mut Mps, q: usize, gate: &[Complex64; 16]) {
        mps.apply_2q_with_svd_flat(
            q,
            gate,
            |matrix, m, n, stride, max_rank, threshold| match self
                .truncated_svd_flat_reasoned(matrix, m, n, stride, max_rank, threshold)
            {
                Ok(out) => {
                    crate::note_gpu_call();
                    out
                }
                Err(reason) => {
                    crate::note_cpu_fallback(reason);
                    omega_backend_mps::svd::truncated_svd_flat(
                        matrix, m, n, stride, max_rank, threshold,
                    )
                }
            },
        );
    }

    /// Compute the truncated SVD of `matrix` via cuSOLVER `Zgesvd` (exact).
    /// Input is row-major flat (`matrix[i * stride + j]`); output is
    /// row-major flat ([`SvdResultFlat`]).
    ///
    /// Truncation matches `omega_backend_mps::svd::truncated_svd_flat`:
    /// keep at most `max_rank` singular values strictly above
    /// `threshold`, never returning fewer than 1; `discarded_weight` is
    /// Σσ² over the dropped singular values.
    ///
    /// `None` on any device failure. [`crate::cuda_svd_flat`] uses the
    /// reasoned form of this call so the failure is counted and named.
    pub fn truncated_svd_flat(
        &self,
        matrix: &[Complex64],
        m_host: usize,
        n_host: usize,
        stride: usize,
        max_rank: usize,
        threshold: f64,
    ) -> Option<SvdResultFlat> {
        self.truncated_svd_flat_reasoned(matrix, m_host, n_host, stride, max_rank, threshold)
            .ok()
    }

    /// [`Self::truncated_svd_flat`], returning WHY the device path failed.
    pub(crate) fn truncated_svd_flat_reasoned(
        &self,
        matrix: &[Complex64],
        m_host: usize,
        n_host: usize,
        stride: usize,
        max_rank: usize,
        threshold: f64,
    ) -> Result<SvdResultFlat, SvdFallback> {
        if m_host == 0 || n_host == 0 {
            return Ok(SvdResultFlat {
                u: vec![],
                s: vec![],
                vt: vec![],
                m: m_host,
                n: n_host,
                discarded_weight: 0.0,
            });
        }
        debug_assert!(stride >= n_host);

        // cuSOLVER `gesvd` requires m >= n. If the host matrix is
        // wide, we SVD A^H instead and swap U/V at the end.
        let transpose = m_host < n_host;
        let (m, n) = if transpose {
            (n_host, m_host)
        } else {
            (m_host, n_host)
        };

        // Pack the host matrix into column-major (re, im) f64 pairs.
        // a_host[col*m + row] in elements; ×2 in f64s.
        let mut a_host: Vec<f64> = vec![0.0; 2 * m * n];
        for row_h in 0..m_host {
            let row_base = row_h * stride;
            for col_h in 0..n_host {
                let z = matrix[row_base + col_h];
                let (i, j, val) = if transpose {
                    // A^H[col_h, row_h] = conj(matrix[row_h][col_h])
                    (col_h, row_h, z.conj())
                } else {
                    (row_h, col_h, z)
                };
                let idx = (j * m + i) * 2;
                a_host[idx] = val.re;
                a_host[idx + 1] = val.im;
            }
        }

        // Acquire (or rebuild) the per-shape buffer cache. When (m, n)
        // matches the previous call, this is a no-op — the same device
        // buffers and `lwork` workspace are reused.
        let stream = &self.stream;
        self.ensure_cache(m, n)?;

        let mut cache_ref = self.cache.borrow_mut();
        let cache = cache_ref.as_mut().expect("ensure_cache populated");
        let k = cache.k;

        stream
            .memcpy_htod(&a_host, &mut cache.a)
            .map_err(|_| FALLBACK_MEMCPY)?;

        // Thin SVD: U is m×k, Vᴴ is k×n.
        let job_thin = b'S' as core::ffi::c_schar;
        {
            let (a_ptr, _a_sync) = cache.a.device_ptr_mut(stream);
            let (s_ptr, _s_sync) = cache.s.device_ptr_mut(stream);
            let (u_ptr, _u_sync) = cache.u.device_ptr_mut(stream);
            let (vt_ptr, _vt_sync) = cache.vt.device_ptr_mut(stream);
            let (work_ptr, _work_sync) = cache.work.device_ptr_mut(stream);
            let (rwork_ptr, _rwork_sync) = cache.rwork.device_ptr_mut(stream);
            let (info_ptr, _info_sync) = cache.info.device_ptr_mut(stream);

            unsafe {
                cusolver_sys::cusolverDnZgesvd(
                    self.handle.cu(),
                    job_thin,
                    job_thin,
                    m as i32,
                    n as i32,
                    a_ptr as *mut _,
                    m as i32,
                    s_ptr as *mut _,
                    u_ptr as *mut _,
                    m as i32,
                    vt_ptr as *mut _,
                    k as i32,
                    work_ptr as *mut _,
                    cache.lwork,
                    rwork_ptr as *mut _,
                    info_ptr as *mut _,
                )
                .result()
                .map_err(|_| FALLBACK_SOLVER_STATUS)?;
            }
        }

        // Pull all k singular triplets back to host.
        let mut s_host = vec![0.0f64; k];
        let mut u_host = vec![0.0f64; 2 * m * k];
        let mut vt_host = vec![0.0f64; 2 * k * n];
        let mut info_host = [0i32; 1];
        let copied = stream
            .memcpy_dtoh(&cache.s, &mut s_host)
            .and_then(|_| stream.memcpy_dtoh(&cache.u, &mut u_host))
            .and_then(|_| stream.memcpy_dtoh(&cache.vt, &mut vt_host))
            .and_then(|_| stream.memcpy_dtoh(&cache.info, &mut info_host))
            .and_then(|_| stream.synchronize());
        drop(cache_ref);
        copied.map_err(|_| FALLBACK_MEMCPY)?;

        // info != 0 → solve failed; fall back to CPU. Positive
        // info means non-convergence; negative means a bad argument.
        if info_host[0] != 0 {
            return Err(FALLBACK_INFO);
        }

        // Truncate exactly as the CPU kernel does. gesvd already returns σ
        // descending; the explicit ordering is the CPU kernel's own, kept so a
        // tie or a NaN is handled identically rather than "probably the same".
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&i, &j| s_host[j].total_cmp(&s_host[i]));
        let kept = order
            .iter()
            .take(max_rank)
            .take_while(|&&i| s_host[i] > threshold)
            .count()
            .max(1);
        // Truncation certificate: Σσ² over exactly the σ dropped — the CPU
        // kernel's quantity, now computable because all k σ were fetched.
        let discarded_weight: f64 = order[kept..].iter().map(|&i| s_host[i] * s_host[i]).sum();
        let s_out: Vec<f64> = order[..kept].iter().map(|&i| s_host[i]).collect();

        // Cross-check against the Frobenius identity Σσ²_all = ‖A‖²_F, which
        // is what the gesvda path had to use as its formula.
        #[cfg(debug_assertions)]
        {
            let mut frob_sq = 0.0;
            for row_h in 0..m_host {
                let base = row_h * stride;
                for col_h in 0..n_host {
                    frob_sq += matrix[base + col_h].norm_sqr();
                }
            }
            let kept_sq: f64 = s_out.iter().map(|&sv| sv * sv).sum();
            debug_assert!(
                (kept_sq + discarded_weight - frob_sq).abs() <= 1e-9 * frob_sq.max(1e-300),
                "gesvd not norm-preserving: kept {kept_sq} + dropped {discarded_weight} \
                 vs ‖A‖²_F {frob_sq}"
            );
        }

        // Solver factors, column-major: U_s[r, c] at (c*m + r), Vᴴ_s[r, c] at
        // (c*k + r). Host A = U Σ Vᴴ:
        //   !transpose: U[i, c] = U_s[i, c],        Vᴴ[c, j] = Vᴴ_s[c, j]
        //    transpose: U[i, c] = conj(Vᴴ_s[c, i]), Vᴴ[c, j] = conj(U_s[j, c])
        let at = |buf: &[f64], idx: usize| Complex64::new(buf[2 * idx], buf[2 * idx + 1]);
        let mut u_out = vec![Complex64::new(0.0, 0.0); m_host * kept];
        let mut vt_out = vec![Complex64::new(0.0, 0.0); kept * n_host];
        for (col, &src) in order[..kept].iter().enumerate() {
            for row in 0..m_host {
                u_out[row * kept + col] = if transpose {
                    at(&vt_host, row * k + src).conj()
                } else {
                    at(&u_host, src * m + row)
                };
            }
            for j in 0..n_host {
                vt_out[col * n_host + j] = if transpose {
                    at(&u_host, src * m + j).conj()
                } else {
                    at(&vt_host, j * k + src)
                };
            }
        }

        Ok(SvdResultFlat {
            u: u_out,
            s: s_out,
            vt: vt_out,
            m: m_host,
            n: n_host,
            discarded_weight,
        })
    }

    /// Nested-Vec adaptor for the flat SVD. Retained for back-compat
    /// with the legacy `truncated_svd(&[Vec<Complex64>], ...)` callers
    /// (mostly tests and one-shot users via `try_cuda_svd`).
    pub fn truncated_svd(
        &self,
        matrix: &[Vec<Complex64>],
        max_rank: usize,
        threshold: f64,
    ) -> Option<SvdResult> {
        let m = matrix.len();
        if m == 0 {
            return Some(SvdResult {
                u: vec![],
                s: vec![],
                vt: vec![],
            });
        }
        let n = matrix[0].len();
        if n == 0 {
            return Some(SvdResult {
                u: vec![],
                s: vec![],
                vt: vec![],
            });
        }
        let mut flat = Vec::with_capacity(m * n);
        for row in matrix {
            flat.extend_from_slice(row);
        }
        self.truncated_svd_flat(&flat, m, n, n, max_rank, threshold)
            .map(Into::into)
    }
}

/// One-shot path: build a context, run one SVD, tear down. Used by
/// the [`crate::cuda_truncated_svd`] convenience wrapper. Returns
/// `None` on any CUDA / cuSOLVER error → caller falls back to CPU.
pub(crate) fn try_cuda_svd(
    matrix: &[Vec<Complex64>],
    max_rank: usize,
    threshold: f64,
) -> Option<SvdResult> {
    let ctx = CudaSvdContext::new()?;
    ctx.truncated_svd(matrix, max_rank, threshold)
}
