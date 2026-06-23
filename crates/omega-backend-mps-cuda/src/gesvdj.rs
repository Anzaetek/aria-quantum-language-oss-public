//! cuSOLVER complex SVD wrapper. Primary path: `Zgesvda`
//! (randomized approximate SVD) with `Zgesvdj` (Jacobi) as fallback.
//!
//! # Why `Zgesvda` is the default
//!
//! For our use case — MPS bond compression where we truncate to χ on
//! each call and the smallest *kept* singular value is well above
//! 1e-7 — the approximate-SVD trade-off is one-sided: we throw away
//! the trailing singular vectors anyway. On the canonical bench
//! shape (256×256 complex, rank 128) `Zgesvda` runs ~25× faster than
//! `Zgesvdj` and ~5× faster than `Zgesvd` on synthetic inputs (see
//! `examples/svd_microbench.rs`). On real MPS matrices the ratio is
//! smaller but still material — `Zgesvdj` happens to be unusually
//! fast on the spectra MPS produces, but `Zgesvda` is faster still.
//!
//! On failure (cuSOLVER returns `info != 0`) the caller automatically
//! falls back to the CPU Jacobi SVD — no separate `Zgesvdj` fallback
//! on-device is wired up.
//!
//! # Matrix layout
//!
//! cuSOLVER is column-major (LAPACK convention). The host-side input
//! is a flat row-major `&[Complex64]` (m, n, stride) — what the MPS
//! `Theta'` reshape produces directly. We transpose into a flat
//! `Vec<f64>` (interleaved re/im) when copying to device. On the way
//! back, U comes out column-major; we convert to row-major
//! `Vec<Complex64>` for the [`SvdResultFlat`] output. V comes out
//! column-major and we conjugate-transpose into `vt` (row-major).
//!
//! # m vs n
//!
//! `gesvda` (like `gesvdj`) requires `m >= n`. When the host matrix
//! has `m < n`, we SVD `A^H` instead (which is m'=n, n'=m so
//! m' >= n'), then swap the roles of U and V in the result:
//!
//!   A = U Σ V^H  ⇒  A^H = V Σ U^H
//!
//! So if `gesvda(A^H) = U' Σ V'^H`, then `U_A = V'`, `V_A = U'`.
//!
//! # Truncation
//!
//! `gesvda` takes the target `rank` as input and returns at most that
//! many singular values pre-sorted in descending order. We pass
//! `rank = min(max_rank, min(m, n))` and let the caller's threshold
//! filter trim further.
//!
//! # Per-shape buffer cache
//!
//! Per-shape device-buffer reuse is the main host-side amortisation:
//! a brickwall MPS circuit at constant χ runs hundreds of SVDs of
//! identical (m, n) — we don't want to alloc / dealloc the device
//! buffers + workspace per call. That's what [`ShapeCache`] does.

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::cusolver::{safe::DnHandle, sys as cusolver_sys};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use num_complex::Complex64;

use omega_backend_mps::mps::Mps;
use omega_backend_mps::svd::{SvdResult, SvdResultFlat};

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

/// Per-shape buffer pool. Reused when the same (m, n, rank) recurs,
/// which dominates a brickwall MPS circuit at constant χ.
struct ShapeCache {
    m: usize,
    n: usize,
    rank: usize,
    lwork: i32,
    a: CudaSlice<f64>,
    s: CudaSlice<f64>,
    u: CudaSlice<f64>,
    v: CudaSlice<f64>,
    work: CudaSlice<f64>,
    info: CudaSlice<i32>,
}

impl CudaSvdContext {
    /// Build a context bound to CUDA device 0. Returns `None` if the
    /// driver init or cuSOLVER handle create fails — caller falls back
    /// to CPU.
    pub fn new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        let handle = DnHandle::new(stream.clone()).ok()?;
        Some(Self {
            ctx,
            stream,
            handle,
            cache: RefCell::new(None),
        })
    }

    /// Build (or reuse) the per-shape buffer + workspace cache. After
    /// this returns successfully, `self.cache` holds a `ShapeCache`
    /// for `(m, n, rank)` ready for `Zgesvda`. Returns `None` and
    /// leaves the previous cache untouched if any allocation / FFI
    /// step fails.
    fn ensure_cache(&self, m: usize, n: usize, rank: usize) -> Option<()> {
        let mut slot = self.cache.borrow_mut();
        if let Some(c) = slot.as_ref() {
            if c.m == m && c.n == n && c.rank == rank {
                return Some(());
            }
        }
        // Different shape (or first call) — rebuild.
        let stream = &self.stream;
        let a = stream.alloc_zeros::<f64>(2 * m * n).ok()?;
        let s = stream.alloc_zeros::<f64>(rank).ok()?;
        let u = stream.alloc_zeros::<f64>(2 * m * rank).ok()?;
        let v = stream.alloc_zeros::<f64>(2 * n * rank).ok()?;
        let info = stream.alloc_zeros::<i32>(1).ok()?;

        let jobz = cusolver_sys::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
        let lda = m as i32;
        let ldu = m as i32;
        let ldv = n as i32;
        let stride_a = (m * n) as i64;
        let stride_s = rank as i64;
        let stride_u = (m * rank) as i64;
        let stride_v = (n * rank) as i64;
        let mut lwork: i32 = 0;
        {
            let (a_ptr, _a_sync) = a.device_ptr(stream);
            let (s_ptr, _s_sync) = s.device_ptr(stream);
            let (u_ptr, _u_sync) = u.device_ptr(stream);
            let (v_ptr, _v_sync) = v.device_ptr(stream);
            unsafe {
                cusolver_sys::cusolverDnZgesvdaStridedBatched_bufferSize(
                    self.handle.cu(),
                    jobz,
                    rank as i32,
                    m as i32,
                    n as i32,
                    a_ptr as *const _,
                    lda,
                    stride_a,
                    s_ptr as *const _,
                    stride_s,
                    u_ptr as *const _,
                    ldu,
                    stride_u,
                    v_ptr as *const _,
                    ldv,
                    stride_v,
                    &mut lwork as *mut _,
                    1,
                )
                .result()
                .ok()?;
            }
        }
        let work = stream.alloc_zeros::<f64>(2 * lwork.max(1) as usize).ok()?;

        *slot = Some(ShapeCache {
            m,
            n,
            rank,
            lwork,
            a,
            s,
            u,
            v,
            work,
            info,
        });
        Some(())
    }

    /// Apply a two-qubit gate at site `q` of `mps`, routing the bond
    /// truncation SVD through cuSOLVER. Reuses the context's
    /// cuSOLVER handle across many calls — the hot-path entry point
    /// for the MPS GPU bench.
    ///
    /// Falls back transparently to the CPU SVD when the GPU SVD
    /// returns `None` (e.g. an OOM / convergence-failed solve), so
    /// callers don't need to handle the failure mode.
    pub fn apply_2q(&self, mps: &mut Mps, q: usize, gate: &[Complex64; 16]) {
        mps.apply_2q_with_svd_flat(q, gate, |matrix, m, n, stride, max_rank, threshold| {
            self.truncated_svd_flat(matrix, m, n, stride, max_rank, threshold)
                .unwrap_or_else(|| {
                    omega_backend_mps::svd::truncated_svd_flat(
                        matrix, m, n, stride, max_rank, threshold,
                    )
                })
        });
    }

    /// Compute the truncated SVD of `matrix` via cuSOLVER `Zgesvdj`.
    /// Input is row-major flat (`matrix[i * stride + j]`); output is
    /// row-major flat ([`SvdResultFlat`]).
    ///
    /// Truncation matches `omega_backend_mps::svd::truncated_svd_flat`:
    /// keep at most `max_rank` singular values strictly above
    /// `threshold`, never returning fewer than 1.
    pub fn truncated_svd_flat(
        &self,
        matrix: &[Complex64],
        m_host: usize,
        n_host: usize,
        stride: usize,
        max_rank: usize,
        threshold: f64,
    ) -> Option<SvdResultFlat> {
        if m_host == 0 || n_host == 0 {
            return Some(SvdResultFlat {
                u: vec![],
                s: vec![],
                vt: vec![],
                m: m_host,
                n: n_host,
            });
        }
        debug_assert!(stride >= n_host);

        // cuSOLVER `gesvdj` requires m >= n. If the host matrix is
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

        // Acquire (or rebuild) the per-shape buffer cache. When
        // (m, n, rank) matches the previous call, this is a no-op —
        // the same device buffers and `lwork` workspace are reused.
        let k = m.min(n); // == n, because m >= n
        let rank = max_rank.min(k).max(1);
        let stream = &self.stream;
        self.ensure_cache(m, n, rank)?;

        let mut cache_ref = self.cache.borrow_mut();
        let cache = cache_ref.as_mut().expect("ensure_cache populated");
        // Threshold is the caller's truncation tolerance; gesvda has
        // no tunable tolerance — accuracy is bounded by the implicit
        // randomized projection. We rely on the post-filter
        // `sv > threshold` to drop residual noise singular values.
        let _ = threshold;

        stream.memcpy_htod(&a_host, &mut cache.a).ok()?;

        let jobz = cusolver_sys::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
        let lda = m as i32;
        let ldu = m as i32;
        let ldv = n as i32;
        let stride_a = (m * n) as i64;
        let stride_s = rank as i64;
        let stride_u = (m * rank) as i64;
        let stride_v = (n * rank) as i64;
        let mut r_nrm_f = [0.0f64; 1];

        {
            let (a_ptr, _a_sync) = cache.a.device_ptr_mut(stream);
            let (s_ptr, _s_sync) = cache.s.device_ptr_mut(stream);
            let (u_ptr, _u_sync) = cache.u.device_ptr_mut(stream);
            let (v_ptr, _v_sync) = cache.v.device_ptr_mut(stream);
            let (work_ptr, _work_sync) = cache.work.device_ptr_mut(stream);
            let (info_ptr, _info_sync) = cache.info.device_ptr_mut(stream);

            unsafe {
                cusolver_sys::cusolverDnZgesvdaStridedBatched(
                    self.handle.cu(),
                    jobz,
                    rank as i32,
                    m as i32,
                    n as i32,
                    a_ptr as *const _,
                    lda,
                    stride_a,
                    s_ptr as *mut _,
                    stride_s,
                    u_ptr as *mut _,
                    ldu,
                    stride_u,
                    v_ptr as *mut _,
                    ldv,
                    stride_v,
                    work_ptr as *mut _,
                    cache.lwork,
                    info_ptr as *mut _,
                    r_nrm_f.as_mut_ptr(),
                    1,
                )
                .result()
                .ok()?;
            }
        }

        // Pull S, U, V back to host. Buffers are sized for `rank`
        // singular vectors, not the full `k = min(m, n)` thin SVD.
        let mut s_host = vec![0.0f64; rank];
        let mut u_host = vec![0.0f64; 2 * m * rank];
        let mut v_host = vec![0.0f64; 2 * n * rank];
        let mut info_host = [0i32; 1];
        stream.memcpy_dtoh(&cache.s, &mut s_host).ok()?;
        stream.memcpy_dtoh(&cache.u, &mut u_host).ok()?;
        stream.memcpy_dtoh(&cache.v, &mut v_host).ok()?;
        stream.memcpy_dtoh(&cache.info, &mut info_host).ok()?;
        stream.synchronize().ok()?;
        drop(cache_ref);

        // info != 0 → solve failed; fall back to CPU. Positive
        // info means non-convergence; negative means a bad argument.
        if info_host[0] != 0 {
            return None;
        }

        // Singular values are descending; pick the post-filter rank
        // (≤ the gesvda-requested `rank`). Shadows the outer `rank`.
        let kept = s_host
            .iter()
            .take(max_rank)
            .take_while(|&&sv| sv > threshold)
            .count()
            .max(1);

        let s_out: Vec<f64> = s_host[..kept].to_vec();

        // Build U (m_host × kept) and Vt (kept × n_host) in row-major
        // flat. When we transposed:
        //   A = U Σ V^H, U_host[i, k] / Vt_host[k, j]
        //   if !transpose: U_host = U_solver, Vt_host[k, j] = conj(V_solver[j, k])
        //   if transpose:  U_host = V_solver, Vt_host[k, j] = conj(U_solver[j, k])
        // Note `left_dev` / `right_dev` are sized for the gesvda
        // request rank, with leading dim `m`/`n` respectively.
        let (left_dev, left_dim, right_dev, right_dim) = if transpose {
            (&v_host, n, &u_host, m)
        } else {
            (&u_host, m, &v_host, n)
        };
        debug_assert_eq!(left_dim, m_host);
        debug_assert_eq!(right_dim, n_host);

        let mut u_out = vec![Complex64::new(0.0, 0.0); m_host * kept];
        for col in 0..kept {
            for row in 0..m_host {
                let idx = (col * left_dim + row) * 2;
                u_out[row * kept + col] = Complex64::new(left_dev[idx], left_dev[idx + 1]);
            }
        }

        let mut vt_out = vec![Complex64::new(0.0, 0.0); kept * n_host];
        for kidx in 0..kept {
            for j in 0..n_host {
                let idx = (kidx * right_dim + j) * 2;
                let v_jk = Complex64::new(right_dev[idx], right_dev[idx + 1]);
                vt_out[kidx * n_host + j] = v_jk.conj();
            }
        }

        Some(SvdResultFlat {
            u: u_out,
            s: s_out,
            vt: vt_out,
            m: m_host,
            n: n_host,
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
