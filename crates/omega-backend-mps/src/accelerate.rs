//! Truncated SVD for MPS bond compression through Accelerate's LAPACK
//! `zgesdd` — the CPU-side half of `STATUS.md` §5 item 16, in f64 throughout.
//!
//! # Why this, and not a Metal kernel
//!
//! Item 16 measured the SVD at 90-97% of MPS evolution and then measured the
//! GPU answer to it: on an RTX PRO 6000 Blackwell, cuSOLVER loses on six of
//! eight shapes. CUDA is the best case for a GPU SVD here (native f64,
//! sub-microsecond dispatch, mature Jacobi) and Metal starts strictly worse —
//! a ~100 µs dispatch floor against ~12 700 sub-sweep dispatches per SVD at
//! n = 256, and no f64 against a 1e-10 forward tolerance. So the entry's own
//! conclusion is that the remaining prize is on the CPU, and names this:
//! "Accelerate/LAPACK on the CPU side is the cheaper first step and keeps
//! f64."
//!
//! Accelerate is part of the base macOS SDK, so unlike the `metal` and `cuda`
//! arms there is no optional dependency to gate and nothing to probe at
//! runtime: `cfg(target_os = "macos")` is the whole availability question, and
//! the tests below therefore run on every macOS build of this crate rather
//! than only under an opt-in feature.
//!
//! # Why this lives here and not in `omega-backend-mps-metal`
//!
//! It was written there, beside the Metal θ-contraction, because that is where
//! the item it belongs to was being worked. That home made it reachable only
//! through `omega-cli`'s `metal` feature — and that feature exists to keep a
//! stock `cargo build` free of GPU crates, which is a question about
//! `dep:metal`, not about a framework in the base SDK. So the gate was wrong by
//! construction: it withheld a 4-8x speedup from every default macOS build for
//! a reason that did not apply to it. Rehomed 2026-10-02 as the DEFAULT macOS
//! kernel; see [`crate::SvdKernel`] for the selection and
//! `STATUS.md` §5 item 16 for the decision and what it costs.
//!
//! # Why `zgesdd` beats the one-sided Jacobi kernel on work, not on tuning
//!
//! [`crate::svd::truncated_svd_flat`] orthogonalises columns by
//! plane rotations: O(sweeps · cols² · rows), with ~6-10 sweeps measured. A
//! bidiagonalisation-based SVD is O(rows · cols²) once. At the 64-120 column
//! θ blocks these shapes produce that is an algorithmic gap, not a constant,
//! and it is spent in a vendor BLAS-3 kernel rather than in our rotation loop.
//!
//! # The certificate is the constraint, not the speed
//!
//! `SvdResultFlat::discarded_weight` feeds `Mps::discarded_weight`, which is a
//! truncation BOUND and not an estimate. The f32 Metal θ-contraction is kept
//! off the production path for exactly this reason — it reports a SMALLER
//! discarded weight than the exact-f64 CPU path on roughly a quarter of the
//! splits it touches (`STATUS.md` §5.3). A tighter-looking number that is not
//! a bound is worse than no change at all. Two things here keep the bound
//! sound, and both are asserted in `tests` rather than argued:
//!
//! 1. The weight is the LARGER of the two computable residuals — the dropped
//!    singular-value tail, and the Frobenius identity ‖A‖²_F − Σσ²_kept. The
//!    two agree to rounding when the factors are orthonormal; taking the max
//!    cannot pick the smaller one. (`zgesdd` returns all min(m, n) singular
//!    values, so unlike the CUDA `gesvda` arm the tail is actually available
//!    here and does not have to be inferred.)
//! 2. A backward-error allowance is added on top, so the reported weight is
//!    above the Jacobi reference's by construction rather than by luck. See
//!    [`backward_error_allowance`].

use crate::svd::{truncated_svd_flat as jacobi_svd_flat, SvdResultFlat};
use num_complex::Complex64;
use std::cell::RefCell;

// `Complex64` is `#[repr(C)]`, so a `[Complex64]` is a LAPACK `complex*16`
// array. Checked rather than assumed: every pointer handed to `zgesdd_` below
// is a cast of one of these buffers.
const _: () = assert!(std::mem::size_of::<Complex64>() == 2 * std::mem::size_of::<f64>());
const _: () = assert!(std::mem::align_of::<Complex64>() == std::mem::align_of::<f64>());

// Accelerate's legacy (32-bit integer) LAPACK interface. These are the
// Fortran symbols, present in the base system framework — no `build.rs`, no
// optional crate, no `ACCELERATE_NEW_LAPACK` opt-in.
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    // `ZGESDD` — divide-and-conquer SVD of a complex double matrix.
    #[allow(clippy::too_many_arguments)]
    fn zgesdd_(
        jobz: *const i8,
        m: *const i32,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        s: *mut f64,
        u: *mut Complex64,
        ldu: *const i32,
        vt: *mut Complex64,
        ldvt: *const i32,
        work: *mut Complex64,
        lwork: *const i32,
        rwork: *mut f64,
        iwork: *mut i32,
        info: *mut i32,
    );
}

/// Per-shape scratch, reused across calls.
///
/// A brickwall circuit at fixed χ runs hundreds of SVDs of identical (m, n) —
/// the same amortisation the CUDA arm's `ShapeCache` exists for, and the
/// reason the `lwork` query is not paid per call. Thread-local because the
/// θ-contraction above this is rayon-parallel and the pool's threads call in
/// concurrently; a shared cache would need a lock on the hot path.
#[derive(Default)]
struct Scratch {
    shape: Option<(usize, usize)>,
    a: Vec<Complex64>,
    s: Vec<f64>,
    u: Vec<Complex64>,
    vt: Vec<Complex64>,
    work: Vec<Complex64>,
    rwork: Vec<f64>,
    iwork: Vec<i32>,
    lwork: i32,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// How much to inflate a computed discarded weight so that it is a BOUND.
///
/// `zgesdd` is backward stable: the computed factorisation satisfies
/// ‖A − UΣVᴴ‖_F ≤ p(m, n)·ε·‖A‖_F for a modest polynomial p. So the computed
/// Σσ² over the dropped tail is the true discarded weight up to O(ε·‖A‖²_F),
/// and "up to" cuts both ways — without an allowance the reported number can
/// sit a few ulps BELOW the truth, which is precisely the failure mode that
/// keeps the f32 θ-contraction off the production path.
///
/// `(m + n)·ε·‖A‖²_F` covers it with room: the per-σ² disagreement between two
/// stable kernels is ~ε·σ²_max, there are min(m, n) ≤ (m+n)/2 of them, so the
/// allowance is about twice the worst case it has to absorb. In absolute terms
/// it is ~1e-13 relative at the 256-column shapes against discarded weights of
/// 1e-3 to 1e-6 relative on any run that actually truncates — far too small to
/// flatter or to spoil the reported fidelity, and far larger than the noise it
/// has to dominate for [`SvdResultFlat::discarded_weight`] to stay above the
/// Jacobi reference's on every input.
fn backward_error_allowance(m: usize, n: usize, frob_sq: f64) -> f64 {
    ((m + n) as f64) * f64::EPSILON * frob_sq
}

/// Truncated SVD of a row-major `m × n` complex matrix (`matrix[i * stride + j]`)
/// via Accelerate `zgesdd`, in the calling convention of
/// [`crate::SvdFlatFn`].
///
/// Returns `None` when LAPACK declined the solve (`info != 0`: non-convergence,
/// or a bad argument), so the caller can fall back to the Jacobi kernel exactly
/// as the CUDA arm does. Truncation semantics are the Jacobi kernel's verbatim:
/// keep at most `max_rank` singular values strictly above `threshold`, never
/// fewer than one.
pub fn truncated_svd_flat(
    matrix: &[Complex64],
    m: usize,
    n: usize,
    stride: usize,
    max_rank: usize,
    threshold: f64,
) -> Option<SvdResultFlat> {
    if m == 0 || n == 0 {
        return Some(SvdResultFlat {
            u: vec![],
            s: vec![],
            vt: vec![],
            m,
            n,
            discarded_weight: 0.0,
        });
    }
    debug_assert!(stride >= n, "stride {stride} < n {n}");
    debug_assert!(matrix.len() >= (m - 1) * stride + n);
    // The legacy interface is 32-bit; these shapes are χ-bounded and nowhere
    // near it, but a silent truncation of the dimension would be a wrong
    // answer rather than a slow one.
    if m > i32::MAX as usize || n > i32::MAX as usize {
        return None;
    }

    let k = m.min(n);
    let frob_sq: f64 = (0..m)
        .flat_map(|i| matrix[i * stride..i * stride + n].iter())
        .map(|z| z.norm_sqr())
        .sum();

    SCRATCH.with(|cell| {
        let sc = &mut *cell.borrow_mut();
        ensure_scratch(sc, m, n)?;

        // LAPACK is column-major; pack A as `a[j * m + i]`, dropping any input
        // stride padding. O(mn) against the solve's O(mn·min(m,n)).
        sc.a.clear();
        sc.a.resize(m * n, Complex64::new(0.0, 0.0));
        for i in 0..m {
            let row = &matrix[i * stride..i * stride + n];
            for (j, &z) in row.iter().enumerate() {
                sc.a[j * m + i] = z;
            }
        }

        // JOBZ = 'S': the thin factors. U is m×k, Vᴴ is k×n, both column-major.
        let jobz = b'S' as i8;
        let (mi, ni) = (m as i32, n as i32);
        let (lda, ldu, ldvt) = (m as i32, m as i32, k as i32);
        let mut info: i32 = 0;
        // The SAFETY note below is a claim about buffer sizes, so check it
        // rather than only assert it in prose. LAPACK cannot: the sizes of
        // `s` / `u` / `vt` / `rwork` / `iwork` are not arguments, they are
        // implied by `m`, `n` and `jobz`. A stale thread-local scratch is the
        // one way they can disagree with the dimensions in the call, and
        // without this that disagreement is a buffer overrun — found by
        // mutating the cache's shape key, which corrupted the heap instead of
        // failing.
        debug_assert_eq!(sc.a.len(), m * n, "scratch A sized for another shape");
        debug_assert_eq!(sc.s.len(), k, "scratch S sized for another shape");
        debug_assert_eq!(sc.u.len(), m * k, "scratch U sized for another shape");
        debug_assert_eq!(sc.vt.len(), k * n, "scratch Vt sized for another shape");
        debug_assert_eq!(
            sc.work.len(),
            sc.lwork as usize,
            "scratch WORK disagrees with the LWORK being passed"
        );
        debug_assert!(
            sc.rwork.len() >= k * (5 * k + 7).max(2 * m.max(n) + 2 * k + 1),
            "scratch RWORK below LAPACK's minimum for this shape"
        );
        debug_assert!(
            sc.iwork.len() >= 8 * k,
            "scratch IWORK below LAPACK's minimum for this shape"
        );
        // SAFETY: every buffer is sized by `ensure_scratch` to the dimensions
        // declared in the arguments beside it — `a` is lda(=m)×n, `s` is k,
        // `u` is ldu(=m)×k, `vt` is ldvt(=k)×n, and `work`/`rwork`/`iwork` are
        // at LAPACK's documented minima for JOBZ='S' (see `ensure_scratch`).
        // `zgesdd` writes only within them and overwrites `a`, which is our
        // copy.
        unsafe {
            zgesdd_(
                &jobz,
                &mi,
                &ni,
                sc.a.as_mut_ptr(),
                &lda,
                sc.s.as_mut_ptr(),
                sc.u.as_mut_ptr(),
                &ldu,
                sc.vt.as_mut_ptr(),
                &ldvt,
                sc.work.as_mut_ptr(),
                &sc.lwork,
                sc.rwork.as_mut_ptr(),
                sc.iwork.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return None;
        }

        // `zgesdd` returns σ descending, which is the `SvdFlatFn` contract the
        // adaptive-truncation path reads `s[0]` under — no sort needed, and no
        // re-ordering of U / Vᴴ columns either.
        let rank =
            sc.s.iter()
                .take(max_rank)
                .take_while(|&&sv| sv > threshold)
                .count()
                .max(1);

        // The bound. `tail` is the dropped Σσ²; `residual` is the same quantity
        // through the Frobenius identity. They agree to rounding and we keep
        // the larger, then add the backward-error allowance — see the module
        // docs and [`backward_error_allowance`].
        //
        // A split that kept every singular value dropped NOTHING, and reports
        // exactly 0.0 — no allowance, because there is no truncation to bound.
        // That is not a shortcut: `Mps::discarded_weight` documents 0.0 as
        // "every split was exact" and `fidelity_estimate` documents 1.0 as
        // "the run was exact", so an allowance applied unconditionally would
        // retire both signals (measured: it turned the exact χ=128 rows at
        // 14 qubits from 0.0 into ~1e-11). The Jacobi kernel's tail sum over
        // an empty range is likewise exactly 0.0, so the two paths stay on one
        // convention, which is what item 16 asks for.
        let discarded_weight = if rank >= k {
            0.0
        } else {
            let tail: f64 = sc.s[rank..k].iter().map(|&sv| sv * sv).sum();
            let kept_sq: f64 = sc.s[..rank].iter().map(|&sv| sv * sv).sum();
            let residual = (frob_sq - kept_sq).max(0.0);
            tail.max(residual) + backward_error_allowance(m, n, frob_sq)
        };

        // Back to row-major: U (m×rank) from the first `rank` columns of the
        // m×k column-major `u`, Vᴴ (rank×n) from the first `rank` rows of the
        // k×n column-major `vt`.
        let mut u = vec![Complex64::new(0.0, 0.0); m * rank];
        for c in 0..rank {
            let col = &sc.u[c * m..c * m + m];
            for (i, &z) in col.iter().enumerate() {
                u[i * rank + c] = z;
            }
        }
        let mut vt = vec![Complex64::new(0.0, 0.0); rank * n];
        for j in 0..n {
            let col = &sc.vt[j * k..j * k + k];
            for (c, &z) in col[..rank].iter().enumerate() {
                vt[c * n + j] = z;
            }
        }

        Some(SvdResultFlat {
            u,
            s: sc.s[..rank].to_vec(),
            vt,
            m,
            n,
            discarded_weight,
        })
    })
}

/// Size the thread-local scratch for `(m, n)`, including the `lwork` query.
/// A no-op when the previous call had the same shape, which is the common case
/// in a fixed-χ sweep.
fn ensure_scratch(sc: &mut Scratch, m: usize, n: usize) -> Option<()> {
    if sc.shape == Some((m, n)) {
        return Some(());
    }
    let k = m.min(n);
    let mx = m.max(n);
    sc.s.clear();
    sc.s.resize(k, 0.0);
    sc.u.clear();
    sc.u.resize(m * k, Complex64::new(0.0, 0.0));
    sc.vt.clear();
    sc.vt.resize(k * n, Complex64::new(0.0, 0.0));
    // LAPACK's documented minima for JOBZ = 'S'. The RWORK expression is the
    // larger of the forms shipped by different LAPACK generations, so it is
    // valid against whichever one Accelerate's legacy interface is built from.
    sc.rwork.clear();
    sc.rwork
        .resize((k * (5 * k + 7).max(2 * mx + 2 * k + 1)).max(1), 0.0);
    sc.iwork.clear();
    sc.iwork.resize((8 * k).max(1), 0);

    // LWORK = -1 is the workspace query: LAPACK writes the optimal size into
    // work[0] and returns without solving.
    let jobz = b'S' as i8;
    let (mi, ni) = (m as i32, n as i32);
    let (lda, ldu, ldvt) = (m as i32, m as i32, k as i32);
    let query_lwork: i32 = -1;
    let mut query = [Complex64::new(0.0, 0.0)];
    let mut info: i32 = 0;
    // SAFETY: the query path reads none of `a` / `s` / `u` / `vt` — LWORK = -1
    // makes `zgesdd` write work[0] and return — but LAPACK still requires
    // non-null, correctly-dimensioned pointers, which the buffers resized
    // above provide. `a` is given as the already-sized previous buffer or a
    // fresh one of m*n.
    sc.a.clear();
    sc.a.resize(m * n, Complex64::new(0.0, 0.0));
    unsafe {
        zgesdd_(
            &jobz,
            &mi,
            &ni,
            sc.a.as_mut_ptr(),
            &lda,
            sc.s.as_mut_ptr(),
            sc.u.as_mut_ptr(),
            &ldu,
            sc.vt.as_mut_ptr(),
            &ldvt,
            query.as_mut_ptr(),
            &query_lwork,
            sc.rwork.as_mut_ptr(),
            sc.iwork.as_mut_ptr(),
            &mut info,
        );
    }
    if info != 0 {
        sc.shape = None;
        return None;
    }
    // work[0] comes back as a complex whose real part is the size.
    let lwork = query[0].re as i64;
    // The documented minimum for JOBZ='S' is 2*k*k + 2*k + mx; never go below
    // it, whatever the query said.
    let minimum = (2 * k * k + 2 * k + mx) as i64;
    let lwork = lwork.max(minimum).max(1);
    if lwork > i32::MAX as i64 {
        sc.shape = None;
        return None;
    }
    sc.lwork = lwork as i32;
    sc.work.clear();
    sc.work.resize(lwork as usize, Complex64::new(0.0, 0.0));
    sc.shape = Some((m, n));
    Some(())
}

/// Smallest `min(m, n)` worth sending to LAPACK, measured rather than guessed
/// (`examples/accelerate_svd_crossover.rs`, M4, θ-block shapes, truncating to
/// half the available rank):
///
/// | shape | jacobi | zgesdd | |
/// |---|---|---|---|
/// | 4×4 | 1.29 µs | 2.54 µs | **0.51x — LAPACK loses** |
/// | 8×8 | 8.29 µs | 7.88 µs | 1.05x — a wash |
/// | 16×16 | 60.96 µs | 29.21 µs | 2.09x |
/// | 32×32 | 504.54 µs | 138.25 µs | 3.65x |
/// | 64×64 | 3008.50 µs | 557.54 µs | 5.40x |
/// | 256×256 | 79018 µs | 13291 µs | 5.95x |
///
/// 16 rather than 8 because 8 is break-even and the absolute times there are
/// tens of microseconds — there is nothing material between the two points to
/// capture, and the gate's job is to make sure a shallow circuit cannot be
/// made slower. A depth-4 brickwall tops out at 8-column blocks, so this gate
/// hands that whole regime back to the Jacobi kernel unchanged, which is
/// exactly what [`accelerate_svd_flat`] measured as 0.76-1.07x without it.
///
/// Keyed on `min(m, n)` and not on the element count, because that is what
/// bounds the factorisation's cost: the 16×32 and 32×16 blocks both measured
/// ~1.9x, with the same `min` as the 2.09x square.
pub const MIN_DIM_FOR_LAPACK: usize = 16;

/// Whether [`accelerate_svd_flat`] will route an `m × n` block to LAPACK.
/// Public so a measurement harness counts the same decision the library
/// makes, rather than a copy of it that can drift.
pub fn lapack_is_worth_it(m: usize, n: usize) -> bool {
    m.min(n) >= MIN_DIM_FOR_LAPACK
}

/// [`truncated_svd_flat`] behind the size gate, with the Jacobi kernel as
/// fallback — the exact shape of [`crate::SvdFlatFn`], and on macOS the kernel
/// [`crate::MpsBackend`] installs unless asked for another
/// ([`crate::SvdKernel::AccelerateZgesdd`]).
///
/// Two fall-throughs, and both land on the Jacobi kernel: a block below
/// [`MIN_DIM_FOR_LAPACK`], where LAPACK is measured to be slower, and a solve
/// LAPACK declined (`info != 0`). So this never changes whether a run
/// completes, and below the gate it does not change the result either — bit for
/// bit, since it delegates.
pub fn accelerate_svd_flat(
    matrix: &[Complex64],
    m: usize,
    n: usize,
    stride: usize,
    max_rank: usize,
    threshold: f64,
) -> SvdResultFlat {
    if !lapack_is_worth_it(m, n) {
        return jacobi_svd_flat(matrix, m, n, stride, max_rank, threshold);
    }
    truncated_svd_flat(matrix, m, n, stride, max_rank, threshold)
        .unwrap_or_else(|| jacobi_svd_flat(matrix, m, n, stride, max_rank, threshold))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Jacobi kernel's own generator, so the two paths are compared on the
    /// same corpus the CPU unit tests use.
    fn seeded_matrix(m: usize, n: usize, seed: u64) -> Vec<Complex64> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let z = z ^ (z >> 31);
            (z as f64) / (u64::MAX as f64) * 2.0 - 1.0
        };
        (0..m * n).map(|_| Complex64::new(next(), next())).collect()
    }

    /// A decaying spectrum: what a deep-circuit θ block actually looks like,
    /// and the only regime where truncation discards anything interesting.
    fn decaying_spectrum(m: usize, n: usize, ratio: f64, seed: u64) -> Vec<Complex64> {
        let k = m.min(n);
        // Q R of a random matrix would need a QR; a diagonal spectrum rotated
        // by the Jacobi kernel's own V is enough and keeps this dependency-free.
        let mut a = vec![Complex64::new(0.0, 0.0); m * n];
        let noise = seeded_matrix(m, n, seed);
        for i in 0..k {
            a[i * n + i] = Complex64::new(ratio.powi(i as i32), 0.0);
        }
        // Mix in a little noise so the matrix is not exactly diagonal (a
        // diagonal input is the one shape where both kernels are trivially
        // exact and the comparison proves nothing).
        for (dst, src) in a.iter_mut().zip(noise.iter()) {
            *dst += src * 1e-3;
        }
        a
    }

    fn orthonormality_defect(mat: &[Complex64], rows: usize, cols: usize) -> f64 {
        let mut worst = 0.0f64;
        for p in 0..cols {
            for q in 0..cols {
                let mut dot = Complex64::new(0.0, 0.0);
                for i in 0..rows {
                    dot += mat[i * cols + p].conj() * mat[i * cols + q];
                }
                let target = if p == q { 1.0 } else { 0.0 };
                worst = worst.max((dot - Complex64::new(target, 0.0)).norm());
            }
        }
        worst
    }

    /// The forward tolerance is 1e-10 and stays 1e-10 — the same bound
    /// this crate's own `jacobi_svd_is_unitary_tall_and_wide_at_bond_64_shapes`
    /// holds its kernel to, on the same shapes, because a split whose factors
    /// are not orthonormal is a non-unitary split whatever produced it.
    #[test]
    fn accelerate_svd_is_unitary_tall_and_wide_at_bond_64_shapes() {
        for (m, n, seed) in [
            (128usize, 96usize, 1u64),
            (96, 128, 2),
            (64, 64, 3),
            (40, 12, 4),
            (12, 40, 5),
        ] {
            let a = seeded_matrix(m, n, seed);
            let r = truncated_svd_flat(&a, m, n, n, m.min(n), 1e-30)
                .expect("zgesdd converges on a generic matrix");
            let k = r.s.len();
            assert_eq!(k, m.min(n), "full rank expected for a generic matrix");
            assert!(
                orthonormality_defect(&r.u, m, k) < 1e-10,
                "UᴴU ≠ I at {m}x{n}: defect {}",
                orthonormality_defect(&r.u, m, k)
            );
            let mut vmat = vec![Complex64::new(0.0, 0.0); n * k];
            for kk in 0..k {
                for j in 0..n {
                    vmat[j * k + kk] = r.vt[kk * n + j].conj();
                }
            }
            assert!(
                orthonormality_defect(&vmat, n, k) < 1e-10,
                "VᴴV ≠ I at {m}x{n}: defect {}",
                orthonormality_defect(&vmat, n, k)
            );
            let mut worst = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for kk in 0..k {
                        acc += r.u[i * k + kk] * r.s[kk] * r.vt[kk * n + j];
                    }
                    worst = worst.max((a[i * n + j] - acc).norm());
                }
            }
            assert!(worst < 1e-9, "reconstruction defect {worst} at {m}x{n}");
        }
    }

    /// **The bound test.** `discarded_weight` is a certificate bound, not an
    /// estimate, so the only acceptable direction of disagreement with the
    /// Jacobi reference is UPWARD. A tighter-looking number that is not
    /// actually a bound is the worst outcome available here — it is why the
    /// f32 θ-contraction is not on the production path — so this asserts the
    /// inequality directly, with no slack, on every truncating shape in the
    /// corpus.
    #[test]
    fn accelerate_discarded_weight_never_undercuts_the_jacobi_reference() {
        // The θ-block shapes these circuits produce (64 columns at χ=32, 120
        // at χ=128, 256 at χ=256), tall and wide, plus a small one. The
        // reference is the real Jacobi kernel, so the 256-column case is
        // deliberately a single spectrum — in a debug build it costs more than
        // the rest of the corpus together.
        let mut cases: Vec<(usize, usize, f64, u64)> = Vec::new();
        for &(m, n) in &[
            (64usize, 64usize),
            (128, 96),
            (96, 128),
            (64, 120),
            (40, 12),
        ] {
            for &ratio in &[0.5f64, 0.8, 0.95] {
                for seed in 0..3u64 {
                    cases.push((m, n, ratio, 0x5EED + seed));
                }
            }
        }
        cases.push((256, 256, 0.95, 0x5EED));

        let mut compared = 0;
        for (m, n, ratio, seed) in cases {
            let a = decaying_spectrum(m, n, ratio, seed);
            let k = m.min(n);
            // Ranks that actually truncate — a rank-k call discards nothing
            // and would compare 0.0 against 0.0.
            for &max_rank in &[1usize, k / 8, k / 4, k / 2] {
                let max_rank = max_rank.max(1);
                let reference = jacobi_svd_flat(&a, m, n, n, max_rank, 1e-14);
                let candidate = truncated_svd_flat(&a, m, n, n, max_rank, 1e-14)
                    .expect("zgesdd converges on this corpus");
                assert!(
                    reference.discarded_weight > 0.0,
                    "{m}x{n} ratio={ratio} rank={max_rank} discards nothing, so this case \
                     cannot witness an under-report"
                );
                assert!(
                    candidate.discarded_weight >= reference.discarded_weight,
                    "UNSOUND BOUND at {m}x{n} ratio={ratio} rank={max_rank}: accelerate \
                     reported {} discarded weight, below the Jacobi reference's {}",
                    candidate.discarded_weight,
                    reference.discarded_weight
                );
                // Above, but not by enough to be a different number: a bound
                // inflated for safety's sake would pass the inequality while
                // making every run look worse than it is. Measured against the
                // BLOCK NORM, which is the quantity the certificate is a
                // fraction of — not against the reference weight, which can
                // itself be arbitrarily close to zero on a nearly exact split
                // and would make any fixed relative margin unmeetable.
                let frob_sq: f64 = a.iter().map(|z| z.norm_sqr()).sum();
                let excess = candidate.discarded_weight - reference.discarded_weight;
                assert!(
                    excess <= 1e-12 * frob_sq,
                    "{m}x{n} ratio={ratio} rank={max_rank}: accelerate's bound is {excess:e} \
                     above the reference's against a block norm of {frob_sq:e} — sound but \
                     no longer the same certificate"
                );
                compared += 1;
            }
        }
        assert!(compared >= 180, "only {compared} truncating cases compared");
    }

    /// Truncation semantics match the Jacobi kernel's to the digit on a known
    /// spectrum: rank 2 of diag(4, 3, 2, 1) drops exactly 2² + 1² = 5.
    ///
    /// Worth knowing what this does and does not pin, because a mutation said
    /// so: summing the dropped σ instead of their squares leaves it GREEN. The
    /// Frobenius route (‖A‖²_F − Σσ²_kept = 30 − 25 = 5) is an independent
    /// computation of the same number and the reported weight is the max of
    /// the two, so one route breaking does not move the answer. That is the
    /// point of taking the max; this test pins the REPORTED certificate, and
    /// the rank it is reported at.
    #[test]
    fn accelerate_discarded_weight_equals_dropped_singular_squares() {
        let s_true = [4.0, 3.0, 2.0, 1.0];
        let n = s_true.len();
        let mut a = vec![Complex64::new(0.0, 0.0); n * n];
        for (i, &sv) in s_true.iter().enumerate() {
            a[i * n + i] = Complex64::new(sv, 0.0);
        }
        let r = truncated_svd_flat(&a, n, n, n, 2, 1e-12).expect("zgesdd converges");
        assert_eq!(r.s.len(), 2);
        assert!((r.s[0] - 4.0).abs() < 1e-9 && (r.s[1] - 3.0).abs() < 1e-9);
        assert!(
            (r.discarded_weight - 5.0).abs() < 1e-9,
            "discarded_weight = {}",
            r.discarded_weight
        );
    }

    /// An untruncated split reports EXACTLY zero, not an allowance. `Mps`
    /// documents `discarded_weight == 0.0` as "every split was exact" and
    /// `fidelity_estimate == 1.0` as "the run was exact"; a path that returns
    /// 4e-14 on a split that dropped nothing retires both signals, and at
    /// 14 qubits with χ ≥ 128 that is EVERY split.
    #[test]
    fn accelerate_reports_exactly_zero_when_nothing_is_truncated() {
        for &(m, n, seed) in &[(20usize, 8usize, 7u64), (64, 64, 8), (96, 128, 9)] {
            let a = seeded_matrix(m, n, seed);
            let k = m.min(n);
            let r = truncated_svd_flat(&a, m, n, n, k, 1e-30).expect("zgesdd converges");
            assert_eq!(r.s.len(), k, "{m}x{n} should keep the full rank");
            assert_eq!(
                r.discarded_weight, 0.0,
                "{m}x{n} dropped nothing but reported {} discarded weight",
                r.discarded_weight
            );
        }
    }

    /// The singular values themselves agree with the reference to the forward
    /// tolerance. Without this, the bound test above could be satisfied by a
    /// path that decomposes something else entirely.
    #[test]
    fn accelerate_singular_values_agree_with_the_jacobi_reference() {
        for &(m, n, seed) in &[(64usize, 64usize, 11u64), (128, 96, 12), (96, 128, 13)] {
            let a = seeded_matrix(m, n, seed);
            let k = m.min(n);
            let reference = jacobi_svd_flat(&a, m, n, n, k, 1e-30);
            let candidate = truncated_svd_flat(&a, m, n, n, k, 1e-30).expect("zgesdd converges");
            assert_eq!(reference.s.len(), candidate.s.len(), "{m}x{n}");
            let smax = reference.s[0];
            for (i, (x, y)) in reference.s.iter().zip(&candidate.s).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-10 * smax,
                    "{m}x{n} sigma[{i}]: reference {x}, accelerate {y}"
                );
            }
        }
    }

    /// The scratch cache is keyed on shape, so a run that alternates shapes —
    /// which every ramping-χ circuit does — must not carry a stale `lwork` or
    /// a stale buffer size into the next solve.
    ///
    /// **The ASCENDING order is load-bearing and was put here by a mutation.**
    /// An earlier version of this list started at 64×64, and a cache that
    /// skipped the shape check entirely (`sc.shape.is_some()`) passed it: every
    /// later shape was smaller, so an oversized workspace served them all and
    /// nothing was observable. Growing into the cache is what exposes the
    /// stale `lwork` — LAPACK checks its arguments before touching any array,
    /// so a workspace sized for 16×8 makes a 64×64 solve decline cleanly
    /// rather than scribbling past the buffer.
    #[test]
    fn alternating_shapes_reuse_the_scratch_without_corrupting_it() {
        let shapes = [(16usize, 8usize), (32, 16), (32, 64), (64, 32), (64, 64)];
        for _ in 0..3 {
            for &(m, n) in &shapes {
                let a = seeded_matrix(m, n, 0xA1);
                let k = m.min(n);
                let reference = jacobi_svd_flat(&a, m, n, n, k, 1e-30);
                let candidate =
                    truncated_svd_flat(&a, m, n, n, k, 1e-30).expect("zgesdd converges");
                let smax = reference.s[0];
                for (x, y) in reference.s.iter().zip(&candidate.s) {
                    assert!((x - y).abs() <= 1e-10 * smax, "{m}x{n}: {x} vs {y}");
                }
            }
        }
    }

    /// Below the gate the shipped entry point IS the Jacobi kernel — bit for
    /// bit, not "to a tolerance". Without this, "the gate changes nothing
    /// below the threshold" is a claim about a delegation nobody checked, and
    /// a gate that silently routed small blocks to LAPACK anyway would perturb
    /// the certificate on exactly the shallow circuits it exists to protect.
    #[test]
    fn below_the_gate_the_result_is_the_jacobi_kernel_bit_for_bit() {
        let mut gated_below = 0;
        for &(m, n) in &[(4usize, 4usize), (8, 8), (15, 15), (8, 64), (64, 8), (2, 2)] {
            assert!(
                !lapack_is_worth_it(m, n),
                "{m}x{n} is meant to be below the gate"
            );
            let a = seeded_matrix(m, n, 0x9A7E);
            let max_rank = (m.min(n) / 2).max(1);
            let shipped = accelerate_svd_flat(&a, m, n, n, max_rank, 1e-14);
            let reference = jacobi_svd_flat(&a, m, n, n, max_rank, 1e-14);
            assert_eq!(
                shipped.discarded_weight.to_bits(),
                reference.discarded_weight.to_bits(),
                "{m}x{n}: certificate differs below the gate"
            );
            assert_eq!(shipped.s.len(), reference.s.len(), "{m}x{n}: rank differs");
            for (x, y) in shipped.s.iter().zip(&reference.s) {
                assert_eq!(x.to_bits(), y.to_bits(), "{m}x{n}: sigma differs");
            }
            for (x, y) in shipped.u.iter().zip(&reference.u) {
                assert_eq!(x.re.to_bits(), y.re.to_bits(), "{m}x{n}: U differs");
                assert_eq!(x.im.to_bits(), y.im.to_bits());
            }
            for (x, y) in shipped.vt.iter().zip(&reference.vt) {
                assert_eq!(x.re.to_bits(), y.re.to_bits(), "{m}x{n}: Vt differs");
                assert_eq!(x.im.to_bits(), y.im.to_bits());
            }
            gated_below += 1;
        }
        assert_eq!(gated_below, 6);

        // And the gate is reachable the other way, or the test above proves
        // only that the gate rejects everything.
        assert!(lapack_is_worth_it(16, 16));
        assert!(lapack_is_worth_it(64, 120));
        let a = seeded_matrix(64, 64, 0x9A7F);
        let shipped = accelerate_svd_flat(&a, 64, 64, 64, 32, 1e-14);
        let reference = jacobi_svd_flat(&a, 64, 64, 64, 32, 1e-14);
        assert!(
            shipped.s.iter().zip(&reference.s).any(|(x, y)| x != y),
            "above the gate the shipped path returned the Jacobi kernel's exact bits, so \
             nothing was dispatched to LAPACK"
        );
        assert!(
            shipped.discarded_weight >= reference.discarded_weight,
            "above the gate the bound must still hold: {} vs {}",
            shipped.discarded_weight,
            reference.discarded_weight
        );
    }

    /// The `SvdFlatFn` contract the adaptive-truncation path depends on:
    /// `s[0]` is σ_max and the rest descend. An unsorted provider silently
    /// mis-truncates.
    #[test]
    fn accelerate_returns_singular_values_descending() {
        let a = seeded_matrix(96, 64, 0xDE);
        let r = truncated_svd_flat(&a, 96, 64, 64, 64, 1e-30).expect("zgesdd converges");
        for w in r.s.windows(2) {
            assert!(w[0] >= w[1], "sigma not descending: {:?}", w);
        }
    }
}
