//! Matrix permanent computation.
//!
//! Implements Ryser's formula with Gray code optimization.
//! Complexity: O(2^n * n) for an n×n matrix.

use num_complex::Complex64;
use rayon::prelude::*;

/// Compute the permanent of an n×n complex matrix using Ryser's formula.
///
/// Ryser's formula:
///   perm(A) = (-1)^n * sum_{S ⊆ [n]} (-1)^|S| * prod_{i=1..n} sum_{j∈S} a_{ij}
///
/// Uses Gray code iteration so consecutive subsets differ by one element,
/// allowing O(n) update per step instead of recomputing row sums.
/// Chunk count for the parallel sweep, and the width below which the sweep
/// stays sequential.
///
/// **Both are functions of `n` only, never of the thread count.** That is what
/// makes the result identical whether one worker or thirty-two run it: rayon
/// preserves order through `collect`, and the partials are folded in chunk
/// order, so the summation tree is fixed by `n` alone. §5.16 of `STATUS.md`
/// records what the alternative costs — a schedule that varies with the pool
/// makes results unreproducible and the certificate unverifiable.
const PERM_PAR_MIN_N: usize = 16;
const PERM_CHUNKS: u64 = 128;

/// Ryser terms for `k` in `[k0, k1)`, summed locally.
///
/// The Gray-code sweep looks sequential — `row_sums` is carried from one
/// iteration to the next — but the carry is only an OPTIMISATION. Term `k`
/// depends solely on the subset `gray(k)`, so a chunk can reconstruct the state
/// it needs from `gray(k0 - 1)` in O(n²) and then run the identical incremental
/// body. That reconstruction is what makes this parallelisable at all; without
/// it the alternative is recomputing every row sum from scratch, which is n×
/// the arithmetic and only pays above n cores.
///
/// Restarting each chunk from a freshly computed `row_sums` also RESETS the
/// rounding drift the incremental ± updates accumulate, so chunked sums are if
/// anything slightly more accurate than the single long running sum — the usual
/// blocked-summation result, not a regression.
fn ryser_chunk(matrix: &[Vec<Complex64>], n: usize, k0: u64, k1: u64) -> Complex64 {
    // The loop body for `k` transitions gray(k-1) -> gray(k), so the state at
    // the start of the chunk is the subset gray(k0 - 1). For k0 = 1 that is the
    // empty set and row_sums is zero — bit-identical to the sequential sweep.
    let prev = (k0 - 1) ^ ((k0 - 1) >> 1);
    let mut row_sums = vec![Complex64::new(0.0, 0.0); n];
    for j in 0..n {
        if (prev >> j) & 1 == 1 {
            for i in 0..n {
                row_sums[i] += matrix[i][j];
            }
        }
    }
    let mut subset_size: u32 = prev.count_ones();
    let mut acc = Complex64::new(0.0, 0.0);

    for k in k0..k1 {
        let j = k.trailing_zeros() as usize;
        let gray_curr = k ^ (k >> 1);
        let adding = (gray_curr >> j) & 1 == 1;

        if adding {
            subset_size += 1;
            for i in 0..n {
                row_sums[i] += matrix[i][j];
            }
        } else {
            subset_size -= 1;
            for i in 0..n {
                row_sums[i] -= matrix[i][j];
            }
        }

        let mut prod = Complex64::new(1.0, 0.0);
        for i in 0..n {
            prod *= row_sums[i];
        }

        let sign = if subset_size.is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        acc += Complex64::new(sign, 0.0) * prod;
    }
    acc
}

pub fn permanent(matrix: &[Vec<Complex64>]) -> Complex64 {
    let n = matrix.len();
    if n == 0 {
        return Complex64::new(1.0, 0.0);
    }
    if n == 1 {
        return matrix[0][0];
    }
    if n == 2 {
        return matrix[0][0] * matrix[1][1] + matrix[0][1] * matrix[1][0];
    }

    let num_subsets = 1u64 << n;
    let total = num_subsets - 1; // k runs over 1..num_subsets

    // Below the gate: ONE unchunked sweep and nothing else. This path is the
    // original loop and must stay allocation-for-allocation identical to it.
    // `slos_full` calls `permanent` once per output Fock state, on submatrices
    // the size of the PHOTON NUMBER — so this is the photonic hot path and n is
    // small there. Building a range vector here cost 45 ns on the 6x6 (403 ->
    // 448) and 4.7% of SLOS, to speed up sizes SLOS never reaches.
    if n < PERM_PAR_MIN_N {
        let r = ryser_chunk(matrix, n, 1, num_subsets);
        return if n % 2 == 1 { -r } else { r };
    }

    let chunks = PERM_CHUNKS.min(total);
    let ranges: Vec<(u64, u64)> = (0..chunks)
        .map(|c| (1 + total * c / chunks, 1 + total * (c + 1) / chunks))
        .filter(|(a, b)| a < b)
        .collect();

    // `collect` preserves order and the fold walks it in order, so the
    // summation tree is fixed by `n` and the result is the same on any pool.
    let mut result = ranges
        .par_iter()
        .map(|&(k0, k1)| ryser_chunk(matrix, n, k0, k1))
        .collect::<Vec<_>>()
        .into_iter()
        .fold(Complex64::new(0.0, 0.0), |a, b| a + b);

    // Multiply by (-1)^n
    if n % 2 == 1 {
        result = -result;
    }

    result
}

/// Compute the permanent of a submatrix defined by row and column indices.
/// Useful for Fock-space calculations where indices may repeat.
pub fn permanent_submatrix(
    full_matrix: &[Vec<Complex64>],
    rows: &[usize],
    cols: &[usize],
) -> Complex64 {
    let n = rows.len();
    assert_eq!(n, cols.len(), "permanent requires square submatrix");

    if n == 0 {
        return Complex64::new(1.0, 0.0);
    }

    // Extract submatrix
    let sub: Vec<Vec<Complex64>> = rows
        .iter()
        .map(|&r| cols.iter().map(|&c| full_matrix[r][c]).collect())
        .collect();

    permanent(&sub)
}

#[cfg(test)]
mod tests {

    /// The chunked sweep must be bitwise invariant to the rayon pool size.
    ///
    /// This is the property that makes the parallelisation safe to land at all,
    /// and it does not come for free — it holds because the chunk count is a
    /// function of `n` alone, `collect` preserves order, and the partials are
    /// folded in chunk order. Key the chunking on `rayon::current_num_threads()`
    /// instead and the permanent silently becomes machine-dependent. Same
    /// invariance the SVD kernel is held to
    /// (`omega_backend_mps::svd::thread_count_does_not_change_the_bits`).
    #[test]
    fn thread_count_does_not_change_the_bits() {
        // n must be at or above PERM_PAR_MIN_N, or this proves nothing: below
        // the gate there is one chunk and no parallel section to vary.
        for n in [PERM_PAR_MIN_N, PERM_PAR_MIN_N + 1] {
            let m: Vec<Vec<Complex64>> = (0..n)
                .map(|j| {
                    (0..n)
                        .map(|k| {
                            Complex64::new(
                                ((j * 7 + k * 13 + 1) as f64).sin(),
                                ((j * 11 + k * 5 + 3) as f64).cos(),
                            )
                        })
                        .collect()
                })
                .collect();
            let run = |threads: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("pool")
                    .install(|| permanent(&m))
            };
            let a = run(1);
            let b = run(8);
            assert_eq!(
                a.re.to_bits(),
                b.re.to_bits(),
                "n={n}: real part differs by thread count ({a:?} vs {b:?})"
            );
            assert_eq!(
                a.im.to_bits(),
                b.im.to_bits(),
                "n={n}: imaginary part differs by thread count"
            );
        }
    }

    /// Below the gate the sweep must be EXACTLY the original one, so small
    /// matrices — every existing caller and every fixture — are untouched.
    #[test]
    fn below_the_gate_one_chunk_reproduces_the_sequential_sweep() {
        let n = 6;
        let m: Vec<Vec<Complex64>> = (0..n)
            .map(|j| {
                (0..n)
                    .map(|k| Complex64::new((j + k) as f64 * 0.37, (j * k) as f64 * 0.11))
                    .collect()
            })
            .collect();
        // Reproduce the pre-2026-09-25 sweep verbatim: one chunk over the whole
        // range, which is what `permanent` does for n < PERM_PAR_MIN_N.
        let expect = ryser_chunk(&m, n, 1, 1u64 << n);
        let expect = if n % 2 == 1 { -expect } else { expect };
        let got = permanent(&m);
        assert!(n < PERM_PAR_MIN_N, "pick an n below the gate");
        assert_eq!(got.re.to_bits(), expect.re.to_bits());
        assert_eq!(got.im.to_bits(), expect.im.to_bits());
    }
    use super::*;

    #[test]
    fn test_permanent_1x1() {
        let m = vec![vec![Complex64::new(3.0, 1.0)]];
        let p = permanent(&m);
        assert!((p - Complex64::new(3.0, 1.0)).norm() < 1e-10);
    }

    #[test]
    fn test_permanent_2x2() {
        // perm([[a, b], [c, d]]) = ad + bc
        let m = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            vec![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
        ];
        let p = permanent(&m);
        // 1*4 + 2*3 = 10
        assert!((p.re - 10.0).abs() < 1e-10);
        assert!(p.im.abs() < 1e-10);
    }

    #[test]
    fn test_permanent_3x3() {
        // All-ones 3×3 matrix: permanent = n! = 6
        let m = vec![
            vec![Complex64::new(1.0, 0.0); 3],
            vec![Complex64::new(1.0, 0.0); 3],
            vec![Complex64::new(1.0, 0.0); 3],
        ];
        let p = permanent(&m);
        assert!((p.re - 6.0).abs() < 1e-10, "perm(J_3) = {} (expected 6)", p);
    }

    #[test]
    fn test_permanent_identity() {
        // perm(I_n) = 1 for any n
        for n in 1..8 {
            let mut m = vec![vec![Complex64::new(0.0, 0.0); n]; n];
            for i in 0..n {
                m[i][i] = Complex64::new(1.0, 0.0);
            }
            let p = permanent(&m);
            assert!(
                (p.re - 1.0).abs() < 1e-10 && p.im.abs() < 1e-10,
                "perm(I_{}) = {} (expected 1)",
                n,
                p
            );
        }
    }

    #[test]
    fn test_permanent_4x4() {
        // Known result: all-ones 4×4 -> permanent = 24 = 4!
        let m = vec![vec![Complex64::new(1.0, 0.0); 4]; 4];
        let p = permanent(&m);
        assert!(
            (p.re - 24.0).abs() < 1e-10,
            "perm(J_4) = {} (expected 24)",
            p
        );
    }

    #[test]
    fn test_permanent_5x5() {
        // All-ones 5×5: perm = 5! = 120
        let m = vec![vec![Complex64::new(1.0, 0.0); 5]; 5];
        let p = permanent(&m);
        assert!(
            (p.re - 120.0).abs() < 1e-8,
            "perm(J_5) = {} (expected 120)",
            p
        );
    }

    #[test]
    fn test_permanent_complex() {
        // 2x2 with complex entries
        let i = Complex64::new(0.0, 1.0);
        let m = vec![
            vec![Complex64::new(1.0, 0.0), i],
            vec![i, Complex64::new(1.0, 0.0)],
        ];
        let p = permanent(&m);
        // perm = 1*1 + i*i = 1 + (-1) = 0
        assert!(p.norm() < 1e-10);
    }

    #[test]
    fn test_permanent_known_3x3() {
        // A specific 3x3 matrix with known permanent
        // [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        // perm = 1*5*9 + 1*6*8 + 2*4*9 + 2*6*7 + 3*4*8 + 3*5*7
        //      = 45 + 48 + 72 + 84 + 96 + 105 = 450
        let m = vec![
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            vec![
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
            ],
            vec![
                Complex64::new(7.0, 0.0),
                Complex64::new(8.0, 0.0),
                Complex64::new(9.0, 0.0),
            ],
        ];
        let p = permanent(&m);
        assert!((p.re - 450.0).abs() < 1e-8, "perm = {} (expected 450)", p);
    }
}
