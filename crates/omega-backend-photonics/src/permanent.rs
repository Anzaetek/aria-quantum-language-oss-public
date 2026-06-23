//! Matrix permanent computation.
//!
//! Implements Ryser's formula with Gray code optimization.
//! Complexity: O(2^n * n) for an n×n matrix.

use num_complex::Complex64;

/// Compute the permanent of an n×n complex matrix using Ryser's formula.
///
/// Ryser's formula:
///   perm(A) = (-1)^n * sum_{S ⊆ [n]} (-1)^|S| * prod_{i=1..n} sum_{j∈S} a_{ij}
///
/// Uses Gray code iteration so consecutive subsets differ by one element,
/// allowing O(n) update per step instead of recomputing row sums.
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

    // row_sums[i] = sum of a[i][j] for j in current subset S
    let mut row_sums = vec![Complex64::new(0.0, 0.0); n];
    let mut result = Complex64::new(0.0, 0.0);
    let num_subsets = 1u64 << n;

    // Track current subset size to compute (-1)^|S|
    let mut subset_size: u32 = 0;

    for k in 1..num_subsets {
        // Gray code: find which bit changed
        let j = k.trailing_zeros() as usize;

        // Check if we're adding or removing column j
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

        // Compute product of row sums
        let mut prod = Complex64::new(1.0, 0.0);
        for i in 0..n {
            prod *= row_sums[i];
        }

        // (-1)^|S|
        let sign = if subset_size.is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        result += Complex64::new(sign, 0.0) * prod;
    }

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
