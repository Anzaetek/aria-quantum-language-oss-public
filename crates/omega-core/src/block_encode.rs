//! Pauli decomposition of a dense matrix (`A = Σᵢ αᵢ Pᵢ`).
//!
//! NUMERIC EXTRACTION of the gate-model toolkit's
//! `quantum-core/src/linalg/block_encode.rs`. Only the host-callable
//! Pauli decomposition (`pauli_decompose` + its trace machinery) ships
//! here — it is pure linear algebra over `{I,X,Y,Z}^⊗n`, which is
//! orthogonal under the Hilbert–Schmidt inner product. The LCU
//! block-encoding *circuit* construction (`block_encode_dense` /
//! `DenseBlockEncoding`) stays in the gate-model toolkit, since it
//! builds an aria `Circuit`; this runtime crate carries no circuit IR.
//! KEEP-IN-SYNC: `pauli_decompose`, `pauli_inner`, `apply_pauli_string`
//! are byte-identical to the toolkit source.

use ndarray::Array2;
use num_complex::Complex64;

/// One Pauli string as a length-`n` byte slice over `{0=I, 1=X, 2=Y, 3=Z}`.
pub type PauliString = Vec<u8>;

/// Decompose an `N×N` complex matrix over `{I, X, Y, Z}^⊗n` (with
/// `N = 2^n`).
///
/// The Pauli operators are orthogonal under the Hilbert–Schmidt inner
/// product `⟨P, Q⟩ = (1/N) · Tr(P†·Q)`, so the coefficients are
///
/// ```text
/// αₚ = (1/N) · Tr(P · A)
/// ```
///
/// (Pauli operators are Hermitian, so `P† = P`.) Returns
/// `(αᵢ, Pᵢ)` for every Pauli string with non-negligible coefficient.
pub fn pauli_decompose(a: &Array2<Complex64>) -> Vec<(Complex64, PauliString)> {
    let dim = a.nrows();
    assert_eq!(dim, a.ncols(), "matrix must be square");
    assert!(
        dim.is_power_of_two(),
        "matrix dimension must be a power of 2"
    );
    let n = (dim as f64).log2().round() as usize;
    let n_pauli = 4usize.pow(n as u32);
    let mut out = Vec::with_capacity(n_pauli);
    for pi in 0..n_pauli {
        let pstr: Vec<u8> = (0..n).map(|q| ((pi >> (2 * q)) & 0b11) as u8).collect();
        let alpha = pauli_inner(a, &pstr);
        if alpha.norm() > 1e-12 {
            out.push((alpha, pstr));
        }
    }
    out
}

/// 1-norm of the Pauli decomposition: `α = Σ |αᵢ|`. This is the LCU
/// block-encoding scaling factor (the encoding is exact for `A / α` in
/// the upper-left block). Computed directly from [`pauli_decompose`]
/// without building any circuit.
pub fn pauli_one_norm(a: &Array2<Complex64>) -> f64 {
    pauli_decompose(a)
        .iter()
        .map(|(c, _)| c.norm())
        .sum::<f64>()
        .max(1e-30)
}

/// Compute `(1/N) · Tr(P · A)` where `P = P_{n-1} ⊗ … ⊗ P_0` is the
/// tensor product of single-qubit Paulis encoded by `pstr`.
fn pauli_inner(a: &Array2<Complex64>, pstr: &[u8]) -> Complex64 {
    let dim = a.nrows();
    let inv_dim = Complex64::new(1.0 / dim as f64, 0.0);
    let mut sum = Complex64::new(0.0, 0.0);
    // Tr(P·A) = Σ_{i,j} P[i,j] · A[j,i]. Since P|j⟩ = phase·|k⟩ has
    // exactly one nonzero entry P[k,j] = phase per column j, the
    // double sum collapses: for each j we know k, and contribute
    // `phase · A[j, k]`.
    for j in 0..dim {
        let (k, phase) = apply_pauli_string(pstr, j);
        sum += phase * a[(j, k)];
    }
    sum * inv_dim
}

/// `pstr` indexes single-qubit Paulis at positions 0..n. Acts on
/// computational-basis state `j` (LSB = qubit 0) and returns the
/// resulting basis index `k` plus the complex phase from Y-bits.
fn apply_pauli_string(pstr: &[u8], j: usize) -> (usize, Complex64) {
    let mut k = j;
    let mut phase = Complex64::new(1.0, 0.0);
    for (q, &p) in pstr.iter().enumerate() {
        let bit = (j >> q) & 1;
        match p {
            0 => { /* I: no flip, no phase */ }
            1 => {
                // X: flip bit q, no phase.
                k ^= 1 << q;
            }
            2 => {
                // Y: flip bit q, phase i if input bit was 0, -i if 1.
                k ^= 1 << q;
                phase *= if bit == 0 {
                    Complex64::new(0.0, 1.0)
                } else {
                    Complex64::new(0.0, -1.0)
                };
            }
            3 => {
                // Z: no flip; phase -1 if bit was 1.
                if bit == 1 {
                    phase = -phase;
                }
            }
            _ => unreachable!("invalid Pauli code"),
        }
    }
    (k, phase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn pauli_decompose_identity() {
        let a: Array2<Complex64> = Array2::eye(2);
        let terms = pauli_decompose(&a);
        // I: αᵢ = 1, X/Y/Z: 0.
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].1, vec![0]);
        assert!((terms[0].0.re - 1.0).abs() < 1e-12);
        assert!(terms[0].0.im.abs() < 1e-12);
    }

    #[test]
    fn pauli_decompose_pauli_x() {
        let a: Array2<Complex64> = array![
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ];
        let terms = pauli_decompose(&a);
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].1, vec![1]);
        assert!((terms[0].0.re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pauli_decompose_pauli_z() {
        let a: Array2<Complex64> = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
        ];
        let terms = pauli_decompose(&a);
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].1, vec![3]);
    }

    #[test]
    fn pauli_decompose_y_picks_up_imaginary() {
        // Y = [[0, -i], [i, 0]]
        let a: Array2<Complex64> = array![
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
            [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
        ];
        let terms = pauli_decompose(&a);
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].1, vec![2]);
        // ⟨Y, Y⟩ = (1/2) Tr(Y · Y) = 1.
        assert!((terms[0].0.re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pauli_decompose_diag_combo() {
        // diag(1, 2) = (3/2)·I + (-1/2)·Z.
        let a: Array2<Complex64> = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(2.0, 0.0)],
        ];
        let terms = pauli_decompose(&a);
        let by_str = |p: u8| {
            terms
                .iter()
                .find(|(_, s)| s == &vec![p])
                .map(|(c, _)| *c)
                .unwrap_or_default()
        };
        assert!((by_str(0).re - 1.5).abs() < 1e-12);
        assert!((by_str(3).re + 0.5).abs() < 1e-12);
    }

    #[test]
    fn pauli_one_norm_diag_combo() {
        // diag(1, 2) = 1.5·I − 0.5·Z → α = 1.5 + 0.5 = 2.0.
        let a: Array2<Complex64> = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(2.0, 0.0)],
        ];
        assert!((pauli_one_norm(&a) - 2.0).abs() < 1e-12);
    }
}
