// SPDX-License-Identifier: Apache-2.0
//! Eigenbasis synthesis for the genuine dense (Hermitian) QSVT solver.
//!
//! A real symmetric positive-definite `A` is diagonalized classically
//! `A = U D Uᵀ` (Jacobi), and the eigenbasis rotation `U` is compiled to a
//! circuit on the system register so the diagonal QSVT inversion
//! ([`crate::linalg::block_encode::block_encode_diagonal`]) can be conjugated
//! into `A`'s eigenbasis: `W_A = U · W_D · U†`, giving `⟨0|_a U_Φ |0⟩_a = P(A)`.
//!
//! - 1 system qubit: `U ∈ O(2)` is a single-qubit gate — a `Z-Y-Z` Euler
//!   rotation.
//! - 2 system qubits: `U ∈ SO(4)` factorizes through the magic basis as
//!   `M·U·M† = A₂ ⊗ B₂` with `A₂, B₂ ∈ U(2)` (Vatan–Williams). `M` is a fixed
//!   4-gate circuit (`S · CX · H · CX`), so `U = M†·(A₂⊗B₂)·M` is two
//!   single-qubit gates wrapped by the magic circuit. The `A₂⊗B₂` factors come
//!   from a no-SVD rank-1 (Pitsianis–Van Loan) extraction.
//!
//! Only real symmetric positive-definite `A` up to 4×4 is supported; that is the
//! class the diagonal inversion polynomial (`1/x` on `[1/κ,1]`, positive
//! spectrum) applies to. Verified end-to-end against the classical solve in
//! `linalg::solver` tests.

use aria_core::ast::{Circuit, CircuitBuilder};
use ndarray::Array2;
use num_complex::Complex64;

/// Eigendecomposition of a real symmetric matrix via cyclic Jacobi rotations.
/// Returns `(eigenvalues, eigenvectors)` with eigenvectors as the columns of the
/// returned matrix (`A = V · diag(eigenvalues) · Vᵀ`).
pub fn eigh_symmetric(a: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut m = a.clone();
    let mut v: Array2<f64> = Array2::eye(n);
    for _sweep in 0..100 {
        // Sum of squared upper off-diagonals — the Jacobi convergence measure.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[(p, q)] * m[(p, q)];
            }
        }
        if off < 1e-26 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if m[(p, q)].abs() < 1e-300 {
                    continue;
                }
                // Givens rotation `G` (identity except the (p,q) 2×2 block
                // `[[c, s], [-s, c]]`) that zeros `m[p][q]` under `Gᵀ·m·G`; the
                // matching angle is `θ = ½·atan2(2·a_pq, a_qq − a_pp)`. Apply as
                // full products (no in-place corner double-update).
                let theta = 0.5 * (2.0 * m[(p, q)]).atan2(m[(q, q)] - m[(p, p)]);
                let (s, c) = theta.sin_cos();
                let mut g: Array2<f64> = Array2::eye(n);
                g[(p, p)] = c;
                g[(q, q)] = c;
                g[(p, q)] = s;
                g[(q, p)] = -s;
                m = g.t().dot(&m).dot(&g);
                v = v.dot(&g);
            }
        }
    }
    let eigenvalues: Vec<f64> = (0..n).map(|i| m[(i, i)]).collect();
    (eigenvalues, v)
}

/// `Z-Y-Z` Euler angles `(φ, θ, ω)` of a 2×2 unitary `g` (up to global phase):
/// `g ∝ Rz(ω)·Ry(θ)·Rz(φ)`, matching `CircuitBuilder::rot_zyz(q, φ, θ, ω)`.
fn zyz_angles(g: &[[Complex64; 2]; 2]) -> (f64, f64, f64) {
    // Strip global phase to land in SU(2): divide by √det.
    let det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
    let phase = det.sqrt();
    let v00 = g[0][0] / phase;
    let v10 = g[1][0] / phase;
    let theta = 2.0 * v10.norm().atan2(v00.norm());
    let (phi, omega);
    if v00.norm() < 1e-9 {
        // θ ≈ π: only ω−φ is fixed; gauge-fix φ = 0.
        phi = 0.0;
        omega = 2.0 * v10.arg();
    } else if v10.norm() < 1e-9 {
        // θ ≈ 0: only ω+φ is fixed; gauge-fix φ = 0.
        phi = 0.0;
        omega = -2.0 * v00.arg();
    } else {
        omega = v10.arg() - v00.arg();
        phi = -v00.arg() - v10.arg();
    }
    (phi, theta, omega)
}

/// Conjugate-transpose of a 2×2 complex matrix.
fn dagger2(g: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    [
        [g[0][0].conj(), g[1][0].conj()],
        [g[0][1].conj(), g[1][1].conj()],
    ]
}

/// Append a single-qubit unitary `g` (2×2) on qubit `q`.
fn append_1q(b: &mut CircuitBuilder, g: &[[Complex64; 2]; 2], q: usize) {
    let (phi, theta, omega) = zyz_angles(g);
    b.rot_zyz(q, phi, theta, omega);
}

/// The magic-basis circuit `M` (`S·CX·H·CX`, up to global phase) on
/// `(msb, lsb)`, in time order. `M · U · M† = A₂ ⊗ B₂` for `U ∈ SO(4)`.
fn append_magic(b: &mut CircuitBuilder, msb: usize, lsb: usize) {
    b.s(lsb);
    b.cx(msb, lsb);
    b.h(msb);
    b.cx(msb, lsb);
}

/// The inverse magic circuit `M†` (`CX·H·CX·S†`) on `(msb, lsb)`, in time order.
fn append_magic_dagger(b: &mut CircuitBuilder, msb: usize, lsb: usize) {
    b.cx(msb, lsb);
    b.h(msb);
    b.cx(msb, lsb);
    b.sdg(lsb);
}

/// Rank-1 (Pitsianis–Van Loan) split of a 4×4 `P = A₂ ⊗ B₂` into its 2×2
/// factors — no SVD: rearrange to `R[2i+j, 2k+l] = P[2i+k, 2j+l]` (rank 1 =
/// `vec(A₂)·vec(B₂)ᵀ`), then read a scaled column/row off its largest entry.
fn kron_factors(p: &Array2<Complex64>) -> ([[Complex64; 2]; 2], [[Complex64; 2]; 2]) {
    let mut r = [[Complex64::new(0.0, 0.0); 4]; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    r[2 * i + j][2 * k + l] = p[(2 * i + k, 2 * j + l)];
                }
            }
        }
    }
    // Largest-magnitude pivot (p0, q0).
    let (mut p0, mut q0, mut best) = (0, 0, -1.0);
    for (i, row) in r.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val.norm() > best {
                best = val.norm();
                p0 = i;
                q0 = j;
            }
        }
    }
    let acol = [r[0][q0], r[1][q0], r[2][q0], r[3][q0]]; // ∝ vec(A₂)
    let piv = r[p0][q0];
    let brow = [
        r[p0][0] / piv,
        r[p0][1] / piv,
        r[p0][2] / piv,
        r[p0][3] / piv,
    ]; // vec(B₂)
       // Frobenius-normalize each 2×2 factor to a unitary (√2 norm; residual global
       // phase cancels in the U·(…)·U† conjugation).
    let anorm = (acol.iter().map(|c| c.norm_sqr()).sum::<f64>() / 2.0).sqrt();
    let bnorm = (brow.iter().map(|c| c.norm_sqr()).sum::<f64>() / 2.0).sqrt();
    let a2 = [
        [acol[0] / anorm, acol[1] / anorm],
        [acol[2] / anorm, acol[3] / anorm],
    ];
    let b2 = [
        [brow[0] / bnorm, brow[1] / bnorm],
        [brow[2] / bnorm, brow[3] / bnorm],
    ];
    (a2, b2)
}

/// 4×4 complex matrix product.
fn matmul4(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    a.dot(b)
}

/// The fixed magic-basis matrix `M` (Vatan–Williams).
fn magic_matrix() -> Array2<Complex64> {
    let r = 1.0 / 2.0_f64.sqrt();
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(r, 0.0);
    let i = Complex64::new(0.0, r);
    Array2::from_shape_vec(
        (4, 4),
        vec![o, z, z, i, z, i, o, z, z, i, -o, z, o, z, z, -i],
    )
    .unwrap()
}

/// Append the eigenbasis rotation `U` (real orthogonal, dim 2 or 4) — or its
/// transpose when `dagger` — onto `b`, acting on the `sys` qubits
/// (`sys[0]` = system LSB, ascending). Used to conjugate the diagonal block
/// encoding into `A`'s eigenbasis.
pub fn append_orthogonal(b: &mut CircuitBuilder, u: &Array2<f64>, sys: &[usize], dagger: bool) {
    let dim = u.nrows();
    if dim == 2 {
        let mut g = [
            [
                Complex64::new(u[(0, 0)], 0.0),
                Complex64::new(u[(0, 1)], 0.0),
            ],
            [
                Complex64::new(u[(1, 0)], 0.0),
                Complex64::new(u[(1, 1)], 0.0),
            ],
        ];
        if dagger {
            g = dagger2(&g);
        }
        append_1q(b, &g, sys[0]);
        return;
    }
    // dim == 4: SO(4) via the magic basis. Ensure det U = +1.
    let mut uc: Array2<Complex64> = u.mapv(|x| Complex64::new(x, 0.0));
    if det4_real(u) < 0.0 {
        for row in 0..4 {
            uc[(row, 0)] = -uc[(row, 0)];
        }
    }
    let m = magic_matrix();
    let mdag = m.t().mapv(|c| c.conj());
    // P = M · U · M†  = A₂ ⊗ B₂.
    let p = matmul4(&matmul4(&m, &uc), &mdag);
    let (mut a2, mut b2) = kron_factors(&p);
    if dagger {
        a2 = dagger2(&a2);
        b2 = dagger2(&b2);
    }
    // U = M† · (A₂ ⊗ B₂) · M  ⇒  time order: apply M, then A₂⊗B₂, then M†.
    // Magic-circuit convention: python-q0 = system MSB, python-q1 = system LSB.
    let (lsb, msb) = (sys[0], sys[1]);
    append_magic(b, msb, lsb);
    append_1q(b, &a2, msb); // A₂ on the MSB
    append_1q(b, &b2, lsb); // B₂ on the LSB
    append_magic_dagger(b, msb, lsb);
}

/// Determinant of a real matrix (dim 2 or 4) via cofactor expansion.
fn det4_real(u: &Array2<f64>) -> f64 {
    let n = u.nrows();
    if n == 2 {
        return u[(0, 0)] * u[(1, 1)] - u[(0, 1)] * u[(1, 0)];
    }
    // Laplace expansion along row 0 for n == 4.
    let mut det = 0.0;
    for c in 0..4 {
        let mut sub = Array2::<f64>::zeros((3, 3));
        for i in 1..4 {
            let mut cc = 0;
            for j in 0..4 {
                if j == c {
                    continue;
                }
                sub[(i - 1, cc)] = u[(i, j)];
                cc += 1;
            }
        }
        let sign = if c % 2 == 0 { 1.0 } else { -1.0 };
        det += sign * u[(0, c)] * det3(&sub);
    }
    det
}

fn det3(m: &Array2<f64>) -> f64 {
    m[(0, 0)] * (m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)])
        - m[(0, 1)] * (m[(1, 0)] * m[(2, 2)] - m[(1, 2)] * m[(2, 0)])
        + m[(0, 2)] * (m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)])
}

/// Build the block-encoding circuit of `A = U·diag(spectrum)·Uᵀ` by conjugating
/// the diagonal block encoding `W_D` with the eigenbasis rotation `U`:
/// `W_A = (I⊗U)·W_D·(I⊗U†)`. `spectrum`/`u` come from [`eigh_symmetric`]; the
/// ancilla is qubit 0, the system is qubits `1..=n_system`.
pub fn conjugated_block_encoding(w_d: &Circuit, u: &Array2<f64>, n_system: usize) -> Circuit {
    let sys: Vec<usize> = (1..=n_system).collect();
    let mut b = CircuitBuilder::new("block_encode_hermitian", 1 + n_system, 0);
    // W_A = U · W_D · U† ⇒ time order: apply U† on system, then W_D, then U.
    append_orthogonal(&mut b, u, &sys, true); // U†
    let mut circ = b.build();
    circ.append_circuit(w_d, &std::collections::HashMap::new(), None); // W_D
    let mut b2 = CircuitBuilder::new("u_forward", 1 + n_system, 0);
    append_orthogonal(&mut b2, u, &sys, false); // U
    circ.append_circuit(&b2.build(), &std::collections::HashMap::new(), None);
    circ
}
