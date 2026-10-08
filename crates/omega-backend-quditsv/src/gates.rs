// SPDX-License-Identifier: Apache-2.0
//! Dense `d×d` (and `d_a·d_b`-square) gate matrices, row-major.
//!
//! Conventions, stated once and pinned by `tests/anchors.rs` and the
//! mqt.qudits harness:
//!
//! * A one-wire matrix `M[r][c]` maps `|c⟩ → Σ_r M[r][c]|r⟩`.
//! * A two-wire matrix on `(a, b)` is indexed `r = ra·d_b + rb`: the FIRST
//!   wire named is the more significant index of the matrix, exactly as
//!   `omega-backend-statevector`'s `Gate2Q` treats `q0`. For `csum a, b`
//!   that makes `a` the control.
//! * `ω_d = e^{2πi/d}`. `F_d[r][c] = ω^{rc}/√d`, `X_d|k⟩ = |k+1 mod d⟩`,
//!   `Z_d|k⟩ = ω^k|k⟩`. At `d = 2` these are `H`, `X`, `Z` exactly.
//! * `rxy(i, j, θ, φ)` on the pair `{|i⟩,|j⟩}` is
//!   `exp(−iθ/2 (cos φ X + sin φ Y))`, identity elsewhere. At `φ = 0` it is
//!   `Rx(θ)`; at `φ = π/2`, `Ry(θ)`. This is mqt.qudits' `R` gate,
//!   Gell-Mann `s`/`a` on `(i, j)` — pinned numerically against it.

use num_complex::Complex64;
use omega_backend_statevector::gates::{Gate1Q, Gate2Q};
use std::f64::consts::PI;

/// Row-major square matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Dense {
    pub dim: usize,
    pub m: Vec<Complex64>,
}

impl Dense {
    pub fn identity(dim: usize) -> Self {
        let mut m = vec![Complex64::new(0.0, 0.0); dim * dim];
        for k in 0..dim {
            m[k * dim + k] = Complex64::new(1.0, 0.0);
        }
        Self { dim, m }
    }

    #[inline]
    pub fn at(&self, r: usize, c: usize) -> Complex64 {
        self.m[r * self.dim + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: Complex64) {
        self.m[r * self.dim + c] = v;
    }

    /// `self · other`.
    pub fn mul(&self, other: &Dense) -> Dense {
        assert_eq!(self.dim, other.dim);
        let d = self.dim;
        let mut out = Dense {
            dim: d,
            m: vec![Complex64::new(0.0, 0.0); d * d],
        };
        for r in 0..d {
            for c in 0..d {
                let mut s = Complex64::new(0.0, 0.0);
                for k in 0..d {
                    s += self.at(r, k) * other.at(k, c);
                }
                out.set(r, c, s);
            }
        }
        out
    }

    /// Conjugate transpose.
    pub fn dagger(&self) -> Dense {
        let d = self.dim;
        let mut out = self.clone();
        for r in 0..d {
            for c in 0..d {
                out.set(r, c, self.at(c, r).conj());
            }
        }
        out
    }
}

fn omega(d: usize) -> Complex64 {
    Complex64::from_polar(1.0, 2.0 * PI / d as f64)
}

/// Fourier gate `F_d[r][c] = ω^{rc}/√d`. `F_2 = H`.
pub fn fourier(d: usize) -> Dense {
    let w = omega(d);
    let norm = 1.0 / (d as f64).sqrt();
    let mut g = Dense::identity(d);
    for r in 0..d {
        for c in 0..d {
            g.set(r, c, w.powu(((r * c) % d) as u32) * norm);
        }
    }
    g
}

/// Cyclic shift `X_d|k⟩ = |k+1 mod d⟩`. `X_2 = X`.
pub fn shift(d: usize) -> Dense {
    let mut g = Dense {
        dim: d,
        m: vec![Complex64::new(0.0, 0.0); d * d],
    };
    for k in 0..d {
        g.set((k + 1) % d, k, Complex64::new(1.0, 0.0));
    }
    g
}

/// Clock `Z_d|k⟩ = ω^k|k⟩`. `Z_2 = Z`.
pub fn clock(d: usize) -> Dense {
    let w = omega(d);
    let mut g = Dense::identity(d);
    for k in 0..d {
        g.set(k, k, w.powu(k as u32));
    }
    g
}

/// `rxy(i, j, θ, φ)`: `exp(−iθ/2 (cos φ X + sin φ Y))` on `{|i⟩, |j⟩}`.
///
/// Requires `i < j < d` — the caller refuses otherwise, naming the levels,
/// because `Y` on `(i, j)` is `−Y` on `(j, i)` and a silent swap would flip
/// the sign of `φ`.
pub fn rxy(d: usize, i: usize, j: usize, theta: f64, phi: f64) -> Dense {
    debug_assert!(i < j && j < d);
    let (c2, s2) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    let mut g = Dense::identity(d);
    g.set(i, i, Complex64::new(c2, 0.0));
    g.set(j, j, Complex64::new(c2, 0.0));
    // −i sin(θ/2) (cos φ X + sin φ Y) on the pair:
    //   X = [[0,1],[1,0]],  Y = [[0,−i],[i,0]]
    //   (i,j) entry: −i s2 (cos φ − i sin φ) = −i s2 e^{−iφ}
    //   (j,i) entry: −i s2 (cos φ + i sin φ) = −i s2 e^{+iφ}
    let mi = Complex64::new(0.0, -1.0);
    g.set(i, j, mi * s2 * Complex64::from_polar(1.0, -phi));
    g.set(j, i, mi * s2 * Complex64::from_polar(1.0, phi));
    g
}

/// `csum`: `|c⟩|t⟩ → |c⟩|t + c mod d_t⟩`, control first (more significant).
/// At `d_c = d_t = 2` this is `CX`.
pub fn csum(d_c: usize, d_t: usize) -> Dense {
    let dim = d_c * d_t;
    let mut g = Dense {
        dim,
        m: vec![Complex64::new(0.0, 0.0); dim * dim],
    };
    for c in 0..d_c {
        for t in 0..d_t {
            let from = c * d_t + t;
            let to = c * d_t + (t + c) % d_t;
            g.set(to, from, Complex64::new(1.0, 0.0));
        }
    }
    g
}

/// A qubit gate matrix, as this crate's dense form.
pub fn embed1(g: &Gate1Q) -> Dense {
    Dense {
        dim: 2,
        m: g.to_vec(),
    }
}

/// A two-qubit gate matrix (row-major `4×4`, `q0` more significant).
pub fn embed2(g: &Gate2Q) -> Dense {
    Dense {
        dim: 4,
        m: g.to_vec(),
    }
}

/// Toffoli on three qubits, `(c0, c1, t)`, index `c0·4 + c1·2 + t`.
pub fn ccx() -> Dense {
    let mut g = Dense::identity(8);
    g.set(6, 6, Complex64::new(0.0, 0.0));
    g.set(7, 7, Complex64::new(0.0, 0.0));
    g.set(6, 7, Complex64::new(1.0, 0.0));
    g.set(7, 6, Complex64::new(1.0, 0.0));
    g
}

/// Fredkin on three qubits, `(c, a, b)`, index `c·4 + a·2 + b`.
pub fn cswap() -> Dense {
    let mut g = Dense::identity(8);
    // c = 1: swap a, b  ⇒  |101⟩ ↔ |110⟩  (indices 5 and 6)
    g.set(5, 5, Complex64::new(0.0, 0.0));
    g.set(6, 6, Complex64::new(0.0, 0.0));
    g.set(5, 6, Complex64::new(1.0, 0.0));
    g.set(6, 5, Complex64::new(1.0, 0.0));
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use omega_backend_statevector::gates as qg;

    fn close(a: &Dense, b: &Dense) -> bool {
        a.dim == b.dim && a.m.iter().zip(&b.m).all(|(x, y)| (x - y).norm() < 1e-12)
    }

    #[test]
    fn the_generalised_gates_are_the_qubit_gates_at_d_2() {
        assert!(close(&fourier(2), &embed1(&qg::h())));
        assert!(close(&shift(2), &embed1(&qg::x())));
        assert!(close(&clock(2), &embed1(&qg::z())));
        assert!(close(&csum(2, 2), &embed2(&qg::cx())));
        assert!(close(&rxy(2, 0, 1, 0.7, 0.0), &embed1(&qg::rx(0.7))));
        assert!(close(&rxy(2, 0, 1, 0.7, PI / 2.0), &embed1(&qg::ry(0.7))));
    }

    #[test]
    fn every_gate_is_unitary_at_d_3_and_5() {
        for d in [3usize, 5] {
            for g in [fourier(d), shift(d), clock(d), rxy(d, 0, 2, 1.1, 0.4)] {
                assert!(close(&g.dagger().mul(&g), &Dense::identity(d)), "d={d}");
            }
        }
        let g = csum(3, 5);
        assert!(close(&g.dagger().mul(&g), &Dense::identity(15)));
    }

    #[test]
    fn shift_cycles_and_clock_phases() {
        let x = shift(3);
        // X_3 |2⟩ = |0⟩
        assert_eq!(x.at(0, 2), Complex64::new(1.0, 0.0));
        assert!(close(&x.mul(&x).mul(&x), &Dense::identity(3)));
        let z = clock(3);
        assert!((z.at(1, 1) - Complex64::from_polar(1.0, 2.0 * PI / 3.0)).norm() < 1e-12);
    }
}
