use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// A 2x2 unitary matrix stored as [row0col0, row0col1, row1col0, row1col1].
pub type Gate1Q = [Complex64; 4];

/// A 4x4 unitary matrix stored row-major.
pub type Gate2Q = [Complex64; 16];

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);
const ISQRT2: Complex64 = Complex64::new(FRAC_1_SQRT_2, 0.0);
const I: Complex64 = Complex64::new(0.0, 1.0);
const NEG_I: Complex64 = Complex64::new(0.0, -1.0);

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn ei(theta: f64) -> Complex64 {
    Complex64::new(theta.cos(), theta.sin())
}

// Single-qubit gate matrices

pub fn h() -> Gate1Q {
    [ISQRT2, ISQRT2, ISQRT2, -ISQRT2]
}

pub fn x() -> Gate1Q {
    [ZERO, ONE, ONE, ZERO]
}

pub fn y() -> Gate1Q {
    [ZERO, -I, I, ZERO]
}

pub fn z() -> Gate1Q {
    [ONE, ZERO, ZERO, -ONE]
}

pub fn s() -> Gate1Q {
    [ONE, ZERO, ZERO, I]
}

pub fn sdg() -> Gate1Q {
    [ONE, ZERO, ZERO, NEG_I]
}

pub fn t() -> Gate1Q {
    [ONE, ZERO, ZERO, ei(PI / 4.0)]
}

pub fn tdg() -> Gate1Q {
    [ONE, ZERO, ZERO, ei(-PI / 4.0)]
}

pub fn _id() -> Gate1Q {
    [ONE, ZERO, ZERO, ONE]
}

pub fn rx(theta: f64) -> Gate1Q {
    let cv = (theta / 2.0).cos();
    let sv = (theta / 2.0).sin();
    [c(cv, 0.0), c(0.0, -sv), c(0.0, -sv), c(cv, 0.0)]
}

pub fn ry(theta: f64) -> Gate1Q {
    let cv = (theta / 2.0).cos();
    let sv = (theta / 2.0).sin();
    [c(cv, 0.0), c(-sv, 0.0), c(sv, 0.0), c(cv, 0.0)]
}

pub fn rz(theta: f64) -> Gate1Q {
    [ei(-theta / 2.0), ZERO, ZERO, ei(theta / 2.0)]
}

pub fn u3(theta: f64, phi: f64, lambda: f64) -> Gate1Q {
    let ct = (theta / 2.0).cos();
    let st = (theta / 2.0).sin();
    [
        c(ct, 0.0),
        -ei(lambda) * st,
        ei(phi) * st,
        ei(phi + lambda) * ct,
    ]
}

pub fn u2(phi: f64, lambda: f64) -> Gate1Q {
    u3(PI / 2.0, phi, lambda)
}

pub fn u1(lambda: f64) -> Gate1Q {
    [ONE, ZERO, ZERO, ei(lambda)]
}

// --- Gate derivative matrices (dU/dθ) for adjoint differentiation ---

/// dRx/dθ = (1/2) * [[-sin(θ/2), -i·cos(θ/2)], [-i·cos(θ/2), -sin(θ/2)]]
///
/// Equivalent matrix-level identity verified in Lean:
/// `Verification/Adjoint/ParamShiftRx.lean::drx_eq_neg_half_i_x_rx`
/// proves `dRx(θ) = -(i/2) · (X · Rx(θ))` — the generator form that
/// underlies the standard 2-term parameter-shift rule for Rx.
pub fn drx(theta: f64) -> Gate1Q {
    let cv = (theta / 2.0).cos();
    let sv = (theta / 2.0).sin();
    [
        c(-sv / 2.0, 0.0),
        c(0.0, -cv / 2.0),
        c(0.0, -cv / 2.0),
        c(-sv / 2.0, 0.0),
    ]
}

/// dRy/dθ = (1/2) * [[-sin(θ/2), -cos(θ/2)], [cos(θ/2), -sin(θ/2)]]
///
/// Equivalent matrix-level identity verified in Lean:
/// `Verification/Adjoint/ParamShiftRy.lean::dry_eq_neg_half_i_y_ry`
/// proves `dRy(θ) = -(i/2) · (Y · Ry(θ))` — same shape as
/// `drx_eq_neg_half_i_x_rx` with Y replacing X.
pub fn dry(theta: f64) -> Gate1Q {
    let cv = (theta / 2.0).cos();
    let sv = (theta / 2.0).sin();
    [
        c(-sv / 2.0, 0.0),
        c(-cv / 2.0, 0.0),
        c(cv / 2.0, 0.0),
        c(-sv / 2.0, 0.0),
    ]
}

/// dRz/dθ = (1/2) * [[-i·e^{-iθ/2}, 0], [0, i·e^{iθ/2}]]
///
/// Equivalent matrix-level identity verified in Lean:
/// `Verification/Adjoint/ParamShiftRz.lean::drz_eq_neg_half_i_z_rz`
/// proves `dRz(θ) = -(i/2) · (Z · Rz(θ))` — same shape as
/// `drx_eq_neg_half_i_x_rx` and `dry_eq_neg_half_i_y_ry`,
/// completing the {Rx, Ry, Rz} eigenvalue-±1/2 generator trio.
pub fn drz(theta: f64) -> Gate1Q {
    let neg_half_i = c(0.0, -0.5);
    let pos_half_i = c(0.0, 0.5);
    [
        neg_half_i * ei(-theta / 2.0),
        ZERO,
        ZERO,
        pos_half_i * ei(theta / 2.0),
    ]
}

/// dU1/dλ = [[0, 0], [0, i·e^{iλ}]]
///
/// Equivalent matrix-level identity verified in Lean:
/// `Verification/Adjoint/ParamShiftU1.lean::du1_dlambda_eq_i_u1_p1`
/// proves `dU1/dλ = i · (U1(λ) · P₁)` (right-projector form), and
/// `du1_dlambda_eq_i_p1_u1` shows the equivalent left-projector
/// form `dU1/dλ = i · (P₁ · U1(λ))` — both hold because U1 is
/// diagonal and commutes with P₁ = diag(0, 1).
pub fn du1_dl(lambda: f64) -> Gate1Q {
    [ZERO, ZERO, ZERO, I * ei(lambda)]
}

/// dU3/dθ = (1/2) * [[-sin(θ/2), -e^{iλ}·cos(θ/2)], [e^{iφ}·cos(θ/2), -e^{i(φ+λ)}·sin(θ/2)]]
///
/// Equivalent matrix-level identity verified in Lean:
/// `Verification/Adjoint/ParamShiftU3.lean::du3_dtheta_eq_half_u3_shifted`
/// proves `dU3/dθ = (1/2) · U3(θ + π, φ, λ)` — the parameter-shift
/// form of the θ derivative. The chain-rule factor 1/2 is folded
/// into a phase-shifted U3 evaluation, which is what underlies the
/// standard ±π/2 parameter-shift rule for U3's θ slot.
pub fn du3_dt(theta: f64, phi: f64, lambda: f64) -> Gate1Q {
    let ct = (theta / 2.0).cos();
    let st = (theta / 2.0).sin();
    [
        c(-st / 2.0, 0.0),
        -ei(lambda) * (ct / 2.0),
        ei(phi) * (ct / 2.0),
        -ei(phi + lambda) * (st / 2.0),
    ]
}

/// dU3/dφ = [[0, 0], [i·e^{iφ}·sin(θ/2), i·e^{i(φ+λ)}·cos(θ/2)]]
///
/// Equivalent matrix-level identity verified in Lean:
/// `Verification/Adjoint/ParamShiftU3.lean::du3_dphi_eq_i_p1_u3`
/// proves `dU3/dφ = i · (P₁ · U3(θ, φ, λ))` where `P₁ = diag(0, 1)`
/// is the projector onto |1⟩ — φ is U3's "left-side" rank-1 generator.
pub fn du3_dp(theta: f64, phi: f64, lambda: f64) -> Gate1Q {
    let ct = (theta / 2.0).cos();
    let st = (theta / 2.0).sin();
    [ZERO, ZERO, I * ei(phi) * st, I * ei(phi + lambda) * ct]
}

/// dU3/dλ = [[0, -i·e^{iλ}·sin(θ/2)], [0, i·e^{i(φ+λ)}·cos(θ/2)]]
///
/// Equivalent matrix-level identity verified in Lean:
/// `Verification/Adjoint/ParamShiftU3.lean::du3_dlambda_eq_i_u3_p1`
/// proves `dU3/dλ = i · (U3(θ, φ, λ) · P₁)` — λ is U3's "right-side"
/// rank-1 generator (mirrors the φ identity). The θ partial doesn't
/// have a single-generator matrix form; spectral / Euler-decomposition
/// proof lives in `Adjoint/Soundness.lean` (Phase B).
pub fn du3_dl(theta: f64, phi: f64, lambda: f64) -> Gate1Q {
    let ct = (theta / 2.0).cos();
    let st = (theta / 2.0).sin();
    [ZERO, -I * ei(lambda) * st, ZERO, I * ei(phi + lambda) * ct]
}

/// dU2/dφ = dU3/dφ(π/2, φ, λ)
///
/// U2(φ, λ) = U3(π/2, φ, λ), so its φ-derivative is U3's φ-derivative
/// at θ=π/2. The matrix-level identity
/// `Verification/Adjoint/ParamShiftU3.lean::du3_dphi_eq_i_p1_u3` —
/// `dU3/dφ = i · (P₁ · U3)` — applies directly with θ fixed at π/2.
pub fn du2_dp(phi: f64, lambda: f64) -> Gate1Q {
    du3_dp(PI / 2.0, phi, lambda)
}

/// dU2/dλ = dU3/dλ(π/2, φ, λ)
///
/// U2(φ, λ) = U3(π/2, φ, λ). The matrix-level identity
/// `Verification/Adjoint/ParamShiftU3.lean::du3_dlambda_eq_i_u3_p1` —
/// `dU3/dλ = i · (U3 · P₁)` — applies directly with θ fixed at π/2.
pub fn du2_dl(phi: f64, lambda: f64) -> Gate1Q {
    du3_dl(PI / 2.0, phi, lambda)
}

/// Conjugate-transpose of a 1Q gate matrix.
pub fn adjoint_1q(g: &Gate1Q) -> Gate1Q {
    [g[0].conj(), g[2].conj(), g[1].conj(), g[3].conj()]
}

/// Conjugate-transpose of a 2Q gate matrix.
pub fn adjoint_2q(g: &Gate2Q) -> Gate2Q {
    let mut adj = [ZERO; 16];
    for i in 0..4 {
        for j in 0..4 {
            adj[i * 4 + j] = g[j * 4 + i].conj();
        }
    }
    adj
}

// Two-qubit gate matrices (4x4, row-major)
// Basis order: |00>, |01>, |10>, |11>

pub fn cx() -> Gate2Q {
    [
        ONE, ZERO, ZERO, ZERO, // |00> -> |00>
        ZERO, ONE, ZERO, ZERO, // |01> -> |01>
        ZERO, ZERO, ZERO, ONE, // |10> -> |11>
        ZERO, ZERO, ONE, ZERO, // |11> -> |10>
    ]
}

pub fn cy() -> Gate2Q {
    [
        ONE, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, -I, ZERO, ZERO, I, ZERO,
    ]
}

pub fn cz() -> Gate2Q {
    [
        ONE, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, -ONE,
    ]
}

pub fn swap() -> Gate2Q {
    [
        ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,
    ]
}

pub fn crz(theta: f64) -> Gate2Q {
    [
        ONE,
        ZERO,
        ZERO,
        ZERO,
        ZERO,
        ONE,
        ZERO,
        ZERO,
        ZERO,
        ZERO,
        ei(-theta / 2.0),
        ZERO,
        ZERO,
        ZERO,
        ZERO,
        ei(theta / 2.0),
    ]
}

pub fn cu3(theta: f64, phi: f64, lambda: f64) -> Gate2Q {
    let g = u3(theta, phi, lambda);
    [
        ONE, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, g[0], g[1], ZERO, ZERO, g[2],
        g[3],
    ]
}

// --- 2Q gate derivatives ---

/// dCRz/dθ: only the lower-right 2x2 block changes
///
/// Equivalent matrix-level identity verified in Lean:
/// `Verification/Adjoint/ParamShiftCRz.lean::dcrz_eq_neg_half_i_m_crz`
/// proves `dCRz(θ) = -(i/2) · (M · CRz(θ))` where `M = diag(0, 0, 1, -1)` is
/// the generator restricted to the |1⟩-control branch (eigenvalues
/// {0, 0, ±1/2} after the (1/2) factor is absorbed).
pub fn dcrz(theta: f64) -> Gate2Q {
    let neg_half_i = c(0.0, -0.5);
    let pos_half_i = c(0.0, 0.5);
    [
        ZERO,
        ZERO,
        ZERO,
        ZERO,
        ZERO,
        ZERO,
        ZERO,
        ZERO,
        ZERO,
        ZERO,
        neg_half_i * ei(-theta / 2.0),
        ZERO,
        ZERO,
        ZERO,
        ZERO,
        pos_half_i * ei(theta / 2.0),
    ]
}

/// dCU3/dθ: lower-right 2x2 = dU3/dθ
///
/// Equivalent matrix-level identity verified in Lean:
/// `Verification/Adjoint/ParamShiftCU3.lean::dcu3_dtheta_eq_half_q_ctl_cu3_shifted`
/// proves `dCU3/dθ = (1/2) · (Q_ctl · CU3(θ+π, φ, λ))` where
/// `Q_ctl = diag(0, 0, 1, 1)` projects onto the |1⟩-control branch.
pub fn dcu3_dt(theta: f64, phi: f64, lambda: f64) -> Gate2Q {
    let dg = du3_dt(theta, phi, lambda);
    [
        ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, dg[0], dg[1], ZERO, ZERO,
        dg[2], dg[3],
    ]
}

/// dCU3/dφ: lower-right 2x2 = dU3/dφ
///
/// Equivalent matrix-level identity verified in Lean:
/// `Verification/Adjoint/ParamShiftCU3.lean::dcu3_dphi_eq_i_p11_cu3`
/// proves `dCU3/dφ = i · (P11 · CU3)` where `P11 = |11⟩⟨11|` —
/// the lift of U3's `P₁ = diag(0, 1)` to the |1⟩-control × |1⟩-
/// target subspace.
pub fn dcu3_dp(theta: f64, phi: f64, lambda: f64) -> Gate2Q {
    let dg = du3_dp(theta, phi, lambda);
    [
        ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, dg[0], dg[1], ZERO, ZERO,
        dg[2], dg[3],
    ]
}

/// dCU3/dλ: lower-right 2x2 = dU3/dλ
///
/// Equivalent matrix-level identity verified in Lean:
/// `Verification/Adjoint/ParamShiftCU3.lean::dcu3_dlambda_eq_i_cu3_p11`
/// proves `dCU3/dλ = i · (CU3 · P11)` (right-projector form).
pub fn dcu3_dl(theta: f64, phi: f64, lambda: f64) -> Gate2Q {
    let dg = du3_dl(theta, phi, lambda);
    [
        ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, dg[0], dg[1], ZERO, ZERO,
        dg[2], dg[3],
    ]
}
