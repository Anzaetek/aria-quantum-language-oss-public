use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

pub type Gate1Q = [Complex64; 4];
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

// Single-qubit gates

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
    [ONE, ZERO, ZERO, ei(std::f64::consts::PI / 4.0)]
}

pub fn tdg() -> Gate1Q {
    [ONE, ZERO, ZERO, ei(-std::f64::consts::PI / 4.0)]
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
    u3(std::f64::consts::PI / 2.0, phi, lambda)
}

pub fn u1(lambda: f64) -> Gate1Q {
    [ONE, ZERO, ZERO, ei(lambda)]
}

// Two-qubit gates

pub fn cx() -> Gate2Q {
    [
        ONE, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ONE, ZERO,
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

/// RBS (Givens rotation): `exp(−i·θ/2·(Y⊗X − X⊗Y))` — identity on
/// {|00⟩, |11⟩}, `[[cos θ, −sin θ], [sin θ, cos θ]]` on span{|01⟩, |10⟩}.
/// Mirrors `omega-backend-statevector/src/gates.rs::rbs`.
pub fn rbs(theta: f64) -> Gate2Q {
    let cv = c(theta.cos(), 0.0);
    let sv = c(theta.sin(), 0.0);
    [
        ONE, ZERO, ZERO, ZERO, //
        ZERO, cv, -sv, ZERO, //
        ZERO, sv, cv, ZERO, //
        ZERO, ZERO, ZERO, ONE,
    ]
}
