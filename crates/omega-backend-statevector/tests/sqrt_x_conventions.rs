// SPDX-License-Identifier: Apache-2.0
//! `sx` / `sxdg` conventions, pinned at the MATRIX level.
//!
//! These are first-class `GateKind` variants rather than aliases for
//! `U3(π/2, −π/2, π/2)`, and both halves of that decision are checked here:
//!
//! 1. **The matrices are exactly Qiskit's**, not `U3` up to a global phase.
//!    The alias is off by `e^{iπ/4}`: `|sx − U3| = 0.541`, `det(sx) = i` vs
//!    `det(U3) = 1`. Invisible in counts and expectations, wrong in any
//!    statevector comparison.
//! 2. **`sx` is CLIFFORD** — `sx·sx = X`. That is the load-bearing half:
//!    `PauliBackend` rejects `U3` outright as non-Clifford, so aliasing would
//!    make the stabilizer backend refuse an all-Clifford circuit. Measured:
//!    with the native variant, `pauli` covers 8 of 14 counts fixtures instead
//!    of 7.
//!
//! Distribution-level checks cannot see (1) at all — a global phase moves no
//! probability — which is why these compare matrices directly. That is the
//! same lesson as the CV/piquasso work, where comparing `⟨n⟩` scalars could
//! not detect a no-op Kerr or a flipped squeezing sign.
//!
//! The Clifford *conjugation* rules the stabilizer tableau implements are
//! proved separately in `proofs/lean4/QuantumProofs/SqrtX.lean`.

use num_complex::Complex64;
use omega_backend_statevector::gates;

type M = [Complex64; 4];

fn mul(a: &M, b: &M) -> M {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

fn max_diff(a: &M, b: &M) -> f64 {
    (0..4).map(|i| (a[i] - b[i]).norm()).fold(0.0, f64::max)
}

fn det(m: &M) -> Complex64 {
    m[0] * m[3] - m[1] * m[2]
}

fn x_gate() -> M {
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    [z, o, o, z]
}

fn identity() -> M {
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    [o, z, z, o]
}

/// The exact Qiskit `SXGate` / `SXdgGate` entries, independently written out.
#[test]
fn matrices_match_the_qiskit_closed_form() {
    let half = 0.5;
    let expect_sx: M = [
        Complex64::new(half, half),
        Complex64::new(half, -half),
        Complex64::new(half, -half),
        Complex64::new(half, half),
    ];
    let expect_sxdg: M = [
        Complex64::new(half, -half),
        Complex64::new(half, half),
        Complex64::new(half, half),
        Complex64::new(half, -half),
    ];
    assert!(max_diff(&gates::sx(), &expect_sx) < 1e-15);
    assert!(max_diff(&gates::sxdg(), &expect_sxdg) < 1e-15);
}

/// `sx·sx = X` — the identity that makes it Clifford, and the reason the
/// stabilizer backend can take it at all.
#[test]
fn sx_squared_is_x() {
    let d = max_diff(&mul(&gates::sx(), &gates::sx()), &x_gate());
    assert!(d < 1e-15, "sx*sx must equal X exactly, got max diff {d:.3e}");
}

/// `sxdg = sx†`, so the adjoint pass may substitute one for the other.
#[test]
fn sxdg_is_the_inverse_of_sx() {
    let d = max_diff(&mul(&gates::sx(), &gates::sxdg()), &identity());
    assert!(d < 1e-15, "sx*sxdg must equal I, got {d:.3e}");
    let d2 = max_diff(&mul(&gates::sxdg(), &gates::sx()), &identity());
    assert!(d2 < 1e-15, "sxdg*sx must equal I, got {d2:.3e}");
}

/// **Guard the guard.** The rejected alias must stay measurably wrong, so a
/// future "simplification" to `U3` cannot pass silently.
///
/// `U3(π/2, −π/2, π/2) = (1/√2)[[1, −i], [−i, 1]]`, which is `sx` divided by
/// `e^{iπ/4}`. Two independent signatures of the difference are asserted: the
/// raw entry-wise gap, and the determinant.
#[test]
fn the_u3_alias_is_measurably_wrong() {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let u3: M = [
        Complex64::new(s, 0.0),
        Complex64::new(0.0, -s),
        Complex64::new(0.0, -s),
        Complex64::new(s, 0.0),
    ];
    let gap = max_diff(&gates::sx(), &u3);
    assert!(
        (gap - 0.5411961).abs() < 1e-6,
        "the sx/U3 gap should be ~0.541 (a global e^{{iπ/4}}), got {gap:.6}"
    );
    // det(sx) = i, det(u3) = 1 — a determinant is basis-independent, so this
    // is a second, structurally different witness to the same fact.
    assert!((det(&gates::sx()) - Complex64::new(0.0, 1.0)).norm() < 1e-15);
    assert!((det(&u3) - Complex64::new(1.0, 0.0)).norm() < 1e-15);

    // ...and confirm it IS only a global phase, so the claim is precise:
    // multiplying u3 by e^{iπ/4} recovers sx exactly.
    let phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    let scaled: M = [u3[0] * phase, u3[1] * phase, u3[2] * phase, u3[3] * phase];
    assert!(
        max_diff(&gates::sx(), &scaled) < 1e-15,
        "sx must equal e^{{iπ/4}}·U3(π/2,−π/2,π/2) exactly"
    );
}

/// Both gates are unitary. Cheap, and it would catch a typo in a single entry
/// that the identities above might not.
#[test]
fn both_gates_are_unitary() {
    for (name, m) in [("sx", gates::sx()), ("sxdg", gates::sxdg())] {
        let dag: M = [m[0].conj(), m[2].conj(), m[1].conj(), m[3].conj()];
        let d = max_diff(&mul(&m, &dag), &identity());
        assert!(d < 1e-15, "{name} is not unitary: U·U† differs by {d:.3e}");
    }
}
