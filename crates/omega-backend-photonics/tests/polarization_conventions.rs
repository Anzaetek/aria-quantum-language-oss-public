// SPDX-License-Identifier: Apache-2.0
//! Pin the polarization conventions at the **matrix** level, from real source.
//!
//! These tests lower actual OPTICQASM text and read the resulting mode unitary.
//! Going through the parser is deliberate: the polarization expansion lives in
//! lowering, so a test built from a hand-assembled op list would exercise
//! everything except the code that could be wrong.
//!
//! ## Why matrix-level, and why this exact assertion
//!
//! The first draft of the plan proposed pinning *Perceval's* HWP matrix. That
//! check is worthless here — it passes regardless of what Aria does. The
//! assertion that bites is the one below: **Aria's lowered unitary must equal
//! Perceval's matrix, global phase included.**
//!
//! That distinction is not academic. The plan's original decomposition,
//! `HWP(θ) = BSrx(2θ,0)·PS(π)`, was verified to 1e-16 — against the *i-less
//! textbook* matrix, while the same document argued the `i` was a correctness
//! issue. Determinants make the contradiction unmissable: `det = −1` for that
//! product, `+1` for Perceval's. These tests exist so that error cannot recur
//! silently.
//!
//! Reference values are taken from Perceval 1.2.4 and pinned in
//! `crates/omega-bridges/python/tests/test_perceval_conventions.py`.

use num_complex::Complex64;
use omega_backend_photonics::components::{build_unitary, PhotonicOp};
use omega_core::circuit::GateKind;
use omega_parser::lower_to_ir;

const TOL: f64 = 1e-12;

/// Lower OPTICQASM source to the mode unitary its ops build.
fn unitary_of(src: &str) -> (usize, Vec<Vec<Complex64>>) {
    let (modes, ops) = lower_to_photonic_ops(src);
    (modes, build_unitary(modes, &ops))
}

fn lower_to_photonic_ops(src: &str) -> (usize, Vec<PhotonicOp>) {
    let ir = lower_to_ir(src).expect("lower");
    let mut ops = Vec::new();
    for op in &ir.ops {
        let p: Vec<f64> = op
            .params
            .iter()
            .map(|e| match e {
                omega_core::circuit::ParamExpr::Concrete(v) => *v,
                other => panic!("symbolic parameter survived lowering: {other:?}"),
            })
            .collect();
        match &op.gate {
            GateKind::PhaseShifter => ops.push(PhotonicOp::PhaseShifter {
                mode: op.qubits[0].0 as usize,
                phi: p[0],
            }),
            GateKind::BeamSplitterRx => ops.push(PhotonicOp::BeamSplitterRx {
                mode0: op.qubits[0].0 as usize,
                mode1: op.qubits[1].0 as usize,
                theta: p[0],
                phi: p[1],
            }),
            other => panic!("unexpected gate in photonic lowering: {other:?}"),
        }
    }
    (ir.num_qubits as usize, ops)
}

fn max_diff(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| (x - y).norm()))
        .fold(0.0f64, f64::max)
}

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Perceval's `HWP(θ)` on one spatial mode's (H, V) pair:
/// `i · [[cos2θ, sin2θ], [sin2θ, −cos2θ]]`.
fn perceval_hwp(theta: f64) -> Vec<Vec<Complex64>> {
    let (c2, s2) = ((2.0 * theta).cos(), (2.0 * theta).sin());
    vec![
        vec![c(0.0, c2), c(0.0, s2)],
        vec![c(0.0, s2), c(0.0, -c2)],
    ]
}

#[test]
fn hwp_matches_perceval_including_the_global_phase() {
    for theta in [0.0, std::f64::consts::FRAC_PI_8, 0.37, std::f64::consts::FRAC_PI_4] {
        let src = format!(
            "OPTICQASM 1.0;\nphoton q[1] pol;\nhwp({theta}) q[0];\n"
        );
        let (modes, u) = unitary_of(&src);
        assert_eq!(modes, 2, "a 1-spatial-mode pol register is 2 optical modes");

        let want = perceval_hwp(theta);
        let d = max_diff(&u, &want);
        assert!(
            d < TOL,
            "hwp({theta}) differs from Perceval by {d:.3e}\n got {u:?}\nwant {want:?}"
        );
    }
}

/// The regression that would have shipped. The i-less textbook matrix must be
/// **rejected**, or this whole file is decoration.
#[test]
fn the_i_less_textbook_hwp_is_rejected() {
    let theta = std::f64::consts::FRAC_PI_8;
    let src = format!("OPTICQASM 1.0;\nphoton q[1] pol;\nhwp({theta}) q[0];\n");
    let (_, u) = unitary_of(&src);

    let (c2, s2) = ((2.0 * theta).cos(), (2.0 * theta).sin());
    let textbook = vec![
        vec![c(c2, 0.0), c(s2, 0.0)],
        vec![c(s2, 0.0), c(-c2, 0.0)],
    ];

    let d = max_diff(&u, &textbook);
    assert!(
        d > 0.5,
        "Aria's hwp now matches the i-LESS textbook matrix (diff {d:.3e}). That \
         is the exact defect FIXES_PLAN.md I0 documents: it disagrees with \
         Perceval by 1.0 and moves 4-mode MZI probabilities by 0.413."
    );
}

/// PBS swaps H between spatial modes and leaves V alone.
///
/// Interleaved ordering `[a_H, a_V, b_H, b_V]`, so the expected matrix is the
/// permutation exchanging indices 0 and 2.
#[test]
fn pbs_swaps_h_and_transmits_v() {
    let src = "OPTICQASM 1.0;\nphoton q[2] pol;\npbs q[0], q[1];\n";
    let (modes, u) = unitary_of(src);
    assert_eq!(modes, 4, "2 spatial modes with polarization = 4 optical modes");

    let mut want = vec![vec![c(0.0, 0.0); 4]; 4];
    want[0][2] = c(1.0, 0.0); // out a_H <- in b_H
    want[2][0] = c(1.0, 0.0); // out b_H <- in a_H
    want[1][1] = c(1.0, 0.0); // a_V transmitted
    want[3][3] = c(1.0, 0.0); // b_V transmitted

    let d = max_diff(&u, &want);
    assert!(d < TOL, "pbs differs from Perceval by {d:.3e}\n got {u:?}");
}

/// A PBS must be its own inverse. Cheap, and it catches a sign error in the
/// `PS(π)·BSrx(π/2,π)` expansion that the permutation test could in principle
/// miss if both were wrong the same way.
#[test]
fn pbs_is_an_involution() {
    let src = "OPTICQASM 1.0;\nphoton q[2] pol;\npbs q[0], q[1];\npbs q[0], q[1];\n";
    let (_, u) = unitary_of(src);
    let mut ident = vec![vec![c(0.0, 0.0); 4]; 4];
    for (i, row) in ident.iter_mut().enumerate() {
        row[i] = c(1.0, 0.0);
    }
    let d = max_diff(&u, &ident);
    assert!(d < TOL, "pbs applied twice is not the identity: {d:.3e}");
}

/// Mode doubling must reach `num_qubits`, because the resource governor prices
/// photonic admission from it. Under-reporting here under-prices by a binomial
/// factor, and photonic jobs are exempt from the qubit ceiling, so pricing is
/// the ONLY guard. See FIXES_PLAN.md I1.
#[test]
fn polarized_declaration_doubles_the_priced_mode_count() {
    let plain_ir = lower_to_ir("OPTICQASM 1.0;\nphoton q[3];\n").expect("lower");
    assert_eq!(plain_ir.num_qubits, 3);

    let pol_ir = lower_to_ir("OPTICQASM 1.0;\nphoton q[3] pol;\n").expect("lower");
    assert_eq!(
        pol_ir.num_qubits, 6,
        "a polarized 3-spatial-mode register must price as 6 optical modes"
    );
}

/// Polarization elements on an unpolarized register must REFUSE.
///
/// Silently treating `q[0]` as spatial would apply the plate to two unrelated
/// optical modes and return a plausible wrong answer — worse than an error.
#[test]
fn polarization_gates_refuse_unpolarized_registers() {
    for src in [
        "OPTICQASM 1.0;\nphoton q[2];\nhwp(0.3) q[0];\n",
        "OPTICQASM 1.0;\nphoton q[2];\npbs q[0], q[1];\n",
    ] {
        let err = lower_to_ir(src).expect_err("must refuse a polarization element without `pol`");
        assert!(
            err.contains("polarized register"),
            "error should name the cause, got: {err}"
        );
    }
}
