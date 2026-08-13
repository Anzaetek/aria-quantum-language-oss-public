// SPDX-License-Identifier: Apache-2.0
//! **What `aria-core` writes as OPTICQASM, `omega-parser` must read.**
//!
//! This is the cross-crate gate for the defect class in
//! `PLAN-OPTICQASM-INTEGRITY.md`, and it lives here rather than in either crate's
//! unit tests because the defect is *between* them: each side was internally
//! consistent and self-testing, and the gap only existed at the seam.
//!
//! Measured before the fix, by driving `omega_parser::lower_to_ir` on the
//! emitter's own output:
//!
//! ```text
//!   ps + bs_rx  (DV core): READS OK
//!   hwp         (Part I):  READS OK
//!   pbs         (Part I):  READS OK
//!   squeeze     (CV):      REJECTED — unknown photonic gate: squeeze
//!   kerr        (CV):      REJECTED — unknown photonic gate: kerr
//!   displace    (CV):      REJECTED — unknown photonic gate: displace
//! ```
//!
//! So a file labelled `OPTICQASM 1.0;`, produced by this workspace, could not be
//! loaded by this workspace. Unlike the `cp` case there was not even a second
//! consumer that accepted it — `omega-backend-cv` has no text front end at all —
//! so the file was readable by nothing.
//!
//! Note what "unknown photonic gate" claimed. `squeeze` is implemented by
//! piquasso (`Squeezing`) and by this workspace's own `omega-backend-cv`. The
//! gate was never unknown; the discrete-variable IR simply cannot express a
//! Fock-space operator. The reader had an executor's limitation written into it
//! as a statement about the language.

use aria_core::ast::nodes::*;
use aria_core::ast::opticqasm::{from_opticqasm, to_opticqasm};
use omega_parser::{lower_opticqasm_cv, parse_opticqasm, CvOp};

/// Emit a one-gate circuit and return the OPTICQASM text.
fn emit(kind: GateKind, params: Vec<f64>, n_modes: usize, on: &[usize]) -> String {
    let mut c = Circuit::new("photonic");
    let m = c.qreg("q", n_modes);
    let qs: Vec<Qubit> = on.iter().map(|&i| m[i].clone()).collect();
    c.apply(GateDef::with_params(kind, params), qs);
    to_opticqasm(&c).expect("photonic gate must be emittable")
}

/// The discrete-variable profile round-trips through `lower_to_ir`.
#[test]
fn dv_profile_emitted_here_lowers_in_omega_parser() {
    for (kind, params, on) in [
        (GateKind::PhaseShifter, vec![0.5], vec![0usize]),
        (GateKind::BeamSplitter, vec![1.2, 0.3], vec![0, 1]),
    ] {
        let text = emit(kind, params, 2, &on);
        let program = parse_opticqasm(&text)
            .unwrap_or_else(|e| panic!("our own OPTICQASM does not PARSE: {e}\n{text}"));
        let ir = omega_parser::lower::lower_opticqasm(&program)
            .unwrap_or_else(|e| panic!("our own OPTICQASM does not LOWER: {e}\n{text}"));
        assert_eq!(ir.ops.len(), 1, "gate lost between emit and lower:\n{text}");
    }
}

/// **The defect.** These three are what `to_opticqasm` writes for the CV gates,
/// and until the CV profile existed nothing in the workspace could read them.
///
/// Asserts on the imported operations, not on "it did not error" — a reader
/// that returned an empty program would satisfy the weaker claim, which is
/// exactly how D7 survived (`Ok`, 0 registers, 0 ops).
#[test]
fn cv_profile_emitted_here_imports_in_omega_parser() {
    let cases: Vec<(GateKind, Vec<f64>, CvOp)> = vec![
        (
            GateKind::Squeezing,
            vec![0.4, 0.2],
            CvOp::Squeeze { mode: 0, r: 0.4, phi: 0.2 },
        ),
        (
            GateKind::Displacement,
            vec![0.7, -0.1],
            CvOp::Displace { mode: 0, re: 0.7, im: -0.1 },
        ),
        (GateKind::Kerr, vec![0.15], CvOp::Kerr { mode: 0, chi: 0.15 }),
    ];

    for (kind, params, want) in cases {
        let text = emit(kind, params, 1, &[0]);
        let program = parse_opticqasm(&text)
            .unwrap_or_else(|e| panic!("our own OPTICQASM does not PARSE: {e}\n{text}"));
        let cv = lower_opticqasm_cv(&program).unwrap_or_else(|e| {
            panic!("our own CV export does not IMPORT: {e}\n{text}")
        });
        assert_eq!(
            cv.ops,
            vec![want],
            "the imported operation differs from what was emitted:\n{text}"
        );
    }
}

/// The DV lowering must still refuse CV gates — but with a route, not the false
/// claim that the gate is unknown.
#[test]
fn dv_lowering_refuses_cv_by_naming_the_right_profile() {
    let text = emit(GateKind::Kerr, vec![0.15], 1, &[0]);
    let program = parse_opticqasm(&text).unwrap();
    let err = omega_parser::lower::lower_opticqasm(&program)
        .expect_err("a Fock-space operator is not expressible in the qubit IR");
    assert!(
        err.contains("continuous-variable"),
        "the refusal must classify the gate, not deny it exists: {err}"
    );
    assert!(
        err.contains("lower_opticqasm_cv"),
        "the refusal must name the import that DOES work: {err}"
    );
    assert!(
        !err.contains("unknown photonic gate"),
        "`kerr` is implemented by piquasso and by omega-backend-cv; calling it \
         unknown is what made this export unreadable: {err}"
    );
}

/// A mixed-profile circuit is emittable, and each profile reads the part it
/// owns. This is the fixture the plan's §5 demands: built from gates on **both**
/// sides of the split, so it is not invariant under the defect.
#[test]
fn a_mixed_circuit_is_readable_by_the_profile_that_owns_each_gate() {
    let mut c = Circuit::new("photonic");
    let m = c.qreg("q", 2);
    c.apply(
        GateDef::with_params(GateKind::PhaseShifter, vec![0.5]),
        vec![m[0].clone()],
    );
    c.apply(
        GateDef::with_params(GateKind::Kerr, vec![0.15]),
        vec![m[1].clone()],
    );
    let text = to_opticqasm(&c).expect("both gates are photonic");
    let program = parse_opticqasm(&text).expect("parses");

    // The CV import takes both: `ps` is a phase-space rotation as well as a
    // linear-optical element, so it belongs to both profiles.
    let cv = lower_opticqasm_cv(&program).expect("CV profile reads ps and kerr");
    assert_eq!(
        cv.ops,
        vec![
            CvOp::PhaseShift { mode: 0, phi: 0.5 },
            CvOp::Kerr { mode: 1, chi: 0.15 },
        ]
    );

    // The DV lowering refuses, because `kerr` has no qubit-IR meaning — and it
    // must refuse the WHOLE program rather than lower the `ps` and drop the
    // `kerr`, which would be the silent-drop defect one more time.
    assert!(omega_parser::lower::lower_opticqasm(&program).is_err());
}

/// `bs` is an accepted alias in `aria-core::from_opticqasm` and is named by the
/// grammar's `gate_name` rule, so it parsed and then died at lowering — a
/// spelling two of the three tables knew.
#[test]
fn the_bs_alias_lowers_on_both_profiles() {
    let src = "OPTICQASM 1.0;\nphoton q[2];\nbs(1.2, 0.3) q[0], q[1];\n";
    let program = parse_opticqasm(src).expect("the grammar names `bs`");
    let ir = omega_parser::lower::lower_opticqasm(&program).expect("`bs` must lower like `bs_rx`");
    assert_eq!(ir.ops.len(), 1);
    let cv = lower_opticqasm_cv(&program).expect("`bs` is two-mode CV too");
    assert_eq!(
        cv.ops,
        vec![CvOp::BeamSplitter { a: 0, b: 1, theta: 1.2, phi: 0.3 }]
    );
}

/// **O4: polarization round-trips, and the `pol` marker travels with it.**
///
/// The mode-count assertion is the point, not decoration. D3 in
/// `PLAN-OPTICQASM-INTEGRITY.md` is that `photon q[N] pol;` means N SPATIAL
/// modes carrying H and V — `2N` optical modes indexed `2s+p` — while
/// `photon q[N];` means N optical modes. A test that emitted `hwp` and asserted
/// the text contains `"hwp"` would pass with the marker dropped, and the file
/// would parse, refuse nothing, and mean something different in every index.
///
/// So this asserts on the LOWERED mode count: 2 spatial modes must become 4
/// optical ones. That number is what changes if the marker goes missing.
#[test]
fn polarization_survives_emit_and_reimport_with_its_mode_semantics() {
    let mut c = Circuit::new("photonic");
    let m = c.qreg_polarized("q", 2);
    c.apply(
        GateDef::with_params(GateKind::HalfWavePlate, vec![0.4]),
        vec![m[0].clone()],
    );
    c.apply(
        GateDef::new(GateKind::PolarizingBeamSplitter),
        vec![m[0].clone(), m[1].clone()],
    );
    let text = to_opticqasm(&c).expect("O4: polarization is emittable");
    assert!(
        text.contains("photon q[2] pol;"),
        "the `pol` marker must be emitted with the gates:\n{text}"
    );

    // The DV lowering is what knows the H/V mapping, so it is the honest place
    // to check the semantics survived.
    let program = parse_opticqasm(&text)
        .unwrap_or_else(|e| panic!("our own polarization output does not parse: {e}\n{text}"));
    let ir = omega_parser::lower::lower_opticqasm(&program)
        .unwrap_or_else(|e| panic!("our own polarization output does not lower: {e}\n{text}"));
    assert_eq!(
        ir.num_qubits, 4,
        "2 spatial modes must occupy 4 optical modes — if this says 2, the `pol` \
         marker was dropped and every mode index now means something else:\n{text}"
    );
    // `hwp` and `pbs` are EXPANDED by the lowering (into phase shifters and
    // beam splitters), so the IR carries neither spelling. That is why the
    // aria-core AST needed the variants at all: it is the only layer that can
    // re-emit `hwp` as `hwp` instead of as its four ops.
    assert!(
        ir.ops.len() > 2,
        "hwp/pbs should expand into several primitive ops, got {}:\n{text}",
        ir.ops.len()
    );

    // And the aria-core reader recovers the SPELLING, which the IR cannot.
    let back = from_opticqasm(&text).expect("re-import");
    let kinds: Vec<_> = back.instructions.iter().map(|i| i.gate.kind).collect();
    assert_eq!(
        kinds,
        vec![GateKind::HalfWavePlate, GateKind::PolarizingBeamSplitter],
        "the spellings must survive the AST round trip:\n{text}"
    );
    assert!(back.registers[0].polarized, "the pol flag was lost on re-import");
}

/// `pbs` is emitted with no parameter list at all, which the grammar allows
/// only for it. `pbs() q[0], q[1];` would be a different, wrong shape.
#[test]
fn pbs_is_emitted_without_a_parameter_list() {
    let mut c = Circuit::new("photonic");
    let m = c.qreg_polarized("q", 2);
    c.apply(
        GateDef::new(GateKind::PolarizingBeamSplitter),
        vec![m[0].clone(), m[1].clone()],
    );
    let text = to_opticqasm(&c).unwrap();
    assert!(text.contains("pbs q[0], q[1];"), "{text}");
    assert!(!text.contains("pbs("), "pbs takes no parameters:\n{text}");
}
