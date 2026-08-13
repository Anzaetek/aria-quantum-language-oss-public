// SPDX-License-Identifier: Apache-2.0
//! **O6 for the photonic lane: every `GateKind` must be answered for OPTICQASM,
//! at compile time.**
//!
//! `every_emitted_gate_is_readable.rs` does this for QASM2. Without the same
//! mechanism here, a new `GateKind` could be added, wired into the QASM2 lane,
//! and silently do nothing sensible in OPTICQASM — `to_opticqasm` refuses
//! unknown kinds at *runtime*, which only helps if someone happens to construct
//! one.
//!
//! The `match` in `profile()` has no `_` arm. Adding a variant fails to compile
//! until someone states which OPTICQASM profile it belongs to, or that it
//! belongs to none.
//!
//! # Why "profile" and not just "emittable"
//!
//! OPTICQASM names two families with different execution models, and a gate's
//! readability depends on which reader you ask (see
//! `PLAN-OPTICQASM-INTEGRITY.md` §3):
//!
//! | profile | gates | reader |
//! |---|---|---|
//! | DV | `hwp`, `pbs` | `lower_opticqasm` only |
//! | CV | `squeeze`, `displace`, `kerr` | `lower_opticqasm_cv` only |
//! | both | `ps`, `bs_rx` | either |
//!
//! A guard that asked only "does some reader accept it" would pass if `kerr`
//! started lowering on the DV path — which would be wrong, since the
//! discrete-variable IR cannot express a Fock-space operator. So each gate's
//! profile is asserted positively **and** the other profile is asserted to
//! refuse it.

use aria_core::ast::nodes::*;
use aria_core::ast::opticqasm::{from_opticqasm, to_opticqasm};
use omega_parser::{lower_opticqasm_cv, parse_opticqasm};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Profile {
    /// Lowers on the discrete-variable path only.
    Dv,
    /// Imports on the continuous-variable path only.
    Cv,
    /// Both — a phase shifter is a linear-optical element and a phase-space
    /// rotation; a beam splitter is two-mode in both pictures.
    Both,
    /// Not part of OPTICQASM. `to_opticqasm` must REFUSE it, not comment it out.
    NotPhotonic,
}

/// Does this gate need a `pol` register, and what are its arities?
struct Shape {
    profile: Profile,
    params: usize,
    modes: usize,
    polarized: bool,
}

/// **Exhaustive on purpose — do not add a `_` arm.**
fn shape(kind: &GateKind) -> Shape {
    use GateKind::*;
    let s = |profile, params, modes, polarized| Shape {
        profile,
        params,
        modes,
        polarized,
    };
    match kind {
        PhaseShifter => s(Profile::Both, 1, 1, false),
        BeamSplitter => s(Profile::Both, 2, 2, false),
        Squeezing => s(Profile::Cv, 2, 1, false),
        Displacement => s(Profile::Cv, 2, 1, false),
        Kerr => s(Profile::Cv, 1, 1, false),
        // Polarization: DV only, and only on a `pol` register.
        HalfWavePlate => s(Profile::Dv, 1, 1, true),
        PolarizingBeamSplitter => s(Profile::Dv, 0, 2, true),
        // Everything qubit-side. Barrier is the one exception that is emitted
        // (as a comment) rather than refused, because it has no operational
        // meaning to lose — handled separately below.
        I | X | Y | Z | H | S | Sdg | T | Tdg | SX | RX | RY | RZ | P | U | CX | CY | CZ
        | SWAP | RXX | RYY | RZZ | CP | CRz | RBS | CCX | CSWAP | Barrier | Reset | Measure => {
            s(Profile::NotPhotonic, 0, 1, false)
        }
    }
}

/// Every variant, once. The compiler checks `shape()` is total; this list
/// checks nothing is left out of the *run*.
const ALL: &[GateKind] = &[
    GateKind::I, GateKind::X, GateKind::Y, GateKind::Z, GateKind::H,
    GateKind::S, GateKind::Sdg, GateKind::T, GateKind::Tdg, GateKind::SX,
    GateKind::RX, GateKind::RY, GateKind::RZ, GateKind::P, GateKind::U,
    GateKind::CX, GateKind::CY, GateKind::CZ, GateKind::SWAP,
    GateKind::RXX, GateKind::RYY, GateKind::RZZ, GateKind::CP, GateKind::CRz,
    GateKind::RBS, GateKind::CCX, GateKind::CSWAP,
    GateKind::Barrier, GateKind::Reset, GateKind::Measure,
    GateKind::BeamSplitter, GateKind::PhaseShifter,
    GateKind::Squeezing, GateKind::Displacement, GateKind::Kerr,
    GateKind::HalfWavePlate, GateKind::PolarizingBeamSplitter,
];

/// Distinct, generic values — no 0, no repeats — so a swapped or dropped
/// parameter cannot coincide with the right answer.
const PARAMS: [f64; 2] = [0.37, 0.61];

fn emit(kind: &GateKind, sh: &Shape) -> Result<String, String> {
    let mut c = Circuit::new("photonic");
    let modes = if sh.polarized {
        c.qreg_polarized("q", sh.modes)
    } else {
        c.qreg("q", sh.modes)
    };
    c.apply(
        GateDef::with_params(kind.clone(), PARAMS[..sh.params].to_vec()),
        modes.to_vec(),
    );
    to_opticqasm(&c)
}

#[test]
fn every_photonic_gate_emits_and_is_read_by_its_own_profile() {
    let mut broken = Vec::new();
    let mut checked = 0usize;

    for kind in ALL {
        let sh = shape(kind);
        if sh.profile == Profile::NotPhotonic {
            continue;
        }
        let text = match emit(kind, &sh) {
            Ok(t) => t,
            Err(e) => {
                broken.push(format!("{kind:?}: photonic but NOT EMITTED: {e}"));
                continue;
            }
        };
        let program = match parse_opticqasm(&text) {
            Ok(p) => p,
            Err(e) => {
                broken.push(format!("{kind:?}: our own output does not PARSE: {e}\n{text}"));
                continue;
            }
        };
        let dv = omega_parser::lower::lower_opticqasm(&program);
        let cv = lower_opticqasm_cv(&program);

        // Positive: the declared profile reads it.
        let (want_dv, want_cv) = match sh.profile {
            Profile::Dv => (true, false),
            Profile::Cv => (false, true),
            Profile::Both => (true, true),
            Profile::NotPhotonic => unreachable!(),
        };
        if dv.is_ok() != want_dv {
            broken.push(format!(
                "{kind:?}: declared {:?}, but the DV lowering {} it: {:?}\n{text}",
                sh.profile,
                if dv.is_ok() { "ACCEPTS" } else { "REFUSES" },
                dv.as_ref().err()
            ));
        }
        // Negative: the other profile refuses. This is the half that stops a
        // gate quietly becoming readable on a path that cannot execute it.
        if cv.is_ok() != want_cv {
            broken.push(format!(
                "{kind:?}: declared {:?}, but the CV import {} it: {:?}\n{text}",
                sh.profile,
                if cv.is_ok() { "ACCEPTS" } else { "REFUSES" },
                cv.as_ref().err()
            ));
        }

        // And the aria-core reader recovers the spelling — the only layer that
        // can, since the DV lowering expands hwp/pbs away.
        match from_opticqasm(&text) {
            Ok(back) => {
                let got: Vec<_> = back.instructions.iter().map(|i| i.gate.kind).collect();
                if got != vec![*kind] {
                    broken.push(format!(
                        "{kind:?}: re-imported as {got:?} — the spelling did not survive\n{text}"
                    ));
                }
            }
            Err(e) => broken.push(format!("{kind:?}: our own output does not re-import: {e}\n{text}")),
        }
        checked += 1;
    }

    assert!(broken.is_empty(), "photonic lane defects:\n{}", broken.join("\n"));
    assert_eq!(
        checked, 7,
        "expected 7 photonic gates (ps, bs_rx, squeeze, displace, kerr, hwp, pbs); \
         got {checked}. If a gate was added, give it a profile; if one was removed, \
         update this count deliberately rather than letting coverage shrink."
    );
}

/// A qubit gate must be REFUSED by `to_opticqasm`, not written as a comment.
///
/// `// unsupported: H q[0];` re-parsed as a circuit with zero operations — an
/// export that succeeded and silently became a different computation. Checked
/// across every non-photonic variant rather than one sample, because the
/// fallback arm is shared and a sample cannot tell which variants reach it.
#[test]
fn no_qubit_gate_is_silently_commented_into_opticqasm() {
    let mut leaked = Vec::new();
    for kind in ALL {
        let sh = shape(kind);
        if sh.profile != Profile::NotPhotonic {
            continue;
        }
        // Barrier is the documented exception: no operational meaning to lose.
        if *kind == GateKind::Barrier {
            continue;
        }
        let mut c = Circuit::new("photonic");
        let q = c.qreg("q", 3);
        // Measure/Reset need a classical bit and one qubit respectively; the
        // emitter refuses before arity matters, which is the point.
        let _ = c.creg("m", 1);
        c.apply(GateDef::new(kind.clone()), vec![q[0].clone()]);
        match to_opticqasm(&c) {
            Err(_) => {}
            Ok(text) => leaked.push(format!("{kind:?}: emitted instead of refused:\n{text}")),
        }
    }
    assert!(
        leaked.is_empty(),
        "OPTICQASM is a photonic dialect; these were not refused:\n{}",
        leaked.join("\n")
    );
}

/// `Barrier` is the one non-photonic kind that is emitted, as a comment.
///
/// Stated as its own test so the exception is deliberate and visible, rather
/// than an unexplained `continue` in the loop above.
#[test]
fn barrier_is_the_only_comment_the_emitter_writes() {
    let mut c = Circuit::new("photonic");
    let q = c.qreg("q", 1);
    c.apply(GateDef::new(GateKind::Barrier), vec![q[0].clone()]);
    let text = to_opticqasm(&c).expect("a barrier is emittable");
    assert!(text.contains("// barrier"), "{text}");
    // It must still re-import — a comment that breaks the parse would be the
    // Aria emitter's `--` defect in a different dialect.
    let back = from_opticqasm(&text).expect("a commented barrier must not break re-import");
    assert!(
        back.instructions.is_empty(),
        "a barrier carries no operation, so re-import should yield none"
    );
}
