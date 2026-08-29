// SPDX-License-Identifier: Apache-2.0
//! An observable naming a qubit the circuit does not have must be REFUSED.
//!
//! # What this was before
//!
//! Nothing validated the qubit index, and `PauliKey`'s setters do no bounds
//! check. On a 4-qubit circuit (`w = 1`, `words.len() = 2`):
//!
//! * `Z99` → `set_z` indexes `words[w + 99/64]` = `words[2]` → **panic**,
//!   "index out of bounds: the len is 2 but the index is 2". For the HTTP
//!   server that panic lands inside a request handler, and there is no
//!   `CatchPanicLayer`.
//! * `X99` → `set_x` indexes `words[99/64]` = `words[1]`, which is the **Z
//!   half** of a 4-qubit key. No panic: it silently corrupts the key and
//!   returns a WRONG NUMBER. That is the worse of the two.
//!
//! Nothing upstream catches it either: `Observable::parse` reads a bare `u32`
//! and never sees the circuit it will be measured against.
//!
//! # Why refusing is the right answer
//!
//! There is no sensible interpretation of "measure qubit 99 of a 4-qubit
//! circuit". Padding the register would invent qubits the caller did not ask
//! for; ignoring the term would silently change the observable. Both are worse
//! than saying no.

use omega_backend_pauliprop::PauliPropBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

fn bell(nq: u32) -> CircuitIR {
    let mut c = CircuitIR::new(nq, CircuitType::GateBased);
    for (g, qs) in [
        (GateKind::H, vec![Qubit(0)]),
        (GateKind::CX, vec![Qubit(0), Qubit(1)]),
    ] {
        c.ops.push(GateOp {
            gate: g,
            qubits: qs.into(),
            params: Default::default(),
            classical_bit: None,
            condition: None,
        });
    }
    c
}

fn run(nq: u32, term: Vec<(u32, PauliOp)>) -> omega_core::error::Result<f64> {
    PauliPropBackend::new().expectation(
        &bell(nq),
        &ParameterBinding::new(),
        &Observable {
            terms: vec![(1.0, term)],
        },
    )
}

/// The panic case: `Z` past the register indexed off the end of the word array.
#[test]
fn a_z_term_past_the_register_is_refused_not_a_panic() {
    let e =
        run(4, vec![(99, PauliOp::Z)]).expect_err("qubit 99 does not exist in a 4-qubit circuit");
    let m = e.to_string();
    assert!(
        m.contains("99") && m.contains('4'),
        "the error should name the offending qubit and the register size, got: {m}"
    );
}

/// The silent-corruption case, which is the more dangerous of the two: `X` past
/// the register wrote into the Z half and returned a plausible wrong number
/// rather than failing.
#[test]
fn an_x_term_past_the_register_is_refused_not_silently_wrong() {
    let e =
        run(4, vec![(99, PauliOp::X)]).expect_err("qubit 99 does not exist in a 4-qubit circuit");
    assert!(e.to_string().contains("99"), "{e}");
}

/// `Y` sets both halves, so it can hit either failure depending on the index.
#[test]
fn a_y_term_past_the_register_is_refused() {
    assert!(run(4, vec![(64, PauliOp::Y)]).is_err());
    assert!(
        run(4, vec![(4, PauliOp::Y)]).is_err(),
        "off-by-one: qubit 4 does not exist in a 4-qubit circuit"
    );
}

/// The boundary must be exact — the last valid qubit still works, and the first
/// invalid one does not. An over-eager check would break every legitimate
/// observable on the top qubit.
#[test]
fn the_boundary_is_exact() {
    assert!(
        run(4, vec![(3, PauliOp::Z)]).is_ok(),
        "qubit 3 IS valid in a 4-qubit circuit and must not be refused"
    );
    assert!(run(4, vec![(4, PauliOp::Z)]).is_err());
}

/// A multi-qubit term where only ONE factor is out of range must still be
/// refused — the check has to cover every factor, not just the first.
#[test]
fn one_bad_factor_among_good_ones_is_still_refused() {
    assert!(run(4, vec![(0, PauliOp::Z), (99, PauliOp::Z)]).is_err());
    assert!(run(4, vec![(99, PauliOp::Z), (0, PauliOp::Z)]).is_err());
}

/// Ordinary observables must be unaffected.
#[test]
fn valid_observables_still_work() {
    let v = run(2, vec![(0, PauliOp::Z), (1, PauliOp::Z)]).expect("valid");
    // Bell state: <Z0 Z1> = 1 exactly, and pauliprop is exact on Clifford.
    assert!(
        (v - 1.0).abs() < 1e-12,
        "expected 1.0 for a Bell pair, got {v}"
    );
}
