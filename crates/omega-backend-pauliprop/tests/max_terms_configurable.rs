// SPDX-License-Identifier: Apache-2.0
//! The Pauli-term ceiling is CONFIGURABLE, and raising it does not change the
//! answer.
//!
//! A bug report measured being refused at 2 097 152 terms — roughly 80 MB — on
//! a 314 GB host, and read the ceiling as hardcoded. It never was:
//! `PauliPropBackend::max_terms` has always been an `Option<usize>` defaulting
//! to [`DEFAULT_MAX_TERMS`]. What did not exist was a way to say so from
//! outside the library, so from the user's side the distinction was invisible.
//!
//! The property that makes this knob different from the other three is asserted
//! here rather than described: `--truncate`, `--max-weight` and `--max-freq`
//! buy completion by DISCARDING terms and move the value, paying for it in
//! `dropped_mass`. Raising the ceiling permits a larger EXACT sum. So a run
//! that completes only because the ceiling was raised must agree, to floating
//! point, with one that never needed raising — and must report no dropped mass.

use omega_backend_pauliprop::{PauliPropBackend, DEFAULT_MAX_TERMS};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

fn op(gate: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
        params: params.iter().map(|p| ParamExpr::Concrete(*p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// Rotations on every qubit each layer, so the term count grows through
/// `branch` rather than staying Clifford-flat.
fn ansatz(nq: u32, layers: usize) -> CircuitIR {
    let mut c = CircuitIR::new(nq, CircuitType::GateBased);
    for l in 0..layers {
        for q in 0..nq {
            c.ops.push(op(GateKind::Ry, &[q], &[0.4 + 0.03 * l as f64]));
            c.ops.push(op(GateKind::Rz, &[q], &[0.7 + 0.05 * q as f64]));
        }
        for q in 0..nq.saturating_sub(1) {
            c.ops.push(op(GateKind::CX, &[q, q + 1], &[]));
        }
    }
    c
}

fn z0() -> Observable {
    Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    }
}

/// Both directions in one test, deliberately.
///
/// Lowering alone would pass if `with_max_terms` were wired to a ceiling that
/// only ever tightens; raising alone would pass if the flag did nothing and the
/// run simply fitted. Together they pin that the number reaching the engine is
/// the number the caller set.
#[test]
fn the_term_ceiling_moves_in_both_directions() {
    let circuit = ansatz(10, 5);
    let params = ParameterBinding::new();
    let obs = z0();

    // Fits under the default.
    let baseline = PauliPropBackend::new()
        .expectation(&circuit, &params, &obs)
        .expect("this circuit must fit under the default ceiling");

    // Lowered below what it needs: refused, and the message names the ceiling
    // that was ACTUALLY in force, not the default.
    let err = PauliPropBackend::new()
        .with_max_terms(Some(64))
        .expectation(&circuit, &params, &obs)
        .expect_err("a ceiling of 64 must refuse this circuit");
    let msg = format!("{err}");
    assert!(
        msg.contains("64 ceiling"),
        "the refusal must name the ceiling in force, not {DEFAULT_MAX_TERMS}; got: {msg}"
    );
    assert!(
        msg.contains("--max-terms"),
        "and must name the flag that raises it, since that is the one option \
         which does not change the answer; got: {msg}"
    );

    // Raised again: completes, and lands on the SAME value. This is the claim
    // that separates this knob from the truncating three.
    let raised = PauliPropBackend::new()
        .with_max_terms(Some(1 << 20))
        .expectation(&circuit, &params, &obs)
        .expect("a raised ceiling must let the exact run complete");
    // Floating-point equality, NOT `to_bits()`.
    //
    // The first version of this asserted bit identity and was flaky: it failed
    // about two runs in three with `0.5694751295245323` against
    // `...324` — one ULP. The cause is not the ceiling. `PauliSum::terms` is a
    // `HashMap`, Rust seeds its hasher randomly PER PROCESS, and the readout
    // sums coefficients in iteration order, so the order of a
    // floating-point sum — and hence its last bit — varies from run to run.
    //
    // So bit-reproducibility is not a property this engine has, and asserting
    // it does not make it one. Worth knowing on its own: `omega-backend-mps`
    // explicitly pins `thread_count_does_not_change_the_bits`, and pauliprop
    // cannot make that promise while the sum iterates a randomly-seeded map.
    // Recorded rather than papered over — see PLAN-OPEN §3b.2.
    assert!(
        (raised - baseline).abs() < 1e-12,
        "raising the ceiling permits a larger EXACT sum, so it must not move \
         the value: {raised} vs {baseline}"
    );
}

/// `None` removes the ceiling rather than setting it to zero.
///
/// `Option<usize>` invites exactly one off-by-one reading — that `None` means
/// "no terms allowed" — and it would show up as a refusal on the first branch,
/// which reads like an unrelated engine bug.
#[test]
fn a_none_ceiling_means_unbounded_not_zero() {
    let circuit = ansatz(8, 4);
    let params = ParameterBinding::new();
    let obs = z0();
    let baseline = PauliPropBackend::new()
        .expectation(&circuit, &params, &obs)
        .expect("fits under the default");
    let unbounded = PauliPropBackend::new()
        .with_max_terms(None)
        .expectation(&circuit, &params, &obs)
        .expect("None must mean unbounded, not a ceiling of zero");
    // Tolerance, not bits — see the note in
    // `the_term_ceiling_moves_in_both_directions`.
    assert!(
        (unbounded - baseline).abs() < 1e-12,
        "{unbounded} vs {baseline}"
    );
}

/// Raising the ceiling must not be mistaken for truncation: it produces no
/// dropped mass, because it discards nothing.
#[test]
fn raising_the_ceiling_reports_no_dropped_mass() {
    let circuit = ansatz(10, 5);
    let params = ParameterBinding::new();
    let obs = z0();
    let (_v, dropped) = PauliPropBackend::new()
        .with_max_terms(Some(1 << 20))
        .expectation_with_budget(&circuit, &params, &obs)
        .expect("must complete");
    assert_eq!(
        dropped, 0.0,
        "the ceiling knob permits a larger exact sum; it must not report an \
         error budget, or a caller cannot tell it apart from `--truncate`"
    );
}
