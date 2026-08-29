// SPDX-License-Identifier: Apache-2.0
//! **The `tch` backend must honour the deferred-measurement contract too.**
//!
//! This was the LAST engine still computing `⟨O⟩` by deleting the measurements
//! and evaluating the remaining unitary. `evolve` skips `Measure`, so
//! `h q0; measure q0 -> c0; if(c==1) x q1` was answered as a pure Bell state:
//! `⟨X₀X₁⟩ = +1` where the truth for that circuit is **0**.
//!
//! # Why this test lives here and not in the cross-engine lane
//!
//! `crates/omega-cli/tests/deferred_expectation.rs` is where the contract is
//! enforced across engines, and every other backend is in it. `aria-backend-tch`
//! cannot be: it is **excluded from the default workspace** because it links
//! libtorch, and only `aria-runtime --features tch` pulls it in. Adding it to
//! that lane would drag libtorch into `omega-cli`'s build.
//!
//! So the lane's assertions are mirrored here for this one engine. That is a
//! weaker arrangement — a second copy of a rule can drift from the first — and
//! the mitigation is that both copies call the SAME
//! `omega_core::defer_measure::prepare_for_expectation`, so what is duplicated
//! is the check, not the rule.
//!
//! # The fixture is read from disk
//!
//! Not hand-copied. `12_feedforward_sometimes_false.qasm` ends with a second
//! `measure q[1] -> c[0]`, and a hand-written version of it that stopped one line
//! early is precisely what concealed a defect that emitted `CX q1,q1` and
//! panicked the CPU backend.

use aria_backend_tch::TchBackend;
use omega_core::circuit::CircuitIR;
use omega_core::error::OmegaError;
use omega_core::executor::{Backend, Observable};
use omega_core::params::ParameterBinding;

fn fixture(fragment: &str) -> (String, CircuitIR) {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("omega-bridges")
        .join("tests")
        .join("fixtures")
        .join("crosscheck");
    let mut hits: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().map(|x| x == "qasm").unwrap_or(false)
                && p.file_name()
                    .and_then(|f| f.to_str())
                    .map(|f| f.contains(fragment))
                    .unwrap_or(false)
        })
        .collect();
    hits.sort();
    assert_eq!(
        hits.len(),
        1,
        "expected exactly one fixture matching {fragment:?} in {}; a test that \
         silently matches nothing asserts nothing",
        dir.display()
    );
    let src = std::fs::read_to_string(&hits[0]).expect("read fixture");
    let ir = omega_parser::lower_to_ir(&src)
        .unwrap_or_else(|e| panic!("{} does not parse: {e}", hits[0].display()));
    (
        hits[0].file_name().unwrap().to_string_lossy().into_owned(),
        ir,
    )
}

fn obs(s: &str) -> Observable {
    Observable::parse(s).unwrap_or_else(|e| panic!("{s} does not parse: {e}"))
}

#[test]
fn the_feedforward_fixture_is_answered_as_a_mixture() {
    let be = TchBackend::cpu();
    let p = ParameterBinding::new();

    for fragment in ["feedforward_sometimes_false", "feedforward_always_true"] {
        let (name, ir) = fixture(fragment);

        // Both fixtures flip q1 exactly when q0 measured 1, so the two qubits
        // always agree and Z0Z1 is +1. An engine that drops the guard gives 0.
        let zz = be
            .expectation(&ir, &p, &obs("Z0 Z1"))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            (zz - 1.0).abs() < 1e-9,
            "{name}: <Z0Z1> = {zz}, expected +1. A value near 0 means the \
             classically-conditioned gate was dropped rather than deferred."
        );

        // THE discriminator, and exact: dephasing DELETES the term rather than
        // computing a small number, so there is nothing to tolerance.
        let xx = be
            .expectation(&ir, &p, &obs("X0 X1"))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            xx, 0.0,
            "{name}: <X0X1> = {xx}, expected EXACTLY 0. This circuit is a mixture \
             of |00> and |11>, not a Bell state — the measurement destroyed the \
             coherence. +1 here is the old semantics: measurement deleted, \
             unitary evaluated, plausible number, silently wrong."
        );
    }
}

/// **A circuit that cannot be deferred is refused, not answered.**
///
/// `h q0; measure q0 -> c0; h q0` — the measured qubit is used coherently
/// afterwards, so there is no deferred form. `evolve` would skip the `Measure`
/// and compute `h; h` = identity, giving `⟨Z₀⟩ = +1` where the truth is 0: the
/// measurement destroys the coherence the second `H` would otherwise restore.
///
/// **No `reset` in this circuit, deliberately.** The first version of this test
/// used `h q0; measure q0; reset q0; h q0` and PASSED with the deferral wiring
/// removed — because `evolve` already refuses `Reset` on its own. It was
/// therefore testing a pre-existing refusal and would have reported the new code
/// as working no matter what it did. Coherent reuse with no reset leaves the
/// deferral pass as the only thing that can refuse.
#[test]
fn a_coherently_reused_measured_qubit_is_refused_not_answered() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\ncreg c[1];\n\
               h q[0];\nmeasure q[0] -> c[0];\nh q[0];\n";
    let ir = omega_parser::lower_to_ir(src).expect("parses");
    let be = TchBackend::cpu();
    match be.expectation(&ir, &ParameterBinding::new(), &obs("Z0")) {
        Err(OmegaError::Unsupported(_)) => {}
        Err(e) => panic!("refused, but not with Unsupported: {e:?}"),
        Ok(v) => panic!(
            "returned {v} for a circuit whose measured qubit is reused \
             coherently. `evolve` skips Measure, so this evaluates h;h = identity \
             and reports +1 where the truth is 0."
        ),
    }
}

/// A circuit with no measurement must be untouched by all of this.
///
/// The regression guard for the common case: `prepare_for_expectation` returns
/// early when there is no `Measure`, so ordinary expectation values must be
/// exactly what they were before this contract existed.
#[test]
fn an_unmeasured_circuit_is_unaffected() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n";
    let ir = omega_parser::lower_to_ir(src).expect("parses");
    let be = TchBackend::cpu();
    let p = ParameterBinding::new();
    // A genuine Bell state: all three correlators are +1/+1/-1.
    for (o, want) in [("Z0 Z1", 1.0), ("X0 X1", 1.0), ("Y0 Y1", -1.0)] {
        let got = be.expectation(&ir, &p, &obs(o)).expect("runs");
        assert!(
            (got - want).abs() < 1e-6,
            "unmeasured Bell state: <{o}> = {got}, expected {want}"
        );
    }
}
