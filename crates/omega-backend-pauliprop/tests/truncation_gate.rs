// SPDX-License-Identifier: Apache-2.0
//! The truncation certificate GATES the result.
//!
//! `dropped_mass` was computed, plumbed, documented as "the CERTIFIED error
//! bound", and printed by the CLI — and consulted by nothing. That is verbatim
//! the defect `omega-backend-mps` records fixing at `sim.rs:528`: *"The
//! truncation certificate gates the RESULT, in every mode. It was computed and
//! printed but consulted by nothing, so a run that discarded 6.5x the state
//! returned a distribution half of which was wrong — with the evidence on
//! screen."* Same sentence, one crate over, never applied here.
//!
//! Measured before this gate, 20 qubits, depth-16 hardware-efficient ansatz,
//! `⟨Z₀⟩`, exact reference `−0.0145160327`:
//!
//! | `--truncate` | returns | `dropped_mass` |
//! |---|---|---|
//! | 1e-3 | −0.0042647207 | 1.41e+03 |
//! | 1e-2 | **0.0000000000** | 1.44e+02 |
//! | 1e-1 | **0.0000000000** | 1.35e+01 |
//!
//! Every bound honoured, every bound useless, and at two cutoffs every term had
//! been truncated away and the engine returned a confident zero.

use omega_backend_pauliprop::PauliPropBackend;
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

/// Deep enough that an aggressive cutoff drives the budget past 1.
fn deep(nq: u32, layers: usize) -> CircuitIR {
    let mut c = CircuitIR::new(nq, CircuitType::GateBased);
    for l in 0..layers {
        for q in 0..nq {
            c.ops.push(op(GateKind::Ry, &[q], &[0.6 + 0.02 * l as f64]));
            c.ops.push(op(GateKind::Rz, &[q], &[0.9 + 0.03 * q as f64]));
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

/// An exact run is never gated: nothing was discarded, so the budget is 0.
#[test]
fn an_exact_run_passes_the_gate_and_certifies_zero() {
    let c = deep(8, 3);
    let (v, cert) = PauliPropBackend::new()
        .expectation_with_certificate(&c, &ParameterBinding::new(), &z0())
        .expect("an exact run must never be gated");
    assert!(cert.is_exact(), "dropped_mass {}", cert.dropped_mass);
    assert!(cert.is_informative());
    assert_eq!(cert.observable_range, 1.0, "|<Z0>| <= 1");
    assert!(v.is_finite());
    assert!(cert.final_terms > 0);
    assert!(
        cert.peak_terms >= cert.final_terms,
        "the peak is a high-water mark, so it cannot be below the final count: \
         peak {} final {}",
        cert.peak_terms,
        cert.final_terms
    );
}

/// THE GATE. A budget that has stopped excluding anything is refused.
#[test]
fn a_run_whose_budget_excludes_nothing_is_refused() {
    let c = deep(12, 12);
    let params = ParameterBinding::new();

    // Establish the premise rather than assuming it: with the gate lifted, this
    // configuration really does produce a vacuous budget. Without this the test
    // could pass because the run failed for some unrelated reason.
    let (value, cert) = PauliPropBackend::with_truncation(1e-1, None)
        .with_max_dropped_mass(Some(f64::INFINITY))
        .expectation_with_certificate(&c, &params, &z0())
        .expect("with the ceiling lifted this must complete");
    assert!(
        !cert.is_informative(),
        "premise: this configuration must produce a VACUOUS budget, else the \
         gate below is not being tested. dropped_mass {:.4e} range {:.4e}",
        cert.dropped_mass,
        cert.observable_range
    );
    assert!(
        value.abs() <= cert.observable_range + 1e-9,
        "sanity: the value itself is in range even though the bound is not"
    );

    // Same run, default ceiling: refused.
    let err = PauliPropBackend::with_truncation(1e-1, None)
        .expectation_with_certificate(&c, &params, &z0())
        .expect_err("a budget wider than the observable's range must be refused");
    let msg = format!("{err}");
    for needle in [
        "excludes nothing",
        "--truncate",
        "--max-terms",
        "--max-dropped-mass",
    ] {
        assert!(
            msg.contains(needle),
            "the refusal must name '{needle}' so the caller knows what to do; got: {msg}"
        );
    }
}

/// Every door is gated. `Backend::expectation` and `expectation_with_budget`
/// both route through the certificate path, so the gate cannot be dodged by
/// picking a different entry point — the mistake `MpsBackend::expectation_multi`
/// made with its Reset check, where one door refused and the other did not.
#[test]
fn the_gate_cannot_be_bypassed_by_choosing_another_door() {
    let c = deep(12, 12);
    let params = ParameterBinding::new();
    let be = || PauliPropBackend::with_truncation(1e-1, None);

    assert!(
        be().expectation(&c, &params, &z0()).is_err(),
        "Backend::expectation must be gated"
    );
    assert!(
        be().expectation_with_budget(&c, &params, &z0()).is_err(),
        "expectation_with_budget must be gated"
    );
    assert!(
        be().expectation_with_certificate(&c, &params, &z0())
            .is_err(),
        "expectation_with_certificate must be gated"
    );
}

/// The escape hatch works, and is the sweep case the gate must not break.
#[test]
fn an_explicit_ceiling_lets_a_deliberate_sweep_keep_its_loose_rows() {
    let c = deep(12, 12);
    let params = ParameterBinding::new();
    let mut informative_seen = false;
    let mut vacuous_seen = false;
    for cutoff in [1e-1, 1e-2, 1e-3] {
        let (_v, cert) = PauliPropBackend::with_truncation(cutoff, None)
            .with_max_dropped_mass(Some(f64::INFINITY))
            .expectation_with_certificate(&c, &params, &z0())
            .expect("an explicit infinite ceiling must never refuse");
        if cert.is_informative() {
            informative_seen = true;
        } else {
            vacuous_seen = true;
        }
    }
    assert!(
        vacuous_seen,
        "the sweep must actually cross the vacuous region, or it is not \
         exercising the escape hatch"
    );
    let _ = informative_seen;
}

/// The ceiling scales with the observable, not with a magic constant.
///
/// `2*Z0` can range over [-2, 2], so a budget of 1.5 excludes something there
/// and nothing for a plain `Z0`. A gate hardcoded to 1.0 would get this wrong
/// in the direction that matters — refusing a run that was still informative.
#[test]
fn the_ceiling_follows_the_observables_own_range() {
    let plain = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };
    let scaled = Observable {
        terms: vec![(2.0, vec![(0, PauliOp::Z)])],
    };
    let summed = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)]), (-1.5, vec![(1, PauliOp::Z)])],
    };
    let c = deep(6, 2);
    let params = ParameterBinding::new();
    let be = PauliPropBackend::new();
    for (obs, want) in [(plain, 1.0), (scaled, 2.0), (summed, 2.5)] {
        let (_v, cert) = be
            .expectation_with_certificate(&c, &params, &obs)
            .expect("exact run");
        assert_eq!(
            cert.observable_range, want,
            "the range is the L1 coefficient norm: |<O>| <= sum|c_i| since \
             |<P>| <= 1 for each Pauli string"
        );
    }
}
