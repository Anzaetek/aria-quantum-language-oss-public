// SPDX-License-Identifier: Apache-2.0
//! **`dropped_mass` must actually bound the error it claims to bound.**
//!
//! Pauli propagation truncates the propagated observable `Σ cₚ·P`, and reports
//! the L1 mass of the coefficients it threw away. Because `|⟨P⟩| ≤ 1` for every
//! Pauli string, the error in `⟨O⟩` is at most that mass:
//!
//! ```text
//!   |⟨O⟩_exact − ⟨O⟩_truncated|  ≤  Σ_dropped |cₚ|  =  dropped_mass
//! ```
//!
//! Unlike the MPS fidelity estimate — which is a proxy, because that MPS is not
//! in canonical gauge, and is printed with a `~` for exactly that reason — this
//! one is a genuine bound with no gauge caveat. That claim is worth a test
//! rather than a comment: it is the number a user sizes their truncation
//! against, and a bound that does not hold is worse than no bound.
//!
//! # What could make this pass for the wrong reason
//!
//! * **A corpus where truncation drops nothing.** `dropped_mass = 0` satisfies
//!   the inequality trivially, and it is the default outcome on shallow or
//!   near-Clifford circuits — measured: `--truncate 1e-2` and `--max-weight 4`
//!   on a 10-qubit depth-3 circuit both dropped exactly zero. The corpus below
//!   therefore counts how many cases had a NON-ZERO budget and fails if too few
//!   did.
//! * **A vacuous bound.** `⟨O⟩ ∈ [−1, 1]`, so any `dropped_mass ≥ 2` is
//!   satisfied by every possible answer. Those are counted separately and are
//!   not allowed to be the only evidence.

use omega_backend_pauliprop::PauliPropBackend;
use omega_core::executor::{Backend, Observable};
use omega_core::params::ParameterBinding;

/// Non-Clifford, entangling, and deep enough that a weight cap bites.
fn circuit(n: usize, depth: usize) -> omega_core::circuit::CircuitIR {
    let mut src = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\n");
    for d in 0..depth {
        for i in 0..n {
            src.push_str(&format!("ry({}) q[{i}];\n", 0.3 + 0.17 * i as f64 + 0.11 * d as f64));
        }
        for i in 0..n - 1 {
            src.push_str(&format!("cx q[{i}], q[{}];\n", i + 1));
        }
    }
    omega_parser::lower_to_ir(&src).expect("lower")
}

#[test]
fn the_dropped_mass_bounds_the_actual_error() {
    let obs = Observable::parse("Z0").expect("observable");
    let params = ParameterBinding::new();

    let (mut compared, mut informative, mut violations) = (0usize, 0usize, Vec::new());
    let mut worst_slack = f64::INFINITY;

    for n in [8usize, 10] {
        for depth in [2usize, 3, 4] {
            let ir = circuit(n, depth);
            let exact = PauliPropBackend::new()
                .expectation(&ir, &params, &obs)
                .expect("exact engine");

            for (label, backend) in [
                ("truncate 0.05", PauliPropBackend::with_truncation_freq(0.05, None, None)),
                ("truncate 0.1", PauliPropBackend::with_truncation_freq(0.1, None, None)),
                ("max_weight 1", PauliPropBackend::with_truncation_freq(0.0, Some(1), None)),
                ("max_weight 2", PauliPropBackend::with_truncation_freq(0.0, Some(2), None)),
                ("max_freq 1", PauliPropBackend::with_truncation_freq(0.0, None, Some(1))),
            ] {
                let (val, budget) = backend
                    .expectation_with_budget(&ir, &params, &obs)
                    .expect("truncated engine");
                let err = (exact - val).abs();
                compared += 1;
                // Only a budget that is both non-zero and non-vacuous is
                // evidence: `⟨O⟩ ∈ [−1, 1]`, so `budget ≥ 2` holds for free.
                if budget > 1e-12 && budget < 2.0 {
                    informative += 1;
                    worst_slack = worst_slack.min(budget - err);
                }
                if err > budget + 1e-9 {
                    violations.push(format!(
                        "n={n} depth={depth} {label}: |Δ⟨O⟩| = {err:.6} exceeds the \
                         claimed bound {budget:.6} (exact {exact:.6}, truncated {val:.6})"
                    ));
                }
            }
        }
    }

    eprintln!(
        "dropped_mass vs actual error: {compared} cases, {informative} with a \
         non-zero non-vacuous budget, tightest slack {worst_slack:.4e}"
    );
    assert!(
        violations.is_empty(),
        "dropped_mass did not bound the error in {} of {compared} cases:\n  {}\n\n\
         This is reported to users as a BOUND, not an estimate — a user sizes \
         their truncation against it. A violation means the claim is wrong, not \
         merely loose.",
        violations.len(),
        violations.join("\n  ")
    );
    assert!(
        informative >= 10,
        "only {informative} of {compared} cases had a non-zero, non-vacuous \
         budget. `dropped_mass = 0` satisfies the inequality trivially and is \
         the default on shallow circuits, so a corpus of those proves nothing"
    );
}

/// **Guard the guard: an untruncated run drops nothing and matches exactly.**
///
/// Without this, "the bound holds" is satisfied by a backend that truncates
/// everything and reports a huge budget.
#[test]
fn an_untruncated_run_reports_zero_and_is_exact() {
    let obs = Observable::parse("Z0").expect("observable");
    let params = ParameterBinding::new();
    let ir = circuit(10, 3);
    let exact = PauliPropBackend::new()
        .expectation(&ir, &params, &obs)
        .expect("exact");
    let (val, budget) = PauliPropBackend::new()
        .expectation_with_budget(&ir, &params, &obs)
        .expect("budget form");
    assert!(budget < 1e-12, "the exact engine must drop nothing, got {budget:.3e}");
    assert!(
        (val - exact).abs() < 1e-12,
        "the budget form must agree with the plain one: {val} vs {exact}"
    );
}

/// **Tighter truncation must not report a LARGER budget.** A budget that moves
/// the wrong way with the knob is not tracking anything.
#[test]
fn the_budget_shrinks_as_truncation_relaxes() {
    let obs = Observable::parse("Z0").expect("observable");
    let params = ParameterBinding::new();
    let ir = circuit(10, 3);
    let mut prev = f64::INFINITY;
    for w in [1usize, 2, 3, 4, 6] {
        let (_, budget) = PauliPropBackend::with_truncation_freq(0.0, Some(w), None)
            .expectation_with_budget(&ir, &params, &obs)
            .expect("truncated");
        assert!(
            budget <= prev + 1e-12,
            "max_weight {w} reported budget {budget:.6}, larger than the tighter \
             cap's {prev:.6} — the budget does not track the knob"
        );
        prev = budget;
    }
    assert!(prev < 1e-12, "max_weight 6 should drop nothing on this circuit, got {prev:.3e}");
}
