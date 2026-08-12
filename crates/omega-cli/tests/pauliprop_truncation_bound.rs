// SPDX-License-Identifier: Apache-2.0
//! `dropped_mass` must be an ERROR BOUND, not a number we print.
//!
//! # Why this test exists
//!
//! `PauliPropBackend::expectation_with_budget` returns `(value, dropped_mass)`,
//! where `dropped_mass` accumulates the L1 magnitude of coefficients discarded
//! by truncation. The whole truncation story rests on that being an upper bound
//! on the resulting error in `⟨O⟩`:
//!
//! ```text
//!   |E_truncated − E_exact| ≤ dropped_mass
//! ```
//!
//! An earlier plan draft proposed *reporting* the budget alongside each cell.
//! That is weak: a number nobody compares against anything cannot be wrong. And
//! it would have been worse than weak — **`PauliPropBackend::new()` has
//! truncation OFF** (`coeff_min = 0.0`, `max_weight = None`), so on these
//! fixtures `dropped_mass ≡ 0.0` and any "refuse when the budget is large"
//! branch could never have executed. A budget subsystem that cannot fire is
//! decoration.
//!
//! So this asserts the inequality, against Qiskit's exact `Statevector` as
//! `E_exact`, sweeping `coeff_min` from "drops nothing" to "drops most of the
//! sum" — because a single truncation value could satisfy the bound by luck.
//!
//! # Why Qiskit and not ppvm for this one
//!
//! ppvm exposes the same truncation knobs (`min_abs_coeff`,
//! `max_pauli_weight`) and is the right anchor for the truncated *value* — same
//! algorithm, independent implementation. But it has **no dropped-mass total**:
//! that instrumentation is ours. So ppvm cannot validate our bound, only our
//! arithmetic. Qiskit gives the exact value the bound is measured against.
//! Conflating the two would leave the bound untested.

#![cfg(feature = "bridge-qiskit")]

use omega_backend_pauliprop::PauliPropBackend;
use omega_bridges::{expectation_qasm2, Backend, WireObservable};
use omega_core::executor::{Observable, PauliOp};
use omega_core::params::ParameterBinding;
use std::path::PathBuf;

fn runner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("omega-bridges")
        .join("python")
}
fn venv_python() -> PathBuf {
    runner_dir().join(".venv-qiskit").join("bin").join("python")
}
fn force_env() {
    std::env::set_var(
        "OMEGA_BRIDGE_QISKIT_CMD",
        runner_dir().join("omega-bridge-qiskit-runner"),
    );
}

/// A circuit with enough non-Clifford rotation to make the Pauli sum branch
/// widely, so truncation has real mass to discard. All-Clifford circuits keep
/// the sum at one term and would make every `coeff_min` a no-op — which is
/// exactly how this test could have been vacuous.
const QASM: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\n\
                    ry(0.7) q[0];\nrx(1.1) q[1];\ncx q[0],q[1];\nrz(0.9) q[1];\n\
                    ry(0.5) q[2];\ncx q[1],q[2];\nrx(0.8) q[0];\nrz(1.3) q[2];\n";

fn observable() -> (Observable, WireObservable) {
    // Z on qubit 0. Sparse (qubit-indexed) for us; dense LSB-first for the wire.
    (
        Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)])],
        },
        vec![("ZII".to_string(), 1.0)],
    )
}

#[test]
fn dropped_mass_bounds_the_truncation_error() {
    if !venv_python().exists() {
        eprintln!(
            "Qiskit venv missing at {} — skipping. Without the exact anchor \
             there is nothing to measure the bound against.",
            venv_python().display()
        );
        return;
    }
    force_env();

    let ir = omega_parser::lower_to_ir(QASM).expect("lower");
    let (obs, wire) = observable();
    let exact = expectation_qasm2(Backend::Qiskit, QASM, &[wire]).expect("qiskit anchor")[0];

    // Sweep chosen by MEASUREMENT, not by guessing. A first attempt used
    // [0, 1e-6, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.5] and the vacuity guard below
    // fired: only 1 of 8 points truncated anything. Probing the backend
    // directly showed why — this circuit's Heisenberg sum keeps few terms with
    // large coefficients, so `dropped_mass` stays 0 until coeff_min ≈ 0.45:
    //
    //   coeff_min ≤ 0.4  ->  dropped 0.000e0, value exact
    //   coeff_min = 0.5  ->  dropped 4.488e-1
    //   coeff_min = 0.7  ->  dropped 6.967e-1, value 0.0
    //
    // The range below straddles that threshold, so the sweep contains both
    // no-op and aggressive truncation and the bound is exercised on both sides.
    let sweep = [0.0, 0.3, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75];
    let mut fired = 0usize;
    let mut worst_slack = f64::INFINITY;

    eprintln!("\n  exact <Z0> = {exact:.12} (Qiskit Statevector)");
    eprintln!(
        "  {:>10}  {:>14}  {:>12}  {:>12}  {:>10}",
        "coeff_min", "E_truncated", "|E - exact|", "dropped_mass", "slack"
    );
    for coeff_min in sweep {
        let backend = PauliPropBackend::with_truncation(coeff_min, None);
        let (val, dropped) = backend
            .expectation_with_budget(&ir, &ParameterBinding::new(), &obs)
            .expect("pauliprop");
        let err = (val - exact).abs();
        let slack = dropped - err;
        eprintln!("  {coeff_min:>10.0e}  {val:>14.9}  {err:>12.3e}  {dropped:>12.3e}  {slack:>10.3e}");

        assert!(
            err <= dropped + 1e-12,
            "BOUND VIOLATED at coeff_min = {coeff_min:e}: |E_trunc − E_exact| = \
             {err:.6e} exceeds the reported dropped_mass {dropped:.6e}. \
             dropped_mass is documented as an upper bound on the expectation \
             error; if it under-reports, every truncated result in the matrix \
             carries an error budget smaller than its actual error."
        );
        if dropped > 1e-12 {
            fired += 1;
            worst_slack = worst_slack.min(slack);
        }
    }

    // The guard that stops this test being decoration: at least some points in
    // the sweep must ACTUALLY TRUNCATE. With the default backend
    // (`PauliPropBackend::new()`) dropped_mass is identically 0 and every
    // assertion above holds trivially.
    assert!(
        fired >= 4,
        "only {fired} of {} sweep points had nonzero dropped_mass — the sweep \
         is not exercising truncation, so the bound assertions are vacuous. \
         Widen the coeff_min range or use a circuit whose Pauli sum branches \
         more.",
        sweep.len()
    );
    eprintln!("  {fired} of {} sweep points truncated; tightest slack {worst_slack:.3e}",
              sweep.len());
}

/// The bound must be reachable, not absurdly loose.
///
/// A bound of `+∞` would satisfy every assertion above. This checks that at the
/// most aggressive truncation the reported mass is within a few orders of the
/// actual error — enough to confirm it tracks reality rather than merely
/// dominating it.
#[test]
fn the_bound_is_informative_not_merely_true() {
    if !venv_python().exists() {
        eprintln!("Qiskit venv missing — skipping");
        return;
    }
    force_env();
    let ir = omega_parser::lower_to_ir(QASM).expect("lower");
    let (obs, wire) = observable();
    let exact = expectation_qasm2(Backend::Qiskit, QASM, &[wire]).expect("qiskit")[0];

    let (val, dropped) = PauliPropBackend::with_truncation(0.7, None)
        .expectation_with_budget(&ir, &ParameterBinding::new(), &obs)
        .expect("pauliprop");
    let err = (val - exact).abs();
    assert!(dropped > 1e-9, "no mass dropped at coeff_min = 0.7: {dropped:e}");
    assert!(
        dropped < 1e4,
        "dropped_mass = {dropped:e} is so large it bounds nothing useful"
    );
    eprintln!(
        "  aggressive truncation: |Δ| = {err:.3e}, bound = {dropped:.3e}, \
         ratio = {:.1}x",
        dropped / err.max(1e-15)
    );
}
