// SPDX-License-Identifier: Apache-2.0
//! Pins the one analytic-`Reset` divergence that changes an answer (ledger A6,
//! STATUS 5.5).
//!
//! Every other row of A6 makes a backend *refuse*. Metal does not: it calls
//! the same predicate as the CPU,
//! `omega_backend_statevector::sim::reset_is_deterministic_within`, but with
//! `tol = 1e-4` (`statevector-metal` `apply_reset_with`) where the CPU passes
//! `1e-9` (`reset_is_deterministic`). A qubit whose reduced purity sits in
//! `(1 − 1e-4, 1 − 1e-9)` is refused by the CPU and accepted by Metal, which
//! then returns one pure state and drops the other reset branch.
//!
//! No Metal device is involved. The test that does observe one is
//! `omega-backend-statevector-metal` `tests/reset_tolerance_observed.rs`
//! (`--features metal`). The disagreement is the `tol` argument, so
//! this calls the shared predicate at both tolerances on a constructed
//! statevector. `reduced_purity` is asserted into that window first: if the
//! construction drifts, the test fails here rather than quietly checking a
//! state both tolerances agree on.
//!
//! The number A6 quotes — purity `1 − 8·10⁻⁶`, a dropped branch of weight up
//! to `5·10⁻⁵` — is the Schmidt weight of the lighter branch. For
//! `√(1−ε)|00⟩ + √ε|11⟩` the purity deficit is `2ε(1−ε)` and the branch Metal
//! discards (the minority outcome; the two post-reset states are orthogonal)
//! weighs `ε`. At `ε = 4·10⁻⁵` that weight is `4·10⁻⁵`, under the ledger's
//! ceiling and inside the window.

use num_complex::Complex64;
use omega_backend_statevector::sim::{reduced_purity, reset_is_deterministic_within};

const CPU_TOL: f64 = 1e-9;
const METAL_TOL: f64 = 1e-4;
/// Ledger A6: Metal drops a branch of weight up to this.
const LEDGER_MAX_DROPPED_WEIGHT: f64 = 5e-5;

/// `√(1−ε)|00⟩ + √ε|11⟩`. Qubit 0's minority outcome has weight `ε`, and the
/// two post-reset states (`|00⟩` vs `|01⟩`) are orthogonal, so that weight is
/// a different answer, not a global phase.
fn slightly_entangled(eps: f64) -> Vec<Complex64> {
    let mut state = vec![Complex64::new(0.0, 0.0); 4];
    state[0] = Complex64::new((1.0 - eps).sqrt(), 0.0);
    state[3] = Complex64::new(eps.sqrt(), 0.0);
    state
}

fn product_zero() -> Vec<Complex64> {
    let mut state = vec![Complex64::new(0.0, 0.0); 4];
    state[0] = Complex64::new(1.0, 0.0);
    state
}

fn bell() -> Vec<Complex64> {
    let mut state = vec![Complex64::new(0.0, 0.0); 4];
    let a = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    state[0] = a;
    state[3] = a;
    state
}

/// Probability of the outcome analytic Metal does not take. It keeps the
/// heavier branch (`p0 < 0.5` selects outcome 1; see `apply_reset_with`).
fn discarded_branch_weight(state: &[Complex64], q: usize) -> f64 {
    let mask = 1usize << q;
    let p0: f64 = state
        .iter()
        .enumerate()
        .filter(|(i, _)| i & mask == 0)
        .map(|(_, a)| a.norm_sqr())
        .sum();
    p0.min(1.0 - p0)
}

#[test]
fn metal_tolerance_accepts_a_state_the_cpu_refuses_and_drops_a_branch() {
    // ε = 4e-5 → purity deficit 2ε(1−ε) ≈ 8e-5, inside (1e-9, 1e-4), and the
    // dropped weight is 4e-5, the order of A6's "up to 5e-5".
    let eps = 4e-5;
    let state = slightly_entangled(eps);
    let purity = reduced_purity(&state, 2, 0);

    assert!(
        purity > 1.0 - METAL_TOL && purity < 1.0 - CPU_TOL,
        "reduced purity {purity} left the window (1 - {METAL_TOL}, 1 - {CPU_TOL}) \
         where the two tolerances disagree; the construction drifted and this \
         test would otherwise pin nothing"
    );

    assert!(
        !reset_is_deterministic_within(&state, 2, 0, CPU_TOL),
        "CPU tol {CPU_TOL} must refuse purity {purity}: the qubit is entangled"
    );
    assert!(
        reset_is_deterministic_within(&state, 2, 0, METAL_TOL),
        "Metal tol {METAL_TOL} must accept purity {purity}: this is the silent \
         accept, the one A6 divergence that returns an answer"
    );

    let dropped = discarded_branch_weight(&state, 0);
    assert!(
        (dropped - eps).abs() < 1e-15,
        "discarded branch weight {dropped} should be the Schmidt weight ε = {eps}"
    );
    assert!(
        dropped > 1e-5 && dropped <= LEDGER_MAX_DROPPED_WEIGHT,
        "A6: Metal drops a branch of weight up to {LEDGER_MAX_DROPPED_WEIGHT} \
         and returns a pure state for it. Measured {dropped}, which is not \
         that order"
    );

    // Controls. A predicate that always returns false would still pass the
    // CPU-refuses assertion above.
    let pure = product_zero();
    let pure_purity = reduced_purity(&pure, 2, 0);
    assert!(
        (pure_purity - 1.0).abs() < 1e-15,
        "control: |00⟩ purity {pure_purity} is not 1"
    );
    assert!(
        reset_is_deterministic_within(&pure, 2, 0, CPU_TOL)
            && reset_is_deterministic_within(&pure, 2, 0, METAL_TOL),
        "a genuinely pure state must be accepted at both tolerances"
    );

    let entangled = bell();
    let bell_purity = reduced_purity(&entangled, 2, 0);
    assert!(
        (bell_purity - 0.5).abs() < 1e-12,
        "control: Bell purity {bell_purity} is not 1/2"
    );
    assert!(
        !reset_is_deterministic_within(&entangled, 2, 0, CPU_TOL)
            && !reset_is_deterministic_within(&entangled, 2, 0, METAL_TOL),
        "a maximally entangled state must be refused at both tolerances"
    );
}
