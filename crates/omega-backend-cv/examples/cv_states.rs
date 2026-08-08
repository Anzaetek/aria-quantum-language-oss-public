// SPDX-License-Identifier: Apache-2.0
//! Continuous-variable states on a truncated Fock space, with the piquasso
//! recipe that reproduces each one.
//!
//! Run it:
//! ```console
//! $ cargo run -p omega-backend-cv --example cv_states
//! ```
//!
//! **Why this is a Rust example and not a CLI invocation.** The DV photonic
//! modality has an OPTICQASM surface and `omega-run --backend photonics`; the CV
//! modality has neither. `omega-backend-cv` is a workspace member that no other
//! crate depends on, so today the Rust API *is* the whole user interface. That
//! is a real gap, tracked as C3 — stating it here beats letting someone discover
//! it after writing a `.cvqasm` file that nothing parses.
//!
//! Every number printed below is cross-checked against piquasso 8.0.1 by
//! `tests/piquasso_xcheck.rs`. The recipes in the comments are the exact ones
//! the fixture generator runs (`tools/cv_cross_check/piquasso_ref.py`), so they
//! are not illustrative prose — they are the thing that was actually compared.

use num_complex::Complex64;
use omega_backend_cv::FockState;

/// Fock cutoff. The truncated space holds levels `0..CUTOFF`, and everything
/// above is cut — which is why every routine here reports what it lost.
const CUTOFF: usize = 20;

fn show(label: &str, state: &FockState) {
    // Report <n> together with the truncation budget, never alone. A mean
    // photon number quoted without its error bar is the failure mode this
    // backend is built to avoid: the plausible number gets used and the caveat
    // does not.
    match state.expect_n(1e-6) {
        Ok(n) => println!(
            "  {label:<34} <n> = {n:.10}   lost mass {:.3e}, weighted {:.3e}",
            state.lost_norm(),
            state.lost_n_weight()
        ),
        Err(e) => println!("  {label:<34} REFUSED: {e}"),
    }
}

fn main() {
    println!("Continuous-variable states, cutoff {CUTOFF}\n");

    // -- Coherent state |alpha> ------------------------------------------
    //
    // piquasso:
    //     with pq.Program() as p:
    //         pq.Q(0) | pq.Vacuum()
    //         pq.Q(0) | pq.Displacement(r=1.0, phi=0.0)
    //     pq.PureFockSimulator(d=1, config=pq.Config(cutoff=20)).execute(p)
    //
    // NOTE the convention difference, which is the single most likely place to
    // introduce a silent error: our surface is CARTESIAN `alpha = re + i*im`
    // (K15), piquasso's `Displacement` is POLAR `(r, phi)`. They agree here
    // because r=1, phi=0 maps to re=1, im=0.
    println!("Coherent states  |alpha>   (analytic <n> = |alpha|^2)");
    for (re, im) in [(0.5, 0.0), (1.0, 0.0), (0.7, 0.7)] {
        let s = FockState::coherent(Complex64::new(re, im), CUTOFF).expect("coherent");
        show(&format!("alpha = {re} + {im}i"), &s);
    }

    // -- Squeezed vacuum S(r)|0> -----------------------------------------
    //
    // piquasso:
    //     pq.Q(0) | pq.Vacuum()
    //     pq.Q(0) | pq.Squeezing(r=0.8)
    //
    // Analytic <n> = sinh^2(r). Watch the gap open as r grows: the state
    // spreads up the Fock ladder until the cutoff starts cutting real weight,
    // and `lost_n_weight` is what tells you before <n> is quietly wrong.
    //
    // This is also where we and piquasso genuinely DIFFER: we evaluate the
    // closed-form amplitudes and then truncate, while piquasso applies an
    // already-truncated squeezing operator. Two defensible readings of the same
    // request; the cross-check asserts the gap stays inside the budget above
    // rather than pretending they agree.
    println!("\nSqueezed vacuum  S(r)|0>   (analytic <n> = sinh^2 r)");
    for r in [0.1, 0.3, 0.5, 0.8] {
        let s = FockState::squeezed_vacuum(r, CUTOFF).expect("squeezed");
        let exact = r.sinh().powi(2);
        let st = format!("r = {r}  (sinh^2 r = {exact:.10})");
        show(&st, &s);
    }

    // -- Diagonal gates ---------------------------------------------------
    //
    // piquasso:
    //     pq.Q(0) | pq.Kerr(xi=0.37)
    //     pq.Q(0) | pq.Phaseshifter(phi=0.7)
    //
    // Both are diagonal in the Fock basis, so both are EXACT on the truncated
    // space: no ladder is climbed, nothing falls past the cutoff, and the
    // probability vector does not move at all. Only the phases change.
    //
    // That invisibility is worth dwelling on. It means <n> cannot see these
    // gates, and neither can any probability-based comparison — a no-op `kerr`
    // would reproduce every number printed on this line. The cross-check
    // therefore compares AMPLITUDES up to global phase; mutation-testing
    // confirmed a no-op Kerr shows up there at 4.9e-1 and nowhere else.
    println!("\nDiagonal gates   (exact; probabilities must NOT move)");
    let mut s = FockState::coherent(Complex64::new(1.0, 0.0), CUTOFF).expect("coherent");
    let before: Vec<f64> = s.amplitudes().iter().map(|a| a.norm_sqr()).collect();
    show("|1.0> before kerr", &s);

    s.kerr(0.37).expect("kerr");
    show("|1.0> after kerr(0.37)", &s);

    s.phase_shift(0.7).expect("phase_shift");
    show("|1.0> after phase_shift(0.7)", &s);

    let after: Vec<f64> = s.amplitudes().iter().map(|a| a.norm_sqr()).collect();
    let moved = before
        .iter()
        .zip(&after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("  probability vector moved by {moved:.3e}  (must be ~0: both gates are diagonal)");

    // -- The cutoff biting ------------------------------------------------
    //
    // Deliberately ask for a state the truncated space cannot hold. The point
    // is that this REFUSES rather than returning a confident wrong number.
    // piquasso, asked the same question, returns a value — see the cross-check,
    // which asserts our leak metric predicts where that value stops being right.
    println!("\nCutoff too small — the refusal path");
    for cutoff in [4, 8, 20] {
        let s = FockState::squeezed_vacuum(1.5, cutoff).expect("squeezed");
        show(&format!("r = 1.5 at cutoff {cutoff}"), &s);
    }

    println!(
        "\nCross-checked against piquasso 8.0.1: cargo test -p omega-backend-cv\n\
         Regenerate/verify the reference: ./.venv-piquasso/bin/python \
         tools/cv_cross_check/verify_fixture.py"
    );
}
