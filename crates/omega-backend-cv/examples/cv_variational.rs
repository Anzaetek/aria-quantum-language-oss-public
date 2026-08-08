// SPDX-License-Identifier: Apache-2.0
//! A QAOA-shaped continuous-variable variational circuit, and an honest account
//! of what it is and is not.
//!
//! Run it:
//! ```console
//! $ cargo run -p omega-backend-cv --example cv_variational
//! ```
//!
//! # What this is
//!
//! The QAOA *shape* is an alternation of two non-commuting layers: a diagonal
//! "cost" layer and a non-diagonal "mixer". On a truncated Fock space this crate
//! can express exactly that:
//!
//! ```text
//!   cost  layer:  exp(i·γ·n²) · exp(i·β·n)     -> kerr(γ) then phase_shift(β)
//!   mixer layer:  D(α)                          -> displace(α)
//! ```
//!
//! `kerr` and `phase_shift` are diagonal in the Fock basis, so on their own they
//! move only phases and can never change a photon-number distribution. `D(α)`
//! does not commute with `n`, so it is what turns those phases into amplitude.
//! That non-commutation is the entire mechanism, and it is why this example
//! could not be written before `displace` existed as an **operator** — the
//! `coherent(α)` constructor can only ever act on vacuum.
//!
//! # What this is NOT
//!
//! **This is not the CV QAOA of Verdon et al.** (arXiv:1902.00409), and calling
//! it that would be an overstatement worth avoiding. That algorithm evolves
//! under `exp(-i·γ·f(x̂))` for an arbitrary objective `f` of the position
//! quadrature — "gradient descent in superposition" — and its photonic
//! realisation is Enomoto et al., Phys. Rev. Research 5, 043005 (2023)
//! (arXiv:2206.07214).
//!
//! This crate has no position-basis operator and no way to build `f(x̂)` for
//! general `f`. What it has is a quadratic-in-`n` diagonal generator, which is a
//! *particular* cost, not an arbitrary one. So: same structural idea, strictly
//! smaller expressive class. Implementing the real thing needs `exp(-i·γ·f(x̂))`,
//! which is a separate piece of work.
//!
//! # The task
//!
//! Prepare the Fock state `|2⟩` from vacuum, maximising `|⟨2|ψ⟩|²`. A single
//! displacement cannot do this — a coherent state has Poissonian statistics and
//! its `|2⟩` weight peaks around 0.27 — so any score meaningfully above that had
//! to come from the interference between layers, which makes the number
//! self-checking rather than merely plausible.

use num_complex::Complex64;
use omega_backend_cv::FockState;

const CUTOFF: usize = 24;
const TARGET: usize = 2;
const LAYERS: usize = 3;

/// Build the ansatz state for a parameter vector `[α₀, γ₀, β₀, α₁, …]`.
///
/// Returns `None` if the circuit leaked more than the tolerance — a refusal is
/// reported rather than a plausible number, so the optimiser cannot wander into
/// a region where the truncation is quietly doing the work.
fn ansatz(params: &[f64]) -> Option<FockState> {
    let mut state = FockState::vacuum(CUTOFF).ok()?;
    for layer in params.chunks(3) {
        state.displace(Complex64::new(layer[0], 0.0)).ok()?;
        state.kerr(layer[1]).ok()?;
        state.phase_shift(layer[2]).ok()?;
    }
    Some(state)
}

/// `|⟨TARGET|ψ⟩|²`, normalised by the represented mass.
fn score(params: &[f64]) -> f64 {
    let Some(state) = ansatz(params) else {
        return f64::NEG_INFINITY;
    };
    // Refuse to score a state whose truncation error is comparable to the
    // quantity being optimised. Without this the optimiser is free to find a
    // "solution" that is an artefact of the cutoff.
    if state.lost_norm() > 1e-6 {
        return f64::NEG_INFINITY;
    }
    let norm = state.norm_sqr();
    if norm <= 0.0 {
        return f64::NEG_INFINITY;
    }
    state.amplitudes()[TARGET].norm_sqr() / norm
}

/// Finite-difference gradient ascent. Deliberately plain: the point of this
/// example is the physics of the ansatz, not the optimiser.
///
/// Returns the **best-scoring** parameters seen, not the last ones. That
/// distinction is not pedantry: gradient ascent here happily walks into the
/// region where the state leaks past the cutoff, where `score` refuses and
/// returns `-inf`. An earlier version of this example returned the final
/// parameters and reported `best` from a separate variable, so the two were out
/// of sync — it printed a headline score of 0.238 alongside a "final state"
/// showing `P(|2>) = 0.7536` and `<n> = NaN`, computed from a state the scorer
/// had already rejected for leaking. A refused state with an attractive-looking
/// number is precisely what the leak guard exists to prevent, so it must not
/// come back in through the reporting path.
fn optimise(mut params: Vec<f64>, steps: usize, lr: f64) -> (Vec<f64>, f64) {
    let eps = 1e-5;
    let mut best = score(&params);
    let mut best_params = params.clone();
    for _ in 0..steps {
        let mut grad = vec![0.0; params.len()];
        for i in 0..params.len() {
            let orig = params[i];
            params[i] = orig + eps;
            let up = score(&params);
            params[i] = orig - eps;
            let down = score(&params);
            params[i] = orig;
            if up.is_finite() && down.is_finite() {
                grad[i] = (up - down) / (2.0 * eps);
            }
        }
        for (p, g) in params.iter_mut().zip(&grad) {
            *p += lr * g;
        }
        let s = score(&params);
        if s.is_finite() && s > best {
            best = s;
            best_params = params.clone();
        }
    }
    (best_params, best)
}

fn main() {
    println!("CV variational state preparation — target |{TARGET}>, cutoff {CUTOFF}\n");

    // Baseline: the best a SINGLE displacement can do. A coherent state's Fock
    // distribution is Poissonian, so P(2) = e^{-|a|^2} |a|^4 / 2, maximal at
    // |a|^2 = 2 giving 2 e^{-2} ~ 0.2707. Any score above this is evidence the
    // layered interference is doing real work.
    let mut best_coherent = (0.0, 0.0);
    for i in 0..=400 {
        let a = i as f64 * 0.01;
        let s = score(&[a, 0.0, 0.0]);
        if s > best_coherent.1 {
            best_coherent = (a, s);
        }
    }
    println!(
        "  single displacement (baseline): P(|{TARGET}>) = {:.6} at alpha = {:.2}",
        best_coherent.1, best_coherent.0
    );
    println!("  analytic Poissonian optimum:    P(|2>) = {:.6}\n", 2.0 * (-2.0f64).exp());

    // The layered ansatz, from several starts — a plain gradient ascent on a
    // non-convex landscape finds different local optima, and reporting only the
    // best of one run would be cherry-picking.
    let starts: [Vec<f64>; 4] = [
        vec![1.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0],
        vec![1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        vec![0.8, -0.3, 0.7, 0.8, 0.3, -0.7, 0.4, 0.1, 0.2],
        vec![1.2, 0.9, -0.4, 0.6, -0.2, 0.8, 0.2, 0.4, -0.1],
    ];

    let mut overall = (Vec::new(), f64::NEG_INFINITY);
    for (i, start) in starts.iter().enumerate() {
        let (p, s) = optimise(start.clone(), 600, 0.4);
        println!("  {LAYERS}-layer ansatz, start {i}: P(|{TARGET}>) = {s:.6}");
        if s > overall.1 {
            overall = (p, s);
        }
    }

    println!("\n  best: P(|{TARGET}>) = {:.6}", overall.1);
    if overall.1 > best_coherent.1 {
        println!(
            "  -> beats the single-displacement baseline by {:.4}, so the gain comes\n     \
             from interference between layers, not from displacement alone.",
            overall.1 - best_coherent.1
        );
    } else {
        println!(
            "  -> did NOT beat the baseline. Reported as-is: an optimiser that fails\n     \
             to find structure is a result, not something to hide."
        );
    }

    if let Some(state) = ansatz(&overall.0) {
        println!(
            "\n  final state: <n> = {:.6}, lost mass {:.3e} (weighted {:.3e})",
            state.expect_n(1e-6).unwrap_or(f64::NAN),
            state.lost_norm(),
            state.lost_n_weight()
        );
        let norm = state.norm_sqr();
        print!("  Fock profile:");
        for k in 0..6 {
            print!(" |{k}>={:.4}", state.amplitudes()[k].norm_sqr() / norm);
        }
        println!();
    }

    println!(
        "\n  NOTE: this is QAOA-SHAPED (alternating diagonal cost / non-diagonal\n  \
         mixer), not the CV QAOA of Verdon et al. (arXiv:1902.00409), which needs\n  \
         exp(-i*gamma*f(x_hat)) for arbitrary f. See the module docs."
    );
}
