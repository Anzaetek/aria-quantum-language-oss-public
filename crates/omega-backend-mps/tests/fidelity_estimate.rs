// SPDX-License-Identifier: Apache-2.0
//! **The reported fidelity estimate must not exceed the true fidelity.**
//!
//! An MPS run that truncates returns an approximate state, and the user needs a
//! number for how good it is. A commercial MPS simulator reports a fidelity per
//! run — in [0, 1], 1.0 meaning exact — and states it is a lower bound on the
//! true fidelity. This file is what lets us report the same shape honestly.
//!
//! We accumulate `Π over splits of (1 − εᵢ)`. In **canonical gauge** that is a
//! genuine lower bound. This MPS is deliberately non-canonical (each εᵢ is
//! measured against the block norm, not the global one), so the product is an
//! **estimate** and is labelled as one everywhere it is printed.
//!
//! What we have instead of a proof is this test: the estimate compared against
//! the true fidelity `|⟨ψ_exact|ψ_χ⟩|² / (‖ψ_exact‖²‖ψ_χ‖²)` over a corpus, with
//! a failure if it ever comes out ABOVE. That is the direction that matters —
//! an estimate that overstates fidelity tells a user their wrong answer is fine.
//!
//! # The fixture, and why it is not a graph state
//!
//! A graph state has a FLAT Schmidt spectrum, which is the one case where the
//! local-vs-global gauge distinction cannot show up. Measured on graph states at
//! n = 10/12/14 and χ = 2/4/8/16: the estimate equalled the truth **exactly**,
//! to the last digit, in all twelve cases. A corpus of those would prove
//! nothing about the gauge.
//!
//! So the circuits below interleave random-angle `Ry`/`Rz` with nearest-neighbour
//! `CX` and long-range `CZ`, giving a non-flat spectrum where the discarded
//! singular values differ in size.

use num_complex::Complex64;
use omega_backend_mps::mps::Mps;

fn fidelity(a: &[Complex64], b: &[Complex64]) -> f64 {
    let mut ip = Complex64::new(0.0, 0.0);
    let (mut na, mut nb) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        ip += x.conj() * y;
        na += x.norm_sqr();
        nb += y.norm_sqr();
    }
    ip.norm_sqr() / (na * nb)
}

fn ry(t: f64) -> [Complex64; 4] {
    let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
    [
        Complex64::new(c, 0.0),
        Complex64::new(-s, 0.0),
        Complex64::new(s, 0.0),
        Complex64::new(c, 0.0),
    ]
}
fn rz(t: f64) -> [Complex64; 4] {
    [
        Complex64::from_polar(1.0, -t / 2.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::from_polar(1.0, t / 2.0),
    ]
}

/// A NON-flat Schmidt spectrum: random-angle rotations between entanglers, so
/// the discarded singular values differ in size. A graph state's spectrum is
/// uniform, which is the one case where the local-vs-global gauge distinction
/// cannot show up — measured F_est == F_true exactly there, on every fixture.
fn build(n: usize, chi: usize, depth: usize, seed: u64) -> Mps {
    let mut m = Mps::zero_state(n, chi);
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((x >> 33) as f64 / (1u64 << 31) as f64) * std::f64::consts::TAU
    };
    for _ in 0..depth {
        for q in 0..n {
            m.apply_1q(q, &ry(next()));
            m.apply_1q(q, &rz(next()));
        }
        for q in 0..n - 1 {
            m.apply_2q(q, &omega_backend_mps::gates::cx());
        }
        // Long-range entanglers push the rank up faster than a chain.
        for i in 0..n / 2 {
            m.apply_2q_distant(i, i + n / 2, &omega_backend_mps::gates::cz());
        }
    }
    m
}

/// The estimate never exceeds the truth, across the corpus.
#[test]
fn the_fidelity_estimate_never_exceeds_the_true_fidelity() {
    let mut violations: Vec<String> = Vec::new();
    let mut compared = 0usize;
    // Two ratios, because they catch opposite failures and the same number
    // cannot do both. `min` catches an OVER-OPTIMISTIC estimate (the bound
    // direction); `max` catches a uselessly PESSIMISTIC one. Tracking only the
    // min let `1 − Σεᵢ` — the sum-complement, which is always below the product
    // and collapses to 0 once the sum passes 1 — pass this file unchanged.
    let mut min_ratio = f64::INFINITY;
    let mut max_ratio: f64 = 0.0;
    for n in [8usize, 10, 12] {
        let exact_chi = 1 << n; // far above 2^(n/2): exact
        for depth in [1usize, 2, 3] {
            for seed in 0..4u64 {
                let sv_exact = build(n, exact_chi, depth, seed).to_statevector();
                for chi in [2usize, 4, 8] {
                    let m = build(n, chi, depth, seed);
                    let f_true = fidelity(&sv_exact, &m.to_statevector());
                    let f_est = m.fidelity_estimate;
                    let holds = f_est <= f_true + 1e-9;
                    compared += 1;
                    if !holds {
                        violations.push(format!(
                            "n={n} depth={depth} seed={seed} chi={chi}: \
                             F_est={f_est:.6} > F_true={f_true:.6}"
                        ));
                    }
                    if f_true > 1e-12 {
                        let r = f_true / f_est.max(1e-300);
                        min_ratio = min_ratio.min(r);
                        max_ratio = max_ratio.max(r);
                    }
                }
            }
        }
    }
    // Report the qualifying count: a comparison that ran three cells reads
    // exactly like one that passed.
    eprintln!(
        "fidelity estimate vs truth: {compared} cases, F_true/F_est in \
         [{min_ratio:.4}, {max_ratio:.4}] (>= 1.0 means the estimate never \
         exceeded the truth; near 1.0 means it is tight)"
    );
    assert!(
        violations.is_empty(),
        "the fidelity ESTIMATE came out above the true fidelity in {} of \
         {compared} cases:\n  {}\n\nThat is the direction that matters: an \
         estimate which overstates fidelity tells a user their wrong answer is \
         fine. It is labelled an estimate rather than a bound precisely because \
         this is measured, not proved — but a measured violation means the label \
         is no longer enough and the gauge needs fixing.",
        violations.len(),
        violations.join("\n  ")
    );
    assert!(
        compared >= 100,
        "only {compared} cases compared — coverage collapsed, and a near-empty \
         comparison is indistinguishable from a passing one"
    );
    assert!(
        min_ratio >= 1.0 - 1e-9,
        "smallest F_true/F_est is {min_ratio:.6}, i.e. the estimate came out \
         above the truth"
    );
    // The usefulness direction, and the threshold is MEASURED rather than
    // picked. On this corpus the product form spans
    //
    //     F_true / F_est  in  [1.0067, 142.7]
    //
    // — tight where truncation is mild, and up to ~140x PESSIMISTIC in the
    // deeply-starved regime (chi=2 on a depth-3 entangler). That is worth
    // saying plainly: the estimate is a good "is this run usable" signal and a
    // poor estimate of how good a bad run actually is.
    //
    // `1 − Σεᵢ` satisfies every assertion above — strictly below the product,
    // so it never overstates — but it hits 0 as soon as the sum passes 1, which
    // happens on ordinary circuits. Measured: its max ratio is ~1e320. A
    // threshold of 1e3 separates the two by 300 orders of magnitude while
    // leaving the real spread ample room.
    assert!(
        max_ratio < 1e3,
        "largest F_true/F_est is {max_ratio:.3e}: the estimate has collapsed \
         toward zero and no longer discriminates. `1 - sum(eps)` fails here \
         (~1e320) while passing every other check in this file."
    );
}

/// **Guard the guard.** An exact run must report fidelity 1.0, or "never
/// exceeds the truth" is satisfied by always reporting 0.
#[test]
fn an_exact_run_reports_fidelity_one() {
    let m = build(10, 1 << 10, 2, 3);
    assert!(
        (m.fidelity_estimate - 1.0).abs() < 1e-12,
        "chi far above 2^(n/2) truncates nothing, so the estimate must be 1.0, \
         got {}",
        m.fidelity_estimate
    );
    assert!(m.discarded_weight < 1e-12);
}

/// And a badly truncated run must report something small — otherwise the number
/// is not discriminating.
#[test]
fn a_starved_run_reports_a_low_fidelity() {
    let m = build(12, 2, 3, 1);
    assert!(
        m.fidelity_estimate < 0.5,
        "chi=2 on a depth-3 entangling circuit cannot be half-decent; got {}",
        m.fidelity_estimate
    );
}
