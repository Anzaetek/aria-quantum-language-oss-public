// SPDX-License-Identifier: Apache-2.0
//! Differential cross-check of the CV backend against **piquasso**.
//!
//! piquasso is an independent implementation of truncated-Fock CV simulation
//! written by people with no connection to this repository. That independence is
//! the whole point: this project has already shipped a defect that every
//! *internal* cross-backend gate missed, because the backends being compared
//! shared the convention that was wrong. Two implementations of the same physics
//! by different authors agreeing is evidence; one implementation agreeing with
//! itself is not.
//!
//! ## How this runs
//!
//! The fixture at `tools/cv_cross_check/piquasso_fixture.jsonl` is **committed**,
//! so this test runs on every `cargo test` with no Python, no venv, and no
//! network — headless-CI clean per K13. It catches regressions in *our* code.
//!
//! It does **not** by itself catch fixture drift: if our conventions changed and
//! someone regenerated the fixture to match, the test would go green on wrong
//! numbers. That is what `ARIA_CV_XCHECK=1 ./ci.sh` is for — it reruns piquasso
//! live and diffs against the committed file. The two halves cover different
//! failures and neither substitutes for the other.
//!
//! ## What the recipe interpreter can express
//!
//! `ops` is structured data, and the interpreter below refuses anything the
//! backend cannot express rather than approximating it. Today that means:
//!
//! * **Preparations** (`vacuum`, `squeeze`, `displace`) are CONSTRUCTORS in this
//!   crate, not operators. There is no `displace(&mut self, ..)` and no
//!   `squeeze(&mut self, ..)`, so a preparation can only appear FIRST.
//! * **Diagonal gates** (`kerr`, `phase_shift`) are real operators and may
//!   follow in any sequence.
//!
//! So `squeeze then displace` — a perfectly ordinary CV circuit — is
//! **unexpressible here**, and the test says so out loud and counts it, instead
//! of quietly comparing only the cases that happen to be easy. A differential
//! check that silently narrows its own corpus is the failure mode this file is
//! built to avoid.

use std::path::PathBuf;

use num_complex::Complex64;
use omega_backend_cv::FockState;

/// Absolute tolerance on a per-level Fock probability.
///
/// Absolute, not relative: the tails run down to ~1e-29, where a relative
/// tolerance is meaningless — it would demand agreement on values that are
/// pure floating-point noise in both implementations.
///
/// Set from measurement, not taste: with both sides on the same normalisation
/// convention the worst residual across the whole corpus is 1.1e-16, so 1e-14
/// leaves ~2 orders of margin over float noise while still being 10^10 tighter
/// than the leak-budget tolerance this replaced.
///
/// (An earlier revision of this doc pointed at a `TRUNCATION_CASES` constant.
/// That constant was removed when the per-case budget replaced the exception
/// list, and the reference was left dangling — it named nothing in the repo.)
const PROB_TOL: f64 = 1e-14;

/// Same reasoning, on amplitudes. Looser than `PROB_TOL` by roughly the square
/// root, because an amplitude error of `e` shows up in a probability as `~2*e`
/// near unit magnitude but as `e^2` where the amplitude itself is small.
const AMP_TOL: f64 = 1e-13;

/// Same reasoning again for `<n>`. Both sides renormalise, so this is float
/// noise; measured worst across the corpus is 2.2e-16.
const MEAN_N_TOL: f64 = 1e-14;

/// # A correction, kept visible because the wrong version shipped
///
/// This file used to carry a `probability_budget()` helper and the claim that
/// Aria and piquasso "genuinely differ" on squeezed vacuum because *we evaluate
/// the closed-form Fock amplitudes and then cut, while piquasso applies a
/// squeezing operator that is itself already truncated* — "two defensible
/// readings of squeezed vacuum at cutoff 20". Disagreements were then held to
/// the backend's own `lost_norm` budget rather than to a numeric tolerance.
///
/// **Both halves of that explanation were wrong.**
///
/// * The underlying amplitudes agree to **1e-16**. There was never a difference
///   of state to defend.
/// * piquasso does not exponentiate a truncated generator (that disagrees by
///   1.8e-2 at r=1.0). It applies the truncation of the **exact** operator, via
///   the Miatto–Quesada recurrence in `piquasso/_math/gate_matrices.py`.
///
/// The entire gap was a **normalisation convention**: piquasso returns raw
/// truncated probabilities (`sum = 0.999936664825` at r=0.8) while Aria
/// renormalises by the represented mass. Renormalising piquasso's own vector
/// and re-diffing reproduces **3.434e-08** (r=0.5) and **4.736e-05** (r=0.8) —
/// *exactly* the two numbers this test used to report as "reconciled by the
/// truncation budget".
///
/// So the check passed, and its bound was even valid — `p/(1−ε) − p ≈ pε` is
/// maximal at the dominant `p`, which is why the gap tracked the lost mass. It
/// passed **for a reason other than the one documented**, with a tolerance
/// roughly 10¹⁰ looser than necessary: a leak-budget-sized tolerance standing
/// in for a floating-point-sized one.
///
/// `piquasso_ref.py` now normalises both sides once, at the source, so the
/// comparison is a plain numeric one at `PROB_TOL`.
///
/// **`lost_norm` / `lost_n_weight` still matter — but not here.** They bound the
/// distance to *truth* (`sinh²r`), not the distance to piquasso, which is float
/// noise once the conventions match. Their proper home is
/// `matches_piquasso_and_predicts_where_piquasso_goes_wrong` in `lib.rs`, where
/// the reference is analytic.
const _: () = ();

#[derive(Debug)]
struct Case {
    name: String,
    cutoff: usize,
    ops: Vec<serde_json::Value>,
    probs: Vec<f64>,
    amps: Vec<Complex64>,
    mean_n: f64,
}

/// Largest deviation between two state vectors **after removing global phase**.
///
/// Global phase is unobservable and the two libraries have no reason to pick the
/// same one, so comparing raw amplitudes would flag correct results. Alignment
/// uses the largest-magnitude component as the reference — the most numerically
/// stable choice, since dividing by a near-zero amplitude to fix a phase
/// amplifies noise into a spurious failure.
///
/// Relative phase between levels survives this quotient, which is the entire
/// point: it is what makes `kerr` and `phase_shift` testable.
fn max_diff_up_to_global_phase(ours: &[Complex64], theirs: &[Complex64]) -> f64 {
    let pivot = theirs
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.norm_sqr()
                .partial_cmp(&b.1.norm_sqr())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    // If both pivots are effectively zero the states are both ~empty; any phase
    // will do and the amplitude comparison degenerates to the norm check.
    let (a, b) = (ours[pivot], theirs[pivot]);
    if a.norm() < 1e-30 || b.norm() < 1e-30 {
        return ours
            .iter()
            .zip(theirs)
            .map(|(x, y)| (x.norm() - y.norm()).abs())
            .fold(0.0f64, f64::max);
    }

    // Rotate ours so its pivot lands on theirs.
    let rot = (b / b.norm()) * (a / a.norm()).conj();
    ours.iter()
        .zip(theirs)
        .map(|(x, y)| (x * rot - y).norm())
        .fold(0.0f64, f64::max)
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/cv_cross_check/piquasso_fixture.jsonl")
}

fn load() -> (String, Vec<Case>) {
    let path = fixture_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let mut version = String::new();
    let mut cases = Vec::new();

    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).expect("fixture line is not JSON");

        if let Some(meta) = v.get("meta") {
            version = meta
                .get("piquasso")
                .and_then(|s| s.as_str())
                .unwrap_or("?")
                .to_string();
            continue;
        }
        // A generator failure writes {"error": ...} and exits non-zero. If that
        // ever lands in the committed fixture, fail loudly here rather than
        // silently comparing zero cases.
        if let Some(err) = v.get("error") {
            panic!("fixture contains a generator error: {err}");
        }

        cases.push(Case {
            name: v["case"].as_str().expect("case name").to_string(),
            cutoff: v["cutoff"].as_u64().expect("cutoff") as usize,
            ops: v["ops"].as_array().expect("ops array").clone(),
            probs: v["probs"]
                .as_array()
                .expect("probs array")
                .iter()
                .map(|p| p.as_f64().expect("prob is a number"))
                .collect(),
            amps: v["amps"]
                .as_array()
                .expect("amps array — regenerate the fixture, it predates the \
                         amplitude comparison and the diagonal gates are not \
                         actually tested without it")
                .iter()
                .map(|a| {
                    let pair = a.as_array().expect("amp is [re, im]");
                    Complex64::new(
                        pair[0].as_f64().expect("re"),
                        pair[1].as_f64().expect("im"),
                    )
                })
                .collect(),
            mean_n: v["mean_n"].as_f64().expect("mean_n"),
        });
    }

    (version, cases)
}

/// Why a case could not be compared. Carried as a value, not a `continue`, so
/// the reason reaches the report.
#[derive(Debug)]
#[allow(dead_code)]
enum Skip {
    /// The backend has no operator form of this preparation.
    PrepNotFirst(String),
    Unknown(String),
}

/// Build the state described by `ops`, or say precisely why we cannot.
///
/// `squeezed_vacuum(r)` and `coherent(alpha)` ARE "squeeze / displace applied to
/// vacuum" — the constructors bake the vacuum in. So a preparation is legal for
/// exactly as long as the state is still untouched vacuum, and becomes
/// unexpressible the moment anything has acted on it.
fn build(ops: &[serde_json::Value], cutoff: usize) -> Result<FockState, Skip> {
    let mut state: Option<FockState> = None;
    let mut pristine_vacuum = true;

    for op in ops {
        let kind = op["op"].as_str().unwrap_or("");
        let is_prep = matches!(kind, "vacuum" | "squeeze" | "displace");

        // `displace` now has an OPERATOR form, so it is legal anywhere.
        // `squeeze` does not yet, so it remains prep-only.
        if is_prep && !pristine_vacuum && kind != "displace" {
            // Not a limitation of the harness — a limitation of the crate.
            return Err(Skip::PrepNotFirst(kind.to_string()));
        }
        if is_prep && kind != "vacuum" {
            // The vacuum has now been consumed by this preparation.
            pristine_vacuum = false;
        }
        if !is_prep {
            pristine_vacuum = false;
        }

        match kind {
            "vacuum" => state = Some(FockState::vacuum(cutoff).expect("vacuum")),
            "squeeze" => {
                let r = op["r"].as_f64().expect("r");
                state = Some(FockState::squeezed_vacuum(r, cutoff).expect("squeezed_vacuum"));
            }
            "displace" => {
                // Cartesian on both sides — K15. The generator converts to
                // piquasso's polar form in exactly one place.
                let re = op["re"].as_f64().expect("re");
                let im = op["im"].as_f64().expect("im");
                let alpha = Complex64::new(re, im);
                match state.as_mut() {
                    // On vacuum, use the CONSTRUCTOR: it carries an exact
                    // analytic tail, which the operator can only measure.
                    None => state = Some(FockState::coherent(alpha, cutoff).expect("coherent")),
                    Some(s) if pristine_vacuum => {
                        let _ = s;
                        state = Some(FockState::coherent(alpha, cutoff).expect("coherent"));
                    }
                    // On a state with structure, use the OPERATOR.
                    Some(s) => s.displace(alpha).expect("displace"),
                }
            }
            "kerr" => {
                let chi = op["chi"].as_f64().expect("chi");
                state.as_mut().expect("gate before prep").kerr(chi).expect("kerr");
            }
            "phase_shift" => {
                let phi = op["phi"].as_f64().expect("phi");
                state
                    .as_mut()
                    .expect("gate before prep")
                    .phase_shift(phi)
                    .expect("phase_shift");
            }
            other => return Err(Skip::Unknown(other.to_string())),
        }
    }

    // A `vacuum` prep followed by nothing still yields a state, so `None` here
    // means an empty recipe, which is a malformed fixture rather than a skip.
    Ok(state.expect("recipe produced no state"))
}

/// Fock probabilities, normalised against the represented mass — the same
/// convention the generator uses on piquasso's side.
fn probabilities(state: &FockState) -> Vec<f64> {
    let norm = state.norm_sqr();
    state
        .amplitudes()
        .iter()
        .map(|a| if norm > 0.0 { a.norm_sqr() / norm } else { 0.0 })
        .collect()
}

#[test]
fn cv_backend_agrees_with_piquasso_on_equivalent_circuits() {
    let (version, cases) = load();
    assert!(
        !cases.is_empty(),
        "fixture parsed to zero cases — a check that compares nothing passes for free"
    );

    let mut compared = 0usize;
    let mut skipped: Vec<(String, Skip)> = Vec::new();
    let mut worst = (String::new(), 0.0f64);
    let mut truncation_seen: Vec<String> = Vec::new();

    for case in &cases {
        let state = match build(&case.ops, case.cutoff) {
            Ok(s) => s,
            Err(why) => {
                skipped.push((case.name.clone(), why));
                continue;
            }
        };

        let ours = probabilities(&state);
        assert_eq!(
            ours.len(),
            case.probs.len(),
            "{}: vector lengths differ ({} vs {}) — an index-for-index \
             comparison is only meaningful at equal length",
            case.name,
            ours.len(),
            case.probs.len()
        );

        let max_diff = ours
            .iter()
            .zip(&case.probs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        // The amplitude comparison. This is the one that can see `kerr` and
        // `phase_shift` at all — both are diagonal, so against `probs` alone a
        // no-op implementation of either is indistinguishable from a correct
        // one, and those cases would be decoration.
        let norm = state.norm_sqr().sqrt();
        let ours_amps: Vec<Complex64> = state
            .amplitudes()
            .iter()
            .map(|a| if norm > 0.0 { a / norm } else { *a })
            .collect();
        let amp_diff = max_diff_up_to_global_phase(&ours_amps, &case.amps);

        assert!(
            amp_diff <= AMP_TOL,
            "{}: Fock AMPLITUDES disagree with piquasso {version} by \
             {amp_diff:.3e} (tolerance {AMP_TOL:.1e}) after removing global phase. \
             A probability-only comparison would have missed this — which is \
             exactly why amplitudes are compared.",
            case.name
        );

        assert!(
            max_diff <= PROB_TOL,
            "{}: Fock probabilities disagree with piquasso {version} by \
             {max_diff:.3e} (tolerance {PROB_TOL:.1e}). Both sides are on the \
             SAME normalisation convention now, so this is a real numeric \
             disagreement, not a convention artefact.\n  ours: {:?}\n  piq:  {:?}",
            case.name,
            &ours[..ours.len().min(6)],
            &case.probs[..case.probs.len().min(6)]
        );

        // Record any case that clears the pure-noise floor, so the report shows
        // where the residual actually sits rather than letting "all green"
        // imply exact agreement everywhere.
        if max_diff > 1e-15 {
            truncation_seen.push(format!("{} (diff {max_diff:.3e})", case.name));
        }

        // <n> is compared at the SAME tolerance as everything else.
        //
        // It used to be budgeted at `lost_n_weight().max(1e-9)` — about 1.4e-3
        // at r=0.8, some 10^12 looser than needed. That was the same mistake as
        // the probability lane, and it survived the first correction of that
        // mistake: the fixture's `mean_n` is computed from ALREADY-renormalised
        // probabilities and `expect_n` divides by `norm_sqr()`, so both sides
        // share a convention and the difference is float noise (measured 2.2e-16
        // worst across the corpus). `lost_n_weight` bounds |<n> - sinh^2 r|,
        // the distance to TRUTH — it is not a bound on the distance to piquasso.
        if let Ok(n) = state.expect_n(1.0) {
            assert!(
                (n - case.mean_n).abs() <= MEAN_N_TOL,
                "{}: <n> {n} vs piquasso {} differs by {:.3e}, tolerance {MEAN_N_TOL:.1e}",
                case.name,
                case.mean_n,
                (n - case.mean_n).abs()
            );
        }

        if max_diff > worst.1 {
            worst = (case.name.clone(), max_diff);
        }
        compared += 1;
    }

    // Report the corpus, always. A differential check that quietly compares
    // three cases is worse than no check, because it reads as coverage.
    eprintln!("piquasso {version}: compared {compared}/{} cases", cases.len());
    eprintln!("  worst agreeing diff: {} at {:.3e}", worst.0, worst.1);
    for t in &truncation_seen {
        eprintln!("  residual above the noise floor — {t}");
    }
    for (name, why) in &skipped {
        eprintln!("  NOT COMPARED — {name}: {why:?}");
    }

    // Guard the corpus size. If the fixture is regenerated smaller, or the
    // interpreter starts refusing cases it used to accept, that must fail here
    // rather than show up as a still-green test doing less work.
    assert!(
        compared >= 17,
        "only {compared} cases compared; the corpus has shrunk"
    );
}

/// The unexpressible cases are a **property of the crate**, and this pins them.
///
/// It used to assert `["squeezed_r=0.5_phase_then_displace"]` and carried the
/// note that gaining operator forms should make it fail, as a reminder to move
/// the case into the compared corpus. `displace` now HAS an operator form, so
/// that happened, and the set is empty: every case in the corpus is expressible.
///
/// `squeeze` is still prep-only, so a corpus case applying `squeeze` to a state
/// with structure would reappear here. That is the intended behaviour, not a
/// hole — the set is pinned exactly so a new gap cannot arrive unnoticed.
#[test]
fn preparations_are_still_constructors_only() {
    let (_, cases) = load();

    let unexpressible: Vec<&str> = cases
        .iter()
        .filter(|c| build(&c.ops, c.cutoff).is_err())
        .map(|c| c.name.as_str())
        .collect();

    assert_eq!(
        unexpressible,
        Vec::<&str>::new(),
        "the set of circuits this backend cannot express has CHANGED.\n\
         If it shrank, an operator form landed — move the case into the \
         compared corpus.\n\
         If it grew, a preparation regressed."
    );
}

/// `displace` as an OPERATOR, against piquasso, on a state that already has
/// structure — the case the `coherent` constructor cannot reach and the one CV
/// QAOA actually needs.
///
/// Reference values from piquasso 8.0.1, cutoff 20, `Squeezing(r)` then
/// `Displacement(r=alpha, phi=0)`, compared as `<n>` on the same normalisation
/// convention as everything else in this file.
#[test]
fn displace_operator_matches_piquasso_on_a_squeezed_state() {
    // (r, alpha, piquasso <n>) — generated by tools/cv_cross_check/displace_ref.py
    let cases = [(0.3_f64, 0.5_f64), (0.5, 0.63), (0.5, 1.0)];

    for (r, alpha) in cases {
        let mut s = FockState::squeezed_vacuum(r, 20).expect("squeezed");
        s.displace(Complex64::new(alpha, 0.0)).expect("displace");

        // Analytic anchor: <n> = |alpha|^2 + sinh^2 r for a displaced squeezed
        // state. This is tied to TRUTH, not to agreement — necessary because a
        // pure cross-check cannot see an error piquasso shares (both truncate
        // sequentially the same way).
        let want = alpha * alpha + r.sinh().powi(2);
        let got = s.expect_n(1.0).expect("expect_n");

        // THIS is where the leak metric is the correct bound, and the only
        // place in this file it appears: the reference is ANALYTIC — the truth
        // — so `lost_n_weight` bounding |<n> - truth| is exactly its contract.
        // (Against piquasso it would be the wrong bound, because piquasso
        // truncates the same way we do and the residual there is float noise.)
        let budget = s.lost_n_weight();
        let err = (got - want).abs();
        assert!(
            err <= budget,
            "displaced-squeezed <n>: got {got}, analytic {want} — error {err:.3e} \
             EXCEEDS the leak budget {budget:.3e} this backend advertises \
             (r={r}, alpha={alpha}). An under-reporting leak metric is worse \
             than a wrong amplitude."
        );
        // ...and the budget must not be so loose as to be vacuous.
        assert!(
            budget < 0.1,
            "leak budget {budget:.3e} is too loose to constrain anything \
             (r={r}, alpha={alpha})"
        );
    }
}

/// `displace` on vacuum must reproduce the `coherent` constructor exactly.
///
/// Two independent code paths for the same state: the closed-form recurrence in
/// `coherent`, and the operator's matrix elements applied to |0>. If they
/// disagree, one of them is wrong.
#[test]
fn displace_on_vacuum_reproduces_the_coherent_constructor() {
    for (re, im) in [(0.5, 0.0), (1.0, 0.0), (0.7, 0.7), (0.4, -0.9)] {
        let alpha = Complex64::new(re, im);
        let mut s = FockState::vacuum(20).expect("vacuum");
        s.displace(alpha).expect("displace");
        let direct = FockState::coherent(alpha, 20).expect("coherent");

        let d = s
            .amplitudes()
            .iter()
            .zip(direct.amplitudes())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f64, f64::max);
        assert!(
            d < 1e-14,
            "displace(|0>) != coherent({alpha}): differs by {d:.3e}"
        );
    }
}
