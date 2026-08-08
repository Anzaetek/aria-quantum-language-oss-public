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
/// The value is set from measurement, not taste. See `TRUNCATION_CASES`: the
/// cases where the two disagree by more than this are exactly the ones where
/// **the disagreement is real physics**, not numerics, and they are listed
/// individually rather than absorbed by loosening this number.
const PROB_TOL: f64 = 1e-9;

/// Same reasoning, on amplitudes. Looser than `PROB_TOL` by roughly the square
/// root, because an amplitude error of `e` shows up in a probability as `~2*e`
/// near unit magnitude but as `e^2` where the amplitude itself is small.
const AMP_TOL: f64 = 1e-8;

/// Where the two implementations genuinely differ, and the claim made about it.
///
/// We evaluate the closed-form Fock amplitudes and then cut. piquasso applies a
/// squeezing operator that is itself already truncated. **Those are not the same
/// state**, and the gap grows with `r` as the cutoff bites — at `r=0.5` it is
/// 3.4e-8, far above any floating-point noise floor. Neither side is "wrong";
/// they are two defensible readings of "squeezed vacuum at cutoff 20".
///
/// So the assertion is not "these agree to 1e-9". It is the stronger and more
/// useful claim:
///
/// > wherever we disagree with piquasso, the disagreement is bounded by the
/// > truncation error **this backend itself advertises** via `lost_norm`.
///
/// That turns an awkward mismatch into a live test of the leak metric. If the
/// gap ever exceeds the budget, the metric is understating the error — and a
/// leak metric that under-reports is worse than none, because callers trust it
/// to decide whether an answer is usable. Bounding it per-case also means no
/// hand-maintained exception list to go stale.
///
/// `lost_norm` is the right unit here: these are probabilities. `lost_n_weight`
/// is photon-weighted and bounds `<n>`, which is checked separately below.
fn probability_budget(state: &FockState) -> f64 {
    state.lost_norm().max(PROB_TOL)
}

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

        if is_prep && !pristine_vacuum {
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
                state = Some(
                    FockState::coherent(Complex64::new(re, im), cutoff).expect("coherent"),
                );
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

        let budget = probability_budget(&state);
        assert!(
            amp_diff <= budget.max(AMP_TOL),
            "{}: Fock AMPLITUDES disagree with piquasso {version} by \
             {amp_diff:.3e} (budget {budget:.3e}) after removing global phase. \
             A probability-only comparison would have missed this — which is \
             exactly why amplitudes are compared.",
            case.name
        );

        assert!(
            max_diff <= budget,
            "{}: Fock probabilities disagree with piquasso {version} by \
             {max_diff:.3e}, EXCEEDING the {budget:.3e} truncation error this \
             backend advertises via lost_norm. Either the numerics are wrong or \
             the leak metric is understating — and an understating leak metric \
             is the worse of the two, because callers trust it to decide whether \
             an answer is usable.\n  ours: {:?}\n  piq:  {:?}",
            case.name,
            &ours[..ours.len().min(6)],
            &case.probs[..case.probs.len().min(6)]
        );

        // Record cases that needed more than the noise floor, so the report
        // shows WHERE truncation is doing the reconciling rather than leaving
        // "all green" to imply exact agreement everywhere.
        if max_diff > PROB_TOL {
            truncation_seen.push(format!(
                "{} (diff {max_diff:.3e} <= budget {budget:.3e})",
                case.name
            ));
        }

        // <n> as a secondary check — a scalar, and scalars collide, so the
        // vector above is what actually discriminates. Bounded by the
        // PHOTON-WEIGHTED leak, which is the quantity that governs <n>.
        if let Ok(n) = state.expect_n(1.0) {
            let n_budget = state.lost_n_weight().max(1e-9);
            assert!(
                (n - case.mean_n).abs() <= n_budget,
                "{}: <n> {n} vs piquasso {} differs by {:.3e}, exceeding the \
                 photon-weighted leak budget {n_budget:.3e}",
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
        eprintln!("  reconciled by truncation budget — {t}");
    }
    for (name, why) in &skipped {
        eprintln!("  NOT COMPARED — {name}: {why:?}");
    }

    // Guard the corpus size. If the fixture is regenerated smaller, or the
    // interpreter starts refusing cases it used to accept, that must fail here
    // rather than show up as a still-green test doing less work.
    assert!(
        compared >= 16,
        "only {compared} cases compared; the corpus has shrunk"
    );
}

/// The unexpressible cases are a **property of the crate**, and this pins them.
///
/// When `displace`/`squeeze` gain operator forms, this test fails — which is the
/// intent. It is the reminder to move those cases into the compared set instead
/// of leaving a gap that has quietly stopped being a gap.
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
        vec!["squeezed_r=0.5_phase_then_displace"],
        "the set of circuits this backend cannot express has CHANGED.\n\
         If it shrank, an operator form landed — move the case into the \
         compared corpus.\n\
         If it grew, a preparation regressed."
    );
}
