// SPDX-License-Identifier: Apache-2.0
//! Differential cross-check of the MULTI-MODE CV backend against **piquasso**.
//!
//! The single-mode corpus (`piquasso_xcheck.rs`) cannot reach any of this. Every
//! operation it exercises acts on one mode, so nothing in it could have caught a
//! wrong beamsplitter — and a beamsplitter is the only operation in the crate
//! that couples modes at all.
//!
//! One convention in particular was untestable before now: **the beamsplitter
//! phase**. A phase is invisible in single-mode Fock probabilities, so a
//! sign error in `φ` produces identical output everywhere in the old corpus. It
//! is observable here only through interference between two modes.
//!
//! # Keyed by occupation, never by index
//!
//! The fixture stores probabilities keyed by `"n0,n1,..."`, and this file looks
//! them up the same way. Neither side ever states a flat index ordering, which
//! removes the convention most likely to differ silently between two
//! implementations. A disagreement about ordering would otherwise surface as a
//! physics-looking mismatch and send someone hunting a numerical bug.
//!
//! # Two truncations, and they are NOT the same truncation
//!
//! piquasso truncates by **total** photon number (`Σ n_i < cutoff`); this crate
//! truncates **per mode** (`n_i < cutoff` for each `i`). So the two bases differ,
//! and states this crate can hold — `|3,3⟩` at cutoff 4 — piquasso does not
//! represent at all.
//!
//! Comparing only over the occupations piquasso emits is therefore the correct
//! intersection, not a convenience: it is every state both bases contain. The
//! corpus is chosen so total photon number stays under the cutoff throughout, so
//! the intersection is the whole physical answer rather than a corner of it —
//! and `every_case_conserves_probability_within_the_shared_basis` checks that
//! the compared mass really does sum to one, which is what would fail if a case
//! were quietly leaking into the non-shared region.

use omega_backend_cv::multimode::MultiFockState;
use serde_json::Value;

const BUDGET: Option<usize> = Some(256 << 20);
/// Loose enough for f64 accumulation over a few operations, tight enough that a
/// real convention error cannot hide: a phase-sign error moves probabilities by
/// O(1), not by 1e-12.
const TOL: f64 = 1e-12;

struct Case {
    name: String,
    cutoff: usize,
    n_modes: usize,
    prep: Vec<usize>,
    ops: Vec<Value>,
    probs: Vec<(Vec<usize>, f64)>,
    mean_n: Vec<f64>,
}

fn load() -> (String, Vec<Case>) {
    let raw = include_str!("../../../tools/cv_cross_check/piquasso_multimode_fixture.jsonl");
    let mut meta = String::new();
    let mut cases = Vec::new();
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).expect("fixture line is JSON");
        if let Some(m) = v.get("meta") {
            meta = m.to_string();
            continue;
        }
        let probs = v["probs"]
            .as_object()
            .expect("probs object")
            .iter()
            .map(|(k, p)| {
                let occ = k
                    .split(',')
                    .map(|x| x.parse::<usize>().expect("occupation digit"))
                    .collect::<Vec<_>>();
                (occ, p.as_f64().expect("probability"))
            })
            .collect();
        cases.push(Case {
            name: v["case"].as_str().expect("case").to_string(),
            cutoff: v["cutoff"].as_u64().expect("cutoff") as usize,
            n_modes: v["n_modes"].as_u64().expect("n_modes") as usize,
            prep: v["prep"]
                .as_array()
                .expect("prep")
                .iter()
                .map(|x| x.as_u64().expect("prep digit") as usize)
                .collect(),
            ops: v["ops"].as_array().expect("ops").clone(),
            probs,
            mean_n: v["mean_n"]
                .as_array()
                .expect("mean_n (regenerate the fixture: piquasso_multimode_ref.py)")
                .iter()
                .map(|x| x.as_f64().expect("mean_n entry"))
                .collect(),
        });
    }
    assert!(!cases.is_empty(), "the multimode fixture is empty");
    (meta, cases)
}

fn build(c: &Case) -> MultiFockState {
    let mut st =
        MultiFockState::number_state(c.cutoff, c.n_modes, &c.prep, BUDGET).expect("number_state");
    for op in &c.ops {
        match op["op"].as_str().expect("op kind") {
            "beamsplitter" => st
                .beamsplitter(
                    op["a"].as_u64().expect("a") as usize,
                    op["b"].as_u64().expect("b") as usize,
                    op["theta"].as_f64().expect("theta"),
                    op["phi"].as_f64().expect("phi"),
                )
                .expect("beamsplitter"),
            "squeezing2" => st
                .squeezing2(
                    op["a"].as_u64().expect("a") as usize,
                    op["b"].as_u64().expect("b") as usize,
                    op["r"].as_f64().expect("r"),
                    op["phi"].as_f64().expect("phi"),
                )
                .expect("squeezing2"),
            "phase_shift" => {
                let phi = op["phi"].as_f64().expect("phi");
                st.apply_single_mode(op["mode"].as_u64().expect("mode") as usize, |fin, fout| {
                    for (n, (a, b)) in fin.iter().zip(fout.iter_mut()).enumerate() {
                        *b = a * num_complex::Complex64::from_polar(1.0, phi * n as f64);
                    }
                })
                .expect("phase_shift")
            }
            other => panic!("fixture uses an op this harness cannot express: {other}"),
        }
    }
    st
}

#[test]
fn multimode_cv_agrees_with_piquasso() {
    let (meta, cases) = load();
    let mut worst = 0.0f64;
    let mut worst_case = String::new();
    let mut failures = Vec::new();

    for c in &cases {
        let st = build(c);
        for (occ, want) in &c.probs {
            let idx = st
                .index_of(occ)
                .unwrap_or_else(|| panic!("{}: occupation {occ:?} out of range", c.name));
            let got = st.amplitudes()[idx].norm_sqr();
            let d = (got - want).abs();
            if d > worst {
                worst = d;
                worst_case = format!("{} at {occ:?}", c.name);
            }
            if d > TOL {
                failures.push(format!(
                    "{}: P{occ:?} = {got:.15} but piquasso says {want:.15} (delta {d:.3e})",
                    c.name
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} disagreement(s) with piquasso {meta}:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
    eprintln!(
        "  piquasso {meta}: compared {} multi-mode cases; worst |dP| {worst:.3e} ({worst_case})",
        cases.len()
    );
}

/// `expect_n(mode)` on ENTANGLED states, against piquasso's density-matrix
/// trace-out.
///
/// Before this test, `expect_n` was exercised only on product states, where a
/// per-mode marginal cannot go wrong in an interesting way: each mode's ladder
/// is its own closed system. After a beamsplitter the occupation of one mode is
/// correlated with another's, and a marginal that walks the joint amplitudes
/// with the wrong stride would return a number that LOOKS plausible (finite,
/// positive, under the cutoff) while belonging to no mode at all.
///
/// The oracle column (`mean_n`) is computed on the piquasso side from
/// `state.reduced(modes=(i,))` — their reduction machinery, not their
/// probability map — so the two sides share only the definition of a mean.
///
/// Non-vacuity is pinned: the corpus must contain the HOM output state
/// `(|2,0⟩ + |0,2⟩)/√2`, which is entangled, and a three-mode case, whose
/// strides differ from every two-mode one.
#[test]
fn expect_n_per_mode_agrees_with_piquasso_on_entangled_states() {
    let (meta, cases) = load();
    let mut worst = 0.0f64;
    let mut worst_case = String::new();
    let mut saw_hom = false;
    let mut saw_three_mode = false;

    for c in &cases {
        assert_eq!(
            c.mean_n.len(),
            c.n_modes,
            "{}: fixture mean_n arity is wrong",
            c.name
        );
        saw_hom |= c.name == "hom_50_50";
        saw_three_mode |= c.n_modes >= 3;
        let st = build(c);
        for (mode, want) in c.mean_n.iter().enumerate() {
            let got = st.expect_n(mode).expect("expect_n");
            let d = (got - want).abs();
            if d > worst {
                worst = d;
                worst_case = format!("{} mode {mode}", c.name);
            }
            assert!(
                d <= TOL,
                "{}: <n_{mode}> = {got:.15} but piquasso's reduced state says \
                 {want:.15} (delta {d:.3e})",
                c.name
            );
        }
    }

    assert!(saw_hom, "corpus lost its entangled anchor (hom_50_50)");
    assert!(saw_three_mode, "corpus lost its three-mode stride case");
    eprintln!(
        "  piquasso {meta}: <n_i> compared on {} cases; worst |delta| {worst:.3e} ({worst_case})",
        cases.len()
    );
}

/// The compared mass sums to one.
///
/// The two sides truncate DIFFERENTLY — piquasso by total photon number, this
/// crate per mode — so the comparison runs over their intersection. That is only
/// the whole answer if no probability escapes it. If a case ever leaked into the
/// non-shared region, the per-occupation comparison above could still pass
/// while both engines quietly held a state neither was being asked about.
#[test]
fn every_case_conserves_probability_within_the_shared_basis() {
    let (_, cases) = load();
    for c in &cases {
        let total: f64 = c.probs.iter().map(|(_, p)| p).sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "{}: the compared occupations carry {total:.12}, not 1.0 — this case \
             leaks outside the shared basis, so the comparison is over a corner \
             of the state rather than all of it",
            c.name
        );
        let st = build(c);
        let ours: f64 = c
            .probs
            .iter()
            .map(|(occ, _)| st.amplitudes()[st.index_of(occ).expect("in range")].norm_sqr())
            .sum();
        assert!(
            (ours - 1.0).abs() < 1e-9,
            "{}: our state carries {ours:.12} over the compared occupations",
            c.name
        );
    }
}

/// Hong-Ou-Mandel, called out by name.
///
/// It is already one row of the corpus above, but a bulk comparison reports a
/// worst-case number and this asserts the specific physical fact — so a failure
/// says "the HOM dip is gone" rather than "case 1 of 19 moved by 0.5".
#[test]
fn the_corpus_contains_a_hong_ou_mandel_dip_and_it_is_zero() {
    let (_, cases) = load();
    let hom = cases
        .iter()
        .find(|c| c.name == "hom_50_50")
        .expect("the corpus must contain hom_50_50");
    assert!(
        !hom.probs
            .iter()
            .any(|(occ, p)| occ == &vec![1, 1] && *p > 1e-12),
        "piquasso reports non-zero P(1,1) for a 50:50 beamsplitter — the \
         fixture itself is wrong, not our backend"
    );
    let st = build(hom);
    let p11 = st.amplitudes()[st.index_of(&[1, 1]).expect("in range")].norm_sqr();
    assert!(p11 < 1e-24, "our P(1,1) = {p11:.3e}, expected 0");
}
