// SPDX-License-Identifier: Apache-2.0
//! Differential cross-check of the LOSS channel against **piquasso** — two
//! different representations of the same CP map.
//!
//! piquasso applies `Attenuator` natively on a DENSITY MATRIX
//! (`FockSimulator`; its pure simulator refuses the instruction, measured).
//! This crate has no density matrices: [`MultiFockState::loss`] dilates —
//! one vacuum ancilla per loss, a beamsplitter at `θ = acos(√η)`, and the
//! channel output is the MARGINAL over the original modes. If the two agree
//! it is because Stinespring dilation and the operator-sum form really are
//! the same channel, not because they share code, conventions, or even a
//! state representation.
//!
//! The comparison surface is therefore `marginal_probs` (occupation-keyed,
//! system modes only — the ancillas are this side's business) and per-mode
//! `expect_n`, whose oracle column comes from piquasso's own `reduced()`
//! trace-out.

use omega_backend_cv::multimode::MultiFockState;
use serde_json::Value;

const BUDGET: Option<usize> = Some(256 << 20);
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
    let raw = include_str!("../../../tools/cv_cross_check/piquasso_loss_fixture.jsonl");
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
                .expect("mean_n")
                .iter()
                .map(|x| x.as_f64().expect("mean_n entry"))
                .collect(),
        });
    }
    assert!(!cases.is_empty(), "the loss fixture is empty");
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
            "loss" => st
                .loss(
                    op["mode"].as_u64().expect("mode") as usize,
                    op["eta"].as_f64().expect("eta"),
                    BUDGET,
                )
                .expect("loss"),
            other => panic!("fixture uses an op this harness cannot express: {other}"),
        }
    }
    st
}

#[test]
fn loss_marginals_agree_with_piquasso_density_matrices() {
    let (meta, cases) = load();
    let mut worst = 0.0f64;
    let mut worst_case = String::new();
    let mut saw_entangled_arm = false;
    let mut saw_channel_then_unitary = false;

    for c in &cases {
        saw_entangled_arm |= c.name == "loss_after_hom";
        saw_channel_then_unitary |= c.name == "loss_then_bs";
        let st = build(c);
        assert!(
            st.n_modes() > c.n_modes,
            "{}: a loss case must have dilated at least once",
            c.name
        );
        let keep: Vec<usize> = (0..c.n_modes).collect();
        let ours = st.marginal_probs(&keep).expect("marginal");
        for (occ, want) in &c.probs {
            let got = ours
                .iter()
                .find(|(k, _)| k == occ)
                .map(|(_, p)| *p)
                .unwrap_or(0.0);
            let d = (got - want).abs();
            if d > worst {
                worst = d;
                worst_case = format!("{} at {occ:?}", c.name);
            }
            assert!(
                d <= TOL,
                "{}: P{occ:?} = {got:.15} but piquasso's density matrix says \
                 {want:.15} (delta {d:.3e})",
                c.name
            );
        }
        // And nothing on our side that piquasso does not have: the marginal
        // must not invent probability mass at unlisted occupations.
        let listed: f64 = c.probs.iter().map(|(_, p)| p).sum();
        let ours_total: f64 = ours.iter().map(|(_, p)| p).sum();
        assert!(
            (ours_total - listed).abs() <= 1e-9,
            "{}: our marginal carries {ours_total:.12}, fixture lists {listed:.12}",
            c.name
        );

        for (mode, want) in c.mean_n.iter().enumerate() {
            let got = st.expect_n(mode).expect("expect_n");
            assert!(
                (got - want).abs() <= TOL,
                "{}: <n_{mode}> = {got:.15}, piquasso reduced says {want:.15}",
                c.name
            );
        }
    }

    assert!(saw_entangled_arm, "corpus lost loss_after_hom");
    assert!(saw_channel_then_unitary, "corpus lost loss_then_bs");
    eprintln!(
        "  piquasso {meta}: loss marginals on {} cases; worst |delta| {worst:.3e} ({worst_case})",
        cases.len()
    );
}
