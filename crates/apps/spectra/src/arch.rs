// SPDX-License-Identifier: Apache-2.0
//! arch_search — quantum architecture search over coupling graphs
//! (the first increment of the QML_ROADMAP long-term follow-up).
//!
//! Given ONLY the labelled Heisenberg-substrate dataset (phases → ±1),
//! the search must discover the generator's interaction structure
//! independently: candidate architectures are Trotterised Heisenberg
//! ansätze over different coupling GRAPHS on 7 sites — chain, ring,
//! star, stride-2, disconnected pairs — each with per-bond trainable
//! couplings. Every candidate trains the same way (adjoint gradients +
//! Adam on the bond-correlator readout with a closed-form affine head)
//! and is scored on a held-out half. Ground truth: the data came from
//! the CHAIN graph, so the chain must win, and its learned couplings
//! must approximate the true disorder draw.
//!
//! The candidate circuits are built programmatically; the builder is
//! cross-checked against the lowered `spectra_heisenberg.aria` on a
//! probe row (|Δ| ≤ 1e-9) so the search space is tied to the shipped,
//! oracle-verified example. The winning architecture is printed as
//! Aria source.

use std::collections::{HashMap, HashSet};

use aria_verify_core::Observable;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit, SymbolId};
use omega_core::executor::Backend;
use omega_core::gradient::{compute_gradient_for, GradMethod};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

pub const N_SITES: usize = 7;

/// A candidate architecture: a name and its coupling graph.
pub struct Candidate {
    pub name: &'static str,
    pub bonds: Vec<(u32, u32)>,
}

/// The search space: five coupling graphs on 7 sites. The generator's
/// graph (chain) is one of them; the search is NOT told which.
pub fn candidates() -> Vec<Candidate> {
    vec![
        Candidate {
            name: "chain",
            bonds: (0..6).map(|k| (k, k + 1)).collect(),
        },
        Candidate {
            name: "ring",
            bonds: (0..6).map(|k| (k, k + 1)).chain([(0u32, 6u32)]).collect(),
        },
        Candidate {
            name: "star",
            bonds: (1..7).map(|k| (0, k)).collect(),
        },
        Candidate {
            name: "stride2",
            bonds: (0..5).map(|k| (k, k + 2)).collect(),
        },
        Candidate {
            name: "pairs",
            bonds: vec![(0, 1), (2, 3), (4, 5)],
        },
    ]
}

/// Build the Trotterised Heisenberg ansatz for an arbitrary coupling
/// graph — the same per-slice decomposition as spectra_heisenberg.aria:
/// per bond XX, YY, ZZ blocks with angle 2·(J_bond·dt), then per-site
/// field RZ(2·φ_i·dt). Symbols: 0..bonds = J·dt, then N_SITES φ·dt.
pub fn build_ir(bonds: &[(u32, u32)], steps: usize) -> (CircuitIR, Vec<u32>, Vec<u32>) {
    build_ir_sized(bonds, steps, N_SITES)
}

/// `build_ir` for an arbitrary site count (the scaling harness runs the
/// same chain at n = 7 … 13 sites).
pub fn build_ir_sized(
    bonds: &[(u32, u32)],
    steps: usize,
    n_sites: usize,
) -> (CircuitIR, Vec<u32>, Vec<u32>) {
    let jt_ids: Vec<u32> = (0..bonds.len() as u32).collect();
    let pt_ids: Vec<u32> = (0..n_sites as u32)
        .map(|i| bonds.len() as u32 + i)
        .collect();
    let mut c = CircuitIR::new(n_sites as u32, CircuitType::GateBased);
    for &id in jt_ids.iter().chain(pt_ids.iter()) {
        c.symbols.insert(id, format!("s{id}"));
    }
    let g1 = |gate, q: u32| GateOp {
        gate,
        qubits: smallvec![Qubit(q)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    };
    let rot = |gate, q: u32, p: ParamExpr| GateOp {
        gate,
        qubits: smallvec![Qubit(q)],
        params: smallvec![p],
        classical_bit: None,
        condition: None,
    };
    let cx = |a: u32, b: u32| GateOp {
        gate: GateKind::CX,
        qubits: smallvec![Qubit(a), Qubit(b)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    };
    let two_sym = |id: u32| {
        ParamExpr::Mul(
            Box::new(ParamExpr::Concrete(2.0)),
            Box::new(ParamExpr::Symbol(id)),
        )
    };
    let pi_2 = std::f64::consts::FRAC_PI_2;
    // |+⟩^⊗n
    for q in 0..n_sites as u32 {
        c.add_op(g1(GateKind::H, q));
    }
    for _ in 0..steps {
        for (k, &(a, b)) in bonds.iter().enumerate() {
            let jt = jt_ids[k];
            // exp(-i·J·dt·X⊗X): H-conjugated ZZ block.
            c.add_op(g1(GateKind::H, a));
            c.add_op(g1(GateKind::H, b));
            c.add_op(cx(a, b));
            c.add_op(rot(GateKind::Rz, b, two_sym(jt)));
            c.add_op(cx(a, b));
            c.add_op(g1(GateKind::H, a));
            c.add_op(g1(GateKind::H, b));
            // exp(-i·J·dt·Y⊗Y): RX(π/2)-conjugated ZZ block.
            c.add_op(rot(GateKind::Rx, a, ParamExpr::Concrete(pi_2)));
            c.add_op(rot(GateKind::Rx, b, ParamExpr::Concrete(pi_2)));
            c.add_op(cx(a, b));
            c.add_op(rot(GateKind::Rz, b, two_sym(jt)));
            c.add_op(cx(a, b));
            c.add_op(rot(GateKind::Rx, a, ParamExpr::Concrete(-pi_2)));
            c.add_op(rot(GateKind::Rx, b, ParamExpr::Concrete(-pi_2)));
            // exp(-i·J·dt·Z⊗Z).
            c.add_op(cx(a, b));
            c.add_op(rot(GateKind::Rz, b, two_sym(jt)));
            c.add_op(cx(a, b));
        }
        for i in 0..n_sites as u32 {
            c.add_op(rot(GateKind::Rz, i, two_sym(pt_ids[i as usize])));
        }
    }
    (c, jt_ids, pt_ids)
}

/// One trained candidate: holdout AUC + learned couplings.
pub struct TrainedCandidate {
    pub name: &'static str,
    pub auc: f64,
    pub couplings: Vec<f64>,
    pub bonds: Vec<(u32, u32)>,
}

/// Train a candidate exactly the way the DMQ lane trains (affine head
/// refit + adjoint gradients + Adam over the bond couplings) and score
/// it on the holdout split.
#[allow(clippy::too_many_arguments)]
pub fn train_candidate(
    cand: &Candidate,
    backend: &dyn Backend,
    steps: usize,
    dt: f64,
    correlator: &Observable,
    train_x: &[Vec<f64>],
    train_y: &[f64],
    test_x: &[Vec<f64>],
    test_y: &[f64],
    epochs: usize,
) -> Result<TrainedCandidate, String> {
    let (ir, jt_ids, pt_ids) = build_ir(&cand.bonds, steps);
    let mut couplings = vec![1.0; cand.bonds.len()];
    let mut head = vec![0.0, 0.0];
    let trainable: HashSet<SymbolId> = jt_ids.iter().copied().collect();
    let raw = |couplings: &[f64], phases: &[f64]| -> Result<f64, String> {
        let mut b = ParameterBinding::new();
        for (&id, &j) in jt_ids.iter().zip(couplings) {
            b.bind(id, j * dt);
        }
        for (&id, &p) in pt_ids.iter().zip(phases) {
            b.bind(id, p * dt);
        }
        backend
            .expectation(&ir, &b, correlator)
            .map_err(|e| e.to_string())
    };
    let (mut am, mut av): (HashMap<u32, f64>, HashMap<u32, f64>) = (HashMap::new(), HashMap::new());
    for epoch in 0..epochs {
        let raw_vals: Vec<f64> = train_x
            .iter()
            .map(|p| raw(&couplings, p))
            .collect::<Result<_, _>>()?;
        let feats: Vec<Vec<f64>> = raw_vals.iter().map(|&v| vec![v]).collect();
        head = aria_verify_core::data::ridge_regression(&feats, train_y, 1e-6)?;
        let a = head[0];
        let mut acc: HashMap<u32, f64> = HashMap::new();
        for ((p, &yi), &rv) in train_x.iter().zip(train_y).zip(&raw_vals) {
            let r = aria_verify_core::data::ridge_predict(&head, &[rv]) - yi;
            let mut b = ParameterBinding::new();
            for (&id, &j) in jt_ids.iter().zip(&couplings) {
                b.bind(id, j * dt);
            }
            for (&id, &ph) in pt_ids.iter().zip(p.iter()) {
                b.bind(id, ph * dt);
            }
            let grads = compute_gradient_for(
                backend,
                &ir,
                &b,
                correlator,
                &GradMethod::Adjoint,
                Some(&trainable),
            )
            .map_err(|e| e.to_string())?;
            for (id, g) in grads {
                *acc.entry(id).or_insert(0.0) += 2.0 * r * a * dt * g / train_x.len() as f64;
            }
        }
        let (b1, b2, eps) = (0.9, 0.999, 1e-8);
        let t = (epoch + 1) as f64;
        for (k, &id) in jt_ids.iter().enumerate() {
            let g = acc.get(&id).copied().unwrap_or(0.0);
            let mi = am.entry(id).or_insert(0.0);
            *mi = b1 * *mi + (1.0 - b1) * g;
            let vi = av.entry(id).or_insert(0.0);
            *vi = b2 * *vi + (1.0 - b2) * g * g;
            let m_hat = *mi / (1.0 - b1.powf(t));
            let v_hat = *vi / (1.0 - b2.powf(t));
            couplings[k] -= 0.05 * m_hat / (v_hat.sqrt() + eps);
        }
    }
    // Holdout scores through the final head.
    let raw_vals: Vec<f64> = train_x
        .iter()
        .map(|p| raw(&couplings, p))
        .collect::<Result<_, _>>()?;
    let feats: Vec<Vec<f64>> = raw_vals.iter().map(|&v| vec![v]).collect();
    head = aria_verify_core::data::ridge_regression(&feats, train_y, 1e-6)?;
    let mut scores = Vec::with_capacity(test_x.len());
    for p in test_x {
        scores.push(aria_verify_core::data::ridge_predict(
            &head,
            &[raw(&couplings, p)?],
        ));
    }
    Ok(TrainedCandidate {
        name: cand.name,
        auc: crate::lanes::auc(&scores, test_y),
        couplings,
        bonds: cand.bonds.clone(),
    })
}

/// Render the winning architecture as Aria source (the deliverable a
/// user would paste into examples/ — one bond block per line group).
pub fn to_aria_source(winner: &TrainedCandidate, steps: usize) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "-- discovered by arch_search: '{}' coupling graph, holdout AUC {:.3}\n",
        winner.name, winner.auc
    ));
    s.push_str(&format!(
        "circuit Discovered(steps: int) {{\n    qreg q[{N_SITES}]\n\n    let jt = symbolic[{}]\n    let pt = symbolic[{N_SITES}]\n\n",
        winner.bonds.len()
    ));
    s.push_str("    repeat i from 0 to 6 {\n        apply H on q[i]\n    }\n\n");
    s.push_str(&format!(
        "    repeat s from 0 to steps - 1 {{  -- {} Trotter slices\n",
        steps
    ));
    for (k, (a, b)) in winner.bonds.iter().enumerate() {
        s.push_str(&format!(
            "        -- bond {k}: sites ({a}, {b}), learned J = {:.3}\n",
            winner.couplings[k]
        ));
        s.push_str(&format!(
            "        apply H on q[{a}]\n        apply H on q[{b}]\n        apply CX on q[{a}], q[{b}]\n        apply RZ(2 * jt[{k}]) on q[{b}]\n        apply CX on q[{a}], q[{b}]\n        apply H on q[{a}]\n        apply H on q[{b}]\n"
        ));
        s.push_str(&format!(
            "        apply RX(pi / 2) on q[{a}]\n        apply RX(pi / 2) on q[{b}]\n        apply CX on q[{a}], q[{b}]\n        apply RZ(2 * jt[{k}]) on q[{b}]\n        apply CX on q[{a}], q[{b}]\n        apply RX(-pi / 2) on q[{a}]\n        apply RX(-pi / 2) on q[{b}]\n"
        ));
        s.push_str(&format!(
            "        apply CX on q[{a}], q[{b}]\n        apply RZ(2 * jt[{k}]) on q[{b}]\n        apply CX on q[{a}], q[{b}]\n"
        ));
    }
    s.push_str("        repeat i from 0 to 6 {\n            apply RZ(2 * pt[i]) on q[i]\n        }\n    }\n}\n");
    s
}
