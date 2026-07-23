// SPDX-License-Identifier: Apache-2.0
//! arch_evolve — evolutionary search over layered RBS connectivity
//! masks (the butterfly-mask search space of QML_ROADMAP).
//!
//! Search space: genomes of `LAYERS` mask choices over 8 qubits from a
//! 5-entry menu (none / even pairs / odd pairs / stride-2 / stride-4).
//! The hand-built butterfly of `butterfly_qnn.aria` is the genome
//! [stride-2, even-pairs, stride-4] — the search is NOT seeded with it;
//! it must find something at least as good on held-out data.
//!
//! Task: the same open-data problem the butterfly app solves — impute
//! the squashed `thalach` column of UCI Cleveland heart from 8 encoded
//! features (RY(x_i) angle encoding, ⟨Z⟩ features + closed-form ridge
//! head, RBS angles trained by adjoint + Adam against H_w).
//!
//! Loop: seeded random population, elitist truncation selection,
//! per-layer point mutations — fully deterministic from the printed
//! seed, with genome→fitness memoisation (a re-evaluated genome must
//! reproduce its fitness bit-for-bit, which the harness asserts).

use std::collections::{HashMap, HashSet};

use aria_verify_core::data::{self, SplitMix64};
use aria_verify_core::Observable;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit, SymbolId};
use omega_core::executor::{Backend, PauliOp};
use omega_core::gradient::{compute_gradient_for, GradMethod};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

pub const N_QUBITS: usize = 8;
pub const LAYERS: usize = 3;

/// The mask menu. Index 0 must stay NONE (the ablation genome). The
/// last entry, `chain-full`, has OVERLAPPING pairs — a layer of it is
/// NOT a commuting block, so a genome ending in it loses the
/// one-execution parallel parameter-shift property. The trainability
/// regularizer prices that in.
pub fn mask_menu() -> Vec<(&'static str, Vec<(u32, u32)>)> {
    vec![
        ("none", vec![]),
        ("pairs-even", vec![(0, 1), (2, 3), (4, 5), (6, 7)]),
        ("pairs-odd", vec![(1, 2), (3, 4), (5, 6)]),
        ("stride-2", vec![(0, 2), (1, 3), (4, 6), (5, 7)]),
        ("stride-4", vec![(0, 4), (1, 5), (2, 6), (3, 7)]),
        (
            "chain-full",
            vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
        ),
    ]
}

/// Hardware-trainability score of a genome: the fraction of its
/// trainable angles whose gradients come from the batched trailing
/// commuting block — measured by the ENGINE's own block detector
/// (`omega_core::parallel_shift`), not re-derived here. 1.0 = every
/// gradient in one execution (the arXiv:2606.03517 property); genomes
/// ending in overlapping masks score lower and pay a fitness penalty.
pub fn trainability(g: &Genome, backend: &dyn Backend, probe_x: &[f64]) -> Result<f64, String> {
    let (c, theta_ids) = build(g, probe_x);
    if theta_ids.is_empty() {
        return Ok(1.0); // nothing to train — trivially parallel
    }
    let mut b = ParameterBinding::new();
    for (k, &id) in theta_ids.iter().enumerate() {
        b.bind(id, 0.1 + 0.01 * k as f64);
    }
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };
    let (_grads, report) =
        omega_core::parallel_shift::parallel_parameter_shift_gradient(backend, &c, &b, &obs, None)
            .map_err(|e| e.to_string())?;
    let total = report.block_symbols + report.fallback_symbols;
    if total == 0 {
        return Ok(1.0);
    }
    Ok(report.block_symbols as f64 / total as f64)
}

/// The expert reference: butterfly_qnn.aria's layer structure.
pub const BUTTERFLY_GENOME: [usize; LAYERS] = [3, 1, 4];
/// The no-entanglement ablation.
pub const NONE_GENOME: [usize; LAYERS] = [0, 0, 0];

pub type Genome = [usize; LAYERS];

pub fn genome_name(g: &Genome) -> String {
    let menu = mask_menu();
    g.iter().map(|&m| menu[m].0).collect::<Vec<_>>().join(" | ")
}

/// Deterministic per-genome seed (mixes the layer indices).
fn genome_seed(g: &Genome, base: u64) -> u64 {
    g.iter().fold(base, |acc, &m| {
        acc.wrapping_mul(31).wrapping_add(m as u64 + 1)
    })
}

/// Build the circuit for a genome: RY(x_i) encoding, then per layer the
/// RBS gates of its mask (one trainable angle per gate).
fn build(g: &Genome, x: &[f64]) -> (CircuitIR, Vec<u32>) {
    let menu = mask_menu();
    let mut c = CircuitIR::new(N_QUBITS as u32, CircuitType::GateBased);
    for (i, &xi) in x.iter().enumerate() {
        c.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(i as u32)],
            params: smallvec![ParamExpr::Concrete(xi)],
            classical_bit: None,
            condition: None,
        });
    }
    let mut theta_ids = Vec::new();
    let mut next: u32 = 0;
    for &m in g.iter() {
        for &(a, b) in &menu[m].1 {
            let id = next;
            next += 1;
            c.symbols.insert(id, format!("t{id}"));
            theta_ids.push(id);
            c.add_op(GateOp {
                gate: GateKind::Rbs,
                qubits: smallvec![Qubit(a), Qubit(b)],
                params: smallvec![ParamExpr::Symbol(id)],
                classical_bit: None,
                condition: None,
            });
        }
    }
    (c, theta_ids)
}

/// Train a genome on (train) and return its held-out MSE — the fitness
/// (lower is better). Fully deterministic from `seed`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    g: &Genome,
    backend: &dyn Backend,
    train_x: &[Vec<f64>],
    train_y: &[f64],
    test_x: &[Vec<f64>],
    test_y: &[f64],
    epochs: usize,
    seed: u64,
) -> Result<f64, String> {
    let z_obs: Vec<Observable> = (0..N_QUBITS as u32)
        .map(|q| Observable {
            terms: vec![(1.0, vec![(q, PauliOp::Z)])],
        })
        .collect();
    // Trainable angles, seeded small init.
    let n_theta = build(g, &train_x[0]).1.len();
    let mut rng = SplitMix64(genome_seed(g, seed));
    let mut theta: HashMap<u32, f64> = (0..n_theta as u32)
        .map(|id| (id, (rng.next_f64() * 2.0 - 1.0) * 0.2))
        .collect();
    let binding = |theta: &HashMap<u32, f64>| {
        let mut b = ParameterBinding::new();
        for (&id, &v) in theta {
            b.bind(id, v);
        }
        b
    };
    let features = |theta: &HashMap<u32, f64>, x: &[f64]| -> Result<Vec<f64>, String> {
        let (c, _) = build(g, x);
        backend
            .expectation_multi(&c, &binding(theta), &z_obs)
            .map_err(|e| e.to_string())
    };
    let fit_head = |theta: &HashMap<u32, f64>| -> Result<Vec<f64>, String> {
        let f: Vec<Vec<f64>> = train_x
            .iter()
            .map(|x| features(theta, x))
            .collect::<Result<_, _>>()?;
        data::ridge_regression(&f, train_y, 1e-3)
    };
    let trainable: HashSet<SymbolId> = theta.keys().copied().collect();
    let (mut am, mut av): (HashMap<u32, f64>, HashMap<u32, f64>) = (HashMap::new(), HashMap::new());
    let mut head = fit_head(&theta)?;
    if !trainable.is_empty() {
        for epoch in 0..epochs {
            if epoch > 0 {
                head = fit_head(&theta)?;
            }
            let hw = Observable {
                terms: (0..N_QUBITS as u32)
                    .map(|q| (head[q as usize], vec![(q, PauliOp::Z)]))
                    .collect(),
            };
            let mut acc: HashMap<u32, f64> = HashMap::new();
            for (x, &yi) in train_x.iter().zip(train_y) {
                let r = data::ridge_predict(&head, &features(&theta, x)?) - yi;
                let (c, _) = build(g, x);
                let grads = compute_gradient_for(
                    backend,
                    &c,
                    &binding(&theta),
                    &hw,
                    &GradMethod::Adjoint,
                    Some(&trainable),
                )
                .map_err(|e| e.to_string())?;
                for (id, gr) in grads {
                    *acc.entry(id).or_insert(0.0) += 2.0 * r * gr / train_x.len() as f64;
                }
            }
            let (b1, b2, eps) = (0.9, 0.999, 1e-8);
            let t = (epoch + 1) as f64;
            for (&id, &gr) in &acc {
                let mi = am.entry(id).or_insert(0.0);
                *mi = b1 * *mi + (1.0 - b1) * gr;
                let vi = av.entry(id).or_insert(0.0);
                *vi = b2 * *vi + (1.0 - b2) * gr * gr;
                let m_hat = *mi / (1.0 - b1.powf(t));
                let v_hat = *vi / (1.0 - b2.powf(t));
                *theta.get_mut(&id).unwrap() -= 0.05 * m_hat / (v_hat.sqrt() + eps);
            }
        }
        head = fit_head(&theta)?;
    }
    let mut sq = 0.0;
    for (x, &yi) in test_x.iter().zip(test_y) {
        let p = data::ridge_predict(&head, &features(&theta, x)?);
        sq += (p - yi) * (p - yi);
    }
    Ok(sq / test_x.len() as f64)
}

/// Fitness-penalty weight for lost hardware trainability: a genome
/// whose gradients are fully serial pays this much MSE-equivalent.
pub const TRAINABILITY_LAMBDA: f64 = 0.05;

/// Elitist evolutionary loop with memoisation. Selection uses the
/// penalised score `mse + λ·(1 − trainability)`; the raw MSE and the
/// trainability fraction are both reported. Returns the winner and the
/// evaluation cache.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn evolve(
    backend: &dyn Backend,
    train_x: &[Vec<f64>],
    train_y: &[f64],
    test_x: &[Vec<f64>],
    test_y: &[f64],
    epochs: usize,
    seed: u64,
    population: usize,
    generations: usize,
) -> Result<(Genome, (f64, f64, f64), HashMap<Genome, (f64, f64, f64)>), String> {
    let menu_len = mask_menu().len();
    let mut rng = SplitMix64(seed);
    let rand_genome = |rng: &mut SplitMix64| -> Genome {
        let mut g = [0usize; LAYERS];
        for slot in g.iter_mut() {
            *slot = (rng.next_f64() * menu_len as f64) as usize % menu_len;
        }
        g
    };
    // cache: genome → (raw MSE, trainability fraction, penalised score).
    let mut cache: HashMap<Genome, (f64, f64, f64)> = HashMap::new();
    let mut pop: Vec<Genome> = (0..population).map(|_| rand_genome(&mut rng)).collect();
    for gen in 0..generations {
        for g in &pop {
            if !cache.contains_key(g) {
                let mse = evaluate(g, backend, train_x, train_y, test_x, test_y, epochs, seed)?;
                let tr = trainability(g, backend, &train_x[0])?;
                let score = mse + TRAINABILITY_LAMBDA * (1.0 - tr);
                cache.insert(*g, (mse, tr, score));
            }
        }
        let mut ranked: Vec<(Genome, (f64, f64, f64))> =
            pop.iter().map(|g| (*g, cache[g])).collect();
        ranked.sort_by(|a, b| a.1 .2.partial_cmp(&b.1 .2).unwrap());
        ranked.dedup_by_key(|(g, _)| *g);
        let elite: Vec<Genome> = ranked
            .iter()
            .take(population / 2)
            .map(|(g, _)| *g)
            .collect();
        println!(
            "    gen {gen}: best score {:.5} (mse {:.5}, trainability {:.2}) [{}]  \
             ({} unique genomes so far)",
            ranked[0].1 .2,
            ranked[0].1 .0,
            ranked[0].1 .1,
            genome_name(&ranked[0].0),
            cache.len()
        );
        if gen + 1 == generations {
            break;
        }
        // Next generation: elites + point-mutated children.
        let mut next = elite.clone();
        while next.len() < population {
            let parent = elite[(rng.next_f64() * elite.len() as f64) as usize % elite.len()];
            let mut child = parent;
            let slot = (rng.next_f64() * LAYERS as f64) as usize % LAYERS;
            child[slot] = (rng.next_f64() * menu_len as f64) as usize % menu_len;
            next.push(child);
        }
        pop = next;
    }
    let (winner, stats) = cache
        .iter()
        .min_by(|a, b| a.1 .2.partial_cmp(&b.1 .2).unwrap())
        .map(|(g, f)| (*g, *f))
        .unwrap();
    Ok((winner, stats, cache))
}
