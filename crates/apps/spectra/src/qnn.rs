// SPDX-License-Identifier: Apache-2.0
//! The two quantum lanes of the SPECTRA demo.
//!
//! * [`TabularQnn`] — the generic Fourier-encoding QNN of the paper:
//!   re-uploaded RY(f_j·φ_j) encodings with **trainable non-integer
//!   frequency scalars** f_j (a `ParamExpr::Mul(Symbol, Concrete)` per
//!   sample — the engine chain-rules through it), RBS-ring entanglers,
//!   ⟨Z⟩ features + closed-form ridge head folded into H_w = Σ wᵢZᵢ for
//!   one-call adjoint gradients (same hybrid-sandwich pattern as the
//!   butterfly-qnn app). The `entangle = false` ablation removes the
//!   RBS layers — the ON−OFF gate of the certification rule.
//!
//! * [`DmqLane`] — the dynamics-matched quantum lane for the Heisenberg
//!   substrate: the SAME lowered `spectra_heisenberg.aria` circuit with
//!   the bond couplings J_k as trainable symbols; score = the bond
//!   correlator expectation under the learned couplings.

use std::collections::{HashMap, HashSet};

use aria_verify_core::data;
use aria_verify_core::Observable;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit, SymbolId};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, PauliOp};
use omega_core::gradient::{compute_gradient_for, GradMethod};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

use crate::gen::heisenberg_correlator;

fn adam_step(
    m: &mut HashMap<u32, f64>,
    v: &mut HashMap<u32, f64>,
    t: f64,
    theta: &mut HashMap<u32, f64>,
    grads: &HashMap<u32, f64>,
    lr: f64,
) {
    let (b1, b2, eps) = (0.9, 0.999, 1e-8);
    for (&id, &g) in grads {
        let mi = m.entry(id).or_insert(0.0);
        *mi = b1 * *mi + (1.0 - b1) * g;
        let vi = v.entry(id).or_insert(0.0);
        *vi = b2 * *vi + (1.0 - b2) * g * g;
        let m_hat = *mi / (1.0 - b1.powf(t));
        let v_hat = *vi / (1.0 - b2.powf(t));
        *theta.get_mut(&id).unwrap() -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

// ---------------------------------------------------------------------------
// Generic tabular QNN lane
// ---------------------------------------------------------------------------

pub struct TabularQnn {
    d: usize,
    blocks: usize,
    entangle: bool,
    /// symbol ids: 0..d = frequency scalars f_j; then per-block RBS ring
    /// angles.
    freq_ids: Vec<u32>,
    /// Trainable per-feature phase offsets c_j — RY(f_j·φ_j + c_j).
    /// Without them the readout only reaches zero-phase cosine
    /// combinations and a planted cos(Σf·φ + b) with b ≈ ±π/2 is
    /// invisible no matter how good the frequency prior is.
    offset_ids: Vec<u32>,
    rbs_ids: Vec<Vec<u32>>,
    params: HashMap<u32, f64>,
    head: Vec<f64>,
    /// Readout observables. Default: single-qubit ⟨Z_q⟩. With
    /// `rich_readout()`, all Z-strings up to weight 3 (d ≤ 4 only) —
    /// the multi-qubit correlators that carry joint-frequency harmonics
    /// cos(Σ f_j·φ_j) the single-qubit profile cannot express.
    readout: Vec<Observable>,
}

impl TabularQnn {
    pub fn new(d: usize, seed: u64, entangle: bool) -> Self {
        Self::with_freq_prior(d, seed, entangle, None)
    }

    /// `freq_prior`: per-feature initial values for the trainable
    /// frequency scalars — how SPECTRA's classical periodogram
    /// statistics enter the quantum lane (QML_ROADMAP: "Tier-1
    /// statistics → prior over frequencies"). `None` = flat jittered
    /// init in the middle of the scan window.
    pub fn with_freq_prior(
        d: usize,
        seed: u64,
        entangle: bool,
        freq_prior: Option<&[f64]>,
    ) -> Self {
        let blocks = 2;
        let mut rng = data::SplitMix64(seed);
        let freq_ids: Vec<u32> = (0..d as u32).collect();
        let offset_ids: Vec<u32> = (d as u32..2 * d as u32).collect();
        let mut next = 2 * d as u32;
        let mut rbs_ids = Vec::new();
        for _ in 0..blocks {
            let ring: Vec<u32> = (0..(d as u32 - 1)).map(|k| next + k).collect();
            next += d as u32 - 1;
            rbs_ids.push(ring);
        }
        let mut params = HashMap::new();
        for (j, &id) in freq_ids.iter().enumerate() {
            let flat = 2.5 + rng.next_f64();
            let v = freq_prior.and_then(|f| f.get(j).copied()).unwrap_or(flat);
            params.insert(id, v);
        }
        for &id in &offset_ids {
            params.insert(id, 0.0);
        }
        for ring in &rbs_ids {
            for &id in ring {
                params.insert(id, (rng.next_f64() * 2.0 - 1.0) * 0.2);
            }
        }
        let readout = (0..d as u32)
            .map(|q| Observable {
                terms: vec![(1.0, vec![(q, PauliOp::Z)])],
            })
            .collect();
        Self {
            d,
            blocks,
            entangle,
            freq_ids,
            offset_ids,
            rbs_ids,
            params,
            head: vec![],
            readout,
        }
    }

    /// Set the number of re-upload blocks (default 2). One block keeps
    /// the harmonic coefficients in {0, ±1} — the cleanest structure
    /// for a frequency prior to act on.
    pub fn upload_blocks(mut self, blocks: usize) -> Self {
        self.blocks = blocks;
        self
    }

    /// Switch to the rich Z-string readout (all subsets of weight ≤ 3;
    /// d ≤ 4 to keep the feature count small).
    pub fn rich_readout(mut self) -> Self {
        assert!(self.d <= 4, "rich readout is for small d");
        let mut obs = Vec::new();
        let d = self.d as u32;
        for a in 0..d {
            obs.push(vec![(a, PauliOp::Z)]);
        }
        for a in 0..d {
            for b in (a + 1)..d {
                obs.push(vec![(a, PauliOp::Z), (b, PauliOp::Z)]);
            }
        }
        for a in 0..d {
            for b in (a + 1)..d {
                for c_ in (b + 1)..d {
                    obs.push(vec![(a, PauliOp::Z), (b, PauliOp::Z), (c_, PauliOp::Z)]);
                }
            }
        }
        self.readout = obs
            .into_iter()
            .map(|terms| Observable {
                terms: vec![(1.0, terms)],
            })
            .collect();
        self
    }

    /// Per-sample circuit: `blocks` × [RY(f_j·φ_j) ⊗ … ; RBS ring].
    fn circuit(&self, phases: &[f64]) -> CircuitIR {
        let mut c = CircuitIR::new(self.d as u32, CircuitType::GateBased);
        for &id in self
            .freq_ids
            .iter()
            .chain(self.offset_ids.iter())
            .chain(self.rbs_ids.iter().flatten())
        {
            c.symbols.insert(id, format!("s{id}"));
        }
        for b in 0..self.blocks {
            for (j, &phi) in phases.iter().enumerate() {
                c.add_op(GateOp {
                    gate: GateKind::Ry,
                    qubits: smallvec![Qubit(j as u32)],
                    params: smallvec![ParamExpr::Add(
                        Box::new(ParamExpr::Mul(
                            Box::new(ParamExpr::Symbol(self.freq_ids[j])),
                            Box::new(ParamExpr::Concrete(phi)),
                        )),
                        Box::new(ParamExpr::Symbol(self.offset_ids[j])),
                    )],
                    classical_bit: None,
                    condition: None,
                });
            }
            if self.entangle {
                for (k, &id) in self.rbs_ids[b].iter().enumerate() {
                    c.add_op(GateOp {
                        gate: GateKind::Rbs,
                        qubits: smallvec![Qubit(k as u32), Qubit(k as u32 + 1)],
                        params: smallvec![ParamExpr::Symbol(id)],
                        classical_bit: None,
                        condition: None,
                    });
                }
            }
        }
        c
    }

    fn binding(&self) -> ParameterBinding {
        let mut b = ParameterBinding::new();
        for (&id, &v) in &self.params {
            b.bind(id, v);
        }
        b
    }

    fn features(&self, backend: &dyn Backend, phases: &[f64]) -> Result<Vec<f64>, String> {
        backend
            .expectation_multi(&self.circuit(phases), &self.binding(), &self.readout)
            .map_err(|e| e.to_string())
    }

    fn fit_head(
        &self,
        backend: &dyn Backend,
        x: &[Vec<f64>],
        y: &[f64],
    ) -> Result<Vec<f64>, String> {
        let f: Vec<Vec<f64>> = x
            .iter()
            .map(|p| self.features(backend, p))
            .collect::<Result<_, _>>()?;
        data::ridge_regression(&f, y, 1e-3)
    }

    /// Train frequencies + entangler angles (adjoint gradients against
    /// H_w = Σ wᵢZᵢ), refitting the ridge head each epoch.
    pub fn fit(
        &mut self,
        backend: &dyn Backend,
        x: &[Vec<f64>],
        y: &[f64],
        epochs: usize,
        lr: f64,
    ) -> Result<(), String> {
        let trainable: HashSet<SymbolId> = self.params.keys().copied().collect();
        let (mut am, mut av) = (HashMap::new(), HashMap::new());
        self.head = self.fit_head(backend, x, y)?;
        for epoch in 0..epochs {
            if epoch > 0 {
                self.head = self.fit_head(backend, x, y)?;
            }
            let hw = Observable {
                terms: self
                    .readout
                    .iter()
                    .enumerate()
                    .map(|(k, o)| (self.head[k], o.terms[0].1.clone()))
                    .collect(),
            };
            let mut acc: HashMap<u32, f64> = HashMap::new();
            for (p, &yi) in x.iter().zip(y) {
                let pred = data::ridge_predict(&self.head, &self.features(backend, p)?);
                let r = pred - yi;
                let circuit = self.circuit(p);
                let grads = compute_gradient_for(
                    backend,
                    &circuit,
                    &self.binding(),
                    &hw,
                    &GradMethod::Adjoint,
                    Some(&trainable),
                )
                .map_err(|e| e.to_string())?;
                for (id, g) in grads {
                    *acc.entry(id).or_insert(0.0) += 2.0 * r * g / x.len() as f64;
                }
            }
            adam_step(
                &mut am,
                &mut av,
                (epoch + 1) as f64,
                &mut self.params,
                &acc,
                lr,
            );
        }
        self.head = self.fit_head(backend, x, y)?;
        Ok(())
    }

    pub fn scores(&self, backend: &dyn Backend, x: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        x.iter()
            .map(|p| Ok(data::ridge_predict(&self.head, &self.features(backend, p)?)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Dynamics-matched quantum lane (Heisenberg substrate)
// ---------------------------------------------------------------------------

pub struct DmqLane<'a> {
    pub ir: &'a CircuitIR,
    pub jt_ids: &'a [u32],
    pub pt_ids: &'a [u32],
    pub dt: f64,
    pub correlator: &'a Observable,
    /// Learned couplings J_k (NOT pre-scaled by dt).
    pub couplings: Vec<f64>,
    head: Vec<f64>,
}

impl<'a> DmqLane<'a> {
    pub fn new(
        ir: &'a CircuitIR,
        jt_ids: &'a [u32],
        pt_ids: &'a [u32],
        dt: f64,
        correlator: &'a Observable,
    ) -> Self {
        Self {
            ir,
            jt_ids,
            pt_ids,
            dt,
            correlator,
            // Deliberately NOT the generator's draw: flat init at the
            // U[0.5, 1.5] prior mean — the lane must LEARN the disorder.
            couplings: vec![1.0; jt_ids.len()],
            head: vec![0.0, 0.0],
        }
    }

    fn raw(&self, backend: &dyn Backend, phases: &[f64]) -> Result<f64, String> {
        heisenberg_correlator(
            backend,
            self.ir,
            self.jt_ids,
            self.pt_ids,
            &self.couplings,
            self.dt,
            phases,
            self.correlator,
        )
    }

    /// Train the couplings by MSE of the affine-rescaled correlator
    /// against the ±1 labels; the 1-D head (a, b) refits in closed form
    /// each epoch. Gradients: adjoint on the correlator observable, with
    /// the dt chain rule folded in (symbols are J·dt).
    pub fn fit(
        &mut self,
        backend: &dyn Backend,
        x: &[Vec<f64>],
        y: &[f64],
        epochs: usize,
        lr: f64,
    ) -> Result<(), String> {
        let trainable: HashSet<SymbolId> = self.jt_ids.iter().copied().collect();
        let (mut am, mut av) = (HashMap::new(), HashMap::new());
        for epoch in 0..epochs {
            // Closed-form 1-D affine head on the raw correlator.
            let raw: Vec<f64> = x
                .iter()
                .map(|p| self.raw(backend, p))
                .collect::<Result<_, _>>()?;
            let feats: Vec<Vec<f64>> = raw.iter().map(|&v| vec![v]).collect();
            self.head = data::ridge_regression(&feats, y, 1e-6)?;
            let a = self.head[0];

            let mut acc: HashMap<u32, f64> = HashMap::new();
            for ((p, &yi), &rv) in x.iter().zip(y).zip(&raw) {
                let r = data::ridge_predict(&self.head, &[rv]) - yi;
                let mut b = ParameterBinding::new();
                for (&id, &j) in self.jt_ids.iter().zip(&self.couplings) {
                    b.bind(id, j * self.dt);
                }
                for (&id, &ph) in self.pt_ids.iter().zip(p.iter()) {
                    b.bind(id, ph * self.dt);
                }
                let grads = compute_gradient_for(
                    backend,
                    self.ir,
                    &b,
                    self.correlator,
                    &GradMethod::Adjoint,
                    Some(&trainable),
                )
                .map_err(|e| e.to_string())?;
                // d/dJ = dt · d/d(J·dt); dL/dJ = 2·r·a·dt·(d⟨O⟩/d jt).
                for (id, g) in grads {
                    *acc.entry(id).or_insert(0.0) += 2.0 * r * a * self.dt * g / x.len() as f64;
                }
            }
            let mut cur: HashMap<u32, f64> = self
                .jt_ids
                .iter()
                .zip(&self.couplings)
                .map(|(&id, &j)| (id, j))
                .collect();
            adam_step(&mut am, &mut av, (epoch + 1) as f64, &mut cur, &acc, lr);
            for (k, &id) in self.jt_ids.iter().enumerate() {
                self.couplings[k] = cur[&id];
            }
        }
        Ok(())
    }

    pub fn scores(&self, backend: &dyn Backend, x: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        x.iter()
            .map(|p| Ok(data::ridge_predict(&self.head, &[self.raw(backend, p)?])))
            .collect()
    }

    /// Same as [`scores`](Self::scores) but evaluates the per-row correlator
    /// across rayon threads — for the noisy PauliProp backend, whose exact
    /// Heisenberg-adjoint expectation is orders of magnitude slower than the
    /// statevector path. Order-preserving, so it's a drop-in for `scores`.
    pub fn scores_par(
        &self,
        backend: &(dyn Backend + Sync),
        x: &[Vec<f64>],
    ) -> Result<Vec<f64>, String> {
        use rayon::prelude::*;
        x.par_iter()
            .map(|p| Ok(data::ridge_predict(&self.head, &[self.raw(backend, p)?])))
            .collect()
    }

    /// Per-row scores from a FINITE number of measurement shots — the
    /// sampling-noise axis of hardware realism, on top of the exact/decoherence
    /// paths. The correlator ⟨Σ Z_k Z_{k+1}⟩ is estimated by sampling the
    /// evolved state in the computational (Z) basis `shots` times and averaging
    /// Σ_k z_k z_{k+1} per shot (an UNBIASED estimator: variance ∝ 1/shots, no
    /// bias), then run through the fixed exact-trained head. Deterministic given
    /// `seed` (each row is sampled with a distinct sub-seed).
    pub fn scores_shots(
        &self,
        backend: &dyn Backend,
        x: &[Vec<f64>],
        shots: u32,
        seed: u64,
    ) -> Result<Vec<f64>, String> {
        x.iter()
            .enumerate()
            .map(|(i, p)| {
                let raw = self.shot_correlator(
                    backend,
                    p,
                    shots,
                    seed ^ (i as u64).wrapping_mul(0x9E37),
                )?;
                Ok(data::ridge_predict(&self.head, &[raw]))
            })
            .collect()
    }

    /// Sample ⟨Σ Z_k Z_{k+1}⟩ from `shots` Z-basis measurements of the evolved
    /// state for one phase row.
    fn shot_correlator(
        &self,
        backend: &dyn Backend,
        phases: &[f64],
        shots: u32,
        seed: u64,
    ) -> Result<f64, String> {
        let mut b = ParameterBinding::new();
        for (&id, &j) in self.jt_ids.iter().zip(&self.couplings) {
            b.bind(id, j * self.dt);
        }
        for (&id, &ph) in self.pt_ids.iter().zip(phases) {
            b.bind(id, ph * self.dt);
        }
        let cfg = ExecConfig {
            shots: Some(shots),
            seed: Some(seed),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let counts = match backend
            .execute(self.ir, &b, &cfg)
            .map_err(|e| e.to_string())?
        {
            ExecResult::Counts(m) => m,
            other => return Err(format!("expected sampled counts, got {other:?}")),
        };
        // z_q = 1 − 2·bit_q (qubit q is bit q, LSB-first); correlator per shot
        // is Σ_{k=0}^{5} z_k z_{k+1}, averaged over the sampled bitstrings.
        let total: u64 = counts.values().map(|&c| c as u64).sum();
        if total == 0 {
            return Ok(0.0);
        }
        let mut acc = 0.0;
        for (state, &c) in &counts {
            let mut corr = 0.0;
            for k in 0..6u32 {
                let zk = 1 - 2 * state.bit(k) as i64;
                let zk1 = 1 - 2 * state.bit(k + 1) as i64;
                corr += (zk * zk1) as f64;
            }
            acc += corr * c as f64;
        }
        Ok(acc / total as f64)
    }

    /// The ablation lane (certification's ON−OFF gate): couplings pinned
    /// to zero — no entanglement, product-state dynamics only. The head
    /// still refits, so the comparison is fair.
    pub fn ablated(&self) -> Self {
        Self {
            ir: self.ir,
            jt_ids: self.jt_ids,
            pt_ids: self.pt_ids,
            dt: self.dt,
            correlator: self.correlator,
            couplings: vec![0.0; self.jt_ids.len()],
            head: vec![0.0, 0.0],
        }
    }

    /// Fit only the affine head with couplings frozen (used by the
    /// ablated lane).
    pub fn fit_head_only(
        &mut self,
        backend: &dyn Backend,
        x: &[Vec<f64>],
        y: &[f64],
    ) -> Result<(), String> {
        let raw: Vec<f64> = x
            .iter()
            .map(|p| self.raw(backend, p))
            .collect::<Result<_, _>>()?;
        let feats: Vec<Vec<f64>> = raw.iter().map(|&v| vec![v]).collect();
        self.head = data::ridge_regression(&feats, y, 1e-6)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omega_backend_statevector::StatevectorBackend;

    fn gate(kind: GateKind, q: u32) -> GateOp {
        GateOp {
            gate: kind,
            qubits: smallvec![Qubit(q)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        }
    }

    fn zz_correlator() -> Observable {
        Observable::parse("1.0*Z0Z1 + 1.0*Z1Z2 + 1.0*Z2Z3 + 1.0*Z3Z4 + 1.0*Z4Z5 + 1.0*Z5Z6")
            .unwrap()
    }

    // A DmqLane with no trainable symbols over a hand-built 7-qubit circuit, so
    // shot_correlator samples that fixed circuit directly (couplings/phases bind
    // nothing). This pins the estimator's bit convention and 6-bond averaging
    // without the substrate .aria or any training.
    #[test]
    fn shot_correlator_exact_on_a_product_state() {
        // X on qubits {0,2,4} → basis state with bits 0,2,4 set;
        // z = [-1,+1,-1,+1,-1,+1,+1]; Σ_k z_k z_{k+1} = -5 + 1 = -4, deterministic.
        let mut ir = CircuitIR::new(7, CircuitType::GateBased);
        for q in [0u32, 2, 4] {
            ir.add_op(gate(GateKind::X, q));
        }
        let corr = zz_correlator();
        let (jt, pt): (Vec<u32>, Vec<u32>) = (vec![], vec![]);
        let dmq = DmqLane::new(&ir, &jt, &pt, 1.0, &corr);
        let est = dmq
            .shot_correlator(&StatevectorBackend::new(), &[], 1000, 7)
            .unwrap();
        assert!(
            (est - (-4.0)).abs() < 1e-9,
            "product-state ΣZZ estimate {est}, want -4"
        );
    }

    #[test]
    fn shot_correlator_converges_on_a_superposition() {
        // H on every qubit → |+>^7: each ⟨Z_k Z_{k+1}⟩ = 0, so ΣZZ = 0. The
        // sampled mean must converge there (unbiasedness + correct averaging).
        let mut ir = CircuitIR::new(7, CircuitType::GateBased);
        for q in 0..7 {
            ir.add_op(gate(GateKind::H, q));
        }
        let corr = zz_correlator();
        let (jt, pt): (Vec<u32>, Vec<u32>) = (vec![], vec![]);
        let dmq = DmqLane::new(&ir, &jt, &pt, 1.0, &corr);
        let est = dmq
            .shot_correlator(&StatevectorBackend::new(), &[], 100_000, 42)
            .unwrap();
        assert!(est.abs() < 0.1, "|+>^7 ΣZZ estimate {est}, want ~0");
    }
}
