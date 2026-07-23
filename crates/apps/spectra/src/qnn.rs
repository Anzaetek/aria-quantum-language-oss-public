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
use omega_core::executor::{Backend, PauliOp};
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
    rbs_ids: Vec<Vec<u32>>,
    params: HashMap<u32, f64>,
    head: Vec<f64>,
}

impl TabularQnn {
    pub fn new(d: usize, seed: u64, entangle: bool) -> Self {
        let blocks = 2;
        let mut rng = data::SplitMix64(seed);
        let freq_ids: Vec<u32> = (0..d as u32).collect();
        let mut next = d as u32;
        let mut rbs_ids = Vec::new();
        for _ in 0..blocks {
            let ring: Vec<u32> = (0..(d as u32 - 1)).map(|k| next + k).collect();
            next += d as u32 - 1;
            rbs_ids.push(ring);
        }
        let mut params = HashMap::new();
        for &id in &freq_ids {
            // Non-integer init in the middle of the scan window, jittered.
            params.insert(id, 2.5 + rng.next_f64());
        }
        for ring in &rbs_ids {
            for &id in ring {
                params.insert(id, (rng.next_f64() * 2.0 - 1.0) * 0.2);
            }
        }
        Self {
            d,
            blocks,
            entangle,
            freq_ids,
            rbs_ids,
            params,
            head: vec![],
        }
    }

    /// Per-sample circuit: `blocks` × [RY(f_j·φ_j) ⊗ … ; RBS ring].
    fn circuit(&self, phases: &[f64]) -> CircuitIR {
        let mut c = CircuitIR::new(self.d as u32, CircuitType::GateBased);
        for &id in self.freq_ids.iter().chain(self.rbs_ids.iter().flatten()) {
            c.symbols.insert(id, format!("s{id}"));
        }
        for b in 0..self.blocks {
            for (j, &phi) in phases.iter().enumerate() {
                c.add_op(GateOp {
                    gate: GateKind::Ry,
                    qubits: smallvec![Qubit(j as u32)],
                    params: smallvec![ParamExpr::Mul(
                        Box::new(ParamExpr::Symbol(self.freq_ids[j])),
                        Box::new(ParamExpr::Concrete(phi)),
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
        let obs: Vec<Observable> = (0..self.d as u32)
            .map(|q| Observable {
                terms: vec![(1.0, vec![(q, PauliOp::Z)])],
            })
            .collect();
        backend
            .expectation_multi(&self.circuit(phases), &self.binding(), &obs)
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
                terms: (0..self.d as u32)
                    .map(|q| (self.head[q as usize], vec![(q, PauliOp::Z)]))
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
