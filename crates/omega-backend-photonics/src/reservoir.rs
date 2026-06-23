//! Stateful temporal reservoir — a photonic circuit with a memristive feedback
//! loop across time steps.
//!
//! Plain SLOS is one-shot: a single input Fock state through a single unitary.
//! A photonic *reservoir* / quantum-memristor processes a **sequence**: at each
//! step the circuit depends both on the current input `x[t]` and on an internal
//! memory state `R[t]` carried over from previous steps, and the measured output
//! drives the memory update. This is the temporal feedback that one-shot SLOS
//! cannot express.
//!
//! The memristive update is an exponential moving average (low-pass filter) of a
//! readout probability — the model used for the NARMA-style reservoir tasks:
//!
//! ```text
//!   R[t] = clamp( R[t-1] + (P[t] − R[t-1]) / τ ,  ε, 1−ε )
//! ```
//!
//! `τ` (the memory size) sets the time constant: small `τ` → fast response / low
//! memory, large `τ` → long hysteresis. `R[t]` is fed back into the next step's
//! circuit as a phase (`θ = π·R`). Each step records the full per-mode
//! occupation probabilities as the reservoir feature vector; a downstream *linear*
//! readout (trained classically) maps features → target — the reservoir itself is
//! never trained, which is the whole point of reservoir computing.

use crate::components::{self, PhotonicOp};
use crate::slos::{self, FockState};

/// One step of the reservoir: the input-driven feature vector and the memory
/// value after the memristive update.
#[derive(Clone, Debug)]
pub struct ReservoirStep {
    /// Per-mode occupation probability ⟨n_mode⟩ (the reservoir feature vector).
    pub features: Vec<f64>,
    /// Internal memory state R[t] after this step.
    pub memory: f64,
}

/// A photonic reservoir with a memristive feedback memory.
pub struct MemristiveReservoir {
    pub n_modes: usize,
    /// Fixed input Fock state injected every step.
    pub input_state: FockState,
    /// Memory time constant τ (memory size).
    pub tau: f64,
    /// Initial memory state R[0].
    pub r0: f64,
    /// Clamp margin ε keeping R ∈ [ε, 1−ε].
    pub eps: f64,
    /// Which mode's occupation probability drives the memory update.
    pub readout_mode: usize,
}

impl MemristiveReservoir {
    pub fn new(n_modes: usize, input_state: FockState) -> Self {
        Self {
            n_modes,
            input_state,
            tau: 4.0,
            r0: 0.5,
            eps: 1e-6,
            readout_mode: 1,
        }
    }

    /// Map the memory state `R ∈ [0,1]` to a feedback phase `θ = π·R`.
    pub fn r_to_theta(r: f64) -> f64 {
        std::f64::consts::PI * r
    }

    /// Per-mode occupation expectation `⟨n_k⟩ = Σ_t n_k(t)·P(t)` from a SLOS
    /// distribution.
    fn mode_expectations(&self, dist: &[(FockState, f64)]) -> Vec<f64> {
        let mut e = vec![0.0; self.n_modes];
        for (state, p) in dist {
            for (k, &nk) in state.iter().enumerate() {
                e[k] += nk as f64 * p;
            }
        }
        e
    }

    /// Run the reservoir over an input sequence. `build_ops(x, feedback_theta)`
    /// returns the optical components for one step given the current input and
    /// the fed-back memory phase. Returns one [`ReservoirStep`] per input.
    pub fn run<F>(&self, inputs: &[f64], build_ops: F) -> Vec<ReservoirStep>
    where
        F: Fn(f64, f64) -> Vec<PhotonicOp>,
    {
        let mut r = self.r0;
        let mut trace = Vec::with_capacity(inputs.len());
        for &x in inputs {
            let theta = Self::r_to_theta(r);
            let ops = build_ops(x, theta);
            let u = components::build_unitary(self.n_modes, &ops);
            let dist = slos::slos_full(&u, &self.input_state);
            let features = self.mode_expectations(&dist);
            // Memristive EMA update driven by the readout-mode occupation.
            let p_update = features[self.readout_mode];
            r = (r + (p_update - r) / self.tau).clamp(self.eps, 1.0 - self.eps);
            trace.push(ReservoirStep {
                features,
                memory: r,
            });
        }
        trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_4;

    /// A 3-mode, 1-photon memristor cell: encode x, mix, inject feedback phase.
    fn cell(x: f64, fb: f64) -> Vec<PhotonicOp> {
        vec![
            PhotonicOp::BeamSplitterRx {
                mode0: 0,
                mode1: 1,
                theta: FRAC_PI_4,
                phi: 0.0,
            },
            PhotonicOp::PhaseShifter { mode: 1, phi: x },
            PhotonicOp::BeamSplitterRx {
                mode0: 1,
                mode1: 2,
                theta: FRAC_PI_4,
                phi: 0.0,
            },
            PhotonicOp::PhaseShifter { mode: 0, phi: fb }, // feedback injection
            PhotonicOp::BeamSplitterRx {
                mode0: 0,
                mode1: 1,
                theta: FRAC_PI_4,
                phi: 0.0,
            },
        ]
    }

    #[test]
    fn memory_state_stays_bounded_and_responds() {
        let res = MemristiveReservoir::new(3, vec![0, 1, 0]);
        let inputs: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        let trace = res.run(&inputs, cell);
        assert_eq!(trace.len(), inputs.len());
        for step in &trace {
            assert!(step.memory >= res.eps && step.memory <= 1.0 - res.eps);
            // features are mode occupations of a 1-photon state ⇒ sum ≈ 1.
            let total: f64 = step.features.iter().sum();
            assert!((total - 1.0).abs() < 1e-9, "occupations sum to {total}");
        }
    }

    #[test]
    fn larger_tau_means_slower_memory() {
        // The memory variance over the sequence should shrink as τ grows
        // (stronger low-pass / longer hysteresis).
        let inputs: Vec<f64> = (0..80).map(|i| (i as f64 * 0.5).sin()).collect();
        let var = |tau: f64| {
            let mut res = MemristiveReservoir::new(3, vec![0, 1, 0]);
            res.tau = tau;
            let trace = res.run(&inputs, cell);
            let mem: Vec<f64> = trace.iter().map(|s| s.memory).collect();
            let mean = mem.iter().sum::<f64>() / mem.len() as f64;
            mem.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / mem.len() as f64
        };
        assert!(var(2.0) > var(20.0), "small τ should fluctuate more");
    }
}
