//! Composable noise model for trajectory-based simulation.
//!
//! Every channel is applied per-qubit immediately after each gate that
//! touches the qubit, via the Monte-Carlo quantum-jump method:
//! draw the Kraus branch weighted by ⟨ψ|E_k†E_k|ψ⟩, apply that operator,
//! then renormalise. Aggregating many shots reproduces the density-matrix
//! dynamics without ever materialising ρ.

use num_complex::Complex64;
use rand::{Rng, RngExt};

use crate::gates;
use crate::sim::apply_1q;

/// Configurable per-gate noise model. Each rate defaults to `0.0` so the
/// all-zero model reduces to exact noise-free simulation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NoiseModel {
    /// Single-qubit depolarizing rate. After each gate, with probability
    /// `depolarizing`, a uniformly chosen X/Y/Z is applied to each
    /// affected qubit.
    pub depolarizing: f64,
    /// Amplitude-damping rate γ. Kraus operators
    /// `E_0 = [[1,0],[0,√(1-γ)]]`, `E_1 = [[0,√γ],[0,0]]`.
    pub amplitude_damping: f64,
    /// Phase-damping rate γ. Trajectory-form: apply Z with probability γ/2.
    pub phase_damping: f64,
    /// Arbitrary Pauli channel `(p_I, p_X, p_Y, p_Z)`. The four entries
    /// must sum to ≤ 1; the remainder is treated as identity.
    pub pauli: Option<PauliRates>,
    /// Bit-flip probability applied to every classical measurement outcome.
    pub readout_flip: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PauliRates {
    pub p_i: f64,
    pub p_x: f64,
    pub p_y: f64,
    pub p_z: f64,
}

impl NoiseModel {
    pub fn noiseless(&self) -> bool {
        self.depolarizing == 0.0
            && self.amplitude_damping == 0.0
            && self.phase_damping == 0.0
            && self.pauli.is_none()
            && self.readout_flip == 0.0
    }
}

/// Apply the full per-gate channel to qubit `q`.
pub fn apply_channel<R: Rng>(
    model: &NoiseModel,
    state: &mut [Complex64],
    n: usize,
    q: usize,
    rng: &mut R,
) {
    if model.depolarizing > 0.0 {
        let r: f64 = rng.random();
        if r < model.depolarizing {
            let choice: f64 = rng.random();
            if choice < 1.0 / 3.0 {
                apply_1q(state, n, q, &gates::x());
            } else if choice < 2.0 / 3.0 {
                apply_1q(state, n, q, &gates::y());
            } else {
                apply_1q(state, n, q, &gates::z());
            }
        }
    }

    if let Some(rates) = model.pauli {
        let r: f64 = rng.random();
        let c_x = rates.p_x;
        let c_y = c_x + rates.p_y;
        let c_z = c_y + rates.p_z;
        if r < c_x {
            apply_1q(state, n, q, &gates::x());
        } else if r < c_y {
            apply_1q(state, n, q, &gates::y());
        } else if r < c_z {
            apply_1q(state, n, q, &gates::z());
        }
    }

    if model.amplitude_damping > 0.0 {
        apply_amplitude_damping(state, n, q, model.amplitude_damping, rng);
    }

    if model.phase_damping > 0.0 {
        // Trajectory form: Z is applied with probability γ/2.
        let r: f64 = rng.random();
        if r < 0.5 * model.phase_damping {
            apply_1q(state, n, q, &gates::z());
        }
    }
}

/// Amplitude damping via quantum-jump sampling.
///
/// `p_jump = γ * P(qubit is in |1⟩)`. On jump, collapse the qubit to |0⟩
/// (transferring |1⟩ amplitude into |0⟩ on that qubit); otherwise apply
/// `E_0` and renormalise.
fn apply_amplitude_damping<R: Rng>(
    state: &mut [Complex64],
    n: usize,
    q: usize,
    gamma: f64,
    rng: &mut R,
) {
    let dim = 1usize << n;
    let bit = 1usize << q;

    // Probability that qubit q is in |1⟩.
    let mut p1 = 0.0f64;
    for (idx, amp) in state.iter().enumerate() {
        if idx & bit != 0 {
            p1 += amp.norm_sqr();
        }
    }
    if p1 <= 0.0 {
        return;
    }

    let p_jump = gamma * p1;
    let r: f64 = rng.random();
    if r < p_jump {
        // Jump: the qubit emits; transfer amplitude from |1⟩ to |0⟩.
        // Post-jump normaliser = 1 / √(γ p1).
        let norm = 1.0 / (gamma * p1).sqrt();
        for idx in 0..dim {
            if idx & bit != 0 {
                let amp_one = state[idx];
                let partner = idx ^ bit;
                state[partner] = amp_one * Complex64::new((gamma).sqrt() * norm, 0.0);
                state[idx] = Complex64::new(0.0, 0.0);
            }
        }
    } else {
        // No jump: apply E_0 = diag(1, √(1-γ)) and renormalise.
        let sqrt_one_minus_g = (1.0 - gamma).sqrt();
        let mut norm_sq = 0.0f64;
        for idx in 0..dim {
            if idx & bit != 0 {
                state[idx] *= sqrt_one_minus_g;
            }
            norm_sq += state[idx].norm_sqr();
        }
        if norm_sq > 0.0 {
            let inv = 1.0 / norm_sq.sqrt();
            for amp in state.iter_mut() {
                *amp *= inv;
            }
        }
    }
}

/// Flip a classical bit outcome with probability `p`.
pub fn maybe_flip<R: Rng>(p: f64, outcome: u8, rng: &mut R) -> u8 {
    if p <= 0.0 {
        return outcome;
    }
    let r: f64 = rng.random();
    if r < p {
        outcome ^ 1
    } else {
        outcome
    }
}
