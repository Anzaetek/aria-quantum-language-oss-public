//! Trajectory application of the shared [`NoiseModel`] for the statevector
//! backend.
//!
//! The noise *data model* lives in [`omega_core::noise`]; this module is the
//! statevector-specific *application*: each channel is applied per-qubit
//! immediately after each gate that touches the qubit, via the Monte-Carlo
//! quantum-jump method — draw the Kraus branch weighted by ⟨ψ|E_k†E_k|ψ⟩,
//! apply that operator, then renormalise. Aggregating many shots reproduces the
//! density-matrix dynamics without ever materialising ρ. Rates are read
//! per-qubit and per gate arity, so the same code serves a uniform idealized
//! device and a calibrated one.

use num_complex::Complex64;
use rand::{Rng, RngExt};

use omega_core::noise::{NoiseModel, ReadoutError};

use crate::gates;
use crate::sim::apply_1q;

/// Apply the full per-gate channel to qubit `q`, for a gate acting on `arity`
/// qubits (the arity selects the 1q vs 2q depolarizing rate).
pub fn apply_channel<R: Rng>(
    model: &NoiseModel,
    state: &mut [Complex64],
    n: usize,
    q: usize,
    arity: usize,
    rng: &mut R,
) {
    let depol = model.depolarizing.at(q, arity);
    if depol > 0.0 {
        let r: f64 = rng.random();
        if r < depol {
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

    if let Some(pauli) = &model.pauli {
        let (px, py, pz) = (pauli.x.at(q), pauli.y.at(q), pauli.z.at(q));
        if px + py + pz > 0.0 {
            let r: f64 = rng.random();
            let c_x = px;
            let c_y = c_x + py;
            let c_z = c_y + pz;
            if r < c_x {
                apply_1q(state, n, q, &gates::x());
            } else if r < c_y {
                apply_1q(state, n, q, &gates::y());
            } else if r < c_z {
                apply_1q(state, n, q, &gates::z());
            }
        }
    }

    let gamma = model.amplitude_damping.at(q);
    if gamma > 0.0 {
        apply_amplitude_damping(state, n, q, gamma, rng);
    }

    let lambda = model.phase_damping.at(q);
    if lambda > 0.0 {
        // Trajectory form: Z is applied with probability λ/2.
        let r: f64 = rng.random();
        if r < 0.5 * lambda {
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

/// Flip a measured bit on qubit `q` per the (possibly asymmetric, per-qubit)
/// readout error, given the true outcome bit.
pub fn maybe_flip_readout<R: Rng>(
    readout: &ReadoutError,
    q: usize,
    outcome: u8,
    rng: &mut R,
) -> u8 {
    let p = readout.flip_prob(q, outcome);
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
