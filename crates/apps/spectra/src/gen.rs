// SPDX-License-Identifier: Apache-2.0
//! Synthetic dataset generators for the SPECTRA certification demo
//! (arXiv:2607.15815 §6.2–6.3). Both are deterministic from a printed
//! seed; phases live in [−π, π].

use aria_verify_core::data::SplitMix64;
use aria_verify_core::Observable;
use omega_core::executor::Backend;
use omega_core::params::ParameterBinding;

/// A labelled phase-encoded dataset: `phases[i]` ∈ [−π, π]^d, `y[i]` ∈ {−1, +1}.
pub struct Dataset {
    pub phases: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub name: &'static str,
}

fn uniform_phase(rng: &mut SplitMix64) -> f64 {
    (rng.next_f64() * 2.0 - 1.0) * std::f64::consts::PI
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

/// Controlled sparse pocket (paper §6.2): three independent phases, one
/// ground-truth 3-way joint Fourier term over off-grid frequencies
/// f = (3.7, 5.1, 6.8) on top of weak per-feature harmonics:
///
///   z(φ) = w·Σ_j cos(f_j·φ_j + b_j) + cos(f₁φ₁ + f₂φ₂ + f₃φ₃ + b),
///   y ~ Bernoulli(σ(g·(z − median))),  w = 0.25, g = 4.
///
/// This is the *calibration instrument*: a complete classical lane panel
/// (order-matched JOINT scan) must find the single joint term and close
/// the gap — certification must REFUSE here.
pub fn sparse_pocket(n: usize, seed: u64) -> Dataset {
    let f = [3.7, 5.1, 6.8];
    let mut rng = SplitMix64(seed);
    let b: Vec<f64> = (0..4).map(|_| uniform_phase(&mut rng)).collect();
    let phases: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..3).map(|_| uniform_phase(&mut rng)).collect())
        .collect();
    let z: Vec<f64> = phases
        .iter()
        .map(|p| {
            let singles: f64 = (0..3).map(|j| (f[j] * p[j] + b[j]).cos()).sum();
            0.25 * singles + (f[0] * p[0] + f[1] * p[1] + f[2] * p[2] + b[3]).cos()
        })
        .collect();
    let tau = median(&z);
    let y: Vec<f64> = z
        .iter()
        .map(|&zi| {
            let sigma = 1.0 / (1.0 + (-4.0 * (zi - tau)).exp());
            if rng.next_f64() < sigma {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
    Dataset {
        phases,
        y,
        name: "sparse_pocket",
    }
}

/// Fixed disorder draw for the Heisenberg substrate: J_k ~ U[0.5, 1.5],
/// drawn once per seed and held fixed across the dataset (paper §6.3).
pub fn heisenberg_couplings(seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64(seed ^ 0x4A5F_11E1);
    (0..6).map(|_| 0.5 + rng.next_f64()).collect()
}

/// Quantum-generated dense-spectrum substrate (paper §6.3): 7 uniform
/// phases drive the Z-fields of a disordered Heisenberg chain evolved
/// from |+⟩^⊗7 by `steps` first-order Trotter slices of total time `t`
/// (the lowered `spectra_heisenberg.aria` circuit). The label is the
/// sign of the nearest-neighbour correlator ⟨Σ Z_kZ_{k+1}⟩ against its
/// median — a function with dense joint Fourier structure across all 7
/// phases that no small enumerable term set reproduces.
#[allow(clippy::too_many_arguments)]
pub fn heisenberg(
    n: usize,
    seed: u64,
    backend: &dyn Backend,
    ir: &omega_core::circuit::CircuitIR,
    jt_ids: &[u32],
    pt_ids: &[u32],
    couplings: &[f64],
    dt: f64,
    correlator: &Observable,
) -> Result<Dataset, String> {
    let mut rng = SplitMix64(seed);
    let phases: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..7).map(|_| uniform_phase(&mut rng)).collect())
        .collect();
    let mut z = Vec::with_capacity(n);
    for p in &phases {
        z.push(heisenberg_correlator(
            backend, ir, jt_ids, pt_ids, couplings, dt, p, correlator,
        )?);
    }
    let tau = median(&z);
    let y: Vec<f64> = z
        .iter()
        .map(|&v| if v > tau { 1.0 } else { -1.0 })
        .collect();
    Ok(Dataset {
        phases,
        y,
        name: "heisenberg_substrate",
    })
}

/// ⟨Σ Z_kZ_{k+1}⟩ of the Trotterised chain for one phase vector — used
/// by both the generator (fixed true couplings) and the DMQ lane
/// (trainable couplings).
#[allow(clippy::too_many_arguments)]
pub fn heisenberg_correlator(
    backend: &dyn Backend,
    ir: &omega_core::circuit::CircuitIR,
    jt_ids: &[u32],
    pt_ids: &[u32],
    couplings: &[f64],
    dt: f64,
    phases: &[f64],
    correlator: &Observable,
) -> Result<f64, String> {
    let mut b = ParameterBinding::new();
    for (&id, &j) in jt_ids.iter().zip(couplings) {
        b.bind(id, j * dt);
    }
    for (&id, &p) in pt_ids.iter().zip(phases) {
        b.bind(id, p * dt);
    }
    backend
        .expectation(ir, &b, correlator)
        .map_err(|e| e.to_string())
}
