//! Fock-amplitude statevector — coherent linear-optical state with
//! **partial-mode measurement** and **adaptive feed-forward state injection**.
//!
//! `slos_full` gives the output *probability* distribution of a single
//! interferometer; it cannot express a mid-circuit measurement that reads out a
//! subset of modes while keeping the rest coherent, nor a later optic that is
//! *conditioned* on that outcome. Those two operations are exactly what an
//! adaptive photonic circuit (e.g. a photonic QCNN pooling layer) needs.
//!
//! [`FockKet`] carries the full vector of complex amplitudes over the Fock basis
//! of a fixed photon number, so it supports:
//!   * [`FockKet::evolve`]   — apply an interferometer unitary (`a†→U a†`);
//!   * [`FockKet::marginal`] — the probability distribution over the occupation
//!     of a *subset* of modes (partial-mode readout);
//!   * [`FockKet::measure`]  — project on a specific outcome on a subset of
//!     modes, returning its probability and the renormalised *conditional* ket
//!     (the unmeasured modes stay coherent);
//!   * [`FockKet::measure_all`] — every nonzero outcome + conditional ket, the
//!     full feed-forward branch set for adaptive injection.
//!
//! Amplitudes are exact (permanent-based, via [`crate::slos::fock_amplitude`]);
//! everything is deterministic and libtorch-free.

use num_complex::Complex64;

use crate::slos::{enumerate_fock_states, fock_amplitude, total_photons, FockState};

/// A pure linear-optical state: complex amplitudes over the Fock basis of a
/// fixed mode count and photon number.
#[derive(Clone, Debug)]
pub struct FockKet {
    pub n_modes: usize,
    pub n_photons: u32,
    /// Fock basis (lexicographic, from [`enumerate_fock_states`]).
    pub basis: Vec<FockState>,
    /// Amplitude per basis state (same order as `basis`).
    pub amps: Vec<Complex64>,
}

impl FockKet {
    /// A single basis state `|state⟩` with unit amplitude.
    pub fn from_fock(state: &FockState) -> Self {
        let n_modes = state.len();
        let n_photons = total_photons(state);
        let basis = enumerate_fock_states(n_modes, n_photons);
        let amps = basis
            .iter()
            .map(|b| {
                if b == state {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();
        Self {
            n_modes,
            n_photons,
            basis,
            amps,
        }
    }

    /// Probability per basis state, `|amp|²` (same order as `basis`).
    pub fn probabilities(&self) -> Vec<f64> {
        self.amps.iter().map(|a| a.norm_sqr()).collect()
    }

    /// Total norm² (1 for a normalised state).
    pub fn norm_sqr(&self) -> f64 {
        self.amps.iter().map(|a| a.norm_sqr()).sum()
    }

    /// Apply an `m×m` interferometer unitary: `|ψ'⟩ = φ(U)|ψ⟩`.
    ///
    /// `new_amp[t] = Σ_s ⟨t|φ(U)|s⟩ · amp[s]`, using the exact permanent kernel.
    pub fn evolve(&self, unitary: &[Vec<Complex64>]) -> FockKet {
        let basis = enumerate_fock_states(self.n_modes, self.n_photons);
        let mut amps = vec![Complex64::new(0.0, 0.0); basis.len()];
        for (ti, t) in basis.iter().enumerate() {
            let mut acc = Complex64::new(0.0, 0.0);
            for (s, &cs) in self.basis.iter().zip(&self.amps) {
                if cs.norm_sqr() < 1e-30 {
                    continue;
                }
                acc += fock_amplitude(unitary, s, t) * cs;
            }
            amps[ti] = acc;
        }
        FockKet {
            n_modes: self.n_modes,
            n_photons: self.n_photons,
            basis,
            amps,
        }
    }

    /// Marginal probability distribution over the occupation of `modes`
    /// (partial-mode readout): `Vec<(occupation_on_modes, probability)>`,
    /// keeping the unmeasured modes traced out. Only nonzero entries.
    pub fn marginal(&self, modes: &[usize]) -> Vec<(Vec<u32>, f64)> {
        let mut acc: Vec<(Vec<u32>, f64)> = Vec::new();
        for (b, a) in self.basis.iter().zip(&self.amps) {
            let p = a.norm_sqr();
            if p < 1e-15 {
                continue;
            }
            let key: Vec<u32> = modes.iter().map(|&m| b[m]).collect();
            match acc.iter_mut().find(|(k, _)| *k == key) {
                Some((_, pr)) => *pr += p,
                None => acc.push((key, p)),
            }
        }
        acc
    }

    /// Project on a specific `outcome` occupation on `modes` (a mid-circuit
    /// partial-mode measurement). Returns the outcome probability and the
    /// renormalised conditional ket — the unmeasured modes remain coherent.
    /// `None` if the outcome has zero probability.
    pub fn measure(&self, modes: &[usize], outcome: &[u32]) -> Option<(f64, FockKet)> {
        let mut amps = self.amps.clone();
        let mut prob = 0.0;
        for (i, b) in self.basis.iter().enumerate() {
            let matches = modes.iter().zip(outcome).all(|(&m, &o)| b[m] == o);
            if matches {
                prob += amps[i].norm_sqr();
            } else {
                amps[i] = Complex64::new(0.0, 0.0);
            }
        }
        if prob < 1e-15 {
            return None;
        }
        let scale = 1.0 / prob.sqrt();
        for a in &mut amps {
            *a *= scale;
        }
        Some((
            prob,
            FockKet {
                n_modes: self.n_modes,
                n_photons: self.n_photons,
                basis: self.basis.clone(),
                amps,
            },
        ))
    }

    /// Every nonzero outcome on `modes` with `(outcome, probability,
    /// conditional ket)` — the full branch set for adaptive feed-forward.
    /// Probabilities sum to 1 for a normalised state.
    pub fn measure_all(&self, modes: &[usize]) -> Vec<(Vec<u32>, f64, FockKet)> {
        self.marginal(modes)
            .into_iter()
            .filter_map(|(outcome, _)| {
                self.measure(modes, &outcome)
                    .map(|(p, ket)| (outcome, p, ket))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{self, PhotonicOp};
    use std::f64::consts::FRAC_PI_4;

    fn bs5050(m0: usize, m1: usize) -> PhotonicOp {
        PhotonicOp::BeamSplitterRx {
            mode0: m0,
            mode1: m1,
            theta: FRAC_PI_4,
            phi: 0.0,
        }
    }

    #[test]
    fn evolve_matches_slos_probabilities() {
        // FockKet.evolve probabilities must equal slos_full's distribution.
        let u = components::build_unitary(
            3,
            &[
                PhotonicOp::PhaseShifter { mode: 0, phi: 0.5 },
                bs5050(0, 1),
                PhotonicOp::BeamSplitterRx {
                    mode0: 1,
                    mode1: 2,
                    theta: 0.6,
                    phi: 0.3,
                },
            ],
        );
        let input = vec![1, 1, 0];
        let ket = FockKet::from_fock(&input).evolve(&u);
        let slos = crate::slos::slos_full(&u, &input);
        for (b, p) in ket.basis.iter().zip(ket.probabilities()) {
            let want = slos
                .iter()
                .find(|(s, _)| s == b)
                .map(|(_, q)| *q)
                .unwrap_or(0.0);
            assert!((p - want).abs() < 1e-10, "state {b:?}: {p} vs {want}");
        }
    }

    #[test]
    fn measure_branches_are_normalised() {
        // |1,1⟩ on a 50:50 BS ⇒ HOM: only |2,0⟩ and |0,2⟩, each prob 0.5.
        let u = components::build_unitary(2, &[bs5050(0, 1)]);
        let ket = FockKet::from_fock(&vec![1, 1]).evolve(&u);
        let branches = ket.measure_all(&[0]); // measure mode 0
        let total: f64 = branches.iter().map(|(_, p, _)| p).sum();
        assert!((total - 1.0).abs() < 1e-10, "branch probs sum to {total}");
        // measuring mode-0 = 2 photons ⇒ conditional state is |2,0⟩ deterministically
        let (_, p2, ket2) = branches.iter().find(|(o, _, _)| o[0] == 2).unwrap();
        assert!((p2 - 0.5).abs() < 1e-10);
        let amp20 = ket2
            .basis
            .iter()
            .zip(&ket2.amps)
            .find(|(b, _)| **b == vec![2, 0])
            .unwrap()
            .1;
        assert!((amp20.norm_sqr() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn partial_marginal_keeps_rest_coherent() {
        // Measure mode 0 of a 3-mode, 1-photon spread; the conditional ket on the
        // unmeasured modes must still be a valid normalised state.
        let u = components::build_unitary(3, &[bs5050(0, 1), bs5050(1, 2)]);
        let ket = FockKet::from_fock(&vec![1, 0, 0]).evolve(&u);
        let (p, cond) = ket.measure(&[0], &[0]).unwrap(); // photon NOT in mode 0
        assert!(p > 0.0 && p < 1.0);
        assert!((cond.norm_sqr() - 1.0).abs() < 1e-10);
    }
}
