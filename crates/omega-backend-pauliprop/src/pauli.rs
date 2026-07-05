//! Pauli algebra for the propagation engine.
//!
//! A Pauli string is stored in **raw symplectic form**: two bit vectors
//! `(x, z)` with the operator meaning
//!
//! ```text
//!   raw(x, z) = ∏_q  X_q^{x_q} Z_q^{z_q}        (q in increasing order)
//! ```
//!
//! The raw product of two such strings only ever differs by a **sign**
//! `(-1)^{Σ z₁·x₂}` (no factor of `i`), so every `i` — the one in `Y = i·XZ`,
//! the one from a Pauli rotation's `i sinθ` branch — is folded into a
//! `Complex64` **coefficient** carried alongside the string. That keeps the
//! string itself a clean hashable key, and a sum of Paulis a plain map.

use num_complex::Complex64;

/// A raw Pauli string `∏_q X_q^{x_q} Z_q^{z_q}` (no phase — phase lives in the
/// coefficient of whatever sum holds it). Hashable, so it keys a [`PauliSum`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PauliKey {
    pub x: Vec<bool>,
    pub z: Vec<bool>,
}

impl PauliKey {
    pub fn identity(n: usize) -> Self {
        Self {
            x: vec![false; n],
            z: vec![false; n],
        }
    }

    /// Number of qubits on which this string is not the identity.
    pub fn weight(&self) -> usize {
        self.x
            .iter()
            .zip(&self.z)
            .filter(|(a, b)| **a || **b)
            .count()
    }

    /// Does this string act as `I` or `Z` on every qubit (so `⟨0…0|·|0…0⟩ ≠ 0`)?
    pub fn is_all_iz(&self) -> bool {
        self.x.iter().all(|&b| !b)
    }
}

/// Raw product `(x1,z1)·(x2,z2)` → `(x,z)` plus the `±1` sign from commuting
/// the `Z₁` factors past the `X₂` factors: `Z^{z₁}X^{x₂} = (-1)^{z₁·x₂}X^{x₂}Z^{z₁}`.
pub fn mul_raw(x1: &[bool], z1: &[bool], x2: &[bool], z2: &[bool]) -> (PauliKey, f64) {
    let n = x1.len();
    let mut x = vec![false; n];
    let mut z = vec![false; n];
    let mut neg = 0u32;
    for q in 0..n {
        x[q] = x1[q] ^ x2[q];
        z[q] = z1[q] ^ z2[q];
        if z1[q] && x2[q] {
            neg += 1;
        }
    }
    let sign = if neg.is_multiple_of(2) { 1.0 } else { -1.0 };
    (PauliKey { x, z }, sign)
}

/// A term's coefficient plus its **split frequency**: the number of sin-branches
/// (non-Clifford splits) the cheapest path to this Pauli has taken. This is the
/// axis PauliPropagation.jl's `max_freq` truncation acts on. When two paths reach
/// the same Pauli, the coefficients add and the frequency is the **minimum** —
/// the term is reachable within the smaller budget, so that's the honest bound.
#[derive(Clone, Copy, Debug)]
pub struct Weighted {
    pub coeff: Complex64,
    pub freq: u32,
}

/// A weighted sum of Pauli strings `Σ cₚ · P`, the observable as it propagates.
#[derive(Clone, Debug, Default)]
pub struct PauliSum {
    pub terms: std::collections::HashMap<PauliKey, Weighted>,
    /// L1 mass of coefficients dropped by truncation so far (an error budget).
    pub dropped_mass: f64,
}

impl PauliSum {
    pub fn new() -> Self {
        Self {
            terms: std::collections::HashMap::new(),
            dropped_mass: 0.0,
        }
    }

    /// Add `coeff · P` at split-frequency 0 (Clifford / seed terms).
    pub fn add(&mut self, key: PauliKey, coeff: Complex64) {
        self.add_weighted(key, coeff, 0);
    }

    /// Add `coeff · P` carrying split-frequency `freq`, merging onto any existing
    /// entry: coefficients sum, frequency takes the minimum.
    pub fn add_weighted(&mut self, key: PauliKey, coeff: Complex64, freq: u32) {
        self.terms
            .entry(key)
            .and_modify(|w| {
                w.coeff += coeff;
                w.freq = w.freq.min(freq);
            })
            .or_insert(Weighted { coeff, freq });
    }

    /// Iterate terms in a canonical (symplectic-key-sorted) order. The
    /// propagation `HashMap` is order-nondeterministic; callers that need
    /// reproducibility (GPU batch layout, top-K decisions) go through this.
    pub fn sorted_terms(&self) -> Vec<(&PauliKey, &Weighted)> {
        let mut v: Vec<_> = self.terms.iter().collect();
        v.sort_by(|(a, _), (b, _)| a.x.cmp(&b.x).then_with(|| a.z.cmp(&b.z)));
        v
    }

    /// Drop terms below `coeff_min` (by magnitude), above `max_weight` (Pauli
    /// weight), or above `max_freq` (split frequency); accumulate the dropped
    /// magnitude into `dropped_mass`.
    pub fn truncate(&mut self, coeff_min: f64, max_weight: Option<usize>, max_freq: Option<u32>) {
        let mut dropped = 0.0;
        self.terms.retain(|k, w| {
            let too_small = w.coeff.norm() < coeff_min;
            let too_heavy = max_weight.is_some_and(|m| k.weight() > m);
            let too_deep = max_freq.is_some_and(|m| w.freq > m);
            if too_small || too_heavy || too_deep {
                dropped += w.coeff.norm();
                false
            } else {
                true
            }
        });
        self.dropped_mass += dropped;
    }
}
