//! Hardware-flavored Pauli noise models for the transversal QEC layer.
//!
//! At the *code-capacity* altitude a logical memory is exposed to an
//! independent per-data-qubit Pauli channel: each data qubit suffers an X
//! (bit-flip) with probability `p_bit` and a Z (phase-flip) with probability
//! `p_phase`. This is exactly the channel the surface code's two CSS sectors
//! decode independently (see [`crate::ecc::run`]), so it drives the
//! logical-error-rate curves in [`super::memory`].
//!
//! The hardware presets bias that channel to match the dominant error mechanism
//! of each platform:
//! * [`NoiseModel::neutral_atom`] — ZZ-biased dephasing (QuEra Transversal STAR:
//!   ZZ-biased two-qubit errors + spectator errors from global pulses + atom
//!   loss ≈ YY), so phase errors dominate (`p_phase > p_bit`).
//! * [`NoiseModel::trapped_ion`] — transport/shuttling-dominated, closer to
//!   depolarizing (Quantinuum), so `p_bit ≈ p_phase`.
//!
//! Circuit-level (per-gate) channels used by the full physical expansion live
//! with the gadget compiler; this module is the memory-channel foundation and
//! the home of the shared seeded RNG.

/// Deterministic SplitMix64 → uniform `f64` in `[0, 1)`. The single seeded
/// stream shared across the QEC Monte-Carlo code so every backend / run replays
/// identical error draws. (Moved here from `ecc::run`; that module delegates.)
pub fn splitmix64(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Single-data-qubit Pauli channel (code-capacity).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PauliChannel {
    pub p_x: f64,
    pub p_y: f64,
    pub p_z: f64,
}

impl PauliChannel {
    pub fn ideal() -> Self {
        Self {
            p_x: 0.0,
            p_y: 0.0,
            p_z: 0.0,
        }
    }
    /// Symmetric depolarizing: total single-qubit error probability `p`, split
    /// evenly over X/Y/Z.
    pub fn depolarizing(p: f64) -> Self {
        Self {
            p_x: p / 3.0,
            p_y: p / 3.0,
            p_z: p / 3.0,
        }
    }
    /// Total single-qubit error probability.
    pub fn total(&self) -> f64 {
        self.p_x + self.p_y + self.p_z
    }
    /// Probability the qubit picks up a bit-flip (X or Y component).
    pub fn bit_flip_prob(&self) -> f64 {
        self.p_x + self.p_y
    }
    /// Probability the qubit picks up a phase-flip (Z or Y component).
    pub fn phase_flip_prob(&self) -> f64 {
        self.p_z + self.p_y
    }
}

/// A code-capacity noise model: independent bit-flip / phase-flip probabilities
/// per data qubit, plus documented hardware provenance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseModel {
    /// Human-readable preset name.
    pub name: &'static str,
    /// P(X on a data qubit).
    pub p_bit: f64,
    /// P(Z on a data qubit).
    pub p_phase: f64,
    /// Dephasing bias `p_phase / p_bit` recorded for reporting (1.0 = unbiased).
    pub zz_bias: f64,
}

impl NoiseModel {
    pub fn ideal() -> Self {
        Self {
            name: "ideal",
            p_bit: 0.0,
            p_phase: 0.0,
            zz_bias: 1.0,
        }
    }

    /// Unbiased independent bit/phase flip at rate `p` (matches the legacy
    /// `ecc::run::monte_carlo` channel).
    pub fn depolarizing(p: f64) -> Self {
        Self {
            name: "depolarizing",
            p_bit: p,
            p_phase: p,
            zz_bias: 1.0,
        }
    }

    /// Neutral-atom (Transversal STAR) preset: ZZ-biased dephasing. Phase errors
    /// dominate — bit flips suppressed by the bias factor `2`.
    pub fn neutral_atom(p: f64) -> Self {
        Self {
            name: "neutral-atom",
            p_bit: p * 0.5,
            p_phase: p,
            zz_bias: 2.0,
        }
    }

    /// Trapped-ion (Quantinuum) preset: transport-dominated, near-depolarizing
    /// with a mild dephasing bias.
    pub fn trapped_ion(p: f64) -> Self {
        Self {
            name: "trapped-ion",
            p_bit: p,
            p_phase: p * 1.2,
            zz_bias: 1.2,
        }
    }

    /// Parse a preset name at physical rate `p`.
    pub fn preset(name: &str, p: f64) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "ideal" => Some(Self::ideal()),
            "depolarizing" | "depol" => Some(Self::depolarizing(p)),
            "neutral-atom" | "neutral_atom" | "atom" => Some(Self::neutral_atom(p)),
            "trapped-ion" | "trapped_ion" | "ion" => Some(Self::trapped_ion(p)),
            _ => None,
        }
    }

    /// Sample per-data-qubit X-error and Z-error supports from the shared RNG.
    /// Draws two uniforms per qubit (bit, then phase) so streams stay aligned
    /// with the legacy sampler.
    pub fn sample_data_errors(&self, n_data: usize, rng: &mut u64) -> (Vec<usize>, Vec<usize>) {
        let mut x_err = Vec::new();
        let mut z_err = Vec::new();
        for q in 0..n_data {
            if splitmix64(rng) < self.p_bit {
                x_err.push(q);
            }
            if splitmix64(rng) < self.p_phase {
                z_err.push(q);
            }
        }
        (x_err, z_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_is_deterministic_and_in_range() {
        let mut a = 12345u64;
        let mut b = 12345u64;
        for _ in 0..1000 {
            let x = splitmix64(&mut a);
            let y = splitmix64(&mut b);
            assert_eq!(x, y);
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn presets_have_expected_bias() {
        let na = NoiseModel::neutral_atom(1e-2);
        assert!(na.p_phase > na.p_bit, "neutral-atom should be phase-biased");
        let depol = NoiseModel::depolarizing(1e-2);
        assert_eq!(depol.p_bit, depol.p_phase);
        assert_eq!(NoiseModel::ideal().p_bit, 0.0);
    }

    #[test]
    fn sampler_rate_matches_probability() {
        // Empirical bit-flip frequency ≈ p_bit over many qubits/draws.
        let nm = NoiseModel::depolarizing(0.1);
        let mut rng = 0xDEADBEEFu64;
        let n = 200_000;
        let (x, _z) = nm.sample_data_errors(n, &mut rng);
        let freq = x.len() as f64 / n as f64;
        assert!((freq - 0.1).abs() < 5e-3, "freq = {freq}");
    }
}
