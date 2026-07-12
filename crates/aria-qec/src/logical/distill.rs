//! Magic-state distillation — the non-Clifford resource.
//!
//! Transversal codes give the Clifford group cheaply, but a universal gate set
//! needs a non-Clifford resource: a *magic state* (e.g. |T⟩ = T|+⟩) injected via
//! gate teleportation. Raw magic states are prepared with a relatively high
//! error `p_in`; **distillation** trades many noisy copies for one much cleaner
//! copy by post-selecting on stabilizer checks of a small code.
//!
//! We model the canonical **15-to-1** protocol (Bravyi–Kitaev, equivalently the
//! punctured Reed–Muller [[15,1,3]] code — the same code Quantinuum switches
//! into for a transversal T). To leading order it maps input infidelity `p` to
//!
//! ```text
//!   p_out ≈ 35 · p³            (cubic error suppression)
//!   accept ≈ (1 − p)¹⁵         (all 15 inputs pass the checks)
//! ```
//!
//! and can be concatenated over several `rounds` for deeper suppression. This is
//! the analytic resource model; it feeds both the physical T-gadget (as the
//! [`PauliChannel`] injected on teleportation) and the effective-logical-channel
//! path. Distilling the actual 15-qubit circuit on a backend is a later option.

use super::noise::PauliChannel;

/// A 15-to-1 distillation protocol. Both variants share the [[15,1,3]] code and
/// therefore the same leading-order error polynomial; they differ only in the
/// (documented) provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistillProtocol {
    /// Bravyi–Kitaev 15-to-1 |T⟩ distillation.
    BravyiKitaev15to1,
    /// Reed–Muller [[15,1,3]] distillation (transversal-T code switch).
    ReedMuller15to1,
}

impl DistillProtocol {
    /// Number of raw input magic states per distilled output.
    pub fn n_input(&self) -> usize {
        15
    }
    /// Leading coefficient `a` in `p_out ≈ a · p_in³`.
    pub fn cubic_coeff(&self) -> f64 {
        35.0
    }
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "15to1-bk" | "bravyi-kitaev" | "bk" => Some(Self::BravyiKitaev15to1),
            "15to1-rm" | "reed-muller" | "rm" => Some(Self::ReedMuller15to1),
            _ => None,
        }
    }
}

/// A concatenated 15-to-1 distillation factory at a given raw input infidelity.
#[derive(Clone, Copy, Debug)]
pub struct MagicStateProtocol {
    pub protocol: DistillProtocol,
    /// Infidelity `p_in` of a raw (undistilled) magic state.
    pub input_infidelity: f64,
    /// Concatenation levels (≥ 1).
    pub rounds: usize,
}

impl MagicStateProtocol {
    pub fn new(protocol: DistillProtocol, input_infidelity: f64, rounds: usize) -> Self {
        Self {
            protocol,
            input_infidelity,
            rounds,
        }
    }

    /// One distillation level: `p ↦ a · p³`.
    fn step(&self, p: f64) -> f64 {
        self.protocol.cubic_coeff() * p * p * p
    }

    /// Output infidelity after `rounds` concatenated levels.
    pub fn output_infidelity(&self) -> f64 {
        let mut p = self.input_infidelity;
        for _ in 0..self.rounds {
            p = self.step(p);
        }
        p
    }

    /// Post-selection acceptance probability (product over levels of the
    /// per-level `(1 − p)¹⁵`, with `p` updated each level).
    pub fn acceptance(&self) -> f64 {
        let mut p = self.input_infidelity;
        let mut acc = 1.0;
        for _ in 0..self.rounds {
            acc *= (1.0 - p).powi(15);
            p = self.step(p);
        }
        acc
    }

    /// Expected raw states consumed per accepted output — `15` per level scaled
    /// up by rejections (`1/accept` at that level), concatenated.
    pub fn raw_states_per_output(&self) -> f64 {
        let mut ratio = 1.0;
        let mut p = self.input_infidelity;
        for _ in 0..self.rounds {
            ratio *= 15.0 / (1.0 - p).powi(15);
            p = self.step(p);
        }
        ratio
    }

    /// The residual Pauli channel injected when the distilled |T⟩ is consumed by
    /// a T gadget via gate teleportation: a logical Z (phase) error at rate
    /// `output_infidelity`.
    pub fn injected_channel(&self) -> PauliChannel {
        PauliChannel {
            p_x: 0.0,
            p_y: 0.0,
            p_z: self.output_infidelity(),
        }
    }

    /// The distillation *threshold*: raw states are improved only when
    /// `p_in < 1/√a`. Above it, distillation makes things worse.
    pub fn threshold(&self) -> f64 {
        1.0 / self.protocol.cubic_coeff().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REL: f64 = 0.01;

    fn rel_close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-300)
    }

    #[test]
    fn single_round_is_cubic() {
        for &p in &[1e-2_f64, 1e-3, 5e-3] {
            let f = MagicStateProtocol::new(DistillProtocol::BravyiKitaev15to1, p, 1);
            assert!(
                rel_close(f.output_infidelity(), 35.0 * p * p * p, REL),
                "p_out({p}) = {}",
                f.output_infidelity()
            );
        }
    }

    #[test]
    fn acceptance_is_one_minus_p_to_the_fifteen() {
        for &p in &[1e-2_f64, 1e-3] {
            let f = MagicStateProtocol::new(DistillProtocol::ReedMuller15to1, p, 1);
            assert!(
                rel_close(f.acceptance(), (1.0 - p).powi(15), REL),
                "accept({p}) = {}",
                f.acceptance()
            );
        }
    }

    #[test]
    fn two_rounds_compose() {
        let p = 1e-2;
        let one = MagicStateProtocol::new(DistillProtocol::BravyiKitaev15to1, p, 1);
        let two = MagicStateProtocol::new(DistillProtocol::BravyiKitaev15to1, p, 2);
        let expected = 35.0 * one.output_infidelity().powi(3);
        assert!(rel_close(two.output_infidelity(), expected, 1e-9));
        // Deeper suppression: two rounds beats one.
        assert!(two.output_infidelity() < one.output_infidelity());
    }

    #[test]
    fn suppresses_below_threshold_only() {
        let f = MagicStateProtocol::new(DistillProtocol::BravyiKitaev15to1, 1e-2, 1);
        assert!(f.output_infidelity() < f.input_infidelity);
        // Above threshold (~0.169) distillation increases error.
        let bad = MagicStateProtocol::new(DistillProtocol::BravyiKitaev15to1, 0.3, 1);
        assert!(bad.output_infidelity() > bad.input_infidelity);
        assert!((f.threshold() - 1.0 / 35.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn injected_channel_is_phase_only() {
        let f = MagicStateProtocol::new(DistillProtocol::BravyiKitaev15to1, 1e-3, 1);
        let ch = f.injected_channel();
        assert_eq!(ch.p_x, 0.0);
        assert_eq!(ch.p_y, 0.0);
        assert!((ch.p_z - f.output_infidelity()).abs() < 1e-18);
    }
}
