// SPDX-License-Identifier: Apache-2.0
//! Continuous-variable photonics on a **truncated Fock space**.
//!
//! A CV mode lives in an infinite-dimensional Hilbert space. Simulating it
//! means cutting the ladder at some `cutoff`, and that cut is not a detail —
//! it is the dominant source of error and the thing this module is most careful
//! about.
//!
//! # Why truncation is the first thing built, not the last
//!
//! `fixes/PLAN-CV-BACKEND.md` R6 asks for a stated truncation policy, and the
//! reason is specific: clipping the top of the ladder **loses norm**. A state
//! that has leaked probability still returns perfectly plausible expectation
//! values — they are simply wrong, and nothing about them looks wrong. That is
//! the failure mode piquasso users are said to learn the hard way, and it is
//! the same shape as several defects already found in this repository:
//! confident output with nothing behind it.
//!
//! So [`FockState`] tracks its own norm loss and [`FockState::expect_n`]
//! **refuses** rather than returning a number from a leaking state.
//!
//! # What is here, and what is not
//!
//! Present: the Fock-space representation, photon-number readout, and the
//! truncation policy — all verifiable *today* against closed-form states.
//!
//! Absent: the gates. `Displacement` and `Squeezing` on a truncated space are
//! built from the Fréchet derivative of a matrix exponential of ladder
//! operators, which is where a subtle sign or ordering error would hide. Those
//! land next, and land against the anchors this module already checks:
//! displaced vacuum gives `⟨n⟩ = |α|²`, squeezed vacuum gives `⟨n⟩ = sinh²r`.
//! Building the measuring stick before the thing being measured is deliberate.

use num_complex::Complex64;

/// A single CV mode truncated at `cutoff` Fock levels: `|0⟩ .. |cutoff-1⟩`.
#[derive(Clone, Debug)]
pub struct FockState {
    amps: Vec<Complex64>,
    /// Probability mass discarded by truncation, accumulated.
    ///
    /// Mirrors `omega-backend-mps`'s `discarded_weight`: an explicit error
    /// budget beats an implicit assumption that truncation was harmless.
    lost_norm: f64,
}

/// Refuse a readout once this much probability has leaked past the cutoff.
///
/// 1e-6 is strict on purpose, and stricter than it first appears. **Lost mass
/// is a LOWER bound on the error in `⟨n⟩`, not an upper one:** the discarded
/// tail sits at high photon numbers, so losing mass `p` at level `k` removes
/// `k·p` from the numerator. The cross-check against piquasso measured the gap
/// directly — at `r = 0.3, cutoff 20` the lost mass is `3.6e-12` while the
/// resulting `⟨n⟩` error is `7.1e-11`, about 20× larger.
///
/// So this threshold must sit well below the accuracy actually wanted. A CV
/// expectation value is not obviously wrong when the state has leaked — it is
/// just wrong — so the check has to fire before the error reaches the answer.
pub const DEFAULT_LEAK_TOLERANCE: f64 = 1e-6;

/// Why a readout was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum CvError {
    /// Truncation discarded more probability than the caller allows.
    TruncationLeak { lost: f64, tolerance: f64 },
    /// The cutoff is too small to represent the requested state at all.
    CutoffTooSmall { cutoff: usize, needed: usize },
    /// A parameter has no finite representation (NaN, or a magnitude the
    /// truncated ladder cannot express).
    Unrepresentable { what: &'static str },
}

impl std::fmt::Display for CvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CvError::TruncationLeak { lost, tolerance } => write!(
                f,
                "truncation lost {lost:.3e} of the probability mass (tolerance {tolerance:.3e}) — \
                 the Fock cutoff is too low for this state. Expectation values from a leaking \
                 state look plausible and are wrong, so this is refused rather than reported. \
                 Raise the cutoff or reduce the squeezing/displacement magnitude."
            ),
            CvError::CutoffTooSmall { cutoff, needed } => write!(
                f,
                "cutoff {cutoff} cannot represent this state; it needs at least {needed} levels"
            ),
            CvError::Unrepresentable { what } => {
                write!(
                    f,
                    "parameter is not representable on the truncated space: {what}"
                )
            }
        }
    }
}

impl FockState {
    /// Vacuum `|0⟩` at the given cutoff.
    pub fn vacuum(cutoff: usize) -> Result<Self, CvError> {
        if cutoff == 0 {
            return Err(CvError::CutoffTooSmall { cutoff, needed: 1 });
        }
        let mut amps = vec![Complex64::new(0.0, 0.0); cutoff];
        amps[0] = Complex64::new(1.0, 0.0);
        Ok(Self {
            amps,
            lost_norm: 0.0,
        })
    }

    pub fn cutoff(&self) -> usize {
        self.amps.len()
    }

    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amps
    }

    /// Probability mass lost to truncation so far.
    pub fn lost_norm(&self) -> f64 {
        self.lost_norm
    }

    /// Build from unnormalised Fock amplitudes, recording what the cutoff cut.
    ///
    /// `tail_mass` is the probability the caller knows lies **beyond** the
    /// cutoff. Passing it explicitly is what makes the leak measurable: a
    /// closed-form state knows its own tail, and a state built by applying
    /// gates accumulates it as it goes.
    pub fn from_amplitudes(amps: Vec<Complex64>, tail_mass: f64) -> Result<Self, CvError> {
        if amps.is_empty() {
            return Err(CvError::CutoffTooSmall {
                cutoff: 0,
                needed: 1,
            });
        }
        if !tail_mass.is_finite() || tail_mass < 0.0 {
            return Err(CvError::Unrepresentable {
                what: "tail mass is not a finite non-negative number",
            });
        }
        if amps.iter().any(|a| !a.re.is_finite() || !a.im.is_finite()) {
            return Err(CvError::Unrepresentable {
                what: "amplitude is not finite",
            });
        }
        Ok(Self {
            amps,
            lost_norm: tail_mass,
        })
    }

    /// Squared norm of the represented (in-cutoff) part.
    pub fn norm_sqr(&self) -> f64 {
        self.amps.iter().map(|a| a.norm_sqr()).sum()
    }

    /// Mean photon number `⟨n⟩`, or a refusal if truncation has leaked.
    ///
    /// The leak check comes **first**. Returning a number and letting the
    /// caller consult `lost_norm` afterwards would be the same mistake as a
    /// governor reporting headroom it does not have: the plausible answer gets
    /// used and the caveat does not.
    pub fn expect_n(&self, tolerance: f64) -> Result<f64, CvError> {
        if self.lost_norm > tolerance {
            return Err(CvError::TruncationLeak {
                lost: self.lost_norm,
                tolerance,
            });
        }
        let norm = self.norm_sqr();
        if norm <= 0.0 {
            return Err(CvError::Unrepresentable {
                what: "state has zero norm",
            });
        }
        // Normalise against the represented mass: the cut tail is accounted for
        // by the leak check above, not by silently renormalising it away.
        let n: f64 = self
            .amps
            .iter()
            .enumerate()
            .map(|(k, a)| k as f64 * a.norm_sqr())
            .sum();
        Ok(n / norm)
    }

    /// Kerr interaction `exp(i·χ·n²)`.
    ///
    /// **Exact on the truncated space**, unlike displacement and squeezing.
    /// `n` is diagonal in the Fock basis, so `n²` is too, and the gate is a
    /// pure phase per level: no ladder is climbed, nothing falls past the
    /// cutoff, and no norm is lost. That is worth stating because it makes
    /// Kerr the one CV gate whose correctness owes nothing to the cutoff.
    pub fn kerr(&mut self, chi: f64) -> Result<(), CvError> {
        if !chi.is_finite() {
            return Err(CvError::Unrepresentable {
                what: "Kerr chi is not finite",
            });
        }
        for (k, a) in self.amps.iter_mut().enumerate() {
            let k2 = (k as f64) * (k as f64);
            *a *= Complex64::from_polar(1.0, chi * k2);
        }
        Ok(())
    }

    /// Phase rotation `exp(i·φ·n)` — the CV phase shifter.
    ///
    /// Also diagonal, so also exact and also norm-preserving.
    pub fn phase_shift(&mut self, phi: f64) -> Result<(), CvError> {
        if !phi.is_finite() {
            return Err(CvError::Unrepresentable {
                what: "phase is not finite",
            });
        }
        for (k, a) in self.amps.iter_mut().enumerate() {
            *a *= Complex64::from_polar(1.0, phi * (k as f64));
        }
        Ok(())
    }

    /// Coherent state `|α⟩` in the Fock basis: `e^{-|α|²/2} α^n / √(n!)`.
    ///
    /// Closed form, so it needs no gate machinery — which is precisely why it
    /// can serve as the anchor that *validates* the gate machinery later.
    /// Analytic result: `⟨n⟩ = |α|²`.
    pub fn coherent(alpha: Complex64, cutoff: usize) -> Result<Self, CvError> {
        if cutoff == 0 {
            return Err(CvError::CutoffTooSmall { cutoff, needed: 1 });
        }
        if !alpha.re.is_finite() || !alpha.im.is_finite() {
            return Err(CvError::Unrepresentable {
                what: "alpha is not finite",
            });
        }
        let n2 = alpha.norm_sqr();
        let mut amps = vec![Complex64::new(0.0, 0.0); cutoff];
        // Recurrence rather than powers/factorials: alpha^n and n! both
        // overflow long before the amplitude does, and the ratio is stable.
        let mut term = Complex64::new((-0.5 * n2).exp(), 0.0);
        amps[0] = term;
        for (k, slot) in amps.iter_mut().enumerate().skip(1) {
            term *= alpha / (k as f64).sqrt();
            *slot = term;
        }
        let kept: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
        // Total mass is 1 for a true coherent state, so whatever is missing
        // fell past the cutoff.
        let tail = (1.0 - kept).max(0.0);
        Self::from_amplitudes(amps, tail)
    }

    /// Squeezed vacuum `S(r)|0⟩` in the Fock basis (real `r`, zero phase).
    ///
    /// Only even levels are populated:
    /// `c_{2m} = √(cosh r)⁻¹ · √((2m)!)/(2^m m!) · (−tanh r)^m`.
    /// Analytic result: `⟨n⟩ = sinh²r`.
    pub fn squeezed_vacuum(r: f64, cutoff: usize) -> Result<Self, CvError> {
        if cutoff == 0 {
            return Err(CvError::CutoffTooSmall { cutoff, needed: 1 });
        }
        if !r.is_finite() {
            return Err(CvError::Unrepresentable {
                what: "squeezing parameter is not finite",
            });
        }
        let t = r.tanh();
        let mut amps = vec![Complex64::new(0.0, 0.0); cutoff];
        // Recurrence on the even sub-ladder:
        //   c_0 = 1/√(cosh r),  c_{2m} = c_{2m-2} · (−t) · √((2m−1)/(2m))
        let mut c = 1.0 / r.cosh().sqrt();
        amps[0] = Complex64::new(c, 0.0);
        let mut m = 1usize;
        while 2 * m < cutoff {
            let two_m = 2 * m;
            c *= -t * (((two_m - 1) as f64) / (two_m as f64)).sqrt();
            amps[two_m] = Complex64::new(c, 0.0);
            m += 1;
        }
        let kept: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
        let tail = (1.0 - kept).max(0.0);
        Self::from_amplitudes(amps, tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R5 anchor: displaced vacuum has `⟨n⟩ = |α|²`.
    ///
    /// This is the measuring stick for the displacement gate that does not
    /// exist yet — built first on purpose, so the gate is checked against
    /// something independent of how the gate is written.
    #[test]
    fn coherent_state_mean_photon_number_is_alpha_squared() {
        for (re, im) in [(0.5, 0.0), (1.0, 0.0), (0.0, 1.5), (1.2, -0.7)] {
            let alpha = Complex64::new(re, im);
            // Cutoff generous relative to |alpha|^2 so the tail is negligible.
            let st = FockState::coherent(alpha, 60).expect("builds");
            let got = st.expect_n(DEFAULT_LEAK_TOLERANCE).expect("no leak");
            let want = alpha.norm_sqr();
            assert!(
                (got - want).abs() < 1e-9,
                "alpha={alpha}: <n> = {got}, want |alpha|^2 = {want}"
            );
        }
    }

    /// R5 anchor: squeezed vacuum has `⟨n⟩ = sinh²r`.
    #[test]
    fn squeezed_vacuum_mean_photon_number_is_sinh_squared() {
        for r in [0.1, 0.3, 0.5, 0.8] {
            let st = FockState::squeezed_vacuum(r, 200).expect("builds");
            let got = st.expect_n(DEFAULT_LEAK_TOLERANCE).expect("no leak");
            let want = r.sinh().powi(2);
            assert!(
                (got - want).abs() < 1e-9,
                "r={r}: <n> = {got}, want sinh^2 r = {want}"
            );
        }
    }

    /// Squeezed vacuum populates only EVEN Fock levels. A gate implementation
    /// that leaks amplitude into odd levels is wrong in a way `⟨n⟩` alone can
    /// hide, so pin the structure as well as the number.
    #[test]
    fn squeezed_vacuum_populates_only_even_levels() {
        let st = FockState::squeezed_vacuum(0.6, 40).expect("builds");
        for (k, a) in st.amplitudes().iter().enumerate() {
            if k % 2 == 1 {
                assert!(a.norm() < 1e-15, "odd level {k} populated: {a}");
            }
        }
        assert!(st.amplitudes()[0].norm() > 0.5, "vacuum component missing");
    }

    /// R6, the property this module exists for: a state that has leaked past
    /// the cutoff must REFUSE to report `⟨n⟩`, not return a plausible number.
    #[test]
    fn a_leaking_state_refuses_to_report_a_number() {
        // |alpha|^2 = 100 mean photons against a cutoff of 10: the ladder is
        // cut far below where the state lives.
        let st = FockState::coherent(Complex64::new(10.0, 0.0), 10).expect("builds");
        assert!(
            st.lost_norm() > 0.5,
            "this state should have leaked badly, lost {}",
            st.lost_norm()
        );
        let err = st
            .expect_n(DEFAULT_LEAK_TOLERANCE)
            .expect_err("must refuse, not answer");
        match err {
            CvError::TruncationLeak { lost, .. } => assert!(lost > 0.5),
            other => panic!("expected TruncationLeak, got {other:?}"),
        }
        // And the refusal must say what to do about it.
        let msg = st.expect_n(DEFAULT_LEAK_TOLERANCE).unwrap_err().to_string();
        assert!(msg.contains("cutoff"), "{msg}");
    }

    /// The dangerous case, made explicit: a leaking state's `⟨n⟩` is not
    /// obviously wrong. Here it under-reports by ~90%, which is exactly why
    /// the refusal above is a refusal and not a warning.
    #[test]
    fn a_leaking_state_would_have_returned_a_plausible_but_wrong_number() {
        let st = FockState::coherent(Complex64::new(10.0, 0.0), 10).expect("builds");
        // Bypass the guard to see what the caller would have been told.
        let unguarded = st.expect_n(f64::INFINITY).expect("guard bypassed");
        let truth = 100.0_f64;
        assert!(
            unguarded < 0.2 * truth,
            "expected a badly wrong value, got {unguarded} vs {truth}"
        );
        assert!(
            unguarded.is_finite() && unguarded > 0.0,
            "and it is finite and positive — i.e. it looks like an answer: {unguarded}"
        );
    }

    /// Kerr and the phase shifter commute with `n`, so they change phases and
    /// leave photon statistics **exactly** alone. A gate that perturbs `⟨n⟩`
    /// here is wrong, and the check is exact rather than approximate because
    /// neither gate climbs the ladder.
    #[test]
    fn diagonal_gates_preserve_photon_statistics_exactly() {
        let alpha = Complex64::new(1.1, -0.4);
        let before = FockState::coherent(alpha, 60).expect("builds");
        let n0 = before.expect_n(DEFAULT_LEAK_TOLERANCE).expect("no leak");

        let mut st = before.clone();
        st.kerr(0.37).expect("finite chi");
        st.phase_shift(-1.2).expect("finite phi");
        let n1 = st.expect_n(DEFAULT_LEAK_TOLERANCE).expect("no leak");

        assert!((n1 - n0).abs() < 1e-12, "<n> moved: {n0} -> {n1}");
        assert!(
            (n1 - alpha.norm_sqr()).abs() < 1e-9,
            "and must still be |alpha|^2"
        );
    }

    /// Being diagonal, they lose no norm — the property that makes them exact
    /// on a truncated space where displacement and squeezing are not.
    #[test]
    fn diagonal_gates_lose_no_norm() {
        let mut st = FockState::coherent(Complex64::new(0.9, 0.3), 40).expect("builds");
        let before = st.norm_sqr();
        let leaked_before = st.lost_norm();
        st.kerr(2.5).expect("ok");
        st.phase_shift(0.8).expect("ok");
        assert!((st.norm_sqr() - before).abs() < 1e-14, "norm changed");
        assert_eq!(st.lost_norm(), leaked_before, "diagonal gates cannot leak");
    }

    /// `exp(i·χ·n²)` at `χ = 2π` is the identity on every level, since `k²` is
    /// an integer. A useful exact check on the phase convention itself.
    #[test]
    fn kerr_at_two_pi_is_the_identity() {
        let mut st = FockState::coherent(Complex64::new(1.0, 0.5), 30).expect("builds");
        let before: Vec<Complex64> = st.amplitudes().to_vec();
        st.kerr(2.0 * std::f64::consts::PI).expect("ok");
        for (a, b) in st.amplitudes().iter().zip(before.iter()) {
            assert!((a - b).norm() < 1e-12, "{a} vs {b}");
        }
    }

    /// A phase shift of 2*pi is likewise the identity; pi flips odd levels.
    #[test]
    fn phase_shift_conventions_are_pinned() {
        let mut st = FockState::coherent(Complex64::new(0.7, 0.0), 20).expect("builds");
        let before: Vec<Complex64> = st.amplitudes().to_vec();
        st.phase_shift(std::f64::consts::PI).expect("ok");
        for (k, (a, b)) in st.amplitudes().iter().zip(before.iter()).enumerate() {
            let want = if k % 2 == 0 { *b } else { -*b };
            assert!((a - want).norm() < 1e-12, "level {k}: {a} vs {want}");
        }
    }

    #[test]
    fn non_finite_gate_parameters_are_refused() {
        let mut st = FockState::vacuum(4).expect("builds");
        assert!(matches!(
            st.kerr(f64::NAN),
            Err(CvError::Unrepresentable { .. })
        ));
        assert!(matches!(
            st.phase_shift(f64::INFINITY),
            Err(CvError::Unrepresentable { .. })
        ));
    }

    /// Cross-check against **piquasso 8.0.1** (`tools/cv_cross_check/`), an
    /// independent implementation of truncated-Fock CV simulation.
    ///
    /// The fixtures are piquasso's own `⟨n⟩` for squeezed vacuum at **cutoff
    /// 20**, and the interesting part is that they are *not* all `sinh²r`:
    ///
    /// | r | piquasso − sinh²r |
    /// |---|---|
    /// | 0.1 | 0 |
    /// | 0.3 | 7e-11 |
    /// | 0.5 | 7.8e-7 |
    /// | 0.8 | **1.3e-3** |
    ///
    /// That growing gap *is* truncation error. piquasso returns 0.7874 for
    /// r=0.8 where the true answer is 0.7887, with nothing in the return value
    /// to say so — exactly the silent-leak failure R6 exists to prevent.
    ///
    /// So this asserts two things at once: that this implementation agrees with
    /// piquasso where the cutoff is adequate, and that its **leak metric
    /// predicts where piquasso stops being right**.
    #[test]
    fn matches_piquasso_and_predicts_where_piquasso_goes_wrong() {
        // (r, piquasso <n> at cutoff 20)
        let fixtures = [
            (0.1_f64, 0.010_033_377_809_537_924_f64),
            (0.3, 0.092_732_609_049_771_14),
            (0.5, 0.271_539_533_552_898_34),
            (0.8, 0.787_422_536_949_700_4),
        ];
        for (r, pq_n) in fixtures {
            let st = FockState::squeezed_vacuum(r, 20).expect("builds");
            let leak = st.lost_norm();
            let analytic = r.sinh().powi(2);
            let pq_err = (pq_n - analytic).abs();

            // Agreement with an INDEPENDENT implementation, on its own terms:
            // compare against piquasso at the same cutoff, using the same
            // truncated normalisation.
            let ours = st
                .expect_n(f64::INFINITY)
                .expect("unguarded for comparison");
            assert!(
                (ours - pq_n).abs() < 1e-9,
                "r={r}: ours {ours} vs piquasso {pq_n}"
            );

            // The leak metric must MOVE WITH piquasso's error — that is what
            // makes it a usable guard rather than a decorative counter.
            //
            // But note the direction, which this test discovered: lost MASS
            // UNDERESTIMATES the error in <n>. The discarded tail sits at high
            // photon numbers, so losing mass p at level k removes k*p from the
            // numerator — the <n> error is mass weighted by k, hence larger.
            // At r=0.3 the mass lost is 3.6e-12 while piquasso is off by
            // 7.1e-11, a factor of ~20.
            //
            // So `lost_norm` is a LOWER BOUND on the <n> error, never an upper
            // one, and DEFAULT_LEAK_TOLERANCE must be read with that in mind:
            // it is deliberately far stricter than the accuracy actually
            // wanted. Asserting the reverse would have encoded the mistake.
            if pq_err > 0.0 {
                assert!(
                    leak > 0.0,
                    "r={r}: piquasso is off by {pq_err:.3e} so some mass must have been lost"
                );
            }

            // The decisive case: where piquasso is visibly wrong, we refuse.
            if pq_err > DEFAULT_LEAK_TOLERANCE {
                assert!(
                    st.expect_n(DEFAULT_LEAK_TOLERANCE).is_err(),
                    "r={r}: piquasso is off by {pq_err:.3e} here and we must refuse, not agree"
                );
            }
        }
    }

    /// The same states at an adequate cutoff agree with BOTH piquasso's target
    /// and the analytic value — confirming the refusals above are about the
    /// cutoff, not about the state being unrepresentable in principle.
    #[test]
    fn a_generous_cutoff_removes_the_disagreement() {
        for r in [0.5_f64, 0.8] {
            let st = FockState::squeezed_vacuum(r, 200).expect("builds");
            let n = st
                .expect_n(DEFAULT_LEAK_TOLERANCE)
                .expect("no leak at cutoff 200");
            assert!((n - r.sinh().powi(2)).abs() < 1e-9, "r={r}: {n}");
        }
    }

    #[test]
    fn vacuum_has_no_photons_and_no_leak() {
        let st = FockState::vacuum(8).expect("builds");
        assert_eq!(st.lost_norm(), 0.0);
        assert!(st.expect_n(DEFAULT_LEAK_TOLERANCE).unwrap().abs() < 1e-15);
    }

    #[test]
    fn degenerate_inputs_are_refused_not_papered_over() {
        assert!(matches!(
            FockState::vacuum(0),
            Err(CvError::CutoffTooSmall { .. })
        ));
        assert!(matches!(
            FockState::coherent(Complex64::new(f64::NAN, 0.0), 8),
            Err(CvError::Unrepresentable { .. })
        ));
        assert!(matches!(
            FockState::squeezed_vacuum(f64::INFINITY, 8),
            Err(CvError::Unrepresentable { .. })
        ));
    }
}
