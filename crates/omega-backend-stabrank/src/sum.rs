// SPDX-License-Identifier: Apache-2.0
//! The sum over Cliffords: `|ψ⟩ = Σᵢ cᵢ|φᵢ⟩` over CH-form branches, and the
//! χ² pairwise readout `Σᵢⱼ c̄ᵢcⱼ⟨φᵢ|O|φⱼ⟩`.
//!
//! # The one mechanism
//!
//! Every gate this engine executes is a product of **Pauli rotations**
//! `exp(−iφ/2 P)` times a global phase, and one rotation expands as
//!
//! ```text
//! exp(−iφ/2 P) = cos(φ/2)·I  −  i sin(φ/2)·P.
//! ```
//!
//! `I` and `P` are both Clifford, so a rotation turns each branch into two
//! branches of the *same* kind — the 2-term split of PLAN-MAJORANA-STIM.md
//! §S1, and the naive rank `χ = 2^t` of §1.1. `T = e^{iπ/8} exp(−iπ/8 Z)`,
//! `T† = e^{−iπ/8} exp(+iπ/8 Z)` and `Rz(θ) = exp(−iθ/2 Z)` are the same
//! rotation at three angles; nothing in [`StabilizerSum`] knows which gate it
//! came from.
//!
//! The split is exact at **every** angle. The approximation, when there is
//! one, is [`StabilizerSum::truncate`] alone: `χ` is the whole cost, and
//! dropping branches is what buys a sub-`2^t` rank. Every drop is banked in
//! [`StabilizerSum::dropped_mass`], which is the `m` the certificate's bound
//! `R·m·(2+m)` is a function of.
//!
//! # Quarter turns do not split
//!
//! At `φ = kπ/2` the rotation is itself Clifford and is applied to each
//! branch in place, leaving `χ` alone. That is the §1.4 cost claim made real:
//! a Clifford gate is free for this engine, and a Givens rotation `Rbs(θ)` is
//! Clifford exactly at multiples of `π/2`.
//!
//! **The Clifford angles are recognised by exact `f64` equality against
//! `k·π/2`, not by a tolerance.** Both paths are exact, so this looks like a
//! free choice, and it is not: reading an angle that is merely *near* `π/2`
//! as `π/2` would move the answer by the angle error, silently. Exact
//! equality cannot do that — a missed recognition costs one branch and
//! nothing else, which is the direction an error is allowed to go.
//!
//! The cases, with `V` any Clifford taking `P` to `Z_a` ([`sqrt_pauli`]
//! builds one) and `√P = (I − iP)/√2 = e^{−iπ/4} V† S_a V`:
//!
//! ```text
//! k ≡ 0:  I          k ≡ 4:  −I
//! k ≡ 1:  √P         k ≡ 5:  −√P
//! k ≡ 2:  −iP        k ≡ 6:  +iP
//! k ≡ 3:  −√P†       k ≡ 7:  +√P†
//! ```
//!
//! **Reduced mod 8, not mod 4.** `exp(−iφ/2 P)` has period `4π` in `φ`, not
//! `2π`: `P² = I` makes it `cos(φ/2)·I − i sin(φ/2)·P`, and a half-angle has
//! twice the period. Folding `k` mod 4 loses the sign on the second lap —
//! `Rz(2π) = −I`, not `I`. That sign is a scalar on a *branch*, so in a sum
//! over Cliffords it is a relative phase between branches and does not cancel.
//!
//! # The readout, and why the whole square
//!
//! `O` Hermitian makes `⟨φⱼ|O|φᵢ⟩ = conj(⟨φᵢ|O|φⱼ⟩)`, so half the Gram matrix
//! determines the other half and the sum could be done in half the time. It
//! is not, because that symmetry is **the thing being checked**: summed over
//! the full square the imaginary part has to cancel to rounding, and
//! [`StabRankBackend::expectation`] refuses when it does not. Build the
//! symmetry in and that check becomes vacuous — it would report `0.0i` on a
//! kernel whose phases were wrong.
//!
//! [`StabRankBackend::expectation`]: crate::StabRankBackend::expectation

use num_complex::Complex64;
use omega_core::error::{OmegaError, Result};
use omega_core::executor::{Observable, PauliOp};

use crate::chform::ChForm;

/// A site of a Pauli string: one wire, one letter.
pub type PauliSite = (usize, PauliOp);

/// `Σ|cᵢ|` over an observable's terms — the range `⟨O⟩` is confined to a
/// priori, since `|⟨P⟩| ≤ 1` for every Pauli string.
///
/// The same quantity `omega_backend_majoranaprop::engine::observable_l1_norm`
/// computes, deliberately recomputed rather than imported: the two backends
/// are siblings with no dependency edge between them, and inventing one so
/// that a sum of absolute values has a single home would be the larger cost.
/// What keeps them from drifting is a test, not a shared symbol —
/// `tests/three_way_fermionic.rs` asserts the two engines report the same
/// `observable_range` on the same post-JW observable.
pub fn observable_l1_norm(observable: &Observable) -> f64 {
    observable.terms.iter().map(|(c, _)| c.abs()).sum()
}

/// `Some(k mod 8)` when `φ` is **exactly** `k·π/2` in `f64`; `None` otherwise.
///
/// Exact equality, and mod 8, by the two arguments in this module's doc
/// comment.
pub(crate) fn quarter_turns(phi: f64) -> Option<u32> {
    use std::f64::consts::FRAC_PI_2;
    let k = (phi / FRAC_PI_2).round();
    // `rem_euclid` on a float past 2^53 has no integer meaning left, and an
    // angle that large is not a Clifford angle anyone wrote on purpose.
    if !k.is_finite() || k.abs() > 9e15 || k * FRAC_PI_2 != phi {
        return None;
    }
    Some(k.rem_euclid(8.0) as u32)
}

/// Apply a Pauli string to one branch. The letters sit on distinct wires and
/// commute, so the product is the gate sequence in any order — and
/// [`ChForm::y_gate`] carries its own `i`, so no phase is owed on top.
fn apply_pauli(state: &mut ChForm, sites: &[PauliSite]) {
    for &(q, p) in sites {
        match p {
            PauliOp::I => {}
            PauliOp::X => state.x_gate(q),
            PauliOp::Y => state.y_gate(q),
            PauliOp::Z => state.z_gate(q),
        }
    }
}

/// `exp(∓iπ/4 P) = (I ∓ iP)/√2` on one branch — the Clifford square root of
/// the Pauli string `P`, for `dagger = false` / `true` respectively.
///
/// `P = V† Z_a V` for the Clifford `V = L·B` built below, so
/// `exp(∓iπ/4 P) = V† exp(∓iπ/4 Z_a) V = e^{∓iπ/4} V† S_a^{(†)} V`:
///
/// * `B = ⊗_q B_q` takes each letter to `Z`. `H X H = Z`; for `Y`,
///   `B = H S†` and `S† Y S = X`, so `B Y B† = H X H = Z`. Applying `B` to a
///   state is `S†` then `H` on that wire, and `B†` is `H` then `S`.
/// * `L = ∏_{q ≠ a} CX(q → a)` collects the result onto the pivot `a`:
///   `CX(q→a)` conjugates `Z_a` to `Z_q Z_a` and leaves `Z_q` alone, so
///   `L (Z_a ∏_q Z_q) L† = Z_a`. The factors share a target and commute, and
///   each is its own inverse, so `L† = L`.
///
/// The leftover scalar is where the Clifford-angle rotation keeps its global
/// phase: `exp(−iπ/4 Z) = diag(e^{−iπ/4}, e^{iπ/4}) = e^{−iπ/4}·S`, and
/// dropping the `e^{∓iπ/4}` would be a per-*branch* phase error, which is
/// exactly the class of defect a single-state check cannot see.
fn sqrt_pauli(state: &mut ChForm, sites: &[PauliSite], dagger: bool) {
    let pivot = sites[0].0;
    for &(q, p) in sites {
        match p {
            PauliOp::X => state.h_gate(q),
            PauliOp::Y => {
                state.sdg_gate(q);
                state.h_gate(q);
            }
            _ => {}
        }
    }
    for &(q, _) in sites {
        if q != pivot {
            state.cx_gate(q, pivot);
        }
    }
    if dagger {
        state.sdg_gate(pivot);
    } else {
        state.s_gate(pivot);
    }
    for &(q, _) in sites {
        if q != pivot {
            state.cx_gate(q, pivot);
        }
    }
    for &(q, p) in sites {
        match p {
            PauliOp::X => state.h_gate(q),
            PauliOp::Y => {
                state.h_gate(q);
                state.s_gate(q);
            }
            _ => {}
        }
    }
    let sign = if dagger { 1.0 } else { -1.0 };
    state.scale_phase(Complex64::from_polar(
        1.0,
        sign * std::f64::consts::FRAC_PI_4,
    ));
}

/// `|ψ⟩ = Σᵢ cᵢ|φᵢ⟩`: a stabilizer-rank decomposition in the Schrödinger
/// picture. `χ` is [`StabilizerSum::chi`], the number of branches.
#[derive(Debug, Clone)]
pub struct StabilizerSum {
    n: usize,
    branches: Vec<(Complex64, ChForm)>,
    peak_chi: usize,
    max_branches: usize,
    coeff_min: f64,
    max_chi: Option<usize>,
    dropped_mass: f64,
}

impl StabilizerSum {
    /// `|0…0⟩` as a one-branch sum, exact: no coefficient floor, no `χ` cut.
    pub fn zero(n: usize, max_branches: usize) -> Self {
        Self {
            n,
            branches: vec![(Complex64::new(1.0, 0.0), ChForm::zero(n))],
            peak_chi: 1,
            max_branches,
            coeff_min: 0.0,
            max_chi: None,
            dropped_mass: 0.0,
        }
    }

    /// Truncating sum: after every split, drop branches whose `|cᵢ|` is below
    /// `coeff_min` and then all but the `max_chi` heaviest.
    ///
    /// `max_chi` **truncates**; `max_branches` ([`StabilizerSum::zero`])
    /// **refuses**. They are both ceilings on `χ` and they are not the same
    /// knob: one trades accuracy for size and owes a reported mass, the other
    /// is a memory stop that discards nothing (see `over_cap` below).
    pub fn with_truncation(mut self, coeff_min: f64, max_chi: Option<usize>) -> Self {
        self.coeff_min = coeff_min;
        self.max_chi = max_chi;
        self
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// The stabilizer rank currently carried.
    pub fn chi(&self) -> usize {
        self.branches.len()
    }

    /// The largest `χ` held at any point in the run.
    pub fn peak_chi(&self) -> usize {
        self.peak_chi
    }

    /// The branches themselves, for a caller that wants the decomposition
    /// rather than a number.
    pub fn branches(&self) -> &[(Complex64, ChForm)] {
        &self.branches
    }

    /// `Σᵢ|cᵢ|` — the coefficient mass the sum is still carrying.
    pub fn coefficient_mass(&self) -> f64 {
        self.branches.iter().map(|(c, _)| c.norm()).sum()
    }

    /// `m = Σ_dropped |cᵢ|`, accumulated over every truncation this sum has
    /// been through. The raw mass, not a bound on anything by itself — the
    /// certificate derives `R·m·(2+m)` from it.
    pub fn dropped_mass(&self) -> f64 {
        self.dropped_mass
    }

    /// The coefficient floor in force (`0.0` = none).
    pub fn coeff_min(&self) -> f64 {
        self.coeff_min
    }

    /// The truncating `χ` ceiling in force (`None` = none). Not
    /// `max_branches`, which refuses instead of cutting.
    pub fn max_chi(&self) -> Option<usize> {
        self.max_chi
    }

    /// Drop what the knobs say to drop and bank its L1 mass.
    ///
    /// Two things this deliberately does **not** do.
    ///
    /// It does not renormalise. `|ψ′⟩` is left exactly as the surviving terms
    /// make it — `⟨ψ′|ψ′⟩ < 1` after a real cut — because the bound
    /// `R·m·(2+m)` is derived for the raw truncated state
    /// ([`crate::StabRankCertificate`]): `‖ψ′‖ ≤ ‖ψ‖ + m` is the step that produces
    /// the `(2+m)` factor, and a renormalised state satisfies a *different*
    /// inequality that nothing here has derived. Scaling the survivors back up
    /// would usually move the value *closer* to the exact one, which is why
    /// this is a soundness rule and not a quality one, and why
    /// `StabRankCertificate::truncated_norm_sqr` is reported: a consumer can
    /// see that it was not done.
    ///
    /// It does not compare against the *exact* state's coefficients. The mass
    /// is measured in the running (already sub-normalised) decomposition, and
    /// that is what the derivation wants. With `ψ_k` exact after `k` steps and
    /// `ψ′_k` this engine's,
    /// `‖ψ_k − ψ′_k‖ ≤ ‖U_k(ψ_{k−1} − ψ′_{k−1})‖ + ‖U_kψ′_{k−1} − T_k U_kψ′_{k−1}‖`
    /// `≤ ‖ψ_{k−1} − ψ′_{k−1}‖ + m_k`, since `U_k` is unitary and `T_k`
    /// removes exactly `m_k` of L1 mass from unit-norm branches. Induction
    /// gives `‖ψ − ψ′‖ ≤ Σₖ mₖ`, so the masses add, and this field adds them.
    pub fn truncate(&mut self) {
        let floor = self.coeff_min;
        if floor > 0.0 {
            let mut dropped = 0.0;
            self.branches.retain(|(c, _)| {
                if c.norm() < floor {
                    dropped += c.norm();
                    false
                } else {
                    true
                }
            });
            self.dropped_mass += dropped;
        }
        // The floor first, so the ceiling cuts what survived it. The other
        // order would charge the same branch to both knobs or neither,
        // depending on where the sort landed it.
        if let Some(cap) = self.max_chi {
            let cap = cap.max(1);
            if self.branches.len() > cap {
                // Heaviest first. `partial_cmp` cannot fail here: a branch
                // coefficient is a product of gate factors and `ChForm` keeps
                // its own amplitudes finite (`check_invariants`).
                self.branches.sort_by(|a, b| {
                    b.0.norm()
                        .partial_cmp(&a.0.norm())
                        .expect("branch coefficients are finite")
                });
                self.dropped_mass += self
                    .branches
                    .drain(cap..)
                    .map(|(c, _)| c.norm())
                    .sum::<f64>();
            }
        }
    }

    /// Apply a Clifford gate to every branch. The coefficients do not move:
    /// a Clifford is one unitary applied to the whole sum.
    pub fn clifford(&mut self, gate: impl Fn(&mut ChForm)) {
        for (_, state) in &mut self.branches {
            gate(state);
        }
    }

    /// Multiply the whole state by a scalar — the global phases the gate
    /// table owes, kept rather than dropped.
    pub fn scale(&mut self, z: Complex64) {
        for (c, _) in &mut self.branches {
            *c *= z;
        }
    }

    /// `exp(−iφ/2 P)` for the Pauli string `P`, by this module's one
    /// mechanism. Doubles `χ` unless `φ` is a quarter turn.
    ///
    /// `I` sites are dropped; an all-identity `P` is a global phase
    /// `e^{−iφ/2}` and no branching. The wires must be distinct, which is a
    /// property of the caller's gate table, not of the input.
    pub fn pauli_rotation(&mut self, sites: &[PauliSite], phi: f64) -> Result<()> {
        let sites: Vec<PauliSite> = sites
            .iter()
            .copied()
            .filter(|&(_, p)| p != PauliOp::I)
            .collect();
        debug_assert!(
            sites.windows(2).all(|w| w[0].0 != w[1].0) && sites.iter().all(|&(q, _)| q < self.n),
            "pauli_rotation: sites must be distinct wires inside the register"
        );
        if sites.is_empty() {
            self.scale(Complex64::from_polar(1.0, -phi / 2.0));
            return Ok(());
        }
        let minus_one = Complex64::new(-1.0, 0.0);
        match quarter_turns(phi) {
            Some(0) => {}
            Some(1) => self.clifford(|st| sqrt_pauli(st, &sites, false)),
            Some(2) => {
                self.clifford(|st| apply_pauli(st, &sites));
                self.scale(Complex64::new(0.0, -1.0));
            }
            Some(3) => {
                self.clifford(|st| sqrt_pauli(st, &sites, true));
                self.scale(minus_one);
            }
            Some(4) => self.scale(minus_one),
            Some(5) => {
                self.clifford(|st| sqrt_pauli(st, &sites, false));
                self.scale(minus_one);
            }
            Some(6) => {
                self.clifford(|st| apply_pauli(st, &sites));
                self.scale(Complex64::new(0.0, 1.0));
            }
            Some(7) => self.clifford(|st| sqrt_pauli(st, &sites, true)),
            Some(k) => unreachable!("quarter_turns is reduced mod 8, got {k}"),
            None => self.split(&sites, phi)?,
        }
        Ok(())
    }

    /// The 2-term split itself: `cos(φ/2)·I − i sin(φ/2)·P`.
    fn split(&mut self, sites: &[PauliSite], phi: f64) -> Result<()> {
        let keep = Complex64::new((phi / 2.0).cos(), 0.0);
        let flip = Complex64::new(0.0, -(phi / 2.0).sin());
        // Exact zeros only. A leg whose coefficient is 0.0 contributes
        // nothing and dropping it is an identity; dropping one that is merely
        // small is truncation, which is S2 and owes a reported mass.
        let (drop_keep, drop_flip) = (keep.re == 0.0, flip.im == 0.0);
        let grown = self.branches.len() * (2 - usize::from(drop_keep || drop_flip));
        if grown > self.max_branches {
            return Err(over_cap(grown, self.max_branches));
        }
        let mut out = Vec::with_capacity(grown);
        for (c, state) in self.branches.drain(..) {
            if !drop_flip {
                let mut flipped = state.clone();
                apply_pauli(&mut flipped, sites);
                out.push((c * flip, flipped));
            }
            if !drop_keep {
                out.push((c * keep, state));
            }
        }
        self.branches = out;
        // `peak_chi` records the pre-truncation width, because that is the
        // memory the run actually asked for.
        self.peak_chi = self.peak_chi.max(self.branches.len());
        // A split is the only thing in this engine that moves a coefficient's
        // magnitude — a Clifford moves no coefficient at all and `scale` is
        // applied to every branch alike — so truncating here covers every
        // point at which there is something new to truncate.
        self.truncate();
        Ok(())
    }

    /// `⟨ψ|P|ψ⟩` for one Pauli string: the χ² pairwise sum
    /// `Σᵢⱼ c̄ᵢcⱼ⟨φᵢ|P|φⱼ⟩`, complex, with no symmetry assumed.
    pub fn pauli_expectation(&self, sites: &[PauliSite]) -> Complex64 {
        self.pauli_expectations(&[sites.to_vec()])[0]
    }

    /// The same sum for several Pauli strings at once.
    ///
    /// The bra programs ([`ChForm::bra_program`]) depend on the branch alone,
    /// not on the string, so the `χ` of them are built once and reused across
    /// every term of the observable.
    pub fn pauli_expectations(&self, terms: &[Vec<PauliSite>]) -> Vec<Complex64> {
        let programs: Vec<_> = self.branches.iter().map(|(_, s)| s.bra_program()).collect();
        terms
            .iter()
            .map(|sites| {
                let kets: Vec<ChForm> = self
                    .branches
                    .iter()
                    .map(|(_, s)| {
                        let mut k = s.clone();
                        apply_pauli(&mut k, sites);
                        k
                    })
                    .collect();
                let mut acc = Complex64::new(0.0, 0.0);
                for (i, (ci, _)) in self.branches.iter().enumerate() {
                    let bra = ci.conj();
                    for (j, (cj, _)) in self.branches.iter().enumerate() {
                        acc += bra * cj * programs[i].overlap(&kets[j]);
                    }
                }
                acc
            })
            .collect()
    }

    /// `⟨ψ|ψ⟩`. Unit for any state this engine produces; a run that lost a
    /// `1/√2` somewhere says so here.
    pub fn norm_sqr(&self) -> f64 {
        self.pauli_expectation(&[]).re
    }

    /// Every amplitude, in the indexing [`ChForm::to_statevector`] uses.
    /// `O(χ·2^n)`; for tests and small cross-checks only.
    pub fn to_statevector(&self) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); 1usize << self.n];
        for (c, state) in &self.branches {
            for (slot, a) in out.iter_mut().zip(state.to_statevector()) {
                *slot += c * a;
            }
        }
        out
    }
}

/// The branch ceiling is a **memory stop, not a truncation knob**: it refuses
/// the run rather than dropping branches, so nothing is discarded and no
/// bound is owed. `max_chi` is the other thing — a cut that drops branches
/// and reports their mass — which is why this one is not called that, and the
/// refusal says so, since a reader who meets this message while holding both
/// knobs needs to know which one changes the answer.
fn over_cap(grown: usize, cap: usize) -> OmegaError {
    OmegaError::Unsupported(format!(
        "stabrank: the sum-over-Cliffords decomposition reached {grown} CH-form \
         branches, above the ceiling of {cap}. χ doubles per non-Clifford \
         rotation and the readout is χ² inner products, so this is a stop, not \
         a wall: raise it with `with_max_branches(N)` (costs memory and time, \
         not accuracy), set a truncating ceiling with \
         `with_truncation(coeff_min, max_chi)` (costs accuracy, and reports \
         what it cost in `state_dropped_mass`), or put the circuit on \
         majoranaprop, whose cost axis is the propagated term count rather \
         than the non-Clifford gate count."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    /// The angles that must be recognised, and the near misses that must not
    /// be. Three separate claims: the grid is found; it is reduced **mod 8**,
    /// so the second lap keeps its sign; and nothing off the grid is pulled
    /// onto it, because a tolerance there would turn an angle error into a
    /// silently wrong answer.
    #[test]
    fn quarter_turns_are_recognised_exactly_and_never_approximately() {
        for (k, phi) in [
            (0u32, 0.0),
            (1, FRAC_PI_2),
            (2, PI),
            (3, 3.0 * FRAC_PI_2),
            (4, 4.0 * FRAC_PI_2),
            (5, 5.0 * FRAC_PI_2),
            (7, -FRAC_PI_2),
            (6, -PI),
            (0, 8.0 * FRAC_PI_2),
        ] {
            assert_eq!(quarter_turns(phi), Some(k), "φ = {phi}");
        }
        for phi in [
            FRAC_PI_4,
            0.3,
            FRAC_PI_2 + f64::EPSILON,
            FRAC_PI_2 - 1e-13,
            PI + 1e-15,
            f64::NAN,
            f64::INFINITY,
            1e300,
        ] {
            assert_eq!(
                quarter_turns(phi),
                None,
                "φ = {phi} was read as a Clifford angle, which would apply an \
                 exactly-Clifford gate in place of a rotation that is not one"
            );
        }
    }

    /// `exp(−iφ/2 P)` on one wire, both paths, against the 2×2 matrix. The
    /// Clifford-angle path and the splitting path have to agree with the same
    /// closed form, which is what lets the angle test above be a performance
    /// decision rather than a correctness one.
    #[test]
    fn the_clifford_path_and_the_split_path_agree_with_the_rotation_matrix() {
        for letter in [PauliOp::X, PauliOp::Y, PauliOp::Z] {
            for k in 0..8 {
                // Quarter turns, and the same angles nudged off the grid.
                for phi in [k as f64 * FRAC_PI_2, k as f64 * FRAC_PI_2 + 0.37] {
                    // Start from a state with weight on both basis vectors and
                    // a phase already on it, so no axis is an eigenstate.
                    let mut sum = StabilizerSum::zero(1, 1024);
                    sum.clifford(|st| st.h_gate(0));
                    sum.clifford(|st| st.s_gate(0));
                    let before = sum.to_statevector();
                    sum.pauli_rotation(&[(0, letter)], phi).unwrap();
                    let after = sum.to_statevector();

                    let (c, s) = ((phi / 2.0).cos(), (phi / 2.0).sin());
                    let p = match letter {
                        PauliOp::X => [0.0, 1.0, 1.0, 0.0].map(|r| Complex64::new(r, 0.0)),
                        PauliOp::Y => [
                            Complex64::new(0.0, 0.0),
                            Complex64::new(0.0, -1.0),
                            Complex64::new(0.0, 1.0),
                            Complex64::new(0.0, 0.0),
                        ],
                        _ => [1.0, 0.0, 0.0, -1.0].map(|r| Complex64::new(r, 0.0)),
                    };
                    let eye = [1.0, 0.0, 0.0, 1.0].map(|r| Complex64::new(r, 0.0));
                    let u: Vec<Complex64> = (0..4)
                        .map(|i| eye[i] * c - Complex64::new(0.0, s) * p[i])
                        .collect();
                    let want = [
                        u[0] * before[0] + u[1] * before[1],
                        u[2] * before[0] + u[3] * before[1],
                    ];
                    for (i, (g, w)) in after.iter().zip(want.iter()).enumerate() {
                        assert!(
                            (g - w).norm() < 1e-13,
                            "{letter:?} at φ={phi} (k={k}, χ={}) amplitude {i}: \
                             {g} vs {w}",
                            sum.chi()
                        );
                    }
                }
            }
        }
    }

    /// A quarter turn leaves χ alone; anything else doubles it. The cost
    /// claim of plan §1.4, as an assertion rather than a remark.
    #[test]
    fn only_a_non_clifford_angle_costs_a_branch() {
        let mut sum = StabilizerSum::zero(3, 1024);
        for k in 0..8 {
            sum.pauli_rotation(&[(0, PauliOp::Y), (2, PauliOp::X)], k as f64 * FRAC_PI_2)
                .unwrap();
            assert_eq!(sum.chi(), 1, "a quarter turn at k={k} branched");
        }
        sum.pauli_rotation(&[(0, PauliOp::Z)], FRAC_PI_4).unwrap();
        assert_eq!(sum.chi(), 2);
        sum.pauli_rotation(&[(1, PauliOp::Z)], FRAC_PI_4).unwrap();
        assert_eq!(sum.chi(), 4);
        assert_eq!(sum.peak_chi(), 4);
        assert!((sum.norm_sqr() - 1.0).abs() < 1e-12);
    }

    /// The ceiling refuses; it does not quietly drop a branch.
    #[test]
    fn the_branch_ceiling_refuses_and_names_the_way_out() {
        let mut sum = StabilizerSum::zero(4, 4);
        for q in 0..2 {
            sum.pauli_rotation(&[(q, PauliOp::Z)], FRAC_PI_4).unwrap();
        }
        assert_eq!(sum.chi(), 4);
        let err = sum
            .pauli_rotation(&[(2, PauliOp::Z)], FRAC_PI_4)
            .expect_err("the third rotation must not fit");
        let msg = err.to_string();
        assert!(
            msg.contains("with_max_branches") && msg.contains("majoranaprop"),
            "the refusal must name the knob and the other engine; got: {msg}"
        );
        assert_eq!(sum.chi(), 4, "a refused rotation must leave the sum intact");
        assert_eq!(
            sum.dropped_mass(),
            0.0,
            "the memory stop must discard nothing — it is not a truncation knob"
        );
    }

    /// `|+++⟩` through three `Rz(π/4)` gates, cut to four branches.
    ///
    /// The branches are `Z`-flip patterns on `|+++⟩` and so mutually
    /// orthogonal, which is what makes the norm leg below readable: every
    /// number here is hand-computable from `cos(π/8)` and `sin(π/8)`.
    ///
    /// `χ = 8` exactly; sorted, the coefficients are `c³`, `c²s` three times,
    /// `cs²` three times and `s³`, so a cut at four keeps `c³` and the three
    /// `c²s` and drops `m = 3cs² + s³`.
    fn three_rotations(max_chi: Option<usize>) -> StabilizerSum {
        let mut sum = StabilizerSum::zero(3, 1024).with_truncation(0.0, max_chi);
        for q in 0..3 {
            sum.clifford(move |st| st.h_gate(q));
        }
        for q in 0..3 {
            sum.pauli_rotation(&[(q, PauliOp::Z)], FRAC_PI_4).unwrap();
        }
        sum
    }

    /// The mass accounting, against the closed form, and the two counts that
    /// say the cut happened where it was asked for.
    #[test]
    fn a_chi_cut_keeps_the_heaviest_branches_and_banks_what_it_dropped() {
        let sum = three_rotations(Some(4));
        let (c, s) = ((FRAC_PI_4 / 2.0).cos(), (FRAC_PI_4 / 2.0).sin());
        let want = 3.0 * c * s * s + s * s * s;
        assert_eq!(sum.chi(), 4, "the cut did not hold χ at its ceiling");
        assert_eq!(sum.peak_chi(), 8, "peak χ is the width before the cut");
        assert!(
            (sum.dropped_mass() - want).abs() < 1e-12,
            "dropped {} where 3cs² + s³ = {want}",
            sum.dropped_mass()
        );
        // The survivors, too: a cut that banked the right mass and kept the
        // wrong branches would pass the line above.
        assert!(
            (sum.coefficient_mass() - (c * c * c + 3.0 * c * c * s)).abs() < 1e-12,
            "the four survivors carry {}, not c³ + 3c²s",
            sum.coefficient_mass()
        );
    }

    /// **The truncated state is not renormalised**, pinned by comparing the
    /// survivors against the same run with no cut — bit-for-bit, because
    /// truncation is a deletion and arithmetic on what is left would be a
    /// different operation.
    ///
    /// The norm leg is the consequence a consumer can see: `⟨ψ′|ψ′⟩` is
    /// `c⁶ + 3c⁴s²`, which is `0.942…`, not `1`. Renormalising would make it
    /// `1` and would invalidate the `(2+m)` factor in the certificate's bound,
    /// which comes from `‖ψ′‖ ≤ ‖ψ‖ + m`.
    #[test]
    fn truncation_deletes_branches_and_does_not_rescale_the_survivors() {
        let mut exact: Vec<f64> = three_rotations(None)
            .branches()
            .iter()
            .map(|(c, _)| c.norm())
            .collect();
        let mut cut: Vec<f64> = three_rotations(Some(4))
            .branches()
            .iter()
            .map(|(c, _)| c.norm())
            .collect();
        exact.sort_by(|a, b| b.partial_cmp(a).unwrap());
        cut.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(exact.len(), 8);
        assert_eq!(
            cut,
            exact[..4].to_vec(),
            "the surviving coefficients are not the exact run's own, so \
             truncation rescaled something"
        );

        let (c, s) = ((FRAC_PI_4 / 2.0).cos(), (FRAC_PI_4 / 2.0).sin());
        let norm = three_rotations(Some(4)).norm_sqr();
        let want = c.powi(6) + 3.0 * c.powi(4) * s * s;
        assert!(
            (norm - want).abs() < 1e-12,
            "⟨ψ′|ψ′⟩ = {norm}, not c⁶ + 3c⁴s² = {want}"
        );
        assert!(
            norm < 0.95,
            "⟨ψ′|ψ′⟩ = {norm} after dropping half the branches: the state was \
             renormalised, and the bound R·m·(2+m) is not derived for that state"
        );
    }

    /// The coefficient floor, and the order the two knobs are applied in.
    #[test]
    fn the_coefficient_floor_cuts_by_magnitude_and_runs_before_the_chi_cut() {
        let (c, s) = ((FRAC_PI_4 / 2.0).cos(), (FRAC_PI_4 / 2.0).sin());
        // A floor just above cs² removes the four lightest branches on its
        // own, so the χ ceiling at 4 finds nothing left to do.
        let mut sum = StabilizerSum::zero(3, 1024).with_truncation(c * s * s + 1e-9, Some(4));
        for q in 0..3 {
            sum.clifford(move |st| st.h_gate(q));
        }
        for q in 0..3 {
            sum.pauli_rotation(&[(q, PauliOp::Z)], FRAC_PI_4).unwrap();
        }
        assert_eq!(sum.chi(), 4);
        assert!(
            (sum.dropped_mass() - (3.0 * c * s * s + s * s * s)).abs() < 1e-12,
            "the floor and the ceiling together banked {}",
            sum.dropped_mass()
        );
        assert_eq!(sum.coeff_min(), c * s * s + 1e-9);
        assert_eq!(sum.max_chi(), Some(4));
    }

    /// An exact sum reports a dropped mass of exactly zero, and that is the
    /// only way the field can be zero: nowhere to drop anything.
    #[test]
    fn without_knobs_nothing_is_dropped_at_any_chi() {
        let sum = three_rotations(None);
        assert_eq!(sum.chi(), 8);
        assert_eq!(sum.dropped_mass(), 0.0);
        assert!((sum.norm_sqr() - 1.0).abs() < 1e-12);
    }
}
