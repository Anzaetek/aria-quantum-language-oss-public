// SPDX-License-Identifier: Apache-2.0
//! The `stabrank` backend door, S2 edition: Clifford + T, with truncation and
//! the bound that has to come with it.
//!
//! Expectation values only, exactly as `majoranaprop`: [`Backend::execute`]
//! refuses and names the backend to use instead. What has changed since S0 is
//! what lies behind the door — one CH-form branch has become a sum of them
//! ([`StabilizerSum`]), so `T`, `T†` and a generic `Rz(θ)` are *executed*
//! rather than refused, by the one mechanism that module documents.
//!
//! A run made with [`StabRankBackend::new`] is still **exact**, and
//! `state_dropped_mass` is `0.0` on it because there is nowhere to drop
//! anything. [`StabRankBackend::with_truncation`] is the other engine: it
//! buys a sub-`2^t` rank by dropping branches, and it pays for them in
//! [`StabRankCertificate::state_dropped_mass`] and
//! [`StabRankCertificate::expectation_error_bound`]. A certificate that
//! reported a zero drop while quietly discarding branches would be the
//! certificate lying, which PLAN-MAJORANA-STIM.md §1.2 is written against.
//!
//! Three things about the truncating engine, each with its own place below:
//! the state is **not renormalised** (the bound is derived for the raw
//! truncated state, and `truncated_norm_sqr` lets a consumer check that it
//! was left raw); `R` is the observable's L1 mass **in the basis actually
//! read out**, after measurement deferral and after any Jordan–Wigner
//! mapping; and a result whose bound excludes nothing is **refused, not
//! returned**.

use num_complex::Complex64;
use omega_core::circuit::{CircuitIR, GateKind, GateOp, ParamExpr};
use omega_core::defer_measure::prepare_for_expectation;
use omega_core::error::{OmegaError, Result};
use omega_core::executor::{Backend, ExecConfig, ExecResult, Observable, PauliOp};
use omega_core::params::ParameterBinding;

use crate::chform::ChForm;
use crate::sum::{observable_l1_norm, quarter_turns, PauliSite, StabilizerSum};

/// Largest imaginary part tolerated on `⟨ψ|P|ψ⟩` before the run is refused.
/// The quantity is real by construction (`P` Hermitian), so anything above
/// rounding is a kernel defect, not a physical answer.
///
/// At S1 this is a live check and not a formality: the χ² sum is taken over
/// the whole square rather than half of it precisely so that the Hermiticity
/// of the Gram matrix has to *hold* rather than be assumed — see the readout
/// section of [`crate::sum`].
const IMAG_TOL: f64 = 1e-9;

/// Default branch ceiling. χ doubles per non-Clifford rotation and the
/// readout is χ² inner products, so this sits far below majoranaprop's
/// `DEFAULT_MAX_TERMS`: at 2¹⁶ branches the Gram matrix alone is 2³² entries.
/// It stops a run that has left this engine's regime; it is not a truncation
/// knob (see `over_cap` in [`crate::sum`]).
pub const DEFAULT_MAX_BRANCHES: usize = 1 << 16;

/// The basis the engine's seed was built in, recorded by the engine so a
/// consumer learns it from the run rather than from whoever dispatched it —
/// the majoranaprop contract, in a picture that has one entry.
///
/// There is exactly one variant, and that is plan §1.4 at the type level:
/// stabrank is Schrödinger, so its seed is the *state* `|0…0⟩` and the
/// observable is read out at the end. A fermionic observable reaches this
/// engine already mapped to Pauli by Jordan–Wigner (exact, so nothing is
/// lost), and there is no ladder-native seeding to choose instead — the state
/// is not an operator sum. majoranaprop's `Ladder` has no counterpart here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedBasis {
    /// A Pauli `Observable`, read out term by term against the branch sum.
    Pauli,
}

impl SeedBasis {
    pub fn as_str(self) -> &'static str {
        match self {
            SeedBasis::Pauli => "pauli",
        }
    }
}

/// What a `stabrank` run did and where it stood.
///
/// # The derivation
///
/// The inequality itself is PLAN-MAJORANA-STIM.md §1.1. Write the run's
/// non-Clifford content as a sum of χ stabilizer states,
/// `|ψ⟩ = Σᵢ cᵢ|φᵢ⟩`. Truncating terms of total coefficient mass
/// `m = Σ_dropped |cᵢ|` gives `‖ψ − ψ′‖ ≤ m` (each `|φᵢ⟩` is unit-norm,
/// triangle inequality), hence deterministically
///
/// ```text
/// |⟨ψ|O|ψ⟩ − ⟨ψ′|O|ψ′⟩| ≤ ‖O‖·‖ψ−ψ′‖·(‖ψ‖+‖ψ′‖) ≤ R·m·(2+m),
/// ```
///
/// with `R` the Pauli-L1 of the observable (`‖O‖ ≤ R`). The last step uses
/// `‖ψ‖ = 1` and `‖ψ′‖ ≤ ‖ψ‖ + ‖ψ−ψ′‖ ≤ 1 + m`; that is where the `(2+m)`
/// comes from, and it is why the truncated state must be left **raw**. The
/// masses of successive truncations add — [`StabilizerSum::truncate`] carries
/// that induction.
///
/// # Why this is not majoranaprop's additive `dropped_mass`
///
/// Verbatim, PLAN-MAJORANA-STIM.md §1.2, *The bound: its own accounting, same
/// gate pattern*:
///
/// > Stabilizer-rank truncation does **not** plug into majoranaprop's
/// > `dropped_mass` field, because the two bounds have different functional
/// > forms and live in different pictures:
/// >
/// > * majoranaprop is **Heisenberg, observable-side**: dropped L1 mass enters
/// >   the error **additively** (`|Δ⟨O⟩| ≤ m` because `|⟨M⟩| ≤ 1`).
/// > * stabilizer rank is **Schrödinger, state-side**: the state enters the
/// >   expectation **bilinearly**, so the same L1 mass enters as
/// >   `R·m·(2+m)` — quadratic correction, multiplied by the observable range.
/// >
/// > Pouring the state-side `m` into a field whose documented meaning is
/// > "additive bound on |Δ⟨O⟩|" would make the certificate lie. So the new
/// > engine gets its own certificate struct carrying `state_dropped_mass`
/// > (the raw `m`) **and** `expectation_error_bound = R·m·(2+m)` (the derived
/// > bound), with the derivation in the doc comment. What is *shared* is the
/// > contract pattern: `observable_range = R`, `is_exact()`,
/// > `is_informative()` ⇔ `expectation_error_bound < R + |value|`, and refusal
/// > (not return) on vacuity — identical shape, different arithmetic.
/// >
/// > **What a combined certificate means:** when both engines run the same
/// > job, each yields a certified interval `[v − b, v + b]`. Both are sound,
/// > so the intersection is sound and tighter; an **empty intersection is a
/// > bug in one engine** and is exactly the cross-lane failure signal S3 tests
/// > for. No phase feeds one engine's truncated output into the other, so the
/// > bounds never compose sequentially.
/// >
/// > Two pins carried over: `R` is computed in the basis actually read out
/// > (post-JW Pauli — the §1.3 lesson of PLAN-FERMIONIC), and the engine does
/// > **not** renormalize `ψ′` after truncation (renormalizing invalidates the
/// > derivation; the bound above is for the raw truncated state).
#[derive(Debug, Clone)]
pub struct StabRankCertificate {
    /// Which basis the observable was seeded in — see [`SeedBasis`].
    pub seed_basis: SeedBasis,
    /// `m`: the L1 coefficient mass of the *state* discarded by truncation,
    /// raw and summed over every cut the run made. `0.0` on an exact run, and
    /// the only way it is zero is that nothing was dropped.
    ///
    /// Not a bound on `|Δ⟨O⟩|` — that is majoranaprop's field, in the other
    /// picture. This one is an input to [`Self::expectation_error_bound`].
    pub state_dropped_mass: f64,
    /// `R·m·(2+m)`: the deterministic bound on `|⟨O⟩ − ⟨O⟩_exact|`, derived
    /// above. Always recomputable from [`Self::observable_range`] and
    /// [`Self::state_dropped_mass`], so the derived number cannot drift from
    /// the raw one.
    pub expectation_error_bound: f64,
    /// Branches in the decomposition at readout.
    pub final_chi: usize,
    /// Largest number of branches held at any point, **before** truncation —
    /// the width the run actually paid memory for.
    pub peak_chi: usize,
    /// Coefficient floor in force (`0.0` = none).
    pub coeff_min: f64,
    /// Truncating `χ` ceiling in force (`None` = none). Distinct from
    /// [`Self::max_branches`], which refuses instead of cutting.
    pub max_chi: Option<usize>,
    /// `Σ|cᵢ|` of the observable — the range `⟨O⟩` is confined to a priori,
    /// computed in the basis actually read out (post-JW, §1.2's pin).
    pub observable_range: f64,
    /// Branch ceiling in force. Not a truncation knob: the run is refused,
    /// not trimmed, when it is reached, and nothing is discarded.
    pub max_branches: usize,
    /// `⟨ψ′|ψ′⟩` of the state actually read out. `1.0` to rounding on an
    /// exact run, and something else after a real cut.
    ///
    /// Reported so the no-renormalisation pin of §1.2 is checkable rather
    /// than merely promised: an engine that scaled `ψ′` back to unit norm
    /// would return a number that is often *closer* to the exact one while
    /// satisfying a bound nobody derived, and this field is where that shows.
    ///
    /// Usually below `1`, and **not always**: the branches of a decomposition
    /// are not orthogonal, so `‖ψ′‖² = 1 − ‖Δ‖² − 2Re⟨ψ′|Δ⟩` for the dropped
    /// part `Δ`, and a `Δ` that overlaps the survivors negatively leaves the
    /// remainder *longer* than the state it came from. Measured at two cells
    /// of `tests/truncation_witness.rs`, at `1.2285` where the other 38 sat at
    /// `0.7285` or `0.9785`. Read this field as "different from 1", not as a
    /// deficit.
    pub truncated_norm_sqr: f64,
    /// The returned value; with [`Self::expectation_error_bound`] gives the
    /// certified interval.
    pub value: f64,
}

impl StabRankCertificate {
    /// True when the run discarded nothing.
    pub fn is_exact(&self) -> bool {
        self.state_dropped_mass == 0.0
    }

    /// True when the bound still excludes something:
    /// `expectation_error_bound < R + |v|`. Same derivation as majoranaprop's
    /// and pauliprop's, with this engine's arithmetic in place of theirs — the
    /// result interval `[v−b, v+b]` contains the a-priori `[−R, R]` iff
    /// `b ≥ R + |v|`.
    pub fn is_informative(&self) -> bool {
        self.expectation_error_bound < self.observable_range + self.value.abs()
    }
}

/// Expectation-value backend over the exact-phase CH-form kernel.
#[derive(Debug, Clone, Copy)]
pub struct StabRankBackend {
    max_branches: usize,
    coeff_min: f64,
    max_chi: Option<usize>,
}

impl Default for StabRankBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StabRankBackend {
    /// Exact engine: no coefficient floor, no `χ` cut, default branch ceiling.
    pub fn new() -> Self {
        Self {
            max_branches: DEFAULT_MAX_BRANCHES,
            coeff_min: 0.0,
            max_chi: None,
        }
    }

    /// Truncating engine: after every split, drop branches whose `|cᵢ|` is
    /// below `coeff_min` and then all but the `max_chi` heaviest.
    ///
    /// majoranaprop's `with_truncation(coeff_min, max_length)` in the same
    /// position, with the second knob changed to the one this picture has: a
    /// Schrödinger branch has a coefficient and no length.
    ///
    /// **`max_chi` is not [`Self::with_max_branches`].** `max_chi` truncates:
    /// it drops branches, changes the answer, and owes a reported mass and a
    /// bound. `max_branches` refuses: it stops a run that has left this
    /// engine's regime without discarding anything, so two runs under
    /// different `max_branches` that both complete return the same number.
    pub fn with_truncation(coeff_min: f64, max_chi: Option<usize>) -> Self {
        Self {
            coeff_min,
            max_chi,
            ..Self::new()
        }
    }

    /// Builder: branch ceiling. Raising it costs memory and time, never
    /// accuracy — a run that fits under either ceiling returns the same
    /// number, because nothing between them is dropped.
    pub fn with_max_branches(mut self, max_branches: usize) -> Self {
        self.max_branches = max_branches;
        self
    }

    pub fn max_branches(&self) -> usize {
        self.max_branches
    }

    pub fn coeff_min(&self) -> f64 {
        self.coeff_min
    }

    /// The truncating ceiling. See [`Self::with_truncation`] for why this is
    /// a different thing from [`Self::max_branches`].
    pub fn max_chi(&self) -> Option<usize> {
        self.max_chi
    }

    /// True when this engine can drop a branch. An engine for which this is
    /// false returns an exact value and a zero bound.
    pub fn truncating(&self) -> bool {
        self.coeff_min > 0.0 || self.max_chi.is_some()
    }

    /// Run the circuit and hand back the whole decomposition. The branch sum
    /// is the product here; a caller that wants amplitudes, a cross term or
    /// the rank itself takes it rather than a number.
    pub fn simulate_sum(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
    ) -> Result<StabilizerSum> {
        self.run(circuit, params, false)
    }

    /// The same run restricted to a single CH-form branch: the S0 door, kept
    /// because a caller who means "this circuit is Clifford" should be told
    /// when it is not, by name, rather than handed a `χ > 1` sum to discover
    /// it from.
    pub fn simulate(&self, circuit: &CircuitIR, params: &ParameterBinding) -> Result<ChForm> {
        let sum = self.run(circuit, params, true)?;
        debug_assert_eq!(sum.chi(), 1, "single-branch run produced a sum");
        let (coeff, mut state) = sum.branches()[0].clone();
        state.scale_phase(coeff);
        Ok(state)
    }

    fn run(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        single_branch: bool,
    ) -> Result<StabilizerSum> {
        circuit.refuse_qudits("stabrank")?;
        self.check_knobs()?;
        let n = circuit.num_qubits as usize;
        let mut sum =
            StabilizerSum::zero(n, self.max_branches).with_truncation(self.coeff_min, self.max_chi);
        for op in &circuit.ops {
            apply(&mut sum, op, params, single_branch)?;
        }
        Ok(sum)
    }

    /// Reject a truncation setting that has no run behind it, rather than
    /// reinterpreting it. `max_chi = Some(0)` asks for a state with no
    /// branches; a non-finite or negative floor makes the comparison in
    /// [`StabilizerSum::truncate`] mean something other than what its doc
    /// says. Both are caller errors and are named as such.
    fn check_knobs(&self) -> Result<()> {
        if self.max_chi == Some(0) {
            return Err(OmegaError::Unsupported(
                "stabrank: `with_truncation(_, Some(0))` asks for a decomposition \
                 with no branches in it, which is not a state and has no \
                 expectation. Use `Some(1)` for the hardest cut this engine can \
                 make, and read `state_dropped_mass` to see what it cost."
                    .into(),
            ));
        }
        if self.coeff_min < 0.0 || self.coeff_min.is_nan() {
            return Err(OmegaError::Unsupported(format!(
                "stabrank: `coeff_min` is {}, and the truncation rule is \
                 `|cᵢ| < coeff_min`. A negative floor drops nothing and a NaN \
                 floor drops nothing while reading as if it might; `0.0` is how \
                 'no floor' is spelled.",
                self.coeff_min
            )));
        }
        Ok(())
    }

    /// The expectation and its certificate — the entry point the trait method
    /// and every future door route through, so the certificate is built in
    /// exactly one place.
    pub fn expectation_with_certificate(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<(f64, StabRankCertificate)> {
        circuit.refuse_qudits(self.name())?;
        let (deferred, observable) = prepare_for_expectation(circuit, observable)?;
        observable.validate_qubits(deferred.num_qubits)?;
        let sum = self.run(&deferred, params, false)?;

        // The observable's terms, and then the identity: `⟨ψ′|ψ′⟩` rides along
        // in the same pairwise pass (one extra term, the bra programs already
        // built) and becomes the certificate's no-renormalisation evidence.
        let mut terms: Vec<Vec<PauliSite>> = observable
            .terms
            .iter()
            .map(|(_, paulis)| paulis.iter().map(|&(w, p)| (w as usize, p)).collect())
            .collect();
        terms.push(Vec::new());
        let values = sum.pauli_expectations(&terms);
        let truncated_norm_sqr = values[values.len() - 1].re;

        let mut value = 0.0;
        for ((coeff, paulis), v) in observable.terms.iter().zip(&values) {
            if v.im.abs() > IMAG_TOL {
                return Err(OmegaError::Backend(format!(
                    "stabrank: the χ² pairwise sum for the Pauli term {paulis:?} came \
                     back with imaginary part {:.3e} on {} branches. The Gram matrix \
                     ⟨φᵢ|P|φⱼ⟩ of a Hermitian P is Hermitian, so its weighted sum is \
                     real; an imaginary part means the CH-form phase bookkeeping \
                     disagrees with itself between branches. That is a kernel defect, \
                     not a property of the circuit; refusing rather than returning the \
                     real part.",
                    v.im,
                    sum.chi()
                )));
            }
            value += coeff * v.re;
        }

        // `R` in the basis actually read out: `observable` here is the one
        // `prepare_for_expectation` handed back, after measurement deferral,
        // and a fermionic observable has already been mapped to Pauli by
        // Jordan–Wigner before it reaches this door. §1.2's first pin.
        let observable_range = observable_l1_norm(&observable);
        let m = sum.dropped_mass();
        let cert = StabRankCertificate {
            seed_basis: SeedBasis::Pauli,
            state_dropped_mass: m,
            expectation_error_bound: observable_range * m * (2.0 + m),
            final_chi: sum.chi(),
            peak_chi: sum.peak_chi(),
            coeff_min: self.coeff_min,
            max_chi: self.max_chi,
            observable_range,
            max_branches: self.max_branches,
            truncated_norm_sqr,
            value,
        };
        if !cert.is_informative() {
            return Err(vacuous(&cert));
        }
        Ok((value, cert))
    }
}

/// The vacuity refusal: a bound that excludes nothing is not a result.
///
/// majoranaprop's `finish` in shape and in reasoning — correct bound, no
/// information — with this engine's arithmetic and this engine's knobs named.
/// Returning the number instead would hand a caller a value whose certified
/// interval already contains every value the observable can take.
fn vacuous(cert: &StabRankCertificate) -> OmegaError {
    let ceiling = cert.observable_range + cert.value.abs();
    OmegaError::Unsupported(format!(
        "stabrank: truncation discarded a state-side L1 coefficient mass of \
         m = {:.4e}, so the derived bound on |Δ⟨O⟩| is R·m·(2+m) = {:.4e}, which \
         exceeds the ceiling of R + |value| = {:.4e}. That bound is correct and \
         it excludes nothing: ⟨O⟩ cannot leave [-{:.4e}, {:.4e}] to begin with. \
         Refusing rather than returning a number with no information in it. \
         Options: raise `max_chi` or lower `coeff_min` in \
         `with_truncation(coeff_min, max_chi)` — both discard less mass and \
         both cost χ² inner products; raise `with_max_branches(N)` so the run \
         needs no truncation at all, which does not change the answer; or put \
         the circuit on majoranaprop, whose cost axis is the propagated term \
         count rather than the non-Clifford gate count. Final χ {}, peak χ {}, \
         coeff_min {:.4e}, max_chi {:?}, max_branches {}.",
        cert.state_dropped_mass,
        cert.expectation_error_bound,
        ceiling,
        cert.observable_range,
        cert.observable_range,
        cert.final_chi,
        cert.peak_chi,
        cert.coeff_min,
        cert.max_chi,
        cert.max_branches,
    ))
}

/// Lower one IR op onto the branch sum.
///
/// Two kinds of entry. A Clifford gate goes straight to a [`ChForm`] method on
/// every branch, exactly as at S0 — no coefficient moves and χ does not grow.
/// Everything else is written as a product of Pauli rotations `exp(−iφ/2 P)`
/// times a global phase, and handed to
/// [`StabilizerSum::pauli_rotation`], which splits it or recognises it as
/// Clifford. There is one non-Clifford mechanism in this engine and this is
/// its only caller.
///
/// `single_branch` is the S0 door: it refuses a rotation that would split
/// rather than silently returning a sum the caller cannot hold.
fn apply(
    sum: &mut StabilizerSum,
    op: &GateOp,
    params: &ParameterBinding,
    single_branch: bool,
) -> Result<()> {
    use std::f64::consts::FRAC_PI_4;
    let q = |i: usize| op.qubits[i].0 as usize;
    let p = |i: usize| -> Result<f64> { params.resolve(&op.params[i]) };

    // `exp(−iφ/2 P)` with the single-branch gate refused by name.
    let rot = |sum: &mut StabilizerSum, sites: &[PauliSite], phi: f64| -> Result<()> {
        if single_branch && quarter_turns(phi).is_none() {
            return Err(splits(op, phi));
        }
        sum.pauli_rotation(sites, phi)
    };

    match &op.gate {
        GateKind::Id | GateKind::Barrier => {}
        GateKind::X => sum.clifford(|st| st.x_gate(q(0))),
        GateKind::Y => sum.clifford(|st| st.y_gate(q(0))),
        GateKind::Z => sum.clifford(|st| st.z_gate(q(0))),
        GateKind::H => sum.clifford(|st| st.h_gate(q(0))),
        GateKind::S => sum.clifford(|st| st.s_gate(q(0))),
        GateKind::Sdg => sum.clifford(|st| st.sdg_gate(q(0))),
        GateKind::Sx => sum.clifford(|st| st.sx_gate(q(0))),
        GateKind::Sxdg => sum.clifford(|st| st.sxdg_gate(q(0))),
        GateKind::CX => sum.clifford(|st| st.cx_gate(q(0), q(1))),
        GateKind::CY => sum.clifford(|st| st.cy_gate(q(0), q(1))),
        GateKind::CZ => sum.clifford(|st| st.cz_gate(q(0), q(1))),
        GateKind::Swap => sum.clifford(|st| st.swap_gate(q(0), q(1))),

        // T = diag(1, e^{iπ/4}) = e^{iπ/8}·exp(−iπ/8·Z), and T† its
        // conjugate. The leading scalar is what makes these the gate rather
        // than the gate up to phase; `omega-backend-statevector`'s `t()` is
        // the literal diagonal, and the amplitude pin compares against it.
        GateKind::T => {
            rot(sum, &[(q(0), PauliOp::Z)], FRAC_PI_4)?;
            sum.scale(Complex64::from_polar(1.0, FRAC_PI_4 / 2.0));
        }
        GateKind::Tdg => {
            rot(sum, &[(q(0), PauliOp::Z)], -FRAC_PI_4)?;
            sum.scale(Complex64::from_polar(1.0, -FRAC_PI_4 / 2.0));
        }
        // Rz(θ) = exp(−iθ/2·Z) exactly — the rotation with no scalar owed.
        GateKind::Rz => rot(sum, &[(q(0), PauliOp::Z)], p(0)?)?,
        // U1(λ) = diag(1, e^{iλ}) = e^{iλ/2}·Rz(λ). Same rotation, different
        // phase convention, and the two are distinct gates in this IR.
        GateKind::U1 => {
            let lam = p(0)?;
            rot(sum, &[(q(0), PauliOp::Z)], lam)?;
            sum.scale(Complex64::from_polar(1.0, lam / 2.0));
        }
        GateKind::Rx => rot(sum, &[(q(0), PauliOp::X)], p(0)?)?,
        GateKind::Ry => rot(sum, &[(q(0), PauliOp::Y)], p(0)?)?,
        // Rbs(θ) = exp(−iθ/2·(Y⊗X − X⊗Y)); the two generators commute
        // (both products are Z⊗Z), the generator annihilates |00⟩ and |11⟩,
        // so the gate is exactly the two rotations and no scalar. Clifford at
        // multiples of π/2 (§1.4) — and there it costs no branch, which is
        // what makes a Clifford-angle Givens circuit cheap here.
        GateKind::Rbs => {
            let th = p(0)?;
            rot(sum, &[(q(0), PauliOp::Y), (q(1), PauliOp::X)], th)?;
            rot(sum, &[(q(0), PauliOp::X), (q(1), PauliOp::Y)], -th)?;
        }

        GateKind::Reset | GateKind::Measure => {
            return Err(OmegaError::Unsupported(format!(
                "stabrank: {:?} is not a unitary this expectation path can carry. \
                 Trailing measurements are deferred before the circuit reaches here; \
                 a mid-circuit reset is a channel. Use the statevector backend.",
                op.gate
            )))
        }
        other => {
            return Err(OmegaError::Unsupported(format!(
                "stabrank: {other:?} has no entry in the S1 gate table. The engine \
                 executes the Clifford group, the Pauli rotations (rx/ry/rz/u1) and \
                 the Givens rotation `rbs`, each as `exp(-i phi/2 P)`; a gate that is \
                 not yet written as a product of those — the multi-angle (u2/u3) and \
                 controlled-rotation (crz/cu3) forms, the multi-controlled gates \
                 (ccx/cswap), and the photonic and qudit operations — is refused by \
                 name rather than approximated. Use the statevector backend, or \
                 majoranaprop."
            )))
        }
    }
    Ok(())
}

/// The single-branch door's refusal: the gate is executable, just not into
/// one CH form. Names the gate — a `T` silently run as an `S` is K8 trap 2
/// one layer in, and plan §1.3 exists to keep that out of this engine.
fn splits(op: &GateOp, phi: f64) -> OmegaError {
    let angle = if op.params.is_empty() {
        String::new()
    } else {
        match &op.params[0] {
            ParamExpr::Concrete(v) => format!(" at angle {v}"),
            e => format!(" at angle {e:?}"),
        }
    };
    OmegaError::Unsupported(format!(
        "stabrank: {:?}{angle} is a rotation by φ = {phi} radians, which is not a \
         multiple of π/2 and so is not Clifford. It splits the state into two \
         CH-form branches, and `simulate` returns a single branch. Use \
         `simulate_sum` for the decomposition, or `expectation_with_certificate` \
         for the value — both execute this gate exactly.",
        op.gate
    ))
}

impl Backend for StabRankBackend {
    fn name(&self) -> &str {
        "stabrank"
    }

    fn execute(
        &self,
        _circuit: &CircuitIR,
        _params: &ParameterBinding,
        _config: &ExecConfig,
    ) -> Result<ExecResult> {
        Err(OmegaError::Unsupported(
            "stabrank is an expectation-value backend; use `expectation()`, not \
             execute/sampling. For shots on a Clifford circuit use the `pauli` \
             backend, and for a statevector use `statevector`."
                .into(),
        ))
    }

    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<f64> {
        self.expectation_with_certificate(circuit, params, observable)
            .map(|(v, _)| v)
    }
}
