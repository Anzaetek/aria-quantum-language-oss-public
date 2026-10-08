// SPDX-License-Identifier: Apache-2.0
//! The Majorana-propagation engine: Heisenberg back-propagation of a Pauli
//! observable through a gate-based circuit, with every gate expressed as a
//! short sequence of **Majorana rotations** `exp(−iφ M_c/2)`.
//!
//! Contract (identical to `omega-backend-pauliprop`, on purpose): the run
//! returns `⟨0|U† O U|0⟩` together with a certificate whose `dropped_mass`
//! is the L1 coefficient mass discarded by truncation. Because every
//! monomial is Hermitian with `M² = 1`, `|⟨M⟩| ≤ 1` and the dropped mass is
//! a genuine **bound** on `|Δ⟨O⟩|`. An untruncated run reports `0`. A run
//! whose bound has stopped excluding anything is refused, not returned.
//!
//! What differs from pauliprop is only the basis, and therefore the
//! truncation knob: `max_length` cuts on the number of Majoranas in a
//! monomial, not on Pauli weight. For a JW-encoded fermionic circuit built
//! from Givens rotations (`Rbs`) a length-`m` monomial stays length-`m`
//! forever (each rotation swaps one Majorana for another), so `max_length`
//! set to the observable's own length makes the run exact with zero drop —
//! where the same circuit in the Pauli basis grows weight through the JW
//! strings. Interaction gates (`CP`, `CU3(0,0,λ)`) *do* grow length, by two
//! per split, and that growth is exactly what the cut then bounds.
//!
//! Every gate in the table below is a product of Pauli rotations. A Pauli
//! rotation `exp(−iθ/2 P)` is a Majorana rotation because a Pauli string is
//! `±M_b` under JW ([`MajoranaKey::from_pauli`]). Clifford gates become
//! rotations at `±π/2` where `cos = 0`, so they never split a term; they
//! cost one pass over the sum and no growth in term count.

use std::collections::HashMap;

use num_complex::Complex64;
use omega_core::circuit::{CircuitIR, GateKind, GateOp, ParamExpr};
use omega_core::defer_measure::{prepare_circuit_for_expectation, prepare_for_expectation};
use omega_core::error::{OmegaError, Result};
use omega_core::executor::{Backend, ExecConfig, ExecResult, Observable, PauliOp};
use omega_core::fermion::{FermionicOp, COEFF_EPS};
use omega_core::params::ParameterBinding;
use serde::{Deserialize, Serialize};

use crate::majorana::MajoranaKey;

/// Default ceiling on the number of live monomials. Same value and same
/// rationale as pauliprop's: a stop for a run that has left the regime the
/// engine is for, not a wall — raise it with `with_max_terms`.
pub const DEFAULT_MAX_TERMS: usize = 1 << 21;

/// The basis the engine's seed was built in. Recorded by [`finish`] from
/// which door called it, so a consumer learns it from the engine rather than
/// from whoever dispatched to the engine.
///
/// [`finish`]: MajoranaPropBackend::finish
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedBasis {
    /// A Pauli `Observable`, mapped monomial by monomial ([`seed_from_pauli`]).
    Pauli,
    /// A `FermionicOp`, seeded straight from its ladder terms
    /// ([`seed_from_fermionic`]) — PLAN-FERMIONIC F3. No Pauli image
    /// was built on this path.
    Ladder,
}

impl SeedBasis {
    pub fn as_str(self) -> &'static str {
        match self {
            SeedBasis::Pauli => "pauli",
            SeedBasis::Ladder => "ladder",
        }
    }
}

/// What a run discarded and where it stood. `dropped_mass` is a bound on
/// `|Δ⟨O⟩|`; the rest is context to read it by.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MajoranaPropCertificate {
    /// Which basis the seed was built in — see [`SeedBasis`].
    pub seed_basis: SeedBasis,
    /// L1 coefficient mass truncated away. `0.0` for an exact run.
    pub dropped_mass: f64,
    /// Monomials in the sum at the end of propagation.
    pub final_terms: usize,
    /// Largest number of monomials held at any point.
    pub peak_terms: usize,
    /// Coefficient floor in force (`0.0` = none).
    pub coeff_min: f64,
    /// Majorana-length cut in force (`None` = none). NOT a Pauli weight.
    pub max_length: Option<usize>,
    /// Term ceiling in force.
    pub max_terms: Option<usize>,
    /// `Σ|cᵢ|` of the observable — the range `⟨O⟩` is confined to a priori.
    pub observable_range: f64,
    /// The returned value; with `dropped_mass` gives the certified interval.
    pub value: f64,
    /// Rotations whose generator has length ≥ 4 and `sin φ ≠ 0` — the only
    /// ones that can change a monomial's length (arXiv:2503.18939v4,
    /// Thm 2(1): a length-2 generator leaves every length unchanged, which
    /// is why Givens-only circuits are exact at any cut ≥ the observable's).
    /// This is the `L` that Theorem 1's bound is a function of.
    pub branching_gates: usize,
    /// Theorem 1's a-priori bound on `E[(f − f^{(w₀)})²] / E‖O‖₂²` at
    /// `w₀ = max_length`, `N` = modes, `L = branching_gates`; `None` without a
    /// length cut. **Average-case over the paper's random ensemble, not a
    /// statement about this circuit** — see [`theorem1_mse_ratio`].
    pub apriori_mse_ratio: Option<f64>,
}

impl MajoranaPropCertificate {
    /// True when the run discarded nothing.
    pub fn is_exact(&self) -> bool {
        self.dropped_mass == 0.0
    }

    /// True when the bound still excludes something: `m < R + |v|`. The
    /// exact vacuity condition, same derivation as pauliprop's — the result
    /// interval `[v−m, v+m]` contains the a-priori `[−R, R]` iff `m ≥ R+|v|`.
    pub fn is_informative(&self) -> bool {
        self.dropped_mass < self.observable_range + self.value.abs()
    }

    /// `Some(true)` when the a-priori ratio is ≥ 1 — the ensemble bound then
    /// says nothing, which at small `N` (`w₀ ≥ N/e`) is the normal case.
    pub fn apriori_is_vacuous(&self) -> Option<bool> {
        self.apriori_mse_ratio.map(|r| r >= 1.0)
    }
}

/// Theorem 1 of Miller et al., *Simulation of Fermionic circuits using
/// Majorana Propagation*, arXiv:2503.18939v4 (eqs. 8–9): for a circuit of
/// `L` gates whose generators are random length-4 monomials with angles
/// uniform on `[0, 2π)`, a Fock initial state and a random homogeneous
/// observable `O`, the mean squared error of length-`w₀` truncation obeys
///
/// ```text
/// Δ(w₀,L) / E‖O‖₂²  ≤  2^{−(N−1)} + (e·w₀/N)^{w₀/2}                        L ≤ L*
///                    ≤  2^{−(N−1)} + 5e·w₀ · (16e·w₀/N)^{w₀/2} · e^{−L/2N}   L > L*
/// L* = 2N · ln(5e·w₀·4^{w₀})
/// ```
///
/// `‖O‖₂²` is the coefficient 2-norm `Σ c_b²` (normalised Hilbert–Schmidt),
/// which the propagation conserves. The two branches agree at `L*`.
///
/// What this is and is not: an **expectation over the ensemble**, so it
/// predicts nothing about one circuit — the measured `dropped_mass` is the
/// per-run bound; this is the prior. `L` must count only branching gates
/// (length ≥ 4); a length-2 generator never changes length (Thm 2(1)).
/// The `(e·w₀/N)^{w₀/2}` term is not monotone in `w₀`: its minimum is at
/// `w₀ = N/e²` and it is `< 1` only for `w₀ < N/e`, so a longer cut stops
/// buying anything at `N/e²` and the prior is vacuous past `N/e`. At
/// `N = 12, w₀ = 4` it is `0.82`; at `N = 52, w₀ = 6` it is `0.03`. The
/// bound is useful at chemistry sizes and near-vacuous at test sizes, and
/// callers must show it that way rather than as a guarantee.
pub fn theorem1_mse_ratio(n_modes: usize, branching_gates: usize, w0: usize) -> Option<f64> {
    if n_modes == 0 || w0 < 2 {
        return None;
    }
    Some(theorem1_mse_ratio_f(
        n_modes as f64,
        branching_gates as f64,
        w0 as f64,
    ))
}

/// The continuous form of [`theorem1_mse_ratio`], for checking the regime
/// switch is seamless. `L` real so `L = L*` can be hit exactly.
pub fn theorem1_mse_ratio_f(n: f64, l: f64, w0: f64) -> f64 {
    use std::f64::consts::E;
    let base = 2f64.powf(-(n - 1.0));
    let l_star = 2.0 * n * (5.0 * E * w0 * 4f64.powf(w0)).ln();
    if l <= l_star {
        base + (E * w0 / n).powf(w0 / 2.0)
    } else {
        base + 5.0 * E * w0 * (16.0 * E * w0 / n).powf(w0 / 2.0) * (-l / (2.0 * n)).exp()
    }
}

/// `Σ|cᵢ|` over an observable's terms — bounds `|⟨O⟩|` since `|⟨P⟩| ≤ 1`.
pub fn observable_l1_norm(observable: &Observable) -> f64 {
    observable.terms.iter().map(|(c, _)| c.abs()).sum()
}

/// Seed a sum from a Pauli observable: each string becomes `±M` via
/// [`MajoranaKey::from_pauli`]. The Pauli door.
pub fn seed_from_pauli(observable: &Observable, n: usize) -> Result<MajoranaSum> {
    let mut sum = MajoranaSum::new();
    for (coeff, paulis) in &observable.terms {
        let mut x = vec![false; n];
        let mut z = vec![false; n];
        for (q, op) in paulis {
            let q = *q as usize;
            if q >= n {
                return Err(OmegaError::Unsupported(format!(
                    "observable acts on qubit {q}, but the circuit has {n} \
                     (valid indices are 0..{}). Pauli terms must name \
                     qubits that exist in the circuit they are measured on.",
                    n.saturating_sub(1)
                )));
            }
            match op {
                PauliOp::I => {}
                PauliOp::X => x[q] = true,
                PauliOp::Z => z[q] = true,
                PauliOp::Y => {
                    x[q] = true;
                    z[q] = true;
                }
            }
        }
        let (key, e) = MajoranaKey::from_pauli(&x, &z);
        let s = if e == 0 { 1.0 } else { -1.0 };
        sum.add(key, coeff * s);
    }
    Ok(sum)
}

/// `i^e` for `e mod 4`.
#[inline]
fn ipow(e: u32) -> Complex64 {
    match e % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

/// Seed a sum straight from the ladder algebra — PLAN-FERMIONIC F3.
///
/// `a_p = (γ_{2p} + i γ_{2p+1})/2`, `a†_p = (γ_{2p} − i γ_{2p+1})/2`: the
/// same Jordan–Wigner encoding [`MajoranaKey::from_pauli`] realises
/// (`γ_{2p} = Z^{⊗p} X_p`, `γ_{2p+1} = Z^{⊗p} Y_p`), evaluated in the
/// engine's own basis. Each ladder splits every partial product in two and
/// [`MajoranaKey::mul`] carries the phase, so a `k`-ladder product costs
/// `2^{k+1} − 2` monomial products and nothing else.
///
/// What the Pauli detour costs on top, and what this path skips: the same
/// `2^{k+1} − 2` string products in `jordan_wigner_terms`, and then, for
/// every one of the `2^k` strings that survive, one monomial product per
/// non-identity site to re-condense it into a `MajoranaKey` — `O(n)` per
/// string for a hopping term, because of the `Z` string. The plan's phrase
/// "linear in terms" oversells it: a `k`-ladder product is `2^k` monomials
/// in either basis; the saving is the re-condensation, not the expansion.
///
/// Returns the sum and the number of intermediate partial products built,
/// observed rather than computed, so a test can tell whether this path ran or
/// was quietly routed through JW (`tests/fermionic_seed.rs` (iii)).
///
/// Refuses a ladder on a mode the circuit has no qubit for, and a
/// non-Hermitian operator (a complex coefficient on a Hermitian monomial),
/// each naming the offender — the same two refusals the Pauli door makes,
/// in this basis.
pub fn seed_from_fermionic(op: &FermionicOp, n: usize) -> Result<(MajoranaSum, usize)> {
    let mut acc: HashMap<MajoranaKey, Complex64> = HashMap::new();
    let mut expanded = 0usize;
    let half = Complex64::new(0.5, 0.0);
    for (coeff, prod) in &op.terms {
        let mut partial: Vec<(Complex64, MajoranaKey)> = vec![(*coeff, MajoranaKey::identity(n))];
        for l in prod {
            let p = l.mode as usize;
            if p >= n {
                return Err(OmegaError::Unsupported(format!(
                    "observable acts on mode {p}, but the circuit has {n} qubits \
                     (one mode per qubit under Jordan–Wigner; valid modes are \
                     0..{}). Ladder operators must name modes that exist in the \
                     circuit they are measured on.",
                    n.saturating_sub(1)
                )));
            }
            let g_re = MajoranaKey::from_indices(n, &[2 * p]);
            let g_im = MajoranaKey::from_indices(n, &[2 * p + 1]);
            let w_im = if l.dagger {
                Complex64::new(0.0, -0.5)
            } else {
                Complex64::new(0.0, 0.5)
            };
            let mut next = Vec::with_capacity(partial.len() * 2);
            for (c, key) in &partial {
                for (w, g) in [(half, &g_re), (w_im, &g_im)] {
                    let (k2, e) = key.mul(g);
                    next.push((c * w * ipow(e), k2));
                }
            }
            expanded += next.len();
            partial = next;
        }
        for (c, k) in partial {
            *acc.entry(k).or_insert(Complex64::new(0.0, 0.0)) += c;
        }
    }
    let mut sum = MajoranaSum::new();
    for (k, c) in acc {
        if c.norm() < COEFF_EPS {
            continue;
        }
        if c.im.abs() > 1e-10 {
            return Err(OmegaError::Unsupported(format!(
                "the Majorana image of this fermionic operator has a complex \
                 coefficient ({c} on γ{:?}), so the operator is not Hermitian \
                 and has no real expectation value. Symmetrise it (`op + \
                 op.dagger()`) if that is what was meant.",
                k.indices()
            )));
        }
        sum.add(k, c.re);
    }
    Ok((sum, expanded))
}

/// The Heisenberg-evolved observable as a real-coefficient sum of Hermitian
/// monomials. Coefficients are real throughout: the seed is real (a Pauli
/// string is `±M`), and the rotation rule mixes with `cos φ`, `sin φ`.
#[derive(Debug, Clone, Default)]
pub struct MajoranaSum {
    pub terms: HashMap<MajoranaKey, f64>,
    /// L1 mass discarded so far.
    pub dropped_mass: f64,
}

impl MajoranaSum {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn add(&mut self, key: MajoranaKey, coeff: f64) {
        let e = self.terms.entry(key).or_insert(0.0);
        *e += coeff;
    }

    /// `Σ|cᵢ|` over the sum — bounds `|⟨O⟩|` since `|⟨M⟩| ≤ 1`.
    ///
    /// Summed in key order, not map order: `terms` is a `HashMap`, whose
    /// walk order is randomised per instance, and a float sum taken in a
    /// random order lands on different ulps each run. This value is the
    /// certificate's `observable_range`, and a certificate has to be the same
    /// bits on every run.
    pub fn l1_norm(&self) -> f64 {
        let mut ordered: Vec<(&MajoranaKey, f64)> =
            self.terms.iter().map(|(k, c)| (k, c.abs())).collect();
        ordered.sort_unstable_by(|a, b| a.0.cmp(b.0));
        ordered.into_iter().map(|(_, a)| a).sum()
    }

    /// [`Observable::dephase`] in the Majorana basis: drop every monomial
    /// that acts as `X` or `Y` on a measured qubit. Under JW a monomial's
    /// Pauli on qubit `q` is `X`/`Y` exactly when one of `γ_{2q}, γ_{2q+1}`
    /// is present and the other absent — the `Z` strings contributed by
    /// higher Majoranas only turn `X` into `Y`, never into `I`/`Z` — so the
    /// test is on two bits. Dropped terms are not `dropped_mass`: they are
    /// removed before evolution, as the Pauli door removes them before
    /// seeding.
    pub fn dephase(&mut self, measured: &[u32]) {
        if measured.is_empty() {
            return;
        }
        self.terms.retain(|k, _| {
            measured
                .iter()
                .all(|&q| k.bit(2 * q as usize) == k.bit(2 * q as usize + 1))
        });
    }

    /// Drop terms below `coeff_min` (magnitude) or longer than `max_length`;
    /// bank the dropped magnitude.
    pub fn truncate(&mut self, coeff_min: f64, max_length: Option<usize>) {
        let mut dropped = 0.0;
        self.terms.retain(|k, c| {
            let too_small = c.abs() < coeff_min;
            let too_long = max_length.is_some_and(|m| k.length() as usize > m);
            if too_small || too_long {
                dropped += c.abs();
                false
            } else {
                true
            }
        });
        self.dropped_mass += dropped;
    }

    /// `⟨0|Σ c_b M_b|0⟩` — the vacuum readout.
    pub fn vacuum_expectation(&self) -> f64 {
        self.terms
            .iter()
            .map(|(k, c)| c * k.vacuum_expectation())
            .sum()
    }
}

/// Majorana propagation backend. Expectation values only.
#[derive(Debug, Clone)]
pub struct MajoranaPropBackend {
    coeff_min: f64,
    max_length: Option<usize>,
    max_terms: Option<usize>,
    max_dropped_mass: Option<f64>,
}

impl Default for MajoranaPropBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MajoranaPropBackend {
    /// Exact engine: no coefficient floor, no length cut, default term ceiling.
    pub fn new() -> Self {
        Self {
            coeff_min: 0.0,
            max_length: None,
            max_terms: Some(DEFAULT_MAX_TERMS),
            max_dropped_mass: None,
        }
    }

    /// Truncating engine: drop coefficients below `coeff_min` and monomials
    /// longer than `max_length` Majoranas after every rotation.
    pub fn with_truncation(coeff_min: f64, max_length: Option<usize>) -> Self {
        Self {
            coeff_min,
            max_length,
            ..Self::new()
        }
    }

    /// Builder: term ceiling (`None` = unlimited). Does not change the answer.
    pub fn with_max_terms(mut self, max_terms: Option<usize>) -> Self {
        self.max_terms = max_terms;
        self
    }

    /// Builder: explicit dropped-mass ceiling for the vacuity gate. `None`
    /// means the exact vacuity point `R + |v|` of each result.
    pub fn with_max_dropped_mass(mut self, ceiling: Option<f64>) -> Self {
        self.max_dropped_mass = ceiling;
        self
    }

    pub fn coeff_min(&self) -> f64 {
        self.coeff_min
    }
    pub fn max_length(&self) -> Option<usize> {
        self.max_length
    }
    pub fn max_terms(&self) -> Option<usize> {
        self.max_terms
    }

    fn truncating(&self) -> bool {
        self.coeff_min > 0.0 || self.max_length.is_some()
    }

    /// The expectation and its dropped-mass budget.
    pub fn expectation_with_budget(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<(f64, f64)> {
        circuit.refuse_qudits(self.name())?;
        self.expectation_with_certificate(circuit, params, observable)
            .map(|(v, c)| (v, c.dropped_mass))
    }

    /// The gated entry point every other door routes through: a result whose
    /// bound excludes nothing is refused rather than returned.
    pub fn expectation_with_certificate(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<(f64, MajoranaPropCertificate)> {
        circuit.refuse_qudits(self.name())?;
        let (deferred, observable) = prepare_for_expectation(circuit, observable)?;
        let seed = seed_from_pauli(&observable, deferred.num_qubits as usize)?;
        let range = observable_l1_norm(&observable);
        self.finish(&deferred, params, seed, range, SeedBasis::Pauli)
    }

    /// The same gated entry point for a fermionic observable, seeded straight
    /// into the Majorana basis by [`seed_from_fermionic`] — PLAN-FERMIONIC
    /// F3. No Pauli `Observable` is built anywhere on this path; the
    /// measurement deferral dephases the Majorana sum directly
    /// ([`MajoranaSum::dephase`]), and `observable_range` is the L1 mass of
    /// the seeded sum, which equals the Pauli-basis L1 the other door reports
    /// because a Pauli string and its Majorana monomial differ by a sign only
    /// (`tests/fermionic_seed.rs` (i) holds the two doors to the same
    /// certificate).
    pub fn expectation_fermionic_with_certificate(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        op: &FermionicOp,
    ) -> Result<(f64, MajoranaPropCertificate)> {
        // The Pauli door refuses qudits and so must this one: the two share
        // `finish`, not their entrance, and the CLI's qudit matrix caught
        // exactly this door running a qutrit circuit as qubits (0.585, exit
        // 0) when only the Pauli door had the check.
        circuit.refuse_qudits(self.name())?;
        let (deferred, measured) = prepare_circuit_for_expectation(circuit)?;
        let (mut seed, _) = seed_from_fermionic(op, deferred.num_qubits as usize)?;
        seed.dephase(&measured);
        let range = seed.l1_norm();
        self.finish(&deferred, params, seed, range, SeedBasis::Ladder)
    }

    /// Propagate a seeded sum and gate the result — shared by both doors so
    /// the certificate is built in exactly one place. `seed_basis` is the
    /// caller's identity, written into the certificate.
    fn finish(
        &self,
        deferred: &CircuitIR,
        params: &ParameterBinding,
        seed: MajoranaSum,
        observable_range: f64,
        seed_basis: SeedBasis,
    ) -> Result<(f64, MajoranaPropCertificate)> {
        let (sum, peak, branching) = self.propagate(deferred, params, seed)?;
        let value = sum.vacuum_expectation();
        let cert = MajoranaPropCertificate {
            seed_basis,
            value,
            dropped_mass: sum.dropped_mass,
            final_terms: sum.terms.len(),
            peak_terms: peak,
            coeff_min: self.coeff_min,
            max_length: self.max_length,
            max_terms: self.max_terms,
            observable_range,
            branching_gates: branching,
            apriori_mse_ratio: self
                .max_length
                .and_then(|w0| theorem1_mse_ratio(deferred.num_qubits as usize, branching, w0)),
        };
        let ceiling = self
            .max_dropped_mass
            .unwrap_or(cert.observable_range + cert.value.abs());
        if cert.dropped_mass >= ceiling {
            return Err(OmegaError::Unsupported(format!(
                "majoranaprop: truncation discarded an L1 coefficient mass of \
                 {:.4e}, which exceeds the ceiling of {:.4e}. That bound is \
                 correct and it excludes nothing: |<O>| cannot leave \
                 [-{:.4e}, {:.4e}] to begin with. Refusing rather than returning \
                 a number with no information in it. Options: tighten \
                 `--truncate`, raise `--max-length` (a longer cut discards less), \
                 raise `--max-terms N` so the run needs less truncation — that \
                 one does not change the answer — or set the ceiling explicitly \
                 via `--max-dropped-mass`. Final terms {}, peak {}.",
                cert.dropped_mass,
                ceiling,
                cert.observable_range,
                cert.observable_range,
                cert.final_terms,
                cert.peak_terms,
            )));
        }
        Ok((value, cert))
    }

    /// Walk the circuit backwards over a seeded sum. Returns the evolved sum,
    /// the peak term count, and how many rotations had a length-≥4 generator
    /// with a non-zero angle (the branching ones).
    fn propagate(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        mut sum: MajoranaSum,
    ) -> Result<(MajoranaSum, usize, usize)> {
        let n = circuit.num_qubits as usize;
        let mut peak = sum.terms.len();
        let mut branching = 0usize;
        for op in circuit.ops.iter().rev() {
            if op.condition.is_some() {
                return Err(OmegaError::Unsupported(
                    "majoranaprop: a classically-conditioned gate makes the circuit a \
                     mixture over measurement outcomes, which cannot be represented by \
                     observable conjugation; use the statevector or MPS backend"
                        .into(),
                ));
            }
            let rots = rotations_for(op, params, n)?;
            // Forward gate U = U_k ⋯ U_1 ⇒ U† O U applies U_k† … first, U_1† last.
            for (c, phi) in rots.iter().rev() {
                // A π rotation (sin = 0) only flips a sign — an `x` prep gate
                // is one, and under JW it is a long string — so it cannot move
                // mass across a length cut. ±π/2 relabellings do change length.
                if phi.sin().abs() > 1e-15 && c.length() as usize >= 4 {
                    branching += 1;
                }
                self.branch(&mut sum, c, *phi)?;
                peak = peak.max(sum.terms.len());
            }
            if self.truncating() {
                sum.truncate(self.coeff_min, self.max_length);
            }
        }
        Ok((sum, peak, branching))
    }

    /// `O ← U† O U` for `U = exp(−iφ M_c/2)`.
    fn branch(&self, sum: &mut MajoranaSum, c: &MajoranaKey, phi: f64) -> Result<()> {
        if phi == 0.0 {
            return Ok(());
        }
        let (cos, sin) = (phi.cos(), phi.sin());
        // Snap Clifford angles so a ±π/2 rotation is a pure relabelling and
        // never leaves a 6e-17 ghost term behind.
        let cos = if cos.abs() < 1e-15 { 0.0 } else { cos };
        let sin = if sin.abs() < 1e-15 { 0.0 } else { sin };
        let mut out = MajoranaSum {
            terms: HashMap::with_capacity(sum.terms.len()),
            dropped_mass: sum.dropped_mass,
        };
        for (key, coeff) in sum.terms.drain() {
            if key.commutes(c) {
                out.add(key, coeff);
            } else {
                let (k2, s) = key.rotate_by(c);
                if cos != 0.0 {
                    out.add(key, coeff * cos);
                }
                if sin != 0.0 {
                    out.add(k2, coeff * sin * s);
                }
            }
            if let Some(cap) = self.max_terms {
                if out.terms.len() > cap {
                    return Err(over_cap(out.terms.len(), cap, self.truncating()));
                }
            }
        }
        *sum = out;
        Ok(())
    }
}

fn over_cap(len: usize, cap: usize, truncating: bool) -> OmegaError {
    let hint = if truncating {
        "tighten `--truncate` or lower `--max-length` to discard more (reported in \
         dropped_mass), or raise `--max-terms`"
    } else {
        "this is an exact run; raise `--max-terms N` (costs memory, not accuracy) or \
         truncate with `--truncate C` / `--max-length L` (reported in dropped_mass)"
    };
    OmegaError::Unsupported(format!(
        "majoranaprop: the Heisenberg-evolved observable reached {len} Majorana \
         monomials, above the ceiling of {cap}; {hint}"
    ))
}

fn resolve(expr: &ParamExpr, params: &ParameterBinding) -> Result<f64> {
    params.resolve(expr)
}

/// One Majorana rotation `exp(−iφ M_c/2)`.
type Rot = (MajoranaKey, f64);

/// `exp(−iθ/2 P)` for the Pauli string `P` given as `(qubit, 'X'|'Y'|'Z')`
/// sites, as the Majorana rotation `(M, ±θ)` with the JW sign folded in.
fn pauli_rot(n: usize, sites: &[(usize, char)], theta: f64) -> Rot {
    let mut x = vec![false; n];
    let mut z = vec![false; n];
    for &(q, p) in sites {
        match p {
            'X' => x[q] = true,
            'Y' => {
                x[q] = true;
                z[q] = true;
            }
            'Z' => z[q] = true,
            _ => unreachable!("pauli_rot site must be X, Y or Z"),
        }
    }
    let (key, e) = MajoranaKey::from_pauli(&x, &z);
    let s = if e == 0 { 1.0 } else { -1.0 };
    (key, theta * s)
}

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// The gate table: each gate as Pauli rotations in **forward** order, global
/// phases dropped (they cancel under conjugation).
fn rotations_for(op: &GateOp, params: &ParameterBinding, n: usize) -> Result<Vec<Rot>> {
    let q = |i: usize| op.qubits[i].0 as usize;
    let p = |i: usize| resolve(&op.params[i], params);
    let r1 = |a: usize, ax: char, th: f64| pauli_rot(n, &[(a, ax)], th);
    let r2 =
        |a: usize, ax: char, b: usize, bx: char, th: f64| pauli_rot(n, &[(a, ax), (b, bx)], th);
    // CP(λ) = e^{iλ/4} · Rz(λ/2)⊗Rz(λ/2) · exp(+iλ/4 ZZ)
    let cp = |a: usize, b: usize, lam: f64| -> Vec<Rot> {
        vec![
            r1(a, 'Z', lam / 2.0),
            r1(b, 'Z', lam / 2.0),
            r2(a, 'Z', b, 'Z', -lam / 2.0),
        ]
    };
    // H = Rx(π/2) Rz(π/2) Rx(π/2) up to phase.
    let h = |a: usize| -> Vec<Rot> {
        vec![
            r1(a, 'X', FRAC_PI_2),
            r1(a, 'Z', FRAC_PI_2),
            r1(a, 'X', FRAC_PI_2),
        ]
    };
    // CX = (I⊗H) CZ (I⊗H).
    let cx = |c: usize, t: usize| -> Vec<Rot> {
        let mut v = h(t);
        v.extend(cp(c, t, PI));
        v.extend(h(t));
        v
    };
    // CCZ = exp(iπ n1 n2 n3): seven commuting diagonal rotations at ±π/4.
    let ccz = |a: usize, b: usize, c: usize| -> Vec<Rot> {
        vec![
            r1(a, 'Z', FRAC_PI_4),
            r1(b, 'Z', FRAC_PI_4),
            r1(c, 'Z', FRAC_PI_4),
            r2(a, 'Z', b, 'Z', -FRAC_PI_4),
            r2(a, 'Z', c, 'Z', -FRAC_PI_4),
            r2(b, 'Z', c, 'Z', -FRAC_PI_4),
            pauli_rot(n, &[(a, 'Z'), (b, 'Z'), (c, 'Z')], FRAC_PI_4),
        ]
    };
    let ccx = |a: usize, b: usize, t: usize| -> Vec<Rot> {
        let mut v = h(t);
        v.extend(ccz(a, b, t));
        v.extend(h(t));
        v
    };

    Ok(match op.gate {
        // Qudit gates (PLAN-QUDIT.md Q2): a qudit circuit is refused at this
        // engine's door by `refuse_qudits`, and on d = 2 wires these gates
        // have no Majorana rotation table entry yet — refused by name rather
        // than swept into a wildcard.
        GateKind::Rxy | GateKind::CSum => {
            return Err(OmegaError::Unsupported(format!(
                "majoranaprop: {:?} is a qudit gate with no rotation-table entry; on qubit \
                 wires use rx/ry (for rxy) or cx (for csum)",
                op.gate
            )))
        }
        GateKind::Id | GateKind::Barrier | GateKind::Measure => vec![],
        GateKind::X => vec![r1(q(0), 'X', PI)],
        GateKind::Y => vec![r1(q(0), 'Y', PI)],
        GateKind::Z => vec![r1(q(0), 'Z', PI)],
        GateKind::S => vec![r1(q(0), 'Z', FRAC_PI_2)],
        GateKind::Sdg => vec![r1(q(0), 'Z', -FRAC_PI_2)],
        GateKind::T => vec![r1(q(0), 'Z', FRAC_PI_4)],
        GateKind::Tdg => vec![r1(q(0), 'Z', -FRAC_PI_4)],
        GateKind::Sx => vec![r1(q(0), 'X', FRAC_PI_2)],
        GateKind::Sxdg => vec![r1(q(0), 'X', -FRAC_PI_2)],
        GateKind::H => h(q(0)),
        GateKind::Rx => vec![r1(q(0), 'X', p(0)?)],
        GateKind::Ry => vec![r1(q(0), 'Y', p(0)?)],
        GateKind::Rz | GateKind::U1 => vec![r1(q(0), 'Z', p(0)?)],
        // U3(θ,φ,λ) = Rz(φ) Ry(θ) Rz(λ) up to phase; forward order Rz(λ) first.
        GateKind::U3 => {
            let (th, ph, lam) = (p(0)?, p(1)?, p(2)?);
            vec![r1(q(0), 'Z', lam), r1(q(0), 'Y', th), r1(q(0), 'Z', ph)]
        }
        GateKind::U2 => {
            let (ph, lam) = (p(0)?, p(1)?);
            vec![
                r1(q(0), 'Z', lam),
                r1(q(0), 'Y', FRAC_PI_2),
                r1(q(0), 'Z', ph),
            ]
        }
        GateKind::CZ => cp(q(0), q(1), PI),
        GateKind::CX => cx(q(0), q(1)),
        // CY = (I⊗S) CX (I⊗S†), forward: S† on target first.
        GateKind::CY => {
            let mut v = vec![r1(q(1), 'Z', -FRAC_PI_2)];
            v.extend(cx(q(0), q(1)));
            v.push(r1(q(1), 'Z', FRAC_PI_2));
            v
        }
        // SWAP = exp(iπ/4 (XX + YY + ZZ)) up to phase; the three commute.
        GateKind::Swap => vec![
            r2(q(0), 'X', q(1), 'X', -FRAC_PI_2),
            r2(q(0), 'Y', q(1), 'Y', -FRAC_PI_2),
            r2(q(0), 'Z', q(1), 'Z', -FRAC_PI_2),
        ],
        // CRz(θ) = exp(−iθ/2 |1⟩⟨1|⊗Z) = exp(−iθ/4 Z_t) exp(+iθ/4 Z_c Z_t).
        GateKind::CRz => {
            let th = p(0)?;
            vec![r1(q(1), 'Z', th / 2.0), r2(q(0), 'Z', q(1), 'Z', -th / 2.0)]
        }
        GateKind::CU3 => {
            let (th, ph, lam) = (p(0)?, p(1)?, p(2)?);
            if th.abs() > 1e-12 || ph.abs() > 1e-12 {
                return Err(OmegaError::Unsupported(format!(
                    "majoranaprop supports CU3 only in its diagonal form CU3(0, 0, lambda) \
                     — the controlled phase `cp`/`cu1` lower to — and this one is \
                     CU3({th}, {ph}, {lam}). Use the statevector or MPS backend."
                )));
            }
            cp(q(0), q(1), lam)
        }
        // Rbs(θ) = exp(−iθ/2 (Y_pX_q − X_pY_q)); the two terms commute.
        GateKind::Rbs => {
            let th = p(0)?;
            vec![r2(q(0), 'Y', q(1), 'X', th), r2(q(0), 'X', q(1), 'Y', -th)]
        }
        GateKind::CCX => ccx(q(0), q(1), q(2)),
        // CSwap(c; a, b) = CX(b→a) · CCX(c, a → b) · CX(b→a).
        GateKind::CSwap => {
            let mut v = cx(q(2), q(1));
            v.extend(ccx(q(0), q(1), q(2)));
            v.extend(cx(q(2), q(1)));
            v
        }
        GateKind::Reset => {
            return Err(OmegaError::Unsupported(
                "majoranaprop: `reset` is a non-unitary channel (trace out and \
                 re-prepare) that observable conjugation cannot express; use the \
                 statevector or MPS backend"
                    .into(),
            ));
        }
        GateKind::PhaseShifter | GateKind::BeamSplitterRx => {
            return Err(OmegaError::Unsupported(
                "majoranaprop: photonic operations are not qubit gates; use the \
                 photonics backend"
                    .into(),
            ));
        }
        GateKind::Custom(_) => {
            return Err(OmegaError::Unsupported(
                "majoranaprop: custom gates must be lowered to the built-in gate set \
                 first"
                    .into(),
            ));
        }
    })
}

impl Backend for MajoranaPropBackend {
    fn name(&self) -> &str {
        "majoranaprop"
    }

    fn execute(
        &self,
        _circuit: &CircuitIR,
        _params: &ParameterBinding,
        _config: &ExecConfig,
    ) -> Result<ExecResult> {
        Err(OmegaError::Unsupported(
            "majoranaprop is an expectation-value backend; use `expectation()`, not \
             execute/sampling"
                .into(),
        ))
    }

    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<f64> {
        circuit.refuse_qudits(self.name())?;
        self.expectation_with_certificate(circuit, params, observable)
            .map(|(v, _)| v)
    }
}
