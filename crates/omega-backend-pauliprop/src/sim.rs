//! Pauli-propagation backend: backward Heisenberg evolution of an observable.
//!
//! `⟨O⟩ = ⟨0…0| U† O U |0…0⟩` is computed by conjugating the observable (a
//! [`PauliSum`]) by each gate from last to first, then reading off the all-`I/Z`
//! terms. Clifford gates map one Pauli to one Pauli (no growth — exact and
//! width-unbounded); Pauli rotations branch one Pauli into two (the tree),
//! truncated by coefficient magnitude / Pauli weight.
//!
//! See `PAULI_PROPAGATION_PLAN.md` in the `quantum` repo (item 17).

use num_complex::Complex64;
use omega_core::circuit::{CircuitIR, GateKind, GateOp, ParamExpr};
use omega_core::defer_measure::prepare_for_expectation;
use omega_core::error::{OmegaError, Result};
use omega_core::executor::{Backend, ExecConfig, ExecResult, Observable, PauliOp};
use omega_core::noise::NoiseModel;
use omega_core::params::ParameterBinding;

use crate::pauli::{mul_raw_packed, PauliKey, PauliSum};

const I: Complex64 = Complex64::new(0.0, 1.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// Accelerator hook for the non-Clifford **branch** step — the hot per-term loop
/// `P → cosθ·P + i sinθ·(R·P)`. The generator `R` is passed in packed u64-word
/// symplectic form (`rx`, `rz`, `factor`), with the already-computed `cos`/`sin`,
/// the `max_freq` cap, and qubit count `n`. The hook replaces `sum` with the
/// branched result and returns `true` iff it handled the work; returning `false`
/// (e.g. no GPU present, or too few terms to be worth a device round-trip) lets
/// the CPU path run unchanged, so semantics are identical either way. The CUDA
/// implementation lives in `omega-backend-pauliprop-cuda`.
/// # Deferred: device residency, and why the cheap version is WRONG
///
/// This signature is why the GPU branch hook cannot win. It takes a **host**
/// `&mut PauliSum` and is called **once per rotation gate**, so every gate pays
/// a full round trip: rebuild the SoA, upload, launch, synchronise, download,
/// merge. Measured with `PAULIPROP_GPU_PROFILE=1` after the packed-key and
/// buffer-reuse work, the GPU arm sits at **0.33x-0.81x** of the CPU, and the
/// remaining cost is that round trip — `launch+sync` 42-51%, `download`
/// 27-32% in the wide/shallow regime; `merge` 86-89% in the deep regime.
///
/// The fix is a contract change: an engine that OWNS the sum for a run —
/// `load` once, then `map_one`/`map_two`/`branch` on device, `finish` once —
/// so the sum crosses the bus twice per CIRCUIT instead of twice per GATE.
/// That also makes the multi-rotation gates cheap: `CRz` issues 2 branches,
/// `U3`/`CU1` 3, and `CCX`/`CSwap` **7** (the CCZ decomposition), each of which
/// is a separate round trip today.
///
/// ## The shortcut that does not work
///
/// The obvious cheap version — batch a gate's `k` rotations into one hook call
/// and merge once at the end instead of `k` times — is **incorrect**, not merely
/// approximate. When a Pauli is reached by two paths the merge keeps
/// `freq = min` ([`Weighted`]), and a later rotation's `max_freq` check reads
/// that frequency. Unmerged, the copy carrying the LARGER `freq` can be
/// truncated where the merged term would have survived. The final expectation
/// would differ, and so would `dropped_mass` — the CERTIFIED error bound.
///
/// So batching requires a **device-side merge** (sort-by-key + segmented
/// reduce, or a device hash table). There is no way around it, and that merge
/// must preserve `freq = min` and the `dropped_mass` bound exactly. Both are
/// mutation-checked on the host side today
/// (`omega-backend-pauliprop-cuda/tests/parity.rs`); a device implementation
/// needs the same treatment before it is trusted.
///
/// ## Whether it is worth doing
///
/// Not obviously. The merge is 86-89% of the branch step and the CPU performs
/// the *same* operation, so a device merge has to beat a host `HashMap` at the
/// thing that dominates — and `BACKEND-CROSSOVER.md` already recommends the CPU
/// for PauliProp on this hardware. A faster hasher was tried and was SLOWER on
/// wide keys (see [`crate::pauli::PauliSum`]), which points at the map rather
/// than the hash. **Speeding up the CPU merge would help the path that actually
/// runs**; a device merge only helps the path we currently tell people not to
/// use. Revisit if either changes: an amd64 box with a discrete GPU (where
/// per-gate transfers cost real time and the verdict could invert), or a
/// workload whose sums are large enough that the kernel stops being noise.
pub type BranchHook = fn(
    sum: &mut PauliSum,
    rx: &[u64],
    rz: &[u64],
    factor: Complex64,
    cos: f64,
    sin: f64,
    max_freq: Option<u32>,
    n: usize,
) -> bool;

/// Default ceiling on the number of Pauli terms held at once.
///
/// # Why this one truncation knob is ON by default
///
/// Every other knob here (`coeff_min`, `max_weight`, `max_freq`) is an
/// *approximation*: it discards amplitude and reports the loss as
/// `dropped_mass`. Leaving those off by default is right — an exact engine
/// should be exact until asked otherwise.
///
/// This one is not an approximation. It is a refusal, and it is on by default
/// because the alternative is not a worse answer but no machine. Each
/// non-Clifford rotation can double the term count, and the worst case is not
/// exotic: a T gate is θ = π/4, where `cos θ = sin θ = 0.7071`, so **both
/// children carry equal weight and coefficient truncation cannot prune either
/// one**. A circuit with k T-like rotations is 2^k terms with nothing to drop.
///
/// Until now that was bounded only by accident — the gates that would produce
/// it (`CCX`, and anything decomposing into a T ladder) happened to be refused
/// as unsupported. Any work that widens gate coverage removes that accident, so
/// the bound has to become explicit first.
///
/// The value is a term count rather than a byte budget because a term's cost
/// depends on `n`; at 2^21 terms and a typical 100–200 B per term this is a few
/// hundred MB, which leaves room on a small host while permitting far more
/// branching than any exact run needs.
pub const DEFAULT_MAX_TERMS: usize = 1 << 21;

thread_local! {
    /// Gates skipped because they act entirely outside the Pauli sum's support.
    ///
    /// **Thread-local, deliberately.** A process-wide counter cannot give a
    /// per-test delta: cargo runs tests in parallel threads, so a `before`/
    /// `after` pair around one propagation silently absorbs every skip that
    /// other tests performed at the same time. The first version of this was a
    /// global `AtomicU64` and it reported 98 skips for a run that performed
    /// none of them. Propagation is single-threaded (no rayon, no spawning
    /// anywhere in this engine), so a thread-local count is exact.
    ///
    /// Always on: it increments at most once per GATE, never per term, on a
    /// path that just avoided rebuilding an entire map.
    ///
    /// It exists because the skip is otherwise **unobservable** — it changes no
    /// value a caller can see — so a value-comparison test passes whether or
    /// not the guard is present, and would keep passing if the guard were
    /// deleted. Same reason `gpu_branch_count()` exists in the CUDA crate.
    static GATES_SKIPPED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

thread_local! {
    /// Peak `terms.len()` seen on this thread, for testing the bound in
    /// `bound.rs` against what the engine actually reaches.
    ///
    /// Sampled at the end of every `branch`, which is the ONLY place the sum
    /// grows — and crucially that means it is sampled MID-GATE. A gate like
    /// `CCX` calls `branch` seven times, and the engine's own ceiling check
    /// runs once after the whole dispatch, so a peak-at-gate-boundary sample
    /// would miss the real maximum by up to 2^7 and would "confirm" a bound the
    /// engine violates in flight.
    ///
    /// Thread-local for the same reason as `GATES_SKIPPED`: cargo runs tests in
    /// parallel and a process-wide counter reports other tests' work.
    static PEAK_TERMS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Peak term count seen on this thread since [`reset_peak_terms`].
pub fn peak_terms() -> usize {
    PEAK_TERMS.with(|c| c.get())
}

/// Reset the per-thread peak.
pub fn reset_peak_terms() {
    PEAK_TERMS.with(|c| c.set(0));
}

/// Everything a pauliprop run must be able to say about its own accuracy.
///
/// The engine has always computed `dropped_mass` — an L1 bound on `|Δ⟨O⟩|`, and
/// a genuine bound rather than an estimate, since `|⟨P⟩| ≤ 1` for every Pauli
/// string. What it never did was **consult it**, and what the CLI never did was
/// put it anywhere a machine could read: it went to stderr as prose, so a JSON
/// consumer got a bare number with no error bar and no way to tell a truncated
/// run from an exact one.
///
/// That is verbatim the defect `omega-backend-mps` records fixing at
/// `sim.rs:528` — *"The truncation certificate gates the RESULT, in every mode.
/// It was computed and printed but consulted by nothing, so a run that
/// discarded 6.5x the state returned a distribution half of which was wrong —
/// with the evidence on screen."* Same sentence, one crate over.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PauliPropCertificate {
    /// L1 mass of discarded coefficients: a **bound** on `|Δ⟨O⟩|`, not an
    /// estimate. `0.0` for an exact run.
    pub dropped_mass: f64,
    /// Terms in the final Pauli sum.
    pub final_terms: usize,
    /// Largest term count reached at any point during propagation. Sampled
    /// mid-gate, so it bounds the true peak rather than the gate-boundary one.
    pub peak_terms: usize,
    /// Coefficient floor in force; `0.0` means none.
    pub coeff_min: f64,
    /// Pauli-weight cap in force.
    pub max_weight: Option<usize>,
    /// Split-frequency cap in force.
    pub max_freq: Option<u32>,
    /// Term ceiling in force.
    pub max_terms: Option<usize>,
    /// `Σ|cᵢ|` over the observable's terms — `⟨O⟩` cannot leave `[−R, R]`,
    /// since `|⟨P⟩| ≤ 1` for each Pauli string.
    pub observable_range: f64,
    /// The value this certificate belongs to. Needed by
    /// [`Self::is_informative`], which asks whether `[v − m, v + m]` is
    /// narrower than `[−R, R]` — a question about where the interval sits, not
    /// only how wide it is.
    pub value: f64,
}

impl PauliPropCertificate {
    /// True when the run discarded nothing.
    pub fn is_exact(&self) -> bool {
        self.dropped_mass == 0.0
    }

    /// True when the error bound still excludes *something*.
    ///
    /// A bound is only worth reading if it rules out a value the answer could
    /// otherwise have taken. The result asserts `⟨O⟩ ∈ [v − m, v + m]`; it was
    /// already known that `⟨O⟩ ∈ [−R, R]`. So the run learned nothing exactly
    /// when the first interval contains the second, which happens iff
    ///
    /// ```text
    /// m >= R + |v|
    /// ```
    ///
    /// and this returns `m < R + |v|`. **That is the exact condition, not a
    /// chosen threshold**, which matters because two plausible thresholds
    /// differ by 2× and the repo already contained both intuitions: this gate
    /// was first written as `m < R`, while `omega-cli/tests/dropped_mass_is_a_bound.rs`
    /// had independently used `m < 2R` ("`⟨O⟩ ∈ [−1, 1]`, so `budget ≥ 2` holds
    /// for free"). `m < R` is the special case at `v = 0` and refuses runs that
    /// do still exclude something; `m < 2R` is the loosest form and admits some
    /// that do not. The exact test agrees with `m < R` on the case that
    /// prompted all this — the engine truncating every term and returning
    /// exactly `0.0` — and is never stricter than it.
    ///
    /// Measured on a 20-qubit depth-16 hardware-efficient ansatz, `⟨Z₀⟩`
    /// (`observable_range = 1.0`), against a statevector reference:
    ///
    /// | `--truncate` | returns | `dropped_mass` |
    /// |---|---|---|
    /// | 1e-3 | −0.0042647207 | **1.41e+03** |
    /// | 1e-2 | **0.0000000000** | 1.44e+02 |
    /// | 1e-1 | **0.0000000000** | 1.35e+01 |
    ///
    /// The bound was honoured every time and told the caller nothing every
    /// time, and at two cutoffs the engine had truncated away *every term* and
    /// returned a confident `0.0`. "The bound was never violated" is trivially
    /// true of a bound that cannot be violated.
    pub fn is_informative(&self) -> bool {
        self.dropped_mass < self.observable_range + self.value.abs()
    }
}

/// `Σ|cᵢ|` over an observable's terms.
///
/// Since `|⟨P⟩| ≤ 1` for any Pauli string, this bounds `|⟨O⟩|` — so it is both
/// the range the answer lives in and the threshold past which an error bound
/// stops excluding anything.
pub fn observable_l1_norm(observable: &Observable) -> f64 {
    observable.terms.iter().map(|(c, _)| c.abs()).sum()
}

#[inline]
fn note_peak(n: usize) {
    PEAK_TERMS.with(|c| {
        if n > c.get() {
            c.set(n);
        }
    });
}

/// Gates skipped as out-of-support **on the calling thread**.
pub fn gates_skipped() -> u64 {
    GATES_SKIPPED.with(|c| c.get())
}

#[inline]
fn note_skipped() {
    GATES_SKIPPED.with(|c| c.set(c.get() + 1));
}

/// Pauli-propagation simulator. Truncation off (`coeff_min = 0`, `max_weight =
/// None`) ⇒ exact; tighten either knob to approximate deep non-Clifford
/// circuits. Exact and cheap for Clifford circuits at any width.
#[derive(Clone, Debug)]
pub struct PauliPropBackend {
    /// Drop terms whose coefficient magnitude falls below this.
    pub coeff_min: f64,
    /// Drop terms whose Pauli weight exceeds this (`None` = no cap).
    pub max_weight: Option<usize>,
    /// Drop terms whose split frequency (number of non-Clifford sin-branches on
    /// the cheapest path to them) exceeds this (`None` = no cap). This is
    /// PauliPropagation.jl's `max_freq` truncation axis. Over-budget sin-branches
    /// are never even created, so it also bounds the tree fan-out directly.
    pub max_freq: Option<u32>,
    /// Hard ceiling on the number of Pauli terms, refused rather than
    /// allocated. See [`DEFAULT_MAX_TERMS`] for why this defaults to *on* while
    /// every other truncation knob defaults to off.
    pub max_terms: Option<usize>,
    /// Ceiling on `dropped_mass`, above which the expectation is REFUSED.
    ///
    /// `None` — the default — means the observable's own L1 coefficient norm:
    /// refuse only when the error bound has grown so wide it excludes nothing,
    /// which is the point at which the returned number stops being an
    /// approximation and becomes an absence of one. `Some(x)` sets an explicit
    /// ceiling; `Some(f64::INFINITY)` restores the old behaviour of never
    /// refusing.
    ///
    /// Mirrors [`MpsBackend::with_max_discarded_weight`] in shape and in
    /// default-on posture, because it is the same problem. The default differs
    /// deliberately: MPS refuses above `1e-6`, a tight accuracy standard, while
    /// this refuses only at *total* information loss. Adopting MPS's strictness
    /// here would turn every currently-working truncated run into a refusal;
    /// this default turns only the meaningless ones into refusals.
    ///
    /// # The objection, and why it did not win
    ///
    /// A peer argued this should not fire merely because a loose cutoff has a
    /// loose budget — that `dropped_mass = 2.06` at `--truncate 1e-1` is
    /// truncation working as designed, and refusing there punishes correct
    /// behaviour. They are right that it is working as designed. But the
    /// *returned value* is then consistent with every value the observable can
    /// take, and shipping that as an expectation is the defect, not the
    /// truncation that produced it. The escape hatch is what serves the sweep
    /// case: a caller deliberately scanning cutoffs sets the ceiling once and
    /// keeps every row.
    max_dropped_mass: Option<f64>,
    /// Optional GPU accelerator for the branch step (see [`BranchHook`]). `None`
    /// = pure CPU. Skipped from `Debug`/equality — it's a code pointer.
    branch_hook: Option<BranchHook>,
    /// Optional per-gate + readout noise model. `None` = exact, noise-free
    /// Heisenberg evolution. When set, each channel's *adjoint* is applied to
    /// the propagating observable — deterministically and exactly (no
    /// trajectories), which is the natural home for Pauli-Lindblad noise.
    noise: Option<NoiseModel>,
}

impl Default for PauliPropBackend {
    fn default() -> Self {
        Self {
            coeff_min: 0.0,
            max_weight: None,
            max_freq: None,
            max_terms: Some(DEFAULT_MAX_TERMS),
            max_dropped_mass: None,
            branch_hook: None,
            noise: None,
        }
    }
}

impl PauliPropBackend {
    /// Exact engine (no truncation).
    pub fn new() -> Self {
        Self::default()
    }

    /// Engine with coefficient-magnitude and Pauli-weight truncation.
    pub fn with_truncation(coeff_min: f64, max_weight: Option<usize>) -> Self {
        Self {
            coeff_min,
            max_weight,
            max_freq: None,
            max_terms: Some(DEFAULT_MAX_TERMS),
            max_dropped_mass: None,
            branch_hook: None,
            noise: None,
        }
    }

    /// Install a GPU accelerator for the branch step (see [`BranchHook`]).
    /// The truncation actually in force, formatted for the ceiling refusal.
    /// `None` for an exact run. See [`over_cap`] for why the distinction is
    /// load-bearing rather than cosmetic.
    fn truncation_in_force(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if self.coeff_min > 0.0 {
            parts.push(format!("--truncate {:e}", self.coeff_min));
        }
        if let Some(w) = self.max_weight {
            parts.push(format!("--max-weight {w}"));
        }
        if let Some(f) = self.max_freq {
            parts.push(format!("--max-freq {f}"));
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(", "))
        }
    }

    /// Set the ceiling on `dropped_mass` above which a run is refused.
    ///
    /// `None` restores the default (the observable's L1 norm — refuse only at
    /// total information loss); `Some(f64::INFINITY)` disables the gate, which
    /// is what a deliberate cutoff sweep wants. See the field docs for why the
    /// default is where it is.
    pub fn with_max_dropped_mass(mut self, ceiling: Option<f64>) -> Self {
        self.max_dropped_mass = ceiling;
        self
    }

    /// Raise (or remove, with `None`) the hard term ceiling.
    ///
    /// The one truncation knob that does **not** change the answer. `coeff_min`,
    /// `max_weight` and `max_freq` all buy completion by discarding terms, and
    /// pay for it in `dropped_mass`; this one just permits a larger EXACT sum.
    /// So it is the knob to reach for first when a run is refused, and the
    /// refusal message names it for that reason.
    ///
    /// `None` disables the ceiling entirely — the run is then bounded only by
    /// host memory, which is the trade the caller is explicitly making.
    pub fn with_max_terms(mut self, max_terms: Option<usize>) -> Self {
        self.max_terms = max_terms;
        self
    }

    pub fn with_branch_hook(mut self, hook: BranchHook) -> Self {
        self.branch_hook = Some(hook);
        self
    }

    /// Attach a per-gate + readout noise model (applied via its Heisenberg
    /// adjoint during propagation). Composes with truncation and the branch
    /// accelerator.
    pub fn with_noise(mut self, model: NoiseModel) -> Self {
        self.noise = Some(model);
        self
    }

    /// Engine with the full truncation triple, including PauliPropagation.jl's
    /// split-frequency cap (`max_freq`).
    pub fn with_truncation_freq(
        coeff_min: f64,
        max_weight: Option<usize>,
        max_freq: Option<u32>,
    ) -> Self {
        Self {
            coeff_min,
            max_weight,
            max_freq,
            max_terms: Some(DEFAULT_MAX_TERMS),
            max_dropped_mass: None,
            branch_hook: None,
            noise: None,
        }
    }

    /// Builder: set the split-frequency cap.
    pub fn max_freq(mut self, max_freq: Option<u32>) -> Self {
        self.max_freq = max_freq;
        self
    }

    /// L1 dropped-coefficient mass from the *last* `expectation` call is not
    /// retained on the backend; callers wanting the error budget should use
    /// [`PauliPropBackend::expectation_with_budget`].
    pub fn expectation_with_budget(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<(f64, f64)> {
        let (deferred, observable) = prepare_for_expectation(circuit, observable)?;
        let circuit = &deferred;
        let observable = &observable;
        self.expectation_with_certificate(circuit, params, observable)
            .map(|(v, cert)| (v, cert.dropped_mass))
    }

    /// The expectation together with its full [`PauliPropCertificate`], and the
    /// **gated** entry point: a run whose error bound has stopped excluding
    /// anything is refused here rather than returned.
    ///
    /// This is the whole of 3b.2 R1. The bound existed, was correct, was
    /// plumbed and was printed; nothing consulted it. Measured consequence on a
    /// 20-qubit depth-16 ansatz: `--truncate 1e-2` truncated away every term
    /// and returned `0.0000000000` as an ordinary expectation value, carrying a
    /// formally-correct budget of ±144 on a quantity confined to [−1, 1].
    ///
    /// `Backend::expectation` and [`Self::expectation_with_budget`] both route
    /// through here, so the gate cannot be bypassed by choosing a different
    /// door — which is the mistake `MpsBackend::expectation_multi` made with
    /// its Reset check.
    pub fn expectation_with_certificate(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<(f64, PauliPropCertificate)> {
        let (deferred, observable) = prepare_for_expectation(circuit, observable)?;
        let circuit = &deferred;
        let observable = &observable;
        // Reset before propagating: the peak is a thread-local high-water mark,
        // so a previous run's peak would otherwise be reported as this one's —
        // the same defect `MpsBackend` fixed by resetting stats per execute.
        reset_peak_terms();
        let sum = self.propagate(circuit, params, observable)?;
        let mut val = Complex64::new(0.0, 0.0);
        for (k, w) in &sum.terms {
            if k.is_all_iz() {
                val += w.coeff;
            }
        }
        let cert = PauliPropCertificate {
            value: val.re,
            dropped_mass: sum.dropped_mass,
            final_terms: sum.terms.len(),
            peak_terms: peak_terms(),
            coeff_min: self.coeff_min,
            max_weight: self.max_weight,
            max_freq: self.max_freq,
            max_terms: self.max_terms,
            observable_range: observable_l1_norm(observable),
        };
        // Default ceiling is the exact vacuity point for THIS result; an
        // explicit one overrides it wholesale.
        let ceiling = self
            .max_dropped_mass
            .unwrap_or(cert.observable_range + cert.value.abs());
        if cert.dropped_mass >= ceiling {
            return Err(OmegaError::Unsupported(format!(
                "pauliprop: truncation discarded an L1 coefficient mass of \
                 {:.4e}, which exceeds the ceiling of {:.4e}. That bound is \
                 correct and it excludes nothing: |<O>| cannot leave \
                 [-{:.4e}, {:.4e}] to begin with, so every value the observable \
                 can take is consistent with this result. Refusing rather than \
                 returning a number with no information in it. Options: tighten \
                 `--truncate` (a smaller cutoff discards less), raise \
                 `--max-terms N` so the run needs less truncation — that one \
                 does not change the answer — or, if you are deliberately \
                 sweeping cutoffs and want the loose rows, set the ceiling \
                 explicitly via `--max-dropped-mass`. Final terms {}, peak {}.",
                cert.dropped_mass,
                ceiling,
                cert.observable_range,
                cert.observable_range,
                cert.final_terms,
                cert.peak_terms,
            )));
        }
        Ok((val.re, cert))
    }

    /// Backward-propagate `observable` through `circuit`, returning the
    /// resulting Pauli sum (the Heisenberg-evolved observable).
    fn propagate(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<PauliSum> {
        let n = circuit.num_qubits as usize;
        let mut sum = PauliSum::new();
        for (coeff, paulis) in &observable.terms {
            let mut key = PauliKey::identity(n);
            let mut c = Complex64::new(*coeff, 0.0);
            for (q, op) in paulis {
                let q = *q as usize;
                // REJECT, do not index. `PauliKey::set_x`/`set_z` do no bounds
                // check: `set_z` indexes `words[w + q/64]`, so an observable
                // naming a qubit past the register PANICS inside whatever is
                // calling us — for the HTTP server that is a request handler,
                // and there is no CatchPanicLayer. Verified: a 4-qubit circuit
                // with observable `Z99` gave
                // "index out of bounds: the len is 2 but the index is 2".
                //
                // `set_x` is worse than a panic for some indices: on a 4-qubit
                // key it silently writes into the Z half and returns a WRONG
                // ANSWER instead of failing.
                //
                // Nothing upstream validates this. `Observable::parse` takes a
                // bare u32 and never sees the circuit.
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
                    PauliOp::X => key.set_x(q, true),
                    PauliOp::Z => key.set_z(q, true),
                    PauliOp::Y => {
                        key.set_x(q, true);
                        key.set_z(q, true);
                        c *= I; // Y = i·XZ
                    }
                }
            }
            sum.add(key, c);
        }
        // Readout error is the *last* thing to happen in forward time
        // (measurement), so in the backward (adjoint) propagation it is applied
        // *first* — to the raw observable, before any gate.
        if let Some(model) = &self.noise {
            self.apply_readout_adjoint(model, n, &mut sum)?;
        }
        for op in circuit.ops.iter().rev() {
            // A classically-conditioned gate is NOT a plain gate. Its action
            // depends on a measurement outcome, so the circuit is a classical
            // mixture over branches — not one unitary — and observable
            // conjugation cannot express it.
            //
            // This backend used to ignore `op.condition` entirely (zero
            // references to it in the whole crate), which meant a guarded gate
            // was applied **unconditionally and silently** — answering a
            // different circuit, exactly as skipping `Reset` did before the
            // refusal below was added.
            //
            // Measured on `h q0; measure q0 -> c0; if (c==1) x q1` (the shape
            // of `12_feedforward_sometimes_false.qasm`): ignoring the guard
            // gives ⟨Z₁⟩ = −1, "q1 definitely flipped", where the statevector
            // backend in collapse mode gives 0 because the X fires on about
            // half the shots. A gap of 1.0 on an observable bounded in
            // [−1, +1] — the largest error reachable when the truth is 0.
            // Pinned in `tests/conditional_refusal.rs`.
            //
            // Refuse, so the caller falls back to a backend that models it.
            // Found by the N-way matrix work (`FIXES_PLAN.md` K7): statevector,
            // MPS and Pauli all call `condition_satisfied`; this backend was
            // the only one that did not.
            if op.condition.is_some() {
                return Err(OmegaError::Unsupported(
                    "pauliprop: a classically-conditioned gate makes the circuit a mixture \
                     over measurement outcomes, which cannot be represented by observable \
                     conjugation; use the statevector or MPS backend"
                        .into(),
                ));
            }
            // The per-gate channel follows its gate in forward time, so its
            // adjoint precedes the gate's adjoint here.
            if let Some(model) = &self.noise {
                self.apply_gate_noise_adjoint(model, op, &mut sum);
            }
            self.conjugate(op, params, n, &mut sum)?;
            if self.coeff_min > 0.0 || self.max_weight.is_some() || self.max_freq.is_some() {
                sum.truncate(self.coeff_min, self.max_weight, self.max_freq);
            }
        }
        Ok(sum)
    }

    /// Apply the adjoint of the per-gate noise channel that follows `op`.
    ///
    /// Each single-qubit Pauli-Lindblad channel acts diagonally on the Pauli
    /// basis — `N†(P) = λ_P · P` — so the adjoint just rescales each term by a
    /// factor set by its local Pauli on the affected qubit. Amplitude damping is
    /// the one non-unital channel: `Z → (1−γ)Z + γ·I`, which additionally spawns
    /// a lower-weight identity term.
    ///
    /// **Order matters.** The statevector backend applies the forward channels
    /// in the order depolarizing → Pauli → amplitude-damping → phase, so the
    /// Heisenberg adjoint on the observable must apply them in *reverse*:
    /// amplitude damping first, then the diagonal channels. If a diagonal
    /// channel ran first it would rescale the parent `Z` before amplitude
    /// damping spawns its identity term, wrongly scaling that identity by the
    /// diagonal factor. The diagonal channels commute with each other, so only
    /// amplitude damping's leading position is load-bearing.
    fn apply_gate_noise_adjoint(&self, model: &NoiseModel, op: &GateOp, sum: &mut PauliSum) {
        if matches!(op.gate, GateKind::Measure | GateKind::Barrier) {
            return;
        }
        let gate_qubits: Vec<usize> = op.qubits.iter().map(|q| q.0 as usize).collect();
        for &q in &gate_qubits {
            // Amplitude damping γ (non-unital): X,Y → √(1−γ)·; Z → (1−γ)Z + γ·I.
            // Applied FIRST (see the ordering note above) so the spawned identity
            // term is not rescaled by the diagonal channels below.
            let gamma = model.amplitude_damping.at(q);
            if gamma > 0.0 {
                apply_amplitude_damping_adjoint(sum, q, gamma);
            }

            // Depolarizing: identity untouched, any non-I Pauli → (1 − 4p/3).
            // A two-qubit gate's pair selects a per-pair rate when configured.
            let p = model.depolarizing.at_gate(q, &gate_qubits);
            if p > 0.0 {
                let lam = 1.0 - 4.0 * p / 3.0;
                scale_by_local_pauli(sum, q, |x, z| if x || z { lam } else { 1.0 });
            }

            // Phase damping λ: X, Y → (1 − λ); I, Z unchanged.
            let lambda = model.phase_damping.at(q);
            if lambda > 0.0 {
                scale_by_local_pauli(sum, q, |x, _z| if x { 1.0 - lambda } else { 1.0 });
            }

            // Explicit Pauli channel (p_x, p_y, p_z): Pauli-transfer eigenvalues.
            if let Some(pauli) = &model.pauli {
                let (px, py, pz) = (pauli.x.at(q), pauli.y.at(q), pauli.z.at(q));
                if px + py + pz > 0.0 {
                    let pi = 1.0 - px - py - pz;
                    scale_by_local_pauli(sum, q, |x, z| match (x, z) {
                        (false, false) => 1.0,              // I
                        (true, false) => pi + px - py - pz, // X
                        (false, true) => pi - px - py + pz, // Z
                        (true, true) => pi - px + py - pz,  // Y
                    });
                }
            }
        }
    }

    /// Apply the adjoint of readout error to the freshly-seeded observable.
    ///
    /// Readout is modelled as an independent bit-flip on each qubit's Z-basis
    /// outcome, so its adjoint scales the measured `Z`/`Y` components by
    /// `1 − 2p` (leaving `X`), matching `readout_flip` on the statevector
    /// sampler. This backend produces expectation values (no shots), so only
    /// **symmetric** readout has an unambiguous meaning here; asymmetric readout
    /// (`p10 ≠ p01`) is rejected — sample it on `--backend sim` instead.
    fn apply_readout_adjoint(
        &self,
        model: &NoiseModel,
        n: usize,
        sum: &mut PauliSum,
    ) -> Result<()> {
        if model.readout.is_zero() {
            return Ok(());
        }
        for q in 0..n {
            let p10 = model.readout.p10.at(q);
            let p01 = model.readout.p01.at(q);
            if (p10 - p01).abs() > 1e-12 {
                return Err(OmegaError::Unsupported(format!(
                    "pauliprop expectation backend: asymmetric readout (p10={p10}, p01={p01}) \
                     on qubit {q} is not representable as an expectation-value rescale; \
                     sample it on --backend sim instead"
                )));
            }
            let p = p10;
            if p > 0.0 {
                // Bit-flip adjoint: Z,Y → (1−2p)·; X unchanged; I unchanged.
                let lam = 1.0 - 2.0 * p;
                scale_by_local_pauli(sum, q, |_x, z| if z { lam } else { 1.0 });
            }
        }
        Ok(())
    }

    /// Conjugate the whole sum by one gate `Gᵏ`: `O ← Gᵏ† O Gᵏ`.
    fn conjugate(
        &self,
        op: &GateOp,
        params: &ParameterBinding,
        n: usize,
        sum: &mut PauliSum,
    ) -> Result<()> {
        let q = |i: usize| op.qubits[i].0 as usize;
        match op.gate {
            // ----- single-qubit Cliffords (one Pauli → one Pauli) -----
            GateKind::H => self.map_single(sum, q(0), CLIFF_H),
            GateKind::X => self.map_single(sum, q(0), CLIFF_X),
            GateKind::Y => self.map_single(sum, q(0), CLIFF_Y),
            GateKind::Z => self.map_single(sum, q(0), CLIFF_Z),
            GateKind::S => self.map_single(sum, q(0), CLIFF_S),
            GateKind::Sdg => self.map_single(sum, q(0), CLIFF_SDG),
            // sx / sxdg ARE Clifford (sx*sx = X), so they conjugate a single
            // Pauli to a single Pauli exactly like S/Sdg. They used to fall
            // into the generic "not yet supported" arm below — honest, but a
            // missed capability on a backend whose whole domain is Clifford
            // conjugation.
            GateKind::Sx => self.map_single(sum, q(0), CLIFF_SX),
            GateKind::Sxdg => self.map_single(sum, q(0), CLIFF_SXDG),

            // ----- two-qubit Cliffords -----
            GateKind::CX => self.map_two(sum, q(0), q(1), CX_IMG),
            GateKind::CZ => self.map_two(sum, q(0), q(1), CZ_IMG),
            GateKind::Swap => self.map_two(sum, q(0), q(1), SWAP_IMG),
            // CY is Clifford. It used to hit the catch-all and be refused as
            // unsupported, which cost a whole gate for no reason: it maps one
            // Pauli to one Pauli exactly, with no branching and no truncation.
            GateKind::CY => self.map_two(sum, q(0), q(1), CY_IMG),

            // ----- Pauli rotations (one Pauli → two: the branch / tree) -----
            GateKind::Rz => {
                self.branch(
                    sum,
                    &gen_single(n, q(0), 'z'),
                    resolve(&op.params[0], params)?,
                )?;
            }
            GateKind::Rx => {
                self.branch(
                    sum,
                    &gen_single(n, q(0), 'x'),
                    resolve(&op.params[0], params)?,
                )?;
            }
            GateKind::Ry => {
                self.branch(
                    sum,
                    &gen_single(n, q(0), 'y'),
                    resolve(&op.params[0], params)?,
                )?;
            }
            // U1(λ) = diag(1, e^{iλ}) = e^{iλ/2}·Rz(λ); the global phase cancels
            // under conjugation, so it propagates exactly as Rz(λ).
            GateKind::U1 => {
                self.branch(
                    sum,
                    &gen_single(n, q(0), 'z'),
                    resolve(&op.params[0], params)?,
                )?;
            }
            GateKind::T => {
                self.branch(sum, &gen_single(n, q(0), 'z'), std::f64::consts::FRAC_PI_4)?;
            }
            GateKind::Tdg => {
                self.branch(sum, &gen_single(n, q(0), 'z'), -std::f64::consts::FRAC_PI_4)?;
            }
            // CRz(θ) = Rz_t(θ/2) · Rzz_{c,t}(-θ/2) — two commuting Pauli rotations.
            GateKind::CRz => {
                let theta = resolve(&op.params[0], params)?;
                self.branch(sum, &gen_single(n, q(1), 'z'), theta / 2.0)?;
                self.branch(sum, &gen_zz(n, q(0), q(1)), -theta / 2.0)?;
            }

            // U3(θ,φ,λ) = e^{i(φ+λ)/2} · Rz(φ)·Ry(θ)·Rz(λ). The phase is NOT
            // part of the identity, and it is exactly why this expansion lives
            // HERE rather than in the shared lowering: conjugation `O -> U†OU`
            // cancels any global phase, so the decomposition is exact for this
            // backend, while a lowering-level rewrite would hand the same
            // phase-shifted circuit to the statevector backend — where the
            // phase becomes physical the moment the result is controlled or
            // interfered — and would change what `to_qasm` re-emits. Same
            // reasoning the `U1` arm above already relies on.
            GateKind::U3 => {
                let theta = resolve(&op.params[0], params)?;
                let phi = resolve(&op.params[1], params)?;
                let lam = resolve(&op.params[2], params)?;
                // Order matters and is easy to reverse. With
                // `U = Rz(φ)·Ry(θ)·Rz(λ)`,
                //
                //   U† O U = Rz(λ)† Ry(θ)† [ Rz(φ)† O Rz(φ) ] Ry(θ) Rz(λ)
                //
                // so the INNERMOST conjugation is by `Rz(φ)` and the outermost
                // by `Rz(λ)` — φ, then θ, then λ. Writing them in the order the
                // gates appear in the product (λ first) is the natural mistake
                // and gives a wrong expectation, not an error: measured
                // -0.2922 against the statevector backend's -0.6154 before this
                // was corrected.
                self.branch(sum, &gen_single(n, q(0), 'z'), phi)?;
                self.branch(sum, &gen_single(n, q(0), 'y'), theta)?;
                self.branch(sum, &gen_single(n, q(0), 'z'), lam)?;
            }

            // `cp`/`cu1` arrive here widened to CU3(0, 0, λ) by the parser, and
            // in THAT case the gate is a controlled phase:
            //
            //   CP(λ) = e^{iλ/4} · Rz_a(λ/2) · Rz_b(λ/2) · Rzz(-λ/2)
            //
            // which is two `branch` calls, exactly like `CRz` above. A general
            // CU3 is NOT that, so the diagonal case is asserted rather than
            // assumed: applying the CP identity to a genuine `cu3(θ,φ,λ)` would
            // return a wrong number for every circuit that used one, silently.
            GateKind::CU3 => {
                let theta = resolve(&op.params[0], params)?;
                let phi = resolve(&op.params[1], params)?;
                let lam = resolve(&op.params[2], params)?;
                if theta.abs() > 1e-12 || phi.abs() > 1e-12 {
                    return Err(OmegaError::Unsupported(format!(
                        "pauliprop supports CU3 only in its diagonal form \
                         CU3(0, 0, lambda) — the controlled phase that `cp`/`cu1` \
                         lower to — and this one is CU3({theta}, {phi}, {lam}). A \
                         general controlled-U3 is not a controlled phase, and \
                         applying the phase identity to it would return a wrong \
                         value rather than an error. Use the statevector or MPS \
                         backend for this circuit."
                    )));
                }
                // CP(λ): Rz(λ/2) on each qubit, then ZZ(-λ/2).
                self.branch(sum, &gen_single(n, q(0), 'z'), lam / 2.0)?;
                self.branch(sum, &gen_single(n, q(1), 'z'), lam / 2.0)?;
                self.branch(sum, &gen_zz(n, q(0), q(1)), -lam / 2.0)?;
            }

            // CCX and CSwap, via CCZ.
            //
            // The textbook route — 6 CX plus a 7-gate T ladder — is available
            // and is the wrong one here. Those 6 CX are 6 full Clifford passes
            // over the whole sum for no benefit, and the T ladder's generators
            // do not commute with the CX between them, so the Pauli weight
            // scrambles as it goes.
            //
            // `CCZ` is exactly seven DIAGONAL rotations at ±π/4, on
            // Z1, Z2, Z3, Z1Z2, Z1Z3, Z2Z3, Z1Z2Z3 — and every one of those
            // generators is a product of Z, so they all commute. No Clifford
            // maps in between, no scrambling. `CCX = H_t · CCZ · H_t`.
            //
            // It is still seven branchings, and at θ = π/4 `cos θ = sin θ`, so
            // both children of each split carry equal weight and coefficient
            // truncation cannot prune either. That is a property of the gate,
            // not of this implementation: a Toffoli is genuinely expensive for
            // Pauli propagation. The term ceiling (`max_terms`) is what keeps
            // it from becoming a hung machine, and `--truncate`/`--max-weight`
            // remain the way to trade accuracy for room — reported, as always,
            // through `dropped_mass`.
            GateKind::CCX => {
                let (a, b, t) = (q(0), q(1), q(2));
                let eighth = std::f64::consts::FRAC_PI_4;
                let is_ccx = true;
                if is_ccx {
                    self.map_single(sum, t, CLIFF_H);
                }
                // Signs: CCZ = exp(iπ/8 (1 - Z1)(1 - Z2)(1 - Z3)) expanded.
                self.branch(sum, &gen_z_product(n, &[a]), -eighth)?;
                self.branch(sum, &gen_z_product(n, &[b]), -eighth)?;
                self.branch(sum, &gen_z_product(n, &[t]), -eighth)?;
                self.branch(sum, &gen_z_product(n, &[a, b]), eighth)?;
                self.branch(sum, &gen_z_product(n, &[a, t]), eighth)?;
                self.branch(sum, &gen_z_product(n, &[b, t]), eighth)?;
                self.branch(sum, &gen_z_product(n, &[a, b, t]), -eighth)?;
                if is_ccx {
                    self.map_single(sum, t, CLIFF_H);
                }
            }

            // CSWAP(c, a, b) = CX(b, a) · CCX(c, a, b) · CX(b, a), so it costs
            // the same seven branchings plus two Clifford maps.
            GateKind::CSwap => {
                let (c, a, b) = (q(0), q(1), q(2));
                let eighth = std::f64::consts::FRAC_PI_4;
                self.map_two(sum, b, a, CX_IMG);
                self.map_single(sum, b, CLIFF_H);
                self.branch(sum, &gen_z_product(n, &[c]), -eighth)?;
                self.branch(sum, &gen_z_product(n, &[a]), -eighth)?;
                self.branch(sum, &gen_z_product(n, &[b]), -eighth)?;
                self.branch(sum, &gen_z_product(n, &[c, a]), eighth)?;
                self.branch(sum, &gen_z_product(n, &[c, b]), eighth)?;
                self.branch(sum, &gen_z_product(n, &[a, b]), eighth)?;
                self.branch(sum, &gen_z_product(n, &[c, a, b]), -eighth)?;
                self.map_single(sum, b, CLIFF_H);
                self.map_two(sum, b, a, CX_IMG);
            }

            // ----- no-ops for the unitary-conjugation picture -----
            //
            // `Measure` is here only because it can no longer reach this point
            // with any consequence: `prepare_for_expectation` either removed it
            // (it was inert, i.e. nothing reads it) or refused the circuit. It is
            // NOT a no-op in general — treating it as one is what returned
            // ⟨Z₀⟩ = +1 for `h q0; measure q0; h q0`, where the truth is 0.
            GateKind::Id | GateKind::Barrier | GateKind::Measure => {}

            // Reset is NOT a no-op. It is a non-unitary channel
            // (rho -> |0><0|_q (x) Tr_q(rho)) and this backend evolves
            // observables by unitary conjugation, which cannot express it.
            // Silently skipping it answered a DIFFERENT circuit: after
            // Bell + reset(q0) it reported <Z_0> = 0 where the channel gives
            // +1. Refuse so the CLI falls back to a backend that models it.
            GateKind::Reset => {
                return Err(OmegaError::Unsupported(
                    "pauliprop: Reset is a non-unitary channel and cannot be represented by \
                     observable conjugation; use the statevector or MPS backend"
                        .into(),
                ));
            }

            ref other => {
                return Err(OmegaError::Unsupported(format!(
                    "pauliprop: gate {other:?} not yet supported \
                     (Clifford H/X/Y/Z/S/Sdg/CX/CZ/Swap + Rx/Ry/Rz/U1/T/Tdg/CRz)"
                )));
            }
        }
        // One check after the dispatch catches every growth path, present and
        // future: Clifford arms cannot grow the sum, and every arm that can
        // goes through `branch`. Checking here rather than inside `branch`
        // means a gate that branches twice (CRz) is measured once, at its real
        // peak, instead of mid-way through its own expansion.
        if let Some(cap) = self.max_terms {
            if sum.terms.len() > cap {
                // Routed through `over_cap` rather than duplicating its text.
                // These were two hand-maintained copies of the same message and
                // only one of them learned about `--max-terms`; the R4 clause
                // would have gone into one of them too.
                return Err(over_cap(
                    sum.terms.len(),
                    cap,
                    self.truncation_in_force().as_deref(),
                ));
            }
        }
        Ok(())
    }

    /// Apply a single-qubit Clifford's local `(X→, Z→)` images to every term.
    /// Cliffords map one Pauli to one Pauli, so the split frequency is carried
    /// through unchanged.
    fn map_single(&self, sum: &mut PauliSum, q: usize, img: SingleImg) {
        // Nothing has support on `q`, so every key's local operator is the
        // identity and every Clifford maps I -> I with unit factor: the rebuild
        // below would produce a bit-identical copy. Skip it.
        //
        // This is not a micro-optimisation. With a LOCAL observable the light
        // cone bounds the work, not the register — measured, the term dynamics
        // of this corpus are identical at 10 and at 20 qubits, while the insert
        // count rises 58% purely from gates outside the support.
        sum.debug_assert_support_is_superset();
        if !sum.touches_qubit(q) {
            note_skipped();
            return;
        }
        let mut out = PauliSum::with_capacity(sum.terms.len());
        out.dropped_mass = sum.dropped_mass;
        for (key, w) in sum.terms.drain() {
            let (nx, nz, f) = img.apply(key.x(q), key.z(q));
            let mut k = key;
            k.set_x(q, nx);
            k.set_z(q, nz);
            out.add_weighted(k, w.coeff * f, w.freq);
        }
        *sum = out;
    }

    /// Apply a two-qubit Clifford's four generator images to every term.
    fn map_two(&self, sum: &mut PauliSum, c: usize, t: usize, img: TwoImg) {
        // Neither qubit is in the support: both local operators are the
        // identity, so this Clifford is the identity on every term. See
        // `map_single`.
        sum.debug_assert_support_is_superset();
        if !sum.touches_qubit(c) && !sum.touches_qubit(t) {
            note_skipped();
            return;
        }
        let mut out = PauliSum::with_capacity(sum.terms.len());
        out.dropped_mass = sum.dropped_mass;
        for (key, w) in sum.terms.drain() {
            // Compose the four generator images present on (c, t).
            let mut acc = Gen2::ident();
            if key.x(c) {
                acc = acc.mul(&img.xc);
            }
            if key.z(c) {
                acc = acc.mul(&img.zc);
            }
            if key.x(t) {
                acc = acc.mul(&img.xt);
            }
            if key.z(t) {
                acc = acc.mul(&img.zt);
            }
            let mut k = key;
            k.set_x(c, acc.xc);
            k.set_z(c, acc.zc);
            k.set_x(t, acc.xt);
            k.set_z(t, acc.zt);
            out.add_weighted(k, w.coeff * acc.f, w.freq);
        }
        *sum = out;
    }

    /// Conjugate by a Pauli rotation `exp(-iθ/2 · R)` with Hermitian generator
    /// `R` ([`Gen`]). Terms that anticommute with `R` branch into two,
    /// `P → cosθ·P + i sinθ·(R·P)` — the non-Clifford tree step. Works for any
    /// single- or multi-qubit Pauli `R` (Rz/Rx/Ry → 1-qubit; the ZZ factor of
    /// CRz → 2-qubit), so the fan-out is bounded only by truncation.
    fn branch(&self, sum: &mut PauliSum, r: &Gen, theta: f64) -> Result<()> {
        let (cos, sin) = (theta.cos(), theta.sin());
        // The generator's support is disjoint from every term's, so every term
        // COMMUTES with it and the expansion below reproduces the sum exactly:
        // anticommutation is popcount(x & gz) + popcount(z & gx) odd, and both
        // popcounts are zero when the supports do not overlap.
        //
        // Checked BEFORE the accelerator hook, so a disjoint rotation costs no
        // device round-trip either.
        sum.debug_assert_support_is_superset();
        if !sum.overlaps(&r.gx, &r.gz) {
            note_skipped();
            return Ok(());
        }
        // Offer the branch expansion to an accelerator (GPU). If it declines
        // (no device / too few terms), fall through to the CPU loop below — the
        // two are semantically identical, so results never depend on the path.
        if let Some(hook) = self.branch_hook {
            // `Gen` is already packed, so this hands the device its words with
            // no conversion — the `pack_bits` that used to happen here was
            // measured at 37-40% of the whole GPU branch step.
            if hook(sum, &r.gx, &r.gz, r.factor, cos, sin, self.max_freq, r.n) {
                note_peak(sum.terms.len());
                return self.check_cap(sum);
            }
        }
        // Sized at n, NOT at the 2n branch worst case. Measured, because the
        // obvious choice is wrong: 2n is SLOWER than not pre-sizing at all
        // (0.91-0.93x on the large non-truncating runs it was meant to help),
        // while n wins 1.19-1.39x across the board. An over-sized table is a
        // sparser table -- you pay to allocate and first-touch twice the
        // memory, then probe it with worse locality, and it never fills enough
        // to earn that back. The rehash cascade costs less than the locality.
        //
        // n is also a true lower bound on the result: every term contributes at
        // least one entry (the cos child always survives), so this never
        // over-allocates. And it keeps the server's `CostKind::PauliProp`
        // pricing honest -- that prices peak as two live sums, which a 2n map
        // would have quietly made three.
        let mut out = PauliSum::with_capacity(sum.terms.len());
        out.dropped_mass = sum.dropped_mass;
        for (key, w) in sum.terms.drain() {
            // Anticommute ⇔ odd symplectic product ⟨P, R⟩ = Σ (Pₓ·R_z ⊕ P_z·Rₓ).
            // Packed: one popcount per 64 qubits instead of a per-qubit loop.
            if !key.anticommutes_with(&r.gx, &r.gz) {
                out.add_weighted(key, w.coeff, w.freq); // commutes → unchanged
                                                        // Tested on THIS path too, not just the branching one. A
                                                        // commuting add is +1 with no children, so skipping the test
                                                        // here let a commuting add precede an anticommuting pair
                                                        // between two checks and put the transient at cap+3 rather
                                                        // than cap+2. Measured, not reasoned: the test caught it.
                if let Some(cap) = self.max_terms {
                    if out.terms.len() > cap {
                        note_peak(out.terms.len());
                        return Err(over_cap(
                            out.terms.len(),
                            cap,
                            self.truncation_in_force().as_deref(),
                        ));
                    }
                }
                continue;
            }
            // cosθ·P keeps the same frequency; the sin child gains one split.
            out.add_weighted(key.clone(), w.coeff * cos, w.freq);
            let child_freq = w.freq + 1;
            let sin_coeff = w.coeff * I * sin * r.factor;
            if self.max_freq.is_some_and(|m| child_freq > m) {
                // Over the split-frequency budget: don't create the child, but
                // certify the discarded L1 mass so the error stays bounded.
                out.dropped_mass += sin_coeff.norm();
                continue;
            }
            // R · P (R on the left): raw product carries the ± sign.
            let (rk, sign) = mul_raw_packed(&r.gx, &r.gz, &key);
            out.add_weighted(rk, sin_coeff * sign, child_freq);

            // Tested HERE, inside the loop, not after it. After the loop bounds
            // the peak at 2·cap; here it is cap+1 — one term's children past
            // the line. The cost is one integer compare per anticommuting term,
            // against a hash insert that already happened.
            if let Some(cap) = self.max_terms {
                if out.terms.len() > cap {
                    note_peak(out.terms.len());
                    return Err(over_cap(
                        out.terms.len(),
                        cap,
                        self.truncation_in_force().as_deref(),
                    ));
                }
            }
        }
        *sum = out;
        note_peak(sum.terms.len());
        self.check_cap(sum)
    }

    /// Refuse if the sum is past `max_terms`.
    ///
    /// Called after EVERY `branch`, not once per gate. The difference is not
    /// cosmetic: `CCX`/`CSwap` expand a CCZ into seven consecutive branchings,
    /// so a per-gate check let a sum enter at `cap - 1` and take seven
    /// doublings — `2^7 · cap`, about 36 GB at n=40 against a 285 MB
    /// reservation — before anything fired. It could exhaust memory BEFORE it
    /// refused, which is the one failure this ceiling exists to prevent.
    ///
    /// Per-branch, the worst case is one doubling past the cap (`2 · cap`),
    /// since a single `branch` can at most double. Bounding it tighter would
    /// mean testing inside the per-term loop, which is the hottest loop in the
    /// engine; one comparison per branch call is free by comparison.
    fn check_cap(&self, sum: &PauliSum) -> Result<()> {
        if let Some(cap) = self.max_terms {
            if sum.terms.len() > cap {
                return Err(over_cap(
                    sum.terms.len(),
                    cap,
                    self.truncation_in_force().as_deref(),
                ));
            }
        }
        Ok(())
    }
}

/// The refusal, shared by the in-loop test and [`PauliPropBackend::check_cap`].
///
/// `in_force` names the truncation the caller actually requested, or `None` for
/// an exact run. The two cases are **different facts and must not read the
/// same**: a bug report measured `--truncate 1e-6` hitting this ceiling and
/// getting a message that never mentioned the cutoff, so it read as though no
/// cutoff had been given. It had been given, was being applied, and simply did
/// not bound the growth — which is what the caller needs to know, because the
/// remedy is the opposite direction (tighten it, not add one).
fn over_cap(reached: usize, cap: usize, in_force: Option<&str>) -> OmegaError {
    let head = match in_force {
        Some(t) => format!(
            "pauliprop: the Pauli sum reached {reached} terms, past the {cap} \
             ceiling, DESPITE the truncation you requested ({t}). That cutoff \
             is being applied and is not aggressive enough to bound the growth, \
             so this is NOT the same as running without it — tightening it is \
             the remedy, not adding one."
        ),
        None => format!(
            "pauliprop: the Pauli sum reached {reached} terms, past the {cap} \
             ceiling — refusing rather than exhausting memory."
        ),
    };
    OmegaError::Unsupported(format!(
        "{head} Each non-Clifford rotation can double the term count, and a T \
         gate (θ = π/4) is the worst case: cos θ = sin θ, so both branches \
         carry equal weight and coefficient truncation cannot prune either. \
         Options: `--truncate C` to drop small coefficients, `--max-weight W` \
         or `--max-freq F` to bound the tree (all three report `dropped_mass`, \
         a BOUND on the resulting error in <O>, and all three CHANGE THE \
         ANSWER — they are approximations), `--max-terms N` to raise THIS \
         ceiling — the only one of the four that does not change the answer, \
         since it permits a larger exact sum rather than discarding terms — a \
         shallower non-Clifford section, or a backend whose cost is not \
         exponential in T-count."
    ))
}

// --------------------------------------------------------------------------
// Single-qubit Clifford images: G† X G and G† Z G, each as (x, z, factor).
// --------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct SingleImg {
    /// G†XG = factor · raw(ax, az)
    ax: bool,
    az: bool,
    fx: Complex64,
    /// G†ZG = factor · raw(bx, bz)
    bx: bool,
    bz: bool,
    fz: Complex64,
}

impl SingleImg {
    /// Image of the local raw operator `X^{x} Z^{z}` under this Clifford.
    fn apply(&self, x: bool, z: bool) -> (bool, bool, Complex64) {
        match (x, z) {
            (false, false) => (false, false, ONE),
            (true, false) => (self.ax, self.az, self.fx),
            (false, true) => (self.bx, self.bz, self.fz),
            (true, true) => {
                // (G†XG)(G†ZG): raw-multiply the two images. One qubit, so
                // this is the packed rule written out rather than a call into
                // the word-wise path — sign = (-1)^{z_a . x_b}.
                let sign = if self.az && self.bx { -1.0 } else { 1.0 };
                (
                    self.ax ^ self.bx,
                    self.az ^ self.bz,
                    self.fx * self.fz * sign,
                )
            }
        }
    }
}

const CLIFF_H: SingleImg = SingleImg {
    ax: false,
    az: true,
    fx: ONE,
    bx: true,
    bz: false,
    fz: ONE,
};
const CLIFF_X: SingleImg = SingleImg {
    ax: true,
    az: false,
    fx: ONE,
    bx: false,
    bz: true,
    fz: Complex64::new(-1.0, 0.0),
};
const CLIFF_Y: SingleImg = SingleImg {
    ax: true,
    az: false,
    fx: Complex64::new(-1.0, 0.0),
    bx: false,
    bz: true,
    fz: Complex64::new(-1.0, 0.0),
};
const CLIFF_Z: SingleImg = SingleImg {
    ax: true,
    az: false,
    fx: Complex64::new(-1.0, 0.0),
    bx: false,
    bz: true,
    fz: ONE,
};
// S† X S = -Y = (-i)·raw(1,1);  S† Z S = Z.
const CLIFF_S: SingleImg = SingleImg {
    ax: true,
    az: true,
    fx: Complex64::new(0.0, -1.0),
    bx: false,
    bz: true,
    fz: ONE,
};
// S X S† = +Y = (+i)·raw(1,1);  Z → Z.
const CLIFF_SDG: SingleImg = SingleImg {
    ax: true,
    az: true,
    fx: Complex64::new(0.0, 1.0),
    bx: false,
    bz: true,
    fz: ONE,
};

// √X and √X†.
//
// NOTE THE DIRECTION. This table stores `G† P G`, which is the OPPOSITE of the
// `G P G†` form proved in `proofs/lean4/QuantumProofs/SqrtX.lean` — an easy
// place to install a sign error that no self-consistent check would see. The
// entries below are the theorems `sqrtX_conj_X` / `sqrtXdg_conj_Z` read in the
// right direction, and were re-derived numerically before being written:
//
//   sx† X sx  = +1 · raw(1,0)        (X is the fixed axis)
//   sx† Z sx  = +i · raw(1,1) = +Y
//   sxdg† X sxdg = +1 · raw(1,0)
//   sxdg† Z sxdg = −i · raw(1,1) = −Y
//
// The two differ ONLY in the sign of `fz`, which is exactly the bit that
// distinguishes the gate from its inverse — so copying one to the other is
// both the easiest mistake and an invisible one.
const CLIFF_SX: SingleImg = SingleImg {
    ax: true,
    az: false,
    fx: ONE,
    bx: true,
    bz: true,
    fz: Complex64::new(0.0, 1.0),
};
const CLIFF_SXDG: SingleImg = SingleImg {
    ax: true,
    az: false,
    fx: ONE,
    bx: true,
    bz: true,
    fz: Complex64::new(0.0, -1.0),
};

// --------------------------------------------------------------------------
// Two-qubit Clifford images: G†·{Xc,Zc,Xt,Zt}·G as 2-qubit raw operators.
// --------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Gen2 {
    xc: bool,
    zc: bool,
    xt: bool,
    zt: bool,
    f: Complex64,
}

impl Gen2 {
    const fn new(xc: bool, zc: bool, xt: bool, zt: bool) -> Self {
        Self {
            xc,
            zc,
            xt,
            zt,
            f: ONE,
        }
    }
    fn ident() -> Self {
        Self {
            xc: false,
            zc: false,
            xt: false,
            zt: false,
            f: ONE,
        }
    }
    /// Raw-multiply two 2-qubit operators (sign per qubit: Z_left past X_right).
    fn mul(&self, o: &Gen2) -> Gen2 {
        let mut neg = 0u32;
        if self.zc && o.xc {
            neg += 1;
        }
        if self.zt && o.xt {
            neg += 1;
        }
        let sign = if neg.is_multiple_of(2) {
            ONE
        } else {
            Complex64::new(-1.0, 0.0)
        };
        Gen2 {
            xc: self.xc ^ o.xc,
            zc: self.zc ^ o.zc,
            xt: self.xt ^ o.xt,
            zt: self.zt ^ o.zt,
            f: self.f * o.f * sign,
        }
    }
}

#[derive(Clone, Copy)]
struct TwoImg {
    xc: Gen2,
    zc: Gen2,
    xt: Gen2,
    zt: Gen2,
}

// CX: Xc→XcXt, Zc→Zc, Xt→Xt, Zt→ZcZt.
const CX_IMG: TwoImg = TwoImg {
    xc: Gen2::new(true, false, true, false),
    zc: Gen2::new(false, true, false, false),
    xt: Gen2::new(false, false, true, false),
    zt: Gen2::new(false, true, false, true),
};
// CZ: Xc→XcZt, Zc→Zc, Xt→ZcXt, Zt→Zt.
const CZ_IMG: TwoImg = TwoImg {
    xc: Gen2::new(true, false, false, true),
    zc: Gen2::new(false, true, false, false),
    xt: Gen2::new(false, true, true, false),
    zt: Gen2::new(false, false, false, true),
};
// CY = |0><0| (x) I + |1><1| (x) Y, Hermitian and self-inverse, so it is a
// Clifford and maps one Pauli to one Pauli — no branching, unlike a
// decomposition into rotations. Derived, then verified numerically against the
// statevector backend (see `tests/cy_is_clifford.rs`), because the Xc image
// carries a phase that hand-derivation gets wrong easily:
//
//   Xc -> Xc (x) Y     (controlled-U conjugation: Xc -> Xc (x) U)
//   Zc -> Zc           (commutes with the control)
//   Xt -> Zc Xt        (Y X Y = -X, and the sign is exactly Zc)
//   Zt -> Zc Zt        (Y Z Y = -Z)
//
// The phase: this encoding stores Y as the bit pair (x=1, z=1), which is
// literally `X·Z = -i·Y`. So the bits (xc, xt, zt) spell `Xc (x) (-i Y_t)`, and
// the image needs a factor of `i` to be `Xc (x) Y_t`. Consistency check that
// the four images are one map and not four guesses: Yt = i·Xt·Zt must be
// fixed, and (Zc Xt)(Zc Zt)·i = i·Xt·Zt = Yt.
const CY_IMG: TwoImg = TwoImg {
    xc: Gen2 {
        xc: true,
        zc: false,
        xt: true,
        zt: true,
        f: I,
    },
    zc: Gen2::new(false, true, false, false),
    xt: Gen2::new(false, true, true, false),
    zt: Gen2::new(false, true, false, true),
};
// Swap: Xc↔Xt, Zc↔Zt.
const SWAP_IMG: TwoImg = TwoImg {
    xc: Gen2::new(false, false, true, false),
    zc: Gen2::new(false, false, false, true),
    xt: Gen2::new(true, false, false, false),
    zt: Gen2::new(false, true, false, false),
};

// --------------------------------------------------------------------------
// Pauli rotation generators (Hermitian, over the full register).
// --------------------------------------------------------------------------

/// A Hermitian Pauli `R = factor · raw(gx, gz)` generating a rotation
/// `exp(-iθ/2 R)`. Stored full-register so single- and multi-qubit generators
/// (e.g. the `ZZ` factor of `CRz`) share one code path.
/// A Pauli rotation generator, stored **packed** for the same reason
/// [`PauliKey`] is: the anticommute test in `branch` runs once per term per
/// gate, and packed it is one popcount per 64 qubits instead of a loop.
struct Gen {
    gx: Vec<u64>,
    gz: Vec<u64>,
    n: usize,
    factor: Complex64,
}

/// Set bit `q` in a packed word vector.
#[inline]
fn set_bit(words: &mut [u64], q: usize) {
    words[q / 64] |= 1u64 << (q % 64);
}

/// Zeroed packed words for `n` qubits.
#[inline]
fn zero_words(n: usize) -> Vec<u64> {
    vec![0u64; n.div_ceil(64)]
}

/// Single-qubit rotation generator on qubit `q`: `'z'`→Z, `'x'`→X, `'y'`→Y.
fn gen_single(n: usize, q: usize, axis: char) -> Gen {
    let mut gx = zero_words(n);
    let mut gz = zero_words(n);
    let factor = match axis {
        'z' => {
            set_bit(&mut gz, q);
            ONE
        }
        'x' => {
            set_bit(&mut gx, q);
            ONE
        }
        'y' => {
            set_bit(&mut gx, q);
            set_bit(&mut gz, q);
            I // Y = i·XZ
        }
        _ => unreachable!("bad axis"),
    };
    Gen { gx, gz, n, factor }
}

/// Two-qubit `Z⊗Z` generator on `(c, t)` (the entangling factor of `CRz`).
fn gen_zz(n: usize, c: usize, t: usize) -> Gen {
    let gx = zero_words(n);
    let mut gz = zero_words(n);
    set_bit(&mut gz, c);
    set_bit(&mut gz, t);
    Gen {
        gx,
        gz,
        n,
        factor: ONE,
    }
}

/// A product of `Z` on an arbitrary set of qubits, as a rotation generator.
///
/// Generalises [`gen_zz`] to any arity, which is what makes `CCZ` cheap: the
/// gate decomposes into seven rotations whose generators are all products of
/// `Z`, and products of `Z` all commute with one another.
fn gen_z_product(n: usize, qubits: &[usize]) -> Gen {
    let gx = zero_words(n);
    let mut gz = zero_words(n);
    for &q in qubits {
        set_bit(&mut gz, q);
    }
    Gen {
        gx,
        gz,
        n,
        factor: ONE,
    }
}

/// Pack a per-qubit boolean vector into little-endian u64 words: qubit `i` is
/// bit `i % 64` of word `i / 64`. The GPU branch kernel consumes this SoA layout.
pub fn pack_bits(bits: &[bool]) -> Vec<u64> {
    let words = bits.len().div_ceil(64);
    let mut out = vec![0u64; words];
    for (i, &b) in bits.iter().enumerate() {
        if b {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

/// Inverse of [`pack_bits`] for `n` qubits.
pub fn unpack_bits(words: &[u64], n: usize) -> Vec<bool> {
    (0..n)
        .map(|i| (words[i / 64] >> (i % 64)) & 1 == 1)
        .collect()
}

/// Resolve a (concrete or symbolic) parameter to a number.
fn resolve(expr: &ParamExpr, params: &ParameterBinding) -> Result<f64> {
    params.resolve(expr)
}

/// Rescale every term's coefficient by a factor determined by its local Pauli
/// `(x, z)` on qubit `q`. Used for the diagonal (Pauli-Lindblad) channel
/// adjoints, which map each Pauli to a scalar multiple of itself — so the keys
/// don't change and we can mutate coefficients in place.
fn scale_by_local_pauli<F: Fn(bool, bool) -> f64>(sum: &mut PauliSum, q: usize, factor: F) {
    for (key, w) in sum.terms.iter_mut() {
        let f = factor(key.x(q), key.z(q));
        if f != 1.0 {
            w.coeff *= f;
        }
    }
}

/// Adjoint of amplitude damping on qubit `q`: `X,Y → √(1−γ)·`, `I → I`, and the
/// non-unital `Z → Z + γ·I` (which spawns a lower-weight identity companion).
/// Rebuilds the term map so spawned/merged keys accumulate correctly.
fn apply_amplitude_damping_adjoint(sum: &mut PauliSum, q: usize, gamma: f64) {
    let s = (1.0 - gamma).sqrt();
    let old = std::mem::take(&mut sum.terms);
    for (key, w) in old {
        match (key.x(q), key.z(q)) {
            (false, false) => sum.add_weighted(key, w.coeff, w.freq), // I
            (true, false) | (true, true) => sum.add_weighted(key, w.coeff * s, w.freq), // X, Y
            (false, true) => {
                // Z → (1−γ)·Z + γ·I: the Z term is attenuated and a companion
                // identity-on-q term is added with weight γ. (Λ†(Z) =
                // diag(1, 2γ−1) = γ·I + (1−γ)·Z.)
                let mut ident = key.clone();
                ident.set_z(q, false);
                sum.add_weighted(key, w.coeff * (1.0 - gamma), w.freq);
                sum.add_weighted(ident, w.coeff * gamma, w.freq);
            }
        }
    }
}

impl Backend for PauliPropBackend {
    fn name(&self) -> &str {
        "pauliprop"
    }

    fn execute(
        &self,
        _circuit: &CircuitIR,
        _params: &ParameterBinding,
        _config: &ExecConfig,
    ) -> Result<ExecResult> {
        Err(OmegaError::Unsupported(
            "pauliprop is an expectation-value backend; use `expectation()`, not execute/sampling"
                .into(),
        ))
    }

    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<f64> {
        self.expectation_with_budget(circuit, params, observable)
            .map(|(v, _)| v)
    }
}
