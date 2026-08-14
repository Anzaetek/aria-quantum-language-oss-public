use std::collections::HashMap;

use num_complex::Complex64;

use crate::circuit::{CircuitIR, SymbolId};
use crate::device::DeviceKind;
use crate::error::Result;
use crate::params::ParameterBinding;

/// How mid-circuit measurements are handled during simulation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum MidCircuitMode {
    /// Skip measurement gates (current/legacy behavior, backward compatible).
    #[default]
    Skip,
    /// Collapse the statevector on each measurement (projective, single-shot).
    Collapse,
}

/// Configuration for circuit execution.
#[derive(Clone, Debug)]
pub struct ExecConfig {
    /// Number of shots. None = exact statevector / probabilities.
    pub shots: Option<u32>,
    /// Random seed for measurement sampling.
    pub seed: Option<u64>,
    /// How mid-circuit measurements are handled.
    pub mid_circuit_mode: MidCircuitMode,
}

impl Default for ExecConfig {
    fn default() -> Self {
        Self {
            shots: Some(1024),
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        }
    }
}

/// Result of circuit execution.
#[derive(Clone, Debug)]
pub enum ExecResult {
    /// Full statevector (2^n complex amplitudes).
    Statevector(Vec<Complex64>),
    /// Measurement counts: bitstring -> count.
    Counts(HashMap<u64, u32>),
    /// Probability distribution over computational basis states.
    Probabilities(Vec<f64>),
}

impl ExecResult {
    /// Get counts, panicking if this isn't a Counts result.
    pub fn counts(&self) -> &HashMap<u64, u32> {
        match self {
            ExecResult::Counts(c) => c,
            _ => panic!("expected Counts result"),
        }
    }

    /// Get statevector, panicking if this isn't a Statevector result.
    pub fn statevector(&self) -> &[Complex64] {
        match self {
            ExecResult::Statevector(sv) => sv,
            _ => panic!("expected Statevector result"),
        }
    }

    /// Format counts as a sorted string for display.
    pub fn format_counts(&self, num_qubits: u32) -> String {
        match self {
            ExecResult::Counts(counts) => {
                let mut sorted: Vec<_> = counts.iter().collect();
                sorted.sort_by(|a, b| b.1.cmp(a.1));
                sorted
                    .iter()
                    .map(|(bits, count)| {
                        format!(
                            "|{:0>width$b}>: {}",
                            bits,
                            count,
                            width = num_qubits as usize
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            ExecResult::Statevector(sv) => {
                let nonzero: Vec<_> = sv
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| a.norm() > 1e-10)
                    .collect();
                nonzero
                    .iter()
                    .map(|(i, a)| {
                        format!("|{:0>width$b}>: {:.6}", i, a, width = num_qubits as usize)
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            ExecResult::Probabilities(probs) => {
                let nonzero: Vec<_> = probs
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| **p > 1e-10)
                    .collect();
                nonzero
                    .iter()
                    .map(|(i, p)| {
                        format!("|{:0>width$b}>: {:.6}", i, p, width = num_qubits as usize)
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
    }
}

/// Whether a circuit must be simulated with [`MidCircuitMode::Collapse`]
/// rather than the cheaper end-of-circuit sampling path.
///
/// True when the circuit has any classically conditioned gate, any classical
/// bit written by more than one `measure` (Qiskit is last-write-wins on the
/// creg; basis-state sampling would double-count), or any operation after a
/// `measure`. Otherwise the measurements are terminal, the qubit→cbit mapping
/// is a pure relabelling, and `Skip` mode plus
/// [`project_counts_onto_creg`] gives the same distribution more cheaply and
/// without per-shot trajectories.
///
/// Extracted from `omega-cli`'s execute path so the N-way counts matrix
/// (`crates/omega-cli/tests/nway_counts.rs`) drives the **same** decision the
/// CLI ships. A matrix that picked its own mode would be validating an
/// execution path no user gets — and `Skip` vs `Collapse` is exactly where the
/// counts-keying convention changes, so the difference is not academic.
pub fn needs_collapse(circuit: &CircuitIR) -> bool {
    use crate::circuit::GateKind;
    if circuit.ops.iter().any(|op| op.condition.is_some()) {
        return true;
    }
    let mut seen = std::collections::HashSet::new();
    for op in &circuit.ops {
        if matches!(op.gate, GateKind::Measure) {
            if let Some(c) = op.classical_bit {
                if !seen.insert(c) {
                    return true; // cbit written twice
                }
            }
        }
    }
    // Any non-measure operation after any measure. (The CLI additionally
    // tested "ops after the LAST measure", which this subsumes; the two were
    // kept as separate disjuncts there and are merged here without changing
    // the predicate's value on any input.)
    circuit.ops.iter().enumerate().any(|(i, op)| {
        matches!(op.gate, GateKind::Measure)
            && circuit.ops[i + 1..]
                .iter()
                .any(|next| !matches!(next.gate, GateKind::Measure))
    })
}

/// Pack a classical register (bit `i` = cbit `i`, LSB-first) into a `u64`
/// counts key.
///
/// This is the encoding every `Collapse`-mode backend must use: the mid-circuit
/// `measure` statements already ran and recorded their outcomes, so the creg —
/// not a fresh sample of the qubit register — is the shot's result. Qiskit
/// reports over the creg too, which is what makes the two comparable.
///
/// Lives here because it had been copy-pasted into three backends, and a
/// fourth copy is a fourth chance to key counts differently from the rest.
pub fn creg_to_u64(classical_bits: &[u8]) -> u64 {
    // The condition is on the bits that are SET, not on the register's declared
    // size. `creg c[70]; measure q[0] -> c[0];` is a perfectly representable
    // 1-bit outcome that happens to live in a 70-entry slice — asserting on
    // `classical_bits.len()` panicked on it in debug builds, while
    // `counts_outcome_width` (which gates the same runs) correctly reported 1.
    // The two disagreed about what "too wide" means.
    //
    // What actually loses information is a set bit at or above index 64.
    debug_assert!(
        classical_bits.iter().skip(MAX_COUNTS_QUBITS).all(|b| *b & 1 == 0),
        "creg_to_u64: a classical bit at or above index {MAX_COUNTS_QUBITS} is \
         set, so the {MAX_COUNTS_QUBITS}-bit counts key cannot represent this \
         outcome. Call `check_counts_width(counts_outcome_width(..))` first."
    );
    let mut bits = 0u64;
    for (i, b) in classical_bits.iter().enumerate() {
        if i >= MAX_COUNTS_QUBITS {
            break;
        }
        bits |= ((*b as u64) & 1) << i;
    }
    bits
}

/// How many qubits an [`ExecResult::Counts`] key can represent.
///
/// The key is a `u64`, so 64 — one bit per qubit.
pub const MAX_COUNTS_QUBITS: usize = 64;

/// Refuse to produce counts that the key cannot represent.
///
/// # The defect this exists for
///
/// `ExecResult::Counts` is keyed by `u64` and nothing checked the qubit count
/// against it, so every bit above 63 was silently dropped and a **confident
/// wrong answer** was returned. Measured on a GHZ chain, one lost bit per qubit
/// above 64:
///
/// ```text
///   63 qubits:  |0…0>, |1…1>        correct
///   64 qubits:  |0…0>, |1…1>        correct
///   65 qubits:  |0…0>, |0 1^64>     one leading zero
///   66 qubits:  |0…0>, |00 1^64>    two leading zeros
///  128 qubits:  |0…0>, |0^64 1^64>  the reported symptom
/// ```
///
/// It was reported as an MPS defect. It is not: the same GHZ at 65 qubits
/// truncates identically on the stabilizer backend. Any backend returning counts
/// above 64 qubits is affected, and the physics in all of them is correct — only
/// the reporting was lossy.
///
/// # Why refusing, rather than widening the key, comes first
///
/// Widening to a bitset touches every backend and the wire format, and needs a
/// decision that is not obviously "wider is better": counts over a 2^128 space
/// are a sparse sample, and a caller at that scale usually wants marginals or an
/// expectation value. That decision can be taken carefully. Returning a wrong
/// answer in the meantime cannot wait for it.
///
/// Expectation values, statevectors and probability vectors are unaffected —
/// this is a property of the counts KEY, not of any simulation.
/// Is this run's counts key packed from the CLASSICAL register?
///
/// Three places need this answer and they must agree: the width guard, the
/// sampling path that builds the key, and the front end that renders it. A
/// disagreement shows up as a correct key printed at the wrong width, or a
/// refusal for a key that was never going to be built.
///
/// Two ways it becomes true:
/// * **collapse** — the mid-circuit measures already ran, so the creg IS the
///   result;
/// * **above the 64-qubit cliff with measurements** — the full-register key
///   cannot be represented at all, so a measured circuit is keyed on its creg
///   instead. Below the cliff the full register is kept, so nothing a caller
///   already relies on moves.
pub fn counts_keyed_on_creg(circuit: &CircuitIR, collapse: bool) -> bool {
    collapse
        || (circuit.num_qubits as usize > MAX_COUNTS_QUBITS && !measure_pairs(circuit).is_empty())
}

/// The number of bits a shot outcome of `circuit` actually occupies.
///
/// **Not the qubit count.** Which register keys the outcome decides the width:
///
/// * keyed on the creg — the width is the highest classical bit used;
/// * keyed on the qubit register — the width is `num_qubits`.
///
/// `by_creg` must be [`counts_keyed_on_creg`]'s answer, **not** the raw
/// collapse flag. The two differ above the 64-qubit cliff, where a measured
/// circuit is keyed on its creg even in skip mode — passing `collapse` there
/// returns `num_qubits`, which then refuses a run that was going to produce a
/// perfectly representable 2-bit key. The parameter was called `collapse` and
/// at least one caller obliged.
///
/// Gating on `num_qubits` unconditionally over-refuses badly. Measured: a
/// 1024-qubit circuit measuring two qubits into `creg c[2]` was refused, though
/// its outcome needs two bits — and that is the natural shape of a large run.
/// A 70-qubit circuit with a 2-bit creg was refused for the same reason.
pub fn counts_outcome_width(circuit: &CircuitIR, by_creg: bool) -> usize {
    if by_creg {
        measure_pairs(circuit)
            .iter()
            .map(|&(_, c)| c as usize + 1)
            .max()
            .unwrap_or(0)
    } else {
        circuit.num_qubits as usize
    }
}

pub fn check_counts_width(n_qubits: usize) -> crate::error::Result<()> {
    if n_qubits > MAX_COUNTS_QUBITS {
        return Err(crate::error::OmegaError::Unsupported(format!(
            "counts are keyed by a {MAX_COUNTS_QUBITS}-bit integer, so a shot \
             outcome over {n_qubits} qubits cannot be represented — every bit \
             above {MAX_COUNTS_QUBITS} would be silently dropped and the counts \
             would be wrong rather than merely truncated. \
             Use `--expectation` (unaffected), reduce the register, or measure \
             at most {MAX_COUNTS_QUBITS} qubits into a classical register."
        )));
    }
    Ok(())
}

/// The `(qubit, classical_bit)` pairs a circuit's `measure` statements
/// declare, in program order.
///
/// Lives here rather than in a front end because **every** consumer of
/// [`ExecResult::Counts`] needs it to interpret the keys, and a second copy is
/// a second convention. See [`project_counts_onto_creg`].
pub fn measure_pairs(circuit: &CircuitIR) -> Vec<(u32, u32)> {
    circuit
        .ops
        .iter()
        .filter(|op| op.gate == crate::circuit::GateKind::Measure)
        .filter_map(|op| {
            let q = op.qubits.first()?.0;
            op.classical_bit.map(|c| (q, c))
        })
        .collect()
}

/// Project full-register sampled counts onto the classical register via the
/// program's `measure → creg` statements (OpenQASM semantics).
///
/// Backends sample the **full qubit register** at the end of the circuit, so a
/// raw counts key is a basis index with bit `q` = qubit `q`. When the program
/// declares an explicit mapping, the reported counts must instead be keyed
/// over creg bits, one bit per `measure`, in `c[j]` order. A later measure
/// into the same classical bit overwrites the earlier one.
///
/// **This is not cosmetic.** A qubit that is never measured must not appear in
/// the key at all. On `08_partial_measure.qasm` — 3 qubits, a 2-bit creg, and
/// an unmeasured `h q[2]` — skipping the projection turns Qiskit's two
/// outcomes `{00, 11}` into four, because the unmeasured qubit's coin flip
/// leaks into the key. A differential test comparing raw keys against a bridge
/// would report an L2 near 1.0 and blame the backend.
///
/// Counts keys are `u64`, so a measure targeting bit ≥ 64 of either register
/// cannot be represented — that's a loud error, not a masked shift.
pub fn project_counts_onto_creg(
    res: ExecResult,
    pairs: &[(u32, u32)],
) -> std::result::Result<ExecResult, String> {
    if pairs.is_empty() {
        return Ok(res);
    }
    if let Some(&(q, c)) = pairs.iter().find(|&&(q, c)| q >= 64 || c >= 64) {
        return Err(format!(
            "measure q[{q}] -> c[{c}]: sampled-count keys are u64, so register \
             indices ≥ 64 cannot be reported; reduce the register or drop --shots"
        ));
    }
    match res {
        ExecResult::Counts(counts) => {
            let mut projected: HashMap<u64, u32> = HashMap::new();
            for (outcome, n) in counts {
                let mut key = 0u64;
                for &(q, c) in pairs {
                    let bit = (outcome >> q) & 1;
                    key = (key & !(1u64 << c)) | (bit << c);
                }
                *projected.entry(key).or_insert(0) += n;
            }
            Ok(ExecResult::Counts(projected))
        }
        other => Ok(other),
    }
}

/// A Pauli operator for defining observables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PauliOp {
    I,
    X,
    Y,
    Z,
}

/// An observable expressed as a sum of Pauli strings with real coefficients.
#[derive(Clone, Debug)]
pub struct Observable {
    pub terms: Vec<(f64, Vec<(u32, PauliOp)>)>,
}

impl Observable {
    /// Identity observable (scalar 1).
    pub fn identity() -> Self {
        Self {
            terms: vec![(1.0, vec![])],
        }
    }

    /// Single Pauli operator on one qubit.
    pub fn pauli(qubit: u32, op: PauliOp) -> Self {
        Self {
            terms: vec![(1.0, vec![(qubit, op)])],
        }
    }

    /// Z operator on a single qubit.
    pub fn z(qubit: u32) -> Self {
        Self::pauli(qubit, PauliOp::Z)
    }

    /// X operator on a single qubit.
    pub fn x(qubit: u32) -> Self {
        Self::pauli(qubit, PauliOp::X)
    }

    /// Y operator on a single qubit.
    pub fn y(qubit: u32) -> Self {
        Self::pauli(qubit, PauliOp::Y)
    }

    /// Z⊗Z on two qubits.
    pub fn zz(q0: u32, q1: u32) -> Self {
        Self {
            terms: vec![(1.0, vec![(q0, PauliOp::Z), (q1, PauliOp::Z)])],
        }
    }

    /// X⊗X on two qubits.
    pub fn xx(q0: u32, q1: u32) -> Self {
        Self {
            terms: vec![(1.0, vec![(q0, PauliOp::X), (q1, PauliOp::X)])],
        }
    }

    /// Y⊗Y on two qubits.
    pub fn yy(q0: u32, q1: u32) -> Self {
        Self {
            terms: vec![(1.0, vec![(q0, PauliOp::Y), (q1, PauliOp::Y)])],
        }
    }

    /// Parse a Pauli-sum string like `"0.3979*Z0+-0.3979*Z1+-0.0112*Z0Z1+0.1809*X0X1"`.
    ///
    /// - Terms are joined with `+`. Negative coefficients are written `+-c*…`
    ///   (the `+-` is not pretty but matches the on-wire form used by
    ///   `omega-run --expectation` and the WASM driver).
    /// - Each term is `coeff*pauli` or just `pauli` (coefficient 1.0).
    /// - `pauli` is a concatenation of `(X|Y|Z|I)<qubit>` fragments;
    ///   `I` fragments are dropped (they are the identity).
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        let s = s.replace(' ', "");
        let mut terms = Vec::new();
        for raw_part in s.split('+') {
            if raw_part.is_empty() {
                continue;
            }
            let (coeff, pauli_part) = if let Some(idx) = raw_part.find('*') {
                let c: f64 = raw_part[..idx]
                    .parse()
                    .map_err(|e| format!("coefficient '{}': {e}", &raw_part[..idx]))?;
                (c, &raw_part[idx + 1..])
            } else if raw_part.starts_with(['X', 'Y', 'Z', 'I']) {
                (1.0, raw_part)
            } else {
                let idx = raw_part
                    .find(['X', 'Y', 'Z', 'I'])
                    .ok_or_else(|| format!("no Pauli letter in term '{raw_part}'"))?;
                let c: f64 = raw_part[..idx]
                    .parse()
                    .map_err(|e| format!("coefficient '{}': {e}", &raw_part[..idx]))?;
                (c, &raw_part[idx..])
            };
            terms.push((coeff, parse_pauli_string(pauli_part)?));
        }
        if terms.is_empty() {
            return Err("empty observable".into());
        }
        Ok(Observable { terms })
    }
}

/// Parse the Pauli-string portion of a term, e.g. `"Z0Z1"` → `[(0,Z),(1,Z)]`.
/// `I` fragments are skipped (they are the identity factor).
pub fn parse_pauli_string(s: &str) -> std::result::Result<Vec<(u32, PauliOp)>, String> {
    let mut out = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let op = match chars[i] {
            'X' | 'x' => PauliOp::X,
            'Y' | 'y' => PauliOp::Y,
            'Z' | 'z' => PauliOp::Z,
            'I' | 'i' => PauliOp::I,
            other => return Err(format!("bad Pauli '{other}' in '{s}'")),
        };
        i += 1;
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if start == i {
            return Err(format!("missing qubit index after Pauli in '{s}'"));
        }
        let q: u32 = s[start..i].parse().map_err(|e| format!("qubit idx: {e}"))?;
        if !matches!(op, PauliOp::I) {
            out.push((q, op));
        }
    }
    Ok(out)
}

impl std::ops::Add for Observable {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self.terms.extend(rhs.terms);
        self
    }
}

impl std::ops::Mul<f64> for Observable {
    type Output = Self;
    fn mul(mut self, scalar: f64) -> Self {
        for term in &mut self.terms {
            term.0 *= scalar;
        }
        self
    }
}

impl std::ops::Mul<Observable> for f64 {
    type Output = Observable;
    fn mul(self, mut obs: Observable) -> Observable {
        for term in &mut obs.terms {
            term.0 *= self;
        }
        obs
    }
}

impl std::ops::Neg for Observable {
    type Output = Self;
    fn neg(mut self) -> Self {
        for term in &mut self.terms {
            term.0 = -term.0;
        }
        self
    }
}

impl std::ops::Sub for Observable {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl std::ops::AddAssign for Observable {
    fn add_assign(&mut self, rhs: Self) {
        self.terms.extend(rhs.terms);
    }
}

impl std::ops::MulAssign<f64> for Observable {
    fn mul_assign(&mut self, scalar: f64) {
        for term in &mut self.terms {
            term.0 *= scalar;
        }
    }
}

/// Trait for quantum simulation backends.
pub trait Backend {
    fn name(&self) -> &str;

    /// Compute device this backend dispatches to. Defaults to `Cpu`;
    /// GPU-backed backends override to report `Metal` / `Cuda` /
    /// `OpenCl`. Read by `QmlTrainer::device()` to validate that the
    /// trainer's explicit device pick matches the backend the caller
    /// actually passes to `fit()`.
    fn device(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    /// CPU-equivalent backend the caller can swap to when this backend
    /// fails an allocation at runtime (`MetalError::AllocationRefused`
    /// / `CudaError::AllocationRefused` etc.). Defaults to `None` —
    /// CPU backends and stub backends have no fallback. GPU backends
    /// override to return `Some(Box::new(StatevectorBackend::new()))`.
    /// Used by `QmlTrainer::fit` to silently rescue the training loop
    /// when a GPU buffer-pool lease refuses (n ≥ 22 on memory-tight
    /// devices). The trainer emits one stderr notice and continues on
    /// the returned backend.
    fn cpu_fallback(&self) -> Option<Box<dyn Backend>> {
        None
    }

    /// Execute a circuit and return results.
    fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> Result<ExecResult>;

    /// Compute expectation value of an observable. Default uses statevector.
    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<f64> {
        let _ = (circuit, params, observable);
        Err(crate::error::OmegaError::Unsupported(
            "expectation not implemented for this backend".into(),
        ))
    }

    /// Compute expectation values for *multiple* observables against the
    /// same circuit + params, in one shot. The default implementation
    /// just loops [`Backend::expectation`], which re-runs the forward
    /// sweep per observable — backends that prepare a residual on-device
    /// statevector should override this so the forward sweep amortises
    /// across the whole batch (see the Metal backend for the
    /// fused-forward + per-observable-reduction implementation).
    ///
    /// Used by the QML trainer to evaluate `⟨Z_q⟩` on each measurement
    /// qubit per training point: that's one circuit + N observables,
    /// previously N forward sweeps.
    fn expectation_multi(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observables: &[Observable],
    ) -> Result<Vec<f64>> {
        observables
            .iter()
            .map(|o| self.expectation(circuit, params, o))
            .collect()
    }

    /// Expectation of ONE observable across MANY parameter bindings (one per
    /// row), against the same circuit. This is the row-batch counterpart of
    /// [`Backend::expectation_multi`] (which batches observables): scoring N
    /// data rows is N independent forward sweeps, so a backend can run them in
    /// parallel. The default loops [`Backend::expectation`] sequentially;
    /// backends override with a data-parallel implementation. The returned
    /// vector is in the SAME order as `bindings` (index-preserving), so seeded
    /// training runs stay bit-stable regardless of parallelism.
    fn expectation_batch(
        &self,
        circuit: &CircuitIR,
        bindings: &[&ParameterBinding],
        observable: &Observable,
    ) -> Result<Vec<f64>> {
        bindings
            .iter()
            .map(|b| self.expectation(circuit, b, observable))
            .collect()
    }

    /// Compute all gradients via adjoint differentiation in a single forward+backward pass.
    /// Returns `Ok(None)` if the backend doesn't support AD (triggers fallback to parameter-shift).
    fn adjoint_gradient(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<Option<Vec<(SymbolId, f64)>>> {
        let _ = (circuit, params, observable);
        Ok(None)
    }

    /// Adjoint gradients across MANY parameter bindings (one per row), against
    /// the same circuit and observable — the row-batch counterpart of
    /// [`Backend::adjoint_gradient`]. This is where a supervised training
    /// loop spends most of its time (one adjoint pass per data row per step),
    /// so parallelising over rows is the main throughput lever. The default
    /// loops sequentially; the returned vector is index-aligned with
    /// `bindings`, and each element carries the same `Ok(None)` /
    /// `Ok(Some(..))` contract as [`Backend::adjoint_gradient`].
    fn adjoint_gradient_batch(
        &self,
        circuit: &CircuitIR,
        bindings: &[&ParameterBinding],
        observable: &Observable,
    ) -> Result<Vec<AdjointGradient>> {
        bindings
            .iter()
            .map(|b| self.adjoint_gradient(circuit, b, observable))
            .collect()
    }

    /// Compute per-observable expectations *and* a gradient against a
    /// caller-provided observable that depends on those expectations,
    /// in a single combined call. Used by the QML trainer's per-train-
    /// point loop:
    /// - `observables` are the per-measurement-qubit `⟨Z_q⟩` to
    ///   produce predictions `y_hat`.
    /// - `gradient_observable_factory` is `|y_hat| → Σ 2·(y_hat_i −
    ///   y_i)·Z_{q_i}` — the residual-weighted observable whose
    ///   adjoint gradient is `dL/dθ`.
    ///
    /// The default implementation just calls `expectation_multi` then
    /// `adjoint_gradient` separately — semantically identical, but
    /// runs the forward sweep twice (~3 ms × 320 train-pt-epochs ≈
    /// 1 s of redundant GPU compute on the n=18 QML bench at the
    /// post-round-13 baseline).
    ///
    /// Backends that can fuse the two forward sweeps (Metal does)
    /// should override this. The closure is taken as
    /// `Box<dyn FnOnce>` so the trait stays object-safe via
    /// `&dyn Backend`.
    fn expectation_multi_then_gradient(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observables: &[Observable],
        gradient_observable_factory: GradientObservableFactory<'_>,
    ) -> Result<ExpectationsAndGradient> {
        let predictions = self.expectation_multi(circuit, params, observables)?;
        let obs = gradient_observable_factory(&predictions);
        let gradient = self.adjoint_gradient(circuit, params, &obs)?;
        Ok((predictions, gradient))
    }
}

/// One binding's adjoint gradient: `Some(per-symbol gradients)`, or `None`
/// when the backend can't run adjoint AD on the circuit (same contract as
/// [`Backend::adjoint_gradient`]). The element type of
/// [`Backend::adjoint_gradient_batch`].
pub type AdjointGradient = Option<Vec<(SymbolId, f64)>>;

/// Closure shape for [`Backend::expectation_multi_then_gradient`] —
/// builds the gradient `Observable` from the just-computed
/// predictions. Owned (`FnOnce`) because the closure typically
/// captures one of the trainer's per-train-point label vectors by
/// value to compute the residual.
pub type GradientObservableFactory<'a> = Box<dyn FnOnce(&[f64]) -> Observable + 'a>;

/// Return shape of [`Backend::expectation_multi_then_gradient`]:
/// `(predictions, optional gradient vector)`. The gradient is
/// `None` if the backend doesn't support adjoint AD on this
/// circuit (matches the contract of `adjoint_gradient`).
pub type ExpectationsAndGradient = (Vec<f64>, Option<Vec<(SymbolId, f64)>>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observable_builder_z() {
        let obs = Observable::z(0);
        assert_eq!(obs.terms.len(), 1);
        assert_eq!(obs.terms[0].0, 1.0);
        assert_eq!(obs.terms[0].1.len(), 1);
        assert_eq!(obs.terms[0].1[0].0, 0);
        assert!(matches!(obs.terms[0].1[0].1, PauliOp::Z));
    }

    #[test]
    fn test_observable_add() {
        let obs = Observable::z(0) + Observable::z(1);
        assert_eq!(obs.terms.len(), 2);
    }

    #[test]
    fn test_observable_scale() {
        let obs = Observable::z(0) * 0.5;
        assert_eq!(obs.terms[0].0, 0.5);

        let obs2 = 0.5 * Observable::z(0);
        assert_eq!(obs2.terms[0].0, 0.5);
    }

    #[test]
    fn test_observable_neg_sub_assign() {
        // -Z(0) flips the scalar.
        let neg_z = -Observable::z(0);
        assert_eq!(neg_z.terms[0].0, -1.0);

        // Sub distributes through term concatenation: Z0 - Z1 = Z0 + (-1)Z1.
        let diff = Observable::z(0) - Observable::z(1);
        assert_eq!(diff.terms.len(), 2);
        assert_eq!(diff.terms[0].0, 1.0);
        assert_eq!(diff.terms[1].0, -1.0);

        // AddAssign / MulAssign let builder loops accumulate without
        // shadowing the binding on every term.
        let mut h = Observable::z(0);
        h += Observable::z(1);
        assert_eq!(h.terms.len(), 2);
        h *= 0.5;
        assert!(h.terms.iter().all(|(c, _)| (*c - 0.5).abs() < 1e-15));
    }

    #[test]
    fn test_observable_h2_hamiltonian() {
        // H₂ at equilibrium: g₀I + g₁Z₀ + g₂Z₁ + g₃Z₀Z₁ + g₄X₀X₁ + g₅Y₀Y₁
        let g = [-1.0523, 0.3979, -0.3979, -0.0112, 0.1809, 0.1809];
        let h2 = g[0] * Observable::identity()
            + g[1] * Observable::z(0)
            + g[2] * Observable::z(1)
            + g[3] * Observable::zz(0, 1)
            + g[4] * Observable::xx(0, 1)
            + g[5] * Observable::yy(0, 1);
        assert_eq!(h2.terms.len(), 6);
    }

    #[test]
    fn test_observable_identity() {
        let obs = Observable::identity();
        assert_eq!(obs.terms.len(), 1);
        assert_eq!(obs.terms[0].0, 1.0);
        assert!(obs.terms[0].1.is_empty()); // identity has no Pauli operators
    }

    #[test]
    fn parse_pauli_string_drops_identity() {
        let v = parse_pauli_string("I0X1").unwrap();
        assert_eq!(v, vec![(1, PauliOp::X)]);
    }

    #[test]
    fn parse_pauli_string_two_digit_qubit() {
        let v = parse_pauli_string("X10Z11").unwrap();
        assert_eq!(v, vec![(10, PauliOp::X), (11, PauliOp::Z)]);
    }

    #[test]
    fn parse_observable_h2_like_sum() {
        let obs = Observable::parse("0.3979*Z0+-0.3979*Z1+-0.0112*Z0Z1+0.1809*X0X1").unwrap();
        assert_eq!(obs.terms.len(), 4);
        let coeffs: Vec<f64> = obs.terms.iter().map(|t| t.0).collect();
        assert!((coeffs[0] - 0.3979).abs() < 1e-12);
        assert!((coeffs[1] + 0.3979).abs() < 1e-12);
        assert!((coeffs[2] + 0.0112).abs() < 1e-12);
        assert!((coeffs[3] - 0.1809).abs() < 1e-12);
    }

    #[test]
    fn parse_observable_implicit_one_coeff() {
        let obs = Observable::parse("Z0Z1").unwrap();
        assert_eq!(obs.terms.len(), 1);
        assert_eq!(obs.terms[0].0, 1.0);
    }

    #[test]
    fn parse_observable_rejects_empty() {
        assert!(Observable::parse("").is_err());
    }

    // --- counts-keying convention -------------------------------------
    //
    // These pin the two functions that decide what a `Counts` key MEANS.
    // Both were extracted from front ends so the N-way matrix could reuse
    // them; the tests below are what makes the extraction safe to trust.

    use crate::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};

    fn op(gate: GateKind, qubits: &[u32], cbit: Option<u32>) -> GateOp {
        GateOp {
            gate,
            qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
            params: smallvec::smallvec![],
            classical_bit: cbit,
            condition: None,
        }
    }

    fn circuit(n: u32, ncl: u32, ops: Vec<GateOp>) -> CircuitIR {
        let mut ir = CircuitIR::new(n, CircuitType::GateBased);
        ir.num_classical_bits = ncl;
        ir.ops = ops;
        ir
    }

    /// `h q0; cx q0,q1; h q2; measure q0->c0; measure q1->c1` — the shape of
    /// `08_partial_measure.qasm`. Terminal measures, no conditionals, no
    /// overwrite, so `Skip` + projection is the correct (and cheaper) route.
    #[test]
    fn terminal_measures_do_not_need_collapse() {
        let c = circuit(
            3,
            2,
            vec![
                op(GateKind::H, &[0], None),
                op(GateKind::CX, &[0, 1], None),
                op(GateKind::H, &[2], None),
                op(GateKind::Measure, &[0], Some(0)),
                op(GateKind::Measure, &[1], Some(1)),
            ],
        );
        assert!(!needs_collapse(&c));
    }

    #[test]
    fn a_gate_after_a_measure_needs_collapse() {
        let c = circuit(
            2,
            2,
            vec![
                op(GateKind::Measure, &[0], Some(0)),
                op(GateKind::X, &[1], None),
                op(GateKind::Measure, &[1], Some(1)),
            ],
        );
        assert!(
            needs_collapse(&c),
            "a gate between two measures must force collapse — this is the case \
             where the CLI's now-merged 'ops after the LAST measure' disjunct \
             was false and only the 'ops after ANY measure' one fired"
        );
    }

    #[test]
    fn a_conditioned_gate_needs_collapse_even_with_terminal_measures() {
        let mut conditioned = op(GateKind::X, &[1], None);
        conditioned.condition = Some((0, 1, 1));
        let c = circuit(
            2,
            2,
            vec![
                op(GateKind::Measure, &[0], Some(0)),
                conditioned,
                op(GateKind::Measure, &[1], Some(1)),
            ],
        );
        assert!(needs_collapse(&c));
    }

    #[test]
    fn a_twice_written_cbit_needs_collapse() {
        // Both measures are terminal, so the "op after a measure" disjunct is
        // false. Only the overwrite check catches this. Without it, Skip-mode
        // projection would apply last-write-wins to a SINGLE sample of both
        // qubits, which is not the same distribution as Qiskit's.
        let c = circuit(
            2,
            1,
            vec![
                op(GateKind::H, &[0], None),
                op(GateKind::H, &[1], None),
                op(GateKind::Measure, &[0], Some(0)),
                op(GateKind::Measure, &[1], Some(0)),
            ],
        );
        assert!(needs_collapse(&c));
    }

    #[test]
    fn no_measures_at_all_does_not_need_collapse() {
        let c = circuit(2, 0, vec![op(GateKind::H, &[0], None)]);
        assert!(!needs_collapse(&c));
    }

    /// The projection must DROP unmeasured qubits, not merely reorder bits.
    /// This is the failure that would otherwise be blamed on a backend.
    #[test]
    fn projection_drops_the_unmeasured_qubit() {
        let c = circuit(
            3,
            2,
            vec![
                op(GateKind::Measure, &[0], Some(0)),
                op(GateKind::Measure, &[1], Some(1)),
            ],
        );
        let pairs = measure_pairs(&c);
        assert_eq!(pairs, vec![(0, 0), (1, 1)]);

        // Full-register keys: bit2 = q2 (unmeasured, a coin flip), bits 1..0
        // = the correlated Bell pair. Four raw keys must collapse to two.
        let mut raw: HashMap<u64, u32> = HashMap::new();
        raw.insert(0b000, 25); // q2=0, q1q0=00
        raw.insert(0b100, 25); // q2=1, q1q0=00
        raw.insert(0b011, 25); // q2=0, q1q0=11
        raw.insert(0b111, 25); // q2=1, q1q0=11
        let out = project_counts_onto_creg(ExecResult::Counts(raw), &pairs).unwrap();
        let got = out.counts();
        assert_eq!(got.len(), 2, "expected 2 creg outcomes, got {got:?}");
        assert_eq!(got.get(&0b00), Some(&50));
        assert_eq!(got.get(&0b11), Some(&50));
    }

    /// A permuted qubit→cbit map must actually permute. The vendored corpus
    /// is all-identity, so nothing there can distinguish a correct projection
    /// from one that ignores `classical_bit` and uses the qubit index.
    #[test]
    fn projection_honours_a_permuted_qubit_to_cbit_map() {
        let c = circuit(
            2,
            2,
            vec![
                op(GateKind::Measure, &[0], Some(1)),
                op(GateKind::Measure, &[1], Some(0)),
            ],
        );
        assert_eq!(measure_pairs(&c), vec![(0, 1), (1, 0)]);
        let mut raw: HashMap<u64, u32> = HashMap::new();
        raw.insert(0b01, 7); // q0=1, q1=0
        let out = project_counts_onto_creg(ExecResult::Counts(raw), &measure_pairs(&c)).unwrap();
        // q0=1 lands in c1, q1=0 in c0 → creg value 0b10.
        assert_eq!(out.counts().get(&0b10), Some(&7));
        assert_eq!(out.counts().get(&0b01), None);
    }

    /// Last-write-wins on an overwritten cbit, matching Qiskit.
    #[test]
    fn projection_is_last_write_wins_on_a_reused_cbit() {
        let c = circuit(
            2,
            1,
            vec![
                op(GateKind::Measure, &[0], Some(0)),
                op(GateKind::Measure, &[1], Some(0)),
            ],
        );
        let mut raw: HashMap<u64, u32> = HashMap::new();
        raw.insert(0b01, 3); // q0=1, q1=0 → c0 written 1 then 0
        let out = project_counts_onto_creg(ExecResult::Counts(raw), &measure_pairs(&c)).unwrap();
        assert_eq!(out.counts().get(&0), Some(&3), "q1's later write must win");
    }

    /// No pairs → pass through untouched. A no-measure circuit's counts are
    /// already keyed over the full register, which is what the Qiskit
    /// runner's synthesised `measure_all()` produces.
    #[test]
    fn projection_without_measures_is_identity() {
        let mut raw: HashMap<u64, u32> = HashMap::new();
        raw.insert(0b101, 9);
        let out = project_counts_onto_creg(ExecResult::Counts(raw), &[]).unwrap();
        assert_eq!(out.counts().get(&0b101), Some(&9));
    }

    #[test]
    fn projection_refuses_registers_beyond_u64() {
        let err = project_counts_onto_creg(ExecResult::Counts(HashMap::new()), &[(0, 64)])
            .expect_err("cbit 64 does not fit a u64 key");
        assert!(err.contains("u64"), "unhelpful message: {err}");
    }
}
