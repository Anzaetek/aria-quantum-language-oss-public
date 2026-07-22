//! Generic gradient computation via the parameter-shift rule.
//!
//! Works with any backend that implements `Backend::execute`.

use crate::circuit::{CircuitIR, GateKind, ParamExpr, SymbolId};
use crate::error::{OmegaError, Result};
use crate::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable, PauliOp};
use crate::params::ParameterBinding;
use crate::qubo::Qubo;

/// A scalar function of a bit-string outcome — the kind of thing operators
/// actually want to differentiate when they say "the gradient of the cost".
///
/// Two shapes today:
/// - `Qubo(Q)`: f(x) = x^T Q x. The exact diagonal observable factorises
///   into a sparse Pauli-Z sum (linear in `n` plus quadratic), so adjoint
///   AD on the built observable has the same cost as on the underlying
///   Ising Hamiltonian.
/// - `Table(rows)`: f given by enumeration over a small bit-space. Useful
///   for ad-hoc CVaR / KL / variance estimands. Builds a full 2^n Pauli-Z
///   sum via the Walsh-Hadamard transform; only practical for `n <= ~12`.
#[derive(Clone, Debug)]
pub enum Functional {
    /// Quadratic Unconstrained Binary Optimisation cost x^T Q x.
    Qubo(Qubo),
    /// Lookup table: each row pairs a bit-vector with f(x).
    /// Bits not in any row default to f = 0.
    Table {
        num_qubits: u32,
        rows: Vec<(Vec<bool>, f64)>,
    },
}

impl Functional {
    /// Number of qubits the functional acts on.
    pub fn num_qubits(&self) -> u32 {
        match self {
            Functional::Qubo(q) => q.n as u32,
            Functional::Table { num_qubits, .. } => *num_qubits,
        }
    }

    /// Evaluate f at a bit-vector. `bits[i]` is the value of bit i (LSB).
    /// Bits beyond the configured `num_qubits` are ignored.
    pub fn evaluate(&self, bits: &[bool]) -> f64 {
        match self {
            Functional::Qubo(q) => q.evaluate(bits),
            Functional::Table { num_qubits, rows } => {
                let n = *num_qubits as usize;
                for (k, v) in rows {
                    if k.len() == n && k.iter().zip(bits.iter()).all(|(a, b)| a == b) {
                        return *v;
                    }
                }
                0.0
            }
        }
    }

    /// Build the diagonal observable `O_f = Σ_x f(x) |x⟩⟨x|` as a Pauli-Z sum.
    ///
    /// For `Qubo`, this reuses `Qubo::to_ising().to_observable()` — the
    /// resulting observable is sparse: at most `n + n^2/2 + 1` terms.
    /// For `Table`, we run the Walsh-Hadamard transform and emit all
    /// non-zero Pauli-Z coefficients (up to 2^n terms).
    pub fn to_diagonal_observable(&self) -> Observable {
        match self {
            Functional::Qubo(q) => q.to_ising().to_observable(),
            Functional::Table { num_qubits, rows } => {
                let n = *num_qubits as usize;
                assert!(
                    n <= 16,
                    "Table → Pauli-Z transform is 2^n; refusing n={} (use Qubo for sparse cases)",
                    n
                );
                let dim = 1usize << n;
                // f[x] indexed by integer encoding of bit-vector
                let mut f_vals = vec![0.0; dim];
                for (bits, v) in rows {
                    assert_eq!(bits.len(), n, "row bit-length mismatch");
                    let mut idx = 0usize;
                    for (i, &b) in bits.iter().enumerate() {
                        if b {
                            idx |= 1 << i;
                        }
                    }
                    f_vals[idx] = *v;
                }
                // Pauli-Z coefficient for subset S:
                //   c_S = (1/2^n) Σ_x f(x) Π_{i ∈ S} (-1)^{x_i}
                // i.e. c_S = (1/2^n) Σ_x f(x) (-1)^{popcount(S & x)}
                let inv_dim = 1.0 / dim as f64;
                let mut terms: Vec<(f64, Vec<(u32, PauliOp)>)> = Vec::new();
                for s in 0..dim {
                    let mut acc = 0.0;
                    for (x, &fx) in f_vals.iter().enumerate() {
                        let sign = if (s & x).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        acc += fx * sign;
                    }
                    let coeff = acc * inv_dim;
                    if coeff.abs() < 1e-12 {
                        continue;
                    }
                    let mut pauli = Vec::new();
                    for i in 0..n {
                        if (s >> i) & 1 == 1 {
                            pauli.push((i as u32, PauliOp::Z));
                        }
                    }
                    terms.push((coeff, pauli));
                }
                if terms.is_empty() {
                    // All-zero functional → trivial identity·0 observable.
                    return Observable {
                        terms: vec![(0.0, vec![])],
                    };
                }
                Observable { terms }
            }
        }
    }
}

/// Parse a functional spec from JSON. Accepts either:
///   `{"qubo":{"n":N,"Q":[[i,j,c],...]}}`              → [`Functional::Qubo`]
///   `{"table":[["bits",value],...],"num_qubits":N}`   → [`Functional::Table`]
///
/// Returns the functional and a static label (`"qubo"` / `"table"`).
/// Shared between the CLI's `--gradient-of-fn` flag and the WASM
/// host hook so both call sites stay in sync.
pub fn parse_functional_spec(s: &str) -> std::result::Result<(Functional, &'static str), String> {
    let v: serde_json::Value =
        serde_json::from_str(s).map_err(|e| format!("invalid JSON: {}", e))?;
    if let Some(qobj) = v.get("qubo") {
        let qubo = Qubo::from_json(&qobj.to_string()).map_err(|e| format!("qubo: {}", e))?;
        return Ok((Functional::Qubo(qubo), "qubo"));
    }
    if let Some(rows_v) = v.get("table") {
        let num_qubits = v
            .get("num_qubits")
            .and_then(|x| x.as_u64())
            .ok_or("table: missing `num_qubits`")? as u32;
        let arr = rows_v.as_array().ok_or("table: not an array")?;
        let mut rows: Vec<(Vec<bool>, f64)> = Vec::with_capacity(arr.len());
        for row in arr {
            let r = row.as_array().ok_or("table: row not array")?;
            if r.len() != 2 {
                return Err("table row must be [bits, value]".into());
            }
            let bits_s = r[0].as_str().ok_or("table: bits not string")?;
            if bits_s.len() != num_qubits as usize {
                return Err(format!(
                    "table: bits `{}` length ≠ num_qubits {}",
                    bits_s, num_qubits
                ));
            }
            let bits: Vec<bool> = bits_s
                .chars()
                .map(|c| match c {
                    '0' => Ok(false),
                    '1' => Ok(true),
                    other => Err(format!("table: bad bit char `{}`", other)),
                })
                .collect::<std::result::Result<_, _>>()?;
            let val = r[1].as_f64().ok_or("table: value not float")?;
            rows.push((bits, val));
        }
        return Ok((Functional::Table { num_qubits, rows }, "table"));
    }
    Err("expected `qubo` or `table` key in the spec".into())
}

/// Gradient computation method.
#[derive(Clone, Debug, Default)]
pub enum GradMethod {
    /// Parameter-shift rule: df/dθ = [f(θ+π/2) - f(θ-π/2)] / 2
    /// Requires 2 circuit evaluations per parameter.
    #[default]
    ParameterShift,
    /// Finite difference: df/dθ ≈ [f(θ+ε) - f(θ)] / ε
    FiniteDifference { epsilon: f64 },
    /// Adjoint differentiation: all gradients in 1 forward + 1 backward pass.
    /// Falls back to ParameterShift if the backend doesn't support it.
    Adjoint,
    /// Stochastic parameter-shift: average over many shots to handle mid-circuit measurements.
    StochasticParameterShift { shots: u32 },
    /// Parallelised parameter-shift (arXiv:2606.03517): symbols confined
    /// to the trailing commuting block get their gradients from a single
    /// batched evaluation of commutator observables `i·[G_k, H]`; other
    /// symbols fall back to the serial per-slot shift rules. See
    /// [`crate::parallel_shift`].
    ParallelParameterShift,
    /// Automatic: selects the best method based on circuit properties.
    /// - Circuits with measurements → StochasticParameterShift
    /// - Unitary circuits → Adjoint (falling back to ParameterShift)
    Auto,
}

/// Independent entry point for "differentiate `f(measurement_outcomes)`".
///
/// Two sub-methods, picked by the caller:
///
/// - [`FunctionalGradMethod::DiagonalObservable`]: build the Pauli-Z
///   observable `O_f = Σ_x f(x) |x⟩⟨x|` once, then run adjoint AD on it.
///   Exact, cost = single adjoint pass independent of `shots`. Covers
///   ~90% of "function-of-measurement" requests in practice — anything
///   that's linear in bit-string probabilities (QUBO cost, MaxCut
///   value, any diagonal observable).
///
/// - [`FunctionalGradMethod::ScoreFunction`]: REINFORCE estimator
///   `∇E[f] ≈ (1/S) Σ_s f(x_s) · ∇log p(x_s|θ)`. Unbiased but noisy.
///   Use for genuinely nonlinear `f` (variance, CVaR, KL, entropy).
///
/// The two are *not* parts of [`GradMethod`] because they take a
/// different input — a [`Functional`] rather than an [`Observable`] —
/// and produce gradients of `E[f]` rather than `<O>`. Wrapping them in
/// a common enum would only force callers to construct an `Observable`
/// they don't have.
#[derive(Clone, Debug)]
pub enum FunctionalGradMethod {
    /// Convert `f` to its Pauli-Z observable, run adjoint AD on it.
    DiagonalObservable,
    /// REINFORCE estimator over `shots` samples per parameter.
    ScoreFunction { shots: u32 },
}

/// Compute the gradient of `E[f(x)]` with respect to all free parameters,
/// where `x` is the measurement outcome of `circuit` and `f` is a
/// [`Functional`] over bit-strings.
///
/// Returns a `Vec` of `(symbol_id, gradient_value)` pairs, sorted by
/// `symbol_id` for deterministic ordering.
pub fn compute_functional_gradient(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    f: &Functional,
    method: &FunctionalGradMethod,
) -> Result<Vec<(SymbolId, f64)>> {
    if f.num_qubits() != circuit.num_qubits {
        return Err(OmegaError::Unsupported(format!(
            "functional num_qubits={} ≠ circuit num_qubits={}",
            f.num_qubits(),
            circuit.num_qubits
        )));
    }
    match method {
        FunctionalGradMethod::DiagonalObservable => {
            let observable = f.to_diagonal_observable();
            // Diagonal observables commute with mid-circuit measurements,
            // but the adjoint path still rejects them. Fall back to
            // parameter-shift if the user has wired one in.
            compute_gradient(backend, circuit, params, &observable, &GradMethod::Adjoint)
        }
        FunctionalGradMethod::ScoreFunction { shots } => {
            score_function_gradient(backend, circuit, params, f, *shots)
        }
    }
}

/// REINFORCE estimator
/// `∇E[f] = E_x[f(x) · ∇log p(x|θ)] ≈ (1/S) Σ_s f(x_s) · ∇log p(x_s|θ)`
/// where the gradient of the log-prob is itself estimated by the
/// parameter-shift rule on the projector `|x_s⟩⟨x_s|` (a diagonal Pauli-Z
/// observable on a single bit-string).
///
/// Cost: `2 · num_params · shots` circuit evaluations. Honest:
/// you only want this when `DiagonalObservable` doesn't apply.
fn score_function_gradient(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    f: &Functional,
    shots: u32,
) -> Result<Vec<(SymbolId, f64)>> {
    let mut symbol_ids: Vec<SymbolId> = circuit.symbols.keys().copied().collect();
    symbol_ids.sort();
    let active_symbols = find_active_symbols(circuit);
    let n = circuit.num_qubits as usize;

    let config = ExecConfig {
        shots: Some(shots),
        seed: Some(0),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let result = backend.execute(circuit, params, &config)?;
    let counts = match result {
        ExecResult::Counts(c) => c,
        _ => {
            return Err(OmegaError::Unsupported(
                "ScoreFunction requires a backend that returns Counts".into(),
            ))
        }
    };

    // For each unique sampled bit-string, compute f(x) and the parameter
    // gradient of `log p(x|θ)` via parameter-shift on the indicator
    // observable `|x⟩⟨x|`. Then weight by frequency.
    let mut grads: Vec<f64> = vec![0.0; symbol_ids.len()];
    let total_shots = shots as f64;

    for (bitstring, count) in counts.iter() {
        let bits = u64_to_bools(*bitstring, n);
        let fx = f.evaluate(&bits);
        let weight = (*count as f64) / total_shots;
        if fx.abs() < 1e-15 {
            continue;
        }
        let p_obs = bitstring_projector_observable(&bits);
        let p_x = backend.expectation(circuit, params, &p_obs)?;
        if p_x.abs() < 1e-12 {
            // Sampled but the analytic probability is ~0 — likely numerical
            // noise; skip rather than divide by zero.
            continue;
        }
        for (idx, &sym_id) in symbol_ids.iter().enumerate() {
            if !active_symbols.contains(&sym_id) {
                continue;
            }
            let dp = parameter_shift_gradient(backend, circuit, params, &p_obs, sym_id)?;
            grads[idx] += weight * fx * (dp / p_x);
        }
    }

    Ok(symbol_ids.into_iter().zip(grads).collect())
}

fn u64_to_bools(x: u64, n: usize) -> Vec<bool> {
    (0..n).map(|i| (x >> i) & 1 == 1).collect()
}

/// Diagonal observable `|x⟩⟨x| = Π_i (1 + (-1)^{x_i} Z_i) / 2`.
/// Returns the equivalent Pauli-Z sum.
fn bitstring_projector_observable(bits: &[bool]) -> Observable {
    let n = bits.len();
    let dim = 1usize << n;
    let inv_dim = 1.0 / dim as f64;
    let mut terms = Vec::new();
    for s in 0..dim {
        let mut sign = 1.0;
        for (i, &b) in bits.iter().enumerate() {
            if (s >> i) & 1 == 1 && b {
                sign = -sign;
            }
        }
        let coeff = sign * inv_dim;
        let mut pauli = Vec::new();
        for i in 0..n {
            if (s >> i) & 1 == 1 {
                pauli.push((i as u32, PauliOp::Z));
            }
        }
        terms.push((coeff, pauli));
    }
    Observable { terms }
}

/// Compute the gradient of an expectation value with respect to all free parameters.
///
/// Returns a Vec of (symbol_id, gradient_value) pairs.
pub fn compute_gradient(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    method: &GradMethod,
) -> Result<Vec<(SymbolId, f64)>> {
    compute_gradient_for(backend, circuit, params, observable, method, None)
}

/// Like [`compute_gradient`] but restricted to a subset of symbols.
/// With `only = Some(set)`, only those symbols' gradients are computed
/// and returned (frozen/layer-wise training skips the rest — for the
/// per-symbol shift methods this also skips their circuit evaluations).
/// With `only = None` this is exactly [`compute_gradient`].
pub fn compute_gradient_for(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    method: &GradMethod,
    only: Option<&std::collections::HashSet<SymbolId>>,
) -> Result<Vec<(SymbolId, f64)>> {
    let has_measurements = circuit
        .ops
        .iter()
        .any(|op| matches!(op.gate, GateKind::Measure));

    // Auto: select best method based on circuit
    if matches!(method, GradMethod::Auto) {
        let resolved = if has_measurements {
            GradMethod::StochasticParameterShift { shots: 100 }
        } else {
            GradMethod::Adjoint
        };
        return compute_gradient_for(backend, circuit, params, observable, &resolved, only);
    }

    // Adjoint: reject circuits with measurements, then try batch AD
    if matches!(method, GradMethod::Adjoint) {
        if has_measurements {
            return Err(OmegaError::Unsupported(
                "adjoint gradient not supported for circuits with mid-circuit measurements; \
                 use StochasticParameterShift or Auto instead"
                    .into(),
            ));
        }
        if let Some(grads) = try_adjoint_gradient(backend, circuit, params, observable)? {
            // Adjoint computes all gradients in one pass; restrict the
            // *output* to the requested subset.
            return Ok(match only {
                Some(set) => grads.into_iter().filter(|(s, _)| set.contains(s)).collect(),
                None => grads,
            });
        }
        // Backend doesn't support AD — fall back to parameter-shift
        return compute_gradient_for(
            backend,
            circuit,
            params,
            observable,
            &GradMethod::ParameterShift,
            only,
        );
    }

    // Parallel commuting-block path (arXiv:2606.03517).
    if matches!(method, GradMethod::ParallelParameterShift) {
        let (grads, _report) = crate::parallel_shift::parallel_parameter_shift_gradient(
            backend, circuit, params, observable, only,
        )?;
        return Ok(grads);
    }

    let mut gradients = Vec::new();
    let mut symbol_ids: Vec<SymbolId> = circuit.symbols.keys().copied().collect();
    symbol_ids.sort();

    let active_symbols = find_active_symbols(circuit);

    for &sym_id in &symbol_ids {
        if let Some(set) = only {
            if !set.contains(&sym_id) {
                continue;
            }
        }
        if !active_symbols.contains(&sym_id) {
            gradients.push((sym_id, 0.0));
            continue;
        }

        let grad = match method {
            GradMethod::ParameterShift => {
                parameter_shift_gradient(backend, circuit, params, observable, sym_id)?
            }
            GradMethod::FiniteDifference { epsilon } => {
                finite_difference_gradient(backend, circuit, params, observable, sym_id, *epsilon)?
            }
            GradMethod::StochasticParameterShift { shots } => stochastic_parameter_shift_gradient(
                backend, circuit, params, observable, sym_id, *shots,
            )?,
            GradMethod::Adjoint | GradMethod::Auto | GradMethod::ParallelParameterShift => {
                unreachable!("handled above")
            }
        };

        gradients.push((sym_id, grad));
    }

    Ok(gradients)
}

/// Attempt adjoint differentiation, falling back to parameter-shift if unsupported.
fn try_adjoint_gradient(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
) -> Result<Option<Vec<(SymbolId, f64)>>> {
    backend.adjoint_gradient(circuit, params, observable)
}

/// Per-slot parameter-shift rule. Picked from the slot's gate kind: the
/// 2-term `±π/2` rule applies to single-Pauli rotation generators
/// (spectrum `±1/2` → frequency 1); the 4-term Banchi-Crooks variant
/// applies to controlled rotations whose generator spectrum is
/// `{0, 0, ±1/2}` → frequencies `{1/2, 1}` — see
/// `verification/Verification/Adjoint/AdjointEqShift.lean` (CRz block).
enum SlotShiftRule {
    /// `f'(θ) = (1/2)·(f(θ+π/2) − f(θ-π/2))`.
    TwoTerm,
    /// `f'(θ) = c₁·(f(θ+π/2) − f(θ-π/2)) + c₂·(f(θ-3π/2) − f(θ+3π/2))`
    /// with `c₁ = (√2+1)/(4√2)`, `c₂ = (√2-1)/(4√2)`. Inverts the
    /// trig system `Σ_j 2·d_j·sin(ω_k·x_j) = ω_k` for shifts
    /// `x ∈ {π/2, 3π/2}` and frequencies `ω ∈ {1/2, 1}`.
    FourTermBanchiCrooks,
    /// `f'(θ) = (f(θ+π/4) − f(θ-π/4)) − ((√2-1)/2)·(f(θ+π/2) − f(θ-π/2))`.
    /// For Givens-type generators with spectrum `{0, 0, ±1}` →
    /// frequencies `{1, 2}` (RBS: `G = (Y⊗X − X⊗Y)/2`). This is the
    /// parallelisable rule of Mathur–Kerenidis et al.
    /// (arXiv:2606.03517, Eq. 6); the naive 2-term ±π/2 rule is
    /// *wrong* here — it would require `sin(2·π/2) = 2`, false.
    FourTermGivens,
}

fn slot_shift_rule(gate: &GateKind) -> SlotShiftRule {
    match gate {
        GateKind::CRz | GateKind::CU3 => SlotShiftRule::FourTermBanchiCrooks,
        GateKind::Rbs => SlotShiftRule::FourTermGivens,
        _ => SlotShiftRule::TwoTerm,
    }
}

/// Coordinates of a parametric occurrence of a symbol inside a circuit:
/// the gate-op index and the parameter-slot index within that gate.
#[derive(Clone, Copy, Debug)]
struct SlotRef {
    op_idx: usize,
    param_idx: usize,
}

/// Does `expr` mention `sym_id` anywhere?
fn param_uses_symbol(expr: &ParamExpr, sym_id: SymbolId) -> bool {
    match expr {
        ParamExpr::Concrete(_) => false,
        ParamExpr::Symbol(id) => *id == sym_id,
        ParamExpr::Negate(inner) => param_uses_symbol(inner, sym_id),
        ParamExpr::Add(a, b) | ParamExpr::Mul(a, b) => {
            param_uses_symbol(a, sym_id) || param_uses_symbol(b, sym_id)
        }
    }
}

/// Every (op_idx, param_idx) slot where the symbol appears in the
/// param expression. A symbol can drive multiple slots (shared
/// parameter) and even multiple params within the same gate.
fn find_symbol_slots(circuit: &CircuitIR, sym_id: SymbolId) -> Vec<SlotRef> {
    let mut out = Vec::new();
    for (op_idx, op) in circuit.ops.iter().enumerate() {
        for (param_idx, p) in op.params.iter().enumerate() {
            if param_uses_symbol(p, sym_id) {
                out.push(SlotRef { op_idx, param_idx });
            }
        }
    }
    out
}

/// Clone `circuit`, replacing slot `slot.op_idx`/`slot.param_idx`'s
/// param expression with `Concrete(value)`. Other params (including
/// other slots that also reference the same symbol) are untouched, so
/// `params` continues to resolve them at the current binding.
fn shifted_circuit(circuit: &CircuitIR, slot: SlotRef, value: f64) -> CircuitIR {
    let mut c = circuit.clone();
    c.ops[slot.op_idx].params[slot.param_idx] = ParamExpr::Concrete(value);
    c
}

/// Banchi-Crooks 4-term coefficients for the two-frequency spectrum
/// `{1/2, 1}` with shifts `{π/2, 3π/2}`. Returned as `(c₁, c₂)`.
fn banchi_crooks_two_freq_coeffs() -> (f64, f64) {
    let sqrt2 = std::f64::consts::SQRT_2;
    let inv_4sqrt2 = 1.0 / (4.0 * sqrt2);
    ((sqrt2 + 1.0) * inv_4sqrt2, (sqrt2 - 1.0) * inv_4sqrt2)
}

/// Compute gradient for a single symbol via the parameter-shift rule,
/// dispatching on each occurrence's generator spectrum.
///
/// Chain rule: for a symbol `s` driving slots `k = 1..K`, each slot
/// with effective angle `v_k(s)` (linear or nonlinear in `s`) inside a
/// gate `G_k`, the gradient is
///     `df/ds = Σ_k (∂v_k/∂s) · (∂f/∂v_k)`
/// where `∂f/∂v_k` is the per-slot expectation derivative obtained by
/// shifting *only* slot `k`'s effective angle with the rule appropriate
/// to `G_k`'s generator. Pre-fix code assumed the 2-term rule and a
/// global shift of `s`; for CRz/CU3 slots the spectrum has frequencies
/// `{1/2, 1}` and the 2-term ±π/2 rule is wrong (it would set
/// `sin(Δ·π/2) = Δ` for `Δ = 1/2`, false at `sin(π/4) = √2/2`).
fn parameter_shift_gradient(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    sym_id: SymbolId,
) -> Result<f64> {
    let slots = find_symbol_slots(circuit, sym_id);
    if slots.is_empty() {
        return Ok(0.0);
    }
    let mut grad = 0.0;
    for slot in slots {
        let gate = circuit.ops[slot.op_idx].gate.clone();
        let pe = &circuit.ops[slot.op_idx].params[slot.param_idx];
        let v = params.resolve(pe)?;
        let alpha = params.resolve_derivative(pe, sym_id)?;
        if alpha == 0.0 {
            continue;
        }
        let slot_grad = match slot_shift_rule(&gate) {
            SlotShiftRule::TwoTerm => slot_two_term(backend, circuit, params, observable, slot, v)?,
            SlotShiftRule::FourTermBanchiCrooks => {
                slot_four_term(backend, circuit, params, observable, slot, v)?
            }
            SlotShiftRule::FourTermGivens => {
                slot_four_term_givens(backend, circuit, params, observable, slot, v)?
            }
        };
        grad += alpha * slot_grad;
    }
    Ok(grad)
}

fn slot_two_term(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    slot: SlotRef,
    v: f64,
) -> Result<f64> {
    let shift = std::f64::consts::FRAC_PI_2;
    let c_plus = shifted_circuit(circuit, slot, v + shift);
    let c_minus = shifted_circuit(circuit, slot, v - shift);
    let e_plus = backend.expectation(&c_plus, params, observable)?;
    let e_minus = backend.expectation(&c_minus, params, observable)?;
    Ok((e_plus - e_minus) / 2.0)
}

fn slot_four_term(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    slot: SlotRef,
    v: f64,
) -> Result<f64> {
    let pi_2 = std::f64::consts::FRAC_PI_2;
    let three_pi_2 = 3.0 * pi_2;
    let (c1, c2) = banchi_crooks_two_freq_coeffs();
    let c_p1 = shifted_circuit(circuit, slot, v + pi_2);
    let c_m1 = shifted_circuit(circuit, slot, v - pi_2);
    let c_p3 = shifted_circuit(circuit, slot, v + three_pi_2);
    let c_m3 = shifted_circuit(circuit, slot, v - three_pi_2);
    let e_p1 = backend.expectation(&c_p1, params, observable)?;
    let e_m1 = backend.expectation(&c_m1, params, observable)?;
    let e_p3 = backend.expectation(&c_p3, params, observable)?;
    let e_m3 = backend.expectation(&c_m3, params, observable)?;
    Ok(c1 * (e_p1 - e_m1) + c2 * (e_m3 - e_p3))
}

/// Givens 4-term rule coefficient `(√2 − 1)/2` for the two-frequency
/// spectrum `{1, 2}` with shifts `{π/4, π/2}` (arXiv:2606.03517 Eq. 6).
/// Solves `2d₁·sin(ω·π/4) + 2d₂·sin(ω·π/2)·c = ω` exactly for both
/// `ω ∈ {1, 2}` with `d₁ = 1`, `d₂ = −(√2−1)/2`.
fn givens_two_freq_coeff() -> f64 {
    (std::f64::consts::SQRT_2 - 1.0) / 2.0
}

fn slot_four_term_givens(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    slot: SlotRef,
    v: f64,
) -> Result<f64> {
    let pi_4 = std::f64::consts::FRAC_PI_4;
    let pi_2 = std::f64::consts::FRAC_PI_2;
    let c2 = givens_two_freq_coeff();
    let e_p1 = backend.expectation(
        &shifted_circuit(circuit, slot, v + pi_4),
        params,
        observable,
    )?;
    let e_m1 = backend.expectation(
        &shifted_circuit(circuit, slot, v - pi_4),
        params,
        observable,
    )?;
    let e_p2 = backend.expectation(
        &shifted_circuit(circuit, slot, v + pi_2),
        params,
        observable,
    )?;
    let e_m2 = backend.expectation(
        &shifted_circuit(circuit, slot, v - pi_2),
        params,
        observable,
    )?;
    Ok((e_p1 - e_m1) - c2 * (e_p2 - e_m2))
}

/// Compute gradient for a single parameter using finite differences.
fn finite_difference_gradient(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    sym_id: SymbolId,
    epsilon: f64,
) -> Result<f64> {
    let current_val = params.get(sym_id).unwrap_or(0.0);

    let e_current = backend.expectation(circuit, params, observable)?;

    let mut params_shifted = params.clone();
    params_shifted.bind(sym_id, current_val + epsilon);
    let e_shifted = backend.expectation(circuit, &params_shifted, observable)?;

    Ok((e_shifted - e_current) / epsilon)
}

/// Compute gradient for a single parameter using stochastic parameter-shift.
///
/// Runs `shots` circuit executions at each shift point, averaging expectation
/// values across measurement outcomes. Works for circuits with mid-circuit
/// measurements. Dispatches per slot the same way the deterministic
/// `parameter_shift_gradient` does — CRz/CU3 slots get the 4-term
/// Banchi-Crooks rule, everything else the 2-term ±π/2 rule.
fn stochastic_parameter_shift_gradient(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    sym_id: SymbolId,
    shots: u32,
) -> Result<f64> {
    let slots = find_symbol_slots(circuit, sym_id);
    if slots.is_empty() {
        return Ok(0.0);
    }
    let mut grad = 0.0;
    for slot in slots {
        let gate = circuit.ops[slot.op_idx].gate.clone();
        let pe = &circuit.ops[slot.op_idx].params[slot.param_idx];
        let v = params.resolve(pe)?;
        let alpha = params.resolve_derivative(pe, sym_id)?;
        if alpha == 0.0 {
            continue;
        }
        let slot_grad = match slot_shift_rule(&gate) {
            SlotShiftRule::TwoTerm => {
                stochastic_slot_two_term(backend, circuit, params, observable, slot, v, shots)?
            }
            SlotShiftRule::FourTermBanchiCrooks => {
                stochastic_slot_four_term(backend, circuit, params, observable, slot, v, shots)?
            }
            SlotShiftRule::FourTermGivens => stochastic_slot_four_term_givens(
                backend, circuit, params, observable, slot, v, shots,
            )?,
        };
        grad += alpha * slot_grad;
    }
    Ok(grad)
}

/// Shot-averaged expectation of `observable` after running `circuit`
/// with `params`. Helper shared by the stochastic-PSR slot routines.
fn stochastic_expectation_avg(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    shots: u32,
) -> Result<f64> {
    let n = circuit.num_qubits;
    let base_config = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let mut sum = 0.0;
    for shot in 0..shots {
        let mut config_shot = base_config.clone();
        config_shot.seed = Some(shot as u64);
        let result = backend.execute(circuit, params, &config_shot)?;
        let sv = match &result {
            ExecResult::Statevector(sv) => sv,
            _ => unreachable!(),
        };
        sum += expectation_from_sv(sv, n, observable);
    }
    Ok(sum / shots as f64)
}

fn stochastic_slot_two_term(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    slot: SlotRef,
    v: f64,
    shots: u32,
) -> Result<f64> {
    let shift = std::f64::consts::FRAC_PI_2;
    let c_plus = shifted_circuit(circuit, slot, v + shift);
    let c_minus = shifted_circuit(circuit, slot, v - shift);
    let e_plus = stochastic_expectation_avg(backend, &c_plus, params, observable, shots)?;
    let e_minus = stochastic_expectation_avg(backend, &c_minus, params, observable, shots)?;
    Ok((e_plus - e_minus) / 2.0)
}

fn stochastic_slot_four_term(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    slot: SlotRef,
    v: f64,
    shots: u32,
) -> Result<f64> {
    let pi_2 = std::f64::consts::FRAC_PI_2;
    let three_pi_2 = 3.0 * pi_2;
    let (c1, c2) = banchi_crooks_two_freq_coeffs();
    let c_p1 = shifted_circuit(circuit, slot, v + pi_2);
    let c_m1 = shifted_circuit(circuit, slot, v - pi_2);
    let c_p3 = shifted_circuit(circuit, slot, v + three_pi_2);
    let c_m3 = shifted_circuit(circuit, slot, v - three_pi_2);
    let e_p1 = stochastic_expectation_avg(backend, &c_p1, params, observable, shots)?;
    let e_m1 = stochastic_expectation_avg(backend, &c_m1, params, observable, shots)?;
    let e_p3 = stochastic_expectation_avg(backend, &c_p3, params, observable, shots)?;
    let e_m3 = stochastic_expectation_avg(backend, &c_m3, params, observable, shots)?;
    Ok(c1 * (e_p1 - e_m1) + c2 * (e_m3 - e_p3))
}

#[allow(clippy::too_many_arguments)]
fn stochastic_slot_four_term_givens(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    slot: SlotRef,
    v: f64,
    shots: u32,
) -> Result<f64> {
    let pi_4 = std::f64::consts::FRAC_PI_4;
    let pi_2 = std::f64::consts::FRAC_PI_2;
    let c2 = givens_two_freq_coeff();
    let e_p1 = stochastic_expectation_avg(
        backend,
        &shifted_circuit(circuit, slot, v + pi_4),
        params,
        observable,
        shots,
    )?;
    let e_m1 = stochastic_expectation_avg(
        backend,
        &shifted_circuit(circuit, slot, v - pi_4),
        params,
        observable,
        shots,
    )?;
    let e_p2 = stochastic_expectation_avg(
        backend,
        &shifted_circuit(circuit, slot, v + pi_2),
        params,
        observable,
        shots,
    )?;
    let e_m2 = stochastic_expectation_avg(
        backend,
        &shifted_circuit(circuit, slot, v - pi_2),
        params,
        observable,
        shots,
    )?;
    Ok((e_p1 - e_m1) - c2 * (e_p2 - e_m2))
}

/// Compute <ψ|O|ψ> from a statevector and observable (for stochastic gradient use).
fn expectation_from_sv(
    sv: &[num_complex::Complex64],
    num_qubits: u32,
    observable: &Observable,
) -> f64 {
    use crate::executor::PauliOp;
    use num_complex::Complex64;

    let n = num_qubits as usize;
    let dim = 1usize << n;
    let mut total = 0.0;

    for (coeff, pauli_string) in &observable.terms {
        let mut result = Complex64::new(0.0, 0.0);
        for i in 0..dim {
            let mut j = i;
            let mut c = Complex64::new(1.0, 0.0);
            for (qubit, pauli) in pauli_string {
                let q = *qubit as usize;
                let bit = (i >> q) & 1;
                match pauli {
                    PauliOp::I => {}
                    PauliOp::Z => {
                        if bit == 1 {
                            c *= -1.0;
                        }
                    }
                    PauliOp::X => {
                        j ^= 1 << q;
                    }
                    PauliOp::Y => {
                        j ^= 1 << q;
                        if bit == 0 {
                            c *= Complex64::new(0.0, 1.0);
                        } else {
                            c *= Complex64::new(0.0, -1.0);
                        }
                    }
                }
            }
            result += sv[i].conj() * c * sv[j];
        }
        total += coeff * result.re;
    }
    total
}

/// Find all symbol IDs that actually appear in the circuit's gate parameters.
fn find_active_symbols(circuit: &CircuitIR) -> Vec<SymbolId> {
    let mut active = Vec::new();
    for op in &circuit.ops {
        for param in &op.params {
            collect_symbols(param, &mut active);
        }
    }
    active.sort();
    active.dedup();
    active
}

fn collect_symbols(expr: &ParamExpr, out: &mut Vec<SymbolId>) {
    match expr {
        ParamExpr::Symbol(id) => out.push(*id),
        ParamExpr::Negate(inner) => collect_symbols(inner, out),
        ParamExpr::Add(a, b) | ParamExpr::Mul(a, b) => {
            collect_symbols(a, out);
            collect_symbols(b, out);
        }
        ParamExpr::Concrete(_) => {}
    }
}

// Backend-touching tests live in omega-backend-statevector to avoid a
// circular dev-dependency. Pure parsers + Functional helpers are
// self-contained and tested below.

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Functional::evaluate / num_qubits ----

    #[test]
    fn functional_qubo_num_qubits_matches_inner_n() {
        let q = Qubo::new(5);
        let f = Functional::Qubo(q);
        assert_eq!(f.num_qubits(), 5);
    }

    #[test]
    fn functional_table_num_qubits_returns_field() {
        let f = Functional::Table {
            num_qubits: 3,
            rows: vec![],
        };
        assert_eq!(f.num_qubits(), 3);
    }

    #[test]
    fn functional_qubo_evaluate_matches_qubo_evaluate() {
        // f(x) = -x_0 - x_1 + 2 x_0 x_1.
        let mut q = Qubo::new(2);
        q.set(0, 0, -1.0);
        q.set(1, 1, -1.0);
        q.set(0, 1, 2.0);
        let f = Functional::Qubo(q);
        assert_eq!(f.evaluate(&[false, false]), 0.0);
        assert_eq!(f.evaluate(&[true, false]), -1.0);
        assert_eq!(f.evaluate(&[false, true]), -1.0);
        assert_eq!(f.evaluate(&[true, true]), 0.0);
    }

    #[test]
    fn functional_table_evaluate_matches_listed_row() {
        // Two-bit table: {(0,1) → 7, (1,0) → -3}; everything else → 0.
        let f = Functional::Table {
            num_qubits: 2,
            rows: vec![(vec![false, true], 7.0), (vec![true, false], -3.0)],
        };
        assert_eq!(f.evaluate(&[false, true]), 7.0);
        assert_eq!(f.evaluate(&[true, false]), -3.0);
        // Unlisted bit-vectors default to 0 (documented contract).
        assert_eq!(f.evaluate(&[false, false]), 0.0);
        assert_eq!(f.evaluate(&[true, true]), 0.0);
    }

    // ---- to_diagonal_observable invariants ----

    #[test]
    fn functional_table_observable_has_at_most_2n_terms() {
        let f = Functional::Table {
            num_qubits: 2,
            rows: vec![(vec![true, true], 1.0)],
        };
        let obs = f.to_diagonal_observable();
        assert!(obs.terms.len() <= 4, "n=2 → at most 2^n = 4 terms");
        // Single |11⟩ → 1 means O = (1/4)(I - Z₀ - Z₁ + Z₀Z₁). Pin
        // each coefficient.
        let mut coeffs: std::collections::HashMap<Vec<u32>, f64> = std::collections::HashMap::new();
        for (c, p) in &obs.terms {
            let key: Vec<u32> = p.iter().map(|(q, _)| *q).collect();
            coeffs.insert(key, *c);
        }
        assert!((coeffs[&vec![]] - 0.25).abs() < 1e-12);
        assert!((coeffs[&vec![0]] - -0.25).abs() < 1e-12);
        assert!((coeffs[&vec![1]] - -0.25).abs() < 1e-12);
        assert!((coeffs[&vec![0, 1]] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn functional_table_zero_functional_emits_trivial_observable() {
        // Empty rows → all f(x) = 0. Documented to return a single
        // identity·0 term so downstream code doesn't divide by zero
        // or hit an empty-terms branch.
        let f = Functional::Table {
            num_qubits: 2,
            rows: vec![],
        };
        let obs = f.to_diagonal_observable();
        assert_eq!(obs.terms.len(), 1);
        assert_eq!(obs.terms[0].0, 0.0);
        assert!(obs.terms[0].1.is_empty());
    }

    // ---- parse_functional_spec ----

    #[test]
    fn parse_qubo_spec_round_trips_into_qubo() {
        let spec = r#"{"qubo":{"n":2,"Q":[[0,0,-1.0],[0,1,2.0]]}}"#;
        let (f, kind) = parse_functional_spec(spec).unwrap();
        assert_eq!(kind, "qubo");
        let Functional::Qubo(q) = f else {
            panic!("expected Qubo variant");
        };
        assert_eq!(q.n, 2);
        assert_eq!(q.entries.get(&(0, 0)).copied(), Some(-1.0));
        assert_eq!(q.entries.get(&(0, 1)).copied(), Some(2.0));
    }

    #[test]
    fn parse_table_spec_decodes_each_row() {
        let spec = r#"{"table":[["00",0.0],["10",1.5],["11",-2.0]],"num_qubits":2}"#;
        let (f, kind) = parse_functional_spec(spec).unwrap();
        assert_eq!(kind, "table");
        let Functional::Table { num_qubits, rows } = f else {
            panic!("expected Table variant");
        };
        assert_eq!(num_qubits, 2);
        assert_eq!(rows.len(), 3);
        // Row 0 was "00" → both bits false.
        assert_eq!(rows[0].0, vec![false, false]);
        assert_eq!(rows[0].1, 0.0);
        // Row 1 was "10" → bit 0 = true, bit 1 = false.
        assert_eq!(rows[1].0, vec![true, false]);
        assert_eq!(rows[1].1, 1.5);
        // Row 2 was "11" → both bits true.
        assert_eq!(rows[2].0, vec![true, true]);
        assert_eq!(rows[2].1, -2.0);
    }

    #[test]
    fn parse_functional_spec_rejects_invalid_json() {
        let err = parse_functional_spec("not json").expect_err("invalid json");
        assert!(err.contains("invalid JSON"), "msg: {err}");
    }

    #[test]
    fn parse_functional_spec_rejects_unknown_top_key() {
        let err = parse_functional_spec(r#"{"unknown":{}}"#).expect_err("no qubo/table");
        assert!(err.contains("`qubo` or `table`"), "msg: {err}");
    }

    #[test]
    fn parse_table_spec_requires_num_qubits() {
        let err = parse_functional_spec(r#"{"table":[]}"#).expect_err("missing num_qubits");
        assert!(err.contains("num_qubits"), "msg: {err}");
    }

    #[test]
    fn parse_table_spec_rejects_wrong_bit_length() {
        // num_qubits=3 but row has 2 bits.
        let spec = r#"{"table":[["10",1.0]],"num_qubits":3}"#;
        let err = parse_functional_spec(spec).expect_err("bit-length mismatch");
        assert!(err.contains("length"), "msg: {err}");
    }

    #[test]
    fn parse_table_spec_rejects_bad_bit_chars() {
        // Bits must be '0' or '1'; '?' is invalid.
        let spec = r#"{"table":[["1?",1.0]],"num_qubits":2}"#;
        let err = parse_functional_spec(spec).expect_err("bad bit char");
        assert!(err.contains("bit char"), "msg: {err}");
    }

    #[test]
    fn parse_table_spec_rejects_wrong_arity_row() {
        let spec = r#"{"table":[["10"]],"num_qubits":2}"#;
        let err = parse_functional_spec(spec).expect_err("arity");
        assert!(err.contains("[bits, value]"), "msg: {err}");
    }
}
