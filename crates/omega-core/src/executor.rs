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

/// A Pauli operator for defining observables.
#[derive(Clone, Debug, PartialEq, Eq)]
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
}
