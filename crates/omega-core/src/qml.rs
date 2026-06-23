//! QML (Quantum Machine Learning) inference + training pipeline.
//!
//! Provides data encoding strategies, model definition, inference, and a
//! training loop ([`QmlTrainer`]) for hybrid quantum-classical ML.
//! Supports mid-circuit measurements for surrogate-driven QML
//! approaches (arXiv:2505.05249).
//!
//! Pipeline: encode(data) → ansatz(params) → measure → extract features.
//!
//! Encoding strategies:
//! - Angle: Ry(x_i) on qubit i (one qubit per feature)
//! - IQP: H → Rz(x_i) → CZ entangling, repeated k layers
//!
//! Measurement modes:
//! - Expectation: compute ⟨Z_i⟩ on measurement qubits (exact, no collapse)
//! - Sample: mid-circuit collapse, extract classical bit outcomes
//!
//! Training:
//! - [`QmlTrainer`] — builder for an MSE-loss SGD trainer that drives
//!   the gradient loop through any [`Backend`]. The Metal backend's
//!   adjoint AD makes this end-to-end GPU when paired with
//!   `omega_backend_statevector_metal::MetalStatevectorBackend`.

use smallvec::smallvec;
use std::collections::HashMap;

use crate::circuit::*;
use crate::device::DeviceKind;
use crate::error::{OmegaError, Result};
use crate::executor::*;
use crate::params::ParameterBinding;

/// Data encoding strategy.
#[derive(Clone, Debug)]
pub enum Encoding {
    /// Ry(x_i) on qubit i. One qubit per feature.
    Angle,
    /// H → Rz(x_i) → CZ entangling, repeated `layers` times.
    Iqp { layers: usize },
}

/// How to extract output from the quantum circuit.
#[derive(Clone, Debug)]
pub enum OutputMode {
    /// Compute ⟨Z_i⟩ expectation values (exact, no mid-circuit measurement needed).
    Expectation,
    /// Run with mid-circuit measurements, average classical outcomes over `shots`.
    Sample { shots: u32 },
}

/// A hybrid quantum-classical model for inference.
#[derive(Clone, Debug)]
pub struct QmlModel {
    /// Number of qubits in the quantum circuit.
    pub num_qubits: u32,
    /// Data encoding strategy.
    pub encoding: Encoding,
    /// Parameterised quantum circuit (ansatz) appended after encoding.
    /// Can contain Measure gates for mid-circuit measurement.
    pub ansatz: CircuitIR,
    /// Which qubits to extract output from.
    pub measurement_qubits: Vec<u32>,
    /// How to extract output.
    pub output_mode: OutputMode,
}

/// Build an encoding circuit from classical data.
pub fn encode_data(data: &[f64], encoding: &Encoding) -> CircuitIR {
    match encoding {
        Encoding::Angle => encode_angle(data),
        Encoding::Iqp { layers } => encode_iqp(data, *layers),
    }
}

/// Angle encoding: Ry(x_i) on qubit i.
fn encode_angle(data: &[f64]) -> CircuitIR {
    let n = data.len();
    let mut ops = Vec::with_capacity(n);
    for (i, &x) in data.iter().enumerate() {
        ops.push(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(i as u32)],
            params: smallvec![ParamExpr::Concrete(x)],
            classical_bit: None,
            condition: None,
        });
    }
    CircuitIR {
        num_qubits: n as u32,
        num_classical_bits: 0,
        ops,
        circuit_type: CircuitType::GateBased,
        symbols: HashMap::new(),
        custom_gates: HashMap::new(),
    }
}

/// IQP encoding: H → Rz(x_i) → CZ entangling, repeated k layers.
fn encode_iqp(data: &[f64], layers: usize) -> CircuitIR {
    let n = data.len();
    let mut ops = Vec::new();

    for _ in 0..layers {
        // H on all qubits
        for i in 0..n {
            ops.push(GateOp {
                gate: GateKind::H,
                qubits: smallvec![Qubit(i as u32)],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        // Rz(x_i) on each qubit
        for (i, &x) in data.iter().enumerate() {
            ops.push(GateOp {
                gate: GateKind::Rz,
                qubits: smallvec![Qubit(i as u32)],
                params: smallvec![ParamExpr::Concrete(x)],
                classical_bit: None,
                condition: None,
            });
        }
        // CZ entangling: nearest-neighbour
        for i in 0..n.saturating_sub(1) {
            ops.push(GateOp {
                gate: GateKind::CZ,
                qubits: smallvec![Qubit(i as u32), Qubit((i + 1) as u32)],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
    }

    CircuitIR {
        num_qubits: n as u32,
        num_classical_bits: 0,
        ops,
        circuit_type: CircuitType::GateBased,
        symbols: HashMap::new(),
        custom_gates: HashMap::new(),
    }
}

/// Run QML inference: encode data, apply ansatz, extract output.
///
/// Returns a vector of output features (one per measurement qubit).
pub fn infer(
    model: &QmlModel,
    data: &[f64],
    params: &ParameterBinding,
    backend: &dyn Backend,
) -> Result<Vec<f64>> {
    // Build the full circuit: encoding + ansatz
    let encoding_circuit = encode_data(data, &model.encoding);
    let mut circuit = CircuitIR {
        num_qubits: model.num_qubits,
        num_classical_bits: model.ansatz.num_classical_bits,
        ops: Vec::new(),
        circuit_type: CircuitType::GateBased,
        symbols: model.ansatz.symbols.clone(),
        custom_gates: model.ansatz.custom_gates.clone(),
    };

    // Append encoding ops
    circuit.ops.extend(encoding_circuit.ops);
    // Append ansatz ops (may include Measure gates for mid-circuit measurement)
    circuit.ops.extend(model.ansatz.ops.iter().cloned());

    match &model.output_mode {
        OutputMode::Expectation => {
            infer_expectation(&circuit, params, backend, &model.measurement_qubits)
        }
        OutputMode::Sample { shots } => {
            infer_sample(&circuit, params, backend, &model.measurement_qubits, *shots)
        }
    }
}

/// Expectation-based output: compute ⟨Z_i⟩ for each measurement qubit.
fn infer_expectation(
    circuit: &CircuitIR,
    params: &ParameterBinding,
    backend: &dyn Backend,
    measurement_qubits: &[u32],
) -> Result<Vec<f64>> {
    // Batch the per-qubit ⟨Z_q⟩ requests into a single
    // `expectation_multi` call so backends that build a single forward
    // sweep can amortise it across all measurement qubits. The default
    // trait impl falls back to a per-observable loop, preserving the
    // pre-batch behaviour for backends that don't override.
    let observables: Vec<Observable> = measurement_qubits
        .iter()
        .map(|&q| Observable {
            terms: vec![(1.0, vec![(q, PauliOp::Z)])],
        })
        .collect();
    backend.expectation_multi(circuit, params, &observables)
}

/// Sample-based output: run circuit with mid-circuit measurements,
/// average outcomes.
///
/// Semantics: for each shot, sample the post-circuit qubit-basis state of
/// the measurement qubits and accumulate ⟨Z_q⟩ = (n0 - n1)/N. For
/// circuits with mid-circuit measurements we loop one trajectory per
/// shot (each `execute` collapses on its mid-circuit measures with a
/// fresh seed), then sample one basis state from the post-collapse
/// statevector. For pure-unitary circuits we run `execute` once and
/// sample N basis states from the analytic final state — same answer at
/// a fraction of the cost.
///
/// This is independent of how `ExecResult::Counts` keys its entries in
/// Collapse mode (creg state, for Qiskit alignment); ⟨Z⟩ is computed from
/// the qubit basis here, not from the creg key.
fn infer_sample(
    circuit: &CircuitIR,
    params: &ParameterBinding,
    backend: &dyn Backend,
    measurement_qubits: &[u32],
    shots: u32,
) -> Result<Vec<f64>> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let has_measure = circuit
        .ops
        .iter()
        .any(|op| matches!(op.gate, GateKind::Measure));

    let mut n0 = vec![0u32; measurement_qubits.len()];
    let mut n1 = vec![0u32; measurement_qubits.len()];

    let mut bump = |sampled: u64| {
        for (i, &q) in measurement_qubits.iter().enumerate() {
            if (sampled >> q) & 1 == 0 {
                n0[i] += 1;
            } else {
                n1[i] += 1;
            }
        }
    };

    let sample_one = |sv: &[num_complex::Complex64], rng: &mut StdRng| -> u64 {
        // Cumulative-probability sampling. Deterministic given `rng`.
        let mut acc = 0.0_f64;
        let r: f64 = rng.random();
        for (i, a) in sv.iter().enumerate() {
            acc += a.norm_sqr();
            if r <= acc {
                return i as u64;
            }
        }
        (sv.len().saturating_sub(1)) as u64
    };

    if !has_measure {
        // Pure unitary: one analytic forward sweep, sample shots times.
        let cfg = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend.execute(circuit, params, &cfg)?;
        let sv = match &result {
            ExecResult::Statevector(s) => s.clone(),
            _ => {
                return Err(OmegaError::Backend(
                    "infer_sample: backend returned non-statevector for shots:None".into(),
                ));
            }
        };
        let mut rng = StdRng::seed_from_u64(0x00C0_FFEE_BEEF);
        for _ in 0..shots {
            bump(sample_one(&sv, &mut rng));
        }
    } else {
        // Mid-circuit measurement present: per-shot trajectory with Collapse.
        let base_seed: u64 = 0x00C0_FFEE_BEEF;
        for s in 0..shots as u64 {
            let traj_seed = base_seed.wrapping_add(s).wrapping_mul(2654435761);
            let cfg = ExecConfig {
                shots: None,
                seed: Some(traj_seed),
                mid_circuit_mode: MidCircuitMode::Collapse,
            };
            let result = backend.execute(circuit, params, &cfg)?;
            let sv = match &result {
                ExecResult::Statevector(s) => s.clone(),
                _ => {
                    return Err(OmegaError::Backend(
                        "infer_sample: backend returned non-statevector for shots:None".into(),
                    ));
                }
            };
            // The collapse already pinned every measured qubit; sampling
            // here resolves only the residual unmeasured-qubit
            // superposition (e.g. the post-reset qubit after Ry(θ)).
            let mut rng = StdRng::seed_from_u64(traj_seed.wrapping_add(0xA5A5));
            bump(sample_one(&sv, &mut rng));
        }
    }

    let total = shots as f64;
    if total == 0.0 {
        return Ok(vec![0.0; measurement_qubits.len()]);
    }
    Ok((0..measurement_qubits.len())
        .map(|i| (n0[i] as f64 - n1[i] as f64) / total)
        .collect())
}

// -----------------------------------------------------------------------
// Training
// -----------------------------------------------------------------------

/// Recognise the GPU-backend "I refused to allocate a buffer" error
/// shape so the trainer can swap to CPU and keep going. Both
/// `MetalError::AllocationRefused` and `CudaError::AllocationRefused`
/// flatten into `OmegaError::Backend("... allocation refused: ...")`
/// via their `From<MetalError|CudaError> for OmegaError` impls — we
/// match on the lower-cased substring rather than typed variants
/// because `omega-core` can't depend on the GPU crates without a
/// circular dep. The defensive "out of memory" branch catches future
/// driver-pass-through error strings that may surface via
/// `CudaError::Driver(...)` etc. once we wire OOM detection deeper
/// into the kernel-launch paths.
fn is_allocation_refused(err: &OmegaError) -> bool {
    let OmegaError::Backend(s) = err else {
        return false;
    };
    let lower = s.to_lowercase();
    lower.contains("allocation refused") || lower.contains("out of memory")
}

/// History returned by [`QmlTrainer::fit`].
#[derive(Clone, Debug, Default)]
pub struct TrainHistory {
    /// Mean MSE loss per epoch (averaged across training points).
    pub loss_per_epoch: Vec<f64>,
}

/// MSE-loss SGD trainer for a [`QmlModel`].
///
/// Gradient computation goes through `Backend::adjoint_gradient`, so
/// the training loop runs on whatever device the backend implements —
/// CPU via `StatevectorBackend`, GPU via
/// `omega_backend_statevector_metal::MetalStatevectorBackend`.
///
/// ```ignore
/// let trainer = QmlTrainer::new(&model)
///     .epochs(100)
///     .learning_rate(0.05)
///     .seed(42);
/// let history = trainer.fit(&backend, &mut params, &train_x, &train_y)?;
/// ```
#[derive(Clone, Debug)]
pub struct QmlTrainer<'a> {
    model: &'a QmlModel,
    epochs: usize,
    learning_rate: f64,
    seed: Option<u64>,
    /// When true, run `omega_core::optimize::optimize` on every
    /// per-training-point circuit (encoding ++ ansatz) before
    /// dispatch. Folds `Ry(x_q)` (encoding) + `Ry(theta_q)`
    /// (ansatz layer 1) into a single `Ry(x_q + theta_q)` via
    /// `ParamExpr::Add` — correctness-preserving (the adjoint
    /// chain-rules through Add) and pinned by
    /// `test_qml_trainer_optimize_matches_unoptimized`.
    ///
    /// **Defaults to `false`** — per-call optimise overhead at the
    /// 50-op HEA shape exceeds the saved kernel launches on the
    /// CUDA Phase 4c bench (14q / 16-param / 256 pts × 100 ep:
    /// optimize=on gives 16.77 s vs optimize=off's 16.11 s,
    /// p<0.05). Keep it opt-in for callers who care about
    /// circuit-shape compactness over per-step wallclock; the
    /// real Phase 4c lever is CUDA Graphs (capture+replay), not
    /// rotation merging.
    optimize: bool,
    /// Optional explicit compute-device pick. When `Some(kind)`,
    /// `fit()` checks `backend.device() == kind` and refuses to run
    /// if the caller passes a backend that doesn't match. `None`
    /// (the default) keeps the legacy "trust the backend you pass"
    /// behaviour. Resolved against `DeviceKind::resolve` so an
    /// unavailable requested device falls back to `Cpu` with the
    /// standard stderr notice.
    requested_device: Option<DeviceKind>,
    /// When true (the default), an `AllocationRefused`-shaped error
    /// from `backend.expectation_multi_then_gradient` at training
    /// time switches the rest of the loop onto whatever
    /// `backend.cpu_fallback()` returns (the CPU `StatevectorBackend`
    /// for our GPU crates). A single stderr notice fires on the
    /// switch. Set to `false` via `.cpu_fallback_on_oom(false)` to
    /// surface the OOM as a hard error (useful for benchmark runs
    /// where silent CPU fallback would skew timings).
    cpu_fallback_on_oom: bool,
}

impl<'a> QmlTrainer<'a> {
    pub fn new(model: &'a QmlModel) -> Self {
        Self {
            model,
            epochs: 100,
            learning_rate: 0.05,
            seed: None,
            optimize: false,
            requested_device: None,
            cpu_fallback_on_oom: true,
        }
    }

    pub fn epochs(mut self, n: usize) -> Self {
        self.epochs = n;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Toggle the per-point CircuitIR optimisation pass (default
    /// off — see the field doc for why). Pin via
    /// `test_qml_trainer_optimize_matches_unoptimized`: on/off
    /// must produce identical loss curves and final parameters
    /// to f64 precision.
    pub fn optimize(mut self, enable: bool) -> Self {
        self.optimize = enable;
        self
    }

    /// Pin the compute device the trainer expects to run on. The
    /// requested device is resolved through `DeviceKind::resolve` —
    /// if the feature flag isn't compiled in, the resolved device
    /// is `Cpu` and a single stderr notice fires. At `fit()` time
    /// the trainer verifies `backend.device()` matches the resolved
    /// device and returns `OmegaError::Unsupported` on mismatch, so
    /// callers can't accidentally hand the Metal trainer a CPU
    /// backend (or vice versa).
    ///
    /// Leaving this unset (the default) preserves the legacy
    /// behaviour where `fit()` trusts whatever backend you pass.
    pub fn device(mut self, kind: DeviceKind) -> Self {
        self.requested_device = Some(DeviceKind::resolve(Some(kind)));
        self
    }

    /// Toggle the trainer's CPU-fallback-on-GPU-OOM behaviour. With
    /// the default `true`, an allocation-refused error from the
    /// gradient call switches the rest of training onto the backend
    /// returned by `backend.cpu_fallback()` (and emits one stderr
    /// notice). Set to `false` to surface OOM as a hard error
    /// instead — useful for benches where silent CPU rescue would
    /// skew the timing or hide a regression. Non-OOM errors always
    /// surface as hard errors regardless of this setting.
    pub fn cpu_fallback_on_oom(mut self, enable: bool) -> Self {
        self.cpu_fallback_on_oom = enable;
        self
    }

    /// Run the SGD training loop. MSE loss against `train_y` labels;
    /// gradient via `Backend::adjoint_gradient`. `params` is updated
    /// in place with the optimised parameter values.
    ///
    /// Currently only `OutputMode::Expectation` is supported for
    /// training — the adjoint-AD gradient lives in the expectation
    /// path. `OutputMode::Sample` returns `OmegaError::Unsupported`.
    pub fn fit(
        &self,
        backend: &dyn Backend,
        params: &mut ParameterBinding,
        train_x: &[Vec<f64>],
        train_y: &[Vec<f64>],
    ) -> Result<TrainHistory> {
        if train_x.len() != train_y.len() {
            return Err(OmegaError::InvalidCircuit(format!(
                "train_x ({}) and train_y ({}) length mismatch",
                train_x.len(),
                train_y.len()
            )));
        }
        if train_x.is_empty() {
            return Err(OmegaError::InvalidCircuit("empty training set".into()));
        }
        if !matches!(self.model.output_mode, OutputMode::Expectation) {
            return Err(OmegaError::Unsupported(
                "QmlTrainer.fit currently supports OutputMode::Expectation only".into(),
            ));
        }
        if let Some(want) = self.requested_device {
            let got = backend.device();
            if got != want {
                return Err(OmegaError::Unsupported(format!(
                    "QmlTrainer.device requested `{}` but backend `{}` reports `{}`",
                    want.name(),
                    backend.name(),
                    got.name(),
                )));
            }
        }

        let n_train = train_x.len() as f64;
        let n_outputs = self.model.measurement_qubits.len();
        let mut history = TrainHistory::default();

        // CPU fallback slot — populated lazily on the first OOM. Once
        // set, every subsequent `expectation_multi_then_gradient`
        // call routes through `fallback.as_ref().unwrap()` instead of
        // the caller-supplied `backend`. We do not flip back to the
        // original backend even if a later call would have fit — the
        // trainer assumes the OOM signal is sticky (memory pressure
        // doesn't fluctuate within a single training run) and a
        // half-GPU-half-CPU loop would muddy any wall-clock
        // comparison the caller is doing.
        let mut fallback: Option<Box<dyn Backend>> = None;

        for _epoch in 0..self.epochs {
            let mut epoch_loss = 0.0;
            let mut epoch_grad: HashMap<SymbolId, f64> = HashMap::new();

            for (x, y) in train_x.iter().zip(train_y.iter()) {
                if y.len() != n_outputs {
                    return Err(OmegaError::InvalidCircuit(format!(
                        "label vector length {} != measurement_qubits {}",
                        y.len(),
                        n_outputs
                    )));
                }
                // Build the per-point circuit (encode + ansatz). Encoding
                // is data-dependent so we rebuild each pass — cheap
                // compared to the gate-apply phase.
                let circuit = self.build_per_point_circuit(x);

                // Per-measurement-qubit ⟨Z_q⟩ as observables for the
                // prediction step.
                let pred_observables: Vec<Observable> = self
                    .model
                    .measurement_qubits
                    .iter()
                    .map(|&q| Observable {
                        terms: vec![(1.0, vec![(q, PauliOp::Z)])],
                    })
                    .collect();

                // Build the residual-weighted observable from
                // predictions inside the closure — the Metal backend's
                // override fuses the prediction sweep with the
                // gradient backward sweep, eliminating one full
                // forward pass per training point. CPU/MPS/Pauli use
                // the default trait impl which calls
                // `expectation_multi` + `adjoint_gradient`
                // separately — semantically identical.
                let measurement_qubits = self.model.measurement_qubits.clone();
                let y_clone = y.clone();
                let make_factory = || -> GradientObservableFactory<'_> {
                    let mqs = measurement_qubits.clone();
                    let yc = y_clone.clone();
                    Box::new(move |y_hat: &[f64]| {
                        let mut obs_terms: Vec<(f64, Vec<(u32, PauliOp)>)> =
                            Vec::with_capacity(mqs.len());
                        for (i, &q) in mqs.iter().enumerate() {
                            let r = y_hat[i] - yc[i];
                            let coeff = 2.0 * r;
                            if coeff != 0.0 {
                                obs_terms.push((coeff, vec![(q, PauliOp::Z)]));
                            }
                        }
                        Observable { terms: obs_terms }
                    })
                };

                let active: &dyn Backend = match fallback.as_deref() {
                    Some(fb) => fb,
                    None => backend,
                };
                let attempt = active.expectation_multi_then_gradient(
                    &circuit,
                    params,
                    &pred_observables,
                    make_factory(),
                );

                let (y_hat, gradient) = match attempt {
                    Ok(p) => p,
                    Err(err)
                        if self.cpu_fallback_on_oom
                            && fallback.is_none()
                            && is_allocation_refused(&err) =>
                    {
                        let fb = match backend.cpu_fallback() {
                            Some(b) => b,
                            None => return Err(err),
                        };
                        eprintln!(
                            "omega: backend `{}` refused an allocation at training time ({}); \
                             falling back to cpu backend `{}` for the rest of fit()",
                            backend.name(),
                            err,
                            fb.name(),
                        );
                        fallback = Some(fb);
                        fallback
                            .as_deref()
                            .expect("fallback just set")
                            .expectation_multi_then_gradient(
                                &circuit,
                                params,
                                &pred_observables,
                                make_factory(),
                            )?
                    }
                    Err(e) => return Err(e),
                };

                // MSE loss for this point.
                let residual: Vec<f64> = y_hat.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
                epoch_loss += residual.iter().map(|r| r * r).sum::<f64>();

                // Empty residual ⇒ perfect prediction ⇒ no gradient.
                let all_zero = residual.iter().all(|&r| r == 0.0);
                if all_zero {
                    continue;
                }

                let active_name = match fallback.as_deref() {
                    Some(fb) => fb.name().to_string(),
                    None => backend.name().to_string(),
                };
                let grads = match gradient {
                    Some(g) => g,
                    None => {
                        return Err(OmegaError::Unsupported(format!(
                            "backend `{active_name}` does not implement adjoint_gradient"
                        )));
                    }
                };
                for (sym, g) in grads {
                    *epoch_grad.entry(sym).or_insert(0.0) += g;
                }
            }

            // SGD step: θ ← θ − lr · (Σ grads) / n_train.
            for (&sym, &grad) in &epoch_grad {
                let current = params.resolve(&ParamExpr::Symbol(sym))?;
                let new = current - self.learning_rate * (grad / n_train);
                params.bind(sym, new);
            }

            history.loss_per_epoch.push(epoch_loss / n_train);
        }

        Ok(history)
    }

    /// Compose the encoding circuit for a single training point with
    /// the model's parametric ansatz, producing one `CircuitIR` whose
    /// gate list is `encode(x) ++ ansatz`.
    fn build_per_point_circuit(&self, x: &[f64]) -> CircuitIR {
        let encoding = encode_data(x, &self.model.encoding);
        let mut circuit = CircuitIR {
            num_qubits: self.model.num_qubits,
            num_classical_bits: self.model.ansatz.num_classical_bits,
            ops: Vec::with_capacity(encoding.ops.len() + self.model.ansatz.ops.len()),
            circuit_type: CircuitType::GateBased,
            symbols: self.model.ansatz.symbols.clone(),
            custom_gates: self.model.ansatz.custom_gates.clone(),
        };
        circuit.ops.extend(encoding.ops);
        circuit.ops.extend(self.model.ansatz.ops.iter().cloned());
        if self.optimize {
            // Folds Ry(x_q) ++ Ry(theta_q) → Ry(x_q + theta_q) via
            // the rotation-merging pass, dropping one kernel launch
            // per qubit per sweep on the QML hot path. Correctness-
            // preserving — the adjoint already chain-rules through
            // ParamExpr::Add.
            crate::optimize::optimize(&mut circuit);
        }
        circuit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle_encoding() {
        let circuit = encode_data(&[0.5, 1.0, 1.5], &Encoding::Angle);
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.ops.len(), 3);
        for op in &circuit.ops {
            assert_eq!(op.gate, GateKind::Ry);
        }
        if let ParamExpr::Concrete(v) = &circuit.ops[0].params[0] {
            assert!((*v - 0.5).abs() < 1e-15);
        }
    }

    #[test]
    fn test_iqp_encoding() {
        let circuit = encode_data(&[0.5, 1.0], &Encoding::Iqp { layers: 1 });
        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.ops.len(), 5); // 2H + 2Rz + 1CZ
        assert_eq!(circuit.ops[0].gate, GateKind::H);
        assert_eq!(circuit.ops[4].gate, GateKind::CZ);
    }

    #[test]
    fn test_iqp_encoding_2_layers() {
        let circuit = encode_data(&[0.5, 1.0], &Encoding::Iqp { layers: 2 });
        assert_eq!(circuit.ops.len(), 10);
    }

    /// Minimal Backend stub that reports a configurable `DeviceKind`.
    /// Used to pin `QmlTrainer::device` validation regardless of which
    /// GPU features are compiled in.
    struct DeviceStubBackend {
        device: DeviceKind,
    }

    impl Backend for DeviceStubBackend {
        fn name(&self) -> &str {
            "device-stub"
        }
        fn device(&self) -> DeviceKind {
            self.device
        }
        fn execute(
            &self,
            _circuit: &CircuitIR,
            _params: &ParameterBinding,
            _config: &ExecConfig,
        ) -> Result<ExecResult> {
            Err(OmegaError::Unsupported("stub".into()))
        }
    }

    fn make_trivial_model() -> QmlModel {
        let mut ansatz = CircuitIR::new(1, CircuitType::GateBased);
        ansatz.symbols.insert(0, "theta".to_string());
        ansatz.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(0)],
            params: smallvec![ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });
        QmlModel {
            num_qubits: 1,
            encoding: Encoding::Angle,
            ansatz,
            measurement_qubits: vec![0],
            output_mode: OutputMode::Expectation,
        }
    }

    #[test]
    fn test_qml_trainer_device_mismatch_rejected() {
        // Stub reports Cuda; trainer.device(Cpu) → resolved Cpu;
        // mismatch path must error before any backend dispatch.
        let model = make_trivial_model();
        let backend = DeviceStubBackend {
            device: DeviceKind::Cuda,
        };
        let mut params = ParameterBinding::new();
        params.bind(0, 0.0);
        let trainer = QmlTrainer::new(&model).device(DeviceKind::Cpu);
        let err = trainer
            .fit(&backend, &mut params, &[vec![0.0]], &[vec![0.0]])
            .expect_err("mismatched device must error");
        assert!(
            matches!(err, OmegaError::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("cpu"),
            "msg should name requested device: {msg}"
        );
        assert!(
            msg.contains("cuda"),
            "msg should name backend device: {msg}"
        );
    }

    #[test]
    fn test_qml_trainer_device_default_unset_skips_validation() {
        // No .device() call → field stays None → fit() doesn't check
        // the backend's device. The stub's execute always errors, but
        // the mismatch path would error *before* dispatch with a
        // specific message; here the error must come from somewhere
        // else (or training proceeds far enough to hit the stub).
        // We pin the absence of the device-mismatch error message.
        let model = make_trivial_model();
        let backend = DeviceStubBackend {
            device: DeviceKind::Cuda,
        };
        let mut params = ParameterBinding::new();
        params.bind(0, 0.0);
        let trainer = QmlTrainer::new(&model);
        let err = trainer
            .fit(&backend, &mut params, &[vec![0.0]], &[vec![0.0]])
            .expect_err("stub backend has no expectation_multi_then_gradient");
        let msg = format!("{err}");
        assert!(
            !msg.contains("QmlTrainer.device requested"),
            "unset device must not produce mismatch error: {msg}"
        );
    }

    #[test]
    fn test_qml_trainer_device_unavailable_resolves_to_cpu() {
        // .device(Metal) on a build without the metal feature resolves
        // to Cpu (one stderr notice fires). A Cpu-reporting backend
        // then matches and the validation passes.
        let model = make_trivial_model();
        let backend = DeviceStubBackend {
            device: DeviceKind::Cpu,
        };
        let mut params = ParameterBinding::new();
        params.bind(0, 0.0);
        // Only meaningful when metal isn't compiled in.
        if DeviceKind::Metal.is_available() {
            return;
        }
        let trainer = QmlTrainer::new(&model).device(DeviceKind::Metal);
        let err = trainer
            .fit(&backend, &mut params, &[vec![0.0]], &[vec![0.0]])
            .expect_err("stub fails at execute time, but device check should pass");
        let msg = format!("{err}");
        assert!(
            !msg.contains("QmlTrainer.device requested"),
            "fallback-to-cpu must not produce mismatch error: {msg}"
        );
    }

    // ---- CPU-fallback-on-OOM stubs and tests ----

    /// CPU stand-in used as the GPU stub's fallback. Returns a fixed
    /// `y_hat = 0.0` and a single-symbol gradient on sym 0 so the
    /// trainer's residual/grad math runs end-to-end without depending
    /// on a real CPU sim.
    struct CpuStubBackend;
    impl Backend for CpuStubBackend {
        fn name(&self) -> &str {
            "cpu-stub-fallback"
        }
        fn device(&self) -> DeviceKind {
            DeviceKind::Cpu
        }
        fn execute(
            &self,
            _circuit: &CircuitIR,
            _params: &ParameterBinding,
            _config: &ExecConfig,
        ) -> Result<ExecResult> {
            Err(OmegaError::Unsupported(
                "cpu-stub: execute not used by trainer test".into(),
            ))
        }
        fn expectation_multi_then_gradient(
            &self,
            _circuit: &CircuitIR,
            _params: &ParameterBinding,
            observables: &[Observable],
            _factory: GradientObservableFactory<'_>,
        ) -> Result<(Vec<f64>, Option<Vec<(SymbolId, f64)>>)> {
            // y_hat = 0 for every observable; constant grad 0.1 on
            // sym=0 — enough for the trivial single-symbol model in
            // these tests to take a real SGD step each epoch.
            Ok((vec![0.0; observables.len()], Some(vec![(0u32, 0.1)])))
        }
    }

    /// Stub GPU backend whose gradient call always returns the
    /// canonical `MetalError::AllocationRefused`-flattened message.
    /// `with_fallback` controls whether its `cpu_fallback()` returns
    /// the CPU stub or `None`.
    struct OomingStubBackend {
        with_fallback: bool,
    }
    impl Backend for OomingStubBackend {
        fn name(&self) -> &str {
            "gpu-stub-ooming"
        }
        fn device(&self) -> DeviceKind {
            DeviceKind::Metal
        }
        fn execute(
            &self,
            _circuit: &CircuitIR,
            _params: &ParameterBinding,
            _config: &ExecConfig,
        ) -> Result<ExecResult> {
            Err(OmegaError::Unsupported(
                "ooming-stub: execute not used by trainer test".into(),
            ))
        }
        fn expectation_multi_then_gradient(
            &self,
            _circuit: &CircuitIR,
            _params: &ParameterBinding,
            _observables: &[Observable],
            _factory: GradientObservableFactory<'_>,
        ) -> Result<(Vec<f64>, Option<Vec<(SymbolId, f64)>>)> {
            Err(OmegaError::Backend(
                "Metal allocation refused: synthetic test OOM (num_qubits=42)".into(),
            ))
        }
        fn cpu_fallback(&self) -> Option<Box<dyn Backend>> {
            if self.with_fallback {
                Some(Box::new(CpuStubBackend))
            } else {
                None
            }
        }
    }

    /// Stub whose gradient call returns a non-OOM backend error. Its
    /// `cpu_fallback()` is wired so a false-positive fallback would
    /// rescue the test — we explicitly assert the fallback is *not*
    /// invoked.
    struct GenericFailingStub;
    impl Backend for GenericFailingStub {
        fn name(&self) -> &str {
            "gpu-stub-non-oom"
        }
        fn device(&self) -> DeviceKind {
            DeviceKind::Metal
        }
        fn execute(
            &self,
            _circuit: &CircuitIR,
            _params: &ParameterBinding,
            _config: &ExecConfig,
        ) -> Result<ExecResult> {
            Err(OmegaError::Unsupported("non-oom-stub: execute".into()))
        }
        fn expectation_multi_then_gradient(
            &self,
            _circuit: &CircuitIR,
            _params: &ParameterBinding,
            _observables: &[Observable],
            _factory: GradientObservableFactory<'_>,
        ) -> Result<(Vec<f64>, Option<Vec<(SymbolId, f64)>>)> {
            Err(OmegaError::Backend(
                "synthetic non-OOM kernel failure for test".into(),
            ))
        }
        fn cpu_fallback(&self) -> Option<Box<dyn Backend>> {
            Some(Box::new(CpuStubBackend))
        }
    }

    #[test]
    fn test_is_allocation_refused_recognises_known_messages() {
        assert!(is_allocation_refused(&OmegaError::Backend(
            "Metal allocation refused: cap exceeded (num_qubits=30)".into()
        )));
        assert!(is_allocation_refused(&OmegaError::Backend(
            "CUDA allocation refused: device OOM (num_qubits=27)".into()
        )));
        assert!(is_allocation_refused(&OmegaError::Backend(
            "cuBLAS error: out of memory while launching gemm".into()
        )));
        assert!(!is_allocation_refused(&OmegaError::Backend(
            "kernel compile failure: nvrtc returned 7".into()
        )));
        // Wrong variant — the trainer only cares about Backend(...).
        assert!(!is_allocation_refused(&OmegaError::Unsupported(
            "allocation refused".into()
        )));
    }

    #[test]
    fn test_qml_trainer_oom_triggers_cpu_fallback_by_default() {
        // GPU stub always OOMs; its cpu_fallback returns a CpuStub
        // that produces a fixed gradient. fit() must rescue silently
        // and complete every epoch.
        let model = make_trivial_model();
        let backend = OomingStubBackend {
            with_fallback: true,
        };
        let mut params = ParameterBinding::new();
        params.bind(0, 0.5);
        let history = QmlTrainer::new(&model)
            .epochs(2)
            .learning_rate(0.1)
            .fit(&backend, &mut params, &[vec![0.5]], &[vec![1.0]])
            .expect("OOM should be rescued by cpu fallback");
        assert_eq!(history.loss_per_epoch.len(), 2);
        // CpuStub returns y_hat=0, y=1.0 ⇒ MSE = 1.0 every epoch.
        for (i, &l) in history.loss_per_epoch.iter().enumerate() {
            assert!((l - 1.0).abs() < 1e-12, "epoch {i}: loss={l}");
        }
        // Param updated each epoch by lr·grad/n_train = 0.1·0.1/1 = 0.01.
        let final_val = params.resolve(&ParamExpr::Symbol(0)).unwrap();
        assert!(
            (final_val - 0.48).abs() < 1e-12,
            "expected param 0.48 after 2 epochs, got {final_val}"
        );
    }

    #[test]
    fn test_qml_trainer_oom_propagates_when_fallback_disabled() {
        // Same stub, but the trainer was told not to rescue. The OOM
        // must surface verbatim.
        let model = make_trivial_model();
        let backend = OomingStubBackend {
            with_fallback: true,
        };
        let mut params = ParameterBinding::new();
        params.bind(0, 0.0);
        let err = QmlTrainer::new(&model)
            .epochs(2)
            .cpu_fallback_on_oom(false)
            .fit(&backend, &mut params, &[vec![0.5]], &[vec![1.0]])
            .expect_err("disabled-fallback must propagate OOM");
        let msg = format!("{err}");
        assert!(
            msg.contains("allocation refused"),
            "expected raw OOM error, got: {msg}"
        );
    }

    #[test]
    fn test_qml_trainer_oom_propagates_when_no_fallback_available() {
        // OOM with no fallback backend ⇒ trainer can't rescue ⇒
        // error propagates even with the default `true` setting.
        let model = make_trivial_model();
        let backend = OomingStubBackend {
            with_fallback: false,
        };
        let mut params = ParameterBinding::new();
        params.bind(0, 0.0);
        let err = QmlTrainer::new(&model)
            .epochs(2)
            .fit(&backend, &mut params, &[vec![0.5]], &[vec![1.0]])
            .expect_err("missing fallback must propagate OOM");
        let msg = format!("{err}");
        assert!(
            msg.contains("allocation refused"),
            "expected raw OOM error, got: {msg}"
        );
    }

    #[test]
    fn test_qml_trainer_non_oom_error_does_not_trigger_fallback() {
        // Non-OOM backend error must not be rescued — would mask real
        // kernel/correctness failures.
        let model = make_trivial_model();
        let backend = GenericFailingStub;
        let mut params = ParameterBinding::new();
        params.bind(0, 0.0);
        let err = QmlTrainer::new(&model)
            .epochs(2)
            .fit(&backend, &mut params, &[vec![0.5]], &[vec![1.0]])
            .expect_err("non-OOM must propagate even with fallback wired");
        let msg = format!("{err}");
        assert!(
            msg.contains("synthetic non-OOM kernel failure"),
            "expected original error, got: {msg}"
        );
        assert!(
            !msg.contains("allocation refused"),
            "non-OOM rescue must not have run: {msg}"
        );
    }

    #[test]
    fn test_model_with_ansatz_structure() {
        let mut ansatz = CircuitIR::new(2, CircuitType::GateBased);
        ansatz.num_classical_bits = 1;
        ansatz.symbols.insert(0, "theta".to_string());
        ansatz.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(0)],
            params: smallvec![ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });
        ansatz.add_op(GateOp {
            gate: GateKind::Measure,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: Some(0),
            condition: None,
        });

        let model = QmlModel {
            num_qubits: 2,
            encoding: Encoding::Angle,
            ansatz,
            measurement_qubits: vec![0, 1],
            output_mode: OutputMode::Sample { shots: 100 },
        };

        assert_eq!(model.measurement_qubits.len(), 2);
        assert_eq!(model.num_qubits, 2);
    }
}
// Execution tests are in crates/omega-backend-statevector/tests/qml_integration.rs
