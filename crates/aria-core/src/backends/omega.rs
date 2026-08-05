// SPDX-License-Identifier: Apache-2.0
//! Bridge from the Aria `Circuit` to the omega-functions wire IR.
//!
//! Converts the Aria AST into the JSON format expected by the
//! omega-functions execution runtime / omega-server HTTP bridge.
//!
//! Two deployment paths:
//! 1. Direct Rust: `to_omega_ir()` produces an `OmegaCircuitIR`
//! 2. QASM: `ast::to_qasm()` → omega-parser (already works)

use crate::ast::nodes::*;

/// Omega-compatible gate kind (mirrors omega_core::circuit::GateKind).
/// We define our own copy so aria-core has no hard dependency on omega-core.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OmegaGateKind {
    H,
    X,
    Y,
    Z,
    S,
    Sdg,
    T,
    Tdg,
    Id,
    Rx,
    Ry,
    Rz,
    U3,
    U2,
    U1,
    CX,
    CY,
    CZ,
    Swap,
    CRz,
    CU3,
    CCX,
    CSwap,
    PhaseShifter,
    BeamSplitterRx,
    Measure,
    Barrier,
    Reset,
}

/// A flattened gate operation for omega-functions.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OmegaGateOp {
    pub gate: OmegaGateKind,
    pub qubits: Vec<u32>,
    pub params: Vec<f64>,
    pub classical_bit: Option<u32>,
    /// Conditional execution: (classical_bit_index, expected_value).
    /// Gate only executes if classical_bits[bit] == value.
    pub condition: Option<(u32, u64)>,
}

/// Execution mode for mid-circuit measurements.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum OmegaMidCircuitMode {
    /// Skip measurements (backward compatible).
    Skip,
    /// Projective collapse on measurement.
    Collapse,
}

/// Execution backend to target inside omega-functions. Matches the
/// backend selection exposed by `omega-runtime` (statevector / MPS /
/// Clifford+stabilizer / photonic). `Auto` asks the runtime to pick the
/// most efficient backend the circuit is compatible with.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum OmegaBackend {
    #[default]
    Auto,
    Statevector,
    Mps {
        max_bond_dim: u32,
    },
    Stabilizer,
    Photonic,
}

/// Omega-compatible circuit IR.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OmegaCircuitIR {
    pub num_qubits: u32,
    pub num_classical_bits: u32,
    pub ops: Vec<OmegaGateOp>,
    pub is_photonic: bool,
    pub mid_circuit_mode: OmegaMidCircuitMode,
    /// Which omega backend should execute this IR. Defaults to `Auto`
    /// (runtime picks). Use `recommended_backend()` to preview the choice
    /// the bridge would make from the circuit shape.
    pub backend: OmegaBackend,
}

impl OmegaCircuitIR {
    /// Override the target backend.
    pub fn with_backend(mut self, backend: OmegaBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Return the backend the bridge would recommend given the circuit's
    /// shape. This is heuristic and deterministic; it does not execute.
    pub fn recommended_backend(&self) -> OmegaBackend {
        if self.is_photonic {
            return OmegaBackend::Photonic;
        }
        // If every gate is Clifford (H/X/Y/Z/S/CX/CZ/SWAP), prefer
        // stabilizer simulation — it's exponentially cheaper.
        let clifford_only = self.ops.iter().all(|op| {
            matches!(
                op.gate,
                OmegaGateKind::H
                    | OmegaGateKind::X
                    | OmegaGateKind::Y
                    | OmegaGateKind::Z
                    | OmegaGateKind::S
                    | OmegaGateKind::Sdg
                    | OmegaGateKind::CX
                    | OmegaGateKind::CY
                    | OmegaGateKind::CZ
                    | OmegaGateKind::Swap
                    | OmegaGateKind::Measure
                    | OmegaGateKind::Barrier
                    | OmegaGateKind::Reset
                    | OmegaGateKind::Id
            )
        });
        if clifford_only {
            return OmegaBackend::Stabilizer;
        }
        // Large qubit counts: switch to MPS. Omega's MPS backend handles
        // mid-circuit measure via `measure_site`, so even measurement-
        // heavy ansätze work.
        if self.num_qubits >= 20 {
            return OmegaBackend::Mps { max_bond_dim: 64 };
        }
        OmegaBackend::Statevector
    }
}

/// Convert a quantum-core GateKind to omega GateKind.
fn map_gate_kind(kind: GateKind) -> Option<OmegaGateKind> {
    match kind {
        GateKind::H => Some(OmegaGateKind::H),
        GateKind::X => Some(OmegaGateKind::X),
        GateKind::Y => Some(OmegaGateKind::Y),
        GateKind::Z => Some(OmegaGateKind::Z),
        GateKind::S => Some(OmegaGateKind::S),
        GateKind::Sdg => Some(OmegaGateKind::Sdg),
        GateKind::T => Some(OmegaGateKind::T),
        GateKind::Tdg => Some(OmegaGateKind::Tdg),
        GateKind::I => Some(OmegaGateKind::Id),
        GateKind::RX => Some(OmegaGateKind::Rx),
        GateKind::RY => Some(OmegaGateKind::Ry),
        GateKind::RZ => Some(OmegaGateKind::Rz),
        GateKind::U => Some(OmegaGateKind::U3),
        GateKind::P => Some(OmegaGateKind::U1),
        GateKind::CX => Some(OmegaGateKind::CX),
        GateKind::CY => Some(OmegaGateKind::CY),
        GateKind::CZ => Some(OmegaGateKind::CZ),
        GateKind::SWAP => Some(OmegaGateKind::Swap),
        // CP(λ) == controlled-U1(λ) == CU3(0, 0, λ); params synthesized below.
        GateKind::CP => Some(OmegaGateKind::CU3),
        GateKind::CCX => Some(OmegaGateKind::CCX),
        GateKind::CSWAP => Some(OmegaGateKind::CSwap),
        GateKind::Measure => Some(OmegaGateKind::Measure),
        GateKind::Barrier => Some(OmegaGateKind::Barrier),
        GateKind::Reset => Some(OmegaGateKind::Reset),
        GateKind::PhaseShifter => Some(OmegaGateKind::PhaseShifter),
        GateKind::BeamSplitter => Some(OmegaGateKind::BeamSplitterRx),
        _ => None,
    }
}

/// Convert a quantum-core Circuit to omega-functions CircuitIR.
///
/// Flattens register-based qubit addressing into global indices.
///
/// # Panics
///
/// Panics if `circuit` contains a gate with no omega IR equivalent — in
/// practice the continuous-variable photonic gates (`Squeezing`,
/// `Displacement`, `Kerr`), which no backend executes yet.
///
/// This infallible wrapper exists for callers whose circuits are gate-model
/// **by construction** and therefore cannot contain CV gates — the QEC bridge
/// (`aria_qec::ecc::run::to_omega_core_ir`) builds its circuits from an
/// encoder/decoder gate set that has no photonic member, so the panic is
/// unreachable there rather than merely unlikely.
///
/// Any caller that can be handed a *user-authored* circuit must use
/// [`try_to_omega_ir`] and surface the error. Dropping an unmapped gate and
/// executing the remainder returns confident wrong numbers; that is exactly the
/// defect this split removes.
pub fn to_omega_ir(circuit: &Circuit) -> OmegaCircuitIR {
    try_to_omega_ir(circuit).expect("to_omega_ir: circuit contains a gate with no omega lowering")
}

/// Convert a quantum-core Circuit to omega-functions CircuitIR, refusing any
/// gate that has no omega IR equivalent.
///
/// Flattens register-based qubit addressing into global indices.
///
/// Prefer this over [`to_omega_ir`] wherever the circuit came from user input.
/// The continuous-variable gates parse and type-check (OPTICQASM emits and
/// accepts `squeeze` / `displace` / `kerr`) but have no executor, so lowering
/// must refuse them rather than silently produce a circuit missing its physics.
pub fn try_to_omega_ir(circuit: &Circuit) -> Result<OmegaCircuitIR, String> {
    // Build global qubit index map
    let mut qubit_map: std::collections::HashMap<Qubit, u32> = std::collections::HashMap::new();
    let mut global_idx = 0u32;
    for reg in &circuit.registers {
        if reg.kind == RegisterKind::Quantum {
            for i in 0..reg.size {
                qubit_map.insert(Qubit::new(&reg.name, i), global_idx);
                global_idx += 1;
            }
        }
    }

    // Build classical bit map
    let mut clbit_map: std::collections::HashMap<Clbit, u32> = std::collections::HashMap::new();
    let mut cl_idx = 0u32;
    for reg in &circuit.registers {
        if reg.kind == RegisterKind::Classical {
            for i in 0..reg.size {
                clbit_map.insert(Clbit::new(&reg.name, i), cl_idx);
                cl_idx += 1;
            }
        }
    }

    let is_photonic = circuit.instructions.iter().any(|inst| {
        matches!(
            inst.gate.kind,
            GateKind::BeamSplitter
                | GateKind::PhaseShifter
                | GateKind::Squeezing
                | GateKind::Displacement
                | GateKind::Kerr
        )
    });

    let mut ops = Vec::new();
    for inst in &circuit.instructions {
        // An unmapped gate is a hard error, never a skip. `is_photonic` above
        // is computed from gates including Squeezing/Displacement/Kerr, so
        // dropping them here would mark the circuit photonic *because* of gates
        // that are no longer in it, and hand the backend a circuit whose
        // physics is missing — with no diagnostic anywhere.
        let Some(omega_gate) = map_gate_kind(inst.gate.kind) else {
            return Err(format!(
                "gate {:?} has no omega IR lowering, so this circuit cannot be executed \
                 (continuous-variable photonic gates — Squeezing, Displacement, Kerr — \
                 parse and export but have no backend yet). Refusing rather than \
                 executing the circuit without it. Discrete-variable photonics runs via \
                 OPTICQASM on `omega-run --backend photonics`.",
                inst.gate.kind
            ));
        };
        {
            let qubits: Vec<u32> = inst
                .qubits
                .iter()
                .map(|q| qubit_map.get(q).copied().unwrap_or(0))
                .collect();

            let classical_bit = inst.clbits.first().and_then(|c| clbit_map.get(c).copied());

            // Symbolic params get materialized as 0.0 placeholders; the
            // caller (e.g. SampleRequest / OmegaGradientRequest) supplies
            // the real bindings in `param_values`.
            let params_f64: Vec<f64> = if inst.gate.kind == GateKind::CP {
                // CP(λ) → CU3(0, 0, λ): controlled phase as controlled-U1.
                let lam = inst
                    .gate
                    .params
                    .first()
                    .and_then(|p| p.try_as_f64())
                    .unwrap_or(0.0);
                vec![0.0, 0.0, lam]
            } else {
                inst.gate
                    .params
                    .iter()
                    .map(|p| p.try_as_f64().unwrap_or(0.0))
                    .collect()
            };
            let condition = inst
                .condition
                .as_ref()
                .and_then(|(cl, v)| clbit_map.get(cl).copied().map(|idx| (idx, *v)));
            ops.push(OmegaGateOp {
                gate: omega_gate,
                qubits,
                params: params_f64,
                classical_bit,
                condition,
            });
        }
    }

    // Detect if circuit uses mid-circuit measurement
    let has_measure = circuit
        .instructions
        .iter()
        .any(|i| i.gate.kind == GateKind::Measure);
    let has_gates_after_measure = if has_measure {
        let measure_pos = circuit
            .instructions
            .iter()
            .position(|i| i.gate.kind == GateKind::Measure)
            .unwrap();
        circuit.instructions[measure_pos + 1..]
            .iter()
            .any(|i| !i.gate.kind.is_meta())
    } else {
        false
    };

    let mid_circuit_mode = if has_gates_after_measure {
        OmegaMidCircuitMode::Collapse
    } else {
        OmegaMidCircuitMode::Skip
    };

    Ok(OmegaCircuitIR {
        num_qubits: circuit.n_qubits() as u32,
        num_classical_bits: circuit.n_clbits() as u32,
        ops,
        mid_circuit_mode,
        is_photonic,
        backend: OmegaBackend::Auto,
    })
}

// ---------------------------------------------------------------------------
// Observable and gradient types (mirrors omega-core)
// ---------------------------------------------------------------------------

/// Pauli operator.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OmegaPauliOp {
    I,
    X,
    Y,
    Z,
}

/// Observable as a sum of weighted Pauli strings.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OmegaObservable {
    pub terms: Vec<(f64, Vec<(u32, OmegaPauliOp)>)>,
}

impl OmegaObservable {
    pub fn identity() -> Self {
        Self {
            terms: vec![(1.0, vec![])],
        }
    }
    pub fn z(qubit: u32) -> Self {
        Self {
            terms: vec![(1.0, vec![(qubit, OmegaPauliOp::Z)])],
        }
    }
    pub fn x(qubit: u32) -> Self {
        Self {
            terms: vec![(1.0, vec![(qubit, OmegaPauliOp::X)])],
        }
    }
    pub fn zz(q0: u32, q1: u32) -> Self {
        Self {
            terms: vec![(1.0, vec![(q0, OmegaPauliOp::Z), (q1, OmegaPauliOp::Z)])],
        }
    }
    pub fn xx(q0: u32, q1: u32) -> Self {
        Self {
            terms: vec![(1.0, vec![(q0, OmegaPauliOp::X), (q1, OmegaPauliOp::X)])],
        }
    }
    pub fn yy(q0: u32, q1: u32) -> Self {
        Self {
            terms: vec![(1.0, vec![(q0, OmegaPauliOp::Y), (q1, OmegaPauliOp::Y)])],
        }
    }
    pub fn scale(mut self, c: f64) -> Self {
        for term in &mut self.terms {
            term.0 *= c;
        }
        self
    }
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, other: Self) -> Self {
        self.terms.extend(other.terms);
        self
    }
}

/// Gradient method selection (mirrors omega-core).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum OmegaGradMethod {
    Auto,
    ParameterShift,
    FiniteDifference(f64),
    Adjoint,
    StochasticParameterShift { shots: u32 },
}

/// Gradient request: which circuit, parameters, observable, and method.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OmegaGradientRequest {
    pub circuit: OmegaCircuitIR,
    pub param_values: Vec<(String, f64)>, // (symbol_name, value)
    pub observable: OmegaObservable,
    pub method: OmegaGradMethod,
}

/// Serialize an `OmegaCircuitIR` to JSON. This is the wire format that
/// `omega-runtime`'s `execute_with_backend` endpoint consumes when the
/// two sides talk over an HTTP / socket boundary.
pub fn omega_ir_to_json(ir: &OmegaCircuitIR) -> Result<String, serde_json::Error> {
    serde_json::to_string(ir)
}

/// Deserialize an `OmegaCircuitIR` from the JSON wire format.
pub fn omega_ir_from_json(s: &str) -> Result<OmegaCircuitIR, serde_json::Error> {
    serde_json::from_str(s)
}

/// Serialize an `OmegaGradientRequest` to JSON (observable + method + circuit).
pub fn omega_gradient_request_to_json(
    req: &OmegaGradientRequest,
) -> Result<String, serde_json::Error> {
    serde_json::to_string(req)
}

/// Deserialize an `OmegaGradientRequest` from JSON.
pub fn omega_gradient_request_from_json(
    s: &str,
) -> Result<OmegaGradientRequest, serde_json::Error> {
    serde_json::from_str(s)
}

/// Build an H₂ Hamiltonian observable at bond length R=0.735 Å.
pub fn h2_observable() -> OmegaObservable {
    OmegaObservable::identity()
        .scale(-0.4804)
        .add(OmegaObservable::z(0).scale(0.3435))
        .add(OmegaObservable::z(1).scale(-0.4347))
        .add(OmegaObservable::zz(0, 1).scale(0.5716))
        .add(OmegaObservable::xx(0, 1).scale(0.0910))
        .add(OmegaObservable::yy(0, 1).scale(0.0910))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::CircuitBuilder;

    #[test]
    fn test_bell_to_omega() {
        let circ = CircuitBuilder::new("bell", 2, 2)
            .h(0)
            .cx(0, 1)
            .measure_all()
            .build();

        let omega = to_omega_ir(&circ);
        assert_eq!(omega.num_qubits, 2);
        assert_eq!(omega.num_classical_bits, 2);
        assert_eq!(omega.ops.len(), 4); // H, CX, M, M
        assert_eq!(omega.ops[0].gate, OmegaGateKind::H);
        assert_eq!(omega.ops[0].qubits, vec![0]);
        assert_eq!(omega.ops[1].gate, OmegaGateKind::CX);
        assert_eq!(omega.ops[1].qubits, vec![0, 1]);
        assert_eq!(omega.ops[2].gate, OmegaGateKind::Measure);
        assert_eq!(omega.ops[2].classical_bit, Some(0));
        assert!(!omega.is_photonic);
    }

    #[test]
    fn test_parametric_to_omega() {
        let circ = CircuitBuilder::new("test", 1, 0)
            .rx(0, 1.5)
            .rz(0, 0.7)
            .build();

        let omega = to_omega_ir(&circ);
        assert_eq!(omega.ops.len(), 2);
        assert_eq!(omega.ops[0].gate, OmegaGateKind::Rx);
        assert_eq!(omega.ops[0].params, vec![1.5]);
        assert_eq!(omega.ops[1].gate, OmegaGateKind::Rz);
        assert_eq!(omega.ops[1].params, vec![0.7]);
    }

    #[test]
    fn recommended_backend_is_stabilizer_for_pure_clifford() {
        // Bell state: H + CX only → Clifford → stabilizer backend.
        let circ = CircuitBuilder::new("bell", 2, 0).h(0).cx(0, 1).build();
        let omega = to_omega_ir(&circ);
        assert_eq!(omega.recommended_backend(), OmegaBackend::Stabilizer);
    }

    #[test]
    fn recommended_backend_is_statevector_for_small_nonclifford() {
        // T gate breaks Clifford; small qubit count → statevector.
        let circ = CircuitBuilder::new("t", 2, 0).h(0).t(0).cx(0, 1).build();
        let omega = to_omega_ir(&circ);
        assert_eq!(omega.recommended_backend(), OmegaBackend::Statevector);
    }

    #[test]
    fn recommended_backend_is_mps_for_large_nonclifford() {
        // 20-qubit T-containing circuit → MPS fallback.
        let mut b = CircuitBuilder::new("large", 20, 0);
        for q in 0..20 {
            b.h(q).t(q);
        }
        let circ = b.build();
        let omega = to_omega_ir(&circ);
        assert!(matches!(
            omega.recommended_backend(),
            OmegaBackend::Mps { max_bond_dim: 64 }
        ));
    }

    #[test]
    fn with_backend_overrides_choice() {
        let circ = CircuitBuilder::new("bell", 2, 0).h(0).cx(0, 1).build();
        let omega = to_omega_ir(&circ).with_backend(OmegaBackend::Mps { max_bond_dim: 32 });
        assert_eq!(omega.backend, OmegaBackend::Mps { max_bond_dim: 32 });
    }

    #[test]
    fn omega_ir_json_roundtrip_preserves_shape() {
        let circ = CircuitBuilder::new("bell", 2, 2)
            .h(0)
            .cx(0, 1)
            .measure_all()
            .build();
        let mut omega = to_omega_ir(&circ);
        omega.backend = OmegaBackend::Mps { max_bond_dim: 64 };

        let json = omega_ir_to_json(&omega).expect("serialize");
        assert!(json.contains("\"num_qubits\""));
        assert!(json.contains("\"Mps\""));

        let parsed = omega_ir_from_json(&json).expect("deserialize");
        assert_eq!(parsed.num_qubits, omega.num_qubits);
        assert_eq!(parsed.num_classical_bits, omega.num_classical_bits);
        assert_eq!(parsed.ops.len(), omega.ops.len());
        assert_eq!(parsed.is_photonic, omega.is_photonic);
        assert_eq!(parsed.mid_circuit_mode, omega.mid_circuit_mode);
        assert_eq!(parsed.backend, omega.backend);
        for (a, b) in parsed.ops.iter().zip(omega.ops.iter()) {
            assert_eq!(a.gate, b.gate);
            assert_eq!(a.qubits, b.qubits);
            assert_eq!(a.params, b.params);
            assert_eq!(a.classical_bit, b.classical_bit);
        }
    }

    #[test]
    fn conditional_gate_maps_clbit_through_bridge() {
        // teleport-style classically-controlled X/Z: m[0] drives an X on q[1].
        const SRC: &str = r#"
circuit C {
    qreg q[2]
    creg m[1]
    apply H on q[0]
    measure q[0] -> m[0]
    when m[0] == 1 { apply X on q[1] }
}
"#;
        let circ = crate::ast::aria::parse_aria_circuit(SRC, "C").expect("aria parse");
        let x_inst = circ
            .instructions
            .iter()
            .find(|i| i.gate.kind == GateKind::X)
            .expect("x gate");
        assert!(x_inst.condition.is_some(), "aria lowering set condition");

        let omega = to_omega_ir(&circ);
        let x_op = omega
            .ops
            .iter()
            .find(|o| o.gate == OmegaGateKind::X)
            .expect("x in omega ir");
        assert_eq!(x_op.condition, Some((0u32, 1u64)));
    }

    #[test]
    fn omega_gradient_request_json_roundtrip() {
        let circ = CircuitBuilder::new("rot", 1, 0).ry(0, 0.0).build();
        let req = OmegaGradientRequest {
            circuit: to_omega_ir(&circ),
            param_values: vec![("theta".into(), 0.5)],
            observable: OmegaObservable::z(0),
            method: OmegaGradMethod::StochasticParameterShift { shots: 256 },
        };
        let json = omega_gradient_request_to_json(&req).unwrap();
        let back = omega_gradient_request_from_json(&json).unwrap();
        assert_eq!(back.param_values, req.param_values);
        assert!(matches!(
            back.method,
            OmegaGradMethod::StochasticParameterShift { shots: 256 }
        ));
        assert_eq!(back.observable.terms.len(), 1);
    }

    /// Continuous-variable gates must be REFUSED by lowering, never dropped.
    ///
    /// Regression: `map_gate_kind` returns `None` for Squeezing/Displacement/
    /// Kerr, and the lowering loop used to be `if let Some(..)` — so a circuit
    /// using them lowered "successfully" with those gates deleted, and executed
    /// to confident wrong numbers. Worse, `is_photonic` is computed from the
    /// very same gate list, so the IR was marked photonic *because* of gates it
    /// no longer contained. Reached over the wire via `remote.rs`, this was a
    /// silent wrong answer on a remote server.
    #[test]
    fn cv_gates_are_refused_not_silently_dropped() {
        for kind in [GateKind::Squeezing, GateKind::Displacement, GateKind::Kerr] {
            let mut circ = Circuit::new("cv");
            circ.qreg("q", 1);
            circ.apply(
                GateDef::with_params(kind, vec![0.5, 0.0]),
                vec![Qubit::new("q", 0)],
            );

            let err = try_to_omega_ir(&circ).expect_err(&format!(
                "{kind:?} must be refused by lowering, not dropped"
            ));
            assert!(
                err.contains("no omega IR lowering"),
                "{kind:?}: unhelpful error {err:?}"
            );
        }
    }

    /// The gate-model path must be untouched by the refusal above: every gate
    /// in a normal circuit still lowers, and none are lost.
    #[test]
    fn gate_model_circuits_still_lower_completely() {
        let circ = CircuitBuilder::new("bell", 2, 2).h(0).cx(0, 1).build();
        let ir = try_to_omega_ir(&circ).expect("gate-model circuit must lower");
        assert_eq!(ir.ops.len(), circ.instructions.len());
        assert!(!ir.is_photonic);
    }
}
