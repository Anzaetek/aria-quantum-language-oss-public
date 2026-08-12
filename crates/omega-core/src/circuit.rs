use smallvec::SmallVec;
use std::collections::HashMap;

/// A qubit (or photonic mode) identifier.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Qubit(pub u32);

/// Unique identifier for a symbol (parameter name).
pub type SymbolId = u32;

/// Unique identifier for a custom gate definition.
pub type CustomGateId = u32;

/// A symbolic or concrete parameter expression.
#[derive(Clone, Debug)]
pub enum ParamExpr {
    Concrete(f64),
    Symbol(SymbolId),
    Negate(Box<ParamExpr>),
    Add(Box<ParamExpr>, Box<ParamExpr>),
    Mul(Box<ParamExpr>, Box<ParamExpr>),
}

impl ParamExpr {
    pub fn is_concrete(&self) -> bool {
        matches!(self, ParamExpr::Concrete(_))
    }
}

/// Standard gate kinds plus photonic operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GateKind {
    // Single-qubit gates (no params)
    H,
    X,
    Y,
    Z,
    S,
    Sdg,
    T,
    Tdg,
    /// `√X` — Stim's `SQRT_X`, Qiskit's `SXGate`.
    ///
    /// **A first-class variant rather than an alias for
    /// `U3(π/2, −π/2, π/2)`, for two independent reasons.**
    ///
    /// 1. **It is CLIFFORD.** `sx·sx = X` exactly. Lowering it to `U3` throws
    ///    that away at the type level: `PauliBackend` rejects `U3` outright as
    ///    non-Clifford, so an all-Clifford `sx` circuit would be refused by the
    ///    stabilizer backend — precisely the engine such circuits belong on.
    /// 2. **The alias is off by a global phase**, `sx = e^{iπ/4}·U3(π/2, −π/2,
    ///    π/2)`; measured `|sx − U3| = 0.541`, `det(sx) = i` vs `det(U3) = 1`.
    ///    Invisible in counts and expectations, but a global factor on a
    ///    *sub-block* becomes a **relative** phase under control — the same
    ///    argument this repo already settled for the photonics `hwp` global
    ///    `i`, which moved 0.413 of probability in a 4-mode MZI.
    ///
    /// Clifford action (Stim's canonical tableau and an independent
    /// conjugation both agree): `X → +X`, `Y → +Z`, `Z → −Y`.
    Sx,
    /// `√X†` — Stim's `SQRT_X_DAG`, Qiskit's `SXdgGate`. See [`GateKind::Sx`].
    ///
    /// Clifford action: `X → +X`, `Y → −Z`, `Z → +Y`. Verified
    /// `sxdg = sx†` and `sx·sxdg = I` to 0.000e+00.
    Sxdg,
    Id,

    // Single-qubit rotation gates (1 param)
    Rx,
    Ry,
    Rz,

    // General single-qubit gate (3 params: theta, phi, lambda)
    U3,
    // 2-param variant (phi, lambda) with theta=pi/2
    U2,
    // 1-param variant (lambda)
    U1,

    // Two-qubit gates (no params)
    CX,
    CY,
    CZ,
    Swap,

    // Two-qubit gates with params
    CRz,
    CU3,
    /// Reconfigurable Beam Splitter / Givens rotation (1 param):
    /// `RBS(θ) = exp(−i·θ/2·(Y⊗X − X⊗Y))` — identity on {|00⟩, |11⟩},
    /// `[[cos θ, −sin θ], [sin θ, cos θ]]` on span{|01⟩, |10⟩}.
    /// Hamming-weight preserving; the primitive of butterfly / unary QML
    /// circuits (Kerenidis et al., arXiv:2606.03517).
    Rbs,

    // Three-qubit gates
    CCX,
    CSwap,

    // Photonic operations
    PhaseShifter,   // ps: 1 param (phi)
    BeamSplitterRx, // bs_rx: 2 params (theta, phi_tr)

    // Custom gate
    Custom(CustomGateId),

    // Measurement
    Measure,

    // Barrier (no-op for simulation, prevents optimization)
    Barrier,

    // Reset qubit to |0⟩
    Reset,
}

impl GateKind {
    /// Number of qubits this gate acts on.
    pub fn num_qubits(&self) -> usize {
        match self {
            GateKind::H
            | GateKind::X
            | GateKind::Y
            | GateKind::Z
            | GateKind::S
            | GateKind::Sdg
            | GateKind::T
            | GateKind::Tdg
            | GateKind::Sx
            | GateKind::Sxdg
            | GateKind::Id
            | GateKind::Rx
            | GateKind::Ry
            | GateKind::Rz
            | GateKind::U3
            | GateKind::U2
            | GateKind::U1
            | GateKind::PhaseShifter
            | GateKind::Reset => 1,

            GateKind::CX
            | GateKind::CY
            | GateKind::CZ
            | GateKind::Swap
            | GateKind::CRz
            | GateKind::CU3
            | GateKind::Rbs
            | GateKind::BeamSplitterRx => 2,

            GateKind::CCX | GateKind::CSwap => 3,

            // Measure and Barrier can act on variable qubits
            GateKind::Measure | GateKind::Barrier | GateKind::Custom(_) => 0,
        }
    }

    /// Number of parameters this gate takes.
    pub fn num_params(&self) -> usize {
        match self {
            GateKind::Rx
            | GateKind::Ry
            | GateKind::Rz
            | GateKind::U1
            | GateKind::CRz
            | GateKind::Rbs => 1,
            GateKind::U2 | GateKind::BeamSplitterRx => 2,
            GateKind::U3 | GateKind::CU3 => 3,
            GateKind::PhaseShifter => 1,
            GateKind::Custom(_) => 0, // variable, checked at use site
            _ => 0,
        }
    }
}

/// A single gate operation in the circuit.
#[derive(Clone, Debug)]
pub struct GateOp {
    pub gate: GateKind,
    pub qubits: SmallVec<[Qubit; 3]>,
    pub params: SmallVec<[ParamExpr; 3]>,
    /// For Measure: which classical bit to store result in.
    pub classical_bit: Option<u32>,
    /// Classical condition: only execute this gate if the creg's
    /// integer value (read from `num_bits` consecutive classical
    /// bits starting at `start_bit`, LSB-first) equals `expected`.
    /// Equivalent to QASM2's `if(c == V)` with `c = classical_bits
    /// [start_bit..start_bit + num_bits]`.
    ///
    /// Format: `Some((start_bit, num_bits, expected))`. For a 1-bit
    /// creg this collapses to the prior `(start_bit, 1, expected)`
    /// shape; multi-bit cregs (e.g. `creg c[2]; if(c == 2)`) use
    /// `num_bits = 2` and `expected = 2` to match Qiskit semantics
    /// against the whole register.
    pub condition: Option<(u32, u32, u64)>,
}

impl GateOp {
    /// Evaluate the classical condition against a slice of classical
    /// bits. Returns `true` if the gate should run (no condition or
    /// the creg's value matches `expected`); `false` if the gate
    /// should be skipped.
    ///
    /// Reads `num_bits` consecutive bits from `classical_bits`
    /// starting at `start_bit` and assembles the LSB-first integer
    /// value, matching QASM2's `if(c == V)` semantics. Bits past
    /// the end of `classical_bits` are treated as 0 (best-effort
    /// for malformed input — the lowering should never produce
    /// out-of-range conditions, but this avoids a panic).
    pub fn condition_satisfied(&self, classical_bits: &[u8]) -> bool {
        let Some((start_bit, num_bits, expected)) = self.condition else {
            return true;
        };
        let mut value: u64 = 0;
        for i in 0..num_bits {
            let idx = (start_bit + i) as usize;
            if idx < classical_bits.len() {
                value |= (classical_bits[idx] as u64 & 1) << i;
            }
        }
        value == expected
    }
}

/// Whether this circuit is gate-based (qubit) or photonic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CircuitType {
    GateBased,
    Photonic,
}

/// The circuit intermediate representation.
#[derive(Clone, Debug)]
pub struct CircuitIR {
    pub num_qubits: u32,
    pub num_classical_bits: u32,
    pub ops: Vec<GateOp>,
    pub circuit_type: CircuitType,
    /// Symbol id -> name mapping for free (unbound) symbols.
    pub symbols: HashMap<SymbolId, String>,
    /// Custom gate definitions: id -> (param_symbol_ids, sub-circuit).
    pub custom_gates: HashMap<CustomGateId, CustomGateDef>,
}

/// A custom gate definition with bound parameter symbols.
#[derive(Clone, Debug)]
pub struct CustomGateDef {
    pub name: String,
    pub param_symbols: Vec<SymbolId>,
    pub body: CircuitIR,
}

impl CircuitIR {
    pub fn new(num_qubits: u32, circuit_type: CircuitType) -> Self {
        Self {
            num_qubits,
            num_classical_bits: 0,
            ops: Vec::new(),
            circuit_type,
            symbols: HashMap::new(),
            custom_gates: HashMap::new(),
        }
    }

    pub fn num_free_symbols(&self) -> usize {
        self.symbols.len()
    }

    pub fn add_op(&mut self, op: GateOp) {
        self.ops.push(op);
    }

    /// Total gate count (excludes Measure, Barrier, Id).
    pub fn gate_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| {
                !matches!(
                    op.gate,
                    GateKind::Measure | GateKind::Barrier | GateKind::Id
                )
            })
            .count()
    }

    /// Circuit depth: longest path through any single qubit.
    pub fn depth(&self) -> usize {
        if self.num_qubits == 0 {
            return 0;
        }
        let mut qubit_depth = vec![0usize; self.num_qubits as usize];
        for op in &self.ops {
            if matches!(
                op.gate,
                GateKind::Measure | GateKind::Barrier | GateKind::Id
            ) {
                continue;
            }
            // Find the max depth across all qubits this gate touches
            let max_d = op
                .qubits
                .iter()
                .map(|q| qubit_depth[q.0 as usize])
                .max()
                .unwrap_or(0);
            // Set all touched qubits to max_d + 1
            for q in &op.qubits {
                qubit_depth[q.0 as usize] = max_d + 1;
            }
        }
        qubit_depth.into_iter().max().unwrap_or(0)
    }

    /// Count of T and T† gates.
    pub fn t_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op.gate, GateKind::T | GateKind::Tdg))
            .count()
    }

    /// Count of CX (CNOT) gates.
    pub fn cx_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op.gate, GateKind::CX))
            .count()
    }

    /// Count of all two-qubit gates.
    pub fn two_qubit_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| op.gate.num_qubits() == 2)
            .count()
    }

    /// Count of all three-qubit gates (CCX, CSwap).
    pub fn three_qubit_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| op.gate.num_qubits() == 3)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    fn simple_op(gate: GateKind, qubits: &[u32]) -> GateOp {
        GateOp {
            gate,
            qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
            params: smallvec![],
            classical_bit: None,
            condition: None,
        }
    }

    #[test]
    fn test_resource_gate_count() {
        let mut c = CircuitIR::new(3, CircuitType::GateBased);
        c.add_op(simple_op(GateKind::H, &[0]));
        c.add_op(simple_op(GateKind::CX, &[0, 1]));
        c.add_op(simple_op(GateKind::Measure, &[0]));
        c.add_op(simple_op(GateKind::Barrier, &[0]));
        c.add_op(simple_op(GateKind::T, &[1]));
        assert_eq!(c.gate_count(), 3); // H + CX + T (not Measure, Barrier)
    }

    #[test]
    fn test_resource_depth() {
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        // H on q0 (depth 1), H on q1 (depth 1) — parallel
        c.add_op(simple_op(GateKind::H, &[0]));
        c.add_op(simple_op(GateKind::H, &[1]));
        // CX on q0,q1 (depth 2)
        c.add_op(simple_op(GateKind::CX, &[0, 1]));
        // H on q0 only (depth 3 on q0, q1 stays at 2)
        c.add_op(simple_op(GateKind::H, &[0]));
        assert_eq!(c.depth(), 3);
    }

    #[test]
    fn test_resource_t_count() {
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        c.add_op(simple_op(GateKind::T, &[0]));
        c.add_op(simple_op(GateKind::Tdg, &[1]));
        c.add_op(simple_op(GateKind::H, &[0]));
        c.add_op(simple_op(GateKind::T, &[0]));
        assert_eq!(c.t_count(), 3);
    }

    #[test]
    fn test_resource_cx_count() {
        let mut c = CircuitIR::new(3, CircuitType::GateBased);
        c.add_op(simple_op(GateKind::CX, &[0, 1]));
        c.add_op(simple_op(GateKind::CX, &[1, 2]));
        c.add_op(simple_op(GateKind::CZ, &[0, 2]));
        assert_eq!(c.cx_count(), 2);
        assert_eq!(c.two_qubit_count(), 3);
    }

    #[test]
    fn test_resource_three_qubit_count() {
        let mut c = CircuitIR::new(3, CircuitType::GateBased);
        c.add_op(simple_op(GateKind::CCX, &[0, 1, 2]));
        c.add_op(simple_op(GateKind::CSwap, &[0, 1, 2]));
        c.add_op(simple_op(GateKind::H, &[0]));
        assert_eq!(c.three_qubit_count(), 2);
    }
}
