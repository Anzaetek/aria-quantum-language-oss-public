use smallvec::SmallVec;
use std::collections::HashMap;

/// A qubit (or photonic mode) identifier.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct Qubit(pub u32);

/// Unique identifier for a symbol (parameter name).
pub type SymbolId = u32;

/// The name a free symbol is reported under when no declared name is known
/// for it. One spelling, shared by every message that mentions a symbol, so
/// a parameter-count refusal and an `UnboundSymbol` error agree.
pub fn fallback_symbol_name(id: SymbolId) -> String {
    format!("sym_{id}")
}

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

    // Qudit gates (DITQASM, PLAN-QUDIT.md Q2). The generalised one-wire
    // gates keep their qubit names — on a `d > 2` wire `H` is the Fourier
    // gate, `X` the cyclic shift, `Z` the clock (mqt.qudits' semantics,
    // verified in Q0; each is the qubit gate at d = 2). These two have no
    // qubit spelling at all.
    /// `rxy(i, j, θ, φ)`: `exp(−iθ/2 (cos φ X + sin φ Y))` on the two-level
    /// subspace `{|i⟩, |j⟩}` of one wire, identity on every other level.
    /// Params: `i`, `j` (integers, `i < j < d`), `θ`, `φ`.
    Rxy,
    /// `csum a, b`: `|c⟩|t⟩ → |c⟩|t + c mod d_b⟩` — the generalised CX
    /// (control first). Equal to `CX` at `d = 2`.
    CSum,

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
            | GateKind::Rxy
            | GateKind::Reset => 1,

            GateKind::CX
            | GateKind::CY
            | GateKind::CZ
            | GateKind::Swap
            | GateKind::CRz
            | GateKind::CU3
            | GateKind::Rbs
            | GateKind::CSum
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
            GateKind::Rxy => 4,
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
            let bit = if idx < classical_bits.len() {
                classical_bits[idx] & 1
            } else {
                0
            };
            if i >= 64 {
                // `1u64 << 64` panics in debug and is masked to `<< 0` in
                // release, so `if (c == 1)` on a `creg c[70]` either crashed or
                // silently OR-ed bit 64 into bit 0 — a condition that fired on
                // the wrong register contents. It is not a representation gap:
                // `expected` is a u64, so ANY set bit at or above 64 makes the
                // register strictly larger than it, and the test is false.
                if bit != 0 {
                    return false;
                }
                continue;
            }
            value |= (bit as u64) << i;
        }
        value == expected
    }
}

/// Whether this circuit is gate-based (qubit), photonic, or fermionic.
///
/// `Fermionic` is a Jordan–Wigner occupation circuit. Its ops are ordinary
/// qubit gates — the same ones `GateBased` carries — and
/// [`CircuitIR::fermionic_registers`] records the `mode` declaration the
/// file made. Engines that evolve qubit gates accept it. Photonics refuses
/// it by name: a fermionic mode is not an optical mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CircuitType {
    GateBased,
    Photonic,
    Fermionic,
}

/// A `mode` register from FermionicQASM, recorded by name so a refusal or a
/// sector-native consumer can name the register and read its block structure.
///
/// `spatial` is the size written in `mode name[spatial]`. When `spin` is
/// false the register occupies wires `start .. start + spatial`. When `spin`
/// is true it occupies `2 * spatial` wires in blocks: `start .. start +
/// spatial` spin-up (alpha), then the next `spatial` wires spin-down
/// (beta). That is the sequential layout pinned in
/// `conventions.json` (`spin_ordering`, qiskit-cold-atom at
/// `ad8893f`), not OPTICQASM `pol`'s interleaved `(s, p) -> 2s + p` map,
/// and not that backend's `norb - 1 - orb` orbital reversal — the fixture
/// records the reversal and leaves it unapplied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FermionicRegister {
    pub name: String,
    pub start: u32,
    pub spatial: u32,
    pub spin: bool,
}

/// A register declared with a DITQASM dimension group — `qreg q [3][3,2,5];`
/// — recorded by name so a refusal can say *which* register asked for a
/// qudit, not just that one did.
///
/// `dims[i]` is the local dimension of wire `start + i`. A group of all 2s
/// (`qreg q [2][2,2];`, which is how `mqt.qudits` re-emits a bare `qreg`)
/// is still recorded here: the declaration was explicit, and hiding it would
/// make the IR lie about its source. Only a dimension ≠ 2 is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuditRegister {
    pub name: String,
    pub start: u32,
    pub dims: Vec<u32>,
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
    /// Registers declared with an explicit per-wire dimension (DITQASM).
    /// Empty for every circuit built by [`CircuitIR::new`] and for every
    /// OPENQASM source — a wire not covered by an entry here is a qubit.
    /// See [`CircuitIR::wire_dim`] and [`CircuitIR::refuse_qudits`]: no
    /// engine in this workspace evolves a `d ≠ 2` wire (PLAN-QUDIT.md Q1),
    /// and each one says so at its own door rather than running the
    /// circuit as qubits and returning a plausible number.
    pub qudit_registers: Vec<QuditRegister>,
    /// FermionicQASM `mode` declarations, in source order. Empty for every
    /// other lane, including a qubit circuit that happens to contain `Rbs`.
    /// A fermionic mode under Jordan–Wigner is a two-level occupation wire:
    /// this is block structure over qubits, not a qudit dimension, and
    /// [`CircuitIR::wire_dim`] stays 2 for every wire it covers.
    pub fermionic_registers: Vec<FermionicRegister>,
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
            qudit_registers: Vec::new(),
            fermionic_registers: Vec::new(),
        }
    }

    /// Local dimension of `wire`: 2 unless a DITQASM register declared it
    /// otherwise.
    pub fn wire_dim(&self, wire: u32) -> u32 {
        self.qudit_registers
            .iter()
            .find_map(|r| {
                let i = wire.checked_sub(r.start)? as usize;
                r.dims.get(i).copied()
            })
            .unwrap_or(2)
    }

    /// Every wire's local dimension, in wire order.
    pub fn wire_dims(&self) -> Vec<u32> {
        (0..self.num_qubits).map(|w| self.wire_dim(w)).collect()
    }

    /// The first wire whose dimension is not 2, with its register — `None`
    /// for a qubit circuit.
    pub fn first_qudit(&self) -> Option<(&QuditRegister, u32, u32)> {
        self.qudit_registers.iter().find_map(|r| {
            r.dims
                .iter()
                .enumerate()
                .find(|(_, &d)| d != 2)
                .map(|(i, &d)| (r, r.start + i as u32, d))
        })
    }

    /// `Ok(())` for a qubit circuit; otherwise the refusal every engine
    /// returns from its door, naming the engine, the register, the wire
    /// and the dimension. One spelling so the CLI tests and the
    /// backend-level tests assert the same sentence.
    ///
    /// This exists because the failure it prevents is silent: a d = 3
    /// register run through a qubit engine does not crash, it evolves
    /// `2^n` amplitudes with the wrong `n`, exits 0 and prints a number.
    pub fn refuse_qudits(&self, engine: &str) -> crate::error::Result<()> {
        match self.first_qudit() {
            None => Ok(()),
            Some((reg, wire, d)) => Err(crate::error::OmegaError::Unsupported(format!(
                "{engine}: register '{}' declares dimension {d} on wire {wire}; \
                 this engine is qubit-only (d = 2) and a qudit circuit run as \
                 qubits would return a plausible wrong number, so it is refused \
                 instead. The engines that evolve qudits are `quditsv` (exact, \
                 dense) and `mps` (PLAN-QUDIT.md Q1–Q3)",
                reg.name
            ))),
        }
    }

    pub fn num_free_symbols(&self) -> usize {
        self.symbols.len()
    }

    /// The free symbol IDs in ascending order: the order a flat parameter
    /// vector binds them in (`ParameterBinding::from_flat`, the CLI's
    /// `--params`, the WASM host ABI).
    pub fn sorted_symbol_ids(&self) -> Vec<SymbolId> {
        let mut ids: Vec<SymbolId> = self.symbols.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Display name for a symbol: its declared name, else `sym_{id}`.
    pub fn symbol_name(&self, id: SymbolId) -> String {
        self.symbols
            .get(&id)
            .cloned()
            .unwrap_or_else(|| fallback_symbol_name(id))
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

/// Is every operation in this circuit a Clifford (or a classical no-op)?
///
/// The Clifford group is exactly where the stabilizer backend is both **exact
/// and cheap** — polynomial in the qubit count rather than exponential — so
/// this predicate is what lets a dispatcher route a circuit to it instead of a
/// dense statevector.
///
/// # What counts, and one that is easy to miss
///
/// `H`, `X`, `Y`, `Z`, `S`, `Sdg`, `CX`, `CY`, `CZ`, `Swap`, plus `Sx`/`Sxdg`.
/// **The square roots of X are Clifford** — `sx·sx = X` exactly — and they are
/// routinely left out of these lists because they look like rotations. Omitting
/// them does not produce a wrong answer, it silently sends a perfectly good
/// Clifford circuit to an exponential backend, which is the kind of miss that
/// never shows up as a failure.
///
/// `Measure`, `Barrier`, `Reset` and `Id` are admitted as well: none of them is
/// a unitary outside the group, and the stabilizer formalism represents all of
/// them exactly.
///
/// Rotations are **not** admitted even at Clifford angles. `Rz(π/2)` equals `S`
/// up to a phase, but recognising that requires trusting a float comparison to
/// decide whether a circuit is exactly representable, and being wrong there
/// means a wrong answer rather than a slow one.
pub fn is_clifford_only(circuit: &CircuitIR) -> bool {
    circuit.ops.iter().all(|op| {
        matches!(
            op.gate,
            GateKind::H
                | GateKind::X
                | GateKind::Y
                | GateKind::Z
                | GateKind::S
                | GateKind::Sdg
                | GateKind::Sx
                | GateKind::Sxdg
                | GateKind::CX
                | GateKind::CY
                | GateKind::CZ
                | GateKind::Swap
                | GateKind::Measure
                | GateKind::Barrier
                | GateKind::Reset
                | GateKind::Id
        )
    })
}

#[cfg(test)]
mod clifford_predicate_tests {
    use super::*;
    use smallvec::smallvec;

    fn op(gate: GateKind) -> GateOp {
        GateOp {
            gate,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        }
    }

    fn circuit(gates: &[GateKind]) -> CircuitIR {
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        for g in gates {
            c.add_op(op(g.clone()));
        }
        c
    }

    #[test]
    fn the_clifford_generators_are_recognised() {
        assert!(is_clifford_only(&circuit(&[
            GateKind::H,
            GateKind::S,
            GateKind::Sdg,
            GateKind::CX,
            GateKind::CZ,
            GateKind::CY,
            GateKind::Swap,
        ])));
    }

    /// The square roots of X are Clifford (`sx·sx = X`) and are the ones these
    /// lists usually forget — a miss here silently routes a Clifford circuit
    /// to an exponential backend rather than producing a wrong answer, so it
    /// would never surface as a failure.
    #[test]
    fn the_square_roots_of_x_are_clifford() {
        assert!(is_clifford_only(&circuit(&[GateKind::Sx, GateKind::Sxdg])));
    }

    #[test]
    fn non_clifford_gates_are_rejected() {
        for g in [GateKind::T, GateKind::Tdg, GateKind::Rz, GateKind::U3] {
            assert!(
                !is_clifford_only(&circuit(&[GateKind::H, g.clone()])),
                "{g:?} must not be treated as Clifford"
            );
        }
    }

    /// Not admitted even at a Clifford angle: deciding that from a float would
    /// mean a wrong answer when the comparison is wrong, rather than a slow one.
    #[test]
    fn rotations_are_rejected_even_at_clifford_angles() {
        let mut c = CircuitIR::new(1, CircuitType::GateBased);
        let mut o = op(GateKind::Rz);
        o.params = smallvec![ParamExpr::Concrete(std::f64::consts::FRAC_PI_2)];
        c.add_op(o);
        assert!(!is_clifford_only(&c));
    }

    #[test]
    fn classical_operations_do_not_disqualify_a_circuit() {
        assert!(is_clifford_only(&circuit(&[
            GateKind::H,
            GateKind::Measure,
            GateKind::Reset,
            GateKind::Barrier,
            GateKind::Id,
        ])));
    }
}
