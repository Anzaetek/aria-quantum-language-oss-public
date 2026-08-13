use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Standard gate set (universal + common shorthands + discrete optics).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GateKind {
    // Single-qubit
    I,
    X,
    Y,
    Z,
    H,
    S,
    Sdg,
    T,
    Tdg,
    SX,
    // Parametric single-qubit
    RX,
    RY,
    RZ,
    P,
    U,
    // Two-qubit
    CX,
    CY,
    CZ,
    SWAP,
    // Parametric two-qubit
    RXX,
    RYY,
    RZZ,
    CP,
    /// Controlled-RZ: `CRz(λ) = diag(1, 1, e^{−iλ/2}, e^{+iλ/2})`, control
    /// first and target second — qelib1's `crz`.
    ///
    /// Distinct from [`GateKind::CP`], which is `diag(1, 1, 1, e^{iλ})`. They
    /// differ by a relative phase `e^{-iλ/2}` on the controlled block, so
    /// substituting one for the other is NOT a global phase and is visible in
    /// any interference. `CRz(λ) = CP(λ)` preceded by `P(−λ/2)` on the
    /// control, exactly.
    ///
    /// The omega IR and the QASM lane both had this gate; the Aria language
    /// simply could not spell it, so a circuit its own backends execute was
    /// inexpressible in its own source form.
    CRz,
    /// Reconfigurable Beam Splitter / Givens rotation:
    /// `RBS(θ) = exp(−i·θ/2·(Y⊗X − X⊗Y))`. Hamming-weight preserving;
    /// the primitive of butterfly / unary QML circuits
    /// (Kerenidis et al., arXiv:2606.03517).
    RBS,
    // Three-qubit
    CCX,
    CSWAP,
    // Special
    Barrier,
    Reset,
    Measure,
    // Discrete optics (omega-functions extensions)
    BeamSplitter,
    PhaseShifter,
    Squeezing,
    Displacement,
    Kerr,
    /// Half-wave plate at angle θ, acting on ONE spatial mode's H/V pair.
    ///
    /// Not a primitive downstream: `omega-parser` expands it into phase
    /// shifters and a beam splitter on the polarization sub-modes
    /// (`PS(π)` on V, `BSrx(2θ, 0)`, then `PS(π/2)` on both). It exists in this
    /// AST so that an OPTICQASM file naming `hwp` can be re-emitted as `hwp`
    /// rather than as its expansion — the AST is the only layer that can
    /// round-trip the spelling, because the IR never sees it.
    ///
    /// Only meaningful on a register declared `polarized`; see
    /// [`RegisterDecl::polarized`].
    HalfWavePlate,
    /// Polarizing beam splitter across TWO spatial modes.
    ///
    /// **Swaps H between the two spatial modes and transmits V** — Perceval's
    /// convention, pinned in `test_perceval_conventions.py`. Note this is the
    /// opposite of the common "transmits H, reflects V" phrasing, which is why
    /// the pin is against a read of the actual matrix rather than a description.
    ///
    /// Takes no parameters, unlike every other photonic gate here.
    PolarizingBeamSplitter,
}

impl GateKind {
    pub fn n_qubits(self) -> usize {
        match self {
            Self::CX
            | Self::CY
            | Self::CZ
            | Self::SWAP
            | Self::RXX
            | Self::RYY
            | Self::RZZ
            | Self::CP
            | Self::RBS
            | Self::BeamSplitter => 2,
            Self::CCX | Self::CSWAP => 3,
            _ => 1,
        }
    }

    pub fn is_self_inverse(self) -> bool {
        matches!(
            self,
            Self::H | Self::X | Self::Y | Self::Z | Self::CX | Self::CZ | Self::SWAP | Self::CCX
        )
    }

    pub fn is_rotation(self) -> bool {
        // Every parametric gate of the form U(θ) with U(θ)† = U(-θ) and
        // U(θ₁) · U(θ₂) = U(θ₁+θ₂) — i.e. things the optimizer can
        // negate for adjoints and additively merge when repeated on the
        // same qubit set.
        matches!(
            self,
            Self::RX
                | Self::RY
                | Self::RZ
                | Self::P
                | Self::CP
                | Self::RXX
                | Self::RYY
                | Self::RZZ
                | Self::RBS
        )
    }

    pub fn is_meta(self) -> bool {
        matches!(self, Self::Barrier | Self::Measure | Self::Reset)
    }
}

/// A gate definition: kind + parameters (symbolic or concrete).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateDef {
    pub kind: GateKind,
    pub params: Vec<super::expr::ParamExpr>,
    pub label: Option<String>,
}

impl PartialEq for GateDef {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.params == other.params
    }
}

impl GateDef {
    pub fn new(kind: GateKind) -> Self {
        Self {
            kind,
            params: vec![],
            label: None,
        }
    }

    /// Create with concrete f64 parameters (backward compatible).
    pub fn with_params(kind: GateKind, params: Vec<f64>) -> Self {
        Self {
            kind,
            params: params
                .into_iter()
                .map(super::expr::ParamExpr::from)
                .collect(),
            label: None,
        }
    }

    /// Create with symbolic ParamExpr parameters.
    pub fn with_exprs(kind: GateKind, params: Vec<super::expr::ParamExpr>) -> Self {
        Self {
            kind,
            params,
            label: None,
        }
    }

    pub fn n_qubits(&self) -> usize {
        self.kind.n_qubits()
    }

    /// Get parameters as concrete f64 values (panics if any are symbolic).
    pub fn params_as_f64(&self) -> Vec<f64> {
        self.params
            .iter()
            .map(|p| {
                p.try_as_f64()
                    .expect("cannot convert symbolic param to f64")
            })
            .collect()
    }

    /// Try to get parameters as f64, returning None if symbolic.
    pub fn try_params_as_f64(&self) -> Option<Vec<f64>> {
        self.params.iter().map(|p| p.try_as_f64()).collect()
    }
}

// Convenient gate constructors
pub fn h() -> GateDef {
    GateDef::new(GateKind::H)
}
pub fn x() -> GateDef {
    GateDef::new(GateKind::X)
}
pub fn y() -> GateDef {
    GateDef::new(GateKind::Y)
}
pub fn z() -> GateDef {
    GateDef::new(GateKind::Z)
}
pub fn s() -> GateDef {
    GateDef::new(GateKind::S)
}
pub fn t() -> GateDef {
    GateDef::new(GateKind::T)
}
pub fn cx() -> GateDef {
    GateDef::new(GateKind::CX)
}
pub fn cz() -> GateDef {
    GateDef::new(GateKind::CZ)
}
pub fn ccx() -> GateDef {
    GateDef::new(GateKind::CCX)
}
pub fn swap() -> GateDef {
    GateDef::new(GateKind::SWAP)
}
pub fn rx(theta: f64) -> GateDef {
    GateDef::with_params(GateKind::RX, vec![theta])
}
pub fn ry(theta: f64) -> GateDef {
    GateDef::with_params(GateKind::RY, vec![theta])
}
pub fn rz(theta: f64) -> GateDef {
    GateDef::with_params(GateKind::RZ, vec![theta])
}
pub fn p(lam: f64) -> GateDef {
    GateDef::with_params(GateKind::P, vec![lam])
}
pub fn cp(lam: f64) -> GateDef {
    GateDef::with_params(GateKind::CP, vec![lam])
}
pub fn u(theta: f64, phi: f64, lam: f64) -> GateDef {
    GateDef::with_params(GateKind::U, vec![theta, phi, lam])
}
pub fn measure() -> GateDef {
    GateDef::new(GateKind::Measure)
}
pub fn reset() -> GateDef {
    GateDef::new(GateKind::Reset)
}
pub fn barrier() -> GateDef {
    GateDef::new(GateKind::Barrier)
}

/// Reference to a qubit inside a register.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Qubit {
    pub register: String,
    pub index: usize,
}

impl Qubit {
    pub fn new(register: &str, index: usize) -> Self {
        Self {
            register: register.to_string(),
            index,
        }
    }
}

impl fmt::Display for Qubit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]", self.register, self.index)
    }
}

/// Reference to a classical bit inside a register.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Clbit {
    pub register: String,
    pub index: usize,
}

impl Clbit {
    pub fn new(register: &str, index: usize) -> Self {
        Self {
            register: register.to_string(),
            index,
        }
    }
}

impl fmt::Display for Clbit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]", self.register, self.index)
    }
}

/// Register kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegisterKind {
    Quantum,
    Classical,
}

/// Declaration of a named register.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterDecl {
    pub name: String,
    pub size: usize,
    pub kind: RegisterKind,
    /// Photonic only: this register's modes each carry an H and a V
    /// polarization, so it occupies `2 * size` OPTICAL modes indexed `2s + p`
    /// with `p = 0` meaning H. `size` continues to count SPATIAL modes.
    ///
    /// A flag rather than a naming convention on purpose: a convention does not
    /// survive a rename, and getting this wrong is silent — `photon q[N] pol;`
    /// and `photon q[N];` both parse, both refuse nothing, and every mode index
    /// means something different between them.
    pub polarized: bool,
}

impl RegisterDecl {
    pub fn qubits(&self) -> Vec<Qubit> {
        assert_eq!(self.kind, RegisterKind::Quantum);
        (0..self.size).map(|i| Qubit::new(&self.name, i)).collect()
    }

    pub fn clbits(&self) -> Vec<Clbit> {
        assert_eq!(self.kind, RegisterKind::Classical);
        (0..self.size).map(|i| Clbit::new(&self.name, i)).collect()
    }
}

/// A single quantum instruction: gate applied to specific qubits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    pub gate: GateDef,
    pub qubits: Vec<Qubit>,
    pub clbits: Vec<Clbit>,
    /// Runtime condition: gate only executes when the referenced classical
    /// bit equals the given value. Lowered from Aria `when m[k] == v { … }`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<(Clbit, u64)>,
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let qstr: Vec<String> = self.qubits.iter().map(|q| q.to_string()).collect();
        if !self.gate.params.is_empty() {
            let pstr: Vec<String> = self.gate.params.iter().map(|p| format!("{p}")).collect();
            write!(
                f,
                "{:?}({}) {}",
                self.gate.kind,
                pstr.join(", "),
                qstr.join(", ")
            )?;
        } else {
            write!(f, "{:?} {}", self.gate.kind, qstr.join(", "))?;
        }
        if !self.clbits.is_empty() {
            let cstr: Vec<String> = self.clbits.iter().map(|c| c.to_string()).collect();
            write!(f, " -> {}", cstr.join(", "))?;
        }
        Ok(())
    }
}

/// A quantum circuit — the top-level AST node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Circuit {
    pub name: String,
    pub registers: Vec<RegisterDecl>,
    pub instructions: Vec<Instruction>,
    pub metadata: HashMap<String, String>,
    pub annotations: Vec<super::annotation::Annotation>,
}

impl Circuit {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            registers: vec![],
            instructions: vec![],
            metadata: HashMap::new(),
            annotations: vec![],
        }
    }

    /// Add a formal annotation to this circuit.
    pub fn annotate(&mut self, ann: super::annotation::Annotation) {
        self.annotations.push(ann);
    }

    /// Get all free symbols in this circuit's gate parameters.
    pub fn free_symbols(&self) -> std::collections::HashSet<String> {
        let mut syms = std::collections::HashSet::new();
        for inst in &self.instructions {
            for p in &inst.gate.params {
                syms.extend(p.free_symbols());
            }
        }
        syms
    }

    /// Bind symbolic parameters to concrete values. Returns a new circuit.
    pub fn bind_params(
        &self,
        bindings: &std::collections::HashMap<String, f64>,
    ) -> Result<Circuit, String> {
        let mut result = self.clone();
        for inst in &mut result.instructions {
            let mut resolved = Vec::new();
            for p in &inst.gate.params {
                let val = p.eval(bindings)?;
                resolved.push(super::expr::ParamExpr::Concrete(val));
            }
            inst.gate.params = resolved;
        }
        Ok(result)
    }

    /// Add a quantum register and return its qubits.
    pub fn qreg(&mut self, name: &str, size: usize) -> Vec<Qubit> {
        self.qreg_inner(name, size, false)
    }

    /// Add a POLARIZED photonic register: `size` spatial modes, each carrying
    /// an H and a V sub-mode, so `2 * size` optical modes downstream.
    ///
    /// Separate constructor rather than a parameter on [`Circuit::qreg`] so
    /// that every existing caller keeps its meaning — a register that silently
    /// became polarized would reinterpret every mode index in the circuit.
    pub fn qreg_polarized(&mut self, name: &str, size: usize) -> Vec<Qubit> {
        self.qreg_inner(name, size, true)
    }

    fn qreg_inner(&mut self, name: &str, size: usize, polarized: bool) -> Vec<Qubit> {
        let reg = RegisterDecl {
            name: name.to_string(),
            size,
            kind: RegisterKind::Quantum,
            polarized,
        };
        let qubits = reg.qubits();
        self.registers.push(reg);
        qubits
    }

    /// Add a classical register and return its clbits.
    pub fn creg(&mut self, name: &str, size: usize) -> Vec<Clbit> {
        let reg = RegisterDecl {
            name: name.to_string(),
            size,
            kind: RegisterKind::Classical,
            polarized: false,
        };
        let clbits = reg.clbits();
        self.registers.push(reg);
        clbits
    }

    /// Apply a gate to the given qubits.
    pub fn apply(&mut self, gate: GateDef, qubits: Vec<Qubit>) {
        self.instructions.push(Instruction {
            gate,
            qubits,
            clbits: vec![],
            condition: None,
        });
    }

    /// Apply a gate with classical bits (for measurement).
    pub fn apply_with_clbits(&mut self, gate: GateDef, qubits: Vec<Qubit>, clbits: Vec<Clbit>) {
        self.instructions.push(Instruction {
            gate,
            qubits,
            clbits,
            condition: None,
        });
    }

    /// Measure a qubit into a classical bit.
    pub fn measure(&mut self, qubit: &Qubit, clbit: &Clbit) {
        self.apply_with_clbits(measure(), vec![qubit.clone()], vec![clbit.clone()]);
    }

    /// Measure all qubits into corresponding classical bits.
    pub fn measure_all(&mut self, qubits: &[Qubit], clbits: &[Clbit]) {
        for (q, c) in qubits.iter().zip(clbits.iter()) {
            self.measure(q, c);
        }
    }

    /// Reset a qubit.
    pub fn reset_qubit(&mut self, qubit: &Qubit) {
        self.apply(reset(), vec![qubit.clone()]);
    }

    /// Add a barrier on the given qubits.
    pub fn barrier(&mut self, qubits: &[Qubit]) {
        self.apply(barrier(), qubits.to_vec());
    }

    /// Inline another circuit's instructions with qubit/clbit remapping.
    pub fn append_circuit(
        &mut self,
        other: &Circuit,
        qubit_map: &HashMap<Qubit, Qubit>,
        clbit_map: Option<&HashMap<Clbit, Clbit>>,
    ) {
        let empty_clbit_map = HashMap::new();
        let cmap = clbit_map.unwrap_or(&empty_clbit_map);
        for inst in &other.instructions {
            let mapped_q: Vec<Qubit> = inst
                .qubits
                .iter()
                .map(|q| qubit_map.get(q).cloned().unwrap_or_else(|| q.clone()))
                .collect();
            let mapped_c: Vec<Clbit> = inst
                .clbits
                .iter()
                .map(|c| cmap.get(c).cloned().unwrap_or_else(|| c.clone()))
                .collect();
            self.instructions.push(Instruction {
                gate: inst.gate.clone(),
                qubits: mapped_q,
                clbits: mapped_c,
                condition: inst.condition.clone(),
            });
        }
    }

    /// Total number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.registers
            .iter()
            .filter(|r| r.kind == RegisterKind::Quantum)
            .map(|r| r.size)
            .sum()
    }

    /// Total number of classical bits.
    pub fn n_clbits(&self) -> usize {
        self.registers
            .iter()
            .filter(|r| r.kind == RegisterKind::Classical)
            .map(|r| r.size)
            .sum()
    }

    /// Circuit depth (longest qubit timeline, excluding meta gates).
    pub fn depth(&self) -> usize {
        let mut qubit_depth: HashMap<Qubit, usize> = HashMap::new();
        for inst in &self.instructions {
            if inst.gate.kind.is_meta() {
                continue;
            }
            let current = inst
                .qubits
                .iter()
                .map(|q| qubit_depth.get(q).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            for q in &inst.qubits {
                qubit_depth.insert(q.clone(), current + 1);
            }
        }
        qubit_depth.values().copied().max().unwrap_or(0)
    }

    /// Number of non-meta gates.
    pub fn gate_count(&self) -> usize {
        self.instructions
            .iter()
            .filter(|i| !i.gate.kind.is_meta())
            .count()
    }

    // -- Aria-style composition operators --

    /// Sequential composition: apply self then other on the same qubits (pipe).
    /// Both circuits must have the same number of qubits.
    pub fn pipe(&self, other: &Circuit) -> Circuit {
        let mut result = self.clone();
        result.name = format!("{}_{}", self.name, other.name);
        // Map other's qubits to self's qubits by position
        let self_qubits = self.all_qubits();
        let other_qubits = other.all_qubits();
        let mut qmap = HashMap::new();
        for (oq, sq) in other_qubits.iter().zip(self_qubits.iter()) {
            qmap.insert(oq.clone(), sq.clone());
        }
        result.append_circuit(other, &qmap, None);
        result
    }

    /// Parallel composition: tensor product of two circuits on disjoint qubits.
    pub fn tensor(&self, other: &Circuit) -> Circuit {
        let mut result = self.clone();
        result.name = format!("{}_{}", self.name, other.name);
        // Add other's registers with offset names
        let _offset = self.n_qubits();
        for reg in &other.registers {
            if reg.kind == RegisterKind::Quantum {
                let new_name = format!("{}_t", reg.name);
                let new_reg = RegisterDecl {
                    name: new_name,
                    size: reg.size,
                    // Carried, not defaulted: dropping it here would turn a
                    // polarization circuit into a plain one on composition,
                    // halving every mode's meaning with no diagnostic.
                    polarized: reg.polarized,
                    kind: RegisterKind::Quantum,
                };
                result.registers.push(new_reg);
            }
        }
        // Add instructions with remapped qubits
        for inst in &other.instructions {
            let mapped_q: Vec<Qubit> = inst
                .qubits
                .iter()
                .map(|q| Qubit::new(&format!("{}_t", q.register), q.index))
                .collect();
            result.instructions.push(Instruction {
                gate: inst.gate.clone(),
                qubits: mapped_q,
                clbits: inst.clbits.clone(),
                condition: inst.condition.clone(),
            });
        }
        result
    }

    /// Repeat the circuit n times.
    pub fn repeat(&self, n: usize) -> Circuit {
        let mut result = self.clone();
        result.name = format!("{}_x{}", self.name, n);
        let qubits = self.all_qubits();
        let mut qmap = HashMap::new();
        for q in &qubits {
            qmap.insert(q.clone(), q.clone());
        }
        for _ in 1..n {
            result.append_circuit(self, &qmap, None);
        }
        result
    }

    /// Inverse (adjoint/dagger): reverse instruction order, negate rotation angles.
    pub fn inverse(&self) -> Circuit {
        let mut result = self.clone();
        result.name = format!("{}_dag", self.name);
        result.instructions.reverse();
        for inst in &mut result.instructions {
            // Negate rotation parameters for the adjoint
            if inst.gate.kind.is_rotation() {
                inst.gate.params = inst.gate.params.iter().map(|p| -p.clone()).collect();
            }
            // S → Sdg, T → Tdg
            inst.gate.kind = match inst.gate.kind {
                GateKind::S => GateKind::Sdg,
                GateKind::Sdg => GateKind::S,
                GateKind::T => GateKind::Tdg,
                GateKind::Tdg => GateKind::T,
                other => other,
            };
        }
        result
    }

    /// Collect all qubits in register order.
    fn all_qubits(&self) -> Vec<Qubit> {
        let mut qubits = Vec::new();
        for reg in &self.registers {
            if reg.kind == RegisterKind::Quantum {
                qubits.extend(reg.qubits());
            }
        }
        qubits
    }
}

// Operator overloads for Aria-style composition
// a >> b = sequential pipe
impl std::ops::Shr for Circuit {
    type Output = Circuit;
    fn shr(self, rhs: Circuit) -> Circuit {
        self.pipe(&rhs)
    }
}
// a | b = parallel tensor
impl std::ops::BitOr for Circuit {
    type Output = Circuit;
    fn bitor(self, rhs: Circuit) -> Circuit {
        self.tensor(&rhs)
    }
}

impl fmt::Display for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Circuit('{}', qubits={}, depth={})",
            self.name,
            self.n_qubits(),
            self.depth()
        )?;
        for inst in &self.instructions {
            writeln!(f, "  {inst}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_manual_construction() {
        let mut circ = Circuit::new("bell");
        let q = circ.qreg("q", 2);
        let c = circ.creg("c", 2);

        circ.apply(h(), vec![q[0].clone()]);
        circ.apply(cx(), vec![q[0].clone(), q[1].clone()]);
        circ.measure(&q[0], &c[0]);
        circ.measure(&q[1], &c[1]);

        assert_eq!(circ.n_qubits(), 2);
        assert_eq!(circ.n_clbits(), 2);
        assert_eq!(circ.gate_count(), 2);
        assert_eq!(circ.depth(), 2);
        assert_eq!(circ.instructions.len(), 4);
    }

    #[test]
    fn test_gate_kind_properties() {
        assert!(GateKind::H.is_self_inverse());
        assert!(GateKind::RZ.is_rotation());
        assert!(GateKind::Measure.is_meta());
        assert_eq!(GateKind::CX.n_qubits(), 2);
        assert_eq!(GateKind::CCX.n_qubits(), 3);
    }

    #[test]
    fn test_pipe_composition() {
        // H then CX on 2 qubits
        let mut a = Circuit::new("a");
        let qa = a.qreg("q", 2);
        a.apply(h(), vec![qa[0].clone()]);

        let mut b = Circuit::new("b");
        let qb = b.qreg("q", 2);
        b.apply(cx(), vec![qb[0].clone(), qb[1].clone()]);

        let piped = a.pipe(&b);
        assert_eq!(piped.gate_count(), 2); // H + CX
        assert_eq!(piped.n_qubits(), 2);
    }

    #[test]
    fn test_pipe_operator() {
        let mut a = Circuit::new("a");
        let qa = a.qreg("q", 1);
        a.apply(h(), vec![qa[0].clone()]);

        let mut b = Circuit::new("b");
        let qb = b.qreg("q", 1);
        b.apply(x(), vec![qb[0].clone()]);

        let piped = a >> b;
        assert_eq!(piped.gate_count(), 2);
    }

    #[test]
    fn test_tensor_composition() {
        let mut a = Circuit::new("a");
        let qa = a.qreg("q", 1);
        a.apply(h(), vec![qa[0].clone()]);

        let mut b = Circuit::new("b");
        let qb = b.qreg("q", 1);
        b.apply(x(), vec![qb[0].clone()]);

        let par = a.tensor(&b);
        assert_eq!(par.n_qubits(), 2);
        assert_eq!(par.gate_count(), 2);
    }

    #[test]
    fn test_repeat() {
        let mut circ = Circuit::new("layer");
        let q = circ.qreg("q", 1);
        circ.apply(h(), vec![q[0].clone()]);

        let repeated = circ.repeat(3);
        assert_eq!(repeated.gate_count(), 3);
    }

    #[test]
    fn test_inverse() {
        let mut circ = Circuit::new("fwd");
        let q = circ.qreg("q", 1);
        circ.apply(
            GateDef::with_params(GateKind::RZ, vec![1.5]),
            vec![q[0].clone()],
        );
        circ.apply(GateDef::new(GateKind::S), vec![q[0].clone()]);

        let inv = circ.inverse();
        assert_eq!(inv.gate_count(), 2);
        // Reversed order: S was last → now first (as Sdg), RZ was first → now second
        assert_eq!(inv.instructions[0].gate.kind, GateKind::Sdg);
        // RZ angle negated
        assert!((inv.instructions[1].gate.params[0].try_as_f64().unwrap() - (-1.5)).abs() < 1e-10);
    }

    #[test]
    fn test_circuit_composition() {
        let mut sub = Circuit::new("sub");
        let sq = sub.qreg("s", 1);
        sub.apply(h(), vec![sq[0].clone()]);

        let mut main = Circuit::new("main");
        let mq = main.qreg("q", 2);
        main.apply(x(), vec![mq[0].clone()]);

        let mut qmap = HashMap::new();
        qmap.insert(sq[0].clone(), mq[1].clone());
        main.append_circuit(&sub, &qmap, None);

        assert_eq!(main.gate_count(), 2);
        assert_eq!(main.instructions[1].gate.kind, GateKind::H);
        assert_eq!(main.instructions[1].qubits[0], mq[1]);
    }
}
