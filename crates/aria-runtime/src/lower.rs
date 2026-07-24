// SPDX-License-Identifier: Apache-2.0
//! Lower an Aria [`Circuit`] into omega-core's in-process [`CircuitIR`].
//!
//! Register-relative qubit/classical-bit addressing is flattened to global
//! indices (matching [`aria_core::backends::omega::to_omega_ir`]), and Aria's
//! symbolic [`ParamExpr`](aria_core::ast::ParamExpr) is mapped to omega's
//! `ParamExpr` so that **trainable parameters survive** as omega `SymbolId`s —
//! the basis for pure-Rust QML training.
//!
//! Gate mapping notes:
//! - `P(λ) → U1(λ)` and `CP(λ) → CU3(0, 0, λ)` are exact (verified against the
//!   omega statevector gate matrices: `u1(λ)=diag(1,e^{iλ})`,
//!   `cu3(θ,φ,λ)=diag(I₂, U3)`).
//! - `SX`, `RXX`, `RYY`, `RZZ` and the continuous-variable photonic gates
//!   (`BeamSplitter`, `PhaseShifter`, `Squeezing`, `Displacement`, `Kerr`) have
//!   no direct gate-model statevector equivalent and produce a clear error
//!   rather than a silently-wrong result. (Decompositions can be added later.)

use std::collections::HashMap;

use aria_core::ast::ParamExpr as AExpr;
use aria_core::ast::{Circuit, Clbit, GateKind as AKind, Qubit, RegisterKind};

use omega_core::circuit::{
    CircuitIR, CircuitType, GateKind as OKind, GateOp, ParamExpr as OExpr, Qubit as OQubit,
    SymbolId,
};

/// A lowered circuit plus the symbol table linking Aria parameter names to the
/// omega `SymbolId`s used inside [`CircuitIR::symbols`].
///
/// # Binding-order contract (stable)
///
/// `SymbolId`s are assigned in **first-appearance order** as the lowering
/// walks the circuit, so they form a contiguous `0..symbol_ids.len()` range.
/// Consumers that pass parameters as a flat slice (the wasm transport, the
/// verify-core oracle) MUST order that slice by **ascending `SymbolId`** —
/// `params[id]` is the value bound to the symbol with that id. `symbol_ids`
/// is the authoritative name → id map; never assume a name's id from source
/// order, always look it up. This ordering is a tested guarantee
/// (`aria-runtime/tests/symbol_order.rs`), not an incidental behaviour.
pub struct Lowered {
    pub ir: CircuitIR,
    /// Aria symbol name → omega `SymbolId` (ids are dense, `0..len`, in
    /// first-appearance order — see the type-level binding-order contract).
    pub symbol_ids: HashMap<String, SymbolId>,
    /// True when a non-meta gate follows a measurement (projective
    /// mid-circuit semantics required at execution time).
    pub needs_collapse: bool,
}

#[derive(Default)]
struct SymTab {
    ids: HashMap<String, SymbolId>,
    next: SymbolId,
}

impl SymTab {
    fn id(&mut self, name: &str) -> SymbolId {
        if let Some(i) = self.ids.get(name) {
            return *i;
        }
        let i = self.next;
        self.next += 1;
        self.ids.insert(name.to_string(), i);
        i
    }
}

/// Convert one Aria parameter expression to an omega parameter expression,
/// allocating `SymbolId`s for free symbols. Fully-concrete sub-trees (incl.
/// `Pi`, concrete `Div`/`FnCall`) collapse to `Concrete`.
fn conv(p: &AExpr, syms: &mut SymTab) -> Result<OExpr, String> {
    if p.is_concrete() {
        let v = p
            .try_as_f64()
            .ok_or_else(|| format!("non-finite concrete parameter: {p}"))?;
        return Ok(OExpr::Concrete(v));
    }
    match p {
        AExpr::Symbol(name) => Ok(OExpr::Symbol(syms.id(name))),
        AExpr::Neg(a) => Ok(OExpr::Negate(Box::new(conv(a, syms)?))),
        AExpr::Add(a, b) => Ok(OExpr::Add(
            Box::new(conv(a, syms)?),
            Box::new(conv(b, syms)?),
        )),
        AExpr::Mul(a, b) => Ok(OExpr::Mul(
            Box::new(conv(a, syms)?),
            Box::new(conv(b, syms)?),
        )),
        // a / c with concrete c becomes a * (1/c) — omega has no Div node.
        AExpr::Div(a, b) if b.is_concrete() => {
            let d = b
                .try_as_f64()
                .ok_or_else(|| format!("non-finite divisor: {b}"))?;
            if d == 0.0 {
                return Err("division by zero in parameter".into());
            }
            Ok(OExpr::Mul(
                Box::new(conv(a, syms)?),
                Box::new(OExpr::Concrete(1.0 / d)),
            ))
        }
        other => Err(format!(
            "symbolic parameter not representable in omega ParamExpr: {other}"
        )),
    }
}

fn conv_all(params: &[AExpr], syms: &mut SymTab) -> Result<Vec<OExpr>, String> {
    params.iter().map(|p| conv(p, syms)).collect()
}

/// Lower an Aria [`Circuit`] to an omega [`CircuitIR`].
pub fn lower(circuit: &Circuit) -> Result<Lowered, String> {
    // Global qubit / classical-bit index maps.
    let mut qubit_map: HashMap<Qubit, u32> = HashMap::new();
    let mut clbit_map: HashMap<Clbit, u32> = HashMap::new();
    let mut q_idx = 0u32;
    let mut c_idx = 0u32;
    for reg in &circuit.registers {
        match reg.kind {
            RegisterKind::Quantum => {
                for i in 0..reg.size {
                    qubit_map.insert(Qubit::new(&reg.name, i), q_idx);
                    q_idx += 1;
                }
            }
            RegisterKind::Classical => {
                for i in 0..reg.size {
                    clbit_map.insert(Clbit::new(&reg.name, i), c_idx);
                    c_idx += 1;
                }
            }
        }
    }

    let mut syms = SymTab::default();
    let mut ops: Vec<GateOp> = Vec::with_capacity(circuit.instructions.len());

    for inst in &circuit.instructions {
        let p = &inst.gate.params;
        let (gate, params): (OKind, Vec<OExpr>) = match inst.gate.kind {
            AKind::I => (OKind::Id, vec![]),
            AKind::X => (OKind::X, vec![]),
            AKind::Y => (OKind::Y, vec![]),
            AKind::Z => (OKind::Z, vec![]),
            AKind::H => (OKind::H, vec![]),
            AKind::S => (OKind::S, vec![]),
            AKind::Sdg => (OKind::Sdg, vec![]),
            AKind::T => (OKind::T, vec![]),
            AKind::Tdg => (OKind::Tdg, vec![]),
            AKind::RX => (OKind::Rx, conv_all(p, &mut syms)?),
            AKind::RY => (OKind::Ry, conv_all(p, &mut syms)?),
            AKind::RZ => (OKind::Rz, conv_all(p, &mut syms)?),
            AKind::P => (OKind::U1, conv_all(p, &mut syms)?),
            AKind::U => (OKind::U3, conv_all(p, &mut syms)?),
            AKind::CX => (OKind::CX, vec![]),
            AKind::CY => (OKind::CY, vec![]),
            AKind::CZ => (OKind::CZ, vec![]),
            AKind::SWAP => (OKind::Swap, vec![]),
            // CP(λ) == controlled-U1(λ) == CU3(0, 0, λ).
            AKind::CP => {
                let lam = conv(p.first().ok_or("CP requires 1 parameter")?, &mut syms)?;
                (
                    OKind::CU3,
                    vec![OExpr::Concrete(0.0), OExpr::Concrete(0.0), lam],
                )
            }
            AKind::RBS => (OKind::Rbs, conv_all(p, &mut syms)?),
            AKind::CCX => (OKind::CCX, vec![]),
            AKind::CSWAP => (OKind::CSwap, vec![]),
            AKind::Barrier => (OKind::Barrier, vec![]),
            AKind::Reset => (OKind::Reset, vec![]),
            AKind::Measure => (OKind::Measure, vec![]),
            other => {
                return Err(format!(
                    "gate {other:?} has no gate-model statevector lowering (Aria run \
                     supports the universal gate set; decompose or use a photonic backend)"
                ));
            }
        };

        let qubits: Vec<OQubit> = inst
            .qubits
            .iter()
            .map(|q| {
                qubit_map
                    .get(q)
                    .copied()
                    .map(OQubit)
                    .ok_or_else(|| format!("unknown qubit {q:?} in gate {gate:?}"))
            })
            .collect::<Result<_, _>>()?;

        let classical_bit = inst.clbits.first().and_then(|c| clbit_map.get(c).copied());

        let condition = match &inst.condition {
            Some((cl, v)) => {
                let idx = clbit_map
                    .get(cl)
                    .copied()
                    .ok_or_else(|| format!("unknown classical bit {cl:?} in condition"))?;
                Some((idx, 1u32, *v))
            }
            None => None,
        };

        ops.push(GateOp {
            gate,
            qubits: qubits.into(),
            params: params.into(),
            classical_bit,
            condition,
        });
    }

    // Projective mid-circuit semantics are required iff a non-meta gate
    // follows the first measurement.
    let needs_collapse = circuit
        .instructions
        .iter()
        .position(|i| i.gate.kind == AKind::Measure)
        .map(|pos| {
            circuit.instructions[pos + 1..]
                .iter()
                .any(|i| !is_meta(i.gate.kind))
        })
        .unwrap_or(false);

    let mut ir = CircuitIR::new(circuit.n_qubits() as u32, CircuitType::GateBased);
    ir.num_classical_bits = circuit.n_clbits() as u32;
    ir.ops = ops;
    ir.symbols = syms.ids.iter().map(|(n, i)| (*i, n.clone())).collect();

    Ok(Lowered {
        ir,
        symbol_ids: syms.ids,
        needs_collapse,
    })
}

fn is_meta(k: AKind) -> bool {
    matches!(k, AKind::Barrier | AKind::Measure | AKind::I)
}
