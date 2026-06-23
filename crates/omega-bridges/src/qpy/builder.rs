//! Map a fully-decoded QPY blob to omega's
//! [`omega_core::circuit::CircuitIR`].
//!
//! The chain:
//!
//! ```text
//! bytes → file_header → program_table → circuit_header →
//!         circuit_body → register_table → circuit_extras →
//!         instructions → CircuitIR
//! ```
//!
//! Today this builder handles single-circuit blobs (`num_circuits ==
//! 1`) with the gate set omega represents natively (H/X/Y/Z/S/Sdg/T
//! /Tdg/Id, Rx/Ry/Rz, U1/U2/U3, CX/CY/CZ/Swap, CRz/CU3, CCX/CSwap,
//! Measure, Reset). Anything else surfaces as
//! `QpyError::UnsupportedGate { name }` so the caller can fall back
//! to the `qpy_to_qasm2` subprocess.
//!
//! Parameters and `ParameterExpression` payloads (`b'p'` / `b'e'`)
//! decode to omega's [`ParamExpr`] tree. The PE evaluator supports
//! ADD / SUB / MUL / RSUB plus DIV / RDIV by a concrete divisor;
//! POW / trig / EXP / LOG / SUBSTITUTE and the substitution-marker
//! framing types (`b's'` / `b'e'` / `b'u'`) surface as `Unsupported`
//! so the caller can fall back to the qpy_to_qasm2 subprocess.

use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use smallvec::SmallVec;

use super::circuit_body::read_circuit_body;
use super::circuit_extras::read_circuit_extras;
use super::circuit_header::{read_circuit_header, FIXED_LEN as CIRCUIT_HEADER_LEN};
use super::header::read_header;
use super::instruction::read_circuit_instructions;
use super::instruction_args::InstructionArg;
use super::instruction_header::ConditionKind;
use super::instruction_params::InstructionParam;
use super::parameter_expression::{
    DecodedParameterExpression, ParamExprOpCode, ParamExprOperand, ParamExprSymbolKind,
};
use super::program_table::read_program_table;
use super::registers::read_register_table;
use super::QpyError;

/// Top-level entry point: parse a QPY blob and return omega's
/// `CircuitIR` for its first circuit. Asserts `num_circuits == 1`;
/// for multi-circuit blobs use [`read_qpy_circuit_irs`] instead.
pub fn read_qpy_circuit_ir(bytes: &[u8]) -> Result<CircuitIR, QpyError> {
    let mut all = read_qpy_circuit_irs(bytes)?;
    if all.len() != 1 {
        return Err(QpyError::Unsupported {
            what: "read_qpy_circuit_ir on a multi-circuit blob",
            detail: "use read_qpy_circuit_irs to receive all circuits; this entry takes the single-circuit case",
        });
    }
    Ok(all.remove(0))
}

/// Multi-circuit variant: returns a `Vec<CircuitIR>` with one entry
/// per circuit in the blob (preserving the file's order).
pub fn read_qpy_circuit_irs(bytes: &[u8]) -> Result<Vec<CircuitIR>, QpyError> {
    let h = read_header(bytes)?;
    let t = read_program_table(bytes, &h)?;
    let mut out = Vec::with_capacity(t.circuit_offsets.len());

    if t.circuit_offsets.is_empty() {
        // Pre-V16 blob — circuits are written sequentially with no
        // offset table. We don't have a fixture for this layout
        // today; surface as Unsupported until one shows up.
        if h.qpy_version < 16 {
            return Err(QpyError::Unsupported {
                what: "pre-V16 sequential circuit layout",
                detail: "the pure-Rust reader needs a circuit-walker for the no-offset-table path; only V16+ tested today",
            });
        }
        return Ok(out);
    }

    for &offset in &t.circuit_offsets {
        out.push(read_one_circuit_ir(bytes, &h, offset as usize)?);
    }
    Ok(out)
}

fn read_one_circuit_ir(
    bytes: &[u8],
    file_header: &super::header::QpyHeader,
    circ_off: usize,
) -> Result<CircuitIR, QpyError> {
    let ch = read_circuit_header(bytes, circ_off)?;
    let body_off = circ_off + CIRCUIT_HEADER_LEN;
    let body = read_circuit_body(bytes, &ch, body_off)?;
    let regs_off = body_off + body.body_len;
    let regs = read_register_table(bytes, ch.num_registers, regs_off)?;
    let extras_off = regs_off + regs.table_len;
    let extras = read_circuit_extras(bytes, file_header, extras_off)?;
    let inst_off = extras_off + extras.extras_len;
    let (insts, _consumed) = read_circuit_instructions(bytes, ch.num_instructions, inst_off)?;

    let mut ir = CircuitIR::new(ch.num_qubits, CircuitType::GateBased);
    ir.num_classical_bits = ch.num_clbits;
    // Symbol-id table: deterministic, allocated in encounter order
    // and shared across instructions so the same Parameter UUID
    // always resolves to the same `SymbolId`.
    let mut sym_lookup = SymbolLookup::default();
    for inst in &insts {
        ir.ops
            .push(decoded_to_gate_op(inst, &mut ir.symbols, &mut sym_lookup)?);
    }
    Ok(ir)
}

/// UUID → `SymbolId` lookup so a `Parameter('θ')` referenced by N
/// instructions produces N references to the same omega symbol.
#[derive(Default)]
struct SymbolLookup {
    by_uuid: std::collections::HashMap<[u8; 16], omega_core::circuit::SymbolId>,
    next_id: omega_core::circuit::SymbolId,
}

impl SymbolLookup {
    fn intern(
        &mut self,
        uuid: [u8; 16],
        name: &str,
        symbols: &mut std::collections::HashMap<omega_core::circuit::SymbolId, String>,
    ) -> omega_core::circuit::SymbolId {
        if let Some(&id) = self.by_uuid.get(&uuid) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.by_uuid.insert(uuid, id);
        symbols.insert(id, name.to_owned());
        id
    }
}

/// Map a Qiskit gate class name to omega's `GateKind`.
pub fn gate_name_to_kind(name: &str) -> Result<GateKind, QpyError> {
    Ok(match name {
        "HGate" => GateKind::H,
        "XGate" => GateKind::X,
        "YGate" => GateKind::Y,
        "ZGate" => GateKind::Z,
        "SGate" => GateKind::S,
        "SdgGate" => GateKind::Sdg,
        "TGate" => GateKind::T,
        "TdgGate" => GateKind::Tdg,
        "IGate" | "IdGate" => GateKind::Id,
        "RXGate" => GateKind::Rx,
        "RYGate" => GateKind::Ry,
        "RZGate" => GateKind::Rz,
        "U1Gate" | "PhaseGate" => GateKind::U1,
        "U2Gate" => GateKind::U2,
        "U3Gate" | "UGate" => GateKind::U3,
        "CXGate" | "CNOTGate" => GateKind::CX,
        "CYGate" => GateKind::CY,
        "CZGate" => GateKind::CZ,
        "SwapGate" => GateKind::Swap,
        "CRZGate" => GateKind::CRz,
        "CU3Gate" => GateKind::CU3,
        "CCXGate" | "ToffoliGate" => GateKind::CCX,
        "CSwapGate" | "FredkinGate" => GateKind::CSwap,
        "Measure" => GateKind::Measure,
        "Barrier" => GateKind::Barrier,
        "Reset" => GateKind::Reset,
        other => {
            return Err(QpyError::UnsupportedGate {
                name: other.to_owned(),
            })
        }
    })
}

fn decoded_to_gate_op(
    inst: &super::instruction::DecodedInstruction<'_>,
    symbols: &mut std::collections::HashMap<omega_core::circuit::SymbolId, String>,
    sym_lookup: &mut SymbolLookup,
) -> Result<GateOp, QpyError> {
    let omega_condition = match inst.condition {
        ConditionKind::None => None,
        ConditionKind::Bit => Some(decode_bit_condition(inst)?),
        ConditionKind::RegisterOrExpr => {
            return Err(QpyError::Unsupported {
                what: "instruction.condition (register / expression variant)",
                detail: "multi-bit register conditions need a register table; classical-expression conditions need the symbolic INSTRUCTION_PARAM slice",
            });
        }
    };

    // Some Qiskit gates omega doesn't represent natively can be
    // decomposed into ones it does — at the cost of a (usually
    // unobservable) global phase that omega's `CircuitIR` doesn't
    // track today. Run those substitutions before consulting
    // `gate_name_to_kind`.
    if let Some(gate_op) = decompose_known_alias(inst)? {
        return Ok(gate_op);
    }

    let gate = gate_name_to_kind(inst.name)?;

    // Split args: qargs first then cargs. omega's GateOp puts qubits
    // in `qubits`; for Measure the single carg goes into `classical_bit`.
    let mut qubits: SmallVec<[Qubit; 3]> = SmallVec::new();
    let mut clbit: Option<u32> = None;
    for (i, a) in inst.args.iter().enumerate() {
        let is_q = i < inst.num_qargs as usize;
        match (is_q, a) {
            (true, InstructionArg::Qubit(q)) => qubits.push(Qubit(*q)),
            (false, InstructionArg::Clbit(c)) => {
                if matches!(gate, GateKind::Measure) {
                    if clbit.is_some() {
                        return Err(QpyError::Unsupported {
                            what: "Measure with > 1 classical args",
                            detail: "omega's GateOp supports a single classical_bit; multi-clbit measures need expansion",
                        });
                    }
                    clbit = Some(*c);
                } else {
                    return Err(QpyError::Unsupported {
                        what: "non-Measure instructions with classical args",
                        detail: "today omega only models cargs on Measure",
                    });
                }
            }
            _ => {
                return Err(QpyError::InstructionArgKindMismatch {
                    expected_kind: if is_q { 'q' } else { 'c' },
                    position: i,
                });
            }
        }
    }

    let mut params: SmallVec<[ParamExpr; 3]> = SmallVec::new();
    for p in &inst.params {
        params.push(param_to_expr(p, symbols, sym_lookup)?);
    }

    Ok(GateOp {
        gate,
        qubits,
        params,
        classical_bit: clbit,
        condition: omega_condition,
    })
}

/// Decode a Qiskit TWO_TUPLE-with-single-clbit condition payload into
/// omega's `(start_bit, num_bits, expected)`. Qiskit tags single-clbit
/// conditions with a `condition_register` of `b"\x00<decimal_index>"`;
/// register-name conditions (where the payload is the register's UTF-8
/// name) require an in-memory register table that omega's reader
/// doesn't carry today, so they surface as `Unsupported` here.
fn decode_bit_condition(
    inst: &super::instruction::DecodedInstruction<'_>,
) -> Result<(u32, u32, u64), QpyError> {
    let bytes = &inst.condition_register;
    if bytes.first() != Some(&0u8) {
        return Err(QpyError::Unsupported {
            what: "instruction.condition with register-name payload",
            detail: "non-null-prefixed condition_register references a named ClassicalRegister; the reader needs the register table to resolve it",
        });
    }
    let idx_str = std::str::from_utf8(&bytes[1..]).map_err(|e| QpyError::InvalidUtf8 {
        what: "instruction.condition_register single-clbit index",
        valid_up_to: e.valid_up_to(),
        len: bytes.len() - 1,
    })?;
    let start_bit: u32 = idx_str.parse().map_err(|_| QpyError::Unsupported {
        what: "instruction.condition_register non-integer payload",
        detail: "single-clbit condition_register must be \\x00 followed by a base-10 clbit index",
    })?;
    Ok((start_bit, 1, inst.condition_value as u64))
}

/// Map alias gates that omega doesn't have natively but can be
/// expressed via U3. Returns `Ok(Some(_))` when the alias was
/// decomposed, `Ok(None)` when the gate isn't an alias, and
/// `Err(_)` when the alias's args/params are malformed. Global
/// phase is dropped — omega's `CircuitIR` doesn't track it and
/// it's unobservable for measurement-based outputs.
fn decompose_known_alias(
    inst: &super::instruction::DecodedInstruction<'_>,
) -> Result<Option<GateOp>, QpyError> {
    use std::f64::consts::FRAC_PI_2;

    // Both alias gates here are 1-qubit zero-param decompositions
    // into U3 with concrete angles.
    let (theta, phi, lambda) = match inst.name {
        // SX = sqrt(X) ≃ U3(π/2, -π/2, π/2). Differs from sqrt(X)
        // by a global phase of e^(iπ/4).
        "SXGate" => (FRAC_PI_2, -FRAC_PI_2, FRAC_PI_2),
        // SXdg = sqrt(X)† ≃ U3(π/2, π/2, -π/2). Same comment on
        // global phase.
        "SXdgGate" => (FRAC_PI_2, FRAC_PI_2, -FRAC_PI_2),
        _ => return Ok(None),
    };
    let total_args = inst.args.len();
    if inst.num_qargs != 1 || total_args != 1 {
        return Err(QpyError::InstructionArgKindMismatch {
            expected_kind: 'q',
            position: 0,
        });
    }
    if !inst.params.is_empty() {
        return Err(QpyError::Unsupported {
            what: "alias-decomposed gate with non-empty params",
            detail: "SXGate / SXdgGate take no parameters; producer wrote some",
        });
    }
    let qubit = match inst.args.first() {
        Some(InstructionArg::Qubit(q)) => Qubit(*q),
        _ => {
            return Err(QpyError::InstructionArgKindMismatch {
                expected_kind: 'q',
                position: 0,
            })
        }
    };
    let mut qubits: SmallVec<[Qubit; 3]> = SmallVec::new();
    qubits.push(qubit);
    let mut params: SmallVec<[ParamExpr; 3]> = SmallVec::new();
    params.push(ParamExpr::Concrete(theta));
    params.push(ParamExpr::Concrete(phi));
    params.push(ParamExpr::Concrete(lambda));
    Ok(Some(GateOp {
        gate: GateKind::U3,
        qubits,
        params,
        classical_bit: None,
        condition: None,
    }))
}

fn param_to_expr(
    p: &InstructionParam,
    symbols: &mut std::collections::HashMap<omega_core::circuit::SymbolId, String>,
    sym_lookup: &mut SymbolLookup,
) -> Result<ParamExpr, QpyError> {
    match p {
        InstructionParam::Float(f) => Ok(ParamExpr::Concrete(*f)),
        InstructionParam::Integer { value, .. } => Ok(ParamExpr::Concrete(*value as f64)),
        InstructionParam::Complex { re, im } => {
            if *im != 0.0 {
                return Err(QpyError::Unsupported {
                    what: "non-real complex INSTRUCTION_PARAM",
                    detail: "omega's ParamExpr is real-valued; only zero-imaginary complex params are accepted today",
                });
            }
            Ok(ParamExpr::Concrete(*re))
        }
        InstructionParam::Parameter { uuid, name } => {
            // Same UUID across instructions resolves to the same
            // omega SymbolId so binding works correctly.
            let id = sym_lookup.intern(*uuid, name, symbols);
            Ok(ParamExpr::Symbol(id))
        }
        InstructionParam::ParameterVectorElement {
            uuid,
            vector_name,
            index,
            ..
        } => {
            // Each indexed element gets its own SymbolId, keyed off
            // the element's UUID (which differs per index, by the
            // `element.uuid = root_uuid + index` convention). The
            // SymbolLookup contract is identical to the plain
            // Parameter case — same UUID across instructions, same
            // SymbolId in the IR.
            let display_name = format!("{vector_name}[{index}]");
            let id = sym_lookup.intern(*uuid, &display_name, symbols);
            Ok(ParamExpr::Symbol(id))
        }
        InstructionParam::ParameterExpression(pe) => {
            evaluate_parameter_expression(pe, symbols, sym_lookup)
        }
        InstructionParam::Null => Err(QpyError::Unsupported {
            what: "InstructionParam::Null as a gate parameter",
            detail: "omega's ParamExpr requires a numeric or symbolic value",
        }),
        InstructionParam::CaseDefault => Err(QpyError::Unsupported {
            what: "InstructionParam::CaseDefault as a gate parameter",
            detail: "switch-case sentinel, not valid as a gate angle",
        }),
        InstructionParam::String(_) => Err(QpyError::Unsupported {
            what: "InstructionParam::String as a gate parameter",
            detail: "omega's ParamExpr requires a numeric or symbolic value",
        }),
        InstructionParam::Opaque { .. } => Err(QpyError::Unsupported {
            what: "opaque INSTRUCTION_PARAM payload as a gate parameter",
            detail: "non-scalar parameter type — per-type decoder lands in a follow-up",
        }),
    }
}

/// Fold a decoded `ParameterExpression` (a stack-machine program) into
/// omega's `ParamExpr`. Supports the algebraic ops omega's IR can
/// represent natively: ADD, SUB, MUL, RSUB, DIV-by-constant, and
/// RDIV-with-constant-numerator. POW / RPOW / SUBSTITUTE / GRAD /
/// trig and other transcendental ops fall outside `ParamExpr` and
/// surface as `Unsupported` so the caller can fall back to the
/// Qiskit-subprocess path.
///
/// Mirrors Qiskit's `_read_parameter_expr_v13` interpreter: each
/// record pushes its LHS / RHS to the stack (skipping `None`) and
/// then, if `op_code != 255`, pops the operands and pushes the
/// applied result. Substitution markers (`b's'` / `b'e'` / `b'u'`)
/// are not yet decoded — surfaced as Unsupported.
fn evaluate_parameter_expression(
    pe: &DecodedParameterExpression,
    symbols: &mut std::collections::HashMap<omega_core::circuit::SymbolId, String>,
    sym_lookup: &mut SymbolLookup,
) -> Result<ParamExpr, QpyError> {
    // Intern every Parameter UUID the PE references so the rest of
    // the circuit's instructions reach the same SymbolId on lookup.
    // ParameterVector elements get the same treatment — each indexed
    // element has its own UUID and is interned with display name
    // `"<vector_name>[<index>]"` to match the plain b'v'
    // INSTRUCTION_PARAM path.
    let mut uuid_to_sym: std::collections::HashMap<[u8; 16], omega_core::circuit::SymbolId> =
        std::collections::HashMap::with_capacity(pe.symbols.len());
    for sym in &pe.symbols {
        let display_name = match &sym.kind {
            ParamExprSymbolKind::Parameter { name } => name.clone(),
            ParamExprSymbolKind::VectorElement {
                vector_name, index, ..
            } => format!("{vector_name}[{index}]"),
        };
        let id = sym_lookup.intern(sym.uuid, &display_name, symbols);
        uuid_to_sym.insert(sym.uuid, id);
    }

    let mut stack: Vec<ParamExpr> = Vec::with_capacity(pe.elements.len() * 2);
    for elem in &pe.elements {
        push_operand(&elem.lhs, &uuid_to_sym, &mut stack)?;
        push_operand(&elem.rhs, &uuid_to_sym, &mut stack)?;

        if elem.op_code == 255 {
            // Wire-level no-op sentinel — stack has whatever the
            // record pushed; the apply step is intentionally skipped.
            continue;
        }

        let op = ParamExprOpCode::from_u8(elem.op_code).ok_or(QpyError::Unsupported {
            what: "ParameterExpression with unrecognised op_code byte",
            detail: "the QPY producer emitted an op_code outside qiskit's OpCode enum",
        })?;

        match op {
            ParamExprOpCode::Add => {
                let rhs = pop(&mut stack)?;
                let lhs = pop(&mut stack)?;
                stack.push(ParamExpr::Add(Box::new(lhs), Box::new(rhs)));
            }
            ParamExprOpCode::Sub => {
                let rhs = pop(&mut stack)?;
                let lhs = pop(&mut stack)?;
                stack.push(ParamExpr::Add(
                    Box::new(lhs),
                    Box::new(ParamExpr::Negate(Box::new(rhs))),
                ));
            }
            ParamExprOpCode::Mul => {
                let rhs = pop(&mut stack)?;
                let lhs = pop(&mut stack)?;
                stack.push(ParamExpr::Mul(Box::new(lhs), Box::new(rhs)));
            }
            ParamExprOpCode::Rsub => {
                // `lhs.__rsub__(rhs)` = rhs - lhs.
                let rhs = pop(&mut stack)?;
                let lhs = pop(&mut stack)?;
                stack.push(ParamExpr::Add(
                    Box::new(rhs),
                    Box::new(ParamExpr::Negate(Box::new(lhs))),
                ));
            }
            ParamExprOpCode::Div => {
                // omega's `ParamExpr` has no `Div` node — only safe
                // when the divisor is a known concrete value, which
                // collapses to `Mul(numerator, 1/divisor)`.
                let rhs = pop(&mut stack)?;
                let lhs = pop(&mut stack)?;
                let div_by_const = match &rhs {
                    ParamExpr::Concrete(d) if *d != 0.0 => Some(1.0 / *d),
                    _ => None,
                };
                if let Some(reciprocal) = div_by_const {
                    stack.push(ParamExpr::Mul(
                        Box::new(lhs),
                        Box::new(ParamExpr::Concrete(reciprocal)),
                    ));
                } else {
                    return Err(QpyError::Unsupported {
                        what: "ParameterExpression DIV with non-constant or zero divisor",
                        detail: "omega's ParamExpr lacks a Div node; constant divisors collapse to Mul-by-reciprocal",
                    });
                }
            }
            ParamExprOpCode::Rdiv => {
                // `lhs.__rdiv__(rhs)` = rhs / lhs. Only safe when the
                // divisor (lhs after the operands are popped) is a
                // concrete value.
                let rhs = pop(&mut stack)?;
                let lhs = pop(&mut stack)?;
                let div_by_const = match &lhs {
                    ParamExpr::Concrete(d) if *d != 0.0 => Some(1.0 / *d),
                    _ => None,
                };
                if let Some(reciprocal) = div_by_const {
                    stack.push(ParamExpr::Mul(
                        Box::new(rhs),
                        Box::new(ParamExpr::Concrete(reciprocal)),
                    ));
                } else {
                    return Err(QpyError::Unsupported {
                        what: "ParameterExpression RDIV with non-constant divisor",
                        detail: "omega's ParamExpr lacks a Div node; only concrete divisors collapse cleanly",
                    });
                }
            }
            ParamExprOpCode::NoOp => {
                // Unreachable — handled before this match by the
                // raw-byte 255 check above. Kept here for exhaustiveness.
                continue;
            }
            ParamExprOpCode::Pow
            | ParamExprOpCode::Rpow
            | ParamExprOpCode::Sin
            | ParamExprOpCode::Cos
            | ParamExprOpCode::Tan
            | ParamExprOpCode::Asin
            | ParamExprOpCode::Acos
            | ParamExprOpCode::Atan
            | ParamExprOpCode::Exp
            | ParamExprOpCode::Log
            | ParamExprOpCode::Abs
            | ParamExprOpCode::Conj
            | ParamExprOpCode::Sign
            | ParamExprOpCode::Grad
            | ParamExprOpCode::Substitute => {
                return Err(QpyError::Unsupported {
                    what: "ParameterExpression op outside omega's algebraic ParamExpr surface",
                    detail: "POW / trig / EXP / LOG / ABS / CONJ / SIGN / GRAD / SUBSTITUTE need a richer ParamExpr — fall back to qpy_to_qasm2",
                });
            }
        }
    }

    if stack.len() != 1 {
        return Err(QpyError::Unsupported {
            what: "ParameterExpression evaluator finished with non-singleton stack",
            detail: "expected a single residual expression on the replay stack — the program is either malformed or uses substitution sequences not yet supported",
        });
    }
    Ok(stack.pop().expect("len == 1 just checked above"))
}

fn push_operand(
    operand: &ParamExprOperand,
    uuid_to_sym: &std::collections::HashMap<[u8; 16], omega_core::circuit::SymbolId>,
    stack: &mut Vec<ParamExpr>,
) -> Result<(), QpyError> {
    match operand {
        ParamExprOperand::Float(f) => {
            stack.push(ParamExpr::Concrete(*f));
            Ok(())
        }
        ParamExprOperand::Int(i) => {
            stack.push(ParamExpr::Concrete(*i as f64));
            Ok(())
        }
        ParamExprOperand::Complex { re, im } => {
            if *im != 0.0 {
                return Err(QpyError::Unsupported {
                    what: "ParameterExpression with non-real complex literal",
                    detail: "omega's ParamExpr is real-valued; only zero-imaginary complex constants accepted",
                });
            }
            stack.push(ParamExpr::Concrete(*re));
            Ok(())
        }
        ParamExprOperand::ParameterUuid(uuid) => {
            let id = uuid_to_sym.get(uuid).ok_or(QpyError::Unsupported {
                what: "ParameterExpression references a UUID not in its symbol map",
                detail: "the producer emitted a Parameter that wasn't declared in the trailing PARAM_EXPR_MAP_ELEM_V3 table",
            })?;
            stack.push(ParamExpr::Symbol(*id));
            Ok(())
        }
        ParamExprOperand::None => {
            // `b'n'` — operand comes from a previous record's stack
            // residual; nothing to push.
            Ok(())
        }
        ParamExprOperand::SubstitutionMarker { .. } | ParamExprOperand::SubstitutionMapSize(_) => {
            Err(QpyError::Unsupported {
                what: "ParameterExpression with substitution markers (b's' / b'e' / b'u')",
                detail: "subs sequences need a follow-on decoder; route through qpy_to_qasm2 today",
            })
        }
    }
}

fn pop(stack: &mut Vec<ParamExpr>) -> Result<ParamExpr, QpyError> {
    stack.pop().ok_or(QpyError::Unsupported {
        what: "ParameterExpression replay stack underflow",
        detail: "the producer emitted a binary op without enough preceding operand pushes — corrupt blob or a feature the reader doesn't yet handle",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_common_qiskit_gates_to_omega_kinds() {
        assert_eq!(gate_name_to_kind("HGate").unwrap(), GateKind::H);
        assert_eq!(gate_name_to_kind("CXGate").unwrap(), GateKind::CX);
        assert_eq!(gate_name_to_kind("CNOTGate").unwrap(), GateKind::CX);
        assert_eq!(gate_name_to_kind("RXGate").unwrap(), GateKind::Rx);
        assert_eq!(gate_name_to_kind("Measure").unwrap(), GateKind::Measure);
        assert_eq!(gate_name_to_kind("Reset").unwrap(), GateKind::Reset);
        assert_eq!(gate_name_to_kind("CCXGate").unwrap(), GateKind::CCX);
        assert_eq!(gate_name_to_kind("ToffoliGate").unwrap(), GateKind::CCX);
        assert_eq!(gate_name_to_kind("UGate").unwrap(), GateKind::U3);
        // PhaseGate ≡ U1Gate — same semantic, just a different
        // class name in qiskit (qc.p(θ, q) emits PhaseGate).
        assert_eq!(gate_name_to_kind("PhaseGate").unwrap(), GateKind::U1);
        assert_eq!(gate_name_to_kind("U1Gate").unwrap(), GateKind::U1);
    }

    #[test]
    fn rejects_unknown_gate_with_typed_error_carrying_the_name() {
        match gate_name_to_kind("RXXGate").unwrap_err() {
            QpyError::UnsupportedGate { name } => assert_eq!(name, "RXXGate"),
            other => panic!("expected UnsupportedGate, got {other:?}"),
        }
    }

    // -------- ParameterExpression evaluator --------

    use super::super::parameter_expression::{
        DecodedParameterExpression, ParamExprElement, ParamExprOperand, ParamExprSymbol,
        ParamExprSymbolKind,
    };

    fn theta_uuid() -> [u8; 16] {
        [0xAA; 16]
    }

    fn pe_with_theta(elements: Vec<ParamExprElement>) -> DecodedParameterExpression {
        DecodedParameterExpression {
            symbols: vec![ParamExprSymbol {
                kind: ParamExprSymbolKind::Parameter {
                    name: "θ".to_owned(),
                },
                value_type: b'p',
                uuid: theta_uuid(),
                value_bytes: vec![],
            }],
            elements,
        }
    }

    fn fold(pe: &DecodedParameterExpression) -> Result<ParamExpr, QpyError> {
        let mut symbols = std::collections::HashMap::new();
        let mut sym_lookup = SymbolLookup::default();
        evaluate_parameter_expression(pe, &mut symbols, &mut sym_lookup)
    }

    #[test]
    fn evaluator_folds_add_float_parameter_into_paramexpr_add() {
        let pe = pe_with_theta(vec![ParamExprElement {
            op_code: 0, // ADD
            lhs: ParamExprOperand::Float(1.5),
            rhs: ParamExprOperand::ParameterUuid(theta_uuid()),
        }]);
        match fold(&pe).unwrap() {
            ParamExpr::Add(lhs, rhs) => {
                assert!(matches!(*lhs, ParamExpr::Concrete(f) if f == 1.5));
                assert!(matches!(*rhs, ParamExpr::Symbol(_)));
            }
            other => panic!("expected Add, got {other:?}"),
        }
    }

    #[test]
    fn evaluator_folds_sub_to_add_negate() {
        let pe = pe_with_theta(vec![ParamExprElement {
            op_code: 1, // SUB → lhs - rhs = Add(lhs, Negate(rhs))
            lhs: ParamExprOperand::ParameterUuid(theta_uuid()),
            rhs: ParamExprOperand::Float(2.0),
        }]);
        match fold(&pe).unwrap() {
            ParamExpr::Add(lhs, rhs) => {
                assert!(matches!(*lhs, ParamExpr::Symbol(_)));
                match *rhs {
                    ParamExpr::Negate(inner) => {
                        assert!(matches!(*inner, ParamExpr::Concrete(f) if f == 2.0))
                    }
                    other => panic!("expected Negate, got {other:?}"),
                }
            }
            other => panic!("expected Add(Negate), got {other:?}"),
        }
    }

    #[test]
    fn evaluator_folds_mul_into_paramexpr_mul() {
        let pe = pe_with_theta(vec![ParamExprElement {
            op_code: 2, // MUL
            lhs: ParamExprOperand::Float(3.0),
            rhs: ParamExprOperand::ParameterUuid(theta_uuid()),
        }]);
        assert!(matches!(fold(&pe).unwrap(), ParamExpr::Mul(_, _)));
    }

    #[test]
    fn evaluator_folds_rsub_to_rhs_minus_lhs() {
        // 1.5 - theta encoded as RSUB with LHS=theta, RHS=1.5.
        let pe = pe_with_theta(vec![ParamExprElement {
            op_code: 18, // RSUB
            lhs: ParamExprOperand::ParameterUuid(theta_uuid()),
            rhs: ParamExprOperand::Float(1.5),
        }]);
        match fold(&pe).unwrap() {
            ParamExpr::Add(lhs, rhs) => {
                // The encoding is `Add(rhs_pushed, Negate(lhs_pushed))`.
                assert!(matches!(*lhs, ParamExpr::Concrete(f) if f == 1.5));
                match *rhs {
                    ParamExpr::Negate(inner) => assert!(matches!(*inner, ParamExpr::Symbol(_))),
                    other => panic!("expected Negate, got {other:?}"),
                }
            }
            other => panic!("expected Add(Concrete, Negate(Symbol)), got {other:?}"),
        }
    }

    #[test]
    fn evaluator_folds_div_by_constant_to_mul_by_reciprocal() {
        // theta / 2 — DIV, LHS=theta, RHS=2.
        let pe = pe_with_theta(vec![ParamExprElement {
            op_code: 3, // DIV
            lhs: ParamExprOperand::ParameterUuid(theta_uuid()),
            rhs: ParamExprOperand::Int(2),
        }]);
        match fold(&pe).unwrap() {
            ParamExpr::Mul(lhs, rhs) => {
                assert!(matches!(*lhs, ParamExpr::Symbol(_)));
                assert!(matches!(*rhs, ParamExpr::Concrete(f) if f == 0.5));
            }
            other => panic!("expected Mul(Symbol, Concrete(0.5)), got {other:?}"),
        }
    }

    #[test]
    fn evaluator_rejects_div_by_non_constant() {
        // theta / theta — both sides are the symbol; no concrete divisor.
        let pe = pe_with_theta(vec![ParamExprElement {
            op_code: 3,
            lhs: ParamExprOperand::ParameterUuid(theta_uuid()),
            rhs: ParamExprOperand::ParameterUuid(theta_uuid()),
        }]);
        match fold(&pe).unwrap_err() {
            QpyError::Unsupported { what, .. } => {
                assert!(what.contains("DIV with non-constant"));
            }
            other => panic!("expected Unsupported(DIV non-constant), got {other:?}"),
        }
    }

    #[test]
    fn evaluator_rejects_pow_op() {
        let pe = pe_with_theta(vec![ParamExprElement {
            op_code: 4, // POW
            lhs: ParamExprOperand::ParameterUuid(theta_uuid()),
            rhs: ParamExprOperand::Int(2),
        }]);
        match fold(&pe).unwrap_err() {
            QpyError::Unsupported { what, .. } => {
                assert!(what.contains("outside omega's algebraic"));
            }
            other => panic!("expected Unsupported(POW), got {other:?}"),
        }
    }

    #[test]
    fn evaluator_handles_two_record_stack_carry() {
        // (theta + 1) * 2 — two records: ADD, then MUL with LHS=None.
        let pe = pe_with_theta(vec![
            ParamExprElement {
                op_code: 0, // ADD
                lhs: ParamExprOperand::Int(1),
                rhs: ParamExprOperand::ParameterUuid(theta_uuid()),
            },
            ParamExprElement {
                op_code: 2, // MUL
                lhs: ParamExprOperand::None,
                rhs: ParamExprOperand::Int(2),
            },
        ]);
        match fold(&pe).unwrap() {
            ParamExpr::Mul(inner, two) => {
                assert!(matches!(*inner, ParamExpr::Add(_, _)));
                assert!(matches!(*two, ParamExpr::Concrete(f) if f == 2.0));
            }
            other => panic!("expected Mul(Add(...), 2.0), got {other:?}"),
        }
    }

    #[test]
    fn evaluator_rejects_subs_marker() {
        let pe = pe_with_theta(vec![ParamExprElement {
            op_code: 255,
            lhs: ParamExprOperand::SubstitutionMarker { which: 's' },
            rhs: ParamExprOperand::None,
        }]);
        match fold(&pe).unwrap_err() {
            QpyError::Unsupported { what, .. } => {
                assert!(what.contains("substitution markers"));
            }
            other => panic!("expected Unsupported(subs markers), got {other:?}"),
        }
    }

    #[test]
    fn evaluator_rejects_uuid_not_in_symbol_map() {
        // ADD using a UUID that wasn't declared in the symbol table.
        let pe = DecodedParameterExpression {
            symbols: vec![],
            elements: vec![ParamExprElement {
                op_code: 0,
                lhs: ParamExprOperand::Float(1.0),
                rhs: ParamExprOperand::ParameterUuid([0x99; 16]),
            }],
        };
        match fold(&pe).unwrap_err() {
            QpyError::Unsupported { what, .. } => {
                assert!(what.contains("UUID not in its symbol map"));
            }
            other => panic!("expected Unsupported(uuid not in map), got {other:?}"),
        }
    }
}
