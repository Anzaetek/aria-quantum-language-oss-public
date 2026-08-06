//! Pure-Rust QPY *writer* — V16 single-circuit emission.
//!
//! Mirrors the reader's chain in reverse:
//!
//! ```text
//! CircuitIR → circuit_header + body + register_table + extras +
//!             instructions → bytes
//! ```
//!
//! ## Scope (2026-05-13)
//!
//! - QPY version 16 only — the smallest version with the circuit-start
//!   offset table the reader's `builder::read_qpy_circuit_irs`
//!   exercises. Pre-V16 sequential layout is unimplemented on both
//!   sides; targeting V16 lets every round-trip test re-use the
//!   existing reader.
//! - Single circuit per blob.
//! - Empty `metadata` ("{}"), empty name, `GlobalPhase::None`.
//! - No registers (CircuitIR's qubits flow through `num_qubits`; the
//!   reader infers a default register from `num_qubits` / `num_clbits`
//!   when the register table is empty).
//! - Instructions: the gate set in [`gate_kind_to_qiskit_name`] —
//!   H/X/Y/Z/S/Sdg/T/Tdg/Id, Rx/Ry/Rz, U1/U2/U3, CX/CY/CZ/Swap, CRz,
//!   CU3, CCX, CSwap, plus Measure / Reset / Barrier. Params can be
//!   `Concrete(_)` (emitted as the `b'f'` scalar payload) or any
//!   algebraic combination of `Symbol`/`Add`/`Mul`/`Negate` (emitted as
//!   the `b'e'` PARAMETER_EXPRESSION payload — PARAM_EXPR_ELEM_V13
//!   records). `Negate(a)` round-trips as `Mul(a, -1)` because QPY has
//!   no NEGATE op_code.
//!
//! ## Round-trip guarantee
//!
//! For a `ir: CircuitIR` whose ops are all in the supported gate set
//! and whose params are within omega's algebraic ParamExpr surface:
//!
//! ```text
//! read_qpy_circuit_ir(write_qpy_circuit_ir(&ir)).num_qubits         == ir.num_qubits
//! read_qpy_circuit_ir(write_qpy_circuit_ir(&ir)).num_classical_bits == ir.num_classical_bits
//! read_qpy_circuit_ir(write_qpy_circuit_ir(&ir)).ops.len()          == ir.ops.len()
//!   // op-by-op: same GateKind, same qubits, equivalent param values
//!   // (concrete params bit-exact; symbolic params evaluate to the
//!   // same number under any ParameterBinding — SymbolId may renumber
//!   // since the reader assigns fresh ids in encounter order).
//! ```
//!
//! Pinned by `bell_pair_round_trips_through_reader`,
//! `rx_concrete_param_round_trips`,
//! `symbolic_add_round_trips_with_concrete_offset`,
//! `symbolic_negate_round_trips_via_mul_minus_one`.

use std::collections::HashMap;

use omega_core::circuit::{CircuitIR, GateKind, GateOp, ParamExpr, SymbolId};

use super::circuit_header::FIXED_LEN as CIRCUIT_HEADER_LEN;

/// QPY format version this writer emits. Set to the same value
/// `program_table.rs` flags as the V16+ offset-table threshold so the
/// reader's V16+ codepath handles our output.
pub const WRITER_QPY_VERSION: u8 = 16;

/// Qiskit producer-version stamp embedded in the file header. We
/// don't pretend to be a specific Qiskit version; the reader uses
/// this only for diagnostics. `1.0.0` is the lowest sensible value
/// for a writer emitting V16-format blobs.
pub const WRITER_QISKIT_VERSION: (u8, u8, u8) = (1, 0, 0);

/// File-level header total length for V10+ (with symbolic_encoding).
const FILE_HEADER_LEN: usize = 19;
/// Program type key length (always 1 byte: `b'q'`).
const PROGRAM_TYPE_KEY_LEN: usize = 1;
/// Each circuit offset entry in the V16+ offset table is a u64.
const OFFSET_ENTRY_LEN: usize = 8;
/// `ANNOTATION_HEADER_STATIC` size (V15+ only). u32 = 0 namespaces.
const ANNOTATION_HEADER_STATIC_LEN: usize = 4;
/// `CUSTOM_CIRCUIT_DEF_HEADER` size. u64 = 0 custom definitions.
const CUSTOM_DEF_HEADER_LEN: usize = 8;

/// Encode a single-circuit `CircuitIR` to a V16 QPY blob. Returns the
/// binary blob; the reader at `read_qpy_circuit_ir` will reconstruct
/// an equivalent `CircuitIR`. Panics with a clear message when an op
/// uses a gate kind outside [`gate_kind_to_qiskit_name`] — the
/// less-common photonic / Custom gates land with the photonic-mode
/// encoder.
pub fn write_qpy_circuit_ir(ir: &CircuitIR) -> Vec<u8> {
    let mut buf = Vec::new();
    write_file_header(&mut buf, /* num_circuits */ 1);
    write_program_type_key(&mut buf);
    // V16+ offset table: one u64 per circuit, pointing to the
    // absolute byte offset where its payload begins.
    let circuit_offset =
        FILE_HEADER_LEN + PROGRAM_TYPE_KEY_LEN + OFFSET_ENTRY_LEN /* one circuit */;
    buf.extend_from_slice(&(circuit_offset as u64).to_be_bytes());
    write_circuit(&mut buf, ir);
    buf
}

/// Reverse map of `builder::gate_name_to_kind`. Emits the canonical
/// Qiskit class name for each native omega `GateKind`. Aliases on the
/// reader side (e.g. `CNOTGate → CX`) collapse to the primary on this
/// side — `CX → CXGate`, never `CNOTGate`.
pub fn gate_kind_to_qiskit_name(gate: &GateKind) -> Option<&'static str> {
    Some(match gate {
        GateKind::H => "HGate",
        GateKind::X => "XGate",
        GateKind::Y => "YGate",
        GateKind::Z => "ZGate",
        GateKind::S => "SGate",
        GateKind::Sdg => "SdgGate",
        GateKind::T => "TGate",
        GateKind::Tdg => "TdgGate",
        GateKind::Id => "IGate",
        GateKind::Rx => "RXGate",
        GateKind::Ry => "RYGate",
        GateKind::Rz => "RZGate",
        GateKind::U1 => "U1Gate",
        GateKind::U2 => "U2Gate",
        GateKind::U3 => "U3Gate",
        GateKind::CX => "CXGate",
        GateKind::CY => "CYGate",
        GateKind::CZ => "CZGate",
        GateKind::Swap => "SwapGate",
        GateKind::CRz => "CRZGate",
        GateKind::CU3 => "CU3Gate",
        GateKind::CCX => "CCXGate",
        GateKind::CSwap => "CSwapGate",
        GateKind::Measure => "Measure",
        GateKind::Barrier => "Barrier",
        GateKind::Reset => "Reset",
        // Photonic + Custom + the as-yet-untested Pauli-frame variants
        // surface as None so the writer panics with a clear "gate not
        // yet supported" message rather than silently encoding garbage.
        // Rbs has no canonical Qiskit gate class (XXPlusYY/XXMinusYY use
        // a different generator and phase convention) — decompose before
        // export rather than encode a near-miss.
        GateKind::PhaseShifter | GateKind::BeamSplitterRx | GateKind::Rbs | GateKind::Custom(_) => {
            return None
        }
    })
}

fn write_file_header(buf: &mut Vec<u8>, num_circuits: u64) {
    buf.extend_from_slice(b"QISKIT");
    buf.push(WRITER_QPY_VERSION);
    buf.push(WRITER_QISKIT_VERSION.0);
    buf.push(WRITER_QISKIT_VERSION.1);
    buf.push(WRITER_QISKIT_VERSION.2);
    buf.extend_from_slice(&num_circuits.to_be_bytes());
    // V10+ symbolic_encoding byte. We emit `b'p'` (sympy) — the reader
    // accepts both b'p' and b'e' identically since the symbolic
    // expression decoder is encoding-agnostic. Pick sympy because that's
    // the format Qiskit < 1.0 wrote by default.
    buf.push(b'p');
}

fn write_program_type_key(buf: &mut Vec<u8>) {
    buf.push(b'q'); // QuantumCircuit
}

fn write_circuit(buf: &mut Vec<u8>, ir: &CircuitIR) {
    write_circuit_header(buf, ir);
    write_circuit_body(buf); // empty name + global_phase=None + metadata="{}"
                             // No registers (num_registers=0 in the header). The reader infers
                             // qubit / clbit indices straight from num_qubits / num_clbits when
                             // the register table is empty.
    write_circuit_extras(buf); // annotation u32=0, custom_def u64=0
    for op in &ir.ops {
        write_instruction(buf, ir, op);
    }
}

fn write_instruction(buf: &mut Vec<u8>, ir: &CircuitIR, op: &GateOp) {
    let name = gate_kind_to_qiskit_name(&op.gate).unwrap_or_else(|| {
        panic!(
            "write_qpy_circuit_ir: gate {:?} not in the writer's supported set; \
             extend gate_kind_to_qiskit_name or fall back to qpy_to_qasm2",
            op.gate
        )
    });
    let name_bytes = name.as_bytes();
    let label_bytes: &[u8] = b"";
    let num_parameters: u16 = op.params.len() as u16;
    // Measure has 1 carg; every other supported gate has 0.
    let (num_qargs, num_cargs): (u32, u32) = match op.gate {
        GateKind::Measure => (op.qubits.len() as u32, 1),
        _ => (op.qubits.len() as u32, 0),
    };
    // Conditional gates: omega's `Some((start_bit, num_bits, expected))`
    // maps to Qiskit's TWO_TUPLE condition. Only `num_bits == 1` is
    // representable today — multi-bit register conditions need a named
    // register table, which the writer doesn't yet emit. Encoding for
    // a single-clbit condition mirrors Qiskit's
    // `_py_serialize_register_param` output: `b"\x00<decimal_index>"`.
    let (extras_key, condition_register, condition_value): (u8, Vec<u8>, i64) = match op.condition {
        None => (0, Vec::new(), 0),
        Some((start_bit, 1, expected)) => {
            let mut reg = Vec::with_capacity(8);
            reg.push(0u8);
            reg.extend_from_slice(start_bit.to_string().as_bytes());
            (1, reg, expected as i64)
        }
        Some((start_bit, num_bits, _)) => {
            panic!(
                "write_qpy_circuit_ir: gate condition ({start_bit}, {num_bits}, _) needs a \
                 named ClassicalRegister; only single-clbit (num_bits == 1) conditions are \
                 encoded by the V16 writer today",
            );
        }
    };
    let condition_register_size: u16 = condition_register.len() as u16;
    let num_ctrl_qubits: u32 = 0;
    let ctrl_state: u32 = 0;

    // INSTRUCTION_V2 fixed header (33 bytes big-endian).
    buf.extend_from_slice(&(name_bytes.len() as u16).to_be_bytes());
    buf.extend_from_slice(&(label_bytes.len() as u16).to_be_bytes());
    buf.extend_from_slice(&num_parameters.to_be_bytes());
    buf.extend_from_slice(&num_qargs.to_be_bytes());
    buf.extend_from_slice(&num_cargs.to_be_bytes());
    buf.push(extras_key);
    buf.extend_from_slice(&condition_register_size.to_be_bytes());
    buf.extend_from_slice(&condition_value.to_be_bytes());
    buf.extend_from_slice(&num_ctrl_qubits.to_be_bytes());
    buf.extend_from_slice(&ctrl_state.to_be_bytes());

    buf.extend_from_slice(name_bytes);
    buf.extend_from_slice(label_bytes);
    // condition_register bytes (empty when ConditionKind::None).
    buf.extend_from_slice(&condition_register);

    // qargs (5 bytes each: type byte + u32 index), then cargs.
    for qubit in &op.qubits {
        buf.push(b'q');
        buf.extend_from_slice(&(qubit.0).to_be_bytes());
    }
    if matches!(op.gate, GateKind::Measure) {
        let cbit = op.classical_bit.unwrap_or(0);
        buf.push(b'c');
        buf.extend_from_slice(&cbit.to_be_bytes());
    }

    // Parameters: Concrete → `b'f'`, everything else → `b'e'`
    // PARAMETER_EXPRESSION payload (PARAM_EXPR_ELEM_V13 stack-machine).
    for p in &op.params {
        write_instruction_param(buf, ir, p);
    }
}

fn write_instruction_param(buf: &mut Vec<u8>, ir: &CircuitIR, p: &ParamExpr) {
    match p {
        ParamExpr::Concrete(v) => {
            // INSTRUCTION_PARAM header: type byte + u64 size (big-endian).
            buf.push(b'f'); // float
            buf.extend_from_slice(&8u64.to_be_bytes());
            // Payload: 8-byte f64 little-endian (QPY ≤ 17 scalar-payload
            // quirk; header above is big-endian, payload is little-endian).
            buf.extend_from_slice(&v.to_le_bytes());
        }
        ParamExpr::Symbol(_)
        | ParamExpr::Negate(_)
        | ParamExpr::Add(_, _)
        | ParamExpr::Mul(_, _) => {
            // INSTRUCTION_PARAM header: type byte + u64 size (big-endian).
            // Type `b'e'` carries a PARAMETER_EXPRESSION payload; the
            // size is the PE blob length so the reader can slice past it
            // without parsing.
            let payload = encode_parameter_expression(ir, p);
            buf.push(b'e');
            buf.extend_from_slice(&(payload.len() as u64).to_be_bytes());
            buf.extend_from_slice(&payload);
        }
    }
}

/// Encode an omega `ParamExpr` into a QPY V13+ `PARAMETER_EXPRESSION`
/// payload. Used for any non-Concrete parameter — the writer emits a
/// stack-machine program of [`PARAM_EXPR_ELEM_V13`] records that, when
/// replayed by Qiskit's `_read_parameter_expr_v13` interpreter (and the
/// in-tree builder's `evaluate_parameter_expression`), rebuilds the
/// original expression.
///
/// Encoding rules:
/// - Leaves (`Concrete`, `Symbol`) emit a single `op_code = 255` record
///   with the leaf packed into the LHS slot, RHS = `b'n'` (None).
/// - Binary ops (`Add`, `Mul`) recurse on both operands then emit a
///   single op record with `op_code = ADD/MUL` and both slots = None
///   (operands come from the stack residual).
/// - `Negate(a)` rewrites as `Mul(a, Concrete(-1))` — omega has no
///   NEGATE op_code, and the reader interprets MUL with a concrete -1
///   operand as a clean Negate (no global-phase quirk).
///
/// The trailing symbol map allocates one entry per unique `SymbolId`
/// the expression references, using a deterministic UUID derived from
/// the SymbolId so blob bytes are stable across writer invocations.
fn encode_parameter_expression(ir: &CircuitIR, expr: &ParamExpr) -> Vec<u8> {
    let mut symbol_uuids: HashMap<SymbolId, [u8; 16]> = HashMap::new();
    let mut ordered_symbols: Vec<SymbolId> = Vec::new();
    let mut elements: Vec<u8> = Vec::new();
    encode_pe_node(expr, &mut elements, &mut symbol_uuids, &mut ordered_symbols);

    let mut out = Vec::new();
    // PARAMETER_EXPR header: u64 map_elements + u64 expr_size (BE).
    out.extend_from_slice(&(ordered_symbols.len() as u64).to_be_bytes());
    out.extend_from_slice(&(elements.len() as u64).to_be_bytes());
    out.extend_from_slice(&elements);

    // Symbol map: one PARAM_EXPR_MAP_ELEM_V3 per unique referenced symbol.
    // Each entry: u8 symbol_type=b'p' + u8 value_type=b'n' + u64 value_size=0
    // + PARAMETER struct (u16 name_size + 16-byte uuid + name bytes).
    for sym_id in &ordered_symbols {
        let uuid = symbol_uuids[sym_id];
        let name = ir
            .symbols
            .get(sym_id)
            .map(String::as_str)
            .unwrap_or("omega_anon");
        out.push(b'p'); // symbol_type = Parameter
        out.push(b'n'); // value_type = None (no per-symbol value payload)
        out.extend_from_slice(&0u64.to_be_bytes()); // value_size = 0
        out.extend_from_slice(&(name.len() as u16).to_be_bytes());
        out.extend_from_slice(&uuid);
        out.extend_from_slice(name.as_bytes());
    }
    out
}

/// Postorder walk: emit records that leave this node's value on top of
/// the replay stack. Each `PARAM_EXPR_ELEM_V13` record is 35 bytes
/// (`!Bc16sc16s` = u8 op_code + u8 lhs_type + 16 lhs_bytes + u8 rhs_type
/// + 16 rhs_bytes).
fn encode_pe_node(
    expr: &ParamExpr,
    out: &mut Vec<u8>,
    symbol_uuids: &mut HashMap<SymbolId, [u8; 16]>,
    ordered: &mut Vec<SymbolId>,
) {
    match expr {
        ParamExpr::Concrete(_) | ParamExpr::Symbol(_) => {
            let (lhs_type, lhs_bytes) = pe_leaf(expr, symbol_uuids, ordered);
            // NoOp (op_code = 255): push LHS only.
            push_pe_record(out, 255, lhs_type, lhs_bytes, b'n', [0u8; 16]);
        }
        ParamExpr::Add(a, b) => {
            encode_pe_node(a, out, symbol_uuids, ordered);
            encode_pe_node(b, out, symbol_uuids, ordered);
            // ADD (op_code = 0): both operands come from the stack.
            push_pe_record(out, 0, b'n', [0u8; 16], b'n', [0u8; 16]);
        }
        ParamExpr::Mul(a, b) => {
            encode_pe_node(a, out, symbol_uuids, ordered);
            encode_pe_node(b, out, symbol_uuids, ordered);
            // MUL (op_code = 2): both operands from the stack.
            push_pe_record(out, 2, b'n', [0u8; 16], b'n', [0u8; 16]);
        }
        ParamExpr::Negate(a) => {
            // No NEGATE op_code in QPY — fold `Negate(a)` into `a * -1`.
            // After recursion, `a` is on the stack. Push -1 as the RHS
            // of an inline MUL so the operand order on the stack is
            // [a, -1], and the reader's `lhs=pop(); rhs=pop()` reassembly
            // gives `Mul(a, Concrete(-1))` — which evaluates to -a.
            encode_pe_node(a, out, symbol_uuids, ordered);
            let neg_one = pe_float_bytes(-1.0);
            push_pe_record(out, 2, b'n', [0u8; 16], b'f', neg_one);
        }
    }
}

/// Build a 35-byte PARAM_EXPR_ELEM_V13 record and append it to `out`.
fn push_pe_record(
    out: &mut Vec<u8>,
    op_code: u8,
    lhs_type: u8,
    lhs_bytes: [u8; 16],
    rhs_type: u8,
    rhs_bytes: [u8; 16],
) {
    out.push(op_code);
    out.push(lhs_type);
    out.extend_from_slice(&lhs_bytes);
    out.push(rhs_type);
    out.extend_from_slice(&rhs_bytes);
}

/// Return the `(type_byte, 16-byte payload)` for a leaf node. Concrete
/// floats pack into the trailing 8 bytes of the 16-byte slot (big-endian
/// IEEE 754) per Qiskit's `!Qd` format; Symbol refs use the 16-byte
/// UUID slot directly.
fn pe_leaf(
    expr: &ParamExpr,
    symbol_uuids: &mut HashMap<SymbolId, [u8; 16]>,
    ordered: &mut Vec<SymbolId>,
) -> (u8, [u8; 16]) {
    match expr {
        ParamExpr::Concrete(v) => (b'f', pe_float_bytes(*v)),
        ParamExpr::Symbol(id) => {
            let uuid = *symbol_uuids.entry(*id).or_insert_with(|| {
                ordered.push(*id);
                symbol_id_to_uuid(*id)
            });
            (b'p', uuid)
        }
        _ => unreachable!("pe_leaf called on non-leaf {expr:?}"),
    }
}

fn pe_float_bytes(v: f64) -> [u8; 16] {
    // `!Qd` — first 8 bytes are filler zeros, last 8 are big-endian f64.
    let mut bytes = [0u8; 16];
    bytes[8..16].copy_from_slice(&v.to_be_bytes());
    bytes
}

/// Deterministic UUID assignment so the same SymbolId always maps to
/// the same 16-byte UUID across writer invocations. Layout: 4-byte
/// `b"OmgS"` marker, 8 zero bytes, then the 4-byte big-endian SymbolId
/// — keeps the UUIDs identifiable in hex dumps and prevents collisions
/// with random Qiskit-generated UUIDs.
fn symbol_id_to_uuid(id: SymbolId) -> [u8; 16] {
    let mut uuid = [0u8; 16];
    uuid[0..4].copy_from_slice(b"OmgS");
    uuid[12..16].copy_from_slice(&id.to_be_bytes());
    uuid
}

fn write_circuit_header(buf: &mut Vec<u8>, ir: &CircuitIR) {
    let name_size: u16 = 0;
    let global_phase_type: u8 = b'n'; // None
    let global_phase_size: u16 = 0;
    let metadata_size: u64 = 2; // "{}" JSON
    let num_registers: u32 = 0;
    let num_instructions: u64 = ir.ops.len() as u64;
    let num_vars: u32 = 0;

    let start = buf.len();
    buf.extend_from_slice(&name_size.to_be_bytes());
    buf.push(global_phase_type);
    buf.extend_from_slice(&global_phase_size.to_be_bytes());
    buf.extend_from_slice(&ir.num_qubits.to_be_bytes());
    buf.extend_from_slice(&ir.num_classical_bits.to_be_bytes());
    buf.extend_from_slice(&metadata_size.to_be_bytes());
    buf.extend_from_slice(&num_registers.to_be_bytes());
    buf.extend_from_slice(&num_instructions.to_be_bytes());
    buf.extend_from_slice(&num_vars.to_be_bytes());
    debug_assert_eq!(
        buf.len() - start,
        CIRCUIT_HEADER_LEN,
        "circuit_header must be exactly {CIRCUIT_HEADER_LEN} bytes"
    );
}

fn write_circuit_body(buf: &mut Vec<u8>) {
    // name (empty), global_phase (size 0 for 'n'), metadata = "{}".
    buf.extend_from_slice(b"{}");
}

fn write_circuit_extras(buf: &mut Vec<u8>) {
    // V15+: 4-byte u32 annotation namespace count = 0.
    buf.extend_from_slice(&0u32.to_be_bytes());
    // Always: 8-byte u64 custom-circuit-def count = 0.
    buf.extend_from_slice(&0u64.to_be_bytes());
    let _ = ANNOTATION_HEADER_STATIC_LEN + CUSTOM_DEF_HEADER_LEN; // used in lengths above
}

#[cfg(test)]
mod tests {
    use super::super::{is_qpy, read_header, read_qpy_circuit_ir, SymbolicEncoding};
    use super::*;
    use omega_core::circuit::CircuitType;

    #[test]
    fn empty_circuit_has_qiskit_magic() {
        let ir = CircuitIR::new(2, CircuitType::GateBased);
        let blob = write_qpy_circuit_ir(&ir);
        assert!(is_qpy(&blob), "writer must emit the QISKIT magic bytes");
    }

    #[test]
    fn empty_circuit_file_header_round_trips() {
        let ir = CircuitIR::new(3, CircuitType::GateBased);
        let blob = write_qpy_circuit_ir(&ir);
        let h = read_header(&blob).unwrap();
        assert_eq!(h.qpy_version, WRITER_QPY_VERSION);
        assert_eq!(h.qiskit_version, WRITER_QISKIT_VERSION);
        assert_eq!(h.num_circuits, 1);
        assert_eq!(h.symbolic_encoding, Some(SymbolicEncoding::Sympy));
    }

    #[test]
    fn empty_circuit_full_round_trip_preserves_qubit_count() {
        // The end-to-end contract: write then read must give back a
        // CircuitIR with the same shape (no ops, same num_qubits,
        // same num_clbits).
        let mut ir = CircuitIR::new(5, CircuitType::GateBased);
        ir.num_classical_bits = 3;
        let blob = write_qpy_circuit_ir(&ir);
        let decoded = read_qpy_circuit_ir(&blob).expect("write→read must succeed");
        assert_eq!(decoded.num_qubits, 5);
        assert_eq!(decoded.num_classical_bits, 3);
        assert!(decoded.ops.is_empty());
    }

    #[test]
    fn empty_circuit_zero_qubits_round_trips() {
        // Edge case: 0-qubit, 0-cbit circuit — sometimes used as a
        // sentinel for "circuit slot exists but has no content".
        let ir = CircuitIR::new(0, CircuitType::GateBased);
        let blob = write_qpy_circuit_ir(&ir);
        let decoded = read_qpy_circuit_ir(&blob).unwrap();
        assert_eq!(decoded.num_qubits, 0);
        assert_eq!(decoded.num_classical_bits, 0);
    }

    #[test]
    fn bell_pair_round_trips_through_reader() {
        // H q[0]; CX q[0], q[1]; — the canonical 2-qubit entangler.
        // After round-trip the reconstructed IR must have the same
        // GateKinds in the same order on the same qubits.
        use omega_core::circuit::{GateKind, GateOp, Qubit};
        use smallvec::smallvec;
        let mut ir = CircuitIR::new(2, CircuitType::GateBased);
        ir.add_op(GateOp {
            gate: GateKind::H,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
        ir.add_op(GateOp {
            gate: GateKind::CX,
            qubits: smallvec![Qubit(0), Qubit(1)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
        let blob = write_qpy_circuit_ir(&ir);
        let decoded = read_qpy_circuit_ir(&blob).expect("Bell pair must round-trip");
        assert_eq!(decoded.num_qubits, 2);
        assert_eq!(decoded.ops.len(), 2);
        assert_eq!(decoded.ops[0].gate, GateKind::H);
        assert_eq!(decoded.ops[0].qubits[0], Qubit(0));
        assert_eq!(decoded.ops[1].gate, GateKind::CX);
        assert_eq!(decoded.ops[1].qubits[0], Qubit(0));
        assert_eq!(decoded.ops[1].qubits[1], Qubit(1));
    }

    #[test]
    fn rx_concrete_param_round_trips() {
        // Single-qubit rotation with a concrete parameter — exercises
        // the b'f' INSTRUCTION_PARAM path (8-byte f64 LE payload).
        use omega_core::circuit::{GateKind, GateOp, ParamExpr, Qubit};
        use omega_core::params::ParameterBinding;
        use smallvec::smallvec;
        let mut ir = CircuitIR::new(1, CircuitType::GateBased);
        ir.add_op(GateOp {
            gate: GateKind::Rx,
            qubits: smallvec![Qubit(0)],
            params: smallvec![ParamExpr::Concrete(0.31)],
            classical_bit: None,
            condition: None,
        });
        let blob = write_qpy_circuit_ir(&ir);
        let decoded = read_qpy_circuit_ir(&blob).unwrap();
        assert_eq!(decoded.ops.len(), 1);
        assert_eq!(decoded.ops[0].gate, GateKind::Rx);
        let binding = ParameterBinding::new();
        let v = binding.resolve(&decoded.ops[0].params[0]).unwrap();
        assert!(
            (v - 0.31).abs() < 1e-12,
            "rx parameter round-trip got {v}, expected 0.31"
        );
    }

    #[test]
    fn measure_round_trips_with_cbit_link() {
        // Measure q[0] -> c[0]: the writer emits one qarg + one carg;
        // the reader rebuilds the GateOp with classical_bit = Some(0).
        use omega_core::circuit::{GateKind, GateOp, Qubit};
        use smallvec::smallvec;
        let mut ir = CircuitIR::new(1, CircuitType::GateBased);
        ir.num_classical_bits = 1;
        ir.add_op(GateOp {
            gate: GateKind::Measure,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: Some(0),
            condition: None,
        });
        let blob = write_qpy_circuit_ir(&ir);
        let decoded = read_qpy_circuit_ir(&blob).unwrap();
        assert_eq!(decoded.ops.len(), 1);
        assert_eq!(decoded.ops[0].gate, GateKind::Measure);
        assert_eq!(decoded.ops[0].classical_bit, Some(0));
    }

    #[test]
    fn symbolic_param_round_trips_as_a_single_symbol_reference() {
        // The simplest symbolic case: `Rx(theta)` where `theta` is a
        // bare Symbol. Encoder emits a single 35-byte NoOp record with
        // the symbol UUID in the LHS slot, plus a 1-element symbol map.
        use omega_core::circuit::{GateKind, GateOp, ParamExpr, Qubit};
        use smallvec::smallvec;
        let mut ir = CircuitIR::new(1, CircuitType::GateBased);
        ir.symbols.insert(0, "theta".to_string());
        ir.add_op(GateOp {
            gate: GateKind::Rx,
            qubits: smallvec![Qubit(0)],
            params: smallvec![ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });
        let blob = write_qpy_circuit_ir(&ir);
        let decoded = read_qpy_circuit_ir(&blob).expect("symbolic Rx must round-trip");
        assert_eq!(decoded.ops.len(), 1);
        assert_eq!(decoded.ops[0].gate, GateKind::Rx);
        match &decoded.ops[0].params[0] {
            ParamExpr::Symbol(sid) => {
                assert_eq!(decoded.symbols.get(sid).map(String::as_str), Some("theta"));
            }
            other => panic!("expected ParamExpr::Symbol(...), got {other:?}"),
        }
    }

    #[test]
    fn symbolic_add_round_trips_with_concrete_offset() {
        // `Rx(theta + 0.5)` — exercises the binary-op path: two leaf
        // pushes followed by an ADD record. After round-trip the reader
        // rebuilds an `Add(Symbol(theta), Concrete(0.5))`, which evaluates
        // to `theta + 0.5` for any given binding.
        use omega_core::circuit::{GateKind, GateOp, ParamExpr, Qubit};
        use omega_core::params::ParameterBinding;
        use smallvec::smallvec;
        let mut ir = CircuitIR::new(1, CircuitType::GateBased);
        ir.symbols.insert(0, "theta".to_string());
        let expr = ParamExpr::Add(
            Box::new(ParamExpr::Symbol(0)),
            Box::new(ParamExpr::Concrete(0.5)),
        );
        ir.add_op(GateOp {
            gate: GateKind::Rx,
            qubits: smallvec![Qubit(0)],
            params: smallvec![expr],
            classical_bit: None,
            condition: None,
        });
        let blob = write_qpy_circuit_ir(&ir);
        let decoded = read_qpy_circuit_ir(&blob).expect("Rx(theta+0.5) must round-trip");
        let mut binding = ParameterBinding::new();
        // The reader picks a fresh SymbolId for the round-tripped theta;
        // find it via the symbols table so the binding hits the right id.
        let theta_id = decoded
            .symbols
            .iter()
            .find(|(_, name)| name.as_str() == "theta")
            .map(|(id, _)| *id)
            .expect("decoded circuit must preserve the symbol name");
        binding.bind(theta_id, 1.25);
        let v = binding.resolve(&decoded.ops[0].params[0]).unwrap();
        assert!(
            (v - (1.25 + 0.5)).abs() < 1e-12,
            "theta+0.5 with theta=1.25 must evaluate to 1.75, got {v}"
        );
    }

    #[test]
    fn conditional_gate_round_trips_with_single_clbit_register() {
        // `if(c0 == 1) X q[0]` — exercises Stage 3b: writer emits the
        // TWO_TUPLE condition with a `\x00<idx>` condition_register and
        // a condition_value, reader folds it back into
        // `GateOp.condition = Some((0, 1, 1))`.
        use omega_core::circuit::{GateKind, GateOp, Qubit};
        use smallvec::smallvec;
        let mut ir = CircuitIR::new(1, CircuitType::GateBased);
        ir.num_classical_bits = 1;
        ir.add_op(GateOp {
            gate: GateKind::X,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: None,
            condition: Some((0, 1, 1)),
        });
        let blob = write_qpy_circuit_ir(&ir);
        let decoded = read_qpy_circuit_ir(&blob).expect("conditional X must round-trip end-to-end");
        assert_eq!(decoded.ops.len(), 1);
        assert_eq!(decoded.ops[0].gate, GateKind::X);
        assert_eq!(decoded.ops[0].condition, Some((0, 1, 1)));
    }

    #[test]
    fn conditional_gate_at_higher_clbit_index_round_trips() {
        // Clbit index 12 with expected value 0 — exercises the decimal
        // encoding path and a non-trivial condition value.
        use omega_core::circuit::{GateKind, GateOp, Qubit};
        use smallvec::smallvec;
        let mut ir = CircuitIR::new(1, CircuitType::GateBased);
        ir.num_classical_bits = 16;
        ir.add_op(GateOp {
            gate: GateKind::H,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: None,
            condition: Some((12, 1, 0)),
        });
        let blob = write_qpy_circuit_ir(&ir);
        let decoded = read_qpy_circuit_ir(&blob).unwrap();
        assert_eq!(decoded.ops[0].condition, Some((12, 1, 0)));
    }

    #[test]
    #[should_panic(expected = "needs a named ClassicalRegister")]
    fn conditional_gate_with_multi_bit_register_panics() {
        // num_bits > 1 needs a named ClassicalRegister; the writer
        // panics rather than silently corrupting the wire format.
        use omega_core::circuit::{GateKind, GateOp, Qubit};
        use smallvec::smallvec;
        let mut ir = CircuitIR::new(1, CircuitType::GateBased);
        ir.num_classical_bits = 2;
        ir.add_op(GateOp {
            gate: GateKind::X,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: None,
            condition: Some((0, 2, 3)),
        });
        let _ = write_qpy_circuit_ir(&ir);
    }

    #[test]
    fn symbolic_negate_round_trips_via_mul_minus_one() {
        // `Rx(-theta)` → encoder emits `Mul(theta, -1)`. After
        // round-trip the reader rebuilds a Mul with a -1 operand, which
        // evaluates to -theta_value for any binding.
        use omega_core::circuit::{GateKind, GateOp, ParamExpr, Qubit};
        use omega_core::params::ParameterBinding;
        use smallvec::smallvec;
        let mut ir = CircuitIR::new(1, CircuitType::GateBased);
        ir.symbols.insert(0, "theta".to_string());
        let expr = ParamExpr::Negate(Box::new(ParamExpr::Symbol(0)));
        ir.add_op(GateOp {
            gate: GateKind::Rx,
            qubits: smallvec![Qubit(0)],
            params: smallvec![expr],
            classical_bit: None,
            condition: None,
        });
        let blob = write_qpy_circuit_ir(&ir);
        let decoded = read_qpy_circuit_ir(&blob).expect("Rx(-theta) must round-trip");
        let theta_id = decoded
            .symbols
            .iter()
            .find(|(_, name)| name.as_str() == "theta")
            .map(|(id, _)| *id)
            .unwrap();
        let mut binding = ParameterBinding::new();
        binding.bind(theta_id, 0.7);
        let v = binding.resolve(&decoded.ops[0].params[0]).unwrap();
        assert!(
            (v - (-0.7)).abs() < 1e-12,
            "Rx(-theta) with theta=0.7 must evaluate to -0.7, got {v}"
        );
    }
}
