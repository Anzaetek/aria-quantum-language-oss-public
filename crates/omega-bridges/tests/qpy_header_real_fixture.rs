//! Header-parser test against a real Qiskit-generated QPY blob.
//!
//! `tests/fixtures/bell_qiskit_2_4_1.qpy` was produced via:
//!
//! ```text
//! from qiskit import QuantumCircuit, qpy
//! qc = QuantumCircuit(2, 2)
//! qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
//! qpy.dump(qc, open("bell_qiskit_2_4_1.qpy", "wb"))
//! ```
//!
//! under Qiskit 2.4.1. The fixture is a frozen artifact — the
//! synthetic-bytes tests in `qpy::header::tests` cover the format
//! variants; this one pins the parser against an actual upstream
//! producer so we'd notice if the layout drifted between QPY
//! versions in a way the hand-rolled fixtures missed.

use omega_bridges::qpy::{
    is_qpy, read_circuit_body, read_circuit_extras, read_circuit_header, read_circuit_instructions,
    read_header, read_instruction_args, read_instruction_header, read_instruction_params,
    read_program_table, read_qpy_circuit_ir, read_qpy_circuit_irs, read_register_table,
    ConditionKind, GlobalPhase, InstructionArg, InstructionParam, ProgramType, RegisterKind,
    SymbolicEncoding,
};
use omega_core::circuit::{GateKind, Qubit};

const FIXTURE: &[u8] = include_bytes!("fixtures/bell_qiskit_2_4_1.qpy");
const TWO_CIRCUIT_FIXTURE: &[u8] = include_bytes!("fixtures/two_circuits_qiskit_2_4_1.qpy");
/// Single-circuit fixture with one parametric gate: `Rx(π/4)` on
/// `qreg q[1]`. Captured under Qiskit 2.4.1 via:
///
/// ```text
/// from qiskit import QuantumCircuit, qpy
/// import math
/// qc = QuantumCircuit(1); qc.rx(math.pi/4, 0)
/// qpy.dump(qc, open("rx_param_qiskit_2_4_1.qpy", "wb"))
/// ```
const RX_PARAM_FIXTURE: &[u8] = include_bytes!("fixtures/rx_param_qiskit_2_4_1.qpy");
/// Single-circuit fixture exercising most gates omega represents
/// natively: H/X/Y/Z/S/Sdg/T/Tdg/Id, Rx/Ry/Rz, U3, CX/CY/CZ/Swap,
/// CRz, CCX, CSwap. 20 ops on 3 qubits.
const WIDE_GATES_FIXTURE: &[u8] = include_bytes!("fixtures/wide_gates_qiskit_2_4_1.qpy");
/// `qc.rx(Parameter('θ'), 0)` — a single Rx gate with a free
/// symbolic parameter (sympy by default in Qiskit 2.4.1).
const RX_SYMBOLIC_FIXTURE: &[u8] = include_bytes!("fixtures/rx_symbolic_qiskit_2_4_1.qpy");
/// Two free parameters `θ` and `φ`; θ is referenced from two
/// different gates (Rx + CRz) so the SymbolLookup must produce the
/// same `SymbolId` for both occurrences.
const TWO_PARAMS_FIXTURE: &[u8] = include_bytes!("fixtures/two_params_qiskit_2_4_1.qpy");
/// `qc.rx(Parameter('θ') + 1.5, 0)` — exercises a single
/// `ParameterExpression` (`b'e'`) INSTRUCTION_PARAM. Captured under
/// Qiskit 2.4.1 via:
///
/// ```text
/// from qiskit import QuantumCircuit, qpy
/// from qiskit.circuit import Parameter
/// theta = Parameter("θ")
/// qc = QuantumCircuit(1); qc.rx(theta + 1.5, 0)
/// qpy.dump(qc, open("rx_paramexpr_qiskit_2_4_1.qpy", "wb"))
/// ```
const RX_PARAMEXPR_FIXTURE: &[u8] = include_bytes!("fixtures/rx_paramexpr_qiskit_2_4_1.qpy");
/// `qc.rx((Parameter('θ') + 1.0) * 2.0, 0)` — two replay-stack
/// records: ADD then MUL with `LHS = b'n'` to carry the residual.
/// Pins the multi-record path against a real producer.
const RX_PARAMEXPR_COMPOUND_FIXTURE: &[u8] =
    include_bytes!("fixtures/rx_paramexpr_compound_qiskit_2_4_1.qpy");
/// `qc.rx(1.5 - Parameter('φ'), 0)` — single RSUB record. Pins the
/// `lhs.__rsub__(rhs) = rhs - lhs` interpretation against a real
/// producer.
const RX_PARAMEXPR_RSUB_FIXTURE: &[u8] =
    include_bytes!("fixtures/rx_paramexpr_rsub_qiskit_2_4_1.qpy");
/// `pv = ParameterVector('x', 3); qc.rx(pv[0], 0); qc.ry(pv[1], 1);
///  qc.rz(pv[2], 2); qc.rx(pv[0], 0)` — exercises the b'v'
/// INSTRUCTION_PARAM payload (PARAMETER_VECTOR_ELEMENT struct) and
/// SymbolLookup deduplication for the repeated `pv[0]` reference.
const RX_PARAMVECTOR_FIXTURE: &[u8] = include_bytes!("fixtures/rx_paramvector_qiskit_2_4_1.qpy");
/// `gammas = ParameterVector('gamma', 2); qc.rx(2.0 * gammas[0] +
/// gammas[1], 0)` — pins PE evaluation over a symbol map that
/// contains b'v' PARAMETER_VECTOR_ELEMENT entries.
const RX_PARAMVECTOR_EXPR_FIXTURE: &[u8] =
    include_bytes!("fixtures/rx_paramvector_expr_qiskit_2_4_1.qpy");

#[test]
fn fixture_passes_magic_byte_check() {
    assert!(is_qpy(FIXTURE));
}

#[test]
fn fixture_parses_to_qpy_v17_qiskit_2_4_1() {
    let h = read_header(FIXTURE).expect("real Qiskit-2.4.1 blob must parse");
    // Qiskit 2.4.1 writes QPY v17 by default; if Qiskit ever bumps
    // the default, this assertion catches it.
    assert_eq!(h.qpy_version, 17);
    assert_eq!(h.qiskit_version, (2, 4, 1));
    assert_eq!(h.num_circuits, 1);
    // 2.4.1 defaults to sympy-encoded symbols.
    assert_eq!(h.symbolic_encoding, Some(SymbolicEncoding::Sympy));
    // The body must start exactly past the 19-byte V10+ header.
    assert_eq!(h.header_len(), 19);
    // Sanity: the byte after the header is the start of the first
    // per-circuit block, which V17 documents as beginning with the
    // CIRCUIT_HEADER struct. The current parser doesn't consume it,
    // but pinning the boundary catches off-by-ones if the file
    // header ever grows again.
    assert!(FIXTURE.len() > h.header_len());
}

#[test]
fn fixture_program_table_resolves_first_circuit_offset() {
    let h = read_header(FIXTURE).unwrap();
    let t = read_program_table(FIXTURE, &h).unwrap();
    assert_eq!(t.program_type, ProgramType::Circuit);
    // V17 has a circuit_start_table; for one circuit it's a single
    // u64 offset. Captured by hand from the fixture: the per-circuit
    // CIRCUIT_HEADER begins at byte 28 (file_header[0..19] +
    // type_key[19..20] + offset_table[20..28]).
    assert_eq!(t.circuit_offsets, vec![28]);
    // table_len = type_key (1) + circuit_table (1 × u64 = 8) = 9.
    assert_eq!(t.table_len(), 9);
    // Combined header + table consumes exactly 28 bytes — same as
    // the offset entry, confirming the table is a self-locating
    // pointer to the circuit immediately after itself.
    assert_eq!(
        h.header_len() + t.table_len(),
        t.circuit_offsets[0] as usize
    );

    // Sanity: the byte at the resolved offset is the start of
    // CIRCUIT_HEADER, whose first field is `name_size: u16`. For the
    // Bell-state fixture the name is "circuit-41" (10 chars).
    let first = t.circuit_offsets[0] as usize;
    let name_size = u16::from_be_bytes([FIXTURE[first], FIXTURE[first + 1]]);
    assert_eq!(name_size, 10);
}

#[test]
fn two_circuit_fixture_resolves_both_offsets() {
    // Two-circuit fixture: a Bell (n=2) and a 3-qubit circuit, both
    // dumped through `qpy.dump([qc1, qc2], f)`.
    let h = read_header(TWO_CIRCUIT_FIXTURE).unwrap();
    assert_eq!(h.qpy_version, 17);
    assert_eq!(h.num_circuits, 2);

    let t = read_program_table(TWO_CIRCUIT_FIXTURE, &h).unwrap();
    assert_eq!(t.program_type, ProgramType::Circuit);
    // Captured by hand: 1st circuit at byte 36, 2nd at byte 372. The
    // table itself is 1 (type key) + 2 × 8 (offsets) = 17 bytes
    // following the 19-byte file header → first circuit must start
    // at byte 36, which the fixture confirms.
    assert_eq!(t.circuit_offsets, vec![36, 372]);
    assert_eq!(t.table_len(), 1 + 2 * 8);
    assert_eq!(h.header_len() + t.table_len(), 36);

    // Both circuit headers begin with name_size: u16 = 10
    // ("circuit-X" naming).
    for &off in &t.circuit_offsets {
        let n = off as usize;
        let name_size = u16::from_be_bytes([TWO_CIRCUIT_FIXTURE[n], TWO_CIRCUIT_FIXTURE[n + 1]]);
        assert_eq!(name_size, 10, "name_size mismatch at offset {n}");
    }
}

#[test]
fn fixture_circuit_header_decodes_bell_state_layout() {
    // Bell circuit dumped above: QuantumCircuit(2, 2); h(0); cx(0,1);
    // measure([0,1], [0,1]). 4 instructions, 2 registers
    // (qreg + creg), no global phase metadata, no input/output vars.
    let h = read_header(FIXTURE).unwrap();
    let t = read_program_table(FIXTURE, &h).unwrap();
    let off = t.circuit_offsets[0] as usize;
    let ch = read_circuit_header(FIXTURE, off).unwrap();
    assert_eq!(ch.name_size, 10);
    assert_eq!(ch.global_phase_type, b'f');
    assert_eq!(ch.global_phase_size, 8);
    assert_eq!(ch.num_qubits, 2);
    assert_eq!(ch.num_clbits, 2);
    // Default Qiskit metadata is `{}` → 2 bytes of JSON.
    assert_eq!(ch.metadata_size, 2);
    assert_eq!(ch.num_registers, 2);
    assert_eq!(ch.num_instructions, 4);
    assert_eq!(ch.num_vars, 0);
}

#[test]
fn two_circuit_fixture_decodes_both_circuit_headers() {
    let h = read_header(TWO_CIRCUIT_FIXTURE).unwrap();
    let t = read_program_table(TWO_CIRCUIT_FIXTURE, &h).unwrap();

    // First circuit: same Bell as the single-circuit fixture.
    let ch1 = read_circuit_header(TWO_CIRCUIT_FIXTURE, t.circuit_offsets[0] as usize).unwrap();
    assert_eq!(ch1.num_qubits, 2);
    assert_eq!(ch1.num_clbits, 2);
    assert_eq!(ch1.num_instructions, 4);
    assert_eq!(ch1.num_registers, 2);

    // Second circuit: QuantumCircuit(3, 3); x(0); h(1); cx(1,2);
    // measure_all() → 3 + 3 = 6 instruction-equivalents (the
    // exact count depends on whether measure_all expands inline);
    // we only assert the structural counts that don't depend on
    // measure_all() internals.
    let ch2 = read_circuit_header(TWO_CIRCUIT_FIXTURE, t.circuit_offsets[1] as usize).unwrap();
    assert_eq!(ch2.num_qubits, 3);
    // Note: Qiskit's measure_all() adds a fresh classical register
    // ("meas") of size num_qubits but **does not** preserve any
    // pre-existing classical register from the constructor. So the
    // 3-clbit register passed to QuantumCircuit(3, 3) gets shadowed
    // by measure_all's "meas" register; the producer wrote
    // num_clbits=6 (3 original + 3 new). Pin it here so a future
    // qiskit version that changes that behaviour gets caught.
    assert_eq!(ch2.num_clbits, 6);
    // Sanity: at least the 3 explicit gates plus measurements.
    assert!(ch2.num_instructions >= 4);
}

#[test]
fn fixture_circuit_body_decodes_bell_state_name_phase_metadata() {
    let h = read_header(FIXTURE).unwrap();
    let t = read_program_table(FIXTURE, &h).unwrap();
    let circ_off = t.circuit_offsets[0] as usize;
    let ch = read_circuit_header(FIXTURE, circ_off).unwrap();
    // Variable section starts immediately after the 37-byte fixed
    // CIRCUIT_HEADER struct.
    let body_off = circ_off + 37;
    let body = read_circuit_body(FIXTURE, &ch, body_off).unwrap();
    // Name: Qiskit auto-names ad-hoc circuits "circuit-NN" where NN
    // is a global counter; the fixture was captured with NN=41.
    // Pinning the prefix not the full string protects us if the
    // counter rolls over between regenerations of the fixture.
    assert!(
        body.name.starts_with("circuit-"),
        "unexpected name: {:?}",
        body.name
    );
    assert_eq!(body.name.len(), 10);
    // No explicit global phase set → 0.0.
    match body.global_phase {
        GlobalPhase::Float(f) => assert_eq!(f, 0.0),
        other => panic!("expected Float(0.0), got {other:?}"),
    }
    assert_eq!(body.metadata, "{}");
    assert_eq!(body.body_len, 10 + 8 + 2);
}

#[test]
fn fixture_register_table_decodes_qreg_and_creg() {
    let h = read_header(FIXTURE).unwrap();
    let t = read_program_table(FIXTURE, &h).unwrap();
    let circ_off = t.circuit_offsets[0] as usize;
    let ch = read_circuit_header(FIXTURE, circ_off).unwrap();
    let body_off = circ_off + 37;
    let body = read_circuit_body(FIXTURE, &ch, body_off).unwrap();
    let reg_off = body_off + body.body_len;
    // Bell circuit registers: qreg q[2] + creg c[2].
    let regs = read_register_table(FIXTURE, ch.num_registers, reg_off).unwrap();
    assert_eq!(regs.registers.len(), 2);

    let q = &regs.registers[0];
    assert_eq!(q.kind, RegisterKind::Quantum);
    assert!(q.standalone);
    assert_eq!(q.size, 2);
    assert!(q.in_circuit);
    assert_eq!(q.name, "q");
    assert_eq!(q.bit_index, vec![0, 1]);

    let c = &regs.registers[1];
    assert_eq!(c.kind, RegisterKind::Classical);
    assert!(c.standalone);
    assert_eq!(c.size, 2);
    assert!(c.in_circuit);
    assert_eq!(c.name, "c");
    assert_eq!(c.bit_index, vec![0, 1]);

    // Each register: 9 (header) + 1 (name) + 16 (2 × i64) = 26 bytes;
    // table_len = 52.
    assert_eq!(regs.table_len, 52);
}

#[test]
fn fixture_decodes_through_to_first_instruction() {
    // Walks the full layered chain on the bell QPY blob:
    //   file header → program table → circuit header → circuit
    //   body → register table → extras (annotation + custom_def
    //   counts) → first INSTRUCTION_V2 header.
    let h = read_header(FIXTURE).unwrap();
    let t = read_program_table(FIXTURE, &h).unwrap();
    let circ_off = t.circuit_offsets[0] as usize;
    let ch = read_circuit_header(FIXTURE, circ_off).unwrap();
    let body_off = circ_off + 37;
    let body = read_circuit_body(FIXTURE, &ch, body_off).unwrap();
    let regs_off = body_off + body.body_len;
    let regs = read_register_table(FIXTURE, ch.num_registers, regs_off).unwrap();
    let extras_off = regs_off + regs.table_len;
    let extras = read_circuit_extras(FIXTURE, &h, extras_off).unwrap();
    // Bell circuit has neither annotations nor custom-defined gates.
    assert_eq!(extras.num_namespaces, Some(0));
    assert_eq!(extras.num_custom_definitions, 0);
    assert_eq!(extras.extras_len, 4 + 8); // V17 → annotation_header + custom_def_header

    let inst_off = extras_off + extras.extras_len;
    let inst = read_instruction_header(FIXTURE, inst_off).unwrap();
    // First instruction in the bell sequence is the Hadamard.
    // Qiskit serialises gates by their Python class name.
    assert_eq!(inst.name, "HGate");
    assert_eq!(inst.label, "");
    assert_eq!(inst.num_parameters, 0);
    assert_eq!(inst.num_qargs, 1);
    assert_eq!(inst.num_cargs, 0);
    assert_eq!(inst.extras_key.condition, ConditionKind::None);
    assert!(!inst.extras_key.annotations);
    assert_eq!(inst.num_ctrl_qubits, 0);
    assert_eq!(inst.ctrl_state, 0);
    // Fixed 33 bytes + name "HGate" (5 bytes) + empty label = 38.
    assert_eq!(inst.header_len, 33 + 5);
}

#[test]
fn fixture_decodes_instruction_args_for_h_and_cx() {
    // Walk to the H instruction's qarg, then to the CX
    // instruction's two qargs. Each INSTRUCTION_ARG is 5 bytes.
    let h = read_header(FIXTURE).unwrap();
    let t = read_program_table(FIXTURE, &h).unwrap();
    let circ_off = t.circuit_offsets[0] as usize;
    let ch = read_circuit_header(FIXTURE, circ_off).unwrap();
    let body_off = circ_off + 37;
    let body = read_circuit_body(FIXTURE, &ch, body_off).unwrap();
    let regs_off = body_off + body.body_len;
    let regs = read_register_table(FIXTURE, ch.num_registers, regs_off).unwrap();
    let extras_off = regs_off + regs.table_len;
    let extras = read_circuit_extras(FIXTURE, &h, extras_off).unwrap();
    let mut cursor = extras_off + extras.extras_len;

    // ----- H instruction: 1 qarg, 0 cargs -----
    let h_inst = read_instruction_header(FIXTURE, cursor).unwrap();
    cursor += h_inst.header_len;
    assert_eq!(h_inst.name, "HGate");
    assert_eq!(h_inst.num_qargs, 1);
    assert_eq!(h_inst.num_cargs, 0);
    let (h_args, h_args_len) =
        read_instruction_args(FIXTURE, h_inst.num_qargs, h_inst.num_cargs, cursor).unwrap();
    assert_eq!(h_args, vec![InstructionArg::Qubit(0)]);
    assert_eq!(h_args_len, 5);
    cursor += h_args_len;
    // H takes no parameters → next instruction follows immediately.

    // ----- CX instruction: 2 qargs, 0 cargs -----
    let cx_inst = read_instruction_header(FIXTURE, cursor).unwrap();
    cursor += cx_inst.header_len;
    assert_eq!(cx_inst.name, "CXGate");
    assert_eq!(cx_inst.num_qargs, 2);
    assert_eq!(cx_inst.num_cargs, 0);
    assert_eq!(cx_inst.num_ctrl_qubits, 1);
    assert_eq!(cx_inst.ctrl_state, 1);
    let (cx_args, cx_args_len) =
        read_instruction_args(FIXTURE, cx_inst.num_qargs, cx_inst.num_cargs, cursor).unwrap();
    assert_eq!(
        cx_args,
        vec![InstructionArg::Qubit(0), InstructionArg::Qubit(1)]
    );
    assert_eq!(cx_args_len, 10);
}

#[test]
fn fixture_rx_decodes_through_to_pi_quarter_param() {
    // Walk the Rx(π/4) parametric fixture all the way through the
    // INSTRUCTION_PARAM list. This is the first place the reader
    // touches a non-empty parameters section.
    let h = read_header(RX_PARAM_FIXTURE).unwrap();
    let t = read_program_table(RX_PARAM_FIXTURE, &h).unwrap();
    let circ_off = t.circuit_offsets[0] as usize;
    let ch = read_circuit_header(RX_PARAM_FIXTURE, circ_off).unwrap();
    assert_eq!(ch.num_qubits, 1);
    assert_eq!(ch.num_clbits, 0);
    assert_eq!(ch.num_registers, 1);
    assert_eq!(ch.num_instructions, 1);

    let body_off = circ_off + 37;
    let body = read_circuit_body(RX_PARAM_FIXTURE, &ch, body_off).unwrap();
    let regs_off = body_off + body.body_len;
    let regs = read_register_table(RX_PARAM_FIXTURE, ch.num_registers, regs_off).unwrap();
    let extras_off = regs_off + regs.table_len;
    let extras = read_circuit_extras(RX_PARAM_FIXTURE, &h, extras_off).unwrap();
    let mut cursor = extras_off + extras.extras_len;

    // ----- Rx(π/4) instruction: 1 qarg, 0 cargs, 1 param -----
    let rx_inst = read_instruction_header(RX_PARAM_FIXTURE, cursor).unwrap();
    cursor += rx_inst.header_len;
    assert_eq!(rx_inst.name, "RXGate");
    assert_eq!(rx_inst.num_qargs, 1);
    assert_eq!(rx_inst.num_cargs, 0);
    assert_eq!(rx_inst.num_parameters, 1);

    let (rx_args, rx_args_len) = read_instruction_args(
        RX_PARAM_FIXTURE,
        rx_inst.num_qargs,
        rx_inst.num_cargs,
        cursor,
    )
    .unwrap();
    assert_eq!(rx_args, vec![InstructionArg::Qubit(0)]);
    cursor += rx_args_len;

    let (rx_params, _params_len) =
        read_instruction_params(RX_PARAM_FIXTURE, rx_inst.num_parameters, cursor).unwrap();
    assert_eq!(rx_params.len(), 1);
    match rx_params[0] {
        InstructionParam::Float(f) => {
            // π/4 round-trips bit-exact through QPY's little-endian
            // f64 encoding.
            assert_eq!(f, std::f64::consts::FRAC_PI_4);
        }
        ref other => panic!("expected Float(π/4), got {other:?}"),
    }
}

#[test]
fn fixture_walks_full_bell_circuit_instruction_list() {
    // Drive the full instruction stream via the per-instruction
    // aggregator and the multi-instruction walker. Bell circuit
    // emits (H, CX, Measure, Measure) — all parameterless.
    let h = read_header(FIXTURE).unwrap();
    let t = read_program_table(FIXTURE, &h).unwrap();
    let circ_off = t.circuit_offsets[0] as usize;
    let ch = read_circuit_header(FIXTURE, circ_off).unwrap();
    let body_off = circ_off + 37;
    let body = read_circuit_body(FIXTURE, &ch, body_off).unwrap();
    let regs_off = body_off + body.body_len;
    let regs = read_register_table(FIXTURE, ch.num_registers, regs_off).unwrap();
    let extras_off = regs_off + regs.table_len;
    let extras = read_circuit_extras(FIXTURE, &h, extras_off).unwrap();
    let inst_off = extras_off + extras.extras_len;

    let (insts, _consumed) =
        read_circuit_instructions(FIXTURE, ch.num_instructions, inst_off).unwrap();
    assert_eq!(insts.len(), 4);

    // Index 0: H gate on qubit 0.
    assert_eq!(insts[0].name, "HGate");
    assert_eq!(insts[0].args, vec![InstructionArg::Qubit(0)]);
    assert!(insts[0].params.is_empty());

    // Index 1: CX with control qubit 0, target qubit 1.
    assert_eq!(insts[1].name, "CXGate");
    assert_eq!(
        insts[1].args,
        vec![InstructionArg::Qubit(0), InstructionArg::Qubit(1)]
    );
    assert_eq!(insts[1].num_ctrl_qubits, 1);
    assert_eq!(insts[1].ctrl_state, 1);

    // Indices 2 and 3: Measure(qubit i, clbit i) for i ∈ {0, 1}.
    for (i, m) in insts.iter().enumerate().skip(2) {
        let bit = (i - 2) as u32;
        assert_eq!(m.name, "Measure");
        assert_eq!(
            m.args,
            vec![InstructionArg::Qubit(bit), InstructionArg::Clbit(bit)]
        );
        assert!(m.params.is_empty());
        assert_eq!(m.condition, ConditionKind::None);
    }
}

#[test]
fn fixture_bell_decodes_to_full_circuit_ir() {
    // Top-level entry point: bytes → CircuitIR.
    let ir = read_qpy_circuit_ir(FIXTURE).unwrap();
    assert_eq!(ir.num_qubits, 2);
    assert_eq!(ir.num_classical_bits, 2);
    assert_eq!(ir.ops.len(), 4);

    assert_eq!(ir.ops[0].gate, GateKind::H);
    assert_eq!(&ir.ops[0].qubits[..], &[Qubit(0)]);
    assert!(ir.ops[0].params.is_empty());

    assert_eq!(ir.ops[1].gate, GateKind::CX);
    assert_eq!(&ir.ops[1].qubits[..], &[Qubit(0), Qubit(1)]);

    for (i, op) in ir.ops.iter().enumerate().skip(2) {
        let bit = (i - 2) as u32;
        assert_eq!(op.gate, GateKind::Measure);
        assert_eq!(&op.qubits[..], &[Qubit(bit)]);
        assert_eq!(op.classical_bit, Some(bit));
    }
}

#[test]
fn fixture_rx_decodes_to_circuit_ir_with_concrete_angle_param() {
    let ir = read_qpy_circuit_ir(RX_PARAM_FIXTURE).unwrap();
    assert_eq!(ir.num_qubits, 1);
    assert_eq!(ir.num_classical_bits, 0);
    assert_eq!(ir.ops.len(), 1);

    assert_eq!(ir.ops[0].gate, GateKind::Rx);
    assert_eq!(&ir.ops[0].qubits[..], &[Qubit(0)]);
    assert_eq!(ir.ops[0].params.len(), 1);
    match &ir.ops[0].params[0] {
        omega_core::circuit::ParamExpr::Concrete(f) => {
            assert_eq!(*f, std::f64::consts::FRAC_PI_4);
        }
        other => panic!("expected Concrete(π/4), got {other:?}"),
    }
}

#[test]
fn two_circuit_fixture_decodes_to_two_circuit_irs() {
    let irs = read_qpy_circuit_irs(TWO_CIRCUIT_FIXTURE).unwrap();
    assert_eq!(irs.len(), 2);

    // First circuit: same Bell as the single-fixture case.
    assert_eq!(irs[0].num_qubits, 2);
    assert_eq!(irs[0].num_classical_bits, 2);
    assert_eq!(irs[0].ops[0].gate, GateKind::H);
    assert_eq!(irs[0].ops[1].gate, GateKind::CX);

    // Second circuit: x(0); h(1); cx(1, 2); measure_all().
    assert_eq!(irs[1].num_qubits, 3);
    assert_eq!(irs[1].ops[0].gate, GateKind::X);
    assert_eq!(&irs[1].ops[0].qubits[..], &[Qubit(0)]);
    assert_eq!(irs[1].ops[1].gate, GateKind::H);
    assert_eq!(&irs[1].ops[1].qubits[..], &[Qubit(1)]);
    assert_eq!(irs[1].ops[2].gate, GateKind::CX);
    assert_eq!(&irs[1].ops[2].qubits[..], &[Qubit(1), Qubit(2)]);
    // The remaining ops are the Barrier from `measure_all()` plus
    // the three Measure ops (q[i] → c[i+3], since measure_all adds
    // its own "meas" register on top of the constructor's clbits).
    assert!(irs[1].ops.len() >= 4);
}

#[test]
fn read_qpy_circuit_ir_rejects_two_circuit_blob_with_unsupported() {
    match read_qpy_circuit_ir(TWO_CIRCUIT_FIXTURE).unwrap_err() {
        omega_bridges::qpy::QpyError::Unsupported { what, .. } => {
            assert!(what.contains("multi-circuit"));
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn fixture_wide_gates_decode_to_expected_native_gate_kinds() {
    // Pins the gate-name → GateKind mapping against a real
    // Qiskit-2.4.1 blob exercising 20 gates across most of the
    // single + two + three-qubit gate set omega represents
    // natively. Catches accidental string typos in the mapping
    // table the moment a fixture regeneration drifts.
    let ir = read_qpy_circuit_ir(WIDE_GATES_FIXTURE).unwrap();
    assert_eq!(ir.num_qubits, 3);
    assert_eq!(ir.num_classical_bits, 0);
    let kinds: Vec<_> = ir.ops.iter().map(|op| op.gate.clone()).collect();
    assert_eq!(
        kinds,
        vec![
            GateKind::H,
            GateKind::X,
            GateKind::Y,
            GateKind::Z,
            GateKind::S,
            GateKind::Sdg,
            GateKind::T,
            GateKind::Tdg,
            GateKind::Id,
            GateKind::Rx,
            GateKind::Ry,
            GateKind::Rz,
            GateKind::U3,
            GateKind::CX,
            GateKind::CY,
            GateKind::CZ,
            GateKind::Swap,
            GateKind::CRz,
            GateKind::CCX,
            GateKind::CSwap,
        ]
    );
    // Spot-check parametric gates: Rx(π/4) and U3(π/2, π/4, π/8).
    if let omega_core::circuit::ParamExpr::Concrete(f) = ir.ops[9].params[0] {
        assert_eq!(f, std::f64::consts::FRAC_PI_4);
    } else {
        panic!("Rx param should be Concrete");
    }
    assert_eq!(ir.ops[12].params.len(), 3);
}

#[test]
fn fixture_rx_symbolic_decodes_to_a_named_omega_symbol() {
    // `qc.rx(Parameter('θ'), 0)` — the pure-Rust reader resolves
    // the QPY 'p' (PARAMETER) payload via the H16s + utf-8 layout
    // and registers a `SymbolId` in the resulting CircuitIR.
    let ir = read_qpy_circuit_ir(RX_SYMBOLIC_FIXTURE).unwrap();
    assert_eq!(ir.num_qubits, 1);
    assert_eq!(ir.ops.len(), 1);
    assert_eq!(ir.ops[0].gate, GateKind::Rx);
    assert_eq!(ir.ops[0].params.len(), 1);

    let sym_id = match ir.ops[0].params[0] {
        omega_core::circuit::ParamExpr::Symbol(id) => id,
        ref other => panic!("expected Symbol, got {other:?}"),
    };
    assert_eq!(ir.symbols.get(&sym_id).map(String::as_str), Some("θ"));
    assert_eq!(ir.symbols.len(), 1);
}

#[test]
fn two_params_fixture_dedups_repeated_parameter_to_one_symbol_id() {
    // θ appears twice (Rx + CRz). Both occurrences must resolve to
    // the same omega SymbolId so a single `--params θ=v φ=v'`
    // binding works for the whole circuit. φ is used once.
    let ir = read_qpy_circuit_ir(TWO_PARAMS_FIXTURE).unwrap();
    assert_eq!(ir.num_qubits, 2);
    assert_eq!(ir.ops.len(), 3);

    // Pull the SymbolIds out of the three ops' params.
    let sym_id =
        |op: &omega_core::circuit::GateOp| match op.params.first().expect("each op has 1 param") {
            omega_core::circuit::ParamExpr::Symbol(id) => *id,
            other => panic!("expected Symbol, got {other:?}"),
        };
    let theta_in_rx = sym_id(&ir.ops[0]);
    let phi_in_ry = sym_id(&ir.ops[1]);
    let theta_in_crz = sym_id(&ir.ops[2]);

    // θ → same SymbolId across both uses.
    assert_eq!(theta_in_rx, theta_in_crz, "θ must dedup across uses");
    // φ → different SymbolId.
    assert_ne!(theta_in_rx, phi_in_ry);

    // Symbol table has exactly two entries with the right names.
    assert_eq!(ir.symbols.len(), 2);
    assert_eq!(ir.symbols.get(&theta_in_rx).map(String::as_str), Some("θ"));
    assert_eq!(ir.symbols.get(&phi_in_ry).map(String::as_str), Some("φ"));
}

#[test]
fn rx_paramexpr_fixture_decodes_to_add_concrete_symbol() {
    // `qc.rx(Parameter('θ') + 1.5, 0)`. Qiskit's writer normalises
    // the binary expression to ADD(1.5, theta); the omega reader
    // mirrors that order in the resulting ParamExpr tree.
    let ir = read_qpy_circuit_ir(RX_PARAMEXPR_FIXTURE).unwrap();
    assert_eq!(ir.num_qubits, 1);
    assert_eq!(ir.ops.len(), 1);
    assert_eq!(ir.ops[0].gate, GateKind::Rx);
    assert_eq!(ir.ops[0].params.len(), 1);
    match &ir.ops[0].params[0] {
        omega_core::circuit::ParamExpr::Add(lhs, rhs) => {
            match lhs.as_ref() {
                omega_core::circuit::ParamExpr::Concrete(f) => assert_eq!(*f, 1.5),
                other => panic!("expected lhs Concrete(1.5), got {other:?}"),
            }
            match rhs.as_ref() {
                omega_core::circuit::ParamExpr::Symbol(id) => {
                    assert_eq!(ir.symbols.get(id).map(String::as_str), Some("θ"));
                }
                other => panic!("expected rhs Symbol(θ), got {other:?}"),
            }
        }
        other => panic!("expected Add, got {other:?}"),
    }
    assert_eq!(ir.symbols.len(), 1);
}

#[test]
fn rx_paramexpr_compound_fixture_decodes_to_two_record_stack_carry() {
    // `(θ + 1.0) * 2.0` — Qiskit emits two PARAM_EXPR_ELEM_V13
    // records: ADD(1, θ) then MUL(None, 2). The omega evaluator
    // walks the replay stack and produces Mul(Add(1.0, θ), 2.0).
    let ir = read_qpy_circuit_ir(RX_PARAMEXPR_COMPOUND_FIXTURE).unwrap();
    assert_eq!(ir.ops.len(), 1);
    assert_eq!(ir.ops[0].gate, GateKind::Rx);
    assert_eq!(ir.ops[0].params.len(), 1);
    match &ir.ops[0].params[0] {
        omega_core::circuit::ParamExpr::Mul(inner, two) => {
            match two.as_ref() {
                omega_core::circuit::ParamExpr::Concrete(f) => assert_eq!(*f, 2.0),
                other => panic!("expected outer rhs Concrete(2.0), got {other:?}"),
            }
            match inner.as_ref() {
                omega_core::circuit::ParamExpr::Add(a, b) => {
                    assert!(matches!(**a, omega_core::circuit::ParamExpr::Concrete(f) if f == 1.0));
                    assert!(matches!(**b, omega_core::circuit::ParamExpr::Symbol(_)));
                }
                other => panic!("expected inner Add, got {other:?}"),
            }
        }
        other => panic!("expected Mul(Add, Concrete), got {other:?}"),
    }
}

#[test]
fn rx_paramexpr_rsub_fixture_decodes_to_one_minus_phi() {
    // `1.5 - Parameter('φ')` — Qiskit emits RSUB(LHS=φ, RHS=1.5)
    // with the semantic `lhs.__rsub__(rhs) = rhs - lhs`. omega
    // collapses this to `Add(rhs_pushed, Negate(lhs_pushed))` which
    // reads as `Add(Concrete(1.5), Negate(Symbol(φ)))`.
    let ir = read_qpy_circuit_ir(RX_PARAMEXPR_RSUB_FIXTURE).unwrap();
    assert_eq!(ir.ops.len(), 1);
    assert_eq!(ir.ops[0].gate, GateKind::Rx);
    assert_eq!(ir.ops[0].params.len(), 1);
    match &ir.ops[0].params[0] {
        omega_core::circuit::ParamExpr::Add(constant, negated_phi) => {
            match constant.as_ref() {
                omega_core::circuit::ParamExpr::Concrete(f) => assert_eq!(*f, 1.5),
                other => panic!("expected Concrete(1.5), got {other:?}"),
            }
            match negated_phi.as_ref() {
                omega_core::circuit::ParamExpr::Negate(inner) => {
                    assert!(matches!(**inner, omega_core::circuit::ParamExpr::Symbol(_)));
                }
                other => panic!("expected Negate(Symbol(φ)), got {other:?}"),
            }
        }
        other => panic!("expected Add(Concrete, Negate(Symbol)), got {other:?}"),
    }
    assert_eq!(ir.symbols.len(), 1);
    let sym_name: Vec<&str> = ir.symbols.values().map(String::as_str).collect();
    assert!(sym_name.contains(&"φ"));
}

#[test]
fn rx_paramvector_fixture_decodes_to_three_indexed_symbols_with_dedup() {
    // ParameterVector('x', 3) used across Rx(pv[0]), Ry(pv[1]),
    // Rz(pv[2]), Rx(pv[0]). Each indexed element interns to its own
    // SymbolId, but the repeated `pv[0]` reference must dedup to the
    // same SymbolId as its first occurrence.
    let ir = read_qpy_circuit_ir(RX_PARAMVECTOR_FIXTURE).unwrap();
    assert_eq!(ir.num_qubits, 3);
    assert_eq!(ir.ops.len(), 4);
    assert_eq!(ir.ops[0].gate, GateKind::Rx);
    assert_eq!(ir.ops[1].gate, GateKind::Ry);
    assert_eq!(ir.ops[2].gate, GateKind::Rz);
    assert_eq!(ir.ops[3].gate, GateKind::Rx);

    // Pull the SymbolIds out of each op's single param.
    let sym_id =
        |op: &omega_core::circuit::GateOp| match op.params.first().expect("each op has 1 param") {
            omega_core::circuit::ParamExpr::Symbol(id) => *id,
            other => panic!("expected Symbol, got {other:?}"),
        };
    let pv0_first = sym_id(&ir.ops[0]);
    let pv1 = sym_id(&ir.ops[1]);
    let pv2 = sym_id(&ir.ops[2]);
    let pv0_repeat = sym_id(&ir.ops[3]);

    assert_eq!(pv0_first, pv0_repeat, "pv[0] must dedup to one SymbolId");
    assert_ne!(pv0_first, pv1, "pv[0] and pv[1] must be distinct symbols");
    assert_ne!(pv1, pv2, "pv[1] and pv[2] must be distinct symbols");

    // Symbol table has exactly three entries with conventional names.
    assert_eq!(ir.symbols.len(), 3);
    assert_eq!(ir.symbols.get(&pv0_first).map(String::as_str), Some("x[0]"));
    assert_eq!(ir.symbols.get(&pv1).map(String::as_str), Some("x[1]"));
    assert_eq!(ir.symbols.get(&pv2).map(String::as_str), Some("x[2]"));
}

#[test]
fn rx_paramvector_expr_fixture_evaluates_pe_with_pv_symbols() {
    // ParameterVector('gamma', 2) used inside a ParameterExpression:
    // `2.0 * gammas[0] + gammas[1]`. The PE symbol map carries two
    // b'v' PARAMETER_VECTOR_ELEMENT entries — the evaluator must
    // intern both elements through SymbolLookup with display names
    // `gamma[0]` and `gamma[1]`, then assemble the algebraic tree.
    let ir = read_qpy_circuit_ir(RX_PARAMVECTOR_EXPR_FIXTURE).unwrap();
    assert_eq!(ir.num_qubits, 1);
    assert_eq!(ir.ops.len(), 1);
    assert_eq!(ir.ops[0].gate, GateKind::Rx);
    assert_eq!(ir.ops[0].params.len(), 1);

    // Symbol table holds gamma[0] and gamma[1] as distinct entries.
    let names: std::collections::HashSet<&str> = ir.symbols.values().map(String::as_str).collect();
    assert_eq!(ir.symbols.len(), 2);
    assert!(names.contains("gamma[0]"));
    assert!(names.contains("gamma[1]"));

    // The expression tree must reference both gamma symbols. The
    // exact tree shape depends on how Qiskit normalises the
    // expression — assert structural invariants instead of an exact
    // shape: at least one Add/Mul node, and both gamma SymbolIds
    // appear somewhere as a leaf.
    fn collect_symbols(expr: &omega_core::circuit::ParamExpr) -> Vec<u32> {
        let mut out = Vec::new();
        fn walk(e: &omega_core::circuit::ParamExpr, out: &mut Vec<u32>) {
            match e {
                omega_core::circuit::ParamExpr::Symbol(id) => out.push(*id),
                omega_core::circuit::ParamExpr::Concrete(_) => {}
                omega_core::circuit::ParamExpr::Negate(a) => walk(a, out),
                omega_core::circuit::ParamExpr::Add(a, b)
                | omega_core::circuit::ParamExpr::Mul(a, b) => {
                    walk(a, out);
                    walk(b, out);
                }
            }
        }
        walk(expr, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }
    let referenced = collect_symbols(&ir.ops[0].params[0]);
    assert_eq!(
        referenced.len(),
        2,
        "expected both gamma elements in the expression"
    );
}

// -------------------- Corruption-resistance tests --------------------
//
// Fuzz-style: chop a byte off the end / flip a magic / blank a
// region in the middle of a known-good fixture, confirm the reader
// surfaces a typed error rather than panicking. These pin the
// "hostile input" surface so a follow-up that adds bounds checks
// in a new layer can't silently regress error reporting.

#[test]
fn truncated_fixture_at_each_layer_returns_typed_truncated_error() {
    // For each "structural boundary" in the bell fixture, slice the
    // input one byte short of that boundary and confirm we get a
    // typed `Truncated` error.
    use omega_bridges::qpy::QpyError;
    for &cut in &[5usize, 18, 27, 30, 64, 75, 100, 137, 150, 188] {
        let truncated = &FIXTURE[..cut];
        let res = read_qpy_circuit_ir(truncated);
        match res {
            Err(QpyError::Truncated { .. }) | Err(QpyError::NotAQpyBlob) => {}
            Err(other) => panic!("cut={cut}: expected Truncated or NotAQpyBlob, got {other:?}"),
            Ok(_) => panic!("cut={cut}: somehow decoded a truncated blob"),
        }
    }
}

#[test]
fn corrupted_magic_returns_not_a_qpy_blob_without_panic() {
    use omega_bridges::qpy::QpyError;
    let mut blob = FIXTURE.to_vec();
    // Flip the second magic byte from 'I' to '!' so the magic check
    // fails fast.
    blob[1] = b'!';
    match read_qpy_circuit_ir(&blob).unwrap_err() {
        QpyError::NotAQpyBlob => {}
        other => panic!("expected NotAQpyBlob, got {other:?}"),
    }
}

#[test]
fn corrupted_qpy_version_byte_returns_unsupported_or_invalid() {
    use omega_bridges::qpy::QpyError;
    // Set qpy_version byte to 0xFF (way past MAX_SUPPORTED_VERSION).
    let mut blob = FIXTURE.to_vec();
    blob[6] = 0xFF;
    match read_qpy_circuit_ir(&blob).unwrap_err() {
        QpyError::UnsupportedVersion {
            version: 0xFF,
            supported_max: _,
        } => {}
        other => panic!("expected UnsupportedVersion, got {other:?}"),
    }

    // Set qpy_version to 0 — the explicit "invalid version 0" path.
    let mut blob = FIXTURE.to_vec();
    blob[6] = 0;
    match read_qpy_circuit_ir(&blob).unwrap_err() {
        QpyError::InvalidVersionZero => {}
        other => panic!("expected InvalidVersionZero, got {other:?}"),
    }
}
