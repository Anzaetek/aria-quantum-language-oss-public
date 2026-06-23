//! Per-instruction aggregator: bundles
//! [`super::instruction_header::InstructionHeader`] +
//! [`super::instruction_args`] + [`super::instruction_params`] into
//! one [`DecodedInstruction`] and walks `num_instructions` of them
//! in sequence via [`read_circuit_instructions`].
//!
//! This is the layer the omega CircuitIR builder hooks into. Today
//! the aggregator decodes single-clbit `Bit` conditions (TWO_TUPLE
//! with a `\x00<index>` condition_register payload) and surfaces them
//! via [`DecodedInstruction::condition_register`]; the
//! classical-expression variant (extras_key low bits == 2) and
//! annotations (extras_key high bit set) still surface as
//! `Unsupported` — the per-payload decoders for both ride the same
//! INSTRUCTION_PARAM machinery and land in the next slice.

use super::instruction_args::{read_instruction_args, InstructionArg};
use super::instruction_header::{read_instruction_header, ConditionKind};
use super::instruction_params::{read_instruction_params, InstructionParam};
use super::QpyError;

/// One fully decoded instruction. Borrows the input slice for the
/// `name` and `label` fields (cheap to keep around) but owns the
/// arg + param lists since they're already small.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedInstruction<'a> {
    /// Qiskit Python class name (e.g. `"HGate"`, `"CXGate"`,
    /// `"RXGate"`, `"Measure"`).
    pub name: &'a str,
    /// Optional user-set label, empty by default.
    pub label: &'a str,
    /// Quantum args first, then classical args. Each carries its
    /// circuit-level qubit / clbit index.
    pub args: Vec<InstructionArg>,
    /// `num_qargs` — useful so the caller doesn't have to count
    /// `args.iter().filter(...)` to split qargs from cargs.
    pub num_qargs: u32,
    /// Decoded parameter list (scalar variants only today; richer
    /// types fall through to `InstructionParam::Opaque`).
    pub params: Vec<InstructionParam>,
    /// Whether the instruction carries a non-trivial condition
    /// (classical-bit or register/expression). `None` and `Bit` round-trip
    /// through the builder today; `RegisterOrExpr` still surfaces as
    /// `Unsupported` from [`read_decoded_instruction`].
    pub condition: ConditionKind,
    /// Condition value (only meaningful when `condition !=
    /// ConditionKind::None`).
    pub condition_value: i64,
    /// UTF-8 bytes of the `condition_register` payload that immediately
    /// follows `label` on the wire (only present when condition is
    /// `Bit`; empty otherwise). For a single-clbit condition the bytes
    /// are `b"\x00<index>"` where `<index>` is the decimal clbit index;
    /// for a register condition they hold the register's UTF-8 name.
    pub condition_register: Vec<u8>,
    /// Number of control qubits (V2+). 0 for non-controlled gates.
    pub num_ctrl_qubits: u32,
    /// Controlled-gate state (V2+). `1` for the standard `|1⟩`
    /// control; non-1 only for controlled gates that fire on a
    /// `|0⟩` (or mixed) control state.
    pub ctrl_state: u32,
    /// Total bytes consumed by header + args + params.
    pub total_len: usize,
}

/// Decode a single instruction starting at `offset`.
pub fn read_decoded_instruction(
    bytes: &[u8],
    offset: usize,
) -> Result<DecodedInstruction<'_>, QpyError> {
    let header = read_instruction_header(bytes, offset)?;
    if header.extras_key.condition == ConditionKind::RegisterOrExpr {
        return Err(QpyError::Unsupported {
            what: "instruction.condition_register payload",
            detail: "extras_key low bits == 2; per-register-condition decoder lands with the symbolic INSTRUCTION_PARAM slice",
        });
    }
    if header.extras_key.annotations {
        return Err(QpyError::Unsupported {
            what: "instruction.annotations",
            detail: "extras_key high bit set; INSTRUCTION_ANNOTATIONS_HEADER decoder lands with the namespace-payload slice",
        });
    }

    let mut cursor = offset + header.header_len;

    // V5+: a `condition_register` UTF-8 blob of `condition_register_size`
    // bytes sits between the (name+label) section and the args section.
    // Qiskit writes it for both TWO_TUPLE (`Bit`) and EXPRESSION
    // conditions; the EXPRESSION variant carries an additional encoded
    // value payload, which is the part still unsupported here.
    let condition_register_len = header.condition_register_size as usize;
    let condition_register_end =
        cursor
            .checked_add(condition_register_len)
            .ok_or(QpyError::Truncated {
                what: "instruction.condition_register",
                need: usize::MAX,
                got: bytes.len(),
            })?;
    if bytes.len() < condition_register_end {
        return Err(QpyError::Truncated {
            what: "instruction.condition_register",
            need: condition_register_end,
            got: bytes.len(),
        });
    }
    let condition_register = bytes[cursor..condition_register_end].to_vec();
    cursor = condition_register_end;

    let (args, args_len) =
        read_instruction_args(bytes, header.num_qargs, header.num_cargs, cursor)?;
    cursor += args_len;
    let (params, params_len) = read_instruction_params(bytes, header.num_parameters, cursor)?;
    cursor += params_len;

    Ok(DecodedInstruction {
        name: header.name,
        label: header.label,
        args,
        num_qargs: header.num_qargs,
        params,
        condition: header.extras_key.condition,
        condition_value: header.condition_value,
        condition_register,
        num_ctrl_qubits: header.num_ctrl_qubits,
        ctrl_state: header.ctrl_state,
        total_len: cursor - offset,
    })
}

/// Walk `num_instructions` consecutive instructions starting at
/// `offset`. Returns the decoded list plus the total bytes
/// consumed (= sum of each instruction's `total_len`).
pub fn read_circuit_instructions(
    bytes: &[u8],
    num_instructions: u64,
    offset: usize,
) -> Result<(Vec<DecodedInstruction<'_>>, usize), QpyError> {
    let mut cursor = offset;
    let mut out = Vec::with_capacity(num_instructions as usize);
    for _ in 0..num_instructions {
        let inst = read_decoded_instruction(bytes, cursor)?;
        cursor += inst.total_len;
        out.push(inst);
    }
    Ok((out, cursor - offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic INSTRUCTION_V2 header (33 bytes) + name +
    /// label + qargs + cargs + params, all back-to-back. Sized so a
    /// caller can chain multiple of these to test
    /// `read_circuit_instructions`.
    #[allow(clippy::too_many_arguments)]
    fn synth_instruction_bytes(
        name: &str,
        label: &str,
        num_parameters: u16,
        qargs: &[u32],
        cargs: &[u32],
        extras_key: u8,
        condition_value: i64,
        num_ctrl_qubits: u32,
        ctrl_state: u32,
        param_payloads: &[(u8, Vec<u8>)],
    ) -> Vec<u8> {
        synth_instruction_bytes_with_condition_register(
            name,
            label,
            num_parameters,
            qargs,
            cargs,
            extras_key,
            condition_value,
            num_ctrl_qubits,
            ctrl_state,
            b"",
            param_payloads,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn synth_instruction_bytes_with_condition_register(
        name: &str,
        label: &str,
        num_parameters: u16,
        qargs: &[u32],
        cargs: &[u32],
        extras_key: u8,
        condition_value: i64,
        num_ctrl_qubits: u32,
        ctrl_state: u32,
        condition_register: &[u8],
        param_payloads: &[(u8, Vec<u8>)],
    ) -> Vec<u8> {
        assert_eq!(num_parameters as usize, param_payloads.len());
        let mut out = Vec::new();
        out.extend_from_slice(&(name.len() as u16).to_be_bytes());
        out.extend_from_slice(&(label.len() as u16).to_be_bytes());
        out.extend_from_slice(&num_parameters.to_be_bytes());
        out.extend_from_slice(&(qargs.len() as u32).to_be_bytes());
        out.extend_from_slice(&(cargs.len() as u32).to_be_bytes());
        out.push(extras_key);
        out.extend_from_slice(&(condition_register.len() as u16).to_be_bytes());
        out.extend_from_slice(&condition_value.to_be_bytes());
        out.extend_from_slice(&num_ctrl_qubits.to_be_bytes());
        out.extend_from_slice(&ctrl_state.to_be_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(label.as_bytes());
        out.extend_from_slice(condition_register);
        for &q in qargs {
            out.push(b'q');
            out.extend_from_slice(&q.to_be_bytes());
        }
        for &c in cargs {
            out.push(b'c');
            out.extend_from_slice(&c.to_be_bytes());
        }
        for (type_byte, payload) in param_payloads {
            out.push(*type_byte);
            out.extend_from_slice(&(payload.len() as u64).to_be_bytes());
            out.extend_from_slice(payload);
        }
        out
    }

    #[test]
    fn decodes_a_bare_h_gate_to_a_one_qarg_zero_param_instruction() {
        let blob = synth_instruction_bytes("HGate", "", 0, &[0], &[], 0, 0, 0, 0, &[]);
        let inst = read_decoded_instruction(&blob, 0).unwrap();
        assert_eq!(inst.name, "HGate");
        assert_eq!(inst.label, "");
        assert_eq!(inst.args, vec![InstructionArg::Qubit(0)]);
        assert_eq!(inst.num_qargs, 1);
        assert!(inst.params.is_empty());
        assert_eq!(inst.condition, ConditionKind::None);
        assert_eq!(inst.num_ctrl_qubits, 0);
        // Header = 33 + 5 (name) + 0 (label) = 38; args = 5; params = 0.
        assert_eq!(inst.total_len, 33 + 5 + 5);
    }

    #[test]
    fn decodes_a_cx_gate_with_two_qargs_and_ctrl_metadata() {
        let blob = synth_instruction_bytes("CXGate", "", 0, &[0, 1], &[], 0, 0, 1, 1, &[]);
        let inst = read_decoded_instruction(&blob, 0).unwrap();
        assert_eq!(inst.name, "CXGate");
        assert_eq!(
            inst.args,
            vec![InstructionArg::Qubit(0), InstructionArg::Qubit(1)]
        );
        assert_eq!(inst.num_ctrl_qubits, 1);
        assert_eq!(inst.ctrl_state, 1);
    }

    #[test]
    fn decodes_an_rx_gate_with_one_float_param() {
        let phase = std::f64::consts::FRAC_PI_4;
        let blob = synth_instruction_bytes(
            "RXGate",
            "",
            1,
            &[0],
            &[],
            0,
            0,
            0,
            0,
            &[(b'f', phase.to_le_bytes().to_vec())],
        );
        let inst = read_decoded_instruction(&blob, 0).unwrap();
        assert_eq!(inst.name, "RXGate");
        assert_eq!(inst.params, vec![InstructionParam::Float(phase)]);
    }

    #[test]
    fn decodes_a_measure_with_one_qarg_and_one_carg() {
        let blob = synth_instruction_bytes("Measure", "", 0, &[0], &[0], 0, 0, 0, 0, &[]);
        let inst = read_decoded_instruction(&blob, 0).unwrap();
        assert_eq!(inst.name, "Measure");
        assert_eq!(
            inst.args,
            vec![InstructionArg::Qubit(0), InstructionArg::Clbit(0)]
        );
        assert_eq!(inst.num_qargs, 1);
    }

    #[test]
    fn rejects_register_condition_until_per_register_decoder_lands() {
        let blob = synth_instruction_bytes(
            "XGate",
            "",
            0,
            &[0],
            &[],
            0b0000_0010, // condition kind = RegisterOrExpr
            7,
            0,
            0,
            &[],
        );
        match read_decoded_instruction(&blob, 0).unwrap_err() {
            QpyError::Unsupported { what, .. } => {
                assert!(what.contains("condition_register"));
            }
            other => panic!("expected Unsupported(condition_register), got {other:?}"),
        }
    }

    #[test]
    fn rejects_annotations_until_per_annotation_decoder_lands() {
        let blob = synth_instruction_bytes("XGate", "", 0, &[0], &[], 0b1000_0000, 0, 0, 0, &[]);
        match read_decoded_instruction(&blob, 0).unwrap_err() {
            QpyError::Unsupported { what, .. } => {
                assert!(what.contains("annotations"));
            }
            other => panic!("expected Unsupported(annotations), got {other:?}"),
        }
    }

    #[test]
    fn keeps_bit_condition_through_the_aggregator() {
        // Bit-condition with a single-clbit register payload. The
        // `\x00`-prefixed register name is Qiskit's tag for "single
        // clbit; decimal index follows".
        let blob = synth_instruction_bytes_with_condition_register(
            "XGate",
            "",
            0,
            &[0],
            &[],
            0b0000_0001,
            42,
            0,
            0,
            b"\x002",
            &[],
        );
        let inst = read_decoded_instruction(&blob, 0).unwrap();
        assert_eq!(inst.condition, ConditionKind::Bit);
        assert_eq!(inst.condition_value, 42);
        assert_eq!(inst.condition_register, b"\x002");
    }

    #[test]
    fn walks_a_two_instruction_circuit() {
        let mut blob = synth_instruction_bytes("HGate", "", 0, &[0], &[], 0, 0, 0, 0, &[]);
        blob.extend_from_slice(&synth_instruction_bytes(
            "CXGate",
            "",
            0,
            &[0, 1],
            &[],
            0,
            0,
            1,
            1,
            &[],
        ));
        let (insts, total) = read_circuit_instructions(&blob, 2, 0).unwrap();
        assert_eq!(insts.len(), 2);
        assert_eq!(insts[0].name, "HGate");
        assert_eq!(insts[1].name, "CXGate");
        assert_eq!(total, blob.len());
    }

    #[test]
    fn empty_instruction_count_returns_empty() {
        let (insts, total) = read_circuit_instructions(b"", 0, 0).unwrap();
        assert!(insts.is_empty());
        assert_eq!(total, 0);
    }

    #[test]
    fn handles_offset_into_a_larger_blob() {
        let mut blob = vec![0xCCu8; 25];
        blob.extend_from_slice(&synth_instruction_bytes(
            "HGate",
            "",
            0,
            &[3],
            &[],
            0,
            0,
            0,
            0,
            &[],
        ));
        let inst = read_decoded_instruction(&blob, 25).unwrap();
        assert_eq!(inst.name, "HGate");
        assert_eq!(inst.args, vec![InstructionArg::Qubit(3)]);
    }
}
