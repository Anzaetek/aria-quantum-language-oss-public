//! Per-instruction `CIRCUIT_INSTRUCTION_V2` header parser.
//!
//! Layout (33 bytes fixed, big-endian throughout):
//!
//! ```text
//! struct CIRCUIT_INSTRUCTION_V2 {
//!     u16  name_size;
//!     u16  label_size;
//!     u16  num_parameters;
//!     u32  num_qargs;
//!     u32  num_cargs;
//!     u8   extras_key;             // V15+: low 2 bits = condition type,
//!                                  // high bit = annotations flag
//!     u16  condition_register_size;
//!     i64  condition_value;
//!     u32  num_ctrl_qubits;
//!     u32  ctrl_state;
//! }
//! u8[name_size]   name             // UTF-8
//! u8[label_size]  label            // UTF-8
//! ```
//!
//! Then `condition_register` (only if extras_key low bits == 2), the
//! qargs / cargs arrays, parameter list, and (extras_key high bit
//! set) annotations follow — those land in their own modules.

use super::QpyError;

/// Decoded `extras_key` byte: condition kind (low 2 bits) + V15+
/// annotations flag (high bit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtrasKey {
    /// Raw byte for round-trip.
    pub raw: u8,
    /// Condition variant (low 2 bits).
    pub condition: ConditionKind,
    /// `true` ⇒ INSTRUCTION_ANNOTATIONS_HEADER follows the
    /// parameter list (V15+ only).
    pub annotations: bool,
}

/// Condition encoding for the instruction (low two bits of
/// `extras_key`):
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConditionKind {
    /// `0` — no condition attached.
    None,
    /// `1` — classical-bit condition: gate runs only when the
    /// referenced clbit equals `condition_value`.
    Bit,
    /// `2` — register / classical-expression condition: gate runs
    /// only when the named register's value equals
    /// `condition_value`. The register name follows the variable
    /// section after `label`, plus an INSTRUCTION_PARAM expression
    /// payload.
    RegisterOrExpr,
}

impl ExtrasKey {
    fn parse(byte: u8) -> Result<Self, QpyError> {
        let condition = match byte & 0b11 {
            0 => ConditionKind::None,
            1 => ConditionKind::Bit,
            2 => ConditionKind::RegisterOrExpr,
            other => return Err(QpyError::UnknownConditionKind { byte: other }),
        };
        let annotations = (byte & 0b1000_0000) != 0;
        // Bits 2..6 are reserved for future use; surface them via
        // `raw` so a forward-incompatible change is at least
        // recoverable for the caller that wants the byte verbatim.
        Ok(Self {
            raw: byte,
            condition,
            annotations,
        })
    }
}

/// Parsed instruction header (fixed struct + variable name + label).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstructionHeader<'a> {
    pub name: &'a str,
    pub label: &'a str,
    pub num_parameters: u16,
    pub num_qargs: u32,
    pub num_cargs: u32,
    pub extras_key: ExtrasKey,
    pub condition_register_size: u16,
    pub condition_value: i64,
    pub num_ctrl_qubits: u32,
    pub ctrl_state: u32,
    /// Total bytes consumed by header struct + name + label. Caller
    /// adds this to the start offset to find where the next
    /// variable section (condition_register / qargs / cargs / params)
    /// begins.
    pub header_len: usize,
}

const FIXED_INSTRUCTION_HEADER_LEN: usize = 2 + 2 + 2 + 4 + 4 + 1 + 2 + 8 + 4 + 4; // 33

/// Parse one INSTRUCTION_V2 fixed header + name + label. Caller
/// passes `offset` = start byte of this instruction in the blob.
pub fn read_instruction_header(
    bytes: &[u8],
    offset: usize,
) -> Result<InstructionHeader<'_>, QpyError> {
    let fixed_end =
        offset
            .checked_add(FIXED_INSTRUCTION_HEADER_LEN)
            .ok_or(QpyError::Truncated {
                what: "instruction_header",
                need: usize::MAX,
                got: bytes.len(),
            })?;
    if bytes.len() < fixed_end {
        return Err(QpyError::Truncated {
            what: "instruction_header",
            need: fixed_end,
            got: bytes.len(),
        });
    }
    let s = &bytes[offset..fixed_end];

    let name_size = u16::from_be_bytes([s[0], s[1]]) as usize;
    let label_size = u16::from_be_bytes([s[2], s[3]]) as usize;
    let num_parameters = u16::from_be_bytes([s[4], s[5]]);
    let num_qargs = u32::from_be_bytes([s[6], s[7], s[8], s[9]]);
    let num_cargs = u32::from_be_bytes([s[10], s[11], s[12], s[13]]);
    let extras_key = ExtrasKey::parse(s[14])?;
    let condition_register_size = u16::from_be_bytes([s[15], s[16]]);
    let mut buf8 = [0u8; 8];
    buf8.copy_from_slice(&s[17..25]);
    let condition_value = i64::from_be_bytes(buf8);
    let num_ctrl_qubits = u32::from_be_bytes([s[25], s[26], s[27], s[28]]);
    let ctrl_state = u32::from_be_bytes([s[29], s[30], s[31], s[32]]);

    // Variable: name + label.
    let name_end = fixed_end + name_size;
    let label_end = name_end + label_size;
    if bytes.len() < label_end {
        return Err(QpyError::Truncated {
            what: "instruction_name_label",
            need: label_end,
            got: bytes.len(),
        });
    }
    let name =
        std::str::from_utf8(&bytes[fixed_end..name_end]).map_err(|e| QpyError::InvalidUtf8 {
            what: "instruction_name",
            valid_up_to: e.valid_up_to(),
            len: name_size,
        })?;
    let label =
        std::str::from_utf8(&bytes[name_end..label_end]).map_err(|e| QpyError::InvalidUtf8 {
            what: "instruction_label",
            valid_up_to: e.valid_up_to(),
            len: label_size,
        })?;

    Ok(InstructionHeader {
        name,
        label,
        num_parameters,
        num_qargs,
        num_cargs,
        extras_key,
        condition_register_size,
        condition_value,
        num_ctrl_qubits,
        ctrl_state,
        header_len: label_end - offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn make_instruction_bytes(
        name: &str,
        label: &str,
        num_parameters: u16,
        num_qargs: u32,
        num_cargs: u32,
        extras_key: u8,
        condition_register_size: u16,
        condition_value: i64,
        num_ctrl_qubits: u32,
        ctrl_state: u32,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(name.len() as u16).to_be_bytes());
        out.extend_from_slice(&(label.len() as u16).to_be_bytes());
        out.extend_from_slice(&num_parameters.to_be_bytes());
        out.extend_from_slice(&num_qargs.to_be_bytes());
        out.extend_from_slice(&num_cargs.to_be_bytes());
        out.push(extras_key);
        out.extend_from_slice(&condition_register_size.to_be_bytes());
        out.extend_from_slice(&condition_value.to_be_bytes());
        out.extend_from_slice(&num_ctrl_qubits.to_be_bytes());
        out.extend_from_slice(&ctrl_state.to_be_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(label.as_bytes());
        out
    }

    #[test]
    fn round_trips_a_typical_h_gate_header() {
        let blob = make_instruction_bytes("HGate", "", 0, 1, 0, 0, 0, 0, 0, 0);
        let h = read_instruction_header(&blob, 0).unwrap();
        assert_eq!(h.name, "HGate");
        assert_eq!(h.label, "");
        assert_eq!(h.num_parameters, 0);
        assert_eq!(h.num_qargs, 1);
        assert_eq!(h.num_cargs, 0);
        assert_eq!(h.extras_key.condition, ConditionKind::None);
        assert!(!h.extras_key.annotations);
        assert_eq!(h.num_ctrl_qubits, 0);
        assert_eq!(h.ctrl_state, 0);
        assert_eq!(h.header_len, 33 + 5);
    }

    #[test]
    fn round_trips_a_cx_gate_header_with_label() {
        let blob = make_instruction_bytes("CXGate", "my-cnot", 0, 2, 0, 0, 0, 0, 1, 1);
        let h = read_instruction_header(&blob, 0).unwrap();
        assert_eq!(h.name, "CXGate");
        assert_eq!(h.label, "my-cnot");
        assert_eq!(h.num_qargs, 2);
        assert_eq!(h.num_ctrl_qubits, 1);
        assert_eq!(h.ctrl_state, 1);
    }

    #[test]
    fn parses_bit_condition_low_bits_equal_one() {
        // extras_key = 0b00000001 = condition kind Bit, no annotations.
        let blob = make_instruction_bytes("XGate", "", 0, 1, 0, 0b0000_0001, 0, 42, 0, 0);
        let h = read_instruction_header(&blob, 0).unwrap();
        assert_eq!(h.extras_key.condition, ConditionKind::Bit);
        assert!(!h.extras_key.annotations);
        assert_eq!(h.condition_value, 42);
    }

    #[test]
    fn parses_register_condition_low_bits_equal_two() {
        let blob = make_instruction_bytes("XGate", "", 0, 1, 0, 0b0000_0010, 1, 7, 0, 0);
        let h = read_instruction_header(&blob, 0).unwrap();
        assert_eq!(h.extras_key.condition, ConditionKind::RegisterOrExpr);
        assert_eq!(h.condition_register_size, 1);
        assert_eq!(h.condition_value, 7);
    }

    #[test]
    fn detects_annotations_flag_high_bit() {
        let blob = make_instruction_bytes("Anno", "", 0, 1, 0, 0b1000_0000, 0, 0, 0, 0);
        let h = read_instruction_header(&blob, 0).unwrap();
        assert!(h.extras_key.annotations);
        assert_eq!(h.extras_key.condition, ConditionKind::None);
    }

    #[test]
    fn rejects_reserved_condition_kind_three() {
        // low bits = 0b11 = 3 — reserved by the QPY spec.
        let blob = make_instruction_bytes("X", "", 0, 1, 0, 0b0000_0011, 0, 0, 0, 0);
        match read_instruction_header(&blob, 0).unwrap_err() {
            QpyError::UnknownConditionKind { byte } => assert_eq!(byte, 3),
            other => panic!("expected UnknownConditionKind, got {other:?}"),
        }
    }

    #[test]
    fn handles_negative_condition_value() {
        let blob = make_instruction_bytes("X", "", 0, 1, 0, 0b0000_0001, 0, -42, 0, 0);
        let h = read_instruction_header(&blob, 0).unwrap();
        assert_eq!(h.condition_value, -42);
    }

    #[test]
    fn truncation_inside_fixed_header_is_typed_error() {
        let blob = vec![0u8; 32]; // one short of 33
        match read_instruction_header(&blob, 0).unwrap_err() {
            QpyError::Truncated { what, need, got } => {
                assert_eq!(what, "instruction_header");
                assert_eq!(need, 33);
                assert_eq!(got, 32);
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn truncation_in_name_or_label_is_typed_error() {
        let mut blob = vec![0u8; 33];
        // Pretend name_size=5 but only 3 trailing bytes follow.
        blob[0] = 0;
        blob[1] = 5;
        blob.extend_from_slice(b"abc");
        match read_instruction_header(&blob, 0).unwrap_err() {
            QpyError::Truncated { what, .. } => assert_eq!(what, "instruction_name_label"),
            other => panic!("expected Truncated(instruction_name_label), got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_utf8_instruction_name() {
        let mut blob = vec![0u8; 33];
        blob[0] = 0;
        blob[1] = 2;
        blob.extend_from_slice(&[0xFF, 0xFE]);
        match read_instruction_header(&blob, 0).unwrap_err() {
            QpyError::InvalidUtf8 { what, .. } => assert_eq!(what, "instruction_name"),
            other => panic!("expected InvalidUtf8(instruction_name), got {other:?}"),
        }
    }

    #[test]
    fn handles_offset_into_a_larger_blob() {
        let mut blob = vec![0xCCu8; 100];
        blob.extend_from_slice(&make_instruction_bytes(
            "Test", "label", 2, 3, 1, 0, 0, 0, 0, 0,
        ));
        let h = read_instruction_header(&blob, 100).unwrap();
        assert_eq!(h.name, "Test");
        assert_eq!(h.label, "label");
        assert_eq!(h.num_parameters, 2);
        assert_eq!(h.num_qargs, 3);
        assert_eq!(h.num_cargs, 1);
    }
}
