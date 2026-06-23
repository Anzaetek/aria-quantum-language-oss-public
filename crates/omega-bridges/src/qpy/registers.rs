//! Per-circuit REGISTER table parser (V4+).
//!
//! Layout for each register (`num_registers` of them, back-to-back,
//! all multi-byte fields big-endian):
//!
//! ```text
//! struct REGISTER {
//!     u8   type;            // b'q' = quantum, b'c' = classical
//!     u8   standalone;      // bool: register owns its bits (1) vs.
//!                           //   borrows them from the circuit (0)
//!     u32  size;            // number of bits in this register
//!     u16  name_size;       // UTF-8 name length
//!     u8   in_circuit;      // bool: is this register part of the
//!                           //   parent circuit (1) vs. orphaned (0)
//! }                         // 9 bytes total
//! u8[name_size]   name      // UTF-8
//! i64[size]       bit_index_array  // negative ⇒ bit not in circuit
//! ```
//!
//! Pre-V4 used `u32` for the bit-index array. We accept V12+ today
//! (per [`super::MAX_SUPPORTED_VERSION`]); the i64 layout applies
//! throughout that range.

use super::QpyError;

/// `b'q'` (quantum) or `b'c'` (classical).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisterKind {
    Quantum,
    Classical,
}

impl RegisterKind {
    fn from_byte(b: u8) -> Result<Self, QpyError> {
        match b {
            b'q' => Ok(RegisterKind::Quantum),
            b'c' => Ok(RegisterKind::Classical),
            other => Err(QpyError::UnknownRegisterKind { byte: other }),
        }
    }
}

/// One parsed register entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Register<'a> {
    pub kind: RegisterKind,
    /// `true` ⇒ the register owns its underlying bits (the typical
    /// `qreg q[2];` case); `false` ⇒ the bits are borrowed from
    /// elsewhere in the circuit.
    pub standalone: bool,
    /// Number of bits in the register.
    pub size: u32,
    /// `false` for registers serialised on a circuit but not part of
    /// it (the `in_circuit` bit). The qiskit producer only emits
    /// these in advanced layout-tracking scenarios.
    pub in_circuit: bool,
    /// UTF-8 name (borrowed from the input slice).
    pub name: &'a str,
    /// Maps register bit position → circuit qubit/clbit index.
    /// Negative entries indicate bits that aren't part of the
    /// parent circuit (e.g. ancillas dropped by a transpiler pass
    /// but kept in the register definition).
    pub bit_index: Vec<i64>,
}

/// Parsed register table for one circuit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisterTable<'a> {
    pub registers: Vec<Register<'a>>,
    /// Total bytes consumed by every register entry combined. Add
    /// to the start-of-table offset to find the first INSTRUCTION.
    pub table_len: usize,
}

const FIXED_REGISTER_HEADER_LEN: usize = 1 + 1 + 4 + 2 + 1; // 9

/// Read `num_registers` consecutive REGISTER entries starting at
/// `offset`. Caller is expected to derive `offset` as
/// `circuit_offsets[i] + circuit_header::FIXED_LEN + body_len`.
pub fn read_register_table<'a>(
    bytes: &'a [u8],
    num_registers: u32,
    offset: usize,
) -> Result<RegisterTable<'a>, QpyError> {
    let mut cursor = offset;
    let mut registers = Vec::with_capacity(num_registers as usize);

    for _ in 0..num_registers {
        // Fixed 9-byte header.
        let header_end =
            cursor
                .checked_add(FIXED_REGISTER_HEADER_LEN)
                .ok_or(QpyError::Truncated {
                    what: "register_header",
                    need: usize::MAX,
                    got: bytes.len(),
                })?;
        if bytes.len() < header_end {
            return Err(QpyError::Truncated {
                what: "register_header",
                need: header_end,
                got: bytes.len(),
            });
        }
        let h = &bytes[cursor..header_end];
        let kind = RegisterKind::from_byte(h[0])?;
        let standalone = byte_to_bool(h[1], "register.standalone")?;
        let size = u32::from_be_bytes([h[2], h[3], h[4], h[5]]);
        let name_size = u16::from_be_bytes([h[6], h[7]]) as usize;
        let in_circuit = byte_to_bool(h[8], "register.in_circuit")?;

        // Variable: name.
        let name_end = header_end + name_size;
        if bytes.len() < name_end {
            return Err(QpyError::Truncated {
                what: "register_name",
                need: name_end,
                got: bytes.len(),
            });
        }
        let name = std::str::from_utf8(&bytes[header_end..name_end]).map_err(|e| {
            QpyError::InvalidUtf8 {
                what: "register_name",
                valid_up_to: e.valid_up_to(),
                len: name_size,
            }
        })?;

        // Variable: bit_index array of i64 (size × 8 bytes).
        let bit_array_bytes = (size as usize).checked_mul(8).ok_or(QpyError::Truncated {
            what: "register_bit_index",
            need: usize::MAX,
            got: bytes.len(),
        })?;
        let bit_end = name_end + bit_array_bytes;
        if bytes.len() < bit_end {
            return Err(QpyError::Truncated {
                what: "register_bit_index",
                need: bit_end,
                got: bytes.len(),
            });
        }
        let mut bit_index = Vec::with_capacity(size as usize);
        for i in 0..size as usize {
            let start = name_end + i * 8;
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[start..start + 8]);
            bit_index.push(i64::from_be_bytes(buf));
        }

        registers.push(Register {
            kind,
            standalone,
            size,
            in_circuit,
            name,
            bit_index,
        });

        cursor = bit_end;
    }

    Ok(RegisterTable {
        registers,
        table_len: cursor - offset,
    })
}

fn byte_to_bool(b: u8, what: &'static str) -> Result<bool, QpyError> {
    match b {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(QpyError::InvalidBoolByte { what, byte: other }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the bytes for one register entry.
    fn register_bytes(
        kind: u8,
        standalone: bool,
        size: u32,
        name: &str,
        in_circuit: bool,
        bit_index: &[i64],
    ) -> Vec<u8> {
        assert_eq!(bit_index.len(), size as usize);
        let mut out = Vec::new();
        out.push(kind);
        out.push(if standalone { 1 } else { 0 });
        out.extend_from_slice(&size.to_be_bytes());
        out.extend_from_slice(&(name.len() as u16).to_be_bytes());
        out.push(if in_circuit { 1 } else { 0 });
        out.extend_from_slice(name.as_bytes());
        for &b in bit_index {
            out.extend_from_slice(&b.to_be_bytes());
        }
        out
    }

    #[test]
    fn parses_a_two_bit_quantum_register_named_q() {
        let blob = register_bytes(b'q', true, 2, "q", true, &[0, 1]);
        let t = read_register_table(&blob, 1, 0).unwrap();
        assert_eq!(t.registers.len(), 1);
        let r = &t.registers[0];
        assert_eq!(r.kind, RegisterKind::Quantum);
        assert!(r.standalone);
        assert_eq!(r.size, 2);
        assert!(r.in_circuit);
        assert_eq!(r.name, "q");
        assert_eq!(r.bit_index, vec![0, 1]);
        // 9 (header) + 1 (name) + 16 (2 × i64) = 26
        assert_eq!(t.table_len, 26);
    }

    #[test]
    fn parses_qreg_then_creg_back_to_back() {
        let mut blob = register_bytes(b'q', true, 2, "q", true, &[0, 1]);
        blob.extend_from_slice(&register_bytes(b'c', true, 2, "c", true, &[0, 1]));
        let t = read_register_table(&blob, 2, 0).unwrap();
        assert_eq!(t.registers.len(), 2);
        assert_eq!(t.registers[0].kind, RegisterKind::Quantum);
        assert_eq!(t.registers[1].kind, RegisterKind::Classical);
        // Two registers, same shape: 26 + 26 = 52
        assert_eq!(t.table_len, 52);
    }

    #[test]
    fn preserves_negative_bit_indices() {
        let blob = register_bytes(b'q', false, 3, "ancilla", false, &[0, -1, 2]);
        let r = &read_register_table(&blob, 1, 0).unwrap().registers[0];
        assert_eq!(r.bit_index, vec![0, -1, 2]);
        assert!(!r.standalone);
        assert!(!r.in_circuit);
    }

    #[test]
    fn handles_offset_into_a_larger_blob() {
        let mut blob = vec![0xCCu8; 50];
        blob.extend_from_slice(&register_bytes(b'q', true, 1, "q", true, &[0]));
        let t = read_register_table(&blob, 1, 50).unwrap();
        assert_eq!(t.registers[0].name, "q");
        assert_eq!(t.registers[0].bit_index, vec![0]);
        assert_eq!(t.table_len, 9 + 1 + 8);
    }

    #[test]
    fn empty_register_count_returns_empty_table() {
        let t = read_register_table(b"", 0, 0).unwrap();
        assert!(t.registers.is_empty());
        assert_eq!(t.table_len, 0);
    }

    #[test]
    fn unknown_register_kind_byte_is_typed_error() {
        let blob = register_bytes(b'X', true, 0, "x", true, &[]);
        match read_register_table(&blob, 1, 0).unwrap_err() {
            QpyError::UnknownRegisterKind { byte } => assert_eq!(byte, b'X'),
            other => panic!("expected UnknownRegisterKind, got {other:?}"),
        }
    }

    #[test]
    fn out_of_range_bool_byte_is_typed_error() {
        // standalone byte = 2 (neither 0 nor 1).
        let mut blob = vec![b'q', 2u8];
        blob.extend_from_slice(&0u32.to_be_bytes());
        blob.extend_from_slice(&0u16.to_be_bytes());
        blob.push(0);
        match read_register_table(&blob, 1, 0).unwrap_err() {
            QpyError::InvalidBoolByte { what, byte } => {
                assert_eq!(what, "register.standalone");
                assert_eq!(byte, 2);
            }
            other => panic!("expected InvalidBoolByte, got {other:?}"),
        }
    }

    #[test]
    fn truncation_in_register_header_is_typed_error() {
        // Only 5 bytes — short of the 9-byte fixed header.
        let blob = vec![b'q', 1, 0, 0, 0];
        match read_register_table(&blob, 1, 0).unwrap_err() {
            QpyError::Truncated { what, need, got } => {
                assert_eq!(what, "register_header");
                assert_eq!(need, 9);
                assert_eq!(got, 5);
            }
            other => panic!("expected Truncated(register_header), got {other:?}"),
        }
    }

    #[test]
    fn truncation_in_name_section_is_typed_error() {
        // 9-byte header says name_size=10 but no name bytes follow.
        let mut blob = vec![b'q', 1];
        blob.extend_from_slice(&0u32.to_be_bytes());
        blob.extend_from_slice(&10u16.to_be_bytes());
        blob.push(1);
        match read_register_table(&blob, 1, 0).unwrap_err() {
            QpyError::Truncated { what, .. } => assert_eq!(what, "register_name"),
            other => panic!("expected Truncated(register_name), got {other:?}"),
        }
    }

    #[test]
    fn truncation_in_bit_index_array_is_typed_error() {
        // size=3 → 24 bytes of i64 expected, but only 8 supplied.
        let mut blob = vec![b'q', 1];
        blob.extend_from_slice(&3u32.to_be_bytes());
        blob.extend_from_slice(&0u16.to_be_bytes()); // empty name
        blob.push(1);
        blob.extend_from_slice(&0i64.to_be_bytes());
        match read_register_table(&blob, 1, 0).unwrap_err() {
            QpyError::Truncated { what, .. } => assert_eq!(what, "register_bit_index"),
            other => panic!("expected Truncated(register_bit_index), got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_utf8_register_name() {
        let mut blob = vec![b'q', 1];
        blob.extend_from_slice(&0u32.to_be_bytes());
        blob.extend_from_slice(&2u16.to_be_bytes());
        blob.push(1);
        blob.extend_from_slice(&[0xFF, 0xFE]);
        match read_register_table(&blob, 1, 0).unwrap_err() {
            QpyError::InvalidUtf8 { what, .. } => assert_eq!(what, "register_name"),
            other => panic!("expected InvalidUtf8(register_name), got {other:?}"),
        }
    }
}
