//! Program type key + V16+ circuit-start offset table parsers.
//!
//! Layout (after the [`super::header`]-parsed file header):
//!
//! ```text
//! u8        program_type_key       // b'q' = CIRCUIT (only one we handle today)
//!                                  // b's' = SCHEDULE (legacy, rejected)
//!
//! V16+ only — circuit_start_table:
//!     [u64 ; num_circuits]         // big-endian absolute file offsets
//!                                  //   to each circuit's payload.
//! ```
//!
//! Pre-V16 blobs go straight from the program type key into the
//! first circuit's payload (no offset table).
//!
//! Reference: `qiskit/qpy/interface.py` `dump()` writes the file
//! header, then `common.write_type_key(file_obj, type_keys.Program.CIRCUIT)`,
//! then for V16+ the `CIRCUIT_TABLE_ENTRY_PACK` entries. The QPY HTML
//! docs leave out the type key — confirmed against a real Qiskit
//! 2.4.1-generated blob (see `tests/qpy_offset_table_real_fixture.rs`).

use super::header::QpyHeader;
use super::QpyError;

/// Program type key: which kind of payload the rest of the blob
/// describes. Today we only handle [`ProgramType::Circuit`]; the
/// legacy `Schedule` key is rejected with a typed error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramType {
    /// `b'q'` — `QuantumCircuit` payload(s) (the common case).
    Circuit,
}

impl ProgramType {
    fn from_byte(b: u8) -> Result<Self, QpyError> {
        match b {
            b'q' => Ok(ProgramType::Circuit),
            b's' => Err(QpyError::UnsupportedProgramType {
                byte: b,
                kind: "Schedule (legacy, removed in newer Qiskit)",
            }),
            other => Err(QpyError::UnknownProgramType { byte: other }),
        }
    }
}

/// Parsed program type key + (V16+) circuit start offset table.
///
/// `circuit_offsets` is empty for pre-V16 blobs; the per-circuit
/// reader walks the payload sequentially in that case. For V16+,
/// each entry is an absolute byte offset into the blob.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramTable {
    pub program_type: ProgramType,
    pub circuit_offsets: Vec<u64>,
    /// Total byte length of (program_type_key + circuit_table). The
    /// per-circuit data begins at `header.header_len() + table.table_len()`.
    table_len: usize,
}

impl ProgramTable {
    pub fn table_len(&self) -> usize {
        self.table_len
    }
}

/// Parse the program type key and (V16+) circuit start offset table
/// that follow the file-level header. `bytes` is the full QPY blob;
/// `header` is the result of [`super::read_header`] applied to it.
pub fn read_program_table(bytes: &[u8], header: &QpyHeader) -> Result<ProgramTable, QpyError> {
    let after_header = header.header_len();
    if bytes.len() < after_header + 1 {
        return Err(QpyError::Truncated {
            what: "program_type_key",
            need: after_header + 1,
            got: bytes.len(),
        });
    }
    let program_type = ProgramType::from_byte(bytes[after_header])?;

    let mut cursor = after_header + 1;
    let mut circuit_offsets = Vec::new();

    if header.qpy_version >= 16 {
        let needed = (header.num_circuits as usize)
            .checked_mul(8)
            .ok_or(QpyError::Truncated {
                what: "circuit_start_table",
                need: usize::MAX,
                got: bytes.len(),
            })?;
        let end = cursor + needed;
        if bytes.len() < end {
            return Err(QpyError::Truncated {
                what: "circuit_start_table",
                need: end,
                got: bytes.len(),
            });
        }
        circuit_offsets.reserve_exact(header.num_circuits as usize);
        for _ in 0..header.num_circuits {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[cursor..cursor + 8]);
            circuit_offsets.push(u64::from_be_bytes(buf));
            cursor += 8;
        }
    }

    Ok(ProgramTable {
        program_type,
        circuit_offsets,
        table_len: cursor - after_header,
    })
}

#[cfg(test)]
mod tests {
    use super::super::header::SymbolicEncoding;
    use super::*;

    fn header_v17_with_n(num_circuits: u64) -> QpyHeader {
        QpyHeader {
            qpy_version: 17,
            qiskit_version: (2, 4, 1),
            num_circuits,
            symbolic_encoding: Some(SymbolicEncoding::Sympy),
        }
    }

    fn header_v9_with_n(num_circuits: u64) -> QpyHeader {
        QpyHeader {
            qpy_version: 9,
            qiskit_version: (0, 45, 3),
            num_circuits,
            symbolic_encoding: None,
        }
    }

    /// Build a blob with `header.header_len()` zero bytes (placeholder
    /// file header) + a program key + an offset table + extra body.
    fn synth_blob(header: &QpyHeader, type_key: u8, offsets: &[u64], extra: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; header.header_len()];
        buf.push(type_key);
        for off in offsets {
            buf.extend_from_slice(&off.to_be_bytes());
        }
        buf.extend_from_slice(extra);
        buf
    }

    #[test]
    fn parses_v17_circuit_table_with_one_offset() {
        let h = header_v17_with_n(1);
        let blob = synth_blob(&h, b'q', &[28], &[0xAA]);
        let t = read_program_table(&blob, &h).unwrap();
        assert_eq!(t.program_type, ProgramType::Circuit);
        assert_eq!(t.circuit_offsets, vec![28]);
        assert_eq!(t.table_len(), 1 + 8); // type key + 1 × u64
    }

    #[test]
    fn parses_v17_multi_circuit_table() {
        let h = header_v17_with_n(3);
        let offsets = vec![100u64, 250, 999];
        let blob = synth_blob(&h, b'q', &offsets, &[]);
        let t = read_program_table(&blob, &h).unwrap();
        assert_eq!(t.circuit_offsets, offsets);
        assert_eq!(t.table_len(), 1 + 3 * 8);
    }

    #[test]
    fn pre_v16_skips_circuit_table_entirely() {
        // V9 has no circuit_start_table — only the program type key.
        let h = header_v9_with_n(1);
        let blob = synth_blob(&h, b'q', &[], &[0x00]);
        let t = read_program_table(&blob, &h).unwrap();
        assert_eq!(t.program_type, ProgramType::Circuit);
        assert!(t.circuit_offsets.is_empty());
        assert_eq!(t.table_len(), 1);
    }

    #[test]
    fn schedule_program_type_is_typed_error() {
        let h = header_v17_with_n(1);
        let blob = synth_blob(&h, b's', &[], &[]);
        match read_program_table(&blob, &h).unwrap_err() {
            QpyError::UnsupportedProgramType { byte, kind } => {
                assert_eq!(byte, b's');
                assert!(kind.contains("Schedule"));
            }
            other => panic!("expected UnsupportedProgramType, got {other:?}"),
        }
    }

    #[test]
    fn unknown_program_type_byte_is_typed_error() {
        let h = header_v17_with_n(1);
        let blob = synth_blob(&h, b'x', &[], &[]);
        match read_program_table(&blob, &h).unwrap_err() {
            QpyError::UnknownProgramType { byte } => assert_eq!(byte, b'x'),
            other => panic!("expected UnknownProgramType, got {other:?}"),
        }
    }

    #[test]
    fn truncation_after_header_reports_program_type_truncation() {
        let h = header_v17_with_n(1);
        let blob = vec![0u8; h.header_len()]; // just the file header, nothing else
        match read_program_table(&blob, &h).unwrap_err() {
            QpyError::Truncated { what, .. } => assert_eq!(what, "program_type_key"),
            other => panic!("expected Truncated(program_type_key), got {other:?}"),
        }
    }

    #[test]
    fn truncation_inside_circuit_table_reports_circuit_table_truncation() {
        let h = header_v17_with_n(2);
        // Type key + only 1 of 2 expected u64 offsets.
        let mut blob = vec![0u8; h.header_len()];
        blob.push(b'q');
        blob.extend_from_slice(&100u64.to_be_bytes());
        match read_program_table(&blob, &h).unwrap_err() {
            QpyError::Truncated { what, need, got } => {
                assert_eq!(what, "circuit_start_table");
                assert_eq!(need, h.header_len() + 1 + 16);
                assert_eq!(got, h.header_len() + 1 + 8);
            }
            other => panic!("expected Truncated(circuit_start_table), got {other:?}"),
        }
    }

    #[test]
    fn body_starts_immediately_after_table_len() {
        let h = header_v17_with_n(1);
        let body: Vec<u8> = (0..16).collect();
        let blob = synth_blob(&h, b'q', &[100], &body);
        let t = read_program_table(&blob, &h).unwrap();
        let body_start = h.header_len() + t.table_len();
        assert_eq!(&blob[body_start..], &body[..]);
    }
}
