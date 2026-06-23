//! Per-circuit `CIRCUIT_HEADER` parser (fixed-size portion).
//!
//! Layout for QPY V12+ (37 bytes total, big-endian throughout):
//!
//! ```text
//! struct CIRCUIT_HEADER {
//!     uint16_t name_size;
//!     char     global_phase_type;        // 1 byte: 'i', 'f', 'p', 'e', …
//!     uint16_t global_phase_size;
//!     uint32_t num_qubits;
//!     uint32_t num_clbits;
//!     uint64_t metadata_size;
//!     uint32_t num_registers;
//!     uint64_t num_instructions;
//!     uint32_t num_vars;                 // V12+ only
//! }
//! ```
//!
//! Variable-length data follows immediately:
//!   1. `name`: `name_size` bytes (UTF-8).
//!   2. `global_phase`: `global_phase_size` bytes; layout depends on
//!      `global_phase_type` (e.g. `'f'` is an 8-byte f64, little-endian
//!      for QPY ≤ 17).
//!   3. `metadata`: `metadata_size` bytes (JSON-encoded UTF-8).
//!   4. registers, instructions, vars …
//!
//! Today this module parses only the fixed-size 37-byte struct; the
//! variable-length name / global_phase / metadata land in a follow-up
//! commit.

use super::QpyError;

/// Fixed-size portion of the per-circuit header. All counts are
/// what the producer wrote; no normalisation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitHeader {
    pub name_size: u16,
    /// One of `b'i'`, `b'f'`, `b'p'`, `b'e'`, … — see the GLOBAL_PHASE
    /// section of the QPY spec for the full type-byte table.
    pub global_phase_type: u8,
    pub global_phase_size: u16,
    pub num_qubits: u32,
    pub num_clbits: u32,
    pub metadata_size: u64,
    pub num_registers: u32,
    pub num_instructions: u64,
    pub num_vars: u32,
}

/// Total byte size of the fixed `CIRCUIT_HEADER` struct.
pub const FIXED_LEN: usize = 2 + 1 + 2 + 4 + 4 + 8 + 4 + 8 + 4; // 37

/// Parse the fixed-size CIRCUIT_HEADER struct starting at `offset`.
/// Caller is expected to pass the absolute offset from the
/// `ProgramTable::circuit_offsets[i]` entry (or compute it manually
/// for pre-V16 blobs).
pub fn read_circuit_header(bytes: &[u8], offset: usize) -> Result<CircuitHeader, QpyError> {
    let end = offset.checked_add(FIXED_LEN).ok_or(QpyError::Truncated {
        what: "circuit_header",
        need: usize::MAX,
        got: bytes.len(),
    })?;
    if bytes.len() < end {
        return Err(QpyError::Truncated {
            what: "circuit_header",
            need: end,
            got: bytes.len(),
        });
    }
    let s = &bytes[offset..end];

    let name_size = u16::from_be_bytes([s[0], s[1]]);
    let global_phase_type = s[2];
    let global_phase_size = u16::from_be_bytes([s[3], s[4]]);

    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    buf4.copy_from_slice(&s[5..9]);
    let num_qubits = u32::from_be_bytes(buf4);
    buf4.copy_from_slice(&s[9..13]);
    let num_clbits = u32::from_be_bytes(buf4);
    buf8.copy_from_slice(&s[13..21]);
    let metadata_size = u64::from_be_bytes(buf8);
    buf4.copy_from_slice(&s[21..25]);
    let num_registers = u32::from_be_bytes(buf4);
    buf8.copy_from_slice(&s[25..33]);
    let num_instructions = u64::from_be_bytes(buf8);
    buf4.copy_from_slice(&s[33..37]);
    let num_vars = u32::from_be_bytes(buf4);

    Ok(CircuitHeader {
        name_size,
        global_phase_type,
        global_phase_size,
        num_qubits,
        num_clbits,
        metadata_size,
        num_registers,
        num_instructions,
        num_vars,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_header_bytes(h: &CircuitHeader) -> Vec<u8> {
        let mut out = Vec::with_capacity(FIXED_LEN);
        out.extend_from_slice(&h.name_size.to_be_bytes());
        out.push(h.global_phase_type);
        out.extend_from_slice(&h.global_phase_size.to_be_bytes());
        out.extend_from_slice(&h.num_qubits.to_be_bytes());
        out.extend_from_slice(&h.num_clbits.to_be_bytes());
        out.extend_from_slice(&h.metadata_size.to_be_bytes());
        out.extend_from_slice(&h.num_registers.to_be_bytes());
        out.extend_from_slice(&h.num_instructions.to_be_bytes());
        out.extend_from_slice(&h.num_vars.to_be_bytes());
        debug_assert_eq!(out.len(), FIXED_LEN);
        out
    }

    #[test]
    fn round_trips_a_typical_bell_state_header() {
        let want = CircuitHeader {
            name_size: 10,
            global_phase_type: b'f',
            global_phase_size: 8,
            num_qubits: 2,
            num_clbits: 2,
            metadata_size: 2,
            num_registers: 2,
            num_instructions: 4,
            num_vars: 0,
        };
        let bytes = make_header_bytes(&want);
        assert_eq!(read_circuit_header(&bytes, 0).unwrap(), want);
    }

    #[test]
    fn handles_offset_into_a_larger_blob() {
        let want = CircuitHeader {
            name_size: 1234,
            global_phase_type: b'p',
            global_phase_size: 64,
            num_qubits: 100,
            num_clbits: 50,
            metadata_size: 0xDEAD_BEEF,
            num_registers: 7,
            num_instructions: 0x0102_0304_0506_0708,
            num_vars: 9,
        };
        // Pad with 100 leading zero bytes so the parser actually has
        // to honour the offset.
        let mut blob = vec![0u8; 100];
        blob.extend_from_slice(&make_header_bytes(&want));
        assert_eq!(read_circuit_header(&blob, 100).unwrap(), want);
    }

    #[test]
    fn truncation_inside_the_struct_reports_circuit_header_kind() {
        // 36 bytes — one short of the 37-byte struct.
        let blob = vec![0u8; 36];
        match read_circuit_header(&blob, 0).unwrap_err() {
            QpyError::Truncated { what, need, got } => {
                assert_eq!(what, "circuit_header");
                assert_eq!(need, 37);
                assert_eq!(got, 36);
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn truncation_at_offset_past_blob_end_reports_truncation() {
        let blob = vec![0u8; 50];
        match read_circuit_header(&blob, 100).unwrap_err() {
            QpyError::Truncated { what, .. } => assert_eq!(what, "circuit_header"),
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn empty_blob_reports_truncation() {
        match read_circuit_header(b"", 0).unwrap_err() {
            QpyError::Truncated { what, need, got } => {
                assert_eq!(what, "circuit_header");
                assert_eq!(need, 37);
                assert_eq!(got, 0);
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn fixed_len_constant_matches_struct_size() {
        // Mirror of the byte-count comment on the constant — keeps
        // a future field addition from drifting the constant
        // silently.
        assert_eq!(FIXED_LEN, 37);
    }
}
