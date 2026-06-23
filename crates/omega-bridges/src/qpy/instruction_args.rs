//! `CIRCUIT_INSTRUCTION_ARG` parser. Each argument is 5 bytes
//! (big-endian):
//!
//! ```text
//! struct CIRCUIT_INSTRUCTION_ARG {
//!     u8   type;   // b'q' = quantum bit, b'c' = classical bit
//!     u32  size;   // index into the circuit's qubit / clbit table
//! }
//! ```
//!
//! Each per-instruction args section is `num_qargs` quantum entries
//! followed by `num_cargs` classical entries, written back-to-back
//! with no padding.

use super::QpyError;

/// One decoded INSTRUCTION_ARG entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstructionArg {
    /// `b'q'` — circuit-level qubit index.
    Qubit(u32),
    /// `b'c'` — circuit-level clbit index.
    Clbit(u32),
}

/// Total bytes per arg entry.
pub const ARG_LEN: usize = 1 + 4;

/// Read a contiguous run of `num_qargs + num_cargs` argument
/// entries. Asserts the qarg entries come first (type=`'q'`) and
/// the carg entries follow (type=`'c'`); any out-of-order entry
/// surfaces as `QpyError::InstructionArgKindMismatch` with the
/// position so a producer-bug doesn't silently shift bit indices.
pub fn read_instruction_args(
    bytes: &[u8],
    num_qargs: u32,
    num_cargs: u32,
    offset: usize,
) -> Result<(Vec<InstructionArg>, usize), QpyError> {
    let total = (num_qargs as usize)
        .checked_add(num_cargs as usize)
        .and_then(|n| n.checked_mul(ARG_LEN))
        .ok_or(QpyError::Truncated {
            what: "instruction_args",
            need: usize::MAX,
            got: bytes.len(),
        })?;
    let end = offset.checked_add(total).ok_or(QpyError::Truncated {
        what: "instruction_args",
        need: usize::MAX,
        got: bytes.len(),
    })?;
    if bytes.len() < end {
        return Err(QpyError::Truncated {
            what: "instruction_args",
            need: end,
            got: bytes.len(),
        });
    }
    let mut args = Vec::with_capacity(num_qargs as usize + num_cargs as usize);
    let mut cursor = offset;
    for i in 0..num_qargs as usize {
        let entry = parse_arg(&bytes[cursor..cursor + ARG_LEN])?;
        if !matches!(entry, InstructionArg::Qubit(_)) {
            return Err(QpyError::InstructionArgKindMismatch {
                expected_kind: 'q',
                position: i,
            });
        }
        args.push(entry);
        cursor += ARG_LEN;
    }
    for i in 0..num_cargs as usize {
        let entry = parse_arg(&bytes[cursor..cursor + ARG_LEN])?;
        if !matches!(entry, InstructionArg::Clbit(_)) {
            return Err(QpyError::InstructionArgKindMismatch {
                expected_kind: 'c',
                position: num_qargs as usize + i,
            });
        }
        args.push(entry);
        cursor += ARG_LEN;
    }
    Ok((args, total))
}

fn parse_arg(bytes: &[u8]) -> Result<InstructionArg, QpyError> {
    debug_assert_eq!(bytes.len(), ARG_LEN);
    let kind = bytes[0];
    let size = u32::from_be_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
    match kind {
        b'q' => Ok(InstructionArg::Qubit(size)),
        b'c' => Ok(InstructionArg::Clbit(size)),
        other => Err(QpyError::UnknownInstructionArgKind { byte: other }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arg_bytes(kind: u8, size: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(ARG_LEN);
        out.push(kind);
        out.extend_from_slice(&size.to_be_bytes());
        out
    }

    #[test]
    fn parses_a_single_qubit_arg() {
        let blob = arg_bytes(b'q', 0);
        let (args, len) = read_instruction_args(&blob, 1, 0, 0).unwrap();
        assert_eq!(args, vec![InstructionArg::Qubit(0)]);
        assert_eq!(len, ARG_LEN);
    }

    #[test]
    fn parses_two_qubit_args_for_a_cx_gate() {
        let mut blob = arg_bytes(b'q', 0);
        blob.extend_from_slice(&arg_bytes(b'q', 1));
        let (args, len) = read_instruction_args(&blob, 2, 0, 0).unwrap();
        assert_eq!(
            args,
            vec![InstructionArg::Qubit(0), InstructionArg::Qubit(1)]
        );
        assert_eq!(len, 2 * ARG_LEN);
    }

    #[test]
    fn parses_qargs_followed_by_cargs_for_a_measure() {
        let mut blob = arg_bytes(b'q', 0);
        blob.extend_from_slice(&arg_bytes(b'c', 0));
        let (args, len) = read_instruction_args(&blob, 1, 1, 0).unwrap();
        assert_eq!(
            args,
            vec![InstructionArg::Qubit(0), InstructionArg::Clbit(0)]
        );
        assert_eq!(len, 2 * ARG_LEN);
    }

    #[test]
    fn empty_args_returns_empty_vec() {
        let (args, len) = read_instruction_args(b"", 0, 0, 0).unwrap();
        assert!(args.is_empty());
        assert_eq!(len, 0);
    }

    #[test]
    fn handles_offset_into_a_larger_blob() {
        let mut blob = vec![0xCCu8; 50];
        blob.extend_from_slice(&arg_bytes(b'q', 7));
        let (args, _) = read_instruction_args(&blob, 1, 0, 50).unwrap();
        assert_eq!(args, vec![InstructionArg::Qubit(7)]);
    }

    #[test]
    fn unknown_arg_kind_byte_is_typed_error() {
        let blob = arg_bytes(b'x', 0);
        match read_instruction_args(&blob, 1, 0, 0).unwrap_err() {
            QpyError::UnknownInstructionArgKind { byte } => assert_eq!(byte, b'x'),
            other => panic!("expected UnknownInstructionArgKind, got {other:?}"),
        }
    }

    #[test]
    fn classical_byte_in_qarg_position_is_kind_mismatch() {
        // Producer wrote a classical entry where a qarg was
        // expected — surface it explicitly so the caller doesn't
        // map the index to the wrong bit set.
        let blob = arg_bytes(b'c', 0);
        match read_instruction_args(&blob, 1, 0, 0).unwrap_err() {
            QpyError::InstructionArgKindMismatch {
                expected_kind,
                position,
            } => {
                assert_eq!(expected_kind, 'q');
                assert_eq!(position, 0);
            }
            other => panic!("expected InstructionArgKindMismatch, got {other:?}"),
        }
    }

    #[test]
    fn quantum_byte_in_carg_position_is_kind_mismatch() {
        let mut blob = arg_bytes(b'q', 0);
        // Then a 'q' where a 'c' is expected.
        blob.extend_from_slice(&arg_bytes(b'q', 0));
        match read_instruction_args(&blob, 1, 1, 0).unwrap_err() {
            QpyError::InstructionArgKindMismatch {
                expected_kind,
                position,
            } => {
                assert_eq!(expected_kind, 'c');
                assert_eq!(position, 1);
            }
            other => panic!("expected InstructionArgKindMismatch, got {other:?}"),
        }
    }

    #[test]
    fn truncation_is_typed_error() {
        // num_qargs=2 needs 10 bytes; only 7 supplied.
        let mut blob = arg_bytes(b'q', 0);
        blob.extend_from_slice(&[0u8, 0]);
        match read_instruction_args(&blob, 2, 0, 0).unwrap_err() {
            QpyError::Truncated { what, need, got } => {
                assert_eq!(what, "instruction_args");
                assert_eq!(need, 2 * ARG_LEN);
                assert_eq!(got, 7);
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
    }
}
