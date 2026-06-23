//! Per-circuit "extras" sections that sit between the register
//! table and the instruction list:
//!
//!   ANNOTATION_HEADER_STATIC  (V15+)  — 4-byte `num_namespaces`
//!     followed by per-namespace payloads.
//!   CUSTOM_CIRCUIT_DEF_HEADER          — 8-byte `size` (number of
//!     custom instruction definitions that follow), then `size` ×
//!     CUSTOM_CIRCUIT_INST_DEF_V2 payloads.
//!
//! Both can be empty (count = 0), which is the common case for any
//! circuit that doesn't carry custom-defined gates or annotated
//! operators. Today this module only parses the count headers and
//! short-circuits with `QpyError::Unsupported` when either count is
//! non-zero — the per-payload decoders land in their own slice
//! together with the full INSTRUCTION_PARAM machinery they share
//! with annotation values.

use super::header::QpyHeader;
use super::QpyError;

/// Sizes of the "extras" sections combined. The instruction list
/// begins at `offset + extras_len`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CircuitExtras {
    /// `num_namespaces` from `ANNOTATION_HEADER_STATIC` (V15+ only).
    /// `None` for older blobs where the section isn't written.
    pub num_namespaces: Option<u32>,
    /// `size` from `CUSTOM_CIRCUIT_DEF_HEADER` — number of custom
    /// instruction definitions that follow.
    pub num_custom_definitions: u64,
    /// Total bytes consumed by the extras section. Add to the
    /// post-register offset to get the start of the instruction list.
    pub extras_len: usize,
}

const ANNOTATION_HEADER_STATIC_LEN: usize = 4;
const CUSTOM_CIRCUIT_DEF_HEADER_LEN: usize = 8;
const ANNOTATION_HEADER_INTRODUCED_VERSION: u8 = 15;

/// Read the annotation + custom-circuit-def count headers. Caller
/// passes `offset` = `circuit_offsets[i] + circuit_header::FIXED_LEN
/// + circuit_body.body_len + register_table.table_len`.
pub fn read_circuit_extras(
    bytes: &[u8],
    header: &QpyHeader,
    offset: usize,
) -> Result<CircuitExtras, QpyError> {
    let mut cursor = offset;

    let num_namespaces = if header.qpy_version >= ANNOTATION_HEADER_INTRODUCED_VERSION {
        let end = cursor
            .checked_add(ANNOTATION_HEADER_STATIC_LEN)
            .ok_or(QpyError::Truncated {
                what: "annotation_header_static",
                need: usize::MAX,
                got: bytes.len(),
            })?;
        if bytes.len() < end {
            return Err(QpyError::Truncated {
                what: "annotation_header_static",
                need: end,
                got: bytes.len(),
            });
        }
        let n = u32::from_be_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ]);
        cursor = end;
        if n != 0 {
            return Err(QpyError::Unsupported {
                what: "ANNOTATION_HEADER_STATIC namespace payloads",
                detail: "num_namespaces > 0; per-namespace decoder lands with the INSTRUCTION_PARAM slice",
            });
        }
        Some(n)
    } else {
        None
    };

    let custom_def_end =
        cursor
            .checked_add(CUSTOM_CIRCUIT_DEF_HEADER_LEN)
            .ok_or(QpyError::Truncated {
                what: "custom_circuit_def_header",
                need: usize::MAX,
                got: bytes.len(),
            })?;
    if bytes.len() < custom_def_end {
        return Err(QpyError::Truncated {
            what: "custom_circuit_def_header",
            need: custom_def_end,
            got: bytes.len(),
        });
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[cursor..custom_def_end]);
    let num_custom_definitions = u64::from_be_bytes(buf);
    cursor = custom_def_end;
    if num_custom_definitions != 0 {
        return Err(QpyError::Unsupported {
            what: "CUSTOM_CIRCUIT_DEF_HEADER payloads",
            detail: "size > 0; per-definition decoder lands alongside the INSTRUCTION decoder",
        });
    }

    Ok(CircuitExtras {
        num_namespaces,
        num_custom_definitions,
        extras_len: cursor - offset,
    })
}

#[cfg(test)]
mod tests {
    use super::super::header::SymbolicEncoding;
    use super::*;

    fn header_v17() -> QpyHeader {
        QpyHeader {
            qpy_version: 17,
            qiskit_version: (2, 4, 1),
            num_circuits: 1,
            symbolic_encoding: Some(SymbolicEncoding::Sympy),
        }
    }

    fn header_v14() -> QpyHeader {
        QpyHeader {
            qpy_version: 14,
            qiskit_version: (1, 3, 0),
            num_circuits: 1,
            symbolic_encoding: Some(SymbolicEncoding::Sympy),
        }
    }

    #[test]
    fn empty_extras_for_v17_blob() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&0u32.to_be_bytes()); // 0 namespaces
        blob.extend_from_slice(&0u64.to_be_bytes()); // 0 custom defs
        let e = read_circuit_extras(&blob, &header_v17(), 0).unwrap();
        assert_eq!(e.num_namespaces, Some(0));
        assert_eq!(e.num_custom_definitions, 0);
        assert_eq!(e.extras_len, 4 + 8);
    }

    #[test]
    fn pre_v15_skips_annotation_header_entirely() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&0u64.to_be_bytes()); // only custom_def header
        let e = read_circuit_extras(&blob, &header_v14(), 0).unwrap();
        assert_eq!(e.num_namespaces, None);
        assert_eq!(e.num_custom_definitions, 0);
        assert_eq!(e.extras_len, 8);
    }

    #[test]
    fn nonzero_namespaces_returns_unsupported_for_now() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&1u32.to_be_bytes());
        blob.extend_from_slice(&0u64.to_be_bytes());
        match read_circuit_extras(&blob, &header_v17(), 0).unwrap_err() {
            QpyError::Unsupported { what, .. } => {
                assert!(what.contains("ANNOTATION_HEADER_STATIC"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn nonzero_custom_definitions_returns_unsupported_for_now() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&0u32.to_be_bytes());
        blob.extend_from_slice(&3u64.to_be_bytes());
        match read_circuit_extras(&blob, &header_v17(), 0).unwrap_err() {
            QpyError::Unsupported { what, .. } => {
                assert!(what.contains("CUSTOM_CIRCUIT_DEF_HEADER"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn truncation_in_annotation_header_is_typed_error() {
        // Only 2 bytes — short of the 4-byte u32.
        let blob = vec![0u8, 0];
        match read_circuit_extras(&blob, &header_v17(), 0).unwrap_err() {
            QpyError::Truncated { what, .. } => assert_eq!(what, "annotation_header_static"),
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn truncation_in_custom_def_header_is_typed_error() {
        // 4 bytes for annotation header + only 5 bytes (need 8) for custom_def.
        let mut blob = vec![0u8; 4];
        blob.extend_from_slice(&[0u8; 5]);
        match read_circuit_extras(&blob, &header_v17(), 0).unwrap_err() {
            QpyError::Truncated { what, .. } => assert_eq!(what, "custom_circuit_def_header"),
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn handles_offset_into_a_larger_blob() {
        let mut blob = vec![0xCCu8; 25];
        blob.extend_from_slice(&0u32.to_be_bytes());
        blob.extend_from_slice(&0u64.to_be_bytes());
        let e = read_circuit_extras(&blob, &header_v17(), 25).unwrap();
        assert_eq!(e.extras_len, 12);
    }
}
