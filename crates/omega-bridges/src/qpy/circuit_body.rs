//! Variable-length per-circuit fields immediately following
//! [`super::circuit_header::CircuitHeader`]: name, global_phase,
//! metadata.
//!
//! Layout (in this order, all back-to-back, no padding):
//!
//! ```text
//! u8[name_size]            name           // UTF-8 circuit name
//! u8[global_phase_size]    global_phase   // see GlobalPhase below
//! u8[metadata_size]        metadata       // JSON-encoded UTF-8
//! ```
//!
//! ## Global-phase encoding
//!
//! `global_phase_type` (1 byte, picked up from the fixed CIRCUIT_HEADER)
//! tells the reader how to interpret the next `global_phase_size`
//! bytes:
//!
//! - `b'n'` — None / no phase (size 0).
//! - `b'i'` — Python integer, little-endian for QPY ≤ 17. Size in
//!   bytes; we decode up to 8 bytes as `i64`. Larger ints are
//!   surfaced as opaque bytes via [`GlobalPhase::Integer`].
//! - `b'f'` — IEEE 754 double, little-endian for QPY ≤ 17 (size 8).
//! - `b'p'` / `b'e'` — `ParameterExpression` symbolic blob (sympy
//!   or symengine). Today returned as opaque bytes — decoding the
//!   expression tree itself lives in a follow-up alongside the
//!   per-circuit parameter table.

use super::circuit_header::CircuitHeader;
use super::QpyError;

/// Decoded `global_phase` value. Three concrete variants today plus
/// `Symbolic` for the parameter-expression cases that need their own
/// decoder.
#[derive(Clone, Debug, PartialEq)]
pub enum GlobalPhase {
    /// `b'n'` — explicitly absent.
    None,
    /// `b'f'` — 8-byte IEEE 754 double.
    Float(f64),
    /// `b'i'` — Python int, decoded from the wire bytes (little-
    /// endian for QPY ≤ 17). The QPY format permits arbitrary-
    /// precision Python ints; if the producer wrote an int that
    /// doesn't fit in `i64`, the raw bytes show up here for the
    /// caller to handle.
    Integer { value: i64, raw: Vec<u8> },
    /// `b'p'` (sympy) or `b'e'` (symengine) — opaque expression
    /// bytes. The first byte tells the caller which encoding the
    /// blob uses; decoding the expression tree itself is a separate
    /// concern.
    Symbolic { encoding: u8, bytes: Vec<u8> },
    /// Unknown / forward-compatibility variant — surfaces the type
    /// byte and the raw payload so the caller can decide whether
    /// to error or pass through.
    Other { type_byte: u8, bytes: Vec<u8> },
}

/// Variable-length portion of the per-circuit block. Borrows from
/// the input slice for the name + metadata UTF-8 to avoid copying;
/// owns the global_phase variant since it's small.
#[derive(Clone, Debug, PartialEq)]
pub struct CircuitBody<'a> {
    /// Circuit name (UTF-8).
    pub name: &'a str,
    /// Decoded global phase.
    pub global_phase: GlobalPhase,
    /// JSON-encoded metadata (UTF-8). Caller can `serde_json::from_str`
    /// when they need it; raw bytes work for round-trip.
    pub metadata: &'a str,
    /// Total bytes consumed by name + global_phase + metadata. Add
    /// to `circuit_offsets[i] + circuit_header::FIXED_LEN` to find
    /// the start of the register table.
    pub body_len: usize,
}

/// Read the name / global_phase / metadata triple that follows the
/// fixed CIRCUIT_HEADER struct. `offset` is the absolute byte
/// offset into `bytes` where the variable section begins (i.e.
/// `circuit_offsets[i] + circuit_header::FIXED_LEN`).
pub fn read_circuit_body<'a>(
    bytes: &'a [u8],
    header: &CircuitHeader,
    offset: usize,
) -> Result<CircuitBody<'a>, QpyError> {
    let name_size = header.name_size as usize;
    let phase_size = header.global_phase_size as usize;
    let metadata_size = header.metadata_size as usize;
    let total = name_size
        .checked_add(phase_size)
        .and_then(|n| n.checked_add(metadata_size))
        .ok_or(QpyError::Truncated {
            what: "circuit_body",
            need: usize::MAX,
            got: bytes.len(),
        })?;
    let end = offset.checked_add(total).ok_or(QpyError::Truncated {
        what: "circuit_body",
        need: usize::MAX,
        got: bytes.len(),
    })?;
    if bytes.len() < end {
        return Err(QpyError::Truncated {
            what: "circuit_body",
            need: end,
            got: bytes.len(),
        });
    }

    let name_end = offset + name_size;
    let phase_end = name_end + phase_size;
    let metadata_end = phase_end + metadata_size;

    let name = decode_utf8(&bytes[offset..name_end], "circuit_name")?;
    let global_phase = decode_global_phase(header.global_phase_type, &bytes[name_end..phase_end])?;
    let metadata = decode_utf8(&bytes[phase_end..metadata_end], "circuit_metadata")?;

    Ok(CircuitBody {
        name,
        global_phase,
        metadata,
        body_len: total,
    })
}

fn decode_utf8<'a>(bytes: &'a [u8], what: &'static str) -> Result<&'a str, QpyError> {
    std::str::from_utf8(bytes).map_err(|e| QpyError::InvalidUtf8 {
        what,
        valid_up_to: e.valid_up_to(),
        len: bytes.len(),
    })
}

fn decode_global_phase(type_byte: u8, payload: &[u8]) -> Result<GlobalPhase, QpyError> {
    match type_byte {
        b'n' => {
            if !payload.is_empty() {
                return Err(QpyError::InconsistentGlobalPhaseSize {
                    type_byte,
                    expected: 0,
                    got: payload.len(),
                });
            }
            Ok(GlobalPhase::None)
        }
        b'f' => {
            if payload.len() != 8 {
                return Err(QpyError::InconsistentGlobalPhaseSize {
                    type_byte,
                    expected: 8,
                    got: payload.len(),
                });
            }
            // QPY ≤ 17 documents integer + float encoding in
            // INSTRUCTION_PARAM as little-endian; the same convention
            // applies to global_phase per a side-by-side decode of a
            // real fixture (see `qpy_header_real_fixture` integration
            // test). u64::from_le_bytes → f64::from_bits keeps NaN
            // bits intact.
            let mut buf = [0u8; 8];
            buf.copy_from_slice(payload);
            Ok(GlobalPhase::Float(f64::from_bits(u64::from_le_bytes(buf))))
        }
        b'i' => {
            // Python int is arbitrary precision. We try to fit into
            // i64; if the wire encoding is larger or doesn't fit
            // sign-extended, surface the raw bytes for the caller.
            let value = if payload.len() <= 8 {
                let mut buf = [0u8; 8];
                buf[..payload.len()].copy_from_slice(payload);
                // Sign-extend if the high byte's MSB is set.
                if let Some(&last) = payload.last() {
                    if last & 0x80 != 0 {
                        for slot in &mut buf[payload.len()..] {
                            *slot = 0xFF;
                        }
                    }
                }
                i64::from_le_bytes(buf)
            } else {
                0
            };
            Ok(GlobalPhase::Integer {
                value,
                raw: payload.to_vec(),
            })
        }
        b'p' | b'e' => Ok(GlobalPhase::Symbolic {
            encoding: type_byte,
            bytes: payload.to_vec(),
        }),
        other => Ok(GlobalPhase::Other {
            type_byte: other,
            bytes: payload.to_vec(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(name_size: u16, gp_type: u8, gp_size: u16, metadata_size: u64) -> CircuitHeader {
        CircuitHeader {
            name_size,
            global_phase_type: gp_type,
            global_phase_size: gp_size,
            num_qubits: 0,
            num_clbits: 0,
            metadata_size,
            num_registers: 0,
            num_instructions: 0,
            num_vars: 0,
        }
    }

    #[test]
    fn decodes_float_global_phase_round_trip() {
        let h = header(0, b'f', 8, 2);
        let mut blob = Vec::new();
        // empty name, then 8-byte f64 little-endian, then `{}`.
        let phase = 1.7320508075688772_f64; // √3 — distinct from any common constant
        blob.extend_from_slice(&phase.to_le_bytes());
        blob.extend_from_slice(b"{}");
        let body = read_circuit_body(&blob, &h, 0).unwrap();
        assert_eq!(body.name, "");
        assert_eq!(body.global_phase, GlobalPhase::Float(phase));
        assert_eq!(body.metadata, "{}");
        assert_eq!(body.body_len, 8 + 2);
    }

    #[test]
    fn decodes_named_circuit_with_no_phase() {
        // name_size=4 ("bell"), 'n' / 0, metadata={}.
        let h = header(4, b'n', 0, 2);
        let mut blob = Vec::new();
        blob.extend_from_slice(b"bell");
        blob.extend_from_slice(b"{}");
        let body = read_circuit_body(&blob, &h, 0).unwrap();
        assert_eq!(body.name, "bell");
        assert_eq!(body.global_phase, GlobalPhase::None);
        assert_eq!(body.metadata, "{}");
        assert_eq!(body.body_len, 4 + 2);
    }

    #[test]
    fn decodes_integer_global_phase_within_i64_range() {
        let h = header(0, b'i', 8, 0);
        let blob = (-12345i64).to_le_bytes().to_vec();
        let body = read_circuit_body(&blob, &h, 0).unwrap();
        match body.global_phase {
            GlobalPhase::Integer { value, raw } => {
                assert_eq!(value, -12345);
                assert_eq!(raw.len(), 8);
            }
            other => panic!("expected Integer, got {other:?}"),
        }
    }

    #[test]
    fn decodes_short_integer_global_phase_with_sign_extension() {
        // 1 byte = -1 (0xFF). Should sign-extend to i64::from(-1).
        let h = header(0, b'i', 1, 0);
        let blob = vec![0xFFu8];
        let body = read_circuit_body(&blob, &h, 0).unwrap();
        match body.global_phase {
            GlobalPhase::Integer { value, raw } => {
                assert_eq!(value, -1);
                assert_eq!(raw, vec![0xFF]);
            }
            other => panic!("expected Integer, got {other:?}"),
        }
    }

    #[test]
    fn decodes_symbolic_phase_payload_opaquely() {
        let h = header(0, b'p', 5, 0);
        let blob = vec![1u8, 2, 3, 4, 5];
        let body = read_circuit_body(&blob, &h, 0).unwrap();
        assert_eq!(
            body.global_phase,
            GlobalPhase::Symbolic {
                encoding: b'p',
                bytes: blob.clone(),
            }
        );
    }

    #[test]
    fn decodes_unknown_phase_type_into_other_variant() {
        let h = header(0, b'X', 3, 0);
        let blob = vec![10u8, 20, 30];
        let body = read_circuit_body(&blob, &h, 0).unwrap();
        assert_eq!(
            body.global_phase,
            GlobalPhase::Other {
                type_byte: b'X',
                bytes: blob.clone(),
            }
        );
    }

    #[test]
    fn rejects_non_utf8_circuit_name() {
        let h = header(2, b'n', 0, 0);
        let blob = vec![0xFFu8, 0xFE]; // not valid UTF-8
        match read_circuit_body(&blob, &h, 0).unwrap_err() {
            QpyError::InvalidUtf8 { what, .. } => assert_eq!(what, "circuit_name"),
            other => panic!("expected InvalidUtf8(circuit_name), got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_utf8_metadata() {
        let h = header(0, b'n', 0, 2);
        let blob = vec![0xFFu8, 0xFE];
        match read_circuit_body(&blob, &h, 0).unwrap_err() {
            QpyError::InvalidUtf8 { what, .. } => assert_eq!(what, "circuit_metadata"),
            other => panic!("expected InvalidUtf8(circuit_metadata), got {other:?}"),
        }
    }

    #[test]
    fn rejects_inconsistent_size_for_float_phase() {
        // global_phase_type='f' implies size 8, but header lies.
        let h = header(0, b'f', 4, 0);
        let blob = vec![0u8; 4];
        match read_circuit_body(&blob, &h, 0).unwrap_err() {
            QpyError::InconsistentGlobalPhaseSize {
                type_byte,
                expected,
                got,
            } => {
                assert_eq!(type_byte, b'f');
                assert_eq!(expected, 8);
                assert_eq!(got, 4);
            }
            other => panic!("expected InconsistentGlobalPhaseSize, got {other:?}"),
        }
    }

    #[test]
    fn rejects_nonzero_size_for_none_phase() {
        let h = header(0, b'n', 1, 0);
        let blob = vec![0u8];
        match read_circuit_body(&blob, &h, 0).unwrap_err() {
            QpyError::InconsistentGlobalPhaseSize {
                type_byte,
                expected,
                got,
            } => {
                assert_eq!(type_byte, b'n');
                assert_eq!(expected, 0);
                assert_eq!(got, 1);
            }
            other => panic!("expected InconsistentGlobalPhaseSize, got {other:?}"),
        }
    }

    #[test]
    fn truncated_body_reports_circuit_body_kind() {
        let h = header(10, b'f', 8, 2); // needs 20 bytes
        let blob = vec![0u8; 5];
        match read_circuit_body(&blob, &h, 0).unwrap_err() {
            QpyError::Truncated { what, need, got } => {
                assert_eq!(what, "circuit_body");
                assert_eq!(need, 20);
                assert_eq!(got, 5);
            }
            other => panic!("expected Truncated(circuit_body), got {other:?}"),
        }
    }

    #[test]
    fn handles_offset_into_a_larger_blob() {
        let h = header(2, b'n', 0, 4);
        let mut blob = vec![0xCCu8; 30];
        blob.extend_from_slice(b"hi");
        blob.extend_from_slice(b"null");
        let body = read_circuit_body(&blob, &h, 30).unwrap();
        assert_eq!(body.name, "hi");
        assert_eq!(body.metadata, "null");
        assert_eq!(body.body_len, 6);
    }
}
