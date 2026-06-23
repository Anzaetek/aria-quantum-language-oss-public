//! `INSTRUCTION_PARAM` decoder. Each parameter has a 9-byte
//! header (big-endian) followed by a `size`-byte payload whose
//! interpretation depends on the type byte:
//!
//! ```text
//! struct INSTRUCTION_PARAM {
//!     u8   type;       // see [`InstructionParam`] for the table
//!     u64  size;       // payload byte length
//! }
//! u8[size]  payload
//! ```
//!
//! Per-instruction the parameter list is just `num_parameters`
//! entries written back-to-back.
//!
//! **Endianness quirk:** "for QPY format versions ≤ 17 the encoding
//! of integers and floats as part of `INSTRUCTION_PARAM` is little
//! endian." So the payload of `i`, `f`, `c` parameters is decoded
//! little-endian, even though the 9-byte header itself is
//! big-endian. The same convention applies to `global_phase`
//! payloads (handled in [`super::circuit_body`]).
//!
//! This first slice decodes the simple scalar types (None / int /
//! float / complex / string). Container, symbolic, and pulse-shape
//! types (numpy array, Range, ParameterExpression, sub-circuit,
//! PauliEvolutionGate operator, …) are surfaced as
//! [`InstructionParam::Opaque`] so the caller can either round-trip
//! them or ask the user to fall back to the Qiskit-subprocess path
//! while the per-type decoders land.

use super::parameter_expression::{read_parameter_expression, DecodedParameterExpression};
use super::QpyError;

/// One decoded parameter. The `Opaque` variant carries the raw
/// payload bytes plus the type byte for any future decoder.
///
/// Type-key reference (`qiskit/qpy/type_keys.py::Value`):
/// `b'i'` int, `b'f'` float, `b'c'` complex, `b's'` string,
/// `b'p'` Parameter, `b'e'` ParameterExpression, `b'v'`
/// ParameterVector, `b'z'` Null/None, `b'n'` numpy obj,
/// `b'd'` case-default sentinel, `b'R'` Register, `b'x'`
/// Expression, `b'm'` Modifier.
#[derive(Clone, Debug, PartialEq)]
pub enum InstructionParam {
    /// `b'z'` — Python `None`. Always size 0.
    Null,
    /// `b'd'` — Switch-case default sentinel. Always size 0.
    CaseDefault,
    /// `b'i'` — Python int. Decoded into `i64` when `size <= 8`;
    /// the raw payload is preserved for the larger arbitrary-
    /// precision case.
    Integer { value: i64, raw: Vec<u8> },
    /// `b'f'` — IEEE 754 double. 8 bytes, little-endian for
    /// QPY ≤ 17.
    Float(f64),
    /// `b'c'` — complex double. 16 bytes total (re + im), each
    /// little-endian for QPY ≤ 17.
    Complex { re: f64, im: f64 },
    /// `b's'` — UTF-8 string.
    String(String),
    /// `b'p'` — Qiskit `Parameter` (single named symbol). Decoded
    /// from `!H16s` + UTF-8 name: `{ name_size: u16, uuid: [u8;
    /// 16], name: u8[name_size] }`.
    Parameter { uuid: [u8; 16], name: String },
    /// `b'v'` — Qiskit `ParameterVectorElement`. Decoded from
    /// `!HQ16sQ` + UTF-8 vector name:
    /// `{ vector_name_size: u16, vector_size: u64, uuid: [u8; 16],
    /// index: u64, vector_name: u8[vector_name_size] }`.
    ///
    /// Each element has its own UUID; the convention is
    /// `element.uuid = root_uuid + index`, so an element on its own
    /// uniquely identifies both its position and its parent vector.
    /// The builder treats each element as a fresh
    /// [`omega_core::circuit::SymbolId`] keyed off the element UUID,
    /// matching the per-Parameter behaviour for plain `b'p'`
    /// payloads.
    ParameterVectorElement {
        uuid: [u8; 16],
        vector_name: String,
        vector_size: u64,
        index: u64,
    },
    /// `b'e'` — Qiskit `ParameterExpression`. The replay-stack
    /// program + symbol map is decoded by
    /// [`super::parameter_expression`]; folding it into omega's
    /// `ParamExpr` happens in the builder (see
    /// `builder.rs::param_to_expr`).
    ParameterExpression(DecodedParameterExpression),
    /// Any type byte the simple decoder doesn't yet handle. The
    /// raw payload is preserved verbatim for the future decoder.
    /// Includes `b'n'` (numpy array), `b'a'` (numpy ndarray),
    /// `b'r'` (range), `b'R'`, `b'x'`, `b'm'`.
    Opaque { type_byte: u8, bytes: Vec<u8> },
}

const PARAM_HEADER_LEN: usize = 1 + 8;

/// Read `count` consecutive INSTRUCTION_PARAM entries starting at
/// `offset`. Returns the parsed parameters plus the total byte
/// length consumed (= sum of headers + payloads).
pub fn read_instruction_params(
    bytes: &[u8],
    count: u16,
    offset: usize,
) -> Result<(Vec<InstructionParam>, usize), QpyError> {
    let mut cursor = offset;
    let mut params = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let header_end = cursor
            .checked_add(PARAM_HEADER_LEN)
            .ok_or(QpyError::Truncated {
                what: "instruction_param_header",
                need: usize::MAX,
                got: bytes.len(),
            })?;
        if bytes.len() < header_end {
            return Err(QpyError::Truncated {
                what: "instruction_param_header",
                need: header_end,
                got: bytes.len(),
            });
        }
        let type_byte = bytes[cursor];
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[cursor + 1..header_end]);
        let size = u64::from_be_bytes(buf) as usize;

        let payload_end = header_end.checked_add(size).ok_or(QpyError::Truncated {
            what: "instruction_param_payload",
            need: usize::MAX,
            got: bytes.len(),
        })?;
        if bytes.len() < payload_end {
            return Err(QpyError::Truncated {
                what: "instruction_param_payload",
                need: payload_end,
                got: bytes.len(),
            });
        }
        let payload = &bytes[header_end..payload_end];
        params.push(decode_payload(type_byte, payload)?);
        cursor = payload_end;
    }
    Ok((params, cursor - offset))
}

fn decode_payload(type_byte: u8, payload: &[u8]) -> Result<InstructionParam, QpyError> {
    match type_byte {
        b'z' => {
            if !payload.is_empty() {
                return Err(QpyError::InconsistentParamSize {
                    type_byte,
                    expected: 0,
                    got: payload.len(),
                });
            }
            Ok(InstructionParam::Null)
        }
        b'd' => {
            if !payload.is_empty() {
                return Err(QpyError::InconsistentParamSize {
                    type_byte,
                    expected: 0,
                    got: payload.len(),
                });
            }
            Ok(InstructionParam::CaseDefault)
        }
        b'p' => {
            // PARAMETER struct: !H16s = u16 name_size + [u8; 16] uuid,
            // then u8[name_size] name.
            if payload.len() < 18 {
                return Err(QpyError::Truncated {
                    what: "instruction_param.parameter",
                    need: 18,
                    got: payload.len(),
                });
            }
            let name_size = u16::from_be_bytes([payload[0], payload[1]]) as usize;
            let total = 2 + 16 + name_size;
            if payload.len() != total {
                return Err(QpyError::InconsistentParamSize {
                    type_byte,
                    expected: total,
                    got: payload.len(),
                });
            }
            let mut uuid = [0u8; 16];
            uuid.copy_from_slice(&payload[2..18]);
            let name =
                std::str::from_utf8(&payload[18..total]).map_err(|e| QpyError::InvalidUtf8 {
                    what: "instruction_param_parameter_name",
                    valid_up_to: e.valid_up_to(),
                    len: name_size,
                })?;
            Ok(InstructionParam::Parameter {
                uuid,
                name: name.to_owned(),
            })
        }
        b'i' => Ok(InstructionParam::Integer {
            value: decode_int_le(payload),
            raw: payload.to_vec(),
        }),
        b'f' => {
            if payload.len() != 8 {
                return Err(QpyError::InconsistentParamSize {
                    type_byte,
                    expected: 8,
                    got: payload.len(),
                });
            }
            let mut buf = [0u8; 8];
            buf.copy_from_slice(payload);
            Ok(InstructionParam::Float(f64::from_bits(u64::from_le_bytes(
                buf,
            ))))
        }
        b'c' => {
            if payload.len() != 16 {
                return Err(QpyError::InconsistentParamSize {
                    type_byte,
                    expected: 16,
                    got: payload.len(),
                });
            }
            let mut re_buf = [0u8; 8];
            let mut im_buf = [0u8; 8];
            re_buf.copy_from_slice(&payload[0..8]);
            im_buf.copy_from_slice(&payload[8..16]);
            Ok(InstructionParam::Complex {
                re: f64::from_bits(u64::from_le_bytes(re_buf)),
                im: f64::from_bits(u64::from_le_bytes(im_buf)),
            })
        }
        b's' => {
            let s = std::str::from_utf8(payload).map_err(|e| QpyError::InvalidUtf8 {
                what: "instruction_param_string",
                valid_up_to: e.valid_up_to(),
                len: payload.len(),
            })?;
            Ok(InstructionParam::String(s.to_owned()))
        }
        b'e' => Ok(InstructionParam::ParameterExpression(
            read_parameter_expression(payload)?,
        )),
        b'v' => {
            // PARAMETER_VECTOR_ELEMENT struct: !HQ16sQ + UTF-8 name.
            const FIXED_LEN: usize = 2 + 8 + 16 + 8;
            if payload.len() < FIXED_LEN {
                return Err(QpyError::Truncated {
                    what: "instruction_param.parameter_vector_element_header",
                    need: FIXED_LEN,
                    got: payload.len(),
                });
            }
            let vector_name_size = u16::from_be_bytes([payload[0], payload[1]]) as usize;
            let mut size_buf = [0u8; 8];
            size_buf.copy_from_slice(&payload[2..10]);
            let vector_size = u64::from_be_bytes(size_buf);
            let mut uuid = [0u8; 16];
            uuid.copy_from_slice(&payload[10..26]);
            let mut idx_buf = [0u8; 8];
            idx_buf.copy_from_slice(&payload[26..34]);
            let index = u64::from_be_bytes(idx_buf);

            let total = FIXED_LEN + vector_name_size;
            if payload.len() != total {
                return Err(QpyError::InconsistentParamSize {
                    type_byte,
                    expected: total,
                    got: payload.len(),
                });
            }
            let vector_name = std::str::from_utf8(&payload[FIXED_LEN..total]).map_err(|e| {
                QpyError::InvalidUtf8 {
                    what: "instruction_param.parameter_vector_element_name",
                    valid_up_to: e.valid_up_to(),
                    len: vector_name_size,
                }
            })?;
            Ok(InstructionParam::ParameterVectorElement {
                uuid,
                vector_name: vector_name.to_owned(),
                vector_size,
                index,
            })
        }
        other => Ok(InstructionParam::Opaque {
            type_byte: other,
            bytes: payload.to_vec(),
        }),
    }
}

fn decode_int_le(payload: &[u8]) -> i64 {
    if payload.len() > 8 {
        // Arbitrary precision int — caller can read the raw bytes.
        return 0;
    }
    let mut buf = [0u8; 8];
    buf[..payload.len()].copy_from_slice(payload);
    if let Some(&last) = payload.last() {
        if last & 0x80 != 0 {
            for slot in &mut buf[payload.len()..] {
                *slot = 0xFF;
            }
        }
    }
    i64::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn param_bytes(type_byte: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(PARAM_HEADER_LEN + payload.len());
        out.push(type_byte);
        out.extend_from_slice(&(payload.len() as u64).to_be_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn decodes_a_single_float_param() {
        let phase = std::f64::consts::FRAC_PI_4;
        let blob = param_bytes(b'f', &phase.to_le_bytes());
        let (params, len) = read_instruction_params(&blob, 1, 0).unwrap();
        assert_eq!(params.len(), 1);
        match params[0] {
            InstructionParam::Float(f) => assert_eq!(f, phase),
            ref other => panic!("expected Float, got {other:?}"),
        }
        assert_eq!(len, PARAM_HEADER_LEN + 8);
    }

    #[test]
    fn decodes_a_single_complex_param() {
        let mut payload = Vec::with_capacity(16);
        payload.extend_from_slice(&1.5_f64.to_le_bytes());
        payload.extend_from_slice(&(-2.25_f64).to_le_bytes());
        let blob = param_bytes(b'c', &payload);
        let (params, len) = read_instruction_params(&blob, 1, 0).unwrap();
        assert_eq!(
            params,
            vec![InstructionParam::Complex { re: 1.5, im: -2.25 }]
        );
        assert_eq!(len, PARAM_HEADER_LEN + 16);
    }

    #[test]
    fn decodes_an_integer_param_within_i64_range() {
        let blob = param_bytes(b'i', &(-12345i64).to_le_bytes());
        let (params, _) = read_instruction_params(&blob, 1, 0).unwrap();
        match &params[0] {
            InstructionParam::Integer { value, raw } => {
                assert_eq!(*value, -12345);
                assert_eq!(raw.len(), 8);
            }
            other => panic!("expected Integer, got {other:?}"),
        }
    }

    #[test]
    fn decodes_a_short_integer_with_sign_extension() {
        // 1 byte = 0xFF → -1 with sign extension.
        let blob = param_bytes(b'i', &[0xFFu8]);
        let (params, _) = read_instruction_params(&blob, 1, 0).unwrap();
        match &params[0] {
            InstructionParam::Integer { value, raw } => {
                assert_eq!(*value, -1);
                assert_eq!(raw, &vec![0xFF]);
            }
            other => panic!("expected Integer, got {other:?}"),
        }
    }

    #[test]
    fn decodes_a_null_param() {
        let blob = param_bytes(b'z', &[]);
        let (params, _) = read_instruction_params(&blob, 1, 0).unwrap();
        assert_eq!(params, vec![InstructionParam::Null]);
    }

    #[test]
    fn decodes_a_case_default_sentinel() {
        let blob = param_bytes(b'd', &[]);
        let (params, _) = read_instruction_params(&blob, 1, 0).unwrap();
        assert_eq!(params, vec![InstructionParam::CaseDefault]);
    }

    #[test]
    fn decodes_a_parameter_with_uuid_and_name() {
        // PARAMETER struct: H16s + name. Build a 20-byte payload
        // for "θ" (2 UTF-8 bytes) with a known UUID.
        let uuid = [
            0x32, 0xfe, 0xac, 0x7d, 0x74, 0x64, 0x49, 0xe7, 0xb5, 0xc2, 0xf3, 0xa1, 0x67, 0x3f,
            0x86, 0xcc,
        ];
        let mut payload = Vec::with_capacity(20);
        payload.extend_from_slice(&2u16.to_be_bytes());
        payload.extend_from_slice(&uuid);
        payload.extend_from_slice("θ".as_bytes());
        let blob = param_bytes(b'p', &payload);
        let (params, _) = read_instruction_params(&blob, 1, 0).unwrap();
        match &params[0] {
            InstructionParam::Parameter { uuid: u, name } => {
                assert_eq!(*u, uuid);
                assert_eq!(name, "θ");
            }
            other => panic!("expected Parameter, got {other:?}"),
        }
    }

    #[test]
    fn rejects_truncated_parameter_payload() {
        // 18 bytes: name_size=2 + uuid + 0 of the 2 expected name bytes.
        let mut payload = vec![0u8, 2];
        payload.extend_from_slice(&[0u8; 16]);
        let blob = param_bytes(b'p', &payload);
        match read_instruction_params(&blob, 1, 0).unwrap_err() {
            QpyError::InconsistentParamSize {
                type_byte,
                expected,
                got,
            } => {
                assert_eq!(type_byte, b'p');
                assert_eq!(expected, 20);
                assert_eq!(got, 18);
            }
            other => panic!("expected InconsistentParamSize, got {other:?}"),
        }
    }

    #[test]
    fn decodes_a_string_param() {
        let blob = param_bytes(b's', b"omega");
        let (params, _) = read_instruction_params(&blob, 1, 0).unwrap();
        assert_eq!(params, vec![InstructionParam::String("omega".into())]);
    }

    #[test]
    fn unknown_type_byte_falls_through_to_opaque() {
        let blob = param_bytes(b'X', &[1, 2, 3]);
        let (params, _) = read_instruction_params(&blob, 1, 0).unwrap();
        assert_eq!(
            params,
            vec![InstructionParam::Opaque {
                type_byte: b'X',
                bytes: vec![1, 2, 3],
            }]
        );
    }

    #[test]
    fn decodes_two_params_back_to_back() {
        let mut blob = param_bytes(b'f', &1.0_f64.to_le_bytes());
        blob.extend_from_slice(&param_bytes(b'i', &7i64.to_le_bytes()));
        let (params, len) = read_instruction_params(&blob, 2, 0).unwrap();
        assert!(matches!(params[0], InstructionParam::Float(f) if f == 1.0));
        assert!(matches!(
            params[1],
            InstructionParam::Integer { value: 7, .. }
        ));
        assert_eq!(len, 2 * (PARAM_HEADER_LEN + 8));
    }

    #[test]
    fn rejects_inconsistent_size_for_float_param() {
        // 'f' implies size 8.
        let blob = param_bytes(b'f', &[0u8; 4]);
        match read_instruction_params(&blob, 1, 0).unwrap_err() {
            QpyError::InconsistentParamSize {
                type_byte,
                expected,
                got,
            } => {
                assert_eq!(type_byte, b'f');
                assert_eq!(expected, 8);
                assert_eq!(got, 4);
            }
            other => panic!("expected InconsistentParamSize, got {other:?}"),
        }
    }

    #[test]
    fn rejects_inconsistent_size_for_null_param() {
        let blob = param_bytes(b'z', &[0u8]);
        match read_instruction_params(&blob, 1, 0).unwrap_err() {
            QpyError::InconsistentParamSize {
                type_byte,
                expected,
                got,
            } => {
                assert_eq!(type_byte, b'z');
                assert_eq!(expected, 0);
                assert_eq!(got, 1);
            }
            other => panic!("expected InconsistentParamSize, got {other:?}"),
        }
    }

    #[test]
    fn n_byte_falls_through_to_opaque_as_numpy_obj_carrier() {
        // 'n' = NUMPY_OBJ in qiskit's type_keys; the per-type
        // numpy decoder lands later. Until then, it falls through
        // to Opaque so the caller can decide whether to drop back
        // to the Qiskit subprocess.
        let blob = param_bytes(b'n', &[1u8, 2, 3]);
        let (params, _) = read_instruction_params(&blob, 1, 0).unwrap();
        assert_eq!(
            params,
            vec![InstructionParam::Opaque {
                type_byte: b'n',
                bytes: vec![1, 2, 3],
            }]
        );
    }

    #[test]
    fn rejects_non_utf8_string_param() {
        let blob = param_bytes(b's', &[0xFFu8, 0xFE]);
        match read_instruction_params(&blob, 1, 0).unwrap_err() {
            QpyError::InvalidUtf8 { what, .. } => {
                assert_eq!(what, "instruction_param_string");
            }
            other => panic!("expected InvalidUtf8, got {other:?}"),
        }
    }

    #[test]
    fn truncation_in_param_header_is_typed_error() {
        // Only 5 bytes — short of the 9-byte header.
        let blob = vec![b'f', 0, 0, 0, 0];
        match read_instruction_params(&blob, 1, 0).unwrap_err() {
            QpyError::Truncated { what, .. } => {
                assert_eq!(what, "instruction_param_header");
            }
            other => panic!("expected Truncated(header), got {other:?}"),
        }
    }

    #[test]
    fn truncation_in_param_payload_is_typed_error() {
        // 9 bytes of header claiming 16-byte payload, only 8 bytes follow.
        let mut blob = vec![b'c'];
        blob.extend_from_slice(&16u64.to_be_bytes());
        blob.extend_from_slice(&[0u8; 8]);
        match read_instruction_params(&blob, 1, 0).unwrap_err() {
            QpyError::Truncated { what, .. } => {
                assert_eq!(what, "instruction_param_payload");
            }
            other => panic!("expected Truncated(payload), got {other:?}"),
        }
    }

    #[test]
    fn empty_count_returns_empty_params() {
        let (params, len) = read_instruction_params(b"", 0, 0).unwrap();
        assert!(params.is_empty());
        assert_eq!(len, 0);
    }

    #[test]
    fn handles_offset_into_a_larger_blob() {
        let mut blob = vec![0xCCu8; 25];
        blob.extend_from_slice(&param_bytes(b'i', &7i64.to_le_bytes()));
        let (params, _) = read_instruction_params(&blob, 1, 25).unwrap();
        match &params[0] {
            InstructionParam::Integer { value, .. } => assert_eq!(*value, 7),
            other => panic!("expected Integer(7), got {other:?}"),
        }
    }

    #[test]
    fn v_byte_decodes_a_parameter_vector_element() {
        // Build a synthetic PARAMETER_VECTOR_ELEMENT payload: name="x"
        // (1 byte UTF-8), vector_size=3, uuid=0xCD repeated, index=1.
        let uuid = [0xCDu8; 16];
        let mut payload = Vec::new();
        payload.extend_from_slice(&1u16.to_be_bytes()); // name_size=1
        payload.extend_from_slice(&3u64.to_be_bytes()); // vector_size=3
        payload.extend_from_slice(&uuid);
        payload.extend_from_slice(&1u64.to_be_bytes()); // index=1
        payload.extend_from_slice(b"x");
        let blob = param_bytes(b'v', &payload);
        let (params, _) = read_instruction_params(&blob, 1, 0).unwrap();
        match &params[0] {
            InstructionParam::ParameterVectorElement {
                uuid: u,
                vector_name,
                vector_size,
                index,
            } => {
                assert_eq!(*u, uuid);
                assert_eq!(vector_name, "x");
                assert_eq!(*vector_size, 3);
                assert_eq!(*index, 1);
            }
            other => panic!("expected ParameterVectorElement, got {other:?}"),
        }
    }

    #[test]
    fn v_byte_rejects_truncated_header() {
        // Only 20 bytes (need 34 for the fixed header).
        let payload = vec![0u8; 20];
        let blob = param_bytes(b'v', &payload);
        match read_instruction_params(&blob, 1, 0).unwrap_err() {
            QpyError::Truncated { what, .. } => {
                assert_eq!(what, "instruction_param.parameter_vector_element_header");
            }
            other => panic!("expected Truncated header, got {other:?}"),
        }
    }

    #[test]
    fn v_byte_rejects_inconsistent_size_when_name_bytes_dont_match() {
        // 34-byte fixed header claims name_size=5 but no name bytes follow.
        let mut payload = Vec::new();
        payload.extend_from_slice(&5u16.to_be_bytes());
        payload.extend_from_slice(&[0u8; 32]); // vector_size + uuid + index
        let blob = param_bytes(b'v', &payload);
        match read_instruction_params(&blob, 1, 0).unwrap_err() {
            QpyError::InconsistentParamSize {
                type_byte,
                expected,
                got,
            } => {
                assert_eq!(type_byte, b'v');
                assert_eq!(expected, 39); // 34 + 5
                assert_eq!(got, 34);
            }
            other => panic!("expected InconsistentParamSize, got {other:?}"),
        }
    }

    #[test]
    fn e_byte_routes_into_parameter_expression_decoder() {
        // Build a minimal ADD(1.5, theta) PARAMETER_EXPR payload and
        // wrap it in the 9-byte INSTRUCTION_PARAM header.
        let uuid = [0xBBu8; 16];
        let mut elem = [0u8; 35];
        elem[0] = 0; // ADD
        elem[1] = b'f';
        // !Qd: first 8 bytes filler, last 8 = 1.5 big-endian.
        elem[10..18].copy_from_slice(&1.5_f64.to_be_bytes());
        elem[18] = b'p';
        elem[19..35].copy_from_slice(&uuid);

        let mut payload = Vec::new();
        payload.extend_from_slice(&1u64.to_be_bytes()); // map_elements=1
        payload.extend_from_slice(&35u64.to_be_bytes()); // expr_size=35
        payload.extend_from_slice(&elem);
        // Symbol map: PARAM_EXPR_MAP_ELEM_V3 + PARAMETER struct
        payload.push(b'p'); // symbol_type
        payload.push(b'p'); // value_type
        payload.extend_from_slice(&0u64.to_be_bytes()); // value_size
        payload.extend_from_slice(&2u16.to_be_bytes()); // name_size
        payload.extend_from_slice(&uuid);
        payload.extend_from_slice("θ".as_bytes());

        let blob = param_bytes(b'e', &payload);
        let (params, _) = read_instruction_params(&blob, 1, 0).unwrap();
        match &params[0] {
            InstructionParam::ParameterExpression(pe) => {
                use super::super::parameter_expression::ParamExprSymbolKind;
                assert_eq!(pe.elements.len(), 1);
                assert_eq!(pe.elements[0].op_code, 0);
                assert_eq!(pe.symbols.len(), 1);
                match &pe.symbols[0].kind {
                    ParamExprSymbolKind::Parameter { name } => assert_eq!(name, "θ"),
                    other => panic!("expected Parameter, got {other:?}"),
                }
                assert_eq!(pe.symbols[0].uuid, uuid);
            }
            other => panic!("expected ParameterExpression, got {other:?}"),
        }
    }
}
