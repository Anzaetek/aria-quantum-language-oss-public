//! `INSTRUCTION_PARAM` payload decoder for the `b'e'` type byte
//! (`ParameterExpression`).
//!
//! Layout (V13+ — the only one Qiskit 2.x emits):
//!
//! ```text
//! struct PARAMETER_EXPR {
//!     u64 map_elements;   // big-endian
//!     u64 expr_size;      // big-endian
//! }
//! u8[expr_size]  expr_bytes        // sequence of 35-byte
//!                                  // PARAM_EXPR_ELEM_V13 records
//! map_elements × {
//!     PARAM_EXPR_MAP_ELEM_V3 {
//!         u8 symbol_type;          // see [`ParamExprOperand`] for the type-key table
//!         u8 value_type;
//!         u64 size;                // bytes of value-payload that follow the symbol
//!     }
//!     symbol_struct                // PARAMETER (`!H16s` + name) for symbol_type=b'p'
//!                                  // PARAMETER_VECTOR_ELEMENT (...) for symbol_type=b'v'
//!     u8[size] value_bytes         // size encodes any per-symbol value payload
//! }
//! ```
//!
//! Each `PARAM_EXPR_ELEM_V13` is `!Bc16sc16s` = 35 bytes:
//!
//! ```text
//! u8 op_code        // see [`ParamExprOpCode`] table
//! u8 lhs_type       // type-key for the LHS payload
//! u8[16] lhs        // LHS payload (interpretation per lhs_type)
//! u8 rhs_type
//! u8[16] rhs
//! ```
//!
//! The 35-byte record is the *fixed-width* form. Some operand types
//! (`b's'` / `b'e'` / `b'u'`) are framing markers that introduce a
//! variable-length follow-on block — those are not produced by simple
//! algebraic ParameterExpressions like `theta + 1.5`. The follow-on
//! decoder lands when we have a fixture that exercises it.
//!
//! Numbers inside the fixed `lhs` / `rhs` slots are *big-endian*
//! (matching Qiskit's `_encode_replay_entry` which uses `!Qd` / `!Qq`
//! / `!dd`), unlike the simple-scalar `INSTRUCTION_PARAM` payloads
//! in the parent module which are little-endian for QPY ≤ 17.

use super::QpyError;

/// One decoded operand — what a 16-byte LHS or RHS payload encodes,
/// indexed by the type byte the producer emitted.
///
/// Type-key reference (`qiskit/qpy/binary_io/value.py::_encode_replay_entry`):
/// `b'p'` Parameter (16-byte UUID), `b'f'` float (`!Qd`), `b'i'` int
/// (`!Qq`), `b'c'` complex (`!dd`), `b'n'` Python None / no-op marker,
/// `b's'` start-of-substitution marker, `b'e'` end-of-substitution
/// marker, `b'u'` substitution-map size carrier (the first 8 bytes of
/// the 16-byte slot are the follow-on block size).
#[derive(Clone, Debug, PartialEq)]
pub enum ParamExprOperand {
    /// `b'p'` — reference to a Parameter by UUID. The UUID indexes
    /// into [`DecodedParameterExpression::symbols`].
    ParameterUuid([u8; 16]),
    /// `b'f'` — IEEE 754 double, big-endian, packed via `!Qd` so the
    /// first 8 bytes of the 16-byte slot are zero filler.
    Float(f64),
    /// `b'i'` — signed 64-bit int, big-endian, packed via `!Qq`.
    Int(i64),
    /// `b'c'` — complex double `(re, im)`, both big-endian, packed
    /// via `!dd` — uses the full 16-byte slot.
    Complex { re: f64, im: f64 },
    /// `b'n'` — None / no-op. Producer used this when one side of the
    /// op is unused (e.g. unary ops that pack the sole operand into LHS
    /// only). Always 16 zero bytes.
    None,
    /// `b's'` / `b'e'` — substitution-block start/end markers. Not
    /// produced by simple algebraic expressions; included so the
    /// decoder can surface them as `Unsupported` rather than silently
    /// misinterpret.
    SubstitutionMarker { which: char },
    /// `b'u'` — substitution-map size carrier. The first 8 bytes of
    /// the 16-byte slot are a `u64` (big-endian) byte count of a
    /// follow-on block in the expr stream.
    SubstitutionMapSize(u64),
}

impl ParamExprOperand {
    fn decode(type_byte: u8, bytes: &[u8; 16]) -> Result<Self, QpyError> {
        Ok(match type_byte {
            b'p' => ParamExprOperand::ParameterUuid(*bytes),
            b'f' => {
                // `!Qd` — first 8 bytes are filler zeros, last 8 are
                // the big-endian double.
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&bytes[8..16]);
                ParamExprOperand::Float(f64::from_bits(u64::from_be_bytes(buf)))
            }
            b'i' => {
                // `!Qq` — same layout as `!Qd` but the trailing 8
                // bytes are a signed 64-bit int.
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&bytes[8..16]);
                ParamExprOperand::Int(i64::from_be_bytes(buf))
            }
            b'c' => {
                let mut re_b = [0u8; 8];
                let mut im_b = [0u8; 8];
                re_b.copy_from_slice(&bytes[0..8]);
                im_b.copy_from_slice(&bytes[8..16]);
                ParamExprOperand::Complex {
                    re: f64::from_bits(u64::from_be_bytes(re_b)),
                    im: f64::from_bits(u64::from_be_bytes(im_b)),
                }
            }
            b'n' => ParamExprOperand::None,
            b's' => ParamExprOperand::SubstitutionMarker { which: 's' },
            b'e' => ParamExprOperand::SubstitutionMarker { which: 'e' },
            b'u' => {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&bytes[0..8]);
                ParamExprOperand::SubstitutionMapSize(u64::from_be_bytes(buf))
            }
            other => {
                return Err(QpyError::UnknownParamExprOperandKind { byte: other });
            }
        })
    }
}

/// One decoded `PARAM_EXPR_ELEM_V13` record — a single replay-stack
/// step from the producer's serialised expression.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamExprElement {
    /// Raw op-code byte. `255` is the producer's "no-op" sentinel
    /// used to wrap framing markers; otherwise the value matches one
    /// of [`ParamExprOpCode`]. The decoder keeps the raw byte to
    /// preserve any future op codes; [`ParamExprOpCode::from_u8`]
    /// classifies it.
    pub op_code: u8,
    pub lhs: ParamExprOperand,
    pub rhs: ParamExprOperand,
}

/// Per-symbol kind discriminator inside a
/// [`DecodedParameterExpression`]. Mirrors the
/// `PARAM_EXPR_MAP_ELEM_V3.symbol_type` byte: `b'p'` for plain
/// `Parameter`, `b'v'` for one element of a `ParameterVector`.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamExprSymbolKind {
    /// `b'p'` — plain `Parameter`. `name` holds the user-visible
    /// symbol name (e.g. `"θ"`).
    Parameter { name: String },
    /// `b'v'` — one element of a `ParameterVector`. The wire format
    /// is `PARAMETER_VECTOR_ELEMENT` (`!HQ16sQ` + UTF-8 vector name);
    /// each element has its own UUID. `index` is the element's
    /// position in the parent vector and `vector_size` is the
    /// declared length.
    VectorElement {
        vector_name: String,
        vector_size: u64,
        index: u64,
    },
}

/// One entry in the symbol map that follows the expr-bytes block.
/// `value_bytes` is the per-symbol value payload — empty for symbol
/// types that don't carry a value, populated for nested
/// ParameterExpression values etc.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamExprSymbol {
    /// Discriminator: `Parameter` for `b'p'`, `VectorElement` for
    /// `b'v'`.
    pub kind: ParamExprSymbolKind,
    /// Type byte of the value — same key set as
    /// [`ParamExprOperand`].
    pub value_type: u8,
    /// 16-byte UUID identifying the symbol that the rest of the
    /// expr stream references. For `VectorElement` this is the
    /// *element* UUID (related to the parent vector's root by
    /// `element.uuid = root_uuid + index`).
    pub uuid: [u8; 16],
    /// Raw value-payload bytes that followed the symbol struct, of
    /// length `PARAM_EXPR_MAP_ELEM_V3.size`. Preserved verbatim so a
    /// future decoder can interpret nested ParameterExpression
    /// value payloads.
    pub value_bytes: Vec<u8>,
}

/// Top-level decoded ParameterExpression: the symbols it references
/// plus the stack-machine program that builds the expression value.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedParameterExpression {
    pub symbols: Vec<ParamExprSymbol>,
    pub elements: Vec<ParamExprElement>,
}

/// Recognised op-code byte values. Mirrors
/// `qiskit.circuit.parameterexpression.OpCode` (Qiskit 2.4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamExprOpCode {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Pow = 4,
    Sin = 5,
    Cos = 6,
    Tan = 7,
    Asin = 8,
    Acos = 9,
    Exp = 10,
    Log = 11,
    Grad = 13,
    Conj = 14,
    Substitute = 15,
    Abs = 16,
    Atan = 17,
    Rsub = 18,
    Rdiv = 19,
    Rpow = 20,
    Sign = 21,
    /// Wire sentinel — Qiskit's interpreter skips the apply step when
    /// `op_code == 255`, treating the record as pure operand pushes.
    NoOp = 255,
}

impl ParamExprOpCode {
    pub fn from_u8(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::Add,
            1 => Self::Sub,
            2 => Self::Mul,
            3 => Self::Div,
            4 => Self::Pow,
            5 => Self::Sin,
            6 => Self::Cos,
            7 => Self::Tan,
            8 => Self::Asin,
            9 => Self::Acos,
            10 => Self::Exp,
            11 => Self::Log,
            13 => Self::Grad,
            14 => Self::Conj,
            15 => Self::Substitute,
            16 => Self::Abs,
            17 => Self::Atan,
            18 => Self::Rsub,
            19 => Self::Rdiv,
            20 => Self::Rpow,
            21 => Self::Sign,
            255 => Self::NoOp,
            _ => return None,
        })
    }
}

const PARAM_EXPR_HEADER_LEN: usize = 16;
const PARAM_EXPR_ELEM_LEN: usize = 35;
const PARAM_EXPR_MAP_ELEM_LEN: usize = 10;
const PARAMETER_STRUCT_FIXED_LEN: usize = 2 + 16; // !H16s

/// Parse a `b'e'` INSTRUCTION_PARAM payload into a structured
/// expression program + symbol map. Does not interpret the program
/// — see `builder.rs::param_to_expr` for the omega-IR fold.
pub fn read_parameter_expression(payload: &[u8]) -> Result<DecodedParameterExpression, QpyError> {
    if payload.len() < PARAM_EXPR_HEADER_LEN {
        return Err(QpyError::Truncated {
            what: "parameter_expression_header",
            need: PARAM_EXPR_HEADER_LEN,
            got: payload.len(),
        });
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&payload[0..8]);
    let map_elements = u64::from_be_bytes(buf) as usize;
    buf.copy_from_slice(&payload[8..16]);
    let expr_size = u64::from_be_bytes(buf) as usize;

    let expr_end = PARAM_EXPR_HEADER_LEN
        .checked_add(expr_size)
        .ok_or(QpyError::Truncated {
            what: "parameter_expression_expr_bytes",
            need: usize::MAX,
            got: payload.len(),
        })?;
    if payload.len() < expr_end {
        return Err(QpyError::Truncated {
            what: "parameter_expression_expr_bytes",
            need: expr_end,
            got: payload.len(),
        });
    }
    if !expr_size.is_multiple_of(PARAM_EXPR_ELEM_LEN) {
        return Err(QpyError::Unsupported {
            what: "parameter_expression expr_size not a multiple of 35",
            detail: "follow-on substitution / nested PE blocks land in a follow-up commit",
        });
    }

    let mut elements = Vec::with_capacity(expr_size / PARAM_EXPR_ELEM_LEN);
    let mut elem_off = PARAM_EXPR_HEADER_LEN;
    while elem_off < expr_end {
        let op_code = payload[elem_off];
        let lhs_type = payload[elem_off + 1];
        let mut lhs_bytes = [0u8; 16];
        lhs_bytes.copy_from_slice(&payload[elem_off + 2..elem_off + 18]);
        let rhs_type = payload[elem_off + 18];
        let mut rhs_bytes = [0u8; 16];
        rhs_bytes.copy_from_slice(&payload[elem_off + 19..elem_off + 35]);
        let lhs = ParamExprOperand::decode(lhs_type, &lhs_bytes)?;
        let rhs = ParamExprOperand::decode(rhs_type, &rhs_bytes)?;
        elements.push(ParamExprElement { op_code, lhs, rhs });
        elem_off += PARAM_EXPR_ELEM_LEN;
    }

    let mut symbols = Vec::with_capacity(map_elements);
    let mut cursor = expr_end;
    for _ in 0..map_elements {
        if payload.len() < cursor + PARAM_EXPR_MAP_ELEM_LEN {
            return Err(QpyError::Truncated {
                what: "parameter_expression_map_elem_header",
                need: cursor + PARAM_EXPR_MAP_ELEM_LEN,
                got: payload.len(),
            });
        }
        let symbol_type = payload[cursor];
        let value_type = payload[cursor + 1];
        let mut size_buf = [0u8; 8];
        size_buf.copy_from_slice(&payload[cursor + 2..cursor + 10]);
        let value_size = u64::from_be_bytes(size_buf) as usize;
        cursor += PARAM_EXPR_MAP_ELEM_LEN;

        let (uuid, kind, consumed) = match symbol_type {
            b'p' => {
                let (uuid, name, consumed) = decode_parameter_struct(payload, cursor)?;
                (uuid, ParamExprSymbolKind::Parameter { name }, consumed)
            }
            b'v' => decode_parameter_vector_element_struct(payload, cursor)?,
            other => {
                return Err(QpyError::UnknownParamExprSymbolKind { byte: other });
            }
        };
        cursor += consumed;

        if payload.len() < cursor + value_size {
            return Err(QpyError::Truncated {
                what: "parameter_expression_map_elem_value",
                need: cursor + value_size,
                got: payload.len(),
            });
        }
        let value_bytes = payload[cursor..cursor + value_size].to_vec();
        cursor += value_size;
        symbols.push(ParamExprSymbol {
            kind,
            value_type,
            uuid,
            value_bytes,
        });
    }

    if cursor != payload.len() {
        return Err(QpyError::Unsupported {
            what: "parameter_expression payload has trailing bytes",
            detail: "decoder consumed less than the producer wrote — possible follow-up structure",
        });
    }

    Ok(DecodedParameterExpression { symbols, elements })
}

/// Decode the `PARAMETER` struct (`!H16s` + UTF-8 name) starting at
/// `offset` in `bytes`. Returns `(uuid, name, consumed_bytes)`.
fn decode_parameter_struct(
    bytes: &[u8],
    offset: usize,
) -> Result<([u8; 16], String, usize), QpyError> {
    if bytes.len() < offset + PARAMETER_STRUCT_FIXED_LEN {
        return Err(QpyError::Truncated {
            what: "parameter_expression_symbol_struct",
            need: offset + PARAMETER_STRUCT_FIXED_LEN,
            got: bytes.len(),
        });
    }
    let name_size = u16::from_be_bytes([bytes[offset], bytes[offset + 1]]) as usize;
    let mut uuid = [0u8; 16];
    uuid.copy_from_slice(&bytes[offset + 2..offset + 18]);
    let name_end = offset
        .checked_add(PARAMETER_STRUCT_FIXED_LEN + name_size)
        .ok_or(QpyError::Truncated {
            what: "parameter_expression_symbol_name",
            need: usize::MAX,
            got: bytes.len(),
        })?;
    if bytes.len() < name_end {
        return Err(QpyError::Truncated {
            what: "parameter_expression_symbol_name",
            need: name_end,
            got: bytes.len(),
        });
    }
    let name = std::str::from_utf8(&bytes[offset + PARAMETER_STRUCT_FIXED_LEN..name_end])
        .map_err(|e| QpyError::InvalidUtf8 {
            what: "parameter_expression_symbol_name",
            valid_up_to: e.valid_up_to(),
            len: name_size,
        })?
        .to_owned();
    Ok((uuid, name, PARAMETER_STRUCT_FIXED_LEN + name_size))
}

const PARAMETER_VECTOR_ELEMENT_FIXED_LEN: usize = 2 + 8 + 16 + 8;

/// Decode the `PARAMETER_VECTOR_ELEMENT` struct (`!HQ16sQ` + UTF-8
/// vector name) starting at `offset` in `bytes`. Returns
/// `(element_uuid, ParamExprSymbolKind::VectorElement, consumed_bytes)`.
fn decode_parameter_vector_element_struct(
    bytes: &[u8],
    offset: usize,
) -> Result<([u8; 16], ParamExprSymbolKind, usize), QpyError> {
    if bytes.len() < offset + PARAMETER_VECTOR_ELEMENT_FIXED_LEN {
        return Err(QpyError::Truncated {
            what: "parameter_expression_vector_element_struct",
            need: offset + PARAMETER_VECTOR_ELEMENT_FIXED_LEN,
            got: bytes.len(),
        });
    }
    let vector_name_size = u16::from_be_bytes([bytes[offset], bytes[offset + 1]]) as usize;
    let mut sz_buf = [0u8; 8];
    sz_buf.copy_from_slice(&bytes[offset + 2..offset + 10]);
    let vector_size = u64::from_be_bytes(sz_buf);
    let mut uuid = [0u8; 16];
    uuid.copy_from_slice(&bytes[offset + 10..offset + 26]);
    let mut idx_buf = [0u8; 8];
    idx_buf.copy_from_slice(&bytes[offset + 26..offset + 34]);
    let index = u64::from_be_bytes(idx_buf);

    let name_end = offset
        .checked_add(PARAMETER_VECTOR_ELEMENT_FIXED_LEN + vector_name_size)
        .ok_or(QpyError::Truncated {
            what: "parameter_expression_vector_element_name",
            need: usize::MAX,
            got: bytes.len(),
        })?;
    if bytes.len() < name_end {
        return Err(QpyError::Truncated {
            what: "parameter_expression_vector_element_name",
            need: name_end,
            got: bytes.len(),
        });
    }
    let vector_name =
        std::str::from_utf8(&bytes[offset + PARAMETER_VECTOR_ELEMENT_FIXED_LEN..name_end])
            .map_err(|e| QpyError::InvalidUtf8 {
                what: "parameter_expression_vector_element_name",
                valid_up_to: e.valid_up_to(),
                len: vector_name_size,
            })?
            .to_owned();
    Ok((
        uuid,
        ParamExprSymbolKind::VectorElement {
            vector_name,
            vector_size,
            index,
        },
        PARAMETER_VECTOR_ELEMENT_FIXED_LEN + vector_name_size,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_elem(op: u8, lhs_t: u8, lhs: [u8; 16], rhs_t: u8, rhs: [u8; 16]) -> [u8; 35] {
        let mut out = [0u8; 35];
        out[0] = op;
        out[1] = lhs_t;
        out[2..18].copy_from_slice(&lhs);
        out[18] = rhs_t;
        out[19..35].copy_from_slice(&rhs);
        out
    }

    fn float_operand_bytes(f: f64) -> [u8; 16] {
        let mut b = [0u8; 16];
        b[8..16].copy_from_slice(&f.to_be_bytes());
        b
    }

    fn int_operand_bytes(i: i64) -> [u8; 16] {
        let mut b = [0u8; 16];
        b[8..16].copy_from_slice(&i.to_be_bytes());
        b
    }

    /// Test fixture row for a `b'p'` symbol: `(value_type, uuid,
    /// name, value_bytes)`. Tucked behind a type alias to keep
    /// clippy's type-complexity check happy.
    type SymbolFixture<'a> = (u8, [u8; 16], &'a str, &'a [u8]);

    fn build_payload(elements: &[[u8; 35]], symbols: &[SymbolFixture<'_>]) -> Vec<u8> {
        let mut out = Vec::new();
        let map_elements = symbols.len() as u64;
        let expr_size = (elements.len() * 35) as u64;
        out.extend_from_slice(&map_elements.to_be_bytes());
        out.extend_from_slice(&expr_size.to_be_bytes());
        for e in elements {
            out.extend_from_slice(e);
        }
        for (val_t, uuid, name, value) in symbols {
            out.push(b'p');
            out.push(*val_t);
            out.extend_from_slice(&(value.len() as u64).to_be_bytes());
            // PARAMETER struct: u16 name_size + 16-byte uuid + name
            let name_bytes = name.as_bytes();
            out.extend_from_slice(&(name_bytes.len() as u16).to_be_bytes());
            out.extend_from_slice(uuid);
            out.extend_from_slice(name_bytes);
            out.extend_from_slice(value);
        }
        out
    }

    /// Test fixture row for a `b'v'` symbol: `(value_type, uuid,
    /// vector_name, vector_size, index, value_bytes)`.
    type VectorSymbolFixture<'a> = (u8, [u8; 16], &'a str, u64, u64, &'a [u8]);

    fn build_payload_with_vec_symbols(
        elements: &[[u8; 35]],
        symbols: &[VectorSymbolFixture<'_>],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(symbols.len() as u64).to_be_bytes());
        out.extend_from_slice(&((elements.len() * 35) as u64).to_be_bytes());
        for e in elements {
            out.extend_from_slice(e);
        }
        for (val_t, uuid, vector_name, vector_size, index, value) in symbols {
            out.push(b'v');
            out.push(*val_t);
            out.extend_from_slice(&(value.len() as u64).to_be_bytes());
            // PARAMETER_VECTOR_ELEMENT struct: !HQ16sQ + name.
            let name_bytes = vector_name.as_bytes();
            out.extend_from_slice(&(name_bytes.len() as u16).to_be_bytes());
            out.extend_from_slice(&vector_size.to_be_bytes());
            out.extend_from_slice(uuid);
            out.extend_from_slice(&index.to_be_bytes());
            out.extend_from_slice(name_bytes);
            out.extend_from_slice(value);
        }
        out
    }

    #[test]
    fn decodes_a_minimal_add_float_parameter() {
        let uuid = [0xAAu8; 16];
        let elems = [pack_elem(
            0, // ADD
            b'f',
            float_operand_bytes(1.5),
            b'p',
            uuid,
        )];
        let payload = build_payload(&elems, &[(b'p', uuid, "θ", &[])]);
        let pe = read_parameter_expression(&payload).unwrap();
        assert_eq!(pe.symbols.len(), 1);
        match &pe.symbols[0].kind {
            ParamExprSymbolKind::Parameter { name } => assert_eq!(name, "θ"),
            other => panic!("expected Parameter, got {other:?}"),
        }
        assert_eq!(pe.symbols[0].uuid, uuid);
        assert!(pe.symbols[0].value_bytes.is_empty());
        assert_eq!(pe.elements.len(), 1);
        assert_eq!(pe.elements[0].op_code, 0);
        assert_eq!(pe.elements[0].lhs, ParamExprOperand::Float(1.5));
        assert_eq!(pe.elements[0].rhs, ParamExprOperand::ParameterUuid(uuid));
    }

    #[test]
    fn decodes_an_int_operand() {
        let elems = [pack_elem(
            2, // MUL
            b'i',
            int_operand_bytes(-7),
            b'n',
            [0u8; 16],
        )];
        let payload = build_payload(&elems, &[]);
        let pe = read_parameter_expression(&payload).unwrap();
        assert_eq!(pe.symbols.len(), 0);
        assert_eq!(pe.elements[0].lhs, ParamExprOperand::Int(-7));
        assert_eq!(pe.elements[0].rhs, ParamExprOperand::None);
    }

    #[test]
    fn decodes_a_complex_operand() {
        let mut lhs_b = [0u8; 16];
        lhs_b[0..8].copy_from_slice(&1.5_f64.to_be_bytes());
        lhs_b[8..16].copy_from_slice(&(-2.25_f64).to_be_bytes());
        let elems = [pack_elem(0, b'c', lhs_b, b'n', [0u8; 16])];
        let payload = build_payload(&elems, &[]);
        let pe = read_parameter_expression(&payload).unwrap();
        assert_eq!(
            pe.elements[0].lhs,
            ParamExprOperand::Complex { re: 1.5, im: -2.25 }
        );
    }

    #[test]
    fn rejects_unknown_operand_type_byte() {
        let elems = [pack_elem(0, b'X', [0u8; 16], b'n', [0u8; 16])];
        let payload = build_payload(&elems, &[]);
        match read_parameter_expression(&payload).unwrap_err() {
            QpyError::UnknownParamExprOperandKind { byte } => assert_eq!(byte, b'X'),
            other => panic!("expected UnknownParamExprOperandKind, got {other:?}"),
        }
    }

    #[test]
    fn rejects_truncated_header() {
        let payload = vec![0u8; 8];
        match read_parameter_expression(&payload).unwrap_err() {
            QpyError::Truncated { what, .. } => {
                assert_eq!(what, "parameter_expression_header");
            }
            other => panic!("expected Truncated header, got {other:?}"),
        }
    }

    #[test]
    fn rejects_expr_size_that_isnt_a_multiple_of_35() {
        // Two elements (70 bytes) but claim 71 — non-multiple.
        let mut payload = vec![];
        payload.extend_from_slice(&0u64.to_be_bytes()); // map_elements=0
        payload.extend_from_slice(&71u64.to_be_bytes()); // expr_size=71
        payload.extend_from_slice(&[0u8; 71]);
        match read_parameter_expression(&payload).unwrap_err() {
            QpyError::Unsupported { what, .. } => {
                assert!(what.contains("not a multiple of 35"));
            }
            other => panic!("expected Unsupported(non-multiple), got {other:?}"),
        }
    }

    #[test]
    fn rejects_truncated_map_elem_header() {
        let mut payload = vec![];
        payload.extend_from_slice(&1u64.to_be_bytes()); // map_elements=1
        payload.extend_from_slice(&0u64.to_be_bytes()); // expr_size=0
        payload.extend_from_slice(&[0u8; 5]); // 5 bytes of map elem (need 10)
        match read_parameter_expression(&payload).unwrap_err() {
            QpyError::Truncated { what, .. } => {
                assert_eq!(what, "parameter_expression_map_elem_header");
            }
            other => panic!("expected Truncated map elem, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_symbol_kind() {
        let mut payload = vec![];
        payload.extend_from_slice(&1u64.to_be_bytes()); // 1 symbol
        payload.extend_from_slice(&0u64.to_be_bytes()); // 0 expr_size
        payload.push(b'X'); // unknown symbol_type
        payload.push(b'p');
        payload.extend_from_slice(&0u64.to_be_bytes());
        match read_parameter_expression(&payload).unwrap_err() {
            QpyError::UnknownParamExprSymbolKind { byte } => assert_eq!(byte, b'X'),
            other => panic!("expected UnknownParamExprSymbolKind, got {other:?}"),
        }
    }

    #[test]
    fn decodes_a_parameter_vector_symbol() {
        // ADD(2.0, pv[1]) — the LHS is a literal, the RHS references a
        // ParameterVector element by element-uuid. The symbol map's
        // single entry is a `b'v'` PARAMETER_VECTOR_ELEMENT struct.
        let elem_uuid = [0x77u8; 16];
        let elems = [pack_elem(
            0,
            b'f',
            float_operand_bytes(2.0),
            b'p',
            elem_uuid,
        )];
        let payload =
            build_payload_with_vec_symbols(&elems, &[(b'p', elem_uuid, "gamma", 4, 1, &[])]);
        let pe = read_parameter_expression(&payload).unwrap();
        assert_eq!(pe.symbols.len(), 1);
        match &pe.symbols[0].kind {
            ParamExprSymbolKind::VectorElement {
                vector_name,
                vector_size,
                index,
            } => {
                assert_eq!(vector_name, "gamma");
                assert_eq!(*vector_size, 4);
                assert_eq!(*index, 1);
            }
            other => panic!("expected VectorElement, got {other:?}"),
        }
        assert_eq!(pe.symbols[0].uuid, elem_uuid);
    }

    #[test]
    fn opcode_table_round_trips_via_from_u8() {
        for &(code, expected) in &[
            (0u8, ParamExprOpCode::Add),
            (1, ParamExprOpCode::Sub),
            (2, ParamExprOpCode::Mul),
            (3, ParamExprOpCode::Div),
            (4, ParamExprOpCode::Pow),
            (15, ParamExprOpCode::Substitute),
            (17, ParamExprOpCode::Atan),
            (18, ParamExprOpCode::Rsub),
            (255, ParamExprOpCode::NoOp),
        ] {
            assert_eq!(ParamExprOpCode::from_u8(code), Some(expected));
        }
        // 12 is the gap in qiskit's enum.
        assert_eq!(ParamExprOpCode::from_u8(12), None);
        // 200 is well past any defined op.
        assert_eq!(ParamExprOpCode::from_u8(200), None);
    }

    #[test]
    fn decodes_substitution_marker_kinds() {
        let elems = [pack_elem(255, b's', [0u8; 16], b'e', [0u8; 16])];
        let payload = build_payload(&elems, &[]);
        let pe = read_parameter_expression(&payload).unwrap();
        assert_eq!(
            pe.elements[0].lhs,
            ParamExprOperand::SubstitutionMarker { which: 's' }
        );
        assert_eq!(
            pe.elements[0].rhs,
            ParamExprOperand::SubstitutionMarker { which: 'e' }
        );
    }

    #[test]
    fn decodes_substitution_map_size_carrier() {
        let mut lhs_b = [0u8; 16];
        lhs_b[0..8].copy_from_slice(&42u64.to_be_bytes());
        let elems = [pack_elem(255, b'u', lhs_b, b'n', [0u8; 16])];
        let payload = build_payload(&elems, &[]);
        let pe = read_parameter_expression(&payload).unwrap();
        assert_eq!(
            pe.elements[0].lhs,
            ParamExprOperand::SubstitutionMapSize(42)
        );
    }
}
