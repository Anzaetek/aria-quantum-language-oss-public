//! QPY file-level header parser.
//!
//! Layout, after the 6-byte `b"QISKIT"` magic (network byte order):
//!
//! ```text
//! V1..=V9  (13 trailing bytes):
//!     u8  qpy_version
//!     u8  qiskit_major
//!     u8  qiskit_minor
//!     u8  qiskit_patch
//!     u64 num_circuits        // big-endian
//!
//! V10..   (14 trailing bytes — adds `symbolic_encoding`):
//!     u8  qpy_version
//!     u8  qiskit_major
//!     u8  qiskit_minor
//!     u8  qiskit_patch
//!     u64 num_circuits        // big-endian
//!     u8  symbolic_encoding   // b'p' = sympy, b'e' = symengine
//! ```
//!
//! V16+ adds a circuit-start offset table immediately after the
//! header — that lives in the per-circuit module, not here.

use thiserror::Error;

use super::MAX_SUPPORTED_VERSION;

/// Symbolic-parameter encoding indicator (QPY V10+ only).
///
/// Both encodings produce equivalent expression trees once decoded;
/// the byte just tells the reader which Python library serialised
/// the symbols. omega doesn't need either library — the per-circuit
/// parser will read whichever encoding the producer used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolicEncoding {
    /// `b'p'` — symbols serialised via the `sympy` library.
    Sympy,
    /// `b'e'` — symbols serialised via the `symengine` library.
    Symengine,
}

/// Parsed file-level header. Every field is what the producer wrote;
/// no normalisation. Use [`SymbolicEncoding`] to dispatch the
/// parameter decoder once that lands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QpyHeader {
    /// QPY format version (`u8`).
    pub qpy_version: u8,
    /// Qiskit version that produced the blob: `(major, minor, patch)`.
    pub qiskit_version: (u8, u8, u8),
    /// Number of circuits in the blob. Read from a big-endian `u64`.
    pub num_circuits: u64,
    /// `Some` when `qpy_version >= 10`; `None` for older blobs that
    /// pre-date the symbolic-encoding indicator.
    pub symbolic_encoding: Option<SymbolicEncoding>,
}

/// Errors surfaced by the QPY reader.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum QpyError {
    /// Input doesn't start with the `b"QISKIT"` magic.
    #[error("not a QPY blob: missing `QISKIT` magic bytes")]
    NotAQpyBlob,
    /// Input ends before the field at `offset` could be read.
    #[error("truncated QPY blob: needed at least {need} bytes for {what}, got {got}")]
    Truncated {
        what: &'static str,
        need: usize,
        got: usize,
    },
    /// QPY version is newer than the reader knows about. Caller can
    /// fall back to the Qiskit-subprocess path.
    #[error(
        "unsupported QPY version {version}; reader supports up to v{supported_max} \
         (route through `qpy_to_qasm2` instead, or update the omega QPY reader)"
    )]
    UnsupportedVersion { version: u8, supported_max: u8 },
    /// QPY version is 0 — sentinel never used by Qiskit, almost
    /// certainly a corrupt blob.
    #[error("QPY version 0 is not a valid version (corrupt blob?)")]
    InvalidVersionZero,
    /// `symbolic_encoding` byte (V10+) was something other than
    /// `b'p'` or `b'e'`.
    #[error(
        "unknown QPY symbolic encoding byte 0x{byte:02x}; expected b'p' (sympy) or b'e' (symengine)"
    )]
    UnknownSymbolicEncoding { byte: u8 },
    /// Program type key byte was a recognised legacy value (e.g.
    /// `b's'` for the deprecated Schedule program type) that the
    /// reader explicitly does not support.
    #[error(
        "unsupported QPY program type 0x{byte:02x} ({kind}); only QuantumCircuit (b'q') is read today"
    )]
    UnsupportedProgramType { byte: u8, kind: &'static str },
    /// Program type key byte was something the reader doesn't know
    /// about at all (potentially a future Qiskit addition).
    #[error("unknown QPY program type byte 0x{byte:02x}; expected b'q' (CIRCUIT)")]
    UnknownProgramType { byte: u8 },
    /// A length-prefixed UTF-8 field (circuit name, metadata, …)
    /// contained bytes that are not valid UTF-8.
    #[error("{what} is not valid UTF-8 (valid up to byte {valid_up_to} of {len})")]
    InvalidUtf8 {
        what: &'static str,
        valid_up_to: usize,
        len: usize,
    },
    /// A `global_phase_size` value disagreed with the type byte's
    /// implicit size (e.g. type=`'f'` with size != 8, or type=`'n'`
    /// with size > 0). Indicates a corrupt or producer-bug blob.
    #[error(
        "inconsistent global_phase_size for type 0x{type_byte:02x}: expected {expected}, got {got}"
    )]
    InconsistentGlobalPhaseSize {
        type_byte: u8,
        expected: usize,
        got: usize,
    },
    /// Register `type` byte was something other than `b'q'` or `b'c'`.
    #[error("unknown QPY register kind byte 0x{byte:02x}; expected b'q' or b'c'")]
    UnknownRegisterKind { byte: u8 },
    /// A wire-encoded boolean field (register.standalone /
    /// register.in_circuit / …) was something other than 0 or 1.
    #[error("{what} encoded as 0x{byte:02x}; expected 0 (false) or 1 (true)")]
    InvalidBoolByte { what: &'static str, byte: u8 },
    /// Low 2 bits of `extras_key` decoded to the reserved value 3.
    #[error("unknown condition kind {byte} (low 2 bits of extras_key); valid: 0 / 1 / 2")]
    UnknownConditionKind { byte: u8 },
    /// The reader encountered a structure it doesn't yet decode
    /// (e.g. a non-zero count of annotation namespaces). Caller can
    /// fall back to the Qiskit subprocess (`qpy_to_qasm2`) until
    /// the in-Rust decoder catches up.
    #[error("unsupported QPY feature: {what} ({detail})")]
    Unsupported {
        what: &'static str,
        detail: &'static str,
    },
    /// `INSTRUCTION_ARG.type` byte was something other than `b'q'`
    /// or `b'c'`.
    #[error("unknown INSTRUCTION_ARG type byte 0x{byte:02x}; expected b'q' or b'c'")]
    UnknownInstructionArgKind { byte: u8 },
    /// A `c` arg appeared where a `q` was expected (or vice versa).
    /// The QPY format mandates qargs first, then cargs, in the
    /// counts the instruction header reported.
    #[error("INSTRUCTION_ARG kind mismatch at position {position}: expected '{expected_kind}'")]
    InstructionArgKindMismatch {
        expected_kind: char,
        position: usize,
    },
    /// An INSTRUCTION_PARAM payload size disagreed with the type
    /// byte's implicit size (e.g. type=`'f'` with size != 8).
    #[error(
        "inconsistent INSTRUCTION_PARAM payload size for type 0x{type_byte:02x}: expected {expected}, got {got}"
    )]
    InconsistentParamSize {
        type_byte: u8,
        expected: usize,
        got: usize,
    },
    /// QPY decoded an instruction whose Qiskit class name doesn't
    /// map to one of omega's `GateKind` variants. Caller can fall
    /// back to the `qpy_to_qasm2` subprocess (which handles every
    /// Qiskit-defined gate via `qasm2.dumps`).
    #[error(
        "unsupported QPY gate: {name} has no native omega GateKind mapping (route through qpy_to_qasm2)"
    )]
    UnsupportedGate { name: String },
    /// A `PARAM_EXPR_ELEM_V13` operand type byte was something the
    /// reader doesn't know about. Valid bytes:
    /// `b'p'` / `b'f'` / `b'i'` / `b'c'` / `b'n'` / `b's'` / `b'e'`
    /// / `b'u'`.
    #[error("unknown PARAM_EXPR operand type byte 0x{byte:02x}")]
    UnknownParamExprOperandKind { byte: u8 },
    /// A `PARAM_EXPR_MAP_ELEM_V3.symbol_type` byte was something the
    /// reader doesn't know about. Valid bytes: `b'p'` (Parameter),
    /// `b'v'` (ParameterVectorElement; not yet decoded).
    #[error("unknown PARAM_EXPR symbol type byte 0x{byte:02x}")]
    UnknownParamExprSymbolKind { byte: u8 },
}

const MAGIC: &[u8] = b"QISKIT";
const MAGIC_LEN: usize = 6;
// MAGIC (6) + qpy_version (1) + qiskit_{major,minor,patch} (3) + num_circuits (8)
const PRE_V10_TOTAL_LEN: usize = MAGIC_LEN + 1 + 3 + 8; // 18
const V10_PLUS_TOTAL_LEN: usize = PRE_V10_TOTAL_LEN + 1; // 19, adds symbolic_encoding

/// Parse the QPY file-level header. Returns the parsed [`QpyHeader`]
/// on success; the rest of the blob (the per-circuit data) starts at
/// byte `header_len` — see [`QpyHeader::header_len`] on the returned
/// value.
///
/// Does not allocate. Does not consume the body — caller is free to
/// hand the same `bytes` slice to a future per-circuit reader.
pub fn read_header(bytes: &[u8]) -> Result<QpyHeader, QpyError> {
    if bytes.len() < MAGIC_LEN || &bytes[..MAGIC_LEN] != MAGIC {
        return Err(QpyError::NotAQpyBlob);
    }

    // Read the QPY version byte first so we can decide whether the
    // symbolic_encoding byte is part of the header.
    if bytes.len() < MAGIC_LEN + 1 {
        return Err(QpyError::Truncated {
            what: "qpy_version",
            need: MAGIC_LEN + 1,
            got: bytes.len(),
        });
    }
    let qpy_version = bytes[MAGIC_LEN];
    if qpy_version == 0 {
        return Err(QpyError::InvalidVersionZero);
    }
    if qpy_version > MAX_SUPPORTED_VERSION {
        return Err(QpyError::UnsupportedVersion {
            version: qpy_version,
            supported_max: MAX_SUPPORTED_VERSION,
        });
    }

    let needs_symbolic = qpy_version >= 10;
    let total_len = if needs_symbolic {
        V10_PLUS_TOTAL_LEN
    } else {
        PRE_V10_TOTAL_LEN
    };
    if bytes.len() < total_len {
        return Err(QpyError::Truncated {
            what: "file header",
            need: total_len,
            got: bytes.len(),
        });
    }

    let qiskit_major = bytes[MAGIC_LEN + 1];
    let qiskit_minor = bytes[MAGIC_LEN + 2];
    let qiskit_patch = bytes[MAGIC_LEN + 3];

    // num_circuits is u64 big-endian at offset MAGIC_LEN + 4.
    let mut nc = [0u8; 8];
    nc.copy_from_slice(&bytes[MAGIC_LEN + 4..MAGIC_LEN + 12]);
    let num_circuits = u64::from_be_bytes(nc);

    let symbolic_encoding = if needs_symbolic {
        let byte = bytes[MAGIC_LEN + 12];
        Some(match byte {
            b'p' => SymbolicEncoding::Sympy,
            b'e' => SymbolicEncoding::Symengine,
            other => return Err(QpyError::UnknownSymbolicEncoding { byte: other }),
        })
    } else {
        None
    };

    Ok(QpyHeader {
        qpy_version,
        qiskit_version: (qiskit_major, qiskit_minor, qiskit_patch),
        num_circuits,
        symbolic_encoding,
    })
}

impl QpyHeader {
    /// Total byte length of the file-level header in the blob this
    /// header was parsed from. The per-circuit data starts at this
    /// offset.
    pub fn header_len(&self) -> usize {
        if self.symbolic_encoding.is_some() {
            V10_PLUS_TOTAL_LEN
        } else {
            PRE_V10_TOTAL_LEN
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic V10+ header for tests. `extra` lets a test
    /// append per-circuit bytes that are ignored by the header reader
    /// but show up in `header_len()` arithmetic.
    fn make_v10_header(
        qpy_version: u8,
        qiskit: (u8, u8, u8),
        num_circuits: u64,
        sym: u8,
        extra: &[u8],
    ) -> Vec<u8> {
        assert!(qpy_version >= 10);
        let mut out = Vec::with_capacity(V10_PLUS_TOTAL_LEN + extra.len());
        out.extend_from_slice(MAGIC);
        out.push(qpy_version);
        out.push(qiskit.0);
        out.push(qiskit.1);
        out.push(qiskit.2);
        out.extend_from_slice(&num_circuits.to_be_bytes());
        out.push(sym);
        out.extend_from_slice(extra);
        out
    }

    fn make_pre_v10_header(
        qpy_version: u8,
        qiskit: (u8, u8, u8),
        num_circuits: u64,
        extra: &[u8],
    ) -> Vec<u8> {
        assert!((1..10).contains(&qpy_version));
        let mut out = Vec::with_capacity(PRE_V10_TOTAL_LEN + extra.len());
        out.extend_from_slice(MAGIC);
        out.push(qpy_version);
        out.push(qiskit.0);
        out.push(qiskit.1);
        out.push(qiskit.2);
        out.extend_from_slice(&num_circuits.to_be_bytes());
        out.extend_from_slice(extra);
        out
    }

    #[test]
    fn reads_v17_header_with_sympy_encoding() {
        let blob = make_v10_header(17, (2, 4, 1), 1, b'p', &[0xAA, 0xBB]);
        let h = read_header(&blob).unwrap();
        assert_eq!(h.qpy_version, 17);
        assert_eq!(h.qiskit_version, (2, 4, 1));
        assert_eq!(h.num_circuits, 1);
        assert_eq!(h.symbolic_encoding, Some(SymbolicEncoding::Sympy));
        assert_eq!(h.header_len(), V10_PLUS_TOTAL_LEN);
    }

    #[test]
    fn reads_v10_header_with_symengine_encoding() {
        let blob = make_v10_header(10, (1, 0, 0), 42, b'e', &[]);
        let h = read_header(&blob).unwrap();
        assert_eq!(h.qpy_version, 10);
        assert_eq!(h.qiskit_version, (1, 0, 0));
        assert_eq!(h.num_circuits, 42);
        assert_eq!(h.symbolic_encoding, Some(SymbolicEncoding::Symengine));
    }

    #[test]
    fn reads_pre_v10_header_without_symbolic_encoding() {
        // V9 was the last version without the symbolic_encoding byte.
        let blob = make_pre_v10_header(9, (0, 45, 3), 1, &[0x01, 0x02, 0x03]);
        let h = read_header(&blob).unwrap();
        assert_eq!(h.qpy_version, 9);
        assert_eq!(h.qiskit_version, (0, 45, 3));
        assert_eq!(h.num_circuits, 1);
        assert_eq!(h.symbolic_encoding, None);
        assert_eq!(h.header_len(), PRE_V10_TOTAL_LEN);
    }

    #[test]
    fn handles_multi_circuit_count_correctly() {
        // num_circuits is read as big-endian u64. Pick a value with
        // bytes in the high half so a wrong-endian read would flip
        // it to something obviously different.
        let nc = 0x0123_4567_89ab_cdef_u64;
        let blob = make_v10_header(15, (1, 1, 0), nc, b'e', &[]);
        assert_eq!(read_header(&blob).unwrap().num_circuits, nc);
    }

    #[test]
    fn missing_magic_is_typed_error() {
        let err = read_header(b"NOTQPY").unwrap_err();
        assert_eq!(err, QpyError::NotAQpyBlob);
    }

    #[test]
    fn empty_input_is_typed_error() {
        let err = read_header(b"").unwrap_err();
        assert_eq!(err, QpyError::NotAQpyBlob);
    }

    #[test]
    fn short_input_with_partial_magic_is_not_a_qpy_blob() {
        // 5 bytes, all matching, but missing the final 'T'. Doesn't
        // satisfy the magic check.
        let err = read_header(b"QISKI").unwrap_err();
        assert_eq!(err, QpyError::NotAQpyBlob);
    }

    #[test]
    fn truncated_after_magic_reports_qpy_version_truncation() {
        let err = read_header(MAGIC).unwrap_err();
        assert!(
            matches!(
                err,
                QpyError::Truncated {
                    what: "qpy_version",
                    ..
                }
            ),
            "unexpected err: {err:?}"
        );
    }

    #[test]
    fn truncated_v10_header_reports_file_header_truncation() {
        // Magic + qpy_version=10 + only 2 of the 4 qiskit bytes.
        let mut blob = Vec::new();
        blob.extend_from_slice(MAGIC);
        blob.push(10);
        blob.extend_from_slice(&[1, 2]);
        let err = read_header(&blob).unwrap_err();
        match err {
            QpyError::Truncated { what, need, got } => {
                assert_eq!(what, "file header");
                assert_eq!(need, V10_PLUS_TOTAL_LEN);
                assert_eq!(got, 9);
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn truncated_pre_v10_header_uses_pre_v10_length() {
        // Magic + qpy_version=5 + qiskit_major only — needs to know
        // we don't ask for the V10 symbolic byte.
        let mut blob = Vec::new();
        blob.extend_from_slice(MAGIC);
        blob.push(5);
        blob.push(0);
        let err = read_header(&blob).unwrap_err();
        match err {
            QpyError::Truncated { what, need, got } => {
                assert_eq!(what, "file header");
                assert_eq!(need, PRE_V10_TOTAL_LEN);
                assert_eq!(got, 8);
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn version_zero_rejected_explicitly() {
        let mut blob = Vec::new();
        blob.extend_from_slice(MAGIC);
        blob.push(0);
        // Pad enough so we don't get a Truncated error first.
        blob.extend_from_slice(&[0; 16]);
        assert_eq!(
            read_header(&blob).unwrap_err(),
            QpyError::InvalidVersionZero
        );
    }

    #[test]
    fn version_above_max_supported_returns_typed_error() {
        let unsupported = MAX_SUPPORTED_VERSION + 1;
        let mut blob = Vec::new();
        blob.extend_from_slice(MAGIC);
        blob.push(unsupported);
        blob.extend_from_slice(&[0; 16]);
        assert_eq!(
            read_header(&blob).unwrap_err(),
            QpyError::UnsupportedVersion {
                version: unsupported,
                supported_max: MAX_SUPPORTED_VERSION,
            }
        );
    }

    #[test]
    fn unknown_symbolic_encoding_byte_is_typed_error() {
        // V10+ header but symbolic_encoding byte is neither 'p' nor 'e'.
        let blob = make_v10_header(11, (1, 0, 0), 0, b'x', &[]);
        assert_eq!(
            read_header(&blob).unwrap_err(),
            QpyError::UnknownSymbolicEncoding { byte: b'x' },
        );
    }

    #[test]
    fn header_does_not_consume_per_circuit_body() {
        // Body bytes after the header should be ignored — header
        // parser cares only about the first `header_len()` bytes.
        let body = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x42];
        let blob = make_v10_header(12, (0, 46, 0), 1, b'e', &body);
        let h = read_header(&blob).unwrap();
        assert_eq!(h.header_len(), V10_PLUS_TOTAL_LEN);
        assert_eq!(&blob[h.header_len()..], &body[..]);
    }

    #[test]
    fn current_existing_magic_byte_fixture_is_truncated_not_invalid() {
        // The `is_qpy_magic_bytes_detection` test in lib.rs uses
        // `b"QISKIT\x0c\x00\x00\x00\x00"` — only 11 bytes, so
        // version 12 is recognised but the rest is short. This pins
        // the boundary so the header parser surfaces Truncated rather
        // than erroring elsewhere.
        let blob: &[u8] = b"QISKIT\x0c\x00\x00\x00\x00";
        let err = read_header(blob).unwrap_err();
        match err {
            QpyError::Truncated { what, need, got } => {
                assert_eq!(what, "file header");
                assert_eq!(need, V10_PLUS_TOTAL_LEN);
                assert_eq!(got, 11);
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
    }
}
