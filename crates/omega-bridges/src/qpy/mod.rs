//! Pure-Rust reader for Qiskit's QPY binary serialisation format.
//!
//! Today: the file-level header (magic, QPY version, Qiskit version,
//! `num_circuits`, V10+ symbolic encoding). Per-circuit blocks —
//! registers, instructions, parameters, layouts, calibrations — land
//! in subsequent commits.
//!
//! The full reader's eventual goal is to replace the
//! `omega_bridges::qpy_to_qasm2` subprocess hop (which spawns Qiskit
//! to do `qpy.load + qasm2.dumps`) with a direct
//! `read_circuit(&[u8]) -> CircuitIR` decode — letting omega-cli
//! consume `.qpy` files on hosts without a Qiskit install.
//!
//! Format reference: <https://docs.quantum.ibm.com/api/qiskit/qpy>.
//! Network byte order (big-endian) throughout, except for
//! `INSTRUCTION_PARAM` integer/float encoding in QPY versions ≤ 17,
//! which is little-endian — that quirk only affects the per-circuit
//! parser, not the file header.

mod builder;
mod circuit_body;
mod circuit_extras;
mod circuit_header;
mod header;
mod instruction;
mod instruction_args;
mod instruction_header;
mod instruction_params;
mod parameter_expression;
mod program_table;
mod registers;
mod write;

pub use builder::{gate_name_to_kind, read_qpy_circuit_ir, read_qpy_circuit_irs};
pub use circuit_body::{read_circuit_body, CircuitBody, GlobalPhase};
pub use circuit_extras::{read_circuit_extras, CircuitExtras};
pub use circuit_header::{read_circuit_header, CircuitHeader};
pub use header::{read_header, QpyError, QpyHeader, SymbolicEncoding};
pub use instruction::{read_circuit_instructions, read_decoded_instruction, DecodedInstruction};
pub use instruction_args::{read_instruction_args, InstructionArg};
pub use instruction_header::{
    read_instruction_header, ConditionKind, ExtrasKey, InstructionHeader,
};
pub use instruction_params::{read_instruction_params, InstructionParam};
pub use parameter_expression::{
    read_parameter_expression, DecodedParameterExpression, ParamExprElement, ParamExprOpCode,
    ParamExprOperand, ParamExprSymbol,
};
pub use program_table::{read_program_table, ProgramTable, ProgramType};
pub use registers::{read_register_table, Register, RegisterKind, RegisterTable};
pub use write::{write_qpy_circuit_ir, WRITER_QISKIT_VERSION, WRITER_QPY_VERSION};

/// QPY format versions this reader claims to handle. Accepting up to
/// the latest documented version; rejecting anything newer with a
/// typed error so a caller using a freshly-released QPY blob gets a
/// clear "version too new" message rather than a silent misparse.
pub const MAX_SUPPORTED_VERSION: u8 = 17;

/// Magic-byte detection for Qiskit's QPY format. The header is
/// `b"QISKIT"` (6 bytes) followed by a `u8` QPY version. Cheap to
/// check before deciding whether to call [`read_header`].
pub fn is_qpy(bytes: &[u8]) -> bool {
    bytes.len() >= 6 && &bytes[..6] == b"QISKIT"
}
