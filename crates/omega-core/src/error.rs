use thiserror::Error;

#[derive(Error, Debug)]
pub enum OmegaError {
    #[error("unbound symbol: {name} (id={id})")]
    UnboundSymbol { id: u32, name: String },

    /// A flat parameter vector had the wrong length for the circuit's free
    /// symbols. Refused rather than padded or truncated: a missing value
    /// bound to 0.0 yields a plausible wrong number, and a dropped extra
    /// hides a caller that disagrees with the circuit about its arity.
    #[error(
        "parameter count mismatch: got {got} value(s) but the circuit has {expected} free \
         parameter(s); unbound: [{}]. Pass exactly one value per symbol, in symbol-ID order \
         (missing values would otherwise bind to 0.0 and extras be dropped, silently)",
        .unbound.join(", ")
    )]
    ParameterCount {
        expected: usize,
        got: usize,
        /// Names of the symbols a short vector would have left unbound
        /// (empty when the vector was too long).
        unbound: Vec<String>,
    },

    #[error("parse error: {0}")]
    Parse(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid circuit: {0}")]
    InvalidCircuit(String),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    /// GPU device ran out of memory during allocation. Distinct from
    /// `Backend(...)` so callers (notably `QmlTrainer.fit`) can catch
    /// it and fall back to a CPU backend instead of propagating.
    /// GPU backends produce this when their underlying driver returns
    /// the OOM error code (`CUDA_ERROR_OUT_OF_MEMORY` on CUDA).
    #[error("GPU out of memory: {0}")]
    OutOfMemory(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, OmegaError>;
