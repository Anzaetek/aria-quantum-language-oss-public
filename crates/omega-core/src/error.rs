use thiserror::Error;

#[derive(Error, Debug)]
pub enum OmegaError {
    #[error("unbound symbol: {name} (id={id})")]
    UnboundSymbol { id: u32, name: String },

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
