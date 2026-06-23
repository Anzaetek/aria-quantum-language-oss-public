//! WASM runtime for hybrid classical-quantum loops.
//!
//! Uses wasmtime to run user-submitted optimization code (VQE, QAOA, etc.)
//! in a sandboxed WASM environment. The WASM guest calls host functions
//! to execute quantum circuits and compute gradients.

pub mod host;
pub mod runtime;

pub use runtime::WasmRunner;
