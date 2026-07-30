//! C ABI types for the backend plugin system.
//!
//! These types cross DLL boundaries and must remain ABI-stable.

use std::os::raw::c_char;

/// ABI version exported by every plugin via `omega_backend_abi_version()`.
///
/// The loader refuses to load a plugin whose reported version differs, so a
/// stale cdylib fails loudly instead of reading a mismatched vtable layout.
/// Bump this only for a layout-breaking change to the FFI types below;
/// additive growth is carved from [`BackendVTable::reserved`] and does **not**
/// bump the version.
pub const OMEGA_BACKEND_ABI_VERSION: u32 = 1;

/// Circuit type enum for C ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfiCircuitType {
    GateBased = 0,
    Photonic = 1,
}

/// A flattened gate operation for C ABI.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct FfiGateOp {
    /// Gate kind (see gate_kind constants below)
    pub gate_kind: u32,
    /// Number of qubits
    pub num_qubits: u32,
    /// Qubit indices (up to 3)
    pub qubits: [u32; 3],
    /// Number of resolved parameters
    pub num_params: u32,
    /// Resolved parameter values (up to 3)
    pub params: [f64; 3],
}

/// A flattened circuit for C ABI. All parameters must be resolved.
#[repr(C)]
pub struct FfiCircuit {
    pub num_qubits: u32,
    pub circuit_type: FfiCircuitType,
    pub num_ops: u32,
    pub ops: *const FfiGateOp,
}

/// Execution configuration for C ABI.
#[repr(C)]
pub struct FfiExecConfig {
    /// Number of shots. 0 = exact statevector/probabilities.
    pub shots: u32,
    /// Random seed. 0 = random.
    pub seed: u64,
}

/// Result type tag.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfiResultType {
    Counts = 0,
    Statevector = 1,
    Probabilities = 2,
}

/// Execution result for C ABI.
#[repr(C)]
pub struct FfiExecResult {
    pub result_type: FfiResultType,
    /// For Counts: pairs of (bitstring, count)
    pub num_entries: u32,
    pub bitstrings: *mut u64,
    pub counts: *mut u32,
    /// For Statevector: complex amplitudes (interleaved re, im)
    pub num_amplitudes: u32,
    pub amplitudes: *mut f64,
    /// For Probabilities
    pub num_probs: u32,
    pub probs: *mut f64,
}

/// Backend vtable: function pointers exported by each backend DLL.
#[repr(C)]
pub struct BackendVTable {
    /// Returns the backend name as a null-terminated string.
    pub name: extern "C" fn() -> *const c_char,

    /// Returns whether this backend supports the given circuit type.
    pub supports: extern "C" fn(circuit_type: FfiCircuitType) -> bool,

    /// Execute a circuit.
    /// Returns 0 on success, non-zero on error.
    pub execute: extern "C" fn(
        circuit: *const FfiCircuit,
        config: *const FfiExecConfig,
        result_out: *mut FfiExecResult,
    ) -> i32,

    /// Free a result allocated by execute.
    pub free_result: extern "C" fn(result: *mut FfiExecResult),

    /// Optional capability descriptor. `None` means the plugin declares no
    /// capabilities and the host treats it permissively. Carved from the
    /// reserved tail — an `Option<extern "C" fn ...>` is `usize`-sized and
    /// null-optimized, so the struct size and `OMEGA_BACKEND_ABI_VERSION`
    /// are unchanged and pre-caps plugins keep loading.
    pub caps: Option<extern "C" fn() -> BackendCaps>,

    /// Reserved for further additive evolution within this ABI version. A
    /// plugin must zero-initialize this tail.
    pub reserved: [usize; 7],
}

/// Backend capability kinds.
pub const CAPS_KIND_SIMULATOR: u32 = 0;
pub const CAPS_KIND_HARDWARE: u32 = 1;

/// Backend noise support levels.
pub const CAPS_NOISE_NONE: u32 = 0;
pub const CAPS_NOISE_MODEL: u32 = 1;
pub const CAPS_NOISE_PHYSICAL: u32 = 2;

/// Capability descriptor a plugin may expose via [`BackendVTable::caps`].
///
/// `#[repr(C)]` and ABI-stable. `native_gates` is a bitmask over the `GATE_*`
/// constants (bit `k` set ⇔ the plugin runs gate kind `k`); the host uses it,
/// with `max_qubits`, to refuse a circuit a plugin cannot handle **loudly**
/// rather than dispatch it and get a wrong answer. `engine_version` is an
/// optional null-terminated string (may be null).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BackendCaps {
    /// `size_of::<BackendCaps>()` as seen by the plugin — a forward-compat guard.
    pub struct_size: u32,
    /// Maximum qubit count the backend accepts. 0 means "unspecified/unbounded".
    pub max_qubits: u32,
    /// `CAPS_KIND_SIMULATOR` or `CAPS_KIND_HARDWARE`.
    pub kind: u32,
    /// Whether shot-based sampling is supported.
    pub supports_shots: bool,
    /// Whether an expectation fast path is supported.
    pub supports_expectation: bool,
    /// One of the `CAPS_NOISE_*` levels.
    pub noise: u32,
    /// Compute device (see `omega_core::device`); 0 = cpu.
    pub device: u32,
    /// Bitmask over the `GATE_*` constants of natively supported gate kinds.
    pub native_gates: u64,
    /// Whether the host may apply the trait `cpu_fallback` default for this
    /// plugin. Off by default: plugins error loudly instead of falling back.
    pub opt_in_cpu_fallback: bool,
    /// Optional engine version string (null-terminated); may be null.
    pub engine_version: *const c_char,
}

/// The bitmask bit for a `GATE_*` constant, for building/testing
/// [`BackendCaps::native_gates`]. All gate constants are < 64.
#[inline]
pub const fn gate_bit(gate_kind: u32) -> u64 {
    1u64 << gate_kind
}

// Gate kind constants (matching GateKind enum order)
pub const GATE_H: u32 = 0;
pub const GATE_X: u32 = 1;
pub const GATE_Y: u32 = 2;
pub const GATE_Z: u32 = 3;
pub const GATE_S: u32 = 4;
pub const GATE_SDG: u32 = 5;
pub const GATE_T: u32 = 6;
pub const GATE_TDG: u32 = 7;
pub const GATE_ID: u32 = 8;
pub const GATE_RX: u32 = 10;
pub const GATE_RY: u32 = 11;
pub const GATE_RZ: u32 = 12;
pub const GATE_U3: u32 = 13;
pub const GATE_U2: u32 = 14;
pub const GATE_U1: u32 = 15;
pub const GATE_CX: u32 = 20;
pub const GATE_CY: u32 = 21;
pub const GATE_CZ: u32 = 22;
pub const GATE_SWAP: u32 = 23;
pub const GATE_CRZ: u32 = 24;
pub const GATE_CU3: u32 = 25;
pub const GATE_RBS: u32 = 26;
pub const GATE_CCX: u32 = 30;
pub const GATE_CSWAP: u32 = 31;
pub const GATE_PS: u32 = 40;
pub const GATE_BS_RX: u32 = 41;
pub const GATE_MEASURE: u32 = 50;
pub const GATE_BARRIER: u32 = 51;
pub const GATE_RESET: u32 = 52;

impl FfiCircuitType {
    pub fn from_circuit_type(ct: &crate::circuit::CircuitType) -> Self {
        match ct {
            crate::circuit::CircuitType::GateBased => FfiCircuitType::GateBased,
            crate::circuit::CircuitType::Photonic => FfiCircuitType::Photonic,
        }
    }
}
