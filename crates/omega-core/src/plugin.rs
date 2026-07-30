//! Dynamic backend plugin loading via libloading.

use std::collections::HashMap;
use std::ffi::CStr;
use std::path::Path;

use crate::circuit::{CircuitIR, CircuitType, GateKind};
use crate::error::{OmegaError, Result};
use crate::executor::{ExecConfig, ExecResult};
use crate::ffi_types::*;
use crate::params::ParameterBinding;

/// A dynamically loaded backend plugin.
pub struct LoadedBackend {
    _lib: libloading::Library,
    vtable: BackendVTable,
    name: String,
}

impl LoadedBackend {
    /// Load a backend from a shared library path.
    ///
    /// The library must export `omega_backend_abi_version() -> u32` (checked
    /// first, and rejected loudly on mismatch) and
    /// `omega_backend_init() -> BackendVTable`.
    pub fn load(path: &Path) -> Result<Self> {
        unsafe {
            let lib = libloading::Library::new(path).map_err(|e| {
                OmegaError::Backend(format!("failed to load {}: {}", path.display(), e))
            })?;

            // ABI handshake first — a version mismatch means the vtable layout
            // may differ, so refuse before calling any other exported symbol.
            let abi_fn: libloading::Symbol<unsafe extern "C" fn() -> u32> =
                lib.get(b"omega_backend_abi_version").map_err(|e| {
                    OmegaError::Backend(format!(
                        "no omega_backend_abi_version in {} (unversioned or pre-ABI plugin): {}",
                        path.display(),
                        e
                    ))
                })?;
            let abi = abi_fn();
            if abi != OMEGA_BACKEND_ABI_VERSION {
                return Err(OmegaError::Backend(format!(
                    "plugin {} reports ABI version {} but this build requires {}",
                    path.display(),
                    abi,
                    OMEGA_BACKEND_ABI_VERSION
                )));
            }

            let init_fn: libloading::Symbol<unsafe extern "C" fn() -> BackendVTable> =
                lib.get(b"omega_backend_init").map_err(|e| {
                    OmegaError::Backend(format!(
                        "no omega_backend_init in {}: {}",
                        path.display(),
                        e
                    ))
                })?;

            let vtable = init_fn();
            let name_ptr = (vtable.name)();
            let name = CStr::from_ptr(name_ptr).to_string_lossy().into_owned();

            Ok(Self {
                _lib: lib,
                vtable,
                name,
            })
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn supports(&self, circuit_type: &CircuitType) -> bool {
        let ffi_type = FfiCircuitType::from_circuit_type(circuit_type);
        (self.vtable.supports)(ffi_type)
    }

    /// The plugin's capability descriptor, or `None` if it declares none.
    pub fn caps(&self) -> Option<BackendCaps> {
        self.vtable.caps.map(|f| f())
    }

    /// Check a circuit against the plugin's declared capabilities, returning a
    /// loud error naming the first unsupported feature. A plugin that declares
    /// no capabilities (`caps() == None`) is treated permissively — the check
    /// passes and any real limitation surfaces as an execute-time error.
    pub fn check_circuit_supported(&self, circuit: &CircuitIR) -> Result<()> {
        let Some(caps) = self.caps() else {
            return Ok(());
        };
        if caps.max_qubits > 0 && circuit.num_qubits > caps.max_qubits {
            return Err(OmegaError::Backend(format!(
                "plugin '{}' supports at most {} qubits, circuit has {}",
                self.name, caps.max_qubits, circuit.num_qubits
            )));
        }
        for op in &circuit.ops {
            // Custom gates have no FFI encoding; execute would reject them too.
            if let Ok(bit_index) = gate_kind_to_ffi(&op.gate) {
                if caps.native_gates & gate_bit(bit_index) == 0 {
                    return Err(OmegaError::Backend(format!(
                        "plugin '{}' does not declare support for gate {:?}",
                        self.name, op.gate
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> Result<ExecResult> {
        // Flatten circuit to FFI format
        let ffi_ops = flatten_circuit(circuit, params)?;
        let ffi_circuit = FfiCircuit {
            num_qubits: circuit.num_qubits,
            circuit_type: FfiCircuitType::from_circuit_type(&circuit.circuit_type),
            num_ops: ffi_ops.len() as u32,
            ops: ffi_ops.as_ptr(),
        };
        let ffi_config = FfiExecConfig {
            shots: config.shots.unwrap_or(0),
            seed: config.seed.unwrap_or(0),
        };

        let mut ffi_result = std::mem::MaybeUninit::<FfiExecResult>::uninit();
        let rc = (self.vtable.execute)(&ffi_circuit, &ffi_config, ffi_result.as_mut_ptr());

        if rc != 0 {
            return Err(OmegaError::Backend(format!(
                "backend {} returned error code {}",
                self.name, rc
            )));
        }

        let ffi_result = unsafe { ffi_result.assume_init() };
        let result = unflatten_result(&ffi_result);

        // Free the FFI result
        (self.vtable.free_result)(&ffi_result as *const _ as *mut _);

        result
    }
}

/// Registry of loaded backend plugins.
pub struct BackendRegistry {
    backends: Vec<LoadedBackend>,
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    /// Load all .so/.dll files from a directory.
    pub fn load_dir(&mut self, dir: &Path) -> Result<usize> {
        let mut count = 0;
        if !dir.exists() {
            return Ok(0);
        }

        let entries = std::fs::read_dir(dir).map_err(OmegaError::Io)?;
        for entry in entries {
            let entry = entry.map_err(OmegaError::Io)?;
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "so" || ext == "dylib" || ext == "dll" {
                match LoadedBackend::load(&path) {
                    Ok(backend) => {
                        eprintln!("Loaded backend: {} from {}", backend.name(), path.display());
                        self.backends.push(backend);
                        count += 1;
                    }
                    Err(e) => {
                        eprintln!("Warning: failed to load {}: {}", path.display(), e);
                    }
                }
            }
        }
        Ok(count)
    }

    /// Register a pre-loaded backend.
    pub fn register(&mut self, backend: LoadedBackend) {
        self.backends.push(backend);
    }

    /// Find a backend that supports the given circuit type.
    pub fn find_backend(&self, circuit_type: &CircuitType) -> Option<&LoadedBackend> {
        self.backends.iter().find(|b| b.supports(circuit_type))
    }

    /// Find a loaded backend by its exact name (as reported by the plugin's
    /// `name()` vtable entry). Used by the CLI/server to resolve
    /// `--backend NAME` against plugins *after* the compiled-in backends.
    pub fn find_by_name(&self, name: &str) -> Option<&LoadedBackend> {
        self.backends.iter().find(|b| b.name() == name)
    }

    /// List all loaded backends.
    pub fn list(&self) -> Vec<&str> {
        self.backends.iter().map(|b| b.name()).collect()
    }
}

/// Map a `GateKind` to its `GATE_*` FFI constant. `Custom` has no FFI encoding.
/// Shared by `flatten_circuit` and the capability check so the two never drift.
pub fn gate_kind_to_ffi(gate: &GateKind) -> Result<u32> {
    Ok(match gate {
        GateKind::H => GATE_H,
        GateKind::X => GATE_X,
        GateKind::Y => GATE_Y,
        GateKind::Z => GATE_Z,
        GateKind::S => GATE_S,
        GateKind::Sdg => GATE_SDG,
        GateKind::T => GATE_T,
        GateKind::Tdg => GATE_TDG,
        GateKind::Id => GATE_ID,
        GateKind::Rx => GATE_RX,
        GateKind::Ry => GATE_RY,
        GateKind::Rz => GATE_RZ,
        GateKind::U3 => GATE_U3,
        GateKind::U2 => GATE_U2,
        GateKind::U1 => GATE_U1,
        GateKind::CX => GATE_CX,
        GateKind::CY => GATE_CY,
        GateKind::CZ => GATE_CZ,
        GateKind::Swap => GATE_SWAP,
        GateKind::CRz => GATE_CRZ,
        GateKind::CU3 => GATE_CU3,
        GateKind::Rbs => GATE_RBS,
        GateKind::CCX => GATE_CCX,
        GateKind::CSwap => GATE_CSWAP,
        GateKind::PhaseShifter => GATE_PS,
        GateKind::BeamSplitterRx => GATE_BS_RX,
        GateKind::Measure => GATE_MEASURE,
        GateKind::Barrier => GATE_BARRIER,
        GateKind::Reset => GATE_RESET,
        GateKind::Custom(_) => return Err(OmegaError::Unsupported("custom gates in FFI".into())),
    })
}

/// Flatten a CircuitIR + params into a Vec<FfiGateOp>.
fn flatten_circuit(circuit: &CircuitIR, params: &ParameterBinding) -> Result<Vec<FfiGateOp>> {
    let mut ffi_ops = Vec::with_capacity(circuit.ops.len());

    for op in &circuit.ops {
        let gate_kind = gate_kind_to_ffi(&op.gate)?;

        let mut qubits = [0u32; 3];
        for (i, q) in op.qubits.iter().enumerate().take(3) {
            qubits[i] = q.0;
        }

        let mut resolved_params = [0.0f64; 3];
        for (i, p) in op.params.iter().enumerate().take(3) {
            resolved_params[i] = params.resolve(p)?;
        }

        ffi_ops.push(FfiGateOp {
            gate_kind,
            num_qubits: op.qubits.len() as u32,
            qubits,
            num_params: op.params.len() as u32,
            params: resolved_params,
        });
    }

    Ok(ffi_ops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_is_pinned() {
        // A bump here is a deliberate layout-breaking change; the reserved
        // tail is for additive growth and must not require it.
        assert_eq!(OMEGA_BACKEND_ABI_VERSION, 1);
    }

    #[test]
    fn vtable_layout_canary() {
        // 4 function pointers + an 8-slot reserved tail, all usize-sized under
        // repr(C). If this changes, every compiled plugin must be rebuilt and
        // OMEGA_BACKEND_ABI_VERSION bumped.
        assert_eq!(
            std::mem::size_of::<BackendVTable>(),
            12 * std::mem::size_of::<usize>()
        );
    }
}

/// Convert FFI result back to ExecResult.
fn unflatten_result(ffi: &FfiExecResult) -> Result<ExecResult> {
    match ffi.result_type {
        FfiResultType::Counts => {
            let mut counts = HashMap::new();
            unsafe {
                for i in 0..ffi.num_entries as usize {
                    let bs = *ffi.bitstrings.add(i);
                    let ct = *ffi.counts.add(i);
                    counts.insert(bs, ct);
                }
            }
            Ok(ExecResult::Counts(counts))
        }
        FfiResultType::Statevector => {
            let mut sv = Vec::with_capacity(ffi.num_amplitudes as usize);
            unsafe {
                for i in 0..ffi.num_amplitudes as usize {
                    let re = *ffi.amplitudes.add(i * 2);
                    let im = *ffi.amplitudes.add(i * 2 + 1);
                    sv.push(num_complex::Complex64::new(re, im));
                }
            }
            Ok(ExecResult::Statevector(sv))
        }
        FfiResultType::Probabilities => {
            let mut probs = Vec::with_capacity(ffi.num_probs as usize);
            unsafe {
                for i in 0..ffi.num_probs as usize {
                    probs.push(*ffi.probs.add(i));
                }
            }
            Ok(ExecResult::Probabilities(probs))
        }
    }
}
