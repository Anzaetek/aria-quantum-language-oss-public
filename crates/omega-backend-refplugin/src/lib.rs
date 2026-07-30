//! Reference backend plugin for the omega plugin ABI.
//!
//! This cdylib exports the two symbols the loader
//! (`omega_core::plugin::LoadedBackend::load`) requires:
//! `omega_backend_abi_version()` and `omega_backend_init()`. It is a *correct*
//! backend — it reconstructs a [`CircuitIR`] from the flattened FFI circuit and
//! runs the in-tree dense statevector simulator — so it doubles as the fixture
//! the plugin CLI wiring and the conformance kit load and check against.
//!
//! It is intentionally small and gate-model only; it is not meant to be a
//! performance path, and it declines photonic circuits.

use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::OnceLock;

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::ffi_types::*;
use omega_core::params::ParameterBinding;
use smallvec::SmallVec;

/// ABI handshake — must match the host's [`OMEGA_BACKEND_ABI_VERSION`].
///
/// # Safety
/// Exported for the dynamic loader; takes no arguments and only returns a
/// constant, so it is trivially sound to call.
#[no_mangle]
pub extern "C" fn omega_backend_abi_version() -> u32 {
    OMEGA_BACKEND_ABI_VERSION
}

/// Construct the vtable the loader reads. The `reserved` tail is zeroed, as the
/// ABI requires.
///
/// # Safety
/// Exported for the dynamic loader; returns a vtable of `'static` function
/// pointers.
#[no_mangle]
pub extern "C" fn omega_backend_init() -> BackendVTable {
    BackendVTable {
        name: backend_name,
        supports,
        execute,
        free_result,
        reserved: [0usize; 8],
    }
}

extern "C" fn backend_name() -> *const c_char {
    static NAME: OnceLock<CString> = OnceLock::new();
    NAME.get_or_init(|| CString::new("refplugin").unwrap())
        .as_ptr()
}

extern "C" fn supports(circuit_type: FfiCircuitType) -> bool {
    matches!(circuit_type, FfiCircuitType::GateBased)
}

extern "C" fn execute(
    circuit: *const FfiCircuit,
    config: *const FfiExecConfig,
    result_out: *mut FfiExecResult,
) -> i32 {
    if circuit.is_null() || config.is_null() || result_out.is_null() {
        return 1;
    }
    // A panic must never unwind across the C boundary; convert it to a code.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let circuit = unsafe { &*circuit };
        let config = unsafe { &*config };
        run(circuit, config)
    }));
    match outcome {
        Ok(Ok(ffi)) => {
            unsafe { result_out.write(ffi) };
            0
        }
        Ok(Err(_)) => 2,
        Err(_) => 1,
    }
}

extern "C" fn free_result(result: *mut FfiExecResult) {
    if result.is_null() {
        return;
    }
    let r = unsafe { &*result };
    unsafe {
        match r.result_type {
            FfiResultType::Counts => {
                reclaim(r.bitstrings, r.num_entries as usize);
                reclaim(r.counts, r.num_entries as usize);
            }
            FfiResultType::Statevector => {
                reclaim(r.amplitudes, r.num_amplitudes as usize * 2);
            }
            FfiResultType::Probabilities => {
                reclaim(r.probs, r.num_probs as usize);
            }
        }
    }
}

/// Reconstruct a boxed slice from a leaked pointer and drop it. `len` must be
/// exactly the length handed out by [`leak`] for this pointer.
unsafe fn reclaim<T>(ptr: *mut T, len: usize) {
    if !ptr.is_null() && len > 0 {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// Leak a `Vec` as a thin data pointer + length. The allocation is reclaimed by
/// [`reclaim`] (called from `free_result`), matching capacity to length via
/// `into_boxed_slice`.
fn leak<T>(v: Vec<T>) -> (*mut T, usize) {
    let boxed = v.into_boxed_slice();
    let len = boxed.len();
    let ptr = Box::into_raw(boxed) as *mut T;
    (ptr, len)
}

/// The plugin body, in safe Rust: rebuild the IR, run statevector, flatten.
///
/// `pub` so the in-repo tests can call it directly (via the rlib) without a
/// dynamic load.
pub fn run(ffi: &FfiCircuit, cfg: &FfiExecConfig) -> Result<FfiExecResult, String> {
    let ir = rebuild_ir(ffi)?;
    let config = ExecConfig {
        shots: if cfg.shots == 0 {
            None
        } else {
            Some(cfg.shots)
        },
        seed: if cfg.seed == 0 { None } else { Some(cfg.seed) },
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let result = StatevectorBackend
        .execute(&ir, &ParameterBinding::default(), &config)
        .map_err(|e| e.to_string())?;
    Ok(flatten_result(result))
}

fn rebuild_ir(ffi: &FfiCircuit) -> Result<CircuitIR, String> {
    if !matches!(ffi.circuit_type, FfiCircuitType::GateBased) {
        return Err("refplugin only supports gate-based circuits".to_string());
    }
    let mut ir = CircuitIR::new(ffi.num_qubits, CircuitType::GateBased);
    let ops = if ffi.num_ops == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(ffi.ops, ffi.num_ops as usize) }
    };
    for op in ops {
        let gate = ffi_gate_to_kind(op.gate_kind)?;
        let qubits: SmallVec<[Qubit; 3]> = op.qubits[..op.num_qubits.min(3) as usize]
            .iter()
            .map(|&q| Qubit(q))
            .collect();
        let params: SmallVec<[ParamExpr; 3]> = op.params[..op.num_params.min(3) as usize]
            .iter()
            .map(|&p| ParamExpr::Concrete(p))
            .collect();
        ir.add_op(GateOp {
            gate,
            qubits,
            params,
            classical_bit: None,
            condition: None,
        });
    }
    Ok(ir)
}

fn flatten_result(result: ExecResult) -> FfiExecResult {
    // Every arm fills its own fields and nulls the rest.
    let mut out = FfiExecResult {
        result_type: FfiResultType::Counts,
        num_entries: 0,
        bitstrings: std::ptr::null_mut(),
        counts: std::ptr::null_mut(),
        num_amplitudes: 0,
        amplitudes: std::ptr::null_mut(),
        num_probs: 0,
        probs: std::ptr::null_mut(),
    };
    match result {
        ExecResult::Counts(map) => {
            let mut bitstrings = Vec::with_capacity(map.len());
            let mut counts = Vec::with_capacity(map.len());
            for (bs, ct) in map {
                bitstrings.push(bs);
                counts.push(ct);
            }
            let (bp, n) = leak(bitstrings);
            let (cp, _) = leak(counts);
            out.result_type = FfiResultType::Counts;
            out.num_entries = n as u32;
            out.bitstrings = bp;
            out.counts = cp;
        }
        ExecResult::Statevector(sv) => {
            let mut interleaved = Vec::with_capacity(sv.len() * 2);
            for z in &sv {
                interleaved.push(z.re);
                interleaved.push(z.im);
            }
            let (ap, _) = leak(interleaved);
            out.result_type = FfiResultType::Statevector;
            out.num_amplitudes = sv.len() as u32;
            out.amplitudes = ap;
        }
        ExecResult::Probabilities(probs) => {
            let n = probs.len();
            let (pp, _) = leak(probs);
            out.result_type = FfiResultType::Probabilities;
            out.num_probs = n as u32;
            out.probs = pp;
        }
    }
    out
}

/// Inverse of `omega_core::plugin::flatten_circuit`'s gate-kind mapping.
fn ffi_gate_to_kind(k: u32) -> Result<GateKind, String> {
    Ok(match k {
        GATE_H => GateKind::H,
        GATE_X => GateKind::X,
        GATE_Y => GateKind::Y,
        GATE_Z => GateKind::Z,
        GATE_S => GateKind::S,
        GATE_SDG => GateKind::Sdg,
        GATE_T => GateKind::T,
        GATE_TDG => GateKind::Tdg,
        GATE_ID => GateKind::Id,
        GATE_RX => GateKind::Rx,
        GATE_RY => GateKind::Ry,
        GATE_RZ => GateKind::Rz,
        GATE_U3 => GateKind::U3,
        GATE_U2 => GateKind::U2,
        GATE_U1 => GateKind::U1,
        GATE_CX => GateKind::CX,
        GATE_CY => GateKind::CY,
        GATE_CZ => GateKind::CZ,
        GATE_SWAP => GateKind::Swap,
        GATE_CRZ => GateKind::CRz,
        GATE_CU3 => GateKind::CU3,
        GATE_RBS => GateKind::Rbs,
        GATE_CCX => GateKind::CCX,
        GATE_CSWAP => GateKind::CSwap,
        GATE_MEASURE => GateKind::Measure,
        GATE_BARRIER => GateKind::Barrier,
        GATE_RESET => GateKind::Reset,
        other => return Err(format!("refplugin: unknown FFI gate kind {other}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Bell circuit flattened to FFI ops, run through `run`, must yield only
    /// the |00> and |11> basis states — exercising rebuild_ir + statevector +
    /// flatten_result end to end without a dynamic load.
    #[test]
    fn refplugin_runs_bell_to_correlated_counts() {
        let ops = [
            FfiGateOp {
                gate_kind: GATE_H,
                num_qubits: 1,
                qubits: [0, 0, 0],
                num_params: 0,
                params: [0.0; 3],
            },
            FfiGateOp {
                gate_kind: GATE_CX,
                num_qubits: 2,
                qubits: [0, 1, 0],
                num_params: 0,
                params: [0.0; 3],
            },
        ];
        let circuit = FfiCircuit {
            num_qubits: 2,
            circuit_type: FfiCircuitType::GateBased,
            num_ops: ops.len() as u32,
            ops: ops.as_ptr(),
        };
        let config = FfiExecConfig {
            shots: 2048,
            seed: 7,
        };
        let ffi = run(&circuit, &config).unwrap();
        assert_eq!(ffi.result_type, FfiResultType::Counts);
        let entries = ffi.num_entries as usize;
        assert!(entries >= 1 && entries <= 2);
        let bitstrings = unsafe { std::slice::from_raw_parts(ffi.bitstrings, entries) };
        for &bs in bitstrings {
            assert!(bs == 0b00 || bs == 0b11, "unexpected basis state {bs:#b}");
        }
        // Free through the same path the host uses.
        let mut ffi = ffi;
        free_result(&mut ffi as *mut _);
    }

    #[test]
    fn refplugin_rejects_photonic() {
        assert!(!supports(FfiCircuitType::Photonic));
        assert!(supports(FfiCircuitType::GateBased));
    }
}
