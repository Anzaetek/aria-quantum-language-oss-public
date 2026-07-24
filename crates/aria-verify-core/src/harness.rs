// SPDX-License-Identifier: Apache-2.0
//! Glue that turns a shipped `.aria` file into something the omega-functions
//! runtime can execute, and runs the quantum side either:
//!   * through a WASM guest loaded in-process (`Transport::WasmInProcess`) —
//!     the "no socket, all in one" case the README/TESTING headline; or
//!   * natively on the host (`Transport::Native`) — a fallback used when the
//!     `wasm32-wasip1` guest has not been built, so a numeric verdict is still
//!     produced.
//!
//! The bridge is exact: `parse_aria → instantiate → aria_runtime::lower` gives
//! the symbol-preserving `omega_core::CircuitIR` that `HostState` registers.

use std::path::PathBuf;

use aria_core::ast::aria::parse_aria;
use aria_runtime::lower::{lower, Lowered};
use omega_core::circuit::{CircuitIR, GateKind, GateOp, Qubit};
use omega_core::executor::Observable;
use omega_wasm_runtime::host::{HostState, WasmResult};
use omega_wasm_runtime::WasmRunner;
use smallvec::smallvec;

const FUEL: u64 = 200_000_000_000;

/// How the quantum primitives are dispatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    /// WASM guest loaded into omega-wasm-runtime, in this process. No socket.
    WasmInProcess,
    /// Host-side native execution (fallback when the guest isn't built).
    Native,
}

impl Transport {
    pub fn label(self, guest: &str) -> String {
        match self {
            Transport::WasmInProcess => {
                format!("{guest}.wasm loaded into omega-wasm-runtime, in-process")
            }
            Transport::Native => "native host execution (omega-backend-statevector)".to_string(),
        }
    }
}

/// Workspace root (two levels up from this crate's manifest).
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Absolute path to a shipped example, e.g. `examples/aria/qsvd.aria`.
pub fn aria_path(file: &str) -> PathBuf {
    repo_root().join("examples/aria").join(file)
}

/// Absolute path to a built guest wasm, e.g. guest `vqe` → `…/vqe.wasm`.
pub fn guest_wasm(guest: &str) -> PathBuf {
    repo_root()
        .join("examples/wasm-guests")
        .join(guest)
        .join("target/wasm32-wasip1/release")
        .join(format!("{guest}.wasm"))
}

/// Parse + instantiate a shipped `.aria` file to an aria `Circuit` (unlowered).
/// Used by the socket transport, which lowers to the JSON wire IR itself.
#[cfg(feature = "remote")]
pub fn load_circuit(
    file: &str,
    circuit: &str,
    int_params: &[(&str, i64)],
) -> Result<aria_core::ast::Circuit, String> {
    let src = std::fs::read_to_string(aria_path(file)).map_err(|e| format!("read {file}: {e}"))?;
    let prog = parse_aria(&src)?;
    prog.instantiate(circuit, int_params)
}

/// Parse + instantiate + lower a shipped `.aria` file (resolved under
/// `examples/aria/`) to the omega IR. For an `.aria` file anywhere else —
/// e.g. an external integration shipping its own circuits — use
/// [`load_lowered_path`] with an absolute or working-directory-relative path.
pub fn load_lowered(
    file: &str,
    circuit: &str,
    int_params: &[(&str, i64)],
) -> Result<Lowered, String> {
    load_lowered_path(&aria_path(file), circuit, int_params)
}

/// Parse + instantiate + lower an `.aria` file at an arbitrary path to the
/// omega IR. Unlike [`load_lowered`], the path is not resolved against the
/// repo's `examples/aria/` directory — pass an absolute path or one relative
/// to the current working directory, so callers outside this repository can
/// use the loading helpers on their own circuits.
pub fn load_lowered_path(
    path: &std::path::Path,
    circuit: &str,
    int_params: &[(&str, i64)],
) -> Result<Lowered, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let prog = parse_aria(&src)?;
    let circ = prog.instantiate(circuit, int_params)?;
    lower(&circ)
}

/// Pick the transport: WASM if the guest binary exists, else native fallback.
pub fn transport_for(guest: &str) -> Transport {
    if guest_wasm(guest).exists() {
        Transport::WasmInProcess
    } else {
        Transport::Native
    }
}

/// Minimize `⟨observable⟩` over `n_params` rotation angles of `ir`, returning
/// `(min_value, best_params)`. Dispatches per `transport`.
#[allow(clippy::too_many_arguments)]
pub fn minimize(
    transport: Transport,
    guest: &str,
    ir: CircuitIR,
    observable: Observable,
    n_params: usize,
    init: Vec<f64>,
    max_iters: usize,
    lr: f64,
) -> Result<(f64, Vec<f64>), String> {
    match transport {
        Transport::WasmInProcess => {
            minimize_wasm(guest, ir, observable, n_params, init, max_iters, lr)
        }
        Transport::Native => minimize_native(ir, observable, n_params, init, max_iters, lr),
    }
}

/// Run the optimizer loop inside the WASM guest (e.g. `vqe.wasm`). The guest
/// reads `circuit_id`/`observable_id`/`num_params`/`max_iters`/`lr` from the
/// staged input JSON, calls back into omega for `execute`/`gradient`, and
/// reports the best `(value, params)` via `omega_report_result`.
fn minimize_wasm(
    guest: &str,
    ir: CircuitIR,
    observable: Observable,
    n_params: usize,
    init: Vec<f64>,
    max_iters: usize,
    lr: f64,
) -> Result<(f64, Vec<f64>), String> {
    let mut host = HostState::new();
    let cid = host.register_circuit(ir);
    let oid = host.register_observable(observable);
    let input = serde_json::json!({
        "circuit_id": cid,
        "observable_id": oid,
        "num_params": n_params,
        "init": init,
        "max_iters": max_iters,
        "lr": lr,
        "optimizer": "gd",
        "log_every": 0,
    });
    host.set_input(input.to_string().into_bytes());

    let wasm_bytes =
        std::fs::read(guest_wasm(guest)).map_err(|e| format!("read {guest}.wasm: {e}"))?;
    let runner = WasmRunner::new(host).map_err(|e| format!("wasm runner: {e}"))?;
    let result: Option<WasmResult> = runner
        .run(&wasm_bytes, FUEL)
        .map_err(|e| format!("wasm run: {e}"))?;
    match result {
        Some(r) => Ok((r.optimal_value, r.optimal_params)),
        None => Err(format!("{guest}.wasm returned no result")),
    }
}

/// Host-side gradient descent (finite-difference) — the native fallback. Uses
/// the SAME host primitive (`execute_expectation`) the guest would call.
fn minimize_native(
    ir: CircuitIR,
    observable: Observable,
    n_params: usize,
    init: Vec<f64>,
    max_iters: usize,
    lr: f64,
) -> Result<(f64, Vec<f64>), String> {
    let mut host = HostState::new();
    let cid = host.register_circuit(ir);
    let oid = host.register_observable(observable);
    let eval = |p: &[f64]| -> Result<f64, String> {
        host.execute_expectation(cid, p, oid)
            .map_err(|e| e.to_string())
    };

    let eps = 1e-6;
    let mut params = init;
    params.resize(n_params, 0.0);
    let mut best_val = eval(&params)?;
    let mut best = params.clone();
    for _ in 0..max_iters {
        // Finite-difference gradient.
        let mut grad = vec![0.0; n_params];
        for i in 0..n_params {
            let mut pp = params.clone();
            pp[i] += eps;
            let mut pm = params.clone();
            pm[i] -= eps;
            grad[i] = (eval(&pp)? - eval(&pm)?) / (2.0 * eps);
        }
        for i in 0..n_params {
            params[i] -= lr * grad[i];
        }
        let v = eval(&params)?;
        if v < best_val {
            best_val = v;
            best = params.clone();
        }
    }
    Ok((best_val, best))
}

/// Prepend `X` gates so the circuit starts from computational basis state
/// `|x⟩` (bit `b` of `x` → `X` on qubit `b`), instead of `|0…0⟩`.
pub fn prepend_basis_state(ir: &mut CircuitIR, x: u64) {
    let mut prep: Vec<GateOp> = Vec::new();
    for b in 0..ir.num_qubits {
        if (x >> b) & 1 == 1 {
            prep.push(GateOp {
                gate: GateKind::X,
                qubits: smallvec![Qubit(b)],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
    }
    prep.extend(std::mem::take(&mut ir.ops));
    ir.ops = prep;
}

/// Append a measurement of `qubit` into classical bit `cbit`, growing the
/// classical register if needed. Used to read an otherwise-unmeasured output
/// qubit (e.g. Bob's qubit in teleportation).
pub fn append_measure(ir: &mut CircuitIR, qubit: u32, cbit: u32) {
    if cbit + 1 > ir.num_classical_bits {
        ir.num_classical_bits = cbit + 1;
    }
    ir.ops.push(GateOp {
        gate: GateKind::Measure,
        qubits: smallvec![Qubit(qubit)],
        params: smallvec![],
        classical_bit: Some(cbit),
        condition: None,
    });
}

/// A copy of `template` with ops replaced by just the `X` gates that prepare
/// `|x⟩` — used to discover the backend's index for `|x⟩` (ordering-agnostic).
pub fn basis_prep_ir(template: &CircuitIR, x: u64) -> CircuitIR {
    let mut ir = template.clone();
    ir.ops.clear();
    prepend_basis_state(&mut ir, x);
    ir
}

/// What the `omega_app` guest should measure on a registered circuit.
pub enum AppMode {
    /// Exact statevector — payload is interleaved (re, im) pairs.
    Statevector,
    /// Sampled counts — payload is `[bits0, count0, bits1, count1, ...]`.
    Counts { shots: u32 },
    /// Expectation values of the given observables — payload is `[⟨O_i⟩, ...]`.
    Expectations(Vec<Observable>),
    /// One Collapse-mode trajectory's creg bits — payload is `[bit0, bit1, ...]`.
    ClassicalBits { seed: i64 },
}

impl AppMode {
    fn tag(&self) -> &'static str {
        match self {
            AppMode::Statevector => "statevector",
            AppMode::Counts { .. } => "counts",
            AppMode::Expectations(_) => "expectations",
            AppMode::ClassicalBits { .. } => "classical_bits",
        }
    }
}

/// Run a single measurement on `ir` through the omega runtime and return the
/// `(payload, value)` the `omega_app` guest reports (or the native equivalent).
/// `params` binds the circuit's free symbols (empty for non-parametric ones).
pub fn execute_report(
    transport: Transport,
    ir: CircuitIR,
    mode: AppMode,
    params: &[f64],
) -> Result<(Vec<f64>, f64), String> {
    match transport {
        Transport::WasmInProcess => execute_report_wasm(ir, mode, params),
        Transport::Native => execute_report_native(ir, mode, params),
    }
}

fn execute_report_wasm(
    ir: CircuitIR,
    mode: AppMode,
    params: &[f64],
) -> Result<(Vec<f64>, f64), String> {
    let mut host = HostState::new();
    let cid = host.register_circuit(ir);
    let obs_ids: Vec<u32> = match &mode {
        AppMode::Expectations(obs) => obs
            .iter()
            .map(|o| host.register_observable(o.clone()))
            .collect(),
        _ => Vec::new(),
    };
    let mut job = serde_json::json!({
        "circuit_id": cid,
        "mode": mode.tag(),
        "params": params,
    });
    match &mode {
        AppMode::Counts { shots } => job["shots"] = (*shots as i64).into(),
        AppMode::ClassicalBits { seed } => job["seed"] = (*seed).into(),
        AppMode::Expectations(_) => {
            job["observable_ids"] = obs_ids.iter().map(|&i| i as i64).collect::<Vec<_>>().into()
        }
        AppMode::Statevector => {}
    }
    host.set_input(job.to_string().into_bytes());

    let wasm_bytes =
        std::fs::read(guest_wasm("omega_app")).map_err(|e| format!("read omega_app.wasm: {e}"))?;
    let runner = WasmRunner::new(host).map_err(|e| format!("wasm runner: {e}"))?;
    let result = runner
        .run(&wasm_bytes, FUEL)
        .map_err(|e| format!("wasm run: {e}"))?;
    match result {
        Some(r) => Ok((r.optimal_params, r.optimal_value)),
        None => Err("omega_app.wasm returned no result".to_string()),
    }
}

fn execute_report_native(
    ir: CircuitIR,
    mode: AppMode,
    params: &[f64],
) -> Result<(Vec<f64>, f64), String> {
    let mut host = HostState::new();
    let cid = host.register_circuit(ir);
    match mode {
        AppMode::Statevector => {
            let payload = host
                .statevector_interleaved(cid, params)
                .map_err(|e| e.to_string())?;
            let n = (payload.len() / 2) as f64;
            Ok((payload, n))
        }
        AppMode::Counts { shots } => {
            let counts = host
                .execute_with_shots(cid, params, shots, Some(42))
                .map_err(|e| e.to_string())?;
            let mut sorted: Vec<_> = counts.into_iter().collect();
            sorted.sort_by_key(|&(k, _)| k);
            let mut payload = Vec::with_capacity(sorted.len() * 2);
            for (bits, c) in &sorted {
                payload.push(*bits as f64);
                payload.push(*c as f64);
            }
            Ok((payload, sorted.len() as f64))
        }
        AppMode::Expectations(obs) => {
            let ids: Vec<u32> = obs
                .into_iter()
                .map(|o| host.register_observable(o))
                .collect();
            let vals = host
                .execute_multi(cid, params, &ids)
                .map_err(|e| e.to_string())?;
            let first = vals.first().copied().unwrap_or(0.0);
            Ok((vals, first))
        }
        AppMode::ClassicalBits { seed } => {
            let bits = host
                .execute_classical(cid, params, Some(seed as u64))
                .map_err(|e| e.to_string())?;
            let mut value = 0u64;
            for (i, &b) in bits.iter().enumerate() {
                if b != 0 {
                    value |= 1u64 << i;
                }
            }
            let payload: Vec<f64> = bits.iter().map(|&b| b as f64).collect();
            Ok((payload, value as f64))
        }
    }
}

/// Native helper: exact statevector amplitudes of a lowered circuit (no params).
pub fn native_statevector(ir: &CircuitIR) -> Result<Vec<num_complex::Complex64>, String> {
    use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
    use omega_core::params::ParameterBinding;
    let backend = omega_backend_statevector::StatevectorBackend::new();
    let cfg = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    match backend
        .execute(ir, &ParameterBinding::new(), &cfg)
        .map_err(|e| e.to_string())?
    {
        ExecResult::Statevector(sv) => Ok(sv),
        other => Err(format!("expected statevector, got {other:?}")),
    }
}
