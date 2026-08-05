//! Bridge endpoint for the `quantum-core` toolkit.
//!
//! `quantum-core` produces an `OmegaCircuitIR` JSON document (defined in
//! `crates/aria-core/src/backends/omega.rs` on the quantum side) with
//! an explicit `backend` selector. This module mirrors the wire types,
//! translates them into `omega_core::circuit::CircuitIR`, and dispatches
//! execution to the matching omega backend:
//!
//! | `backend` field         | runtime backend                     |
//! |-------------------------|-------------------------------------|
//! | `"Auto"`                | heuristic (Clifford→Pauli, large→MPS, else statevector) |
//! | `"Statevector"`         | `omega_backend_statevector::StatevectorBackend` |
//! | `{ "Mps": { … } }`      | `omega_backend_mps::MpsBackend{max_bond_dim}` |
//! | `"Stabilizer"`          | `omega_backend_pauli::PauliBackend`  |
//! | `"Photonic"`            | `omega_backend_photonics::PhotonicsBackend` |
//!
//! It also exposes a sibling endpoint for **MBQC one-way patterns** (C1.3):
//! `POST /v1/quantum/execute_pattern` takes an [`OmegaPatternIR`] (the
//! `quantum-core` measurement-pattern wire type) and runs it on the photonic
//! graph-state executor (`omega_backend_photonics::mbqc`), returning the
//! canonical output statevector.

use crate::timing::{PhaseTimer, RowTimer};
use crate::worker::{governor, CostKind, JobShape, Reservation};
use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable};
use omega_core::params::ParameterBinding;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::auth::middleware::check_rights;
use crate::auth::rights;
use crate::auth::token::TokenClaims;
use crate::AppState;

type SharedState = Arc<RwLock<AppState>>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub enum OmegaGateKind {
    H,
    X,
    Y,
    Z,
    S,
    Sdg,
    T,
    Tdg,
    Id,
    Rx,
    Ry,
    Rz,
    U3,
    U2,
    U1,
    CX,
    CY,
    CZ,
    Swap,
    CRz,
    CU3,
    CCX,
    CSwap,
    PhaseShifter,
    BeamSplitterRx,
    Measure,
    Barrier,
    Reset,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OmegaGateOp {
    pub gate: OmegaGateKind,
    pub qubits: Vec<u32>,
    pub params: Vec<f64>,
    pub classical_bit: Option<u32>,
    pub condition: Option<(u32, u64)>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OmegaMidCircuitMode {
    Skip,
    Collapse,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OmegaBackendSel {
    Auto,
    Statevector,
    Mps {
        max_bond_dim: u32,
    },
    Stabilizer,
    Photonic,
    /// A dynamically-loaded backend plugin, selected by name. Never chosen by
    /// `Auto` resolution — a client must request it explicitly.
    Plugin {
        name: String,
    },
}

/// Lazily-loaded backend-plugin registry, populated once from
/// `OMEGA_BACKEND_DIR` (`:`-separated). Plugin loading is **opt-in** — the
/// server never probes a home directory implicitly, so no plugin is dlopened
/// unless an operator configured `OMEGA_BACKEND_DIR`.
///
/// Plugin backends must be thread-safe: the registry is shared across job
/// handlers and plugin execution is not serialized. (This mirrors the
/// requirement stated on `BackendVTable` in the ABI.)
fn plugin_registry() -> &'static omega_core::plugin::BackendRegistry {
    use std::sync::OnceLock;
    static REGISTRY: OnceLock<omega_core::plugin::BackendRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut reg = omega_core::plugin::BackendRegistry::new();
        if let Ok(env_dirs) = std::env::var("OMEGA_BACKEND_DIR") {
            for d in env_dirs.split(':').filter(|s| !s.is_empty()) {
                if let Err(e) = reg.load_dir(std::path::Path::new(d)) {
                    eprintln!("[quantum] backend-dir {d}: {e}");
                }
            }
        }
        reg
    })
}

/// Names of the plugin backends currently loaded, for `/v1/backends`.
pub fn loaded_plugin_names() -> Vec<String> {
    plugin_registry()
        .list()
        .iter()
        .map(|s| s.to_string())
        .collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OmegaCircuitIR {
    pub num_qubits: u32,
    pub num_classical_bits: u32,
    pub ops: Vec<OmegaGateOp>,
    pub is_photonic: bool,
    pub mid_circuit_mode: OmegaMidCircuitMode,
    pub backend: OmegaBackendSel,
}

impl OmegaGateKind {
    fn to_core(&self) -> GateKind {
        match self {
            OmegaGateKind::H => GateKind::H,
            OmegaGateKind::X => GateKind::X,
            OmegaGateKind::Y => GateKind::Y,
            OmegaGateKind::Z => GateKind::Z,
            OmegaGateKind::S => GateKind::S,
            OmegaGateKind::Sdg => GateKind::Sdg,
            OmegaGateKind::T => GateKind::T,
            OmegaGateKind::Tdg => GateKind::Tdg,
            OmegaGateKind::Id => GateKind::Id,
            OmegaGateKind::Rx => GateKind::Rx,
            OmegaGateKind::Ry => GateKind::Ry,
            OmegaGateKind::Rz => GateKind::Rz,
            OmegaGateKind::U3 => GateKind::U3,
            OmegaGateKind::U2 => GateKind::U2,
            OmegaGateKind::U1 => GateKind::U1,
            OmegaGateKind::CX => GateKind::CX,
            OmegaGateKind::CY => GateKind::CY,
            OmegaGateKind::CZ => GateKind::CZ,
            OmegaGateKind::Swap => GateKind::Swap,
            OmegaGateKind::CRz => GateKind::CRz,
            OmegaGateKind::CU3 => GateKind::CU3,
            OmegaGateKind::CCX => GateKind::CCX,
            OmegaGateKind::CSwap => GateKind::CSwap,
            OmegaGateKind::PhaseShifter => GateKind::PhaseShifter,
            OmegaGateKind::BeamSplitterRx => GateKind::BeamSplitterRx,
            OmegaGateKind::Measure => GateKind::Measure,
            OmegaGateKind::Barrier => GateKind::Barrier,
            OmegaGateKind::Reset => GateKind::Reset,
        }
    }
}

/// Translate the quantum-core wire IR into omega-core's `CircuitIR`.
pub fn translate_to_core_ir(ir: &OmegaCircuitIR) -> CircuitIR {
    let circuit_type = if ir.is_photonic {
        CircuitType::Photonic
    } else {
        CircuitType::GateBased
    };
    let mut core = CircuitIR::new(ir.num_qubits, circuit_type);
    core.num_classical_bits = ir.num_classical_bits;
    for op in &ir.ops {
        let qubits: SmallVec<[Qubit; 3]> = op.qubits.iter().map(|&q| Qubit(q)).collect();
        let params: SmallVec<[ParamExpr; 3]> =
            op.params.iter().map(|&p| ParamExpr::Concrete(p)).collect();
        core.add_op(GateOp {
            gate: op.gate.to_core(),
            qubits,
            params,
            classical_bit: op.classical_bit,
            // Wire format from `quantum-core` carries the legacy
            // single-bit `(start_bit, expected)` shape. The internal
            // `GateOp::condition` was widened to `(start_bit,
            // num_bits, expected)` in commit f47c9e7 (multi-bit creg
            // `if(c == V)` support). Default `num_bits = 1` keeps the
            // pre-widening single-bit semantics — a multi-bit `if`
            // from `quantum-core` would need a wire-schema bump on
            // both sides; defer until a real caller needs it.
            condition: op
                .condition
                .map(|(start_bit, expected)| (start_bit, 1, expected)),
        });
    }
    core
}

/// Map a resolved backend to the cost curve the governor should price it with.
///
/// `Auto` is resolved first (see [`admit_ir`]) rather than assumed dense —
/// pricing a 30-qubit Clifford circuit as `2^30 × 16` would refuse a job the
/// stabilizer backend runs in kilobytes.
fn cost_kind_for(sel: &OmegaBackendSel) -> CostKind {
    match sel {
        OmegaBackendSel::Statevector => CostKind::DenseStatevector,
        OmegaBackendSel::Mps { max_bond_dim } => CostKind::Mps {
            max_bond_dim: *max_bond_dim,
        },
        OmegaBackendSel::Stabilizer => CostKind::Stabilizer,
        OmegaBackendSel::Photonic => CostKind::Photonic,
        OmegaBackendSel::Plugin { .. } => CostKind::Opaque,
        // resolve_backend never returns Auto; price defensively if it ever does.
        OmegaBackendSel::Auto => CostKind::DenseStatevector,
    }
}

/// Reserve capacity for `ir` or turn the refusal into an HTTP response.
///
/// Held for the duration of execution: dropping the returned [`Reservation`]
/// is what returns the budget, so callers must keep it alive across the run.
/// Where a statevector circuit will actually execute, so the governor debits
/// the right memory pool and prices at the right element width (device kernels
/// are f32). Mirrors `exec_statevector`'s routing exactly — if the two ever
/// disagree, the reservation is against the wrong pool.
fn exec_target_for(sel: &OmegaBackendSel) -> crate::worker::ExecTarget {
    #[cfg(feature = "opencl")]
    {
        use omega_core::device::DeviceKind;
        if matches!(sel, OmegaBackendSel::Statevector | OmegaBackendSel::Auto)
            && DeviceKind::resolve(None) == DeviceKind::OpenCl
        {
            return crate::worker::ExecTarget::Device(0);
        }
    }
    let _ = sel;
    crate::worker::ExecTarget::Cpu
}

fn shape_for(ir: &OmegaCircuitIR, densifies: bool, batch: usize) -> JobShape {
    let resolved = resolve_backend(ir);
    let mut shape = JobShape::new(ir.num_qubits, cost_kind_for(&resolved));
    shape.densifies = densifies;
    shape.batch = batch;
    shape.target = exec_target_for(&resolved);
    shape
}

/// Shape for `/execute`. `shots: None` ships every amplitude back (and so pays
/// for the JSON encoding); a shot run samples and returns counts.
fn execute_shape(ir: &OmegaCircuitIR, shots: Option<u32>) -> JobShape {
    let base = shape_for(ir, true, 1);
    match shots {
        None => base.returning_statevector(),
        Some(_) => base.with_shots(),
    }
}

/// Reserve for a whole batch by pricing **every** row and taking the worst.
///
/// Pricing only the widest row was a hole: rows carry independent `backend`
/// selections, so a 40-qubit Clifford row (cheap, stabilizer) could be picked
/// as "widest" while a 30-qubit Statevector row in the same request allocated
/// 16 GiB against that reservation. Width does not imply cost.
fn admit_batch(
    circuits: &[OmegaCircuitIR],
    densifies: bool,
) -> Result<Reservation, Box<axum::response::Response>> {
    let worst = circuits
        .iter()
        .map(|ir| shape_for(ir, densifies, circuits.len()))
        // An unpriceable row sorts above every priced one: it must not be
        // silently out-ranked by a row we happen to have a number for.
        .max_by_key(|sh| crate::worker::estimate_peak_bytes(sh).unwrap_or(u64::MAX))
        .expect("caller checked the batch is non-empty");
    admit_shape(&worst)
}

fn admit_shape(shape: &JobShape) -> Result<Reservation, Box<axum::response::Response>> {
    governor().admit(shape).map_err(|rej| {
        // Boxed: an axum Response is large and this is the cold path.
        // Busy is the only refusal waiting can fix; everything else is a
        // property of the request and must not invite a retry storm.
        let (status, retry_after) = if rej.is_transient() {
            (StatusCode::TOO_MANY_REQUESTS, Some("1"))
        } else {
            (StatusCode::PAYLOAD_TOO_LARGE, None)
        };
        let body = Json(serde_json::json!({
            "error": rej.message(),
            "capacity_bytes": governor().config().capacity_bytes,
            "available_bytes": governor().available_bytes(),
        }));
        Box::new(match retry_after {
            Some(secs) => (status, [("Retry-After", secs)], body).into_response(),
            None => (status, body).into_response(),
        })
    })
}

/// Decide which concrete backend should execute this IR.
fn resolve_backend(ir: &OmegaCircuitIR) -> OmegaBackendSel {
    if let OmegaBackendSel::Auto = ir.backend {
        if ir.is_photonic {
            return OmegaBackendSel::Photonic;
        }
        let clifford_only = ir.ops.iter().all(|op| {
            matches!(
                op.gate,
                OmegaGateKind::H
                    | OmegaGateKind::X
                    | OmegaGateKind::Y
                    | OmegaGateKind::Z
                    | OmegaGateKind::S
                    | OmegaGateKind::Sdg
                    | OmegaGateKind::CX
                    | OmegaGateKind::CY
                    | OmegaGateKind::CZ
                    | OmegaGateKind::Swap
                    | OmegaGateKind::Measure
                    | OmegaGateKind::Barrier
                    | OmegaGateKind::Reset
                    | OmegaGateKind::Id
            )
        });
        if clifford_only {
            return OmegaBackendSel::Stabilizer;
        }
        if ir.num_qubits >= 20 {
            return OmegaBackendSel::Mps { max_bond_dim: 64 };
        }
        OmegaBackendSel::Statevector
    } else {
        ir.backend.clone()
    }
}

/// Execute an `OmegaCircuitIR` on the runtime backend selected by its
/// `backend` field. Returns the raw `ExecResult` variant so the caller
/// can render it to the wire.
pub fn execute_quantum_ir(
    ir: &OmegaCircuitIR,
    shots: Option<u32>,
    seed: Option<u64>,
) -> omega_core::error::Result<(ExecResult, OmegaBackendSel)> {
    let core = translate_to_core_ir(ir);
    let binding = ParameterBinding::new();
    let mid_circuit_mode = match ir.mid_circuit_mode {
        OmegaMidCircuitMode::Skip => MidCircuitMode::Skip,
        OmegaMidCircuitMode::Collapse => MidCircuitMode::Collapse,
    };
    let config = ExecConfig {
        shots,
        seed,
        mid_circuit_mode,
    };

    let resolved = resolve_backend(ir);
    let result = match &resolved {
        OmegaBackendSel::Auto => unreachable!("resolve_backend never returns Auto"),
        OmegaBackendSel::Statevector => exec_statevector(&core, &binding, &config)?,
        OmegaBackendSel::Mps { max_bond_dim } => {
            omega_backend_mps::MpsBackend::new(*max_bond_dim as usize)
                .execute(&core, &binding, &config)?
        }
        OmegaBackendSel::Stabilizer => {
            omega_backend_pauli::PauliBackend::new().execute(&core, &binding, &config)?
        }
        OmegaBackendSel::Photonic => {
            omega_backend_photonics::PhotonicsBackend::new().execute(&core, &binding, &config)?
        }
        OmegaBackendSel::Plugin { name } => {
            let registry = plugin_registry();
            let plugin = registry.find_by_name(name).ok_or_else(|| {
                omega_core::error::OmegaError::Backend(format!("no plugin backend named '{name}'"))
            })?;
            // Refuse a circuit the plugin does not declare support for, loudly.
            plugin.check_circuit_supported(&core)?;
            plugin.execute(&core, &binding, &config)?
        }
    };
    Ok((result, resolved))
}

/// Compute `⟨ψ|O|ψ⟩` of a Pauli observable on an `OmegaCircuitIR`, on the
/// backend its `backend` field selects. Returns `(value, resolved_backend)`.
/// This is the remote counterpart of a local `Backend::expectation` — it lets
/// a client get a scalar back instead of pulling the full statevector and
/// reducing it client-side. The photonic backend has no Pauli-expectation
/// contract, so it is rejected with a clear error.
pub fn expectation_quantum_ir(
    ir: &OmegaCircuitIR,
    observable: &Observable,
) -> omega_core::error::Result<(f64, OmegaBackendSel)> {
    let core = translate_to_core_ir(ir);
    let binding = ParameterBinding::new();
    let resolved = resolve_backend(ir);
    let value = match &resolved {
        OmegaBackendSel::Auto => unreachable!("resolve_backend never returns Auto"),
        OmegaBackendSel::Statevector => omega_backend_statevector::StatevectorBackend::new()
            .expectation(&core, &binding, observable)?,
        OmegaBackendSel::Mps { max_bond_dim } => {
            omega_backend_mps::MpsBackend::new(*max_bond_dim as usize)
                .expectation(&core, &binding, observable)?
        }
        OmegaBackendSel::Stabilizer => {
            omega_backend_pauli::PauliBackend::new().expectation(&core, &binding, observable)?
        }
        OmegaBackendSel::Photonic => {
            return Err(omega_core::error::OmegaError::Unsupported(
                "expectation is not defined on the photonic backend".into(),
            ))
        }
        OmegaBackendSel::Plugin { name } => {
            // The plugin ABI has no expectation fast path; report it loudly
            // rather than silently pulling a statevector and reducing here.
            return Err(omega_core::error::OmegaError::Unsupported(format!(
                "expectation is not supported over the plugin ABI (backend '{name}')"
            )));
        }
    };
    Ok((value, resolved))
}

/// Execute a Statevector circuit, routing to the OpenCL device when the
/// server is built `--features opencl` AND `OMEGA_DEVICE=opencl` resolves
/// to a usable OpenCL device. Falls back to the CPU statevector backend
/// (a) without the feature, (b) when OMEGA_DEVICE isn't opencl, or (c) when
/// no OpenCL device can be opened — never an error path for the caller.
fn exec_statevector(
    core: &CircuitIR,
    binding: &ParameterBinding,
    config: &ExecConfig,
) -> omega_core::error::Result<ExecResult> {
    #[cfg(feature = "opencl")]
    {
        use omega_core::device::DeviceKind;
        if DeviceKind::resolve(None) == DeviceKind::OpenCl {
            match omega_backend_statevector_opencl::OpenClStatevectorBackend::new() {
                Ok(backend) => {
                    eprintln!("[quantum] statevector via OpenCL device");
                    return backend.execute(core, binding, config);
                }
                Err(e) => {
                    eprintln!("[quantum] OpenCL device unavailable ({e}); CPU statevector");
                }
            }
        }
    }
    omega_backend_statevector::StatevectorBackend::new().execute(core, binding, config)
}

fn backend_name(sel: &OmegaBackendSel) -> String {
    match sel {
        OmegaBackendSel::Auto => "auto".into(),
        OmegaBackendSel::Statevector => "statevector".into(),
        OmegaBackendSel::Mps { max_bond_dim } => format!("mps(bond={})", max_bond_dim),
        OmegaBackendSel::Stabilizer => "stabilizer".into(),
        OmegaBackendSel::Photonic => "photonic".into(),
        OmegaBackendSel::Plugin { name } => format!("plugin({name})"),
    }
}

fn exec_result_to_json(result: &ExecResult, num_qubits: u32) -> serde_json::Value {
    match result {
        ExecResult::Counts(counts) => {
            let map: std::collections::HashMap<String, u32> = counts
                .iter()
                .map(|(bs, ct)| {
                    (
                        format!("{:0>width$b}", bs, width = num_qubits as usize),
                        *ct,
                    )
                })
                .collect();
            serde_json::json!({ "type": "counts", "counts": map })
        }
        ExecResult::Statevector(sv) => {
            let amps: Vec<[f64; 2]> = sv.iter().map(|c| [c.re, c.im]).collect();
            serde_json::json!({ "type": "statevector", "amplitudes": amps })
        }
        ExecResult::Probabilities(probs) => {
            serde_json::json!({ "type": "probabilities", "probabilities": probs })
        }
    }
}

// ---------- MBQC pattern IR (C1.3) ----------
//
// `quantum-core` ships one-way measurement patterns as a *sibling* wire type
// (`OmegaPatternIR`, decision C1.1) rather than a gate list, because photonic
// hardware runs the one-way model natively. These mirror
// `quantum-core/src/backends/omega.rs`'s `OmegaPatternIR`/`OmegaMeasurement`
// (u32 wire indices); the bridge converts them to the photonics backend's
// `MbqcPattern` and dispatches to its cluster-state executor.

/// One adaptive single-qubit measurement on the pattern wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OmegaMeasurement {
    pub qubit: u32,
    pub angle: f64,
    pub x_corr_from: Vec<u32>,
    pub z_corr_from: Vec<u32>,
}

/// MBQC measurement-pattern wire type (mirrors `mbqc::Pattern`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OmegaPatternIR {
    pub vertices: Vec<u32>,
    pub edges: Vec<(u32, u32)>,
    pub layers: Vec<Vec<OmegaMeasurement>>,
    pub output: Vec<u32>,
    /// Always true for now — patterns target the photonic graph-state backend.
    #[serde(default)]
    pub is_photonic: bool,
}

/// Translate the pattern wire IR into the photonics backend's `MbqcPattern`.
fn to_photonics_pattern(ir: &OmegaPatternIR) -> omega_backend_photonics::MbqcPattern {
    omega_backend_photonics::MbqcPattern {
        vertices: ir.vertices.iter().map(|&v| v as usize).collect(),
        edges: ir
            .edges
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect(),
        layers: ir
            .layers
            .iter()
            .map(|layer| {
                layer
                    .iter()
                    .map(|m| omega_backend_photonics::mbqc::Measurement {
                        qubit: m.qubit as usize,
                        angle: m.angle,
                        x_corr_from: m.x_corr_from.iter().map(|&i| i as usize).collect(),
                        z_corr_from: m.z_corr_from.iter().map(|&i| i as usize).collect(),
                    })
                    .collect()
            })
            .collect(),
        output: ir.output.iter().map(|&v| v as usize).collect(),
    }
}

/// Execute an MBQC `OmegaPatternIR` on the photonic one-way backend (C1.3).
/// Returns the canonical deterministic output statevector over the output
/// qubits — the same value `quantum-core`'s `simulate_pattern_deterministic`
/// produces (the C1.4 cross-wire equality).
pub fn execute_pattern_ir(ir: &OmegaPatternIR) -> ExecResult {
    let pattern = to_photonics_pattern(ir);
    let amps = omega_backend_photonics::simulate_mbqc_pattern(&pattern);
    ExecResult::Statevector(amps)
}

// ---------- HTTP endpoint ----------

#[derive(Deserialize)]
pub struct QuantumExecuteReq {
    pub circuit: OmegaCircuitIR,
    #[serde(default)]
    pub shots: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

pub async fn execute_quantum_route(
    Extension(claims): Extension<TokenClaims>,
    State(_state): State<SharedState>,
    Json(req): Json<QuantumExecuteReq>,
) -> impl IntoResponse {
    if let Err(resp) = check_rights(&claims, rights::EXECUTE) {
        return resp;
    }
    let num_qubits = req.circuit.num_qubits;
    let mut timer = PhaseTimer::new();
    // Price and reserve before allocating anything. `_reservation` must stay
    // alive for the whole execution — dropping it returns the budget.
    let _reservation = match admit_shape(&execute_shape(&req.circuit, req.shots)) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };
    timer.mark("admit");
    match execute_quantum_ir(&req.circuit, req.shots, req.seed) {
        Ok((result, resolved)) => {
            timer.mark("exec");
            let body = serde_json::json!({
                "backend": backend_name(&resolved),
                "result": exec_result_to_json(&result, num_qubits),
            });
            timer.mark("serialize");
            // `Server-Timing` is a standard header, so devtools and existing
            // tooling read this without bespoke support. It is what lets a
            // caller tell "the server was slow" from "the wire was slow".
            (
                StatusCode::OK,
                [("Server-Timing", timer.server_timing_header())],
                Json(body),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
pub struct QuantumPatternReq {
    pub pattern: OmegaPatternIR,
}

/// `POST /v1/quantum/execute_pattern` — execute an MBQC measurement pattern on
/// the photonic one-way backend (C1.3). Returns the canonical output
/// statevector. Requires the `EXECUTE` right.
pub async fn execute_pattern_route(
    Extension(claims): Extension<TokenClaims>,
    State(_state): State<SharedState>,
    Json(req): Json<QuantumPatternReq>,
) -> impl IntoResponse {
    if let Err(resp) = check_rights(&claims, rights::EXECUTE) {
        return resp;
    }
    let n_out = req.pattern.output.len() as u32;
    // MBQC was executing with NO admission at all. The simulator's state
    // doubles per activated vertex (`omega-backend-photonics/src/mbqc.rs:148`)
    // and both `vertices` and `output` are caller-supplied with no ceiling, so
    // a few KB of JSON — well under the body limit — asked for 2^40 amplitudes.
    // Price it as the dense array it really builds, on whichever of the two
    // counts is larger.
    let width = req.pattern.vertices.len().max(req.pattern.output.len()) as u32;
    let shape = JobShape::new(width, CostKind::DenseStatevector).returning_statevector();
    let _reservation = match admit_shape(&shape) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };
    let result = execute_pattern_ir(&req.pattern);
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "backend": "photonic",
            "result": exec_result_to_json(&result, n_out),
        })),
    )
        .into_response()
}

#[derive(Deserialize)]
pub struct QuantumExpectationReq {
    /// A single bound circuit (its gate params are concrete).
    #[serde(default)]
    pub circuit: Option<OmegaCircuitIR>,
    /// A batch of bound circuits — e.g. one per data row, each with its
    /// features baked into the gate parameters. One value is returned per
    /// circuit, in order.
    #[serde(default)]
    pub circuits: Option<Vec<OmegaCircuitIR>>,
    /// Pauli-sum observable, e.g. `"Z0"` or `"0.5*X0 + Z1 Z2"`.
    pub observable: String,
}

/// `POST /v1/quantum/expectation` — compute `⟨O⟩` of a Pauli observable on one
/// bound circuit (`circuit`) or a batch of them (`circuits`, one value each).
/// Returns scalars, so a remote QML client never has to pull the full
/// statevector and reduce it locally. Requires the `EXECUTE` right.
pub async fn expectation_quantum_route(
    Extension(claims): Extension<TokenClaims>,
    State(_state): State<SharedState>,
    Json(req): Json<QuantumExpectationReq>,
) -> impl IntoResponse {
    if let Err(resp) = check_rights(&claims, rights::EXECUTE) {
        return resp;
    }
    let bad = |msg: String| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": msg })),
        )
            .into_response()
    };

    // Exactly one of `circuit` / `circuits`.
    let circuits: Vec<OmegaCircuitIR> = match (req.circuit, req.circuits) {
        (Some(c), None) => vec![c],
        (None, Some(cs)) => cs,
        (Some(_), Some(_)) => return bad("provide either `circuit` or `circuits`, not both".into()),
        (None, None) => return bad("provide `circuit` or `circuits`".into()),
    };
    if circuits.is_empty() {
        return bad("`circuits` is empty".into());
    }
    let observable = match Observable::parse(&req.observable) {
        Ok(o) => o,
        Err(e) => return bad(format!("bad observable '{}': {e}", req.observable)),
    };

    let mut timer = PhaseTimer::new();
    // Rows run as a sequential loop below, so peak cost is the *widest* row,
    // not their sum — price that one and hold the reservation for the batch.
    // Every expectation densifies: MpsBackend::expectation and the stabilizer's
    // analytic mode both contract to a full statevector, whatever they store
    // internally. Price all rows, not just the widest.
    let _reservation = match admit_batch(&circuits, true) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };
    timer.mark("admit");

    let mut values = Vec::with_capacity(circuits.len());
    // Per-row execution cost. A batch-level total tells a search driver
    // nothing about which trial was expensive; this is the scheduling signal
    // it actually needs, and it cannot ride in a header for 256 rows.
    let mut row_ms = Vec::with_capacity(circuits.len());
    let mut backend: Option<String> = None;
    for (i, ir) in circuits.iter().enumerate() {
        let row = RowTimer::new();
        match expectation_quantum_ir(ir, &observable) {
            Ok((v, resolved)) => {
                if backend.is_none() {
                    backend = Some(backend_name(&resolved));
                }
                values.push(v);
                row_ms.push(row.finish_ms());
            }
            Err(e) => return bad(format!("circuit[{i}]: {e}")),
        }
    }
    timer.mark("exec");
    let mut timing_json = timer.to_json();
    if let Some(map) = timing_json.as_object_mut() {
        map.insert("row_ms".into(), serde_json::json!(row_ms));
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "backend": backend.unwrap_or_else(|| "statevector".to_string()),
            "values": values,
            "timing": timing_json,
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bell_ir(backend: OmegaBackendSel) -> OmegaCircuitIR {
        OmegaCircuitIR {
            num_qubits: 2,
            num_classical_bits: 0,
            is_photonic: false,
            mid_circuit_mode: OmegaMidCircuitMode::Skip,
            backend,
            ops: vec![
                OmegaGateOp {
                    gate: OmegaGateKind::H,
                    qubits: vec![0],
                    params: vec![],
                    classical_bit: None,
                    condition: None,
                },
                OmegaGateOp {
                    gate: OmegaGateKind::CX,
                    qubits: vec![0, 1],
                    params: vec![],
                    classical_bit: None,
                    condition: None,
                },
            ],
        }
    }

    /// Single-qubit `Ry(theta)` circuit (params baked in — the wire form).
    fn ry_ir(theta: f64, backend: OmegaBackendSel) -> OmegaCircuitIR {
        OmegaCircuitIR {
            num_qubits: 1,
            num_classical_bits: 0,
            is_photonic: false,
            mid_circuit_mode: OmegaMidCircuitMode::Skip,
            backend,
            ops: vec![OmegaGateOp {
                gate: OmegaGateKind::Ry,
                qubits: vec![0],
                params: vec![theta],
                classical_bit: None,
                condition: None,
            }],
        }
    }

    #[test]
    fn translate_preserves_shape() {
        let ir = bell_ir(OmegaBackendSel::Statevector);
        let core = translate_to_core_ir(&ir);
        assert_eq!(core.num_qubits, 2);
        assert_eq!(core.ops.len(), 2);
        assert!(matches!(core.circuit_type, CircuitType::GateBased));
    }

    #[test]
    fn expectation_bell_zz_is_one() {
        let ir = bell_ir(OmegaBackendSel::Statevector);
        let obs = Observable::parse("Z0 Z1").unwrap();
        let (v, sel) = expectation_quantum_ir(&ir, &obs).unwrap();
        assert!((v - 1.0).abs() < 1e-12, "⟨Z0 Z1⟩ on Bell = {v}");
        assert_eq!(sel, OmegaBackendSel::Statevector);
    }

    #[test]
    fn expectation_ry_batch_matches_cos() {
        // Each "row" is a bound Ry(theta) circuit; ⟨Z0⟩ = cos(theta). This is
        // the batch shape a remote QML client sends (one circuit per row).
        let obs = Observable::parse("Z0").unwrap();
        for theta in [0.0, 0.5, 1.3, std::f64::consts::PI] {
            let (v, _) =
                expectation_quantum_ir(&ry_ir(theta, OmegaBackendSel::Statevector), &obs).unwrap();
            assert!((v - theta.cos()).abs() < 1e-12, "θ={theta}: {v} != cos");
        }
    }

    #[test]
    fn expectation_matches_across_backends() {
        // MPS (χ high enough) and statevector agree on the same circuit.
        let obs = Observable::parse("Z0").unwrap();
        let sv = expectation_quantum_ir(&ry_ir(0.9, OmegaBackendSel::Statevector), &obs)
            .unwrap()
            .0;
        let mps =
            expectation_quantum_ir(&ry_ir(0.9, OmegaBackendSel::Mps { max_bond_dim: 8 }), &obs)
                .unwrap()
                .0;
        assert!((sv - mps).abs() < 1e-12, "sv {sv} vs mps {mps}");
    }

    #[test]
    fn expectation_photonic_is_rejected() {
        let obs = Observable::parse("Z0").unwrap();
        let mut ir = ry_ir(0.5, OmegaBackendSel::Photonic);
        ir.is_photonic = true;
        assert!(expectation_quantum_ir(&ir, &obs).is_err());
    }

    #[test]
    fn auto_picks_stabilizer_for_clifford_bell() {
        let ir = bell_ir(OmegaBackendSel::Auto);
        assert_eq!(resolve_backend(&ir), OmegaBackendSel::Stabilizer);
    }

    #[test]
    fn auto_picks_statevector_for_small_non_clifford() {
        let mut ir = bell_ir(OmegaBackendSel::Auto);
        ir.ops.push(OmegaGateOp {
            gate: OmegaGateKind::T,
            qubits: vec![0],
            params: vec![],
            classical_bit: None,
            condition: None,
        });
        assert_eq!(resolve_backend(&ir), OmegaBackendSel::Statevector);
    }

    #[test]
    fn auto_picks_mps_for_large_non_clifford() {
        let mut ir = bell_ir(OmegaBackendSel::Auto);
        ir.num_qubits = 20;
        ir.ops.push(OmegaGateOp {
            gate: OmegaGateKind::T,
            qubits: vec![0],
            params: vec![],
            classical_bit: None,
            condition: None,
        });
        assert!(matches!(
            resolve_backend(&ir),
            OmegaBackendSel::Mps { max_bond_dim: 64 }
        ));
    }

    #[test]
    fn auto_picks_photonic_when_flagged() {
        let mut ir = bell_ir(OmegaBackendSel::Auto);
        ir.is_photonic = true;
        assert_eq!(resolve_backend(&ir), OmegaBackendSel::Photonic);
    }

    #[test]
    fn explicit_statevector_execution_returns_counts() {
        let ir = bell_ir(OmegaBackendSel::Statevector);
        let (result, resolved) = execute_quantum_ir(&ir, Some(256), Some(42)).unwrap();
        assert_eq!(resolved, OmegaBackendSel::Statevector);
        match result {
            ExecResult::Counts(_) => {}
            other => panic!("expected Counts, got {:?}", other),
        }
    }

    #[test]
    fn explicit_stabilizer_execution_on_clifford_circuit() {
        let ir = bell_ir(OmegaBackendSel::Stabilizer);
        let (_result, resolved) = execute_quantum_ir(&ir, Some(64), Some(7)).unwrap();
        assert_eq!(resolved, OmegaBackendSel::Stabilizer);
    }

    #[test]
    fn auto_dispatch_executes_without_error() {
        let ir = bell_ir(OmegaBackendSel::Auto);
        let (_result, resolved) = execute_quantum_ir(&ir, Some(64), Some(1)).unwrap();
        // Auto → Stabilizer for a Clifford circuit.
        assert_eq!(resolved, OmegaBackendSel::Stabilizer);
    }

    #[test]
    fn json_roundtrip_wire_format() {
        let ir = bell_ir(OmegaBackendSel::Mps { max_bond_dim: 32 });
        let s = serde_json::to_string(&ir).unwrap();
        assert!(s.contains("\"Mps\""));
        let parsed: OmegaCircuitIR = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.ops.len(), 2);
        assert_eq!(parsed.backend, OmegaBackendSel::Mps { max_bond_dim: 32 });
    }

    #[test]
    fn plugin_backend_sel_serde_roundtrip() {
        let sel = OmegaBackendSel::Plugin {
            name: "refplugin".to_string(),
        };
        let s = serde_json::to_string(&sel).unwrap();
        assert!(s.contains("Plugin"));
        assert!(s.contains("refplugin"));
        let parsed: OmegaBackendSel = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed, sel);
    }

    #[test]
    fn backend_name_renders_plugin() {
        let sel = OmegaBackendSel::Plugin {
            name: "acme".to_string(),
        };
        assert_eq!(backend_name(&sel), "plugin(acme)");
    }

    #[test]
    fn auto_never_selects_a_plugin() {
        // Auto resolution must never pick a plugin — a client opts in by name.
        let ir = bell_ir(OmegaBackendSel::Auto);
        assert!(!matches!(
            resolve_backend(&ir),
            OmegaBackendSel::Plugin { .. }
        ));
    }

    #[test]
    fn expectation_over_plugin_is_rejected() {
        // The plugin ABI has no expectation fast path — loud error, not a
        // silent statevector reduction.
        let obs = Observable::parse("Z0").unwrap();
        let ir = ry_ir(
            0.5,
            OmegaBackendSel::Plugin {
                name: "refplugin".to_string(),
            },
        );
        let err = expectation_quantum_ir(&ir, &obs).unwrap_err();
        assert!(format!("{err}").contains("plugin"));
    }

    #[test]
    fn execute_unknown_plugin_errors_loudly() {
        // No plugin by this name is loaded in the test environment.
        let ir = bell_ir(OmegaBackendSel::Plugin {
            name: "definitely-not-loaded-xyz".to_string(),
        });
        let err = execute_quantum_ir(&ir, Some(16), Some(1)).unwrap_err();
        assert!(format!("{err}").contains("no plugin backend named"));
    }
}
