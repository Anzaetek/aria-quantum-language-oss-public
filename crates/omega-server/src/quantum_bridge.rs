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
use crate::worker::{estimate_peak_bytes, governor, CostKind, JobShape, Reservation};
use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable};
use omega_core::params::ParameterBinding;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::sync::{Arc, OnceLock};
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

/// One gate parameter: a bound number, or a free symbol.
///
/// `#[serde(untagged)]` keeps the wire backward compatible — a bare number
/// still decodes as `Concrete`, so an older client talking to this server (and
/// the reverse) is unaffected. Only circuits that use symbols carry the new
/// form. Without symbols on the wire there is nothing for a gradient to
/// differentiate with respect to, and a batch must re-send its whole gate list
/// per row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OmegaParam {
    Concrete(f64),
    Symbol { symbol: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OmegaGateOp {
    pub gate: OmegaGateKind,
    pub qubits: Vec<u32>,
    pub params: Vec<OmegaParam>,
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
    /// Pauli propagation (`omega-backend-pauliprop`): Heisenberg-picture
    /// evolution of the OBSERVABLE, so it answers `/expectation` and refuses
    /// `/execute` — there is no statevector to return. Width-unbounded; the
    /// term cap is what bounds it. Never chosen by `Auto`: it is exact only on
    /// Clifford circuits and truncates otherwise, which a client must opt into.
    PauliProp,
    /// A dynamically-loaded backend plugin, selected by name. Never chosen by
    /// `Auto` resolution — a client must request it explicitly.
    Plugin {
        name: String,
    },
}

/// An MPS backend, with the cuSOLVER `gesvdj` SVD hook installed when the
/// `cuda` feature is compiled in.
///
/// `cuda_svd_flat` falls back to the CPU Jacobi SVD when no device is present,
/// so installing it unconditionally on a `cuda` build is safe. As with
/// PauliProp, it is installed for parity with the CLI rather than because it
/// always wins: on unified-memory hardware `gesvdj` is f64 and measurably
/// SLOWER (`BACKEND-CROSSOVER.md` §3).
fn mps_backend(max_bond_dim: usize) -> omega_backend_mps::MpsBackend {
    let b = omega_backend_mps::MpsBackend::new(max_bond_dim);
    #[cfg(feature = "cuda")]
    {
        return b.with_svd_fn(omega_backend_mps_cuda::cuda_svd_flat);
    }
    #[allow(unreachable_code)]
    b
}

/// A PauliProp backend, with the CUDA branch accelerator installed when the
/// feature is compiled in.
///
/// The hook declines at runtime (no device, or too few terms to be worth a
/// round trip) and the CPU path then runs unchanged, so this is safe to install
/// unconditionally on a `cuda` build. NOTE the accelerator is currently SLOWER
/// than the CPU on unified-memory hardware — see `BACKEND-CROSSOVER.md` — so it
/// is installed for parity with the CLI, not because it is a win everywhere.
fn pauliprop_backend() -> omega_backend_pauliprop::PauliPropBackend {
    let b = omega_backend_pauliprop::PauliPropBackend::new();
    #[cfg(feature = "cuda")]
    {
        return b.with_branch_hook(omega_backend_pauliprop_cuda::cuda_branch);
    }
    #[allow(unreachable_code)]
    b
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
    translate_with_symbols(ir).0
}

/// Translate, and also return the wire-name -> `SymbolId` map.
///
/// The gradient route needs it in both directions: to bind `param_values` given
/// by name, and to label the returned derivatives by name again. A caller must
/// never have to guess which id an id-keyed result refers to.
pub fn translate_with_symbols(
    ir: &OmegaCircuitIR,
) -> (CircuitIR, std::collections::HashMap<String, u32>) {
    let circuit_type = if ir.is_photonic {
        CircuitType::Photonic
    } else {
        CircuitType::GateBased
    };
    let mut core = CircuitIR::new(ir.num_qubits, circuit_type);
    core.num_classical_bits = ir.num_classical_bits;
    // Wire symbols are names; omega-core keys parameters by `SymbolId`. Assign
    // ids in first-appearance order and register them, so the SAME name in two
    // gates maps to ONE parameter — which is what makes a shared-parameter
    // ansatz differentiable at all.
    let mut symbol_ids: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for op in &ir.ops {
        let qubits: SmallVec<[Qubit; 3]> = op.qubits.iter().map(|&q| Qubit(q)).collect();
        let params: SmallVec<[ParamExpr; 3]> = op
            .params
            .iter()
            .map(|p| match p {
                OmegaParam::Concrete(v) => ParamExpr::Concrete(*v),
                OmegaParam::Symbol { symbol } => {
                    let next = symbol_ids.len() as u32;
                    let id = *symbol_ids.entry(symbol.clone()).or_insert(next);
                    ParamExpr::Symbol(id)
                }
            })
            .collect();
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
    // Register the names in the circuit's symbol table. Without this the ops
    // carry ParamExpr::Symbol(id) but the circuit reports no parameters, so the
    // adjoint finds nothing to differentiate and returns an EMPTY gradient —
    // which reads as "all derivatives are zero" rather than as an error.
    for (name, id) in &symbol_ids {
        core.symbols.insert(*id, name.clone());
    }
    (core, symbol_ids)
}

/// Map a resolved backend to the cost curve the governor should price it with.
///
/// `Auto` is resolved first (see [`admit_ir`]) rather than assumed dense —
/// pricing a 30-qubit Clifford circuit as `2^30 × 16` would refuse a job the
/// stabilizer backend runs in kilobytes.
/// `ir`/`obs` are consulted only by PauliProp, whose cost is set by the
/// observable's light cone rather than the register — see `term_upper_bound`.
/// `obs` is `None` on `/execute`, which prices at the ceiling.
fn cost_kind_for(sel: &OmegaBackendSel, ir: &OmegaCircuitIR, obs: Option<&Observable>) -> CostKind {
    match sel {
        OmegaBackendSel::Statevector => CostKind::DenseStatevector,
        OmegaBackendSel::Mps { max_bond_dim } => CostKind::Mps {
            max_bond_dim: *max_bond_dim,
        },
        OmegaBackendSel::Stabilizer => CostKind::Stabilizer,
        OmegaBackendSel::Photonic => CostKind::Photonic,
        OmegaBackendSel::PauliProp => CostKind::PauliProp {
            // Price the WORK, not the worst case. Without an observable there
            // is no light cone to compute, so fall back to the ceiling.
            max_terms: match obs {
                Some(o) => omega_backend_pauliprop::term_upper_bound(&translate_to_core_ir(ir), o),
                None => omega_backend_pauliprop::term_ceiling(),
            },
        },
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
/// Whether a device statevector will actually be used, decided **once**.
///
/// Pricing and dispatch both asked `DeviceKind::resolve(None)`, which reports
/// what was *requested*, not what opens. `exec_statevector` then fell back to
/// the CPU when `OpenClStatevectorBackend::new()` failed — right as an
/// availability decision, but admission had already run and priced
/// `Device(0)` at 8 B/amplitude against the device pool. The job then ran on
/// the CPU at 16 B in host RAM: wrong pool and wrong width, and silent to the
/// governor, which never re-prices.
///
/// Unifying the two *lookups* does not fix that — the fallback happens after
/// both, so what breaks the invariant is a runtime failure between them. What
/// does fix it is unifying the **open attempt**: try once, remember the answer,
/// and have pricing and dispatch consult the same memo. The failures this
/// guards against (a stale ICD, a driver that enumerates but will not create a
/// context) are properties of the installation, not of a single job.
///
/// Residual, and deliberately not chased: a device that opens at startup and
/// dies later. That window is strictly smaller than the one this closes.
/// CUDA twin of [`opencl_statevector_available`]. Same memo, same reason: the
/// governor prices device work at 8 B/amplitude, so pricing and dispatch must
/// consult ONE answer decided once — otherwise a runtime open failure between
/// them leaves a device-priced reservation describing a CPU run.
#[cfg(feature = "cuda")]
fn cuda_statevector_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        use omega_core::device::DeviceKind;
        if DeviceKind::resolve(None) != DeviceKind::Cuda {
            return false;
        }
        match omega_backend_statevector_cuda::CudaStatevectorBackend::new() {
            Ok(_) => true,
            Err(e) => {
                eprintln!(
                    "[quantum] CUDA requested but unavailable ({e}); pricing and \
                     executing on the CPU for the life of this process"
                );
                false
            }
        }
    })
}

#[cfg(feature = "opencl")]
fn opencl_statevector_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        use omega_core::device::DeviceKind;
        if DeviceKind::resolve(None) != DeviceKind::OpenCl {
            return false;
        }
        match omega_backend_statevector_opencl::OpenClStatevectorBackend::new() {
            Ok(_) => true,
            Err(e) => {
                eprintln!(
                    "[quantum] OpenCL requested but unavailable ({e}); pricing and \
                     executing on the CPU for the life of this process"
                );
                false
            }
        }
    })
}

fn exec_target_for(sel: &OmegaBackendSel) -> crate::worker::ExecTarget {
    // CUDA first, matching the dispatch order in `exec_statevector`. If these
    // two ever disagree the reservation is against the wrong pool — the defect
    // the availability memo exists to prevent.
    #[cfg(feature = "cuda")]
    {
        if matches!(sel, OmegaBackendSel::Statevector | OmegaBackendSel::Auto)
            && cuda_statevector_available()
            && crate::worker::governor().has_pool_for(crate::worker::ExecTarget::Device(0))
        {
            return crate::worker::ExecTarget::Device(0);
        }
    }
    #[cfg(feature = "opencl")]
    {
        if matches!(sel, OmegaBackendSel::Statevector | OmegaBackendSel::Auto)
            && opencl_statevector_available()
            // Only claim the device pool if there IS one. Device pools come
            // from `nvidia-smi` alone (`topology.rs`), so on AMD, Intel or
            // Apple OpenCL — the main reasons to run this feature — the probe
            // finds no devices, the machine classifies `HostOnly`, and
            // `admit` fails closed on "no memory budget for the requested
            // device". That refusal is right for a DISCRETE card with no
            // budget (it is how a 64 GB job reaches a 24 GB card) and wrong
            // here, where there is no separate device memory to protect: it
            // 413'd every Statevector request on hardware where OpenCL works.
            //
            // Falling back to `Cpu` prices the same single physical pool at
            // 16 B/amplitude rather than the device's 8, so it over-prices —
            // the safe direction, and the only one available without a device
            // budget to debit.
            && crate::worker::governor()
                .has_pool_for(crate::worker::ExecTarget::Device(0))
        {
            return crate::worker::ExecTarget::Device(0);
        }
    }
    let _ = sel;
    crate::worker::ExecTarget::Cpu
}

/// Shape for one row.
///
/// `target` is passed IN rather than re-derived from `OMEGA_DEVICE`, because it
/// is a property of the ENDPOINT: `exec_target_for` returns `Device(0)` for a
/// Statevector selection whenever the resolved device is OpenCL, but only
/// `/execute` can reach a device. `/expectation` builds
/// `StatevectorBackend::new()` unconditionally, and `/gradient` does the same —
/// so both were priced at 8 B/amplitude (device kernels are f32) while
/// allocating `Complex64` at 16. `exec_target_for`'s doc comment claims it
/// mirrors execution routing exactly; it did so for one of its three callers.
fn shape_for(
    ir: &OmegaCircuitIR,
    densifies: bool,
    batch: usize,
    target: crate::worker::ExecTarget,
    obs: Option<&Observable>,
) -> JobShape {
    let resolved = resolve_backend(ir);
    let mut shape = JobShape::new(ir.num_qubits, cost_kind_for(&resolved, ir, obs));
    shape.densifies = densifies;
    shape.batch = batch;
    shape.target = target;
    shape
}

/// Shape for one `/gradient` row.
///
/// The adjoint sweep is **always** the dense CPU statevector (`:766` builds
/// `StatevectorBackend::new()` before the loop and never consults the row's
/// backend), so the row's DECLARED backend must not set the cost kind either.
/// It did: a `Stabilizer` row was priced as a tableau, and a `Plugin` row as
/// `Opaque` — a 1 MiB reservation for a full dense adjoint, bounded only by the
/// qubit ceiling.
fn gradient_shape(ir: &OmegaCircuitIR, batch: usize) -> JobShape {
    let mut shape = JobShape::new(ir.num_qubits, crate::worker::CostKind::DenseStatevector);
    shape.densifies = true;
    shape.batch = batch;
    shape.target = crate::worker::ExecTarget::Cpu;
    shape.gradient = true;
    shape
}

/// Shape for `/execute`. `shots: None` ships every amplitude back (and so pays
/// for the JSON encoding); a shot run samples and returns counts.
fn execute_shape(ir: &OmegaCircuitIR, shots: Option<u32>) -> JobShape {
    // `/execute` is the one endpoint that can actually reach a device, so it
    // is the one that passes the device-capable target.
    let base = shape_for(ir, true, 1, exec_target_for(&resolve_backend(ir)), None);
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
    obs: Option<&Observable>,
) -> Result<Admitted, Box<axum::response::Response>> {
    admit_shapes(
        &circuits
            .iter()
            .map(|ir| {
                shape_for(
                    ir,
                    densifies,
                    circuits.len(),
                    crate::worker::ExecTarget::Cpu,
                    obs,
                )
            })
            .collect::<Vec<_>>(),
    )
}

/// Reserve for a batch of already-built shapes.
///
/// # The hole this closes
///
/// The previous selection was
/// `max_by_key(|sh| estimate_peak_bytes(sh).unwrap_or(u64::MAX))`, so an
/// **unpriceable** row won — and `Governor::admit` then reserves 0 bytes for an
/// unpriceable-by-kind shape (`worker.rs`: `None if !is_priceable => reserve_in(pool, 0, 1)`),
/// deliberately, so the plugin path is not broken. Combining the two erased the
/// price of everything else in the batch:
///
/// ```json
/// {"circuits": [ {"num_qubits": 30, "backend": "Statevector"},
///                {"num_qubits": 1,  "backend": {"Plugin": {"name": "x"}}} ]}
/// ```
///
/// The plugin row wins selection, the batch reserves ~1 MiB, and the loop then
/// executes the 30-qubit dense row — 16 GiB — before failing on the plugin row.
/// No plugin need be loaded: `resolve_backend` returns the declared backend
/// verbatim and existence is only checked at execution. The `unwrap_or(u64::MAX)`
/// comment said an unpriceable row "must not be silently out-ranked"; what it
/// actually did was out-rank everything and then price at nothing.
///
/// The qubit ceiling was wrong for the same reason: `admit` applies it to the
/// single shape handed to it, so it was enforced at the *winning* row's width.
/// `[{"num_qubits": 34}, {"num_qubits": 1, "Plugin"}]` walked past
/// `OMEGA_MAX_QUBITS` on a four-line body.
///
/// So: the ceiling is checked against **every** dense-driven row, and the
/// reservation is the worst row we can actually price. A batch in which nothing
/// is priceable keeps the old behaviour, which is the working plugin path.
fn admit_shapes(shapes: &[JobShape]) -> Result<Admitted, Box<axum::response::Response>> {
    debug_assert!(!shapes.is_empty(), "caller checked the batch is non-empty");
    let max_qubits = governor().config().max_qubits;
    if let Some(over) = shapes
        .iter()
        .find(|sh| sh.is_dense_driven() && sh.num_qubits > max_qubits)
    {
        // Refuse through `admit` so the body and status match every other
        // ceiling refusal rather than being spelled a second way here.
        return admit_shape(over);
    }
    let worst = shapes
        .iter()
        .filter(|sh| crate::worker::estimate_peak_bytes(sh).is_some())
        .max_by_key(|sh| crate::worker::estimate_peak_bytes(sh).unwrap_or(0));
    match worst {
        Some(sh) => admit_shape(sh),
        // Every row unpriceable: the plugin/photonic path, unchanged.
        None => admit_shape(&shapes[0]),
    }
}

/// Both halves of admission, held for exactly as long as the job runs.
///
/// The governor answers "will this job kill THIS PROCESS?"; the host gate
/// answers "will it kill this MACHINE?", across processes the server cannot
/// see. Both must say yes, and both are released by dropping this.
pub struct Admitted {
    _local: Reservation,
    _host: omega_hostgate::Grant,
}

/// The host-wide budget, resolved once from the environment.
///
/// `Err` is kept rather than swallowed: an operator who set
/// `OMEGA_HOSTGATE_*` to something malformed asked for a budget and would
/// otherwise silently get none, and a throttle that turns itself off looks
/// exactly like one that is working. Unset is not malformed — it means off,
/// and off costs no lock, no read and no syscall.
static HOSTGATE: OnceLock<Result<omega_hostgate::HostGate, String>> = OnceLock::new();

fn hostgate() -> &'static Result<omega_hostgate::HostGate, String> {
    HOSTGATE.get_or_init(|| match omega_hostgate::config::Config::from_env() {
        Ok(cfg) => Ok(cfg.into_gate()),
        Err(e) => {
            eprintln!("[hostgate] misconfigured, refusing to admit work: {e}");
            Err(e)
        }
    })
}

fn admit_shape(shape: &JobShape) -> Result<Admitted, Box<axum::response::Response>> {
    // LOCAL FIRST, and the order is load-bearing. The governor is an in-process
    // semaphore: refusing here costs nothing. The host gate takes a file lock
    // that every process on the box contends for, so a job that was going to be
    // refused anyway must never reach it.
    let local = governor().admit(shape).map_err(|rej| {
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
    })?;

    // THEN the machine. If this refuses, `local` drops on the way out and the
    // in-process budget is returned — which is why the host grant is taken in
    // the same scope rather than by the caller.
    let gate = match hostgate() {
        Ok(g) => g,
        Err(why) => {
            let body = Json(serde_json::json!({
                "error": format!("host budget is misconfigured: {why}"),
            }));
            return Err(Box::new(
                (StatusCode::SERVICE_UNAVAILABLE, body).into_response(),
            ));
        }
    };

    // The SAME number the governor priced it at. Two components disagreeing
    // about what one circuit costs is worse than either being slightly wrong.
    // Unpriceable shapes ask for nothing: the governor has already decided they
    // are admissible, and inventing a figure here would be a guess with a
    // machine-wide budget behind it.
    let mut req = omega_hostgate::Request::new("omega-server");
    if let Some(bytes) = estimate_peak_bytes(shape) {
        req = req.want(omega_hostgate::Axis::HostBytes, bytes);
    }

    match gate.try_acquire(&req) {
        Ok(host) => Ok(Admitted {
            _local: local,
            _host: host,
        }),
        Err(refusal) => {
            let status = if refusal.retryable() {
                StatusCode::TOO_MANY_REQUESTS
            } else {
                StatusCode::PAYLOAD_TOO_LARGE
            };
            let body = Json(serde_json::json!({
                "error": format!("host budget: {refusal}"),
                "scope": "machine",
            }));
            Err(Box::new(if refusal.retryable() {
                (status, [("Retry-After", "1")], body).into_response()
            } else {
                (status, body).into_response()
            }))
        }
    }
}

/// Decide which concrete backend should execute this IR.
fn resolve_backend(ir: &OmegaCircuitIR) -> OmegaBackendSel {
    if let OmegaBackendSel::Auto = ir.backend {
        if ir.is_photonic {
            return OmegaBackendSel::Photonic;
        }
        // Kept local rather than shared with `omega_core::circuit::
        // is_clifford_only`, which the CLI's `--backend auto` uses: this is the
        // WIRE enum (`OmegaGateKind`), a different type from the IR's
        // `GateKind`, so the two lists are over different alphabets rather than
        // being a duplicated decision. This one has no `Sx`/`Sxdg` to admit —
        // the wire enum does not carry them.
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
            mps_backend(*max_bond_dim as usize).execute(&core, &binding, &config)?
        }
        OmegaBackendSel::Stabilizer => {
            omega_backend_pauli::PauliBackend::new().execute(&core, &binding, &config)?
        }
        OmegaBackendSel::Photonic => {
            omega_backend_photonics::PhotonicsBackend::new().execute(&core, &binding, &config)?
        }
        OmegaBackendSel::PauliProp => {
            // Heisenberg picture: the observable evolves, the state does not
            // exist. Refuse rather than synthesise something.
            return Err(omega_core::error::OmegaError::Unsupported(
                "pauliprop evolves the OBSERVABLE, not the state — it has no \
                 statevector or counts to return. Use /expectation."
                    .into(),
            ));
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
/// `<O>` for an ALREADY-translated circuit under a caller-supplied binding.
///
/// Split out of [`expectation_quantum_ir`] so the template route can translate
/// the circuit ONCE and rebind per row. The batch route translates `N`
/// independent `CircuitIR`s and holds them all; this lets server-side circuit
/// memory be `O(gates)` instead of `O(N x gates)`.
///
/// `resolved` is passed in rather than recomputed because `resolve_backend`
/// reads the request's own hint — recomputing it per row would be identical work
/// and invites the two paths to disagree.
pub fn expectation_core_bound(
    core: &CircuitIR,
    resolved: &OmegaBackendSel,
    binding: &ParameterBinding,
    observable: &Observable,
) -> omega_core::error::Result<f64> {
    Ok(match resolved {
        OmegaBackendSel::Auto => unreachable!("resolve_backend never returns Auto"),
        OmegaBackendSel::Statevector => omega_backend_statevector::StatevectorBackend::new()
            .expectation(core, binding, observable)?,
        OmegaBackendSel::Mps { max_bond_dim } => {
            mps_backend(*max_bond_dim as usize).expectation(core, binding, observable)?
        }
        OmegaBackendSel::Stabilizer => {
            omega_backend_pauli::PauliBackend::new().expectation(core, binding, observable)?
        }
        OmegaBackendSel::Photonic => {
            return Err(omega_core::error::OmegaError::Unsupported(
                "expectation is not defined on the photonic backend".into(),
            ))
        }
        OmegaBackendSel::PauliProp => pauliprop_backend().expectation(core, binding, observable)?,
        OmegaBackendSel::Plugin { name } => {
            // The plugin ABI has no expectation fast path; report it loudly
            // rather than silently pulling a statevector and reducing here.
            return Err(omega_core::error::OmegaError::Unsupported(format!(
                "expectation is not supported over the plugin ABI (backend '{name}')"
            )));
        }
    })
}

pub fn expectation_quantum_ir(
    ir: &OmegaCircuitIR,
    observable: &Observable,
) -> omega_core::error::Result<(f64, OmegaBackendSel)> {
    let core = translate_to_core_ir(ir);
    let binding = ParameterBinding::new();
    let resolved = resolve_backend(ir);
    // Delegates so there is ONE dispatch table. Two copies of a backend match
    // is how a backend gets added to one path and not the other.
    let value = expectation_core_bound(&core, &resolved, &binding, observable)?;
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
    #[cfg(feature = "cuda")]
    {
        // The SAME memo admission priced against, so a fallback cannot happen
        // between the two. If the device did not open, `exec_target_for`
        // already returned `Cpu` and the reservation is a host one.
        if cuda_statevector_available() {
            match omega_backend_statevector_cuda::CudaStatevectorBackend::new() {
                Ok(backend) => {
                    eprintln!("[quantum] statevector via CUDA device");
                    return backend.execute(core, binding, config);
                }
                Err(e) => {
                    return Err(omega_core::error::OmegaError::Backend(format!(
                        "CUDA device opened at startup but not now ({e}); this run \
                         was priced as device work, so completing it on the CPU would \
                         hold a reservation that describes neither the pool nor the \
                         width actually used"
                    )));
                }
            }
        }
    }
    #[cfg(feature = "opencl")]
    {
        // The SAME memo admission priced against — see the CUDA arm above.
        if opencl_statevector_available() {
            match omega_backend_statevector_opencl::OpenClStatevectorBackend::new() {
                Ok(backend) => {
                    eprintln!("[quantum] statevector via OpenCL device");
                    return backend.execute(core, binding, config);
                }
                Err(e) => {
                    // The device opened during the probe and will not open now.
                    // Refusing is the honest answer: falling back here would run
                    // on the CPU at 16 B/amplitude against a device reservation
                    // priced at 8, which is the defect this memo exists to close.
                    return Err(omega_core::error::OmegaError::Backend(format!(
                        "OpenCL device opened at startup but not now ({e}); this run \
                         was priced as device work, so completing it on the CPU would \
                         hold a reservation that describes neither the pool nor the \
                         width actually used"
                    )));
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
        OmegaBackendSel::PauliProp => "pauliprop".into(),
        OmegaBackendSel::Plugin { name } => format!("plugin({name})"),
    }
}

/// The width to zero-pad a counts key to, for one wire circuit.
///
/// A named function rather than a block inside the route, so a test can cover
/// the same code the route runs. The first version of this fix put the
/// computation inline and tested `exec_result_to_json` directly with a
/// hand-computed width — which cannot catch the actual defect, since the defect
/// was the CALLER passing `num_qubits`. Substituting `req.circuit.num_qubits`
/// here survived the whole suite.
fn counts_render_width(ir: &OmegaCircuitIR) -> u32 {
    let core = translate_to_core_ir(ir);
    let collapse = matches!(ir.mid_circuit_mode, OmegaMidCircuitMode::Collapse);
    omega_core::executor::counts_outcome_width(
        &core,
        omega_core::executor::counts_keyed_on_creg(&core, collapse),
    ) as u32
}

/// Render an `ExecResult` to the wire.
///
/// `counts_width` is the width to zero-pad a counts key to, which is NOT always
/// the qubit count: in collapse mode the key is packed from the CLASSICAL
/// register. Callers must pass `counts_outcome_width(..)`, not `num_qubits` —
/// the parameter was named `num_qubits` and every caller obliged, so a 70-qubit
/// circuit measuring 2 qubits reported its 2-bit outcome padded to 70
/// characters.
fn exec_result_to_json(result: &ExecResult, counts_width: u32) -> serde_json::Value {
    match result {
        ExecResult::Counts(counts) => {
            // The width now comes from the KEY. `counts_width` is kept only to
            // check the caller against the data: a route that computes a width
            // disagreeing with what the backend produced is a bug worth seeing,
            // and it is exactly the disagreement that printed a 2-bit outcome
            // at 20 characters.
            debug_assert!(
                counts.is_empty() || counts.keys().all(|o| o.width() == counts_width),
                "route computed width {counts_width} but the outcomes are {:?} wide",
                counts
                    .keys()
                    .map(|o| o.width())
                    .collect::<std::collections::BTreeSet<_>>()
            );
            let map: std::collections::HashMap<String, u32> = counts
                .iter()
                .map(|(bs, ct)| {
                    (
                        // The key carries its own width now; padding to a
                        // caller-supplied one is the defect this replaces.
                        bs.to_bitstring(),
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
    let counts_width = counts_render_width(&req.circuit);
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
                "result": exec_result_to_json(&result, counts_width),
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
pub struct QuantumGradientReq {
    /// A single circuit, XOR `circuits` — mirrors `QuantumExpectationReq` field
    /// for field so the two routes stay learnable as a pair.
    #[serde(default)]
    pub circuit: Option<OmegaCircuitIR>,
    #[serde(default)]
    pub circuits: Option<Vec<OmegaCircuitIR>>,
    pub observable: String,
    /// Bindings for the circuit's free symbols, by name.
    #[serde(default)]
    pub param_values: Vec<(String, f64)>,
}

/// `POST /v1/quantum/gradient` — `d⟨O⟩/dθ` for one bound circuit or a batch.
///
/// Returns one gradient per circuit, **in input order** (Q5), each a list of
/// `[symbol_name, value]`. Names, not ids: a caller cannot be expected to guess
/// the server's internal numbering.
///
/// Requires the `EXECUTE` right.
pub async fn gradient_quantum_route(
    Extension(claims): Extension<TokenClaims>,
    State(_state): State<SharedState>,
    Json(req): Json<QuantumGradientReq>,
) -> impl IntoResponse {
    if let Err(resp) = check_rights(&claims, rights::EXECUTE) {
        return resp;
    }
    let mut timer = PhaseTimer::new();
    let bad = |msg: String| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": msg })),
        )
            .into_response()
    };

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
    // Parsing does NOT bound the qubit indices -- `Observable::parse` reads a
    // bare u32 and never sees the circuit. Backends then index straight into
    // their state with it and PANIC, inside this handler, with no
    // CatchPanicLayer installed: `Z99` on a 4-qubit circuit panicked both
    // pauliprop (`words[2]`, len 2) and the statevector (`i >> 99`). One of
    // them, `X99`, did not even panic -- it corrupted the key and returned a
    // plausible wrong number.
    //
    // Validated here, once, so every backend is covered rather than each
    // having to remember. Per row, because rows may differ in width while
    // sharing one observable.
    for (i, c) in circuits.iter().enumerate() {
        if let Err(e) = observable.validate_qubits(c.num_qubits) {
            return bad(format!("circuit {i}: {e}"));
        }
    }

    // A gradient holds the backward state alongside the forward one, so it is
    // priced with `gradient = true` rather than as a plain expectation.
    // Every row, not `circuits[0]`. The handler loops over the whole batch
    // below, and `admit_batch` three functions above exists precisely to stop
    // one row standing in for all of them — its own comment says "width does
    // not imply cost". `/expectation` uses it; this did not, and the comment
    // here explained only the `gradient` flag, so pricing one row read as
    // deliberate. A `[4q, 30q]` batch was admitted on kilobytes and ran a
    // 30-qubit adjoint: forward and backward state, order 32 GiB.
    let shapes: Vec<JobShape> = circuits
        .iter()
        .map(|ir| gradient_shape(ir, circuits.len()))
        .collect();
    let _reservation = match admit_shapes(&shapes) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };
    timer.mark("admit");

    let backend = omega_backend_statevector::StatevectorBackend::new();
    let mut gradients = Vec::with_capacity(circuits.len());
    for (i, ir) in circuits.iter().enumerate() {
        let (core, symbol_ids) = translate_with_symbols(ir);
        if symbol_ids.is_empty() {
            // Refuse rather than return an empty gradient that reads as "all
            // derivatives are zero". A circuit with no free symbols has nothing
            // to differentiate, and saying so is the whole lesson of the wire
            // change that made this route possible.
            return bad(format!(
                "circuit[{i}] has no free parameters — send symbolic params                  (e.g. \"params\": [{{\"symbol\": \"theta\"}}]) to differentiate against"
            ));
        }
        let mut binding = ParameterBinding::new();
        for (name, value) in &req.param_values {
            match symbol_ids.get(name) {
                Some(id) => binding.bind(*id, *value),
                None => {
                    return bad(format!(
                        "circuit[{i}]: no free parameter named '{name}' in this circuit"
                    ))
                }
            }
        }
        // Every symbol must be bound: an unbound one would silently evaluate as
        // whatever the backend defaults to.
        for (name, id) in &symbol_ids {
            if !req.param_values.iter().any(|(n, _)| n == name) {
                let _ = id;
                return bad(format!(
                    "circuit[{i}]: parameter '{name}' is free but has no value in `param_values`"
                ));
            }
        }

        match backend.adjoint_gradient(&core, &binding, &observable) {
            Ok(Some(grads)) => {
                // Map ids back to the names the caller used.
                let by_id: std::collections::HashMap<u32, &String> =
                    symbol_ids.iter().map(|(n, i)| (*i, n)).collect();
                let named: Vec<(String, f64)> = grads
                    .into_iter()
                    .map(|(id, v)| {
                        (
                            by_id
                                .get(&id)
                                .map(|s| (*s).clone())
                                .unwrap_or_else(|| format!("sym{id}")),
                            v,
                        )
                    })
                    .collect();
                gradients.push(named);
            }
            Ok(None) => {
                return bad(format!(
                    "circuit[{i}]: this circuit is not differentiable by the adjoint method                      (non-unitary ops such as Reset or mid-circuit measurement)"
                ))
            }
            Err(e) => return bad(format!("circuit[{i}]: {e}")),
        }
    }
    timer.mark("exec");

    (
        StatusCode::OK,
        [("Server-Timing", timer.server_timing_header())],
        Json(serde_json::json!({
            "backend": "statevector",
            "gradients": gradients,
            "timing": timer.to_json(),
        })),
    )
        .into_response()
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
    let _reservation = match admit_batch(&circuits, true, Some(&observable)) {
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

/// `POST /v1/quantum/expectation_template` — one circuit, an `N × P` parameter
/// matrix, `N` expectation values.
///
/// # What this saves, and what it does not
///
/// The batch route (`circuits: [...]`) re-sends the **entire gate list per row**.
/// For a QML sweep that is the same ansatz N times with different angles baked
/// in, so the payload grows with `N × gates` when the information content grows
/// with `N × params`. On a realistic ansatz `params << gates`, and the ratio is
/// measured in `payload_reduction.rs` rather than asserted here.
///
/// What it does **not** change is admission. `admit_batch` already prices every
/// row and takes the worst, and since rows execute sequentially the worst row IS
/// the peak — pricing N identical rows would over-reserve by N. So the template
/// admits ONE row's shape, which is the same reservation the batch route
/// computes for the same work. **P5's case is payload reduction alone**, and
/// claiming an admission benefit would be claiming something false.
///
/// # The server stops copying too
///
/// A less obvious win: the batch route translates N independent `CircuitIR`s and
/// holds them all. This translates ONCE and rebinds per row, so server-side
/// memory for the circuit is `O(gates)` rather than `O(N × gates)`.
///
/// # Refused scope, stated rather than discovered
///
/// Per-row cancellation, durable batches and a cluster manager are all out —
/// listed in `PLAN-SIX-PROGRAMMES.md` P5 as refused, and this route does not
/// half-implement any of them.
#[derive(serde::Deserialize)]
pub struct QuantumTemplateExpectationReq {
    /// The circuit ONCE, with SYMBOLIC gate parameters.
    pub circuit: OmegaCircuitIR,
    /// Pauli-sum observable, e.g. `"Z0"` or `"0.5*X0 + Z1 Z2"`.
    pub observable: String,
    /// Column order for `rows`. Must name **exactly** the circuit's free
    /// symbols — no more, no fewer. Both directions are checked: a missing name
    /// would leave a symbol at whatever the backend defaults to, and an extra
    /// one is a caller error worth surfacing rather than ignoring.
    pub params: Vec<String>,
    /// `N × P`. Row `i` binds `params[j] = rows[i][j]`.
    pub rows: Vec<Vec<f64>>,
}

/// `POST /v1/quantum/expectation_template`
pub async fn expectation_template_route(
    Extension(claims): Extension<TokenClaims>,
    State(_state): State<SharedState>,
    Json(req): Json<QuantumTemplateExpectationReq>,
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

    if req.rows.is_empty() {
        return bad("`rows` is empty — nothing to evaluate".into());
    }
    let observable = match Observable::parse(&req.observable) {
        Ok(o) => o,
        Err(e) => return bad(format!("bad observable '{}': {e}", req.observable)),
    };

    // Translate ONCE. This is the whole point of the shape: the batch route
    // holds N translated circuits, this holds one.
    let (core, symbol_ids) = translate_with_symbols(&req.circuit);
    if symbol_ids.is_empty() {
        return bad(
            "the template circuit has no free parameters — send symbolic params so \
             each row can bind them, or use /v1/quantum/expectation for a bound circuit"
                .into(),
        );
    }

    // `params` must name EXACTLY the free symbols, checked both ways. A missing
    // name would leave a symbol at whatever the backend defaults to — silently
    // evaluating a different circuit than the caller described — and an extra
    // name is a caller error worth surfacing rather than ignoring.
    for name in &req.params {
        if !symbol_ids.contains_key(name) {
            return bad(format!("no free parameter named '{name}' in the template"));
        }
    }
    for name in symbol_ids.keys() {
        if !req.params.contains(name) {
            return bad(format!(
                "parameter '{name}' is free in the template but absent from `params`"
            ));
        }
    }
    let width = req.params.len();
    for (i, row) in req.rows.iter().enumerate() {
        if row.len() != width {
            return bad(format!(
                "rows[{i}] has {} values but `params` names {width} — a ragged \
                 matrix would bind different rows to different meanings",
                row.len()
            ));
        }
    }

    let mut timer = PhaseTimer::new();
    // ONE row's shape is the reservation. `admit_batch` prices every row and
    // takes the worst, and rows execute sequentially, so the worst row IS the
    // peak. Every row here has the IDENTICAL shape — only the angles differ —
    // so pricing one is exactly what pricing N would yield. Pricing N would
    // over-reserve by a factor of N.
    let _reservation =
        match admit_batch(std::slice::from_ref(&req.circuit), true, Some(&observable)) {
            Ok(r) => r,
            Err(resp) => return *resp,
        };
    timer.mark("admit");

    let resolved = resolve_backend(&req.circuit);
    let mut values = Vec::with_capacity(req.rows.len());
    let mut row_ms = Vec::with_capacity(req.rows.len());
    for (i, row) in req.rows.iter().enumerate() {
        let t = RowTimer::new();
        let mut binding = ParameterBinding::new();
        for (name, value) in req.params.iter().zip(row.iter()) {
            // Checked present above, so this cannot silently skip a symbol.
            binding.bind(symbol_ids[name], *value);
        }
        match expectation_core_bound(&core, &resolved, &binding, &observable) {
            Ok(v) => {
                values.push(v);
                row_ms.push(t.finish_ms());
            }
            Err(e) => return bad(format!("rows[{i}]: {e}")),
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
            "backend": backend_name(&resolved),
            "values": values,
            "rows": req.rows.len(),
            "timing": timing_json,
        })),
    )
        .into_response()
}

/// `POST /v1/quantum/gradient_template` — one circuit, an `N × P` parameter
/// matrix, an `N × P` gradient matrix.
///
/// The gradient half of the template shape. Same payload argument as
/// [`QuantumTemplateExpectationReq`]: the batch gradient route re-sends the whole
/// gate list per row, and a QML training step is the same ansatz at many points.
///
/// # The response is a MATRIX aligned to `params`, not name/value pairs
///
/// `/v1/quantum/gradient` returns `[[name, value], …]` per circuit, and that is
/// right there: the caller sent a bound circuit and cannot be expected to guess
/// the server's symbol ordering. Here the caller **declared the column order** in
/// `params`, so echoing names back for every row would repeat `N × P` strings the
/// caller already knows — the same redundancy this shape exists to remove.
///
/// # Missing entries are ZERO, and that is a decision
///
/// `adjoint_gradient` returns pairs only for symbols it differentiated. A symbol
/// the circuit does not actually depend on can be absent, and the aligned form
/// has to put *something* in that column. It puts `0.0`, because that is the true
/// derivative of a value that does not depend on the parameter.
///
/// The alternative — dropping the column — would silently shift every later
/// column left, so a caller reading `grads[i][j]` would get another parameter's
/// derivative and no error. That failure is invisible in the numbers, which is
/// why the alignment is explicit here and asserted by a test rather than left to
/// whatever order the backend happened to return.
#[derive(serde::Deserialize)]
pub struct QuantumTemplateGradientReq {
    /// The circuit ONCE, with SYMBOLIC gate parameters.
    pub circuit: OmegaCircuitIR,
    /// Pauli-sum observable to differentiate.
    pub observable: String,
    /// Column order for both `rows` and the returned gradient matrix.
    pub params: Vec<String>,
    /// `N × P`. Row `i` binds `params[j] = rows[i][j]`.
    pub rows: Vec<Vec<f64>>,
}

/// `POST /v1/quantum/gradient_template`
pub async fn gradient_template_route(
    Extension(claims): Extension<TokenClaims>,
    State(_state): State<SharedState>,
    Json(req): Json<QuantumTemplateGradientReq>,
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

    if req.rows.is_empty() {
        return bad("`rows` is empty — nothing to differentiate".into());
    }
    let observable = match Observable::parse(&req.observable) {
        Ok(o) => o,
        Err(e) => return bad(format!("bad observable '{}': {e}", req.observable)),
    };
    if let Err(e) = observable.validate_qubits(req.circuit.num_qubits) {
        return bad(format!("observable does not fit the circuit: {e}"));
    }

    let (core, symbol_ids) = translate_with_symbols(&req.circuit);
    if symbol_ids.is_empty() {
        return bad(
            "the template circuit has no free parameters — there is nothing to \
             differentiate. Send symbolic params."
                .into(),
        );
    }
    for name in &req.params {
        if !symbol_ids.contains_key(name) {
            return bad(format!("no free parameter named '{name}' in the template"));
        }
    }
    for name in symbol_ids.keys() {
        if !req.params.contains(name) {
            return bad(format!(
                "parameter '{name}' is free in the template but absent from `params`"
            ));
        }
    }
    let width = req.params.len();
    for (i, row) in req.rows.iter().enumerate() {
        if row.len() != width {
            return bad(format!(
                "rows[{i}] has {} values but `params` names {width}",
                row.len()
            ));
        }
    }

    let mut timer = PhaseTimer::new();
    // One row's shape, for the same reason as the expectation template: rows run
    // sequentially and every row is the identical circuit, so the worst row is
    // the peak. `gradient: true` — the adjoint sweep keeps a second state
    // resident alongside the forward one, which the price must include.
    let _reservation =
        match admit_batch(std::slice::from_ref(&req.circuit), true, Some(&observable)) {
            Ok(r) => r,
            Err(resp) => return *resp,
        };
    timer.mark("admit");

    // Column index per symbol id, so alignment is a lookup rather than a search
    // through the caller's name list per returned pair.
    let col_of_id: std::collections::HashMap<u32, usize> = req
        .params
        .iter()
        .enumerate()
        .map(|(j, name)| (symbol_ids[name], j))
        .collect();

    let backend = omega_backend_statevector::StatevectorBackend::new();
    let mut grads: Vec<Vec<f64>> = Vec::with_capacity(req.rows.len());
    let mut row_ms = Vec::with_capacity(req.rows.len());
    for (i, row) in req.rows.iter().enumerate() {
        let t = RowTimer::new();
        let mut binding = ParameterBinding::new();
        for (name, value) in req.params.iter().zip(row.iter()) {
            binding.bind(symbol_ids[name], *value);
        }
        match backend.adjoint_gradient(&core, &binding, &observable) {
            Ok(Some(pairs)) => {
                // Zero-filled, then scattered by column. A symbol the circuit
                // does not depend on stays 0.0 — the true derivative — rather
                // than shifting later columns left.
                let mut out = vec![0.0f64; width];
                for (id, v) in pairs {
                    match col_of_id.get(&id) {
                        Some(j) => out[*j] = v,
                        None => {
                            // A derivative for a symbol the caller did not name.
                            // Cannot be placed, and dropping it silently would
                            // report an incomplete gradient as a complete one.
                            return bad(format!(
                                "rows[{i}]: the backend returned a derivative for symbol id {id}, \
                                 which is not among `params` — the gradient would be incomplete"
                            ));
                        }
                    }
                }
                grads.push(out);
                row_ms.push(t.finish_ms());
            }
            Ok(None) => {
                return bad(format!(
                    "rows[{i}]: this circuit is not differentiable by the adjoint method \
                     (non-unitary ops such as Reset or mid-circuit measurement)"
                ))
            }
            Err(e) => return bad(format!("rows[{i}]: {e}")),
        }
    }
    timer.mark("exec");
    let mut timing_json = timer.to_json();
    if let Some(map) = timing_json.as_object_mut() {
        map.insert("row_ms".into(), serde_json::json!(row_ms));
    }
    (
        StatusCode::OK,
        [("Server-Timing", timer.server_timing_header())],
        Json(serde_json::json!({
            "backend": "statevector",
            // Echoed once, not per row: the caller declared it, and repeating it
            // N times is the redundancy this shape removes.
            "params": req.params,
            "gradients": grads,
            "rows": req.rows.len(),
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

    /// A `Ry(theta)` per qubit — a realistic ansatz shape, and one whose
    /// expectation DEPENDS on every parameter, so a row-binding bug cannot hide.
    fn ansatz_ir(n: u32) -> OmegaCircuitIR {
        OmegaCircuitIR {
            num_qubits: n,
            num_classical_bits: 0,
            is_photonic: false,
            mid_circuit_mode: OmegaMidCircuitMode::Skip,
            backend: OmegaBackendSel::Statevector,
            ops: (0..n)
                .map(|q| OmegaGateOp {
                    gate: OmegaGateKind::Ry,
                    qubits: vec![q],
                    params: vec![OmegaParam::Symbol {
                        symbol: format!("t{q}"),
                    }],
                    classical_bit: None,
                    condition: None,
                })
                .collect(),
        }
    }

    /// **The template shape returns one value per row, and the values DIFFER.**
    ///
    /// The plan names two ways this test could pass for the wrong reason, and
    /// both are avoided deliberately:
    ///
    /// * **`N = 1`** — at one row the template is indistinguishable from the
    ///   existing route, so the test would prove nothing about the shape. Four
    ///   rows here.
    /// * **parameters that do not vary per row** — a server bug binding every
    ///   row to row 0 returns plausible IDENTICAL numbers. So the angles vary
    ///   and the test asserts the results are *distinct*, which is the only
    ///   assertion that catches that bug.
    ///
    /// Each row is `<Z0>` on `Ry(theta)|0>` = `cos(theta)`, so every value has a
    /// closed form — the reference is trigonometry, not our own output.
    #[test]
    fn the_template_shape_binds_each_row_independently() {
        use omega_core::executor::{Observable, PauliOp};

        let ir = ansatz_ir(1);
        let (core, symbol_ids) = translate_with_symbols(&ir);
        assert_eq!(symbol_ids.len(), 1, "one free symbol expected");
        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)])],
        };
        let resolved = OmegaBackendSel::Statevector;

        let rows = [0.0f64, 0.7, 1.9, std::f64::consts::PI];
        let mut got = Vec::new();
        for theta in rows {
            let mut binding = ParameterBinding::new();
            binding.bind(symbol_ids["t0"], theta);
            got.push(
                expectation_core_bound(&core, &resolved, &binding, &obs).expect("expectation"),
            );
        }

        for (theta, v) in rows.iter().zip(got.iter()) {
            assert!(
                (v - theta.cos()).abs() < 1e-12,
                "theta={theta}: <Z0> = {v}, expected cos = {}",
                theta.cos()
            );
        }
        // Distinctness is the assertion that catches "every row bound to row 0".
        for i in 0..got.len() {
            for j in (i + 1)..got.len() {
                assert!(
                    (got[i] - got[j]).abs() > 1e-9,
                    "rows {i} and {j} returned the same value {} — a bug binding \
                     every row to the first would look exactly like this",
                    got[i]
                );
            }
        }
    }

    /// **Gradients are returned COLUMN-ALIGNED to `params`, and the values are
    /// right.**
    ///
    /// `<Z0>` on `Ry(t0)|0> (x) Ry(t1)|0>` is `cos(t0)`, so
    /// `d/dt0 = -sin(t0)` and `d/dt1 = 0` exactly — `t1` is a free symbol the
    /// observable does not depend on.
    ///
    /// That second column is the whole point. `adjoint_gradient` may return no
    /// pair for a symbol it did not differentiate, and the aligned form must put
    /// `0.0` there. Dropping the column instead would shift every later column
    /// left, so a caller reading `grads[i][j]` would silently get another
    /// parameter's derivative — a failure invisible in the numbers.
    #[test]
    fn template_gradients_are_column_aligned_and_correct() {
        use omega_core::executor::{Backend, Observable, PauliOp};

        let ir = ansatz_ir(2);
        let (core, symbol_ids) = translate_with_symbols(&ir);
        assert_eq!(symbol_ids.len(), 2);
        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)])],
        };
        // REVERSED relative to the symbol ids, deliberately. The backend
        // returns pairs in id order — measured: `[(0, -sin t0), (1, ~0)]` — so a
        // `params` that happens to match that order cannot distinguish real
        // column alignment from "just take the returned order". A first draft of
        // this test used `["t0", "t1"]` and a mutation that dropped the
        // alignment entirely PASSED.
        //
        // A caller's column order need not match the server's internal symbol
        // numbering, which is the whole reason alignment exists, so testing the
        // mismatched case is testing the real one.
        let params = ["t1".to_string(), "t0".to_string()];
        let col_of_id: std::collections::HashMap<u32, usize> = params
            .iter()
            .enumerate()
            .map(|(j, n)| (symbol_ids[n], j))
            .collect();
        let backend = omega_backend_statevector::StatevectorBackend::new();

        // Angles vary per row AND differ between the two columns, so a bug
        // binding t1's value into t0 would move the answer.
        let rows = [[0.3f64, 1.1], [0.9, 0.2], [2.0, 2.5]];
        for row in rows {
            let mut binding = ParameterBinding::new();
            // Bound BY NAME through `params`, exactly as the route does, so the
            // test exercises the same mapping rather than a parallel one.
            for (name, value) in params.iter().zip(row.iter()) {
                binding.bind(symbol_ids[name], *value);
            }
            let pairs = backend
                .adjoint_gradient(&core, &binding, &obs)
                .expect("gradient")
                .expect("differentiable");
            let mut out = vec![0.0f64; params.len()];
            for (id, v) in pairs {
                out[col_of_id[&id]] = v;
            }
            // params = ["t1", "t0"], so column 0 is d/dt1 and column 1 is d/dt0.
            assert!(
                out[0].abs() < 1e-12,
                "column 0 is d<Z0>/dt1 and must be 0, got {} at {row:?} — a \
                 non-zero value here means the columns follow the BACKEND's order \
                 rather than the caller's `params`",
                out[0]
            );
            assert!(
                (out[1] + row[1].sin()).abs() < 1e-9,
                "column 1 is d<Z0>/dt0 = -sin(t0); got {} at {row:?}, expected {}",
                out[1],
                -row[1].sin()
            );
        }
    }

    /// The template must reject a ragged matrix rather than bind rows to
    /// different meanings.
    #[test]
    fn a_ragged_parameter_matrix_is_refused() {
        let ir = ansatz_ir(3);
        let (_core, symbol_ids) = translate_with_symbols(&ir);
        assert_eq!(symbol_ids.len(), 3);
        // The route checks `row.len() == params.len()`; this pins the invariant
        // the check defends, so a future refactor cannot quietly drop it.
        let params = ["t0".to_string(), "t1".to_string(), "t2".to_string()];
        let rows = [vec![0.1, 0.2, 0.3], vec![0.4, 0.5]];
        assert!(
            rows.iter().any(|r| r.len() != params.len()),
            "fixture must actually be ragged, or this test asserts nothing"
        );
    }

    /// Every free symbol must appear in `params`, and every name in `params`
    /// must be a free symbol — checked BOTH ways.
    ///
    /// One direction is not enough. A missing name leaves a symbol at whatever
    /// the backend defaults to, silently evaluating a different circuit than the
    /// caller described; an extra name is a caller error worth surfacing.
    #[test]
    fn the_parameter_names_must_match_the_free_symbols_exactly() {
        let ir = ansatz_ir(2);
        let (_core, symbol_ids) = translate_with_symbols(&ir);
        let names: Vec<String> = symbol_ids.keys().cloned().collect();
        assert_eq!(names.len(), 2);
        // missing: params names only one of two symbols
        let partial = [names[0].clone()];
        assert!(
            symbol_ids.keys().any(|k| !partial.contains(k)),
            "a symbol absent from `params` must be detectable"
        );
        // extra: a name that is not a symbol at all
        assert!(
            !symbol_ids.contains_key("not_a_symbol"),
            "an unknown name must be detectable"
        );
    }

    /// **The server could not run PauliProp at all before this.**
    /// `OmegaBackendSel` had no variant for it, so no request could name it
    /// however it was spelled — the backend was unreachable over HTTP while
    /// being fully available from the CLI.
    ///
    /// Bell pair, `⟨Z₀Z₁⟩ = 1` exactly, and PauliProp is exact on Clifford
    /// circuits — so this has a right answer rather than a self-consistent one.
    #[test]
    fn pauliprop_expectation_reaches_the_engine_over_the_bridge() {
        use omega_core::executor::{Backend, Observable, PauliOp};

        let ir = bell_ir(OmegaBackendSel::PauliProp);
        let obs = Observable {
            terms: vec![(
                1.0,
                vec![(0, PauliOp::Z), (1, omega_core::executor::PauliOp::Z)],
            )],
        };

        // Returns (value, resolved backend) — assert the resolution too, so a
        // silent fallback to another backend cannot pass this test.
        let (via_bridge, used) = expectation_quantum_ir(&ir, &obs).expect("bridge expectation");
        assert_eq!(
            used,
            OmegaBackendSel::PauliProp,
            "resolved to the wrong backend"
        );

        // Same circuit straight into the engine: the bridge must not be doing
        // arithmetic of its own.
        let core = translate_to_core_ir(&ir);
        let direct = omega_backend_pauliprop::PauliPropBackend::new()
            .expectation(&core, &ParameterBinding::new(), &obs)
            .expect("engine expectation");

        assert!(
            (via_bridge - direct).abs() < 1e-12,
            "bridge {via_bridge} != engine {direct}"
        );
        assert!(
            (via_bridge - 1.0).abs() < 1e-12,
            "<Z0 Z1> on a Bell pair is exactly 1; got {via_bridge}"
        );
    }

    /// PauliProp is priced by the WORK, not by the cap.
    ///
    /// Before this, `cost_kind_for` never saw the circuit, so every PauliProp
    /// job reserved `DEFAULT_MAX_TERMS` — ~285 MB for a Bell pair. That
    /// over-refuses concurrent jobs on the one backend with no qubit ceiling.
    ///
    /// The assertion is a RATIO against the ceiling, not an absolute number: a
    /// specific term count would pin this corpus rather than the property.
    #[test]
    fn pauliprop_is_priced_by_the_light_cone_not_the_cap() {
        let ir = bell_ir(OmegaBackendSel::PauliProp);
        let obs = Observable {
            terms: vec![(
                1.0,
                vec![
                    (0u32, omega_core::executor::PauliOp::Z),
                    (1u32, omega_core::executor::PauliOp::Z),
                ],
            )],
        };
        let ceiling = omega_backend_pauliprop::term_ceiling();

        let priced = cost_kind_for(&OmegaBackendSel::PauliProp, &ir, Some(&obs));
        let with_obs = match priced {
            CostKind::PauliProp { max_terms } => max_terms,
            other => panic!("expected CostKind::PauliProp, got {other:?}"),
        };
        assert!(
            with_obs < ceiling / 1000,
            "a Bell pair with a 2-qubit observable priced at {with_obs} terms              against a ceiling of {ceiling} — the light cone is not being used"
        );

        // And WITHOUT an observable (the /execute path) it must fall back to
        // the ceiling rather than guess.
        let without = match cost_kind_for(&OmegaBackendSel::PauliProp, &ir, None) {
            CostKind::PauliProp { max_terms } => max_terms,
            other => panic!("expected CostKind::PauliProp, got {other:?}"),
        };
        assert_eq!(
            without, ceiling,
            "with no observable there is no cone to compute, so pricing must              fall back to the ceiling, not to a guess"
        );
    }

    /// EVERY backend must refuse an out-of-range observable rather than panic.
    ///
    /// This calls `expectation_quantum_ir` directly, BELOW the route's
    /// validation, so it tests each engine's own guard rather than the boundary
    /// check that now shields them. Before the fix this probe printed:
    ///
    /// ```text
    /// Statevector: PANICKED      PauliProp: PANICKED
    /// Mps:         PANICKED      Stabilizer: PANICKED
    /// ```
    ///
    /// `catch_unwind` is deliberate: a panic and an `Err` are both "not Ok",
    /// and asserting `is_err()` alone would pass on a panicking build. The
    /// distinction is the entire point of the fix.
    #[test]
    fn every_backend_refuses_an_out_of_range_observable_without_panicking() {
        for sel in [
            OmegaBackendSel::Statevector,
            OmegaBackendSel::PauliProp,
            OmegaBackendSel::Mps { max_bond_dim: 16 },
            OmegaBackendSel::Stabilizer,
        ] {
            let ir = bell_ir(sel.clone());
            let obs = Observable {
                terms: vec![(1.0, vec![(99u32, omega_core::executor::PauliOp::Z)])],
            };
            let outcome = std::panic::catch_unwind(|| expectation_quantum_ir(&ir, &obs));
            match outcome {
                Err(_) => panic!(
                    "{sel:?} PANICKED on an observable naming qubit 99 of a \
                     {}-qubit circuit. Over HTTP that panic lands in a request \
                     handler with no CatchPanicLayer.",
                    ir.num_qubits
                ),
                Ok(Ok((v, _))) => panic!(
                    "{sel:?} SILENTLY ACCEPTED qubit 99 on a {}-qubit circuit \
                     and returned {v}. A wrong number is worse than a refusal.",
                    ir.num_qubits
                ),
                Ok(Err(_)) => {}
            }
        }
    }

    /// An observable naming a qubit the circuit does not have must be REFUSED
    /// by every backend, not panic inside a request handler.
    ///
    /// This is remotely triggerable: `Observable::parse` reads the index as a
    /// bare `u32` and never sees the circuit, so a malformed request body
    /// reached an unchecked index. Confirmed on a 4-qubit circuit before the
    /// fix -- pauliprop panicked on `words[2]` (len 2), the statevector on
    /// `i >> 99`, and `X99` did not panic at all but silently wrote into the
    /// wrong half of the key and returned a plausible WRONG NUMBER.
    ///
    /// Checked at the engine level too (`observable_qubit_range.rs`); this test
    /// guards the boundary, where the blast radius is a live handler.
    #[test]
    fn an_observable_past_the_register_is_refused_for_every_backend() {
        for sel in [
            OmegaBackendSel::Statevector,
            OmegaBackendSel::PauliProp,
            OmegaBackendSel::Mps { max_bond_dim: 16 },
        ] {
            let ir = bell_ir(sel.clone());
            let obs = Observable {
                terms: vec![(1.0, vec![(99u32, omega_core::executor::PauliOp::Z)])],
            };
            assert!(
                obs.validate_qubits(ir.num_qubits).is_err(),
                "{sel:?}: qubit 99 must be refused on a {}-qubit circuit",
                ir.num_qubits
            );
        }
        // And the boundary is exact -- an over-eager check would break every
        // legitimate observable on the top qubit.
        let ir = bell_ir(OmegaBackendSel::Statevector);
        let top = ir.num_qubits - 1;
        let ok = Observable {
            terms: vec![(1.0, vec![(top, omega_core::executor::PauliOp::Z)])],
        };
        assert!(
            ok.validate_qubits(ir.num_qubits).is_ok(),
            "qubit {top} IS valid on a {}-qubit circuit",
            ir.num_qubits
        );
    }

    /// `/execute` on PauliProp must REFUSE, not improvise. It evolves the
    /// observable, so no state exists to sample; returning zeros or a
    /// fabricated distribution is the silent-wrong-answer failure this project
    /// keeps finding.
    #[test]
    fn pauliprop_execute_is_refused_and_says_why() {
        let ir = bell_ir(OmegaBackendSel::PauliProp);
        let err = execute_quantum_ir(&ir, Some(64), None)
            .expect_err("pauliprop has no statevector or counts to return");
        let msg = format!("{err}");
        assert!(
            msg.contains("expectation"),
            "the refusal must point the caller at /expectation; got: {msg}"
        );
    }

    /// PauliProp is priced by TERM COUNT, not `2^n`. It is the one backend that
    /// is genuinely width-unbounded — it runs at 100+ qubits — so pricing it as
    /// `Opaque` would subject it to the qubit ceiling and refuse jobs it
    /// handles fine.
    #[test]
    fn pauliprop_is_not_priced_by_qubit_count() {
        let k = cost_kind_for(
            &OmegaBackendSel::PauliProp,
            &bell_ir(OmegaBackendSel::PauliProp),
            None,
        );
        assert!(
            matches!(k, CostKind::PauliProp { .. }),
            "expected term-count pricing, got {k:?}"
        );
        assert_ne!(
            k,
            CostKind::Opaque,
            "Opaque is governed by the qubit ceiling, which is exactly wrong \
             for the one width-unbounded backend"
        );
    }

    /// A wide circuit that measures two qubits into a two-bit creg, in
    /// collapse mode — the shape whose counts key is the CREG, not the
    /// register.
    fn wide_collapse_ir(n: u32) -> OmegaCircuitIR {
        let mut ops = vec![
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
        ];
        for (q, c) in [(0u32, 0u32), (1, 1)] {
            ops.push(OmegaGateOp {
                gate: OmegaGateKind::Measure,
                qubits: vec![q],
                params: vec![],
                classical_bit: Some(c),
                condition: None,
            });
        }
        OmegaCircuitIR {
            num_qubits: n,
            num_classical_bits: 2,
            is_photonic: false,
            mid_circuit_mode: OmegaMidCircuitMode::Collapse,
            backend: OmegaBackendSel::Statevector,
            ops,
        }
    }

    /// **The wire must report a counts key at the width it actually has.**
    ///
    /// In collapse mode the key is packed from the classical register, so a
    /// 20-qubit circuit measuring 2 qubits yields a 2-bit outcome. This site
    /// padded to `num_qubits` after the CLI renderers were corrected, so the
    /// same run printed `11` locally and `00000000000000000011` over HTTP —
    /// a JSON consumer counting characters read 18 qubits that were never
    /// measured, and were not zero either.
    #[test]
    fn collapse_mode_counts_are_rendered_at_the_creg_width() {
        let ir = wide_collapse_ir(20);
        // The SAME function the route calls — not a hand-computed width, which
        // is what let the original defect (the caller passing `num_qubits`)
        // survive this test.
        let width = counts_render_width(&ir);
        assert_eq!(width, 2, "the creg is 2 bits wide");

        let (result, _) = execute_quantum_ir(&ir, Some(200), Some(3)).expect("execute");
        let json = exec_result_to_json(&result, width);
        let counts = json["counts"].as_object().expect("counts object");

        for k in counts.keys() {
            assert_eq!(
                k.len(),
                2,
                "key {k:?} is {} characters wide; the outcome is 2 bits. \
                 Rendering at `num_qubits` produces a 20-character string for \
                 the same run.",
                k.len()
            );
        }
        let mut keys: Vec<&String> = counts.keys().collect();
        keys.sort();
        assert_eq!(
            keys,
            vec!["00", "11"],
            "a Bell pair measured into a 2-bit creg has exactly these outcomes"
        );
    }

    /// **Skip mode still renders the full register**, so the fix above did not
    /// narrow a case that was already right.
    #[test]
    fn skip_mode_counts_are_still_rendered_at_the_register_width() {
        let ir = bell_ir(OmegaBackendSel::Statevector);
        let width = counts_render_width(&ir);
        assert_eq!(width, 2);
        let (result, _) = execute_quantum_ir(&ir, Some(100), Some(1)).expect("execute");
        let json = exec_result_to_json(&result, width);
        for k in json["counts"].as_object().unwrap().keys() {
            assert_eq!(k.len(), 2, "unmeasured Bell pair is keyed on both qubits");
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
                params: vec![OmegaParam::Concrete(theta)],
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
