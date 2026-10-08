// SPDX-License-Identifier: Apache-2.0
//! Run an Aria circuit on a pluggable execution backend.
//!
//! Every backend implements `omega_core::executor::Backend`. The default
//! `sim` backend is the pure-Rust statevector simulator; additional backends
//! (MPS, GPU, libtorch, remote) are added as features in later phases and
//! selected through [`BackendSel`].

use std::collections::{HashMap, HashSet};

use aria_core::ast::Circuit;
use num_complex::Complex64;
use omega_backend_mps::{MpsBackend, MpsRunStats, NoisyMpsBackend};
use omega_backend_pauliprop::PauliPropBackend;
use omega_backend_statevector::{NoisyStatevectorBackend, StatevectorBackend};
use omega_core::circuit::SymbolId;
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable};
use omega_core::gradient::compute_gradient_for;
use omega_core::noise::NoiseModel;
use omega_core::params::ParameterBinding;

// Re-exported so callers of [`expectation_with_gradient`] can name the gradient
// method without a direct `omega_core` dependency (matches the rest of `run`).
pub use omega_core::gradient::GradMethod;

use crate::lower::{lower, Lowered};

/// Default MPS bond dimension (χ) when `--backend mps` is selected without an
/// explicit `mps:<chi>`. Ample for the small examples; raise it for circuits
/// whose entanglement exceeds it (χ = 2^(n/2) is exact for n qubits).
pub const DEFAULT_MPS_CHI: usize = 64;

/// Ceiling (χ budget) for `--backend mps:auto` when no explicit
/// `mps:auto:<ceiling>` is given — the bond grows adaptively only as far as the
/// entanglement needs, never past this.
pub const DEFAULT_MPS_AUTO_CEILING: usize = 1024;

/// Relative singular-value tolerance for adaptive (`mps:auto`) truncation: a
/// split keeps the singular values above `ε · σ_max`. Small enough to be
/// effectively lossless on low-entanglement states while letting the bond stay
/// far below the ceiling when it can.
pub const MPS_AUTO_EPS: f64 = 1e-10;

/// Which execution backend to dispatch to. Every variant is an implementation
/// of `omega_core::executor::Backend` — the Aria plugin contract. New backends
/// (GPU, libtorch `tch`, remote omega-server) slot in here behind features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendSel {
    /// Pure-Rust CPU statevector (`omega-backend-statevector`). Exact, ≤~24 qubits.
    Sim,
    /// Pure-Rust matrix-product-state (`omega-backend-mps`) with bond
    /// dimension `chi`. Scales to many qubits when entanglement is bounded;
    /// `chi = 2^(n/2)` reproduces the dense statevector exactly. Selected as
    /// `--backend mps` (χ = [`DEFAULT_MPS_CHI`]) or `--backend mps:<chi>`.
    Mps { chi: usize },
    /// Adaptive-bond MPS: the bond grows with the actual entanglement (relative
    /// singular-value tolerance [`MPS_AUTO_EPS`]) up to the `max_chi` ceiling,
    /// instead of always filling a fixed χ. Selected as `--backend mps:auto`
    /// (ceiling [`DEFAULT_MPS_AUTO_CEILING`]) or `--backend mps:auto:<ceiling>`.
    MpsAuto { max_chi: usize },
    /// GPU statevector (Metal / CUDA / OpenCL, whichever feature is compiled in).
    /// Falls back to [`BackendSel::Sim`] if the device is unavailable at runtime.
    Gpu,
    /// libtorch (`tch`) statevector backend (feature `tch`).
    Tch,
    /// Pauli-propagation (`omega-backend-pauliprop`): Heisenberg-picture
    /// evolution of the observable. Expectation values only — `--shots` /
    /// `--statevector` are rejected by the backend with a clear error.
    /// Exact and width-unbounded on Clifford circuits.
    PauliProp,
}

impl BackendSel {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "sim" | "statevector" => Ok(Self::Sim),
            "mps" => Ok(Self::Mps {
                chi: DEFAULT_MPS_CHI,
            }),
            "gpu" => Ok(Self::Gpu),
            "tch" => Ok(Self::Tch),
            "pauliprop" => Ok(Self::PauliProp),
            "mps:auto" => Ok(Self::MpsAuto {
                max_chi: DEFAULT_MPS_AUTO_CEILING,
            }),
            other => {
                // `mps:auto:<ceiling>` — adaptive bond with an explicit ceiling.
                if let Some(cap_str) = other.strip_prefix("mps:auto:") {
                    let max_chi: usize = cap_str.parse().map_err(|_| {
                        format!("bad MPS ceiling in '{other}' (want mps:auto:<int>)")
                    })?;
                    if max_chi == 0 {
                        return Err("MPS bond ceiling must be ≥ 1".into());
                    }
                    return Ok(Self::MpsAuto { max_chi });
                }
                // `mps:<chi>` — explicit fixed bond dimension.
                if let Some(chi_str) = other.strip_prefix("mps:") {
                    let chi: usize = chi_str.parse().map_err(|_| {
                        format!("bad MPS bond dimension in '{other}' (want mps:<int>)")
                    })?;
                    if chi == 0 {
                        return Err("MPS bond dimension must be ≥ 1".into());
                    }
                    return Ok(Self::Mps { chi });
                }
                Err(format!(
                    "unknown backend '{other}' (available: sim, mps, mps:<chi>, mps:auto, \
                     mps:auto:<ceiling>, gpu, tch, pauliprop; remote via --url)"
                ))
            }
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Sim => "sim",
            Self::Mps { .. } => "mps",
            Self::MpsAuto { .. } => "mps:auto",
            Self::Gpu => "gpu",
            Self::Tch => "tch",
            Self::PauliProp => "pauliprop",
        }
    }
}

/// Construct the selected backend (`omega_core::Backend`). Shared by `run` and
/// `train` so the chosen backend drives both execution and gradients.
pub(crate) fn make_backend(sel: BackendSel) -> Result<Box<dyn Backend>, String> {
    Ok(match sel {
        BackendSel::Sim => Box::new(StatevectorBackend::new()),
        BackendSel::Mps { chi } => Box::new(make_mps(chi)),
        BackendSel::MpsAuto { max_chi } => Box::new(make_mps(max_chi).with_adaptive(MPS_AUTO_EPS)),
        BackendSel::Gpu => make_gpu()?,
        BackendSel::Tch => make_tch()?,
        // Exact engine (no truncation: coeff_min = 0, max_weight = None), with
        // the GPU branch accelerator wired under a cuda build.
        BackendSel::PauliProp => Box::new(make_pauliprop(0.0, None, None)),
    })
}

std::thread_local! {
    /// Truncation certificate of the most recent MPS run through the `run::*`
    /// wrappers on this thread — a side channel that keeps the wrappers'
    /// signatures unchanged (the binding-order / API contract) while letting a
    /// front end report it. `None` after a non-MPS run. Read (and cleared) with
    /// [`take_last_mps_stats`].
    static LAST_MPS_STATS: std::cell::Cell<Option<MpsRunStats>> = const { std::cell::Cell::new(None) };
}

/// Take (and clear) the MPS truncation certificate of the most recent `run::*`
/// call on this thread; `None` if that run did not use an MPS backend. Used by
/// the CLIs to report discarded weight without changing any run signature.
pub fn take_last_mps_stats() -> Option<MpsRunStats> {
    LAST_MPS_STATS.with(|c| c.take())
}

/// Clear the thread-local certificate. Every `run::*` entry point that does NOT
/// produce MPS truncation stats (the noisy trajectory paths, remote) calls this
/// so a later `take_last_mps_stats` can never return a *previous* MPS run's
/// value — the side channel means "the most recent run", not "the most recent
/// MPS run ago".
fn clear_mps_stats() {
    LAST_MPS_STATS.with(|c| c.set(None));
}

/// Run `f` against the selected backend, recording the MPS truncation
/// certificate into the thread-local side channel when the backend is MPS (so
/// [`take_last_mps_stats`] can surface it) and clearing it otherwise. Holds the
/// concrete `MpsBackend` for the MPS arms — stats live on the concrete type,
/// never the `Backend` trait.
fn with_backend_stats<T>(
    sel: BackendSel,
    f: impl FnOnce(&dyn Backend) -> Result<T, String>,
) -> Result<T, String> {
    match sel {
        BackendSel::Mps { chi } => {
            let b = make_mps(chi);
            let out = f(&b);
            // Record stats only on success — a failed run must not leave a clean
            // zero certificate that reads as "no truncation".
            LAST_MPS_STATS.with(|c| {
                c.set(out.is_ok().then(|| b.last_run_stats()));
            });
            out
        }
        BackendSel::MpsAuto { max_chi } => {
            let b = make_mps(max_chi).with_adaptive(MPS_AUTO_EPS);
            let out = f(&b);
            LAST_MPS_STATS.with(|c| {
                c.set(out.is_ok().then(|| b.last_run_stats()));
            });
            out
        }
        other => {
            let b = make_backend(other)?;
            clear_mps_stats();
            f(b.as_ref())
        }
    }
}

/// Construct the Pauli-propagation backend with the given truncation, installing
/// the CUDA branch accelerator (`omega-backend-pauliprop-cuda`) under a `cuda`
/// build. The accelerator falls back to the CPU branch per-gate when no device
/// is present or the term count is small, so results are identical either way.
pub(crate) fn make_pauliprop(
    coeff_min: f64,
    max_weight: Option<usize>,
    max_freq: Option<u32>,
) -> PauliPropBackend {
    let backend = PauliPropBackend::with_truncation_freq(coeff_min, max_weight, max_freq);
    #[cfg(feature = "cuda")]
    let backend = backend.with_branch_hook(omega_backend_pauliprop_cuda::cuda_branch);
    // Metal arm: symplectic branch work on the GPU, f64 coefficients on the CPU
    // (Apple has no native f64), so the result is exact. Falls back per-gate when
    // no device is present or the term count is small. `not(cuda)` because a hook
    // is single-valued — a cuda box never also wires the metal arm.
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let backend = backend.with_branch_hook(omega_backend_pauliprop_metal::metal_branch);
    backend
}

#[allow(unreachable_code, unused)]
fn make_tch() -> Result<Box<dyn Backend>, String> {
    #[cfg(feature = "tch")]
    {
        // Auto-select the GPU when the linked libtorch was built with CUDA and a
        // device is present; otherwise this is exactly `cpu()`. CUDA keeps
        // complex128, so the numerics match the CPU path (unlike the MPS arm).
        // Set ARIA_TCH_CPU=1 to force CPU even on a CUDA libtorch (debugging).
        if std::env::var("ARIA_TCH_CPU").is_ok_and(|v| v == "1") {
            return Ok(Box::new(aria_backend_tch::TchBackend::cpu()));
        }
        return Ok(Box::new(aria_backend_tch::TchBackend::cuda_or_cpu()));
    }
    Err(
        "aria was built without the libtorch backend; rebuild with `--features tch` \
         (and set LIBTORCH — see INSTALL_LIBTORCH.md)"
            .to_string(),
    )
}

/// Process-wide ceiling on an MPS run's truncation certificate.
///
/// Stored as `f64` bits so a front end can set it once, before any run.
static MPS_DISCARD_CEILING: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(
    // `f64::to_bits` is not const-callable on all supported toolchains, so the
    // sentinel 0 means "unset" and resolves to the backend default on read.
    0,
);

/// Raise (or lower) the discarded-weight ceiling every MPS backend built here
/// will enforce. `f64::INFINITY` accepts any truncation.
///
/// **This is a library-level policy, deliberately.** It briefly lived in the
/// `aria` CLI's post-run reporting instead, and that left `train`, `tune`,
/// `predict` and every library caller of `run_counts`/`expectation` with NO
/// truncation gate — they never call the CLI's reporter. Measured: `aria train
/// --backend mps:2` trained to completion on a state whose certificate was
/// 4.375, optimizing an observable that was already wrong at step 0
/// (`<O>` = -0.029 against -0.009 exact). Fixing one gate had removed another.
pub fn set_mps_discard_ceiling(w: f64) {
    MPS_DISCARD_CEILING.store(w.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// The ceiling in force, defaulting to the backend's.
pub fn mps_discard_ceiling() -> f64 {
    let bits = MPS_DISCARD_CEILING.load(std::sync::atomic::Ordering::Relaxed);
    if bits == 0 {
        omega_backend_mps::DEFAULT_MAX_DISCARDED_WEIGHT
    } else {
        f64::from_bits(bits)
    }
}

/// `MPS_METAL_CONTRACT=1` (or `true` / `yes`) opts into the f32 Metal two-site
/// contraction. Unset is the production default. Same predicate as `omega-run`.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn mps_metal_contract_opt_in() -> bool {
    match std::env::var("MPS_METAL_CONTRACT") {
        Ok(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        }
        Err(_) => false,
    }
}

/// The two MPS samplers take one Metal policy, so `make_noisy_mps` cannot
/// drift from `make_mps`. The f32 contraction is not installed: on a quantity
/// documented as a bound, occasionally-low is not a bound (`STATUS.md` §5
/// item 3). `not(cuda)` because a dual-vendor build keeps CUDA's native-f64
/// SVD. The opt-in warns; unset is silent, because this constructor has no
/// device request to answer.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
trait InstallMetalContract: Sized {
    fn install_metal_contract(self, f: omega_backend_mps::Contract2qFn) -> Self;
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
impl InstallMetalContract for MpsBackend {
    fn install_metal_contract(self, f: omega_backend_mps::Contract2qFn) -> Self {
        self.with_contract_fn(f)
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
impl InstallMetalContract for NoisyMpsBackend {
    fn install_metal_contract(self, f: omega_backend_mps::Contract2qFn) -> Self {
        self.with_contract_fn(f)
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn apply_mps_metal_contract<B: InstallMetalContract>(backend: B) -> B {
    if mps_metal_contract_opt_in() {
        // Loud on purpose. The opt-in exists so the kernel can still be
        // measured; it must not look like a normal run.
        eprintln!(
            "WARNING: MPS_METAL_CONTRACT is set, so the MPS two-site contraction \
             is routed through Metal (f32 above the bond threshold). \
             discarded_weight can come out SMALLER than the exact-f64 CPU value. \
             On this run the truncation certificate is NOT a bound."
        );
        backend.install_metal_contract(omega_backend_mps_metal::metal_contract_2q)
    } else {
        backend
    }
}

#[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
fn apply_mps_metal_contract<B>(backend: B) -> B {
    backend
}

/// Construct the MPS backend.
///
/// Under a `cuda` build the bond-compression SVD goes through cuSOLVER
/// `gesvdj`, which falls back to the CPU Jacobi SVD when no device is present.
/// That path is native f64, so the certificate is unchanged.
///
/// Under a `metal` build the two-site contraction stays on exact f64. The
/// Metal kernel is f32 — Apple GPUs have no f64 — and `discarded_weight` can
/// come back smaller than the CPU value, which is not a bound. SVD stays on
/// the CPU either way. `MPS_METAL_CONTRACT=1` opts into the f32 kernel and
/// warns; that is the same variable `omega-run` honours.
fn make_mps(chi: usize) -> MpsBackend {
    // Every MPS backend built in this crate carries the ceiling, so the gate
    // applies on every path — run, train, tune, predict, and library callers —
    // not only the one CLI subcommand that happens to report afterwards.
    let backend = MpsBackend::new(chi).with_max_discarded_weight(mps_discard_ceiling());
    #[cfg(feature = "cuda")]
    let backend = backend.with_svd_fn(omega_backend_mps_cuda::cuda_svd_flat);
    apply_mps_metal_contract(backend)
}

/// Construct the compiled-in GPU statevector backend, falling back to the
/// pure-Rust CPU statevector if the device cannot be initialized at runtime.
#[allow(unreachable_code, unused)]
fn make_gpu() -> Result<Box<dyn Backend>, String> {
    #[cfg(feature = "metal")]
    {
        return Ok(
            match omega_backend_statevector_metal::MetalStatevectorBackend::new() {
                Ok(b) => Box::new(b),
                Err(e) => {
                    eprintln!("note: Metal unavailable ({e:?}); falling back to CPU statevector");
                    Box::new(StatevectorBackend::new())
                }
            },
        );
    }
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    {
        return Ok(
            match omega_backend_statevector_cuda::CudaStatevectorBackend::new() {
                Ok(b) => Box::new(b),
                Err(e) => {
                    eprintln!("note: CUDA unavailable ({e:?}); falling back to CPU statevector");
                    Box::new(StatevectorBackend::new())
                }
            },
        );
    }
    #[cfg(all(feature = "opencl", not(any(feature = "metal", feature = "cuda"))))]
    {
        return Ok(
            match omega_backend_statevector_opencl::OpenClStatevectorBackend::new() {
                Ok(b) => Box::new(b),
                Err(e) => {
                    eprintln!("note: OpenCL unavailable ({e:?}); falling back to CPU statevector");
                    Box::new(StatevectorBackend::new())
                }
            },
        );
    }
    Err("aria was built without a GPU backend; rebuild with \
         `--features metal` (or cuda/opencl)"
        .to_string())
}

/// Bind every free Aria symbol to a concrete value — `bindings` first, then any
/// remaining free symbol defaults to `0.0` — and lower to a concrete omega IR.
fn concrete_ir(circuit: &Circuit, bindings: &HashMap<String, f64>) -> Result<Lowered, String> {
    let mut full = bindings.clone();
    for s in circuit.free_symbols() {
        full.entry(s).or_insert(0.0);
    }
    let bound = circuit.bind_params(&full)?;
    lower(&bound)
}

/// `measure qubit -> clbit` pairs of a lowered circuit, in program order.
/// Empty when the program declares no measure-to-creg mapping.
///
/// Thin adapter over [`omega_core::executor::measure_pairs`], which is where
/// the semantics live. It moved there when the N-way counts matrix
/// (`crates/omega-cli/tests/nway_counts.rs`) needed the same projection for
/// QASM2-sourced circuits: two copies of a counts-keying convention is how a
/// matrix ends up validating a convention no shipped path uses.
pub(crate) fn measure_pairs(low: &Lowered) -> Vec<(u32, u32)> {
    omega_core::executor::measure_pairs(&low.ir)
}

pub(crate) use omega_core::executor::project_counts_onto_creg;

/// Width in bits of the outcome keys produced by [`run_counts`] for the same
/// `(circuit, bindings)`: the creg width when counts are projected onto the
/// classical register, otherwise the full qubit register width. CLI front
/// ends use this to format bitstrings. Takes the same `bindings` as
/// [`run_counts`] so the two lower the circuit identically — deciding from
/// the unbound circuit would disagree whenever parameters are only
/// lowerable after binding (e.g. `sin(theta)`).
pub fn counts_width(circuit: &Circuit, bindings: &HashMap<String, f64>) -> usize {
    match concrete_ir(circuit, bindings) {
        Ok(low) if !low.needs_collapse && !measure_pairs(&low).is_empty() => {
            (low.ir.num_classical_bits as usize).max(1)
        }
        _ => circuit.n_qubits().max(1),
    }
}

/// Execute and return measurement counts (basis-state integer → count).
///
/// When the program measures into a classical register and all measurements
/// are terminal, counts are keyed over the creg (see
/// [`project_counts_onto_creg`]); a program with no `measure` statements
/// keeps the legacy full-register keying. Mid-circuit-measurement programs
/// (gates after a measure) also keep full-register keying, since the final
/// sample of a measured qubit can legitimately differ from the recorded
/// classical bit.
pub fn run_counts(
    circuit: &Circuit,
    bindings: &HashMap<String, f64>,
    shots: u32,
    seed: Option<u64>,
    sel: BackendSel,
) -> Result<ExecResult, String> {
    let low = concrete_ir(circuit, bindings)?;
    let cfg = ExecConfig {
        shots: Some(shots),
        seed,
        mid_circuit_mode: if low.needs_collapse {
            MidCircuitMode::Collapse
        } else {
            MidCircuitMode::Skip
        },
    };
    let res = with_backend_stats(sel, |b| {
        b.execute(&low.ir, &ParameterBinding::new(), &cfg)
            .map_err(|e| e.to_string())
    })?;
    if low.needs_collapse {
        return Ok(res);
    }
    // Do NOT project a result the backend already keyed on the creg.
    //
    // Above 64 qubits the MPS and stabilizer backends pack the outcome from the
    // classical register themselves, because the full-register key does not
    // exist there. Re-projecting reads QUBIT positions out of a CREG key, and
    // the result is not merely mislabelled — measured on a 70-qubit circuit
    // whose true marginal is 0.79/0.21, this front end reported |00> with
    // probability 1.000 while `omega-run` on the identical circuit reported
    // 1577/423. Certainty, on a distribution that has none.
    if omega_core::executor::counts_keyed_on_creg(&low.ir, low.needs_collapse) {
        return Ok(res);
    }
    project_counts_onto_creg(res, &measure_pairs(&low))
}

/// Parse a `--noise '{...}'` JSON string into a [`NoiseModel`].
///
/// Delegates to [`omega_core::noise::NoiseModel::from_json`], which accepts
/// both the scalar (idealized-machine) and per-qubit (calibrated-hardware)
/// forms and rejects unknown keys.
pub fn parse_noise_model(s: &str) -> Result<NoiseModel, String> {
    NoiseModel::from_json(s)
}

/// Execute with a per-gate noise model, returning measurement counts.
///
/// Noise is a trajectory (Monte-Carlo quantum-jump) simulation, so it is only
/// meaningful on the statevector backend (`--backend sim`/`statevector`). Any
/// other selection is a hard error rather than a silent noiseless run — the
/// whole point of this path is that `--noise` never gets dropped on the floor.
/// Otherwise mirrors [`run_counts`]: same lowering, same creg projection.
pub fn run_counts_noisy(
    circuit: &Circuit,
    bindings: &HashMap<String, f64>,
    shots: u32,
    seed: Option<u64>,
    sel: BackendSel,
    model: &NoiseModel,
) -> Result<ExecResult, String> {
    clear_mps_stats(); // trajectory path tracks no truncation stats
    let low = concrete_ir(circuit, bindings)?;
    let cfg = ExecConfig {
        shots: Some(shots),
        seed,
        mid_circuit_mode: if low.needs_collapse {
            MidCircuitMode::Collapse
        } else {
            MidCircuitMode::Skip
        },
    };
    // Both trajectory samplers apply the same shared model. CUDA's native-f64
    // SVD hook is wired when this build has it. The Metal f32 contraction is
    // not: see [`make_mps`]. Channels stay on the CPU either way.
    let backend: Box<dyn Backend> = match sel {
        BackendSel::Sim => Box::new(NoisyStatevectorBackend::with_model(model.clone(), seed)),
        BackendSel::Mps { chi } => Box::new(make_noisy_mps(chi, model.clone())),
        other => {
            return Err(format!(
                "--noise sampling is supported on --backend sim or mps; backend '{}' cannot \
                 apply a noise model to sampled counts (pauliprop applies noise to --expectation)",
                other.name()
            ));
        }
    };
    let res = backend
        .execute(&low.ir, &ParameterBinding::new(), &cfg)
        .map_err(|e| e.to_string())?;
    if low.needs_collapse {
        return Ok(res);
    }
    // Do NOT project a result the backend already keyed on the creg.
    //
    // Above 64 qubits the MPS and stabilizer backends pack the outcome from the
    // classical register themselves, because the full-register key does not
    // exist there. Re-projecting reads QUBIT positions out of a CREG key, and
    // the result is not merely mislabelled — measured on a 70-qubit circuit
    // whose true marginal is 0.79/0.21, this front end reported |00> with
    // probability 1.000 while `omega-run` on the identical circuit reported
    // 1577/423. Certainty, on a distribution that has none.
    if omega_core::executor::counts_keyed_on_creg(&low.ir, low.needs_collapse) {
        return Ok(res);
    }
    project_counts_onto_creg(res, &measure_pairs(&low))
}

/// Construct the noisy MPS trajectory backend with the same accelerators as
/// [`make_mps`]. Noise does not make an under-reported `discarded_weight` into
/// a bound, so the Metal f32 contraction stays off unless `MPS_METAL_CONTRACT`
/// is set — the same policy, the same warning.
///
/// NOTE(cuda): the CUDA SVD arm is compiled and wired identically to
/// [`make_mps`]. It is not executed on the CI host (no NVIDIA device).
fn make_noisy_mps(chi: usize, model: NoiseModel) -> NoisyMpsBackend {
    let backend = NoisyMpsBackend::with_model(chi, model);
    #[cfg(feature = "cuda")]
    let backend = backend.with_svd_fn(omega_backend_mps_cuda::cuda_svd_flat);
    apply_mps_metal_contract(backend)
}

/// Expectation value under a noise model. Only the Pauli-propagation backend
/// folds noise into an expectation value *exactly* (its Heisenberg adjoint), so
/// this routes there; `sim`/`mps` compute analytic (noiseless) expectations, so
/// noisy expectation on them is rejected — sample with `--shots` instead.
pub fn expectation_noisy(
    circuit: &Circuit,
    observable: &str,
    bindings: &HashMap<String, f64>,
    sel: BackendSel,
    model: &NoiseModel,
) -> Result<f64, String> {
    clear_mps_stats(); // pauliprop path tracks no MPS truncation stats
    if sel != BackendSel::PauliProp {
        return Err(format!(
            "--noise with --expectation is only supported on --backend pauliprop (its Heisenberg \
             adjoint folds noise into the expectation exactly); backend '{}' computes an analytic \
             noiseless expectation — use --shots for a noisy estimate instead",
            sel.name()
        ));
    }
    let low = concrete_ir(circuit, bindings)?;
    let obs = Observable::parse(observable)?;
    make_pauliprop(0.0, None, None)
        .with_noise(model.clone())
        .expectation(&low.ir, &ParameterBinding::new(), &obs)
        .map_err(|e| e.to_string())
}

/// Exact statevector (no sampling; measurement gates are skipped).
pub fn statevector(
    circuit: &Circuit,
    bindings: &HashMap<String, f64>,
    sel: BackendSel,
) -> Result<Vec<Complex64>, String> {
    let low = concrete_ir(circuit, bindings)?;
    let cfg = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let res = with_backend_stats(sel, |b| {
        b.execute(&low.ir, &ParameterBinding::new(), &cfg)
            .map_err(|e| e.to_string())
    })?;
    match res {
        ExecResult::Statevector(sv) => Ok(sv),
        other => Err(format!(
            "backend returned non-statevector result: {other:?}"
        )),
    }
}

pub fn expectation(
    circuit: &Circuit,
    observable: &str,
    bindings: &HashMap<String, f64>,
    sel: BackendSel,
) -> Result<f64, String> {
    let low = concrete_ir(circuit, bindings)?;
    // This path used to REFUSE any circuit with a `when c == v` gate, on the
    // grounds that the analytic backends evaluate without mid-circuit collapse
    // and would silently skip the guard. The error told the caller to "rewrite
    // the feedforward as coherent control" — which is precisely what
    // `omega_core::defer_measure` now does for them, so the refusal had become a
    // lie about a limitation that no longer exists. A circuit that genuinely
    // cannot be deferred is still refused, by the backend, naming the construct.
    let obs = Observable::parse(observable)?;
    with_backend_stats(sel, |b| {
        b.expectation(&low.ir, &ParameterBinding::new(), &obs)
            .map_err(|e| e.to_string())
    })
}

/// One-shot expectation **and** gradient of `observable` for `circuit` at the
/// given parameter values — the name-keyed front end for
/// `omega_core::gradient::compute_gradient_for`, so a caller never has to lower
/// by hand or touch `SymbolId`.
///
/// `bindings` maps Aria symbol names to values; any free symbol the map omits
/// defaults to `0.0` (as elsewhere in `run::*`). `only`, when `Some`, restricts
/// the returned gradient to those symbol names (frozen/layer-wise training) —
/// and for the per-symbol shift methods also skips the other symbols'
/// evaluations; `None` returns the gradient for every symbol. The result is
/// `(expectation, name → ∂⟨O⟩/∂name)`.
///
/// Like its `run::*` siblings this **re-lowers per call** — it is the one-shot
/// entry point, not a training inner loop (use `train_supervised` /
/// `train_expectation`, which lower once, for that).
pub fn expectation_with_gradient(
    circuit: &Circuit,
    observable: &str,
    bindings: &HashMap<String, f64>,
    sel: BackendSel,
    method: GradMethod,
    only: Option<&[&str]>,
) -> Result<(f64, HashMap<String, f64>), String> {
    clear_mps_stats(); // gradient path doesn't surface MPS truncation stats
                       // Lower WITHOUT folding the bindings into the IR: gradients differentiate
                       // w.r.t. the live symbols, so they must survive lowering (unlike
                       // `expectation`, which uses `concrete_ir`).
    let low = lower(circuit)?;
    let obs = Observable::parse(observable)?;

    let known = || {
        let mut names: Vec<&String> = low.symbol_ids.keys().collect();
        names.sort();
        names
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    // Validate every provided name up front (bindings and `only`), same error
    // shape as the trainer, so a typo is a clear message not a silent no-op.
    for name in bindings.keys() {
        if !low.symbol_ids.contains_key(name) {
            return Err(format!(
                "unknown symbol '{name}' in bindings (circuit has: {})",
                known()
            ));
        }
    }
    if let Some(sel_names) = only {
        for name in sel_names {
            if !low.symbol_ids.contains_key(*name) {
                return Err(format!(
                    "unknown symbol '{name}' in `only` (circuit has: {})",
                    known()
                ));
            }
        }
    }

    // Bind every symbol: provided value, else 0.0 (matching `concrete_ir`).
    let mut binding = ParameterBinding::new();
    for (name, &id) in &low.symbol_ids {
        binding.bind(id, bindings.get(name).copied().unwrap_or(0.0));
    }

    let backend = make_backend(sel)?;
    let value = backend
        .expectation(&low.ir, &binding, &obs)
        .map_err(|e| e.to_string())?;

    // Restrict to the named subset, if any (names already validated above).
    let only_ids: Option<HashSet<SymbolId>> = only.map(|names| {
        names
            .iter()
            .map(|n| low.symbol_ids[*n])
            .collect::<HashSet<_>>()
    });
    let grads = compute_gradient_for(
        backend.as_ref(),
        &low.ir,
        &binding,
        &obs,
        &method,
        only_ids.as_ref(),
    )
    .map_err(|e| e.to_string())?;

    // Map SymbolIds back to names — SymbolId never crosses the API boundary.
    let id_to_name: HashMap<SymbolId, &String> =
        low.symbol_ids.iter().map(|(n, &i)| (i, n)).collect();
    let mut out = HashMap::with_capacity(grads.len());
    for (id, g) in grads {
        // Every id from `compute_gradient_for` is a symbol of this lowered
        // circuit, so it must be in the map; a miss would be an engine bug, not
        // a value to silently drop.
        debug_assert!(
            id_to_name.contains_key(&id),
            "gradient for unknown SymbolId {id}"
        );
        if let Some(name) = id_to_name.get(&id) {
            out.insert((*name).clone(), g);
        }
    }
    Ok((value, out))
}

/// Truncation knobs for the Pauli-propagation backend (PauliPropagation.jl's
/// three axes). All-default = the exact engine.
#[derive(Clone, Copy, Default)]
pub struct PauliPropTruncation {
    /// Drop terms with coefficient magnitude below this (`0.0` = keep all).
    pub coeff_min: f64,
    /// Drop terms with Pauli weight above this (`None` = no cap).
    pub max_weight: Option<usize>,
    /// Drop terms above this split frequency / number of sin-branches
    /// (`None` = no cap).
    pub max_freq: Option<u32>,
}

/// Pauli-propagation expectation with explicit truncation, returning
/// `(value, dropped_mass)` — the certified L1 error budget. This is the Aria
/// front-end for the deep-non-Clifford regime PauliPropagation.jl targets; the
/// exact (all-default) engine is reachable through [`expectation`] as usual.
pub fn expectation_pauliprop(
    circuit: &Circuit,
    observable: &str,
    bindings: &HashMap<String, f64>,
    trunc: PauliPropTruncation,
) -> Result<(f64, f64), String> {
    clear_mps_stats();
    let low = concrete_ir(circuit, bindings)?;
    let obs = Observable::parse(observable)?;
    make_pauliprop(trunc.coeff_min, trunc.max_weight, trunc.max_freq)
        .expectation_with_budget(&low.ir, &ParameterBinding::new(), &obs)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_parse_covers_mps_fixed_and_adaptive() {
        assert_eq!(
            BackendSel::parse("mps").unwrap(),
            BackendSel::Mps {
                chi: DEFAULT_MPS_CHI
            }
        );
        assert_eq!(
            BackendSel::parse("mps:128").unwrap(),
            BackendSel::Mps { chi: 128 }
        );
        assert_eq!(
            BackendSel::parse("mps:auto").unwrap(),
            BackendSel::MpsAuto {
                max_chi: DEFAULT_MPS_AUTO_CEILING
            }
        );
        assert_eq!(
            BackendSel::parse("mps:auto:512").unwrap(),
            BackendSel::MpsAuto { max_chi: 512 }
        );
        assert_eq!(BackendSel::MpsAuto { max_chi: 512 }.name(), "mps:auto");
        // Malformed / zero are loud errors, not silent fallbacks.
        assert!(BackendSel::parse("mps:0").is_err());
        assert!(BackendSel::parse("mps:auto:0").is_err());
        assert!(BackendSel::parse("mps:auto:xyz").is_err());
        assert!(BackendSel::parse("wat").is_err());
    }
}

/// Production Metal policy for both MPS constructors. Re-adding
/// `with_contract_fn(metal_contract_2q)` at either site reddens the test that
/// calls it: `discarded_weight` stops matching the bare CPU backend, and the
/// Metal dispatch counter moves.
///
/// The two tests share one lock. `MPS_METAL_CONTRACT` and the discard ceiling
/// are process-wide, and cargo runs tests in this binary in parallel.
#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod mps_metal_certificate {
    use super::{make_mps, make_noisy_mps, set_mps_discard_ceiling};
    use omega_backend_mps::{MpsBackend, NoisyMpsBackend};
    use omega_core::executor::{Backend, ExecConfig, MidCircuitMode};
    use omega_core::noise::NoiseModel;
    use omega_core::params::ParameterBinding;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        // A failed sibling must not poison the others: the mutation check is
        // which assertion fired, and a poisoned lock turns every later test
        // into the same panic.
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Brickwall deep enough that the middle bond saturates χ = 32 and the
    /// run truncates. Below the Metal threshold, or with nothing discarded,
    /// reinstalling the f32 hook would leave `discarded_weight` unchanged and
    /// these tests could not go red.
    fn truncating_brickwall() -> omega_core::circuit::CircuitIR {
        let n = 12usize;
        let depth = 12usize;
        let mut src = format!("circuit BW() {{\n  qreg q[{n}]\n");
        for q in 0..n {
            src.push_str(&format!("  apply H on q[{q}]\n"));
        }
        for d in 0..depth {
            for q in 0..n {
                let ang = 0.17 + 0.01 * (q + d * n) as f64;
                src.push_str(&format!("  apply RY({ang}) on q[{q}]\n"));
                src.push_str(&format!("  apply RZ(0.23) on q[{q}]\n"));
            }
            let mut q = d % 2;
            while q + 1 < n {
                src.push_str(&format!("  apply CX on q[{q}], q[{}]\n", q + 1));
                q += 2;
            }
        }
        src.push_str("}\n");
        let circ = aria_core::ast::parse_aria(&src)
            .expect("parse")
            .instantiate("BW", &[])
            .expect("instantiate");
        crate::lower::lower(&circ).expect("lower").ir
    }

    fn production_env() {
        // SAFETY: test-only. `ENV_LOCK` is held by every test that touches
        // these, and no other test in this harness reads them.
        unsafe {
            std::env::remove_var("MPS_METAL_CONTRACT");
            std::env::remove_var("MPS_METAL_MIN_BOND");
        }
        set_mps_discard_ceiling(f64::INFINITY);
    }

    fn discarded_bits_mps(circuit: &omega_core::circuit::CircuitIR) -> (u64, f64, usize, u64) {
        let before = omega_backend_mps_metal::metal_contraction_count();
        let backend = make_mps(32);
        let obs = omega_core::executor::Observable::parse("Z0").expect("Z0");
        backend
            .expectation(circuit, &ParameterBinding::new(), &obs)
            .expect("mps expectation");
        let st = backend.last_run_stats();
        let ran = omega_backend_mps_metal::metal_contraction_count() - before;
        (
            st.discarded_weight.to_bits(),
            st.discarded_weight,
            st.max_bond_reached,
            ran,
        )
    }

    #[test]
    fn make_mps_discarded_weight_matches_bare_cpu() {
        let _guard = env_lock();
        production_env();
        let circuit = truncating_brickwall();

        let bare = MpsBackend::new(32).with_max_discarded_weight(f64::INFINITY);
        let obs = omega_core::executor::Observable::parse("Z0").expect("Z0");
        bare.expectation(&circuit, &ParameterBinding::new(), &obs)
            .expect("bare mps expectation");
        let bare_st = bare.last_run_stats();

        let (bits, dw, bond, ran) = discarded_bits_mps(&circuit);
        assert!(
            bare_st.max_bond_reached >= 32,
            "fixture max_bond_reached={} never reached MIN_BOND_DIM_FOR_METAL; \
             reinstalling the f32 hook would not change discarded_weight",
            bare_st.max_bond_reached
        );
        assert!(
            bare_st.discarded_weight > 0.0,
            "fixture discarded nothing ({}); both paths report zero and the hook \
             would not redden this test",
            bare_st.discarded_weight
        );
        assert_eq!(
            ran, 0,
            "make_mps dispatched {ran} Metal contractions; the f32 hook is installed"
        );
        assert_eq!(
            bits,
            bare_st.discarded_weight.to_bits(),
            "make_mps discarded_weight {dw} != bare CPU {} (bond {bond}, cpu {}). \
             The f32 contraction hook is on the production path, and a smaller \
             certificate is not a bound.",
            bare_st.discarded_weight,
            bare_st.max_bond_reached
        );
    }

    #[test]
    fn make_noisy_mps_discarded_weight_matches_bare_cpu() {
        let _guard = env_lock();
        production_env();
        let circuit = truncating_brickwall();
        let cfg = ExecConfig {
            shots: None,
            seed: Some(1),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let model = NoiseModel::default();

        let bare =
            NoisyMpsBackend::with_model(32, model.clone()).with_max_discarded_weight(f64::INFINITY);
        bare.execute(&circuit, &ParameterBinding::new(), &cfg)
            .expect("bare noisy mps");
        let bare_st = bare.last_run_stats();

        let before = omega_backend_mps_metal::metal_contraction_count();
        // The ceiling is applied inside `make_mps` only. Raise it after the
        // hook decision so a truncating fixture can finish and report.
        let hooked = make_noisy_mps(32, model).with_max_discarded_weight(f64::INFINITY);
        hooked
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .expect("noisy mps");
        let ran = omega_backend_mps_metal::metal_contraction_count() - before;
        let st = hooked.last_run_stats();

        assert!(
            bare_st.max_bond_reached >= 32 && bare_st.discarded_weight > 0.0,
            "noisy fixture did not truncate (bond {}, discarded {}); the hook \
             would not redden this test",
            bare_st.max_bond_reached,
            bare_st.discarded_weight
        );
        assert_eq!(
            ran, 0,
            "make_noisy_mps dispatched {ran} Metal contractions; the f32 hook is installed"
        );
        assert_eq!(
            st.discarded_weight.to_bits(),
            bare_st.discarded_weight.to_bits(),
            "make_noisy_mps discarded_weight {} != bare CPU {} (bond {}, cpu {}). \
             The f32 contraction hook is on the noisy production path, and a \
             smaller certificate is not a bound.",
            st.discarded_weight,
            bare_st.discarded_weight,
            st.max_bond_reached,
            bare_st.max_bond_reached
        );
    }

    #[test]
    fn mps_metal_contract_opt_in_dispatches() {
        let _guard = env_lock();
        production_env();
        // SAFETY: same lock as the production tests. Restored before return.
        unsafe { std::env::set_var("MPS_METAL_CONTRACT", "1") };
        let circuit = truncating_brickwall();
        let before = omega_backend_mps_metal::metal_contraction_count();
        let backend = make_mps(32);
        let obs = omega_core::executor::Observable::parse("Z0").expect("Z0");
        backend
            .expectation(&circuit, &ParameterBinding::new(), &obs)
            .expect("opt-in mps expectation");
        let ran = omega_backend_mps_metal::metal_contraction_count() - before;
        unsafe { std::env::remove_var("MPS_METAL_CONTRACT") };
        assert!(
            ran > 0,
            "MPS_METAL_CONTRACT=1 dispatched no Metal contraction; the opt-in is a no-op"
        );
    }
}
