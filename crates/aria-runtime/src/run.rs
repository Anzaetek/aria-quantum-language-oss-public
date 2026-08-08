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
        return Ok(Box::new(aria_backend_tch::TchBackend::cpu()));
    }
    Err(
        "aria was built without the libtorch backend; rebuild with `--features tch` \
         (and set LIBTORCH — see INSTALL_LIBTORCH.md)"
            .to_string(),
    )
}

/// Construct the MPS backend. Under a `cuda` build its bond-compression SVD is
/// routed through the cuSOLVER `gesvdj` accelerator (`omega-backend-mps-cuda`),
/// which itself falls back to the CPU Jacobi SVD when no CUDA device is present
/// — so `--backend mps` transparently uses the GPU when one is available and
/// the same code is exact-identical otherwise. Under a `metal` build the
/// two-site θ-contraction is routed to the GPU instead (SVD stays on CPU —
/// Apple has no native f64, so on-GPU Jacobi SVD is deferred; see
/// GPU_BACKEND_PLAN.md), engaging only above the bond-dim threshold.
fn make_mps(chi: usize) -> MpsBackend {
    let backend = MpsBackend::new(chi);
    #[cfg(feature = "cuda")]
    let backend = backend.with_svd_fn(omega_backend_mps_cuda::cuda_svd_flat);
    // Metal arm: the two-site θ-contraction runs on the GPU (SVD stays on CPU —
    // Apple has no native f64, so on-GPU Jacobi SVD is deferred; see
    // GPU_BACKEND_PLAN.md). Only engages above the bond-dim threshold; below it,
    // and when no device is present, `--backend mps` is exact-f64 as before.
    // `not(cuda)` so a (hypothetical) dual-vendor build keeps CUDA's native-f64
    // gesvdj SVD rather than the f32 Metal contraction.
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let backend = backend.with_contract_fn(omega_backend_mps_metal::metal_contract_2q);
    backend
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
    // Both trajectory samplers apply the same shared model. MPS composes with
    // whatever GPU accelerators this build wired (Metal θ-contraction / CUDA
    // SVD) — the channels are CPU-side, the heavy contraction stays on device.
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
    project_counts_onto_creg(res, &measure_pairs(&low))
}

/// Construct the noisy MPS trajectory backend with the same GPU accelerators as
/// [`make_mps`] — noise composes with them (see [`NoisyMpsBackend`]). The
/// channels run on the CPU (f64, exact); the heavy two-qubit contraction / bond
/// SVD stays on the device.
///
/// NOTE(cuda): the CUDA SVD arm is compiled and wired identically to
/// [`make_mps`], but — unlike the Metal arm, which is exercised on macOS in this
/// repo's tests — it is not executed on the CI host (no NVIDIA device). Its
/// composition with noise is code-identical to the verified Metal path.
fn make_noisy_mps(chi: usize, model: NoiseModel) -> NoisyMpsBackend {
    let backend = NoisyMpsBackend::with_model(chi, model);
    #[cfg(feature = "cuda")]
    let backend = backend.with_svd_fn(omega_backend_mps_cuda::cuda_svd_flat);
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let backend = backend.with_contract_fn(omega_backend_mps_metal::metal_contract_2q);
    backend
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

/// Expectation value `⟨ψ|O|ψ⟩` of a Pauli observable string (e.g. `"Z0"`,
/// `"1.0*Z0 Z1"`), parsed by `omega_core::executor::Observable::parse`.
/// Refuse an expectation/gradient on a circuit whose classically-conditioned
/// gates would be silently skipped.
///
/// `Backend::expectation` hardcodes `MidCircuitMode::Skip`
/// (`omega-backend-statevector/src/sim.rs`), so a `when c == v { .. }` gate
/// never fires on this path. The counts path handles feedforward correctly —
/// only expectation and gradient do not — which makes the failure especially
/// easy to miss: the same circuit gives right answers with `--shots` and wrong
/// ones with `--expectation`.
///
/// Measured: `X q0; measure q0 -> c0; when c0 == 1 { X q1 }` returns
/// `⟨Z1⟩ = +1.0` where the correct value is `−1.0`, because the conditional
/// `X` is dropped. A plausible number, silently wrong, on the path QML uses.
///
/// So refuse. Returning a value here and documenting the caveat elsewhere is
/// the same mistake as a governor advertising headroom it does not have: the
/// number gets used, the caveat does not.
fn reject_feedforward_on_analytic_path(low: &Lowered, what: &str) -> Result<(), String> {
    if low.needs_collapse {
        return Err(format!(
            "{what} cannot be computed for a circuit with classically-conditioned gates: \
             the analytic path evaluates without mid-circuit collapse, so every \
             `when c == v` gate would be silently skipped and the result would be \
             plausible but wrong. Use `--shots` (the sampled path executes feedforward \
             correctly), or rewrite the feedforward as coherent control."
        ));
    }
    Ok(())
}

pub fn expectation(
    circuit: &Circuit,
    observable: &str,
    bindings: &HashMap<String, f64>,
    sel: BackendSel,
) -> Result<f64, String> {
    let low = concrete_ir(circuit, bindings)?;
    reject_feedforward_on_analytic_path(&low, "expectation")?;
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
