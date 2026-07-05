// SPDX-License-Identifier: Apache-2.0
//! Run an Aria circuit on a pluggable execution backend.
//!
//! Every backend implements `omega_core::executor::Backend`. The default
//! `sim` backend is the pure-Rust statevector simulator; additional backends
//! (MPS, GPU, libtorch, remote) are added as features in later phases and
//! selected through [`BackendSel`].

use std::collections::HashMap;

use aria_core::ast::Circuit;
use num_complex::Complex64;
use omega_backend_mps::MpsBackend;
use omega_backend_pauliprop::PauliPropBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::GateKind as OGateKind;
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable};
use omega_core::params::ParameterBinding;

use crate::lower::{lower, Lowered};

/// Default MPS bond dimension when the MPS backend is selected.
const MPS_BOND: usize = 64;

/// Which execution backend to dispatch to. Every variant is an implementation
/// of `omega_core::executor::Backend` — the Aria plugin contract. New backends
/// (GPU, libtorch `tch`, remote omega-server) slot in here behind features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendSel {
    /// Pure-Rust CPU statevector (`omega-backend-statevector`). Exact, ≤~24 qubits.
    Sim,
    /// Pure-Rust matrix-product-state (`omega-backend-mps`). Scales to many
    /// qubits when entanglement is bounded.
    Mps,
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
            "mps" => Ok(Self::Mps),
            "gpu" => Ok(Self::Gpu),
            "tch" => Ok(Self::Tch),
            "pauliprop" => Ok(Self::PauliProp),
            other => Err(format!(
                "unknown backend '{other}' (available: sim, mps, gpu, tch, pauliprop; remote via --url)"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Sim => "sim",
            Self::Mps => "mps",
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
        BackendSel::Mps => Box::new(make_mps()),
        BackendSel::Gpu => make_gpu()?,
        BackendSel::Tch => make_tch()?,
        // Exact engine (no truncation: coeff_min = 0, max_weight = None), with
        // the GPU branch accelerator wired under a cuda build.
        BackendSel::PauliProp => Box::new(make_pauliprop(0.0, None, None)),
    })
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
/// the same code is exact-identical otherwise. Metal has no on-GPU SVD arm yet
/// (see GPU_BACKEND_PLAN.md), so the CPU SVD is used there.
fn make_mps() -> MpsBackend {
    let backend = MpsBackend::new(MPS_BOND);
    #[cfg(feature = "cuda")]
    let backend = backend.with_svd_fn(omega_backend_mps_cuda::cuda_svd_flat);
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
/// Empty when the program declares no measure-to-creg mapping. The single
/// source of the pair-extraction semantics — the remote path lowers the
/// same bound circuit and calls this too.
pub(crate) fn measure_pairs(low: &Lowered) -> Vec<(u32, u32)> {
    low.ir
        .ops
        .iter()
        .filter(|op| op.gate == OGateKind::Measure)
        .filter_map(|op| {
            let q = op.qubits.first()?.0;
            op.classical_bit.map(|c| (q, c))
        })
        .collect()
}

/// Project full-register sampled counts onto the classical register via the
/// program's `measure → creg` statements (OpenQASM semantics). Backends
/// sample the full qubit register at the end of the circuit; when the
/// program declares an explicit mapping, the reported counts must be keyed
/// over creg bits, one bit per `measure`, in `c[j]` order. A later measure
/// into the same classical bit overwrites the earlier one. Shared by the
/// local and remote (omega-server) execution paths.
///
/// Counts keys are `u64`, so a measure targeting bit ≥ 64 of either register
/// cannot be represented — that's a loud error, not a masked shift.
pub(crate) fn project_counts_onto_creg(
    res: ExecResult,
    pairs: &[(u32, u32)],
) -> Result<ExecResult, String> {
    if pairs.is_empty() {
        return Ok(res);
    }
    if let Some(&(q, c)) = pairs.iter().find(|&&(q, c)| q >= 64 || c >= 64) {
        return Err(format!(
            "measure q[{q}] -> c[{c}]: sampled-count keys are u64, so register \
             indices ≥ 64 cannot be reported; reduce the register or drop --shots"
        ));
    }
    match res {
        ExecResult::Counts(counts) => {
            let mut projected: HashMap<u64, u32> = HashMap::new();
            for (outcome, n) in counts {
                let mut key = 0u64;
                for &(q, c) in pairs {
                    let bit = (outcome >> q) & 1;
                    key = (key & !(1u64 << c)) | (bit << c);
                }
                *projected.entry(key).or_insert(0) += n;
            }
            Ok(ExecResult::Counts(projected))
        }
        other => Ok(other),
    }
}

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
    let res = make_backend(sel)?
        .execute(&low.ir, &ParameterBinding::new(), &cfg)
        .map_err(|e| e.to_string())?;
    if low.needs_collapse {
        return Ok(res);
    }
    project_counts_onto_creg(res, &measure_pairs(&low))
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
    match make_backend(sel)?
        .execute(&low.ir, &ParameterBinding::new(), &cfg)
        .map_err(|e| e.to_string())?
    {
        ExecResult::Statevector(sv) => Ok(sv),
        other => Err(format!(
            "backend returned non-statevector result: {other:?}"
        )),
    }
}

/// Expectation value `⟨ψ|O|ψ⟩` of a Pauli observable string (e.g. `"Z0"`,
/// `"1.0*Z0 Z1"`), parsed by `omega_core::executor::Observable::parse`.
pub fn expectation(
    circuit: &Circuit,
    observable: &str,
    bindings: &HashMap<String, f64>,
    sel: BackendSel,
) -> Result<f64, String> {
    let low = concrete_ir(circuit, bindings)?;
    let obs = Observable::parse(observable)?;
    make_backend(sel)?
        .expectation(&low.ir, &ParameterBinding::new(), &obs)
        .map_err(|e| e.to_string())
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
    let low = concrete_ir(circuit, bindings)?;
    let obs = Observable::parse(observable)?;
    make_pauliprop(trunc.coeff_min, trunc.max_weight, trunc.max_freq)
        .expectation_with_budget(&low.ir, &ParameterBinding::new(), &obs)
        .map_err(|e| e.to_string())
}
