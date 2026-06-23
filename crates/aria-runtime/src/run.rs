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
use omega_backend_statevector::StatevectorBackend;
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
}

impl BackendSel {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "sim" | "statevector" => Ok(Self::Sim),
            "mps" => Ok(Self::Mps),
            "gpu" => Ok(Self::Gpu),
            "tch" => Ok(Self::Tch),
            other => Err(format!(
                "unknown backend '{other}' (available: sim, mps, gpu, tch; remote via --url)"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Sim => "sim",
            Self::Mps => "mps",
            Self::Gpu => "gpu",
            Self::Tch => "tch",
        }
    }
}

/// Construct the selected backend (`omega_core::Backend`). Shared by `run` and
/// `train` so the chosen backend drives both execution and gradients.
pub(crate) fn make_backend(sel: BackendSel) -> Result<Box<dyn Backend>, String> {
    Ok(match sel {
        BackendSel::Sim => Box::new(StatevectorBackend::new()),
        BackendSel::Mps => Box::new(MpsBackend::new(MPS_BOND)),
        BackendSel::Gpu => make_gpu()?,
        BackendSel::Tch => make_tch()?,
    })
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

/// Execute and return measurement counts (basis-state integer → count).
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
    make_backend(sel)?
        .execute(&low.ir, &ParameterBinding::new(), &cfg)
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
