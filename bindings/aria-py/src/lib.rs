// SPDX-License-Identifier: Apache-2.0
//! Python bindings for the Aria runtime (pyo3). Lower an Aria circuit ONCE,
//! then evaluate expectations and adjoint gradients over parameter vectors —
//! the surface a PyTorch `autograd.Function` needs. Parameter vectors align to
//! `Model.symbols` (ascending SymbolId order — the documented binding contract).
//!
//! Pure Rust underneath (no libtorch): the compiled extension links the
//! statevector / MPS backends directly.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use aria_core::ast::parse_aria;
use aria_runtime::lower::lower;
use omega_backend_mps::MpsBackend;
use omega_backend_pauliprop::PauliPropBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::CircuitIR;
use omega_core::executor::{Backend, Observable};
use omega_core::gradient::{compute_gradient_for, GradMethod};
use omega_core::params::ParameterBinding;

fn err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Build the MPS backend, routing its bond-compression SVD through cuSOLVER
/// `gesvdj` under a `cuda` build (mirrors `aria_runtime::run::make_mps`). The
/// accelerator itself falls back to the CPU Jacobi SVD when no device is
/// present, so the result is exact-identical either way.
fn make_mps(chi: usize) -> MpsBackend {
    let backend = MpsBackend::new(chi);
    #[cfg(feature = "cuda")]
    let backend = backend.with_svd_fn(omega_backend_mps_cuda::cuda_svd_flat);
    // Metal arm: the two-site θ-contraction on the GPU, SVD on the CPU. `not(cuda)`
    // for the same reason the runtime has it — a dual-vendor build keeps CUDA's
    // native-f64 gesvdj over the f32 Metal contraction.
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let backend = backend.with_contract_fn(omega_backend_mps_metal::metal_contract_2q);
    backend
}

/// Build the Pauli-propagation backend, installing the CUDA branch-expansion
/// accelerator under a `cuda` build (mirrors `aria_runtime::run::make_pauliprop`).
/// Truncation defaults match the runtime's `--backend pauliprop`: none, i.e.
/// `coeff_min = 0.0` and no weight/frequency cap, so Python gets the same
/// backend the CLI does rather than a quietly-truncated variant. As with the
/// MPS SVD, the accelerator falls back to the CPU branch per gate when no
/// device is present or the term count is small, so the result is identical
/// either way — which is why this spec does not error on a missing device.
fn make_pauliprop() -> PauliPropBackend {
    let backend = PauliPropBackend::with_truncation_freq(0.0, None, None);
    #[cfg(feature = "cuda")]
    let backend = backend.with_branch_hook(omega_backend_pauliprop_cuda::cuda_branch);
    // Metal arm, `not(cuda)` because a branch hook is single-valued.
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let backend = backend.with_branch_hook(omega_backend_pauliprop_metal::metal_branch);
    backend
}

/// Which GPU accelerator this wheel was compiled with, if any.
///
/// Mirrors `aria_runtime::run`'s cfg priority so the Python lane and the CLI
/// resolve `gpu` to the same arm on the same build.
const ACCELERATOR: Option<&str> = if cfg!(feature = "metal") {
    Some("metal")
} else if cfg!(feature = "cuda") {
    Some("cuda")
} else if cfg!(feature = "opencl") {
    Some("opencl")
} else {
    None
};

/// The accelerated statevector, whichever accelerator was compiled in.
///
/// Errors — never falls back to the CPU — when the device is unusable. A silent
/// fallback here would make a "GPU" benchmark quietly measure the CPU, which is
/// how misleading numbers get published. `mps` and `pauliprop` behave the other
/// way on purpose (see `make_mps` / `make_pauliprop`): there the accelerator
/// speeds up one step inside an otherwise-identical algorithm, so falling back
/// changes speed, not semantics.
#[allow(unreachable_code, unused_variables)]
fn make_gpu(pin: Option<&str>) -> PyResult<Box<dyn Backend + Send + Sync>> {
    if let (Some(want), Some(have)) = (pin, ACCELERATOR) {
        if want != have {
            return Err(err(format!(
                "backend 'gpu:{want}' was requested but this wheel was built with \
                 '{have}'. Build with `maturin build --features {want}`, or use the \
                 vendor-neutral 'gpu' spec."
            )));
        }
    }
    #[cfg(feature = "metal")]
    {
        return omega_backend_statevector_metal::MetalStatevectorBackend::new()
            .map(|b| Box::new(b) as Box<dyn Backend + Send + Sync>)
            .map_err(|e| err(format!("Metal statevector unavailable: {e:?}")));
    }
    #[cfg(feature = "cuda")]
    {
        return omega_backend_statevector_cuda::CudaStatevectorBackend::new()
            .map(|b| Box::new(b) as Box<dyn Backend + Send + Sync>)
            .map_err(|e| err(format!("CUDA statevector unavailable: {e:?}")));
    }
    #[cfg(feature = "opencl")]
    {
        return omega_backend_statevector_opencl::OpenClStatevectorBackend::new()
            .map(|b| Box::new(b) as Box<dyn Backend + Send + Sync>)
            .map_err(|e| err(format!("OpenCL statevector unavailable: {e:?}")));
    }
    Err(err(
        "this aria_py wheel has no GPU accelerator compiled in. Rebuild with \
         `maturin build --features cuda` (NVIDIA), `--features metal` (Apple), \
         or `--features opencl`.",
    ))
}

/// Backend from a spec string. **The spec names the engine, never the vendor.**
///
/// `"sv"`/`"statevector"` · `"mps"`/`"mps:<chi>"` · `"pauliprop"`/`"pp"` · `"gpu"`
///
/// These are the same names `aria_runtime::run::BackendSel` accepts, so a script
/// does not change when it moves between an NVIDIA box and an Apple one: the
/// accelerator is chosen at build time and applied inside the spec. `mps` and
/// `pauliprop` need no GPU variant — their accelerator (the SVD, the branch
/// expansion) is transparent and wired in by the same cfg the runtime uses.
///
/// For deliberately benchmarking one arm, pin it: `"gpu:cuda"`, `"gpu:metal"`,
/// `"gpu:opencl"`. A pin this wheel was not built with is an error naming what
/// it *was* built with — never a silent downgrade.
///
/// `"sv:cuda"`, `"mps:cuda[:<chi>]"` and `"pauliprop:cuda"` are **deprecated
/// aliases** kept so wheels already in use keep working; prefer the neutral
/// spellings.
fn make_backend(spec: &str) -> PyResult<Box<dyn Backend + Send + Sync>> {
    match spec {
        "sv" | "sim" | "statevector" => Ok(Box::new(StatevectorBackend::new())),
        "mps" => Ok(Box::new(make_mps(64))),
        "pauliprop" | "pp" => Ok(Box::new(make_pauliprop())),
        "gpu" => make_gpu(None),
        s if s.starts_with("gpu:") => make_gpu(Some(&s[4..])),
        // --- deprecated vendor-named aliases (see the doc comment) ---
        "sv:cuda" | "statevector:cuda" => make_gpu(Some("cuda")),
        "pauliprop:cuda" => Ok(Box::new(make_pauliprop())),
        "mps:cuda" => Ok(Box::new(make_mps(64))),
        s if s.starts_with("mps:cuda:") => {
            let chi: usize = s[9..]
                .parse()
                .map_err(|_| err(format!("bad mps spec '{s}' (want mps:cuda:<int>)")))?;
            if chi == 0 {
                return Err(err("mps bond dimension must be >= 1"));
            }
            Ok(Box::new(make_mps(chi)))
        }
        s if s.starts_with("mps:") => {
            let chi: usize = s[4..]
                .parse()
                .map_err(|_| err(format!("bad mps spec '{s}' (want mps:<int>)")))?;
            if chi == 0 {
                return Err(err("mps bond dimension must be >= 1"));
            }
            Ok(Box::new(make_mps(chi)))
        }
        _ => Err(err(format!(
            "unknown backend '{spec}' (sv | mps | mps:<chi> | pauliprop | gpu | \
             gpu:<cuda|metal|opencl>)"
        ))),
    }
}

/// A backend, constructed once and reused across calls.
///
/// This exists because construction is not free: a CUDA backend builds a device
/// handle, and on a GB10 that costs **316 ms** (median of 12; `"sv"` costs
/// 0.000 ms). Paying it per call made one `gradient_batch` at n=6, L=3, batch 32
/// take 429-604 ms where a reused backend takes 51-52 ms — 8.4x to 11.5x across
/// two runs, the spread being machine load on the per-call side only. A training
/// loop calls these methods thousands of times, so construction, not the
/// simulation, was the cost.
///
///     be = aria_py.Backend("gpu")
///     for _ in range(steps):
///         g = model.gradient_batch(rows, "Z0", be)   # one device handle, reused
///
/// Passing a spec string still works and still builds a fresh backend per call,
/// which is the right default for `sv` (construction is trivial) and the wrong
/// one for anything with a device behind it.
///
/// **Deliberately `unsendable`: a `Backend` belongs to the thread that built
/// it.** For parallel work build **one `Backend` per thread or per process**.
///
/// The reason has CHANGED, and the old one is no longer true. This used to say
/// `CudaStatevectorBackend` implements neither `Send` nor `Sync`; it now does,
/// via `unsafe impl`s that were written blind on an Apple host and then
/// **verified on a CUDA host** (DGX Spark GB10, CUDA 13.0.88, 2026-08-17). So
/// `unsendable` is now a POLICY choice, not a soundness necessity:
///
/// * The soundness argument for the CUDA impls reduces to one `Mutex` — the
///   graph cache — and holds only while every use of a cached `CUgraphExec`
///   stays inside that guard. That invariant is upheld today and the compiler
///   will not notice if a future change breaks it.
/// * Sharing one backend across threads buys nothing anyway: all CUDA calls go
///   through ONE stream, so two host threads serialise on the device even with
///   the GIL released.
///
/// `unsendable` costs nothing given the per-thread guidance, and keeps a
/// mis-shared handle raising a Python exception instead of relying on that
/// `Mutex` invariant surviving future edits.
///
/// Note this is independent of the GIL release added to `Model`'s compute
/// methods: [`Python::allow_threads`] runs its closure on the SAME thread, so
/// `unsendable` does not block it. What the release needed was
/// `dyn Backend + Send + Sync`, which is a property of the trait object, not of
/// this pyclass.
#[pyclass(name = "Backend", unsendable)]
struct PyBackend {
    inner: Box<dyn Backend + Send + Sync>,
    spec: String,
}

#[pymethods]
impl PyBackend {
    #[new]
    fn new(spec: &str) -> PyResult<Self> {
        Ok(Self {
            inner: make_backend(spec)?,
            spec: spec.to_string(),
        })
    }

    /// The spec string this backend was built from.
    #[getter]
    fn spec(&self) -> &str {
        &self.spec
    }

    /// Which accelerator this backend actually runs on: the compiled-in one for
    /// a `gpu` spec, `"cpu"` for everything else.
    ///
    /// `mps` and `pauliprop` report `"cpu"` even on an accelerated wheel, and
    /// that is the considered answer rather than an oversight: the algorithm
    /// runs on the CPU and the device accelerates one step inside it (a
    /// bond-compression SVD, a branch expansion), transparently and with a
    /// per-operation fallback. Reporting `"cuda"` there would imply the whole
    /// engine moved.
    ///
    /// An earlier version keyed off the spec STRING and so contradicted itself:
    /// `mps:cuda` said `"cuda"` while `mps:cuda:16` and plain `mps` said
    /// `"cpu"`, for three backends with the identical SVD hook wired in.
    #[getter]
    fn accelerator(&self) -> &str {
        if self.spec == "gpu" || self.spec.starts_with("gpu:") {
            ACCELERATOR.unwrap_or("cpu")
        } else {
            "cpu"
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Backend({:?}, accelerator={:?})",
            self.spec,
            self.accelerator()
        )
    }
}

/// Either a backend built for this one call, or a borrowed reusable one.
enum Be<'py> {
    Owned(Box<dyn Backend + Send + Sync>),
    Reused(PyRef<'py, PyBackend>),
}

impl Be<'_> {
    /// The `+ Send + Sync` is what lets the returned reference cross into a
    /// [`Python::allow_threads`] closure: that closure must be `Send`, so the
    /// `&dyn Backend` it captures must be `Send`, which requires the trait
    /// object to be `Sync`.
    ///
    /// Measured, not assumed — `crates/omega-cli/tests/backends_are_send_sync.rs`
    /// asserts it for every backend at compile time and names the ones a given
    /// host could not check.
    fn as_dyn(&self) -> &(dyn Backend + Send + Sync) {
        match self {
            Be::Owned(b) => b.as_ref(),
            Be::Reused(r) => r.inner.as_ref(),
        }
    }
}

/// Accept `None` (default `sv`), a spec string, or a reusable `Backend`.
fn resolve_backend<'py>(arg: Option<&Bound<'py, PyAny>>) -> PyResult<Be<'py>> {
    let Some(obj) = arg else {
        return Ok(Be::Owned(make_backend("sv")?));
    };
    if let Ok(be) = obj.downcast::<PyBackend>() {
        return Ok(Be::Reused(be.borrow()));
    }
    let spec: String = obj
        .extract()
        .map_err(|_| err("backend must be a spec string or an aria_py.Backend built from one"))?;
    Ok(Be::Owned(make_backend(&spec)?))
}

/// A lowered Aria circuit: evaluate it over parameter vectors.
#[pyclass]
struct Model {
    ir: CircuitIR,
    /// Symbol names in ascending SymbolId order (params align to this).
    symbols: Vec<String>,
    /// SymbolIds ascending, aligned with `symbols`.
    ids: Vec<u32>,
}

impl Model {
    fn binding(&self, params: &[f64]) -> PyResult<ParameterBinding> {
        if params.len() != self.ids.len() {
            return Err(err(format!(
                "expected {} params (one per symbol), got {}",
                self.ids.len(),
                params.len()
            )));
        }
        let mut b = ParameterBinding::new();
        for (&id, &v) in self.ids.iter().zip(params) {
            b.bind(id, v);
        }
        Ok(b)
    }

    /// Reorder a `(SymbolId, grad)` list to ascending-id order (aligns with
    /// `symbols`); symbols with no gradient entry get 0.
    fn align(&self, g: Vec<(u32, f64)>) -> Vec<f64> {
        let m: HashMap<u32, f64> = g.into_iter().collect();
        self.ids
            .iter()
            .map(|id| m.get(id).copied().unwrap_or(0.0))
            .collect()
    }
}

#[pymethods]
impl Model {
    /// Ordered symbol names (ascending SymbolId). Parameter vectors align here.
    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.symbols.clone()
    }

    #[getter]
    fn num_symbols(&self) -> usize {
        self.symbols.len()
    }

    #[getter]
    fn num_qubits(&self) -> u32 {
        self.ir.num_qubits
    }

    /// `⟨O⟩` for one parameter vector. `backend` is a spec string or a
    /// reusable `Backend` (see that class — reuse matters on a device).
    #[pyo3(signature = (params, observable, backend=None))]
    fn expectation(
        &self,
        py: Python<'_>,
        params: Vec<f64>,
        observable: &str,
        backend: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        let obs = Observable::parse(observable).map_err(err)?;
        let be = resolve_backend(backend)?;
        let bnd = self.binding(&params)?;
        let dyn_be = be.as_dyn();
        py.allow_threads(|| dyn_be.expectation(&self.ir, &bnd, &obs))
            .map_err(|e| err(e.to_string()))
    }

    /// `∂⟨O⟩/∂param` for one parameter vector (adjoint AD; aligned to `symbols`).
    #[pyo3(signature = (params, observable, backend=None))]
    fn gradient(
        &self,
        py: Python<'_>,
        params: Vec<f64>,
        observable: &str,
        backend: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<f64>> {
        let obs = Observable::parse(observable).map_err(err)?;
        let be = resolve_backend(backend)?;
        let bnd = self.binding(&params)?;
        let dyn_be = be.as_dyn();
        let g = py
            .allow_threads(|| {
                compute_gradient_for(dyn_be, &self.ir, &bnd, &obs, &GradMethod::Adjoint, None)
            })
            .map_err(|e| err(e.to_string()))?;
        Ok(self.align(g))
    }

    /// `⟨O⟩` across many parameter rows → one value per row (backend-parallel
    /// on the statevector backend).
    #[pyo3(signature = (rows, observable, backend=None))]
    fn expectation_batch(
        &self,
        py: Python<'_>,
        rows: Vec<Vec<f64>>,
        observable: &str,
        backend: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<f64>> {
        let obs = Observable::parse(observable).map_err(err)?;
        let be = resolve_backend(backend)?;
        let bnds: Vec<ParameterBinding> = rows
            .iter()
            .map(|r| self.binding(r))
            .collect::<PyResult<_>>()?;
        let refs: Vec<&ParameterBinding> = bnds.iter().collect();
        let dyn_be = be.as_dyn();
        py.allow_threads(|| dyn_be.expectation_batch(&self.ir, &refs, &obs))
            .map_err(|e| err(e.to_string()))
    }

    /// `∂⟨O⟩/∂param` across many rows → `(rows × params)`, aligned to `symbols`.
    /// Uses the backend's batched adjoint (parallel on statevector), falling
    /// back to per-row parameter-shift when a backend lacks adjoint AD.
    #[pyo3(signature = (rows, observable, backend=None))]
    fn gradient_batch(
        &self,
        py: Python<'_>,
        rows: Vec<Vec<f64>>,
        observable: &str,
        backend: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let obs = Observable::parse(observable).map_err(err)?;
        let be = resolve_backend(backend)?;
        let bnds: Vec<ParameterBinding> = rows
            .iter()
            .map(|r| self.binding(r))
            .collect::<PyResult<_>>()?;
        let refs: Vec<&ParameterBinding> = bnds.iter().collect();
        let dyn_be = be.as_dyn();

        // ONE release around the whole loop, not one per row. The
        // parameter-shift fallback runs `2 * num_params` circuits per row and is
        // the slowest of the four methods, so re-acquiring the GIL between rows
        // would hand most of the benefit back — and `align` is pure arithmetic
        // on `Vec<f64>`, needing no GIL, so there is nothing forcing a break.
        let raw: Result<Vec<Vec<(u32, f64)>>, _> = py.allow_threads(|| {
            let batched = dyn_be.adjoint_gradient_batch(&self.ir, &refs, &obs)?;
            let mut out = Vec::with_capacity(batched.len());
            for (i, row) in batched.into_iter().enumerate() {
                match row {
                    Some(g) => out.push(g),
                    None => out.push(compute_gradient_for(
                        dyn_be,
                        &self.ir,
                        &bnds[i],
                        &obs,
                        &GradMethod::ParameterShift,
                        None,
                    )?),
                }
            }
            Ok::<_, omega_core::error::OmegaError>(out)
        });
        Ok(raw
            .map_err(|e| err(e.to_string()))?
            .into_iter()
            .map(|g| self.align(g))
            .collect())
    }
}

/// Load + lower an Aria circuit from a `.aria` file.
#[pyfunction]
#[pyo3(signature = (path, circuit, ints=None))]
fn load(path: &str, circuit: &str, ints: Option<Vec<(String, i64)>>) -> PyResult<Model> {
    let src = std::fs::read_to_string(path).map_err(|e| err(format!("read {path}: {e}")))?;
    load_source(&src, circuit, ints)
}

/// Load + lower an Aria circuit from source text.
#[pyfunction]
#[pyo3(signature = (source, circuit, ints=None))]
fn load_source(source: &str, circuit: &str, ints: Option<Vec<(String, i64)>>) -> PyResult<Model> {
    let ints = ints.unwrap_or_default();
    let int_refs: Vec<(&str, i64)> = ints.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let prog = parse_aria(source).map_err(err)?;
    let circ = prog.instantiate(circuit, &int_refs).map_err(err)?;
    let low = lower(&circ).map_err(err)?;
    let mut pairs: Vec<(u32, String)> = low
        .symbol_ids
        .iter()
        .map(|(n, &i)| (i, n.clone()))
        .collect();
    pairs.sort_by_key(|(i, _)| *i);
    let ids = pairs.iter().map(|(i, _)| *i).collect();
    let symbols = pairs.into_iter().map(|(_, n)| n).collect();
    Ok(Model {
        ir: low.ir,
        symbols,
        ids,
    })
}

/// The GPU accelerator this wheel was built with: `"cuda"`, `"metal"`,
/// `"opencl"`, or `None` for a pure-CPU wheel.
///
/// Branch on this rather than on a vendor spec string — the point of the
/// neutral `gpu` spec is that the same code runs on either machine.
#[pyfunction]
fn accelerator() -> Option<&'static str> {
    ACCELERATOR
}

/// Backend specs this wheel accepts, neutral spellings only. `gpu` appears only
/// when an accelerator was compiled in, so this doubles as a capability probe.
#[pyfunction]
fn backends() -> Vec<&'static str> {
    let mut v = vec!["sv", "mps", "mps:<chi>", "pauliprop"];
    if ACCELERATOR.is_some() {
        v.push("gpu");
    }
    v
}

#[pymodule]
fn _aria_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<PyBackend>()?;
    m.add_function(wrap_pyfunction!(accelerator, m)?)?;
    m.add_function(wrap_pyfunction!(backends, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(load_source, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
