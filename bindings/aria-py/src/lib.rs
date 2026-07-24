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
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::CircuitIR;
use omega_core::executor::{Backend, Observable};
use omega_core::gradient::{compute_gradient_for, GradMethod};
use omega_core::params::ParameterBinding;

fn err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Backend from a spec string: `"sv"`/`"statevector"` or `"mps"`/`"mps:<chi>"`.
fn make_backend(spec: &str) -> PyResult<Box<dyn Backend>> {
    match spec {
        "sv" | "sim" | "statevector" => Ok(Box::new(StatevectorBackend::new())),
        "mps" => Ok(Box::new(MpsBackend::new(64))),
        s if s.starts_with("mps:") => {
            let chi: usize = s[4..]
                .parse()
                .map_err(|_| err(format!("bad mps spec '{s}' (want mps:<int>)")))?;
            if chi == 0 {
                return Err(err("mps bond dimension must be >= 1"));
            }
            Ok(Box::new(MpsBackend::new(chi)))
        }
        _ => Err(err(format!(
            "unknown backend '{spec}' (sv | mps | mps:<chi>)"
        ))),
    }
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

    /// `⟨O⟩` for one parameter vector.
    #[pyo3(signature = (params, observable, backend="sv"))]
    fn expectation(&self, params: Vec<f64>, observable: &str, backend: &str) -> PyResult<f64> {
        let obs = Observable::parse(observable).map_err(err)?;
        let be = make_backend(backend)?;
        be.expectation(&self.ir, &self.binding(&params)?, &obs)
            .map_err(|e| err(e.to_string()))
    }

    /// `∂⟨O⟩/∂param` for one parameter vector (adjoint AD; aligned to `symbols`).
    #[pyo3(signature = (params, observable, backend="sv"))]
    fn gradient(&self, params: Vec<f64>, observable: &str, backend: &str) -> PyResult<Vec<f64>> {
        let obs = Observable::parse(observable).map_err(err)?;
        let be = make_backend(backend)?;
        let g = compute_gradient_for(
            be.as_ref(),
            &self.ir,
            &self.binding(&params)?,
            &obs,
            &GradMethod::Adjoint,
            None,
        )
        .map_err(|e| err(e.to_string()))?;
        Ok(self.align(g))
    }

    /// `⟨O⟩` across many parameter rows → one value per row (backend-parallel
    /// on the statevector backend).
    #[pyo3(signature = (rows, observable, backend="sv"))]
    fn expectation_batch(
        &self,
        rows: Vec<Vec<f64>>,
        observable: &str,
        backend: &str,
    ) -> PyResult<Vec<f64>> {
        let obs = Observable::parse(observable).map_err(err)?;
        let be = make_backend(backend)?;
        let bnds: Vec<ParameterBinding> = rows
            .iter()
            .map(|r| self.binding(r))
            .collect::<PyResult<_>>()?;
        let refs: Vec<&ParameterBinding> = bnds.iter().collect();
        be.expectation_batch(&self.ir, &refs, &obs)
            .map_err(|e| err(e.to_string()))
    }

    /// `∂⟨O⟩/∂param` across many rows → `(rows × params)`, aligned to `symbols`.
    /// Uses the backend's batched adjoint (parallel on statevector), falling
    /// back to per-row parameter-shift when a backend lacks adjoint AD.
    #[pyo3(signature = (rows, observable, backend="sv"))]
    fn gradient_batch(
        &self,
        rows: Vec<Vec<f64>>,
        observable: &str,
        backend: &str,
    ) -> PyResult<Vec<Vec<f64>>> {
        let obs = Observable::parse(observable).map_err(err)?;
        let be = make_backend(backend)?;
        let bnds: Vec<ParameterBinding> = rows
            .iter()
            .map(|r| self.binding(r))
            .collect::<PyResult<_>>()?;
        let refs: Vec<&ParameterBinding> = bnds.iter().collect();
        let batched = be
            .adjoint_gradient_batch(&self.ir, &refs, &obs)
            .map_err(|e| err(e.to_string()))?;
        let mut out = Vec::with_capacity(rows.len());
        for (i, row) in batched.into_iter().enumerate() {
            match row {
                Some(g) => out.push(self.align(g)),
                None => {
                    let g = compute_gradient_for(
                        be.as_ref(),
                        &self.ir,
                        &bnds[i],
                        &obs,
                        &GradMethod::ParameterShift,
                        None,
                    )
                    .map_err(|e| err(e.to_string()))?;
                    out.push(self.align(g));
                }
            }
        }
        Ok(out)
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

#[pymodule]
fn _aria_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(load_source, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
