//! Analytic derivatives of the photonic unitary transfer matrix.
//!
//! For a circuit U = U_n · ... · U_2 · U_1, the derivative with respect to
//! parameter k of component j is:
//!   dU/dk = U_n · ... · U_{j+1} · dU_j/dk · U_{j-1} · ... · U_1
//!
//! The SLOS expectation is too expensive to differentiate through (permanent),
//! so photonics gradients use parameter-shift. This module provides the unitary
//! derivative infrastructure for future use (process fidelity, etc.).

use num_complex::Complex64;

use omega_core::circuit::*;
use omega_core::error::Result;
use omega_core::params::ParameterBinding;

use crate::components::{self, PhotonicOp};

/// Derivative info for one parameter: which component, which sub-parameter.
struct ParamLocation {
    /// Index into the ops array.
    op_idx: usize,
    /// 0 for the first parameter (phi for PS, theta for BS), 1 for the second (phi for BS).
    sub_param: usize,
    /// The SymbolId this parameter derives from.
    symbol: SymbolId,
    /// Chain rule factor: d(resolved_angle)/d(symbol).
    chain_factor: f64,
}

/// Compute dU/d(symbol) for each active symbol in the circuit.
///
/// Returns pairs of (SymbolId, derivative_matrix).
pub fn circuit_unitary_derivatives(
    num_modes: usize,
    circuit: &CircuitIR,
    params: &ParameterBinding,
) -> Result<Vec<(SymbolId, Vec<Vec<Complex64>>)>> {
    // Extract photonic ops with resolved parameters
    let ops = extract_ops(circuit, params)?;

    // Build prefix products: prefix[i] = U_{i-1} · ... · U_0
    // prefix[0] = I, prefix[k] = U_{k-1} · prefix[k-1]
    let n_ops = ops.len();
    let mut prefixes: Vec<Vec<Vec<Complex64>>> = Vec::with_capacity(n_ops + 1);
    prefixes.push(components::identity(num_modes));
    for i in 0..n_ops {
        let u_i = component_matrix(num_modes, &ops[i]);
        let prev = &prefixes[i];
        prefixes.push(components::mat_mul(&u_i, prev));
    }

    // Build suffix products: suffix[i] = U_{n-1} · ... · U_i
    // suffix[n] = I, suffix[k] = suffix[k+1] · U_k
    let mut suffixes: Vec<Vec<Vec<Complex64>>> = vec![vec![]; n_ops + 1];
    suffixes[n_ops] = components::identity(num_modes);
    for i in (0..n_ops).rev() {
        let u_i = component_matrix(num_modes, &ops[i]);
        suffixes[i] = components::mat_mul(&suffixes[i + 1], &u_i);
    }

    // Find all parameter locations with chain factors
    let locations = find_param_locations(circuit, params)?;

    // For each symbol, accumulate dU/d(symbol) = Σ suffix[j+1] · dU_j/d(sub) · prefix[j] · chain
    let mut grad_map: std::collections::HashMap<SymbolId, Vec<Vec<Complex64>>> =
        std::collections::HashMap::new();

    for loc in &locations {
        let du_j = component_derivative_matrix(num_modes, &ops[loc.op_idx], loc.sub_param);
        // dU/d(raw_param) = suffix[j+1] · dU_j · prefix[j]
        let tmp = components::mat_mul(&du_j, &prefixes[loc.op_idx]);
        let full = components::mat_mul(&suffixes[loc.op_idx + 1], &tmp);

        let entry = grad_map
            .entry(loc.symbol)
            .or_insert_with(|| zero_matrix(num_modes));

        let factor = Complex64::new(loc.chain_factor, 0.0);
        for i in 0..num_modes {
            for j in 0..num_modes {
                entry[i][j] += factor * full[i][j];
            }
        }
    }

    let mut result: Vec<(SymbolId, Vec<Vec<Complex64>>)> = grad_map.into_iter().collect();
    result.sort_by_key(|(id, _)| *id);
    Ok(result)
}

/// Build the m×m matrix for a single photonic component.
fn component_matrix(m: usize, op: &PhotonicOp) -> Vec<Vec<Complex64>> {
    let mut u = components::identity(m);
    match op {
        PhotonicOp::PhaseShifter { mode, phi } => {
            u[*mode][*mode] = Complex64::new(phi.cos(), phi.sin());
        }
        PhotonicOp::BeamSplitterRx {
            mode0,
            mode1,
            theta,
            phi,
        } => {
            let ct = theta.cos();
            let st = theta.sin();
            let eip = Complex64::new(phi.cos(), phi.sin());
            let eim = Complex64::new(phi.cos(), -phi.sin());
            u[*mode0][*mode0] = Complex64::new(ct, 0.0);
            u[*mode0][*mode1] = -eip * st;
            u[*mode1][*mode0] = eim * st;
            u[*mode1][*mode1] = Complex64::new(ct, 0.0);
        }
    }
    u
}

/// Build the m×m derivative matrix dU_j/d(sub_param) for a single component.
fn component_derivative_matrix(m: usize, op: &PhotonicOp, sub_param: usize) -> Vec<Vec<Complex64>> {
    let mut du = zero_matrix(m);
    match (op, sub_param) {
        // d(PhaseShifter)/dφ: d(e^{iφ})/dφ = i·e^{iφ}
        (PhotonicOp::PhaseShifter { mode, phi }, 0) => {
            let i_unit = Complex64::new(0.0, 1.0);
            du[*mode][*mode] = i_unit * Complex64::new(phi.cos(), phi.sin());
        }
        // d(BS)/dθ: [[-sin, -e^{iφ}·cos], [e^{-iφ}·cos, -sin]]
        (
            PhotonicOp::BeamSplitterRx {
                mode0,
                mode1,
                theta,
                phi,
            },
            0,
        ) => {
            let ct = theta.cos();
            let st = theta.sin();
            let eip = Complex64::new(phi.cos(), phi.sin());
            let eim = Complex64::new(phi.cos(), -phi.sin());
            du[*mode0][*mode0] = Complex64::new(-st, 0.0);
            du[*mode0][*mode1] = -eip * ct;
            du[*mode1][*mode0] = eim * ct;
            du[*mode1][*mode1] = Complex64::new(-st, 0.0);
        }
        // d(BS)/dφ: [[0, -i·e^{iφ}·sin], [-i·e^{-iφ}·sin, 0]]
        (
            PhotonicOp::BeamSplitterRx {
                mode0,
                mode1,
                theta,
                phi,
            },
            1,
        ) => {
            let st = theta.sin();
            let i_unit = Complex64::new(0.0, 1.0);
            let eip = Complex64::new(phi.cos(), phi.sin());
            let eim = Complex64::new(phi.cos(), -phi.sin());
            du[*mode0][*mode1] = -i_unit * eip * st;
            du[*mode1][*mode0] = -i_unit * eim * st;
        }
        _ => {} // invalid sub_param — returns zero matrix
    }
    du
}

fn zero_matrix(m: usize) -> Vec<Vec<Complex64>> {
    vec![vec![Complex64::new(0.0, 0.0); m]; m]
}

/// Extract PhotonicOps from circuit (same logic as sim.rs but public for gradient use).
fn extract_ops(circuit: &CircuitIR, params: &ParameterBinding) -> Result<Vec<PhotonicOp>> {
    let mut ops = Vec::new();
    for gate_op in &circuit.ops {
        match &gate_op.gate {
            GateKind::PhaseShifter => {
                let phi = params.resolve(&gate_op.params[0])?;
                let mode = gate_op.qubits[0].0 as usize;
                ops.push(PhotonicOp::PhaseShifter { mode, phi });
            }
            GateKind::BeamSplitterRx => {
                let theta = params.resolve(&gate_op.params[0])?;
                let phi = params.resolve(&gate_op.params[1])?;
                let mode0 = gate_op.qubits[0].0 as usize;
                let mode1 = gate_op.qubits[1].0 as usize;
                ops.push(PhotonicOp::BeamSplitterRx {
                    mode0,
                    mode1,
                    theta,
                    phi,
                });
            }
            GateKind::Barrier | GateKind::Measure => {}
            _ => {}
        }
    }
    Ok(ops)
}

/// Find all (op_idx, sub_param, symbol, chain_factor) tuples.
fn find_param_locations(
    circuit: &CircuitIR,
    params: &ParameterBinding,
) -> Result<Vec<ParamLocation>> {
    let mut locations = Vec::new();
    let mut op_idx = 0usize;

    for gate_op in &circuit.ops {
        match &gate_op.gate {
            GateKind::PhaseShifter | GateKind::BeamSplitterRx => {
                for (sub_param, param_expr) in gate_op.params.iter().enumerate() {
                    for &sym in &collect_symbols(param_expr) {
                        let chain = params.resolve_derivative(param_expr, sym)?;
                        if chain.abs() > 1e-30 {
                            locations.push(ParamLocation {
                                op_idx,
                                sub_param,
                                symbol: sym,
                                chain_factor: chain,
                            });
                        }
                    }
                }
                op_idx += 1;
            }
            _ => {}
        }
    }

    Ok(locations)
}

fn collect_symbols(expr: &ParamExpr) -> Vec<SymbolId> {
    let mut syms = Vec::new();
    collect_inner(expr, &mut syms);
    syms.sort();
    syms.dedup();
    syms
}

fn collect_inner(expr: &ParamExpr, out: &mut Vec<SymbolId>) {
    match expr {
        ParamExpr::Symbol(id) => out.push(*id),
        ParamExpr::Negate(inner) => collect_inner(inner, out),
        ParamExpr::Add(a, b) | ParamExpr::Mul(a, b) => {
            collect_inner(a, out);
            collect_inner(b, out);
        }
        ParamExpr::Concrete(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::build_unitary;

    #[test]
    fn test_phase_shifter_derivative_vs_finite_diff() {
        let phi = 0.7;
        let eps = 1e-7;
        let m = 3;
        let mode = 1;

        let ops_plus = vec![PhotonicOp::PhaseShifter {
            mode,
            phi: phi + eps,
        }];
        let ops_minus = vec![PhotonicOp::PhaseShifter {
            mode,
            phi: phi - eps,
        }];
        let u_plus = build_unitary(m, &ops_plus);
        let u_minus = build_unitary(m, &ops_minus);

        let du_analytic =
            component_derivative_matrix(m, &PhotonicOp::PhaseShifter { mode, phi }, 0);

        for i in 0..m {
            for j in 0..m {
                let fd = (u_plus[i][j] - u_minus[i][j]) / (2.0 * eps);
                assert!(
                    (du_analytic[i][j] - fd).norm() < 1e-5,
                    "PS deriv[{}][{}]: analytic={}, fd={}",
                    i,
                    j,
                    du_analytic[i][j],
                    fd
                );
            }
        }
    }

    #[test]
    fn test_beam_splitter_derivative_theta_vs_finite_diff() {
        let theta = 0.5;
        let phi = 0.3;
        let eps = 1e-7;
        let m = 3;

        let ops_plus = vec![PhotonicOp::BeamSplitterRx {
            mode0: 0,
            mode1: 1,
            theta: theta + eps,
            phi,
        }];
        let ops_minus = vec![PhotonicOp::BeamSplitterRx {
            mode0: 0,
            mode1: 1,
            theta: theta - eps,
            phi,
        }];
        let u_plus = build_unitary(m, &ops_plus);
        let u_minus = build_unitary(m, &ops_minus);

        let op = PhotonicOp::BeamSplitterRx {
            mode0: 0,
            mode1: 1,
            theta,
            phi,
        };
        let du_analytic = component_derivative_matrix(m, &op, 0);

        for i in 0..m {
            for j in 0..m {
                let fd = (u_plus[i][j] - u_minus[i][j]) / (2.0 * eps);
                assert!(
                    (du_analytic[i][j] - fd).norm() < 1e-5,
                    "BS dtheta[{}][{}]: analytic={}, fd={}",
                    i,
                    j,
                    du_analytic[i][j],
                    fd
                );
            }
        }
    }

    #[test]
    fn test_beam_splitter_derivative_phi_vs_finite_diff() {
        let theta = 0.5;
        let phi = 0.3;
        let eps = 1e-7;
        let m = 3;

        let ops_plus = vec![PhotonicOp::BeamSplitterRx {
            mode0: 0,
            mode1: 1,
            theta,
            phi: phi + eps,
        }];
        let ops_minus = vec![PhotonicOp::BeamSplitterRx {
            mode0: 0,
            mode1: 1,
            theta,
            phi: phi - eps,
        }];
        let u_plus = build_unitary(m, &ops_plus);
        let u_minus = build_unitary(m, &ops_minus);

        let op = PhotonicOp::BeamSplitterRx {
            mode0: 0,
            mode1: 1,
            theta,
            phi,
        };
        let du_analytic = component_derivative_matrix(m, &op, 1);

        for i in 0..m {
            for j in 0..m {
                let fd = (u_plus[i][j] - u_minus[i][j]) / (2.0 * eps);
                assert!(
                    (du_analytic[i][j] - fd).norm() < 1e-5,
                    "BS dphi[{}][{}]: analytic={}, fd={}",
                    i,
                    j,
                    du_analytic[i][j],
                    fd
                );
            }
        }
    }

    #[test]
    fn test_circuit_unitary_derivative_chain_rule() {
        // 3-component circuit: PS(φ₀) on mode 0, BS(θ₁, φ₁) on (0,1), PS(φ₂) on mode 1
        // Verify dU/d(symbol) via finite differences
        use std::collections::HashMap;

        let m = 2;
        let mut circuit = CircuitIR::new(m as u32, CircuitType::Photonic);
        circuit.symbols.insert(0, "phi0".to_string());
        circuit.symbols.insert(1, "theta1".to_string());
        circuit.symbols.insert(2, "phi_tr1".to_string());
        circuit.symbols.insert(3, "phi2".to_string());

        circuit.add_op(GateOp {
            gate: GateKind::PhaseShifter,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::BeamSplitterRx,
            qubits: smallvec::smallvec![Qubit(0), Qubit(1)],
            params: smallvec::smallvec![ParamExpr::Symbol(1), ParamExpr::Symbol(2)],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::PhaseShifter,
            qubits: smallvec::smallvec![Qubit(1)],
            params: smallvec::smallvec![ParamExpr::Symbol(3)],
            classical_bit: None,
            condition: None,
        });

        let vals = [0.5, 0.8, 0.3, 1.2];
        let mut params = ParameterBinding::new();
        for (i, &v) in vals.iter().enumerate() {
            params.bind(i as u32, v);
        }

        let derivs = circuit_unitary_derivatives(m, &circuit, &params).unwrap();
        let deriv_map: HashMap<u32, &Vec<Vec<Complex64>>> =
            derivs.iter().map(|(id, d)| (*id, d)).collect();

        // Verify each symbol against finite differences
        let eps = 1e-7;
        for sym in 0..4u32 {
            let mut p_plus = params.clone();
            p_plus.bind(sym, vals[sym as usize] + eps);
            let mut p_minus = params.clone();
            p_minus.bind(sym, vals[sym as usize] - eps);

            let ops_plus = extract_ops(&circuit, &p_plus).unwrap();
            let ops_minus = extract_ops(&circuit, &p_minus).unwrap();
            let u_plus = crate::components::build_unitary(m, &ops_plus);
            let u_minus = crate::components::build_unitary(m, &ops_minus);

            let du_analytic = deriv_map.get(&sym).unwrap();
            for i in 0..m {
                for j in 0..m {
                    let fd = (u_plus[i][j] - u_minus[i][j]) / (2.0 * eps);
                    assert!(
                        (du_analytic[i][j] - fd).norm() < 1e-4,
                        "sym {} du[{}][{}]: analytic={}, fd={}",
                        sym,
                        i,
                        j,
                        du_analytic[i][j],
                        fd
                    );
                }
            }
        }
    }
}
