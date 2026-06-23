//! Cross-CPU adjoint AD parity for the OpenCL backend.
//!
//! Mirror of Metal's `adjoint_metal_matches_cpu_12q_hea`. The
//! adjoint algorithm (forward sweep → ν = O|ψ⟩ → backward sweep
//! with per-param `inner_product` ride-on) is identical to the CPU
//! and Metal versions; the only divergence is the f32-on-device
//! arithmetic, which caps the achievable accuracy at ~1e-5 vs the
//! CPU's f64-throughout 1e-10. The 12q × 16-param HEA shape
//! exercises the forward fusion walker, the per-gate `apply_op_dagger`
//! map, the per-(op, sym) `apply_op_derivative_inplace` map plus the
//! `copy_into` scratch path, and a mixed Z / Z⊗Z / X observable (the
//! X term forces the observable initialiser through the host
//! `apply_observable_host` fallback rather than any diagonal fast
//! path).

#![cfg(feature = "opencl")]

use omega_backend_statevector::StatevectorBackend;
use omega_backend_statevector_opencl::OpenClStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

fn build_12q_16p_hea() -> (CircuitIR, ParameterBinding, Observable) {
    let n: u32 = 12;
    let mut circuit = CircuitIR::new(n, CircuitType::GateBased);
    for s in 0..16u32 {
        circuit.symbols.insert(s, format!("theta_{s}"));
    }
    // Layer 1: Ry on first 8 qubits (params 0..8).
    for q in 0..8 {
        circuit.ops.push(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(q)],
            params: smallvec![ParamExpr::Symbol(q)],
            classical_bit: None,
            condition: None,
        });
    }
    // Linear CX entangling ladder.
    for q in 0..n - 1 {
        circuit.ops.push(GateOp {
            gate: GateKind::CX,
            qubits: smallvec![Qubit(q), Qubit(q + 1)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }
    // Layer 2: Rz on the last 8 qubits (params 8..16). These collapse
    // into one `apply_diagonal_product` dispatch under the forward
    // fusion walker — exercises that path.
    for (i, q) in (4..n).enumerate() {
        circuit.ops.push(GateOp {
            gate: GateKind::Rz,
            qubits: smallvec![Qubit(q)],
            params: smallvec![ParamExpr::Symbol(8 + i as u32)],
            classical_bit: None,
            condition: None,
        });
    }

    let mut params = ParameterBinding::new();
    for s in 0..16u32 {
        let v = ((s as f64) * 0.317 - 0.84).sin() * 1.4;
        params.bind(s, v);
    }
    let obs = Observable {
        terms: vec![
            (1.0, vec![(0, PauliOp::Z)]),
            (0.5, vec![(5, PauliOp::Z)]),
            (-0.7, vec![(0, PauliOp::Z), (11, PauliOp::Z)]),
            (0.3, vec![(3, PauliOp::X)]),
        ],
    };
    (circuit, params, obs)
}

#[test]
fn adjoint_opencl_matches_cpu_12q_hea() {
    let opencl = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return, // no OpenCL ICD — skip.
    };
    let (circuit, params, obs) = build_12q_16p_hea();

    let cpu = StatevectorBackend::new();
    let cpu_grads = cpu
        .adjoint_gradient(&circuit, &params, &obs)
        .expect("cpu adjoint")
        .expect("cpu has adjoint");

    let opencl_grads = opencl
        .adjoint_gradient(&circuit, &params, &obs)
        .expect("opencl adjoint")
        .expect("opencl has adjoint");

    assert_eq!(cpu_grads.len(), 16);
    assert_eq!(opencl_grads.len(), 16);

    let mut max_diff = 0.0_f64;
    for ((sa, ga), (sb, gb)) in cpu_grads.iter().zip(opencl_grads.iter()) {
        assert_eq!(sa, sb, "symbol ordering must match");
        let diff = (ga - gb).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff < 1e-5,
            "symbol {sa}: cpu = {ga:.10}, opencl = {gb:.10}, diff = {diff:.3e}",
        );
    }
    eprintln!("12q/16-param HEA: max abs diff vs CPU adjoint = {max_diff:.3e}");
}

#[test]
fn adjoint_opencl_falls_back_to_param_shift_on_measure() {
    // Circuits with Measure return `Ok(None)` from `adjoint_gradient`
    // so the trainer falls back to parameter-shift. Same shape as
    // CPU + Metal contracts.
    let opencl = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
    circuit.symbols.insert(0, "theta".into());
    circuit.num_classical_bits = 2;
    circuit.ops.push(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });
    circuit.ops.push(GateOp {
        gate: GateKind::Measure,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: Some(0),
        condition: None,
    });
    let mut params = ParameterBinding::new();
    params.bind(0, 0.5);
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };
    let out = opencl
        .adjoint_gradient(&circuit, &params, &obs)
        .expect("adjoint must surface as Ok(_)");
    assert!(
        out.is_none(),
        "expected Ok(None) on Measure to trigger param-shift fallback",
    );
}

#[test]
fn adjoint_opencl_matches_cpu_4q_6p() {
    // Smaller 4q / 6-param shape — single forward pass, simpler
    // observable. Mirrors Metal's second adjoint parity test
    // (`adjoint_metal_matches_cpu`). Exercises the Rx + Rz + CX
    // dagger chain and the per-param derivative on Rx / Ry / Rz.
    let opencl = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let n: u32 = 4;
    let mut circuit = CircuitIR::new(n, CircuitType::GateBased);
    for s in 0..6u32 {
        circuit.symbols.insert(s, format!("p_{s}"));
    }
    let push = |c: &mut CircuitIR, gate, q: u32, s: u32| {
        c.ops.push(GateOp {
            gate,
            qubits: smallvec![Qubit(q)],
            params: smallvec![ParamExpr::Symbol(s)],
            classical_bit: None,
            condition: None,
        });
    };
    push(&mut circuit, GateKind::Rx, 0, 0);
    push(&mut circuit, GateKind::Ry, 1, 1);
    push(&mut circuit, GateKind::Rz, 2, 2);
    circuit.ops.push(GateOp {
        gate: GateKind::CX,
        qubits: smallvec![Qubit(0), Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    circuit.ops.push(GateOp {
        gate: GateKind::CX,
        qubits: smallvec![Qubit(2), Qubit(3)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    push(&mut circuit, GateKind::Rz, 1, 3);
    push(&mut circuit, GateKind::Ry, 2, 4);
    push(&mut circuit, GateKind::Rx, 3, 5);

    let mut params = ParameterBinding::new();
    for s in 0..6u32 {
        params.bind(s, ((s as f64) * 0.41).cos());
    }
    let obs = Observable {
        terms: vec![
            (1.0, vec![(0, PauliOp::Z), (3, PauliOp::Z)]),
            (-0.4, vec![(1, PauliOp::Y)]),
        ],
    };

    let cpu_grads = StatevectorBackend::new()
        .adjoint_gradient(&circuit, &params, &obs)
        .unwrap()
        .unwrap();
    let opencl_grads = opencl
        .adjoint_gradient(&circuit, &params, &obs)
        .unwrap()
        .unwrap();
    assert_eq!(cpu_grads.len(), 6);
    let mut max = 0.0_f64;
    for ((sa, ga), (sb, gb)) in cpu_grads.iter().zip(opencl_grads.iter()) {
        assert_eq!(sa, sb);
        let d = (ga - gb).abs();
        if d > max {
            max = d;
        }
        assert!(
            d < 1e-5,
            "4q/6p: symbol {sa}: cpu={ga:.10}, opencl={gb:.10}, diff={d:.3e}",
        );
    }
    eprintln!("4q/6-param shape: max abs diff vs CPU adjoint = {max:.3e}");
}
