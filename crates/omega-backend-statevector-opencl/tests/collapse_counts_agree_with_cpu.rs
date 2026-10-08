// SPDX-License-Identifier: Apache-2.0
//! **Collapse mode must not be answered differently from the CPU when the
//! circuit has no `Measure`.**
//!
//! The OpenCL collapse refusal looks for a `Measure` only. With shots and a
//! declared creg the CPU keys collapse-mode counts on the CLASSICAL register,
//! even when nothing writes it. A circuit forced into collapse by a
//! conditioned gate alone (`creg c[1]; h q[1]; if (c == 0) x q[0];`) passes
//! the OpenCL check and would be answered by the qubit-register sampler. The
//! CUDA backend had exactly this defect; see
//! `omega-backend-statevector-cuda/tests/counts_width_and_collapse.rs`.
//!
//! Either answer is acceptable here: a refusal naming the unsupported mode, so
//! the caller routes to the CPU, or counts equal to the CPU's.
#![cfg(feature = "opencl")]

use omega_backend_statevector::StatevectorBackend;
use omega_backend_statevector_opencl::{OpenClError, OpenClStatevectorBackend};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::outcome::Outcome;
use omega_core::params::ParameterBinding;
use smallvec::smallvec;
use std::collections::HashMap;

fn conditioned_without_measure_circuit() -> CircuitIR {
    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    c.num_classical_bits = 1;
    c.add_op(GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    // `if (c == 0) x q[0];` — the creg is never written, so the gate fires.
    c.add_op(GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: Some((0, 1, 0)),
    });
    c
}

fn render(m: &HashMap<Outcome, u32>) -> Vec<(String, u32)> {
    let mut v: Vec<(String, u32)> = m.iter().map(|(k, n)| (k.to_bitstring(), *n)).collect();
    v.sort();
    v
}

#[test]
fn collapse_without_a_measure_agrees_with_the_cpu() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(OpenClError::Unavailable(_)) | Err(OpenClError::Runtime(_)) => {
            eprintln!("SKIP (NOT A PASS): OpenCL unavailable");
            return;
        }
        Err(e) => panic!("unexpected backend init error: {e}"),
    };
    let c = conditioned_without_measure_circuit();
    assert!(
        omega_core::executor::needs_collapse(&c),
        "fixture sanity: a conditioned gate forces collapse"
    );
    let cfg = ExecConfig {
        shots: Some(64),
        seed: Some(5),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let ExecResult::Counts(cpu) = StatevectorBackend::new()
        .execute(&c, &ParameterBinding::default(), &cfg)
        .expect("cpu runs collapse mode")
    else {
        panic!("cpu shots run must return Counts");
    };

    match backend.execute(&c, &ParameterBinding::default(), &cfg) {
        Err(e) => assert!(
            {
                let m = format!("{e}");
                m.contains("collapse mode") || m.contains("mid-circuit measurement")
            },
            "a refusal must name what is unsupported; got: {e}"
        ),
        Ok(ExecResult::Counts(got)) => assert_eq!(
            render(&got),
            render(&cpu),
            "OpenCL answered collapse mode differently from the CPU on a \
             circuit with no Measure"
        ),
        Ok(other) => panic!("shots run returned a non-Counts result: {other:?}"),
    }
}
