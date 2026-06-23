//! MPS backend benchmark: deep circuit push into bond-dim truncation.

use criterion::{criterion_group, criterion_main, Criterion};
use smallvec::smallvec;

use omega_backend_mps::MpsBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, MidCircuitMode};
use omega_core::params::ParameterBinding;

const NUM_QUBITS: u32 = 14;

fn entangling_circuit(num_qubits: u32, depth: usize) -> CircuitIR {
    let mut circuit = CircuitIR::new(num_qubits, CircuitType::GateBased);
    for q in 0..num_qubits {
        circuit.ops.push(GateOp {
            gate: GateKind::H,
            qubits: smallvec![Qubit(q)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }
    for d in 0..depth {
        let offset = d as u32 & 1;
        for q in (offset..num_qubits - 1).step_by(2) {
            circuit.ops.push(GateOp {
                gate: GateKind::CX,
                qubits: smallvec![Qubit(q), Qubit(q + 1)],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
            circuit.ops.push(GateOp {
                gate: GateKind::Rz,
                qubits: smallvec![Qubit(q + 1)],
                params: smallvec![ParamExpr::Concrete(0.25)],
                classical_bit: None,
                condition: None,
            });
            circuit.ops.push(GateOp {
                gate: GateKind::CX,
                qubits: smallvec![Qubit(q), Qubit(q + 1)],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        for q in 0..num_qubits {
            circuit.ops.push(GateOp {
                gate: GateKind::Rx,
                qubits: smallvec![Qubit(q)],
                params: smallvec![ParamExpr::Concrete(0.15)],
                classical_bit: None,
                condition: None,
            });
        }
    }
    circuit
}

fn bench_mps(c: &mut Criterion) {
    let params = ParameterBinding::new();
    let cfg = ExecConfig {
        shots: None,
        seed: Some(0),
        mid_circuit_mode: MidCircuitMode::Skip,
    };

    let circuit = entangling_circuit(NUM_QUBITS, 4);
    c.bench_function("mps_14q_brickwall_chi128", |b| {
        b.iter(|| {
            let backend = MpsBackend::new(128);
            backend.execute(&circuit, &params, &cfg).unwrap()
        });
    });
}

criterion_group!(benches, bench_mps);
criterion_main!(benches);
