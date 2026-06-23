//! Pauli (stabilizer) backend bench: 50-qubit Clifford chain sample.

use criterion::{criterion_group, criterion_main, Criterion};
use smallvec::smallvec;

use omega_backend_pauli::PauliBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, ExecConfig, MidCircuitMode};
use omega_core::params::ParameterBinding;

const NUM_QUBITS: u32 = 50;

fn clifford_chain(num_qubits: u32) -> CircuitIR {
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
    for q in 0..num_qubits - 1 {
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
            gate: GateKind::S,
            qubits: smallvec![Qubit(q)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }
    circuit
}

fn bench_pauli(c: &mut Criterion) {
    let circuit = clifford_chain(NUM_QUBITS);
    let params = ParameterBinding::new();
    let cfg = ExecConfig {
        shots: Some(128),
        seed: Some(0),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    c.bench_function("pauli_50q_clifford_chain_128shots", |b| {
        b.iter(|| {
            let backend = PauliBackend::new();
            backend.execute(&circuit, &params, &cfg).unwrap()
        });
    });
}

criterion_group!(benches, bench_pauli);
criterion_main!(benches);
