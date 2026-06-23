//! Statevector backend benchmarks.
//!
//! Baseline metrics the GPU work needs to beat. Keep kernel-focused and
//! sized so a full run fits in a few minutes on commodity hardware.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use smallvec::smallvec;

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, MidCircuitMode, Observable, PauliOp};
use omega_core::params::ParameterBinding;

const NUM_QUBITS: u32 = 20;

/// Qubit counts swept across to expose where memory bandwidth becomes
/// the bottleneck. The statevector at n qubits is 16·2^n bytes
/// (complex64), so n=20 is 16 MiB — past L2 cache on most cores. The
/// bench harness computes per-operation throughput from these.
const SWEEP_QUBITS: &[u32] = &[12, 14, 16, 18, 20];

fn bell_chain(num_qubits: u32) -> CircuitIR {
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
    circuit
}

fn qaoa_layer(num_qubits: u32) -> CircuitIR {
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
    // One layer of ZZ + RX (ring topology).
    for q in 0..num_qubits {
        let a = q;
        let b = (q + 1) % num_qubits;
        circuit.ops.push(GateOp {
            gate: GateKind::CX,
            qubits: smallvec![Qubit(a), Qubit(b)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.ops.push(GateOp {
            gate: GateKind::Rz,
            qubits: smallvec![Qubit(b)],
            params: smallvec![ParamExpr::Concrete(0.37)],
            classical_bit: None,
            condition: None,
        });
        circuit.ops.push(GateOp {
            gate: GateKind::CX,
            qubits: smallvec![Qubit(a), Qubit(b)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }
    for q in 0..num_qubits {
        circuit.ops.push(GateOp {
            gate: GateKind::Rx,
            qubits: smallvec![Qubit(q)],
            params: smallvec![ParamExpr::Concrete(0.49)],
            classical_bit: None,
            condition: None,
        });
    }
    circuit
}

fn bench_statevector(c: &mut Criterion) {
    let cfg = ExecConfig {
        shots: None,
        seed: Some(0),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let params = ParameterBinding::new();

    // Headline benches at 20 qubits — the long-standing baseline numbers
    // recorded in benches/results/cpu-aarch64-darwin.json.
    let circuit = bell_chain(NUM_QUBITS);
    c.bench_function("sv_20q_bell_chain_exec", |b| {
        b.iter(|| {
            let backend = StatevectorBackend::new();
            backend.execute(&circuit, &params, &cfg).unwrap()
        });
    });

    let qaoa = qaoa_layer(NUM_QUBITS);
    c.bench_function("sv_20q_qaoa_layer_exec", |b| {
        b.iter(|| {
            let backend = StatevectorBackend::new();
            backend.execute(&qaoa, &params, &cfg).unwrap()
        });
    });

    let zz = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z), (NUM_QUBITS - 1, PauliOp::Z)])],
    };
    c.bench_function("sv_20q_pauli_zz_expectation", |b| {
        b.iter(|| {
            let backend = StatevectorBackend::new();
            backend.expectation(&circuit, &params, &zz).unwrap()
        });
    });

    // Qubit-count sweep on the Bell chain. Throughput is set in
    // statevector bytes touched per execution so Criterion reports a
    // GiB/s figure that exposes the memory-bandwidth ceiling — the
    // regime where GPU is supposed to pull ahead.
    let mut sweep = c.benchmark_group("sv_bell_chain_sweep");
    for &n in SWEEP_QUBITS {
        let circuit = bell_chain(n);
        let bytes = (1u64 << n) * std::mem::size_of::<num_complex::Complex64>() as u64;
        // Each gate touches the full statevector once (read + write).
        let touched = bytes * circuit.ops.len() as u64 * 2;
        sweep.throughput(Throughput::Bytes(touched));
        sweep.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let backend = StatevectorBackend::new();
                backend.execute(&circuit, &params, &cfg).unwrap()
            });
        });
    }
    sweep.finish();
}

criterion_group!(benches, bench_statevector);
criterion_main!(benches);
