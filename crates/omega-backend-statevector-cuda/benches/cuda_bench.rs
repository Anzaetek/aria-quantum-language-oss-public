//! CUDA statevector backend benchmarks — port of
//! `omega-backend-statevector-metal/benches/metal_bench.rs`.
//!
//! Same circuit set + sweep so the CUDA numbers compare apples-to-
//! apples with both the CPU baseline (`benches/results/cpu-*.json`)
//! and the Metal numbers (`metal-aarch64-darwin.json`).
//!
//! Reproducer:
//!   cargo bench -p omega-backend-statevector-cuda \
//!       --bench cuda_bench --features cuda
//! Output lands in `target/criterion/...` by default; commit the
//! distilled headline numbers to `benches/results/cuda-x86_64-linux.json`.

#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use smallvec::smallvec;

use omega_backend_statevector_cuda::CudaStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, MidCircuitMode, Observable, PauliOp};
use omega_core::params::ParameterBinding;

const NUM_QUBITS: u32 = 20;
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

fn bench_cuda(c: &mut Criterion) {
    let cfg = ExecConfig {
        shots: None,
        seed: Some(0),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let params = ParameterBinding::new();

    // One device handle across all benches — opening the CUDA context
    // and NVRTC-compiling the kernel library is a one-time cost that
    // shouldn't be amortised into per-iter measurements.
    let backend = CudaStatevectorBackend::new().expect("CUDA device");

    let circuit = bell_chain(NUM_QUBITS);
    c.bench_function("cuda_20q_bell_chain_exec", |b| {
        b.iter(|| backend.execute(&circuit, &params, &cfg).unwrap());
    });

    let qaoa = qaoa_layer(NUM_QUBITS);
    c.bench_function("cuda_20q_qaoa_layer_exec", |b| {
        b.iter(|| backend.execute(&qaoa, &params, &cfg).unwrap());
    });

    let zz = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z), (NUM_QUBITS - 1, PauliOp::Z)])],
    };
    c.bench_function("cuda_20q_pauli_zz_expectation", |b| {
        b.iter(|| backend.expectation(&circuit, &params, &zz).unwrap());
    });

    let multi = Observable {
        terms: (0..32u32)
            .map(|k| {
                let q0 = k % NUM_QUBITS;
                let q1 = (k * 7 + 3) % NUM_QUBITS;
                let p0 = match k % 3 {
                    0 => PauliOp::Z,
                    1 => PauliOp::X,
                    _ => PauliOp::Y,
                };
                let p1 = match k % 4 {
                    0 => PauliOp::Z,
                    1 => PauliOp::X,
                    2 => PauliOp::Y,
                    _ => PauliOp::I,
                };
                (
                    1.0 / (1.0 + k as f64),
                    if q0 == q1 {
                        vec![(q0, p0)]
                    } else {
                        vec![(q0, p0), (q1, p1)]
                    },
                )
            })
            .collect(),
    };
    c.bench_function("cuda_20q_pauli_multi_term_expectation", |b| {
        b.iter(|| backend.expectation(&circuit, &params, &multi).unwrap());
    });

    let mut sweep = c.benchmark_group("cuda_bell_chain_sweep");
    for &n in SWEEP_QUBITS {
        let circuit = bell_chain(n);
        let bytes = (1u64 << n) * std::mem::size_of::<num_complex::Complex64>() as u64;
        let touched = bytes * circuit.ops.len() as u64 * 2;
        sweep.throughput(Throughput::Bytes(touched));
        sweep.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| backend.execute(&circuit, &params, &cfg).unwrap());
        });
    }
    sweep.finish();
}

criterion_group!(benches, bench_cuda);
criterion_main!(benches);
