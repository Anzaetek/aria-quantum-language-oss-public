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

/// Width for the per-kernel measurements below.
///
/// **The existing `SWEEP_QUBITS` cannot measure memory traffic on this class of
/// device.** At 20 qubits the f32 state is `2^20 × 8 B = 8 MiB`, and a GB10's L2
/// is 24 MiB — the whole working set is L2-resident, so the sweep is dominated
/// by kernel-launch latency and would rank "fewer launches" above every
/// bandwidth effect. 26 qubits is `2^26 × 8 B = 512 MiB`, ~21× that L2, which
/// puts the kernels where they actually live in a real run.
///
/// Kept separate from `SWEEP_QUBITS` rather than replacing it, because the
/// sweep's job is apples-to-apples comparison against the CPU and Metal result
/// files, and changing its widths would invalidate that.
const MEM_BOUND_QUBITS: u32 = 26;

/// Gates per timed iteration. One 26q gate moves ~1 GiB, so a single-gate
/// iteration would be measuring the sync as much as the kernel; batching
/// amortises the one readback that forces the stream to drain.
const GATES_PER_ITER: u32 = 20;

/// Per-kernel cost at a width that does not fit in cache.
///
/// This is the P1 baseline instrument. `apply_cx`, `apply_cz` and `apply_swap`
/// all currently build a dense `[Complex64; 16]` and dispatch the generic
/// `apply_2q` kernel (`lib.rs:528-584`), which per quad does 4 loads + 4 stores
/// and 16 complex multiplies + 12 complex adds — for gates that are a
/// permutation (CX, SWAP) or a diagonal (CZ) and need no arithmetic at all.
///
/// `apply_h` is the reference point: a genuinely non-trivial 1q gate, 2 loads +
/// 2 stores per pair. A permutation kernel for CX should land at or below it.
///
/// Reported as throughput over the bytes a *correct* implementation must move,
/// so the number is directly comparable across gate kinds.
fn bench_2q_kernel_cost(c: &mut Criterion) {
    let backend = CudaStatevectorBackend::new().expect("CUDA device");
    let n = MEM_BOUND_QUBITS;

    // Device amplitudes are f32 complex = 8 B, NOT the host `Complex64`'s 16 B.
    const BYTES_PER_AMP: u64 = 8;
    let state_bytes = (1u64 << n) * BYTES_PER_AMP;

    let mut group = c.benchmark_group("cuda_kernel_cost_26q");
    // 26q gates run ~ms each; criterion's default 100 samples would take
    // minutes per entry for no extra resolution.
    group.sample_size(10);
    // Denominator is what a DENSE 2q implementation must move: 4 loads + 4
    // stores per quad = 2·dim amplitudes. Held constant across every entry in
    // the group so the entries are directly comparable.
    //
    // READ THE `thrpt` COLUMN AS "dense-equivalent work rate", NOT as achieved
    // memory bandwidth. A kernel that moves fewer bytes than the dense form
    // scores above the device's real bandwidth — CZ reports ~541 GiB/s on a
    // GB10 whose LPDDR5X tops out near 254 GiB/s, because it only touches one
    // amplitude in four. The `time` column is the honest number; divide by the
    // per-kernel traffic below for actual bandwidth:
    //
    //   dense 2q  2·dim amps    CX/SWAP  1·dim amps    CZ  0.5·dim amps
    group.throughput(Throughput::Bytes(state_bytes * 2 * GATES_PER_ITER as u64));

    // Walk qubit pairs so no single (qa, qb) stride is measured in isolation —
    // the addressing cost of `apply_2q` depends on the gap between the two.
    //
    // READ THE RESULTS WITH THIS IN MIND. The byte saving is NOT a property of
    // the gate; it is a property of `min(qa, qb)`, because DRAM/L2 granularity
    // is a 32 B sector = 4 f32 amplitudes:
    //
    //   min(qa,qb) = 0   CZ touches i = 3 mod 4  -> every sector -> NO saving
    //   min(qa,qb) = 1   half the sectors        -> 2x
    //   min(qa,qb) >= 2  one sector in four      -> 4x
    //
    // CX/SWAP (2 slots of 4) reach 2x only at min >= 2; at min in {0,1} they
    // save flops but not bytes. This sweep runs qa = 0..19 at n = 26, so 18 of
    // 20 gates land in the favourable regime and the headline ratios are an
    // upper bound, not a circuit-average. A real CX ladder or QFT uses (0,1)
    // and (1,2) as often as any other pair. `..._low_index` below measures the
    // unfavourable end deliberately.
    let pair = |k: u32| (k % (n - 1), (k % (n - 1)) + 1);
    let low_pair = |k: u32| (k % 2, (k % 2) + 1);

    // A deterministic non-trivial state. Benching on `allocate`'s |0...0> would
    // be measuring these kernels against a buffer of zeros with a single 1.0 —
    // and worse, CX/SWAP/CZ/CRz are all FIXED POINTS of that state (they never
    // touch slot 0), so a kernel that did nothing at all would score the same.
    let seed_state: Vec<num_complex::Complex64> = (0..(1usize << n))
        .map(|i| {
            let a = (i % 4093) as f64 / 4093.0 - 0.5;
            let b = (i % 3079) as f64 / 3079.0 - 0.5;
            num_complex::Complex64::new(a, b)
        })
        .collect();
    let fresh = |b: &CudaStatevectorBackend| {
        let mut st = b.allocate(n).expect("allocate");
        st.write_state(&seed_state).expect("seed");
        st
    };

    // The path CX/SWAP/CZ took before P1-a, still reachable because `apply_2q`
    // is still the generic 2q entry point. Benched IN THE SAME RUN as the
    // specialised kernels rather than against a number from a previous run, so
    // the comparison cannot drift on clocks, thermal state or driver version.
    #[rustfmt::skip]
    let cx_dense: [num_complex::Complex64; 16] = {
        let z = num_complex::Complex64::new(0.0, 0.0);
        let o = num_complex::Complex64::new(1.0, 0.0);
        [
            o, z, z, z,
            z, z, z, o,
            z, z, o, z,
            z, o, z, z,
        ]
    };
    group.bench_function("cx_via_dense_2q_before", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                let (a, t) = pair(k);
                st.apply_2q(a, t, &cx_dense).unwrap();
            }
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    group.bench_function("cx", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                let (a, t) = pair(k);
                st.apply_cx(a, t).unwrap();
            }
            // Scalar readback — forces the stream to drain without pulling
            // 512 MiB back to the host the way `read_state` would.
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    group.bench_function("cz", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                let (a, t) = pair(k);
                st.apply_cz(a, t).unwrap();
            }
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    group.bench_function("swap", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                let (a, t) = pair(k);
                st.apply_swap(a, t).unwrap();
            }
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    // CRz: the two-slot arm of the phase kernel. Its dense form is benched
    // alongside for the same reason as CX's.
    #[rustfmt::skip]
    let crz_dense: [num_complex::Complex64; 16] = {
        let z = num_complex::Complex64::new(0.0, 0.0);
        let o = num_complex::Complex64::new(1.0, 0.0);
        let phn = num_complex::Complex64::from_polar(1.0, -0.37 / 2.0);
        let php = num_complex::Complex64::from_polar(1.0, 0.37 / 2.0);
        [
            o, z,   z, z,
            z, phn, z, z,
            z, z,   o, z,
            z, z,   z, php,
        ]
    };
    group.bench_function("crz_via_dense_2q_before", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                let (a, t) = pair(k);
                st.apply_2q(a, t, &crz_dense).unwrap();
            }
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    group.bench_function("crz", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                let (a, t) = pair(k);
                st.apply_crz(a, t, 0.37).unwrap();
            }
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    // The unfavourable end: qa in {0,1}, where sector granularity means the
    // specialised kernels save flops but few or no bytes. Compare against
    // `cx`/`cz` above to see how much of the headline ratio is qubit-index
    // dependent rather than gate dependent.
    group.bench_function("cx_low_index", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                let (a, t) = low_pair(k);
                st.apply_cx(a, t).unwrap();
            }
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    group.bench_function("cz_low_index", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                let (a, t) = low_pair(k);
                st.apply_cz(a, t).unwrap();
            }
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    #[rustfmt::skip]
    let cy_dense: [num_complex::Complex64; 16] = {
        let z = num_complex::Complex64::new(0.0, 0.0);
        let o = num_complex::Complex64::new(1.0, 0.0);
        let pi = num_complex::Complex64::new(0.0, 1.0);
        let mi = num_complex::Complex64::new(0.0, -1.0);
        [
            o, z,  z, z,
            z, z,  z, mi,
            z, z,  o, z,
            z, pi, z, z,
        ]
    };
    group.bench_function("cy_via_dense_2q_before", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                let (a, t) = pair(k);
                st.apply_2q(a, t, &cy_dense).unwrap();
            }
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    group.bench_function("cy", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                let (a, t) = pair(k);
                st.apply_cy(a, t).unwrap();
            }
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    // Like-for-like dense references AT LOW INDEX. Comparing the low-index
    // arms against the spread `cx_via_dense_2q_before` above is invalid: the
    // dense kernel is itself ~19% slower at low indices, so that comparison
    // makes the specialised kernels look worse than they are and nearly caused
    // CZ to be gated behind a fallback that would have slowed it by 12%.
    group.bench_function("cx_low_index_dense", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                let (a, t) = low_pair(k);
                st.apply_2q(a, t, &cx_dense).unwrap();
            }
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    #[rustfmt::skip]
    let cz_dense: [num_complex::Complex64; 16] = {
        let z = num_complex::Complex64::new(0.0, 0.0);
        let o = num_complex::Complex64::new(1.0, 0.0);
        let m = num_complex::Complex64::new(-1.0, 0.0);
        [
            o, z, z, z,
            z, o, z, z,
            z, z, o, z,
            z, z, z, m,
        ]
    };
    group.bench_function("cz_low_index_dense", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                let (a, t) = low_pair(k);
                st.apply_2q(a, t, &cz_dense).unwrap();
            }
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    // CCX: exact subspace permutation vs the 15-gate decomposition. Both run
    // through `CudaState` directly so the comparison is the gate, not the
    // dispatch around it.
    group.bench_function("ccx_decompose", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                let a = k % (n - 2);
                st.apply_ccx(a, a + 1, a + 2).unwrap();
            }
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    {
        let exact_backend = CudaStatevectorBackend::new()
            .expect("CUDA device")
            .with_multi_control(omega_core::executor::MultiControlMode::Exact);
        group.bench_function("ccx_exact", |b| {
            let mut st = exact_backend.allocate(n).expect("allocate");
            st.write_state(&seed_state).expect("seed");
            b.iter(|| {
                for k in 0..GATES_PER_ITER {
                    let a = k % (n - 2);
                    st.apply_ccx(a, a + 1, a + 2).unwrap();
                }
                st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                    .unwrap()
            });
        });
    }

    // Reference: a real 1q gate, and a real diagonal 1q gate. CX/CZ should not
    // cost more than these once they stop going through the dense path.
    group.bench_function("h_1q_reference", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                st.apply_h(k % n).unwrap();
            }
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    group.bench_function("rz_diagonal_reference", |b| {
        let mut st = fresh(&backend);
        b.iter(|| {
            for k in 0..GATES_PER_ITER {
                st.apply_rz(k % n, 0.37).unwrap();
            }
            st.pauli_expectation(0, 1, num_complex::Complex64::new(1.0, 0.0))
                .unwrap()
        });
    });

    group.finish();
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

criterion_group!(benches, bench_cuda, bench_2q_kernel_cost);
criterion_main!(benches);
