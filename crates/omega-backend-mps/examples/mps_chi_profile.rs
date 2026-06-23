// One-off investigation harness: sweep the brickwall benchmark across
// {num_qubits, depth, χ} to see when MPS wallclock is actually bottlenecked
// on the SVD path vs other operations. Was the empirical basis for
// deferring the MPS-SVD-on-Metal item — see TODO.md "MPS SVD on Metal
// — DEFERRED 2026-05-12".

use omega_backend_mps::MpsBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, MidCircuitMode};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;
use std::time::Instant;

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

fn main() {
    let params = ParameterBinding::new();
    let cfg = ExecConfig {
        shots: None,
        seed: Some(0),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    for &(nq, depth) in &[(14u32, 4usize), (14, 12), (14, 24), (20, 8), (20, 16)] {
        println!("\n=== {}q × depth {} ===", nq, depth);
        let circuit = entangling_circuit(nq, depth);

        for _ in 0..2 {
            let backend = MpsBackend::new(128);
            backend.execute(&circuit, &params, &cfg).unwrap();
        }

        for &chi in &[8usize, 32, 128, 256] {
            let n_iters = if depth > 12 || nq > 14 { 5 } else { 12 };
            let mut times = Vec::new();
            for _ in 0..n_iters {
                let t = Instant::now();
                let backend = MpsBackend::new(chi);
                backend.execute(&circuit, &params, &cfg).unwrap();
                times.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = times[times.len() / 2];
            let min = times[0];
            println!(
                "  chi={:>4} : min {:>9.2} ms  median {:>9.2} ms",
                chi, min, median
            );
        }
    }
}
