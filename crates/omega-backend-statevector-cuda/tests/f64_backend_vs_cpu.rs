// SPDX-License-Identifier: Apache-2.0
//! The f64 CUDA arm (`CudaStatevectorF64Backend`, `omega-run --precision f64`)
//! against the f64 CPU statevector, amplitude by amplitude, at **1e-12**.
//!
//! The bar is the point. The f32 arm is checked at 1e-6 (`gpu_parity.rs`)
//! because f32 cannot do better than ~5e-7; a double arm that only met 1e-6
//! would be indistinguishable from an f32 arm mislabelled. On this card the
//! cuStateVec yardstick reads 1−F = 7.3e-15 at double, so a failure here is a
//! wiring error, not a tolerance that was too tight.
//!
//! The circuits are the ones the f32 arm is already checked on — the 4-qubit
//! parity circuit of `omega-cli/tests/gpu_parity.rs`, the Bell pair, the
//! bit-order circuit of `tools/emu_compare/test_bitorder.py` at n = 3 and 16,
//! and E5's two workload families from `tools/emu_compare/qasm` — plus one
//! circuit that applies every gate kind this arm accepts, on both qubit orders,
//! because a gate-to-matrix table is where a swapped `(qa, qb)` or a dropped
//! phase hides. The comparison is NOT phase-aligned: both sides build every
//! gate from the CPU's `gates` module, so even the global phase must agree.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use num_complex::Complex64;
use omega_backend_statevector::StatevectorBackend;
use omega_backend_statevector_cuda::f64_backend::CudaStatevectorF64Backend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::error::OmegaError;
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable, PauliOp};
use omega_core::params::ParameterBinding;

const TOL: f64 = 1e-12;

fn gpu() -> Option<CudaStatevectorF64Backend> {
    match CudaStatevectorF64Backend::new() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("SKIP: no CUDA device for the f64 arm ({e})");
            None
        }
    }
}

fn state(b: &dyn Backend, c: &CircuitIR) -> Vec<Complex64> {
    let cfg = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    match b.execute(c, &ParameterBinding::default(), &cfg).unwrap() {
        ExecResult::Statevector(sv) => sv,
        other => panic!("shots: None must give a statevector, got {other:?}"),
    }
}

/// Worst per-amplitude modulus difference, no phase alignment.
fn worst(a: &[Complex64], b: &[Complex64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dimensions differ");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

fn agree(label: &str, gpu: &CudaStatevectorF64Backend, c: &CircuitIR) -> f64 {
    let d = worst(&state(gpu, c), &state(&StatevectorBackend::new(), c));
    assert!(
        d <= TOL,
        "{label}: f64 GPU vs CPU worst |Δamp| = {d:e} > {TOL:e}"
    );
    d
}

fn qasm(src: &str) -> CircuitIR {
    omega_parser::lower_to_ir(src).expect("parse")
}

fn workload(file: &str) -> CircuitIR {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/emu_compare/qasm")
        .join(file);
    qasm(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

/// `omega-cli/tests/gpu_parity.rs`'s PARITY_QASM, verbatim.
const PARITY_QASM: &str = r#"
OPENQASM 2.0;
qreg q[4];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
ry(0.31) q[0];
rz(0.42) q[1];
ry(0.55) q[2];
cx q[0], q[2];
rz(-0.21) q[0];
ry(0.77) q[3];
"#;

/// `tools/emu_compare/test_bitorder.py`'s circuit. Not reversal-symmetric, so
/// a qubit-order error moves amplitude to a different index.
fn bitorder(n: u32) -> CircuitIR {
    let last = n - 1;
    qasm(&format!(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\n\
         x q[0];\nh q[1];\ncx q[1],q[{last}];\ncx q[1],q[0];\nu1(0.5) q[{last}];\n"
    ))
}

fn op(gate: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// Every gate kind the arm accepts, each two-qubit gate on BOTH orders of a
/// non-adjacent pair, over a state with no symmetry to hide behind.
fn every_gate() -> CircuitIR {
    use GateKind::*;
    let mut c = CircuitIR::new(5, CircuitType::GateBased);
    let mut seed = [
        op(H, &[0], &[]),
        op(Ry, &[1], &[0.7]),
        op(Rx, &[2], &[1.1]),
        op(H, &[3], &[]),
        op(Ry, &[4], &[-0.4]),
    ]
    .to_vec();
    c.ops.append(&mut seed);
    let one: [(GateKind, &[f64]); 16] = [
        (H, &[]),
        (X, &[]),
        (Y, &[]),
        (Z, &[]),
        (S, &[]),
        (Sdg, &[]),
        (Sx, &[]),
        (Sxdg, &[]),
        (T, &[]),
        (Tdg, &[]),
        (Rx, &[0.37]),
        (Ry, &[1.21]),
        (Rz, &[-0.83]),
        (U1, &[0.61]),
        (U2, &[0.29, -1.13]),
        (U3, &[0.44, 1.7, -0.52]),
    ];
    for (i, (g, p)) in one.iter().enumerate() {
        c.ops.push(op(g.clone(), &[(i % 5) as u32], p));
    }
    let two: [(GateKind, &[f64]); 7] = [
        (CX, &[]),
        (CY, &[]),
        (CZ, &[]),
        (Swap, &[]),
        (CRz, &[0.93]),
        (CU3, &[0.31, -0.77, 1.29]),
        (Rbs, &[0.58]),
    ];
    for (g, p) in two.iter() {
        c.ops.push(op(g.clone(), &[0, 3], p));
        c.ops.push(op(Ry, &[0], &[0.2]));
        c.ops.push(op(g.clone(), &[3, 0], p));
        c.ops.push(op(Rx, &[3], &[0.3]));
    }
    c
}

#[test]
fn the_f64_arm_matches_the_cpu_on_the_f32_arms_circuits() {
    let Some(g) = gpu() else { return };
    agree("parity (gpu_parity.rs)", &g, &qasm(PARITY_QASM));
    agree(
        "bell",
        &g,
        &qasm("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n"),
    );
    agree("bitorder n=3", &g, &bitorder(3));
    agree("bitorder n=16", &g, &bitorder(16));
}

#[test]
fn the_f64_arm_matches_the_cpu_on_e5s_workload_families() {
    let Some(g) = gpu() else { return };
    // The 20-qubit members of E5's two families: the same generator and gate
    // set as the 24/26/28-qubit rows, at a size the CPU reference runs in
    // seconds.
    for f in ["random1_20q_d20_s0.qasm", "hea_20q_d24.qasm"] {
        let d = agree(f, &g, &workload(f));
        eprintln!("{f}: worst |Δamp| = {d:e}");
    }
}

#[test]
fn every_accepted_gate_matches_the_cpu_on_both_qubit_orders() {
    let Some(g) = gpu() else { return };
    let d = agree("every gate", &g, &every_gate());
    eprintln!("every gate: worst |Δamp| = {d:e}");
}

/// A matched-precision claim needs the arm to be measurably better than f32:
/// on the 20-qubit E5 workload the f32 arm must MISS this file's bar and the
/// f64 arm must meet it. If both met it the test above would prove nothing
/// about precision.
#[test]
fn the_bar_separates_the_two_precisions() {
    let Some(g) = gpu() else { return };
    let Ok(f32_arm) = omega_backend_statevector_cuda::CudaStatevectorBackend::new() else {
        return;
    };
    let c = workload("random1_20q_d20_s0.qasm");
    let cpu = state(&StatevectorBackend::new(), &c);
    let d64 = worst(&state(&g, &c), &cpu);
    let d32 = worst(&state(&f32_arm, &c), &cpu);
    eprintln!("random1_20q: f64 {d64:e}, f32 {d32:e}");
    assert!(d64 <= TOL, "f64 {d64:e}");
    assert!(
        d32 > TOL,
        "the f32 arm met the f64 bar ({d32:e}); the bar is not a precision test"
    );
}

#[test]
fn expectation_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    let c = every_gate();
    let obs = Observable {
        terms: vec![
            (0.5, vec![(0, PauliOp::Z)]),
            (-1.25, vec![(0, PauliOp::X), (3, PauliOp::Y)]),
            (
                0.75,
                vec![(1, PauliOp::Y), (2, PauliOp::Z), (4, PauliOp::X)],
            ),
        ],
    };
    let p = ParameterBinding::default();
    let ours = g.expectation(&c, &p, &obs).unwrap();
    let cpu = StatevectorBackend::new().expectation(&c, &p, &obs).unwrap();
    assert!((ours - cpu).abs() <= TOL, "⟨O⟩ f64 GPU {ours} vs CPU {cpu}");
}

#[test]
fn shots_sample_the_f64_state() {
    let Some(g) = gpu() else { return };
    // |101⟩ deterministically: every shot must land on key 0b101.
    let c = qasm("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\nx q[0];\nx q[2];\n");
    let cfg = ExecConfig {
        shots: Some(200),
        seed: Some(7),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    match g.execute(&c, &ParameterBinding::default(), &cfg).unwrap() {
        ExecResult::Counts(counts) => {
            assert_eq!(counts.values().sum::<u32>(), 200);
            assert_eq!(counts.len(), 1, "{counts:?}");
            let (k, _) = counts.iter().next().unwrap();
            assert_eq!(k.width(), 3);
            assert_eq!((k.bit(0), k.bit(1), k.bit(2)), (1, 0, 1), "{k:?}");
        }
        other => panic!("shots must give counts, got {other:?}"),
    }
}

/// Everything outside the arm's scope is refused by name, never answered.
#[test]
fn what_the_f64_path_cannot_express_is_refused() {
    let Some(g) = gpu() else { return };
    let cfg = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let refused = |c: &CircuitIR, cfg: &ExecConfig| -> String {
        match g.execute(c, &ParameterBinding::default(), cfg) {
            Err(OmegaError::Unsupported(m)) => m,
            other => panic!("expected a refusal, got {other:?}"),
        }
    };
    let mut ccx = CircuitIR::new(3, CircuitType::GateBased);
    ccx.ops.push(op(GateKind::CCX, &[0, 1, 2], &[]));
    assert!(refused(&ccx, &cfg).contains("three qubits"));

    let mut reset = CircuitIR::new(1, CircuitType::GateBased);
    reset.ops.push(op(GateKind::Reset, &[0], &[]));
    assert!(refused(&reset, &cfg).contains("channel"));

    let mut cond = CircuitIR::new(1, CircuitType::GateBased);
    cond.num_classical_bits = 1;
    let mut x = op(GateKind::X, &[0], &[]);
    x.condition = Some((0, 1, 1));
    cond.ops.push(x);
    assert!(refused(&cond, &cfg).contains("conditioned"));

    let mut meas = CircuitIR::new(1, CircuitType::GateBased);
    meas.num_classical_bits = 1;
    let mut m = op(GateKind::Measure, &[0], &[]);
    m.classical_bit = Some(0);
    meas.ops.push(op(GateKind::H, &[0], &[]));
    meas.ops.push(m);
    let collapse = ExecConfig {
        mid_circuit_mode: MidCircuitMode::Collapse,
        ..cfg.clone()
    };
    assert!(refused(&meas, &collapse).contains("collapse"));
}
