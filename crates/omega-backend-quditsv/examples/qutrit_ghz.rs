// SPDX-License-Identifier: Apache-2.0
//! Qutrit GHZ through the quditsv API.
//!
//! The same circuit as `examples/qudit/qutrit_ghz.ditqasm`: Fourier on wire
//! 0, then `csum` copies that digit onto wires 1 and 2. Wire 0 is the least
//! significant digit, so `|kkk⟩` sits at index `k + 3k + 9k` = `0, 13, 26`.
//! `F_3|0⟩` is real and positive, and `csum` does not add a phase, so each
//! of those amplitudes is `1/√3`.

use num_complex::Complex64;
use omega_backend_quditsv::sim::run;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit, QuditRegister};
use omega_core::params::ParameterBinding;

fn gate(kind: GateKind, wires: &[u32]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: wires.iter().copied().map(Qubit).collect(),
        params: Default::default(),
        classical_bit: None,
        condition: None,
    }
}

fn ghz() -> CircuitIR {
    let mut circuit = CircuitIR::new(3, CircuitType::GateBased);
    circuit.qudit_registers.push(QuditRegister {
        name: "q".to_string(),
        start: 0,
        dims: vec![3, 3, 3],
    });
    circuit.ops.push(gate(GateKind::H, &[0]));
    circuit.ops.push(gate(GateKind::CSum, &[0, 1]));
    circuit.ops.push(gate(GateKind::CSum, &[0, 2]));
    circuit
}

fn main() {
    let st = run(&ghz(), &ParameterBinding::new()).expect("qutrit GHZ");
    let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
    let peaks = [0usize, 13, 26];
    for (i, a) in st.amp.iter().enumerate() {
        let expect = if peaks.contains(&i) { inv_sqrt3 } else { 0.0 };
        let err = (a.re - expect).abs() + a.im.abs();
        assert!(err < 1e-12, "index {i}: got {a}, expected {expect} (real)");
        if peaks.contains(&i) {
            println!("index {i}: {a:.12}");
        }
    }
    let mass: f64 = st.amp.iter().map(Complex64::norm_sqr).sum();
    assert!((mass - 1.0).abs() < 1e-12, "norm {mass}");
}
