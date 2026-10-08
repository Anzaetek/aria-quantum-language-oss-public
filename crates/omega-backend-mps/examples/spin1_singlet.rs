// SPDX-License-Identifier: Apache-2.0
//! Spin-1 singlet on two qutrits, and ⟨S₁·S₂⟩ = −2.
//!
//! The circuit is `examples/qudit/spin1_singlet.ditqasm`. Level 0, 1, 2 is
//! m = +1, 0, −1, so the state is (|0⟩|2⟩ − |1⟩|1⟩ + |2⟩|0⟩)/√3.
//!
//! The same number is computed two ways. `expectation_site_operators` is one
//! real d×d matrix per wire, so S₁·S₂ is three calls:
//! ⟨Sz⊗Sz⟩ + ½⟨S⁺⊗S⁻⟩ + ½⟨S⁻⊗S⁺⟩. The other call is that contraction on the
//! dense state from quditsv. The CLI has no door for either.

use num_complex::Complex64;
use omega_backend_mps::MpsBackend;
use omega_backend_quditsv::sim::run;
use omega_core::circuit::{
    CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit, QuditRegister,
};
use omega_core::params::ParameterBinding;

fn op(kind: GateKind, wires: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: wires.iter().copied().map(Qubit).collect(),
        params: params.iter().copied().map(ParamExpr::Concrete).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// Angles are the closed forms, written to the same digits as the DITQASM file.
fn singlet() -> CircuitIR {
    let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
    circuit.qudit_registers.push(QuditRegister {
        name: "q".to_string(),
        start: 0,
        dims: vec![3, 3],
    });
    let theta = 2.0 * (1.0 / 3.0_f64.sqrt()).asin();
    circuit.ops.push(op(
        GateKind::Rxy,
        &[0],
        &[0.0, 1.0, theta, -std::f64::consts::FRAC_PI_2],
    ));
    circuit.ops.push(op(
        GateKind::Rxy,
        &[0],
        &[
            0.0,
            2.0,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
        ],
    ));
    circuit.ops.push(op(GateKind::X, &[1], &[]));
    circuit.ops.push(op(GateKind::X, &[1], &[]));
    circuit.ops.push(op(GateKind::CSum, &[0, 1], &[]));
    circuit.ops.push(op(GateKind::CSum, &[0, 1], &[]));
    circuit
}

fn re(x: f64) -> Complex64 {
    Complex64::new(x, 0.0)
}

/// Row-major. Level 0, 1, 2 is m = +1, 0, −1.
fn spin_matrices() -> (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
    let s2 = 2.0_f64.sqrt();
    let sz = vec![
        re(1.0),
        re(0.0),
        re(0.0),
        re(0.0),
        re(0.0),
        re(0.0),
        re(0.0),
        re(0.0),
        re(-1.0),
    ];
    // S⁺ |m⟩ raises m. S⁺|−1⟩ = √2|0⟩, S⁺|0⟩ = √2|+1⟩.
    let sp = vec![
        re(0.0),
        re(s2),
        re(0.0),
        re(0.0),
        re(0.0),
        re(s2),
        re(0.0),
        re(0.0),
        re(0.0),
    ];
    let sm = vec![
        re(0.0),
        re(0.0),
        re(0.0),
        re(s2),
        re(0.0),
        re(0.0),
        re(0.0),
        re(s2),
        re(0.0),
    ];
    (sz, sp, sm)
}

fn dense_product(amp: &[Complex64], a: &[Complex64], b: &[Complex64]) -> Complex64 {
    let mut acc = Complex64::new(0.0, 0.0);
    for j in 0..3 {
        for i in 0..3 {
            let bra = amp[i + 3 * j].conj();
            for l in 0..3 {
                for k in 0..3 {
                    acc += bra * a[i * 3 + k] * b[j * 3 + l] * amp[k + 3 * l];
                }
            }
        }
    }
    acc
}

fn dot(
    sz: &[Complex64],
    sp: &[Complex64],
    sm: &[Complex64],
    term: impl Fn(&[Complex64], &[Complex64]) -> f64,
) -> f64 {
    term(sz, sz) + 0.5 * term(sp, sm) + 0.5 * term(sm, sp)
}

fn main() {
    let circuit = singlet();
    let params = ParameterBinding::new();
    let (sz, sp, sm) = spin_matrices();

    let mps = MpsBackend::new(64);
    let from_mps = dot(&sz, &sp, &sm, |a, b| {
        mps.expectation_site_operators(&circuit, &params, &[a, b])
            .expect("site operators")
    });

    let st = run(&circuit, &params).expect("dense singlet");
    let from_dense = dense_product(&st.amp, &sz, &sz)
        + 0.5 * dense_product(&st.amp, &sp, &sm)
        + 0.5 * dense_product(&st.amp, &sm, &sp);

    println!("mps site operators: {from_mps:.12}");
    println!(
        "dense contraction:  {:.12}{:+.3e}i",
        from_dense.re, from_dense.im
    );
    assert!(
        (from_mps + 2.0).abs() < 1e-10,
        "mps ⟨S1·S2⟩ = {from_mps}, expected -2"
    );
    assert!(
        (from_dense.re + 2.0).abs() < 1e-10 && from_dense.im.abs() < 1e-10,
        "dense ⟨S1·S2⟩ = {from_dense}, expected -2"
    );
}
