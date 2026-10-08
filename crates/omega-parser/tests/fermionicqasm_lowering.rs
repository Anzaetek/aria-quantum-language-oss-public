// SPDX-License-Identifier: Apache-2.0
//! F4.2 lowering. The QASM2 side of the differential is written in this file.
//! Generating it from the fermionic lowering would pass when both sides share
//! a bug, which is the failure this comparison exists to catch.

use omega_backend_photonics::PhotonicsBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;
use omega_parser::lower::{lower_to_ir, lower_to_ir_with_dialect, Qasm2Dialect};
use std::path::PathBuf;

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fermionicqasm")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn bind(ir: &CircuitIR, pairs: &[(&str, f64)]) -> ParameterBinding {
    let mut params = ParameterBinding::new();
    for (name, value) in pairs {
        let id = ir
            .symbols
            .iter()
            .find(|(_, n)| n.as_str() == *name)
            .map(|(id, _)| *id)
            .unwrap_or_else(|| panic!("circuit has no symbol {name}; symbols: {:?}", ir.symbols));
        params.bind(id, *value);
    }
    params
}

fn evolve(ir: &CircuitIR, params: &ParameterBinding) -> ExecResult {
    let config = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    StatevectorBackend::new()
        .execute(ir, params, &config)
        .unwrap_or_else(|e| panic!("statevector refused: {e}"))
}

fn max_delta(a: &ExecResult, b: &ExecResult) -> f64 {
    match (a, b) {
        (ExecResult::Statevector(x), ExecResult::Statevector(y)) => {
            assert_eq!(x.len(), y.len(), "statevector widths differ");
            x.iter()
                .zip(y.iter())
                .map(|(p, q)| (p - q).norm())
                .fold(0.0_f64, f64::max)
        }
        _ => panic!("expected two statevectors, got {a:?} and {b:?}"),
    }
}

fn occupation(result: &ExecResult, wire: u32) -> f64 {
    let ExecResult::Statevector(amps) = result else {
        panic!("expected a statevector");
    };
    let bit = 1usize << wire;
    amps.iter()
        .enumerate()
        .filter(|(i, _)| i & bit != 0)
        .map(|(_, a)| a.norm_sqr())
        .sum()
}

/// (i) Each fixture against a QASM2+`rbs` program written in this test.
///
/// Fails when a mapping is wrong. It cannot pass from both readers sharing
/// a writer: the OpenQASM text below is not produced from the fermionic IR.
#[test]
fn cross_lane_differential_against_handwritten_qasm2() {
    // Hand-written. `load` is leading `x`, `num` is `u1`, `numnum` is `cp`
    // (including the non-adjacent pair), `givens` is `rbs` in the written
    // order, `tunnel(θ) m[p], m[q]` is `u1(-π/2)` on `q`, `rbs(-θ)`, `u1(π/2)`
    // on `q`. The spin register is the blocked layout: `s[2] spin` starts at
    // wire 4, so `s[0], s[1]` are wires 4, 5 and `s[2], s[3]` are wires 6, 7.
    let spinless_qasm = r#"OPENQASM 2.0;
qreg m[4];
creg c[4];
x m[0];
x m[1];
u1(0.7) m[0];
cp(0.7) m[0], m[1];
rbs(0.7) m[0], m[1];
u1(-pi/2) m[2];
rbs(-0.7) m[1], m[2];
u1(pi/2) m[2];
measure m[3] -> c[3];
"#;
    // theta = 0.3, phi = 0.4. Written as those numbers, not as a rewrite of
    // the fermionic ops.
    let all_forms_qasm = r#"OPENQASM 2.0;
qreg q[8];
creg c[4];
x q[0];
x q[1];
u1(0.3) q[0];
cp(0.5) q[0], q[3];
rbs(-0.7) q[4], q[5];
u1(-pi/2) q[7];
rbs(-0.4) q[6], q[7];
u1(pi/2) q[7];
measure q[0] -> c[0];
"#;

    let cases = [
        ("spinless.fqasm", spinless_qasm, &[][..]),
        (
            "all_forms.fqasm",
            all_forms_qasm,
            &[("theta", 0.3), ("phi", 0.4)][..],
        ),
    ];
    for (file, qasm, symbols) in cases {
        let ferm = lower_to_ir(&fixture(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(
            ferm.circuit_type,
            CircuitType::Fermionic,
            "{file} lowered as {:?}",
            ferm.circuit_type
        );
        let qasm_ir = lower_to_ir_with_dialect(qasm, Qasm2Dialect::Lenient)
            .unwrap_or_else(|e| panic!("hand-written QASM for {file}: {e}"));
        let got = evolve(&ferm, &bind(&ferm, symbols));
        let expect = evolve(&qasm_ir, &ParameterBinding::new());
        let delta = max_delta(&got, &expect);
        assert!(
            delta < 1e-12,
            "{file}: statevector |Δ| = {delta} (tolerance 1e-12)"
        );
        // A comparison of two real computational-basis states would pass with
        // every phase-carrying gate dropped on both sides. `num` / `tunnel`
        // put amplitude off the real axis; this fails if they did not run.
        let ExecResult::Statevector(amps) = &got else {
            unreachable!("evolve returns a statevector");
        };
        let imag = amps.iter().map(|a| a.im.abs()).fold(0.0_f64, f64::max);
        assert!(
            imag > 1e-6,
            "{file}: agreed statevector is real (max |Im| = {imag}); phases did not run"
        );
    }
}

/// (ii) Non-adjacent `givens` refuses with the Z-string message, because the
/// lowering calls `fermion::givens_expr`. Non-adjacent `numnum` is accepted:
/// number operators have no Z string, and a blanket adjacency check would
/// refuse a legal gate.
#[test]
fn nonadjacent_givens_refuses_with_z_string_and_numnum_is_accepted() {
    let givens = "FERMIONICQASM 1.0;\nmode m[4];\ngivens(0.7) m[0], m[2];\n";
    let err = lower_to_ir(givens).expect_err("non-adjacent givens must be refused");
    assert!(
        err.contains("Z string"),
        "refusal must be givens_expr's Z-string message, got: {err}"
    );

    let numnum = "FERMIONICQASM 1.0;\nmode m[4];\nnumnum(0.7) m[0], m[2];\n";
    let ir = lower_to_ir(numnum).unwrap_or_else(|e| panic!("non-adjacent numnum refused: {e}"));
    assert!(
        ir.ops.iter().any(|op| op.gate == GateKind::CU3),
        "numnum should lower to CU3, ops: {:?}",
        ir.ops.iter().map(|op| &op.gate).collect::<Vec<_>>()
    );
}

/// (iii) `mode m[2] spin; load m[0];` occupies wire 0 (site 0, up) and leaves
/// wire 2 empty (site 0, down). An interleaved map, or the unapplied
/// `norb - 1 - orb` reversal, moves that particle.
#[test]
fn spin_block_load_occupies_wire_0_not_wire_2() {
    let src = "FERMIONICQASM 1.0;\nmode m[2] spin;\nload m[0];\n";
    let ir = lower_to_ir(src).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(ir.num_qubits, 4, "spin doubles the declared spatial size");
    assert_eq!(ir.fermionic_registers.len(), 1);
    let reg = &ir.fermionic_registers[0];
    assert_eq!(reg.name, "m");
    assert_eq!(reg.start, 0);
    assert_eq!(reg.spatial, 2);
    assert!(reg.spin);
    let result = evolve(&ir, &ParameterBinding::new());
    let n0 = occupation(&result, 0);
    let n1 = occupation(&result, 1);
    let n2 = occupation(&result, 2);
    assert!(
        (n0 - 1.0).abs() < 1e-12,
        "<n> on wire 0 = {n0}, want 1 (site 0 up)"
    );
    // `pol`'s other submode is wire `2*s+1`. Wire 2 is site 0 down.
    assert!((n1 - 0.0).abs() < 1e-12, "<n> on wire 1 = {n1}, want 0");
    assert!(
        (n2 - 0.0).abs() < 1e-12,
        "<n> on wire 2 = {n2}, want 0 (site 0 down)"
    );
}

/// (iv) Fermionic modes are not qudits. A spin block must not show up as a
/// dimension, or the qudit refusal starts seeing these circuits.
#[test]
fn fermionic_lowering_leaves_qudit_registers_empty_and_wire_dim_2() {
    for file in ["spinless.fqasm", "all_forms.fqasm"] {
        let ir = lower_to_ir(&fixture(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert!(
            ir.qudit_registers.is_empty(),
            "{file}: qudit_registers = {:?}",
            ir.qudit_registers
        );
        assert!(ir.num_qubits > 0, "{file} allocated no wires");
        for w in 0..ir.num_qubits {
            assert_eq!(ir.wire_dim(w), 2, "{file}: wire_dim({w})");
        }
    }
}

/// (v) `load` after a gate is refused. The photonics backend refuses a
/// Fermionic circuit by name.
#[test]
fn load_after_a_gate_refuses_and_photonics_refuses_fermionic() {
    let src = "FERMIONICQASM 1.0;\nmode m[2];\nnum(0.1) m[0];\nload m[1];\n";
    let err = lower_to_ir(src).expect_err("load after a gate must be refused");
    assert!(
        err.contains("load") && err.contains("after a gate"),
        "refusal should name load-after-a-gate, got: {err}"
    );

    let ir = lower_to_ir("FERMIONICQASM 1.0;\nmode m[2] spin;\nload m[0];\n")
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(ir.circuit_type, CircuitType::Fermionic);
    let refused = PhotonicsBackend::new().execute(
        &ir,
        &ParameterBinding::new(),
        &ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        },
    );
    let err = refused.expect_err("photonics must refuse a Fermionic circuit");
    let msg = err.to_string();
    assert!(
        msg.contains("Fermionic"),
        "photonics refusal must name Fermionic, got: {msg}"
    );
}

/// Cross-spin `givens` / `tunnel` has no pinned matrix. Adjacent across the
/// block boundary (`m[1], m[2]` on `mode m[2] spin`) is the case `givens_expr`
/// would otherwise accept, so the refusal has to happen before that call.
#[test]
fn cross_spin_givens_and_tunnel_are_refused_as_unpinned() {
    for gate in ["givens", "tunnel"] {
        let src = format!("FERMIONICQASM 1.0;\nmode m[2] spin;\n{gate}(0.7) m[1], m[2];\n");
        let err = match lower_to_ir(&src) {
            Err(e) => e,
            Ok(_) => panic!("{gate} across the spin boundary lowered; it is unpinned"),
        };
        assert!(
            err.contains("unpinned"),
            "{gate} refusal must say the case is unpinned, got: {err}"
        );
    }
    // Same-spin adjacent on each block still lowers. This fails if the
    // cross-spin check is applied to every pair.
    let ok =
        "FERMIONICQASM 1.0;\nmode m[2] spin;\ngivens(0.7) m[0], m[1];\ntunnel(0.7) m[2], m[3];\n";
    lower_to_ir(ok).unwrap_or_else(|e| panic!("same-spin pair refused: {e}"));
}
