// SPDX-License-Identifier: Apache-2.0
//! **`CircuitIR::refuse_qudits` — the one sentence every engine door
//! speaks, and the accessors under it.** (PLAN-QUDIT.md Q1, IR half.)
//!
//! The backends' own tests reach this through their `execute`; the CLI
//! matrix reaches it through the binary. This file pins the helper itself:
//! that a circuit built any way but through the parser is a qubit circuit,
//! that the refusal is a typed `Unsupported` (not `InvalidCircuit`, not a
//! panic — the N-way lane maps `Unsupported` to "cannot express" and
//! everything else to a red cell), and that the plugin ABI check refuses
//! host-side with the same words.

use omega_core::circuit::{CircuitIR, CircuitType, QuditRegister};
use omega_core::error::OmegaError;
use omega_core::plugin::circuit_ffi_limitation;

fn qutrits(n: u32) -> CircuitIR {
    let mut ir = CircuitIR::new(n, CircuitType::GateBased);
    ir.qudit_registers.push(QuditRegister {
        name: "q".into(),
        start: 0,
        dims: vec![3; n as usize],
    });
    ir
}

#[test]
fn every_constructor_builds_a_qubit_circuit() {
    let ir = CircuitIR::new(5, CircuitType::GateBased);
    assert!(ir.qudit_registers.is_empty());
    assert_eq!(ir.wire_dims(), vec![2; 5]);
    assert!(ir.first_qudit().is_none());
    assert!(ir.refuse_qudits("any engine").is_ok());
    let ph = CircuitIR::new(3, CircuitType::Photonic);
    assert!(ph.refuse_qudits("photonics").is_ok());
}

#[test]
fn the_refusal_is_a_typed_unsupported_naming_engine_register_wire_and_dimension() {
    let err = qutrits(2).refuse_qudits("statevector").unwrap_err();
    let OmegaError::Unsupported(msg) = &err else {
        panic!("must be Unsupported, got {err:?}");
    };
    for needle in [
        "statevector:",
        "register 'q'",
        "dimension 3",
        "wire 0",
        "qubit-only",
        "PLAN-QUDIT",
    ] {
        assert!(msg.contains(needle), "{needle:?} missing from:\n{msg}");
    }
}

#[test]
fn a_mixed_register_names_the_first_non_qubit_wire() {
    let mut ir = CircuitIR::new(4, CircuitType::GateBased);
    ir.qudit_registers.push(QuditRegister {
        name: "a".into(),
        start: 0,
        dims: vec![2, 2],
    });
    ir.qudit_registers.push(QuditRegister {
        name: "b".into(),
        start: 2,
        dims: vec![2, 5],
    });
    assert_eq!(ir.wire_dims(), vec![2, 2, 2, 5]);
    let (reg, wire, d) = ir.first_qudit().unwrap();
    assert_eq!((reg.name.as_str(), wire, d), ("b", 3, 5));
    let msg = ir.refuse_qudits("mps").unwrap_err().to_string();
    assert!(
        msg.contains("register 'b'") && msg.contains("wire 3") && msg.contains("dimension 5"),
        "{msg}"
    );
    assert!(!msg.contains("register 'a'"), "{msg}");
}

#[test]
fn an_explicit_all_two_group_is_not_refused() {
    let mut ir = CircuitIR::new(2, CircuitType::GateBased);
    ir.qudit_registers.push(QuditRegister {
        name: "q".into(),
        start: 0,
        dims: vec![2, 2],
    });
    assert!(ir.refuse_qudits("pauli").is_ok());
    assert!(ir.first_qudit().is_none());
}

#[test]
fn the_plugin_abi_refuses_host_side_with_the_same_sentence() {
    let why = circuit_ffi_limitation(&qutrits(1)).expect("must be refused");
    assert!(
        why.contains("plugin ABI:") && why.contains("dimension 3"),
        "{why}"
    );
    assert!(circuit_ffi_limitation(&CircuitIR::new(1, CircuitType::GateBased)).is_none());
}

/// `clone`, and the `ops`-only manipulations everything in the workspace
/// does, carry the registers along: a helper that rebuilt a `CircuitIR`
/// from `num_qubits` and `ops` alone would drop them and the circuit would
/// silently become qubits again. That is the one shape of regression Q1
/// most needs to catch, so it is named here.
#[test]
fn cloning_keeps_the_registers() {
    let ir = qutrits(3);
    let c = ir.clone();
    assert_eq!(c.qudit_registers, ir.qudit_registers);
    assert!(c.refuse_qudits("x").is_err());
}
