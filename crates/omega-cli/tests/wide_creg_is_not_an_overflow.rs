// SPDX-License-Identifier: Apache-2.0
//! **A classical register wider than 64 bits is legal as long as the bits in
//! USE fit.**
//!
//! `creg c[70]; measure q[0] -> c[0];` produces a one-bit outcome. Nothing
//! about it is unrepresentable. But two places conflated "the register is
//! declared wide" with "the outcome is wide":
//!
//! * `creg_to_u64` asserted on `classical_bits.len()`, which is the DECLARED
//!   size — so it panicked in debug builds on a circuit that
//!   `counts_outcome_width` had just measured at 1 bit and admitted. The gate
//!   and the encoder disagreed about the same run.
//! * `condition_satisfied` computed `(bit as u64) << i` for every `i` in the
//!   register. At `i = 64` that panics in debug and is masked to `<< 0` in
//!   release, so `if (c == 1)` on a wide register either crashed or tested a
//!   value with bit 64 folded into bit 0.
//!
//! **These tests must run in a debug build to mean anything** — both defects
//! are `debug_assert` / debug-only arithmetic panics, and `cargo test --release`
//! passes on the broken code. Left as a normal test so the default
//! `cargo test --workspace` exercises them.

use omega_core::circuit::{GateKind, GateOp, Qubit};
use omega_core::executor::{creg_to_u64, Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;

fn counts(src: &str, mode: MidCircuitMode) -> std::collections::HashMap<u64, u32> {
    let ir = omega_parser::lower_to_ir(src).expect("lower");
    let cfg = ExecConfig {
        shots: Some(200),
        seed: Some(11),
        mid_circuit_mode: mode,
    };
    match omega_backend_statevector::StatevectorBackend::new()
        .execute(&ir, &ParameterBinding::default(), &cfg)
        .expect("a 70-bit creg using 2 of its bits must run, not be refused")
    {
        ExecResult::Counts(c) => c,
        o => panic!("{o:?}"),
    }
}

/// A 70-entry register with only bits 0 and 1 written encodes to 2 bits.
#[test]
fn creg_to_u64_reads_the_bits_in_use_not_the_declared_width() {
    let mut bits = vec![0u8; 70];
    bits[0] = 1;
    bits[1] = 1;
    assert_eq!(creg_to_u64(&bits), 0b11);

    // All-zero above 63 is fine at any width.
    let wide = vec![0u8; 4096];
    assert_eq!(creg_to_u64(&wide), 0);
}

/// End to end: 4 qubits, a 70-bit creg, collapse mode.
#[test]
fn a_seventy_bit_creg_with_two_measured_bits_runs() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\ncreg c[70];\n\
               h q[0];\ncx q[0], q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n";
    let c = counts(src, MidCircuitMode::Collapse);
    let bad: Vec<u64> = c.keys().copied().filter(|k| *k != 0 && *k != 0b11).collect();
    assert!(bad.is_empty(), "Bell pair keyed on a wide creg gave {bad:?}");
    assert_eq!(c.values().sum::<u32>(), 200);
}

/// **The guarded branch.** `if (c == 1)` over a 70-bit register.
///
/// `c` holds only bit 0, so the condition is decidable and must fire exactly
/// when q0 measured 1. The `x q[3]` it guards makes that observable: q3 is 1
/// in precisely the shots where q0 was.
#[test]
fn a_condition_over_a_wide_creg_neither_panics_nor_wraps() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\ncreg c[70];\n\
               h q[0];\nmeasure q[0] -> c[0];\nif(c==1) x q[3];\n\
               measure q[3] -> c[1];\n";
    let c = counts(src, MidCircuitMode::Collapse);
    let bad: Vec<u64> = c.keys().copied().filter(|k| *k != 0 && *k != 0b11).collect();
    assert!(
        bad.is_empty(),
        "q3 must equal q0 in every shot (the guard copies it), got keys {bad:?} \
         — a key of 0b01 means the guard fired when it should not have, or the \
         reverse"
    );
    assert_eq!(c.len(), 2, "both branches must occur over 200 shots: {c:?}");
}

/// **A set bit at or above 64 is genuinely unrepresentable**, and the condition
/// must report false rather than fold it into a low bit.
///
/// This is the case the old `<< i` masked: with bit 64 set and `expected == 1`,
/// release-mode `1u64 << 64` becomes `1u64 << 0`, so the register read as 1 and
/// the guard fired. Checked directly on `Operation`, because no QASM2 source
/// can set bit 64 without measuring 65 qubits.
#[test]
fn a_set_bit_above_the_key_width_makes_a_condition_false() {
    let op = GateOp {
        gate: GateKind::X,
        qubits: vec![Qubit(0)].into(),
        params: Default::default(),
        classical_bit: None,
        condition: Some((0, 70, 1)),
    };

    let mut bits = vec![0u8; 70];
    bits[0] = 1;
    assert!(op.condition_satisfied(&bits), "value 1 must satisfy c == 1");

    bits[64] = 1; // value is now 2^64 + 1, which no u64 `expected` can equal
    assert!(
        !op.condition_satisfied(&bits),
        "bit 64 was folded into bit 0 by a masked shift: the register holds \
         2^64 + 1 and the condition tests for 1"
    );
}
