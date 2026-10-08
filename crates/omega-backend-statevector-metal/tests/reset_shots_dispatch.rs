//! The shots-mode `Reset` delegation must be visible after `execute` returns.
//!
//! Correct counts prove nothing: they were already correct when the GPU never
//! ran. This test fails if the counters stop moving. A Reset circuit in shots
//! mode moves `reset_shots_cpu_fallback_count` and not
//! `metal_shots_execute_count`; the same circuit without Reset does the
//! opposite.

#![cfg(all(target_os = "macos", feature = "metal"))]

use omega_backend_statevector_metal::{
    metal_shots_execute_count, reset_shots_cpu_fallback_count, MetalStatevectorBackend,
};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;

fn push(c: &mut CircuitIR, gate: GateKind, qs: Vec<u32>) {
    c.ops.push(GateOp {
        gate,
        qubits: qs.into_iter().map(Qubit).collect(),
        params: vec![].into(),
        classical_bit: None,
        condition: None,
    });
}

fn bell(with_reset: bool) -> CircuitIR {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    push(&mut c, GateKind::H, vec![0]);
    push(&mut c, GateKind::CX, vec![0, 1]);
    if with_reset {
        push(&mut c, GateKind::Reset, vec![0]);
    }
    c
}

/// Which counter moved across one `execute`. Exactly one must.
fn device_that_ran(cpu_before: u64, metal_before: u64) -> &'static str {
    let cpu = reset_shots_cpu_fallback_count().wrapping_sub(cpu_before);
    let metal = metal_shots_execute_count().wrapping_sub(metal_before);
    match (cpu, metal) {
        (1, 0) => "cpu",
        (0, 1) => "metal",
        _ => panic!(
            "expected exactly one device to record the call, cpu delta {cpu}, metal delta {metal}"
        ),
    }
}

#[test]
fn reset_shots_delegation_names_the_device() {
    let backend = match MetalStatevectorBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Metal backend unavailable: {e:?} — nothing to assert.");
            return;
        }
    };
    let params = ParameterBinding::new();
    let cfg = ExecConfig {
        shots: Some(16),
        seed: Some(0),
        mid_circuit_mode: MidCircuitMode::Skip,
    };

    let cpu_before = reset_shots_cpu_fallback_count();
    let metal_before = metal_shots_execute_count();
    let reset = backend
        .execute(&bell(true), &params, &cfg)
        .expect("reset shots");
    assert!(
        matches!(reset, ExecResult::Counts(_)),
        "reset shots returned {reset:?}"
    );
    assert_eq!(
        device_that_ran(cpu_before, metal_before),
        "cpu",
        "shots + Reset must be the CPU fallback, not a Metal sample"
    );

    let cpu_before = reset_shots_cpu_fallback_count();
    let metal_before = metal_shots_execute_count();
    let plain = backend
        .execute(&bell(false), &params, &cfg)
        .expect("shots without reset");
    assert!(
        matches!(plain, ExecResult::Counts(_)),
        "shots without Reset returned {plain:?}"
    );
    assert_eq!(
        device_that_ran(cpu_before, metal_before),
        "metal",
        "shots without Reset must be a Metal sample, not the CPU fallback"
    );
}
