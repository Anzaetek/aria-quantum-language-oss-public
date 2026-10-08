// SPDX-License-Identifier: Apache-2.0
//! Collapse mode without a `Measure`: Metal must not answer where the CPU
//! keys counts differently.
//!
//! CUDA and OpenCL both had this defect and both were measured answering
//! wrongly rather than refusing — on an RTX PRO 6000, `creg c[1]; h q[1];
//! if (c == 0) x q[0];` in collapse mode with 64 shots returned
//! `{"001": 40, "011": 24}` from CUDA and `{"001": 32, "011": 32}` from
//! OpenCL, against the CPU's `{"0": 64}`. Both were reachable through
//! `omega-server` as an **HTTP 200**: a confident wrong answer, not an error.
//!
//! The cause is shared and structural, not a typo in one backend. Each GPU
//! guard refuses collapse mode only when the circuit contains a `Measure`.
//! But a conditioned gate alone forces collapse, and with shots and a declared
//! creg the CPU keys every outcome on the CLASSICAL register even when no
//! `Measure` ever writes it — reading `0` at the creg's width. A backend with
//! only a qubit-register sampler then answers different keys at a different
//! width. Metal's guard at `lib.rs:1220` has the same `Measure`-only shape, so
//! it is the third of three and the only one not yet measured.
//!
//! # What this test accepts
//!
//! Either a refusal naming the mode, or counts equal to the CPU's. It does not
//! presuppose the fix: a Metal backend that grew a creg-keyed sampler and
//! agreed with the CPU would pass, and should.
//!
//! # What it does NOT catch
//!
//! It fixes one shape — one conditioned gate, one classical bit, 64 shots,
//! seed 5. A width-2 creg, or a condition on a bit other than 0, is a
//! different cell and is not covered. It also says nothing about the
//! expectation path, which produces no counts key and is guarded separately.

#![cfg(all(target_os = "macos", feature = "metal"))]

use omega_backend_statevector::StatevectorBackend;
use omega_backend_statevector_metal::MetalStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

/// `creg c[1]; h q[1]; if (c == 0) x q[0];` on three qubits — the reproducer
/// the CUDA and OpenCL defects were found with, unchanged.
fn collapse_without_measure() -> CircuitIR {
    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    c.num_classical_bits = 1;
    c.ops.push(GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    c.ops.push(GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        // (first_bit, num_bits, expected) — `if (c == 0)` over the 1-bit creg.
        condition: Some((0, 1, 0)),
    });
    c
}

/// Counts as a sorted map keyed by the outcome's own rendering, so a width
/// difference shows up as different KEYS rather than as a reordering.
fn counts_of(res: ExecResult) -> std::collections::BTreeMap<String, u32> {
    match res {
        ExecResult::Counts(m) => m
            .into_iter()
            .map(|(outcome, n)| (format!("{outcome}"), n))
            .collect(),
        other => panic!("expected Counts, got {other:?}"),
    }
}

#[test]
fn collapse_without_a_measure_agrees_with_the_cpu_or_is_refused() {
    let Ok(metal) = MetalStatevectorBackend::new() else {
        eprintln!("Metal unavailable — skipping collapse-width test");
        return;
    };
    let circuit = collapse_without_measure();
    let params = ParameterBinding::default();
    let config = ExecConfig {
        shots: Some(64),
        seed: Some(5),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };

    let cpu = counts_of(
        StatevectorBackend::new()
            .execute(&circuit, &params, &config)
            .expect("the CPU backend answers this circuit"),
    );

    match metal.execute(&circuit, &params, &config) {
        Err(e) => {
            let said = e.to_string();
            assert!(
                said.contains("collapse") || said.contains("mid-circuit"),
                "refused, but the message does not name the mode: {said}"
            );
        }
        Ok(res) => {
            let got = counts_of(res);
            assert_eq!(
                got, cpu,
                "Metal ANSWERED collapse mode and disagreed with the CPU. \
                 This is the CUDA/OpenCL defect in the third backend: the CPU \
                 keys on the 1-bit creg, Metal keys on the 3-qubit register, \
                 and omega-server returns it as HTTP 200."
            );
        }
    }
}
