// SPDX-License-Identifier: Apache-2.0
//! Qudit gates must be refused by name, not silently accepted.
//!
//! `GateKind::Rxy` and `GateKind::CSum` arrived with the qudit work and this
//! crate never got the arms. The matches were therefore non-exhaustive and the
//! crate **did not compile** under `--features opencl` — which `ci.sh` builds by
//! default on macOS (`ARIA_OPENCL` defaults to 1), while the workspace clippy in
//! step 2 never sees it because the feature is off. Two different gates, each
//! blind in the other's direction, so a hard compile error sat in the tree.
//!
//! # Why this file exists rather than just the arms
//!
//! Adding `GateKind::Rxy | GateKind::CSum => Ok(())` would also have compiled,
//! and then a qudit circuit handed to this backend would run as a sequence of
//! no-ops and return a confident wrong statevector. That mutation was tried:
//! it compiles, and `cargo clippy --features opencl --all-targets -D warnings`
//! exits 0 with nothing in the crate noticing. So the refusal needs an
//! assertion, or the compiler is the only thing holding it and the compiler
//! cannot tell a refusal from a no-op.
//!
//! # What this test does NOT catch
//!
//! It asserts `execute` refuses. It does not reach `adjoint.rs`'s own arm —
//! `adjoint_opencl_matches_cpu.rs` is where that belongs, and a mutation of the
//! adjoint arm alone would leave this file green. It also cannot run without an
//! OpenCL device: like every other integration test here it skips when the
//! constructor cannot reach one, and `device_present.rs` plus
//! `ARIA_OPENCL_REQUIRE_DEVICE=1` is what stops that skip from being silent.

#![cfg(feature = "opencl")]

use omega_backend_statevector_opencl::{OpenClError, OpenClStatevectorBackend};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::error::OmegaError;
use omega_core::executor::{Backend, ExecConfig, MidCircuitMode};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

fn backend() -> Option<OpenClStatevectorBackend> {
    match OpenClStatevectorBackend::new() {
        Ok(b) => Some(b),
        Err(OpenClError::Unavailable(_)) | Err(OpenClError::Runtime(_)) => {
            eprintln!("OpenCL unavailable — skipping qudit refusal test");
            None
        }
        Err(e) => panic!("unexpected backend init error: {e}"),
    }
}

/// `Rxy` carries (level_a, level_b, theta, phi) — a rotation inside a
/// d-dimensional wire. There is no OpenCL kernel for it and there is no
/// qubit reading of it, so the only honest answer is a refusal.
fn rxy_op() -> GateOp {
    GateOp {
        gate: GateKind::Rxy,
        qubits: smallvec![Qubit(0)],
        params: smallvec![
            ParamExpr::Concrete(0.0),
            ParamExpr::Concrete(1.0),
            ParamExpr::Concrete(0.5),
            ParamExpr::Concrete(0.0),
        ],
        classical_bit: None,
        condition: None,
    }
}

fn csum_op() -> GateOp {
    GateOp {
        gate: GateKind::CSum,
        qubits: smallvec![Qubit(0), Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    }
}

fn refuses(op: GateOp, name: &str) {
    let Some(backend) = backend() else { return };
    let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
    circuit.add_op(op);
    let config = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    match backend.execute(&circuit, &ParameterBinding::default(), &config) {
        Err(OmegaError::Unsupported(msg)) => {
            assert!(
                msg.contains(name),
                "refused, but the message does not name the gate: {msg}"
            );
        }
        Err(e) => panic!("{name} refused, but not with Unsupported: {e:?}"),
        Ok(out) => panic!(
            "{name} was ACCEPTED and returned {out:?}. A qudit gate run as a \
             no-op on a qubit statevector produces a confident wrong answer; \
             this backend must refuse it."
        ),
    }
}

#[test]
fn rxy_is_refused_and_the_message_names_it() {
    refuses(rxy_op(), "Rxy");
}

#[test]
fn csum_is_refused_and_the_message_names_it() {
    refuses(csum_op(), "CSum");
}
