// SPDX-License-Identifier: Apache-2.0
//! **A statevector too big for the machine is refused, not attempted.**
//!
//! Both failures this pins were reachable from an ordinary QASM file, with no
//! flag and no warning:
//!
//! * a 64-qubit circuit panicked the process — `attempt to shift left with
//!   overflow` — because `1usize << 64` is not a shift a 64-bit machine can do.
//!   In release mode it *wraps* instead, which is worse: the run returns a
//!   one-amplitude "statevector" and reports success.
//! * a 36-qubit circuit allocated 2 TiB and got the process OOM-killed, taking
//!   whatever else the box was doing with it.
//!
//! The unit tests next to `capacity::check` cover the arithmetic. These go
//! through the real `Backend::execute`, because the bug was never in the
//! arithmetic — it was that nothing called it.

use omega_backend_statevector::sim::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, ExecConfig};
use omega_core::params::ParameterBinding;

fn one_h(n: u32) -> CircuitIR {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    c.add_op(GateOp {
        gate: GateKind::H,
        qubits: [Qubit(0)].into_iter().collect(),
        params: Default::default(),
        classical_bit: None,
        condition: None,
    });
    c
}

fn run(n: u32, shots: Option<u32>) -> Result<(), String> {
    let backend = StatevectorBackend::new();
    let config = ExecConfig {
        shots,
        ..Default::default()
    };
    backend
        .execute(&one_h(n), &ParameterBinding::default(), &config)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// The exact shape that panicked: 64 qubits through the public entry point.
/// A panic here is a process death, so this test failing is not cosmetic.
#[test]
fn a_sixty_four_qubit_circuit_is_refused_rather_than_panicking() {
    let err = run(64, Some(10)).expect_err("64 qubits must be refused");
    assert!(
        err.contains("64") && err.contains("usize"),
        "the refusal must explain that the index space, not memory, is the \
         limit; got: {err}"
    );
}

/// Release builds wrap instead of panicking, so `n == 64` would otherwise
/// return a 1-amplitude state and *succeed*. Silent wrong answers are the
/// failure mode this whole guard exists to prevent, so check well past the
/// wrap point too.
#[test]
fn widths_past_the_wrap_point_never_succeed() {
    for n in [64u32, 65, 96, 128] {
        assert!(
            run(n, Some(10)).is_err(),
            "{n} qubits must be refused, not silently wrapped to a small state"
        );
    }
}

/// A width that fits the index space but not the machine. 40 qubits is 16 TiB
/// of state — no host running this test has that, so the refusal must be the
/// memory one, naming both figures.
#[test]
fn a_width_that_exceeds_host_memory_is_refused_with_both_numbers() {
    if std::env::var(omega_backend_statevector::capacity::OVERSUBSCRIBE_VAR).is_ok() {
        eprintln!("skipped: the oversubscribe escape hatch is set in this environment");
        return;
    }
    match run(40, Some(10)) {
        Err(err) => {
            assert!(
                err.contains("available"),
                "a memory refusal must say what was available; got: {err}"
            );
            assert!(
                err.contains("mps") || err.contains("pauli"),
                "a refusal should name a backend that would work; got: {err}"
            );
        }
        Ok(()) => panic!(
            "40 qubits (16 TiB of state) was admitted — either the host really \
             has that much, or the memory gate is not being consulted"
        ),
    }
}

/// **The guard must not refuse ordinary work.** A capacity check that fires on
/// a 20-qubit circuit is a worse bug than the panic it replaced, because it
/// breaks every working run rather than an unreachable one.
#[test]
fn ordinary_widths_still_run() {
    for n in [1u32, 4, 12, 20] {
        run(n, Some(16)).unwrap_or_else(|e| panic!("{n} qubits must still run: {e}"));
    }
}

/// Sampling costs twice evolution, so the refusal threshold differs between an
/// analytic run and a sampled one. Both paths must consult the guard — the
/// analytic entry point allocates the same `2^n` vector.
#[test]
fn the_analytic_path_is_guarded_too() {
    assert!(
        run(64, None).is_err(),
        "the analytic path allocates 2^n as well and must be refused at 64"
    );
}
