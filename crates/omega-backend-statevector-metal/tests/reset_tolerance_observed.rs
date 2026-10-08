//! Observes the analytic-`Reset` tolerance split on a real Metal device.
//!
//! `omega-bridges/tests/reset_tolerance_divergence.rs` calls the shared
//! predicate twice and never opens a device, so editing the threshold
//! inside `apply_reset_with` cannot redden it. This test writes one
//! constructed state and asks each backend to Reset qubit 0.
//!
//! `√(1−ε)|00⟩ + √ε|11⟩` at `ε = 4·10⁻⁵` has purity deficit `2ε(1−ε)`.
//! That deficit sits where the two backends disagree: Metal's analytic
//! Reset accepts the state, and the CPU statevector backend refuses it.
//! The assertion is that split. This file does not name either threshold.
//! The only one the Metal side consults is the literal in
//! `apply_reset_with`, read back off the device buffer.
//!
//! `MetalStatevectorBackend::execute` cannot be handed an input state: it
//! leases `|0…0⟩` and, when `shots` is set, a circuit containing `Reset`
//! is delegated to the CPU before any Metal kernel runs. `ExecConfig`'s
//! default is `shots: Some(1024)`, so that path would make both sides the
//! CPU. The analytic entry `execute` calls when `shots` is `None` is
//! `MetalState::apply_reset` (`branch = None`). Allocating from the
//! backend and writing the state reaches that entry on these amplitudes.

#![cfg(all(target_os = "macos", feature = "metal"))]

use num_complex::Complex64;
use omega_backend_statevector::sim::reduced_purity;
use omega_backend_statevector::StatevectorBackend;
use omega_backend_statevector_metal::MetalStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::error::OmegaError;
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;

/// `√(1−ε)|00⟩ + √ε|11⟩`. Same amplitudes as
/// `omega-bridges/tests/reset_tolerance_divergence.rs`. Qubit 0's purity
/// deficit is `2ε(1−ε)`.
fn slightly_entangled(eps: f64) -> Vec<Complex64> {
    let mut state = vec![Complex64::new(0.0, 0.0); 4];
    state[0] = Complex64::new((1.0 - eps).sqrt(), 0.0);
    state[3] = Complex64::new(eps.sqrt(), 0.0);
    state
}

fn push(c: &mut CircuitIR, gate: GateKind, qs: &[u32], params: &[f64]) {
    c.ops.push(GateOp {
        gate,
        qubits: qs.iter().copied().map(Qubit).collect(),
        params: params.iter().copied().map(ParamExpr::Concrete).collect(),
        classical_bit: None,
        condition: None,
    });
}

/// `Ry(θ); CX` with `θ = 2 arcsin(√ε)`, which prepares [`slightly_entangled`]
/// from `|00⟩`. The CPU backend has no write-state entry; this is how the
/// same amplitudes reach `StatevectorBackend::execute` before `Reset`.
fn prepare(eps: f64, reset: bool) -> CircuitIR {
    let theta = 2.0 * eps.sqrt().asin();
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    push(&mut c, GateKind::Ry, &[0], &[theta]);
    push(&mut c, GateKind::CX, &[0, 1], &[]);
    if reset {
        push(&mut c, GateKind::Reset, &[0], &[]);
    }
    c
}

/// Analytic run. `shots: None` is load-bearing on the CPU: with shots, Reset
/// is one trajectory and an entangled qubit is allowed.
fn analytic() -> ExecConfig {
    ExecConfig {
        shots: None,
        seed: Some(0),
        mid_circuit_mode: MidCircuitMode::Skip,
    }
}

fn statevector(result: ExecResult) -> Vec<Complex64> {
    match result {
        ExecResult::Statevector(v) => v,
        other => panic!("expected a statevector, got {other:?}"),
    }
}

/// CPU refused because the analytic Reset guard fired, not because the
/// circuit failed to run.
fn refused_as_entangled(err: &OmegaError) -> bool {
    match err {
        OmegaError::Unsupported(msg) => msg.contains("entangled"),
        _ => false,
    }
}

#[test]
fn metal_reset_accepts_where_the_cpu_refuses() {
    let backend = match MetalStatevectorBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Metal backend unavailable: {e:?} — nothing to assert.");
            return;
        }
    };

    // Same ε as the bridges test. The deficit check is the construction,
    // not a backend threshold: a drifted state fails here, before either
    // Reset, instead of landing where both backends agree.
    let eps = 4e-5;
    let state = slightly_entangled(eps);
    let deficit = 1.0 - reduced_purity(&state, 2, 0);
    let expected = 2.0 * eps * (1.0 - eps);
    assert!(
        (deficit - expected).abs() < 1e-12,
        "purity deficit {deficit} is not 2ε(1−ε) = {expected}"
    );

    let cpu = StatevectorBackend::new();
    let params = ParameterBinding::new();
    let cfg = analytic();
    let prepared = statevector(
        cpu.execute(&prepare(eps, false), &params, &cfg)
            .expect("CPU prepare"),
    );
    for (i, (got, want)) in prepared.iter().zip(state.iter()).enumerate() {
        let d = (*got - *want).norm();
        assert!(
            d < 1e-12,
            "CPU prepare amplitude {i} differs from the constructed state by {d}: \
             the Reset would not be running on that state"
        );
    }

    let cpu_reset = cpu.execute(&prepare(eps, true), &params, &cfg);
    let cpu_refused = match &cpu_reset {
        Err(e) if refused_as_entangled(e) => true,
        Err(e) => panic!("CPU Reset failed for some other reason: {e}"),
        Ok(_) => false,
    };

    // The buffer `apply_reset` reads is this state, stored as the device's
    // f32 amplitudes. The guard's threshold is the one in `apply_reset_with`.
    let mut metal_state = backend.allocate(2).expect("allocate a 2-qubit state");
    metal_state
        .write_state(&state)
        .expect("write the constructed state");
    let metal_reset = metal_state.apply_reset(0);
    let metal_accepted = metal_reset.is_ok();

    assert!(
        metal_accepted && cpu_refused,
        "analytic Reset of qubit 0 on this state must split: Metal {metal_reset:?}, CPU {cpu_reset:?}"
    );
}
