// SPDX-License-Identifier: Apache-2.0
//! CUDA's analytic-Reset criterion, and the divergence it used to have.
//!
//! CUDA used to refuse an analytic reset whenever the **outcome** was random;
//! the CPU and Metal refuse whenever the qubit is **entangled**. Those differ
//! on an unentangled superposition — `H q0; Reset q0` has `p0 = 0.5` but purity
//! 1, so both branches land on `|0⟩⊗rest` and the *result* is deterministic
//! even though the outcome is not. CPU and Metal accepted it; CUDA rejected it.
//!
//! **Why this file exists rather than relying on the suite that was already
//! there.** `reset_matches_cpu` uses `H; CX; Ry; Reset q0` — an *entangled*
//! reset, which every backend refused before this change and still refuses
//! after. It is green either way, so it could not verify the fix. A test that
//! cannot fail is not evidence.
//!
//! The two cases below are the ones that can:
//!
//! * the previously-rejected unentangled superposition must now be **accepted**
//!   and agree with the CPU;
//! * an entangled qubit must still be **refused** — this is the direction that
//!   matters, because the change swapped a criterion that was over-strict but
//!   never wrong for one that *can* be wrong. A false accept here returns a
//!   confident answer for a state a single statevector cannot represent.
//!
//! Comparison is on **⟨Z⟩, not amplitudes**, deliberately. The CPU draws its
//! branch from the RNG while CUDA picks deterministically (`p0 <= 0.5`), so on
//! `|−⟩` the two land on states differing by a global sign. An amplitude
//! assertion would flake; the observable is the physical content.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use omega_backend_statevector::StatevectorBackend;
use omega_backend_statevector_cuda::CudaStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, MidCircuitMode, Observable, PauliOp};
use omega_core::params::ParameterBinding;

fn op(gate: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
        params: params.iter().map(|p| ParamExpr::Concrete(*p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

fn circuit(n: u32, ops: Vec<GateOp>) -> CircuitIR {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    c.ops = ops;
    c
}

fn analytic() -> ExecConfig {
    ExecConfig {
        shots: None,
        seed: Some(3),
        mid_circuit_mode: MidCircuitMode::Skip,
    }
}

fn z_on(q: u32) -> Observable {
    Observable {
        terms: vec![(1.0, vec![(q, PauliOp::Z)])],
    }
}

fn backends() -> Option<(StatevectorBackend, CudaStatevectorBackend)> {
    match CudaStatevectorBackend::new() {
        Ok(c) => Some((StatevectorBackend::new(), c)),
        Err(e) => {
            eprintln!("SKIP (NOT A PASS): no CUDA device ({e})");
            None
        }
    }
}

/// The false rejection this change fixes. `H q0; Reset q0` — unentangled,
/// `p0 = 0.5`, purity 1. CPU accepts; CUDA used to refuse.
#[test]
fn unentangled_superposition_reset_is_accepted_and_matches_cpu() {
    let Some((cpu, cuda)) = backends() else {
        return;
    };
    let c = circuit(
        2,
        vec![
            op(GateKind::H, &[0], &[]),
            op(GateKind::Reset, &[0], &[]),
            // Keep computing after the reset so a wrong reset propagates into
            // the observable rather than being masked by measuring q0 alone.
            op(GateKind::H, &[1], &[]),
        ],
    );
    let p = ParameterBinding::new();

    let want = cpu
        .expectation(&c, &p, &z_on(0))
        .expect("CPU accepts an unentangled reset");
    let got = cuda
        .expectation(&c, &p, &z_on(0))
        .expect("CUDA must now ACCEPT an unentangled reset — this is the fix");

    // Reset drives q0 to |0>, so <Z_0> = +1 exactly on both sides.
    assert!(
        (want - 1.0).abs() < 1e-5,
        "fixture sanity: CPU <Z_0> after reset should be +1, got {want}"
    );
    assert!(
        (got - want).abs() < 1e-5,
        "CUDA <Z_0> {got} != CPU {want} after an unentangled reset"
    );
}

/// The direction that protects correctness. An entangled qubit leaves the
/// register mixed, which one statevector cannot represent, so BOTH backends
/// must refuse. A tolerance too loose would turn this into a confident wrong
/// answer.
#[test]
fn entangled_reset_is_still_refused_by_both() {
    let Some((cpu, cuda)) = backends() else {
        return;
    };
    let c = circuit(
        2,
        vec![
            op(GateKind::H, &[0], &[]),
            op(GateKind::CX, &[0, 1], &[]),
            op(GateKind::Reset, &[0], &[]),
        ],
    );
    let p = ParameterBinding::new();

    assert!(
        cpu.execute(&c, &p, &analytic()).is_err(),
        "CPU must refuse an analytic entangled reset"
    );
    assert!(
        cuda.execute(&c, &p, &analytic()).is_err(),
        "CUDA must STILL refuse an analytic entangled reset — accepting it \
         would return an answer for a state a statevector cannot represent"
    );
}

/// Near the boundary, where a loose tolerance shows up.
///
/// `Ry(θ) q0; CX q0,q1` gives `cos(θ/2)|00⟩ + sin(θ/2)|11⟩`, whose reduced
/// purity on q0 is `cos⁴(θ/2) + sin⁴(θ/2) = 1 − sin²(θ)/2`. So the deviation
/// from pure is `sin²(θ)/2` and the angle dials it directly:
///
///   θ = 0.0045 → ~1.0e-5   (10× the 1e-6 bound — the genuinely near case)
///   θ = 0.05   → ~1.2e-3
///   θ = 0.4    → ~7.6e-2
///
/// The first row is the one with teeth. It is entangled by only 1e-5, which the
/// old 1e-4 tolerance proposed in the code comment would have **accepted** —
/// returning a confident answer for a mixed state. This pins that the boundary
/// is enforced at the value actually chosen rather than at a nominal one.
#[test]
fn weakly_entangled_reset_is_refused_not_waved_through() {
    let Some((_cpu, cuda)) = backends() else {
        return;
    };
    let p = ParameterBinding::new();
    for theta in [0.0045_f64, 0.05, 0.4] {
        let deviation = theta.sin().powi(2) / 2.0;
        let c = circuit(
            2,
            vec![
                op(GateKind::Ry, &[0], &[theta]),
                op(GateKind::CX, &[0, 1], &[]),
                op(GateKind::Reset, &[0], &[]),
            ],
        );
        assert!(
            cuda.execute(&c, &p, &analytic()).is_err(),
            "theta={theta} (purity deviation {deviation:.2e}): a weakly \
             entangled qubit must be refused. A tolerance loose enough to \
             accept this admits states whose amplitude error is orders outside \
             this repo's own gates (5e-7 f32, 1e-9 cross-check)"
        );
    }
}
