// SPDX-License-Identifier: Apache-2.0
//! What the CUDA backend actually promises about mid-circuit measurement and
//! counts width.
//!
//! **`LINUX-CUDA-VERIFICATION.md` §1 asked the wrong question**, and this file
//! exists to close it with the right one. That document flagged a "width note
//! that deserves a second opinion": does the CUDA *collapse* arm key counts on
//! `circuit.num_qubits` or on the creg width, warning that "if the CUDA
//! collapse counts come out at the wrong width, this line is why".
//!
//! There is no collapse arm. `lib.rs` refuses `MidCircuitMode::Collapse`
//! outright for any circuit containing a `Measure`, so the CLI dispatcher falls
//! back to the CPU. There is no `Outcome` produced on that path at all, and
//! therefore no width to get wrong.
//!
//! That mattered more than a paperwork correction. A test written to the
//! original framing — assert the collapse counts are `counts_outcome_width(c,
//! /*by_creg=*/ true)` — fails, because the *assertion* is wrong rather than
//! the code. "Fixing" the backend to satisfy it means keying the sampler on the
//! creg width, and `Outcome::from_u64` masks silently: full-register basis keys
//! would be truncated to the creg width, quietly, on every circuit containing a
//! measure.
//!
//! So this pins the two things that are real:
//!
//! 1. the refusal itself — the actual contract, previously untested;
//! 2. the width the sampler *does* use, against
//!    `counts_outcome_width(c, false)`, which is what the CPU computes for the
//!    same shapes. Pinning it against the shared helper is what stops the two
//!    backends drifting, which was the legitimate worry underneath the
//!    mis-posed question.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use omega_backend_statevector_cuda::CudaStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{counts_outcome_width, Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;

fn op(gate: GateKind, qubits: &[u32], params: &[f64], classical_bit: Option<u32>) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
        params: params.iter().map(|p| ParamExpr::Concrete(*p)).collect(),
        classical_bit,
        condition: None,
    }
}

/// 4 qubits, a 2-bit creg. Deliberately asymmetric: if anything ever keys on
/// the creg the width comes out 2 instead of 4, and the assertions below
/// separate those cases.
fn narrow_creg_circuit() -> CircuitIR {
    let mut c = CircuitIR::new(4, CircuitType::GateBased);
    c.num_classical_bits = 2;
    c.ops = vec![
        op(GateKind::H, &[0], &[], None),
        op(GateKind::CX, &[0, 1], &[], None),
        op(GateKind::Measure, &[0], &[], Some(0)),
        op(GateKind::Measure, &[1], &[], Some(1)),
    ];
    c
}

fn backend() -> Option<CudaStatevectorBackend> {
    match CudaStatevectorBackend::new() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("SKIP (NOT A PASS): no CUDA device ({e})");
            None
        }
    }
}

/// The contract that actually exists. CUDA declines mid-circuit measurement
/// with collapse so the dispatcher can route to the CPU; a silent wrong answer
/// here would be far worse than a refusal, so the refusal is the feature.
#[test]
fn collapse_with_a_measure_is_refused_not_answered() {
    let Some(cuda) = backend() else { return };
    let c = narrow_creg_circuit();
    let cfg = ExecConfig {
        shots: Some(64),
        seed: Some(1),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let err = cuda
        .execute(&c, &ParameterBinding::new(), &cfg)
        .expect_err("CUDA must refuse MidCircuitMode::Collapse, not answer it");
    let msg = format!("{err}");
    assert!(
        msg.contains("mid-circuit measurement"),
        "the refusal must say WHAT is unsupported so a caller can route around \
         it; got: {msg}"
    );
}

/// The width the sampler really uses, pinned against the shared helper rather
/// than against a literal — a literal would still agree if both sides drifted
/// together.
#[test]
fn skip_mode_counts_key_on_the_qubit_register_width() {
    let Some(cuda) = backend() else { return };
    let c = narrow_creg_circuit();
    let cfg = ExecConfig {
        shots: Some(256),
        seed: Some(7),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let res = cuda
        .execute(&c, &ParameterBinding::new(), &cfg)
        .expect("skip mode must run");

    let want = counts_outcome_width(&c, false);
    assert_eq!(want, 4, "fixture sanity: 4 qubits");
    assert_ne!(
        want,
        counts_outcome_width(&c, true),
        "fixture is pointless unless the creg width (2) differs from the \
         register width (4)"
    );

    let ExecResult::Counts(counts) = res else {
        panic!("shots run must return Counts");
    };
    assert!(!counts.is_empty(), "sampler returned nothing to check");
    for (k, _) in counts.iter() {
        assert_eq!(
            k.width() as usize,
            want,
            "counts key width {} != counts_outcome_width(c, false) = {want}. \
             `sample_counts_on_device` samples the full qubit register, so the \
             register width is correct here — a creg-width key would SILENTLY \
             truncate via Outcome::from_u64",
            k.width()
        );
    }
}

/// 3 qubits, a 1-bit creg, a gate conditioned on that creg, and NO `Measure`.
/// `needs_collapse` is true for this shape (any conditioned gate), so every
/// caller that follows the shared predicate runs it in `Collapse` mode.
fn conditioned_without_measure_circuit() -> CircuitIR {
    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    c.num_classical_bits = 1;
    let mut x = op(GateKind::X, &[0], &[], None);
    // `if (c == 0) x q[0];` — the creg is never written, so the gate fires.
    x.condition = Some((0, 1, 0));
    c.ops = vec![op(GateKind::H, &[1], &[], None), x];
    c
}

/// **Collapse mode must not be answered differently from the CPU when the
/// circuit has no `Measure`.**
///
/// The refusal above only looks for a `Measure`. A circuit whose collapse is
/// forced by a classically conditioned gate alone passes that check and is
/// answered by the qubit-register sampler, while the CPU keys the same run on
/// the classical register (the qiskit shape: an unwritten `creg c[1]` reads
/// `0`). The two backends then disagree on the key width AND the key value,
/// and a caller that renders at `counts_outcome_width(c, true)` mislabels
/// the CUDA result.
#[test]
fn collapse_without_a_measure_agrees_with_the_cpu() {
    let Some(cuda) = backend() else { return };
    let c = conditioned_without_measure_circuit();
    assert!(
        omega_core::executor::needs_collapse(&c),
        "fixture sanity: a conditioned gate forces collapse"
    );
    let cfg = ExecConfig {
        shots: Some(64),
        seed: Some(5),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let cpu = omega_backend_statevector::StatevectorBackend::new()
        .execute(&c, &ParameterBinding::new(), &cfg)
        .expect("cpu runs collapse mode");
    let ExecResult::Counts(cpu) = cpu else {
        panic!("cpu shots run must return Counts");
    };
    let want = counts_outcome_width(&c, true);
    for k in cpu.keys() {
        assert_eq!(
            k.width() as usize,
            want,
            "fixture sanity: cpu keys on the creg"
        );
    }

    match cuda.execute(&c, &ParameterBinding::new(), &cfg) {
        // Refusing is an acceptable answer: the caller routes to the CPU.
        Err(e) => assert!(
            {
                let m = format!("{e}");
                m.contains("collapse mode") || m.contains("mid-circuit measurement")
            },
            "a refusal must name what is unsupported; got: {e}"
        ),
        Ok(ExecResult::Counts(got)) => {
            let render = |m: &std::collections::HashMap<omega_core::outcome::Outcome, u32>| {
                let mut v: Vec<(String, u32)> =
                    m.iter().map(|(k, n)| (k.to_bitstring(), *n)).collect();
                v.sort();
                v
            };
            assert_eq!(
                render(&got),
                render(&cpu),
                "CUDA answered collapse mode differently from the CPU on a \
                 circuit with no Measure"
            );
        }
        Ok(other) => panic!("shots run returned a non-Counts result: {other:?}"),
    }
}
