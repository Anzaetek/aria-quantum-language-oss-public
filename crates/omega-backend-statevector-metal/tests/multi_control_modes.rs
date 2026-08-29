// SPDX-License-Identifier: Apache-2.0
//! `MultiControlMode` on Metal: the exact CCX/CSwap octet-permutation kernel vs
//! the 15-gate decomposition, both against the **CPU f64 statevector**.
//!
//! Deliberately the same shape as the CUDA crate's `multi_control_modes.rs`,
//! because the two backends' `Decompose` paths are required to agree
//! bit-for-bit and a differently-shaped test would not notice them drifting.
//!
//! # Why the reference is the CPU and not the decomposition
//!
//! The obvious test — exact kernel vs decomposition — proves almost nothing.
//! Both are this project's own code and share its conventions (qubit ordering,
//! the octet slot layout, the interleaved (re, im) f32 buffer), so they would
//! agree on a shared misreading. This repository has already shipped that exact
//! failure: the `Reset` channel was wrong in three backends in three different
//! bases, and every cross-backend agreement gate passed because each pair
//! coincided in whatever basis was checked.
//!
//! So the reference here is `omega-backend-statevector` — f64, and it walks the
//! CCX subspace directly rather than decomposing — and that backend is in turn
//! gated against Qiskit by `omega-xcheck`, whose corpus contains `ccx` and
//! `cswap` for this reason. The chain is Metal → CPU → Qiskit, with no link
//! sharing an implementation.
//!
//! # What each mode is held to
//!
//! * `Exact` is a **permutation**: it moves amplitudes and computes nothing, so
//!   against an f64 reference it is limited only by the f32 device precision of
//!   whatever produced the state. Held to 1e-6.
//! * `Decompose` is 15 gates of f32 rounding with irrational `e^{±iπ/4}`
//!   factors, so it is held to the same bar but is expected to sit measurably
//!   further from it — which the test reports rather than merely asserting.
#![cfg(all(target_os = "macos", feature = "metal"))]

use omega_backend_statevector::StatevectorBackend;
use omega_backend_statevector_metal::MetalStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, MultiControlMode};
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

fn rnd(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// A circuit that puts real amplitude everywhere before the multi-controlled
/// gate, so a wrong permutation cannot hide in a zero.
fn circuit_with(gate: GateKind, n: u32, seed: &mut u64) -> Option<CircuitIR> {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for q in 0..n {
        c.ops.push(op(GateKind::H, &[q], &[]));
        c.ops
            .push(op(GateKind::Ry, &[q], &[0.13 * (q as f64 + 1.0)]));
    }
    // Three distinct qubits. Retry rather than skip: at n=3 a random triple
    // collides often, and skipping silently thinned the corpus to 12-14
    // circuits — which the count guard caught, but only after the fact.
    let mut tries = 0;
    let (a, b, t) = loop {
        let a = (rnd(seed) % n as u64) as u32;
        let b = (rnd(seed) % n as u64) as u32;
        let t = (rnd(seed) % n as u64) as u32;
        if a != b && a != t && b != t {
            break (a, b, t);
        }
        tries += 1;
        if tries > 200 {
            return None;
        }
    };
    c.ops.push(op(gate.clone(), &[a, b, t], &[]));
    // Keep computing afterwards so an error propagates rather than sitting in
    // one amplitude.
    for q in 0..n {
        c.ops.push(op(GateKind::H, &[q], &[]));
    }
    Some(c)
}

fn analytic() -> ExecConfig {
    ExecConfig {
        shots: None,
        seed: Some(1),
        mid_circuit_mode: MidCircuitMode::Skip,
    }
}

fn probs(r: ExecResult) -> Vec<f64> {
    match r {
        ExecResult::Statevector(v) => v.iter().map(|a| a.norm_sqr()).collect(),
        ExecResult::Probabilities(p) => p,
        other => panic!("expected amplitudes, got {other:?}"),
    }
}

#[test]
fn both_modes_match_the_cpu_statevector_on_ccx_and_cswap() {
    let Ok(base) = MetalStatevectorBackend::new() else {
        eprintln!("SKIP (NOT A PASS): no Metal device");
        return;
    };
    let exact = MetalStatevectorBackend::new()
        .expect("metal")
        .with_multi_control(MultiControlMode::Exact);
    let cpu = StatevectorBackend::new();
    let pb = ParameterBinding::new();

    let mut seed = 0xBEEFu64;
    let mut checked = 0usize;
    let (mut worst_exact, mut worst_dec) = (0.0f64, 0.0f64);

    for gate in [GateKind::CCX, GateKind::CSwap] {
        for n in 3..=6u32 {
            for _ in 0..4 {
                let Some(c) = circuit_with(gate.clone(), n, &mut seed) else {
                    continue;
                };
                let want = probs(cpu.execute(&c, &pb, &analytic()).expect("cpu"));
                let got_dec = probs(base.execute(&c, &pb, &analytic()).expect("decompose"));
                let got_exact = probs(exact.execute(&c, &pb, &analytic()).expect("exact"));

                let d = |a: &[f64]| {
                    a.iter()
                        .zip(want.iter())
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0f64, f64::max)
                };
                let (de, dd) = (d(&got_exact), d(&got_dec));
                worst_exact = worst_exact.max(de);
                worst_dec = worst_dec.max(dd);

                assert!(
                    de < 1e-6,
                    "{gate:?} n={n}: EXACT mode differs from the CPU by {de:.3e}. \
                     A permutation computes nothing, so this is a wrong slot \
                     mapping, not rounding."
                );
                assert!(
                    dd < 1e-6,
                    "{gate:?} n={n}: DECOMPOSE mode differs from the CPU by {dd:.3e}"
                );
                checked += 1;
            }
        }
    }

    assert!(
        checked >= 16,
        "only {checked} circuits ran — corpus too thin"
    );
    // Report both, because the point of the switch is that they differ.
    eprintln!(
        "  {checked} circuits: worst |Δp| vs CPU — exact {worst_exact:.3e}, \
         decompose {worst_dec:.3e}"
    );
}

/// Exact vs Decompose, head to head — how much error the 15-gate path carries.
///
/// The CPU comparison above proves both are *correct*; this one quantifies the
/// gap between them, which is the whole reason the switch exists. It is
/// reported rather than bounded tightly: the decomposition's error depends on
/// the angles and the qubit indices, so pinning a number here would be pinning
/// this corpus, not the property.
///
/// Held only to a loose ceiling — if the two ever diverge by more than f32
/// noise can explain, one of them is wrong, and the CPU test above will say
/// which.
#[test]
fn exact_vs_decompose_head_to_head() {
    let Ok(base) = MetalStatevectorBackend::new() else {
        eprintln!("SKIP (NOT A PASS): no Metal device");
        return;
    };
    let exact = MetalStatevectorBackend::new()
        .expect("metal")
        .with_multi_control(MultiControlMode::Exact);
    let cpu = StatevectorBackend::new();
    let pb = ParameterBinding::new();
    let mut seed = 0x5EEDu64;

    let mut worst_pair = 0.0f64;
    let mut worst_dec_err = 0.0f64;
    let mut worst_exact_err = 0.0f64;
    let mut checked = 0usize;

    for gate in [GateKind::CCX, GateKind::CSwap] {
        for n in 3..=6u32 {
            for _ in 0..4 {
                let Some(c) = circuit_with(gate.clone(), n, &mut seed) else {
                    continue;
                };
                let want = probs(cpu.execute(&c, &pb, &analytic()).expect("cpu"));
                let a = probs(base.execute(&c, &pb, &analytic()).expect("decompose"));
                let b = probs(exact.execute(&c, &pb, &analytic()).expect("exact"));

                let m = |x: &[f64], y: &[f64]| {
                    x.iter()
                        .zip(y.iter())
                        .map(|(p, q)| (p - q).abs())
                        .fold(0.0f64, f64::max)
                };
                worst_pair = worst_pair.max(m(&a, &b));
                worst_dec_err = worst_dec_err.max(m(&a, &want));
                worst_exact_err = worst_exact_err.max(m(&b, &want));
                checked += 1;
            }
        }
    }

    assert!(checked >= 16, "only {checked} circuits ran");
    eprintln!(
        "  exact vs decompose over {checked} circuits: worst |Δp| between them \
         {worst_pair:.3e}\n  error vs CPU f64 — exact {worst_exact_err:.3e}, \
         decompose {worst_dec_err:.3e}  (ratio {:.1}x)",
        if worst_exact_err > 0.0 {
            worst_dec_err / worst_exact_err
        } else {
            f64::INFINITY
        }
    );
    assert!(
        worst_pair < 1e-5,
        "the two modes differ by {worst_pair:.3e}, far beyond f32 noise — one \
         of them is wrong, and the CPU comparison says which"
    );
}

/// The switch must actually switch. If `Exact` silently fell back to the
/// decomposition every assertion above would still pass, and the feature would
/// be a no-op that looks tested.
#[test]
fn the_two_modes_are_not_the_same_computation() {
    let Ok(base) = MetalStatevectorBackend::new() else {
        eprintln!("SKIP (NOT A PASS): no Metal device");
        return;
    };
    let exact = MetalStatevectorBackend::new()
        .expect("metal")
        .with_multi_control(MultiControlMode::Exact);
    let pb = ParameterBinding::new();
    let mut seed = 0x1234u64;

    let mut any_differ = false;
    for _ in 0..8 {
        let Some(c) = circuit_with(GateKind::CCX, 5, &mut seed) else {
            continue;
        };
        let a = probs(base.execute(&c, &pb, &analytic()).expect("decompose"));
        let b = probs(exact.execute(&c, &pb, &analytic()).expect("exact"));
        if a.iter()
            .zip(b.iter())
            .any(|(x, y)| x.to_bits() != y.to_bits())
        {
            any_differ = true;
            break;
        }
    }
    assert!(
        any_differ,
        "Exact and Decompose produced BIT-IDENTICAL results on every circuit. \
         Either the switch is not wired up, or the decomposition's 15 gates of \
         f32 rounding happen to cancel exactly — the first is far likelier and \
         would make this feature a no-op."
    );
}
