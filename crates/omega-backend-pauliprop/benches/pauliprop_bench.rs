// SPDX-License-Identifier: Apache-2.0
//! Pauli-propagation benches — the crate's first, added 2026-08-18.
//!
//! # Why this did not exist, and what it does and does not settle
//!
//! Eight crates in this workspace carry a `benches/`; this one did not, while
//! carrying two of the largest performance claims in the tree (the packed
//! `PauliKey` and the out-of-support skip). Both were measured with ad-hoc
//! patches that were never committed, so neither could be reproduced by anyone
//! — including on the box that produced the numbers.
//!
//! **A bench fixes half of that and cannot fix the other half.** The
//! light-cone claim's real content is that the insert COUNT is flat across
//! register width, and a count is what `tests/insert_count_is_flat_across_width.rs`
//! asserts. A wall-clock figure can never re-derive it, and moves with whatever
//! else the machine is doing — a 3.4x swing from an unrelated browser process
//! nearly produced a false regression report on the sibling machine this week.
//!
//! So read these as: does the shape of the cost still look the way we think it
//! does, and did a change regress it — NOT as the evidence for the claim.
//!
//! # Measurement conditions
//!
//! criterion interleaves and reports a confidence interval, which is what makes
//! this better than the hand-timed tables in `PLAN-PAULISUM-MAP.md` and
//! `BACKEND-CROSSOVER.md` — **those record no idle check, no load average and
//! no interleaving discipline for any of their figures.** Still check the box is
//! quiet before believing a delta: `uptime`, and compare against the previous
//! run's CI rather than against a number in a document.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use omega_backend_pauliprop::PauliPropBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
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

/// Entangling chain confined to the first `span` qubits; rotations everywhere.
/// Widening the register then adds only work OUTSIDE the observable's light
/// cone, which is the situation the skip exists for.
fn layered(nq: u32, layers: usize, span: u32) -> CircuitIR {
    let mut c = CircuitIR::new(nq, CircuitType::GateBased);
    for l in 0..layers {
        for q in 0..nq {
            c.ops.push(op(
                GateKind::Ry,
                &[q],
                &[0.2 + 0.05 * (q + l as u32) as f64],
            ));
            c.ops.push(op(GateKind::Rz, &[q], &[0.3 + 0.04 * q as f64]));
        }
        for q in 0..span.min(nq).saturating_sub(1) {
            c.ops.push(op(GateKind::CX, &[q, q + 1], &[]));
        }
    }
    c
}

fn local_obs() -> Observable {
    Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z), (1, PauliOp::Z)])],
    }
}

/// **The light-cone property, as a curve.** Cost should be near-FLAT in `nq`.
///
/// This is the timing shadow of `insert_count_is_flat_across_width`. If that
/// test passes and this curve slopes, the slope is per-gate overhead on skipped
/// gates, not extra term work — a different and much cheaper problem.
fn width_sweep(c: &mut Criterion) {
    let mut g = c.benchmark_group("pauliprop/width_sweep_local_observable");
    let pb = ParameterBinding::new();
    let o = local_obs();
    for nq in [10u32, 14, 20, 30, 50] {
        let circ = layered(nq, 6, 6);
        g.bench_with_input(BenchmarkId::from_parameter(nq), &circ, |b, circ| {
            b.iter(|| {
                PauliPropBackend::new()
                    .max_freq(Some(3))
                    .expectation(circ, &pb, &o)
                    .expect("propagation")
            })
        });
    }
    g.finish();
}

/// The branch step with the truncation knobs off — the packed-key merge
/// (`add_weighted`: hash, compare, insert) dominates here, which is what the
/// `PauliKey` packing work targeted.
fn branch_merge(c: &mut Criterion) {
    let mut g = c.benchmark_group("pauliprop/branch_merge_exact");
    let pb = ParameterBinding::new();
    let o = local_obs();
    for (nq, layers) in [(12u32, 3usize), (12, 4)] {
        let circ = layered(nq, layers, nq);
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("nq{nq}_l{layers}")),
            &circ,
            |b, circ| {
                b.iter(|| {
                    PauliPropBackend::new()
                        .max_freq(Some(2))
                        .expectation(circ, &pb, &o)
                        .expect("propagation")
                })
            },
        );
    }
    g.finish();
}

/// A WIDE observable has no small light cone, so the skip cannot fire. This is
/// the control arm: it is what the engine costs when the property the other
/// benches measure does not apply.
fn wide_observable(c: &mut Criterion) {
    let mut g = c.benchmark_group("pauliprop/wide_observable_no_skip");
    let pb = ParameterBinding::new();
    for nq in [10u32, 14, 20] {
        let circ = layered(nq, 4, nq);
        let o = Observable {
            terms: vec![(1.0, (0..nq).map(|q| (q, PauliOp::Z)).collect())],
        };
        g.bench_with_input(BenchmarkId::from_parameter(nq), &circ, |b, circ| {
            b.iter(|| {
                PauliPropBackend::new()
                    .max_freq(Some(2))
                    .expectation(circ, &pb, &o)
                    .expect("propagation")
            })
        });
    }
    g.finish();
}

criterion_group!(benches, width_sweep, branch_merge, wide_observable);
criterion_main!(benches);
