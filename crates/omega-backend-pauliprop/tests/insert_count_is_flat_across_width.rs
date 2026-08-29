// SPDX-License-Identifier: Apache-2.0
//! **The work pauliprop does is bounded by the observable's light cone, not by
//! the register — and this is the test that says so.**
//!
//! This backend's whole claim to running at 100+ qubits is that a gate acting
//! outside the observable's support is skipped rather than rebuilding the term
//! map into a bit-identical copy. Before the skip existed, insert counts rose
//! 58% from nq = 10 to nq = 20 on this corpus for no change in the term
//! dynamics.
//!
//! # Why a COUNT and not a timing
//!
//! `PLAN-PAULISUM-MAP.md` records the evidence as a wall-clock table AND an
//! insert count, and says "the ratio is not the point — the flat ON column is".
//! It is right, and the count is the stronger half: a wall-clock ratio moves
//! with whatever else the machine is doing (a 3.4x swing from an unrelated
//! browser tab nearly produced a false regression report on the sibling
//! machine), while a count is identical on an idle box and a loaded one.
//!
//! But until this test there was **no insert counter in the tree**. The
//! published number came from an ad-hoc patch that was never committed, so the
//! load-immune claim was also unreproducible — by anyone, including on the box
//! that measured it. A `benches/` directory would not have fixed that: a bench
//! measures time and can never re-derive a count.

use omega_backend_pauliprop::pauli::{inserts, reset_inserts};
use omega_backend_pauliprop::sim::gates_skipped;
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

/// Layered, 2q-heavy, and — crucially — the ENTANGLING CHAIN DOES NOT REACH
/// past a fixed neighbourhood.
///
/// The chain is deliberately confined to the first `SPAN` qubits. Widening the
/// register then adds single-qubit rotations far from the observable and
/// nothing else, which is precisely the situation the skip exists for: work that
/// the register makes available and the light cone makes irrelevant.
///
/// A nearest-neighbour chain across the WHOLE register would grow the cone by
/// one qubit per layer in each direction, so the count would be flat only until
/// the cone hit the register edge and would then differ at the narrow widths for
/// a legitimate reason. That would test the register boundary, not the skip.
const SPAN: u32 = 6;

fn circuit(nq: u32, layers: usize) -> CircuitIR {
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
        for q in 0..SPAN.min(nq).saturating_sub(1) {
            c.ops.push(op(GateKind::CX, &[q, q + 1], &[]));
        }
    }
    c
}

fn obs() -> Observable {
    Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z), (1, PauliOp::Z)])],
    }
}

/// `(value, inserts, gates_skipped)` for one width.
fn run(nq: u32, layers: usize) -> (f64, u64, u64) {
    let before_skips = gates_skipped();
    reset_inserts();
    let v = PauliPropBackend::new()
        .max_freq(Some(3))
        // This file measures WORK — insert counts and gate skips — not
        // accuracy, and `max_freq(3)` on this circuit is aggressive enough that
        // the dropped-mass budget goes vacuous (~33 on a quantity in [-1, 1]).
        // The informativeness gate is right to refuse that as an *answer*; here
        // the value is only ever compared against itself at another width, so
        // the ceiling is lifted deliberately rather than the truncation loosened
        // — loosening it would change the very work being counted.
        .with_max_dropped_mass(Some(f64::INFINITY))
        .expectation(&circuit(nq, layers), &ParameterBinding::new(), &obs())
        .expect("propagation must succeed");
    (v, inserts(), gates_skipped() - before_skips)
}

/// The property, stated as an equality rather than a ratio: the work is
/// **identical** at 10, 14, 20 and 30 qubits.
#[test]
fn the_insert_count_does_not_grow_with_the_register() {
    let widths = [10u32, 14, 20, 30];
    let results: Vec<_> = widths.iter().map(|&nq| (nq, run(nq, 6))).collect();

    for (nq, (v, ins, skipped)) in &results {
        eprintln!("nq={nq:>3}  inserts={ins:>10}  skipped={skipped:>6}  <O>={v:.12}");
    }

    let (_, (_, baseline, _)) = results[0];
    for (nq, (_, ins, _)) in &results {
        assert_eq!(
            *ins, baseline,
            "insert count at nq={nq} is {ins}, but {baseline} at nq={}. \
             Cost must track the observable's light cone, not the register — \
             that is the property the 100+ qubit claim rests on, and it is \
             silently violated the moment a gate outside the support stops \
             being skipped.",
            widths[0]
        );
    }

    assert!(baseline > 0, "a zero baseline would make this test vacuous");
}

/// The counter must not be measuring nothing, and the skip must actually be
/// firing more as the register widens — otherwise flatness could be an artefact
/// of a circuit that does no work.
#[test]
fn wider_registers_skip_strictly_more_gates() {
    let (_, _, s10) = run(10, 6);
    let (_, _, s30) = run(30, 6);
    assert!(
        s30 > s10,
        "nq=30 skipped {s30} gates and nq=10 skipped {s10}; the extra qubits \
         add gates outside the light cone, so the skip count MUST rise even \
         though the insert count does not"
    );
}

/// Flat work must not mean flat-and-wrong: the value has to agree across widths
/// too, since the extra qubits are genuinely irrelevant to this observable.
#[test]
fn the_value_is_unchanged_by_the_extra_qubits() {
    let (v10, _, _) = run(10, 6);
    let (v30, _, _) = run(30, 6);
    assert!(
        (v10 - v30).abs() < 1e-12,
        "<O> = {v10} at nq=10 and {v30} at nq=30; the added qubits are outside \
         the observable's light cone and cannot change it"
    );
}
