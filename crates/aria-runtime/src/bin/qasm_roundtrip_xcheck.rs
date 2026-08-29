// SPDX-License-Identifier: Apache-2.0
//! **Aria NATIVE execution vs Qiskit running Aria's EXPORTED text.**
//!
//! `OPTIONAL_TESTS.md` gap #5, and the one that document singles out as most
//! valuable. The distinction it insists on is subtle and load-bearing:
//!
//! > comparing *Aria native* against *Qiskit-on-Aria's-export*, not just both
//! > engines on the same text, since a lossy export makes them agree.
//!
//! A differential test conducted THROUGH an export can pass precisely because
//! the export dropped something on both sides. The reference here is therefore
//! Aria's own execution of the circuit it holds in memory — never the exported
//! text — so anything the exporter loses shows up as a disagreement.
//!
//! # Why the existing export checks do not cover this
//!
//! `tools/qiskit_xcheck/qasm2_dialect.py` already loads our export into Qiskit
//! and compares `Operator(loaded)` against a Qiskit-built reference. That is a
//! real check, and it is structurally blind to exactly the defect that
//! motivated this one: **`Operator()` raises on measurement and on classical
//! conditions**, so every circuit containing them is excluded from that harness.
//! The 2026-08-08 defect was `if(c==V)` being dropped on export — a circuit
//! shape that check cannot look at.
//!
//! So this corpus is deliberately made of the shapes `Operator()` cannot handle:
//! mid-circuit measurement, and gates conditioned on a measured bit.
//!
//! # Compared on DISTRIBUTIONS, and the condition is sometimes false
//!
//! Sampling defects are invisible to an analytic vector comparison, so this
//! compares shot distributions. And every conditional here is driven by an `H`,
//! so it is true on roughly half the shots: a condition that is ALWAYS true
//! hides a dropped guard, because both engines then lose the same thing and
//! agree. That is the same trap the feedforward corpus in `omega-xcheck`
//! documents, applied to the export path.
//!
//! Output is one record per circuit, consumed by
//! `tools/qiskit_xcheck/compare_qasm_roundtrip.py`.

use std::collections::HashMap;

use aria_core::ast::nodes::{Circuit, Clbit, GateDef, GateKind, Qubit};
use aria_core::ast::qasm::to_qasm;
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;

fn rnd(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// A circuit whose classical condition has an OBSERVABLE effect.
///
/// # The guard reads a SIZE-1 register, and that is not stylistic
///
/// The first version used one `creg c[n]` for both guard and readout, and every
/// circuit came back `EXPORT_REFUSED`:
///
/// > Aria conditions on a SINGLE classical bit, but QASM 2.0's `if` compares the
/// > WHOLE register … Emitting `if (c == 1)` would assert that the entire
/// > register equals 1, a different predicate — a silent change of meaning.
///
/// That refusal is correct and the corpus was wrong. A separate `creg m[1]` makes
/// `if (m == 1)` mean exactly `m[0] == 1`. Worth recording rather than quietly
/// fixing: a harness that SKIPPED refusals would have reported "0 disagreements"
/// over an empty corpus.
///
/// # The target starts in |0⟩, and that is the whole point
///
/// The second version put the conditional's target in an `H` superposition. The
/// export was fine, the comparison passed — and the corpus was VACUOUS: flipping
/// a uniformly random qubit leaves the distribution uniform, so a DROPPED
/// conditional produces exactly the same counts. It would have passed with the
/// very defect it exists to catch.
///
/// Leaving the target in |0⟩ makes the effect a hard correlation: `c[t] == m` on
/// every shot. Drop the conditional and `c[t]` is 0 always, so half the
/// distribution moves. [`correlation_is_observable`] asserts that signature on
/// the native counts before anything is printed, so the corpus cannot silently
/// go vacuous again.
fn make(seed: &mut u64, idx: usize) -> (Circuit, usize) {
    let n = 2 + (idx % 2); // 2 or 3 qubits
    let mut c = Circuit::new(&format!("rt{idx}"));
    c.qreg("q", n);
    c.creg("m", 1); // the guard bit, alone in its register
    c.creg("c", n); // readout

    // q0 in superposition: the guard is a genuine coin flip, so the condition
    // is FALSE on about half the shots. A condition that is always true hides a
    // dropped guard — both engines lose the same thing and agree.
    c.apply(GateDef::new(GateKind::H), vec![Qubit::new("q", 0)]);
    c.measure(&Qubit::new("q", 0), &Clbit::new("m", 0));

    // Target stays in |0⟩. Never q0, and never given an H.
    let target = 1 + (rnd(seed) % (n as u64 - 1)) as usize;

    // Conditional X — the construct the 2026-08-08 export dropped.
    c.instructions.push(aria_core::ast::nodes::Instruction {
        gate: GateDef::new(GateKind::X),
        qubits: vec![Qubit::new("q", target)],
        clbits: vec![],
        condition: Some((Clbit::new("m", 0), 1)),
    });

    // A spectator gate on a NON-target qubit, so the corpus is not just the
    // same circuit N times. Chosen from the Cliffords that leave |0⟩ alone in
    // the Z basis, so it cannot mask the correlation being asserted.
    if n == 3 {
        let spectator = if target == 1 { 2 } else { 1 };
        if rnd(seed).is_multiple_of(2) {
            c.apply(GateDef::new(GateKind::Z), vec![Qubit::new("q", spectator)]);
        }
    }

    for q in 0..n {
        c.measure(&Qubit::new("q", q), &Clbit::new("c", q));
    }
    (c, target)
}

/// Does the native distribution actually SHOW the conditional?
///
/// `c[t] == m` must hold on every shot. If the conditional were dropped, `c[t]`
/// would be 0 while `m` is 1 on about half the shots, so this returns false.
///
/// Checked on the NATIVE counts, before the record is emitted: a corpus that has
/// gone vacuous must fail loudly here rather than sail through the comparison.
/// The key layout is `m` least-significant, printed last — derived from the
/// invariant `m == c[0]` (q0 is measured into both), not assumed.
fn correlation_is_observable(counts: &[(String, u32)], n: usize, target: usize) -> bool {
    let width = 1 + n; // m[1] then c[n]
    let mut saw_true = false;
    let mut saw_false = false;
    for (key, v) in counts {
        if *v == 0 {
            continue;
        }
        if key.len() != width {
            return false;
        }
        let bit = |i: usize| -> u8 {
            // index 0 is least significant, so it is the LAST character.
            key.as_bytes()[width - 1 - i] - b'0'
        };
        let m = bit(0);
        let ct = bit(1 + target);
        if m != ct {
            return false; // the conditional did not fire when it should have
        }
        if m == 1 {
            saw_true = true;
        } else {
            saw_false = true;
        }
    }
    // Both branches must actually occur, or the guard was not a coin flip.
    saw_true && saw_false
}

const SHOTS: u32 = 8000;

fn main() {
    let n_circ: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let be = omega_backend_statevector::StatevectorBackend::new();
    let pb = ParameterBinding::new();
    let mut seed = 0xA51Au64;

    println!("#BEGIN");
    for i in 0..n_circ {
        let (circ, target) = make(&mut seed, i);
        // n is set by `make`; recompute from the quantum register rather than
        // threading it, so the two cannot disagree.
        let n_q: usize = circ
            .registers
            .iter()
            .filter(|r| r.name == "q")
            .map(|r| r.size)
            .sum();

        // The EXPORT under test. A refusal is a result, not a skip: this
        // exporter is documented to refuse what it cannot express, and a
        // refusal here means the corpus found an inexpressible shape, which is
        // worth printing rather than silently dropping.
        let qasm = match to_qasm(&circ) {
            Ok(q) => q,
            Err(e) => {
                println!("R {i} EXPORT_REFUSED {e}");
                continue;
            }
        };

        // NATIVE execution — the reference. Lowered from the same `Circuit`
        // object that was exported, NOT re-imported from the text.
        let lowered = match aria_runtime::lower::lower(&circ) {
            Ok(l) => l,
            Err(e) => {
                println!("R {i} LOWER_FAILED {e}");
                continue;
            }
        };
        let cfg = ExecConfig {
            shots: Some(SHOTS),
            seed: Some(0xC0FFEE + i as u64),
            mid_circuit_mode: MidCircuitMode::Collapse,
        };
        let counts = match be.execute(&lowered.ir, &pb, &cfg) {
            Ok(ExecResult::Counts(c)) => c,
            other => {
                println!("R {i} NO_COUNTS {other:?}");
                continue;
            }
        };
        let mut agg: HashMap<String, u32> = HashMap::new();
        for (k, v) in &counts {
            *agg.entry(format!("{k}")).or_insert(0) += v;
        }
        let mut items: Vec<_> = agg.into_iter().collect();
        items.sort();

        // The corpus must not be vacuous. Asserted on the NATIVE counts, before
        // anything is printed.
        if !correlation_is_observable(&items, n_q, target) {
            println!(
                "R {i} VACUOUS_CORPUS target={target} counts={}",
                items
                    .iter()
                    .map(|(k, v)| format!("{k}:{v}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            continue;
        }

        // One self-contained record: the exported text, and the distribution
        // Aria produced from the in-memory circuit.
        println!("R {i} SHOTS {SHOTS} TARGET {target}");
        println!("QASM_BEGIN");
        print!("{qasm}");
        println!("QASM_END");
        println!(
            "COUNTS {}",
            items
                .iter()
                .map(|(k, v)| format!("{k}:{v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    println!("#END");
}
