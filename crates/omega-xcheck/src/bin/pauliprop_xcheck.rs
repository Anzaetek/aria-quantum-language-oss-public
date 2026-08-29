// SPDX-License-Identifier: Apache-2.0
//! **PauliProp against Qiskit.** Emits random NON-Clifford circuits plus a Pauli
//! observable, prints this engine's expectation for each, and lets
//! `tools/qiskit_xcheck/compare_pauliprop.py` rebuild the same circuits and
//! compute ⟨ψ|O|ψ⟩ with Qiskit's `Statevector`.
//!
//! **Why this exists: PauliProp had no independent gate at all.**
//! `omega-xcheck`'s main harness drives `StatevectorBackend` (and stabilizer,
//! and Metal) — all statevector-shaped, all compared on probabilities. PauliProp
//! returns expectations and never entered it. Its correctness rested on:
//!
//! * `pauliprop_backend_matches_sim_expectations` — our own CPU statevector,
//! * the `ppvm` anchor — a *same-algorithm* implementation,
//! * GPU-vs-CPU parity — two of our own paths.
//!
//! Every one of those shares this project's conventions. Two implementations
//! that share a convention agree on a shared mistake, and this repository has
//! already shipped a defect that every internal agreement gate missed (the
//! `Reset` channel, wrong in three backends in three different bases, each pair
//! coinciding in whatever basis was being checked).
//!
//! **The corpus is deliberately non-Clifford.** The main harness emits
//! `h/s/sdg/x/z/cx`, which PauliProp handles by Clifford conjugation and which
//! never touches `branch` — the tree-expansion step that is the entire engine.
//! A Clifford-only cross-check of PauliProp would be green and vacuous. So this
//! emits `rz/rx/ry/t/tdg` alongside the Cliffords, with angles that are not
//! multiples of π/2.
//!
//! **Truncation is off.** `max_freq`/`coeff_min` are approximations that report
//! `dropped_mass`; with them on, disagreement with Qiskit is *expected* and
//! bounded rather than a defect, which is a different test. This one runs the
//! engine exact, so any disagreement is a bug.

use omega_backend_pauliprop::PauliPropBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

fn op(g: GateKind, qs: &[u32], ps: &[f64]) -> GateOp {
    GateOp {
        gate: g,
        qubits: qs.iter().map(|&q| Qubit(q)).collect(),
        params: ps.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
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

fn main() {
    let n_circ: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    let pp = PauliPropBackend::new();
    let pb = ParameterBinding::new();
    let mut seed = 0xC0FFEEu64;

    println!("#BEGIN");
    let mut emitted = 0usize;
    for _ in 0..n_circ {
        let n = 2 + (rnd(&mut seed) % 4) as u32; // 2..5 qubits
        let depth = 4 + (rnd(&mut seed) % 8) as usize;
        let mut c = CircuitIR::new(n, CircuitType::GateBased);
        let mut desc: Vec<String> = Vec::new();

        for _ in 0..depth {
            let q = (rnd(&mut seed) % n as u64) as u32;
            // Angles deliberately off the π/2 lattice: a multiple of π/2 makes
            // the rotation Clifford, and a corpus that accidentally collapsed to
            // Clifford would never exercise `branch`.
            let theta = 0.1 + (rnd(&mut seed) % 300) as f64 / 100.0;
            match rnd(&mut seed) % 8 {
                0 => {
                    c.ops.push(op(GateKind::H, &[q], &[]));
                    desc.push(format!("h:{q}"));
                }
                1 => {
                    c.ops.push(op(GateKind::S, &[q], &[]));
                    desc.push(format!("s:{q}"));
                }
                2 => {
                    c.ops.push(op(GateKind::Rz, &[q], &[theta]));
                    desc.push(format!("rz:{q}:{theta}"));
                }
                3 => {
                    c.ops.push(op(GateKind::Rx, &[q], &[theta]));
                    desc.push(format!("rx:{q}:{theta}"));
                }
                4 => {
                    c.ops.push(op(GateKind::Ry, &[q], &[theta]));
                    desc.push(format!("ry:{q}:{theta}"));
                }
                5 => {
                    c.ops.push(op(GateKind::T, &[q], &[]));
                    desc.push(format!("t:{q}"));
                }
                _ => {
                    let b = (rnd(&mut seed) % n as u64) as u32;
                    if q == b {
                        c.ops.push(op(GateKind::H, &[q], &[]));
                        desc.push(format!("h:{q}"));
                    } else {
                        c.ops.push(op(GateKind::CX, &[q, b], &[]));
                        desc.push(format!("cx:{q},{b}"));
                    }
                }
            }
        }

        // A multi-term observable, so the comparison is not one lucky ⟨Z⟩.
        let mut terms: Vec<(f64, Vec<(u32, PauliOp)>)> = Vec::new();
        let mut obs_desc: Vec<String> = Vec::new();
        let n_terms = 1 + (rnd(&mut seed) % 3) as usize;
        for _ in 0..n_terms {
            let coeff = 0.25 + (rnd(&mut seed) % 200) as f64 / 100.0;
            let mut factors = Vec::new();
            let mut label = vec!['I'; n as usize];
            for qi in 0..n {
                match rnd(&mut seed) % 4 {
                    0 => {
                        factors.push((qi, PauliOp::X));
                        label[qi as usize] = 'X';
                    }
                    1 => {
                        factors.push((qi, PauliOp::Y));
                        label[qi as usize] = 'Y';
                    }
                    2 => {
                        factors.push((qi, PauliOp::Z));
                        label[qi as usize] = 'Z';
                    }
                    _ => {}
                }
            }
            // Qiskit's Pauli label is little-endian: leftmost char is the
            // HIGHEST qubit index. Reverse so the two sides mean the same
            // operator — the single most likely place for this whole
            // cross-check to agree on a shared misreading.
            let s: String = label.iter().rev().collect();
            obs_desc.push(format!("{coeff}*{s}"));
            terms.push((coeff, factors));
        }

        let obs = Observable { terms };
        let Ok(val) = pp.expectation(&c, &pb, &obs) else {
            continue; // unsupported gate for this engine — skip, do not fake
        };

        println!(
            "{} {} | {} | {:.17e}",
            n,
            desc.join(" "),
            obs_desc.join(" "),
            val
        );
        emitted += 1;
    }
    println!("#END {emitted}");
}
