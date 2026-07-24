// SPDX-License-Identifier: Apache-2.0
//! Differential fuzzing across the omega backends (item 21e/21f).
//!
//! On *random* circuits, the exact dense (statevector) and tensor (MPS)
//! engines must agree on every single-qubit Pauli expectation, and the
//! statevector engine must preserve normalisation (the unitarity invariant).
//! Exercises the full `.aria` Circuit → omega IR lowering through the public
//! [`expectation`]/[`statevector`] helpers and the [`BackendSel`] plugin
//! contract. (Pauli propagation is not in `BackendSel`, so it is out of scope
//! here; the gate-model toolkit fuzzes the statevector/MPS/pauliprop trio.)

use std::collections::HashMap;

use aria_core::ast::{Circuit, CircuitBuilder};
use aria_runtime::{expectation, statevector, BackendSel};

/// Seeded LCG → uniform `[0,1)`.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

/// Build a random circuit on `n` qubits with `depth` gates over a universal
/// gate set (Clifford + rotations + T) — exact for statevector and MPS.
fn random_circuit(n: usize, depth: usize, s: &mut u64) -> Circuit {
    let mut b = CircuitBuilder::new("fuzz", n, 0);
    for _ in 0..depth {
        if n >= 2 && lcg(s) < 0.4 {
            let a = (lcg(s) * n as f64) as usize;
            let mut c = (lcg(s) * n as f64) as usize;
            if c == a {
                c = (a + 1) % n;
            }
            if lcg(s) < 0.5 {
                b.cx(a, c);
            } else {
                b.cz(a, c);
            }
        } else {
            let q = (lcg(s) * n as f64) as usize;
            match (lcg(s) * 8.0) as usize {
                0 => b.h(q),
                1 => b.s(q),
                2 => b.x(q),
                3 => b.y(q),
                4 => b.z(q),
                5 => b.rx(q, lcg(s) * std::f64::consts::TAU),
                6 => b.ry(q, lcg(s) * std::f64::consts::TAU),
                _ => b.rz(q, lcg(s) * std::f64::consts::TAU),
            };
        }
    }
    b.build()
}

#[test]
fn fuzz_statevector_vs_mps_agree() {
    // Property (21e): on random universal circuits, the exact dense and tensor
    // engines must agree on every single-qubit Z/X expectation, and |⟨P⟩| ≤ 1.
    let binds: HashMap<String, f64> = HashMap::new();
    let mut s = 0xC0FF_EE42u64;
    for _ in 0..120 {
        let n = 2 + (lcg(&mut s) * 3.0) as usize; // 2..=4
        let depth = 4 + (lcg(&mut s) * 16.0) as usize;
        let circ = random_circuit(n, depth, &mut s);
        for q in 0..n {
            for axis in ["Z", "X"] {
                let obs = format!("{axis}{q}");
                let sv = expectation(&circ, &obs, &binds, BackendSel::Sim).expect("sv expectation");
                let mps = expectation(&circ, &obs, &binds, BackendSel::Mps { chi: 64 })
                    .expect("mps expectation");
                assert!(sv.abs() <= 1.0 + 1e-9, "‖ψ‖ violated: ⟨{obs}⟩={sv}");
                assert!(
                    (sv - mps).abs() < 1e-6,
                    "sv/mps disagree on ⟨{obs}⟩: {sv} vs {mps} (n={n}, depth={depth})"
                );
            }
        }
    }
}

#[test]
fn fuzz_statevector_preserves_norm() {
    // Property (21f): a random circuit lowered through omega and run on the
    // statevector engine yields a normalised state — Σ|aᵢ|² = 1 — the numeric
    // witness that the lowering + gate application stayed unitary.
    let binds: HashMap<String, f64> = HashMap::new();
    let mut s = 0xBADC0DEu64;
    for _ in 0..150 {
        let n = 1 + (lcg(&mut s) * 4.0) as usize; // 1..=4
        let depth = 4 + (lcg(&mut s) * 16.0) as usize;
        let circ = random_circuit(n, depth, &mut s);
        let sv = statevector(&circ, &binds, BackendSel::Sim).expect("statevector");
        let norm2: f64 = sv.iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm2 - 1.0).abs() < 1e-9,
            "‖ψ‖²={norm2} (n={n}, depth={depth})"
        );
    }
}
