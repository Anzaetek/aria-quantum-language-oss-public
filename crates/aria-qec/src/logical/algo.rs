//! Algorithms end-to-end on the logical layer: QFT, QPE, Grover.
//!
//! Two altitudes, per the plan:
//! * **Physical-encoded** — a logical algorithm is compiled to a physical
//!   circuit under a code and run so the *logical observables* reproduce the
//!   ideal algorithm. 2-qubit Grover is fully Clifford (H, X, CZ), so it runs
//!   exactly on Pauli-propagation / statevector on encoded Steane qubits — the
//!   faithful demonstration.
//! * **Logical-channel** — the ideal K-qubit logical unitary is simulated
//!   directly (statevector), and the extracted effective logical noise is
//!   injected as a readout bit-flip channel. This is the cheap scale-up path for
//!   QFT / QPE, whose controlled-rotation gates are not transversal.

use std::collections::HashMap;
use std::f64::consts::PI;

use aria_core::ast::nodes::Circuit;
use aria_core::ast::CircuitBuilder;
use crate::ecc::run::{to_omega_core_ir, SimBackend};

use super::compile::LogicalCircuit;

use omega_core::executor::{ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;

/// 2-qubit Grover (one iteration) marking `marked` ∈ 0..4 — pure Clifford
/// (H, X, CZ), so it succeeds with certainty and runs exactly on encoded qubits.
/// Patch 0 is bit 0 (LSB), patch 1 is bit 1.
pub fn logical_grover2(marked: u8) -> LogicalCircuit {
    let m0 = marked & 1;
    let m1 = (marked >> 1) & 1;
    let mut lc = LogicalCircuit::new(2);
    lc.prep_zero(0).prep_zero(1);
    // Uniform superposition.
    lc.h(0).h(1);
    // Oracle: phase-flip |marked⟩. Conjugate CZ by X on the qubits whose marked
    // bit is 0, so the flipped state is |marked⟩.
    if m0 == 0 {
        lc.x(0);
    }
    if m1 == 0 {
        lc.x(1);
    }
    lc.cz(0, 1);
    if m0 == 0 {
        lc.x(0);
    }
    if m1 == 0 {
        lc.x(1);
    }
    // Diffuser: H X·CZ·X H.
    lc.h(0).h(1).x(0).x(1).cz(0, 1).x(0).x(1).h(0).h(1);
    lc
}

/// Exact statevector probability distribution over the `n` qubits of `circuit`
/// (keys are integer basis states, qubit 0 = LSB).
pub fn statevector_distribution(circuit: &Circuit, _n: usize) -> HashMap<u64, f64> {
    let ir = to_omega_core_ir(circuit);
    let be = SimBackend::Statevector.backend();
    let cfg = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    match be.execute(&ir, &ParameterBinding::new(), &cfg) {
        Ok(ExecResult::Statevector(sv)) => {
            let mut m = HashMap::new();
            for (idx, amp) in sv.iter().enumerate() {
                let p = amp.norm_sqr();
                if p > 1e-12 {
                    m.insert(idx as u64, p);
                }
            }
            m
        }
        other => panic!("expected statevector, got {other:?}"),
    }
}

/// Marginalize a distribution over `n` qubits down to its low `k` bits.
pub fn marginal_low_bits(dist: &HashMap<u64, f64>, k: usize) -> HashMap<u64, f64> {
    let mask = (1u64 << k) - 1;
    let mut out: HashMap<u64, f64> = HashMap::new();
    for (&x, &p) in dist {
        *out.entry(x & mask).or_insert(0.0) += p;
    }
    out
}

/// Ideal QFT output distribution on input basis state `input` over `n` qubits.
pub fn qft_distribution(n: usize, input: u64) -> HashMap<u64, f64> {
    let mut b = CircuitBuilder::new("qft", n, 0);
    for i in 0..n {
        if (input >> i) & 1 == 1 {
            b.x(i);
        }
    }
    let qs: Vec<usize> = (0..n).collect();
    b.qft(&qs);
    statevector_distribution(&b.build(), n)
}

/// QFT followed by inverse-QFT — must return the input basis state exactly.
pub fn qft_roundtrip_distribution(n: usize, input: u64) -> HashMap<u64, f64> {
    let mut b = CircuitBuilder::new("qft_rt", n, 0);
    for i in 0..n {
        if (input >> i) & 1 == 1 {
            b.x(i);
        }
    }
    let qs: Vec<usize> = (0..n).collect();
    b.qft(&qs).inverse_qft(&qs);
    statevector_distribution(&b.build(), n)
}

/// Inline inverse-QFT on `qs` (qs[0] = LSB) with the convention that pairs with
/// textbook QPE: input `Σ_y e^{2πi φ y}|y>` ↦ `|round(φ·2^m)>`, read directly as
/// `Σ q_j 2^j`. Self-contained so QPE does not depend on the builder's
/// (swap-carrying) QFT convention.
/// Genuine controlled-phase `diag(1,1,1,e^{iλ})`. In aria-core the builder's `cp`
/// lowers to `CU3(0,0,λ)` = controlled-`diag(1,e^{iλ})`, i.e. already a *true*
/// controlled-phase (unlike the `quantum` toolkit, whose `cp` lowers to CRZ and
/// needs a `P(λ/2)` correction). So this is just `cp`.
fn cphase(b: &mut CircuitBuilder, c: usize, t: usize, lam: f64) {
    b.cp(c, t, lam);
}

/// Standard inverse-QFT (true controlled-phase, qubit0 = LSB, with the leading
/// bit-reversal swap) — the exact dagger of the textbook QFT, so it maps the QPE
/// kickback `Σ_y e^{2πi φ y}|y>` to `|round(φ·2^m)>` read as `Σ q_k 2^k`.
fn inverse_qft_inline(b: &mut CircuitBuilder, qs: &[usize]) {
    // Nielsen–Chuang inverse QFT with a genuine controlled-phase: bit-reversal
    // swap, then for each qubit (ascending) the controlled rotations from the
    // already-processed lower qubits, then its Hadamard.
    let n = qs.len();
    for i in 0..n / 2 {
        b.swap(qs[i], qs[n - 1 - i]);
    }
    for j in 0..n {
        for l in 0..j {
            cphase(b, qs[l], qs[j], -PI / (1u64 << (j - l)) as f64);
        }
        b.h(qs[j]);
    }
}

/// QPE counting-register distribution estimating the phase of `P(2πφ)` with
/// `m` counting qubits (the target eigenstate is |1⟩). Counting qubit `k`
/// applies `U^{2^k}`; the inline inverse-QFT then places `round(φ·2^m)` in the
/// register read as `Σ q_k 2^k`.
pub fn qpe_distribution(m: usize, phase: f64) -> HashMap<u64, f64> {
    let n = m + 1;
    let mut b = CircuitBuilder::new("qpe", n, 0);
    b.x(m); // target |1⟩ (eigenstate of the phase gate)
    for k in 0..m {
        b.h(k);
    }
    for k in 0..m {
        let reps = (1u64 << k) as f64;
        cphase(&mut b, k, m, 2.0 * PI * phase * reps);
    }
    let counting: Vec<usize> = (0..m).collect();
    inverse_qft_inline(&mut b, &counting);
    let full = statevector_distribution(&b.build(), n);
    marginal_low_bits(&full, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical::compile::compile_physical;
    use crate::logical::metrics::{
        grover_success_prob, qpe_phase_error, readout_bitflip, tvd,
    };
    use crate::logical::run::logical_z_expectation;
    use crate::logical::transversal::SteaneTransversal;

    #[test]
    fn encoded_grover2_finds_every_marked_state() {
        // Physical-encoded, Clifford ⇒ exact. For marked = |m1 m0⟩, the logical
        // qubits end in that computational state: ⟨Z̄_i⟩ = (-1)^{m_i}.
        let code = SteaneTransversal::new();
        for marked in 0u8..4 {
            let lc = logical_grover2(marked);
            let prog = compile_physical(&lc, &code);
            let z0 = logical_z_expectation(&prog, 0, SimBackend::PauliProp).unwrap();
            let z1 = logical_z_expectation(&prog, 1, SimBackend::PauliProp).unwrap();
            let want0 = if marked & 1 == 1 { -1.0 } else { 1.0 };
            let want1 = if (marked >> 1) & 1 == 1 { -1.0 } else { 1.0 };
            assert!((z0 - want0).abs() < 1e-9, "marked={marked} ⟨Z̄_0⟩={z0}");
            assert!((z1 - want1).abs() < 1e-9, "marked={marked} ⟨Z̄_1⟩={z1}");
        }
    }

    #[test]
    fn encoded_grover_backends_agree() {
        let code = SteaneTransversal::new();
        let prog = compile_physical(&logical_grover2(0b10), &code);
        for patch in 0..2 {
            let pp = logical_z_expectation(&prog, patch, SimBackend::PauliProp).unwrap();
            let sv = logical_z_expectation(&prog, patch, SimBackend::Statevector).unwrap();
            assert!((pp - sv).abs() < 1e-6, "patch {patch}: {pp} vs {sv}");
        }
    }

    #[test]
    fn qft_of_zero_is_uniform() {
        let n = 4;
        let dist = qft_distribution(n, 0);
        let unit = 1.0 / (1u64 << n) as f64;
        for v in 0u64..(1 << n) {
            assert!((dist.get(&v).copied().unwrap_or(0.0) - unit).abs() < 1e-9);
        }
    }

    #[test]
    fn qft_roundtrip_recovers_input() {
        let n = 4;
        for input in [0u64, 5, 11, 15] {
            let dist = qft_roundtrip_distribution(n, input);
            assert!(
                (grover_success_prob(&dist, input) - 1.0).abs() < 1e-9,
                "input {input} not recovered: {dist:?}"
            );
        }
    }

    #[test]
    fn qpe_recovers_exact_phase() {
        // φ = k/2^m lands exactly on counting value k.
        // φ = j/2^m lands exactly on counting value j (clean delta), for every
        // exact phase at m = 1, 2, 3 counting qubits.
        for m in 1..=3usize {
            for j in 0..(1u64 << m) {
                let phase = j as f64 / (1u64 << m) as f64;
                let dist = qpe_distribution(m, phase);
                let err = qpe_phase_error(&dist, m, phase);
                assert!(err < 1e-9, "m={m} φ={phase}: err {err}, dist {dist:?}");
                assert!(
                    dist.get(&j).copied().unwrap_or(0.0) > 0.999,
                    "m={m} j={j}: not a clean delta: {dist:?}"
                );
            }
        }
    }

    #[test]
    fn logical_channel_noise_degrades_gracefully() {
        // Logical-channel altitude: inject an effective readout bit-flip b into
        // a peaked (QFT∘QFT⁻¹ = delta) distribution. TVD(0) = 0 and grows
        // monotonically with b. (A bare QFT output is uniform, so it is
        // invariant under bit-flip — the round-trip delta is the right probe.)
        let dist = qft_roundtrip_distribution(3, 5);
        let t0 = tvd(&dist, &readout_bitflip(&dist, 3, 0.0));
        let t1 = tvd(&dist, &readout_bitflip(&dist, 3, 0.02));
        let t2 = tvd(&dist, &readout_bitflip(&dist, 3, 0.08));
        assert!(t0 < 1e-12, "TVD at b=0 = {t0}");
        assert!(t1 < t2, "TVD should grow with noise: {t1} !< {t2}");
    }
}
