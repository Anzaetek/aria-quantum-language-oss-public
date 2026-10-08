// SPDX-License-Identifier: Apache-2.0
//! **The MPS truncation certificate is checked against values it did not
//! compute.**
//!
//! Third crate under the A2 gate ("assertions that cannot fail"), after
//! `omega-backend-majoranaprop` and `omega-backend-pauliprop`. The shape of
//! the hole was the same each time: the certificate (`discarded_weight`,
//! `fidelity_estimate`, `max_bond_reached`) is computed, accumulated and
//! gated, and every test that reads it asserts a predicate the arithmetic
//! cannot fail — `max_bond_reached <= χ`, `discarded_weight < 1e-12` on an
//! exact run, "the estimate never exceeds the truth". A `record_stats` emptied
//! to `()`, a `+=` flipped to `-=`, a `1 − ε` flipped to `1 + ε`, a
//! `> ceiling` flipped to `>= ceiling`: all green under those predicates.
//!
//! Here every number comes from a closed form the code never sees:
//!
//! * `ry(2θ)` on qubit 0 then `CX(0,1)` takes |00⟩ to `cos θ|00⟩ + sin θ|11⟩`,
//!   whose Schmidt values are exactly `{cos θ, sin θ}`. Dropping the smaller
//!   one costs weight `min²` and leaves fidelity `max²` at bond 1. A second
//!   `ry(2φ)`+`CX` on the surviving product branch has spectrum
//!   `A·{cos φ, sin φ}` on a block of norm² `A² ≠ 1`, which is what separates
//!   "divide by the block norm" from "multiply by it".
//! * The ceiling is straddled from both sides with the run's own certificate:
//!   a ceiling EQUAL to it accepts, one a hair below refuses — on both
//!   backends, because the noisy one grew its gate later and separately.
//! * The kernel's "free unitarity certificate" (`kept + dropped = ‖Θ'‖²`) is
//!   a `debug_assert!` no test had ever made fire; a kernel that lies about
//!   its dropped weight must trip it.

use num_complex::Complex64;
use omega_backend_mps::gates;
use omega_backend_mps::mps::Mps;
use omega_backend_mps::svd::{truncated_svd_flat, SvdResultFlat};
use omega_backend_mps::{MpsBackend, NoisyMpsBackend, DEFAULT_MAX_DISCARDED_WEIGHT};
use omega_core::circuit::*;
use omega_core::error::OmegaError;
use omega_core::executor::*;
use omega_core::noise::NoiseModel;
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

const TOL: f64 = 1e-9;

fn close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= TOL,
        "{what}: got {got:.15}, closed form says {want:.15} (Δ = {:.3e})",
        got - want
    );
}

/// `ry(2θ)` on qubit 0, then `CX(0,1)`. From |00⟩: `cos θ|00⟩ ± sin θ|11⟩`,
/// Schmidt values `{cos θ, sin θ}`. From the other product branch |11⟩ the
/// same pair gives `∓sin θ|01⟩ + cos θ|10⟩` — still `{cos θ, sin θ}` times the
/// branch amplitude. The sign convention of `ry` changes no modulus.
fn entangle(m: &mut Mps, theta: f64) {
    m.apply_1q(0, &gates::ry(2.0 * theta));
    m.apply_2q(0, &gates::cx());
}

// ---------------------------------------------------------------------------
// 1. The certificate against a known Schmidt spectrum (Mps, adaptive path)
// ---------------------------------------------------------------------------

/// θ = 0.3: spectrum {0.9553, 0.2955}, ratio 0.309. ε = 0.5 drops the small
/// one; the certificate must say `sin²θ` was dropped and `cos²θ` kept. On a
/// 2-site chain the block IS the state, so these are exact values, not the
/// "estimate" the docs hedge on for longer chains. (The state's own norm is
/// not a usable cross-check: `to_statevector` renormalises by design.)
#[test]
fn adaptive_drop_reports_exactly_the_dropped_schmidt_weight() {
    let th = 0.3_f64;
    let mut m = Mps::zero_state(2, 2);
    m.set_adaptive_eps(0.5);
    entangle(&mut m, th);
    close(
        m.discarded_weight,
        th.sin().powi(2),
        "discarded_weight after one drop",
    );
    close(
        m.fidelity_estimate,
        th.cos().powi(2),
        "fidelity_estimate after one drop",
    );
    assert_eq!(m.max_bond_reached, 1, "one Schmidt value survives");
}

/// Same θ, ε = 0.2 (< 0.309): both survive. The certificate must say nothing
/// was dropped — with bond 2, which is what tells this apart from an
/// implementation that drops everything and calls it exact.
#[test]
fn adaptive_keep_reports_nothing_dropped_and_bond_two() {
    let th = 0.3_f64;
    let mut m = Mps::zero_state(2, 2);
    m.set_adaptive_eps(0.2);
    entangle(&mut m, th);
    assert!(
        m.discarded_weight.abs() <= 1e-15,
        "both σ kept, nothing dropped; got {}",
        m.discarded_weight
    );
    assert!((m.fidelity_estimate - 1.0).abs() <= 1e-15);
    assert_eq!(m.max_bond_reached, 2, "both Schmidt values survive");
}

/// Two drops in a row. After the first, the block norm² is `cos²θ ≠ 1`, so the
/// second relative weight is `sin²φ` only if the kernel's absolute dropped
/// weight is DIVIDED by the block norm. Multiplying instead gives
/// `cos⁴θ·sin²φ` — 17% off here, and invisible on a normalized first split.
#[test]
fn a_second_drop_is_measured_relative_to_the_shrunk_block_norm() {
    let (th, ph) = (0.3_f64, 0.4_f64); // tan φ = 0.423 < 0.5: dropped too
    let mut m = Mps::zero_state(2, 2);
    m.set_adaptive_eps(0.5);
    entangle(&mut m, th);
    entangle(&mut m, ph);
    close(
        m.discarded_weight,
        th.sin().powi(2) + ph.sin().powi(2),
        "Σ relative dropped weight over two splits",
    );
    close(
        m.fidelity_estimate,
        th.cos().powi(2) * ph.cos().powi(2),
        "Π(1 − εᵢ) over two splits",
    );
    assert_eq!(m.max_bond_reached, 1);
}

/// The threshold is `ε · σ_max`, relative to the spectrum — not `ε / σ_max`.
/// θ = 1.0 under ε = 0.7 drops cos(1) (0.540 < 0.7·0.841) and leaves a branch
/// of amplitude sin(1) = 0.841. Then φ = 0.576 under ε = 0.5: σ = 0.841·{0.839,
/// 0.545} = {0.706, 0.458}; the relative threshold 0.5·0.706 = 0.353 KEEPS
/// 0.458, while `ε / σ_max` = 0.708 would drop it. The two thresholds agree
/// whenever σ_max ≈ 1, which is why the shrunk branch is needed to tell them
/// apart.
#[test]
fn adaptive_threshold_is_relative_to_the_largest_singular_value() {
    let (th, ph) = (1.0_f64, 0.576_f64);
    let mut m = Mps::zero_state(2, 2);
    m.set_adaptive_eps(0.7);
    entangle(&mut m, th);
    close(m.discarded_weight, th.cos().powi(2), "first drop = cos²(1)");
    assert_eq!(m.max_bond_reached, 1);
    m.set_adaptive_eps(0.5);
    entangle(&mut m, ph);
    close(
        m.discarded_weight,
        th.cos().powi(2),
        "second split keeps both σ: nothing more dropped",
    );
    close(
        m.fidelity_estimate,
        th.sin().powi(2),
        "fidelity unchanged by a kept split",
    );
    assert_eq!(m.max_bond_reached, 2, "the second split keeps both σ");
}

/// The non-adaptive path: χ = 1 makes the KERNEL truncate, and the same closed
/// form must come back through `svd.discarded_weight / total`. Two splits, so
/// the division is again measured on a block of norm² `cos²θ`.
#[test]
fn fixed_rank_truncation_reports_the_same_closed_form() {
    let (th, ph) = (0.3_f64, 0.4_f64);
    let mut m = Mps::zero_state(2, 1);
    entangle(&mut m, th);
    close(m.discarded_weight, th.sin().powi(2), "χ=1 drops sin²θ");
    close(m.fidelity_estimate, th.cos().powi(2), "χ=1 keeps cos²θ");
    assert_eq!(m.max_bond_reached, 1);
    entangle(&mut m, ph);
    close(
        m.discarded_weight,
        th.sin().powi(2) + ph.sin().powi(2),
        "χ=1, two splits: Σ relative dropped weight",
    );
    close(
        m.fidelity_estimate,
        th.cos().powi(2) * ph.cos().powi(2),
        "χ=1, two splits: Π(1 − εᵢ)",
    );
}

// ---------------------------------------------------------------------------
// 2. The backend: worst-case accumulation and the ceiling, from both sides
// ---------------------------------------------------------------------------

fn g(kind: GateKind, qubits: &[u32]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: smallvec![],
        classical_bit: None,
        condition: None,
    }
}

fn cfg() -> ExecConfig {
    ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    }
}

/// The refusal itself, or a panic naming the gate that should have refused.
fn refused<T>(r: std::result::Result<T, OmegaError>, why: &str) -> OmegaError {
    match r {
        Err(e) => e,
        Ok(_) => panic!("{why} — but the run was accepted"),
    }
}

/// |GHZ_n⟩: Schmidt rank exactly 2 at every cut. χ = 2 is exact, and the peak
/// bond is exactly 2 — not "≤ 2".
fn ghz(n: u32) -> CircuitIR {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    c.add_op(g(GateKind::H, &[0]));
    for q in 0..n - 1 {
        c.add_op(g(GateKind::CX, &[q, q + 1]));
    }
    c
}

/// Random-angle `Ry` between `CX` bricks: a non-flat spectrum whose rank grows
/// past 2 within two layers, so χ = 2 must truncate.
fn brickwork(n: u32, layers: usize, seed: u64) -> CircuitIR {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    let mut x = seed;
    let mut angle = || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((x >> 33) as f64 / (1u64 << 31) as f64) * std::f64::consts::TAU
    };
    for layer in 0..layers {
        for q in 0..n {
            let mut op = g(GateKind::Ry, &[q]);
            op.params = smallvec![ParamExpr::Concrete(angle())];
            c.add_op(op);
        }
        let mut q = (layer % 2) as u32;
        while q + 1 < n {
            c.add_op(g(GateKind::CX, &[q, q + 1]));
            q += 2;
        }
    }
    c
}

#[test]
fn an_exact_run_reports_its_exact_peak_bond() {
    let b = MpsBackend::new(2);
    b.execute(&ghz(5), &ParameterBinding::new(), &cfg())
        .expect("GHZ at χ=2 is exact");
    let st = b.last_run_stats();
    assert_eq!(
        st.max_bond_reached, 2,
        "GHZ has Schmidt rank 2 at every cut"
    );
    assert!(
        st.discarded_weight <= 1e-15,
        "exact run dropped {}",
        st.discarded_weight
    );
    assert!((st.fidelity_estimate - 1.0).abs() <= 1e-15);
}

/// A starved run: the certificate must be positive, the fidelity strictly
/// inside (0, 1), the peak bond exactly the ceiling, and the two numbers must
/// satisfy the product inequality `Π(1 − εᵢ) ≥ 1 − Σεᵢ` that holds for any
/// εᵢ ∈ [0, 1] — a relation between the two fields that a sign flip in
/// either one breaks. Then the same backend, run on an exact circuit, must
/// report THAT run: the accumulator is per-`execute`, not per-backend.
#[test]
fn a_starved_run_reports_a_positive_certificate_and_resets_per_execute() {
    let b = MpsBackend::new(2).with_max_discarded_weight(f64::INFINITY);
    b.execute(&brickwork(8, 6, 0xA2), &ParameterBinding::new(), &cfg())
        .expect("ceiling lifted");
    let st = b.last_run_stats();
    assert!(
        st.discarded_weight > DEFAULT_MAX_DISCARDED_WEIGHT,
        "χ=2 on a 6-layer brickwork must truncate well past the default \
         ceiling; got {}",
        st.discarded_weight
    );
    assert!(
        st.fidelity_estimate > 0.0 && st.fidelity_estimate < 1.0,
        "fidelity of a truncating run lies strictly in (0, 1); got {}",
        st.fidelity_estimate
    );
    assert_eq!(
        st.max_bond_reached, 2,
        "a starved run saturates the ceiling"
    );
    assert!(
        st.fidelity_estimate >= 1.0 - st.discarded_weight - 1e-12,
        "Π(1 − εᵢ) = {} < 1 − Σεᵢ = {}: the two fields contradict each other",
        st.fidelity_estimate,
        1.0 - st.discarded_weight
    );
    // Per-execute, not per-backend.
    b.execute(&ghz(5), &ParameterBinding::new(), &cfg())
        .unwrap();
    let st2 = b.last_run_stats();
    assert_eq!(st2.max_bond_reached, 2);
    assert!(
        st2.discarded_weight <= 1e-15,
        "the exact run must not inherit the starved run's {}; got {}",
        st.discarded_weight,
        st2.discarded_weight
    );
}

/// The ceiling, straddled with the run's own certificate: equal accepts
/// (`>`, not `>=`), a hair below refuses, and the default refuses.
#[test]
fn the_ceiling_is_straddled_by_the_certificate_itself() {
    let circuit = brickwork(8, 6, 0xA2);
    let open = MpsBackend::new(2).with_max_discarded_weight(f64::INFINITY);
    open.execute(&circuit, &ParameterBinding::new(), &cfg())
        .unwrap();
    let dw = open.last_run_stats().discarded_weight;
    assert!(dw > DEFAULT_MAX_DISCARDED_WEIGHT);

    MpsBackend::new(2)
        .with_max_discarded_weight(dw)
        .execute(&circuit, &ParameterBinding::new(), &cfg())
        .expect("a ceiling EQUAL to the certificate accepts: the gate is `>`");

    let err = refused(
        MpsBackend::new(2)
            .with_max_discarded_weight(dw * (1.0 - 1e-12))
            .execute(&circuit, &ParameterBinding::new(), &cfg()),
        "a ceiling a hair below the certificate refuses",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("exceeds the ceiling"),
        "refusal must name the certificate and the ceiling; got {msg}"
    );

    // The default ceiling is crossed early, so it is the MID-CIRCUIT abort
    // that refuses — a second gate with its own comparison and its own
    // message. (The hair-below ceiling above is crossed only by the last
    // split, which the pre-op abort never sees; that is why it reached
    // `check_truncation` and said "exceeds".)
    let err = refused(
        MpsBackend::new(2).execute(&circuit, &ParameterBinding::new(), &cfg()),
        "the default ceiling refuses this run",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("exceeded the ceiling") && msg.contains("partway"),
        "the default ceiling must be enforced by the mid-circuit abort, which \
         says so; got {msg}"
    );
}

/// The noisy backend grew its gate separately, so it is straddled separately.
/// Under a zero noise model every trajectory is the noiseless circuit, so its
/// worst-over-trajectories certificate must equal the noiseless one exactly.
#[test]
fn the_noisy_backend_ceiling_is_straddled_too() {
    let circuit = brickwork(8, 6, 0xA2);
    let noiseless = MpsBackend::new(2).with_max_discarded_weight(f64::INFINITY);
    noiseless
        .execute(&circuit, &ParameterBinding::new(), &cfg())
        .unwrap();
    let dw = noiseless.last_run_stats().discarded_weight;

    let shots = ExecConfig {
        shots: Some(8),
        seed: Some(11),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let open = NoisyMpsBackend::with_model(2, NoiseModel::default())
        .with_max_discarded_weight(f64::INFINITY);
    open.execute(&circuit, &ParameterBinding::new(), &shots)
        .expect("ceiling lifted");
    let st = open.last_run_stats();
    assert!(
        (st.discarded_weight - dw).abs() <= 1e-12 * dw,
        "zero noise: the worst trajectory IS the noiseless run ({} vs {})",
        st.discarded_weight,
        dw
    );
    assert_eq!(st.max_bond_reached, 2);

    NoisyMpsBackend::with_model(2, NoiseModel::default())
        .with_max_discarded_weight(st.discarded_weight)
        .execute(&circuit, &ParameterBinding::new(), &shots)
        .expect("a ceiling EQUAL to the certificate accepts");

    let err = refused(
        NoisyMpsBackend::with_model(2, NoiseModel::default())
            .with_max_discarded_weight(st.discarded_weight * (1.0 - 1e-12))
            .execute(&circuit, &ParameterBinding::new(), &shots),
        "a ceiling a hair below refuses",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Noise does not excuse it"),
        "the noisy refusal must say truncation is not a modelled channel; got {msg}"
    );

    let err = refused(
        NoisyMpsBackend::with_model(2, NoiseModel::default()).execute(
            &circuit,
            &ParameterBinding::new(),
            &shots,
        ),
        "the default ceiling refuses this run under noise too",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("exceeded the ceiling") && msg.contains("partway"),
        "under noise the default ceiling is enforced by the trajectory's \
         mid-circuit abort; got {msg}"
    );
}

// ---------------------------------------------------------------------------
// 3. The free unitarity certificate must be able to fire
// ---------------------------------------------------------------------------

/// The real kernel, reporting half the block as dropped on top of the truth.
/// U, S, Vt are correct; only the claim is wrong — exactly what the
/// `kept + dropped = ‖Θ'‖²` guard exists to catch.
fn kernel_that_lies_about_dropped_weight(
    m: &[Complex64],
    rows: usize,
    cols: usize,
    stride: usize,
    max_rank: usize,
    threshold: f64,
) -> SvdResultFlat {
    let mut r = truncated_svd_flat(m, rows, cols, stride, max_rank, threshold);
    r.discarded_weight += 0.5;
    r
}

/// Debug-only, like the guard itself: a `debug_assert!` no test had ever made
/// fire is a guard whose tolerance can be rewritten to anything.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "not norm-preserving")]
fn the_unitarity_guard_fires_on_a_kernel_that_lies() {
    let mut m = Mps::zero_state(2, 2);
    m.set_svd_fn(kernel_that_lies_about_dropped_weight);
    entangle(&mut m, 0.3);
}
