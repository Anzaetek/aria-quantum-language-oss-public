// SPDX-License-Identifier: Apache-2.0
//! **The DV half of `PLAN-OPTICQASM-INTEGRITY.md` O5: one OPTICQASM source,
//! two independent backends, the same numbers.**
//!
//! The CV half already exists (`omega-backend-cv/tests/opticqasm_cv_xcheck.rs`,
//! 17/17 piquasso fixture cases). This is its discrete-variable counterpart:
//! text produced by `aria_core::ast::opticqasm::to_opticqasm`, executed by
//! **Perceval 1.2.4** and by **`omega-backend-photonics`**, compared as
//! distributions.
//!
//! ```text
//!   aria-core Circuit ──> to_opticqasm ──┬──> perceval_runner.py ──> counts
//!                                        │                             │
//!                                        └──> lower_to_ir ──> photonics ┴──> L2
//! ```
//!
//! # Why this is a different claim from the tests around it
//!
//! * `perceval_opticqasm_hom_bunching` drives a **hand-written** source and
//!   checks a physics signature. It cannot see an emitter defect, because the
//!   emitter is not on its path.
//! * `opticqasm_readable_by_parser.rs` proves our text survives our own reader.
//!   Both sides could share a misconception.
//! * This proves the text means the same thing to Perceval as to us. A wrong
//!   beamsplitter convention passes both of the above and fails here.
//!
//! # Polarization is included, and only recently could be
//!
//! `hwp`/`pbs` became emittable in O4. Before that the emitter had zero
//! references to either, so a polarization circuit could not be put on this path
//! at all — the conventions pinned in `tests/test_perceval_conventions.py`
//! (notably that `pbs` **swaps H and transmits V**, the opposite of the common
//! phrasing) had never been checked against a circuit we ourselves wrote.
//!
//! Skips when the perceval venv is absent, so `cargo test` stays Python-free.
//! The skip is reported, never silent.

#![cfg(feature = "bridge-perceval")]

use aria_core::ast::nodes::*;
use aria_core::ast::opticqasm::to_opticqasm;
use omega_bridges::{run_opticqasm, Backend, BridgeError, Counts};
use omega_core::executor::{Backend as _, ExecConfig, ExecResult};
use omega_core::params::ParameterBinding;
use std::path::PathBuf;

const SHOTS: u32 = 8192;

fn runner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")
}

fn venv_python() -> PathBuf {
    runner_dir().join(".venv-perceval").join("bin").join("python")
}

fn force_runner_env() {
    std::env::set_var(
        "OMEGA_BRIDGE_PERCEVAL_CMD",
        runner_dir().join("omega-bridge-perceval-runner"),
    );
}

/// L2 distance between two normalised count distributions.
fn count_l2(a: &Counts, b: &Counts, total_a: u32, total_b: u32) -> f64 {
    let mut keys: std::collections::HashSet<&String> = a.keys().collect();
    keys.extend(b.keys());
    keys.iter()
        .map(|k| {
            let pa = (*a.get(*k).unwrap_or(&0) as f64) / total_a as f64;
            let pb = (*b.get(*k).unwrap_or(&0) as f64) / total_b as f64;
            (pa - pb).powi(2)
        })
        .sum::<f64>()
        .sqrt()
}

/// Statistical gate, not a magic number.
///
/// Both sides are sampled at `SHOTS`, so each bin's estimate carries a standard
/// error `sqrt(p(1-p)/n)`. Summing in quadrature over bins and taking `K = 6`
/// gives a bound that a correct pair of backends crosses with probability far
/// below any realistic flake rate, while a genuine convention error — which
/// moves probability mass by tens of percent — is nowhere near it. The same
/// construction as the N-way counts matrices.
fn l2_gate(a: &Counts, b: &Counts) -> f64 {
    const K: f64 = 6.0;
    let n = SHOTS as f64;
    let mut var = 0.0;
    let mut keys: std::collections::HashSet<&String> = a.keys().collect();
    keys.extend(b.keys());
    for k in keys {
        let pa = (*a.get(k).unwrap_or(&0) as f64) / n;
        let pb = (*b.get(k).unwrap_or(&0) as f64) / n;
        let p = 0.5 * (pa + pb);
        var += p * (1.0 - p) * (2.0 / n);
    }
    K * var.sqrt()
}

/// Run an OPTICQASM source on `omega-backend-photonics`.
fn photonics_counts(source: &str, input_fock: &[u32]) -> Counts {
    let ir = omega_parser::lower_to_ir(source)
        .unwrap_or_else(|e| panic!("our own OPTICQASM does not lower: {e}\n{source}"));
    assert_eq!(
        ir.num_qubits as usize,
        input_fock.len(),
        "the input Fock state must name every optical mode — if this trips on a \
         `pol` circuit, the marker was dropped and N spatial modes were read as N \
         optical ones instead of 2N:\n{source}"
    );
    let backend = omega_backend_photonics::sim::PhotonicsBackend::with_input(input_fock.to_vec());
    let cfg = ExecConfig {
        shots: Some(SHOTS),
        seed: Some(0xA71A),
        ..Default::default()
    };
    match backend
        .execute(&ir, &ParameterBinding::default(), &cfg)
        .unwrap_or_else(|e| panic!("photonics execute failed: {e:?}\n{source}"))
    {
        ExecResult::Counts(c) => c
            .into_iter()
            .map(|(k, v)| {
                // `decode_fock_string` renders `|0,2>`; the perceval runner
                // returns `0,2`. Comparing them raw made every bin disjoint and
                // the L2 came out at 9.0e-1 — which reads exactly like a
                // convention error and was not one: the underlying counts agreed
                // to within sampling noise (3665 vs 3664, 964 vs 953, 3563 vs
                // 3575). Normalised to the bare comma form, and
                // `compare` now asserts the two share bins at all so this
                // failure names its own cause next time.
                let key = omega_backend_photonics::sim::decode_fock_string(k, input_fock.len());
                (key.trim_matches(['|', '>']).to_string(), v as u32)
            })
            .collect(),
        other => panic!("expected Counts, got {other:?}"),
    }
}

/// Build a circuit, emit it, and compare the two backends on that text.
///
/// Returns `None` when the venv is absent, so the caller reports a skip rather
/// than passing quietly.
fn compare(label: &str, circuit: &Circuit, input_fock: &[u32]) -> Option<(f64, f64)> {
    let source = to_opticqasm(circuit)
        .unwrap_or_else(|e| panic!("{label}: circuit is not emittable: {e}"));

    let perceval = match run_opticqasm(
        Backend::Perceval,
        &source,
        SHOTS,
        Some(input_fock),
        None,
    ) {
        Ok(c) => c,
        Err(BridgeError::Unavailable(_, msg)) => {
            eprintln!("{label}: perceval unavailable ({msg})");
            return None;
        }
        Err(e) => panic!("{label}: perceval failed on OUR OPTICQASM: {e}\n{source}"),
    };
    let ours = photonics_counts(&source, input_fock);

    let p_total: u32 = perceval.values().sum();
    let o_total: u32 = ours.values().sum();
    assert_eq!(p_total, SHOTS, "{label}: perceval lost shots");
    assert_eq!(o_total, SHOTS, "{label}: photonics lost shots");
    // A comparison over an empty or single-bin distribution is vacuous: any two
    // deterministic backends agree on it. Every case here must actually spread.
    assert!(
        perceval.len() >= 2,
        "{label}: perceval produced {} bin(s) — a degenerate distribution cannot \
         distinguish a correct backend from a broken one:\n{source}",
        perceval.len()
    );

    // The two must actually share bins. Disjoint key sets produce a large L2
    // that reads like a physics disagreement and is a formatting bug — exactly
    // what happened on the first run here. Asserting the overlap makes the
    // failure name its own cause instead of accusing the backends.
    let shared = perceval.keys().filter(|k| ours.contains_key(*k)).count();
    assert!(
        shared > 0,
        "{label}: perceval and photonics share NO bins, so this is a key-format \
         mismatch rather than a disagreement.\n  perceval : {:?}\n  photonics: {:?}",
        perceval.keys().collect::<Vec<_>>(),
        ours.keys().collect::<Vec<_>>()
    );

    Some((count_l2(&perceval, &ours, p_total, o_total), l2_gate(&perceval, &ours)))
}

fn skip_if_no_venv(what: &str) -> bool {
    if !venv_python().exists() {
        eprintln!(
            "perceval venv missing at {} — skipping {what}. Build with \
             `make -C crates/omega-bridges/python perceval-venv`.",
            venv_python().display()
        );
        return true;
    }
    force_runner_env();
    false
}

/// A beam splitter and a phase shifter — the DV core.
///
/// The phase shifter is not decoration: a bare 50/50 BS with `|1,1⟩` in is
/// symmetric, so a convention error in the *phase* argument is invisible on it.
/// The `ps` breaks that symmetry.
#[test]
fn dv_core_agrees_with_perceval_on_our_own_emission() {
    if skip_if_no_venv("the DV core comparison") {
        return;
    }
    let mut c = Circuit::new("photonic");
    let m = c.qreg("q", 2);
    c.apply(
        GateDef::with_params(GateKind::PhaseShifter, vec![0.37]),
        vec![m[0].clone()],
    );
    c.apply(
        GateDef::with_params(GateKind::BeamSplitter, vec![0.61, 0.23]),
        vec![m[0].clone(), m[1].clone()],
    );

    let Some((l2, gate)) = compare("dv core", &c, &[1, 1]) else {
        return;
    };
    assert!(
        l2 <= gate,
        "our OPTICQASM computes different distributions on perceval and on \
         omega-backend-photonics: L2 = {l2:.4e} against a 6-sigma gate of {gate:.4e}"
    );
}

/// **Polarization**, newly possible: `hwp` and `pbs` on a `pol` register.
///
/// `pbs` swaps H between the two spatial modes and transmits V — Perceval's
/// convention, and the opposite of the usual "transmits H, reflects V"
/// phrasing. This is the first time that convention is checked against a
/// circuit this workspace *emitted* rather than one hand-written to match.
#[test]
fn polarization_agrees_with_perceval_on_our_own_emission() {
    if skip_if_no_venv("the polarization comparison") {
        return;
    }
    let mut c = Circuit::new("photonic");
    let m = c.qreg_polarized("q", 2);
    // **TWO half-wave plates, and the second one is load-bearing.**
    //
    // The `hwp` expansion contains a `PS(pi)` on the V sub-mode. With a single
    // plate on a product Fock input that phase is unobservable — a single-mode
    // phase ahead of a beam splitter factors out as a global phase, so deleting
    // it entirely left this test GREEN (measured, twice: once with `[1,0,0,0]`
    // and again with both H and V populated as `[1,1,0,0]`, which does not help
    // for a product state either).
    //
    // The first plate puts the photon in an H/V superposition; the second's
    // internal phase then acts on populated amplitude on both sides and the
    // interference reaches the counts. With this fixture, deleting the `PS(pi)`
    // fails the test.
    //
    // Generic angles: not 0, not pi/4, so neither the identity nor a clean
    // H<->V swap can stand in for a wrong `2*theta`.
    c.apply(
        GateDef::with_params(GateKind::HalfWavePlate, vec![0.37]),
        vec![m[0].clone()],
    );
    c.apply(
        GateDef::with_params(GateKind::HalfWavePlate, vec![0.21]),
        vec![m[0].clone()],
    );
    c.apply(
        GateDef::new(GateKind::PolarizingBeamSplitter),
        vec![m[0].clone(), m[1].clone()],
    );

    // 2 spatial modes -> 4 optical modes, indexed 2s+p with p=0 meaning H.
    // One H photon in spatial mode 0; the interference comes from the two
    // plates, not from the input state.
    let Some((l2, gate)) = compare("polarization", &c, &[1, 0, 0, 0]) else {
        return;
    };
    assert!(
        l2 <= gate,
        "our polarization OPTICQASM computes different distributions on perceval \
         and on omega-backend-photonics: L2 = {l2:.4e} against a 6-sigma gate of \
         {gate:.4e}. A `pbs` convention flip (swap H vs transmit H) lands here."
    );
}
