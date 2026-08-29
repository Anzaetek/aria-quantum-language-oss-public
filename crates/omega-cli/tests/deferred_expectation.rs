// SPDX-License-Identifier: Apache-2.0
//! **Every engine must agree on what `⟨O⟩` means for a circuit that measures —
//! or refuse. Never a third answer.**
//!
//! The rule itself lives in one place (`omega_core::defer_measure`), but there
//! are more than a dozen expectation entry points across nine backend crates
//! once `expectation_multi`, `expectation_batch` and the fused gradient paths are
//! counted. A rule that has to be *remembered* at that width holds in eleven of
//! twelve after the next change, and the divergence is invisible from inside any
//! one backend: each returns a self-consistent number, and only a comparison
//! across engines notices. So the enforcement is this test, not discipline.
//!
//! # The fixtures are READ FROM DISK, and that is the point
//!
//! `omega-core`'s unit tests build the feedforward circuit by hand, and the first
//! version of that helper stopped one line early — omitting
//! `12_feedforward_sometimes_false.qasm`'s trailing `measure q[1] -> c[0]`, the
//! second write to `c[0]` that made the pass resolve its control to the wrong
//! qubit and emit `CX q1,q1`. That panics the statevector backend outright, and
//! the only test looking for the defect could not see it because its input was
//! an approximation of the fixture rather than the fixture.
//!
//! This file parses the `.qasm` files, so a hand-copy cannot drift from them.
//!
//! # The expected value is MEASURED, not asserted from arithmetic
//!
//! `⟨Z₁⟩` is anchored by sampling the original circuit through the `Collapse`
//! trajectory path — a completely different mechanism from the analytic deferral
//! being tested. Writing the value as a literal would be a second implementation
//! of the thing under test, with no reviewer.
//!
//! **Why `⟨Z₁⟩` and not `⟨Z₀Z₁⟩`, which would be the more obvious choice:** both
//! fixtures declare `creg c[1]` and write q0 *and then* q1 into the same single
//! bit, so the sampled counts contain one bit — q1's final value — and the JOINT
//! outcome of the two qubits is simply not observable through them. An anchor
//! computed as `agree − differ` over those counts would have been reading a
//! qubit that is not in the data. `⟨Z₁⟩` is what the register actually records,
//! and it discriminates on both fixtures: 0 for the sometimes-false one, −1 for
//! the always-true one, against +1 for an engine that drops the guard.
//!
//! `⟨Z₀Z₁⟩` is therefore checked only for agreement ACROSS engines, with no
//! reference value. That is consistency, not correctness — our own backends
//! agreeing proves comparatively little — and it is labelled as such below.
//!
//! `⟨X₀X₁⟩` is asserted as **exactly 0**, with no tolerance: dephasing deletes
//! the term rather than computing a small number, and
//! `AerSimulator(method="density_matrix")` gives exactly 0.0 at every shot count
//! because each collapsed branch individually has zero X-expectation.

use omega_backend_mps::MpsBackend;
use omega_backend_pauli::PauliBackend;
use omega_backend_pauliprop::PauliPropBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::CircuitIR;
use omega_core::error::OmegaError;
use omega_core::executor::{
    Backend as CoreBackend, ExecConfig, ExecResult, MidCircuitMode, Observable,
};
use omega_core::params::ParameterBinding;

/// Read a crosscheck fixture by the distinctive part of its filename.
fn fixture(name_fragment: &str) -> (String, CircuitIR) {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("omega-bridges")
        .join("tests")
        .join("fixtures")
        .join("crosscheck");
    let mut hits: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().map(|x| x == "qasm").unwrap_or(false)
                && p.file_name()
                    .and_then(|f| f.to_str())
                    .map(|f| f.contains(name_fragment))
                    .unwrap_or(false)
        })
        .collect();
    hits.sort();
    assert_eq!(
        hits.len(),
        1,
        "expected exactly one fixture matching {name_fragment:?} in {}, found {hits:?}. \
         This test is worthless if it silently matches nothing.",
        dir.display()
    );
    let src = std::fs::read_to_string(&hits[0]).expect("read fixture");
    let ir = omega_parser::lower_to_ir(&src)
        .unwrap_or_else(|e| panic!("{} does not parse: {e}", hits[0].display()));
    (
        hits[0].file_name().unwrap().to_string_lossy().into_owned(),
        ir,
    )
}

/// `⟨Z₁⟩` measured by SAMPLING the original circuit, trajectory by trajectory.
///
/// The independent anchor. This path honours the guard by actually collapsing and
/// branching, and shares no code with the deferral pass.
///
/// Both fixtures declare a 1-bit `creg` and write q0 then q1 into it, so the
/// recorded bit is q1's final measurement — which is exactly what `Z₁` asks
/// about. Nothing here can recover the joint q0/q1 outcome; see the module doc.
fn sampled_z1(ir: &CircuitIR) -> f64 {
    const SHOTS: u32 = 40_000;
    let cfg = ExecConfig {
        shots: Some(SHOTS),
        seed: Some(4242),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let ExecResult::Counts(counts) = StatevectorBackend::new()
        .execute(ir, &ParameterBinding::new(), &cfg)
        .expect("the trajectory path runs feedforward circuits")
    else {
        panic!("expected counts");
    };
    let total: u32 = counts.values().sum();
    assert_eq!(total, SHOTS, "every shot must be accounted for");
    let ones: i64 = counts
        .iter()
        .filter(|(o, _)| o.bit(0) == 1)
        .map(|(_, n)| *n as i64)
        .sum();
    // ⟨Z⟩ = P(0) − P(1) = 1 − 2·P(1).
    1.0 - 2.0 * (ones as f64 / SHOTS as f64)
}

fn z1() -> Observable {
    Observable::parse("Z1").expect("Z1 parses")
}

fn zz() -> Observable {
    Observable::parse("Z0 Z1").expect("Z0 Z1 parses")
}
fn xx() -> Observable {
    Observable::parse("X0 X1").expect("X0 X1 parses")
}

/// Either the engine gives the mixture value, or it refuses with `Unsupported`.
/// Anything else is a divergence and fails.
/// # Two tolerances, because the two comparisons are not the same kind of claim
///
/// `SAMPLED_TOL` is for a value anchored against 40 000 trajectories: that
/// estimate carries √N noise of about 0.005, so a band of 0.02 is roughly 4σ —
/// wide enough not to flake, far narrower than the 1.0 gap a dropped guard
/// produces.
///
/// `EXACT_TOL` is for `⟨X₀X₁⟩ = 0`, which is not an average of anything.
/// Dephasing DELETES the term, and Aer's density-matrix path returns exactly 0.0
/// at every shot count because each collapsed branch individually has zero
/// X-expectation. Giving that comparison a sampling tolerance would throw away
/// the strongest assertion available here.
const SAMPLED_TOL: f64 = 0.02;
const EXACT_TOL: f64 = 1e-9;

/// Cross-engine agreement spans an f32 GPU and f64 CPUs, so it cannot be exact.
///
/// Metal returned `0.99999994` against the CPU's `1.0000000000000002` on the
/// first run of this lane — f32 round-off, not a disagreement about semantics.
/// `1e-6` is well inside f32's ~1e-7 and nowhere near the 1.0-sized gap a
/// mis-wired engine produces.
const CROSS_ENGINE_TOL: f64 = 1e-6;

fn check(
    engine: &str,
    fixture_name: &str,
    got: omega_core::error::Result<f64>,
    want: f64,
    tol: f64,
) {
    match got {
        Ok(v) if (v - want).abs() < tol => {}
        Ok(v) => panic!(
            "{engine} on {fixture_name}: returned {v}, expected {want} (tol {tol}).\n\
             This is the defect this lane exists to catch: an engine that keeps the \
             old semantics returns a plausible, self-consistent, WRONG number. \
             Check that this backend's expectation path calls \
             `omega_core::defer_measure::prepare_for_expectation`."
        ),
        Err(OmegaError::Unsupported(_)) => {}
        Err(e) => panic!(
            "{engine} on {fixture_name}: failed with {e:?}. A backend that cannot \
             express this must return `Unsupported` — the N-way matrix files that \
             as `cannot-express` rather than as a failure."
        ),
    }
}

#[test]
fn every_engine_agrees_on_the_feedforward_fixtures_or_refuses() {
    for fragment in ["feedforward_sometimes_false", "feedforward_always_true"] {
        let (name, ir) = fixture(fragment);
        let anchor = sampled_z1(&ir);
        eprintln!("  {name}: sampled <Z1> = {anchor:+.4}");

        let sv = StatevectorBackend::new();
        // chi = 2 is the FULL bond dimension at 2 qubits, so this is exact and
        // does not trip the over-provisioning advisory on every call.
        let mps = MpsBackend::new(2);
        let pauli = PauliBackend::new();
        let pp = PauliPropBackend::new();
        #[allow(unused_mut)]
        let mut engines: Vec<(&str, &dyn CoreBackend)> = vec![
            ("statevector", &sv),
            ("mps", &mps),
            ("pauli", &pauli),
            ("pauliprop", &pp),
        ];
        // Metal is the backend the QML trainer actually runs on, so it is in this
        // lane rather than trusted. A machine without a usable device SKIPS it
        // loudly instead of silently passing a four-engine test as if it were
        // five.
        #[cfg(feature = "metal")]
        let metal = omega_backend_statevector_metal::MetalStatevectorBackend::new();
        #[cfg(feature = "metal")]
        match &metal {
            Ok(m) => engines.push(("metal", m)),
            Err(e) => eprintln!("  metal backend unavailable, NOT covered here: {e}"),
        }
        // CUDA is the other backend the QML trainer runs on. Same treatment as
        // Metal: in the lane, not trusted, and a box without a device says so
        // rather than passing a smaller test as though it were the full one.
        #[cfg(feature = "cuda")]
        let cuda = omega_backend_statevector_cuda::CudaStatevectorBackend::new();
        #[cfg(feature = "cuda")]
        match &cuda {
            Ok(c) => engines.push(("cuda", c)),
            Err(e) => eprintln!("  cuda backend unavailable, NOT covered here: {e}"),
        }

        let p = ParameterBinding::new();
        let mut zz_values: Vec<(&str, f64)> = Vec::new();
        for (engine, be) in &engines {
            // Anchored: must match the SAMPLED value, so the deferred control is
            // really applied. An engine that drops the guard gives +1 on both
            // fixtures, against 0 and −1 respectively.
            check(
                engine,
                &name,
                be.expectation(&ir, &p, &z1()),
                anchor,
                SAMPLED_TOL,
            );
            // Off-diagonal, exactly 0. Naive deferral without dephasing gives +1
            // for fixture 12 — a pure Bell state — and this is the only kind of
            // observable that separates the two designs at all.
            check(
                engine,
                &name,
                be.expectation(&ir, &p, &xx()),
                0.0,
                EXACT_TOL,
            );
            // `expectation_multi` is a separate override on every backend and must
            // obey the same rule. The QML trainer calls this one, not the single
            // form, so wiring only the latter would leave every training run wrong.
            match be.expectation_multi(&ir, &p, &[z1(), xx()]) {
                Ok(vals) => {
                    assert_eq!(vals.len(), 2, "{engine}: expectation_multi arity");
                    check(engine, &name, Ok(vals[0]), anchor, SAMPLED_TOL);
                    check(engine, &name, Ok(vals[1]), 0.0, EXACT_TOL);
                }
                Err(OmegaError::Unsupported(_)) => {}
                Err(e) => panic!("{engine} on {name}: expectation_multi failed with {e:?}"),
            }
            // `expectation_batch` must not bypass the rule either.
            match be.expectation_batch(&ir, &[&p], &xx()) {
                Ok(vals) => check(engine, &name, Ok(vals[0]), 0.0, EXACT_TOL),
                Err(OmegaError::Unsupported(_)) => {}
                Err(e) => panic!("{engine} on {name}: expectation_batch failed with {e:?}"),
            }
            if let Ok(v) = be.expectation(&ir, &p, &zz()) {
                zz_values.push((engine, v));
            }
        }

        // CONSISTENCY ONLY: no reference value exists for this one (see the module
        // doc — the fixtures' 1-bit creg cannot record the joint outcome). Engines
        // agreeing with each other is weak evidence, and it is claimed as nothing
        // more; it would still catch one backend being wired differently from the
        // rest, which is the specific failure this file is here to prevent.
        if let Some((first_engine, first)) = zz_values.first().copied() {
            for (engine, v) in &zz_values {
                assert!(
                    (v - first).abs() < CROSS_ENGINE_TOL,
                    "{name}: <Z0Z1> is {v} on {engine} but {first} on {first_engine}. \
                     The engines have diverged on what an expectation MEANS, which is \
                     exactly what a shared rule in one place is supposed to make \
                     impossible."
                );
            }
        }
    }
}

/// **No engine may silently answer a circuit that cannot be deferred.**
///
/// `reset` after a measurement is the QEC ancilla-recycling pattern, and it has
/// no deferred form: the measurement cannot move past work that depends on its
/// having happened. Every engine must refuse.
#[test]
fn a_reset_and_reuse_circuit_is_refused_by_every_engine() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[1];\n\
               h q[0];\nmeasure q[0] -> c[0];\nreset q[0];\nh q[0];\ncx q[0],q[1];\n";
    let ir = omega_parser::lower_to_ir(src).expect("parses");

    let sv = StatevectorBackend::new();
    // chi = 2 is the FULL bond dimension at 2 qubits, so this is exact and
    // does not trip the over-provisioning advisory on every call.
    let mps = MpsBackend::new(2);
    let pauli = PauliBackend::new();
    let pp = PauliPropBackend::new();
    let engines: Vec<(&str, &dyn CoreBackend)> = vec![
        ("statevector", &sv),
        ("mps", &mps),
        ("pauli", &pauli),
        ("pauliprop", &pp),
    ];
    let p = ParameterBinding::new();
    for (engine, be) in &engines {
        match be.expectation(&ir, &p, &zz()) {
            Err(OmegaError::Unsupported(_)) => {}
            Err(e) => panic!("{engine}: refused, but not with Unsupported: {e:?}"),
            Ok(v) => panic!(
                "{engine}: returned {v} for a reset-and-reuse circuit. There is no \
                 deferred form of this circuit, so any number here is the answer to \
                 a DIFFERENT circuit — the same failure the `Reset` refusal in \
                 pauliprop already guards against."
            ),
        }
    }
}
