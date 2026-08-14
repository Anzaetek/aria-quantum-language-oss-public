// SPDX-License-Identifier: Apache-2.0
//! **A shots run is refused if ANY trajectory crosses the truncation ceiling.**
//!
//! `MpsBackend::execute` has two gates on the same quantity, and they used to
//! disagree:
//!
//! * an **early abort** inside `evolve_once`, which fires on the first
//!   trajectory to cross — its comment claimed "stopping at the first crossing
//!   changes nothing about WHICH runs are refused";
//! * a **deferred check** after the shot loop, reading stats that
//!   `record_stats` wrote **last-writer-wins** — so it saw only the LAST
//!   trajectory.
//!
//! On a feed-forward circuit those are different sets. A 200-shot run where
//! shot 3 discarded 2.5 of the state was refused with the early abort and
//! ACCEPTED without it, whenever the final shot took the cheap branch — the
//! comment was false, and the deferred gate was the wrong one, because a
//! histogram is polluted by any bad trajectory, not just its last.
//!
//! `record_stats` now keeps the worst since `reset_stats`, so both gates key on
//! the same number.
//!
//! # Why this fixture can fail
//!
//! `h q[0]; measure q[0]` splits the trajectories 50/50, and the CZ ladder runs
//! only when the bit is 1. The two branches have measured certificates **0.0**
//! and **2.5** at χ=4, so a ceiling of 1.0 sits strictly between them and the
//! verdict genuinely depends on which trajectories are consulted.
//!
//! The ladder is `h` on all sites then **cz** across a 7-rung cut, whose Schmidt
//! rank is 2^7 = 128. It must be CZ: `cx` on |+>|+> is the identity, which
//! makes the whole fixture a product state that truncates nothing and passes
//! every ceiling — measured (bond 1 at every χ) while writing this.

use omega_backend_mps::MpsBackend;
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;

/// Measure q0; run an entangling ladder only if it came out 1.
fn feed_forward() -> String {
    let mut s =
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[14];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n"
            .to_string();
    for i in 1..14 {
        s.push_str(&format!("if(c==1) h q[{i}];\n"));
    }
    for i in 0..7 {
        s.push_str(&format!("if(c==1) cz q[{i}], q[{}];\n", i + 7));
    }
    s
}

fn run(chi: usize, ceiling: f64, shots: u32, seed: u64) -> (Result<ExecResult, omega_core::error::OmegaError>, f64) {
    let ir = omega_parser::lower_to_ir(&feed_forward()).expect("lower");
    let b = MpsBackend::new(chi).with_max_discarded_weight(ceiling);
    let cfg = ExecConfig {
        shots: Some(shots),
        seed: Some(seed),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let r = b.execute(&ir, &ParameterBinding::default(), &cfg);
    let cert = b.last_run_stats().discarded_weight;
    (r, cert)
}

/// **Guard the guard: the two branches must straddle the ceiling.**
///
/// If both branches ever truncate the same amount, every assertion below passes
/// for free.
#[test]
fn the_two_branches_have_different_certificates() {
    // Seeds chosen after measuring which branch each takes at one shot.
    let mut seen: Vec<f64> = Vec::new();
    for seed in 1..=12u64 {
        let (r, cert) = run(4, 1e9, 1, seed);
        assert!(r.is_ok(), "ceiling 1e9 must accept everything");
        if !seen.iter().any(|c| (c - cert).abs() < 1e-9) {
            seen.push(cert);
        }
    }
    seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(
        seen.len(),
        2,
        "expected exactly two trajectory certificates (cheap branch and ladder \
         branch), got {seen:?} — the fixture no longer discriminates and every \
         other test in this file would pass vacuously"
    );
    assert!(
        seen[0] < 1.0 && seen[1] > 1.0,
        "the ceiling used below (1.0) must sit strictly between the two \
         branches; measured {seen:?}"
    );
}

/// **The claim.** Enough shots to hit both branches ⇒ refused, on every seed.
///
/// Not "on some seed": the refusal must not depend on which branch the final
/// trajectory happened to take, and that dependence is exactly the defect.
#[test]
fn any_crossing_trajectory_refuses_the_whole_run() {
    for seed in 1..=8u64 {
        let (r, _) = run(4, 1.0, 40, seed);
        let e = r.expect_err(&format!(
            "seed {seed}: 40 shots hit the ladder branch with overwhelming \
             probability (1 - 2^-40), and that branch discards 2.5 against a \
             ceiling of 1.0. Accepting means the gate consulted one trajectory \
             rather than the worst."
        ));
        assert!(
            e.to_string().contains("truncation certificate"),
            "seed {seed}: refused for the wrong reason: {e}"
        );
    }
}

/// **The converse, and the sharp test of `record_stats`.** A ceiling above
/// both branches accepts, and the reported certificate is the WORST
/// trajectory's on every seed — not the last one's.
///
/// Every seed matters. Under last-writer-wins the reported value is whatever
/// the final shot happened to do, so roughly half the seeds report the cheap
/// branch's 0.0 and half report 2.5. Checking one seed is a coin flip that
/// reads as a pass — the first version of this test did exactly that and
/// survived the mutation it was written to catch.
#[test]
fn the_reported_certificate_is_the_worst_trajectory_on_every_seed() {
    // The ladder branch's certificate, measured at χ=4.
    let ladder = {
        let mut seen: Vec<f64> = (1..=12u64).map(|s| run(4, 1e9, 1, s).1).collect();
        seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
        *seen.last().unwrap()
    };
    assert!(ladder > 1.0, "fixture: ladder branch certificate {ladder}");

    for seed in 1..=8u64 {
        let (r, cert) = run(4, 1e9, 40, seed);
        let counts = match r.expect("must accept") {
            ExecResult::Counts(c) => c,
            o => panic!("{o:?}"),
        };
        assert_eq!(counts.values().sum::<u32>(), 40);
        assert!(
            (cert - ladder).abs() < 1e-9,
            "seed {seed}: reported certificate {cert}, expected the worst \
             trajectory's {ladder}. 40 shots hit the ladder branch with \
             probability 1 - 2^-40, so anything else means the stats record the \
             LAST trajectory rather than the worst — and the deferred gate then \
             refuses or accepts based on a coin flip."
        );
    }
}

/// **Stats do not leak between runs.** `record_stats` accumulates a maximum, so
/// without a reset a previous circuit's certificate would gate the next one.
#[test]
fn stats_are_reset_per_execute() {
    let ir = omega_parser::lower_to_ir(&feed_forward()).expect("lower");
    let b = MpsBackend::new(4).with_max_discarded_weight(1e9);
    let cfg = |shots, seed| ExecConfig {
        shots: Some(shots),
        seed: Some(seed),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    // A run that truncates heavily...
    b.execute(&ir, &ParameterBinding::default(), &cfg(40, 1)).expect("run 1");
    assert!(b.last_run_stats().discarded_weight > 1.0);

    // ...must not colour a subsequent exact one.
    let bell = omega_parser::lower_to_ir(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[1];\nh q[0];\ncx q[0],q[1];\nmeasure q[0]->c[0];\n",
    )
    .expect("lower");
    b.execute(&bell, &ParameterBinding::default(), &cfg(10, 1)).expect("run 2");
    assert_eq!(
        b.last_run_stats().discarded_weight,
        0.0,
        "a Bell pair at χ=4 truncates nothing; a non-zero certificate here is \
         the previous run's, which would also refuse this one at any ceiling \
         below it"
    );
}
