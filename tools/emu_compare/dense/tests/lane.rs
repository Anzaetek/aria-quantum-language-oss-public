// SPDX-License-Identifier: Apache-2.0
//! The dense lane's gates, each reddened by the input it exists to refuse.
//!
//! The first group is pure. The second needs the lane venv (`EMU_DENSE_PY`)
//! and is a registered skip without it; `EMU_DENSE_REQUIRE=1`, as akilles sets
//! for this lane, turns that skip back into a failure.

use emu_compare_dense::client::{Engine, PyArm};
use emu_compare_dense::gate::{classify, nontrivial, spread, state_gate, state_tol, z_gate};
use emu_compare_dense::workload::{
    load_artifact, observable_string, observable_terms, write_npy, Ours,
};
use omega_core::executor::Observable;
use omega_emu_compare::Classification;
use serde_json::json;

#[test]
fn a_value_that_is_zero_for_the_circuit_is_refused() {
    // <Z0> on every committed HEA artifact is 0 to rounding: the gate that
    // would have admitted two arms agreeing at zero.
    assert!(nontrivial(0.0).is_err());
    assert!(nontrivial(1.36e-16).is_err());
    assert!(nontrivial(-5e-4).is_err());
    assert!(nontrivial(14.641578281274).is_ok());
    assert!(nontrivial(0.046283512825).is_ok());
}

#[test]
fn the_state_gate_is_two_sided() {
    // qsim's norm drift: F = 1 + 9.6e-8. The one-sided 1 - F <= tol passes it.
    assert!(state_gate("f64 arm", 9.6e-8, 1e-12, state_tol(true)).is_err());
    assert!(state_gate("f64 arm", 3e-15, 4e-15, state_tol(true)).is_ok());
}

#[test]
fn the_state_gate_sees_an_amplitude_error_fidelity_tolerates() {
    // §4.3a: a fidelity inside tolerance with one amplitude badly wrong.
    assert!(state_gate("f32 arm", 5e-7, 3e-3, state_tol(false)).is_err());
    assert!(state_gate("f32 arm", 5e-7, 8e-8, state_tol(false)).is_ok());
}

#[test]
fn the_sampling_gate_refuses_six_standard_errors() {
    let exact = [0.10, -0.20, 0.0];
    let se0 = ((1.0 - 0.01) / 1000.0_f64).sqrt();
    let bad = [0.10 + 6.0 * se0, -0.20, 0.0];
    let ok = [0.10 + 2.0 * se0, -0.21, 0.01];
    assert!(z_gate("arm", &bad, &exact, 1000).is_err());
    let worst = z_gate("arm", &ok, &exact, 1000).unwrap();
    assert!(worst > 1.9 && worst < 2.1, "worst |z| {worst}");
}

#[test]
fn a_collapsed_repeat_is_flagged() {
    // qulacs QFT-16 on a loaded box: 1.37 s against 0.0015 s.
    assert!(spread(&[0.0015, 0.0016, 1.37]).1);
    assert!(!spread(&[0.010, 0.011, 0.012]).1);
}

#[test]
fn classification_needs_both_the_ratio_and_disjoint_intervals() {
    assert_eq!(classify(2.0, 1.0, 1.1, 2.0, 2.2), Classification::Win);
    assert_eq!(classify(0.4, 1.0, 1.1, 0.4, 0.45), Classification::Loss);
    assert_eq!(classify(1.25, 1.0, 1.1, 1.25, 1.3), Classification::Tie);
    assert_eq!(classify(1.5, 1.0, 1.6, 1.5, 1.7), Classification::Tie); // overlap
}

#[test]
fn the_observable_touches_every_qubit_with_2n_minus_1_terms() {
    let n = 5u32;
    let s = observable_string(n);
    let obs = Observable::parse(&s).unwrap();
    assert_eq!(obs.terms.len(), (2 * n - 1) as usize);
    let terms = observable_terms(n);
    assert_eq!(terms.as_array().unwrap().len(), (2 * n - 1) as usize);
    for q in 0..n {
        assert!(s.contains(&format!("X{q}")), "{s} misses X{q}");
    }
}

/// Venv-gated legs: a registered skip, never a silent one.
fn require() -> bool {
    if std::env::var("EMU_DENSE_PY").is_ok() {
        return true;
    }
    if std::env::var("EMU_DENSE_REQUIRE").as_deref() == Ok("1") {
        panic!("EMU_DENSE_REQUIRE=1 but EMU_DENSE_PY is unset");
    }
    eprintln!("SKIP: EMU_DENSE_PY unset, so the venv legs of the dense lane cannot run");
    false
}

#[test]
fn on_the_committed_hea_z0_is_zero_and_the_pinned_observable_is_not() {
    if !require() {
        return;
    }
    let art = load_artifact("hea-14q-d4").unwrap();
    let mut oracle = PyArm::launch(Engine::AerOracle { threads: 16 }).unwrap();
    oracle.load(&art.id, &art.path).unwrap();
    let z0 = oracle
        .call(json!({"op": "reference", "name": art.id, "terms": [[1.0, [[0, "Z"]]]]}))
        .unwrap()["value"]
        .as_f64()
        .unwrap();
    assert!(nontrivial(z0).is_err(), "<Z0> = {z0} was admitted");
    let o = oracle
        .call(json!({"op": "reference", "name": art.id, "terms": observable_terms(art.qubits)}))
        .unwrap()["value"]
        .as_f64()
        .unwrap();
    assert!(nontrivial(o).is_ok(), "O = {o} was refused");
}

#[test]
fn qsim_refuses_a_generic_or_downgraded_build() {
    if !require() {
        return;
    }
    let py = std::env::var("EMU_DENSE_PY").unwrap();
    let dir = env!("CARGO_MANIFEST_DIR");
    for (module, must_contain) in [
        ("qsimcirq.qsim_basic", "GENERIC"),
        ("qsimcirq.qsim_sse", "downgraded"),
        ("qsimcirq.qsim_avx2", "downgraded"),
    ] {
        let code = format!(
            "import sys, types\n\
             import qsim_arm as A\n\
             A._qs.qsim = types.SimpleNamespace(__name__='{module}')\n\
             A.qsim_decide.detect_instructions = lambda: {{'qsimcirq.qsim_basic': 3, 'qsimcirq.qsim_sse': 2, 'qsimcirq.qsim_avx2': 1}}['{module}']\n\
             try:\n    A.simd_witness()\nexcept RuntimeError as e:\n    print('REFUSED', e); sys.exit(0)\n\
             print('ADMITTED'); sys.exit(1)\n"
        );
        let out = std::process::Command::new(&py)
            .arg("-c")
            .arg(&code)
            .current_dir(dir)
            .env("EMU_DENSE_THREADS", "1")
            .env("EMU_DENSE_QSIM_FUSE", "2")
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "{module} was admitted: {text}");
        assert!(text.contains(must_contain), "{module}: {text}");
    }
    // And the real build on this box is admitted.
    PyArm::launch(Engine::Qsim {
        threads: 1,
        fuse: 2,
    })
    .expect("the box's own qsim build");
}

#[test]
fn every_arm_matches_the_aer_double_state_on_a_committed_cell() {
    if !require() {
        return;
    }
    // RANDOM-1, not HEA: HEA is symmetric under bit reversal, so a competitor
    // with the wrong bit order would pass every HEA state check.
    let art = load_artifact("random1-20q-d20").unwrap();
    let tmp = std::env::temp_dir().join(format!("emu-dense-test-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut oracle = PyArm::launch(Engine::AerOracle { threads: 4 }).unwrap();
    oracle.load(&art.id, &art.path).unwrap();
    let r = tmp.join("ref.npy");
    oracle
        .call(json!({"op": "reference", "name": art.id, "terms": observable_terms(art.qubits), "out": r}))
        .unwrap();
    let mut check = |label: &str, p: &std::path::Path, f64_arm: bool| {
        let v = oracle
            .call(json!({"op": "compare", "ref": r, "other": p}))
            .unwrap();
        let omf = v["one_minus_f_abs"].as_f64().unwrap();
        let md = v["max_abs_diff"].as_f64().unwrap();
        state_gate(label, omf, md, state_tol(f64_arm)).unwrap();
    };
    let ours = Ours::new(4).unwrap();
    let p = tmp.join("ours.npy");
    write_npy(&p, &ours.state(&art.circuit).unwrap()).unwrap();
    check("ours", &p, true);
    for (e, f64_arm) in [
        (Engine::Qulacs { threads: 4 }, true),
        (
            Engine::Qsim {
                threads: 4,
                fuse: 2,
            },
            false,
        ),
    ] {
        let mut a = PyArm::launch(e).unwrap();
        a.load(&art.id, &art.path).unwrap();
        let p = tmp.join(format!("{}.npy", a.engine.label()));
        a.write_state(&art.id, &p).unwrap();
        check(&a.engine.label(), &p, f64_arm);
    }
    // A state off in norm by 1e-7 must FAIL an f64 state gate: |1 - F| is
    // two-sided (qsim's F = 1 + 9.6e-8 passes a one-sided rule).
    let mut st = ours.state(&art.circuit).unwrap();
    let scaled: Vec<_> = st.iter().map(|a| a * (1.0 + 5e-8)).collect();
    let p = tmp.join("scaled.npy");
    write_npy(&p, &scaled).unwrap();
    let v = oracle
        .call(json!({"op": "compare", "ref": r, "other": p}))
        .unwrap();
    assert!(
        state_gate(
            "scaled",
            v["one_minus_f_abs"].as_f64().unwrap(),
            0.0,
            state_tol(true)
        )
        .is_err(),
        "a state with norm 1 + 1e-7 passed the two-sided gate: {v}"
    );
    // A mirrored (bit-reversed) state must FAIL the gate: the bit-order trap.
    let n = art.qubits as usize;
    let rev = |i: usize| (0..n).fold(0usize, |acc, b| acc | (((i >> b) & 1) << (n - 1 - b)));
    let orig = st.clone();
    for (i, a) in st.iter_mut().enumerate() {
        *a = orig[rev(i)];
    }
    let p = tmp.join("mirrored.npy");
    write_npy(&p, &st).unwrap();
    let v = oracle
        .call(json!({"op": "compare", "ref": r, "other": p}))
        .unwrap();
    assert!(
        state_gate(
            "mirrored",
            v["one_minus_f_abs"].as_f64().unwrap(),
            v["max_abs_diff"].as_f64().unwrap(),
            state_tol(true)
        )
        .is_err(),
        "a bit-reversed state passed the state gate: {v}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}
