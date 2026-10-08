// SPDX-License-Identifier: Apache-2.0
//! ppvm's noisy counts against OUR noisy statevector — the noise mapping in
//! `python/stim_noise.py`, checked end to end.
//!
//! The mapping's job is to reproduce **omega's** semantics, not Stim's
//! defaults, so the reference here is `NoisyStatevectorBackend` fed the SAME
//! `--noise` JSON through `NoiseModel::from_json`. Two places where the obvious
//! Stim spelling is wrong are pinned by fixtures that tell them apart:
//!
//! * omega kicks EACH qubit of a two-qubit gate with an independent one-qubit
//!   depolarizing draw — `DEPOLARIZE1` per qubit, not `DEPOLARIZE2`.
//! * omega's readout error flips the RECORD, not the qubit — `M(p)`, not
//!   `X_ERROR(p)` before `M`. `measure` twice on one qubit distinguishes them.
//!
//! **Vacuity guard.** Every agreement fixture also asserts that ppvm's
//! NOISELESS distribution is far (> 0.03 TVD) from its noisy one, so a pass
//! cannot come from the noise being dropped on both sides.

#![cfg(feature = "bridge-ppvm")]

use omega_backend_statevector::{NoiseModel, NoisyStatevectorBackend};
use omega_bridges::{run_qasm2, Backend, BridgeError};
use omega_core::executor::{Backend as _, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn runner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")
}
fn venv(slug: &str) -> PathBuf {
    runner_dir()
        .join(format!(".venv-{slug}"))
        .join("bin")
        .join("python")
}
fn force(slug: &str) {
    std::env::set_var(
        format!("OMEGA_BRIDGE_{}_CMD", slug.to_ascii_uppercase()),
        runner_dir().join(format!("omega-bridge-{slug}-runner")),
    );
}

macro_rules! skip_unless {
    ($($slug:expr),+) => {
        $(if !venv($slug).exists() {
            eprintln!("{} venv missing — skipping", $slug);
            return;
        } force($slug);)+
    };
}

const HDR: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n";
// ppvm is unseeded through `run_qasm2`, so the TVD floor is sampling noise on
// BOTH sides. At 20k shots the worst 8-outcome fixture sat at 0.010–0.014
// against AGREE = 0.02 across five runs — too close for a nightly. 50k shots
// puts the floor near 0.007; the mutants this file exists to catch land at
// 0.04–0.18, so AGREE keeps its margin in both directions.
const SHOTS: u32 = 50_000;
const AGREE: f64 = 0.02;
const NOISE_VISIBLE: f64 = 0.03;

fn measure_all(n: usize) -> String {
    (0..n)
        .map(|i| format!("measure q[{i}] -> c[{i}];\n"))
        .collect()
}

fn bell() -> String {
    format!(
        "{HDR}qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\n{}",
        measure_all(2)
    )
}

fn ghz3() -> String {
    format!(
        "{HDR}qreg q[3];\ncreg c[3];\nh q[0];\ncx q[0], q[1];\ncx q[1], q[2];\n{}",
        measure_all(3)
    )
}

/// Bell state read in the X basis: Z-type errors before the final `h` layer
/// become visible bit flips, X-type errors after it do too. A Z-basis Bell
/// fixture would be blind to half of a Pauli channel.
fn bell_x_basis() -> String {
    format!(
        "{HDR}qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\nh q[0];\nh q[1];\n{}",
        measure_all(2)
    )
}

/// Our noisy statevector, per-shot trajectories, keyed MSB-first by creg.
fn ours(qasm: &str, noise: &Value, seed: u64) -> HashMap<String, u32> {
    let ir = omega_parser::lower_to_ir(qasm).expect("lower");
    let model = NoiseModel::from_json(&noise.to_string()).expect("omega accepts the noise JSON");
    let cfg = ExecConfig {
        shots: Some(SHOTS),
        seed: Some(seed),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    match NoisyStatevectorBackend::with_model(model, Some(seed))
        .execute(&ir, &ParameterBinding::default(), &cfg)
        .expect("noisy statevector must run")
    {
        ExecResult::Counts(c) => c.into_iter().map(|(o, n)| (o.to_bitstring(), n)).collect(),
        o => panic!("{o:?}"),
    }
}

/// ppvm through the bridge. The wire is LSB-first; reversed to MSB-first to
/// match `Outcome::to_bitstring` (see `wide_counts_vs_qiskit_mps.rs`).
fn ppvm(qasm: &str, noise: Option<&Value>) -> HashMap<String, u32> {
    run_qasm2(Backend::Ppvm, qasm, SHOTS, noise)
        .expect("ppvm must run")
        .into_iter()
        .map(|(k, v)| (k.chars().rev().collect::<String>(), v))
        .collect()
}

fn tvd(a: &HashMap<String, u32>, b: &HashMap<String, u32>) -> f64 {
    let ta: f64 = a.values().map(|&v| v as f64).sum();
    let tb: f64 = b.values().map(|&v| v as f64).sum();
    let mut keys: HashSet<&String> = a.keys().collect();
    keys.extend(b.keys());
    0.5 * keys
        .iter()
        .map(|k| {
            let pa = *a.get(*k).unwrap_or(&0) as f64 / ta;
            let pb = *b.get(*k).unwrap_or(&0) as f64 / tb;
            (pa - pb).abs()
        })
        .sum::<f64>()
}

/// ppvm noisy ≈ ours noisy (≤ AGREE), and ppvm noisy is visibly NOT ppvm
/// noiseless (> NOISE_VISIBLE).
fn assert_agrees(label: &str, qasm: &str, noise: Value, seed: u64) {
    let theirs = ppvm(qasm, Some(&noise));
    let mine = ours(qasm, &noise, seed);
    let clean = ppvm(qasm, None);
    let d = tvd(&theirs, &mine);
    let guard = tvd(&theirs, &clean);
    eprintln!(
        "{label} noise={noise}: TVD(ppvm, ours) = {d:.4}; TVD(ppvm noisy, ppvm clean) = {guard:.4}"
    );
    assert!(
        guard > NOISE_VISIBLE,
        "{label}: VACUITY — ppvm's noisy counts are within {guard:.4} of its noiseless \
         ones, so the noise did not reach the sampler.\n noisy {theirs:?}\n clean {clean:?}"
    );
    assert!(
        d <= AGREE,
        "{label}: TVD(ppvm, noisy statevector) = {d:.4} > {AGREE}\n ppvm {theirs:?}\n ours {mine:?}"
    );
}

fn refusal(noise: Value) -> String {
    match run_qasm2(Backend::Ppvm, &bell(), 64, Some(&noise)) {
        Err(BridgeError::CannotExpress(Backend::Ppvm, msg)) => msg,
        other => panic!("noise {noise} must be CannotExpress, got {other:?}"),
    }
}

#[test]
fn amplitude_damping_is_refused_by_name_as_cannot_express() {
    skip_unless!("ppvm");
    let msg = refusal(json!({"amplitude_damping": 0.1}));
    assert!(
        msg.contains("ppvm-noise-not-supported") && msg.contains("amplitude_damping"),
        "the refusal must name the key: {msg}"
    );
    let msg = refusal(json!({"readout": [{"p10": 0.02, "p01": 0.03}]}));
    assert!(
        msg.contains("ppvm-noise-not-supported")
            && msg.contains("asymmetric `readout`")
            && msg.contains("p10=0.02"),
        "asymmetric readout must be refused by name: {msg}"
    );
    // A refused key does not hide behind a mapped one.
    let msg = refusal(json!({"depolarizing": 0.01, "amplitude_damping": [0.0, 0.2]}));
    assert!(msg.contains("amplitude_damping"), "{msg}");
}

#[test]
fn depolarizing_counts_match_the_noisy_statevector() {
    skip_unless!("ppvm");
    assert_agrees("bell", &bell(), json!({"depolarizing": 0.05}), 11);
    assert_agrees("ghz3", &ghz3(), json!({"depolarizing": 0.05}), 12);
    assert_agrees(
        "ghz3 arity split",
        &ghz3(),
        json!({"depolarizing": {"1q": 0.02, "2q": 0.1}}),
        13,
    );
    // Per-pair: (0,1) is hot, (1,2) falls to the per-qubit default, whose
    // entry for q2 differs from q1's — so both the pair lookup and the
    // per-qubit fallback inside one 2q gate are exercised.
    assert_agrees(
        "ghz3 per-pair",
        &ghz3(),
        json!({"depolarizing": {"1q": [0.01, 0.0, 0.0],
                                "2q": {"0,1": 0.15, "default": [0.0, 0.01, 0.08]}}}),
        14,
    );
}

#[test]
fn pauli_channel_counts_match() {
    skip_unless!("ppvm");
    assert_agrees(
        "bell x-basis",
        &bell_x_basis(),
        json!({"pauli": {"X": 0.03, "Y": 0.02, "Z": 0.06}}),
        21,
    );
    assert_agrees(
        "ghz3 per-qubit pauli",
        &ghz3(),
        json!({"pauli": {"X": [0.08, 0.0, 0.03], "Y": 0.01, "Z": [0.0, 0.05, 0.0]}}),
        22,
    );
}

#[test]
fn phase_damping_counts_match() {
    skip_unless!("ppvm");
    // omega unravels phase damping λ as Z w.p. λ/2 — only visible off the Z basis.
    assert_agrees(
        "bell x-basis",
        &bell_x_basis(),
        json!({"phase_damping": 0.2}),
        31,
    );
}

#[test]
fn symmetric_readout_counts_match() {
    skip_unless!("ppvm");
    assert_agrees("bell flip", &bell(), json!({"readout_flip": 0.1}), 41);
    assert_agrees(
        "ghz3 per-qubit",
        &ghz3(),
        json!({"readout": [0.05, 0.0, 0.15]}),
        42,
    );
    assert_agrees(
        "bell symmetric objects",
        &bell(),
        json!({"readout": [{"p10": 0.12, "p01": 0.12}, {"p10": 0.03, "p01": 0.03}]}),
        43,
    );
    // Record-only flip: the second `measure` of the same qubit must see the
    // TRUE state. With `X_ERROR` before each `M` (a state flip) c1 would be
    // wrong w.p. 2p(1-p) = 0.32 instead of p = 0.2, and correlated with c0.
    let twice = format!(
        "{HDR}qreg q[1];\ncreg c[2];\nx q[0];\nmeasure q[0] -> c[0];\nmeasure q[0] -> c[1];\n"
    );
    assert_agrees("measure twice", &twice, json!({"readout_flip": 0.2}), 44);
}

/// Placement, not rates: omega kicks after a `reset` and after an `id` (which
/// lower to `R` and to nothing), and ONCE after a `swap` — which ppvm lowers
/// to three `CX`, so a kick per emitted instruction would be three kicks. Each
/// fixture is built so the wrong placement moves the distribution by ≥ 0.08.
#[test]
fn noise_placement_matches_on_reset_id_and_swap() {
    skip_unless!("ppvm");
    // |1> → reset → |0>, then the kick: P(1) = 2p/3 = 0.10, or 0 with no kick.
    let reset =
        format!("{HDR}qreg q[1];\ncreg c[1];\nx q[0];\nreset q[0];\nmeasure q[0] -> c[0];\n");
    assert_agrees("reset", &reset, json!({"depolarizing": 0.15}), 51);
    // Two kicks (after `x`, after `id`) vs one: P(0) = 0.18 vs 0.10.
    let id = format!("{HDR}qreg q[1];\ncreg c[1];\nx q[0];\nid q[0];\nmeasure q[0] -> c[0];\n");
    assert_agrees("id", &id, json!({"depolarizing": 0.15}), 52);
    // 1q rate 0 so only the swap's own kick shows; three kicks (one per CX of
    // the expansion) would roughly triple the flip rate on each qubit.
    let swap = format!(
        "{HDR}qreg q[2];\ncreg c[2];\nx q[0];\nswap q[0], q[1];\n{}",
        measure_all(2)
    );
    assert_agrees(
        "swap",
        &swap,
        json!({"depolarizing": {"1q": 0.0, "2q": 0.2}}),
        53,
    );
}

/// Every `--noise` key omega's `NoiseModel::from_json` recognises. Private in
/// omega-core (`NOISE_KEYS`), so copied. A key added there and not here is
/// still refused by the runner's "does not map noise key" branch — the safe
/// direction — but this list would then under-cover it.
const OMEGA_NOISE_KEYS: [&str; 6] = [
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "pauli",
    "readout_flip",
    "readout",
];

fn capabilities() -> Value {
    let mut child = Command::new(runner_dir().join("omega-bridge-ppvm-runner"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn ppvm runner");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(br#"{"mode":"capabilities"}"#)
        .unwrap();
    let out = child.wait_with_output().unwrap();
    serde_json::from_slice(&out.stdout).expect("capabilities JSON")
}

#[test]
fn the_capabilities_handshake_lists_the_mapped_keys() {
    skip_unless!("ppvm");
    let caps = capabilities();
    let keys: Vec<String> = caps["capabilities"]["noise_keys"]
        .as_array()
        .expect("noise_keys array")
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let expected = [
        "depolarizing",
        "pauli",
        "phase_damping",
        "readout",
        "readout_flip",
    ];
    assert_eq!(
        keys, expected,
        "handshake must list exactly the mapped keys"
    );

    // Listed ⇒ actually runs. Not listed ⇒ actually refused. Both directions,
    // so the handshake cannot claim a key the mapping drops, or hide one it has.
    let sample = |k: &str| match k {
        "pauli" => json!({"pauli": {"X": 0.1}}),
        other => json!({ other: 0.1 }),
    };
    for k in OMEGA_NOISE_KEYS {
        let r = run_qasm2(Backend::Ppvm, &bell(), 64, Some(&sample(k)));
        if keys.iter().any(|x| x == k) {
            assert!(r.is_ok(), "`{k}` is advertised but does not run: {r:?}");
        } else {
            match r {
                Err(BridgeError::CannotExpress(_, m)) => {
                    assert!(m.contains(k), "refusal of `{k}` must name it: {m}")
                }
                other => panic!("`{k}` is not advertised but was not refused: {other:?}"),
            }
        }
    }
    let notes = caps["capabilities"]["notes"].as_str().unwrap_or_default();
    assert!(
        notes.contains("amplitude_damping") && notes.contains("asymmetric readout"),
        "notes must say what is refused: {notes}"
    );
}

/// The Python resolver (`stim_noise.Depolarizing.at_gate`) against the Rust
/// one it mirrors (`omega_core::noise::Depolarizing::at_gate`), on a table
/// covering scalar, arity split, per-qubit arrays (incl. past the end),
/// per-pair hit in both orders, pair miss → default, no default → 0, and
/// arity 3 ignoring the pair table.
#[test]
fn the_python_rate_resolution_mirrors_at_gate() {
    skip_unless!("ppvm");
    let specs = [
        json!(0.05),
        json!([0.01, 0.02]),
        json!({"1q": 0.001, "2q": 0.012}),
        json!({"1q": [0.1, 0.2, 0.3], "2q": [0.4, 0.5]}),
        json!({"1q": 1e-4, "2q": {"0,1": 7.2e-3, "2,1": 4.1e-3, "default": 3e-3}}),
        json!({"2q": {"0,1": 9e-3}}),
        json!({"1q": 0.01, "2q": {"1,3": 0.2, "default": [0.0, 0.03, 0.04, 0.05]}}),
    ];
    let gates: [&[usize]; 8] = [
        &[0],
        &[2],
        &[5],
        &[0, 1],
        &[1, 0],
        &[2, 1],
        &[3, 1],
        &[0, 1, 2],
    ];
    let mut cases = Vec::new();
    let mut want = Vec::new();
    for spec in &specs {
        let m = NoiseModel::from_json(&json!({ "depolarizing": spec }).to_string()).unwrap();
        for g in gates {
            for &q in g {
                cases.push(json!([spec, q, g]));
                want.push(m.depolarizing.at_gate(q, g));
            }
        }
    }
    let script = format!(
        "import json,sys; sys.path.insert(0, {dir:?}); from stim_noise import Depolarizing; \
         print(json.dumps([Depolarizing(s).at_gate(q, g) for s, q, g in json.loads(sys.stdin.read())]))",
        dir = runner_dir().to_str().unwrap()
    );
    let mut child = Command::new(venv("ppvm"))
        .args(["-c", &script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn python");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(Value::Array(cases.clone()).to_string().as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "python resolver failed");
    let got: Vec<f64> = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(got.len(), want.len());
    for ((c, g), w) in cases.iter().zip(&got).zip(&want) {
        assert!(g == w, "at_gate mismatch for {c}: python {g}, rust {w}");
    }
    // The table is not all zeros — a resolver returning 0.0 everywhere would
    // otherwise agree with half of it.
    assert!(want.iter().filter(|w| **w > 0.0).count() > want.len() / 2);
}
