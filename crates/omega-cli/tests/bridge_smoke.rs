//! Smoke tests for `omega-run --bridge`.
//!
//! These exercise the dispatcher plumbing without requiring a Python
//! venv. The default workspace build has no `bridge-*` features, so the
//! dispatcher returns `NotCompiled` — exactly the "off" path operators
//! see when they haven't opted into a bridge backend.

use std::path::PathBuf;
use std::process::{Command, Stdio};

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_omega-run"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn fixture() -> PathBuf {
    // A self-contained Bell QASM shipped alongside the test. (The earlier path
    // pointed at a private `verify-qiskit/fixtures/` tree that isn't vendored
    // into this repo, so every test failed at file-read before reaching the
    // bridge dispatcher these tests actually exercise.)
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("bell_phi_plus.qasm")
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(binary_path())
        .args(args)
        .current_dir(repo_root())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn omega-run")
}

/// Only meaningful when the feature is ABSENT — the whole point is what
/// happens without it. Compiled in, `--bridge qiskit` succeeds and the
/// assertion `expected failure, got success` fires.
///
/// It was ungated, so `cargo test -p omega-cli --features bridge-qiskit` had
/// always failed here. Nothing noticed because ci.sh never tested omega-cli
/// with bridge features until the N-way matrix stage arrived — the same
/// hand-maintained-crate-list hole as FIXES_PLAN.md K9, one layer down.
#[cfg(not(feature = "bridge-qiskit"))]
#[test]
fn bridge_default_features_reports_not_compiled() {
    // Without `bridge-qiskit`/`bridge-perceval` compiled in, every
    // `--bridge <name>` invocation should fail with the typed
    // NotCompiled message that points operators at the rebuild flag.
    let path = fixture();
    let output = run(&[
        path.to_str().unwrap(),
        "--bridge",
        "qiskit",
        "--shots",
        "100",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "expected failure, got success");
    assert!(
        stderr.contains("not compiled in"),
        "expected NotCompiled message, got: {stderr}"
    );
    assert!(
        stderr.contains("--features bridge-qiskit"),
        "expected rebuild hint, got: {stderr}"
    );
}

#[test]
fn bridge_rejects_statevector_mode() {
    let path = fixture();
    let output = run(&[
        path.to_str().unwrap(),
        "--bridge",
        "qiskit",
        "--statevector",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    // Asserts the two facts, not the sentence: statevector mode is refused,
    // and the refusal says where to go instead. The wording changed when
    // `--expectation` became routable over a bridge, and pinning the old
    // sentence made a deliberate capability gain look like a regression.
    assert!(
        stderr.contains("Statevector") && stderr.contains("in-process backend"),
        "expected mode-rejection notice, got: {stderr}"
    );
}

/// The counterpart: `--expectation` IS supported over a bridge now, so the
/// refusal above must not have widened to cover it. Without this, tightening
/// the guard back to "sampling only" would pass every remaining test.
#[test]
fn bridge_accepts_expectation_mode() {
    let path = fixture();
    let output = run(&[
        path.to_str().unwrap(),
        "--bridge",
        "qiskit",
        "--expectation",
        "Z0",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("require an in-process backend"),
        "expectation must not be refused as an unsupported bridge mode: {stderr}"
    );
}

#[test]
fn bridge_rejects_unknown_backend_name() {
    let path = fixture();
    let output = run(&[
        path.to_str().unwrap(),
        "--bridge",
        "no-such-backend",
        "--shots",
        "100",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("Invalid --bridge value") && stderr.contains("unknown backend"),
        "expected unknown-backend message, got: {stderr}"
    );
}

/// `--noise` with `--bridge --expectation` must be REFUSED, not dropped.
///
/// It was dropped, silently. The counts path parses `--noise` and hands it to
/// the runner; the expectation path returns before reaching that code, and the
/// in-process guard that catches the same mistake for statevector/mps sits
/// ~400 lines further down, past the `return`. So this printed a noiseless
/// expectation to stdout with nothing on stderr — the shape that matters,
/// because `--format json > out.json` captured a plausible wrong number and no
/// trace of the flag that had been ignored.
///
/// `omega_bridges::expectation_qasm2` takes no noise argument at all, so this
/// is not a gap in one backend: no bridge can express it. The refusal is
/// therefore placed before `Backend::parse`, which is also what lets this test
/// run in the default feature set — it asserts the flag combination is
/// rejected, not anything about a particular bridge being installed.
#[test]
fn bridge_expectation_refuses_noise_rather_than_ignoring_it() {
    let path = fixture();
    let output = run(&[
        path.to_str().unwrap(),
        "--bridge",
        "qiskit",
        "--expectation",
        "Z0Z1",
        "--noise",
        r#"{"depolarizing":0.4}"#,
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success(), "must not exit 0: {stdout}");
    assert!(
        stderr.contains("--noise cannot be combined with --bridge --expectation"),
        "expected the noise refusal, got: {stderr}"
    );
    // The point of the bug was a NUMBER on stdout. Nothing may be emitted.
    assert!(
        !stdout.contains("<O> ="),
        "an expectation value was printed despite the dropped noise model: {stdout}"
    );
}

/// The neighbouring paths must keep working — the guard is scoped to the one
/// combination that cannot be expressed, not to `--noise` or to `--bridge`.
#[test]
fn bridge_noise_guard_does_not_touch_the_counts_path() {
    let path = fixture();
    let output = run(&[
        path.to_str().unwrap(),
        "--bridge",
        "qiskit",
        "--shots",
        "100",
        "--noise",
        r#"{"depolarizing":0.4}"#,
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Whatever this build does with the counts path (NotCompiled here, a real
    // run with the feature on), it must NOT be the expectation guard.
    assert!(
        !stderr.contains("--noise cannot be combined"),
        "the counts path must still accept --noise: {stderr}"
    );
}

/// The ppvm runner wrapper, or `None` (with the reason printed) when the venv
/// is absent — a missing venv is a SKIP, never a pass.
#[cfg(feature = "bridge-ppvm")]
fn ppvm_runner_or_skip() -> Option<PathBuf> {
    let py = repo_root().join("crates/omega-bridges/python/.venv-ppvm/bin/python");
    if !py.exists() {
        eprintln!("ppvm venv missing at {} — skipping", py.display());
        return None;
    }
    Some(repo_root().join("crates/omega-bridges/python/omega-bridge-ppvm-runner"))
}

#[cfg(feature = "bridge-ppvm")]
fn run_ppvm(runner: &std::path::Path, noise: &str) -> std::process::Output {
    Command::new(binary_path())
        .args([
            fixture().to_str().unwrap(),
            "--bridge",
            "ppvm",
            "--shots",
            "64",
            "--noise",
            noise,
        ])
        .env("OMEGA_BRIDGE_PPVM_CMD", runner)
        .current_dir(repo_root())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn omega-run")
}

/// `--bridge ppvm --noise` on the counts path: a mapped channel runs and
/// prints counts (it used to be refused wholesale — STATUS §5 #12).
#[cfg(feature = "bridge-ppvm")]
#[test]
fn bridge_ppvm_counts_accept_mapped_noise() {
    let Some(runner) = ppvm_runner_or_skip() else {
        return;
    };
    let out = run_ppvm(&runner, r#"{"depolarizing":0.05}"#);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "must exit 0: {stderr}");
    let total: u64 = stdout
        .lines()
        .filter_map(|l| l.trim().strip_prefix('|'))
        .filter_map(|l| l.split_once(">:"))
        .map(|(_, n)| n.trim().parse::<u64>().expect("count"))
        .sum();
    assert_eq!(total, 64, "expected 64 shots of counts on stdout: {stdout}");
}

/// ...and an unmappable channel is refused by NAME, as CannotExpress, with a
/// non-zero exit — never a noiseless distribution.
#[cfg(feature = "bridge-ppvm")]
#[test]
fn bridge_ppvm_counts_refuse_amplitude_damping_by_name() {
    let Some(runner) = ppvm_runner_or_skip() else {
        return;
    };
    let out = run_ppvm(&runner, r#"{"amplitude_damping":0.1}"#);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "must not exit 0: {stdout}");
    assert!(
        stderr.contains("cannot express this circuit")
            && stderr.contains("ppvm-noise-not-supported")
            && stderr.contains("`amplitude_damping`"),
        "expected the CannotExpress sentence naming amplitude_damping, got: {stderr}"
    );
    assert!(
        !stdout.contains("Results:"),
        "no counts may be printed: {stdout}"
    );
}
