// SPDX-License-Identifier: Apache-2.0
//! `--dump-state-bits`: a measurement artifact, so every guard is a hard
//! refusal and the payload is bit patterns, never decimal.
//!
//! The flag exists for the cross-device bit-equality protocol
//! (`PLAN-BITEQ-CUDA-METAL.md`). What these tests pin, feature-independently:
//!
//! * the artifact's amplitudes are HEX BIT PATTERNS at the width of the
//!   executing arm (cpu → 16 hex digits of `f64::to_bits`), and they decode
//!   to the analytic state — mutation-checked below by decoding rather than
//!   by pattern-matching, so a wrong width or decimal leak fails loudly;
//! * an explicit `--device` is REQUIRED (`OMEGA_DEVICE` must not be able to
//!   choose what the artifact claims), and a device this binary cannot
//!   dispatch to is refused at the dispatch decision, not discovered later;
//! * `--shots` is refused — sampled counts have no amplitudes.
//!
//! (`--noise` needs no test here: `--noise` without `--shots` is refused by
//! an existing guard upstream of the dump flag, and a dump-side noise test
//! would pass even with no dump-side check — a vacuous assertion, per the
//! plan's review.)

use std::path::PathBuf;
use std::process::{Command, Output};

fn bin() -> PathBuf {
    let mut p = std::env::current_exe().expect("current_exe");
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("omega-run")
}

fn circuit() -> PathBuf {
    static FIXTURE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let p = std::env::temp_dir()
                .join(format!("omega_dump_state_bits_{}.qasm", std::process::id()));
            // Bell pair: exactly two non-zero amplitudes of 1/sqrt(2), whose
            // f64 bit pattern is a known constant — decodable, not guessable.
            std::fs::write(
                &p,
                "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n",
            )
            .expect("write fixture");
            p
        })
        .clone()
}

fn run(extra: &[&str], env: &[(&str, &str)]) -> Output {
    let c = circuit();
    let mut cmd = Command::new(bin());
    cmd.arg(&c);
    cmd.args(extra);
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.output().expect("spawn omega-run")
}

fn out_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "omega_dump_state_bits_{}_{}.json",
        tag,
        std::process::id()
    ))
}

#[test]
fn cpu_artifact_decodes_to_the_analytic_bell_state() {
    let path = out_path("cpu");
    let o = run(
        &[
            "--statevector",
            "--device",
            "cpu",
            "--dump-state-bits",
            path.to_str().unwrap(),
        ],
        &[],
    );
    assert!(o.status.success(), "cpu dump failed: {o:?}");
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("artifact written"))
            .expect("artifact is JSON");

    assert_eq!(doc["format"], "biteq-v1");
    assert_eq!(doc["backend"], "cpu-f64");
    assert_eq!(doc["precision"], "f64");
    assert_eq!(doc["n_qubits"], 2);
    assert_eq!(doc["multi_control"], "decompose", "the default mode");
    assert!(
        doc["build"]["rev"].as_str().is_some_and(|r| !r.is_empty()),
        "artifact must carry the build rev"
    );
    assert!(
        doc["ordering"].as_str().unwrap().contains("LOW bit"),
        "amplitude ordering must be stated in the file"
    );

    // Decode the bits — the test that catches a wrong width or a decimal
    // round-trip, which string checks cannot.
    let amps = doc["amps"].as_array().expect("amps");
    assert_eq!(amps.len(), 4);
    let dec = |i: usize, part: usize| {
        let s = amps[i][part].as_str().expect("hex string");
        assert_eq!(s.len(), 16, "cpu-f64 must be 16 hex digits, got {s:?}");
        f64::from_bits(u64::from_str_radix(s, 16).expect("hex"))
    };
    let isqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    assert_eq!(dec(0, 0).to_bits(), isqrt2.to_bits(), "amp[0].re");
    assert_eq!(dec(3, 0).to_bits(), isqrt2.to_bits(), "amp[3].re");
    for (i, want_zero) in [(1usize, true), (2, true)] {
        assert!(
            want_zero && dec(i, 0) == 0.0 && dec(i, 1) == 0.0,
            "amp[{i}] must be zero in a Bell state"
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_missing_device_is_refused_even_when_the_env_var_names_one() {
    let path = out_path("nodev");
    let o = run(
        &["--statevector", "--dump-state-bits", path.to_str().unwrap()],
        &[("OMEGA_DEVICE", "cpu")],
    );
    assert_eq!(o.status.code(), Some(1));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("EXPLICIT --device"),
        "must demand an explicit device, got: {err}"
    );
    assert!(!path.exists(), "no artifact may be written on refusal");
}

#[test]
fn a_device_this_binary_cannot_dispatch_to_is_refused_at_the_decision() {
    // `cuda` is never compiled into the default macOS/Linux test build of
    // this crate's test matrix on Apple hardware; on a CUDA-featured build
    // this test still passes because the refusal text differs from the
    // fallback text only in which guard fired — both name the flag.
    let path = out_path("badd");
    let o = run(
        &[
            "--statevector",
            "--device",
            "opencl",
            "--dump-state-bits",
            path.to_str().unwrap(),
        ],
        &[],
    );
    if cfg!(feature = "opencl") {
        // Featured builds may legitimately dispatch; nothing to assert here
        // beyond "did not silently write a cpu artifact labelled opencl".
        if o.status.success() {
            let doc: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).expect("artifact"))
                    .expect("json");
            assert_eq!(doc["backend"], "opencl-f32");
        }
    } else {
        assert_eq!(o.status.code(), Some(1));
        let err = String::from_utf8_lossy(&o.stderr);
        // The PROPERTY, not one guard's wording: an explicit --device this
        // binary cannot dispatch to must refuse, naming the device. Two
        // guards can produce this — the general explicit-device refusal
        // (which fires first) and the dump-specific one behind it — and
        // pinning either phrase would make this test fail when the other
        // legitimately wins the race to the exit.
        assert!(
            err.contains("--device opencl") || err.contains("will not dispatch"),
            "must refuse and name the device, got: {err}"
        );
        assert!(!path.exists(), "no artifact may be written on refusal");
    }
    let _ = std::fs::remove_file(&path);
}

/// The bridge/expectation/gradient modes return early, far above the run-path
/// dump guards — review found the flag was silently IGNORED there, leaving
/// any stale artifact in place for diff.py to trust. Now a hard refusal,
/// before any mode runs.
#[test]
fn non_execute_modes_are_refused_not_ignored() {
    let path = out_path("mode");
    let o = run(
        &[
            "--expectation",
            "Z0",
            "--device",
            "cpu",
            "--dump-state-bits",
            path.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(o.status.code(), Some(1), "must refuse, not run the mode");
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("no state to dump"),
        "refusal must name the conflict: {err}"
    );
    assert!(!path.exists(), "no artifact may be written on refusal");
}

#[test]
fn shots_are_refused() {
    let path = out_path("shots");
    let o = run(
        &[
            "--shots",
            "10",
            "--device",
            "cpu",
            "--dump-state-bits",
            path.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(o.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("analytic state"),
        "refusal must say why"
    );
    assert!(!path.exists());
}

/// **An explicit `--device` that cannot be honoured must EXIT NON-ZERO, even
/// with no dump flag in sight.**
///
/// Reported from two GPU hosts: `--device cuda` above 28 qubits fell back to
/// the CPU with only a stderr notice, and a benchmark lane published a
/// confident wrong conclusion about GPU performance because the substituted
/// results looked exactly like device results. A silent substitution does not
/// produce missing data — it produces plausible data.
///
/// The three non-regressions matter as much as the refusal, so they are
/// asserted here rather than assumed: automatic selection (no flag) must keep
/// falling back, `OMEGA_DEVICE` must keep falling back (it expresses no
/// per-run expectation the way a flag does), and `--device cpu` must succeed.
#[test]
fn an_explicit_device_that_cannot_be_honoured_is_an_error() {
    // A device no default build compiles in. On a featured build this run
    // legitimately succeeds, so the assertion is scoped the same way as the
    // sibling test above.
    let o = run(&["--shots", "10", "--seed", "1", "--device", "opencl"], &[]);
    if !cfg!(feature = "opencl") {
        assert_eq!(
            o.status.code(),
            Some(1),
            "explicit --device opencl must refuse, not run on the CPU"
        );
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(
            err.contains("--device opencl"),
            "the refusal must name the device: {err}"
        );
        assert!(
            !String::from_utf8_lossy(&o.stdout).contains("counts"),
            "no results may be emitted alongside a device refusal"
        );
    }

    // Non-regression 1: automatic selection still falls back and SUCCEEDS.
    let auto = run(&["--shots", "10", "--seed", "1", "--format", "json"], &[]);
    assert!(auto.status.success(), "auto selection must still run");

    // Non-regression 2: OMEGA_DEVICE is not a per-run expectation, so it
    // keeps falling back rather than refusing.
    let env = run(
        &["--shots", "10", "--seed", "1", "--format", "json"],
        &[("OMEGA_DEVICE", "opencl")],
    );
    assert!(
        env.status.success(),
        "OMEGA_DEVICE must fall back, not refuse: {}",
        String::from_utf8_lossy(&env.stderr)
    );

    // Non-regression 3: an explicit CPU request is always honourable.
    let cpu = run(
        &[
            "--shots", "10", "--seed", "1", "--device", "cpu", "--format", "json",
        ],
        &[],
    );
    assert!(cpu.status.success(), "--device cpu must succeed");
    assert!(
        String::from_utf8_lossy(&cpu.stdout).contains("\"device_used\":\"cpu-f64\""),
        "and must report the arm that ran"
    );
}
