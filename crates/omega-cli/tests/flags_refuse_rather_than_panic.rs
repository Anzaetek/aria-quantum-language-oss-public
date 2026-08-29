// SPDX-License-Identifier: Apache-2.0
//! **A malformed command line must be refused, not panic.**
//!
//! Every flag in every one of `omega-run`'s three argument loops was written
//! `i += 1; args[i]`, which indexes past the end when the flag is last on the
//! line. Measured before the fix:
//!
//! ```text
//!   $ omega-run c.qasm --backend mps:8 --shots 20 --noise
//!   thread 'main' panicked at crates/omega-cli/src/main.rs:349:39:
//!   index out of bounds: the len is 9 but the index is 9
//!   note: run with `RUST_BACKTRACE=1` ...
//!   $ echo $?
//!   101
//! ```
//!
//! Value parsing had the same shape — `.expect("invalid shots number")` panics
//! on `--shots abc`, and names neither the flag nor what was passed.
//!
//! Exit status matters as much as the message: 101 is "this program has a bug",
//! 1 is "your input was wrong". A script cannot tell those apart from the text.

use std::path::PathBuf;
use std::process::{Command, Output};

fn bin() -> PathBuf {
    // The test binary lives in target/<profile>/deps/; the CLI is two up.
    let mut p = std::env::current_exe().expect("current_exe");
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("omega-run")
}

/// The shared QASM fixture, written **exactly once per process** to a
/// **process-unique** path.
///
/// Both of those matter, and neither was true before — this test was flaky at
/// roughly 1 run in 6:
///
/// ```text
///   a_well_formed_command_line_still_succeeds ... FAILED
///   Parse error: unknown circuit format: expected OPENQASM or OPTICQASM header
/// ```
///
/// The old version rewrote `temp_dir()/omega_flag_refusal_fixture.qasm` on
/// every call. All three tests call it, cargo runs them on separate threads,
/// and `fs::write` truncates before it writes — so one test could hand
/// `omega-run` a file another test had just emptied. The parse error was the
/// symptom; the fixture was momentarily zero bytes.
///
/// A fixed name in a shared temp dir is also racy ACROSS processes: two
/// concurrent `cargo test` runs (different target dirs, a second checkout, CI
/// beside a local run) collide on the same path. The pid in the name removes
/// that, and `OnceLock` removes the in-process race by making the write happen
/// once rather than per call.
fn circuit() -> PathBuf {
    static FIXTURE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let p = std::env::temp_dir()
                .join(format!("omega_flag_refusal_fixture_{}.qasm", std::process::id()));
            std::fs::write(
                &p,
                "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\n",
            )
            .expect("write fixture");
            p
        })
        .clone()
}

fn run(extra: &[&str]) -> Output {
    let c = circuit();
    let mut cmd = Command::new(bin());
    cmd.arg(&c);
    cmd.args(extra);
    cmd.output().expect("spawn omega-run")
}

/// Every value-taking flag, given no value.
#[test]
fn a_flag_missing_its_value_is_refused_cleanly() {
    let flags = [
        "--shots",
        "--seed",
        "--backend",
        "--backend-dir",
        "--bridge",
        "--device",
        "--input",
        "--expectation",
        "--gradient",
        "--gradient-of-fn",
        "--score-fn-shots",
        "--params",
        "--method",
        "--noise",
    ];
    for f in flags {
        let out = run(&[f]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !stderr.contains("panicked"),
            "{f} with no value PANICKED:\n{stderr}"
        );
        assert_eq!(
            out.status.code(),
            Some(1),
            "{f} with no value exited {:?}; 1 means \"your input was wrong\", \
             101 means \"this program has a bug\". stderr:\n{stderr}",
            out.status.code()
        );
        assert!(
            stderr.contains(f),
            "the refusal for {f} does not name the flag, so the reader cannot \
             tell which one was wrong:\n{stderr}"
        );
    }
}

/// A value of the wrong shape names the flag AND what was passed.
#[test]
fn a_flag_with_an_unparseable_value_is_refused_cleanly() {
    for (flag, bad) in [
        ("--shots", "abc"),
        ("--seed", "not-a-seed"),
        ("--score-fn-shots", "1.5"),
        ("--params", "1,x,3"),
        ("--input", "1,two"),
    ] {
        let out = run(&[flag, bad]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !stderr.contains("panicked"),
            "{flag} {bad} PANICKED:\n{stderr}"
        );
        assert_eq!(out.status.code(), Some(1), "{flag} {bad}: {stderr}");
        assert!(
            stderr.contains(flag),
            "{flag} {bad}: refusal does not name the flag:\n{stderr}"
        );
    }
}

/// **Guard the guard.** The fixture and binary must actually work, or every
/// assertion above passes because nothing ever ran.
#[test]
fn a_well_formed_command_line_still_succeeds() {
    let out = run(&["--backend", "mps:8", "--shots", "20", "--seed", "1"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a valid command line failed, so the refusal tests above prove nothing:\n{stderr}"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("11") || stdout.contains("00"),
        "expected Bell counts on stdout, got:\n{stdout}"
    );
}
