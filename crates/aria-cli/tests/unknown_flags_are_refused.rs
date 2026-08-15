// SPDX-License-Identifier: Apache-2.0
//! **An unrecognised `--flag` must be refused, and the declared vocabulary must
//! not drift from what the code reads.**
//!
//! `parse_args` treated any `--name` outside `bool_flags` as taking a value, so
//! an unknown option was consumed, stored, and never looked at again. Measured
//! on `examples/aria/bell.aria` before the fix:
//!
//! ```text
//!   aria run … --shot 9999             ran 1024 shots (the default)
//!   aria run … --strict-trunctaion .5  ran with the DEFAULT truncation gate
//!   aria run … --typo xyz              ran, no diagnostic at all
//! ```
//!
//! The middle one is why this is a correctness bug and not a papercut: a typo
//! in a safety flag left the user believing they had tightened a gate they had
//! not touched, and the output looked completely normal.
//!
//! # The part that would rot
//!
//! The fix declares each subcommand's option names in `Vocabulary`, so the
//! refusal happens at parse time — before a file is read or anything is
//! reserved. A declared list can drift from what the accessors actually ask
//! for, and drift here is silent in the dangerous direction: add
//! `a.opt("new-flag")` to `cmd_run` without touching `Vocabulary::RUN` and
//! `--new-flag` is *refused* (loud, fine); remove a read and leave the name
//! declared and it is *accepted and ignored* — the original bug, back again.
//!
//! `the_declared_vocabulary_matches_what_the_code_reads` compares the two by
//! scanning the source, so both directions fail the build.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn bin() -> PathBuf {
    let mut p = std::env::current_exe().expect("current_exe");
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("aria")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("repo root")
        .to_path_buf()
}

fn bell() -> PathBuf {
    repo_root().join("examples/aria/bell.aria")
}

fn aria(args: &[&str]) -> Output {
    Command::new(bin()).args(args).output().expect("spawn aria")
}

/// The three measured cases, plus a transposition.
#[test]
fn a_mistyped_flag_is_refused_and_the_real_name_suggested() {
    let f = bell();
    let f = f.to_str().unwrap();
    for (bad, expect_suggestion) in [
        ("--shot", Some("--shots")),
        ("--strict-trunctaion", Some("--strict-truncation")),
        ("--stpes", None), // not a `run` option at all
        ("--typo", None),
    ] {
        let out = aria(&["run", f, "--circuit", "Bell", "--shots", "20", bad, "9"]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(1),
            "{bad} was ACCEPTED — it is consumed and ignored, which is the \
             defect. stdout:\n{}",
            String::from_utf8_lossy(&out.stdout)
        );
        assert!(
            stderr.contains(bad),
            "the refusal does not name {bad}:\n{stderr}"
        );
        if let Some(sug) = expect_suggestion {
            assert!(
                stderr.contains(sug),
                "{bad} should suggest {sug}:\n{stderr}"
            );
        }
    }
}

/// **Guard the guard.** The same command line without the bad flag must work,
/// or every assertion above passes because the run was broken anyway.
#[test]
fn the_same_command_line_without_the_bad_flag_succeeds() {
    let f = bell();
    let out = aria(&[
        "run",
        f.to_str().unwrap(),
        "--circuit",
        "Bell",
        "--shots",
        "20",
        "--seed",
        "1",
    ]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a valid `aria run` failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("|00>") || stdout.contains("|11>"),
        "expected Bell counts, got:\n{stdout}"
    );
}

/// Flags that DO exist must still be accepted — the refusal must not be a
/// blanket one.
#[test]
fn declared_flags_are_still_accepted() {
    let f = bell();
    let f = f.to_str().unwrap();
    for extra in [
        vec!["--strict-truncation", "5.0"],
        vec!["--backend", "mps:8"],
        vec!["--expectation", "Z0"],
        vec!["--statevector"],
    ] {
        let mut args = vec![
            "run",
            f,
            "--circuit",
            "Bell",
            "--shots",
            "20",
            "--seed",
            "1",
        ];
        args.extend(extra.iter().copied());
        let out = aria(&args);
        assert_eq!(
            out.status.code(),
            Some(0),
            "{extra:?} was refused although it is a real option:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// **Asking for help is never an error.**
///
/// `aria run --help` printed "--help requires a value" — the parser treated it
/// as a value-taking option like any other unknown name.
#[test]
fn every_subcommand_answers_help() {
    for c in [
        "run", "train", "tune", "predict", "export", "import", "list", "parse",
    ] {
        for flag in ["--help", "-h"] {
            let out = aria(&[c, flag]);
            let combined = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            assert_eq!(
                out.status.code(),
                Some(0),
                "`aria {c} {flag}` exited {:?}: {combined}",
                out.status.code()
            );
            assert!(
                combined.contains("aria") && !combined.contains("requires a value"),
                "`aria {c} {flag}` did not print usage: {combined}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The drift check.
// ---------------------------------------------------------------------------

/// Names passed to `.opt(..)`, `.has(..)` or `.all(..)` inside `fn <name>`.
fn keys_read_by(src: &str, func: &str) -> std::collections::BTreeSet<String> {
    let start = src.find(&format!("\nfn {func}(")).unwrap_or_else(|| {
        panic!(
            "fn {func} not found — this test scans the source and the \
                                   source moved; update it rather than deleting it"
        )
    });
    // The next top-level `fn` ends the body.
    let rest = &src[start + 1..];
    let end = rest[1..].find("\nfn ").map(|i| i + 1).unwrap_or(rest.len());
    let body = &rest[..end];

    let mut out = std::collections::BTreeSet::new();
    for m in ["opt(\"", "has(\"", "all(\""] {
        let mut at = 0;
        while let Some(i) = body[at..].find(m) {
            let s = at + i + m.len();
            if let Some(j) = body[s..].find('"') {
                out.insert(body[s..s + j].to_string());
            }
            at = s;
        }
    }
    out
}

/// The `Vocabulary::<NAME>` list, plus the subcommand's boolean flags.
fn declared(src: &str, konst: &str) -> std::collections::BTreeSet<String> {
    let i = src
        .find(&format!("const {konst}: &'static [&'static str] = &["))
        .unwrap_or_else(|| panic!("Vocabulary::{konst} not found"));
    let body = &src[i..];
    let end = body.find("];").expect("unterminated Vocabulary entry");
    body[..end]
        .split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect()
}

/// **Every name the code reads is declared, and every name declared is read.**
///
/// One direction stops a real option being refused; the other stops a declared
/// name outliving its reader, which resurrects the accept-and-ignore bug for
/// exactly that flag.
#[test]
fn the_declared_vocabulary_matches_what_the_code_reads() {
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs"))
        .expect("read main.rs");

    // `strict-truncation` is read by a shared helper; `cmd_train` delegates to
    // `cmd_train_supervised` with the same `Args`.
    let strict = keys_read_by(&src, "apply_strict_truncation");
    let supervised = keys_read_by(&src, "cmd_train_supervised");
    // Boolean flags are passed separately to `parse_args`, not in Vocabulary.
    let bools: std::collections::BTreeSet<String> =
        ["statevector", "qasm", "qasm3", "json", "lean", "gate-model"]
            .iter()
            .map(|s| s.to_string())
            .collect();

    let cases: Vec<(&str, &str, Vec<&std::collections::BTreeSet<String>>)> = vec![
        ("cmd_list", "LIST", vec![]),
        ("cmd_parse", "PARSE", vec![]),
        ("cmd_run", "RUN", vec![&strict]),
        ("cmd_train", "TRAIN", vec![&strict, &supervised]),
        ("cmd_predict", "PREDICT", vec![&strict]),
        ("cmd_export", "EXPORT", vec![]),
        ("cmd_tune", "TUNE", vec![&strict]),
        ("cmd_import", "IMPORT", vec![]),
    ];

    for (func, konst, extra) in cases {
        let mut read = keys_read_by(&src, func);
        for e in extra {
            read.extend(e.iter().cloned());
        }
        read.retain(|k| !bools.contains(k));
        let dec = declared(&src, konst);

        let missing: Vec<&String> = read.difference(&dec).collect();
        assert!(
            missing.is_empty(),
            "{func} reads {missing:?} but Vocabulary::{konst} does not declare \
             them — those options are now REFUSED even though the code handles \
             them"
        );
        let stale: Vec<&String> = dec.difference(&read).collect();
        assert!(
            stale.is_empty(),
            "Vocabulary::{konst} declares {stale:?} but {func} never reads them \
             — those options are accepted and silently ignored, which is the \
             original bug for exactly those flags"
        );
    }
}
