//! Integration tests for `omega-run --shor`.

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

fn run(args: &[&str]) -> serde_json::Value {
    let out = Command::new(binary_path())
        .args(args)
        .current_dir(repo_root())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .expect("spawn omega-run");
    assert!(out.status.success(), "omega-run failed: {:?}", out.status);
    let s = String::from_utf8(out.stdout).unwrap();
    serde_json::from_str(&s).expect("json parse")
}

#[test]
fn shor_factors_15() {
    let v = run(&["--shor", "--N", "15", "--seed", "42", "--format", "json"]);
    assert_eq!(v["mode"], "shor");
    assert_eq!(v["N"], 15);
    let factors = v["factors"].as_array().expect("factors missing");
    assert_eq!(factors.len(), 2);
    let p = factors[0].as_u64().unwrap();
    let q = factors[1].as_u64().unwrap();
    assert_eq!(p * q, 15);
    assert!(p > 1 && q > 1);
}

#[test]
fn shor_factors_21() {
    let v = run(&["--shor", "--N", "21", "--seed", "7", "--format", "json"]);
    let factors = v["factors"].as_array().expect("factors missing");
    let p = factors[0].as_u64().unwrap();
    let q = factors[1].as_u64().unwrap();
    assert_eq!(p * q, 21);
    assert!(p > 1 && q > 1);
}

#[test]
fn shor_rejects_too_large_n() {
    let out = Command::new(binary_path())
        .args(["--shor", "--N", "100", "--format", "json"])
        .current_dir(repo_root())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn omega-run");
    assert!(!out.status.success(), "expected exit code != 0 for N > 63");
}
