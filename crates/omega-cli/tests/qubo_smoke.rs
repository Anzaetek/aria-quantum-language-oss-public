//! Integration tests for `omega-run --qubo`.

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

fn run_qubo(args: &[&str]) -> String {
    let out = Command::new(binary_path())
        .args(args)
        .current_dir(repo_root())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .expect("spawn omega-run");
    assert!(out.status.success(), "omega-run failed: {:?}", out.status);
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn qubo_maxcut_triangle_matches_brute_force() {
    let stdout = run_qubo(&[
        "--qubo",
        "examples/qubo/maxcut_triangle.json",
        "--qaoa-depth",
        "2",
        "--shots",
        "1024",
        "--seed",
        "42",
        "--optimizer",
        "cma-es",
        "--max-iters",
        "30",
        "--format",
        "json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("json parse");
    assert_eq!(v["mode"], "qubo");
    assert_eq!(v["n"], 3);
    let best_value = v["best"]["value"].as_f64().unwrap();
    assert!(
        (best_value - (-1.0)).abs() < 1e-9,
        "best value should be -1, got {}",
        best_value
    );
    assert_eq!(v["brute_force_check"]["matched"], true);
}

#[test]
fn qubo_portfolio_gradient_finds_optimum() {
    let stdout = run_qubo(&[
        "--qubo",
        "examples/qubo/portfolio_4asset.json",
        "--qaoa-depth",
        "2",
        "--shots",
        "1024",
        "--seed",
        "42",
        "--optimizer",
        "gradient",
        "--max-iters",
        "100",
        "--format",
        "json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("json parse");
    assert_eq!(v["n"], 4);
    // Brute force is deterministic; check that QAOA best matches it.
    let brute_val = v["brute_force_check"]["value"].as_f64().unwrap();
    let best_val = v["best"]["value"].as_f64().unwrap();
    assert!(
        (best_val - brute_val).abs() < 1e-9,
        "QAOA best {} should match brute-force optimum {}",
        best_val,
        brute_val,
    );
}

#[test]
fn qubo_json_shape_is_complete() {
    let stdout = run_qubo(&[
        "--qubo",
        "examples/qubo/maxcut_triangle.json",
        "--qaoa-depth",
        "1",
        "--shots",
        "256",
        "--seed",
        "1",
        "--max-iters",
        "10",
        "--format",
        "json",
    ]);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("json parse");
    // Required top-level keys
    for k in ["mode", "n", "best", "samples", "qaoa", "brute_force_check"] {
        assert!(v.get(k).is_some(), "missing key: {}", k);
    }
    // QAOA shape
    for k in [
        "depth",
        "gamma",
        "beta",
        "iterations",
        "final_expected_cost",
    ] {
        assert!(v["qaoa"].get(k).is_some(), "missing qaoa.{}", k);
    }
    assert_eq!(v["qaoa"]["depth"], 1);
    assert_eq!(v["qaoa"]["gamma"].as_array().unwrap().len(), 1);
    assert_eq!(v["qaoa"]["beta"].as_array().unwrap().len(), 1);

    // Samples: each must have bits/value/count/probability
    let samples = v["samples"].as_array().unwrap();
    assert!(!samples.is_empty());
    for s in samples {
        for k in ["bits", "value", "count", "probability"] {
            assert!(s.get(k).is_some(), "sample missing {}", k);
        }
    }
}
