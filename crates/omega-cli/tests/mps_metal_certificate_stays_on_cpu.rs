//! `--device metal` on the MPS backend must keep running, and must not claim
//! the f32 two-site contraction is what ran.
//!
//! The numeric half — `discarded_weight` bit-identical to the CPU on a circuit
//! whose bond exceeds the Metal threshold — lives next to
//! `apply_mps_device_hooks` (`device_metal_discarded_weight_matches_cpu_exactly`).
//! Reinstalling `metal_contract_2q` there reddens that test. This file pins the
//! sentence the CLI prints, so the old banner cannot come back beside a hook
//! that the numeric test already rejects.
#![cfg(all(feature = "metal", not(feature = "cuda")))]

use std::path::PathBuf;
use std::process::Command;

fn omega_run() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_omega-run"))
}

fn bell() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/circuits/bell.qasm")
}

fn run(extra_env: Option<(&str, &str)>) -> (bool, String, String) {
    let mut cmd = Command::new(omega_run());
    cmd.args([
        bell().to_str().unwrap(),
        "--backend",
        "mps",
        "--device",
        "metal",
        "--shots",
        "1",
        "--seed",
        "1",
        "--format",
        "json",
    ]);
    if let Some((k, v)) = extra_env {
        cmd.env(k, v);
    } else {
        cmd.env_remove("MPS_METAL_CONTRACT");
    }
    let out = cmd.output().expect("spawn omega-run");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn device_metal_mps_stays_on_cpu_and_says_so() {
    let (ok, stdout, stderr) = run(None);
    assert!(ok, "omega-run failed:\n{stderr}");
    assert!(
        stderr.contains("stays on the CPU"),
        "banner does not say the contraction stays on the CPU:\n{stderr}"
    );
    assert!(
        stderr.contains("not installed"),
        "banner does not say the Metal contraction is not installed:\n{stderr}"
    );
    assert!(
        !stderr.contains("MPS_METAL_CONTRACT is set"),
        "default run printed the opt-in warning:\n{stderr}"
    );
    assert!(
        !stderr.contains("f32 above the bond threshold"),
        "default run still claims the f32 contraction:\n{stderr}"
    );
    let doc: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert_eq!(
        doc["device_used"], "mps-cpu",
        "artifact names a Metal contraction that did not run: {doc}"
    );
}

#[test]
fn metal_contract_opt_in_warns_the_certificate_is_not_a_bound() {
    let (ok, stdout, stderr) = run(Some(("MPS_METAL_CONTRACT", "1")));
    assert!(ok, "opt-in run failed:\n{stderr}");
    assert!(
        stderr.contains("MPS_METAL_CONTRACT is set"),
        "opt-in did not announce itself:\n{stderr}"
    );
    assert!(
        stderr.contains("NOT a bound"),
        "opt-in did not warn that the certificate is not a bound:\n{stderr}"
    );
    let doc: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert_eq!(
        doc["device_used"], "mps-cpu+metal-contract-hook",
        "opt-in artifact did not name the hook: {doc}"
    );
}
