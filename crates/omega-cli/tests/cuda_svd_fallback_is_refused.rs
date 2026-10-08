// SPDX-License-Identifier: Apache-2.0
//! **An MPS run's CUDA SVD dispatch is reported, and an explicit `--device
//! cuda` run that fell back to the CPU is refused.**
//!
//! `cuda_svd_flat` falls back to the CPU per call, and the result looks the
//! same either way. Before the dispatch counters, `--backend mps --device
//! cuda` printed "Device: cuda" and returned a result whether the SVDs ran on
//! the GPU, partly on the CPU (`STATUS.md` §5.16 measured 20-45% of calls), or
//! entirely on the CPU.
//!
//! The fallback is FORCED here with `CUDA_VISIBLE_DEVICES=""`: the driver then
//! exposes no device, the per-thread context cannot be built, and every SVD
//! falls back with "CUDA context unavailable". That needs no device, so the
//! refusal is exercised on every host with the `cuda` feature; the test that
//! needs a real device skips, with its reason printed, when there is none.
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::process::{Command, Output};

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_omega-run"))
}

/// A process-unique fixture: a 4-qubit entangler, so the MPS chain performs
/// several two-site splits (each one an SVD).
fn circuit() -> PathBuf {
    static FIXTURE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let p = std::env::temp_dir().join(format!(
                "omega_cuda_svd_fixture_{}.qasm",
                std::process::id()
            ));
            std::fs::write(
                &p,
                "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\ncreg c[4];\n\
                 h q[0];\ncx q[0], q[1];\ncx q[1], q[2];\ncx q[2], q[3];\n\
                 rx(0.3) q[1];\ncx q[1], q[2];\n",
            )
            .expect("write fixture");
            p
        })
        .clone()
}

fn run(extra: &[&str], env: &[(&str, &str)]) -> Output {
    Command::new(bin())
        .arg(circuit())
        .args(["--backend", "mps", "--shots", "16", "--seed", "7"])
        .args(extra)
        .envs(env.iter().copied())
        // Keep a caller's OMEGA_DEVICE from deciding what is "explicit".
        .env_remove("OMEGA_DEVICE")
        .output()
        .expect("spawn omega-run")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

const NO_DEVICE: (&str, &str) = ("CUDA_VISIBLE_DEVICES", "");

#[test]
fn an_explicit_cuda_device_refuses_a_run_that_fell_back() {
    let o = run(&["--device", "cuda", "--format", "json"], &[NO_DEVICE]);
    let err = stderr(&o);
    assert_eq!(
        o.status.code(),
        Some(1),
        "an explicit --device cuda run whose SVDs fell back must be refused; stderr:\n{err}"
    );
    assert!(
        o.stdout.is_empty(),
        "a refused run must not print its result; stdout:\n{}",
        String::from_utf8_lossy(&o.stdout)
    );
    // The dispatch line, then the refusal naming the counts and the reason.
    assert!(
        err.contains("cuda svd: 0 on gpu, ")
            && err.contains(" cpu fallbacks (last: CUDA context unavailable"),
        "missing the dispatch line; stderr:\n{err}"
    );
    assert!(
        err.contains("--device cuda was requested but the run could not use it: ")
            && err.contains(
                "MPS bond-compression SVDs fell back to the CPU (last: CUDA context unavailable"
            )
            && err.contains("Refusing rather than returning a CPU result labelled as a device run"),
        "missing the refusal sentence; stderr:\n{err}"
    );
}

/// Automatic selection expressed no expectation to violate: the same fallback
/// under `OMEGA_DEVICE=cuda` runs, and says so on the dispatch line and in the
/// JSON.
#[test]
fn an_automatic_cuda_device_reports_the_fallback_and_runs() {
    let o = Command::new(bin())
        .arg(circuit())
        .args([
            "--backend",
            "mps",
            "--shots",
            "16",
            "--seed",
            "7",
            "--format",
            "json",
        ])
        .env("OMEGA_DEVICE", "cuda")
        .env(NO_DEVICE.0, NO_DEVICE.1)
        .output()
        .expect("spawn omega-run");
    let err = stderr(&o);
    assert!(o.status.success(), "stderr:\n{err}");
    assert!(
        err.contains("cuda svd: 0 on gpu, ")
            && err.contains(" cpu fallbacks (last: CUDA context unavailable"),
        "stderr:\n{err}"
    );
    let doc: serde_json::Value = serde_json::from_slice(&o.stdout).expect("json");
    let d = &doc["cuda_svd"];
    assert_eq!(d["gpu_calls"], 0, "{doc}");
    assert!(d["cpu_fallbacks"].as_u64().unwrap() > 0, "{doc}");
    assert!(
        d["last_fallback"]
            .as_str()
            .unwrap()
            .starts_with("CUDA context unavailable"),
        "{doc}"
    );
    assert!(doc.get("mps_truncation").is_some(), "{doc}");
}

/// With a device, the same explicit run succeeds and reports every SVD on the
/// GPU — the A10 form: it must FAIL, not skip, if a device exists and the GPU
/// path did not run.
#[test]
fn an_explicit_cuda_device_reports_its_gpu_dispatch() {
    if let Err(e) = omega_backend_mps_cuda::CudaSvdContext::try_new() {
        eprintln!("an_explicit_cuda_device_reports_its_gpu_dispatch: SKIPPED — {e}");
        return;
    }
    let o = run(&["--device", "cuda", "--format", "json"], &[]);
    let err = stderr(&o);
    assert!(o.status.success(), "stderr:\n{err}");
    let doc: serde_json::Value = serde_json::from_slice(&o.stdout).expect("json");
    let gpu = doc["cuda_svd"]["gpu_calls"].as_u64().expect("gpu_calls");
    assert!(gpu > 0, "a device is present but no SVD ran on it: {doc}");
    assert_eq!(doc["cuda_svd"]["cpu_fallbacks"], 0, "{doc}");
    assert!(doc["cuda_svd"]["last_fallback"].is_null(), "{doc}");
    assert!(
        err.contains(&format!("cuda svd: {gpu} on gpu, 0 cpu fallbacks\n")),
        "stderr:\n{err}"
    );
}
