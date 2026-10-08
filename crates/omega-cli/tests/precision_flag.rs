// SPDX-License-Identifier: Apache-2.0
//! `omega-run --precision f32|f64`: which CUDA statevector arm runs.
//!
//! The flag only means something on `--device cuda --backend statevector`, so
//! everywhere else it is refused rather than ignored: an ignored flag would let
//! a row claim a precision it did not select. On a CUDA build with a device,
//! `--precision f64` must produce the CPU's state to 1e-12 through the user's
//! own path (`--dump-state-npy`) and label its timing line `cuda-f64`.

use std::path::PathBuf;
use std::process::Command;

fn omega_run() -> Command {
    Command::new(env!("CARGO_BIN_EXE_omega-run"))
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("omega-precision-flag-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// `tools/emu_compare/test_bitorder.py`'s circuit at 16 qubits: not
/// reversal-symmetric, so an ordering error moves amplitude.
fn bitorder_qasm() -> PathBuf {
    let p = scratch("bitorder16.qasm");
    std::fs::write(
        &p,
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[16];\n\
         x q[0];\nh q[1];\ncx q[1],q[15];\ncx q[1],q[0];\nu1(0.5) q[15];\nry(0.3) q[7];\ncx q[7],q[3];\n",
    )
    .unwrap();
    p
}

fn refused(args: &[&str]) -> String {
    let qasm = bitorder_qasm();
    let out = omega_run().arg(&qasm).args(args).output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(2),
        "{args:?} must be refused with exit 2; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn precision_is_refused_off_the_cuda_statevector_arm() {
    for args in [
        &["--statevector", "--precision", "f64"][..],
        &["--statevector", "--device", "cpu", "--precision", "f64"][..],
        &["--statevector", "--device", "cpu", "--precision", "f32"][..],
        &["--backend", "mps", "--device", "cuda", "--precision", "f64"][..],
    ] {
        let err = refused(args);
        assert!(
            err.contains("--precision selects the CUDA statevector arm"),
            "{args:?}: {err}"
        );
    }
    let err = refused(&["--statevector", "--device", "cuda", "--precision", "f16"]);
    assert!(err.contains("f32 or f64"), "{err}");
}

/// Complex128 `.npy` as written by `--dump-state-npy` (format 1.0, `<c16`).
#[cfg(feature = "cuda")]
fn read_npy(p: &std::path::Path) -> Vec<(f64, f64)> {
    let b = std::fs::read(p).unwrap();
    assert_eq!(&b[..6], b"\x93NUMPY");
    let hlen = u16::from_le_bytes([b[8], b[9]]) as usize;
    let header = std::str::from_utf8(&b[10..10 + hlen]).unwrap();
    assert!(header.contains("<c16"), "{header}");
    b[10 + hlen..]
        .chunks(16)
        .map(|c| {
            (
                f64::from_le_bytes(c[..8].try_into().unwrap()),
                f64::from_le_bytes(c[8..].try_into().unwrap()),
            )
        })
        .collect()
}

#[cfg(feature = "cuda")]
#[test]
fn f64_arm_dumps_the_cpu_state_and_labels_its_timing() {
    let qasm = bitorder_qasm();
    let (gpu, cpu) = (scratch("gpu.npy"), scratch("cpu.npy"));
    let out = omega_run()
        .arg(&qasm)
        .args([
            "--statevector",
            "--backend",
            "statevector",
            "--device",
            "cuda",
        ])
        .args([
            "--precision",
            "f64",
            "--timing-reps",
            "1",
            "--dump-state-npy",
        ])
        .arg(&gpu)
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() && err.contains("cuda unavailable") {
        eprintln!("SKIP: no CUDA device: {err}");
        return;
    }
    assert!(out.status.success(), "f64 run failed: {err}");
    let timing = err
        .lines()
        .find(|l| l.starts_with("omega-timing:"))
        .unwrap_or_else(|| panic!("no timing line: {err}"));
    assert!(timing.contains(r#""arm":"cuda-f64""#), "{timing}");

    let st = omega_run()
        .arg(&qasm)
        .args([
            "--statevector",
            "--backend",
            "statevector",
            "--device",
            "cpu",
        ])
        .arg("--dump-state-npy")
        .arg(&cpu)
        .status()
        .unwrap();
    assert!(st.success());
    let (g, c) = (read_npy(&gpu), read_npy(&cpu));
    assert_eq!(g.len(), 1 << 16);
    let worst = g
        .iter()
        .zip(&c)
        .map(|(a, b)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt())
        .fold(0.0f64, f64::max);
    assert!(worst <= 1e-12, "cuda-f64 vs cpu worst |Δamp| = {worst:e}");
}
