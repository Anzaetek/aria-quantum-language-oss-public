// SPDX-License-Identifier: Apache-2.0
//! The GPU lane's checks that need no GPU.
//!
//! The fusion check needs the cuStateVec venv's numpy (`EMU_GPU_CUSV_PY`) and
//! is a registered skip without it; `EMU_GPU_REQUIRE=1`, as akilles sets for
//! this lane, turns that skip back into a failure.

use std::path::PathBuf;
use std::process::Command;

fn venv_python() -> Option<String> {
    match std::env::var("EMU_GPU_CUSV_PY") {
        Ok(p) => Some(p),
        Err(_) => {
            if std::env::var("EMU_GPU_REQUIRE").as_deref() == Ok("1") {
                panic!("EMU_GPU_REQUIRE=1 but EMU_GPU_CUSV_PY is unset");
            }
            eprintln!("SKIP: EMU_GPU_CUSV_PY unset, so the fusion check cannot run");
            None
        }
    }
}

/// The cuStateVec arm's host-side fusion equals gate-by-gate application at
/// f64 on 42 circuits, including a CX joining a block of the opposite
/// orientation and single-qubit gates folding in from both sides. Dropping the
/// orientation permutation, or swapping the embedding's kron order, fails it.
#[test]
fn fused_blocks_equal_gate_by_gate_application() {
    let Some(py) = venv_python() else { return };
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = Command::new(py)
        .arg(dir.join("check_fusion.py"))
        .current_dir(&dir)
        // No .pyc: a cached module written in the same second as an edit of
        // the same size is reused, and would test the wrong fusion.
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .expect("run check_fusion.py");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "fusion check failed: {stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.starts_with("ok: "), "unexpected output: {stdout}");
}
