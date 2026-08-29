// SPDX-License-Identifier: Apache-2.0
//! **`--multi-control` must not be accepted and then silently discarded.**
//!
//! # What this was before
//!
//! `with_multi_control` is implemented by exactly one backend,
//! `CudaStatevectorBackend`. Both call sites in the CLI sit behind
//! `#[cfg(feature = "cuda")]`, so on a default build the flag was parsed,
//! validated against the mode table, and then dropped on the floor without a
//! word. `clippy` had been reporting it for some time as "value assigned to
//! `multi_control` is never read" — in a crate CI never linted.
//!
//! `GATE-EXACTNESS.md` had already refused to put `multi_control` on
//! `QuantumExecuteReq` because an inert field "advertis[es] a capability no code
//! path can honour". The CLI shipped precisely that, one directory over.
//!
//! # Why a notice rather than a refusal
//!
//! The CPU statevector applies `CCX`/`CSwap` as a direct subspace permutation.
//! That makes the two modes asymmetric, and the asymmetry is the point:
//!
//! * `exact` off CUDA gets the caller the numbers they asked for by another
//!   route — refusing would break scripts that pass the flag uniformly across
//!   devices for no numeric reason.
//! * `decompose` off CUDA does NOT — the caller asked for the 15-gate chain and
//!   got the permutation. It is also the default value, so it is what anyone
//!   comparing a CPU run against a CUDA/Metal decomposed reference will hit.
//!
//! So the notice distinguishes the two, and these tests assert that it does.
//!
//! # The property that matters most
//!
//! `numbers_are_unchanged_by_the_flag` is the real gate. A notice that also
//! perturbed the result would be a worse bug than the silence it replaced.

use std::process::Command;

/// Returns (stdout, stdout+stderr, success). `info()` routes to stdout in text
/// mode and stderr in machine modes, so assertions use the combined streams —
/// what matters is that the user is told, not which stream carries it.
fn omega_run(args: &[&str]) -> (String, String, bool) {
    let exe = env!("CARGO_BIN_EXE_omega-run");
    let out = Command::new(exe)
        .args(args)
        .output()
        .expect("run omega-run");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let both = format!("{stdout}\n{stderr}");
    (stdout, both, out.status.success())
}

/// Both controls set, so the Toffoli fires and `<Z2>` is exactly -1. A circuit
/// where CCX does nothing would pass these tests even if the gate were skipped.
fn ccx_circuit() -> String {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\n\
               x q[0];\nx q[1];\nccx q[0], q[1], q[2];\n";
    let path = std::env::temp_dir().join("multi_control_notice_ccx.qasm");
    std::fs::write(&path, src).expect("write fixture");
    path.to_string_lossy().into_owned()
}

/// The misleading case: the caller asked for the 15-gate chain and this run
/// cannot give it. The notice must say NOT honoured, not merely mention itself.
#[test]
fn decompose_off_cuda_says_it_was_not_honoured() {
    let path = ccx_circuit();
    let (_out, both, ok) =
        omega_run(&[&path, "--expectation", "Z2", "--multi-control", "decompose"]);
    assert!(ok, "the run itself must still succeed:\n{both}");
    assert!(
        both.contains("--multi-control decompose"),
        "passing a mode this run cannot honour must be reported:\n{both}"
    );
    assert!(
        both.contains("NOT honoured"),
        "the notice must say the mode was not honoured, since the numbers really \
         do differ from a CUDA/Metal decompose run:\n{both}"
    );
}

/// The benign case: `exact` is what the CPU already does. The caller is still
/// told the switch did not apply, but must not be warned about numbers that are
/// in fact exactly what they asked for.
#[test]
fn exact_off_cuda_says_the_numbers_are_unaffected() {
    let path = ccx_circuit();
    let (_out, both, ok) = omega_run(&[&path, "--expectation", "Z2", "--multi-control", "exact"]);
    assert!(ok, "the run itself must still succeed:\n{both}");
    assert!(
        both.contains("--multi-control exact"),
        "the mode not applying must still be reported:\n{both}"
    );
    assert!(
        both.contains("unaffected"),
        "'exact' off CUDA gets the numbers it asked for; the notice must not \
         imply otherwise:\n{both}"
    );
    assert!(
        !both.contains("NOT honoured"),
        "'exact' IS effectively honoured here — crying wolf trains users to \
         ignore the decompose case, which is the one that matters:\n{both}"
    );
}

/// The over-eager direction. A guard that fires when the flag was never passed
/// would put a line about CUDA in front of every CPU user forever.
#[test]
fn no_flag_means_no_notice() {
    let path = ccx_circuit();
    let (_out, both, ok) = omega_run(&[&path, "--expectation", "Z2"]);
    assert!(ok, "{both}");
    assert!(
        !both.contains("--multi-control"),
        "no notice may be printed when the flag was never passed:\n{both}"
    );
}

/// The notice is a notice. Whatever it says, the answer must be bit-identical
/// to the run without the flag — including the `-1` that proves CCX fired.
#[test]
fn numbers_are_unchanged_by_the_flag() {
    let path = ccx_circuit();
    let value = |args: &[&str]| -> String {
        let mut a = vec![path.as_str(), "--expectation", "Z2"];
        a.extend_from_slice(args);
        let (out, both, ok) = omega_run(&a);
        assert!(ok, "{both}");
        // Drop the notice itself; compare only the computed output.
        out.lines()
            .filter(|l| !l.contains("--multi-control"))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let bare = value(&[]);
    assert!(
        bare.contains("-1"),
        "both controls are set, so <Z2> must be -1 — if it is not, this circuit \
         is not exercising CCX at all:\n{bare}"
    );
    assert_eq!(
        bare,
        value(&["--multi-control", "exact"]),
        "--multi-control exact must not change the result off CUDA"
    );
    assert_eq!(
        bare,
        value(&["--multi-control", "decompose"]),
        "--multi-control decompose must not change the result off CUDA"
    );
}
