// SPDX-License-Identifier: Apache-2.0
//! **A qutrit register is refused by EVERY backend, by name, at the door —
//! never run as qubits.** (PLAN-QUDIT.md Q1)
//!
//! This is an A10-shaped test. If the admission check were absent, a
//! `qreg q [2][3,3];` circuit would evolve `2^2` amplitudes for a `3^2`
//! system, exit 0, and print a plausible number — and every assertion below
//! fails on the exit code, not on a tolerance. The refusal is the whole
//! deliverable: no engine in this workspace evolves a d ≠ 2 wire, and Q1's
//! promise is that each one SAYS so rather than implies it.
//!
//! Two engines have since left the refusing set: `quditsv` (Q2, the exact
//! reference) and `mps` (Q3.2, which now evolves per-site `d`). For both,
//! the tests below assert a positive result — the MPS statevector against
//! quditsv's at 1e-10 — and that where they still refuse (a qubit gate on a
//! qutrit, a Pauli on a qutrit, sampling a qudit circuit) they say the same
//! sentence. Every other engine is still asserted to refuse.
//!
//! Universal means universal. The matrix below is every backend name the
//! CLI dispatches to, in every mode that reaches an engine, plus the three
//! GPU statevector arms. A refusal that lived in "the three backends that
//! happened to be edited" would pass a narrower test and fail this one.
//!
//! Why the message is asserted, not just the status: the CLI's GPU arms
//! catch `Unsupported` and FALL BACK to the CPU statevector unless
//! `--device` was explicit. If the CPU engine did not refuse in turn, a
//! `--backend statevector` run on a CUDA build would land on the CPU path
//! with the qudit circuit and exit 0. The engine sentence in stderr is how
//! this test knows the refusal came from an engine door and not from a
//! flag gate upstream of dispatch.

use std::path::PathBuf;
use std::process::{Command, Output};

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_omega-run"))
}

/// Two qutrits, both touched, with a measure so the sampling mode has
/// something to sample. Gate-wise this is plain QASM 2 (`x`, `cx`, `h`) on
/// purpose: Q1 does not lower DITQASM's own gates, and if the file used one
/// the PARSER would refuse first and this test would pin the wrong door.
const QUTRITS: &str = "DITQASM 2.0;\nqreg q [2][3,3];\ncreg c[2];\n\
                       x q[0];\ncx q[0], q[1];\nh q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n";

/// The same shape with a hopping-friendly body for the sector backend,
/// which refuses `h` on its own grounds — the qudit refusal must win.
const QUTRITS_HOP: &str =
    "DITQASM 2.0;\nqreg q [2][3,3];\ncreg c[2];\nx q[0];\nrbs(0.7) q[0], q[1];\n";

/// One qutrit, one qubit, in one register — wire 1 is a qubit, wire 0 is
/// not. The refusal must name wire 0 and dimension 3, not wire 1.
const MIXED: &str = "DITQASM 2.0;\nqreg q [2][3,2];\ncreg c[2];\nh q[1];\n";

fn fixture(tag: &str, body: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "omega_qudit_refusal_{tag}_{}.qasm",
        std::process::id()
    ));
    std::fs::write(&p, body).expect("write fixture");
    p
}

fn run(path: &std::path::Path, extra: &[&str]) -> Output {
    Command::new(bin())
        .arg(path)
        .args(extra)
        .output()
        .expect("spawn omega-run")
}

/// The one sentence every door speaks. Asserting all four pieces — engine
/// name, register, wire, dimension — is what distinguishes "an engine
/// refused" from "something upstream refused".
fn assert_refused_by(out: &Output, engine: &str, register: &str, wire: u32, dim: u32, what: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stderr.contains("panicked"), "{what} PANICKED:\n{stderr}");
    assert_ne!(
        out.status.code(),
        Some(0),
        "{what}: a d={dim} circuit EXITED 0 — it was run as qubits.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    for needle in [
        format!("{engine}:"),
        format!("register '{register}'"),
        format!("dimension {dim}"),
        format!("wire {wire}"),
        "qubit-only".to_string(),
    ] {
        assert!(
            stderr.contains(&needle),
            "{what}: refusal must contain {needle:?}:\n{stderr}"
        );
    }
    assert!(
        !stdout.contains("<O> =") && !stdout.contains("\"counts\""),
        "{what}: a refusal must not also print a result:\n{stdout}"
    );
}

// ---------------------------------------------------------------------------
// Sampling mode, every backend that samples.
// ---------------------------------------------------------------------------

#[test]
fn sampling_is_refused_on_every_sampling_backend() {
    let path = fixture("sample", QUTRITS);
    // `mps` left this list in Q3.2: it evolves qudits now, and its sampling
    // refusal is a different sentence (the result type, not the engine) —
    // pinned in `mps_sampling_of_a_qudit_circuit_is_refused_for_its_result_type`.
    for (backend, engine) in [("statevector", "statevector"), ("pauli", "pauli")] {
        let out = run(&path, &["--backend", backend, "--shots", "16"]);
        assert_refused_by(
            &out,
            engine,
            "q",
            0,
            3,
            &format!("--backend {backend} --shots"),
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn sector_refuses_the_qudit_before_its_own_gate_check() {
    let path = fixture("sector", QUTRITS_HOP);
    let out = run(
        &path,
        &[
            "--backend",
            "sector",
            "--qasm-dialect",
            "lenient",
            "--shots",
            "16",
        ],
    );
    assert_refused_by(&out, "sector", "q", 0, 3, "--backend sector");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_analytic_statevector_mode_is_refused() {
    let path = fixture("analytic", QUTRITS);
    let out = run(&path, &["--backend", "statevector", "--statevector"]);
    assert_refused_by(&out, "statevector", "q", 0, 3, "--statevector");
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Expectation mode, every backend that has an expectation arm.
// ---------------------------------------------------------------------------

#[test]
fn expectation_is_refused_on_every_expectation_backend() {
    let path = fixture("expect", QUTRITS);
    // `mps` left this list in Q3.2 — see `mps_evolves_a_qudit_circuit_…`
    // below for what it says about this body (the `cx` on qutrits).
    for (backend, engine) in [
        ("statevector", "statevector"),
        ("pauli", "pauli"),
        ("pauliprop", "pauliprop"),
        ("majoranaprop", "majoranaprop"),
        ("stabrank", "stabrank"),
    ] {
        let out = run(&path, &["--backend", backend, "--expectation", "Z0"]);
        assert_refused_by(
            &out,
            engine,
            "q",
            0,
            3,
            &format!("--backend {backend} --expectation"),
        );
    }
    // The sector backend needs a number-conserving body.
    let hop = fixture("expect_sector", QUTRITS_HOP);
    let out = run(
        &hop,
        &[
            "--backend",
            "sector",
            "--qasm-dialect",
            "lenient",
            "--expectation",
            "Z0",
        ],
    );
    assert_refused_by(&out, "sector", "q", 0, 3, "--backend sector --expectation");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&hop);
}

#[test]
fn the_fermionic_expectation_door_is_refused_too() {
    let path = fixture("fermionic", QUTRITS_HOP);
    let out = run(
        &path,
        &[
            "--backend",
            "majoranaprop",
            "--qasm-dialect",
            "lenient",
            "--expectation-fermionic",
            "[0^ 0]",
        ],
    );
    assert_refused_by(&out, "majoranaprop", "q", 0, 3, "--expectation-fermionic");
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Gradient mode.
// ---------------------------------------------------------------------------

#[test]
fn gradient_is_refused() {
    let src = "DITQASM 2.0;\nqreg q [2][3,3];\nrx(theta) q[0];\ncx q[0], q[1];\n";
    let path = fixture("grad", src);
    let out = run(
        &path,
        &[
            "--backend",
            "statevector",
            "--gradient",
            "Z0",
            "--params",
            "0.3",
        ],
    );
    assert_refused_by(&out, "statevector", "q", 0, 3, "--gradient");
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// `auto` must not route around the refusal.
// ---------------------------------------------------------------------------

#[test]
fn auto_backend_selection_routes_a_qudit_circuit_to_quditsv_which_still_refuses_qubit_gates() {
    // Q2 gave `auto` a qudit engine. Before Q2 this row asserted that auto
    // landed on pauli/statevector and was refused there; now it must land
    // on `quditsv` — and quditsv must STILL refuse this body, because `cx`
    // on two qutrits is a qubit gate with no d = 3 meaning (the generalised
    // CX is `csum`). Exit non-zero either way; what changed is who says so,
    // and that it names the gate and the way out.
    let path = fixture("auto_qudit", QUTRITS);
    let out = run(&path, &["--backend", "auto", "--statevector"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    // `info` notices (the preamble) go to stdout; refusals go to stderr.
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert_ne!(
        out.status.code(),
        Some(0),
        "auto ran a qubit gate on qutrits:\n{stderr}"
    );
    assert!(
        stdout.contains("auto: qudit"),
        "auto must say it chose quditsv:\n{stdout}\n{stderr}"
    );
    assert!(
        stderr.contains("quditsv:") && stderr.contains("CX is a qubit gate"),
        "{stderr}"
    );
    assert!(
        stderr.contains("csum"),
        "the refusal must name the generalised gate:\n{stderr}"
    );
    let _ = std::fs::remove_file(&path);

    // And a VALID qutrit body under auto runs — the positive half, without
    // which the row above could pass because DITQASM was unreadable.
    let ok = fixture(
        "auto_qudit_ok",
        "DITQASM 2.0;\nqreg q [2][3,3];\nh q[0];\nx q[1];\ncsum q[0], q[1];\n",
    );
    let out = run(&ok, &["--backend", "auto", "--statevector"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.code(),
        Some(0),
        "auto must run a valid qutrit circuit:\n{stderr}"
    );
    assert!(stdout.contains("auto: qudit"), "{stdout}");
    let _ = std::fs::remove_file(&ok);
}

// ---------------------------------------------------------------------------
// Q3.2: `mps` is no longer a refusing engine. It evolves the qudit circuit
// and must agree with `quditsv` amplitude for amplitude; where it still
// refuses, it refuses for quditsv's reasons in quditsv's words.
// ---------------------------------------------------------------------------

/// A mixed register [3,2,4,3]: the generalised h/x/z on qudit wires, `rxy`
/// on a qutrit and a ququart, `csum` adjacent (3→2) and distant in both
/// orientations (4→3, 3→3 across the chain, 2→3), and qubit gates on the
/// one qubit wire. Every SWAP in the network reshapes between unequal
/// dimensions.
const MIXED_OK: &str = "DITQASM 2.0;\nqreg q [4][3,2,4,3];\n\
                        h q[0];\nh q[2];\nx q[3];\nrxy (0, 2, 1.1, 0.4) q[0];\n\
                        csum q[0], q[1];\nh q[1];\ncsum q[2], q[0];\n\
                        rxy (1, 3, 0.7, -0.2) q[2];\ncsum q[3], q[0];\nz q[3];\n\
                        ry(0.3) q[1];\nt q[1];\ncsum q[1], q[3];\nz q[2];\n";

/// The `amplitudes` of a `--format json` statevector run (the JSON document
/// is the line that starts with `{`; a banner precedes it on stdout).
fn amplitudes(out: &Output, what: &str) -> Vec<(f64, f64)> {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{what} must run:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let line = stdout
        .lines()
        .find(|l| l.starts_with('{'))
        .unwrap_or_else(|| panic!("{what}: no JSON document on stdout:\n{stdout}"));
    let v: serde_json::Value = serde_json::from_str(line).expect("valid JSON");
    v["amplitudes"]
        .as_array()
        .unwrap_or_else(|| panic!("{what}: no amplitudes in {line}"))
        .iter()
        .map(|a| (a[0].as_f64().unwrap(), a[1].as_f64().unwrap()))
        .collect()
}

#[test]
fn mps_statevector_of_a_mixed_qudit_circuit_agrees_with_quditsv() {
    let path = fixture("mps_vs_quditsv", MIXED_OK);
    let reference = amplitudes(
        &run(
            &path,
            &["--backend", "quditsv", "--statevector", "--format", "json"],
        ),
        "quditsv --statevector",
    );
    assert_eq!(reference.len(), 3 * 2 * 4 * 3, "Π dᵢ amplitudes");
    for backend in ["mps", "mps:16", "mps:auto"] {
        let got = amplitudes(
            &run(
                &path,
                &["--backend", backend, "--statevector", "--format", "json"],
            ),
            &format!("{backend} --statevector"),
        );
        assert_eq!(got.len(), reference.len(), "{backend}");
        let worst = got
            .iter()
            .zip(&reference)
            .map(|(a, b)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt())
            .fold(0.0f64, f64::max);
        assert!(
            worst < 1e-10,
            "{backend} vs quditsv: max |Δamp| = {worst:e}"
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn mps_sampling_of_a_qudit_circuit_is_refused_for_its_result_type() {
    let path = fixture("mps_sample", MIXED_OK);
    for backend in ["mps", "mps:8", "quditsv"] {
        let out = run(&path, &["--backend", backend, "--shots", "16"]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!stderr.contains("panicked"), "{backend}: {stderr}");
        assert_ne!(
            out.status.code(),
            Some(0),
            "{backend} sampled a qudit circuit"
        );
        let engine = if backend == "quditsv" {
            "quditsv:"
        } else {
            "mps:"
        };
        assert!(
            stderr.contains(engine)
                && stderr.contains("`Counts` outcomes are bit strings")
                && stderr.contains("--statevector"),
            "{backend}: {stderr}"
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn mps_evolves_a_qudit_circuit_and_refuses_a_qubit_gate_on_a_qutrit_by_name() {
    // The Q1 body (`cx` on two qutrits): mps no longer refuses the REGISTER,
    // it refuses the GATE — the same sentence quditsv gives.
    let path = fixture("mps_cx_qutrit", QUTRITS);
    for backend in ["mps", "quditsv"] {
        let out = run(&path, &["--backend", backend, "--statevector"]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_ne!(out.status.code(), Some(0), "{backend}: {stderr}");
        assert!(
            stderr.contains(&format!(
                "{backend}: CX is a qubit gate and wire 0 has dimension 3"
            )) && stderr.contains("csum"),
            "{backend}: {stderr}"
        );
        assert!(
            !stderr.contains("qubit-only"),
            "{backend}: the Q1 sentence is gone"
        );
    }
    // Expectation: a Pauli on a qutrit wire is refused by both, by name.
    let ok = fixture("mps_expect", MIXED_OK);
    for backend in ["mps", "quditsv"] {
        let out = run(&ok, &["--backend", backend, "--expectation", "Z0"]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_ne!(out.status.code(), Some(0), "{backend}: {stderr}");
        assert!(
            stderr.contains(&format!(
                "{backend}: observable Z0 names wire 0, which has dimension 3"
            )),
            "{backend}: {stderr}"
        );
    }
    // …and on the qubit wire it is a number, the same number.
    let z1 = |backend: &str| {
        let out = run(&ok, &["--backend", backend, "--expectation", "Z1"]);
        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
        assert_eq!(
            out.status.code(),
            Some(0),
            "{backend} Z1:\n{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let line = stdout
            .lines()
            .find(|l| l.contains("<O> ="))
            .unwrap_or_else(|| panic!("{backend}: no <O> line:\n{stdout}"))
            .to_string();
        line.split("<O> =")
            .nth(1)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse::<f64>()
            .unwrap()
    };
    let (a, b) = (z1("mps"), z1("quditsv"));
    assert!((a - b).abs() < 1e-10, "Z1: mps {a} vs quditsv {b}");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&ok);
}

// ---------------------------------------------------------------------------
// The refusal names the RIGHT wire in a mixed register.
// ---------------------------------------------------------------------------

#[test]
fn a_mixed_register_refusal_names_the_qudit_wire_not_the_qubit_one() {
    let path = fixture("mixed", MIXED);
    let out = run(&path, &["--backend", "statevector", "--statevector"]);
    assert_refused_by(&out, "statevector", "q", 0, 3, "mixed [3,2]");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("wire 1"), "wire 1 is a qubit:\n{stderr}");
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// GPU arms: CPU-side admission, before any device is touched.
//
// The CUDA arm is compiled into this binary only under `--features cuda`
// and only dispatches with `--device cuda`, so this test is REAL on akilles
// (`ARIA_CUDA=1 ./ci.sh`) and a registered no-op elsewhere. It says so.
// Metal (macOS) and OpenCL follow the same pattern; the refusal for all
// three lives in the same `refuse_qudits` call at the top of `execute`,
// before `allocate`/`lease`, which the CUDA row here is the only in-CI
// witness for.
// ---------------------------------------------------------------------------

#[test]
fn the_gpu_arms_refuse_on_the_cpu_side_before_allocating() {
    let path = fixture("gpu", QUTRITS);
    let mut checked = 0;
    for (device, engine, feature_on) in [
        ("cuda", "cuda-statevector", cfg!(feature = "cuda")),
        ("metal", "metal-statevector", cfg!(feature = "metal")),
        ("opencl", "opencl-statevector", cfg!(feature = "opencl")),
    ] {
        if !feature_on {
            eprintln!(
                "note: {device} arm not compiled into this omega-run — its qudit refusal \
                 is verifiable only on a box with that feature (cuda: akilles)."
            );
            continue;
        }
        let out = run(
            &path,
            &[
                "--backend",
                "statevector",
                "--device",
                device,
                "--shots",
                "16",
            ],
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !stderr.contains("panicked"),
            "--device {device} PANICKED:\n{stderr}"
        );
        assert_ne!(
            out.status.code(),
            Some(0),
            "--device {device} ran a qudit circuit:\n{stderr}"
        );
        // With an explicit --device the CLI refuses rather than substituting
        // CPU (`refuse_device_fallback`), and forwards the engine's message
        // inside its own. Verified on akilles (RTX PRO 6000): the sentence
        // arrives as `cuda-statevector: register 'q' declares dimension 3 …`
        // — the engine's `name()`, refusing before `allocate`.
        assert!(
            stderr.contains(&format!("{engine}:")),
            "--device {device}: {stderr}"
        );
        assert!(
            stderr.contains("dimension 3"),
            "--device {device}: {stderr}"
        );
        checked += 1;
    }
    eprintln!("gpu arms checked: {checked}");
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Control: the same file with `[2,2]` runs. Without this, every row above
// could pass because DITQASM itself was unreadable.
// ---------------------------------------------------------------------------

#[test]
fn the_same_circuit_declared_as_qubits_runs_on_every_backend() {
    let body = QUTRITS.replace("[2][3,3]", "[2][2,2]");
    let path = fixture("control", &body);
    for backend in ["statevector", "mps", "pauli"] {
        let out = run(&path, &["--backend", backend, "--shots", "16"]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(0),
            "control: DITQASM [2][2,2] must RUN on {backend}:\n{stderr}"
        );
        assert!(
            !stderr.contains("qubit-only"),
            "{backend}: no refusal expected:\n{stderr}"
        );
    }
    let out = run(&path, &["--backend", "statevector", "--expectation", "Z0"]);
    assert_eq!(out.status.code(), Some(0));
    let _ = std::fs::remove_file(&path);
}
