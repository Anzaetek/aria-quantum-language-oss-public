// SPDX-License-Identifier: Apache-2.0
//! Shipped fermionic and qudit examples, through the CLI, against the
//! numbers in docs/QUDIT_EXAMPLES.md and docs/FERMIONIC_EXAMPLES.md.
//!
//! Analytic references are asserted here and need no Python. An OpenFermion
//! or ffsim rerun of the same files is a registered skip in `ci.sh` when
//! that venv is absent — this file does not skip them by going quiet.

use std::path::PathBuf;
use std::process::{Command, Output};

fn bin() -> PathBuf {
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

fn run(args: &[&str]) -> Output {
    Command::new(bin())
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("spawn omega-run")
}

fn json_line(out: &Output, what: &str) -> serde_json::Value {
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
        .unwrap_or_else(|| panic!("{what}: no JSON on stdout:\n{stdout}"));
    serde_json::from_str(line).unwrap_or_else(|e| panic!("{what}: {e}: {line}"))
}

fn amplitudes_of(doc: &serde_json::Value, what: &str) -> Vec<(f64, f64)> {
    doc["amplitudes"]
        .as_array()
        .unwrap_or_else(|| panic!("{what}: no amplitudes"))
        .iter()
        .map(|a| (a[0].as_f64().unwrap(), a[1].as_f64().unwrap()))
        .collect()
}

fn inv_sqrt3() -> f64 {
    1.0 / 3.0_f64.sqrt()
}

const GHZ_TOL: f64 = 1e-12;

fn assert_qutrit_ghz(amps: &[(f64, f64)], what: &str) {
    assert_eq!(amps.len(), 27, "{what}: 3^3 amplitudes, got {}", amps.len());
    for (i, (re, im)) in amps.iter().enumerate() {
        let expect = if i == 0 || i == 13 || i == 26 {
            inv_sqrt3()
        } else {
            0.0
        };
        let err = (re - expect).abs() + im.abs();
        assert!(
            err < GHZ_TOL,
            "{what}: index {i} got ({re}, {im}), expected ({expect}, 0)"
        );
    }
}

#[test]
fn e4_qutrit_ghz_quditsv_matches_one_over_sqrt_three() {
    let out = run(&[
        "examples/qudit/qutrit_ghz.ditqasm",
        "--backend",
        "quditsv",
        "--statevector",
        "--format",
        "json",
    ]);
    let doc = json_line(&out, "E4 quditsv");
    assert_eq!(doc["quditsv"]["exact"], true, "quditsv certificate");
    assert_eq!(doc["quditsv"]["dropped_mass"], 0.0);
    assert_qutrit_ghz(&amplitudes_of(&doc, "E4 quditsv"), "E4 quditsv");
}

#[test]
fn e4_qutrit_ghz_mps_agrees_in_the_exact_regime() {
    let out = run(&[
        "examples/qudit/qutrit_ghz.ditqasm",
        "--backend",
        "mps",
        "--statevector",
        "--format",
        "json",
    ]);
    let doc = json_line(&out, "E4 mps");
    let discarded = doc["mps_truncation"]["discarded_weight"]
        .as_f64()
        .expect("mps certificate");
    assert!(
        discarded == 0.0,
        "E4 mps discarded_weight {discarded} — not the exact regime"
    );
    assert_qutrit_ghz(&amplitudes_of(&doc, "E4 mps"), "E4 mps");
}

#[test]
fn e4_auto_routes_the_qutrit_register_to_quditsv() {
    let out = run(&[
        "examples/qudit/qutrit_ghz.ditqasm",
        "--backend",
        "auto",
        "--statevector",
    ]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "E4 auto:\n{stdout}\n{stderr}");
    assert!(
        stdout.contains(
            "auto: qudit register 'q' (dimension 3 on wire 0) -> quditsv (exact, dense mixed-radix)"
        ),
        "E4 auto preamble:\n{stdout}"
    );
    assert!(
        stdout.contains("|0>:") || stdout.contains("0.577"),
        "E4 auto must print the state:\n{stdout}"
    );
}

#[test]
fn e4_shots_are_refused_because_counts_are_bit_strings() {
    let out = run(&[
        "examples/qudit/qutrit_ghz.ditqasm",
        "--backend",
        "quditsv",
        "--shots",
        "100",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_ne!(
        out.status.code(),
        Some(0),
        "E4 shots ran:\n{stdout}\n{stderr}"
    );
    assert!(
        stderr.contains("`Counts` outcomes are bit strings"),
        "E4 shots refusal:\n{stderr}"
    );
    assert!(
        !stdout.contains("0.577") && !stdout.contains("\"amplitudes\""),
        "E4 shots must not also print the state:\n{stdout}"
    );
}

// ---------------------------------------------------------------------------
// E1 — H2 / STO-3G at R = 0.7414 Å.
// ---------------------------------------------------------------------------

/// Digits printed by `tools/fermionic_examples/gen_h2_op.py` from OpenFermion's
/// shipped MolecularData file. The literature print is a separate constant.
fn h2_script_energies() -> (f64, f64) {
    let text = std::fs::read_to_string(repo_root().join("examples/fermionic/h2_reference.txt"))
        .expect("h2_reference.txt");
    let mut hf = None;
    let mut fci = None;
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("E_HF ") {
            hf = Some(v.trim().parse::<f64>().unwrap());
        }
        if let Some(v) = line.strip_prefix("E_FCI ") {
            fci = Some(v.trim().parse::<f64>().unwrap());
        }
    }
    (hf.expect("E_HF"), fci.expect("E_FCI"))
}

/// Seeley–Richard–Love, J. Chem. Phys. 137, 224109 (2012), as printed for
/// this geometry: six decimals. The OpenFermion file is longer; 1e-4 is the
/// tolerance against this print, 1e-6 against the file.
const H2_LIT_HF: f64 = -1.116685;
const H2_LIT_FCI: f64 = -1.137270;

fn assert_h2_energy(got: f64, script: f64, lit: f64, what: &str) {
    assert!(
        (got - script).abs() < 1e-6,
        "{what}: measured {got} vs the OpenFermion file {script}"
    );
    assert!(
        (got - lit).abs() < 1e-4,
        "{what}: measured {got} vs the literature print {lit}"
    );
}

fn h2_expectation(circuit: &str, backend: &str) -> serde_json::Value {
    fermionic_value(
        &format!("examples/fermionic/{circuit}.qasm"),
        "examples/fermionic/h2_op.txt",
        backend,
        &[],
        &format!("E1 {circuit} {backend}"),
    )
}

fn h2_fqasm_expectation(circuit: &str, backend: &str) -> serde_json::Value {
    fermionic_value(
        &format!("examples/fermionic/{circuit}.fqasm"),
        "examples/fermionic/h2_op.txt",
        backend,
        &[],
        &format!("E1 {circuit}.fqasm {backend}"),
    )
}

/// FermionicQASM and its QASM2 twin must agree with each other, and that
/// common value must sit on the published reference. Agreeing with the twin
/// alone would not say either one is right.
fn assert_twin_and_reference(fq: f64, qasm: f64, reference_ok: impl FnOnce(f64), what: &str) {
    assert!(
        (fq - qasm).abs() < 1e-9,
        "{what}: FermionicQASM {fq} disagrees with QASM2 {qasm}"
    );
    reference_ok(fq);
}

#[test]
fn e1_h2_hf_and_fci_match_the_openfermion_file() {
    let (script_hf, script_fci) = h2_script_energies();
    let mut sv_hf = None;
    let mut sv_fci = None;
    for backend in ["statevector", "mps", "majoranaprop"] {
        let hf = h2_expectation("h2_hf", backend);
        let fci = h2_expectation("h2_ground", backend);
        let hf_v = hf["value"].as_f64().unwrap();
        let fci_v = fci["value"].as_f64().unwrap();
        assert_h2_energy(hf_v, script_hf, H2_LIT_HF, &format!("E1 HF {backend}"));
        assert_h2_energy(fci_v, script_fci, H2_LIT_FCI, &format!("E1 FCI {backend}"));
        if backend == "statevector" {
            sv_hf = Some(hf_v);
            sv_fci = Some(fci_v);
        }
        if backend == "majoranaprop" {
            for (label, doc, sv) in [("HF", &hf, sv_hf.unwrap()), ("FCI", &fci, sv_fci.unwrap())] {
                let cert = &doc["majoranaprop_truncation"];
                assert_eq!(
                    cert["seed_basis"], "ladder",
                    "E1 {label} must be seeded from the ladder operator"
                );
                let dropped = cert["dropped_mass"].as_f64().unwrap();
                assert_eq!(dropped, 0.0, "E1 {label} dropped_mass {dropped}");
                assert_eq!(cert["exact"], true, "E1 {label} certificate");
                let v = doc["value"].as_f64().unwrap();
                assert!(
                    (v - sv).abs() <= dropped + 1e-12,
                    "E1 {label} majoranaprop {v} is outside the statevector value {sv} ± {dropped}"
                );
            }
        }
    }
}

#[test]
fn e1_fermionicqasm_agrees_with_qasm2_and_the_openfermion_file() {
    let (script_hf, script_fci) = h2_script_energies();
    for (circuit, script, lit, label) in [
        ("h2_hf", script_hf, H2_LIT_HF, "HF"),
        ("h2_ground", script_fci, H2_LIT_FCI, "FCI"),
    ] {
        for backend in ["statevector", "mps", "majoranaprop"] {
            let qasm = h2_expectation(circuit, backend);
            let fq = h2_fqasm_expectation(circuit, backend);
            let qv = qasm["value"].as_f64().unwrap();
            let fv = fq["value"].as_f64().unwrap();
            let what = format!("E1 {label} {backend}");
            assert_twin_and_reference(fv, qv, |v| assert_h2_energy(v, script, lit, &what), &what);
        }
    }
}

// ---------------------------------------------------------------------------
// E2 — 2-site Fermi-Hubbard, t = 1, U = 4, two electrons.
// ---------------------------------------------------------------------------

/// (U - sqrt(U^2 + 16 t^2)) / 2 at t = 1, U = 4, which is 2 - 2*sqrt(2).
fn hubbard_closed_form() -> f64 {
    2.0 - 2.0 * 2.0_f64.sqrt()
}

fn fermionic_value(
    file: &str,
    op_rel: &str,
    backend: &str,
    extra: &[&str],
    what: &str,
) -> serde_json::Value {
    let op = std::fs::read_to_string(repo_root().join(op_rel)).expect(op_rel);
    // The circuit path is argv[1]. A flag placed in front of the file is
    // read as the circuit.
    let mut cmd = vec![
        file.to_string(),
        "--backend".to_string(),
        backend.to_string(),
        "--expectation-fermionic".to_string(),
        op,
        "--format".to_string(),
        "json".to_string(),
    ];
    cmd.extend(extra.iter().map(|s| (*s).to_string()));
    let out = Command::new(bin())
        .current_dir(repo_root())
        .args(&cmd)
        .output()
        .expect("spawn omega-run");
    json_line(&out, what)
}

fn assert_hubbard_energy(got: f64, what: &str) {
    let closed = hubbard_closed_form();
    assert!(
        (got - closed).abs() < 1e-9,
        "{what}: measured {got} vs 2 - 2*sqrt(2) = {closed}"
    );
}

fn hubbard_expectation(backend: &str) -> serde_json::Value {
    fermionic_value(
        "examples/fermionic/hubbard2.qasm",
        "examples/fermionic/hubbard2_op.txt",
        backend,
        &["--qasm-dialect", "lenient"],
        &format!("E2 hubbard {backend}"),
    )
}

fn hubbard_fqasm_expectation(backend: &str) -> serde_json::Value {
    fermionic_value(
        "examples/fermionic/hubbard2.fqasm",
        "examples/fermionic/hubbard2_op.txt",
        backend,
        &[],
        &format!("E2 hubbard.fqasm {backend}"),
    )
}

#[test]
fn e2_hubbard_half_filling_matches_the_closed_form() {
    let mut sv = None;
    for backend in ["statevector", "mps", "majoranaprop"] {
        let doc = hubbard_expectation(backend);
        let v = doc["value"].as_f64().unwrap();
        assert_hubbard_energy(v, &format!("E2 {backend}"));
        if backend == "statevector" {
            sv = Some(v);
        }
        if backend == "majoranaprop" {
            let cert = &doc["majoranaprop_truncation"];
            assert_eq!(
                cert["seed_basis"], "ladder",
                "E2 must be seeded from the ladder operator"
            );
            let dropped = cert["dropped_mass"].as_f64().unwrap();
            assert_eq!(dropped, 0.0, "E2 dropped_mass {dropped}");
            assert_eq!(cert["exact"], true, "E2 certificate");
            assert!(
                (v - sv.unwrap()).abs() <= dropped + 1e-12,
                "E2 majoranaprop {v} is outside the statevector value {} ± {dropped}",
                sv.unwrap()
            );
        }
    }
}

#[test]
fn e2_fermionicqasm_agrees_with_qasm2_and_the_closed_form() {
    for backend in ["statevector", "mps", "majoranaprop"] {
        let qasm = hubbard_expectation(backend);
        let fq = hubbard_fqasm_expectation(backend);
        let qv = qasm["value"].as_f64().unwrap();
        let fv = fq["value"].as_f64().unwrap();
        let what = format!("E2 {backend}");
        assert_twin_and_reference(fv, qv, |v| assert_hubbard_energy(v, &what), &what);
    }
}

// ---------------------------------------------------------------------------
// E3 — Kitaev chain at the sweet spot t = Δ = 1, μ = 0.
// ---------------------------------------------------------------------------

fn kitaev_closed_form(n: u32) -> f64 {
    -((n - 1) as f64)
}

fn assert_kitaev_energy(got: f64, n: u32, what: &str) {
    let closed = kitaev_closed_form(n);
    assert!(
        (got - closed).abs() < 1e-9,
        "{what}: measured {got} vs -(N-1) = {closed}"
    );
}

fn kitaev_expectation(circuit: &str, backend: &str, extra: &[&str]) -> serde_json::Value {
    let n = circuit
        .trim_start_matches("kitaev_n")
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>();
    fermionic_value(
        &format!("examples/fermionic/{circuit}.qasm"),
        &format!("examples/fermionic/kitaev_op_n{n}.txt"),
        backend,
        extra,
        &format!("E3 {circuit} {backend}"),
    )
}

fn kitaev_n2_fqasm_expectation(backend: &str) -> serde_json::Value {
    fermionic_value(
        "examples/fermionic/kitaev_n2.fqasm",
        "examples/fermionic/kitaev_op_n2.txt",
        backend,
        &[],
        &format!("E3 kitaev_n2.fqasm {backend}"),
    )
}

#[test]
fn e3_kitaev_matches_the_closed_form() {
    // N = 2 and N = 8 have a dense statevector oracle. N = 64 does not:
    // the assertion there is the analytic value plus this backend's own
    // exact certificate.
    for (circuit, n, dense) in [
        ("kitaev_n2", 2u32, true),
        ("kitaev_n2_partner", 2, true),
        ("kitaev_n8", 8, true),
        ("kitaev_n8_partner", 8, true),
        ("kitaev_n64", 64, false),
        ("kitaev_n64_partner", 64, false),
    ] {
        let mp = kitaev_expectation(circuit, "majoranaprop", &[]);
        let v = mp["value"].as_f64().unwrap();
        assert_kitaev_energy(v, n, &format!("E3 {circuit} majoranaprop"));
        let cert = &mp["majoranaprop_truncation"];
        assert_eq!(cert["seed_basis"], "ladder", "E3 {circuit} seed");
        let dropped = cert["dropped_mass"].as_f64().unwrap();
        assert_eq!(dropped, 0.0, "E3 {circuit} dropped_mass {dropped}");
        assert_eq!(cert["exact"], true, "E3 {circuit} certificate");
        if dense {
            let sv = kitaev_expectation(circuit, "statevector", &[]);
            let sv_v = sv["value"].as_f64().unwrap();
            assert_kitaev_energy(sv_v, n, &format!("E3 {circuit} statevector"));
            assert!(
                (v - sv_v).abs() <= dropped + 1e-12,
                "E3 {circuit} majoranaprop {v} is outside the statevector value {sv_v}"
            );
        }
    }
}

#[test]
fn e3_kitaev_n8_with_one_t_is_covered_by_the_length_cut() {
    let sv = kitaev_expectation("kitaev_n8_t", "statevector", &[]);
    let sv_v = sv["value"].as_f64().unwrap();
    let mp = kitaev_expectation("kitaev_n8_t", "majoranaprop", &["--max-length", "4"]);
    let v = mp["value"].as_f64().unwrap();
    let dropped = mp["majoranaprop_truncation"]["dropped_mass"]
        .as_f64()
        .unwrap();
    assert!(
        dropped > 0.0,
        "E3 T coda dropped nothing, so the cut was not exercised: {dropped}"
    );
    assert!(
        (sv_v - v).abs() <= dropped + 1e-9,
        "E3 T coda: statevector {sv_v} is outside {v} ± {dropped}"
    );
}

#[test]
fn e3_n2_fermionicqasm_agrees_with_qasm2_and_prepares_the_one_particle_state() {
    // N = 8 and N = 64 have no FermionicQASM program: the ground space is
    // span{|+>^N, |->^N}, and no vector in it has one particle number.
    // Every gate after `load` conserves particle number, so the energy
    // -(N-1) is not reachable. N = 2 is the exception. The odd sector is
    // one particle, and its ground state has energy -1, the same number
    // as kitaev_n2.qasm. The states themselves differ.
    for backend in ["statevector", "mps", "majoranaprop"] {
        let qasm = kitaev_expectation("kitaev_n2", backend, &[]);
        let fq = kitaev_n2_fqasm_expectation(backend);
        let qv = qasm["value"].as_f64().unwrap();
        let fv = fq["value"].as_f64().unwrap();
        let what = format!("E3 N=2 {backend}");
        assert_twin_and_reference(fv, qv, |v| assert_kitaev_energy(v, 2, &what), &what);
    }
    let out = run(&[
        "examples/fermionic/kitaev_n2.fqasm",
        "--backend",
        "statevector",
        "--statevector",
        "--format",
        "json",
    ]);
    let doc = json_line(&out, "E3 N=2 state");
    let amps = amplitudes_of(&doc, "E3 N=2 state");
    let s = 1.0 / 2.0_f64.sqrt();
    assert_eq!(amps.len(), 4, "E3 N=2: two qubits");
    for (i, (re, im)) in amps.iter().enumerate() {
        let expect = if i == 1 || i == 2 { s } else { 0.0 };
        let err = (re - expect).abs() + im.abs();
        assert!(
            err < 1e-12,
            "E3 N=2 amplitude {i}: got ({re}, {im}), expected ({expect}, 0)"
        );
    }
}

// ---------------------------------------------------------------------------
// E3, truncated: T gates on the Kitaev ground state |+>^N, read through the
// stabilizer-rank backend with its state forced below the exact rank.
//
// T on a set S of sites keeps a product state, with <X_j> = 1/sqrt(2) on S
// and 1 elsewhere. The energy -sum X_i X_{i+1} is then
// -sum_i c_i c_{i+1} at any N. That closed form is the reference wherever
// the dense statevector cannot run. majoranaprop, a different engine in this
// tree, also returns it exactly, with dropped_mass 0.
//
// One knob is used per row and the test names it. `--max-chi` truncates and
// reports the bound. `--max-branches` refuses and returns no value. Only the
// former appears with a number.
// ---------------------------------------------------------------------------

fn kitaev_t_closed_form(n: usize, t_sites: &[usize]) -> f64 {
    let c = |i: usize| {
        if t_sites.contains(&i) {
            1.0 / 2.0_f64.sqrt()
        } else {
            1.0
        }
    };
    -(0..n - 1).map(|i| c(i) * c(i + 1)).sum::<f64>()
}

/// The three conjuncts of the truncation witness, asserted together:
/// mass was dropped, the bound is narrower than the range it lives in, and
/// the error it bounds is real and inside it. `|value| > 0.1` keeps the row
/// off the zero expectations that would make all three easy.
fn assert_truncation_witness(doc: &serde_json::Value, reference: f64, max_chi: u64, what: &str) {
    let cert = &doc["stabrank_truncation"];
    let v = doc["value"].as_f64().unwrap();
    let m = cert["state_dropped_mass"].as_f64().unwrap();
    let bound = cert["expectation_error_bound"].as_f64().unwrap();
    let r = cert["observable_range"].as_f64().unwrap();
    let err = (v - reference).abs();
    assert_eq!(
        cert["max_chi"], max_chi,
        "{what}: the knob is --max-chi {max_chi}"
    );
    assert_eq!(cert["exact"], false, "{what}: certificate claims exact");
    assert!(v.abs() > 0.1, "{what}: value {v} is too near zero to test");
    assert!(m > 0.0, "{what}: nothing dropped, m = {m}");
    assert!(
        bound < r + v.abs(),
        "{what}: bound {bound} is vacuous against R + |value| = {}",
        r + v.abs()
    );
    assert!(
        err > 0.0 && err <= bound,
        "{what}: |value - reference| = {err} is not in (0, {bound}]; value {v}, reference {reference}"
    );
}

fn fermionic_refusal(file: &str, op_rel: &str, backend: &str, extra: &[&str]) -> String {
    let op = std::fs::read_to_string(repo_root().join(op_rel)).expect(op_rel);
    let mut args = vec![file, "--backend", backend, "--expectation-fermionic", &op];
    args.extend_from_slice(&["--format", "json"]);
    args.extend_from_slice(extra);
    let out = run(&args);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert_ne!(
        out.status.code(),
        Some(0),
        "{file} {extra:?} must be refused:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        !stdout.lines().any(|l| l.starts_with('{')),
        "{file} {extra:?}: a refusal printed a value:\n{stdout}"
    );
    stderr
}

/// Majoranaprop returns the product-state circuits exactly. Checked against
/// the closed form so the engine used as a reference is itself pinned.
fn assert_majoranaprop_exact(circuit: &str, op_rel: &str, reference: f64, what: &str) {
    let doc = fermionic_value(
        &format!("examples/fermionic/{circuit}.qasm"),
        op_rel,
        "majoranaprop",
        &[],
        what,
    );
    let v = doc["value"].as_f64().unwrap();
    let cert = &doc["majoranaprop_truncation"];
    assert_eq!(cert["exact"], true, "{what}: majoranaprop certificate");
    assert_eq!(
        cert["dropped_mass"].as_f64().unwrap(),
        0.0,
        "{what}: dropped"
    );
    assert!(
        (v - reference).abs() < 1e-9,
        "{what}: majoranaprop {v} vs closed form {reference}"
    );
}

#[test]
fn e3_truncated_n8_two_t_bound_holds_against_dense() {
    // Reference: the dense statevector, itself checked against the closed form.
    let sv = kitaev_expectation("kitaev_n8_t2", "statevector", &[]);
    let sv_v = sv["value"].as_f64().unwrap();
    let doc = kitaev_expectation("kitaev_n8_t2", "stabrank", &["--max-chi", "3"]);
    assert_truncation_witness(&doc, sv_v, 3, "E3 N=8 two T, --max-chi 3");
    let reference = kitaev_t_closed_form(8, &[1, 5]);
    assert!(
        (sv_v - reference).abs() < 1e-9,
        "E3 N=8 two T: statevector {sv_v} vs closed form {reference}"
    );
}

#[test]
fn e3_truncated_beyond_dense_reach_bound_holds_against_the_closed_form() {
    // Reference: the closed form, and majoranaprop exact. No dense run:
    // 2^64 and 2^256 amplitudes.
    for (circuit, n, sites) in [
        ("kitaev_n64_t2", 64, [10, 40]),
        ("kitaev_n256_t2", 256, [10, 200]),
    ] {
        let reference = kitaev_t_closed_form(n, &sites);
        let op = format!("examples/fermionic/kitaev_op_n{n}.txt");
        let doc = kitaev_expectation(circuit, "stabrank", &["--max-chi", "3"]);
        assert_truncation_witness(&doc, reference, 3, &format!("E3 {circuit}, --max-chi 3"));
        assert_majoranaprop_exact(circuit, &op, reference, &format!("E3 {circuit}"));
        let exact = kitaev_expectation(circuit, "stabrank", &[]);
        let ev = exact["value"].as_f64().unwrap();
        assert_eq!(
            exact["stabrank_truncation"]["exact"], true,
            "E3 {circuit} untruncated"
        );
        assert!(
            (ev - reference).abs() < 1e-9,
            "E3 {circuit}: untruncated stabrank {ev} vs closed form {reference}"
        );
    }
}

#[test]
fn e3_truncated_single_bond_is_the_tight_row() {
    // Observable range R = 1, so the bound is not inflated by N - 1.
    // Reference: the closed form -c_10 c_11, and majoranaprop exact.
    let op = "examples/fermionic/kitaev_bond_10_11.txt";
    let reference = -1.0 / 2.0_f64.sqrt();
    let doc = fermionic_value(
        "examples/fermionic/kitaev_n64_t2.qasm",
        op,
        "stabrank",
        &["--max-chi", "3"],
        "E3 bond(10,11) stabrank",
    );
    assert_eq!(
        doc["stabrank_truncation"]["observable_range"], 1.0,
        "E3 bond R"
    );
    assert_truncation_witness(&doc, reference, 3, "E3 bond(10,11), --max-chi 3");
    assert_majoranaprop_exact("kitaev_n64_t2", op, reference, "E3 bond(10,11)");
}

#[test]
fn e3_truncated_to_one_branch_is_refused_as_vacuous() {
    // --max-chi 1 drops m = 0.736 > sqrt(3) - 1. No N makes that informative.
    let err = fermionic_refusal(
        "examples/fermionic/kitaev_n64_t2.qasm",
        "examples/fermionic/kitaev_op_n64.txt",
        "stabrank",
        &["--max-chi", "1"],
    );
    assert!(
        err.contains("exceeds the ceiling of R + |value|"),
        "E3 N=64 --max-chi 1 refused for another reason:\n{err}"
    );
}

#[test]
fn e3_full_quench_is_refused_by_stabrank_and_exact_on_majoranaprop() {
    // T on all 64 sites. Sum over Cliffords needs 2^64 branches. At the
    // default --max-branches ceiling stabrank refuses: no value, no bound.
    // The number comes from majoranaprop, against the closed form -63/2.
    let err = fermionic_refusal(
        "examples/fermionic/kitaev_n64_tall.qasm",
        "examples/fermionic/kitaev_op_n64.txt",
        "stabrank",
        &[],
    );
    assert!(
        err.contains("above the ceiling of 65536"),
        "E3 full quench refused for another reason:\n{err}"
    );
    let all: Vec<usize> = (0..64).collect();
    let reference = kitaev_t_closed_form(64, &all);
    assert!((reference + 31.5).abs() < 1e-12, "closed form {reference}");
    assert_majoranaprop_exact(
        "kitaev_n64_tall",
        "examples/fermionic/kitaev_op_n64.txt",
        reference,
        "E3 full quench",
    );
}

// ---------------------------------------------------------------------------
// LiH / STO-3G at R = 1.6 Å, Hartree-Fock from a PySCF FCIDUMP.
// The circuit is the RHF determinant. PySCF FCI is the ground-state
// reference; this state does not reach it. The gap is the correlation
// energy from that FCI calculation.
// ---------------------------------------------------------------------------

fn lih_script_energies() -> (f64, f64) {
    let text = std::fs::read_to_string(repo_root().join("examples/fermionic/lih_reference.txt"))
        .expect("lih_reference.txt");
    let mut hf = None;
    let mut fci = None;
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("E_HF ") {
            hf = Some(v.trim().parse::<f64>().unwrap());
        }
        if let Some(v) = line.strip_prefix("E_FCI ") {
            fci = Some(v.trim().parse::<f64>().unwrap());
        }
    }
    (hf.expect("E_HF"), fci.expect("E_FCI"))
}

fn lih_expectation(backend: &str) -> f64 {
    let dump = repo_root().join("examples/fermionic/lih_sto3g.fcidump");
    let out = Command::new(bin())
        .current_dir(repo_root())
        .args([
            "examples/fermionic/lih_hf.fqasm",
            "--hamiltonian",
            dump.to_str().unwrap(),
            "--backend",
            backend,
            "--format",
            "json",
        ])
        .output()
        .expect("spawn omega-run");
    let doc = json_line(&out, &format!("LiH {backend}"));
    doc["value"].as_f64().expect("LiH value")
}

#[test]
fn lih_rhf_matches_pyscf_and_the_gap_to_fci_is_the_correlation_energy() {
    let (e_hf, e_fci) = lih_script_energies();
    let gap_ref = e_hf - e_fci;
    assert!(
        gap_ref > 1e-3,
        "LiH reference lists HF and FCI as the same number ({e_hf}, {e_fci})"
    );
    for backend in ["statevector", "sector"] {
        let measured = lih_expectation(backend);
        assert!(
            (measured - e_hf).abs() < 1e-8,
            "LiH {backend}: measured {measured} vs PySCF RHF {e_hf}"
        );
        let gap = measured - e_fci;
        assert!(
            (gap - gap_ref).abs() < 1e-8,
            "LiH {backend}: gap to PySCF FCI {gap} vs E_HF - E_FCI {gap_ref}"
        );
        assert!(
            gap.abs() > 1e-3,
            "LiH {backend}: measured {measured} reproduced PySCF FCI {e_fci}"
        );
    }
}

#[test]
fn lih_fcidump_matches_bytes_pyscf_wrote() {
    let py = repo_root().join("crates/omega-bridges/python/.venv-ffsim/bin/python");
    if !py.exists() {
        eprintln!(
            "pyscf venv missing at {} — LiH FCIDUMP vs pyscf did not run. Build with \
             `make -C crates/omega-bridges/python ffsim-venv`.",
            py.display()
        );
        return;
    }
    let out = Command::new(&py)
        .arg(repo_root().join("tools/fermionic_examples/gen_lih.py"))
        .arg("--check")
        .output()
        .expect("spawn pyscf");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "gen_lih.py --check failed:\n{stdout}{stderr}"
    );
}

// ---------------------------------------------------------------------------
// E5 — spin-1 singlet. The CLI has no qudit-observable flag; ⟨S₁·S₂⟩ is
// contracted here from the statevector the CLI printed.
// ---------------------------------------------------------------------------

fn spin_dot(amps: &[(f64, f64)]) -> f64 {
    let s2 = 2.0_f64.sqrt();
    // Row-major. Level 0, 1, 2 is m = +1, 0, -1.
    let sz = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0];
    let sp = [0.0, s2, 0.0, 0.0, 0.0, s2, 0.0, 0.0, 0.0];
    let sm = [0.0, 0.0, 0.0, s2, 0.0, 0.0, 0.0, s2, 0.0];
    let prod = |a: &[f64], b: &[f64]| {
        let mut re = 0.0;
        for j in 0..3 {
            for i in 0..3 {
                let (br, bi) = amps[i + 3 * j];
                for l in 0..3 {
                    for k in 0..3 {
                        let (kr, ki) = amps[k + 3 * l];
                        let coeff = a[i * 3 + k] * b[j * 3 + l];
                        re += coeff * (br * kr + bi * ki);
                    }
                }
            }
        }
        re
    };
    prod(&sz, &sz) + 0.5 * prod(&sp, &sm) + 0.5 * prod(&sm, &sp)
}

fn assert_spin1_singlet(amps: &[(f64, f64)], what: &str) {
    let ss = spin_dot(amps);
    assert!(
        (ss + 2.0).abs() < 1e-10,
        "{what}: measured ⟨S1·S2⟩ = {ss}, expected -2"
    );
    let s = inv_sqrt3();
    assert_eq!(amps.len(), 9, "{what}: two qutrits");
    for (i, (re, im)) in amps.iter().enumerate() {
        let expect = match i {
            2 | 6 => s,
            4 => -s,
            _ => 0.0,
        };
        let err = (re - expect).abs() + im.abs();
        assert!(
            err < 1e-12,
            "{what}: index {i} got ({re}, {im}), expected ({expect}, 0)"
        );
    }
}

fn singlet_state(backend: &str) -> serde_json::Value {
    let out = run(&[
        "examples/qudit/spin1_singlet.ditqasm",
        "--backend",
        backend,
        "--statevector",
        "--format",
        "json",
    ]);
    json_line(&out, &format!("E5 singlet {backend}"))
}

#[test]
fn e5_spin1_singlet_has_bond_energy_minus_two() {
    for backend in ["quditsv", "mps"] {
        let doc = singlet_state(backend);
        let amps = amplitudes_of(&doc, &format!("E5 {backend}"));
        assert_spin1_singlet(&amps, &format!("E5 {backend}"));
    }
}

// ---------------------------------------------------------------------------
// E6 — qutrit Deutsch-Jozsa. P(index 0) is 1 (constant) or 0 (balanced).
// ---------------------------------------------------------------------------

fn prob_index0(amps: &[(f64, f64)]) -> f64 {
    let (re, im) = amps[0];
    re * re + im * im
}

fn assert_dj(circuit: &str, expect: f64) {
    let file = format!("examples/qudit/{circuit}.ditqasm");
    let out = run(&[
        &file,
        "--backend",
        "quditsv",
        "--statevector",
        "--format",
        "json",
    ]);
    let doc = json_line(&out, &format!("E6 {circuit}"));
    let amps = amplitudes_of(&doc, &format!("E6 {circuit}"));
    let p = prob_index0(&amps);
    assert!(
        (p - expect).abs() < 1e-12,
        "E6 {circuit}: P(index 0) = {p}, expected {expect}"
    );
}

#[test]
fn e6_qutrit_deutsch_jozsa_probabilities() {
    assert_dj("qutrit_dj_constant", 1.0);
    assert_dj("qutrit_dj_balanced", 0.0);
}
