// SPDX-License-Identifier: Apache-2.0
//! Parse + instantiate every `examples/aria/*.aria` file.
//!
//! This is the gate that keeps the Aria example programs *real* (not
//! documentation-only): each must round-trip through `parse_aria` and
//! `instantiate` to a non-empty `Circuit`. Adding a new example is a one-row
//! change to the table below — no other Rust needed.

use aria_core::ast::parse_aria;

/// (file, circuit name, instantiation params).
type Example<'a> = (&'a str, &'a str, &'a [(&'a str, i64)]);

/// Keep alphabetical.
const EXAMPLES: &[Example<'static>] = &[
    ("bell.aria", "Bell", &[]),
    ("butterfly_qnn.aria", "ButterflyQNN", &[]),
    ("ghz.aria", "GHZ", &[]),
    ("jl_sketch_digits.aria", "JlSketchDigits", &[("k", 8)]),
    ("circulant.aria", "CyclicShift", &[]),
    ("cqs.aria", "HadamardTestZ", &[]),
    ("qos_oracle.aria", "PhaseOracle", &[("n", 3)]),
    (
        "bernstein_vazirani.aria",
        "BernsteinVazirani",
        &[("n", 4), ("a", 5)],
    ),
    ("deutsch_jozsa.aria", "DeutschJozsa", &[("n", 3)]),
    ("grover3.aria", "Grover3", &[("marked", 5)]),
    ("hhl.aria", "Hhl", &[("n", 2)]),
    ("iqp_born.aria", "IqpBorn", &[("n", 4)]),
    ("qcnn.aria", "QCNN", &[]),
    (
        "qcbm_strongly_entangling.aria",
        "QcbmStronglyEntangling",
        &[("N", 4), ("L", 2)],
    ),
    ("qaoa_maxcut.aria", "QAOAMaxCut", &[("n", 3), ("p", 2)]),
    ("qasm_gpu.aria", "QasmGpu", &[]),
    ("qec_grover.aria", "Grover2", &[("marked", 2)]),
    ("qec_qft.aria", "QFT", &[("n", 4)]),
    ("qec_qpe.aria", "QPEDemo", &[("m", 3)]),
    ("qft.aria", "QFT", &[("n", 4)]),
    ("qgan.aria", "QGANGenerator", &[("n", 2), ("L", 2)]),
    ("qml_classifier.aria", "QMLClassifier", &[("L", 3)]),
    ("qml_tune.aria", "QmlTune", &[("n", 4), ("L", 2)]),
    (
        "qclassifier_rich.aria",
        "QClassifierRich",
        &[("N", 4), ("L", 2)],
    ),
    ("qpe.aria", "QPE", &[("t", 3)]),
    ("qsp.aria", "QSP", &[("d", 4)]),
    ("trotter.aria", "Trotter", &[("steps", 8)]),
    ("qdrift.aria", "QDrift", &[("N", 8)]),
    ("taylor_lcu.aria", "TaylorLcu", &[]),
    ("shor.aria", "Shor15", &[]),
    ("schrodingerize.aria", "Schrodingerize", &[("s", 1)]),
    ("qssl.aria", "QSSLEncoder", &[("N", 2), ("L", 2)]),
    ("qsvd.aria", "QsvdAnsatz", &[]),
    ("qsvt_invert.aria", "QsvtInvert", &[("degree", 4)]),
    ("quantum_kernel.aria", "QuantumKernelMap", &[("n", 2)]),
    ("shor_ecdlp.aria", "ShorECDLP", &[("n", 4), ("t", 4)]),
    ("simon.aria", "Simon", &[("n", 3)]),
    ("sketch_qml.aria", "SketchQml", &[("k", 4)]),
    (
        "spectra_heisenberg.aria",
        "SpectraHeisenberg",
        &[("steps", 3)],
    ),
    (
        "strongly_entangling.aria",
        "StronglyEntangling",
        &[("n", 3), ("L", 2)],
    ),
    ("superdense.aria", "Superdense", &[("b0", 1), ("b1", 0)]),
    ("swap_test.aria", "SwapTest", &[("n", 2)]),
    ("teleport.aria", "Teleport", &[]),
    ("vqe_ansatz.aria", "VQEAnsatz", &[("n_layers", 3)]),
];

fn aria_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/aria")
}

#[test]
fn all_listed_examples_parse_and_instantiate() {
    let dir = aria_dir();
    let mut failures = Vec::new();
    for (file, name, params) in EXAMPLES {
        let src = match std::fs::read_to_string(dir.join(file)) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("{file}: read: {e}"));
                continue;
            }
        };
        let prog = match parse_aria(&src) {
            Ok(p) => p,
            Err(e) => {
                failures.push(format!("{file}: parse: {e}"));
                continue;
            }
        };
        match prog.instantiate(name, params) {
            Ok(c) if c.n_qubits() > 0 => {}
            Ok(_) => failures.push(format!("{file}::{name}: 0 qubits")),
            Err(e) => failures.push(format!("{file}::{name}: instantiate: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "Aria example failures:\n  {}",
        failures.join("\n  ")
    );
}

/// Every `.aria` file on disk must be covered by the table above — so a new
/// example can't silently skip the parse gate.
#[test]
fn every_aria_file_is_covered() {
    let dir = aria_dir();
    let listed: std::collections::HashSet<&str> = EXAMPLES.iter().map(|(f, _, _)| *f).collect();
    let mut uncovered = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read aria dir") {
        let entry = entry.unwrap();
        let fname = entry.file_name().to_string_lossy().into_owned();
        if fname.ends_with(".aria") && !listed.contains(fname.as_str()) {
            uncovered.push(fname);
        }
    }
    assert!(
        uncovered.is_empty(),
        "these .aria files are not in the EXAMPLES parse table: {uncovered:?}"
    );
}
