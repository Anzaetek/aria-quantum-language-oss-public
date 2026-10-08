// SPDX-License-Identifier: Apache-2.0
//! The §3 workload artifacts are what they say they are.
//!
//! §3 pins the HEA family as "the `entangling_circuit` brickwall of
//! `crates/omega-backend-mps/examples/mps_stage_profile.rs:82-108` [...]
//! exported once to QASM2 so every arm ingests the identical file". The export
//! is therefore the contract. Two things can break it:
//!
//! * the exported QASM stops being that circuit — a dropped Rx ring, a changed
//!   angle, a brickwall offset off by one. Caught by lowering each committed
//!   file with the repo's own parser and comparing op-for-op against
//!   `entangling_circuit` rebuilt here.
//! * the committed file stops being what the generator writes, so regenerating
//!   would silently move every row's workload. Caught by re-hashing against
//!   `tools/emu_compare/MANIFEST.json`, which the generator writes in the same
//!   pass as the files.
//!
//! Both matter because the failure they prevent already happened one domain
//! over: `contract_bench.rs` measured a warm-up that was an identity, so the
//! bond never crossed the dispatch threshold and the circuit on the label was
//! not the circuit that ran.

use std::path::{Path, PathBuf};

use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_parser::lower::{lower_to_ir_with_dialect, Qasm2Dialect};
use sha2::{Digest, Sha256};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("repo root")
}

fn manifest() -> serde_json::Map<String, serde_json::Value> {
    let path = repo_root().join("tools/emu_compare/MANIFEST.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let doc: serde_json::Value = serde_json::from_str(&text).expect("MANIFEST.json");
    doc["artifacts"]
        .as_object()
        .expect("artifacts object")
        .clone()
}

/// `entangling_circuit` of `crates/omega-backend-mps/examples/
/// mps_stage_profile.rs:82-108`. A transcription, which is the point: it is the
/// independent statement of the circuit that the exported QASM is checked
/// against, so a drift in either one reddens this test.
fn entangling_circuit(num_qubits: u32, depth: usize) -> CircuitIR {
    let mut circuit = CircuitIR::new(num_qubits, CircuitType::GateBased);
    let push = |c: &mut CircuitIR, gate: GateKind, qs: Vec<u32>, ps: Vec<f64>| {
        c.ops.push(GateOp {
            gate,
            qubits: qs.into_iter().map(Qubit).collect(),
            params: ps.into_iter().map(ParamExpr::Concrete).collect(),
            classical_bit: None,
            condition: None,
        });
    };
    for q in 0..num_qubits {
        push(&mut circuit, GateKind::H, vec![q], vec![]);
    }
    for d in 0..depth {
        let offset = d as u32 & 1;
        for q in (offset..num_qubits - 1).step_by(2) {
            push(&mut circuit, GateKind::CX, vec![q, q + 1], vec![]);
            push(&mut circuit, GateKind::Rz, vec![q + 1], vec![0.25]);
            push(&mut circuit, GateKind::CX, vec![q, q + 1], vec![]);
        }
        for q in 0..num_qubits {
            push(&mut circuit, GateKind::Rx, vec![q], vec![0.15]);
        }
    }
    circuit
}

/// The Clifford brickwall of §3(a): the same skeleton with the two rotations
/// replaced by one S, which is what makes it expressible in a tableau.
fn clifford_circuit(num_qubits: u32, depth: usize) -> CircuitIR {
    let mut circuit = CircuitIR::new(num_qubits, CircuitType::GateBased);
    let push = |c: &mut CircuitIR, gate: GateKind, qs: Vec<u32>| {
        c.ops.push(GateOp {
            gate,
            qubits: qs.into_iter().map(Qubit).collect(),
            params: Default::default(),
            classical_bit: None,
            condition: None,
        });
    };
    for q in 0..num_qubits {
        push(&mut circuit, GateKind::H, vec![q]);
    }
    for d in 0..depth {
        let offset = d as u32 & 1;
        for q in (offset..num_qubits - 1).step_by(2) {
            push(&mut circuit, GateKind::CX, vec![q, q + 1]);
        }
        for q in 0..num_qubits {
            push(&mut circuit, GateKind::S, vec![q]);
        }
    }
    circuit
}

fn concrete(p: &ParamExpr) -> f64 {
    match p {
        ParamExpr::Concrete(v) => *v,
        other => panic!("non-concrete parameter {other:?}"),
    }
}

fn assert_same_circuit(id: &str, got: &CircuitIR, want: &CircuitIR) {
    assert_eq!(got.num_qubits, want.num_qubits, "{id}: qubit count");
    assert_eq!(got.ops.len(), want.ops.len(), "{id}: op count");
    for (i, (a, b)) in got.ops.iter().zip(&want.ops).enumerate() {
        assert_eq!(a.gate, b.gate, "{id}: op {i} gate");
        assert_eq!(a.qubits, b.qubits, "{id}: op {i} qubits");
        assert_eq!(a.params.len(), b.params.len(), "{id}: op {i} param count");
        for (pa, pb) in a.params.iter().zip(&b.params) {
            assert_eq!(concrete(pa), concrete(pb), "{id}: op {i} angle");
        }
    }
}

/// Lower a committed artifact in the dialect the manifest records for it.
///
/// The dialect is a pinned property of the workload, not a detail of this
/// test: `legacy` is `QuantumCircuit.from_qasm_str`, so a `legacy` artifact is
/// one every arm — ours and Qiskit's — reads bit-for-bit. LUCJ-16 is pinned
/// `lenient`, and the manifest says in prose why that file cannot be made
/// `legacy` without breaking our own `sector` arm.
fn lower(id: &str, file: &str, dialect: &str) -> CircuitIR {
    let path = repo_root().join(file);
    let src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{id}: {} {e}", path.display()));
    let dialect = match dialect {
        "legacy" => Qasm2Dialect::Legacy,
        "lenient" => Qasm2Dialect::Lenient,
        "strict" => Qasm2Dialect::Strict,
        other => panic!("{id}: unknown dialect {other:?} in MANIFEST.json"),
    };
    lower_to_ir_with_dialect(&src, dialect).unwrap_or_else(|e| panic!("{id}: lowering {file}: {e}"))
}

fn lower_pinned(id: &str, meta: &serde_json::Value) -> CircuitIR {
    lower(
        id,
        meta["file"].as_str().unwrap(),
        meta["dialect"].as_str().unwrap(),
    )
}

#[test]
fn the_hea_export_is_the_entangling_circuit_it_claims_to_be() {
    let artifacts = manifest();
    let mut checked = 0;
    for (id, meta) in &artifacts {
        if meta["family"] != "hea" {
            continue;
        }
        let n = meta["qubits"].as_u64().unwrap() as u32;
        let depth = meta["depth"].as_u64().unwrap() as usize;
        let got = lower_pinned(id, meta);
        assert_same_circuit(id, &got, &entangling_circuit(n, depth));
        checked += 1;
    }
    // §3 pins 14q x {4,12,24} and 20q x {12,24} for the MPS/dense-CPU lanes,
    // and the GPU lane adds 24/26/28q x depth 24.
    assert_eq!(checked, 8, "the HEA grid of §3 is eight shapes");
}

#[test]
fn the_clifford_brickwall_is_clifford_only() {
    let artifacts = manifest();
    let mut shapes = Vec::new();
    for (id, meta) in &artifacts {
        if meta["family"] != "clifford" {
            continue;
        }
        let n = meta["qubits"].as_u64().unwrap() as u32;
        let depth = meta["depth"].as_u64().unwrap() as usize;
        let got = lower_pinned(id, meta);
        assert_same_circuit(id, &got, &clifford_circuit(n, depth));
        // The stabilizer lane's whole premise: Stim can express this. A single
        // rotation would make the comparison a different one.
        for (i, op) in got.ops.iter().enumerate() {
            assert!(
                matches!(op.gate, GateKind::H | GateKind::S | GateKind::CX),
                "{id}: op {i} is {:?}, which is not H/S/CX",
                op.gate
            );
        }
        shapes.push((n, depth));
    }
    shapes.sort_unstable();
    assert_eq!(shapes, vec![(24, 100), (100, 100)], "§3(a) pins these two");
}

#[test]
fn every_artifact_lowers_and_matches_its_pinned_shape() {
    // RANDOM-1 and LUCJ-16 have no Rust-side statement to compare against —
    // RANDOM-1's angles come from a Mersenne Twister and LUCJ's from ffsim —
    // so what is pinned is that the file parses with this repo's own parser and
    // has the qubit count and gate counts the manifest recorded. A file an arm
    // cannot read is not a workload, and a silently truncated one would show
    // up here as a gate count.
    for (id, meta) in &manifest() {
        let ir = lower_pinned(id, meta);
        assert_eq!(
            ir.num_qubits,
            meta["qubits"].as_u64().unwrap() as u32,
            "{id}: qubit count"
        );
        let mut counted = 0usize;
        for (_, n) in meta["gates"].as_object().unwrap() {
            counted += n.as_u64().unwrap() as usize;
        }
        assert_eq!(ir.ops.len(), counted, "{id}: op count against the manifest");
    }
}

#[test]
fn only_lucj_needs_a_dialect_no_qiskit_reader_has() {
    // The dialect pin has to have teeth both ways: every `legacy` artifact
    // must actually read as `legacy` (that is what makes "every arm ingests
    // the identical file" true for it), and LUCJ-16 must actually fail there
    // (if it ever stopped failing, the manifest's prose about `rbs` would be
    // stale and the weaker contract would be carried for no reason).
    for (id, meta) in &manifest() {
        let path = repo_root().join(meta["file"].as_str().unwrap());
        let src = std::fs::read_to_string(&path).unwrap();
        let legacy = lower_to_ir_with_dialect(&src, Qasm2Dialect::Legacy);
        match meta["dialect"].as_str().unwrap() {
            "legacy" => assert!(
                legacy.is_ok(),
                "{id} is pinned legacy but does not read there: {}",
                legacy.unwrap_err()
            ),
            "lenient" => {
                assert_eq!(
                    id, "lucj16",
                    "a second lenient artifact needs its own reason"
                );
                let err = legacy.expect_err("lucj16 reads in the legacy dialect after all");
                assert!(err.contains("rbs"), "{id}: unexpected refusal {err}");
            }
            other => panic!("{id}: unexpected dialect {other}"),
        }
    }
}

#[test]
fn every_committed_artifact_hashes_to_the_manifest() {
    // §3 asks every row to fix "(circuit id, generator+seed or file, ...)".
    // A row can only name its file honestly if the file is pinned, and
    // `Workload::sha256` is a witness field for that reason.
    for (id, meta) in &manifest() {
        let file = meta["file"].as_str().unwrap();
        let bytes =
            std::fs::read(repo_root().join(file)).unwrap_or_else(|e| panic!("{id}: {file} {e}"));
        let got = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(
            got,
            meta["sha256"].as_str().unwrap(),
            "{id}: {file} is not the file MANIFEST.json pins; regenerate with \
             tools/emu_compare/gen_workloads.py or explain the change"
        );
    }
}

#[test]
fn the_manifest_covers_every_committed_qasm_file() {
    // The other direction: a stray file in the directory is a workload nobody
    // pinned, and §3's point is that the set is fixed before anything runs.
    let dir = repo_root().join("tools/emu_compare/qasm");
    let pinned: Vec<String> = manifest()
        .values()
        .map(|m| m["file"].as_str().unwrap().to_string())
        .collect();
    let mut stray = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("qasm dir") {
        let name = entry.unwrap().file_name().to_string_lossy().to_string();
        if !name.ends_with(".qasm") {
            continue;
        }
        if !pinned
            .iter()
            .any(|p| p == &format!("tools/emu_compare/qasm/{name}"))
        {
            stray.push(name);
        }
    }
    assert!(stray.is_empty(), "unpinned workloads: {stray:?}");
    assert_eq!(pinned.len(), 15, "§3's workload set is fifteen artifacts");
}
