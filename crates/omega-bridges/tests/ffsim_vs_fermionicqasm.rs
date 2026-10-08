// SPDX-License-Identifier: Apache-2.0
//! F4.3 — FermionicQASM against ffsim. The honesty gate for the track.
//!
//! Every circuit in this file is `.fqasm` text passed to [`omega_parser::lower_to_ir`].
//! Building a `CircuitIR` here would test the lowering's output shape, not the
//! language. ffsim never sees our matrices: the live leg ships the lowered
//! circuit as QPY and reads the same Jordan–Wigner observables back, the way
//! `ffsim_vs_statevector.rs` does. The fixture leg compares full statevectors
//! to `conventions.json`, which F4.0 filled with ffsim's own `apply_*` calls.
//!
//! ## What this file does not compare
//!
//! `conventions.json` `unpinned.cross_spin_givens_and_tunnel` (ffsim 0.0.84):
//! `apply_givens_rotation` and `apply_tunneling_interaction` each take one
//! `Spin` and stay inside that species. There is no ffsim matrix for a pair
//! whose wires sit in different spin blocks, including the adjacent pair
//! across the block boundary. F4.2 refuses that statement. This file asserts
//! the refusal and does not invent an oracle. Same-spin agreement is not
//! reported as covering it.
//!
//! Non-adjacent `givens` / `tunnel` (`spinless_{3,4}/{givens,tunnel}_0_2`) are
//! a different gap. ffsim did return those matrices; F4.2 still refuses the
//! statement through `fermion::givens_expr` (the Z string). Those four keys
//! are named below and are not compared. The note on each fixture entry says
//! the matrix is for a routing that does not exist yet.
//!
//! `not_a_pin.apply_givens_rotation_docstring_matrix`: the docstring prints
//! the off-diagonals the other way round from the matrix ffsim executes.
//! This file compares the executed matrix.
//!
//! The N=0 and N=n sectors of an expectation comparison are a global phase on
//! one basis vector. They are executed, and they are not evidence. The
//! per-gate fixture compares those columns as amplitudes, which do carry the
//! phase.

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::OnceLock;

const TOL: f64 = 1e-9;

/// Non-adjacent `givens` / `tunnel`. Shared by the four excluded keys.
const NONADJACENT: &str = "\
F4.2 refuses this statement: fermion::givens_expr will not emit a \
non-adjacent Rbs, because the fermionic rotation carries a Z string that \
gate does not. conventions.json stores the ffsim matrix for a routing that \
is not in this lowering. F4.3 does not compare that matrix, and does not \
invent a circuit for ffsim to accept in its place.";

struct CatalogEntry {
    key: &'static str,
    mode: &'static str,
    n_wires: u32,
    /// Gate statement at the fixture's θ = 0.7. Text, not an IR node.
    gate: &'static str,
    /// `ffsim.<this>(` must occur in the fixture `call`.
    ffsim_fn: &'static str,
    /// `Some` ⇒ do not compare. The statement must be refused, and the
    /// string is why.
    not_compared: Option<&'static str>,
}

const fn pin(
    key: &'static str,
    mode: &'static str,
    n_wires: u32,
    gate: &'static str,
    ffsim_fn: &'static str,
) -> CatalogEntry {
    CatalogEntry {
        key,
        mode,
        n_wires,
        gate,
        ffsim_fn,
        not_compared: None,
    }
}

const fn excluded(
    key: &'static str,
    mode: &'static str,
    n_wires: u32,
    gate: &'static str,
    ffsim_fn: &'static str,
) -> CatalogEntry {
    CatalogEntry {
        key,
        mode,
        n_wires,
        gate,
        ffsim_fn,
        not_compared: Some(NONADJACENT),
    }
}

const CATALOG: &[CatalogEntry] = &[
    pin(
        "spinless_2/num_orb0",
        "mode m[2];",
        2,
        "num(0.7) m[0];",
        "apply_num_interaction",
    ),
    pin(
        "spinless_2/num_orb1",
        "mode m[2];",
        2,
        "num(0.7) m[1];",
        "apply_num_interaction",
    ),
    pin(
        "spinless_2/numnum_0_1",
        "mode m[2];",
        2,
        "numnum(0.7) m[0], m[1];",
        "apply_num_num_interaction",
    ),
    pin(
        "spinless_2/givens_0_1",
        "mode m[2];",
        2,
        "givens(0.7) m[0], m[1];",
        "apply_givens_rotation",
    ),
    pin(
        "spinless_2/tunnel_0_1",
        "mode m[2];",
        2,
        "tunnel(0.7) m[0], m[1];",
        "apply_tunneling_interaction",
    ),
    pin(
        "spinless_2/numnum_1_0",
        "mode m[2];",
        2,
        "numnum(0.7) m[1], m[0];",
        "apply_num_num_interaction",
    ),
    pin(
        "spinless_2/givens_1_0",
        "mode m[2];",
        2,
        "givens(0.7) m[1], m[0];",
        "apply_givens_rotation",
    ),
    pin(
        "spinless_2/tunnel_1_0",
        "mode m[2];",
        2,
        "tunnel(0.7) m[1], m[0];",
        "apply_tunneling_interaction",
    ),
    pin(
        "spinless_3/num_orb0",
        "mode m[3];",
        3,
        "num(0.7) m[0];",
        "apply_num_interaction",
    ),
    pin(
        "spinless_3/num_orb2",
        "mode m[3];",
        3,
        "num(0.7) m[2];",
        "apply_num_interaction",
    ),
    pin(
        "spinless_3/numnum_0_1",
        "mode m[3];",
        3,
        "numnum(0.7) m[0], m[1];",
        "apply_num_num_interaction",
    ),
    pin(
        "spinless_3/givens_0_1",
        "mode m[3];",
        3,
        "givens(0.7) m[0], m[1];",
        "apply_givens_rotation",
    ),
    pin(
        "spinless_3/tunnel_0_1",
        "mode m[3];",
        3,
        "tunnel(0.7) m[0], m[1];",
        "apply_tunneling_interaction",
    ),
    pin(
        "spinless_3/numnum_0_2",
        "mode m[3];",
        3,
        "numnum(0.7) m[0], m[2];",
        "apply_num_num_interaction",
    ),
    excluded(
        "spinless_3/givens_0_2",
        "mode m[3];",
        3,
        "givens(0.7) m[0], m[2];",
        "apply_givens_rotation",
    ),
    excluded(
        "spinless_3/tunnel_0_2",
        "mode m[3];",
        3,
        "tunnel(0.7) m[0], m[2];",
        "apply_tunneling_interaction",
    ),
    pin(
        "spinless_4/num_orb0",
        "mode m[4];",
        4,
        "num(0.7) m[0];",
        "apply_num_interaction",
    ),
    pin(
        "spinless_4/num_orb3",
        "mode m[4];",
        4,
        "num(0.7) m[3];",
        "apply_num_interaction",
    ),
    pin(
        "spinless_4/numnum_0_1",
        "mode m[4];",
        4,
        "numnum(0.7) m[0], m[1];",
        "apply_num_num_interaction",
    ),
    pin(
        "spinless_4/givens_0_1",
        "mode m[4];",
        4,
        "givens(0.7) m[0], m[1];",
        "apply_givens_rotation",
    ),
    pin(
        "spinless_4/tunnel_0_1",
        "mode m[4];",
        4,
        "tunnel(0.7) m[0], m[1];",
        "apply_tunneling_interaction",
    ),
    pin(
        "spinless_4/numnum_0_2",
        "mode m[4];",
        4,
        "numnum(0.7) m[0], m[2];",
        "apply_num_num_interaction",
    ),
    excluded(
        "spinless_4/givens_0_2",
        "mode m[4];",
        4,
        "givens(0.7) m[0], m[2];",
        "apply_givens_rotation",
    ),
    excluded(
        "spinless_4/tunnel_0_2",
        "mode m[4];",
        4,
        "tunnel(0.7) m[0], m[2];",
        "apply_tunneling_interaction",
    ),
    pin(
        "spinful_2/num_orb0_alpha",
        "mode m[2] spin;",
        4,
        "num(0.7) m[0];",
        "apply_num_interaction",
    ),
    pin(
        "spinful_2/num_orb0_beta",
        "mode m[2] spin;",
        4,
        "num(0.7) m[2];",
        "apply_num_interaction",
    ),
    pin(
        "spinful_2/num_orb1_alpha",
        "mode m[2] spin;",
        4,
        "num(0.7) m[1];",
        "apply_num_interaction",
    ),
    pin(
        "spinful_2/numnum_0_1_alpha",
        "mode m[2] spin;",
        4,
        "numnum(0.7) m[0], m[1];",
        "apply_num_num_interaction",
    ),
    pin(
        "spinful_2/givens_0_1_alpha",
        "mode m[2] spin;",
        4,
        "givens(0.7) m[0], m[1];",
        "apply_givens_rotation",
    ),
    pin(
        "spinful_2/givens_0_1_beta",
        "mode m[2] spin;",
        4,
        "givens(0.7) m[2], m[3];",
        "apply_givens_rotation",
    ),
    pin(
        "spinful_2/tunnel_0_1_alpha",
        "mode m[2] spin;",
        4,
        "tunnel(0.7) m[0], m[1];",
        "apply_tunneling_interaction",
    ),
    pin(
        "spinful_2/tunnel_0_1_beta",
        "mode m[2] spin;",
        4,
        "tunnel(0.7) m[2], m[3];",
        "apply_tunneling_interaction",
    ),
    pin(
        "spinful_2/onsite_orb0",
        "mode m[2] spin;",
        4,
        "numnum(0.7) m[0], m[2];",
        "apply_on_site_interaction",
    ),
    pin(
        "spinful_2/numprod_alpha0_beta1",
        "mode m[2] spin;",
        4,
        "numnum(0.7) m[0], m[3];",
        "apply_num_op_prod_interaction",
    ),
];

fn conventions() -> &'static Value {
    static DOC: OnceLock<Value> = OnceLock::new();
    DOC.get_or_init(|| {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../omega-parser/tests/fixtures/fermionicqasm/conventions.json");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
    })
}

fn with_loads(mode: &str, occupied: &[u32], gate: &str) -> String {
    let mut s = String::from("FERMIONICQASM 1.0;\n");
    s.push_str(mode);
    s.push('\n');
    for w in occupied {
        s.push_str(&format!("load m[{w}];\n"));
    }
    s.push_str(gate);
    s.push('\n');
    s
}

/// The only way a circuit enters this file.
fn lower_fqasm(src: &str) -> CircuitIR {
    assert!(
        src.lines().any(|l| l.trim() == "FERMIONICQASM 1.0;"),
        "F4.3 enters from FermionicQASM text, not from an IR built in the test:\n{src}"
    );
    let ir = omega_parser::lower_to_ir(src).unwrap_or_else(|e| panic!("lower `{src}`: {e}"));
    assert_eq!(
        ir.circuit_type,
        CircuitType::Fermionic,
        "text was not lowered as Fermionic — the header was not what decided the lane:\n{src}"
    );
    ir
}

fn evolve(ir: &CircuitIR) -> Vec<num_complex::Complex64> {
    let result = StatevectorBackend::new()
        .execute(
            ir,
            &ParameterBinding::new(),
            &ExecConfig {
                shots: None,
                seed: None,
                mid_circuit_mode: MidCircuitMode::Skip,
            },
        )
        .unwrap_or_else(|e| panic!("statevector: {e}"));
    match result {
        ExecResult::Statevector(v) => v,
        other => panic!("expected a statevector, got {other:?}"),
    }
}

fn system_of(key: &str) -> &str {
    key.split_once('/')
        .unwrap_or_else(|| panic!("gate key `{key}` has no system prefix"))
        .0
}

/// The fixture's bitstrings put orbital 0 on the right, and that index is
/// the statevector address. A later change to the column order has to fail
/// here, before a scrambled matrix is read as a sign error.
fn assert_basis_is_little_endian(doc: &Value) {
    let systems = doc["systems"].as_object().expect("systems");
    assert!(!systems.is_empty(), "conventions.json has no systems");
    for (name, sys) in systems {
        let n = sys["n_qubits"].as_u64().unwrap() as usize;
        let basis = sys["basis"].as_array().expect("basis");
        assert_eq!(basis.len(), 1 << n, "{name}: basis does not cover 2^{n}");
        for rec in basis {
            let idx = rec["qiskit_index"].as_u64().unwrap() as usize;
            let bits = rec["bitstring"].as_str().unwrap();
            assert_eq!(bits.chars().count(), n, "{name} bitstring {bits}");
            let mut mask = 0usize;
            for (i, ch) in bits.chars().rev().enumerate() {
                match ch {
                    '1' => mask |= 1 << i,
                    '0' => {}
                    _ => panic!("{name} bitstring {bits} is not bits"),
                }
            }
            assert_eq!(
                mask, idx,
                "{name}: bitstring {bits} is not qiskit index {idx} (orbital 0 on the right)"
            );
        }
    }
}

fn ids(doc: &Value, field: &str) -> BTreeSet<String> {
    doc[field]
        .as_array()
        .unwrap_or_else(|| panic!("{field} is not a list"))
        .iter()
        .map(|e| {
            e["id"]
                .as_str()
                .unwrap_or_else(|| panic!("{field} entry without id"))
                .to_string()
        })
        .collect()
}

/// Every gate key is either pinned from text or named as not compared.
/// A key added to the fixture and not listed here fails. A key listed here
/// and dropped from the fixture fails the other way. Quiet omission is the
/// defect this function exists to catch.
#[test]
fn every_fixture_gate_is_matched_from_text_or_named_as_not_compared() {
    let doc = conventions();
    assert_eq!(doc["version"], "0.0.84", "F4.3 is pinned to the F4.0 ffsim");
    assert_basis_is_little_endian(doc);

    let not_a_pin = ids(doc, "not_a_pin");
    assert_eq!(
        not_a_pin,
        BTreeSet::from(["apply_givens_rotation_docstring_matrix".to_string()]),
        "the docstring/execution disagreement is a recorded not_a_pin; this test compares the executed matrix, and deleting the record would hide that choice"
    );
    let unpinned = ids(doc, "unpinned");
    assert_eq!(
        unpinned,
        BTreeSet::from(["cross_spin_givens_and_tunnel".to_string()]),
        "cross-spin givens and tunnel are unpinned; F4.3 must keep that name"
    );
    let cross = doc["unpinned"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["id"] == "cross_spin_givens_and_tunnel")
        .unwrap();
    let blocks = cross["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str())
        .collect::<Vec<_>>();
    assert!(
        blocks.contains(&"F4.3"),
        "unpinned.cross_spin_givens_and_tunnel must name F4.3 as blocked, blocks={blocks:?}"
    );

    let gates = doc["gates"].as_object().expect("gates");
    let mut seen = BTreeSet::new();
    let mut pinned = 0usize;
    let mut excluded = 0usize;
    let mut worst = 0.0f64;
    let mut worst_key = String::new();
    for entry in CATALOG {
        assert!(
            seen.insert(entry.key),
            "duplicate catalog key {}",
            entry.key
        );
        let gate = gates.get(entry.key).unwrap_or_else(|| {
            panic!(
                "catalog names {} but conventions.json has no such gate",
                entry.key
            )
        });
        let call = gate["call"].as_str().unwrap();
        let needle = format!("ffsim.{}(", entry.ffsim_fn);
        assert!(
            call.contains(&needle),
            "{} call `{}` does not contain `{needle}`",
            entry.key,
            call
        );
        let system = system_of(entry.key);
        assert_eq!(gate["system"], system, "{}", entry.key);
        let n_json = doc["systems"][system]["n_qubits"].as_u64().unwrap() as u32;
        assert_eq!(n_json, entry.n_wires, "{}", entry.key);

        if entry.not_compared.is_some() {
            excluded += 1;
            let src = with_loads(entry.mode, &[], entry.gate);
            let err = match omega_parser::lower_to_ir(&src) {
                Err(e) => e,
                Ok(_) => panic!(
                    "{} lowered. It is excluded because the lowering refuses it; a successful lower would be a case this test is not comparing.\n{src}",
                    entry.key
                ),
            };
            assert!(
                err.contains("Z string"),
                "{} refusal must be the Z-string message (non-adjacent), got: {err}",
                entry.key
            );
            eprintln!(
                "not compared: {} — non-adjacent, lowering refuses (Z string)",
                entry.key
            );
            continue;
        }

        pinned += 1;
        let matrix = gate["matrix"].as_array().unwrap();
        let dim = 1usize << entry.n_wires;
        assert_eq!(matrix.len(), dim, "{} matrix rows", entry.key);
        for row in matrix {
            assert_eq!(
                row.as_array().unwrap().len(),
                dim,
                "{} matrix cols",
                entry.key
            );
        }
        // `basis` is the occupation mask (bit w set ⇒ load mode w) and the
        // matrix column. Both uses are the point of the loop.
        #[allow(clippy::needless_range_loop)]
        for basis in 0..dim {
            let mut occupied = Vec::new();
            for w in 0..entry.n_wires {
                if basis & (1 << w) != 0 {
                    occupied.push(w);
                }
            }
            let src = with_loads(entry.mode, &occupied, entry.gate);
            let amps = evolve(&lower_fqasm(&src));
            assert_eq!(amps.len(), dim, "{}", entry.key);
            for (row, got) in amps.iter().enumerate() {
                let pair = &matrix[row][basis];
                let want_re = pair[0].as_f64().unwrap();
                let want_im = pair[1].as_f64().unwrap();
                let d = (got.re - want_re).hypot(got.im - want_im);
                if d > worst {
                    worst = d;
                    worst_key = format!("{} col {basis} row {row}", entry.key);
                }
                assert!(
                    d < TOL,
                    "{} col {basis} row {row}: lowered ({}, {}) fixture ({want_re}, {want_im}) |Δ|={d:e}\n{src}",
                    entry.key,
                    got.re,
                    got.im
                );
            }
        }
    }
    let json_keys: BTreeSet<&str> = gates.keys().map(String::as_str).collect();
    assert_eq!(
        seen, json_keys,
        "catalog and conventions.json gates diverged"
    );
    assert!(pinned > 0 && excluded > 0);
    eprintln!(
        "fixture: {pinned} gates matched from .fqasm text across every column, {excluded} named and not compared, worst |Δ| = {worst:e} at {worst_key}"
    );
}

/// Cross-spin `givens` / `tunnel` has no ffsim matrix. The pair exercised
/// here is adjacent (`m[1], m[2]` on `mode m[2] spin`), so `givens_expr`
/// would accept it; the refusal has to be the unpinned check, not the
/// Z-string check. This test does not call ffsim.
#[test]
fn cross_spin_givens_and_tunnel_are_not_compared() {
    let doc = conventions();
    let reason = doc["unpinned"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["id"] == "cross_spin_givens_and_tunnel")
        .expect("unpinned id missing")["reason"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        reason.contains("F4.3 must not treat same-spin agreement as covering that case"),
        "fixture reason no longer says F4.3's same-spin agreement is not coverage: {reason}"
    );

    for gate in ["givens", "tunnel"] {
        let src = format!("FERMIONICQASM 1.0;\nmode m[2] spin;\n{gate}(0.7) m[1], m[2];\n");
        let err = match omega_parser::lower_to_ir(&src) {
            Err(e) => e,
            Ok(_) => panic!(
                "{gate} across the spin boundary lowered; F4.3 would have to invent an oracle"
            ),
        };
        assert!(
            err.contains("unpinned") && err.contains("cross-spin"),
            "{gate} must be refused as unpinned cross-spin, got: {err}"
        );
        eprintln!("not compared: cross-spin {gate} on m[1], m[2] — refused, no ffsim matrix");
    }

    // Same-spin pairs still lower. An over-broad refusal would make the
    // spinful pins in the fixture test unable to run, and this states the
    // boundary directly.
    let same =
        "FERMIONICQASM 1.0;\nmode m[2] spin;\ngivens(0.7) m[0], m[1];\ntunnel(0.7) m[2], m[3];\n";
    let ir = lower_fqasm(same);
    assert!(
        ir.ops.len() >= 2,
        "same-spin givens and tunnel did not lower"
    );

    eprintln!("cross-spin givens and tunnel: blocked, not compared. {reason}");
}

#[cfg(feature = "bridge-ffsim")]
mod live {
    use super::*;
    use num_complex::Complex64;
    use omega_bridges::{ffsim, WireObservable};
    use omega_core::circuit::{GateKind, ParamExpr};
    use omega_core::executor::{Observable, PauliOp};
    use omega_core::fermion::{FermionicOp, Ladder};
    use std::collections::BTreeMap;

    fn runner_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")
    }
    fn venv_python() -> PathBuf {
        runner_dir().join(".venv-ffsim").join("bin").join("python")
    }

    fn skip_without_venv() -> bool {
        if !venv_python().exists() {
            eprintln!(
                "ffsim venv missing at {} — FermionicQASM vs ffsim did not run. Build with \
                 `make -C crates/omega-bridges/python ffsim-venv`.",
                venv_python().display()
            );
            return true;
        }
        std::env::set_var(
            "OMEGA_BRIDGE_FFSIM_CMD",
            runner_dir().join("omega-bridge-ffsim-runner"),
        );
        false
    }

    fn current(p: u32, q: u32) -> FermionicOp {
        let t = FermionicOp::term(Complex64::i(), vec![Ladder::raise(p), Ladder::lower(q)]);
        let d = t.dagger();
        t + d
    }

    fn observables(n: u32) -> Vec<(String, Observable)> {
        let jw = |op: FermionicOp| op.jordan_wigner().expect("jordan_wigner");
        let mut v = Vec::new();
        for p in 0..n {
            v.push((format!("n_{p}"), jw(FermionicOp::number(p))));
        }
        for p in 0..n {
            for q in (p + 1)..n {
                v.push((format!("hop {p}-{q}"), jw(FermionicOp::hopping(p, q, 1.0))));
                v.push((format!("current {p}-{q}"), jw(current(p, q))));
            }
        }
        if n >= 2 {
            v.push(("n_0 n_1".into(), jw(FermionicOp::interaction(0, 1, 1.0))));
        }
        if n >= 4 {
            v.push(("n_0 n_3".into(), jw(FermionicOp::interaction(0, 3, 1.0))));
            v.push(("n_1 n_3".into(), jw(FermionicOp::interaction(1, 3, 1.0))));
        }
        v
    }

    fn to_wire(obs: &Observable, n: u32) -> WireObservable {
        obs.terms
            .iter()
            .map(|(coeff, paulis)| {
                let mut s = vec![b'I'; n as usize];
                for (q, p) in paulis {
                    s[*q as usize] = match p {
                        PauliOp::I => b'I',
                        PauliOp::X => b'X',
                        PauliOp::Y => b'Y',
                        PauliOp::Z => b'Z',
                    };
                }
                (String::from_utf8(s).unwrap(), *coeff)
            })
            .collect()
    }

    struct Reading {
        worst: f64,
        name: String,
        ours: Vec<f64>,
        theirs: Vec<f64>,
    }

    fn read(src: &str, named: &[(String, Observable)]) -> Reading {
        let ir = lower_fqasm(src);
        let obs: Vec<Observable> = named.iter().map(|(_, o)| o.clone()).collect();
        let wires: Vec<WireObservable> = obs.iter().map(|o| to_wire(o, ir.num_qubits)).collect();
        let sv = StatevectorBackend::new();
        let binding = ParameterBinding::new();
        let ours: Vec<f64> = obs
            .iter()
            .map(|o| {
                o.validate_qubits(ir.num_qubits).unwrap();
                sv.expectation(&ir, &binding, o).unwrap()
            })
            .collect();
        let theirs = ffsim::expectation(&ir, &wires)
            .unwrap_or_else(|e| panic!("ffsim refused:\n{e}\n{src}"));
        assert_eq!(ours.len(), theirs.len());
        let mut worst = 0.0;
        let mut name = String::new();
        for ((nm, _), (a, b)) in named.iter().zip(ours.iter().zip(&theirs)) {
            let d = (a - b).abs();
            if d >= worst {
                worst = d;
                name = nm.clone();
            }
        }
        Reading {
            worst,
            name,
            ours,
            theirs,
        }
    }

    fn agree(src: &str, named: &[(String, Observable)], what: &str) -> Reading {
        let reading = read(src, named);
        for (i, ((nm, _), (a, b))) in named
            .iter()
            .zip(reading.ours.iter().zip(&reading.theirs))
            .enumerate()
        {
            let d = (a - b).abs();
            assert!(
                d < TOL,
                "{what}, observable {i} ({nm}): statevector {a}, ffsim {b}, |Δ| = {d:e}\n{src}"
            );
        }
        reading
    }

    fn rbs_params(ir: &CircuitIR) -> Vec<f64> {
        ir.ops
            .iter()
            .filter(|op| op.gate == GateKind::Rbs)
            .map(|op| match op.params.as_slice() {
                [ParamExpr::Concrete(v)] => *v,
                other => panic!("Rbs param is not one concrete angle: {other:?}"),
            })
            .collect()
    }

    /// Layered 4-mode program. Same shape as `ffsim_vs_statevector`'s
    /// `layered`: both Givens orientations, a reversed pair, non-adjacent
    /// `numnum`, plus `num` and `tunnel` so the sweep is not only the two
    /// gates that file already checked.
    const SPINLESS_BODY: &str = "\
givens(0.3) m[0], m[1];
givens(-0.45) m[2], m[3];
numnum(0.7) m[1], m[2];
givens(0.52) m[1], m[2];
numnum(-0.9) m[0], m[3];
tunnel(0.21) m[2], m[3];
numnum(1.3) m[0], m[1];
givens(-0.8) m[1], m[0];
num(-0.3) m[2];
tunnel(-0.55) m[0], m[1];
";

    /// Same-spin `givens` / `tunnel` only. Cross-spin number operators
    /// (`numnum` onsite and num-prod) are in the file because those ffsim
    /// calls exist. Cross-spin `givens` / `tunnel` are not in this string.
    const SPINFUL_BODY: &str = "\
givens(0.3) m[0], m[1];
tunnel(-0.45) m[2], m[3];
numnum(0.7) m[0], m[2];
num(0.25) m[1];
givens(0.52) m[2], m[3];
numnum(-0.9) m[0], m[3];
tunnel(0.21) m[0], m[1];
numnum(1.3) m[1], m[3];
";

    fn occupations(n: u32) -> Vec<Vec<u32>> {
        (0..(1u32 << n))
            .map(|mask| (0..n).filter(|w| mask & (1 << w) != 0).collect())
            .collect()
    }

    fn number_moved(
        named: &[(String, Observable)],
        values: &[f64],
        occupied: &[u32],
        n: u32,
    ) -> bool {
        for p in 0..n {
            let want = if occupied.contains(&p) { 1.0 } else { 0.0 };
            let idx = named
                .iter()
                .position(|(nm, _)| nm == &format!("n_{p}"))
                .unwrap();
            if (values[idx] - want).abs() > 1e-2 {
                return true;
            }
        }
        false
    }

    fn max_coherence(named: &[(String, Observable)], values: &[f64]) -> f64 {
        named
            .iter()
            .zip(values)
            .filter(|((nm, _), _)| nm.starts_with("hop ") || nm.starts_with("current "))
            .map(|(_, v)| v.abs())
            .fold(0.0, f64::max)
    }

    /// All 16 occupations of 4 spinless modes: sectors N = 0, 1, 2, 3, 4
    /// with every basis state of each. N = 0 and N = 4 are run and labelled
    /// as not evidence (one basis vector, global phase only).
    #[test]
    fn every_particle_sector_of_4_modes_agrees_with_ffsim() {
        if skip_without_venv() {
            return;
        }
        let n = 4u32;
        let named = observables(n);
        let mut per_n: BTreeMap<u32, (usize, f64, bool, f64)> = BTreeMap::new();
        for occ in occupations(n) {
            let src = with_loads("mode m[4];", &occ, SPINLESS_BODY);
            let reading = agree(
                &src,
                &named,
                &format!("spinless N={} occ={occ:?}", occ.len()),
            );
            let moved = number_moved(&named, &reading.ours, &occ, n);
            let coh = max_coherence(&named, &reading.ours);
            let e = per_n
                .entry(occ.len() as u32)
                .or_insert((0, 0.0, false, 0.0));
            e.0 += 1;
            e.1 = e.1.max(reading.worst);
            e.2 |= moved;
            e.3 = e.3.max(coh);
        }
        let expect = [(0u32, 1usize), (1, 4), (2, 6), (3, 4), (4, 1)];
        assert_eq!(
            per_n.len(),
            expect.len(),
            "sectors present: {:?}",
            per_n.keys().collect::<Vec<_>>()
        );
        for (npart, count) in expect {
            let (got, worst, moved, coh) = per_n[&npart];
            assert_eq!(got, count, "N={npart} occupation count");
            if (1..4).contains(&npart) {
                assert!(
                    moved,
                    "N={npart}: every occupation stayed on its input numbers — the gates did not run"
                );
            }
            if npart == 2 {
                assert!(
                    coh > 1e-2,
                    "N=2: no hopping or current above 1e-2 (max {coh:e}); a sign error in that sector would be invisible"
                );
            }
            let note = if npart == 0 || npart == 4 {
                " (expectation is a global phase on one basis state; not evidence)"
            } else {
                ""
            };
            eprintln!(
                "spinless 4 modes N={npart}: {got} occupations, worst |Δ| = {worst:e}, max |coherence| = {coh:e}{note}"
            );
        }
    }

    /// 2 spatial modes, spin blocks, 4 wires. Every `(N_alpha, N_beta)` is
    /// in the 16 occupations. The body has no cross-spin `givens` or
    /// `tunnel`; those stay refused and are not sent to ffsim.
    #[test]
    fn every_spin_sector_of_2_spatial_modes_agrees_with_ffsim() {
        if skip_without_venv() {
            return;
        }
        let spatial = 2u32;
        let n = spatial * 2;
        for line in SPINFUL_BODY.lines() {
            let t = line.trim();
            if t.starts_with("givens") || t.starts_with("tunnel") {
                let idx = mode_indices(t);
                assert_eq!(idx.len(), 2, "{t}");
                assert_eq!(
                    idx[0] / spatial,
                    idx[1] / spatial,
                    "spinful sweep must not contain a cross-spin {t} — that case is unpinned and is not compared"
                );
            }
        }
        let cross_numnum = SPINFUL_BODY.lines().any(|line| {
            let t = line.trim();
            if !t.starts_with("numnum") {
                return false;
            }
            let idx = mode_indices(t);
            idx.len() == 2 && idx[0] / spatial != idx[1] / spatial
        });
        assert!(
            cross_numnum,
            "spinful sweep dropped the pinned cross-spin number operators"
        );

        let named = observables(n);
        let mut per: BTreeMap<(u32, u32), (usize, f64, bool, f64)> = BTreeMap::new();
        for occ in occupations(n) {
            let na = occ.iter().filter(|w| **w < spatial).count() as u32;
            let nb = occ.len() as u32 - na;
            let src = with_loads("mode m[2] spin;", &occ, SPINFUL_BODY);
            let reading = agree(
                &src,
                &named,
                &format!("spinful Na={na} Nb={nb} occ={occ:?}"),
            );
            let moved = number_moved(&named, &reading.ours, &occ, n);
            let coh = max_coherence(&named, &reading.ours);
            let e = per.entry((na, nb)).or_insert((0, 0.0, false, 0.0));
            e.0 += 1;
            e.1 = e.1.max(reading.worst);
            e.2 |= moved;
            e.3 = e.3.max(coh);
        }
        for na in 0..=spatial {
            for nb in 0..=spatial {
                let (count, worst, moved, coh) = per[&(na, nb)];
                let binom = |k| match k {
                    0 | 2 => 1,
                    1 => 2,
                    _ => panic!("spatial 2"),
                };
                assert_eq!(count, binom(na) * binom(nb), "Na={na} Nb={nb}");
                // Each species empty or full: a same-spin hop has nowhere to
                // move a particle, and a cross-spin numnum only phases. The
                // expectations of that one basis state do not depend on the
                // gates. (0,0), (0,2), (2,0), (2,2) for two spatial modes.
                let phase_only = (na == 0 || na == spatial) && (nb == 0 || nb == spatial);
                if !phase_only {
                    assert!(
                        moved,
                        "Na={na} Nb={nb}: numbers never left the input occupation"
                    );
                }
                let note = if phase_only {
                    " (expectation is a global phase on one basis state; not evidence)"
                } else {
                    ""
                };
                eprintln!(
                    "spinful Na={na} Nb={nb}: {count} occupations, worst |Δ| = {worst:e}, max |coherence| = {coh:e}{note}"
                );
            }
        }
        let (_, _, _, coh_11) = per[&(1, 1)];
        assert!(
            coh_11 > 1e-2,
            "Na=1 Nb=1: no coherence above 1e-2 (max {coh_11:e})"
        );
        eprintln!(
            "cross-spin givens and tunnel were not in this sweep: unpinned, no ffsim matrix, lowering refuses them"
        );
    }

    fn mode_indices(line: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let b = line.as_bytes();
        let mut i = 0;
        while i + 2 < b.len() {
            if b[i] == b'm' && b[i + 1] == b'[' {
                let mut j = i + 2;
                let mut n = 0u32;
                let mut any = false;
                while j < b.len() && b[j].is_ascii_digit() {
                    any = true;
                    n = n * 10 + u32::from(b[j] - b'0');
                    j += 1;
                }
                if any {
                    out.push(n);
                }
                i = j;
            } else {
                i += 1;
            }
        }
        out
    }

    fn sector_reps(entry: &CatalogEntry) -> Vec<(String, Vec<u32>)> {
        let spin = entry.mode.contains("spin");
        if !spin {
            let n = entry.n_wires;
            let mut v = Vec::new();
            for k in 0..=n {
                v.push((format!("N={k}"), (0..k).collect()));
            }
            if n >= 3 {
                v.push(("N=2 gap 0,2".into(), vec![0, 2]));
            }
            if n >= 4 {
                v.push(("N=2 gap 1,3".into(), vec![1, 3]));
                v.push(("N=3 gap 0,1,3".into(), vec![0, 1, 3]));
            }
            v
        } else {
            let spatial = entry.n_wires / 2;
            let mut v = Vec::new();
            for na in 0..=spatial {
                for nb in 0..=spatial {
                    let mut occ = Vec::new();
                    for i in 0..na {
                        occ.push(i);
                    }
                    for i in 0..nb {
                        occ.push(spatial + i);
                    }
                    v.push((format!("Na={na} Nb={nb}"), occ));
                }
            }
            if spatial >= 2 {
                v.push(("Na=1 Nb=1 orbital 1".into(), vec![1, spatial + 1]));
            }
            v
        }
    }

    /// One state per particle sector (and the gapped occupations).
    ///
    /// Two comparisons, because a single one overclaims.
    ///
    /// Live expectations (`agree`) run our statevector and ffsim on the same
    /// lowered IR. On one basis state, `num` and `numnum` only multiply that
    /// amplitude by a phase. ⟨n⟩, ⟨hop⟩, and ⟨current⟩ do not depend on it,
    /// and omitting the op omits it for both engines, so they still agree.
    /// A dropped `numnum` push left this function green when it stopped there.
    ///
    /// The F4.0 column is ffsim's `apply_*`, computed with no knowledge of
    /// this lowering. The statevector has to match that column, phase
    /// included — a global phase on a one-dimensional sector included.
    /// That is the comparison a missing `numnum` fails. Sectors where the
    /// fixture column is the basis vector itself (the gate acts as the
    /// identity there) are not witnesses; each diagonal gate has to have
    /// at least one sector that is.
    #[test]
    fn each_lowered_gate_matches_ffsim_in_every_sector() {
        if skip_without_venv() {
            return;
        }
        let mut worst = 0.0;
        let mut worst_at = String::new();
        let mut worst_amp = 0.0;
        let mut worst_amp_at = String::new();
        let mut n_compared = 0usize;
        for entry in CATALOG {
            if entry.not_compared.is_some() {
                continue;
            }
            let named = observables(entry.n_wires);
            let mut witnessed = false;
            for (sector, occ) in sector_reps(entry) {
                let src = with_loads(entry.mode, &occ, entry.gate);
                let reading = agree(&src, &named, &format!("{} {sector}", entry.key));
                let amp = fixture_column_delta(entry.key, &occ);
                assert!(
                    amp < TOL,
                    "{} {sector}: statevector column misses the F4.0 ffsim matrix by {amp:e}. \
                     Live expectation agreement does not see a diagonal phase on one basis \
                     state, and both engines share the lowered IR, so a dropped num or numnum \
                     still agrees. This column is the comparison that sees it.\n{src}",
                    entry.key
                );
                n_compared += 1;
                if reading.worst > worst {
                    worst = reading.worst;
                    worst_at = format!("{} {sector} {}", entry.key, reading.name);
                }
                if amp > worst_amp {
                    worst_amp = amp;
                    worst_amp_at = format!("{} {sector}", entry.key);
                }
                if fixture_column_moves_the_basis(entry.key, entry.n_wires, &occ) > 1e-2 {
                    witnessed = true;
                }
            }
            // `num` / `numnum` only. A mixing gate moves amplitudes in every
            // sector that has a particle it can hop; the diagonal gates are
            // the ones whose phase is invisible to `agree`.
            if entry.gate.starts_with("num") {
                assert!(
                    witnessed,
                    "{}: no sector representative has an F4.0 column that differs from \
                     the basis vector, so a dropped {} would not move any amplitude \
                     this sweep looks at",
                    entry.key, entry.gate
                );
            }
        }
        assert!(n_compared > 0);
        eprintln!(
            "per-gate: {n_compared} (gate, sector) pairs, live expectation worst |Δ| = {worst:e} at {worst_at}, F4.0 column worst |Δ| = {worst_amp:e} at {worst_amp_at}"
        );
    }

    /// `num` / `numnum` on one basis vector are a phase of that vector.
    /// Expectations of that one vector do not depend on the phase. These
    /// programs put a real Givens in front of the diagonal gate so the phase
    /// becomes a relative phase, which `current` / `hop` can see. Each
    /// program's coherence has to be nonzero or the agreement is not a pin.
    #[test]
    fn diagonal_phase_is_visible_to_live_ffsim() {
        if skip_without_venv() {
            return;
        }
        let programs = [
            (
                "num after givens",
                "mode m[2];",
                2,
                "load m[0];\ngivens(0.4) m[0], m[1];\nnum(0.7) m[0];\n",
            ),
            (
                "num on the other mode",
                "mode m[2];",
                2,
                "load m[1];\ngivens(0.4) m[0], m[1];\nnum(0.7) m[1];\n",
            ),
            (
                // |110⟩ through givens(1, 2) is a superposition of |110⟩ and |101⟩.
                // numnum(0, 1) phases only |110⟩. On |11⟩ of the pair itself the
                // gate is a global phase and the comparison would be blind.
                "numnum after givens",
                "mode m[3];",
                3,
                "load m[0];\nload m[1];\ngivens(0.4) m[1], m[2];\nnumnum(0.7) m[0], m[1];\n",
            ),
            (
                "numnum pair reversed",
                "mode m[3];",
                3,
                "load m[0];\nload m[1];\ngivens(0.4) m[1], m[2];\nnumnum(0.7) m[1], m[0];\n",
            ),
            (
                "non-adjacent numnum",
                "mode m[3];",
                3,
                "load m[0];\nload m[2];\ngivens(0.4) m[0], m[1];\nnumnum(0.7) m[0], m[2];\n",
            ),
            (
                "onsite after givens",
                "mode m[2] spin;",
                4,
                "load m[0];\nload m[2];\ngivens(0.4) m[0], m[1];\nnumnum(0.7) m[0], m[2];\n",
            ),
            (
                "num-prod after givens",
                "mode m[2] spin;",
                4,
                "load m[0];\nload m[3];\ngivens(0.4) m[0], m[1];\nnumnum(0.7) m[0], m[3];\n",
            ),
            (
                "beta num after beta givens",
                "mode m[2] spin;",
                4,
                "load m[2];\ngivens(0.4) m[2], m[3];\nnum(0.7) m[2];\n",
            ),
        ];
        for (name, mode, n, body) in programs {
            let src = format!("FERMIONICQASM 1.0;\n{mode}\n{body}");
            let named = observables(n);
            let reading = agree(&src, &named, name);
            let coh = max_coherence(&named, &reading.ours);
            assert!(
                coh > 1e-2,
                "{name}: agreed expectations have no coherence (max {coh:e}); the phase is invisible and the agreement is not a pin\n{src}"
            );
            eprintln!(
                "{name}: worst |Δ| = {:e}, |coherence| = {coh:e}",
                reading.worst
            );
        }
    }

    /// How far the F4.0 column sits from the computational-basis vector.
    ///
    /// Zero means the gate does not change that occupation, so matching it is
    /// not evidence the gate was lowered. `numnum` on a state that does not
    /// occupy both modes is that case; the doubly-occupied column is not.
    fn fixture_column_moves_the_basis(key: &str, n_wires: u32, occupied: &[u32]) -> f64 {
        let col = occupied
            .iter()
            .fold(0usize, |mask, w| mask | (1usize << *w));
        let matrix = &conventions()["gates"][key]["matrix"];
        let dim = 1usize << n_wires;
        let mut worst = 0.0f64;
        for row in 0..dim {
            let pair = &matrix[row][col];
            let want_re = if row == col { 1.0 } else { 0.0 };
            let d = (pair[0].as_f64().unwrap() - want_re).hypot(pair[1].as_f64().unwrap());
            worst = worst.max(d);
        }
        worst
    }

    /// Amplitude distance from one column of an F4.0 gate matrix. The matrix
    /// was produced by ffsim's `apply_*`, not by this lowering, so a bug in
    /// the lowering moves this number. Sending the same (wrong) circuit to
    /// both our statevector and the ffsim runner does not: they share it.
    fn fixture_column_delta(key: &str, occupied: &[u32]) -> f64 {
        let entry = CATALOG
            .iter()
            .find(|e| e.key == key)
            .unwrap_or_else(|| panic!("no catalog entry {key}"));
        let src = with_loads(entry.mode, occupied, entry.gate);
        let amps = evolve(&lower_fqasm(&src));
        let col = occupied.iter().fold(0usize, |mask, w| mask | (1usize << w));
        let matrix = &conventions()["gates"][key]["matrix"];
        let mut worst = 0.0f64;
        for (row, amp) in amps.iter().enumerate() {
            let pair = &matrix[row][col];
            let d = (amp.re - pair[0].as_f64().unwrap()).hypot(amp.im - pair[1].as_f64().unwrap());
            worst = worst.max(d);
        }
        worst
    }

    fn ours_only(src: &str, named: &[(String, Observable)]) -> Vec<f64> {
        let ir = lower_fqasm(src);
        let sv = StatevectorBackend::new();
        let binding = ParameterBinding::new();
        named
            .iter()
            .map(|(_, o)| {
                o.validate_qubits(ir.num_qubits).unwrap();
                sv.expectation(&ir, &binding, o).unwrap()
            })
            .collect()
    }

    /// The plan's non-triviality pin.
    ///
    /// Occupying both wires of the tunnelled pair makes `tunnel` the
    /// identity, so a sign flip of θ would not move any amplitude. The
    /// 2-particle case occupies mode 0 and mode 2: one particle on the pair
    /// and a spectator. The 1-particle case is beside it.
    ///
    /// The hook flips θ inside the lowering. Comparing the flipped circuit
    /// to ffsim on that same circuit cannot fail — both engines lower once
    /// and agree on the mistake. The oracle held fixed is the F4.0 matrix
    /// (ffsim's `apply_tunneling_interaction`, computed with no knowledge of
    /// this lowering) and the live ffsim expectations of the *unflipped*
    /// program. The flipped state has to miss both.
    #[test]
    fn tunnel_theta_sign_flip_is_detected_as_disagreement() {
        if skip_without_venv() {
            return;
        }
        let cases = [
            (
                "N=1",
                "spinless_4/tunnel_0_1",
                &[0u32][..],
                "FERMIONICQASM 1.0;\nmode m[4];\nload m[0];\ntunnel(0.7) m[0], m[1];\n",
            ),
            (
                "N=2 spectator on 2",
                "spinless_4/tunnel_0_1",
                &[0u32, 2][..],
                "FERMIONICQASM 1.0;\nmode m[4];\nload m[0];\nload m[2];\ntunnel(0.7) m[0], m[1];\n",
            ),
        ];
        let named = observables(4);
        for (label, fixture_key, occupied, src) in cases {
            let ir = lower_fqasm(src);
            let pinned = rbs_params(&ir);
            assert_eq!(pinned.len(), 1, "{label}: tunnel should lower to one Rbs");
            assert!(
                (pinned[0] - (-0.7)).abs() < 1e-12,
                "{label}: pinned tunnel Rbs angle is {} , want -0.7",
                pinned[0]
            );
            let plain_amp = fixture_column_delta(fixture_key, occupied);
            assert!(
                plain_amp < TOL,
                "{label}: unflipped tunnel misses the F4.0 matrix by {plain_amp:e}"
            );
            let plain = agree(src, &named, &format!("unflipped tunnel {label}"));
            let coh = max_coherence(&named, &plain.ours);
            assert!(
                coh > 1e-2,
                "{label}: unflipped tunnel has no coherence (max {coh:e}); flipping θ could not be seen"
            );

            let (flipped_amp, exp_delta, exp_name, exp_ours, exp_ffsim) =
                omega_parser::with_tunnel_theta_sign_flipped(|| {
                    let ir = lower_fqasm(src);
                    let ang = rbs_params(&ir);
                    assert!(
                        (ang[0] - 0.7).abs() < 1e-12,
                        "{label}: hook did not flip the Rbs angle to +0.7 (got {})",
                        ang[0]
                    );
                    let amp = fixture_column_delta(fixture_key, occupied);
                    let ours = ours_only(src, &named);
                    let mut best = 0.0;
                    let mut best_name = String::new();
                    let mut best_ours = 0.0;
                    let mut best_ffsim = 0.0;
                    for ((nm, _), (a, b)) in named.iter().zip(ours.iter().zip(&plain.theirs)) {
                        let d = (a - b).abs();
                        if d >= best {
                            best = d;
                            best_name = nm.clone();
                            best_ours = *a;
                            best_ffsim = *b;
                        }
                    }
                    (amp, best, best_name, best_ours, best_ffsim)
                });
            assert!(
                flipped_amp > 1e-2,
                "{label}: flipping theta was NOT detected against the F4.0 matrix. \
                 |Δ| = {flipped_amp:e}. The comparison does not have teeth."
            );
            assert!(
                exp_delta > 1e-2,
                "{label}: flipping theta was NOT detected against live ffsim's unflipped \
                 expectations. Worst |Δ| = {exp_delta:e} on {exp_name}. \
                 Same-circuit agreement would hide this; the oracle has to stay the pinned one."
            );
            eprintln!(
                "non-triviality pin {label}: flipped tunnel θ, fixture |Δ| = {flipped_amp:e} (unflipped {plain_amp:e}), live expectation |Δ| = {exp_delta:e} on {exp_name} (statevector {exp_ours}, ffsim {exp_ffsim})"
            );
        }
    }
}
