// SPDX-License-Identifier: Apache-2.0
//! FCIDUMP → `FermionicOp`.
//!
//! The fixtures under `tests/fixtures/fcidump/` are bytes PySCF wrote.
//! `generate.py --check` regenerates them and diffs; a hand-edited file
//! fails that diff. `--expand` is PySCF reading the same file back
//! (`fcidump.read` + `ao2mo.restore`) and printing the chemist Hamiltonian
//! as ladder lines. This file does not write an FCIDUMP.
//!
//! Without the venv the two PySCF legs skip out loud. The hand-assembled
//! Hubbard comparison and the malformed-namelist refusal do not need it:
//! they run against the committed bytes and against inline text.

use std::path::PathBuf;
use std::process::Command;

use num_complex::Complex64;
use omega_core::fcidump;
use omega_core::fermion::{FermionicOp, Ladder};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fcidump")
}

fn venv_python() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../omega-bridges/python/.venv-ffsim/bin/python")
}

fn skip_without_venv() -> bool {
    if !venv_python().exists() {
        eprintln!(
            "pyscf venv missing at {} — FCIDUMP vs pyscf did not run. Build with \
             `make -C crates/omega-bridges/python ffsim-venv`.",
            venv_python().display()
        );
        return true;
    }
    false
}

fn run_generator(args: &[&str]) -> std::process::Output {
    Command::new(venv_python())
        .arg(fixture_dir().join("generate.py"))
        .args(args)
        .output()
        .expect("spawn pyscf")
}

/// t = 1 within each spin block, U = 4 density-density, core energy 0.5.
/// Assembled as ladder text, not by reading the FCIDUMP.
const HUBBARD_BY_HAND: &str = "\
0.5 [] + -1 [0^ 1] + -1 [1^ 0] + -1 [2^ 3] + -1 [3^ 2] \
+ 4 [0^ 0 2^ 2] + 4 [1^ 1 3^ 3]";

fn term_map(op: &FermionicOp) -> std::collections::BTreeMap<Vec<Ladder>, Complex64> {
    op.normal_ordered()
        .terms
        .into_iter()
        .map(|(c, p)| (p, c))
        .collect()
}

fn assert_same_operator(what: &str, got: &FermionicOp, expected: &FermionicOp) {
    let a = term_map(got);
    let b = term_map(expected);
    let mut keys: Vec<Vec<Ladder>> = a.keys().chain(b.keys()).cloned().collect();
    keys.sort();
    keys.dedup();
    for key in keys {
        let ca = a.get(&key).copied().unwrap_or(Complex64::new(0.0, 0.0));
        let cb = b.get(&key).copied().unwrap_or(Complex64::new(0.0, 0.0));
        if ca.norm() < 1e-8 && cb.norm() < 1e-8 {
            continue;
        }
        let delta = (ca - cb).norm();
        assert!(
            delta < 1e-8,
            "{what}: term {key:?}: reader {ca} vs oracle {cb} (delta {delta})"
        );
    }
}

#[test]
fn fcidump_fixtures_are_bytes_pyscf_wrote() {
    if skip_without_venv() {
        return;
    }
    let out = run_generator(&["--check"]);
    assert!(
        out.status.success(),
        "generate.py --check failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(fixture_dir().join("hubbard.fcidump")).unwrap();
    let op = fcidump::parse(&text).expect("hubbard.fcidump");
    assert_eq!(
        op.num_modes(),
        4,
        "reader did not consume the four modes PySCF wrote"
    );
    let scalar = term_map(&op)
        .get(&Vec::new())
        .copied()
        .unwrap_or(Complex64::new(0.0, 0.0));
    assert!(
        (scalar.re - 0.5).abs() < 1e-12 && scalar.im.abs() < 1e-12,
        "core energy from the PySCF bytes: {scalar}"
    );
}

#[test]
fn fcidump_terms_match_pyscf_read() {
    if skip_without_venv() {
        return;
    }
    for name in ["hubbard.fcidump", "h2_sto3g.fcidump", "random3.fcidump"] {
        let path = fixture_dir().join(name);
        let text = std::fs::read_to_string(&path).unwrap();
        let got = fcidump::parse(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        let out = run_generator(&["--expand", path.to_str().unwrap()]);
        assert!(
            out.status.success(),
            "--expand {name} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let mut expected = FermionicOp::zero();
        let stdout = String::from_utf8(out.stdout).unwrap();
        for line in stdout.lines() {
            if line.is_empty() {
                continue;
            }
            let term = FermionicOp::parse(line)
                .unwrap_or_else(|e| panic!("{name}: oracle line {line:?}: {e}"));
            expected = expected + term;
        }
        assert!(
            !stdout.trim().is_empty(),
            "{name}: pyscf expand produced no terms, so agreement would be vacuous"
        );
        assert_same_operator(name, &got, &expected);
    }
}

#[test]
fn hubbard_fcidump_equals_the_hand_assembled_operator() {
    let text = std::fs::read_to_string(fixture_dir().join("hubbard.fcidump")).unwrap();
    let got = fcidump::parse(&text).expect("hubbard.fcidump");
    let hand = FermionicOp::parse(HUBBARD_BY_HAND).expect("hand string");
    assert_same_operator("hubbard by hand", &got, &hand);
}

#[test]
fn malformed_fcidump_namelist_names_the_field() {
    let ms2 = "\
 &FCI NORB=4,NELEC=2,MS2=1.5,
  ORBSYM=1,1,1,1,
  ISYM=1,
 &END
 0.0  0  0  0  0
";
    let err = fcidump::parse(ms2).expect_err("MS2=1.5 must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("MS2"),
        "refusal must name the field MS2, got: {msg}"
    );

    // `1,no` would be split into a second namelist token (`NO`), because a
    // comma followed by a letter starts the next field — the same cut PySCF's
    // reader makes. `2.5` stays inside ORBSYM, so the refusal can name it.
    let orbsym = "\
 &FCI NORB=2,NELEC=2,MS2=0,
  ORBSYM=1,2.5,
  ISYM=1,
 &END
 0.0  0  0  0  0
";
    let err = fcidump::parse(orbsym).expect_err("ORBSYM=1,2.5 must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("ORBSYM"),
        "refusal must name the field ORBSYM, got: {msg}"
    );
}
