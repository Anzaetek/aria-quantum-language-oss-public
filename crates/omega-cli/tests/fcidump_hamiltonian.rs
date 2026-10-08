// SPDX-License-Identifier: Apache-2.0
//! `--hamiltonian h.fcidump` on a FermionicQASM circuit.
//!
//! The FCIDUMP is bytes PySCF wrote (`tests/fixtures/fcidump/hubbard.fcidump`).
//! The ladder string below is the same Hubbard operator assembled by hand:
//! core energy 0.5, hopping −1 within each spin block, U = 4 on
//! n_0 n_2 and n_1 n_3. The two `omega-run` invocations share the circuit
//! and the sector backend and nothing else, so a reader that drops the ½
//! or the wrong index order disagrees with the hand string, and a missing
//! `--hamiltonian` flag exits on the unknown-argument path.

use std::path::PathBuf;
use std::process::{Command, Stdio};

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_omega-run"))
}

fn hubbard_fcidump() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../omega-core/tests/fixtures/fcidump/hubbard.fcidump")
}

const HUBBARD_BY_HAND: &str = "\
0.5 [] + -1 [0^ 1] + -1 [1^ 0] + -1 [2^ 3] + -1 [3^ 2] \
+ 4 [0^ 0 2^ 2] + 4 [1^ 1 3^ 3]";

const CIRCUIT: &str = "\
FERMIONICQASM 1.0;
mode m[2] spin;
load m[0];
load m[2];
givens(0.3) m[0], m[1];
tunnel(0.2) m[2], m[3];
";

fn write_circuit() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omega_fcidump_hubbard_{}.fqasm",
        std::process::id()
    ));
    std::fs::write(&path, CIRCUIT).expect("write circuit");
    path
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(binary_path())
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn omega-run")
}

fn expectation(args: &[&str]) -> f64 {
    let out = run(args);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "omega-run {args:?} failed (status {:?}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status.code()
    );
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.contains("<O> ="))
        .unwrap_or_else(|| panic!("no `<O> =` line in output:\n{stdout}\nstderr:\n{stderr}"));
    line.split('=')
        .nth(1)
        .unwrap()
        .trim()
        .parse()
        .expect("parse <O>")
}

#[test]
fn hamiltonian_fcidump_matches_hand_assembled_expectation_on_sector() {
    let circuit = write_circuit();
    let file = circuit.to_str().unwrap();
    let dump = hubbard_fcidump();
    let dump = dump.to_str().unwrap();

    let both = run(&[
        file,
        "--hamiltonian",
        dump,
        "--expectation-fermionic",
        HUBBARD_BY_HAND,
        "--backend",
        "sector",
    ]);
    let stderr = String::from_utf8_lossy(&both.stderr);
    assert_eq!(
        both.status.code(),
        Some(2),
        "--hamiltonian with --expectation-fermionic must be refused as two modes, got {:?}: {stderr}",
        both.status.code()
    );
    assert!(
        stderr.contains("--hamiltonian") && stderr.contains("--expectation-fermionic"),
        "the refusal must name both flags, got: {stderr}"
    );

    let from_file = expectation(&[file, "--hamiltonian", dump, "--backend", "sector"]);
    let by_hand = expectation(&[
        file,
        "--expectation-fermionic",
        HUBBARD_BY_HAND,
        "--backend",
        "sector",
    ]);
    assert!(
        (from_file - by_hand).abs() < 1e-9,
        "sector <H>_fcidump {from_file} vs <H>_hand {by_hand}"
    );
    assert!(
        (from_file - 0.5).abs() > 1e-6,
        "expectation {from_file} is only the core energy; the integrals are not in the measurement"
    );
    let _ = std::fs::remove_file(&circuit);
}
