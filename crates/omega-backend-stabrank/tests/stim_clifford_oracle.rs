// SPDX-License-Identifier: Apache-2.0
//! S0 test (iii): the Clifford fragment against **Stim**, exact integers,
//! through the existing QEC venv — registered skip when it is absent.
//!
//! Stim is the oracle and never the implementation (PLAN-MAJORANA-STIM.md
//! §1.3). It is not linked, not vendored, and not a dependency of this
//! crate: this file shells out to `tools/qec_cross_check/.venv`, the venv
//! `tools/qec_cross_check/run.sh` builds, and compares numbers. Stim cannot
//! be the implementation even if someone wanted it to be — its tableau
//! discards the global phase this kernel exists to keep, and it exports no
//! stabilizer-state inner product. What it *can* do, better than anything
//! else available, is tell us whether the Clifford fragment is right, as a
//! genuinely independent implementation of the same algorithm.
//!
//! `peek_observable_expectation` returns exact integers on a stabilizer
//! state. Routing through Stim's `state_vector()` instead would be
//! `complex64` (measured error 1.21e-08, FIXES_PLAN K8 trap 1) and would
//! silently turn an exact anchor into a 1e-7 one.
//!
//! **The oracle side is exact; our side is not, and the assertion says so.**
//! `⟨P⟩` here is the real part of `⟨φ|P|φ⟩`, computed through a chain of
//! `1/√2` factors, and it comes back as `−1.0000000000000004` rather than
//! `−1` often enough that an `==` would be a false alarm about rounding
//! rather than a check on the physics. So each cell asserts two things:
//! the value rounds to Stim's integer (identification, exact), and it is
//! within 1e-12 of it (the accumulated error is rounding and nothing
//! more). That is strictly stronger than comparing two floats at 1e-12,
//! because the right-hand side is an exact integer and not another
//! estimate.
//!
//! # Which mutation this fixture catches, and which it does not
//!
//! Catches: any defect that changes which stabilizer group the kernel
//! ends up in — a wrong `F`/`G`/`M` update on any gate in the table.
//!
//! Does NOT catch **any phase defect at all**. Stim's tableau has no
//! global phase, and `⟨P⟩` on a single state is invariant under one, so
//! this fixture is green for a kernel whose every amplitude carries a
//! spurious `i`. That is exactly why it is the third leg and not the
//! first: `amplitude_vs_statevector.rs` (complex amplitudes) and
//! `global_phase_pin.rs` (an absolute phase) carry what Stim structurally
//! cannot. Conversely those two share the dense backend as their oracle,
//! and this file is the only leg whose oracle is a different codebase
//! entirely.

use omega_backend_stabrank::{ChForm, StabRankBackend};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::PauliOp;
use omega_core::params::ParameterBinding;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Reads the job on stdin, writes one expectation per line.
///
/// Line protocol, so neither side needs a JSON dependency:
/// `N <qubits>`, then `G <stim gate line>` in order, then `P <pauli>` per
/// observable.
const ORACLE: &str = r#"
import sys, stim
n, gates, paulis = 0, [], []
for line in sys.stdin:
    tag, _, rest = line.rstrip("\n").partition(" ")
    if tag == "N":
        n = int(rest)
    elif tag == "G":
        gates.append(rest)
    elif tag == "P":
        paulis.append(rest)
sim = stim.TableauSimulator()
sim.set_num_qubits(n)
sim.do(stim.Circuit("\n".join(gates)))
for p in paulis:
    print(sim.peek_observable_expectation(stim.PauliString(p)))
"#;

fn venv_python() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tools/qec_cross_check/.venv/bin/python")
}

/// The registered skip. Loud, names the path and the script that builds
/// it, and says which half of S0's evidence was not collected.
fn skip_without_stim() -> bool {
    let py = venv_python();
    let importable = py.exists()
        && Command::new(&py)
            .args(["-c", "import stim"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
    if !importable {
        eprintln!(
            "QEC venv with stim missing at {} — the independent Clifford-fragment \
             oracle did NOT run. The dense-statevector legs still ran, but they \
             share one codebase; nothing here was checked against a second \
             implementation. Build it with `tools/qec_cross_check/run.sh`.",
            py.display()
        );
        return true;
    }
    false
}

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, k: usize) -> usize {
        (self.next_u64() % k as u64) as usize
    }
}

/// `(our IR gate, Stim's name for it)`. Written out rather than derived so
/// a rename on either side is a compile error or a visible edit, not a
/// silent remap onto the wrong gate.
const TABLE_1Q: [(GateKind, &str); 8] = [
    (GateKind::H, "H"),
    (GateKind::S, "S"),
    (GateKind::Sdg, "S_DAG"),
    (GateKind::X, "X"),
    (GateKind::Y, "Y"),
    (GateKind::Z, "Z"),
    (GateKind::Sx, "SQRT_X"),
    (GateKind::Sxdg, "SQRT_X_DAG"),
];
const TABLE_2Q: [(GateKind, &str); 4] = [
    (GateKind::CX, "CX"),
    (GateKind::CY, "CY"),
    (GateKind::CZ, "CZ"),
    (GateKind::Swap, "SWAP"),
];

/// One circuit, built once and emitted both ways so the two sides cannot
/// drift.
struct Job {
    n: u32,
    ir: CircuitIR,
    stim: Vec<String>,
}

fn random_job(n: u32, depth: usize, rng: &mut Lcg) -> Job {
    let mut ir = CircuitIR::new(n, CircuitType::GateBased);
    let mut stim = Vec::new();
    for _ in 0..depth {
        let (kind, name, wires) = if n >= 2 && rng.below(3) == 0 {
            let a = rng.below(n as usize) as u32;
            let mut b = rng.below(n as usize) as u32;
            while b == a {
                b = rng.below(n as usize) as u32;
            }
            let (k, s) = &TABLE_2Q[rng.below(TABLE_2Q.len())];
            (k.clone(), *s, vec![a, b])
        } else {
            let a = rng.below(n as usize) as u32;
            let (k, s) = &TABLE_1Q[rng.below(TABLE_1Q.len())];
            (k.clone(), *s, vec![a])
        };
        ir.add_op(GateOp {
            gate: kind,
            qubits: wires.iter().map(|&q| Qubit(q)).collect(),
            params: Default::default(),
            classical_bit: None,
            condition: None,
        });
        let args: Vec<String> = wires.iter().map(|q| q.to_string()).collect();
        stim.push(format!("{name} {}", args.join(" ")));
    }
    Job { n, ir, stim }
}

/// Stim's `PauliString` is LSB-first, the same order as our wire indices,
/// so character `q` names wire `q` with no reversal. (Pinned by
/// `omega-bridges/tests/stim_expectation.rs::stim_reads_pauli_strings_lsb_first`.)
fn pauli_text(n: u32, p: &[(usize, PauliOp)]) -> String {
    let mut s = vec!['_'; n as usize];
    for &(q, l) in p {
        s[q] = match l {
            PauliOp::I => '_',
            PauliOp::X => 'X',
            PauliOp::Y => 'Y',
            PauliOp::Z => 'Z',
        };
    }
    format!("+{}", s.into_iter().collect::<String>())
}

fn ask_stim(job: &Job, observables: &[String]) -> Vec<f64> {
    let mut child = Command::new(venv_python())
        .args(["-c", ORACLE])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn the QEC venv python");
    {
        let stdin = child.stdin.as_mut().expect("stdin");
        writeln!(stdin, "N {}", job.n).unwrap();
        for g in &job.stim {
            writeln!(stdin, "G {g}").unwrap();
        }
        for o in observables {
            writeln!(stdin, "P {o}").unwrap();
        }
    }
    let out = child.wait_with_output().expect("stim oracle");
    assert!(
        out.status.success(),
        "the stim oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().parse::<f64>().expect("an expectation per line"))
        .collect()
}

fn kernel(ir: &CircuitIR) -> ChForm {
    StabRankBackend::new()
        .simulate(ir, &ParameterBinding::new())
        .expect("Clifford circuit")
}

fn observables_for(n: u32, rng: &mut Lcg) -> Vec<Vec<(usize, PauliOp)>> {
    let letters = [PauliOp::X, PauliOp::Y, PauliOp::Z];
    let mut v: Vec<Vec<(usize, PauliOp)>> = Vec::new();
    for q in 0..n as usize {
        for l in letters {
            v.push(vec![(q, l)]);
        }
    }
    for _ in 0..6 {
        let mut term = Vec::new();
        for q in 0..n as usize {
            if rng.below(2) == 0 {
                term.push((q, letters[rng.below(3)]));
            }
        }
        v.push(term);
    }
    v
}

/// The comparison.
#[test]
fn clifford_expectations_equal_stims_exactly() {
    if skip_without_stim() {
        return;
    }
    let mut compared = 0usize;
    let mut decisive = 0usize;
    for n in 1..=6u32 {
        for seed in 0..12u64 {
            let mut rng = Lcg(0x57_1707 + seed * 104_729 + u64::from(n));
            let job = random_job(n, 6 + 4 * n as usize, &mut rng);
            let obs = observables_for(n, &mut rng);
            let text: Vec<String> = obs.iter().map(|p| pauli_text(n, p)).collect();
            let theirs = ask_stim(&job, &text);
            assert_eq!(theirs.len(), obs.len(), "the oracle dropped a row");
            let state = kernel(&job.ir);
            for (p, want) in obs.iter().zip(theirs.iter()) {
                let got = state.pauli_expectation(p);
                assert_eq!(
                    got.im, 0.0,
                    "n={n} seed={seed} P={p:?}: ⟨P⟩ is not real ({got})"
                );
                assert_eq!(
                    got.re.round(),
                    *want,
                    "n={n} seed={seed} P={p:?}: stabrank {} vs stim {want}. Stim's \
                     value is an exact integer, so this is a real disagreement.",
                    got.re
                );
                assert!(
                    (got.re - *want).abs() < 1e-12,
                    "n={n} seed={seed} P={p:?}: stabrank {} rounds to stim's {want} \
                     but is {:.3e} away from it — that is accumulated error, not \
                     rounding",
                    got.re,
                    (got.re - *want).abs()
                );
                if *want != 0.0 {
                    decisive += 1;
                }
                compared += 1;
            }
        }
    }
    eprintln!("stabrank vs stim 1.16.0: {compared} expectations, {decisive} of them ±1");
    assert!(compared >= 300, "only {compared} cells compared");
    assert!(
        decisive >= 100,
        "only {decisive} of {compared} comparisons were ±1. A sweep that is \
         almost all zeros agrees with anything that answers 0, which includes \
         a kernel that lost the circuit entirely."
    );
}

/// The sensitivity control for the leg above. One extra `S` on wire 0,
/// compared against Stim's answer for the *unmodified* circuit, has to
/// disagree somewhere — otherwise the agreement test is measuring nothing.
#[test]
fn an_inserted_s_gate_makes_the_two_disagree() {
    if skip_without_stim() {
        return;
    }
    let mut rng = Lcg(0x5e_1151);
    let n = 4u32;
    let job = random_job(n, 20, &mut rng);
    let obs = observables_for(n, &mut rng);
    let text: Vec<String> = obs.iter().map(|p| pauli_text(n, p)).collect();
    let theirs = ask_stim(&job, &text);

    let mut perturbed = job.ir.clone();
    perturbed.add_op(GateOp {
        gate: GateKind::S,
        qubits: vec![Qubit(0)].into(),
        params: Default::default(),
        classical_bit: None,
        condition: None,
    });
    let state = kernel(&perturbed);
    let disagreements = obs
        .iter()
        .zip(theirs.iter())
        .filter(|(p, want)| state.pauli_expectation(p).re.round() != **want)
        .count();
    assert!(
        disagreements > 0,
        "inserting an S changed nothing the oracle can see, so the equality \
         test above is not evidence for this circuit"
    );
    eprintln!(
        "sensitivity control: {disagreements} of {} cells moved",
        obs.len()
    );
}
