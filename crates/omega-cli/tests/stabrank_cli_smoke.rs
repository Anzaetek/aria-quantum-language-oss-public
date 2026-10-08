// SPDX-License-Identifier: Apache-2.0
//! S3 test (i): **`--backend stabrank` exists, and a fermionic observable
//! string reaches it.** stabrank against the dense statevector, through the
//! CLI, on the `--expectation-fermionic` spelling and on the hand-written
//! Pauli image of the same operator.
//!
//! # How this fails if the wiring is absent, and why that matters
//!
//! It fails **at the exit code, not at a tolerance**. Without an arm for
//! `stabrank` in the CLI's expectation dispatch the run reaches the `_` arm,
//! prints `Expectation not supported for backend: stabrank` and exits `1`;
//! there is no number to compare and no tolerance that could be loosened to
//! make it pass. So every assertion here reads `status.code()` explicitly
//! rather than `status.success()`, and reads it *before* parsing stdout —
//! `serde_json::from_slice` on the empty stdout of a failed run panics with a
//! parse error, which is a true failure reported as the wrong thing.
//!
//! The statevector leg is run through the same CLI with the same fixture and
//! the same observable string, so what is compared is two backends and not
//! two spellings.
//!
//! # Zero engine changes, which is the point of this test
//!
//! `--expectation-fermionic` works on stabrank without the engine having
//! gained a fermionic door, and that is PLAN-MAJORANA-STIM.md's F2 pattern
//! rather than an accident. The CLI lowers the ladder string to its
//! Jordan–Wigner Pauli image once, for every backend; the image is exact, so
//! nothing is lost; and stabrank is Schrödinger, so its seed is the state
//! `|0…0⟩` and the observable is read out at the end whatever basis it was
//! typed in. There is no ladder-native seeding to choose on the state side —
//! the state is not an operator sum — which is why the certificate's
//! `seed_basis` has exactly one variant (§1.4). majoranaprop needs a direct
//! fermionic door because *its* seed is the operator.
//!
//! The two spellings are therefore required to agree exactly, not
//! approximately: `the_ladder_spelling_and_its_pauli_image_agree` compares
//! them, and the `observable_mapping` block is asserted so a reader can see
//! which image was measured rather than taking it on faith.
//!
//! # Fixture symmetry (plan §2), measured
//!
//! K8's third trap was a `t → tdg` mutation scoring exactly `0.00e+00`
//! because the fixture measured `⟨X⟩ = cos θ`, which cannot tell a `T` from a
//! `T†`. The standing rule is that every T-sensitive fixture measures a leg
//! that can. This one does, and the measurement is on the record here rather
//! than asserted in the abstract: the observable's hopping terms carry both
//! an `X`-string and a `Y`-string leg under Jordan–Wigner
//! (`a†₀a₁ + a†₁a₀ → 0.375·(X₀X₁ + Y₀Y₁)`), and replacing the `t q[2]` of the
//! fixture by `tdg q[2]` moves the dense value from `0.666421356237` to
//! `0.525000000000`. `a_t_dagger_in_the_fixture_is_a_different_number` runs
//! both circuits and asserts the gap, so a stabrank that executed `T†` as `T`
//! — or either as `S` — cannot pass by measuring an even function of the
//! angle.
//!
//! # One mutation this test does NOT catch
//!
//! **A wrong error bound.** Measured, not supposed: with
//! `backend.rs:391`'s `R·m·(2+m)` replaced by `R·m·m`, this file stays
//! entirely green. Every run here is untruncated, so `m = 0` and the bound is
//! `0` under either expression; the certificate is read only for `exact` and
//! for `seed_basis`. The bound's soundness is
//! `stabrank_majoranaprop_intervals.rs`, and its consistency with its own
//! inputs is `stabrank_json_roundtrip.rs`. This file is about the door.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;

/// Clifford + T on four wires, with one Clifford-angle Givens rotation —
/// plan §1.4's overlap regime, where both engines are cheap and exact, so a
/// disagreement is a defect rather than a bound.
///
/// Six `T` gates, hence `χ = 64` branches: large enough that the engine is
/// really running a decomposition and small enough to be instant in a debug
/// build. The `H` layer is load-bearing. A `T` is diagonal and a quarter-turn
/// Givens rotation is a signed permutation of the occupation basis, so
/// without it the circuit never leaves a computational basis state, every
/// expectation collapses to a coefficient of the observable, and all three
/// engines would agree for free.
///
/// `rbs` is in no QASM2 dialect's standard library, hence
/// `--qasm-dialect lenient` on every invocation below.
const QASM: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\n\
h q[0];\nh q[1];\nh q[2];\nh q[3];\nt q[0];\nt q[1];\nt q[2];\nt q[3];\n\
s q[1];\nrbs(1.5707963267948966) q[1],q[2];\ncx q[2],q[3];\nt q[0];\nt q[3];\n";

/// `QASM` with the `t q[2]` of the first T layer replaced by `tdg q[2]`. The
/// §2 fixture-symmetry leg; see this file's module doc.
const QASM_TDG: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\n\
h q[0];\nh q[1];\nh q[2];\nh q[3];\nt q[0];\nt q[1];\ntdg q[2];\nt q[3];\n\
s q[1];\nrbs(1.5707963267948966) q[1],q[2];\ncx q[2],q[3];\nt q[0];\nt q[3];\n";

/// Number, hopping and interaction terms: three different Majorana lengths
/// and, under Jordan–Wigner, both `X`-string and `Y`-string Pauli terms. The
/// hopping pair is what makes the value phase-sensitive at all.
const FERMIONIC: &str = "1.25 [0^ 0] + -0.5 [2^ 2] + 0.75 [0^ 1] + 0.75 [1^ 0] \
                         + 0.4 [1^ 3] + 0.4 [3^ 1] + 0.6 [0^ 0 2^ 2]";

/// The dense value of `FERMIONIC` on `QASM`, and on `QASM_TDG`. Recorded as
/// constants so the gap below is a pin and not whatever the run produced.
const DENSE: f64 = 0.666_421_356_237_309_4;
const DENSE_TDG: f64 = 0.525;

/// Both engines are exact on this workload, so the only error is f64
/// summation order. `clifford_t_vs_statevector.rs` holds the engine to 1e-10
/// on random circuits; this is the same scale.
const TOL: f64 = 1e-10;

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_omega-run"))
}

fn write_fixture(tag: &str, body: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omega_stabrank_smoke_{}_{tag}.qasm",
        std::process::id()
    ));
    std::fs::write(&path, body).expect("write fixture");
    path
}

/// Run `omega-run` and hand back `(exit code, stdout, stderr)` **without
/// asserting anything**. The exit code is the evidence in this file, so it
/// has to reach the test rather than be consumed by a helper's `assert!`.
fn run(fixture: &Path, backend: &str, obs_flag: &str, obs: &str) -> (Option<i32>, String, String) {
    let out = Command::new(binary_path())
        .args([
            fixture.to_str().unwrap(),
            "--qasm-dialect",
            "lenient",
            "--backend",
            backend,
            obs_flag,
            obs,
            "--format",
            "json",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn omega-run");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Exit code first, parse second. Reversing these two reports a missing CLI
/// arm as a JSON parse error, which is the right failure named wrongly.
fn run_ok(fixture: &Path, backend: &str, obs_flag: &str, obs: &str) -> Value {
    let (code, stdout, stderr) = run(fixture, backend, obs_flag, obs);
    assert_eq!(
        code,
        Some(0),
        "omega-run --backend {backend} {obs_flag} exited {code:?}, not 0. An \
         unknown backend leaves the CLI's expectation dispatch at its `_` arm, \
         which prints `Expectation not supported for backend: {backend}` and \
         exits 1 — that is what the absence of this phase's wiring looks like, \
         and it is an exit code and not a tolerance. stderr:\n{stderr}"
    );
    serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!("--backend {backend}: stdout is not one JSON document ({e}): {stdout}")
    })
}

fn value(doc: &Value) -> f64 {
    doc["value"]
        .as_f64()
        .unwrap_or_else(|| panic!("no numeric `value` in {doc}"))
}

#[test]
fn a_fermionic_observable_string_reaches_stabrank_and_matches_the_statevector() {
    let path = write_fixture("fermionic", QASM);

    let stab = run_ok(&path, "stabrank", "--expectation-fermionic", FERMIONIC);
    let dense = run_ok(&path, "statevector", "--expectation-fermionic", FERMIONIC);
    let (vs, vd) = (value(&stab), value(&dense));

    assert!(
        (vs - vd).abs() < TOL,
        "stabrank {vs} vs statevector {vd} on the same fermionic observable, \
         |Δ| = {:.3e}. Both engines are exact on this workload — stabrank's \
         certificate says so below — so this is a defect and not a bound.",
        (vs - vd).abs()
    );
    assert!(
        (vd - DENSE).abs() < TOL,
        "the dense oracle returned {vd} where this fixture's recorded value is \
         {DENSE}. The fixture changed, so every pinned number in this file is \
         stale — including the T† gap."
    );

    // The premise of the comparison above, taken from the engine rather than
    // assumed: nothing was truncated, so the whole difference is f64
    // summation order.
    let cert = &stab["stabrank_truncation"];
    assert_eq!(
        cert["exact"],
        Value::Bool(true),
        "stabrank truncated an untruncated run; the comparison above is then \
         against a bound and not against an exact value: {cert}"
    );
    assert_eq!(
        cert["state_dropped_mass"].as_f64(),
        Some(0.0),
        "nothing was asked to be dropped and something was: {cert}"
    );
    // χ > 1 is the A10 count pin at CLI level: an arm that quietly dispatched
    // the statevector backend has no branches to report.
    assert_eq!(
        cert["final_chi"].as_u64(),
        Some(64),
        "six T gates is χ = 2^6 = 64 branches; the certificate reports {}. A \
         χ of 1 means this arm is not running a decomposition at all.",
        cert["final_chi"]
    );
    assert_eq!(
        cert["seed_basis"],
        Value::from("pauli"),
        "the Schrödinger seed is the state |0…0⟩ and the observable arrives \
         already mapped by Jordan–Wigner, so `pauli` is the only basis this \
         engine can report: {cert}"
    );

    let _ = std::fs::remove_file(&path);
}

/// The ladder spelling and the hand-written Pauli image of the same operator
/// must give the same number through the same backend — exactly, because the
/// JW lowering happens once, above the dispatch, and stabrank reads whatever
/// it is handed.
#[test]
fn the_ladder_spelling_and_its_pauli_image_agree() {
    let path = write_fixture("image", QASM);
    let ladder = run_ok(&path, "stabrank", "--expectation-fermionic", FERMIONIC);

    // Taken from the run rather than transcribed, so this cannot drift from
    // the mapping the CLI actually performed.
    let mapping = &ladder["observable_mapping"];
    assert_eq!(
        mapping["mapping"],
        Value::from("jordan_wigner"),
        "the fermionic door must record how the operator was lowered: {ladder}"
    );
    assert_eq!(
        mapping["input_basis"],
        Value::from("ladder"),
        "the input spelling must be recorded as what was typed: {ladder}"
    );
    let image = mapping["mapped_observable"]
        .as_str()
        .unwrap_or_else(|| panic!("no mapped_observable in {ladder}"))
        .to_string();

    let pauli = run_ok(&path, "stabrank", "--expectation", &image);
    let (vl, vp) = (value(&ladder), value(&pauli));
    assert!(
        (vl - vp).abs() < 1e-12,
        "the ladder spelling gives {vl} and its own reported Pauli image \
         `{image}` gives {vp} through the same backend, |Δ| = {:.3e}. These \
         are the same observable reaching the same engine by two routes; the \
         JW image is exact, so they must agree to rounding.",
        (vl - vp).abs()
    );
    assert!(
        vl.abs() > 1e-3,
        "the fixture's value is {vl}, near enough to zero that agreement here \
         proves little"
    );

    // `R` is computed in the basis actually read out (§1.2's pin): the same
    // number by either spelling, because it is the same Pauli sum.
    let (rl, rp) = (
        ladder["stabrank_truncation"]["observable_range"].as_f64(),
        pauli["stabrank_truncation"]["observable_range"].as_f64(),
    );
    assert_eq!(
        rl, rp,
        "observable_range differs between the ladder spelling ({rl:?}) and its \
         Pauli image ({rp:?}); R must be Σ|cᵢ| over the terms that were read \
         out, which are identical here"
    );

    let _ = std::fs::remove_file(&path);
}

/// Plan §2's fixture symmetry, as a measurement. `t q[2] → tdg q[2]` has to
/// move the number, through stabrank *and* through the dense oracle, by the
/// same amount — otherwise this fixture could not tell a `T` from its
/// adjoint and the agreement above would be evidence of nothing.
#[test]
fn a_t_dagger_in_the_fixture_is_a_different_number() {
    let t_path = write_fixture("sym_t", QASM);
    let dg_path = write_fixture("sym_tdg", QASM_TDG);

    let t_stab = value(&run_ok(
        &t_path,
        "stabrank",
        "--expectation-fermionic",
        FERMIONIC,
    ));
    let dg_stab = value(&run_ok(
        &dg_path,
        "stabrank",
        "--expectation-fermionic",
        FERMIONIC,
    ));
    let dg_dense = value(&run_ok(
        &dg_path,
        "statevector",
        "--expectation-fermionic",
        FERMIONIC,
    ));

    let gap = (t_stab - dg_stab).abs();
    assert!(
        gap > 0.1,
        "`t q[2]` and `tdg q[2]` give {t_stab} and {dg_stab} through stabrank, \
         a gap of {gap:.3e}. This fixture cannot distinguish a T from a T†, so \
         it is K8's third trap and the agreement tests in this file would stay \
         green under a `t → tdg` defect in the engine. The recorded gap is \
         |{DENSE} − {DENSE_TDG}| = {:.6}.",
        (DENSE - DENSE_TDG).abs()
    );
    assert!(
        (dg_stab - dg_dense).abs() < TOL,
        "on the T† fixture stabrank gives {dg_stab} and the statevector \
         {dg_dense}: the engine moved under the swap, but not to the right \
         number"
    );
    assert!(
        (dg_dense - DENSE_TDG).abs() < TOL,
        "the T† fixture's dense value is {dg_dense}, not the recorded \
         {DENSE_TDG}"
    );

    let _ = std::fs::remove_file(&t_path);
    let _ = std::fs::remove_file(&dg_path);
}

/// The other half of "the door exists": the doors that must stay shut.
/// stabrank is an expectation-value backend, so a sampling run is refused by
/// name with a non-zero exit — never answered with the noiseless
/// distribution, and never with a panic.
#[test]
fn sampling_on_stabrank_is_refused_by_name() {
    let path = write_fixture("refuse", QASM);
    let out = Command::new(binary_path())
        .args([
            path.to_str().unwrap(),
            "--qasm-dialect",
            "lenient",
            "--backend",
            "stabrank",
            "--shots",
            "16",
        ])
        .output()
        .expect("spawn omega-run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_ne!(
        out.status.code(),
        Some(101),
        "stabrank --shots PANICKED instead of refusing:\n{stderr}"
    );
    assert_eq!(
        out.status.code(),
        Some(1),
        "stabrank --shots must be refused with exit 1, not answered; \
         stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("expectation-value backend"),
        "the refusal must say why and name the door that works; \
         stderr:\n{stderr}"
    );
    let _ = std::fs::remove_file(&path);
}
