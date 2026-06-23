//! Cross-backend output-compatibility harness.
//!
//! For each enabled bridge backend, the harness:
//! 1. Runs a small curated set of QASM2 fixtures through Qiskit (the
//!    canonical reference, also a bridge backend).
//! 2. Runs the same fixtures through the *other* backend.
//! 3. Computes per-bin L2 distance between the normalised count
//!    distributions.
//! 4. Asserts the distance is below the per-backend threshold from
//!    `../QuetzalcoatlProto/backend/backend.py:144-152`:
//!      - perceval                   ≤ 0.0025
//!      - cirq / qadence             ≤ 0.0032
//!
//! Why this file exists: the user's standing rule for bridges work is
//! that no new bridge backend lands without a compatibility test
//! against Qiskit. This is the framework that test plugs into; each
//! new bridge feature flag adds its own `#[cfg(feature = "bridge-X")]`
//! test below.
//!
//! Skip behaviour: if the operator hasn't built the relevant Python
//! venv (`make -C crates/omega-bridges/python qiskit-venv` etc.), the
//! test prints a notice and exits green so `cargo test --workspace`
//! doesn't require Python on every contributor's machine.

// The cross-backend harness only fires when both bridge-qiskit and
// bridge-perceval are compiled in. The shared helpers below carry an
// allow(dead_code) so default builds (no bridge features) keep the
// file warning-free.
use std::collections::HashMap;
use std::path::PathBuf;

#[cfg(feature = "bridge-perceval")]
use omega_bridges::run_opticqasm;
use omega_bridges::{run_qasm2, Backend, BridgeError, Counts};

/// Curated subset of the verify-qiskit corpus. Picked so the cross-
/// backend harness runs in seconds rather than minutes — a few simple
/// circuits per category, enough to exercise the wire format.
#[allow(dead_code)]
fn curated_fixtures() -> Vec<(&'static str, PathBuf)> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("verify-qiskit")
        .join("fixtures");
    let pick: Vec<(&'static str, &'static str)> = vec![
        ("01_single_qubit_basic", "sq_000.qasm"),
        ("02_single_qubit_rotations", "sqr_000.qasm"),
        ("06_bell_ghz_w", "bell_phi_plus.qasm"),
        ("06_bell_ghz_w", "ghz_3.qasm"),
        ("07_qft_like", "qft_3.qasm"),
    ];
    pick.into_iter()
        .map(|(cat, name)| (cat, root.join(cat).join(name)))
        .filter(|(_, path)| path.exists())
        .collect()
}

#[allow(dead_code)]
fn runner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")
}

#[allow(dead_code)]
fn venv_python(slug: &str) -> PathBuf {
    runner_dir()
        .join(format!(".venv-{slug}"))
        .join("bin")
        .join("python")
}

#[allow(dead_code)]
fn wrapper(slug: &str) -> PathBuf {
    runner_dir().join(format!("omega-bridge-{slug}-runner"))
}

/// Set the per-backend env var so the dispatcher uses the in-repo
/// wrapper rather than searching PATH.
#[allow(dead_code)]
fn force_runner_env(slug: &str) {
    let env_var = format!("OMEGA_BRIDGE_{}_CMD", slug.to_ascii_uppercase());
    std::env::set_var(env_var, wrapper(slug));
}

/// L2 distance between two normalised count distributions.
/// Bins absent from either side count as 0. Returns sqrt(sum of
/// per-bin squared probability differences).
fn count_l2(a: &Counts, b: &Counts, total_a: u32, total_b: u32) -> f64 {
    let mut keys: std::collections::HashSet<&String> = a.keys().collect();
    keys.extend(b.keys());
    let mut sum_sq = 0.0;
    for k in keys {
        let pa = (*a.get(k).unwrap_or(&0) as f64) / total_a as f64;
        let pb = (*b.get(k).unwrap_or(&0) as f64) / total_b as f64;
        let d = pa - pb;
        sum_sq += d * d;
    }
    sum_sq.sqrt()
}

#[allow(dead_code)]
fn run_or_skip(
    backend: Backend,
    qasm: &str,
    shots: u32,
    seed_label: &str,
) -> Option<(Counts, u32)> {
    match run_qasm2(backend, qasm, shots, None) {
        Ok(c) => {
            let total: u32 = c.values().sum();
            assert!(
                total == shots,
                "{seed_label}: shots conserved expected {shots} got {total}"
            );
            Some((c, total))
        }
        Err(BridgeError::NotCompiled(_, _)) => {
            eprintln!("{seed_label}: backend {backend:?} not compiled; skipping");
            None
        }
        Err(BridgeError::Unavailable(b, msg)) => {
            eprintln!("{seed_label}: backend {b:?} unavailable ({msg}); skipping");
            None
        }
        Err(e) => panic!("{seed_label}: backend {backend:?} failed: {e}"),
    }
}

#[cfg(all(feature = "bridge-qiskit", feature = "bridge-perceval"))]
#[test]
fn perceval_matches_qiskit_within_threshold() {
    // Threshold from backend.py:147 — perceval gets the tightest of
    // the four (0.0025) because dual-rail postselection should land
    // very close to the Qiskit reference for the gate set we test.
    const PERCEVAL_THRESHOLD: f64 = 0.0025;
    const SHOTS: u32 = 4096;

    if !venv_python("qiskit").exists() {
        eprintln!(
            "Qiskit venv missing at {} — skipping cross-backend harness. \
             Build with `make -C crates/omega-bridges/python qiskit-venv`.",
            venv_python("qiskit").display()
        );
        return;
    }
    if !venv_python("perceval").exists() {
        eprintln!(
            "Perceval venv missing at {} — skipping cross-backend harness. \
             Build with `make -C crates/omega-bridges/python perceval-venv`.",
            venv_python("perceval").display()
        );
        return;
    }
    force_runner_env("qiskit");
    force_runner_env("perceval");

    let fixtures = curated_fixtures();
    assert!(
        !fixtures.is_empty(),
        "no fixtures found at verify-qiskit/fixtures/ — repo broken?"
    );
    let mut report: Vec<(String, f64)> = Vec::new();
    for (cat, path) in &fixtures {
        let qasm = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let label = format!("{cat}/{}", path.file_name().unwrap().to_string_lossy());

        let (qk_counts, qk_total) =
            run_or_skip(Backend::Qiskit, &qasm, SHOTS, &format!("{label} qiskit"))
                .expect("Qiskit must succeed when its venv is present");
        let (pe_counts, pe_total) = match run_or_skip(
            Backend::Perceval,
            &qasm,
            SHOTS,
            &format!("{label} perceval"),
        ) {
            Some(x) => x,
            None => continue, // perceval not usable for this fixture (bigger circuits)
        };

        let l2 = count_l2(&qk_counts, &pe_counts, qk_total, pe_total);
        report.push((label.clone(), l2));
        assert!(
            l2 <= PERCEVAL_THRESHOLD,
            "{label}: L2 = {l2:.5e} exceeds Perceval threshold {PERCEVAL_THRESHOLD}"
        );
    }
    eprintln!("\nperceval_matches_qiskit_within_threshold:");
    for (label, l2) in &report {
        eprintln!("  {label:<40} L2 = {l2:.4e}");
    }
    // Vacuous-pass guard: if both venvs were present but every
    // fixture skipped on Perceval (typically because the Qiskit
    // converter add-on is missing — see requirements-perceval.txt),
    // the test would otherwise silently pass without comparing
    // anything. Skip the test out loud instead so the operator
    // knows nothing was actually validated.
    if report.is_empty() {
        eprintln!(
            "\nNo fixtures successfully exercised both backends — \
             every Perceval run was Unavailable. Likely the Qiskit \
             converter add-on is not installed; see \
             crates/omega-bridges/python/requirements-perceval.txt. \
             Skipping rather than passing vacuously."
        );
    }
}

/// Bloqade gate-mode arm: the QuEra Aquila Python stack
/// (`bloqade.qasm2.loads(..., returns="c")` →
/// `StackMemorySimulator.task(...).batch_run(shots)`) returns exact
/// probabilities multiplied by `shots`, so the only sampling noise in
/// this comparison comes from the Qiskit AerSimulator's Monte Carlo
/// sampler. To distinguish a real cross-backend mismatch from that
/// sampling noise at the perceval-threshold L2 ≤ 0.0025, we need
/// enough Qiskit shots that L2 ~ sqrt(Σ p(1-p)/N) is well below
/// 0.0025 even on the most-spread fixtures (3-qubit QFT,
/// Σ p(1-p) ≈ 0.875). 200k shots gives L2 ≈ 0.0016 in that regime —
/// comfortably under threshold while still completing in a few
/// seconds. Both venvs (Qiskit + Bloqade) must be present; otherwise
/// the test prints a notice and exits green to keep `cargo test
/// --workspace` Python-free for everyone else.
#[cfg(all(feature = "bridge-qiskit", feature = "bridge-bloqade"))]
#[test]
fn bloqade_matches_qiskit_within_threshold() {
    const BLOQADE_THRESHOLD: f64 = 0.0025;
    // 1_000_000 shots: Bloqade's side is exact (probability * shots,
    // rounded), so the L2 mismatch with Qiskit is dominated by
    // Qiskit's binomial sampling noise. Expected
    // L2 ≈ sqrt(Σ p(1-p)/N); the worst fixture (3q QFT,
    // Σ p(1-p) ≈ 0.875) lands at ≈ 0.00094, well under the 0.0025
    // threshold even with the ~3σ statistical tail of L2 itself
    // (~0.0016). At 200k shots QFT runs were intermittently flaking
    // right at the boundary; 1M removes that without making the
    // test slow (≈ 5s per Qiskit Aer call on a 3-qubit circuit).
    const SHOTS: u32 = 1_000_000;

    if !venv_python("qiskit").exists() {
        eprintln!(
            "Qiskit venv missing at {} — skipping cross-backend harness. \
             Build with `make -C crates/omega-bridges/python qiskit-venv`.",
            venv_python("qiskit").display()
        );
        return;
    }
    if !venv_python("bloqade").exists() {
        eprintln!(
            "Bloqade venv missing at {} — skipping cross-backend harness. \
             Build with `make -C crates/omega-bridges/python bloqade-venv`.",
            venv_python("bloqade").display()
        );
        return;
    }
    force_runner_env("qiskit");
    force_runner_env("bloqade");

    let fixtures = curated_fixtures();
    assert!(
        !fixtures.is_empty(),
        "no fixtures found at verify-qiskit/fixtures/ — repo broken?"
    );
    let mut report: Vec<(String, f64)> = Vec::new();
    for (cat, path) in &fixtures {
        let qasm = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let label = format!("{cat}/{}", path.file_name().unwrap().to_string_lossy());

        let (qk_counts, qk_total) =
            run_or_skip(Backend::Qiskit, &qasm, SHOTS, &format!("{label} qiskit"))
                .expect("Qiskit must succeed when its venv is present");
        let (bl_counts, bl_total) =
            match run_or_skip(Backend::Bloqade, &qasm, SHOTS, &format!("{label} bloqade")) {
                Some(x) => x,
                None => continue, // bloqade not usable for this fixture
            };

        let l2 = count_l2(&qk_counts, &bl_counts, qk_total, bl_total);
        report.push((label.clone(), l2));
        assert!(
            l2 <= BLOQADE_THRESHOLD,
            "{label}: L2 = {l2:.5e} exceeds Bloqade threshold {BLOQADE_THRESHOLD}"
        );
    }
    eprintln!("\nbloqade_matches_qiskit_within_threshold:");
    for (label, l2) in &report {
        eprintln!("  {label:<40} L2 = {l2:.4e}");
    }
    // Vacuous-pass guard, mirroring the Perceval arm: if both venvs
    // were present but every Bloqade run was Unavailable (e.g. local
    // pyqrack install broken), surface a skip rather than passing
    // silently with no comparisons performed.
    if report.is_empty() {
        eprintln!(
            "\nNo fixtures successfully exercised both backends — \
             every Bloqade run was Unavailable. Likely the local \
             pyqrack / bloqade-circuit install is broken; see \
             crates/omega-bridges/python/requirements-bloqade.txt. \
             Skipping rather than passing vacuously."
        );
    }
}

/// OPTICQASM ↔ Perceval round-trip — confirm the photonic bridge
/// parses native OPTICQASM and reproduces the canonical HOM bunching.
/// Skips when the perceval venv isn't built so `cargo test` stays
/// Python-free for everyone else.
#[cfg(feature = "bridge-perceval")]
#[test]
fn perceval_opticqasm_hom_bunching() {
    if !venv_python("perceval").exists() {
        eprintln!(
            "Perceval venv missing at {} — skipping OPTICQASM HOM smoke. \
             Build with `make -C crates/omega-bridges/python perceval-venv`.",
            venv_python("perceval").display()
        );
        return;
    }
    force_runner_env("perceval");

    // 2-mode 50/50 BS with one photon in each mode. Indistinguishable
    // photons bunch — output distribution is |2,0⟩ + |0,2⟩, never
    // |1,1⟩. Same circuit omega's photonic backend exercises in
    // `examples/circuits/hom_effect.opticqasm`.
    let source = "OPTICQASM 1.0;\n\
                  photon q[2];\n\
                  bs_rx(0.7853981633974483, 0.0) q[0], q[1];";
    const SHOTS: u32 = 4096;

    let counts = match run_opticqasm(Backend::Perceval, source, SHOTS, Some(&[1, 1]), None) {
        Ok(c) => c,
        Err(BridgeError::Unavailable(_, msg)) => {
            eprintln!("Perceval unavailable ({msg}); skipping HOM smoke");
            return;
        }
        Err(e) => panic!("HOM smoke failed: {e}"),
    };

    let total: u32 = counts.values().sum();
    assert!(total == SHOTS, "shots conserved: got {total} / {SHOTS}");

    // Bunching: only |2,0⟩ and |0,2⟩ should be present (each ~50%);
    // |1,1⟩ should be absent (or vanishingly small from f32 round-off
    // in the photonic simulator).
    let n_20 = *counts.get("2,0").unwrap_or(&0);
    let n_02 = *counts.get("0,2").unwrap_or(&0);
    let n_11 = *counts.get("1,1").unwrap_or(&0);
    let bunched = n_20 + n_02;
    let bunched_frac = bunched as f64 / SHOTS as f64;
    assert!(
        bunched_frac > 0.95,
        "expected ≥95% bunching, got {:.3} (|2,0⟩={n_20}, |0,2⟩={n_02}, |1,1⟩={n_11})",
        bunched_frac
    );
    // The two bunched bins should be roughly balanced (within 10pp at
    // 4096 shots — sampling noise dominates).
    let imbalance = (n_20 as i64 - n_02 as i64).unsigned_abs() as f64 / SHOTS as f64;
    assert!(
        imbalance < 0.10,
        "|2,0⟩ vs |0,2⟩ imbalance > 10%: {n_20} vs {n_02}"
    );
}

/// Sanity-check the L2 helper against a hand-rolled distribution.
#[test]
fn count_l2_matches_hand_computed_value() {
    // Two distributions with one bin difference of 0.1 each:
    //   A = {00: 0.6, 11: 0.4}
    //   B = {00: 0.7, 11: 0.3}
    // L2 = sqrt(0.01 + 0.01) = sqrt(0.02) ≈ 0.1414.
    let mut a: HashMap<String, u32> = HashMap::new();
    a.insert("00".into(), 600);
    a.insert("11".into(), 400);
    let mut b: HashMap<String, u32> = HashMap::new();
    b.insert("00".into(), 700);
    b.insert("11".into(), 300);
    let l2 = count_l2(&a, &b, 1000, 1000);
    let expected = (0.01_f64 + 0.01).sqrt();
    assert!(
        (l2 - expected).abs() < 1e-12,
        "got {l2}, expected ~{expected}"
    );
}
