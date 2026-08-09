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
use omega_bridges::corpus::{crosscheck_corpus, gates_used};
use omega_bridges::{run_qasm2, Backend, BridgeError, Counts};

/// Small subset of the cross-check corpus, so the Perceval and Bloqade arms
/// run in seconds rather than minutes.
///
/// **This was hard-wired to the private `verify-qiskit/` tree and hard-failed
/// without it.** `curated_fixtures()` filtered its five picks by
/// `path.exists()`, so on any checkout lacking that tree it returned an empty
/// vector and the arms tripped `assert!(!fixtures.is_empty(), "repo broken?")`.
/// The repo was not broken; the corpus simply was not there.
///
/// It survived because `ci.sh`'s bridge stage compiles
/// `bridge-qiskit,bridge-tsim,bridge-ppvm` — **not** `bridge-perceval` — so
/// the two arms that call this are `#[cfg]`'d out of every CI run. A test that
/// CI cannot compile is not a gate, which is the same lesson as `FIXES_PLAN.md`
/// K9 one layer down.
///
/// Now it takes the first `n` entries of whichever corpus
/// [`crosscheck_corpus`] resolves — the vendored one when the private tree is
/// absent — so the arms exercise real fixtures everywhere. Deliberately NOT
/// filtered by gate set: Perceval and Bloqade have no `{"mode":"gates"}`
/// introspection, so an out-of-subset fixture comes back `Unavailable` and is
/// skipped with a reason by `run_or_skip`, which is the existing behaviour.
#[allow(dead_code)]
fn curated_fixtures() -> Vec<(&'static str, PathBuf)> {
    const MAX: usize = 5;
    let corpus = crosscheck_corpus();
    corpus
        .files
        .into_iter()
        .take(MAX)
        .map(|p| (corpus.label, p))
        .collect()
}

/// Ask a runner which QASM2 gate names it can lower (`{"mode":"gates"}`).
///
/// The gate set lives in `python/qasm2_stim.py`; querying it here
/// instead of duplicating the list in Rust means the fixture filter
/// can never claim a coverage number the converter disagrees with.
#[allow(dead_code)]
fn runner_gate_set(slug: &str) -> std::collections::BTreeSet<String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut child = Command::new(wrapper(slug))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn {slug} runner for gate-set query: {e}"));
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(br#"{"mode":"gates"}"#)
        .expect("write gate-set query");
    let out = child.wait_with_output().expect("wait for gate-set query");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let resp: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!(
            "{slug} runner returned non-JSON for the gate-set query: {e} \
             (stdout: {stdout}, stderr: {})",
            String::from_utf8_lossy(&out.stderr)
        )
    });
    assert!(
        resp.get("ok").and_then(|v| v.as_bool()).unwrap_or(false),
        "{slug} runner refused the gate-set query: {resp}"
    );
    resp.get("gates")
        .and_then(|v| v.as_array())
        .expect("gate-set response missing `gates`")
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect()
}

/// Corpus entries whose every gate is inside `gate_set`. Returns
/// `(corpus_label, selected, total)` so the test can report coverage.
#[allow(dead_code)]
fn fixtures_within_gate_set(
    gate_set: &std::collections::BTreeSet<String>,
) -> (&'static str, Vec<PathBuf>, usize) {
    let corpus = crosscheck_corpus();
    let label = corpus.label;
    let total = corpus.files.len();
    let selected = corpus
        .files
        .into_iter()
        .filter(|path| {
            let Ok(qasm) = std::fs::read_to_string(path) else {
                return false;
            };
            gates_used(&qasm).iter().all(|g| gate_set.contains(g))
        })
        .collect();
    (label, selected, total)
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

/// The L2 gate for this circuit: `K` × the RMS L2 expected when both sides
/// sample the same distribution, floored at a few shots' worth of granularity.
///
/// **This replaced a hardcoded `0.0025` that was wrong by 7.6x on the very
/// first fixture it was ever applied to.** That constant came from the
/// reference Python harness and implicitly assumed a shot count and an outcome
/// count; it is roughly 4σ at the tsim/ppvm arms' 4M shots and **7.6× TIGHTER
/// THAN THE NOISE FLOOR** at the Perceval arm's 4096. Measured on
/// `01_single_qubit_basic.qasm` (4 outcomes, 4096 shots): null scale
/// 1.9133e-2, observed L2 2.1382e-2 — 1.1σ, i.e. two correct samplers — and
/// the constant called it a failure.
///
/// For two independent samplers of the same `p`,
/// `E[L2²] = Σₖ pₖ(1−pₖ)·(1/n_a + 1/n_b)`, and the gate is `K` times that RMS.
/// `K = 6` is exactly 6σ in the narrowest two-outcome case and strictly more
/// conservative as the outcome count grows. Same derivation as
/// `omega-cli/tests/nway_counts.rs`; see `FIXES_PLAN.md` K2.
///
/// When one side returns EXACT probabilities rather than samples (Bloqade
/// multiplies probability by shots), its variance term is really zero, so the
/// gate is conservative by up to √2. Erring loose there is the right direction:
/// it cannot manufacture a failure.
#[allow(dead_code)]
fn l2_gate(anchor: &Counts, n_a: u32, n_b: u32) -> f64 {
    let inv = 1.0 / n_a as f64 + 1.0 / n_b as f64;
    let var: f64 = anchor
        .values()
        .map(|&c| {
            let p = c as f64 / n_a as f64;
            p * (1.0 - p)
        })
        .sum();
    let rms = (var * inv).sqrt();
    let floor = 8.0 / n_a.min(n_b) as f64;
    (6.0 * rms).max(floor)
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
        // A typed refusal is a CORRECT answer: the backend understood the
        // circuit and cannot express it. Skipping with a reason is right;
        // panicking would count an honest refusal as a defect and redden the
        // arm. `run_or_skip` predated `BridgeError::CannotExpress` (added in
        // the step-2 taxonomy work) and fell through to the panic arm — so
        // the first refusal this harness ever received crashed it.
        Err(BridgeError::CannotExpress(b, msg)) => {
            eprintln!("{seed_label}: backend {b:?} cannot express this circuit ({msg}); skipping");
            None
        }
        Err(e) => panic!("{seed_label}: backend {backend:?} failed: {e}"),
    }
}

#[cfg(all(feature = "bridge-qiskit", feature = "bridge-perceval"))]
#[test]
fn perceval_matches_qiskit_within_threshold() {
    // Gate derived per circuit — see `l2_gate`. The former hardcoded 0.0025
    // was 7.6x tighter than this arm's own noise floor and failed a correct
    // backend on the first fixture.
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
        let gate = l2_gate(&qk_counts, qk_total, pe_total);
        report.push((label.clone(), l2));
        assert!(
            l2 <= gate,
            "{label}: L2 = {l2:.5e} exceeds derived gate {gate:.5e}"
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
        let gate = l2_gate(&qk_counts, qk_total, bl_total);
        report.push((label.clone(), l2));
        assert!(
            l2 <= gate,
            "{label}: L2 = {l2:.5e} exceeds derived gate {gate:.5e}"
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

/// Shared body for the two QuEra stim-dialect arms (tsim, ppvm).
///
/// Both bridges lower QASM2 through the same `python/qasm2_stim.py`
/// converter and both *sample*, so they share a threshold, a shot
/// count, and a skip policy. The only per-backend input is the slug
/// and the gate set the runner reports, which drives the fixture
/// filter.
///
/// ## Why 4M shots and not Bloqade's 1M
///
/// Bloqade's arm gets away with 1M because its side is exact
/// (probability × shots): the only binomial noise is Qiskit's, so
/// `Var(Δp) = p(1-p)/N`. tsim and ppvm are Monte-Carlo samplers, so
/// *both* sides carry that variance and `Var(Δp) = 2p(1-p)/N`.
///
/// The tightest case is a two-outcome fixture (Bell, GHZ) where
/// `L2 = √2·|Δp|` with `p = 1/2`. At N shots, `sd(Δp) = √(0.5/N)`, so
/// the threshold sits at `0.0025 / (√2 · √(0.5/N))` standard
/// deviations. N = 1M puts that at only 2.5σ — a ~1.2% false-failure
/// rate per fixture, which over a multi-fixture corpus is a flaky
/// test, not a gate. N = 4M puts it at 5σ (≈ 6e-7 per fixture), which
/// is the same safety margin Bloqade's 1M buys. Measured cost at 4M
/// shots on a 3-qubit fixture: qiskit 1.5 s, ppvm 2.7 s, tsim 6.9 s.
///
/// Note the dispatcher does not forward a seed (`RunnerRequest` has no
/// seed field), so these runs are genuinely random every invocation —
/// the margin above is what keeps that honest rather than lucky.
#[cfg(feature = "bridge-qiskit")]
#[allow(dead_code)]
fn quera_stim_bridge_matches_qiskit(backend: Backend, slug: &str) {
    const SHOTS: u32 = 4_000_000;

    if !venv_python("qiskit").exists() {
        eprintln!(
            "Qiskit venv missing at {} — skipping cross-backend harness. \
             Build with `make -C crates/omega-bridges/python qiskit-venv`.",
            venv_python("qiskit").display()
        );
        return;
    }
    if !venv_python(slug).exists() {
        eprintln!(
            "{slug} venv missing at {} — skipping cross-backend harness. \
             Build with `make -C crates/omega-bridges/python {slug}-venv`.",
            venv_python(slug).display()
        );
        return;
    }
    force_runner_env("qiskit");
    force_runner_env(slug);

    let gate_set = runner_gate_set(slug);
    let (corpus, fixtures, total) = fixtures_within_gate_set(&gate_set);
    eprintln!(
        "\n{slug}_matches_qiskit_within_threshold: {} of {total} fixtures in \
         {corpus} are inside the {slug} gate set ({})",
        fixtures.len(),
        gate_set.iter().cloned().collect::<Vec<_>>().join(" ")
    );
    assert!(
        !fixtures.is_empty(),
        "no {slug}-compatible fixtures found in {corpus} — corpus missing or filter broken"
    );

    let mut report: Vec<(String, f64)> = Vec::new();
    for path in &fixtures {
        let qasm = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let label = path.file_name().unwrap().to_string_lossy().to_string();

        let (qk_counts, qk_total) =
            run_or_skip(Backend::Qiskit, &qasm, SHOTS, &format!("{label} qiskit"))
                .expect("Qiskit must succeed when its venv is present");
        let (other_counts, other_total) =
            match run_or_skip(backend, &qasm, SHOTS, &format!("{label} {slug}")) {
                Some(x) => x,
                None => continue,
            };

        let l2 = count_l2(&qk_counts, &other_counts, qk_total, other_total);
        let gate = l2_gate(&qk_counts, qk_total, other_total);
        report.push((label.clone(), l2));
        assert!(
            l2 <= gate,
            "{label}: L2 = {l2:.5e} exceeds derived gate {gate:.5e}"
        );
    }
    for (label, l2) in &report {
        eprintln!("  {label:<40} L2 = {l2:.4e}");
    }
    // Vacuous-pass guard, mirroring the Perceval / Bloqade arms: both
    // venvs present but every run Unavailable would otherwise pass
    // silently with nothing compared.
    if report.is_empty() {
        eprintln!(
            "\nNo fixtures successfully exercised both backends — every \
             {slug} run was Unavailable. Check the venv build; see \
             crates/omega-bridges/python/requirements-{slug}.txt. \
             Skipping rather than passing vacuously."
        );
    }
}

/// tsim arm: QuEra's ZX-calculus stabilizer-rank sampler
/// (`bloqade-tsim`), fed the extended-Stim text that
/// `python/qasm2_stim.py` lowers QASM2 into. Restricted to the
/// fixtures whose gates are inside the converter's tsim set — which
/// includes SWAP and CCX/CCZ, since tsim's `shorthand_to_stim`
/// expands those into Clifford+T.
#[cfg(all(feature = "bridge-qiskit", feature = "bridge-tsim"))]
#[test]
fn tsim_matches_qiskit_within_threshold() {
    quera_stim_bridge_matches_qiskit(Backend::Tsim, "tsim");
}

/// ppvm arm: QuEra's generalized stabilizer tableau, sampled through
/// `ppvm.sample_stim`. Same lowering as the tsim arm, smaller gate
/// set — ppvm's executor rejects SWAP and has no CCX/CCZ sugar, so
/// fixtures using those are filtered out here and only exercised on
/// the tsim side.
#[cfg(all(feature = "bridge-qiskit", feature = "bridge-ppvm"))]
#[test]
fn ppvm_matches_qiskit_within_threshold() {
    quera_stim_bridge_matches_qiskit(Backend::Ppvm, "ppvm");
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
    // `examples/circuits/hom_dip.opticqasm`.
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

/// Report WHICH corpus ran, and assert the vendored one still covers the
/// defect classes it was extended to reach.
///
/// Two distinct problems this guards.
///
/// **1. The corpus identity is not stable.** `crosscheck_corpus()` *prefers* a
/// private `../verify-qiskit/fixtures` tree when it is checked out, falling
/// back to the 11+ vendored fixtures otherwise. So two operators can run "the
/// cross-backend tests", see the same green, and have exercised entirely
/// different circuits — one of them unaudited by anything in this repository.
/// The label was already computed and returned; nothing printed it.
///
/// **2. The vendored corpus had holes exactly the shape of shipped defects.**
/// Measured before this was extended: `measure` in 10/11 fixtures, but
/// **`if(` in 0/11 and `reset` in 0/11**. So it could not reach the defect
/// class fixed in `11888a9` (expectation silently skipped feedforward) or
/// `ae6da5c` (mid-circuit measurement sampled once per run, not per shot).
/// A matrix built on that corpus would have been green by construction.
#[test]
fn corpus_identity_is_reported_and_covers_the_defect_classes() {
    let corpus = crosscheck_corpus();
    let (label, files) = (corpus.label, &corpus.files);

    let strip = |s: &str| -> String {
        s.lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let mut with_if = 0usize;
    let mut with_reset = 0usize;
    let mut with_measure = 0usize;
    for path in files {
        let Ok(raw) = std::fs::read_to_string(path) else {
            continue;
        };
        let src = strip(&raw);
        if src.contains("if") && src.contains("==") {
            with_if += 1;
        }
        if src.contains("reset") {
            with_reset += 1;
        }
        if src.contains("measure") {
            with_measure += 1;
        }
    }

    eprintln!("cross-backend corpus: {label} ({} files)", files.len());
    eprintln!("  measure: {with_measure}   conditionals: {with_if}   reset: {with_reset}");

    assert!(!files.is_empty(), "corpus is empty — nothing would be compared");

    if label != "tests/fixtures/crosscheck" {
        // Not a failure: the private corpus is larger and legitimate. But it is
        // NOT audited here, so a green run against it means something different
        // and must say so rather than read as identical coverage.
        eprintln!(
            "  NOTE: running the PRIVATE corpus, which this repository does not \
             audit. The coverage assertions below are skipped."
        );
        return;
    }

    // Vendored corpus only: keep the holes closed.
    assert!(
        with_if >= 2,
        "the vendored corpus needs at least 2 conditional fixtures (found {with_if}). \
         One must have a SOMETIMES-FALSE condition and one an always-true control: \
         an always-true condition alone cannot distinguish 'honours the guard' from \
         'ignores the guard', because both produce the identical distribution \
         (measured: both {{\"1\": 20000}} on Aer)."
    );
    assert!(
        with_reset >= 1,
        "the vendored corpus needs at least one reset fixture (found {with_reset}); \
         without it the corpus cannot reach the ae6da5c defect class"
    );
}
