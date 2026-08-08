// SPDX-License-Identifier: Apache-2.0
//! N-way counts matrix: one QASM2 corpus, every engine that can sample it, one
//! report (`FIXES_PLAN.md` Part K, step 3).
//!
//! The value is not another green tick — it is that **a defect surviving one
//! engine rarely survives seven**. The risk is the exact opposite: a matrix is
//! the easiest possible place to hide a check that never ran. Everything
//! awkward in this file exists to make that hiding impossible.
//!
//! # Why this lives in `omega-cli` and not in `cross_backend.rs`
//!
//! Part K5 says to build on `crates/omega-bridges/tests/cross_backend.rs`, and
//! for the bridge-to-bridge arms that is still where the work belongs. But the
//! counts lane needs the **in-tree** engines, and `omega-bridges` depends on
//! `omega-core` alone — deliberately, so that a crate whose job is shelling out
//! to external Python does not drag in every simulator in the workspace. Adding
//! `omega-parser` + statevector + mps + pauli as dev-dependencies there would
//! invert that layering for the sake of file placement.
//!
//! `omega-cli` already depends on all of them *and* on `omega-bridges`, so the
//! matrix lives here and the two lanes share what actually matters: the corpus
//! locator (`omega_bridges::corpus`), so they can never disagree about which
//! fixtures ran.
//!
//! # The counts-keying convention, which is the whole difficulty
//!
//! In-tree engines key counts by `u64` over the **full qubit register**;
//! bridges return **creg-width, LSB-first strings**. These are not two spellings
//! of one thing — they disagree about *which bits exist*.
//!
//! `08_partial_measure.qasm` is the case that makes it concrete: 3 qubits, a
//! 2-bit creg, and an unmeasured `h q[2]`. Qiskit reports two outcomes
//! (`{"00": 10047, "11": 9953}`, measured). A raw in-tree key carries q[2]'s
//! coin flip as a third bit, so the same physics arrives as four outcomes and
//! the L2 against Qiskit is ≈ 1.0. Compare without converting and every engine
//! "fails" on a fixture where every engine is right.
//!
//! There are **two** conventions to get right, not one, and the second is the
//! one that bit during development: the string is **LSB-first** (leftmost
//! character = bit 0), because that is what `omega_bridges::Counts` documents
//! and what `qiskit_runner.py` emits by reversing Qiskit's own display order.
//! See [`to_str_counts`].
//!
//! The conversion is [`omega_core::executor::project_counts_onto_creg`], driven
//! by [`omega_core::executor::needs_collapse`] — both extracted from shipped
//! front ends rather than written here, so this matrix validates the path users
//! actually get. See their unit tests in `omega-core/src/executor.rs`.
//!
//! # Thresholds are derived per circuit, not hardcoded
//!
//! `cross_backend.rs` carries a hardcoded L2 ≤ 0.0025 and pays for it with 4M
//! shots. That is fine for a bridge (a subprocess sampling in C++) and
//! impossible for an in-tree engine in `Collapse` mode, which runs **one
//! independent trajectory per shot**.
//!
//! So the threshold here is computed from the anchor's own empirical
//! distribution. For two independent samplers of the same `p`,
//!
//! ```text
//!   E[L2²] = Σₖ pₖ(1−pₖ)·(1/n_a + 1/n_b)
//! ```
//!
//! and the gate is `K` times that RMS. Note what `K` means at each end: for a
//! two-outcome circuit `L2 = √2·|Δp|` and the gate is **exactly `K` standard
//! deviations**; as `d` grows, `L2²` becomes a `d`-term chi-square whose tail
//! at `K²` times its mean is far thinner still. So `K = 6` is 6σ in the
//! narrowest case and strictly more conservative everywhere else.
//!
//! Measured separation on the three real errors this file caught during
//! development, against the gate for the same circuit:
//!
//! | error | L2 | gate | ratio |
//! |---|---|---|---|
//! | MSB/LSB bit order | 0.24 – 0.59 | ~0.05 | 5 – 12x |
//! | creg projection disabled | ~1.0 | ~0.05 | ~20x |
//! | conditional dropped | 1.41 | 4.0e-4 | ~3500x |
//!
//! The tightest of those still clears the gate 5x over, which is the margin
//! that matters — not a round claim about orders of magnitude. This is what
//! K2's correction asked for: a threshold that knows the circuit, rather than
//! a constant that is simultaneously too tight on a Bell state and vacuous on
//! a wide one.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use omega_core::circuit::CircuitIR;
use omega_core::executor::{
    measure_pairs, needs_collapse, project_counts_onto_creg, Backend as CoreBackend, ExecConfig,
    ExecResult, MidCircuitMode,
};

/// Counts keyed the way a bridge reports them: creg-width, LSB-first strings.
type StrCounts = HashMap<String, u32>;

/// Shots for every engine in the lane. Chosen so an in-tree `Collapse`-mode
/// run — one trajectory per shot — finishes in seconds, since the threshold
/// adapts to the shot count rather than demanding a fixed one.
const SHOTS: u32 = 20_000;

/// Gate multiplier on the null RMS. See the module docs.
const K_SIGMA: f64 = 6.0;

/// Why a cell in the matrix is not a comparison. These must not look alike:
/// collapsing them into "skipped" is how a matrix reports coverage it does not
/// have (Part K1).
#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
enum Status {
    /// Compared, within the derived gate. Carries `(l2, threshold)`.
    Agree(f64, f64),
    /// Compared, outside the gate. A defect.
    Disagree(f64, f64),
    /// The engine understood the circuit and **correctly refused** it —
    /// `pauli` on a non-Clifford rotation, a bridge runner reporting
    /// `*-unsupported-gate`. A legitimately empty cell.
    CannotExpress(String),
    /// Environmental: a venv is absent, a feature is not compiled. Says
    /// nothing at all about the engine.
    NotInstalled(String),
    /// A real gap: crashed, or failed for a reason that is neither of the
    /// above.
    Error(String),
}

impl Status {
    #[allow(dead_code)]
    fn tag(&self) -> &'static str {
        match self {
            Status::Agree(..) => "agree",
            Status::Disagree(..) => "disagree",
            Status::CannotExpress(_) => "cannot-express",
            Status::NotInstalled(_) => "not-installed",
            Status::Error(_) => "error",
        }
    }
}

// --------------------------------------------------------------------------
// Key conversion
// --------------------------------------------------------------------------

/// Width of the outcome keys a run of `ir` produces, in bits.
///
/// A circuit that declares `measure q -> c` reports over the creg; one that
/// declares none reports over the full qubit register, matching the Qiskit
/// runner's synthesised `measure_all()`.
fn key_width(ir: &CircuitIR) -> usize {
    if measure_pairs(ir).is_empty() {
        (ir.num_qubits as usize).max(1)
    } else {
        (ir.num_classical_bits as usize).max(1)
    }
}

/// Render `u64` counts as the bridge-convention strings so the two are
/// comparable at all.
///
/// **The bridge convention is LSB-first**: `omega_bridges::Counts` is
/// documented as "outcome bit-string (LSB-first)", and `qiskit_runner.py:162`
/// implements it as `bits = flat[::-1]` — Qiskit's own `c[w-1]…c[0]` string,
/// reversed. So the leftmost character is bit 0, not bit `w-1`.
///
/// The obvious `format!("{k:0width$b}")` is therefore **wrong**, and wrong in a
/// way that hides: it is correct on every palindromic outcome, which includes
/// `{00, 11}` — the entire output of a Bell fixture. Writing it that way here
/// put all three dense in-tree engines at L2 ≈ 0.24–0.59 against Qiskit on the
/// four asymmetric fixtures while they agreed perfectly with each other, which
/// reads exactly like a backend defect and was not one. Hence
/// [`bit_order_is_lsb_first_on_an_asymmetric_outcome`], which pins a
/// deliberately non-palindromic key.
/// Rendering **refuses** a key that does not fit `width` rather than dropping
/// its high bits. That refusal caught a second defect: `PauliBackend` was
/// re-measuring every qubit in collapse mode and keying over the full register,
/// so a 2-qubit circuit with a 1-bit creg emitted key `3`. Truncating to width
/// turned `3` into `"1"` — the right answer, from a backend reporting over the
/// wrong register — and the matrix's first run scored `pauli` as fully in
/// agreement. A conversion that silently discards information will eventually
/// convert a wrong answer into a right-looking one.
fn to_str_counts(counts: &HashMap<u64, u32>, width: usize) -> Result<StrCounts, String> {
    let mut out = StrCounts::new();
    for (k, v) in counts {
        if width < 64 && *k >= (1u64 << width) {
            return Err(format!(
                "counts key {k} does not fit the declared {width}-bit register — \
                 the engine is reporting over a different register than the \
                 program declares. Truncating here would hide that."
            ));
        }
        let s: String = (0..width)
            .map(|b| if (k >> b) & 1 == 1 { '1' } else { '0' })
            .collect();
        *out.entry(s).or_insert(0) += *v;
    }
    Ok(out)
}

/// Run one in-tree engine and return counts in bridge convention.
///
/// The mode decision and the projection both come from `omega-core`, so this
/// exercises the shipped path rather than a convention invented for the test.
///
/// Returns a [`Status`] on failure rather than an error type, because the
/// classification is the point. `OmegaError::Unsupported` from a backend is a
/// **correct refusal** (`pauli` on a T gate) and belongs in `CannotExpress`; a
/// key that does not fit its register is a **defect** and belongs in `Error`.
/// Routing both through one error type would file the second as the first,
/// which is the silent direction — see `docs/BRIDGES.md`.
fn run_in_tree(
    backend: &dyn CoreBackend,
    ir: &CircuitIR,
    seed: u64,
) -> Result<StrCounts, Status> {
    let collapse = needs_collapse(ir);
    let config = ExecConfig {
        shots: Some(SHOTS),
        seed: Some(seed),
        mid_circuit_mode: if collapse {
            MidCircuitMode::Collapse
        } else {
            MidCircuitMode::Skip
        },
    };
    let params = omega_core::params::ParameterBinding::new();
    let res = backend.execute(ir, &params, &config).map_err(|e| match e {
        omega_core::error::OmegaError::Unsupported(msg) => Status::CannotExpress(msg),
        other => Status::Error(other.to_string()),
    })?;

    // In `Collapse` mode the backend already keyed by the classical register
    // (it walked every `measure` and packed the creg), so projecting again
    // would be wrong — the key is no longer a qubit-basis index. In `Skip`
    // mode the key IS a qubit-basis index and must be projected.
    let res = if collapse {
        res
    } else {
        project_counts_onto_creg(res, &measure_pairs(ir)).map_err(Status::Error)?
    };
    let ExecResult::Counts(counts) = res else {
        return Err(Status::Error(
            "engine returned a non-counts result for a shots-mode run".into(),
        ));
    };
    to_str_counts(&counts, key_width(ir)).map_err(Status::Error)
}

// --------------------------------------------------------------------------
// Metric
// --------------------------------------------------------------------------

/// L2 distance between two normalised count distributions. Bins absent from
/// either side count as 0.
#[allow(dead_code)]
fn count_l2(a: &StrCounts, b: &StrCounts, total_a: u32, total_b: u32) -> f64 {
    let mut keys: std::collections::HashSet<&String> = a.keys().collect();
    keys.extend(b.keys());
    let mut sum_sq = 0.0;
    for k in keys {
        let pa = (*a.get(k).unwrap_or(&0) as f64) / total_a as f64;
        let pb = (*b.get(k).unwrap_or(&0) as f64) / total_b as f64;
        sum_sq += (pa - pb) * (pa - pb);
    }
    sum_sq.sqrt()
}

/// The L2 gate for this circuit: `K` × the RMS L2 expected when both sides
/// sample the same distribution, floored at a few shots' worth of granularity.
///
/// The floor matters for deterministic circuits, where every `pₖ(1−pₖ)` is 0
/// and the RMS gate would be exactly 0 — correct in principle, but it would
/// then fail on an engine that returns exact `probability × shots` and rounds
/// one count differently from a sampler.
fn l2_gate(anchor: &StrCounts, n_a: u32, n_b: u32) -> f64 {
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
    (K_SIGMA * rms).max(floor)
}

// --------------------------------------------------------------------------
// Bridge plumbing
// --------------------------------------------------------------------------

#[allow(dead_code)]
fn runner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("omega-bridges")
        .join("python")
}

#[allow(dead_code)]
fn venv_python(slug: &str) -> PathBuf {
    runner_dir()
        .join(format!(".venv-{slug}"))
        .join("bin")
        .join("python")
}

#[allow(dead_code)]
fn force_runner_env(slug: &str) {
    std::env::set_var(
        format!("OMEGA_BRIDGE_{}_CMD", slug.to_ascii_uppercase()),
        runner_dir().join(format!("omega-bridge-{slug}-runner")),
    );
}

/// Map a bridge outcome onto the five-way status taxonomy.
///
/// `CannotExpress` is a *typed* `BridgeError` variant as of the step-2 work
/// (`docs/BRIDGES.md`, "Error `kind` naming"). Before that it was a string
/// prefix inside `BridgeError::Backend`, and every refusal would have landed
/// in `Error` — reddening the matrix for backends that behaved correctly.
#[allow(dead_code)]
fn bridge_status(res: Result<omega_bridges::Counts, omega_bridges::BridgeError>) -> Result<StrCounts, Status> {
    use omega_bridges::BridgeError;
    match res {
        Ok(c) => Ok(c),
        Err(BridgeError::NotCompiled(b, _)) => {
            Err(Status::NotInstalled(format!("{b:?} feature not compiled")))
        }
        Err(BridgeError::Unavailable(_, msg)) => Err(Status::NotInstalled(msg)),
        Err(BridgeError::CannotExpress(_, msg)) => Err(Status::CannotExpress(msg)),
        Err(e) => Err(Status::Error(e.to_string())),
    }
}

// --------------------------------------------------------------------------
// The matrix
// --------------------------------------------------------------------------

/// One row of the report.
#[allow(dead_code)]
struct Row {
    fixture: String,
    engine: &'static str,
    status: Status,
}

/// Fixtures the in-tree QASM2 lowering cannot yet accept, with the reason.
///
/// This is a **tracked gap list, not an excuse list**, and the difference is
/// enforced in both directions:
///
/// * a lowering failure NOT listed here fails the test, so a new gap is loud;
/// * a listed fixture that starts lowering ALSO fails the test, so the list
///   cannot go stale and quietly keep suppressing a fixture that now works.
///
/// Filing these as `cannot-express` instead would be the silent direction
/// `docs/BRIDGES.md` warns about: a real gap disappearing into the matrix as a
/// legitimately empty cell. They are real gaps, they are in the report, and
/// they are counted separately from the cells that were compared.
#[allow(dead_code)]
const NOT_LOWERABLE: &[(&str, &str)] = &[(
    "03_sqrt_x.qasm",
    "`sx` / `sxdg` have no GateKind. They are exactly \
     `e^{±iπ/4}·u3(π/2, ∓π/2, ±π/2)` (verified against Qiskit), so lowering to \
     U3 is correct for counts and WRONG by a global phase for any statevector \
     comparison. Choosing between an aliased lowering and a new GateKind \
     threaded through six backends (CPU/MPS/Pauli/CUDA/Metal/OpenCL) is a \
     design decision, so it is planned rather than slipped in here.",
)];

#[cfg(feature = "bridge-qiskit")]
#[test]
fn nway_counts_matrix_agrees_with_qiskit() {
    use omega_backend_mps::{MpsBackend, NoisyMpsBackend};
    use omega_backend_pauli::PauliBackend;
    use omega_backend_statevector::StatevectorBackend;
    use omega_bridges::corpus::crosscheck_corpus;

    if !venv_python("qiskit").exists() {
        eprintln!(
            "Qiskit venv missing at {} — skipping the N-way counts matrix. The \
             anchor is not optional: without an independent implementation this \
             lane degrades to our engines agreeing with each other, which is the \
             weak-evidence configuration that let the Reset defect through \
             (FIXES_PLAN.md K3). Build with \
             `make -C crates/omega-bridges/python qiskit-venv`.",
            venv_python("qiskit").display()
        );
        return;
    }
    force_runner_env("qiskit");
    for slug in ["perceval", "bloqade", "tsim", "ppvm"] {
        force_runner_env(slug);
    }

    let corpus = crosscheck_corpus();
    assert!(!corpus.files.is_empty(), "corpus is empty — nothing to compare");
    eprintln!(
        "\nN-way counts matrix — corpus {} ({} fixtures), {SHOTS} shots, gate = \
         {K_SIGMA}x the per-circuit null RMS",
        corpus.label,
        corpus.files.len()
    );
    if !corpus.is_vendored {
        eprintln!(
            "  NOTE: the PRIVATE corpus took precedence. This repository does not \
             audit it, so a green run here means something different from a green \
             run on tests/fixtures/crosscheck."
        );
    }

    let statevector = StatevectorBackend::new();
    let mps = MpsBackend::new(64);
    // A NoiseModel of all zeros, so this row is NOT testing noise — Qiskit is
    // the noiseless anchor and a noisy engine could not be compared against it
    // at all. What it does test is the **trajectory path**: `NoisyMpsBackend`
    // evolves one independent shot at a time even at zero noise, which is a
    // different code path from `MpsBackend`'s analytic sample-at-the-end, and
    // is where the `Reset` per-shot defect (`ae6da5c`) lived. Named `(p=0)` so
    // the report cannot be read as noise coverage.
    let noisy_mps = NoisyMpsBackend::with_model(64, Default::default());
    let pauli = PauliBackend::new();
    let in_tree: Vec<(&'static str, &dyn CoreBackend)> = vec![
        ("statevector", &statevector),
        ("mps", &mps),
        ("noisy-mps(p=0)", &noisy_mps),
        ("pauli", &pauli),
    ];

    let mut rows: Vec<Row> = Vec::new();
    let mut lowering_gaps: Vec<(String, String, &'static str)> = Vec::new();

    for (idx, path) in corpus.files.iter().enumerate() {
        let qasm = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let fixture = path.file_name().unwrap().to_string_lossy().to_string();

        // The anchor. A fixture Qiskit itself cannot run is not a fixture this
        // lane can say anything about — record it and move on rather than
        // silently comparing our engines to each other.
        let anchor = match bridge_status(omega_bridges::run_qasm2(
            omega_bridges::Backend::Qiskit,
            &qasm,
            SHOTS,
            None,
        )) {
            Ok(c) => c,
            Err(status) => {
                rows.push(Row {
                    fixture: fixture.clone(),
                    engine: "qiskit(anchor)",
                    status,
                });
                continue;
            }
        };
        let anchor_total: u32 = anchor.values().sum();
        assert_eq!(
            anchor_total, SHOTS,
            "{fixture}: Qiskit did not conserve shots"
        );

        // Feed every engine the SAME source text (Part K4 trap 1: routing
        // through Aria's QASM export would let both sides lose `if(c==V)`
        // together and agree about it).
        let known_gap = NOT_LOWERABLE.iter().find(|(f, _)| *f == fixture);
        let ir = match omega_parser::lower_to_ir(&qasm) {
            Ok(ir) => {
                assert!(
                    known_gap.is_none(),
                    "{fixture} is listed in NOT_LOWERABLE but now lowers cleanly. \
                     Remove the entry — a stale gap list silently suppresses a \
                     fixture that works."
                );
                ir
            }
            Err(e) => {
                // A gap that is on the list is reported and counted; one that
                // is not is a failure.
                let status = match known_gap {
                    Some((_, why)) => {
                        lowering_gaps.push((fixture.clone(), e.clone(), *why));
                        continue;
                    }
                    None => Status::Error(format!("lowering failed: {e}")),
                };
                for (name, _) in &in_tree {
                    rows.push(Row {
                        fixture: fixture.clone(),
                        engine: name,
                        status: status.clone(),
                    });
                }
                continue;
            }
        };

        for (name, backend) in &in_tree {
            let seed = 0x5eed_0000u64 ^ ((idx as u64) << 8) ^ name.len() as u64;
            let status = match run_in_tree(*backend, &ir, seed) {
                Ok(counts) => {
                    let total: u32 = counts.values().sum();
                    let l2 = count_l2(&anchor, &counts, anchor_total, total);
                    let gate = l2_gate(&anchor, anchor_total, total);
                    if l2 <= gate {
                        Status::Agree(l2, gate)
                    } else {
                        Status::Disagree(l2, gate)
                    }
                }
                Err(status) => status,
            };
            rows.push(Row {
                fixture: fixture.clone(),
                engine: name,
                status,
            });
        }

        for (name, backend) in bridge_engines() {
            let status = match bridge_status(omega_bridges::run_qasm2(backend, &qasm, SHOTS, None))
            {
                Ok(counts) => {
                    let total: u32 = counts.values().sum();
                    let l2 = count_l2(&anchor, &counts, anchor_total, total);
                    let gate = l2_gate(&anchor, anchor_total, total);
                    if l2 <= gate {
                        Status::Agree(l2, gate)
                    } else {
                        Status::Disagree(l2, gate)
                    }
                }
                Err(status) => status,
            };
            rows.push(Row {
                fixture: fixture.clone(),
                engine: name,
                status,
            });
        }
    }

    report_and_assert(&rows, corpus.files.len(), &lowering_gaps);
}

/// Bridge engines other than the anchor, as their features allow.
#[allow(unused_mut, dead_code)]
fn bridge_engines() -> Vec<(&'static str, omega_bridges::Backend)> {
    let mut v: Vec<(&'static str, omega_bridges::Backend)> = Vec::new();
    #[cfg(feature = "bridge-perceval")]
    v.push(("perceval", omega_bridges::Backend::Perceval));
    #[cfg(feature = "bridge-tsim")]
    v.push(("tsim", omega_bridges::Backend::Tsim));
    #[cfg(feature = "bridge-ppvm")]
    v.push(("ppvm", omega_bridges::Backend::Ppvm));
    v
}

/// Print the matrix, then fail on any `disagree` or `error`.
///
/// The per-engine **qualifying count** is printed on every line, not just the
/// failures. "17/17 agree" over a matrix where five engines ran two circuits
/// each reads as coverage and is worse than no matrix at all (Part K4 trap 5).
#[allow(dead_code)]
fn report_and_assert(
    rows: &[Row],
    n_fixtures: usize,
    lowering_gaps: &[(String, String, &'static str)],
) {
    let mut per_engine: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
    for row in rows {
        *per_engine
            .entry(row.engine)
            .or_default()
            .entry(row.status.tag())
            .or_insert(0) += 1;
    }

    eprintln!("\n  {:<16} {:>7} {:>9} {:>15} {:>14} {:>6}", "engine", "agree", "disagree", "cannot-express", "not-installed", "error");
    for (engine, tally) in &per_engine {
        let g = |t: &str| tally.get(t).copied().unwrap_or(0);
        eprintln!(
            "  {engine:<16} {:>7} {:>9} {:>15} {:>14} {:>6}   ({} of {n_fixtures} fixtures compared)",
            g("agree"),
            g("disagree"),
            g("cannot-express"),
            g("not-installed"),
            g("error"),
            g("agree") + g("disagree"),
        );
    }

    // Tracked gaps are printed, not hidden. They reduce the fixture count
    // every in-tree engine could reach, and the reader must see that.
    if !lowering_gaps.is_empty() {
        eprintln!(
            "\n  {} of {n_fixtures} fixtures reached NO in-tree engine — the QASM2 \
             lowering refused them (tracked in NOT_LOWERABLE):",
            lowering_gaps.len()
        );
        for (fixture, err, why) in lowering_gaps {
            eprintln!("    {fixture:<28} {err}");
            eprintln!("      {why}");
        }
    }

    let bad: Vec<&Row> = rows
        .iter()
        .filter(|r| matches!(r.status, Status::Disagree(..) | Status::Error(_)))
        .collect();
    if !bad.is_empty() {
        eprintln!("\n  failures:");
        for row in &bad {
            match &row.status {
                Status::Disagree(l2, gate) => eprintln!(
                    "    {:<28} {:<14} L2 = {l2:.4e} > gate {gate:.4e}",
                    row.fixture, row.engine
                ),
                Status::Error(msg) => {
                    eprintln!("    {:<28} {:<14} ERROR {msg}", row.fixture, row.engine)
                }
                _ => unreachable!(),
            }
        }
    }

    // Vacuous-pass guard. Every engine could be `not-installed` and the
    // matrix would otherwise report a clean sheet having compared nothing.
    let compared = rows
        .iter()
        .filter(|r| matches!(r.status, Status::Agree(..) | Status::Disagree(..)))
        .count();
    assert!(
        compared > 0,
        "the matrix compared ZERO cells — every engine was unavailable or \
         refused. Passing here would report coverage that does not exist."
    );

    assert!(
        bad.is_empty(),
        "{} of {} matrix cells disagreed with Qiskit or errored (see above)",
        bad.len(),
        rows.len()
    );
}

// --------------------------------------------------------------------------
// Tests of the harness itself
// --------------------------------------------------------------------------

/// The gate must scale with the circuit, not sit at a constant.
///
/// This is the concrete form of K2's correction: a fixed threshold is
/// simultaneously too tight on a narrow distribution and vacuous on a wide
/// one. A two-outcome fair coin and a 64-outcome uniform distribution at the
/// same shot count must not get the same number.
#[test]
fn the_l2_gate_scales_with_the_circuit() {
    let n = 20_000u32;
    let mut bell: StrCounts = HashMap::new();
    bell.insert("00".into(), n / 2);
    bell.insert("11".into(), n / 2);

    let mut wide: StrCounts = HashMap::new();
    for k in 0..64u32 {
        wide.insert(format!("{k:06b}"), n / 64);
    }

    let g_bell = l2_gate(&bell, n, n);
    let g_wide = l2_gate(&wide, n, n);

    // Σ p(1-p): Bell = 2·(1/2)(1/2) = 0.5; uniform-64 = 64·(1/64)(63/64) ≈ 0.984.
    // So the wide gate is larger by ≈ √(0.984/0.5) ≈ 1.4.
    assert!(
        g_wide > g_bell * 1.3,
        "wide gate {g_wide:.5e} must exceed the Bell gate {g_bell:.5e}"
    );
    // Both must stay far below the O(1) L2 a real convention error produces.
    // The measured widest gate is 5.95e-2 (uniform-64 at 20k shots); the
    // ordering bug this file caught during development sat at 2.4e-1 to 5.9e-1
    // and a dropped conditional at 1.41. A 0.1 ceiling keeps at least a 2.4x
    // margin on the smallest of those.
    assert!(
        g_wide < 0.1,
        "gate {g_wide:.5e} is too loose to catch a real convention error"
    );
}

/// A deterministic circuit gets a floor, not a zero gate.
#[test]
fn the_l2_gate_has_a_floor_for_deterministic_circuits() {
    let mut det: StrCounts = HashMap::new();
    det.insert("1".into(), 20_000);
    let gate = l2_gate(&det, 20_000, 20_000);
    assert!(gate > 0.0, "a zero gate would fail on a single rounded count");
    assert!(gate < 1e-3, "the floor must stay far below any real disagreement");
}

/// The key conversion must be tested against the shape it exists for, not
/// merely exercised. `08_partial_measure` has an unmeasured qubit; converting
/// correctly yields 2-bit keys, and skipping the projection yields 3-bit keys
/// that share no key at all with Qiskit's.
#[test]
fn projected_keys_match_the_bridge_convention_on_partial_measure() {
    let qasm = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("omega-bridges")
            .join("tests")
            .join("fixtures")
            .join("crosscheck")
            .join("08_partial_measure.qasm"),
    )
    .expect("read 08_partial_measure.qasm");
    let ir = omega_parser::lower_to_ir(&qasm).expect("lower");

    assert_eq!(ir.num_qubits, 3);
    assert_eq!(ir.num_classical_bits, 2);
    assert!(
        !needs_collapse(&ir),
        "terminal measures — this fixture must take the Skip + project route, \
         which is the route the projection exists for"
    );
    assert_eq!(key_width(&ir), 2, "keys must be creg-width, not qubit-width");

    let backend = omega_backend_statevector::StatevectorBackend::new();
    let counts = run_in_tree(&backend, &ir, 0xC0FFEE).expect("statevector run");

    let mut keys: Vec<&String> = counts.keys().collect();
    keys.sort();
    assert_eq!(
        keys,
        vec!["00", "11"],
        "expected Qiskit's two 2-bit outcomes; got {keys:?}. Three-character \
         keys mean the unmeasured h q[2] leaked into the key."
    );
    let total: u32 = counts.values().sum();
    assert_eq!(total, SHOTS);
    // Roughly balanced — the Bell pair, with q[2] correctly discarded.
    let n00 = counts["00"] as f64 / total as f64;
    assert!((n00 - 0.5).abs() < 0.02, "P(00) = {n00}, expected ~0.5");
}

/// The bit order must be pinned on an outcome that is NOT a palindrome.
///
/// Every fixture whose outcomes are `{00, 11}` — Bell, GHZ, the partial-measure
/// case — is invariant under reversal, so an ordering bug passes them all. This
/// test exists because exactly that happened: see [`to_str_counts`].
#[test]
fn bit_order_is_lsb_first_on_an_asymmetric_outcome() {
    let mut raw: HashMap<u64, u32> = HashMap::new();
    raw.insert(0b100, 1); // bit 2 set, bits 1 and 0 clear
    let out = to_str_counts(&raw, 3).expect("0b100 fits 3 bits");
    let key = out.keys().next().unwrap();
    assert_eq!(
        key, "001",
        "LSB-first: bit 0 is the LEFTMOST character, so the u64 0b100 renders \
         as \"001\". Getting \"100\" here means the harness is using Qiskit's \
         display order rather than the bridge wire order."
    );
}

/// A key wider than its declared register is refused, not truncated. This is
/// the guard that would have exposed `PauliBackend`'s wrong-register keying on
/// the matrix's first run instead of scoring it as agreement.
#[test]
fn a_key_too_wide_for_its_register_is_refused() {
    let mut raw: HashMap<u64, u32> = HashMap::new();
    raw.insert(3, 10); // 0b11 in a 1-bit register
    let err = to_str_counts(&raw, 1).expect_err("key 3 must not fit 1 bit");
    assert!(err.contains("does not fit"), "unhelpful message: {err}");

    // ...and the correct width still works, so the guard is not just "always
    // refuses".
    assert!(to_str_counts(&raw, 2).is_ok());
}

/// `09_unitary_only` has no creg and no measure, so keys stay full-register
/// width — matching the Qiskit runner's synthesised `measure_all()`.
#[test]
fn no_measure_fixtures_keep_full_register_width() {
    let qasm = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("omega-bridges")
            .join("tests")
            .join("fixtures")
            .join("crosscheck")
            .join("09_unitary_only.qasm"),
    )
    .expect("read 09_unitary_only.qasm");
    let ir = omega_parser::lower_to_ir(&qasm).expect("lower");
    assert!(measure_pairs(&ir).is_empty());
    assert_eq!(key_width(&ir), 3);

    let backend = omega_backend_statevector::StatevectorBackend::new();
    let counts = run_in_tree(&backend, &ir, 0xBEEF).expect("statevector run");
    assert!(
        counts.keys().all(|k| k.len() == 3),
        "expected 3-bit keys, got {:?}",
        counts.keys().collect::<Vec<_>>()
    );
}
