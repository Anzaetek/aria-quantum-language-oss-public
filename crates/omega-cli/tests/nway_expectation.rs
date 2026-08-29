// SPDX-License-Identifier: Apache-2.0
//! N-way **expectation** matrix: one QASM2 corpus, every in-tree engine that
//! can evaluate `⟨O⟩`, anchored on Qiskit (`FIXES_PLAN.md` Part K, step 4).
//!
//! Sibling of `nway_counts.rs`. Same corpus, same five-way status taxonomy,
//! same qualifying-count reporting. What differs is everything about the
//! *quantity*, and each difference is load-bearing.
//!
//! # This lane is analytic, so its tolerance is not the counts lane's
//!
//! Every engine here evaluates an exact contraction — no shots, no sampling
//! noise. K2: "analytic vs stochastic must not share a tolerance." The counts
//! lane derives a per-circuit σ gate from the shot count; this one uses a flat
//! `1e-12`, which is ~10 orders tighter and is the correct scale for
//! double-precision linear algebra on 2–3 qubit states.
//!
//! # The observable set is DERIVED, not chosen
//!
//! An earlier draft of the plan picked observables by reasoning — Z on each
//! qubit, ZZ on adjacent pairs, X/Y on q0 — and argued that made the lane "not
//! all-diagonal". Measured against a planted `t`→`tdg` defect on
//! `09_unitary_only`, **every one of those seven observables returned a
//! difference of exactly 0.00e+00**. Of all 63 non-identity 3-qubit Paulis,
//! only four see that defect (`XYI`, `XYX`, `YXI`, `YXX`), all mixed X⊗Y terms
//! of weight ≥ 2: entanglement traces out single-qubit coherence, so weight-1
//! X or Y on any qubit is blind to it.
//!
//! A greedy cover of all planted defects needs only three observables. That is
//! rejected deliberately — a minimal cover is fitted to the defects I happened
//! to plant, which is overfitting dressed as efficiency. At 2–3 qubits the
//! **full weight-≤2 set** (15 for n=2, 36 for n=3) is free and carries no
//! selection bias, so that is what runs. [`observable_set_detects_planted_defects`]
//! then verifies the harness rather than justifying a hand-picked list.
//!
//! # Mid-circuit constructs are refused from the IR, not filtered by name
//!
//! **This paragraph described a contract that no longer holds.** It said
//! "expectation is a property of unitary evolution", which was this repository's
//! own invention rather than anyone's convention, and it is what let a measured
//! circuit be answered as though the measurement had not happened.
//!
//! What holds now (`omega_core::defer_measure`): an INERT measurement — nothing
//! reads the bit it wrote, nothing touches the qubit again — is elided, which is
//! what `remove_final_measurements()` does on the Qiskit side. A measurement
//! whose bit is read by a later guard is DEFERRED into a quantum control and the
//! observable is dephased on that qubit. A measured qubit that is then used
//! coherently is REFUSED.
//!
//! The lane's admission rule caught up on 2026-08-20: a conditioned circuit is
//! ADMITTED, and its anchor comes from the runner's `expectation-mixture` mode
//! — branch-exact evolution over Qiskit's linear algebra, one branch per
//! measurement outcome, no shots — so fixtures 12 and 14 enter the matrix and
//! the tolerance stays the flat analytic `1e-12` (the earlier plan assumed an
//! Aer shot-based oracle and a √N tolerance split; branch evolution makes both
//! unnecessary). What stays out: `reset` (a channel neither side models here)
//! and a measured qubit later used coherently, which the engines refuse by
//! contract. The 12/14 pair is ALSO the mutation guard for the new oracle: an
//! oracle that drops guards agrees on 14 and disagrees on 12 by design.
//! `crates/omega-cli/tests/deferred_expectation.rs` still covers the contract
//! across engines; this lane is where correctness (an external anchor) lives.
//!
//! The historical divergence that motivated the strict rule, kept because it
//! explains the shape of the code below:
//!
//! - `StatevectorBackend::expectation` runs in `Skip` mode, so conditionals
//!   evaluate against a classical register no measure ever writes — an
//!   `if(c==0)` guard silently *fires*.
//! - `MpsBackend::expectation` rejects any `Reset`; statevector applies
//!   deterministic ones. Different policies, same input.
//! - Qiskit's `Statevector.from_instruction` raises on a leftover measure but
//!   on an entangled `reset` returns **one stochastic trajectory** — measured:
//!   two distinct states over 30 runs of Bell + `reset q[0]`. A nondeterministic
//!   anchor.
//!
//! So the exclusion is a runtime guard, enforced by inspecting the lowered IR.
//! Filtering by filename would let a future fixture through silently.

#[allow(unused_imports)]
use std::collections::BTreeMap;

use omega_bridges::WireObservable;
use omega_core::circuit::{CircuitIR, GateKind};
use omega_core::executor::{Backend as CoreBackend, Observable, PauliOp};

mod common;

/// Analytic gate. Every engine here does exact linear algebra on ≤ 3 qubits;
/// accumulated double-precision error is ~1e-15, so 1e-12 leaves three orders
/// of headroom without being loose enough to hide anything: the planted
/// defects this lane is built to catch move `⟨O⟩` by 0.1–2.0.
const TOL: f64 = 1e-12;

/// Why a cell is not a comparison. Mirrors `nway_counts.rs` — collapsing these
/// into "skipped" is how a matrix reports coverage it does not have (K1).
#[derive(Clone, Debug)]
#[allow(dead_code)]
enum Status {
    Agree(f64),
    Disagree { worst: f64, observable: String },
    CannotExpress(String),
    NotInstalled(String),
    Error(String),
}

impl Status {
    #[allow(dead_code)]
    fn tag(&self) -> &'static str {
        match self {
            Status::Agree(_) => "agree",
            Status::Disagree { .. } => "disagree",
            Status::CannotExpress(_) => "cannot-express",
            Status::NotInstalled(_) => "not-installed",
            Status::Error(_) => "error",
        }
    }
}

// --------------------------------------------------------------------------
// Observables
// --------------------------------------------------------------------------

/// All Pauli strings of weight 1 or 2 on `n` qubits, as `(qubit, PauliOp)`
/// pairs. 3n weight-1 terms plus 9·C(n,2) weight-2 terms: 15 at n=2, 36 at n=3.
fn weight_le_2_paulis(n: u32) -> Vec<Vec<(u32, PauliOp)>> {
    const P: [PauliOp; 3] = [PauliOp::X, PauliOp::Y, PauliOp::Z];
    let mut out = Vec::new();
    for q in 0..n {
        for p in P {
            out.push(vec![(q, p)]);
        }
    }
    for a in 0..n {
        for b in (a + 1)..n {
            for pa in P {
                for pb in P {
                    out.push(vec![(a, pa), (b, pb)]);
                }
            }
        }
    }
    out
}

fn pauli_char(p: PauliOp) -> char {
    match p {
        PauliOp::I => 'I',
        PauliOp::X => 'X',
        PauliOp::Y => 'Y',
        PauliOp::Z => 'Z',
    }
}

/// Render as the **LSB-first** dense string the bridge wire uses: index 0 of
/// the string is qubit 0. Qiskit's own `SparsePauliOp` is the reverse; the
/// runner flips it. Pinned by [`wire_strings_are_lsb_first`].
fn to_wire(term: &[(u32, PauliOp)], n: u32) -> WireObservable {
    let mut s = vec!['I'; n as usize];
    for (q, p) in term {
        s[*q as usize] = pauli_char(*p);
    }
    vec![(s.into_iter().collect::<String>(), 1.0)]
}

fn to_observable(term: &[(u32, PauliOp)]) -> Observable {
    Observable {
        terms: vec![(1.0, term.to_vec())],
    }
}

fn label(term: &[(u32, PauliOp)], n: u32) -> String {
    to_wire(term, n)[0].0.clone()
}

// --------------------------------------------------------------------------
// Fixture admission — from the IR, never from the filename
// --------------------------------------------------------------------------

/// Why a circuit cannot enter the expectation lane, if it cannot.
///
/// Conditioned gates and mid-circuit measurement are ADMITTED since the
/// `expectation-mixture` anchor exists; what remains out is what neither side
/// answers: `reset`, and a measured qubit later used coherently (the product
/// contract refuses it — `omega_core::defer_measure`).
fn rejection_reason(ir: &CircuitIR) -> Option<String> {
    if ir.ops.iter().any(|op| matches!(op.gate, GateKind::Reset)) {
        return Some("reset (non-unitary channel)".into());
    }
    // A measured qubit touched by a later GATE is coherent reuse. A later
    // measure of the same qubit is fine (it re-measures the collapsed state),
    // and a conditioned gate on OTHER qubits is the admitted feedforward case.
    for (i, op) in ir.ops.iter().enumerate() {
        if !matches!(op.gate, GateKind::Measure) {
            continue;
        }
        let q = op.qubits[0];
        if ir.ops[i + 1..].iter().any(|later| {
            !matches!(later.gate, GateKind::Measure | GateKind::Barrier)
                && later.qubits.contains(&q)
        }) {
            return Some(format!(
                "qubit {} is used coherently after being measured (refused by \
                 the deferred-measurement contract)",
                q.0
            ));
        }
    }
    None
}

/// Whether the anchor must be the mixture mode: any conditioned gate, or any
/// measure that is not terminal-inert. (A purely-terminal-measure circuit
/// keeps the plain `Statevector` anchor, unchanged from before.)
fn needs_mixture_anchor(ir: &CircuitIR) -> bool {
    if ir.ops.iter().any(|op| op.condition.is_some()) {
        return true;
    }
    ir.ops.iter().enumerate().any(|(i, op)| {
        matches!(op.gate, GateKind::Measure)
            && ir.ops[i + 1..]
                .iter()
                .any(|next| !matches!(next.gate, GateKind::Measure | GateKind::Barrier))
    })
}

/// Drop terminal `Measure` ops, so the IR handed to each engine is identical to
/// the one the Qiskit runner sees after `remove_final_measurements` and a
/// disagreement cannot be blamed on the two sides having stripped differently.
///
/// This is now REDUNDANT rather than load-bearing: `prepare_for_expectation`
/// elides inert measurements inside the product code, so stripping here removes
/// ops that the engines would have removed themselves. It is retained only
/// because it keeps the two sides' inputs visibly identical at the call site.
///
/// It is worth being clear about what removing it would and would not achieve.
/// It does NOT expose the feedforward divergence: `rejection_reason` refuses
/// conditioned circuits before this function is ever reached, so fixtures 12 and
/// 14 are already out of the lane. An earlier plan claimed deleting this call was
/// the one-line change that would make the new contract visible here; that was
/// wrong on both counts.
fn strip_terminal_measures(ir: &CircuitIR) -> CircuitIR {
    let mut out = ir.clone();
    out.ops.retain(|op| !matches!(op.gate, GateKind::Measure));
    out
}

// --------------------------------------------------------------------------
// Bridge plumbing (mirrors nway_counts.rs)
// --------------------------------------------------------------------------

#[allow(dead_code)]
fn runner_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("omega-bridges")
        .join("python")
}

#[allow(dead_code)]
fn venv_python(slug: &str) -> std::path::PathBuf {
    // Shared resolver: honours ARIA_QISKIT_PY and both venv locations. A
    // bridge-local-only lookup made this lane self-skip on hosts whose venv
    // sits where ci.sh puts it — green, but comparing against nothing.
    common::venv_python(slug)
}

#[allow(dead_code)]
fn force_runner_env(slug: &str) {
    std::env::set_var(
        format!("OMEGA_BRIDGE_{}_CMD", slug.to_ascii_uppercase()),
        runner_dir().join(format!("omega-bridge-{slug}-runner")),
    );
}

#[allow(dead_code)]
struct Row {
    fixture: String,
    engine: &'static str,
    status: Status,
}

/// Evaluate every observable on one engine, against the anchor's values.
#[allow(dead_code)]
fn score_engine(
    backend: &dyn CoreBackend,
    ir: &CircuitIR,
    terms: &[Vec<(u32, PauliOp)>],
    anchor: &[f64],
) -> Status {
    let params = omega_core::params::ParameterBinding::new();
    let mut worst = 0.0_f64;
    let mut worst_obs = String::new();
    for (term, &want) in terms.iter().zip(anchor) {
        match backend.expectation(ir, &params, &to_observable(term)) {
            Ok(got) => {
                if !got.is_finite() {
                    return Status::Error(format!(
                        "{} returned a non-finite value for {}",
                        backend.name(),
                        label(term, ir.num_qubits)
                    ));
                }
                let d = (got - want).abs();
                if d > worst {
                    worst = d;
                    worst_obs = label(term, ir.num_qubits);
                }
            }
            // A typed refusal is a CORRECT answer for an engine outside its
            // class (`pauli` on a T gate, `pauliprop` on a conditional). It is
            // not a defect, and must not redden the row.
            Err(omega_core::error::OmegaError::Unsupported(msg)) => {
                return Status::CannotExpress(msg)
            }
            Err(e) => return Status::Error(e.to_string()),
        }
    }
    if worst <= TOL {
        Status::Agree(worst)
    } else {
        Status::Disagree {
            worst,
            observable: worst_obs,
        }
    }
}

// --------------------------------------------------------------------------
// The matrix
// --------------------------------------------------------------------------

#[cfg(feature = "bridge-qiskit")]
#[test]
fn nway_expectation_matrix_agrees_with_qiskit() {
    use omega_backend_mps::MpsBackend;
    use omega_backend_pauli::PauliBackend;
    use omega_backend_pauliprop::PauliPropBackend;
    use omega_backend_statevector::StatevectorBackend;
    use omega_bridges::corpus::crosscheck_corpus;

    if !venv_python("qiskit").exists() {
        eprintln!(
            "Qiskit venv missing at {} — skipping the N-way expectation matrix. \
             Without the anchor this lane degrades to our own engines agreeing \
             with each other, which is the weak-evidence configuration that let \
             the MPS trajectory defect through (FIXES_PLAN.md K3/K7).",
            venv_python("qiskit").display()
        );
        return;
    }
    force_runner_env("qiskit");

    let corpus = crosscheck_corpus();
    eprintln!(
        "\nN-way EXPECTATION matrix — corpus {} ({} fixtures), analytic gate {TOL:.0e}",
        corpus.label,
        corpus.files.len()
    );

    let statevector = StatevectorBackend::new();
    let mps = MpsBackend::new(64);
    let pauli = PauliBackend::new();
    let pauliprop = PauliPropBackend::new();
    let engines: Vec<(&'static str, &dyn CoreBackend)> = vec![
        ("statevector", &statevector),
        ("mps", &mps),
        ("pauli", &pauli),
        ("pauliprop", &pauliprop),
    ];

    let mut rows: Vec<Row> = Vec::new();
    let mut admitted = 0usize;
    let mut refused: Vec<(String, String)> = Vec::new();

    for path in &corpus.files {
        let fixture = path.file_name().unwrap().to_string_lossy().to_string();
        let Ok(qasm) = std::fs::read_to_string(path) else {
            continue;
        };
        let ir = match omega_parser::lower_to_ir(&qasm) {
            Ok(ir) => ir,
            Err(e) => {
                refused.push((fixture, format!("lowering: {e}")));
                continue;
            }
        };
        if let Some(why) = rejection_reason(&ir) {
            refused.push((fixture, why));
            continue;
        }
        // A mixture fixture keeps its RAW ir: the engines' own
        // `prepare_for_expectation` elides/defers measurements, and stripping
        // here would delete the mid-circuit measure the guard reads. The
        // unitary fixtures keep the historical strip (redundant but visibly
        // identical to what the plain anchor sees).
        let mixture = needs_mixture_anchor(&ir);
        let ir = if mixture {
            ir
        } else {
            strip_terminal_measures(&ir)
        };
        let n = ir.num_qubits;
        let terms = weight_le_2_paulis(n);
        let wire: Vec<_> = terms.iter().map(|t| to_wire(t, n)).collect();

        // The anchor. A fixture Qiskit cannot evaluate says nothing about our
        // engines, so it is recorded and skipped rather than compared
        // internally. A mixture fixture is anchored by branch-exact
        // evolution; both anchors are analytic, so ONE tolerance covers the
        // whole matrix.
        let anchor_result = if mixture {
            omega_bridges::expectation_mixture_qasm2(omega_bridges::Backend::Qiskit, &qasm, &wire)
        } else {
            omega_bridges::expectation_qasm2(omega_bridges::Backend::Qiskit, &qasm, &wire)
        };
        let anchor = match anchor_result {
            Ok(v) => v,
            Err(e) => {
                refused.push((fixture, format!("qiskit anchor: {e}")));
                continue;
            }
        };
        admitted += 1;

        for (name, backend) in &engines {
            let status = score_engine(*backend, &ir, &terms, &anchor);
            rows.push(Row {
                fixture: fixture.clone(),
                engine: name,
                status,
            });
        }
    }

    // Report
    let mut per_engine: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
    for r in &rows {
        *per_engine
            .entry(r.engine)
            .or_default()
            .entry(r.status.tag())
            .or_insert(0) += 1;
    }
    eprintln!(
        "  {:<14}{:>7}{:>10}{:>16}{:>7}",
        "engine", "agree", "disagree", "cannot-express", "error"
    );
    for (engine, tally) in &per_engine {
        let g = |t: &str| tally.get(t).copied().unwrap_or(0);
        eprintln!(
            "  {engine:<14}{:>7}{:>10}{:>16}{:>7}   ({} of {admitted} admitted fixtures compared)",
            g("agree"),
            g("disagree"),
            g("cannot-express"),
            g("error"),
            g("agree") + g("disagree"),
        );
    }
    if !refused.is_empty() {
        eprintln!(
            "\n  {} of {} fixtures not admitted:",
            refused.len(),
            corpus.files.len()
        );
        for (f, why) in &refused {
            eprintln!("    {f:<36} {why}");
        }
    }

    let bad: Vec<&Row> = rows
        .iter()
        .filter(|r| matches!(r.status, Status::Disagree { .. } | Status::Error(_)))
        .collect();
    for r in &bad {
        match &r.status {
            Status::Disagree { worst, observable } => eprintln!(
                "    DISAGREE {:<30} {:<12} worst |Δ| = {worst:.3e} on {observable}",
                r.fixture, r.engine
            ),
            Status::Error(m) => eprintln!("    ERROR    {:<30} {:<12} {m}", r.fixture, r.engine),
            _ => {}
        }
    }

    // Vacuous-pass guards. The counts lane's `compared > 0` is too weak here:
    // this lane admits far fewer fixtures, so "at least one cell" could be
    // satisfied by a single trivial circuit.
    assert!(
        admitted >= 13,
        "only {admitted} fixtures were admitted — the lane is not exercising \
         the corpus. Expected 13 (14 minus 13_reset_midcircuit); the two \
         feedforward fixtures are admitted through the mixture anchor and \
         must not silently fall back out."
    );
    let compared = rows
        .iter()
        .filter(|r| matches!(r.status, Status::Agree(_) | Status::Disagree { .. }))
        .count();
    assert!(compared > 0, "the matrix compared ZERO cells");
    assert!(
        bad.is_empty(),
        "{} cells disagreed with Qiskit or errored",
        bad.len()
    );
}

// --------------------------------------------------------------------------
// Harness self-tests — these need no venv and no feature
// --------------------------------------------------------------------------

/// The wire string must be LSB-first. Pinned on an ASYMMETRIC term, because
/// `ZZ`, `XX` and `II` read identically in both directions — which is how the
/// counts lane shipped a reversed-order bug past its own first test.
#[test]
fn wire_strings_are_lsb_first() {
    // Z on qubit 0 of a 3-qubit register.
    let w = to_wire(&[(0, PauliOp::Z)], 3);
    assert_eq!(w.len(), 1);
    assert_eq!(
        w[0].0, "ZII",
        "LSB-first: qubit 0 is the LEFTMOST character. \"IIZ\" would be \
         Qiskit's MSB-first convention, which the runner applies on its own \
         side — applying it here too would reverse twice and silently cancel."
    );
    assert_eq!(to_wire(&[(2, PauliOp::Y)], 3)[0].0, "IIY");
    assert_eq!(
        to_wire(&[(0, PauliOp::X), (1, PauliOp::Y)], 3)[0].0,
        "XYI",
        "mixed X⊗Y weight-2 terms are the ONLY ones that see phase defects \
         like t→tdg once entanglement traces out single-qubit coherence"
    );
}

/// Counts: 3n weight-1 terms + 9·C(n,2) weight-2 terms.
#[test]
fn the_observable_set_is_the_full_weight_le_2_family() {
    assert_eq!(weight_le_2_paulis(2).len(), 6 + 9, "n=2: 15 terms");
    assert_eq!(weight_le_2_paulis(3).len(), 9 + 27, "n=3: 36 terms");
    // Every mixed X-Y pair must be present — see wire_strings_are_lsb_first.
    let set: std::collections::HashSet<String> =
        weight_le_2_paulis(3).iter().map(|t| label(t, 3)).collect();
    for want in ["XYI", "YXI", "IXY", "IYX"] {
        assert!(set.contains(want), "observable set is missing {want}");
    }
}

/// Admission is decided from the IR: reset and coherent-reuse are rejected,
/// feedforward is admitted through the mixture anchor, terminal measures need
/// no mixture at all.
#[test]
fn admission_is_decided_by_the_ir_not_the_filename() {
    use omega_core::circuit::{CircuitType, GateOp, Qubit};
    let op = |g: GateKind, q: u32, cb: Option<u32>| GateOp {
        gate: g,
        qubits: [Qubit(q)].into_iter().collect(),
        params: Default::default(),
        classical_bit: cb,
        condition: None,
    };
    let mut ir = CircuitIR::new(2, CircuitType::GateBased);
    ir.num_classical_bits = 2;

    // terminal measures -> admitted
    ir.ops = vec![
        op(GateKind::H, 0, None),
        op(GateKind::Measure, 0, Some(0)),
        op(GateKind::Measure, 1, Some(1)),
    ];
    assert!(
        rejection_reason(&ir).is_none(),
        "terminal measures are a no-op"
    );
    assert_eq!(strip_terminal_measures(&ir).ops.len(), 1);

    // and terminal measures do NOT need the mixture anchor.
    ir.ops = vec![
        op(GateKind::H, 0, None),
        op(GateKind::Measure, 0, Some(0)),
        op(GateKind::Measure, 1, Some(1)),
    ];
    assert!(!needs_mixture_anchor(&ir));

    // a gate on ANOTHER qubit after a measure -> admitted (the measure is
    // inert or deferred, both sides handle it), via the mixture anchor.
    ir.ops = vec![op(GateKind::Measure, 0, Some(0)), op(GateKind::X, 1, None)];
    assert!(rejection_reason(&ir).is_none());
    assert!(needs_mixture_anchor(&ir));

    // the measured qubit itself reused coherently -> rejected, matching the
    // deferred-measurement contract's refusal.
    ir.ops = vec![op(GateKind::Measure, 0, Some(0)), op(GateKind::X, 0, None)];
    assert!(rejection_reason(&ir).unwrap().contains("coherently"));

    // reset -> rejected. Qiskit's plain anchor would return a RANDOM
    // trajectory here (measured: 2 distinct states over 30 runs on Bell +
    // reset), and the mixture mode refuses it too.
    ir.ops = vec![op(GateKind::Reset, 0, None)];
    assert!(rejection_reason(&ir).unwrap().contains("reset"));

    // conditional -> ADMITTED (fixtures 12/14), via the mixture anchor.
    let mut guarded = op(GateKind::X, 1, None);
    guarded.condition = Some((0, 1, 1));
    ir.ops = vec![op(GateKind::Measure, 0, Some(0)), guarded];
    assert!(rejection_reason(&ir).is_none());
    assert!(needs_mixture_anchor(&ir));
}
