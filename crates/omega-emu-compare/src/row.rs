// SPDX-License-Identifier: Apache-2.0
//! The row schema of PLAN-EMULATOR-COMPARISON §3, as one serde struct.
//!
//! §3 pins the shape before anything runs:
//!
//! > Row schema, fixed per row and printed into the results doc: `(circuit id,
//! > generator+seed or file, qubits, depth, chi, shots, precision, box, arm
//! > versions, BLAS/threads, witness fields, value, oracle gap, time min/median
//! > of N)`.
//!
//! Everything here is that list, plus the four things the GPU lane found it
//! needed once it tried to produce a real row: the compile-time/worktree rev
//! pair, the fixed startup cost split out from steady state, the device census
//! naming each holder, and a `derivation` block carrying the inputs of every
//! derived field.
//!
//! **A speed row and a capability row are different types, not one type with
//! empty cells.** §1 is explicit — "a row below is either a speed row [...] or
//! a capability row (one arm cannot express the workload — reported as a
//! capability fact, never as an empty speed cell, in either direction)" — and
//! §1's whole subject is a distinction this project blurred once already. A sum
//! type is that rule made mechanical: a [`CapabilityRow`] has nowhere to put a
//! ratio, and a [`SpeedRow`] cannot omit its timings.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::witness::{GitRev, PathWitnesses, Witness};

/// The tie rule of §4.6, pinned before any number exists, which is the only
/// time it can be pinned honestly.
pub const WIN_RATIO_THRESHOLD: f64 = 1.3;

/// §4.6, printed into every speed row so the classification travels with its
/// rule rather than with a reader's memory of it.
pub const TIE_RULE: &str = "win or loss only if the min-of-N ratio >= 1.3x AND the \
     [min, median] intervals of the two arms are disjoint; everything else is a tie, \
     printed with its ratio (§4.6)";

/// §4.7: a row whose competitor time is under this multiple of that
/// competitor's per-call floor measures its binding, not its arithmetic.
pub const OVERHEAD_FLOOR_MULTIPLE: f64 = 10.0;

/// §4.7, printed alongside the flag for the same reason as [`TIE_RULE`].
pub const OVERHEAD_RULE: &str = "annotated overhead-dominated when the competitor's \
     min time is under 10x its own smallest possible invocation (§4.7)";

/// §4.3 value gate on f64 rows.
pub const GATE_REL_F64: f64 = 1e-10;

/// §4.3 value gate on f32 rows.
pub const GATE_REL_F32: f64 = 1e-6;

/// §4.3c: the values a trivial implementation produces without computing
/// anything. `0.0` is the empty sum and the uninitialised buffer; `1.0` is the
/// normalisation it was never asked to change; `-1.0` is `1.0` with one sign
/// slip. A value gate that pins only one of these, and nothing else, admits a
/// stub.
///
/// This exists because four rows were published with one: E2's Clifford
/// expectation rows gate on `<Z0>`, and `<Z0>` on a Clifford brickwall is
/// **exactly** `0.0` on both arms and on the dense oracle. The observable was
/// fine and the circuit made it degenerate, so the gate read `0 == 0` and a
/// backend returning `0.0` unconditionally would have passed all four.
pub const DEGENERATE_GATE_VALUES: [f64; 3] = [0.0, 1.0, -1.0];

/// How close two gate values must be to count as the same value for
/// [`DEGENERATE_GATE_VALUES`]. These are exact-integer and f64-exact gates, so
/// this is a JSON round-trip window, not a tolerance.
pub const DEGENERATE_GATE_WINDOW: f64 = 1e-12;

/// §4.4: N = 5 timed repeats, N = 3 for rows over 30 s. Fewer than three has
/// no median-min spread, so it carries no noise witness.
pub const MIN_REPEATS: u32 = 3;

/// §4.4 addendum: a GPU row is void if utilisation exceeds this before or
/// after the row, because CPU load average cannot see the card.
pub const GPU_UTIL_VOID_PCT: u32 = 10;

/// Which comparison a row belongs to (§2). The lane decides which witnesses
/// are meaningful, so it is on the row rather than inferred from the arm name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Lane {
    /// `omega-backend-statevector` f64 vs qulacs (qsim annotated second opinion).
    DenseCpu,
    /// `omega-backend-statevector-cuda` f32 vs cuStateVec, akilles only.
    DenseGpu,
    /// `omega-backend-mps` vs quimb `CircuitMPS`.
    Mps,
    /// `omega-backend-pauli` vs Stim, promoted from oracle.
    Stabilizer,
    /// `omega-backend-pauliprop` vs monoprop, under the matched-accuracy rule.
    Pauliprop,
    /// `sector` / `majoranaprop` / `sim` vs ffsim, FQE second opinion.
    Fermionic,
    /// `omega-backend-cv` vs piquasso, one row.
    Cv,
    /// `omega-backend-photonics` vs Perceval — closed by item 15, linked not redone.
    Photonic,
}

/// What the timed region computes. §4.2 splits these because sampling doubles
/// our own peak memory and because Stim's `compile_sampler()` cost has to be
/// split out the same way our setup is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quantity {
    /// Drives the library `expectation` path. Never `execute(shots: None)`:
    /// that ends in a dense `to_statevector` and times reconstruction
    /// (`STATUS.md:1408-1419`).
    Expectation,
    /// Times sampling on both arms and nothing else.
    Sampling,
}

/// The §4.6 verdict. Always accompanied by the ratio, which is published
/// whatever the verdict is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Classification {
    /// Ours faster, past the threshold, intervals disjoint.
    Win,
    /// Theirs faster, past the threshold, intervals disjoint. Published at the
    /// moment it is measured (Ruling 3), not in a final editing pass.
    Loss,
    /// Everything else, printed with its ratio.
    Tie,
}

/// Which direction a capability asymmetry runs (§5). Both directions are
/// reported; the section has two halves for a reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// "They cannot, we can."
    OursOnly,
    /// "We cannot, they can."
    CompetitorOnly,
}

/// The pinned workload (§3), identified by the committed artifact it ingests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    /// Artifact id in `tools/emu_compare/MANIFEST.json`, e.g. `hea-14q-d12`.
    pub id: String,
    /// Repo-relative path of the QASM2 file every arm ingests.
    pub file: String,
    /// The artifact's sha256, as a witness: a row that names a file without
    /// its hash cannot be shown to have run the committed circuit, and "the
    /// HEA" was reimplemented per-arm once before this directory existed.
    pub sha256: Witness<String>,
    /// The generator invocation, from the manifest.
    pub generator: String,
    /// Pinned seed where the family has one (RANDOM-1: 0).
    pub seed: Option<u64>,
    /// Qubits (modes, for the fermionic lane).
    pub qubits: u32,
    /// Circuit depth as the generator counts it.
    pub depth: u32,
    /// Bond dimension on truncating lanes; `None` on the dense ones, which are
    /// chi-free, stated rather than left blank.
    pub chi: Option<u32>,
    /// Shots on sampling rows; `None` on expectation rows.
    pub shots: Option<u32>,
    /// What is timed.
    pub quantity: Quantity,
    /// The observable, in `omega_core::Observable::parse` syntax, so both arms
    /// can be shown to have built the same one.
    pub observable: Option<String>,
}

/// §4.5. Precision is matched or the row is marked, and a marked row counts as
/// evidence when we lose and is never quoted as a win.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Precision {
    /// e.g. `f64`, `f32`.
    pub ours: String,
    /// e.g. `f64` (qulacs), `f32` (qsim, cuStateVec).
    pub competitor: String,
    /// Whether the two match.
    pub matched: bool,
    /// Required when they do not: the §4.5 annotation, including that an
    /// unmatched row can only be published as a loss.
    pub note: Option<String>,
}

/// Thread and BLAS configuration (§3's `BLAS/threads`, §8's strawman clause).
///
/// quimb's rows have to record which BLAS numpy links — Accelerate on
/// andromeda, OpenBLAS on akilles — because that choice moves the competitor's
/// number more than most of its own knobs do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Threads {
    /// Threads the arm reports using, not the threads the box has.
    pub threads: u32,
    /// The linked BLAS and its version, or `none` for a BLAS-free arm.
    pub blas: String,
}

/// One arm's identity and configuration. Carries no timing: a capability row
/// has arms too, and the side that cannot express the workload has no time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    /// e.g. `omega-backend-mps`, `quimb-CircuitMPS`, `stim-1.16.0`.
    pub name: String,
    /// Every version this arm's result depends on. §8: unpinned competitor
    /// versions are one of the worthlessness conditions, and
    /// `BACKEND-CROSSOVER.md` is the in-house cautionary tale.
    pub versions: Witness<BTreeMap<String, String>>,
    /// §3's BLAS/threads.
    pub threads: Witness<Threads>,
    /// Every knob that was chosen, printed per row. §8: "where a knob is
    /// ambiguous, choose in the competitor's favor" — which is only checkable
    /// if the choice is in the row.
    pub knobs: BTreeMap<String, String>,
    /// §4.1's per-arm witnesses that the path on the label is the path that
    /// ran. Non-empty, every entry asserted, or the row does not serialize.
    pub executed_path: PathWitnesses,
}

/// §4.4's repetition policy, as measured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timing {
    /// The comparison statistic.
    pub min_s: f64,
    /// The other half of the noise witness: the median-min spread.
    pub median_s: f64,
    /// N. At least [`MIN_REPEATS`].
    pub repeats: u32,
    /// One untimed warm-up invocation per (arm, row) was taken and discarded —
    /// allocator, rayon pool, NVRTC/JIT, Python import.
    pub warmup_discarded: bool,
    /// Arms were run A/B/A/B rather than all-A-then-all-B, so drift hits both.
    pub interleaved: bool,
}

/// An arm on a speed row: identity, plus the three timing facts §4.1 and §4.7
/// require to be separable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedArm {
    /// Who this is and how it was configured.
    pub arm: Arm,
    /// Steady-state min/median over N.
    pub timing: Witness<Timing>,
    /// Startup cost, kept out of the steady-state number and reported beside
    /// it. §4.1's negative-space tell: "a 'GPU' column containing no fixed GPU
    /// startup cost is not a GPU column".
    pub fixed_cost_s: Witness<f64>,
    /// This arm's smallest possible invocation, measured in E1 (§4.7). The
    /// deliberately absent 6x6 permanent row is the receipt: without a floor,
    /// a small-shape comparison measures their binding against our arithmetic.
    pub per_call_floor_s: Witness<f64>,
}

/// What the value gate compared against (§4.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateReference {
    /// A dense oracle, affordable at every shape <= 26q.
    Oracle {
        /// e.g. `aer-cpu-f64`.
        arm: String,
        /// The oracle's own version pin.
        versions: BTreeMap<String, String>,
        /// Its value.
        value: serde_json::Value,
    },
    /// Above the oracle line there is no third party, so the gate is arm vs
    /// arm. The note says exactly which way round, because this is the weaker
    /// gate and it should read as weaker.
    ArmVsArm {
        /// Which arm stood in for the oracle, and why it could.
        note: String,
    },
}

/// Both arms' truncation accounting, printed next to the dense gap where exact
/// reference values exist (§4.3). `discarded_weight` is a bound; quimb's
/// accumulated fidelity is an estimate; the row keeps them distinguishable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Truncation {
    /// Ours: a bound, not an estimate.
    pub ours_discarded_weight: f64,
    /// Ours: 1.0 means the run was exact.
    pub ours_fidelity_estimate: Option<f64>,
    /// Theirs, as whatever they actually report — verified in E1 before any
    /// claim is made about what it guarantees.
    pub competitor_fidelity_estimate: Option<f64>,
}

/// §4.3: a fast wrong number is disqualified. The gate runs before timing, so
/// a failing row is a disagreement finding rather than a silent drop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueGate {
    /// [`GATE_REL_F64`] or [`GATE_REL_F32`].
    pub tolerance_rel: f64,
    /// What both arms were held against.
    pub reference: GateReference,
    /// Our value: scalar on expectation rows, per-qubit `<Z_q>` on sampling.
    pub ours_value: serde_json::Value,
    /// Theirs, same shape.
    pub competitor_value: serde_json::Value,
    /// §3's `oracle gap`, ours.
    pub ours_gap: f64,
    /// §3's `oracle gap`, theirs.
    pub competitor_gap: f64,
    /// Truncating lanes only.
    pub truncation: Option<Truncation>,
    /// Whether the row is admitted to the speed table at all.
    pub passed: bool,
}

/// One process holding the card during the census (§4.4 addendum).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuHolder {
    /// The holder's pid.
    pub pid: u32,
    /// Device memory it holds, MiB.
    pub mib: u64,
    /// Its process name. E5's first end-to-end attempt found an ollama
    /// `llama-server` at 70.4 GB and a `train.py adapter` at 3.6 GB; "who" is
    /// the part of that record that let anyone act on it, so it is a field.
    pub name: String,
}

/// The device-side half of the quiet-box check, added after the CPU-only §4.4
/// missed a card at 98% utilisation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuCensus {
    /// Utilisation before the row, percent.
    pub util_before_pct: u32,
    /// And after.
    pub util_after_pct: u32,
    /// The threshold applied, printed so a reader need not look it up.
    pub void_above_pct: u32,
    /// Everything holding the card before the row. Non-empty voids it.
    pub holders_before: Vec<GpuHolder>,
    /// And after.
    pub holders_after: Vec<GpuHolder>,
}

/// §4.4's load record. A row without one is void (§8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadRecord {
    /// 1-minute load average before the row.
    pub load1_before: f64,
    /// And after.
    pub load1_after: f64,
    /// The threshold applied, from [`crate::load_void_above`] for the host's
    /// core count. Recorded per row because item 15's numbers moved 29% under
    /// another session's load.
    pub void_above: f64,
    /// Cooldown honoured before this row, seconds. andromeda throttles, and a
    /// thermally clipped second arm is a silent bias.
    pub cooldown_s: Option<u32>,
    /// The `omega-hostgate run --watch --host-bytes` declaration this row ran
    /// under, e.g. `--host-bytes 6G`. The wrapper sits outside the in-process
    /// timed region, so it costs neither arm.
    pub hostgate_cap: Option<String>,
    /// Device census, required on [`Lane::DenseGpu`].
    pub gpu: Option<GpuCensus>,
    /// §4.4a: CPU used by processes OUTSIDE the lane's own process tree,
    /// integrated over the row. When present, it — not `load1_after` — is the
    /// contention test, because a multi-threaded lane's own work raises the
    /// load average it would otherwise be voided by. Absent on rows recorded
    /// before §4.4a, which keep the load-average test.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_cpu: Option<ExternalCpu>,
}

/// §4.4a: contention from work that is not the lane's, as cores, over the row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalCpu {
    /// Mean cores used by processes outside the lane's tree during the row:
    /// (Δ system busy CPU time − Δ the lane tree's CPU time) / wall time.
    /// A core count, not a load average.
    pub external_cores_during_row: f64,
    /// Length of the integration window, seconds (first warm-up to last repeat).
    pub window_s: f64,
    /// How it was measured, so a reader can tell an integral from a sample.
    pub method: String,
    /// The pids counted as the lane's own tree.
    pub own_pids: Vec<u32>,
    /// Cores used by work outside the lane in a short census taken **before**
    /// the row, replacing the `load1_before` gate where it exists.
    ///
    /// Optional, and the option is a migration rather than a convenience. The
    /// rows published before this field existed were admitted by a
    /// `load1_before <= void_above` test, and they pass it; a row that carries
    /// this figure is tested on it instead. On Linux the two agree closely
    /// enough that the change is bookkeeping. On macOS `load1_before` is not a
    /// measure of CPU contention — 3.28 then 1.92 a minute apart on a box that
    /// was 85% idle by both this census and `top` — so the old gate blocks
    /// every row on that box regardless of what the box is doing, which is why
    /// the mac half of the dense lane is recorded as owed rather than absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_cores_before: Option<f64>,
    /// How long the pre-row census ran, seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_window_s: Option<f64>,
}

/// Whether our hand-written competitor loop beat the stale wrapper around the
/// same library (the §6 addendum's second mitigation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FloorVerdict {
    /// The authored loop is at least as fast as the wrapper.
    Ok,
    /// It is slower: authoring the competitor's loop means owning its
    /// performance bugs, and this is one. The loop is the finding, not the
    /// hardware.
    LoopIsTheFinding,
}

/// The sanity-floor arm (§6 addendum) — a second wrapper around the same
/// library, kept precisely so a slow hand-written loop is detectable.
///
/// Not to be confused with [`SpeedArm::per_call_floor_s`], which is §4.7's
/// per-call overhead floor. Two different "floors"; this one is an arm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FloorCheck {
    /// e.g. `aer-gpu-0.15.1`.
    pub arm: String,
    /// Its min over N, at the same configuration as the admitted competitor.
    pub min_s: f64,
    /// The verdict.
    pub verdict: FloorVerdict,
}

/// Every input to every derived field on the row, so a reader can re-derive
/// `classification`, `overhead_dominated` and `floor_check` from the row alone
/// and catch drift between inputs and outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Derivation {
    /// [`TIE_RULE`], carried rather than cited.
    pub tie_rule: String,
    /// [`WIN_RATIO_THRESHOLD`]. In the row because §8's last bullet is "a tie
    /// reported as a win because the threshold moved after the numbers
    /// existed".
    pub win_ratio_threshold: f64,
    /// `[min, median]` of our arm.
    pub ours_min_median_s: [f64; 2],
    /// `[min, median]` of the competitor.
    pub competitor_min_median_s: [f64; 2],
    /// `competitor.min / ours.min`; above 1 means ours is faster.
    pub ratio_competitor_over_ours: f64,
    /// Whether the two `[min, median]` intervals are disjoint.
    pub intervals_disjoint: bool,
    /// Whether the ratio is past the threshold in either direction.
    pub ratio_beyond_threshold: bool,
    /// [`OVERHEAD_RULE`].
    pub overhead_rule: String,
    /// The competitor's own per-call floor (§4.7).
    pub competitor_per_call_floor_s: f64,
    /// `competitor.min / floor`; under [`OVERHEAD_FLOOR_MULTIPLE`] annotates.
    pub competitor_min_over_floor: f64,
    /// The sanity-floor arm's min, when the lane has one.
    pub floor_arm_min_s: Option<f64>,
}

/// A timed comparison: same computation, both arms, one number each.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedRow {
    /// Our arm.
    pub ours: SpeedArm,
    /// The lane's yardstick (§2), the one the classification is against.
    pub competitor: SpeedArm,
    /// Arms recorded as context and never promoted to rows: the Aer-CPU number
    /// that falls out for free (§1), the sanity floor, and any competitor
    /// variant that was gated but not admitted.
    pub context_arms: BTreeMap<String, SpeedArm>,
    /// §4.3.
    pub value_gate: Witness<ValueGate>,
    /// §4.4.
    pub load: Witness<LoadRecord>,
    /// Published whatever the classification is.
    pub ratio_competitor_over_ours: f64,
    /// §4.6. Only the classification has a threshold.
    pub classification: Classification,
    /// §4.7.
    pub overhead_dominated: bool,
    /// §6 addendum, where the lane has a sanity-floor arm.
    pub floor_check: Option<FloorCheck>,
    /// The inputs of all four fields above.
    pub derivation: Witness<Derivation>,
}

/// An asymmetry: one arm cannot express the workload. Reported as a fact with
/// one line of evidence, in either direction, never as an empty speed cell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityRow {
    /// Which way the asymmetry runs.
    pub direction: Direction,
    /// The claim, worded as §5 words them — "policy", not "only we have
    /// Reset"; "quimb could; unmeasured here", not "impossible".
    pub claim: String,
    /// §5's one line of evidence. Claimed per competitor only after E1 reads
    /// what each actually reports: the uniqueness claim is verified, not
    /// asserted.
    pub evidence: Witness<String>,
    /// Our arm.
    pub ours: Arm,
    /// Theirs.
    pub competitor: Arm,
    /// What the side that cannot actually does — the exception text, the
    /// missing API, the refusal. Without it the row is an assertion about
    /// someone else's software.
    pub refusal: Witness<String>,
}

/// A row is one or the other. There is no third kind and no overlap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowBody {
    /// Same computation, both arms, timed.
    Speed(Box<SpeedRow>),
    /// One arm cannot express the workload.
    Capability(Box<CapabilityRow>),
}

/// One line of `docs/EMULATOR-COMPARISON.md`'s backing JSONL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    /// Stable id, cited from the doc's prose. E7 checks every citation
    /// resolves to a row here.
    pub row_id: String,
    /// Which comparison (§2).
    pub lane: Lane,
    /// Which phase produced it, e.g. `E5`.
    pub phase: String,
    /// The date measured, ISO. These are dated measurements on named
    /// hardware, not standing truths and not regression gates (§8).
    pub recorded: String,
    /// `andromeda` or `akilles`, as a witness: a row that cannot name its box
    /// is `BACKEND-CROSSOVER.md`, a whole document of numbers from a machine
    /// the estate no longer has.
    pub box_name: Witness<String>,
    /// §4.4's revision pair. Voids on mismatch or a dirty tree.
    pub git: Witness<GitRev>,
    /// The pinned workload (§3).
    pub circuit: Workload,
    /// §4.5.
    pub precision: Precision,
    /// Speed or capability.
    pub body: RowBody,
    /// Standing annotations that belong to the row rather than to the lane:
    /// depth-4 rows are directional only and never headlines; 14q chi >= 128
    /// sits outside MPS's useful regime; a NumPy-hot-path competitor.
    pub notes: Vec<String>,
}

/// A row that was not taken, and why.
///
/// §4.4 publishes these rather than dropping them — "a row killed at its cap
/// is published as 'exceeds declared budget at this shape', which is a result,
/// not a failure to appear" — and the E5 harness keeps void rows because their
/// useful artefact is *who held the card*. This type always serializes, which
/// is the point: the refusals built into [`Row`] have somewhere to land.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoidRow {
    /// The row that would have been.
    pub row_id: String,
    /// Which comparison.
    pub lane: Lane,
    /// Which phase.
    pub phase: String,
    /// When.
    pub recorded: String,
    /// Why, in a sentence a reader can act on.
    pub reason: String,
    /// Whatever was captured before the void: the holders, the rev pair, the
    /// gate gaps. Untyped on purpose — a void row's context is whatever the
    /// lane had at the moment it refused, and constraining it would cost the
    /// detail that makes these worth keeping.
    pub context: serde_json::Value,
}

impl Row {
    /// Every reason this row will not serialize, named by field.
    ///
    /// The refusals are enforced by the `Serialize` impls in
    /// [`crate::witness`], which cannot name their field — serde does not tell
    /// a nested value where it sits. This walks the row to produce the message
    /// a human needs. It is a diagnostic, not the enforcement: deleting it
    /// would not let an unwitnessed row through.
    pub fn refusals(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.box_name.is_present() {
            out.push("box_name: absent".to_string());
        }
        match self.git.get() {
            None => out.push("git: absent".to_string()),
            Some(rev) => {
                if let Some(reason) = rev.void_reason() {
                    out.push(format!("git: {reason}"));
                }
            }
        }
        if !self.circuit.sha256.is_present() {
            out.push("circuit.sha256: absent".to_string());
        }
        match &self.body {
            RowBody::Speed(speed) => {
                for (label, arm) in speed.arms() {
                    arm.refusals(&label, &mut out);
                }
                if !speed.value_gate.is_present() {
                    out.push("body.speed.value_gate: absent".to_string());
                }
                if !speed.load.is_present() {
                    out.push("body.speed.load: absent".to_string());
                }
                if !speed.derivation.is_present() {
                    out.push("body.speed.derivation: absent".to_string());
                }
            }
            RowBody::Capability(cap) => {
                cap.ours.refusals("body.capability.ours", &mut out);
                cap.competitor
                    .refusals("body.capability.competitor", &mut out);
                if !cap.evidence.is_present() {
                    out.push("body.capability.evidence: absent".to_string());
                }
                if !cap.refusal.is_present() {
                    out.push("body.capability.refusal: absent".to_string());
                }
            }
        }
        out
    }
}

impl Arm {
    fn refusals(&self, path: &str, out: &mut Vec<String>) {
        if !self.versions.is_present() {
            out.push(format!("{path}.versions: absent"));
        }
        if !self.threads.is_present() {
            out.push(format!("{path}.threads: absent"));
        }
        if let Some(reason) = self.executed_path.refusal() {
            out.push(format!("{path}.{reason}"));
        }
    }
}

impl SpeedArm {
    fn refusals(&self, path: &str, out: &mut Vec<String>) {
        self.arm.refusals(&format!("{path}.arm"), out);
        if !self.timing.is_present() {
            out.push(format!("{path}.timing: absent"));
        }
        if !self.fixed_cost_s.is_present() {
            out.push(format!("{path}.fixed_cost_s: absent"));
        }
        if !self.per_call_floor_s.is_present() {
            out.push(format!("{path}.per_call_floor_s: absent"));
        }
    }
}

impl SpeedRow {
    /// Our arm, the yardstick, and every context arm, labelled by the path a
    /// refusal message should name.
    fn arms(&self) -> Vec<(String, &SpeedArm)> {
        let mut out = vec![
            ("body.speed.ours".to_string(), &self.ours),
            ("body.speed.competitor".to_string(), &self.competitor),
        ];
        for (name, arm) in &self.context_arms {
            out.push((format!("body.speed.context_arms[{name}]"), arm));
        }
        out
    }
}
