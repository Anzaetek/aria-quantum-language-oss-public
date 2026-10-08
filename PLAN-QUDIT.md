<!-- SPDX-License-Identifier: Apache-2.0 -->
# PLAN — QUDITS (Track Q)

**Status: Q0–Q4 landed (Q3.2 and Q4 on 2026-09-30, STATUS §4i, §4j). Track Q is complete; what it deliberately does not do is §3.**
The paragraph below is Q0's original framing, kept as written. Q0 was verification only
— DITQASM semantics read out of MQT's parser source, eight fixtures written by
`mqt.qudits` itself, and the oracle licences. **There is still no qudit code
in the tree**: `CircuitIR` carries a scalar `num_qubits: u32`, and
`omega-parser/src/` has no DITQASM grammar.

Split out of the former `PLAN-FERMIONIC-QUDIT.md` on 2026-09-27. It was
planned alongside the fermionic observable surface because both were §3c/§3d
unknowns; they share nothing in the end. Track F is closed and lives in
[`PLAN-FERMIONIC.md`](./PLAN-FERMIONIC.md).


Written 2026-09-26 against HEAD `21f0e67`, on **andromeda** (Apple M4, 16 GB,
darwin/arm64). Every command quoted below was re-run on this machine on this
date; every code citation was read at this HEAD, not recalled from the plan
documents. The estate is **andromeda** and **akilles** (linux x86_64, 123 GB,
RTX PRO 6000, CUDA 12.9). **There is no GB10.** Phases that need a box name
it.

This plan covers two tracks that §3c and §3d of `PLAN-OPEN-20260825.md` left
open, in §3c's own shape: tiers stated, oracles split from algorithm
references, each phase with an entry condition, a deliverable, the test that
proves it — including how that test would FAIL if the feature were absent
(§3f A10, `PLAN-OPEN-20260825.md:2537`: a test that cannot fail is not
evidence) — and the certificate story. This project treats truncation
certificates as **bounds, not estimates** (`omega-backend-pauliprop/src/sim.rs:184-186`:
`dropped_mass` is "a genuine bound rather than an estimate, since `|⟨P⟩| ≤ 1`";
tested as such in `omega-cli/tests/dropped_mass_is_a_bound.rs`;
`omega-backend-majoranaprop` shares the contract, `src/lib.rs:5-8`). A phase
that cannot say what it discarded is listed under NOT DOING, not under a
softer heading.

---

## 0. Evidence re-run, and what turned out stale

*Findings relevant to this track; the fermionic ones moved with it.*
entries went stale this week:

* **The §3d.0 qudit evidence still holds.**
  `grep -riE 'qudit|qutrit|ququart' crates/ --include='*.rs'` returned
  **zero lines** on 2026-09-26 at `21f0e67` — exactly what
  `PLAN-OPEN-20260825.md:1399` recorded. No qudit surface, no `d`, no
  per-register dimension: `CircuitIR` still carries a scalar
  `num_qubits: u32` (`omega-core/src/circuit.rs:256`), and
  `omega-parser` has no `DITQASM`/dimension token anywhere.
  code), but any reader of that header should know it describes a previous
  state of the crate.

* **STALE GATE: `FIXES_PLAN.md:1582-1583`.** "B1–B6 are gated on the DGX
  search answering PR-AUC in CV's favour." Two problems: `omega-backend-cv`
  (B1) exists and passes its anchors regardless of that gate, and the DGX
  is gone (`STATUS.md:291-292`: "That GB10 is gone; the estate is now
  akilles … and andromeda"). Likewise `FIXES_PLAN.md:1308`'s platform list
  "CUDA (discrete + GB10/GH200)" names hardware the estate does not have.
  Consequence for this plan: any option inherited from FIXES_PLAN Part F
  must be re-decided on the current estate, not adopted with its dead gate
  attached. §2 does this for the `apply_1q`/`apply_2q` qudit-generalisation
  option (`FIXES_PLAN.md:1301-1304`).

* **Load-bearing and current: `STATUS.md:845-849`.** The MPS-core SVD work
  is already directed to be "shaped for the lanes that will share the MPS
  core: the Majorana/fermionic track (§3c) and the qudit track (§3d, where
  the physical index is `d > 2`, the two-site gate is `d² × d²`, not `4×4`,
  and the theta block handed to the SVD is `(χ_l·d) × (d·χ_r)`)." The qudit
  track below (Q3) is the consumer that sentence anticipates. (The STATUS
  sentence originally called the theta block `d² × d²`; that is the gate's
  shape, and both texts now say so — the theta block is `(χ_l·d) × (d·χ_r)`,
  exactly what `mps.rs` builds as `bl × (d0·d1) × br` and reshapes to
  `(bl·e0) × (e1·br)` for the SVD.)

---

## 1. Qudits: the position

Three options were on the table: (a) generalise `omega-backend-statevector`
to `d`, (b) reuse `omega-backend-cv` (its state is technically
`dim = cutoff^n_modes`), (c) IR surface first, then dedicated qudit engines.

**Position: (c), with MPS as the scale lane and a small dedicated dense
mixed-radix crate as the oracle lane. Not (a) first, and (b) not at all.**

* **Against (a) as the first move**: §3d.1's *measured* blast radius —
  71 occurrences of `1 <<` addressing in `statevector/src/sim.rs` alone,
  genuine rewrite, before Metal/CUDA/OpenCL mirrors. The obvious path is
  the expensive one, and §3d.1's conclusion ("MPS is the cheaper path to
  qudits, not statevector") survived its own recount.
* **Against (b)**: `omega-backend-cv` is *bosonic*, and its entire safety
  design is about the cutoff being a lie: `PAD = 96` headroom per mode
  (`lib.rs:56`), truncated displacement/squeezing that are **not unitary**
  (`FIXES_PLAN.md:1565-1567`), per-state norm-leak accounting with
  refusal-on-leak (`lib.rs:19-20`). A fixed-`d` qudit has none of that:
  gates are exactly unitary `d×d` matrices, there is no ladder above
  level `d−1`, and the leak certificate is identically zero. Running
  qudits through CV means carrying a certificate that can never fire and a
  PAD that means nothing — a certificate field whose meaning depends on
  which caller you are.

**The FIXES_PLAN:1579 objection, taken head-on rather than around.** As
written (`FIXES_PLAN.md:1577-1580`) it argues against folding the CV
statevector into `omega-backend-photonics` because that "would blur two
different state representations behind one name." The principle underneath
is: **one crate name must mean one state model and one certificate
semantics.** That principle *decides* this section, in both directions:

* It kills option (b): CV-as-qudit-backend blurs "bounded-`d` unitary
  evolution, exact by construction" with "cutoff-truncated bosonic
  evolution, honest only via leak accounting" behind one name — the same
  defect one level up.
* It also disciplines `FIXES_PLAN.md:1301-1304` ("the work is
  qudit-generalising `apply_1q`/`apply_2q` to `d = cutoff` rather than new
  physics. Reuse before rewrite."), which pulls the other way. Read
  precisely, that line is about **kernels**, not state models: applying a
  `d×d` matrix along a strided axis of a dense array is arithmetic with no
  certificate semantics in it, and sharing *that* between a qudit engine
  and a future CV-GPU path violates nothing — `truncated_svd_flat` already
  plays exactly this role for MPS (`svd.rs:87`, dimension-generic,
  verified so in §3d.1). The two FIXES_PLAN lines are reconcilable:
  **reuse at the kernel level, never at the state-model level.** And note
  the 1303 item's own gate (the DGX search) is stale per §0 — it is
  re-decided here on merit, not inherited.

So the shape is: T1 IR surface with an all-dims-are-2 admission check in
every existing backend (Q1), a laptop-sized dense mixed-radix oracle crate
in the mould of `omega-backend-sector` — the repo's precedent for "small
dedicated exact engine built to be the reference" (Q2) — then the MPS
`phys: d` generalisation as the lane that scales (Q3), which is also the
lane `STATUS.md:845-849` already tells the SVD rework to leave room for.
DITQASM is adopted, not invented, per the standing §3e.1a decision — after
its VERIFY-BEFORE-IMPLEMENTING items are discharged (Q0), because the
syntax in §3e.1a is transcribed from tutorial prose and the
`RBS → XXPlusYYGate` episode (§3c.0d) is the standing receipt for what an
unverified transcription costs.

---

## 2. Track Q — qudits

### Q0 — verify DITQASM and the oracle licences (the §3e.1a debt)

* **Entry condition**: none; parallel to Track F.
* **Deliverable**: the four §3e.1a unknowns
  (`PLAN-OPEN-20260825.md:1623-1633`) answered **from MQT Qudits' actual
  grammar/parser source**, not its tutorial: bare `qreg matter [2];`
  semantics; whether `h`/`cx` on a qudit are generalised or embedded
  gates; `ctl` level-list ordering; version-lock semantics. Plus committed
  round-trip fixture files *generated by* `mqt.qudits` itself, and the
  licence check §3d.3 records as NOT done for QuickQudits, QuForge,
  QuDiet, Sdim (MQT is verified MIT; the §3c all-Apache result was luck,
  not a rule).
* **Test, and how it fails if the feature were absent**: the fixtures are
  the test — bytes emitted by `mqt.qudits`, committed, with a venv-gated
  harness that re-parses them under `mqt.qudits` and (from Q1 on) under
  our parser. No venv → **registered skip** via the `ci.sh` `skipped`
  helper, never a silent pass (§3f A8 discipline). "Absent" here means
  unverified transcription: any Q1 grammar written before Q0 lands would
  be tested only against our own reading — the exact writer-validated-by-
  its-own-reader defect the QPY lane hit (`PLAN-OPEN-20260825.md:934-943`).
* **Certificate story**: none — no numbers. The honesty artifact is the
  fixture provenance (generated by the reference implementation, not by
  us).
* **Estimate**: 1–2 days, **gated on things outside the plan**: PyPI
  access through the pxpipe proxy (the `PIP_CERT` precedent exists,
  §3c.0e) and on upstream licence facts being acceptable.
* **Landed 2026-09-26** (akilles, `STATUS.md` §4e): the four unknowns
  answered from `qasm.py` and recorded at `PLAN-OPEN-20260825.md` §3e.1a;
  eight fixtures under `crates/omega-parser/tests/fixtures/ditqasm/` written
  by `mqt.qudits` 0.5.2 (`gen_fixtures.py`, `PROVENANCE.json`); harness
  `tools/ditqasm_xcheck/reparse.py` in `ci.sh` as a registered skip, venv
  target `make -C crates/omega-bridges/python mqt-venv`. Licences: three of
  the five were **copyleft** (QuDiet AGPL, Sdim GPL, qiskit-qudits AGPL) —
  black-box oracles only; QuickQudits and QuForge Apache-2.0. Two upstream
  defects pinned (the `ctl` writer, `rxy`/`rz` parameter loss on parse).
  Consequence for Q1: `cx` in DITQASM is an embedded two-level gate, the
  generalised one is `csum`.

### Q1 — T1 surface: per-register dimension, and every backend refuses it

* **Entry condition**: Q0's grammar answers (the `qreg` rule cannot be
  written before its bare-form semantics are known).
* **Deliverable**: `CircuitIR` carries a per-wire dimension (all-2
  default so every existing constructor is untouched); the parser accepts
  the DITQASM subset that costs little by §3e.1a's own cost analysis
  (header token + dimensioned `qreg` — one optional pest group on
  `qasm2.pest`'s `qreg_decl`); **every existing backend refuses any
  circuit with a dimension ≠ 2, naming the dimension and the register**;
  and the QASM2/QPY **emitters refuse** to serialize a `d > 2` circuit
  (a qudit circuit silently exported as qubit QASM2 is the same defect as
  running it). `rxy` and level-controls are **not** in Q1 — §3e.1a costs
  them separately and nothing in Q2 needs them parsed before the engine
  exists.
* **Test, and how it fails if the feature were absent**: one refusal test
  **per backend** (statevector, MPS, pauli, pauliprop, majoranaprop,
  sector, photonics dispatch, plus the three GPU statevector arms' CPU-side
  admission): parse a qutrit register, run, assert non-zero exit and a
  message naming dimension 3. This is §3d.2's "do the assertion first"
  made mechanical, and it is exactly an A10-shaped test: **if the
  admission check is absent, the d=3 circuit runs as a qubit circuit,
  exits 0 with a plausible number, and the test fails on exit code.** The
  parser side: Q0's committed fixtures must parse; a hand-rolled dialect
  diverging from MQT's grammar fails against fixtures we did not write.
* **Certificate story**: no numbers; the certificate IS the refusal. After
  Q1, "this repo does not support qudits" is enforced by every engine
  rather than implied by a grep.
* **Estimate**: 4–8 focused days — the IR touch is small but it ripples
  through every backend's admission and the emitters, and the test matrix
  is wide. andromeda (the GPU arms' admission checks are CPU-side code;
  the CUDA arm's refusal test *runs* on akilles in CI or not at all — name
  it in the test's skip note).
* **Landed 2026-09-27** (akilles, `STATUS.md` §4f). `CircuitIR::qudit_registers`
  + `wire_dim`/`first_qudit`/`refuse_qudits` (`omega-core/src/circuit.rs`);
  `DITQASM` header token and the `qreg` dimension group in `qasm2.pest`,
  refused under `OPENQASM` and on length/`d < 2` mismatch, naming the
  register; `refuse_qudits` at the top of **every** `&CircuitIR` door on
  every `Backend` impl (52 doors across 11 engines incl. the three GPU arms
  and `TchBackend`), both certificate doors, majoranaprop's fermionic door,
  the plugin ABI, the QPY writer and the `aria-verify` oracle. Tests:
  `omega-parser/tests/ditqasm_registers.rs` against Q0's `mqt.qudits`-written
  bytes; `omega-cli/tests/qudits_are_refused_by_every_backend.rs` (10 rows,
  every backend × every mode, non-zero exit AND the engine sentence, with a
  `[2][2,2]` control that must run); `omega-core/tests/qudit_refusal.rs`.
  **CUDA row run for real on akilles** (RTX PRO 6000): refused before
  `allocate`, forwarded through `refuse_device_fallback`. Mutation: one
  door removed reddens 4 rows. Caveat honoured from the plan text: there
  is no in-tree QASM2 *text* emitter for `CircuitIR`, so that clause had
  nothing to attach to; QPY is the serializer and it refuses. `rxy`,
  `csum`, `ctl` and the four-param `cx` stay outside Q1 as costed.

### Q2 — the dense mixed-radix oracle crate

* **Entry condition**: Q1 landed (there is an IR to read dimensions from).
* **Deliverable**: a new small crate (`omega-backend-quditsv`), dense
  mixed-radix statevector, deliberately laptop-sized: capacity ceiling on
  `Π dᵢ` with refusal **before allocation** (the `omega-backend-cv`
  `capacity.rs` precedent), no GPU, no gradients, no noise. Gates:
  generalised shift `X_d`, clock `Z_d`, Fourier `F_d`, two-level
  `rxy(i, j, θ, φ)`, level-controlled application; `d = 2` gates embed.
  Role: the in-tree exact reference for Q3, same role
  `omega-backend-sector` plays for particle-conserving circuits — the
  repo's precedent that a small dedicated exact engine beats 71 rewrite
  sites in a load-bearing one (§3d.1).
* **Test, and how it fails if the feature were absent**: three legs, per
  §3c.4's ladder. (i) **Analytic anchors that a qubit engine cannot fake**:
  `F_3|0⟩` uniform over three levels, `X_3` cycling `|2⟩ → |0⟩`,
  `rxy(0,1)` acting as identity on `|2⟩` — every one requires amplitude
  at level 2 to exist, so any d=2 delegation or mixed-radix addressing
  error fails on the value, not on plumbing. (ii) `d = 2` differential:
  random qubit circuits vs `omega-backend-statevector` at 1e-12 — catches
  stride errors where they are cheapest to see. Note (ii) alone would be
  the A10 trap (it passes under secret delegation); (i) is what makes the
  suite falsifiable. (iii) external oracle: QuickQudits, whose authors
  describe it as "intended as a reference implementation for small
  instances" (§3d.3) — venv-gated, registered skip, and **gated on Q0's
  licence check**; until that clears, (iii) is a lead, not a leg.
* **Certificate story**: exact engine — the certificate is
  `exact: true` and the bound is **zero**, which is only honest because
  the capacity gate refuses anything it cannot hold exactly (mixed-radix
  `Π dᵢ` over ~2²⁴ amplitudes refuses with the product and the ceiling in
  the message, before allocating). Nothing is ever discarded, so there is
  nothing to bound; a leaking-state situation cannot arise by
  construction, which is precisely the state-model difference from CV
  that §2 keeps separate.
* **Estimate**: 5–10 focused days. andromeda.

### Q3 — MPS with `phys = d`

* **Entry condition**: Q1 (dimensions in the IR) **and** Q2 (an exact
  oracle to differ against). Do not start before both — §3c.6's "building
  the engine first means building it twice" applies verbatim.
* **Deliverable**: `MpsTensor` gains the explicit `phys` field §3d.1
  notes it lacks; per-site `d` (mixed dimensions are the normal case, per
  DITQASM); gate application on `d²×d²` θ blocks (the fixed-size
  `[Complex64; 16]`/`[Complex64; 4]` signatures become slices — §3d.1's
  "API change, not a literal edit"); readout/sampling off 2-adic
  addressing (`mps/sim.rs:1493`, `mps.rs:531`). The SVD core is already
  dimension-generic (`truncated_svd_flat`, `svd.rs:87` — verified in
  §3d.1) and is the piece `STATUS.md:845-849` has already reserved the
  shape for. **CPU only.** The Metal/CUDA mirrors (23 hardcode sites in
  `mps-metal/contract.rs` alone) are out — §5.
* **Test, and how it fails if the feature were absent**: (i) qutrit random
  circuits, exact regime (bond unbounded at small `n`), vs Q2 at 1e-10 —
  cannot run at all without the generalisation because Q1 made the
  qubit-MPS refuse `d = 3`, which is the A10 property: **the baseline
  behaviour is a refusal, so a pass requires the new path to have
  executed**; (ii) truncating runs: certified discarded weight vs the
  *measured* error against Q2, asserting the contract direction
  (measured ≤ what the certificate implies) on a sweep of bonds — this can
  fail, and fails specifically if the `d²` block bookkeeping feeds the
  certificate wrong-sized discarded weight; (iii) the existing qubit MPS
  parity suite unchanged and green, so `d = 2` regression is pinned;
  (iv) mutation check in the style of `pauliprop-cuda/tests/parity.rs`:
  perturb the discarded-weight accumulation, assert the gate trips.
* **Certificate story**: the qubit MPS lane's contract transfers —
  accumulated relative discarded singular-value weight, gated against the
  1e-6-class ceiling with refusal (the §3b.2 R1 machinery, exercised at
  28 modes in §3c.6 P4 where `mps:64` *refused* rather than estimated).
  The transfer is legitimate because the SVD/Frobenius argument never
  mentions the physical dimension — but Q3 states that as a claim and
  pins it with test (ii) rather than asserting it, and keeps the repo's
  existing honesty split: `discarded_weight` is the certified quantity,
  `fidelity_estimate` stays **labelled an estimate** (STATUS §1) for
  `d > 2` exactly as for `d = 2`. What Q3 must NOT do is convert the
  weight into an expectation-error bound without a derivation — the qubit
  lane refuses instead of interpreting, and the qudit lane inherits the
  refusal posture, not an interpretation.
* **Estimate**: 10–20 focused days — the largest item in this plan; the
  type-signature ripple through `mps.rs`/`sim.rs` and the test estate is
  the §3d.1 "integer factor" that was understated once already. andromeda
  for development; nothing here needs a GPU.

### Q4 — measure it, on named boxes

* **Entry condition**: Q3 landed.
* **Deliverable**: a comparison in the §3c.6 P4 shape: qutrit chain
  dynamics (the natural first workload; MQT Qudits' tensor-network
  backend as the external comparison point, venv-gated), small shapes on
  **andromeda**, RAM-bound large-bond rows on **akilles** (123 GB).
  Wall time and peak RSS together (`FIXES_PLAN.md:1280` discipline), and
  every number stated with its box.
* **Test, and how it fails if the feature were absent**: the harness
  follows the A10 prescribed form (`PLAN-OPEN-20260825.md:2579-2584`):
  it asserts its precondition — per-site `d > 2` actually present in the
  executed circuit, certificate fields present in every row — and
  **fails, not reports**, when a row would measure something other than
  what its label says. The three 2026-09-25 receipts (CPU-as-GPU speedup,
  silent per-call fallback, `to_statevector` dominating an "MPS" profile)
  are the checklist; in particular Q4 measures through `expectation`, not
  `execute(shots: None)`, because STATUS records that trap explicitly.
* **Certificate story**: no new certificate; Q4 *consumes* Q3's — every
  published row carries its discarded weight or `exact`, and a row that
  refused is published as a refusal (the §3c.6 "honest table" precedent,
  where "no certified number on either side" was itself the result).
* **Estimate**: 2–4 focused days, **gated on akilles availability** for
  the large rows and on the MQT venv (network/proxy, as Q0).

---

---

## 3. What this plan does NOT do, and why

* **Qudit Pauli propagation (Weyl–Heisenberg) and qudit stabiliser — the
  whole of §3d.2 T3.** The branch machinery is a redesign, not a
  re-parameterisation (`anticommutes_with` is mod-2 symplectic; Weyl
  operators ω-commute and split into up to `d` children —
  `PLAN-OPEN-20260825.md:1487-1497`), and qudit Weyl expectations are
  **complex**, so the certificate's `observable_range` realness assumption
  dies (`:1498-1500`). Until someone re-derives a bound for that setting,
  this path **cannot say what it discarded** — and by this project's
  contract that excludes it, full stop. It returns to a plan only with the
  derivation in hand.
* **Reusing `omega-backend-cv` for qudits, or folding either into the
  qubit statevector.** §1's position: one name, one state model, one
  certificate semantics. Kernel-level sharing stays available later
  without violating that.
* **GPU qudit backends (Metal/CUDA MPS mirrors, statevector arms).**
  23 hardcode sites in `mps-metal/contract.rs` before any physics; the
  three A10 receipts are all GPU-benchmark receipts; and each device
  backend carries the FIXES_PLAN F3 obligations (parity gates, governor
  pricing, loud `Unsupported`, verified on the hardware it claims —
  `f11a9f5` is the counterexample receipt). CPU correctness with
  certificates first; GPU only with a workload that demands it, measured
  on akilles or andromeda by name.
* **FERMIONICQASM (the circuit file format).** §3e.2a's draft grammar

---

## 4. Cost, honestly, and the outside gates

| phase | estimate (focused days) | box | gated on outside-the-plan |
|---|---|---|---|
| Q0 DITQASM verify + licences | 1–2 | andromeda | **DONE** — PyPI-via-proxy; licences resolved |
| Q1 IR dims + universal refusal | 4–8 | andromeda (CUDA-arm refusal test verifiable only on akilles) | **DONE** — STATUS §4f; CUDA arm run for real on akilles |
| Q2 dense oracle crate | 5–10 | andromeda | **DONE** — STATUS §4g; both oracle legs ran (mqt.qudits, QuickQudits) |
| Q3 MPS `phys = d` | 10–20 | andromeda | **DONE** — STATUS §4h (Q3.1 core), §4i (Q3.2 backend door); tests (i)–(iv) all in `tests/qudit_backend.rs`; sampling into `Counts` and `NoisyMpsBackend` stated as not in Q3 |
| Q4 measurement | 2–4 | andromeda + **akilles** | **DONE** — STATUS §4j; `examples/qudit_chain_profile.rs` + `tools/qudit_chain_xcheck/tnsim_rows.py`; small profile in `ci.sh` on every box, large rows by hand on akilles; MQT venv built on both boxes |

Q1 → Q2 → Q3 → Q4 strictly; each entry condition above is a hard gate. Q3 is
the only item big enough to be worth abandoning mid-plan.

Remaining: **nothing** — Q0–Q4 landed between 2026-09-26 and 2026-09-30,
five calendar days against a 22–46 focused-day estimate. Q1 and Q2 took two
days on akilles against 9–18; Q3 (10–20) took three; Q4 (2–4) one. The
spreads were admission-test width, not engine width, and §3d.1's "integer
factor" warning for Q3 did not materialise because the SVD core was already
dimension-generic and Q2's oracle existed before the first MPS line was
written — the §3c.6 "build the engine first means building it twice" cost
was paid in the right order. What the plan got wrong is recorded in
STATUS §4j: the large rows are not RAM-bound on the MPS side.
