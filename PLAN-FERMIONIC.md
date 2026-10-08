<!-- SPDX-License-Identifier: Apache-2.0 -->
# PLAN — the fermionic OBSERVABLE surface (Track F)

**Status: the OBSERVABLE surface is closed (2026-09-26).** F1, F2, the §1.3
certificate-basis guard and F3 all landed and were independently
mutation-verified.

**Two tracks reopened by owner decision 2026-09-27**, both reversing a
deferral recorded below:

* **F4 — write our own fermionic circuit language, and document it.** §3
  previously deferred FERMIONICQASM on the grounds that the observable surface
  needed a spelling rather than a file format. That was right about the
  observable surface and is no longer the question being asked: the decision is
  that fermionic *circuits* get a first-class text form of our own.
* **F5 — Majorana: start from Stim.** Rather than grow the Majorana lane purely
  in-house.

Split out of the former `PLAN-FERMIONIC-QUDIT.md` on 2026-09-27. Fermionic
observables and qudits were planned together because both were unknowns in
§3c/§3d. They turned out to share nothing: **Track F touched no IR and is
done; Track Q is entirely an IR change and has not started.** Qudits are now
[`PLAN-QUDIT.md`](./PLAN-QUDIT.md).


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

*Findings relevant to this track; the qudit findings moved with it.*
entries went stale this week:

* **The §3d.0 qudit evidence still holds.**
  `grep -riE 'qudit|qutrit|ququart' crates/ --include='*.rs'` returned
  **zero lines** on 2026-09-26 at `21f0e67` — exactly what
  `PLAN-OPEN-20260825.md:1399` recorded. No qudit surface, no `d`, no
  per-register dimension: `CircuitIR` still carries a scalar
  `num_qubits: u32` (`omega-core/src/circuit.rs:256`), and
  `omega-parser` has no `DITQASM`/dimension token anywhere.

* **The fermionic docs are NOT stale — the gap is exactly where §3c.6 says.**
  The fermionic *surface* T1 exists and is not open work:
  `omega_core::fermion` (`FermionicOp`, normal ordering, `jordan_wigner()`,
  `cphase`/`givens`) landed at `8a0e779` with the dense Fock oracle
  (`omega-core/tests/fermion_jw.rs`). `omega-backend-majoranaprop` is built
  — **21/21 tests re-run green on andromeda today** — with P0–P4 closed in
  §3c.6. What remains open is precisely what §3c.6 P1 and
  `PLAN-OPEN-20260825.md:661-663` record: "`Observable::parse` speaks only
  `X/Y/Z`" — verified below at the code, §1.

* **STALE: `omega-backend-cv/src/lib.rs:22-32`.** The module header says
  "Absent: the gates. `Displacement` and `Squeezing` … land next." They
  landed: `displace_fiber` is at `lib.rs:70`, `squeeze_fiber` at
  `lib.rs:142`, `beamsplitter` at `multimode.rs:268`, and the crate's **70
  tests pass on andromeda today** (incl. `squeeze_operator.rs`,
  `beamsplitter.rs`, three piquasso cross-checks). The doc comment was true
  when written and was never updated. Not fixed here (this plan writes no
## 1. The fermionic observable gap, precisely

### 1.1 What can be expressed today

* **In Rust**: everything. `FermionicOp` builds ladder-operator sums
  (`raise`/`lower`/`number`/`hopping`/`interaction`,
  `omega-core/src/fermion.rs:113-151`), `jordan_wigner()` maps them to
  `Observable` and **refuses non-Hermitian input** naming the offending
  coefficient (`fermion.rs:310-334`). This layer is oracle-checked
  exhaustively (all 584 products ≤ length 3 at 4 modes,
  `omega-core/tests/fermion_jw.rs`).
* **On the wire / CLI**: Pauli strings only. `Observable::parse`
  (`omega-core/src/executor.rs:612-641`) accepts `coeff*` + concatenated
  `(X|Y|Z|I)<index>` fragments and nothing else; the help text states the
  whole format in four lines (`omega-cli/src/main.rs:101-105`). Every
  expectation backend — statevector, MPS, pauliprop, majoranaprop, sector —
  consumes `Observable` and only `Observable` (e.g. majoranaprop seeds its
  Majorana sum from Pauli terms, `omega-backend-majoranaprop/src/engine.rs:334-360`;
  the sector backend reads out per-basis-state *Pauli* products,
  `omega-backend-sector/src/sim.rs:322-365`).

### 1.2 What cannot, and what a user tries that fails

1. `omega-run lucj.qasm --backend majoranaprop --expectation "n0"` →
   `no Pauli letter in term 'n0'` (`executor.rs:629`). There is no CLI or
   text spelling for a number operator, a hopping term, a current, or any
   ladder product. The refusal is honest; the capability is absent.
2. So the user Jordan–Wigners **by hand**. `n_0` is manageable
   (`0.5*I0+-0.5*Z0`). Hopping between non-adjacent modes is not: the
   correct string for `a†_1 a_3 + h.c.` is `0.5*X1Z2X3+0.5*Y1Y3` — no,
   it is `0.5*X1Z2X3+0.5*Y1Z2Y3`, and that deliberate error is the point.
   **Omitting the Z-string runs on every backend and returns a plausible
   wrong number with no refusal anywhere** — the exact §3b.2 failure shape,
   and the sign/string trap `fermion_jw.rs` exists to pin *on the library
   side, which the CLI cannot reach*.
3. The pattern already forced one consumer to invent a private wire:
   `omega-bridges/tests/fqe_vs_statevector.rs:95-121` ships fermionic
   observables to the FQE runner as ad-hoc `[mode, dagger]` JSON, built
   from `FermionicOp` in Rust test code. Every future harness or user
   script repeats that invention or hand-JWs.

The gap is therefore one layer thick and sharply bounded: **a text spelling
for `FermionicOp`, a CLI intake for it, and its routing to the backends that
already do the right thing with the mapped result.** No new engine, no new
mapper, no new physics. (Non-adjacent *gates* remain refused by
`givens` — `fermion.rs:434-443` — and stay out of scope: that is the gate
surface, tracked in §3c.0a, not the observable surface. Observables never
needed routing: `jordan_wigner_terms` already emits the Z-strings.)

### 1.3 One real subtlety: which basis the certificate's norm lives in

The vacuity gate in both propagation engines uses the observable's L1
coefficient norm as `observable_range` (`majoranaprop engine.rs:148,292`;
pauliprop equivalent). A fermionic observable has *three* candidate L1
norms — ladder-coefficient, Pauli-basis (post-JW), Majorana-basis — and
they differ (e.g. `n_0`: ladder L1 = 1; JW image `0.5·I − 0.5·Z0` has
Pauli L1 = 1 but range [0, 1], not [−1, 1]). The bound stays a bound in
every case because it is computed in the basis actually propagated, but the
plan must pin *that* — F2 does — or a future refactor computing the range
from ladder coefficients would silently loosen or tighten the gate.

---

## 2. Track F — the fermionic observable surface

Small track, pays immediately, touches no IR. Do it before Track Q.

### F1 — a text spelling for `FermionicOp`

* **Entry condition**: none. All dependencies (`FermionicOp`, the dense
  oracle) are in-tree and green.
* **Deliverable**: `FermionicOp::parse` + `Display`, round-tripping.
  Spelling: adopt **OpenFermion's `FermionOperator` string convention**
  (`c [p^ q]`, `^` marking creation, terms joined by `+`) rather than
  inventing one — §3e.2 already names OpenFermion the de-facto operator
  representation, and the FQE venv (`make fqe-venv`) has `openfermion`
  installed as a dependency, so the convention can be pinned against the
  library that owns it, not against our reading of its docs
  (VERIFY-BEFORE-IMPLEMENTING, §3e.1a's rule). No named shorthands
  (`hop(p,q)` etc.) in this phase: the raw spelling covers everything and
  sugar is scope.
* **Test, and how it fails if the feature were absent**: (i) parse →
  `jordan_wigner()` → dense Fock matrix equality, as a third leg of
  `fermion_jw.rs`'s exhaustive small-`N` sweep — fails on any sign,
  ordering or Z-string error in the parser because the dense oracle never
  sees the parser; (ii) round-trip `parse(display(op)) == op` under normal
  ordering; (iii) a **cross-library pin**: `openfermion.FermionOperator`
  fed the same strings through the FQE venv, term-for-term coefficient
  agreement (registered skip when the venv is absent, per the `ci.sh`
  discipline). Absent the feature, (i)–(iii) do not compile/run at all —
  and to make presence non-trivial, (i) includes the deliberate
  wrong-string case from §1.2: the parser's output for non-adjacent
  hopping must *differ* from the hand-JW-without-Z-string observable by a
  measured amount on a reference state. A parser that "worked" by
  delegating to `Observable::parse` fails that case.
* **Certificate story**: this phase produces no numbers; its honesty
  obligation is refusal. Non-Hermitian input keeps the existing
  `jordan_wigner` refusal naming the offending coefficient
  (`fermion.rs:322-329`); malformed text refuses naming the token. Nothing
  is discarded, so there is nothing to bound — and the test that the
  parse is *exact* (leg i) is what entitles later phases to say the
  observable intake added zero to any `dropped_mass`.
* **Estimate**: 2–4 focused days. andromeda; no special hardware.
* **Landed 2026-09-25** (akilles, `STATUS.md` §4c): `FermionicOp::parse` /
  `Display` / `FromStr` in `crates/omega-core/src/fermion.rs`; legs (i)
  and (i′, the §1.2 wrong-string case — Δ⟨H⟩ = 2 on the reference state)
  in `tests/fermion_jw.rs`, legs (ii) and (iii) in
  `tests/fermion_spelling.rs`, (iii) wired into `ci.sh` as a registered
  skip without an openfermion Python. Mutation-checked: flipping `^`
  detection reddens 5 tests across both suites.

### F2 — CLI and JSON intake, routed to every expectation backend

* **Entry condition**: F1 landed.
* **Deliverable**: `--expectation-fermionic OBS` on `omega-run`
  (mutually exclusive with `--expectation`, refusal message naming both —
  extending the existing exclusive-flag table at
  `omega-cli/src/main.rs:1293`), lowering through
  `FermionicOp::parse` → `jordan_wigner()` → the existing `Observable`
  dispatch, so statevector, MPS, sector, pauliprop and majoranaprop all
  work on day one with **zero engine changes**. `--format json` gains an
  `observable` block: `{"input_basis": "ladder", "mapping":
  "jordan_wigner", "mapped_terms": N}` so a reader can re-derive what was
  measured.
* **Test, and how it fails if the feature were absent**: (i) CLI smoke:
  `omega-run --backend majoranaprop --expectation-fermionic "1 [1^ 3] + 1 [3^ 1]"`
  on a 4-mode Givens circuit, value vs statevector at 1e-10 — with the
  flag absent, `omega-run` exits non-zero on the unknown flag and the test
  fails at exit code, not at tolerance; (ii) the same fermionic string
  through **all five** expectation backends, pairwise agreement (the
  sector backend is the interesting arm: fermionic-native representation,
  Pauli readout); (iii) the constant-term case: `1 [0^ 0]` (i.e. `n_0`,
  whose JW image carries a `0.5·I` identity term) through every backend —
  pins that identity terms survive each dispatch path identically;
  (iv) extend `fqe_vs_statevector.rs` to build its ladder-JSON *from the
  F1 text spelling* rather than from Rust constructors, closing the loop
  §1.2(3) identified: text → our JW → statevector vs text → FQE's own FCI
  basis, JW on our side only.
* **Certificate story**: the mapping is exact, so every backend's existing
  certificate applies **unchanged** — majoranaprop's `dropped_mass` bound
  survives because JW images of Hermitian ladder sums seed as real
  combinations of Hermitian monomials with `|⟨M⟩| ≤ 1` (`lib.rs:16-18`),
  which is the entire derivation. The one new pin (§1.3): a test asserting
  `observable_range` equals the L1 norm **in the propagated basis**
  (post-mapping), by constructing an observable whose ladder-L1 and
  Pauli-L1 differ and checking the vacuity gate fires at the propagated
  number. That test fails under the plausible future refactor that
  computes the range early.
* **Estimate**: 2–3 focused days. andromeda; FQE leg needs the venv
  already built there (§3c.0e).

### F3 — direct Majorana seeding in majoranaprop

* **Entry condition**: F2 landed (so there is a via-JW path to differ
  against and a user-visible reason: term-count).
* **Deliverable**: `MajoranaPropBackend` accepts a fermionic observable
  and seeds `MajoranaSum` directly via `a_p = (γ_{2p} + iγ_{2p+1})/2`,
  bypassing the ladder → Pauli → `MajoranaKey::from_pauli`
  (`engine.rs:357`) detour. This is the natural basis for the engine
  (its own doc says so, `lib.rs:9-15`); a `k`-factor ladder product
  expands to `2^k` Pauli strings before `from_pauli` re-condenses them,
  and the direct seed is linear in terms.
* **Test, and how it fails if the feature were absent**: (i) equality of
  value **and bit-identical certificate** (`dropped_mass`, `final_terms`,
  `peak_terms`) between direct seeding and the F2 via-JW path across the
  `fermion_jw.rs` product sweep — the two paths must reach the *same*
  monomial seeds, so any divergence is a bug in one of them; (ii) the
  dense `N ≤ 6` oracle as the tie-breaker leg, so (i) cannot pass by
  matching errors; (iii) the non-triviality pin: on a non-adjacent hopping
  observable, assert the direct path's *intermediate* seed-expansion count
  is strictly below the Pauli path's — a "direct" implementation that
  secretly routes through JW fails this count, which is the A10 point:
  the test says whether the claimed path actually ran.
* **Certificate story**: unchanged by construction, and test (i) is the
  proof — seeding is exact in both paths, so certificates must agree
  exactly, and a certificate delta IS the failure signal. Nothing new is
  discarded; `dropped_mass` keeps its meaning and its
  `dropped_mass_is_a_bound` contract.
* **Estimate**: 2–3 focused days. andromeda.

---

---

---

## 3. Open tracks (owner decision 2026-09-27)

### F4 — FermionicQASM 1.0: our own circuit language, documented

**Decision: invent it.** Previously deferred as "coverage, not need". The
groundwork is already done and should not be redone —
`PLAN-OPEN-20260825.md` §3e.2a searched before drafting and established:

* **No fermionic QASM dialect exists.** Not in OpenQASM 2 or 3, not cQASM,
  not Jaqal. The single candidate, `qiskit_cold_atom.fermions.FermiHubbard
  .qasm()`, is a *generic inherited* `Instruction.qasm()` with no fermionic
  override — deprecated in qiskit-terra 0.25.0 and removed in Qiskit 1.0.
  Not a dialect either way.
* **The name `FermionicQASM` has no prior art.**
* **OPTICQASM is in-house.** The bosonic precedent is "we wrote one", not
  "we adopted one" — so writing a fermionic one follows precedent rather
  than departing from it. **Neither carries the authority of a standard,
  and both documents must say so out loud.**

So: **invent the syntax, adopt every established convention** — qiskit-cold-
atom's mode/spin conventions, ffsim's gate set, FCIDUMP for the Hamiltonian.
Mirror `OPTICQASM`'s structure (`header = { "FERMIONICQASM" ~ version ~ ";" }`)
and add `CircuitType::Fermionic` to the IR beside `CircuitType::Photonic`.

**Deliverables:** a `.pest` grammar in `omega-parser` beside `opticqasm.pest`,
lowering into the IR, **and the language documentation** — the owner asked for
the document explicitly, so it is a deliverable, not a by-product. It must
state what was adopted, what was invented, and that this is an in-house format.

**Test, and how it fails if absent:** round-trip fixtures as in Q0 — but there
is no reference implementation to generate them, which is exactly the
writer-validated-by-its-own-reader trap §3e.2a warns about. So the oracle must
be external: a FermionicQASM circuit and its hand-written QASM2 + `rbs`
equivalent must produce the same expectation through the same backend, and the
FCIDUMP path must agree with `FermionicOp` built in Rust.

### F5 — Majorana: start from Stim

**Decision: Stim is the starting point** (`quantumlib/Stim`, **Apache-2.0**,
verified 2026-09-27 — permissive, so it may be read and built on, unlike the
three copyleft qudit oracles). It is already in this tree as an *oracle*:
`FIXES_PLAN.md:2643` uses Stim's `peek_observable_expectation` as the
independent implementation for the `pauli` stabilizer backend, and it is
pinned at 1.16.0 in the QEC cross-check venv (`PREREQUISITES.md:43`).

The step being asked for is different: Stim as a **foundation** for the
Majorana lane rather than only as a check on it — **and explicitly with the
non-Clifford extensions, T gates included** (owner, 2026-09-27).

That second half is the hard half, and it must be said plainly: **Stim is a
Clifford simulator and cannot execute a T gate at all.** Its speed comes from
the stabilizer formalism, which is closed under Clifford operations and not
under T. So "Stim + extensions" is not a configuration flag; it is a decision
about *which* extension, and each one buys universality at a different price:

* **Stabilizer decomposition / stabilizer rank** (Bravyi–Gosset and successors)
  — write the magic state as a sum of χ stabilizer states and run Stim on each.
  Exact at full rank; the cost is exponential in the T count, ~2^{0.23 t}.
  **This is the one that fits this project's contract**, because truncating χ
  gives a bound on what was discarded, and a bound is what a certificate is
  required to be. `dropped_mass`/`discarded_weight` have an obvious analogue.
* **Magic-state injection with gadgets** — universality via distillation, which
  is a QEC-layer answer and overlaps §19 work already done in the sibling repo.
* **Approximate stabilizer sampling** — cheap and gives no bound, so by the
  standing contract it does not ship: a phase that cannot say what it discarded
  is excluded, the same rule that keeps qudit Weyl propagation out of
  PLAN-QUDIT.md.

So F5 has two gates before code: pick the extension (the stabilizer-rank route
is the recommendation, with the reason above), and settle how its truncation
reports a bound. Note the Majorana lane's own structure may help here —
`majoranaprop` already propagates a Majorana sum with a `dropped_mass` bound
and a vacuity gate, so the question is whether stabilizer rank plugs into that
accounting or needs its own. Before any code, establish
and write down which of the two it is to be — vendored/linked dependency, or
a design borrowed and reimplemented — because Apache-2.0 permits both and the
choice changes the build, the packaging and what "independent oracle" then
means. **If Stim becomes the implementation, it stops being an independent
oracle**, and something else has to play that role for the Majorana lane;
`majoranaprop` already has a dense-statevector oracle (F3's
`direct_seed_agrees_with_the_statevector_oracle`), which is the candidate.

## 4. What this track deliberately did NOT do

* **Non-adjacent Givens routing.** A gate-surface gap, correctly refused
  today with the route named in the message (`fermion.rs:434-443`), and
  §3c.0a owns it. Observables never had the problem. **This is the one
  fermionic thing still open** — it is a gate-surface item, not an
  observable one, and it is not tracked here.
* **Bravyi–Kitaev / parity mappers.** JW is the only mapping any in-tree
  backend consumes; F2's JSON records `"mapping": "jordan_wigner"`
  precisely so adding a second mapper later is an enum value, not a
  migration. Adding BK now would be a mapper with no consumer and no
  oracle pressure.
* **Sector-backend native ladder readout.** Its Pauli readout via JW is
  exact today (`sector/sim.rs:322-365`); a ladder-native readout changes
  no answer and adds no certificate — pure optimisation, deferred until a
  profile says otherwise.

---

## 5. What the closed work cost

| phase | estimated | box | outcome |
|---|---|---|---|
| F1 parse/display | 2–4 d | andromeda | landed `33deb56` |
| F2 CLI + routing | 2–3 d | andromeda | landed `42c663b`; §1.3 guard `6ff06e1` |
| F3 direct seeding | 2–3 d | andromeda | landed `238eb90` |

Estimated 6–10 focused days. The estimate held: the shape of each change was
visible in the code cited, which is what made the F-track numbers firm while
Q3's remain a range.
