<!-- SPDX-License-Identifier: Apache-2.0 -->
# External simulator bridges

`omega-bridges` reaches simulators that live outside this repository, over a
JSON-over-stdio subprocess protocol: stdin `{"qasm","shots","seed","noise"}` →
stdout `{"ok","counts"}`.

Each bridge is a Cargo feature plus a runner script. A missing install is
reported as `Unavailable`, never a hard error, so the default `./ci.sh` stays
green on a machine that has none of them.

| backend | feature | what it is | status |
|---|---|---|---|
| Qiskit | `bridge-qiskit` | reference SDK | implemented |
| Perceval | `bridge-perceval` | Quandela photonics | implemented |
| Bloqade | `bridge-bloqade` | QuEra Aquila (gate mode; analog AHS returns `Unavailable`) | implemented |
| ppvm | `bridge-ppvm` | QuEra Pauli-sum propagation | implemented (runner + QASM2→Stim lowering) |
| tsim | `bridge-tsim` | QuEra ZX stabilizer-rank sampler | implemented (runner + QASM2→Stim lowering) |
| Cirq / Qadence | `bridge-cirq`, `bridge-qadence` | — | placeholder |

## What each backend actually is, and what that costs

The bridges are not interchangeable simulators with different speeds. They use
different representations, and which circuits are cheap, expensive or
impossible follows from that — so it is written down here rather than
rediscovered per fixture.

| bridge | representation | non-Clifford | cost driver |
|---|---|---|---|
| qiskit / aer | dense statevector (or its own MPS) | native | 2^n memory |
| **tsim** | ZX-calculus **stabilizer rank** | **yes**, via tag extensions | T-count |
| **ppvm** | **Pauli propagation** (Heisenberg) | **yes**, by branching | T-count |
| perceval | dual-rail photonic (SLOS) | via generic 1-qubit unitaries | mode count |
| bloqade | neutral-atom gate model (pyqrack) | native | 2^n |
| **ffsim** | **fixed-particle-number** fermionic basis (IBM) | Hamming-weight-preserving gates only | C(n, k) memory |

**tsim is not Stim.** Stim is Clifford-only; tsim is a stabilizer-*rank*
sampler that consumes a Stim-derived instruction set **plus two tag extensions**
carrying the gates Stim cannot express:

```text
T                        S[T] <q>
T_DAG                    S_DAG[T] <q>
RX/RY/RZ(theta)          I[R_X(theta=<c>*pi)] <q>
U3(theta, phi, lambda)   I[U3(theta=..., phi=..., lambda=...)] <q>
```

`tsim.Circuit` accepts that text directly; `ppvm.StimProgram.parse` lowers the
same tags into its `ExtendedInstruction::{T, Rotation, U3}`. Both therefore run
non-Clifford circuits — verified, not assumed: `h; t; h` has exact
`P(1) = sin²(π/8) = 0.1464`, a value no Clifford-only simulator can produce
(dropping the T gives 0.0; treating it as S gives 0.5), and tsim returns 0.1447
against Qiskit's 0.1480 at 20k shots.

The cost is the usual one: stabilizer rank for tsim, Pauli-sum branching for
ppvm, both exponential in T-count. A Toffoli carries seven T-family rotations
and sits at `θ = π/4` where `cos θ = sin θ`, so **both branches of every split
carry equal weight and no coefficient truncation can prune either**. That is a
property of the gate, not a defect of either engine.

### Decomposed rather than refused

ppvm's parser rejects `SWAP`, `CCX` and `CCZ` as *instructions* — confirmed by
handing raw text to `ppvm.StimProgram.parse`, which answers
`unsupported instruction 'SWAP'`. It accepts `CX`, `H`, `S[T]` and `S_DAG[T]`,
and those compose into all three, so the converter expands them
(`_PPVM_EXPANSIONS` in `qasm2_stim.py`). Refusing them had been a **lowering**
gap on this side, not a capability gap on theirs, and the refusal said the
wrong thing.

Verified against Qiskit including phases: a Toffoli placed inside interference
(`h;h;h; ccx; h; t; h`) reproduces Qiskit's pattern to sampling noise
(0.43/0.42/0.076/0.074), which a sign error in the T ladder would break even
though a basis-state test would not notice.

**Still refused, correctly:** controlled rotations (`crz`, `cu1`, `cp`, `cu3`).
`u1 = rz` up to a global phase, which is invisible in measurement statistics —
but inside a control that phase becomes *relative*, i.e. observable. Lowering
one through the uncontrolled shortcut would be a silently different circuit.

## Refusals name the reason and the way out

A bridge refusal is a capability statement, and it is worth reading. Every one
now carries three things: **what** was refused, **why this engine** cannot do
it, and **which backend runs it today**. Two that used to leak raw internals:

- bloqade on a feedforward circuit returned `object of type 'float' has no
  len()` — an interpreter type error naming neither the missing feature nor an
  alternative. It is now `bloqade-not-supported`, naming `if (creg == N)`
  lowering to an `if_else` node the pyqrack interpreter does not handle.
- perceval on `CY` / `CRZ` / 3-qubit gates raised `UnknownGateError` /
  `NotImplementedError` as `kind:"execute"` — a *hard error*, which the
  cross-backend harness treats as a broken bridge and panics on rather than
  skipping. Those are now `perceval-not-supported`.

The distinction matters beyond tidiness: the harness skips a typed refusal with
a reason and fails on an execution error, so mis-typing a capability limit turns
"this backend cannot express that" into "this bridge is broken".

## Expectation over a bridge

`--bridge <name> --expectation "<Pauli string>"` now works for the bridges that
implement it (qiskit, ppvm, tsim). The transport always existed — the
cross-check harness used `omega_bridges::expectation_qasm2` — but it was
reachable only from Rust, so the CLI turned the combination away even for
backends that support it.

The CLI's sparse `Z0 X2` form is widened to the wire's full-width LSB-first
Pauli string at the call site, rather than teaching the wire a second encoding.
Verified against the in-process statevector backend on asymmetric observables
(`Z0`, `Z1`, `Z2`, `Z0 Z2` on a circuit where all three qubits differ), which is
what a qubit-order mistake would show up in.

### ffsim — expectation over QPY, not QASM2

ffsim is the **differential anchor for the fermionic surface** (`Rbs`,
`CPhase`, Jordan–Wigner observables; `PLAN-OPEN-20260825.md` §3c.0e item 2) and
is deliberately not on the QASM2 wire. `Rbs` has no QASM2 spelling, and lowering
it to CX + rotations hands ffsim a circuit it either rejects or routes to a
general simulator — the thing the anchor exists not to be. So
`omega_bridges::ffsim::expectation(&CircuitIR, &[WireObservable])` serialises
the circuit with the pure-Rust QPY writer, base64 inside the JSON request, and
`Rbs(θ)` arrives as `XXPlusYYGate(−2θ, β = π/2)` on the same qargs. The sign and
the β were both checked numerically against qiskit 2.5.2 (`qpy/write.rs::
qiskit_params` carries the receipt; `tests/qpy_rbs_vs_qiskit.rs` pins it through
Qiskit's own gate semantics on `⟨X0 X1⟩ = −sin 2θ`, which flips under either
mistake). `CU3(0,0,λ)` is `CPhaseGate(λ)` exactly and is rewritten runner-side.

The runner (`python/ffsim_runner.py`, testable half in `ffsim_prep.py`) turns
the leading `x` layer into a Slater determinant, **refuses** anything not
Hamming-weight preserving with `ffsim-unsupported-gate`, evolves in ffsim's
number-conserving basis, and only then expands to a 2^n statevector so the
observables go through the same `SparsePauliOp` path as `qiskit_runner.py`.
The two anchors therefore differ only in the evolution engine. Counts mode and
`expectation_qasm2(Backend::Ffsim, …)` are `CannotExpress`, with the message
pointing at the real entry point.

### OpenFermion-FQE — the JW-free oracle (test-only)

ffsim and our statevector are read out through the **same** Jordan–Wigner
observables, so a sign error in `jordan_wigner()` cancels out of that
comparison. `python/fqe_runner.py` closes the hole: it takes the circuit as
fermionic generators (`{"kind":"givens","p","q","theta"}` ≡
`exp(θ(a†_p a_q − a†_q a_p))`, `{"kind":"cphase","p","q","phi"}` ≡
`exp(iφ n_p n_q)`) and observables as ladder products
(`[[re, im, [[mode, dagger], …]], …]`), evolves in FQE's FCI basis with a
spinless register mapped to an all-alpha sector, and never touches a qubit.
`tests/fqe_vs_statevector.rs` drives it against `omega-backend-statevector`
with the mapper on our side only; agreement is at machine precision and the
flipped-angle control moves by O(1).

There is **no** `Backend::Fqe` and there must not be one — it is a reference,
not a product surface. `make -C crates/omega-bridges/python fqe-venv` builds
`.venv-fqe` on Python 3.11 through `uv` (FQE 0.3.0 is a 2023 Cython sdist:
`numpy<2`, no 3.13, declared deps the runner never imports installed
`--no-deps`). Without the venv the test skips out loud.

## Capability handshake

Send `{"mode": "capabilities"}` to any runner and it answers without executing
anything:

```json
{"ok": true, "capabilities": {
  "backend": "qiskit",
  "modes": ["execute", "expectation", "qpy_to_qasm2", "qasm2_to_qpy"],
  "noise_keys": ["depolarizing"],
  "notes": "scalar depolarizing only; per-qubit and per-pair forms are refused"
}}
```

Without this, a mismatch between what a caller wants and what a bridge
implements surfaced only as a mid-run error — after the circuit had been
converted and sent — and for noise it did not surface at all.

## Noise: what each bridge accepts, and what it refuses

`noise` is **not** an opaque dict a runner may ignore. A runner that cannot
honour a key must refuse it, because a caller who asks for noise and receives a
noiseless distribution has no way to tell.

| bridge | accepts | refuses with |
|---|---|---|
| qiskit | scalar `depolarizing` | `qiskit-noise-key-not-supported` |
| perceval | — | `perceval-noise-not-supported` |
| bloqade | — | `bloqade-noise-not-supported` |
| tsim | — | `tsim-noise-not-supported` |
| ppvm | `depolarizing` (all forms), `pauli`, `phase_damping`, symmetric `readout` / `readout_flip` — counts mode only | `ppvm-noise-not-supported` (`amplitude_damping`, asymmetric `readout`, unknown keys) |

`perceval` and `bloqade` previously never read `noise` at all and returned a
noiseless result — the same silent-drop defect `--noise` was fixed for
in-tree. The qiskit runner accepted `amplitude_damping`, `phase_damping`,
`pauli` and `readout` and ignored all four; it now refuses them by name. Its
scalar-only restriction also means the per-qubit and per-pair `depolarizing`
forms are refused rather than flattened to one uniform rate, which would be a
different noise model from the one requested.

For the full model — per-pair two-qubit rates, amplitude/phase damping, Pauli
and readout error — use an in-process backend (`statevector`, `mps`); ppvm
covers every part of it except amplitude damping and asymmetric readout.

### ppvm: omega's noise as Stim noise instructions (2026-09-30)

**What ppvm accepts** — established by handing raw text to
`StimProgram.parse` + `sample_stim` (ppvm 0.1.0, the unpinned git install),
not assumed: `DEPOLARIZE1`, `DEPOLARIZE2`, `X_ERROR`, `Y_ERROR`, `Z_ERROR`,
`PAULI_CHANNEL_1`, `PAULI_CHANNEL_2` and the record-flip `M(p)` are accepted
and sample correctly (`DEPOLARIZE1(0.3)` on `|0>` → P(1) = 0.200;
`M(0.3) 0; M 0` → 0.300 then 0.000). `HERALDED_ERASE` ("unsupported
instruction") and `E` ("unknown instruction") are rejected at parse time.

**The mapping** (`python/stim_noise.py`) reproduces omega's placement, not
Stim's idioms. After every gate application — including `id` and each qubit of
a `reset`, and once after a whole `swap`/`ccx` expansion — each touched qubit
`q` gets, in omega's order:

| omega key | emitted | rate |
|---|---|---|
| `depolarizing` | `DEPOLARIZE1(p) q` — per qubit, **also after a 2q gate** | `NoiseModel::at_gate(q, gate)`: 1q → `1q`; 2q → its `"i,j"` pair entry, else `2q`/`default` at `q` |
| `pauli` | `PAULI_CHANNEL_1(px,py,pz) q` | per-qubit `X`/`Y`/`Z` |
| `phase_damping` λ | `Z_ERROR(λ/2) q` | omega unravels it as Z w.p. λ/2 |
| `readout` / `readout_flip` (symmetric) | `M(p) q` in place of `M q` | per qubit |

Two obvious spellings are wrong, and `tests/ppvm_noise.rs` has a fixture that
catches each: `DEPOLARIZE2` is the uniform 15-Pauli channel, whereas omega
kicks each qubit of a 2q gate independently (Bell TVD 0.040 against our noisy
statevector when mutated in); and `X_ERROR` before `M` flips the *qubit*,
whereas omega flips only the *record* — two `measure`s of one qubit tell them
apart (TVD 0.122).

**Refused**, as `ppvm-noise-not-supported` → `CannotExpress`, naming the key:
`amplitude_damping` (non-unital, not a Pauli channel — ppvm's tableau carries
Pauli noise only, and a twirled substitute is a different channel; an all-zero
rate is accepted as the no-op it is) and asymmetric `readout` (`p10 != p01`;
`M(p)` flips with one probability whatever the true bit). Atom loss stays out:
`bits_to_counts` hard-errors on a `LOST` outcome, and giving it a place in the
counts shape is a protocol change, not a mapping. The expectation mode takes no
noise (the CLI refuses `--bridge --expectation --noise`; the runner refuses it
too for a direct caller).

## Discovery

For each backend, in order: `$OMEGA_BRIDGE_<SLUG>_CMD` (full path) → the wrapper
script on `PATH` → a dev fallback inside the repo. Each backend gets **its own
venv**; never system Python.

## What the bridge surface can and cannot carry

The protocol carries **QASM2 in, counts out**. That is the right shape for
"sample this circuit", and the wrong shape for anything richer. Two limits worth
knowing before choosing a bridge:

1. **No expectation values, gradients, or detector/observable records** — only
   measurement counts.
2. **Each backend covers a gate subset** of the fixture corpus. Runners refuse
   out-of-subset gates loudly (`kind: "<slug>-unsupported-gate"`) rather than
   skipping them, and the cross-backend tests report **how many fixtures
   qualified** — a differential check that silently tests three cases is worse
   than none.

## tsim — a real QEC tool, on a surface that cannot express QEC

**tsim is a genuinely strong tool for QEC investigation.** It samples noisy
Clifford+T circuits by ZX stabilizer-rank decomposition at scales the MPS
backend cannot reach, supports the full Stim v1.13 instruction set with its
noise channels, and — the part that matters for QEC — **detectors and
observables**. Magic-state distillation and cultivation, surface-code sampling:
that is what it is built for.

**The limitation is the bridge, not tsim.** Detector and observable sampling has
no expression in a QASM2-plus-counts protocol, so *through this surface* tsim
arrives as a plain noisy sampler with its most useful QEC feature unreachable.

So if the goal is QEC investigation, the bridge is the wrong door. Options, in
increasing order of effort:

- **Drive tsim directly** (its own Python API) for exploratory QEC work. Nothing
  in this repository needs to be involved, and this is the right first move.
- **Extend the protocol** with a detector/observable response shape — scoped
  work, worth doing if QEC sampling becomes a recurring need here rather than an
  investigation.
- **Port the method to Rust** if it is ever needed *inside a loop* — see below.

## Python in a loop: the tier rule

The bridge spawns **a subprocess per call**. That is fine for a one-shot or
CI-time differential check, and unacceptable inside a training, scoring, or
sampling loop, where process spawn plus interpreter startup dominates and puts a
Python VM on every iteration's critical path.

| use | mechanism |
|---|---|
| one-shot / CI cross-check | subprocess bridge — fine |
| in-loop validation (per step, row, or trial) | **must be in-process Rust** |

- **ppvm is already Rust** (`ppvm-pauli-sum`, `ppvm-tableau`), so the in-process
  tier needs no porting: take it as a git Cargo dependency. The bridge is a
  cheap way to get a first number, not the destination.
- **tsim is Python on JAX/XLA**, so it cannot serve in-loop validation as it
  stands. That would require porting ZX stabilizer-rank decomposition to Rust —
  a real project, planned in `FIXES_PLAN.md` E3 (specify in Lean from the method
  paper, port kernel by kernel, prove equivalence with `leanlift`, and use
  Creusot/Kani for overflow, bounds and termination).

## Why ppvm is here at all

ppvm implements the **same algorithm family** as the in-tree
`omega-backend-pauliprop` — Heisenberg Pauli-sum propagation with magnitude
truncation. It is therefore not a new capability but an **independent numeric
reference** that validates pauliprop.

That distinction is worth the trouble. This project has already shipped a defect
that every *internal* cross-backend agreement gate missed, because each pair of
backends happened to coincide in the basis being checked. Two implementations of
the same idea, written by different people, agreeing is evidence. One
implementation agreeing with itself is not.

## Error `kind` naming — it is a contract, not a label

A runner reports failure as `{"ok": false, "kind": "...", "error": "..."}`, and
`classify_failure` (`src/runner.rs`) maps the **suffix** onto a typed
`BridgeError`. Three outcomes look identical from outside — "no result" — and a
cross-backend matrix that collapses them reports coverage it does not have:

| suffix | maps to | meaning |
|---|---|---|
| `*-not-installed` | `Unavailable` | environmental; says nothing about the backend |
| `*-unsupported-gate`, `*-not-supported`, `*-not-implemented` | `CannotExpress` | the backend understood and **correctly refused**; the cell is legitimately empty |
| anything else | `Backend` | a real defect |

`-not-installed` is checked first. Suffix matching is deliberate, so a new
runner following the convention classifies correctly with no change to
`runner.rs` — the cost being that a **badly-named kind is misclassified**, which
is why the convention is written down here.

Getting this wrong in the safe direction is loud (a refusal reported as a
defect). Getting it wrong the other way is silent: a genuine failure excused as
"cannot express" disappears from the matrix as a legitimately empty cell. Name
accordingly.

`classify_failure` is unit-tested against **every kind the runners actually
emit**, harvested by grepping `python/*.py` rather than imagined.

## Adding a bridge

Mirror `crates/omega-bridges/src/bloqade.rs` (34 lines) plus
`python/<slug>_runner.py`, `python/omega-bridge-<slug>-runner` and
`requirements-<slug>.txt`. The shared subprocess plumbing in `runner.rs` handles
discovery, the `Unavailable` mapping, and error shapes.

House rule: **a blocked integration ships as a findings note, not a fake-green
bridge.** If a tool cannot be installed or does not agree, say so.
