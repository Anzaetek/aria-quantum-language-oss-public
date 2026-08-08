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

## Adding a bridge

Mirror `crates/omega-bridges/src/bloqade.rs` (34 lines) plus
`python/<slug>_runner.py`, `python/omega-bridge-<slug>-runner` and
`requirements-<slug>.txt`. The shared subprocess plumbing in `runner.rs` handles
discovery, the `Unavailable` mapping, and error shapes.

House rule: **a blocked integration ships as a findings note, not a fake-green
bridge.** If a tool cannot be installed or does not agree, say so.
