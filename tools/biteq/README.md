<!-- SPDX-License-Identifier: Apache-2.0 -->
# biteq — the CUDA↔Metal `Decompose` bit-equality measurement

The harness for `PLAN-BITEQ-CUDA-METAL.md` (which is the authority; read it
first). It measures a claim two doc comments have asserted since before the
mode switch existed: that the CUDA and Metal statevector backends agree
**bit for bit** in `Decompose` mode, because they run the identical 15-gate
Nielsen-Chuang sequence.

## Protocol in one paragraph

Both machines run the same committed corpus (`corpus/*.qasm`, 3–6 qubits,
CCX + CSwap, including circuits where the multi-control does NOT fire) in
BOTH `--multi-control` modes plus a CPU f64 arbiter run, dumping raw
amplitude **bit patterns** (`--dump-state-bits`, hex `to_bits`, never
decimal, `-0.0` visible). Artifacts cross machines by scp; `diff.py` renders
the verdict. The CPU columns are compared FIRST — if the two hosts' f64
states differ, that is libm, and the GPUs cannot be blamed. Each box must
also show Exact ≠ Decompose somewhere, or its mode switch did not take
effect and its artifacts are invalid.

## Running one side

```sh
cargo build --release -p omega-cli --features metal   # or: cuda
tools/biteq/run_side.sh metal artifacts-mac           # or: cuda artifacts-dgx
```

Then move one side's directory to the other machine (scp, initiated from the
Mac per the standing arrangement) and:

```sh
python3 tools/biteq/diff.py artifacts-mac artifacts-dgx
```

Exit 0: claim confirmed on this corpus for the measured OS/compiler pairs.
Exit 2: claim false — the differ prints the softened wording the doc
comments should adopt. Exit 1: the measurement itself is invalid; fix the
harness before believing anything.

## Cautions that are easy to re-lose

* The corpus's strict files use only gates whose matrices are compile-time
  constants. `ry` lives ONLY in `p90_ry_probe`, which the verdict excludes:
  rotation matrices come from host `cos`/`sin`, and macOS vs glibc rounding
  is not the claim under test. The probe measures that difference instead.
* Signed-zero-only differences still falsify, and are attributed by the
  differ to dense-vs-permutation CX dispatch — a real implementation
  difference outside the 15-gate sequence.
* `--dump-state-bits` refuses every silent-fallback path the CLI has
  (explicit `--device` required, dispatch-decision gate, in-arm fallback
  gate, exact f32-promotion check per amplitude). If it refuses, the answer
  is never to weaken the guard; it is to find out what almost lied to you.

## Self-test

`python3 tools/biteq/self_test.py` exercises `diff.py`'s comparison kernel
on synthetic artifacts: an equal pair, a 1-ULP difference, and a
`-0.0`/`+0.0` pair (which must count as a difference and be classed as a
signed zero).
