<!-- SPDX-License-Identifier: Apache-2.0 -->
# What still needs the Linux / RTX 6000 Pro box — 2026-08-15

Everything in the 2026-08-15 branch was verified on macOS **except the items
below**, which cannot be compiled or run there. This is the checklist for the
Linux box; nothing here is a known failure, it is unverified work.

**Short answer to "is it just the CUDA mods?" — no.** The CUDA edits are the
part that was written blind, but three other things are unverified for
different reasons.

---

## 1. The CUDA `Outcome` edits — WRITTEN BLIND, HIGHEST RISK

`crates/omega-backend-statevector-cuda/src/lib.rs` is `cfg`-gated to
linux/windows + `feature = "cuda"`, so **not one line of these five edits has
ever been through a compiler.** They are mechanically identical to the Metal
and OpenCL fixes (which do compile and pass), but three of the five are
test-side, and test-side is where a blind edit is most likely to be wrong:
`Outcome` is **not `Copy`** and has **no `Display`**.

| line | edit | the way it could be wrong |
|---|---|---|
| 858 | `counts_from_u64(counts, circuit.num_qubits)` — collapse arm | is `circuit` in scope under that `cfg`? is `num_qubits` the right width for the collapse path, or should it be the creg width? |
| 892 | same, shots arm | width choice |
| 3154 | `let key = \|v: u64\| Outcome::from_u64(v, 2)` in the Bell test | the `filter(\|(k, _)\| **k != key(0))` needs exactly two derefs — `iter()` yields `(&Outcome, &u32)` and `filter` adds a reference |
| 3215 | `{k:?}` instead of `{k}` | `Outcome` has `Debug`, not `Display` — `{k}` would not compile |
| 3272 | `for k in …keys()` + `k.as_u64().expect(…)` | was `for &k in` — `Outcome` is not `Copy`, so the pattern had to change too |

**A width note that deserves a second opinion.** At line 858 the collapse arm
keys on `circuit.num_qubits`, because the keys come from
`sample_counts_on_device`, which keys on the **qubit register**. The CPU's
collapse arm instead uses `counts_outcome_width(circuit, collapse)` and can key
on the **creg**. I preserved what the device sampler actually produces rather
than matching the CPU's choice — projection onto the creg happens later and
reads `Outcome::bit(q)`, which needs qubit-register keying. If the CUDA
collapse counts come out at the wrong width, this line is why.

```sh
# The compile is the point; do this first, it is the cheap failure.
cargo check -p omega-backend-statevector-cuda --features cuda --all-targets
cargo test  -p omega-backend-statevector-cuda --features cuda   # ci.sh:365
```

## 2. CUDA f64 — the newest arm, least covered

`f64_path.rs` arrived with patches 0011–0013 (request R8-cuda-f64) and is the
least exercised code in the crate. The `Outcome` edits above sit on the shared
`execute` path, so they affect **both** precisions.

Run the f32 and f64 arms separately and do not let one standing in for the
other:

```sh
cargo test -p omega-backend-mps-cuda        --features cuda   # ci.sh:367
cargo test -p omega-backend-pauliprop-cuda  --features cuda   # ci.sh:369
cargo test -p aria-runtime --features cuda --test run_examples gpu_cuda_agrees_with_sim
cargo test -p aria-runtime --features cuda --test run_examples gpu_mps_cuda_agrees_with_sim
cargo test -p aria-runtime --features cuda --test run_examples rbs   # ci.sh:377
```

The `gpu_cuda_agrees_with_sim` arm is the one that matters most: it is the
CUDA↔CPU differential check, and the CPU side changed this branch (S1). S1 is
bit-identical so **no** movement is expected there — if that test's delta
moves at all, S1's bit-identity claim is wrong on some path the macOS tests do
not reach, and that is a finding, not a tolerance to widen.

## 3. `RUST_TEST_THREADS` — request E7, still open and now load-bearing

`cargo test -p omega-backend-statevector-cuda --features cuda --lib` is flaky
under cargo's default parallelism (~1 run in 4, sometimes SIGSEGV, at
`ForwardGraph::capture` + concurrent `cuda.execute`). `CudaStatevectorBackend`
is neither `Send` nor `Sync` **by construction** (captured `CudaGraph`, raw
`*mut CUgraph_st`), so this is a property of the type.

`ci.sh` still has **no** `RUST_TEST_THREADS` anywhere. So:

- run the CUDA stage with `RUST_TEST_THREADS=1` and record whether that alone
  makes it deterministic across, say, 8 consecutive runs;
- **a flake here must not be filed as a failure of the `Outcome` edits.** Run
  each CUDA test set at least twice before blaming this branch.

This also gates `PLAN-SV-PERF.md` S2: a shared rayon pool must never be handed
a CUDA backend handle, and that constraint should be checked on hardware before
S2 lands, not after.

## 4. Two feature sets that are unswept on ANY machine

I swept all 22 macOS-buildable feature combinations (all clean). Two could not
be swept anywhere available here, and the `Outcome` migration is exactly the
kind of change that hides in an uncompiled feature — six sites did:

- **`tch`** — `aria-cli` / `aria-runtime`. macOS builds fail on the Apple-clang
  `std::is_arithmetic` issue; Linux is where this can be checked at all.
  `cargo check -p aria-runtime --features tch --all-targets`.
- **`cuda` in `bindings/aria-py`** — a **separate workspace with its own
  lockfile**, so no workspace-wide command reaches it. `ci.sh:402` has a stage
  for it now (request E1), but the `cuda` feature there is untested.
  `cargo check --features cuda --all-targets` inside `bindings/aria-py`.

## 5. Not blocked on Linux, but wants a many-core x86 box

`fixes/xxx-2/REQUEST.md` says it plainly: *"The threading gap should widen on a
many-core x86 box; we have not measured that."* The S0/S1 numbers in
`PLAN-SV-PERF.md` §1.5 are from a 12-core arm64 machine. When S2 lands, its
scaling numbers should be taken there too — a 12-core result is not a scaling
law, and the invariance test's `T ∈ {1,2,3,12}` should include a `T` above the
core count of the box it was designed on.

---

## Order

1. `cargo check … --features cuda --all-targets` — compile first, it is the
   cheapest way to find the blind edits.
2. The CUDA test sets, each run twice (§3).
3. `gpu_cuda_agrees_with_sim` specifically, against S1 (§2).
4. The two unswept feature sets (§4).
5. S2 scaling numbers, whenever S2 exists (§5).
