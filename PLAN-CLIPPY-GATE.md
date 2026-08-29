<!-- SPDX-License-Identifier: Apache-2.0 -->
# Plan: make the clippy gate cover the tree it claims to cover

## Where this came from — and the claim that was wrong

From the M5 (macOS) session, 2026-08-17: their `ci.sh` ran `cargo clippy` without
`--all-targets`, so test targets were never linted and two pre-existing errors sat
there. "A `-D warnings` gate that skips half the tree is a partial gate presenting
as complete."

**The first draft of this plan escalated that into a claim that is false, and it
is recorded here rather than deleted because it was told to another machine before
it was checked.** The draft said `omega-backend-pauliprop` "has never been
clippy'd by CI" and that "nothing in `ci.sh` would have caught" the eight dropped
`Result`s in `2eb4c6d`.

Disproved by experiment. Appending to `crates/omega-backend-pauliprop/src/sim.rs`:

```rust
fn __canary_ret() -> core::result::Result<(), ()> { Ok(()) }
fn __canary_drop() { __canary_ret(); }
```

then running stage 2 **verbatim** (`cargo clippy "${ARIA_CRATES[@]}" -- -D
warnings`):

```
error: unused `std::result::Result` that must be used
error: could not compile `omega-backend-pauliprop` (lib) due to 1 previous error
STAGE2_EXIT=101
```

`cargo clippy -p X` sets `RUSTC_WORKSPACE_WRAPPER`, which runs clippy-driver over
**every workspace member in X's dependency graph**, with `-D warnings` applied to
each. `aria-runtime` depends on `omega-backend-pauliprop` unconditionally
(`crates/aria-runtime/Cargo.toml:6`) and is in `ARIA_CRATES`. The gate would have
failed that commit. The eight sites were caught by hand *before* committing, and
"CI never saw it" was turned into "CI could not see it" without testing the
difference — the same unmeasured-claim failure as the MPS `gesvdj` overclaim and
the `ps -eo pcpu` misreading earlier in this session.

M5 asserted the same thing about their own tree, to their user, without testing
it. One unchecked claim, propagated across two machines in under an hour, because
it was plausible and flattering to the change it justified.

## The gap that is actually there

Two holes, both real, neither the one first claimed:

**(a) No `--all-targets`.** Test targets are never linted. This accounts for
**9 of the 13** diagnostics below.

**(b) 18 crates outside the `ARIA_CRATES` dependency closure**, computed with
`cargo tree -e normal` against `cargo metadata` members rather than eyeballed:

```
omega-backend-cv, omega-backend-mps-{cuda,metal}, omega-backend-pauliprop-{cuda,metal},
omega-backend-refplugin, omega-backend-statevector-{cuda,metal,opencl}, omega-bridges,
omega-cli, omega-client, omega-ffi, omega-plugin-conformance, omega-server,
omega-tensor, omega-wasm-cli, omega-xcheck
```

`omega-core`, `omega-parser`, `omega-backend-statevector`, `-mps`, `-pauli`,
`-photonics` and `-pauliprop` are **already linted**. The remaining 4 of 13
diagnostics are lib/bin code in hole-(b) crates. M5 measured 16 on their tree; the
delta is crate-set differences between the boxes, not method.

Worth noting against the "gate" framing: **6 of the 13 are rustc lints**
(`dead_code`, `unused_variables`, `unused_assignments`), not clippy lints. Part of
what is missing is a plain `-D warnings` gate, not a clippy-specific one.

## The 13, and what each turned out to be

| file | diagnostic | verdict |
|---|---|---|
| `omega-cli/src/main.rs:365,422` | `multi_control` assigned, never read | **real defect** — see below |
| `omega-backend-statevector-cuda/src/lib.rs:445` | const `QUAD_KERNEL_MIN_QUBIT_2SLOT` never used | miscfg'd, not dead |
| `aria-core/tests/every_emitted_gate_is_readable.rs:39` | `EmittedButUnreadable` never constructed | deliberate headroom |
| `omega-ffi/src/lib.rs:76` | unnecessary `*const u8` -> `*const u8` cast | mechanical |
| `omega-parser/tests/gate_arity_is_validated.rs:168` | single-pattern `match` | mechanical |
| `omega-cli/tests/projected_sampling_matches.rs:425` | field assignment outside initializer | mechanical |
| `aria-runtime/tests/counts_above_the_cliff.rs:118` | field assignment outside initializer | mechanical |
| `aria-core/tests/qasm2_qiskit_dialect.rs:116` | `clone` on `Copy` `GateKind` | mechanical |
| `aria-core/tests/rxx_rzz_round_trip.rs:131` | useless `vec!` | mechanical |
| `aria-core/tests/every_emitted_gate_is_readable.rs:133` | `clone` on `Copy` | mechanical |
| `aria-core/tests/every_photonic_gate_round_trips.rs:140,255` | `clone` on `Copy` | mechanical |

**11 of 13 are not defects. One is.** The draft called all 13 benign; that was
wrong on `multi_control`.

### `--multi-control` is accepted and silently discarded — fix the behaviour

Both `with_multi_control` call sites (`main.rs:1020, 1390`) are
`#[cfg(feature = "cuda")]`, and `with_multi_control` exists only on
`CudaStatevectorBackend` (`statevector-cuda/src/lib.rs:261`). So on a default
build the flag is parsed, validated, and dropped.

The draft proposed `let _ = multi_control;`. That converts a compiler-reported
defect into a silent one, and the repo has already written down the opposite
standard twice:

* `GATE-EXACTNESS.md:126-130` refused to put `multi_control` on
  `QuantumExecuteReq` precisely because it would be "inert, advertising a
  capability no code path can honour". The CLI shipped exactly that.
* `a246c4a`, landed yesterday: "refuse noise a runner cannot honour" — "a caller
  who asked for noise had no way to tell they did not get it".

**But the harm runs the opposite way from the obvious reading, and that changes
the fix.** The CPU statevector applies `CCX`/`CSwap` **directly**
(`statevector/src/sim.rs:727, 734`; GATE-EXACTNESS.md: "walks the CCX subspace
directly"). So CPU is *always* exact:

* `--multi-control exact` off CUDA — the user gets exactly the numbers they asked
  for, by a different route. No numeric harm.
* `--multi-control decompose` off CUDA — the user asked for the 15-gate chain and
  got the exact permutation. **This** is the silently-unhonoured case, and it is
  the default value, so it is reached by anyone comparing a CPU run against a
  CUDA/Metal decomposed reference.

So **refusing is wrong** — it would break scripted invocations that pass
`--multi-control` uniformly across devices and are numerically fine. The
proportionate fix is the one the neighbouring flag already uses: `--device`
"falls back to cpu ... when not compiled in or available" and says so at runtime
(`main.rs:1028, 1067`).

Fix: track whether the flag was **explicitly passed**, and emit an `info(...)`
line when the resolved execution path cannot honour it, naming what was applied
instead.

Two properties this has and the `cfg` approach does not:

1. **It is a runtime check, so it covers the runtime cases.** `--device cpu` or
   `--backend mps` on a CUDA build also discard the mode; a `cfg` catches none of
   that. `DeviceKind::resolve` already falls back to CPU when the device is
   unavailable (`omega-core/src/device.rs`), so `resolved_device == Cuda` implies
   CUDA is compiled in and present — one predicate works in every build.
2. **It reads the variable**, so the lint dies honestly rather than being
   silenced.

The draft's proposed mechanism was also wrong on its own terms: it copied the
`#[cfg(not(any(feature = "metal", feature = "cuda")))]` predicate from
`resolved_device` two lines up, but Metal has no `with_multi_control`, so under
`--features metal` the variable is still unused and the warning returns. Nothing
in CI builds `omega-cli` with `metal` **or** `cuda`, so that mistake would not
have been caught. (`omega-cli/Cargo.toml:38` names a `ci-linux-cuda.sh` as its
gate; **that file does not exist**, in the tree or in any commit. Corrected in
passing.)

### `QUAD_KERNEL_MIN_QUBIT_2SLOT` — the draft's `cfg` was wrong

The draft said the four use sites are "all inside `#[cfg(feature = "cuda")]`".
They are not. All four (`lib.rs:756, 779, 806, 936`) are under:

```rust
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
```

Applying the draft literally leaves the const dead on **macOS + `--features
cuda`** — a configuration the crate's own manifest promises works
(`statevector-cuda/Cargo.toml:12-14`). Use the full three-part predicate.

The const is private and the two prose references (`lib.rs:795, 2936`) are `//`
comments, not intra-doc links, so cfg'ing it breaks no docs and no build.

### `EmittedButUnreadable` — correct, but the file had drifted

The judgement holds: all 37 `GateKind` variants were checked, every gate filed
`RoundTrips` is genuinely readable, and the nearest candidate for mis-filing —
`RYY` — is verified against Qiskit at max|Δ| = 1.214e-16
(`qasm2_qiskit_dialect.rs:148`). No gate is mis-filed.

Two things in the file that the draft read past:

1. **The module header contradicts the table.** Line 16 still says `ryy | hand
   diff | still unreadable, deliberately`, while `spec()` at :66 classifies
   `RYY => RoundTrips(1, 2)`. The row describes the *bare* spelling, which is no
   longer what is emitted. It reads as evidence for exactly the mis-filing this
   analysis just ruled out.
2. **The "exhaustive" `ALL` table is not exhaustive.** Verified by set difference,
   not by counting: `GateKind` has 37 variants, `ALL` has **35**, missing
   `HalfWavePlate` and `PolarizingBeamSplitter`. Harmless today (both are
   `NotQasm2`) but the file's whole design argument is that the compiler and this
   list keep each other honest, and half of it has already drifted.

Use `#[expect(dead_code, reason = "...")]`, not `#[allow]`. The toolchain is
pinned at 1.95 so `expect` is stable, and it **self-retires**: the moment a gate
maps to the variant, the unfulfilled expectation fires and forces the attribute
out. `#[allow]` is permanent invisible drift — the failure mode this plan exists
to attack.

## The change

1. Fix the 13 per the above.
2. Widen stage 2 to `cargo clippy --workspace --all-targets -- -D warnings`.
   `--workspace`, not a widened hand list, for the reason `ci.sh:22-34` already
   records about the last hand list: "a crate is covered here only if someone
   remembered to type it."
3. Add the **featured** invocations to the `ARIA_CUDA=1` block. `--workspace`
   cannot lint cfg'd-out code, and today the only featured line is
   `-p omega-server`, so `omega-cli`, `aria-runtime`, `statevector-cuda`,
   `mps-cuda` and `pauliprop-cuda` under `--features cuda` are linted by nothing.
   **Without this the `QUAD_KERNEL_MIN_QUBIT_2SLOT` fix ships unverified**, since
   it is a cfg fix and no unfeatured run can check it.
4. Keep the curated arrays for `fmt` and `build`. Formatting genuinely is curated
   (`aria-core` is ported verbatim and deliberately unformatted).

## Cost: 24.5 s cold. Measured, having first refused to.

```
CARGO_TARGET_DIR=<fresh> cargo clippy --workspace --all-targets   ->  24.51 s
```

The draft wrote a paragraph explaining that measuring this "costs more than the
answer is worth", plus a contingency for a cost it never established. It cost one
command and 25 seconds. Warm-registry, aarch64, this box.

One genuinely new compile surface: `--all-targets` includes `--benches`, and the
eight `benches/` directories declare `harness = false`, so stage 5's
`cargo test --workspace` does **not** build them. This is not a re-run of work CI
already does. (GPU benches are `required-features`-gated and correctly skipped.)

## What this still does NOT cover — stated, not implied

`--workspace` "has no holes by construction" is true only of workspace members.
After this change the gate still does not reach:

* **`bindings/aria-py`** — a separate cargo project. `ci.sh:467-470` already calls
  this out by name as "the same coverage hole as the typed crate list, one
  directory over". It gets `cargo build`, never clippy.
* **`crates/aria-backend-tch`** — workspace `exclude`d.
* **`examples/wasm-guests/*`** — five excluded members, built at `ci.sh:227-230`,
  never linted.

A plan whose thesis is "hand-maintained gates grow holes" does not get to leave
three of them unnamed.

## Cross-machine consequence, flagged by M5 before the fact

The five GPU backends are outside the stage-2 closure on **both** boxes. Widening
means `statevector-metal` and `pauliprop-metal` get linted for the first time on
the M5 machine, and `statevector-cuda`/`mps-cuda`/`pauliprop-cuda` here. Each of
us will see first-time diagnostics in crates the other cannot compile. That is the
widening working, not a regression from it — and it is the same platform split
that let `0152` break `pauliprop-metal` past a green CI.

## RESULT — CI r14, `ARIA_CUDA=1 ARIA_QISKIT_XCHECK=1`, EXIT=0

Landed as `6c990aa`..`7d78e7c` (patches 0072-0076). Reading the FAILED count,
not the passed count: **0 failed**, across every stage that ran.

* stage 2 `--workspace --all-targets`: clean, no output
* `OK: clippy --features cuda (cli, runtime, statevector/mps/pauliprop CUDA backends)`
* Qiskit cross-check: 60 agree, 0 disagree, worst |Δp| = 4.441e-16
* PauliProp vs Qiskit: 40 agree, 0 disagree, worst |d| = 1.998e-15
* 9 stages did not run (absent hardware/venvs), unchanged from r13

The `--multi-control` fix is the only behaviour change in the series; the
numeric gates above are what say it did not move any number.

## Gates

* `cargo clippy --workspace --all-targets -- -D warnings` exits 0.
* The featured CUDA clippy lines exit 0 under `ARIA_CUDA=1`.
* `cargo test --workspace` stays green — several fixes touch test files, and a
  lint fix that changes what a test asserts is a lint fix that got away from me.
* `--multi-control` gets a test: passing it on a non-CUDA path must produce the
  notice, and must NOT change the numbers.
* Full CI green (r14).

## Explicitly not in scope

New lint groups (`pedantic`, `nursery`), widening `fmt`, and the Metal crates —
they cannot be linted here at all. That crate's coverage is M5's to gate.
