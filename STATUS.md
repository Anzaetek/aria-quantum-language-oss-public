<!-- SPDX-License-Identifier: Apache-2.0 -->
# Status — 2026-08-15

Snapshot of what changed, what is verified, and what is knowingly left open.
Every number here was measured; estimates are labelled as estimates.

The previous edition of this file covered only the 2026-08-05 backend-correctness
work (§4 below, kept because it is still true). Twelve commits landed after it
without it being updated — that gap is what this edition closes.

## 1. Landed since 2026-08-05

| area | what | evidence |
|---|---|---|
| **Counts width** | `ExecResult::Counts` is keyed by `Outcome`, not `u64`; the 64-qubit cliff is gone | `omega-run ghz1024_full.qasm --backend mps`, all 1024 qubits measured (`89da782`) |
| **MPS fidelity** | per-run `fidelity_estimate` + `discarded_weight`, labelled an estimate | `50b7441` |
| **pauliprop** | `dropped_mass` reported by `omega-run` and tested **as a bound**, not an estimate | `crates/omega-cli/tests/dropped_mass_is_a_bound.rs` (`3725ba3`) |
| **pauliprop ↔ ppvm** | QuEra's `ppvm` as the *same-algorithm* anchor — the earlier "anchor" compared ppvm against Qiskit and never built our backend | 26 (circuit, observable) pairs, worst \|Δ\| = 0.000e0, 0 skipped (`d4d1229`) |
| **MPS ↔ Aer** | our wide MPS sampling against Qiskit Aer's `matrix-product-state` | 4 circuits; worst TVD 0.0327, worst per-qubit \|ΔP(1)\| 0.0158, worst \|Δ⟨ZᵢZⱼ⟩\| 0.0302 (`388cf31`) |
| **Governor** | five admission defects: batch pricing, the `Opaque` hole, the device memo | `PLAN-GOVERNOR-ADMISSION.md`, implemented and reviewed (`9ddf8e1`) |
| **OpenCL** | the feature 413'd *every* statevector run on the hardware it exists for | `c7e3348` |
| **QASM2 import** | `u` was the last import gap; `rxx`/`ryy`/`rzz` land too | `PLAN-CR-20260813.md` B1 |
| **CLI** | an unrecognised `--flag` was accepted and silently ignored | `d10c1cd` |

## 2. Landed 2026-08-15

**The `Outcome` migration had missed four feature-gated backends.** None of them
is built by `cargo test --workspace`, so each survived the migration by not
being compiled at all:

| crate | feature | defect |
|---|---|---|
| `aria-runtime` | `remote` | the wire key decoded through `u64::from_str_radix` — capped a remote run at 64 qubits and discarded the width below it |
| `omega-backend-statevector-metal` | `metal` | `ExecResult::Counts` built from a `u64` map |
| `omega-backend-statevector-opencl` | `opencl` | same |
| `omega-backend-statevector-cuda` | `cuda` | same, two sites |

`PLAN-WIDE-COUNTS.md` named this class of site in advance — *"`from_str_radix`
still funnels through `u64` … the conversion sites are where the defect will
survive if it survives anywhere."* It survived in exactly four of them.

The remote decoder's fix ships with three tests, each **mutation-checked**:
decoding back through `u64::from_str_radix` fails the 70-bit test and only that
one; pinning the width to 64 fails the width test and only that one.

**CI hygiene, same day:** stage 1 (rustfmt) and stage 2 (clippy `-D warnings`)
were both red on `main` — six files of fmt drift, and 16 lints in
`omega-parser/src/lower.rs` — from the three preceding commits.

**CPU statevector, PLAN-SV-PERF S1:** `apply_2q` scanned all `dim` indices and
rejected three of every four. It now walks the `dim/4` groups directly.
Measured **1.30×**, flat across widths and circuit shapes:

| circuit | before | after |
|---|---|---|
| ghz_28 | 15.640 s | 12.040 s |
| qft_26 | 45.490 s | 35.300 s |
| qft_28 | 213.790 s | 163.350 s |

Bit-for-bit identical, and tested that way: the replaced loop is kept verbatim
in `mod group_walk_equivalence` and compared against with `to_bits()` over
`n ∈ 2..=7` × every ordered qubit pair.

## 3. Measured baseline — CPU statevector

12-core Apple silicon, 24 GB, `rustc 1.95.0`, `--release`, one thread, 1000
shots. Full table in `PLAN-SV-PERF.md` §1.5.

- The **sampler is not a time cost** — 1.5 s of 215 s at 28 qubits.
- The **sampler doubles peak memory**: `qft_28` RSS 4099 MiB (evolution) → 8195
  MiB (with sampling). Evolution's 4099 MiB is `2^28 × 16 B` to the megabyte, so
  the state is the only large allocation in the gate loop; the other 4096 MiB is
  `sample_counts` holding two `2^n` f64 vectors. Fixing that is S3.

## 4. Verified (2026-08-05 work, unchanged)

- **`verification/Verification/Backend/PauliAlgebra.lean`** — 8 theorems,
  **proved**: no `sorry`, no Mathlib, no `native_decide`.
- **Qiskit differential cross-check** (`ARIA_QISKIT_XCHECK=1`): aria CPU vs
  Qiskit `4.441e-16`; stabilizer vs CPU `4.441e-16`; Metal vs CPU `1.857e-7`
  (f32).
- **Reset audited across all 11 backends** — no silent skips.
- Five defects fixed and evidenced: `stabilizer_expectation`'s pivot-less group
  test, `pauli_mult_phase`'s inverted `X·Z` (in two copies), the fast-reject on
  non-diagonal stabilizers, and two Metal `Reset` defects.

## 5. Known open — recorded, not fixed

1. **CUDA is unverified for the 2026-08-15 fixes.** Both `cfg`-gated arms were
   edited without being compiled — macOS cannot build them. They need the
   Linux / RTX 6000 Pro box, f32 **and** f64 (`f64_path.rs` is the newest and
   least-covered arm). → `PLAN-SV-PERF.md` §6.5
2. **CUDA's Reset criterion diverges** (refuses on random *outcome*, not
   entanglement). A false rejection, not a wrong answer. → `LIMITATIONS.md`
3. **Metal per-shot GPU trajectories block at ~64 shots** — in-flight
   command-buffer exhaustion. `execute` delegates to CPU meanwhile.
4. **`Reset.lean` / `StabilizerExpectation.lean`: 7 `sorry` targets.** They need
   a real ordered field; core Lean has none.
5. **Five distinct Reset acceptance policies** across backends. Only CPU↔Metal
   is conformance-tested. → ledger A6
6. **`ci.sh` runs clippy without `--all-targets`.** `cbce4a2` cleared the lints
   so CI *could* adopt it; CI never did, and it has since regressed
   (`omega-parser/tests/gate_arity_is_validated.rs:165`). → request E2
7. **`tools/qec_cross_check/run.sh` bootstraps with plain `pip`** —
   `pymatching` cannot source-build on aarch64. → request E6
8. **No `RUST_TEST_THREADS` in `ci.sh`** for the CUDA stage;
   `CudaStatevectorBackend` is neither `Send` nor `Sync` by construction. →
   request E7
9. **`aria_py` bypasses `omega-server` admission entirely** — no way for a
   client to ask what a circuit would cost. → request R2b
10. **CPU statevector is single-threaded per circuit** and 2q-heavy circuits run
    sparse gates as dense 4×4 — the CPU is the only backend without a diagonal
    fast path, which Metal, CUDA and OpenCL all have. → `PLAN-SV-PERF.md` S1b, S2
11. **MPS is ~210× slower than our own statevector on one 19-qubit chain.**
    Reproduced, undiagnosed; its suspected link to the 128q counts bug is
    disproven. → `PLAN-CR-20260813.md` B4
12. **The ppvm bridge refuses any noise request** — ppvm's tableau supports the
    channels including atom loss; the mapping is unwritten.
14. **`hhl.aria` and `proofs/lean4/QuantumProofs/HHL.lean` need cleaning up —
    and the link between them is an annotation, not a check.** The off-by-one
    fixed on 2026-08-15 is the symptom; the structure around it is the item.
    Three separate things to settle:
    - **The example does not implement what the proof proves.** `HHL.lean` is
      sorry-free and genuine — `hhl_solves_system` is a real matrix-vector
      identity `A · o = C · b` for `A = diagonal λ`, with an exact QPE and a
      controlled `RY(2·arcsin(C/λᵢ))`. The Aria example instead uses a
      *proxy* eigenvalue (`theta = 0.5 / (k + 1)`, commented "eigenvalue proxy
      λ = k + 1") and, at `hhl.aria` step 3, an "inverse QFT" that is **H on
      each qubit and nothing else** — a true inverse QFT needs the controlled
      phase ladder, so for `n ≥ 2` that step is incomplete. Whether the example
      should be brought up to the proof, or the header claim narrowed to what
      the example is (a template, not a certified HHL), is a decision, not a
      bug fix.
    - **`@prove "hhl_recovers_inverse"` is not verified against anything.** The
      annotation names a property; `ci.sh:656` separately checks the Lean
      theorems are sorry-free. Nothing connects the two — no check would notice
      if the circuit and the theorem drifted apart, and they have.
    - **The harness cannot see circuit-level defects at all.** It compares the
      WASM path against an in-process oracle **on the same lowered IR**, so it
      is a transport check. It reported `Δmax = 0.000e0 PASS` for as long as the
      off-by-one existed. **The other 48 examples run through the same
      structurally-blind check.** The *loop-bound* class specifically has now
      been audited — see below — but that is one class of circuit defect, and
      the harness is blind to all of them.

    **Loop-bound audit, 2026-08-16 — `hhl` was the only one.** All **115
    `repeat` loops across the 44 `examples/aria/*.aria`** were checked against
    their register declarations. Everything else is right, and several are
    right in a way that shows the inclusive semantics were understood:
    `bernstein_vazirani.aria` and `deutsch_jozsa.aria` deliberately use
    `to n` for the Hadamard sweep (the answer qubit at index `n` gets
    `X` then `H`, making |−⟩) and `to n - 1` for the query register, in the
    same file. `qft.aria` and `qec_qft.aria` implement the real controlled-phase
    ladder with `from i + 1 to n - 1` and reverse it with `step -1`.

    Two things that follow. First, `hhl.aria` was an outlier rather than a
    symptom of a house-wide misunderstanding. Second, **`qft.aria` already
    contains the correct inverse QFT that `hhl.aria` step 3 is missing** — so
    reconciling the example needs no new physics, only reuse.

    Scope of that audit, stated so nobody over-reads it: it covers loop bounds
    against register sizes. It does **not** cover gate choice, angle formulas,
    or eigenvalue proxies — the other three ways `hhl` is still not what
    `HHL.lean` proves.
15. **The photonic backend has never been compared on *speed*, only on
    agreement.** `bridge-perceval` exists and the DV/CV conventions are matched
    verbatim, so the correctness axis is covered — but Perceval's hot path is
    C++ (`quandelibc`) and piquasso's is NumPy/JAX, and neither has been timed
    against `omega-backend-photonics`. Deferred deliberately: the statevector
    lane (`PLAN-SV-PERF.md`) and the adjoint lane
    (`PLAN-ADJOINT-MEMORY.md`) are the measured bottlenecks today, and a
    photonic comparison wants the same discipline they got — a baseline before
    a claim, and per-stage timings rather than wall clock.
