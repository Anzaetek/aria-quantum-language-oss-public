<!-- SPDX-License-Identifier: Apache-2.0 -->
# Status — 2026-08-16

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

## 4b. Landed 2026-08-16 — the Linux/CUDA pass, on a real GB10

First execution of the CUDA stage on **Grace-Blackwell** (`aarch64`, sm_121,
48 SMs, 24 MiB L2, 121.7 GiB unified, CUDA 13.0.88, driver 580.159.03). Every
prior CUDA verification in this repo was x86_64 + RTX PRO 6000 + nvcc 12.9.
`ARIA_CUDA=1 ARIA_QISKIT_XCHECK=1 ./ci.sh` → **exit 0**.

| area | what | evidence |
|---|---|---|
| **CUDA build** | two blind `Outcome` casts blocked the whole stage from compiling | `fc1a549`; the handoff doc listed 5 sites, there were 7 |
| **f64** | exact against an independent implementation on sm_121 | `f64_vs_qiskit` worst \|Δamplitude\| **1.388e-16**, ⟨Z⟩ **2.220e-16** |
| **Qiskit differential** | the mandatory gate, never run on this box | 60 agree, 0 disagree, worst \|Δp\| **4.441e-16** |
| **CUDA 13 / aarch64** | `cudarc 0.19.4` builds; NVRTC compiles 25 `.cu` × {f32,f64} on sm_121 | `precision_compile` |
| **SV 2q perf** | CX **1.73×**, SWAP **1.74×**, CZ **2.91×**, CRz **1.74×** | `31d53f5`, `4325a50`, `+0036` |
| **E7** | `RUST_TEST_THREADS=1`, stage-scoped | `33a40b4` |

**Two findings that are not fixes.**

- **The PauliProp GPU "accelerator" decelerates.** 0.11×–0.79× against the CPU
  branch across 6→100 qubits, **no crossover** — including past the 64-qubit
  symplectic word boundary where one might be expected. It had only ever been
  gated on *agreement*, never timed. Structural, not kernel: `BranchHook` takes
  a **host** `&mut PauliSum` and is called once per rotation gate, so each gate
  pays ~15 device allocations, 7 uploads, a full `synchronize()`, 7 downloads,
  and a host merge costing about what the CPU's whole branch loop costs. The
  `PAULIPROP_GPU_MIN = 256` default is **deliberately left alone** — it looks
  far too low, but no amd64 measurement exists.
- **A "speedup" that was a slowdown at low qubit indices, caught by review.**
  The 2q byte saving is a property of `min(qa, qb)`, not of the gate: DRAM
  granularity is a 32 B sector = 4 amplitudes, so below `min = 2` a two-slot
  gate (CX/SWAP/CRz) touches 2 of every 4 — the same sectors as dense, strided
  within them. Measured like-for-like at `qa ∈ {0,1}`: CX specialised 148.99 ms
  vs dense 135.41 ms, i.e. **1.10× slower**. CX ladders, GHZ chains and QFT all
  start at `(0,1)`. Those gates are now index-gated; CZ is not, because a
  one-slot diagonal touches 1 in 4 and wins at every index (122.28 vs 135.50 ms).
  A blanket threshold was written first and would have made CZ 12% slower.
- **Two claims corrected by measuring them.** "Register pressure" was a
  **local-memory spill** (`cuFuncGetAttribute`: 35 regs / 64 B local → 18 / 0
  after `#pragma unroll`, worth 16.7% on CRz), and CRz "bit-identity" was
  passing by luck of FMA contraction — it went red on this box when an unrelated
  unroll moved the schedule, exactly as it would have on nvcc 12.9 / H100.
- **The CUDA-graph path still builds dense 4×4s**, so the QML training hot path
  (`TrainStepGraph`, live production code) gets none of these speedups. The
  numbers above are the non-graph forward/adjoint path only.
- **The GB10 topology branch cannot fire on a GB10.** `nvidia-smi` reports
  `memory.total` as `[N/A]`, so the device row is dropped and the machine
  classifies `HostOnly`. Fails safe here by luck. The obvious fix was
  **withdrawn as dangerous** — `Unified` means no per-device ceiling
  (`worker.rs:762-765`), so it would over-commit a discrete H100. Needs an
  amd64 box. → `PLAN-LINUX-CUDA-GB10.md` §4

## 4c. Landed 2026-09-25 — PLAN-FERMIONIC F1, the `FermionicOp` text spelling

`FermionicOp::parse` + `Display` (+ `FromStr`), in OpenFermion's
`FermionOperator` string form — `1.0 [1^ 3] + 1.0 [3^ 1]`, `^` marks creation,
terms joined by `+`, the identity spelled `[]`. `parse(display(op)) == op`
exactly under normal ordering. No shorthands.

| leg | what | evidence |
|---|---|---|
| (i) | parsed text → `jordan_wigner_terms` → dense Fock matrix, every product of length 1–3 on 4 modes, text built by hand from the product so the parser shares nothing with the oracle | `fermion_jw.rs::parsed_text_matches_the_oracle…`, **584** products |
| (i′) | the §1.2 wrong-string case: `a†₁a₃ + h.c.` parsed vs the hand `0.5·X1X3 + 0.5·Y1Y3` (no `Z2`) on ψ = (\|q1q2⟩+\|q2q3⟩)/√2 | ⟨H⟩ = **−1** with the string, **+1** without, Δ = 2 |
| (ii) | round-trip is the identity, exactly, on generated operators | `fermion_spelling.rs::parse_of_display_is_the_identity_exactly` |
| (iii) | cross-library pin: our `Display` through `openfermion.FermionOperator(...)`, its `str()` through our parser, term-for-term coefficients | `fermion_spelling.rs::openfermion_reads_our_text_and_we_read_its`; `ci.sh` runs it with `OMEGA_OPENFERMION_PYTHON` / the FQE venv, **registered skip** without one |

Refusal, not certificates: malformed text refuses naming the token; non-Hermitian
input keeps the existing `jordan_wigner` refusal naming the coefficient. Nothing
discarded, nothing to bound.

**Mutation proof.** Flipping the parser's `^` detection (creation ↔ annihilation)
reddens 5 tests — 3 of 6 in `fermion_spelling` including the OpenFermion pin,
2 of 11 in `fermion_jw` — and the file was restored bit-for-bit (`cmp`) after.

Nothing in F1 touches the IR or a backend. F2 and F3 follow in §4d.

## 4d. Landed 2026-09-25 — PLAN-FERMIONIC F2 + F3

**F2 — `--expectation-fermionic` on the CLI** (42c663b, 6ff06e1). Takes the
F1 text, maps through Jordan–Wigner, dispatches to whichever expectation
backend was chosen (statevector, MPS, paulipro, majoranaprop). The
certificate JSON records `"mapping": "jordan_wigner"` next to the parsed
operator, so a second mapper later is an enum value, not a migration.
Mutually exclusive with `--expectation`.

| leg | what | evidence |
|---|---|---|
| value | fermionic spelling vs the same operator hand-spelled as Paulis, on statevector, MPS, paulipro | identical values, identical certificate bounds |
| §1.3 basis guard | `a†₁a₃ + h.c.` through `--expectation-fermionic`: ladder-coefficient L1 is **2.0**, the certificate's `observable_range` must be **1.0** — the Pauli-image L1, the basis the engine actually propagates | `fermionic_certificate_basis.rs::fermionic_certificate_range_is_the_pauli_l1_not_the_ladder_l1`; a computation in the wrong basis moves the vacuity gate silently, and this test is the only thing that sees it |
| same certificate | fermionic vs hand-spelled Pauli, field for field | `fermionic_certificate_basis.rs::…matches_the_hand_spelled_pauli_image` |

**F3 — direct Majorana seeding in majoranaprop** (this commit). A ladder
operator maps into the engine's native Majorana basis without a Pauli
detour: `engine::seed_from_fermionic`, each ladder product expanded as a
Majorana monomial product in the engine's own basis. The
certificate carries `seed_basis: "ladder" | "pauli"`, written by the
engine's `finish` from which door called it — the engine's own record, not
a CLI flag next to the dispatch (that was the first draft of the CLI test,
and it survived the mutation; the certificate field does not).

| leg | what | evidence |
|---|---|---|
| (i) | both doors reach the **same monomial seeds**, keys and coefficients bit-identical, across every symmetrised ladder product of length 1–3 on 3 modes with a complex coefficient — **258** products | `fermionic_seed.rs::direct_seed_matches_the_pauli_seed_bit_for_bit_across_the_product_sweep` |
| (ii) | both doors report the same certificate on that sweep — counts exactly, sums to 1e-12 (`MajoranaSum` is a `HashMap`; even the Pauli door is not bit-reproducible against itself on sums) | `…both_doors_report_the_same_certificate_on_the_sweep` |
| (iii) | direct seed vs the dense statevector oracle | `…direct_seed_agrees_with_the_statevector_oracle` |
| (iv) | the direct door builds strictly fewer intermediates: 12 partial products vs 12 + 6 site products for the JW detour on the same operator | `…direct_seed_builds_strictly_fewer_intermediates_than_the_pauli_detour` |
| CLI | fermionic spelling on majoranaprop reports `seed_basis: "ladder"`, the hand Pauli image `"pauli"`, same value and same `observable_range` | `omega-cli/tests/fermionic_direct_seed.rs` |

Refusals are the engine's, naming the offender: non-Hermitian input names the
monomial, a mode outside the circuit names the mode. Dephasing drops exactly
the X- or Y-monomials in either basis.

**Mutation proof.** Pointing the majoranaprop `Some(op) =>` dispatch arm back
at `expectation_with_certificate` reddens the CLI `seed_basis` assertion — the
engine reports `"pauli"` because that is what it was given.

## 4e. Landed 2026-09-26 — PLAN-QUDIT Q0 (DITQASM verified, licences)

No engine code. The deliverable is evidence, not numbers: eight DITQASM
fixtures under `crates/omega-parser/tests/fixtures/ditqasm/` **written by
`mqt.qudits` 0.5.2 itself** (`tools/ditqasm_xcheck/gen_fixtures.py`, provenance
in `PROVENANCE.json`, two hand-written inputs labelled as such), and a harness
(`reparse.py`) that re-parses them under the library that wrote them.
`ci.sh` runs it with a Python that imports `mqt.qudits` and registers a skip
otherwise (`make -C crates/omega-bridges/python mqt-venv`).

| §3e.1a unknown | answer (from `qasm.py`, confirmed by fixture) |
|---|---|
| `qreg matter [2];` | two qudits of `d = 2`; MQT re-emits `[2][2,2]` |
| `h`/`x`/`z` vs `cx` | `h`/`x`/`z` generalised (Fourier, `X_d`, `Z_d`); **`cx` is an embedded two-level gate with four params**, `csum` is the generalised CX |
| `ctl` level list | per control, positional with the control list |
| version | locked to the literal `DITQASM 2.0;`; `3.0`/`2.1`/`OPENQASM` are `SyntaxError` |

**Licences** (GitHub `repos` + `/license`, `LICENSE` first line read): MQT MIT,
QuickQudits Apache-2.0, QuForge Apache-2.0; **QuDiet AGPL-3.0, Sdim GPL-3.0,
qiskit-qudits AGPL-3.0** — three of six copyleft, black-box result oracles
only, never read. Sdim's paper cites no repository; found via PyPI.

**Two upstream defects, pinned** so a fix is noticed rather than absorbed: MQT's
`ctl` writer emits the target index for every control and cannot re-parse its
own output; `from_qasm` drops `rxy`/`rz` parameters (tuple vs list in
`R.validate_parameter`, `cx` unaffected). So "round-trips under MQT" is not an
acceptance criterion for Q1 — byte identity against the emitted fixtures is.

## 4f. Landed 2026-09-27 — PLAN-QUDIT Q1: the IR knows its wire dimensions, and every engine refuses d ≠ 2

**IR.** `CircuitIR` gains `qudit_registers: Vec<QuditRegister>` (name, start
wire, per-wire dims), empty for every constructor and every OPENQASM source, so
a wire not covered is a qubit. `wire_dim(w)`, `wire_dims()`, `first_qudit()`,
and the one sentence every door speaks: `refuse_qudits(engine)` → typed
`OmegaError::Unsupported` naming **engine, register, wire and dimension**.
Eleven struct literals across the workspace gained the field; `CircuitIR::new`
callers (~110 files) were untouched.

**Parser.** `DITQASM 2.0;` is a header token beside `OPENQASM`; `qreg q
[3][3,2,5];` is one optional pest group on `qreg_decl`. Lowering refuses a
group under an `OPENQASM` header (it is DITQASM's syntax, not a fourth
dialect), a group whose length ≠ size, and `d < 2` — each naming the register.
Gates only DITQASM spells (`csum`, `rxy`, four-param `cx`, `ctl`) are **not**
in Q1, per §3e.1a's cost analysis; Q0's fixtures fail at the *gate*, never at
the register line, and `tests/ditqasm_registers.rs` pins that against bytes
`mqt.qudits` wrote: `[1][3]` → qutrit; bare `[2]` → two qubits, no register
recorded; `[2][2,2]` → recorded, still a qubit circuit; `matter [2][3,3]` +
`field [1][7]` → offsets 0 and 2 in declaration order; `[3][3,2,5]` answers
per wire.

**Every door.** `refuse_qudits` is the first line of every `&CircuitIR`-taking
trait method on every `Backend` impl — 7 doors on `StatevectorBackend`, 4 on
MPS (incl. Noisy), 3 pauli, 3 pauliprop, 3 majoranaprop, 3 sector, 4 photonics,
9 CUDA, 9 Metal, 5 OpenCL, 2 `TchBackend` — plus both inherent
`expectation_with_certificate` doors AND majoranaprop's fermionic door (which
does not share the Pauli door's entrance: the CLI matrix caught it running a
qutrit circuit at 0.585, exit 0, before it had the check). Also the plugin ABI
(`circuit_ffi_limitation`, host-side, before flattening), the QPY writer (a
d = 3 wire would be written and read back as a qubit), and the `aria-verify`
oracle (an oracle that ran qudits as qubits would agree with a backend making
the same mistake). Server, WASM and refplugin paths run the CPU engines
directly and inherit the door.

| leg | what | evidence |
|---|---|---|
| A10 matrix | one refusal row **per backend per mode**: statevector/mps/mps:8/pauli sampling; sector on a hopping body (qudit refusal wins over its gate check); `--statevector`; expectation on statevector/mps/pauli/pauliprop/majoranaprop/sector; `--expectation-fermionic`; `--gradient`; `--backend auto` on Clifford (→pauli) and non-Clifford (→statevector) bodies; a mixed `[3,2]` register names wire 0 not wire 1 | `omega-cli/tests/qudits_are_refused_by_every_backend.rs`, 10 tests; each asserts **non-zero exit** AND the engine sentence (`{engine}:`, `register 'q'`, `wire 0`, `dimension 3`, `qubit-only`) AND no `<O> =`/counts on stdout |
| control | the identical file declared `[2][2,2]` **runs** on every backend — without this every row could pass because DITQASM was unreadable | same file |
| GPU arms | `--device cuda` on a qutrit circuit: `cuda-statevector: register 'q' declares dimension 3 …` refused **before `allocate`**, forwarded through `refuse_device_fallback` (explicit device ⇒ no CPU substitution). Metal/OpenCL are the same call at the same place; the test names which arm it could not run | **run for real on akilles, RTX PRO 6000** (`--features cuda`, `gpu arms checked: 1`); registered note, not silence, elsewhere |
| IR | typed `Unsupported`, first non-qubit wire in a mixed register, all-2 group not refused, plugin ABI same sentence, `clone` keeps the registers | `omega-core/tests/qudit_refusal.rs`, 6 tests |

**Mutation proof.** Deleting the `refuse_qudits` line from `StatevectorBackend::execute`
alone reddens 4 of 10 matrix rows (`--shots`, `--statevector`, mixed `[3,2]`,
`auto` non-Clifford) with "a d=3 circuit EXITED 0 — it was run as qubits";
restored, `cmp`-identical. Workspace `clippy --all-targets -D warnings` exit 0
after `touch` (the F3 lesson), also under `--features cuda`. Q0's
`reparse.py` still green; fixture bytes untouched.

**Not done here, on purpose:** no engine evolves a qudit — Q2 is the first
that does (§4g). No QASM2 *text* emitter exists for `CircuitIR` in-tree (only
aria-core's own `Circuit`), so the plan's "QASM2 emitter refuses" clause had
nothing to attach to; the QPY writer is the one serializer, and it refuses.

## 4g. Landed 2026-09-27 — PLAN-QUDIT Q2: `quditsv`, the exact dense mixed-radix engine

**Crate.** `crates/omega-backend-quditsv` (`capacity.rs`, `gates.rs`,
`sim.rs`; 861 lines): `Π dᵢ` amplitudes read from `CircuitIR::wire_dims()`
(Q1), wire 0 least significant exactly as every qubit engine here, so on an
all-`d = 2` circuit the state is the dense qubit statevector **index for
index**. The `d = 2` gate matrices are the statevector backend's own `gates`
table (this crate never calls its apply path), so the differential below
tests the mixed-radix addressing, not two hand-typed tables. Generalised
one-wire gates keep their qubit names — `H` → Fourier `F_d`, `X` → shift,
`Z` → clock (mqt.qudits' semantics, pinned in Q0; each is the qubit gate at
`d = 2`) — plus two new IR variants with no qubit spelling: `rxy(i, j, θ, φ)`
on a level pair and `csum` (the generalised CX, control first). Every other
exhaustive `GateKind` match in the workspace (plugin ABI, QPY writer,
majoranaprop, CUDA adjoint + lib, the CLI's serializer) gained a **named**
refusal arm, not a wildcard; `aria-verify`'s oracle realises both at `d = 2`
independently of this crate's table, so the embedding has a second reading.
A qubit gate on a `d > 2` wire is refused by name with the way out (`CX is a
qubit gate … csum`); sampling, Pauli observables on qudit wires, channels,
conditions and mid-circuit measurement are refused, never approximated.

**Capacity, and why the certificate is honest.** Hard ceiling
`MAX_PRODUCT_DIM = 2^24` amplitudes (256 MiB), checked by `u128`
checked-multiply **before allocation**; the refusal carries the product and
the ceiling in bytes, and a product that overflows 128 bits is its own
refusal. Then a host-memory comparison (`OMEGA_QUDITSV_OVERSUBSCRIBE=1` skips
only that; the ceiling always holds). Nothing is ever discarded, so the
certificate is `exact: true`, `dropped_mass: 0` **as a bound**
(`dropped_mass_is_a_bound: true`), plus `dims` and `product_dim` — attached
to the JSON as `quditsv`.

**CLI.** `--backend quditsv` (alias `qudit`) for `--statevector` and
`--expectation`; `--backend auto` now routes any circuit with a qudit
register to it and says so in the preamble (`auto: qudit register 'q'
(dimension 3 on wire 0) -> quditsv (exact, dense mixed-radix)`).

| leg | what | evidence |
|---|---|---|
| (i) anchors | `F_3\|0⟩` uniform over three levels; `X_3` cycles `\|2⟩ → \|0⟩`; `Z_3` puts `ω^k` on `\|k⟩`; `rxy(0,1)` is the identity on `\|2⟩` and its image of `\|0⟩` matches mqt.qudits' own numbers (`0.939373, 0.101333 − 0.327583i, 0`, 1e-6); `csum` adds mod d; a `[3,2,5]` register addresses the right wire; the `Backend` door returns the full product state. Every one needs amplitude at level 2 to *exist*, so a secret d = 2 delegation fails on a value, not on plumbing | `omega-backend-quditsv/tests/anchors.rs`, 7 tests |
| (ii) d = 2 differential | 12 random multi-wire and 20 single-wire qubit circuits (seed 20260927), expectation on every weight-≤ 2 Pauli, and sampling — vs `StatevectorBackend` at **1e-12**, index for index | `tests/qubit_differential.rs`, 4 tests |
| refusals | over-ceiling refused by arithmetic before allocation; qubit gate on a qudit wire names gate and alternative; `rxy` levels validated; sampling / Pauli-on-qudit / channels / conditions / mid-circuit refused naming the register | `tests/refusals.rs`, 6 tests |
| (iii) external oracles | 6 DITQASM circuits vs **mqt.qudits** 0.5.2 (MIT; mixed `[3,2,5]`, the non-Clifford `rxy`): max\|Δ\| ≤ 3.76e-16 at 1e-9, global phase aligned. 3 Clifford circuits vs **QuickQudits** 1.0.1 (Apache-2.0, ungated by Q0): ≤ 2.28e-16. Both oracles index big-endian, omega little-endian; `--pin-order` proves the reindexing is not a no-op — the un-reindexed `asymmetric_pin` comparison differs by 1.00 and must | `tools/quditsv_xcheck/xcheck.py`; `ci.sh` runs it under each venv that imports its oracle, a **registered skip** for each that is absent. Ran under both venvs on **both** boxes: akilles (linux x86_64) and andromeda (Apple M4, `make mqt-venv` / `make quickqudits-venv` under `omega-hostgate run --host-bytes 4G`) — mqt.qudits 6/6 at ≤ 2.73e-16 and 18/18 `reparse.py`, QuickQudits 3/3 at ≤ 1.71e-16 on andromeda, 2026-09-30. The cross-language claim rests on two boxes and two independent oracles; no leg is a skip anywhere |
| CLI matrix | `auto` on a qutrit body with `cx` lands on quditsv AND is still refused there (`quditsv: … CX is a qubit gate`, names `csum`); a valid qutrit body (`h`, `x`, `csum`) under `auto` exits 0 with the state | `omega-cli/tests/qudits_are_refused_by_every_backend.rs`, 10 tests; Q1's refusal rows unchanged |

**Mutation proof.** `sim.rs:84` `let stride = self.strides[w]` → `1` (every
one-wire gate lands on wire 0): `mixed_radix_strides_address_the_right_wire`
and `csum_adds_the_control_digit_to_the_target_mod_d` red, 2 of 7 anchors,
cargo halted there; restored, `cmp`-identical, 7/4/6 green again. `clippy
--all-targets -D warnings` exit 0 on quditsv, cli, core, parser.

**Not done here, on purpose:** no gradients, no noise, no GPU — "laptop-sized"
is the role; Q3's MPS is the lane that scales, and it is measured against
this. DITQASM's `ctl` (level-controlled application) and four-param `cx` are
still not parsed. QuickQudits cannot express `rxy` or mixed dimensions, so
`rxy_only`, `mixed_325` and `asymmetric_pin` carry the mqt leg only — printed
as `----` rows, not skipped silently.

## 4h. Landed 2026-09-27 — PLAN-QUDIT Q3.1: the MPS core carries a per-site dimension (WIP, door still closed)

**What changed** (`a98e7e1`, `crates/omega-backend-mps/src/mps.rs` +
`tests/qudit_core.rs`, 907/119 lines): `MpsTensor` has the `phys` field
§3d.1 said it lacked; `Mps` carries `dims` and is built with `with_dims`.
`apply_1`, `apply_2` (adjacent), `apply_2_distant` and `to_statevector` are
dimension-generic — mixed radix, site 0 least significant, the same index
convention as `quditsv` — and take slices; the fixed-size `[Complex64; 4]` /
`[Complex64; 16]` APIs are now wrappers that assert `d = 2`. A SWAP between
sites of different dimension reshapes the pair (`(d0,d1)` in, `(d1,d0)` out)
and carries the dimensions with the state; the distant-pair path swaps in,
applies, swaps back in reverse order, so `dims` is restored. The SVD core was
already generic (`truncated_svd_flat`, STATUS §5 item 3) and is reused as-is.
The Metal/CUDA `Contract2qFn` hook is consulted only for a qubit pair that
stays a qubit pair.

**`d = 2` is bit-identical**, not "close": 120 random runs (5 and 7 qubits,
fixed and adaptive `χ`, 57 of them truncating, dense random two-qubit gates
on adjacent and reversed pairs) against a verbatim copy of `983685a`'s
kernels — statevector, `discarded_weight`, `fidelity_estimate`,
`max_bond_reached` all `==`. The throwaway harness was deleted after the
check; the in-tree pin is `d2_slice_api_is_bit_identical_to_the_qubit_api`.

**Tests** (13 in `qudit_core.rs`): pure qutrit chain vs a dense inline
reference at 1e-12 (exact and χ-capped); `[3,2,4,2]` distant pairs both
orientations, `dims` and every tensor's `phys` checked after every gate;
`[3,3,3,3]` at `χ = 2` asserts `discarded_weight > 0` and raw norm below 1
(the certificate is fed `d²` blocks, not `4`); refusals by name for every
qubit-only API on a qudit. **Mutation:** `swap_matrix_dims` → identity
reddened exactly the three mixed-radix dense comparisons and nothing else;
restored, `cmp`-identical.

**As of this entry the door was still closed** — `MpsBackend` in `sim.rs`
still called `refuse_qudits`, and sampling, `measure_site` and the
product-operator expectation refused any non-qubit chain by assertion. Q3.2
(§4i, three days later) opened it.

## 4i. Landed 2026-09-30 — PLAN-QUDIT Q3.2: `MpsBackend` evolves qudit circuits, differential against `quditsv`

**What changed** (`49b4bf0` + merge `c06a749`; `mps.rs` 213, `sim.rs` 293,
`tests/qudit_backend.rs` 1182 lines, CLI test 183 lines). `execute` no longer
calls `refuse_qudits`: it reads `circuit.wire_dims()`, builds the chain with
`Mps::with_dims`, and dispatches by gate — the generalised gates (`H` =
Fourier, `X` = shift, `Z` = clock, `Rxy`, `CSum`) on any wire, the qubit
gates on `d = 2` wires only. Every qubit-only construct on a `d ≠ 2` wire is
**refused by name, with the same sentence `quditsv` gives after the engine
prefix** (`CX is a qubit gate and wire 0 has dimension 3`, `observable Z0
names wire 0, which has dimension 3`, …); pinned by
`mps_and_quditsv_refuse_the_same_constructs_with_the_same_sentence`. The
gate *matrices* for `d ≠ 2` come from `omega-backend-quditsv::gates` — a
new regular dependency edge, taken on purpose: with one matrix table the
differential is a test of the MPS contraction, truncation and SWAP network,
not of two transcriptions of the same formulas. Only `gates` is used; the
MPS crate never calls quditsv's apply path.

**Plan test (i), exact regime** — `random_mixed_radix_circuits_match_quditsv_in_the_exact_regime`:
dims drawn from {2, 3, 4}, `n = 4..6`, depth 30, adjacent *and* distant
pairs, at a bond that never truncates, statevector vs `quditsv::sim::run` at
1e-10; plus a pure qutrit chain and a `[3,2,4,2]` chain. Before Q3.2 every
one of these runs was refused at the door, so a pass requires the new path
to have executed (the plan's A10 property). Through the CLI,
`omega-run --backend mps --statevector` on a mixed DITQASM register now
agrees with `--backend quditsv` (`mps_statevector_of_a_mixed_qudit_circuit_agrees_with_quditsv`).

**Plan test (ii), the truncation contract transfers to `d = 3`** —
`qutrit_truncation_keeps_the_qubit_lanes_certificate_contract`: the qubit
file's contract verbatim on a qutrit brickwork, `n = 8`, 8 layers (≥ 40
entangling-or-rotation ops), `χ ∈ {2, 3, 4, 6, 8, 12}`, three seeds;
measured `1 − F_true` against `quditsv` vs the certified quantities, contract
direction asserted on every run. `the_ceiling_is_straddled_by_a_qutrit_runs_own_certificate`
is the qubit straddle test at `d = 3`. `qutrit_closed_form_spectrum_is_certified_exactly`
is the half the deep sweep cannot exercise: a two-site qutrit state with
Schmidt spectrum `{c₁, s₁c₂, s₁s₂}` — three values, which a qubit
bookkeeping cannot hold — where `χ = 2` must drop exactly `s₁²s₂²` and
`χ = 1` exactly `s₁²`. (In the deep sweep every run has `Σεᵢ > 2`, so the
field relation `Π(1 − εᵢ) ≥ 1 − Σεᵢ` is vacuous there; the closed form is
what pins the bookkeeping.)

**Plan test (iv), mutation — run twice, two mutants.** The subagent's:
`discarded_weight += 0.5 * rel_discarded` — the sweep stayed green (vacuous
at that depth, as above) and only the closed-form test tripped, which is why
that test exists. Mine, on the merged tree: `discarded_weight += 0.0 *
rel_discarded` at both accumulation sites (`mps.rs:340`, `:552`) → exactly
`qutrit_truncation_keeps_the_qubit_lanes_certificate_contract`,
`the_ceiling_is_straddled_by_a_qutrit_runs_own_certificate` and
`qutrit_closed_form_spectrum_is_certified_exactly` FAILED, the other ten
passed; restored, tree clean. **(iii)** the qubit estate is untouched:
`omega-backend-mps` 66 + 10 + 4 + 3 + 13 + 13 + 5 = 114 passed, clippy
`-D warnings` clean.

**Sampling emits digits, not bits.** `sample_bits_with_envs_into` draws one
digit per site, `< d` on every site; `sampled_digits_match_quditsv_marginals_on_a_mixed_chain`
runs `[3,2,4,2,3]` × 20 000 shots against the exact per-site marginals *and*
the full joint at ≤ 0.02, and asserts a digit ≥ 2 was actually drawn.
`MidCircuitMode::Collapse` on a qutrit wire, with gates after the
measurement, matches the exact vector projected by hand onto the drawn level
(all three levels drawn over the seeds); `Reset` returns a qutrit in level 2
to zero. The Pauli expectation on the *qubit* wires of a mixed chain matches
`quditsv` (`Z1` through the CLI, same number to 1e-10); a Pauli on a `d ≠ 2`
wire is refused by name.

**What is refused, and why — the result type, not the engine.** `--shots`
on a qudit circuit is refused by `mps` with the sentence `quditsv` uses:
`` `Counts` outcomes are bit strings ``. The digits are sampled (above); the
public result type cannot carry them, and encoding digits into bits is not
part of Q3. In `Collapse` mode, measuring a `d ≠ 2` wire into a classical
bit that some condition *reads* is refused (`refuse_digit_into_a_read_cbit`):
a stored `2` would be read as `bit & 1 = 0` and the conditioned gate would
fire on a value nobody measured; where no condition reads the bit the
collapse runs. `the_documented_differences_from_quditsv` pins the other
direction — what MPS runs that `quditsv` refuses (trajectory channels,
mid-circuit measurement, classical control): capabilities the engine already
had on qubits, not disagreements about a gate's meaning.

**Not in Q3.2, stated:** `NoisyMpsBackend` still calls `refuse_qudits`
(`sim.rs:898`); the Metal/CUDA `Contract2qFn` hook is consulted only for a
qubit pair that stays a qubit pair (plan: CPU only, §5); the digit readout
into `Counts` waits on an `Outcome` that can hold a digit. Q4 (measure it on
named boxes) is §4j.

## 4j. Landed 2026-09-30 — PLAN-QUDIT Q4: the qudit MPS measured on akilles, MQT Qudits on the same circuits

**The measurement door first** (`b0948c1`). `Backend::expectation` takes a
Pauli observable and refuses a `d ≠ 2` wire by name, so until this phase the
only number a qutrit chain could yield was `execute(shots: None)` →
`to_statevector`, a `3^n` dense reconstruction that is not MPS evolution and
that the bond does not bound — the trap §5.16 records. `MpsBackend::expectation_site_operators`
takes one `d_q × d_q` operator per wire, checks count and shape against
`wire_dims()` before evolving, and contracts on the chain (`O(χ²)`). Prepared
as `expectation` is: analytic `Reset` refused; a measurement nothing reads is
elided; one a condition reads is deferred and its wire's operator dephased
`Σ_k|k⟩⟨k|O|k⟩⟨k|` (on `d ≠ 2` the read case is refused earlier, so the
dephasing fires on the qubit wires of a mixed chain — both halves pinned
against a hand-deferred circuit). Against a dense hand contraction at 1e-10
over random `[3,2,3,2]`-family circuits; reproduces the norm and every exact
marginal; refuses wrong count and wrong shape naming the wire.

**The harness** (`crates/omega-backend-mps/examples/qudit_chain_profile.rs`,
`tools/qudit_chain_xcheck/tnsim_rows.py`, README there). Qutrit brickwork —
random `rxy` level rotations, `csum` bricks, one distant `csum` per layer,
the family the Q3 contract sweep certified — measured through the door
above with `⟨S_z⟩ = diag(1, 0, −1)` on the middle site. A10, in the harness
itself: `--d 2` exits 2 (a qubit table under a qudit heading); a row with no
certificate from any run of its circuit exits 3. **The second gate fired on
the first real run:** the engine's early abort stops evolution when the
running certificate crosses the ceiling and returns *before* `record_stats`,
so a refused attempt had no stats and the harness would have published a row
with none. Fixed by taking a refused row's certificate — and its uncertified
value — from a second, complete, ceiling-lifted run of the same circuit; the
refusing attempt's own certificate is quoted from its sentence in the
REFUSED cell. That is the receipt A10 asks for: the harness consumed its own
evidence and refused to print. The external leg builds the identical
circuits in mqt.qudits through its API (`from_qasm` drops `rxy` params, Q0
D2), runs `tnsim` per circuit under a hard `RLIMIT_AS` and a timeout, and
**fails** if tnsim disagrees with `quditsv` at 1e-9 or with any certified
MPS row at 1e-8. `ci.sh` runs the small profile on every box and the tnsim
leg where the venv exists (both boxes now have it, §4g).

**Small profile on akilles** (32 threads, seed 20260930, 6 layers). RSS is
the process high-water mark, cumulative down the table. `REFUSED` quotes the
certificate at the point the engine stopped; `discarded_weight` /
`fidelity_est` / `max_bond` are the complete ceiling-lifted run's, and
"uncertified" is its value.

| box | n | ops | χ | wall ms | peak RSS | discarded_weight | fidelity_est | ⟨S_z⟩ certified | exact (quditsv) | \|Δ\| | uncertified |
|---|---|---|---|---|---|---|---|---|---|---|---|
| akilles | 6 | 57 | 4 | 0.1 | 3.3 MiB | 1.237 | 0.2613 | REFUSED (cert 1.219e-6, bond 4) | −0.0611609561 | 1.43e-1 (uncert.) | −0.2045023423 |
| akilles | 6 | 57 | 8 | 0.5 | 3.3 MiB | 3.885e-1 | 0.6618 | REFUSED (cert 1.020e-4) | −0.0611609561 | 1.12e-2 (uncert.) | −0.0724007101 |
| akilles | 6 | 57 | 16 | 1.1 | 3.3 MiB | 2.990e-2 | 0.9703 | REFUSED (cert 1.848e-3) | −0.0611609561 | 2.66e-3 (uncert.) | −0.0585016487 |
| akilles | 6 | 57 | 27 | 2.4 | 3.3 MiB | 0 | 1.0000 | **−0.0611609561** | −0.0611609561 | 2.64e-16 | — |
| akilles | 8 | 75 | 4 | 0.3 | 3.5 MiB | 3.375 | 0.0247 | REFUSED (cert 3.491e-4) | 0.0073518323 | 2.23e-4 (uncert.) | 0.0071285894 |
| akilles | 8 | 75 | 8 | 0.7 | 3.5 MiB | 1.645 | 0.1647 | REFUSED (cert 2.038e-2) | 0.0073518323 | 2.01e-2 (uncert.) | −0.0127629768 |
| akilles | 8 | 75 | 16 | 2.6 | 3.5 MiB | 6.289e-1 | 0.5073 | REFUSED (cert 2.960e-2) | 0.0073518323 | 1.99e-2 (uncert.) | 0.0272251920 |
| akilles | 8 | 75 | 27 | 3.5 | 3.8 MiB | 1.591e-1 | 0.8426 | REFUSED (cert 1.155e-2) | 0.0073518323 | 4.14e-4 (uncert.) | 0.0077653935 |
| akilles | 10 | 93 | 4 | 0.4 | 6.0 MiB | 5.119 | 0.0031 | REFUSED (cert 9.667e-2) | −0.0000671006 | 2.48e-1 (uncert.) | 0.2479340186 |
| akilles | 10 | 93 | 8 | 1.3 | 6.0 MiB | 2.900 | 0.0402 | REFUSED (cert 7.385e-2) | −0.0000671006 | 3.16e-2 (uncert.) | 0.0315331691 |
| akilles | 10 | 93 | 16 | 8.0 | 6.0 MiB | 1.661 | 0.1612 | REFUSED (cert 4.081e-1) | −0.0000671006 | 4.25e-2 (uncert.) | −0.0425541667 |
| akilles | 10 | 93 | 27 | 16.2 | 6.0 MiB | 8.392e-1 | 0.3957 | REFUSED (cert 8.330e-2) | −0.0000671006 | 8.67e-3 (uncert.) | −0.0087363232 |
| akilles | 12 | 111 | 4 | 0.2 | 27.9 MiB | 2.170 | 0.0912 | REFUSED (cert 4.611e-2) | −0.2356675998 | 1.33e-1 (uncert.) | −0.3690051244 |
| akilles | 12 | 111 | 8 | 0.1 | 27.9 MiB | 5.230e-1 | 0.5809 | REFUSED (cert 1.194e-3) | −0.2356675998 | 5.36e-2 (uncert.) | −0.2893084585 |
| akilles | 12 | 111 | 16 | 0.6 | 27.9 MiB | 3.793e-2 | 0.9624 | REFUSED (cert 3.235e-3) | −0.2356675998 | 2.16e-3 (uncert.) | −0.2378259484 |
| akilles | 12 | 111 | 27 | 2.9 | 27.9 MiB | 5.8e-32 | 1.0000 | **−0.2356675998** | −0.2356675998 | 4.22e-15 | — |

**MQT Qudits `tnsim` 0.5.2 on the same four circuits, on akilles:** agrees
with `quditsv` on all four at ≤ 3.6e-15, and with both certified MPS rows at
≤ 6.4e-16 — `OK: tnsim agrees on 6 comparisons`. tnsim wall / peak RSS:
n=6 29 ms / 132 MiB; **n=8 8 726 ms / 4.2 GiB**; n=10 251 ms / 278 MiB;
n=12 24 ms / 169 MiB. Its cost is set by the contraction order the circuit
happens to induce, not by `n` — the one distant `csum` per layer is what the
n=8 instance paid for. Same shape as the §3c.6 P4 receipt.

**Large profile on akilles** (8 layers, χ ∈ {64, 128, 256}) — the honest
table. Every MPS row is **refused** under the 1e-6 ceiling; `quditsv` cannot
hold `3^16`; and tnsim refuses all three within a hard 40 GiB `RLIMIT_AS`
(n=16 fails a 5.77 GiB allocation with the address space already committed —
a first attempt with no rlimit reached ~49 GB resident before it was stopped
for exceeding this box's 40 GB rule; n=20 asks for a 468 GiB array, n=24 for
52 GiB). So at n ≥ 16 there is **no certified number on either side**, and
the uncertified MPS values below are published as exactly that.

| box | n | ops | χ | wall ms | peak RSS | discarded_weight | fidelity_est | max_bond | REFUSED at | uncertified ⟨S_z⟩ |
|---|---|---|---|---|---|---|---|---|---|---|
| akilles | 16 | 196 | 64 | 41.7 | 8.1 MiB | 4.465 | 0.0059 | 64 | cert 6.016e-6 | 0.0446467278 |
| akilles | 16 | 196 | 128 | 202.9 | 24.9 MiB | 2.448 | 0.0617 | 128 | cert 3.466e-2 | −0.0107659258 |
| akilles | 16 | 196 | 256 | 2 081.6 | 99.7 MiB | 9.201e-1 | 0.3084 | 256 | cert 6.268e-2 | −0.0064346166 |
| akilles | 20 | 244 | 64 | 66.2 | 99.7 MiB | 2.453 | 0.0692 | 64 | cert 7.231e-3 | 0.0025514842 |
| akilles | 20 | 244 | 128 | 641.8 | 99.7 MiB | 1.117 | 0.3020 | 128 | cert 5.583e-2 | 0.0128455545 |
| akilles | 20 | 244 | 256 | 2 425.7 | 99.7 MiB | 3.202e-1 | 0.7122 | 256 | cert 3.485e-2 | 0.0084167651 |
| akilles | 24 | 292 | 64 | 458.3 | 99.7 MiB | 7.173 | 0.0003 | 64 | cert 6.800e-2 | 0.0715043840 |
| akilles | 24 | 292 | 128 | 407.9 | 99.7 MiB | 5.015 | 0.0037 | 128 | cert 2.170e-2 | 0.0489779787 |
| akilles | 24 | 292 | 256 | 4 132.2 | 126.9 MiB | 2.894 | 0.0361 | 256 | cert 6.380e-2 | 0.0114017735 |

**What the numbers say.** (1) The plan's premise that the large rows are
"RAM-bound on akilles" was wrong for this workload: 24 qutrits at χ = 256
peak at 127 MiB and 4.1 s. RAM is the *external* engine's problem — dense
`3^n`. (2) The certificate does its job in the direction the contract
promises: on every small row with an exact reference, the uncertified error
tracks the certificate's order (χ=27 at n=12: certificate 5.8e-32, error
4e-15; n=10 χ=4: certificate 5.1, error 0.25), and the two rows the
certificate admitted are exact to 1e-15. It stays pessimistic where the
qubit lane was (n=8 χ=4: certificate 3.4 against an error of 2.2e-4).
(3) At 8 layers of random qutrit brickwork, χ = 256 is not enough for a
1e-6 certificate at n ≥ 16; the certificate shrinks 2–5× per doubling of χ
and is still 3e4× over the ceiling at 256. That is the regime boundary, and
the engine refused rather than estimated it. (4) The cross-language claim
now rests on two boxes and two independent oracles for the exact regime
(§4g), and on two engines' *refusals* beyond it.

**Small profile on andromeda** (Apple M4, 16 GB, same seed; run and
reported by andromeda's session 2026-09-30). The certified set is the same
`{n = 6, n = 12}` at χ = 27: |Δ| vs exact **1.04e-16** and **2.80e-15**; every
other row refused with its uncertified value and error beside it, certificates
1.02e-4 to 4.08e-1 against the 1e-6 ceiling; peak RSS 32.8 MiB at n = 12;
`--d 2` refused. **tnsim on andromeda:** all 6 comparisons agree — n = 6
9.02e-17, n = 8 0.00, n = 10 2.78e-17, n = 12 2.83e-15; wall 13–10 611 ms
with **n = 8 the outlier at 10.6 s and 3.7 GiB peak** against 171 MiB at
n = 12 — the same contraction-order effect as on akilles, on a different
ISA. So the Q4 numbers stand on two boxes and, for the exact regime, on
three engines (quditsv, mps, tnsim).

**Two things the andromeda run found, both about guards, both recorded
rather than smoothed over.** (1) `RLIMIT_AS` cannot be set on macOS at all
("current limit exceeds maximum limit" for every value), so the cap added
after the 49 GB incident turned the tnsim leg into a hard failure on every
Mac — the capacity-test shape again, a correct safety measure that was not
portable. The worker now reports the failed `setrlimit` and runs uncapped.
(2) **Nothing in the estate caps a run at runtime, and a PID-level watchdog
would not either.** The akilles tnsim run was launched under
`omega-hostgate run --host-bytes 40G` and reached ~49 GB resident: hostgate
is an admission *ledger* by its own contract (`lib.rs`: "no cgroup
enforcement, no preemption… cooperative processes on one box") — the 40 G is
a declaration the child is trusted to honour, and nothing measures it.
andromeda then showed the second half: a watchdog on the script's PID saw a
17 MB peak while the worker subprocess held 3.7 GiB. So a runtime cap has to
exist *and* has to measure the process tree (cgroup on Linux; an explicit
child walk on macOS). Raised as §5 item 18 against `omega-hostgate` rather
than patched per caller.

**Not done here, on purpose:** the `large` profile is not CI
(4 s of MPS, but the tnsim leg's failed allocations are real work on a
16 GB box); no GPU, no `NoisyMpsBackend`, no digit `Counts` — unchanged from
§4i.

## 5. Known open — recorded, not fixed

> **Audited 2026-08-26.** Every entry re-checked against the code rather than
> carried forward. Four clauses had gone stale — items 6, 10 (both halves), the
> CUDA `CY` correction nested under 10, and 11 — and are struck through with the
> evidence inline. Two were closed by work that never came back to update this
> list; two were closed by someone else's work months earlier.
>
> Worth stating because of *how* they were found: item 10 is one sentence
> making two claims, and they went stale at different times. Auditing the entry
> as a unit would have kept whichever half was still true and preserved the
> other. Check each clause, not each bullet — a stale open item is worse than
> no list, since it sends the next reader to re-solve closed work and makes the
> live entries look equally doubtful.

1. ~~**CUDA is unverified for the 2026-08-15 fixes.**~~ **CLOSED 2026-08-16 on
   a DGX Spark GB10** — not the RTX 6000 Pro box this expected, but the harder
   target (`aarch64`, sm_121, CUDA 13.0.88). Two of the blind edits did not
   compile; `LINUX-CUDA-VERIFICATION.md` §1 tabulated five and there were
   seven. Fixed in `fc1a549`, then f32 **and** f64 both run:
   `f64_vs_qiskit` worst |Δamplitude| **1.388e-16**, and the mandatory Qiskit
   differential agrees on **60 circuits at 4.441e-16**.
   `ARIA_CUDA=1 ARIA_QISKIT_XCHECK=1 ./ci.sh` → exit 0.
   ~~**Not fully closed:** the `lib.rs:858` collapse-counts *width* question is
   semantic, so compiling it answers nothing, and CUDA still has no equivalent
   of `collapse_counts_use_the_creg_width.rs`.~~ **Both clauses were stale, and
   answering them found a live defect.** Audited 2026-10-02 on an RTX PRO 6000
   (x86_64, sm_120, CUDA 12.9). See item 2's last paragraph.
2. ~~**CUDA's Reset criterion diverges** (refuses on random *outcome*, not
   entanglement)~~ **CLOSED — the entry outlived the fix.** Audited 2026-09-26
   against the code: `statevector-cuda/src/lib.rs:1719` calls
   `omega_backend_statevector::sim::reset_is_deterministic_within`, i.e. the
   same purity predicate as the CPU. The divergence the entry describes is
   past tense in the code's own words — `tests/reset_criterion.rs` opens "the
   divergence it **used to** have" — and that file's
   `unentangled_superposition_reset_is_accepted_and_matches_cpu` is the test
   that proves it: `H q0; Reset q0` has `p0 = 0.5` but purity 1, and CUDA now
   accepts it as CPU and Metal do.
   That test file is also worth reading for its own reason: it exists because
   the pre-existing `reset_matches_cpu` uses an *entangled* reset, which every
   backend refused before the change and still refuses after — green either
   way, so it could not have verified the fix. "A test that cannot fail is not
   evidence" is its wording, and it is the same defect class as §3f A10.
   ~~**Still genuinely open from the old item 1:** CUDA has no equivalent of
   `collapse_counts_use_the_creg_width.rs`, and the `lib.rs:858` collapse-counts
   *width* question is semantic, so compiling it answers nothing.~~ **CLOSED
   2026-10-02, and the clause was stale when it was written.** The equivalent
   is `statevector-cuda/tests/counts_width_and_collapse.rs`, which landed in
   `c12bf07`, the same merge that carried this clause. `lib.rs:858` has since
   drifted onto `apply_reset`'s doc comment. The width answer: CUDA has no
   collapse arm, and every CUDA counts path keys on the qubit register, which
   is `counts_outcome_width(c, false)` and correct for skip mode.
   **Answering it found a defect the clause did not name.** The collapse
   refusal looked for a `Measure` only. With shots and a declared creg the CPU
   keys collapse-mode counts on the classical register even when nothing writes
   it, so `creg c[1]; h q[1]; if (c == 0) x q[0];` read `{"0": 64}` on the CPU
   and `{"001": 40, "011": 24}` on CUDA. It was reachable: `omega-server` with
   `OMEGA_DEVICE=cuda` sends collapse mode straight to the device and answered
   HTTP 200 with the CUDA keys (a debug build panics on the width
   `debug_assert` instead). The CLI was never affected; it runs every
   `needs_collapse` circuit as CPU trajectories. Fixed: CUDA now refuses that
   shape, and the server answers HTTP 400 naming it. Test first, fix second:
   `collapse_without_a_measure_agrees_with_the_cpu`.
   Mutation evidence, each restored bit-for-bit:
   | mutation | test that went red |
   |---|---|
   | collapse guard `if false && …` | `collapse_with_a_measure_is_refused_not_answered` (`expect_err`) |
   | skip-mode key width `num_qubits` → `num_classical_bits` | `skip_mode_counts_key_on_the_qubit_register_width` |
   | new clause `keyed_on_creg && false` | `collapse_without_a_measure_agrees_with_the_cpu` |
   **OpenCL had it too, measured and fixed 2026-10-02** on the same card
   through the NVIDIA OpenCL driver: `{"001": 32, "011": 32}` against the
   CPU's `{"0": 64}`, served as HTTP 200 by `omega-server` with
   `OMEGA_DEVICE=opencl`. It now refuses the same shape and the server
   answers HTTP 400. The test is
   `statevector-opencl/tests/collapse_counts_agree_with_cpu.rs`, committed
   failing before the fix. Making the new guard always false turns it red.
   **Still open, not measured:** Metal (`lib.rs:1220`) has the same
   `Measure`-only guard and needs an Apple device to check.
3. ~~**Metal shots-mode `Reset` delegates to the CPU backend.**~~ **CLOSED
   2026-09-30 — the recorded fix was measured, and it loses. The delegation
   stays.** On an Apple M4 (4 P + 6 E, Metal 4, 16 GB, macOS 26.6.2), Bell +
   `Reset q0` was run on one leased state, re-initialised in place, with
   completion forced each shot and an autorelease pool drained each shot.
   Waiting does not return Metal's ~64-outstanding command-buffer semaphore:
   an autoreleased `MTLCommandBuffer` stays outstanding until the pool
   drains, which is the adjoint sweep's finding. That is the form
   `LIMITATIONS.md` names, plus the drain the semaphore actually requires.
   The loop is not in the tree.

   It does **not** stall. Every count returned. Same seed, and the counts
   matched the CPU at every count — the post-reset state is a basis state,
   so the final sample is fixed by the reset branch. Release build:

   | shots | CPU Reset | GPU, one lease | no `Reset` (GPU) |
   |---:|---:|---:|---:|
   | 32 | 0.03 ms | 88.16 ms | 0.33 ms |
   | 64 | 0.06 ms | 127.17 ms | 0.34 ms |
   | 256 | 0.22 ms | 267.66 ms | 0.33 ms |
   | 1024 | 0.84 ms | 1084.78 ms | 0.37 ms |
   | 65536 | 29.28 ms | 74139.47 ms | 0.93 ms |

   Also completed, same counts as the CPU: 16 (36.19 ms, cold), 96, 128,
   4096 (4.32 s), 16384 (17.26 s). Warm, the GPU arm is ~1.1 ms/shot.
   65536/1024 on the no-`Reset` arm is 2.56× (0.37 ms → 0.93 ms). GPU/CPU
   on the Reset arm is 2968×, 1296× and 2532× at 32, 1024 and 65536. A
   2-qubit trajectory pays several command-buffer round trips; the CPU pays
   a fraction of a microsecond. Shipping the loop would make the path this
   item exists for about three orders of magnitude slower.

   `execute` still returns `StatevectorBackend::execute(..)` whenever
   `shots` is set and the circuit contains a `Reset`. The delegation stays
   observable: `reset_shots_cpu_fallback_count()` moves when it runs,
   `metal_shots_execute_count()` moves when `execute` samples on the GPU, a
   shots-mode call moves exactly one, and `tests/reset_shots_dispatch.rs`
   asserts both directions (Reset moves only the CPU counter; the same
   circuit without Reset moves only the Metal one). Thread-local, not a
   process-global `AtomicU64`: this crate's other shots execute runs under
   `cargo test` in parallel, and a shared counter would report that test's
   GPU sample as this call's (the failure `GATES_SKIPPED` already recorded).
   Deleting either increment fails the test.

   The 2026-09-25 cost of the delegation, on the same circuit, was 0.01 ms
   at 32 shots rising to 17.44 ms at 65536, against a flat 0.45–1.08 ms
   without `Reset`. This machine measured the delegation at 0.03 ms and
   29.28 ms. The shape is the same. Reproduce it with `cargo run --release
   --example reset_shots_probe -p omega-backend-statevector-metal --features
   metal`. The one-lease column above is not that probe: the probe times
   the delegation. The loop was a throwaway and was removed after the table.

   → §3f **A10**. Anyone benchmarking "Metal vs CPU" on a Reset circuit is
   timing the CPU twice and attributing half to Metal.

   **Not closed by the paragraph above.** The Metal theta-contraction is
   STRICTLY WORSE than the CPU path — measured 2026-09-25 on an Apple M4
   (4 P + 6 E, Metal 4, macOS 26.6.2).
   Three findings, in the order they have to be read:

   * *The bench that was supposed to validate it has never once dispatched to
     the GPU.* `contract_bench.rs` prints "0 of N pairs on GPU" beside
     "speedup 1.16x" — a CPU-vs-CPU number. Its warm-up applies CNOT layers to
     an all-`|+>` state, and `CNOT(|+> (x) |+>) = |+> (x) |+>` exactly, so the
     warm-up is the identity and the bond never leaves 1. Confirmed by sweeping
     `MPS_METAL_MIN_BOND`: 78 pairs dispatch at 1, zero at 2. So the G-Metal
     acceptance criterion has never been evaluated against a GPU. → §3f **A10**.
   * *With a bench that actually entangles, Metal is break-even at best.*
     0.909x-1.003x against a 1.3x acceptance bar, across max bonds 16 to 128 —
     including 128, where a GPU should look best. Dispatch happens on only
     27-38% of pairs (`metal_contract_2q` returns `None` when
     `left.bond_right < 32`, and most brickwall pairs sit below that even when
     the state's max bond is 128). De-diluting crudely gives 1.01x on the
     larger shape and **1.37x SLOWER** per dispatched pair on the smaller.
   * *It perturbs the truncation certificate, in the under-reporting
     direction.* The GPU contraction is f32 (Apple has no f64). Calling both
     paths on byte-identical inputs at every split: max relative difference in
     `discarded_weight` 2.8e-8, median ~1e-8 — and the Metal value is SMALLER
     than the CPU value on 5 of 20, 2 of 20 and 1 of 12 dispatched splits. On a
     quantity documented as a BOUND, occasionally-low is not a bound.

   Break-even speed on the pairs it runs, plus a bound it under-reports, makes
   the existing Metal contraction strictly worse than CPU rather than merely
   not-better. It is **not installed** on the production MPS path —
   `omega-run`, `aria-runtime` (`make_mps` / `make_noisy_mps`), and
   `bindings/aria-py`. Certificate soundness is why. `omega-run --backend mps
   --device metal` keeps the two-site contraction on the exact-f64 CPU path
   and says so. The runtime and the Python binding have no device request:
   unset is silent, and the contraction stays on the CPU.
   `aria_py.Backend.accelerator()` stays `"cpu"` for an `mps` spec; the
   binding returns a bare float, not a device field. The kernel stays in
   `omega-backend-mps-metal` for the open measurement
   (`examples/certificate_divergence.rs`). `MPS_METAL_CONTRACT=1` opts back in
   on all three and warns that `discarded_weight` is not a bound under it.

   **The same signature appears in CUDA `gesvdj` (§16): agreement when the bond
   is exact, divergence only when truncating.** Two different GPU backends, two
   different precisions (Metal f32, cuSOLVER f64), same shape of defect. That
   points at doing the truncation DECISION in different arithmetic rather than
   at either vendor's kernel.
4. **`verification/Verification/Backend/{Reset,StabilizerExpectation}.lean`:
   7 `sorry` targets** — 4 + 3, the seven named below; the "9, recounted
   2026-09-26" that stood here was a miscount (the list beside it always had
   seven). Audited 2026-09-30 by elaborating each file under the pinned
   toolchain: Reset.lean 4 warnings, StabilizerExpectation.lean 3,
   PauliAlgebra.lean 0. `ci.sh` now gates on those exact counts wherever
   `lean` is on PATH (no lake, no Mathlib needed), with a negative probe that
   proves the gate rejects an undeclared `sorry`; before that, nothing in CI
   elaborated `verification/` at all.
   Both files open with *"Status: **specification, not yet proved.**
   `theorem`s are targets (`sorry`)"*, so this is a declared state rather than
   abandoned proof work. The named targets are `reset_outcome_irrelevant`,
   `reset_yields_zero`, `fold_is_not_reset`, `entangled_reset_not_pure`;
   `expectation_trichotomy`, `zero_only_when_anticommuting`,
   `echelon_reduction_complete`.
   **Why they cannot be discharged in place — and it is not the ordered
   field.** The earlier "they need a real ordered field; core Lean has none" is
   stale twice over: the files no longer mention `Real` (they declare an
   abstract inhabited carrier `R`), and `proofs/lean4` has Mathlib v4.29.0
   pinned and built. The actual obstacle is that every symbol a target talks
   about (`State`, `p0`, `project`, `applyX`, `reducedPurity`, `EqUpToPhase`;
   `StabState`, `Anticommutes`, `InGroup±`, `expectation`,
   `reducesToIdentity`) is a free axiom with no axiom relating it to any
   other. From that trust boundary no target is provable and none is
   refutable; adding "bridge" axioms until they go through would be assuming
   the theorem. The only honest route is a concrete model — `State n :=
   Fin (2^n) → ℂ`, `project` as a normalised projector, a tableau for the
   stabilizer file — inside `proofs/lean4`, which has the matrix semantics
   (`CircuitSemantics.lean`) but no measurement, Reset or tableau today. That
   is proof work, not a recount, and is left open here with its scope stated.
   **Reset half done in the concrete model, 2026-10-02.**
   `proofs/lean4/QuantumProofs/ResetModel.lean` builds that model (unit
   vectors over `Fin n → Bool`, projector-and-renormalise, bit-flip `X`,
   `tr ρ_q²` purity, equality up to a unit scalar, the coherent `fold`) and
   proves all four Reset targets there: `reset_outcome_irrelevant`,
   `reset_yields_zero`, `fold_is_not_reset`, `entangled_reset_not_pure`, with
   premise witnesses `witness_T1/T2/T4` (`|+⟩`, a Bell state), and a
   fixture `reset_yields_zero_needs_possible` showing the impossible branch is
   reachable (`|0⟩`, outcome 1: reset is the zero vector, `p0 ≠ 1`) so
   `Possible` is load-bearing. `EqUpToPhase` is one scalar `c : ℂ` acting on
   the whole vector — a global phase, never per component. Axioms
   `propext`, `Classical.choice`, `Quot.sound` only; `ci.sh` (ARIA_LEAN=1)
   gates all eight by exact count. Mutations that change a definition, not a
   proof — `reset` without the `X` on outcome 1, `fold` replaced by the real
   channel, the purity cross-term halved — each break the proof. The four
   abstract `sorry`s in `Reset.lean` remain by design (see above); the
   stabilizer three are still open.
   **Stabilizer T1 done in the concrete model, 2026-10-06.**
   `proofs/lean4/QuantumProofs/StabilizerModel.lean` models a Pauli as the
   record `i^k X^x Z^z` acting on amplitudes over bitstrings. The record
   product is proved to be the operator product (`act_mul`), and the
   symplectic form is proved to be operator commutation
   (`act_swap`, `anticommutes_iff_operators`). Group membership is
   combinatorial: a record equal to a signed product of generators, never
   `Pψ = ±ψ`. `expectation_trichotomy` is proved there, with the Bell-state
   witness `witness_T1` and the independent amplitude computation
   `bell_YY_direct` (`⟨Φ⁺|YY|Φ⁺⟩ = −1`). Axioms `propext`,
   `Classical.choice`, `Quot.sound` only; `ci.sh` (ARIA_LEAN=1) gates all seven
   by exact count, and misspelling one member makes it report 6 of 7.
   Definition mutations, each making a theorem false: the product's phase
   using `x·z'` instead of `z·x'` (the shape of the `pauli_mult_phase`
   defect), the adjoint without its `(-1)^{z·x}`, `ω` with one of its two
   terms, the operator in the `Z`-then-`X` order, `InGroupMinus` without its
   sign, `Anticommutes` as `ω = 0`, and `neg` as `i`. Each one breaks a proof.
   **Stabilizer T2 done in the same model, 2026-10-06.**
   `zero_only_when_anticommuting` is proved with full rank stated as a
   hypothesis: the generators' bits are linearly independent over `𝔽₂`.
   `full_rank_needed` shows the hypothesis is load-bearing. On `|0⟩|+⟩` with
   `Z⊗I` listed twice, `⟨I⊗Z⟩ = 0` and `I⊗Z` anticommutes with nothing. The
   proof makes the span Lagrangian with Mathlib's
   `LinearMap.BilinForm.finrank_orthogonal`. The Pauli algebra and the
   commutation↔symplectic bridge are this file's own, since Mathlib has
   neither. `commuting_hermitian_in_group` proves the centralizer is the
   group up to sign, which is what the greedy pass got wrong. `witness_T2`
   inhabits the premises on the Bell state. The `ci.sh` gate is now twelve
   theorems. Definition mutations, each breaking a proof: `Independent :=
   True` (T2 and `full_rank_needed` fail), `ω`'s bilinear form with one term
   (`omega_eq_symp` fails), the subset product selecting `c i = 0`, `vec`
   dropping the `z` bits, and `commuting` only on the diagonal.
   `echelon_reduction_complete` is next.
   **Exhaustiveness, consistency, and an inhabitant at every `n`, same file,
   2026-10-07.** `not_plus_and_minus`: no Pauli is both `+` and `−` a product
   of generators. `exhaustive`: on a full-rank state, every *Hermitian* Pauli
   anticommutes with a generator or is `±` a product of generators. The
   Hermitian hypothesis is necessary in this model, not a convenience:
   `exhaustive_needs_hermitian` shows `i·ZZ` on the Bell state is in none of
   the three branches, because every signed group element is Hermitian
   (`inGroup_hermitian`). `zeroState n` (`|0^n⟩`, generators `Z_i`) is proved
   full rank at every `n` (`zeroState_independent`), with `⟨Z_i⟩ = 1` and
   `⟨X_i⟩ = 0` (`zeroState_Z`, `zeroState_X`). The `ci.sh` gate is now 19
   theorems. Definition mutations, each breaking a proof: `InGroupMinus`
   without its sign (`not_plus_and_minus` fails at its own line), `adj`
   without negating the phase (`exhaustive_needs_hermitian` is then refuted
   by `decide`, and `exhaustive` inherits `sorryAx`), and `zGen` as `−Z_i`
   (`zeroState`'s `stabilizes` fails). This closes no new target: item 4
   stays at 6 of its 7 targets proved in the concrete models, and
   `echelon_reduction_complete` — that the pivoted `𝔽₂` elimination
   algorithm is complete — is proved by no one yet.
   **Statement defect found and fixed 2026-09-30 (Reset.lean).** T1 and T2
   were false in any model where `project` is measure-then-renormalise: on
   `|0⟩` (unentangled, so `Deterministic`) the outcome-`true` branch has
   probability 0 and `project s q true` has nothing to renormalise, so
   `p0 (reset s q true) q = R.one` cannot hold. Both now carry
   `Possible s q b` (the forced outcome has nonzero probability — the
   condition the RNG-driven and RNG-free backends both already respect), and
   T1 is quantified over every pair of possible outcomes instead of the fixed
   pair `(true, false)`. T4 deliberately does not take it: an entangled qubit
   has both outcomes possible, and that is part of what T4 must prove.
   §4 of this file records `PauliAlgebra.lean` as 8 theorems proved with no
   Mathlib; that file is concrete, which is why it could be.
5. ~~**Five distinct Reset acceptance policies** across backends.~~ **COVERED
   2026-09-30 (`4797957`)** — the policies remain, by design (→ ledger A6);
   what this item tracked was that their divergences were untested, and every
   one is now pinned — the Metal tolerance on a real device (below).
   **Coverage as of 2026-09-25** — the "only CPU↔Metal is conformance-tested"
   clause was already stale when written twice over:
   - CPU↔Metal: `statevector-metal::reset_matches_cpu`
   - CPU↔CUDA: `statevector-cuda/tests/reset_criterion.rs` (needs a device)
   - CPU/MPS/pauli: `omega-bridges/tests/reset_acceptance_conformance.rs` — new,
     runs with no GPU, pins all three policies *and their disagreement* on
     `H q0; Reset q0`: CPU accepts (purity 1), MPS refuses (all analytic
     resets), pauli refuses (cannot represent the channel).
   - **The test found a ledger defect on its first run.** A6's `pauli` row said
     "non-Z-eigenstate", implying it accepts `X q0; Reset q0`. It does not.
     `LIMITATIONS.md` was right and A6 was wrong; A6 is corrected.
   Closed 2026-09-30. That looser tolerance is the divergence that changes
   an answer rather than refusing, and it is observed on a real device by
   `omega-backend-statevector-metal/tests/reset_tolerance_observed.rs`.
   The state is `√(1−ε)|00⟩ + √ε|11⟩` at `ε = 4·10⁻⁵` (purity deficit
   `2ε(1−ε) ≈ 8·10⁻⁵`). Reset on qubit 0 is driven through
   `MetalStatevectorBackend` and the CPU `StatevectorBackend` on that
   state: Metal accepts, the CPU refuses. The test asserts the split and
   carries neither tolerance, so changing the threshold in
   `apply_reset_with` reddens it. macOS, `--features metal`; if
   `MetalStatevectorBackend::new()` fails, the test skips out loud.
   `omega-bridges/tests/reset_tolerance_divergence.rs` still records which
   branch is dropped, and does not touch a device.
6. ~~**`ci.sh` runs clippy without `--all-targets`.**~~ **CLOSED — verified
   2026-08-26.** Stage 2 is `Clippy -D warnings (WHOLE WORKSPACE, all targets)`
   and it passes: `grep -c 'clippy --workspace --all-targets' ci.sh` → 2, and a
   full run exits 0. The named regression
   (`omega-parser/tests/gate_arity_is_validated.rs:165`) is gone with it. The
   entry outlived the fix, which is the failure mode this list exists to avoid
   — a stale open item sends the next reader to re-solve closed work and makes
   the live entries look equally doubtful. → request E2
7. ~~**`tools/qec_cross_check/run.sh` bootstraps with plain `pip`**~~
   **ARCHITECTURE-SCOPED, and it RUNS on x86_64 — verified 2026-09-26 on
   akilles.** The original entry was true and is still true *of aarch64*, but it
   was written as though it were universal, and the machine it was measured on
   is no longer in the estate.
   - **aarch64** (measured on a DGX Spark GB10, 2026-08-16): no `pymatching`
     wheel on PyPI at all (`--only-binary=:all:` → "no matching distribution")
     and the cmake source build fails. Cannot run. That GB10 is gone; the estate
     is now akilles (linux x86_64 + RTX 6000) and andromeda (Apple M4).
   - **x86_64** (akilles, 2026-09-26): installs from a wheel —
     `pymatching-2.4.0-cp312…manylinux_2_28_x86_64`, no build — and the
     MANDATORY cross-check **passes in full**, zero-config bootstrap:
     - encoded-algorithm vs qiskit 2.5.2 (+stim tableau), grover/qft/qpe:
       **20 passed, 0 failed**
     - surface-code decoder vs PyMatching 2.4.0, d=3 and d=5, p=0.05,
       20 000 shots: **100.00% shot-for-shot logical-class agreement** in every
       sector; logical rates identical to 4 digits (d=3 X 0.0367 Z 0.0336;
       d=5 X 0.0251 Z 0.0238). **12 passed, 0 failed**
   - **Gotcha — environment, not packaging.** Behind a TLS-intercepting proxy
     plain `pip` fails with `CERTIFICATE_VERIFY_FAILED … self signed certificate
     in certificate chain`, exactly as cargo does. Point `PIP_CERT` at the
     proxy's CA bundle — on akilles that is
     `~/.config/pxpipe/mitm-ca.pem` — and `CARGO_HTTP_CAINFO` at the same file
     for the `cargo build` inside `run.sh`. Note `SSL_CERT_FILE` is NOT set on
     that host, so recipes written against it expand to an empty path. This is
     a proxy setting and not what this item is about. (Kept out of the public
     mirror, which carries the same entry without the internal path.)
   - So the check is available today and should be run on x86_64 rather than
     treated as blocked. It is the fourth item this week that was a
     machine-local limit recorded as a universal one (cf. le-harnais Tier B
     CUDA, lh-linux-arm64, and this file's own CUDA items). **Name the
     architecture in the claim** — an entry that does not say which machine it
     was measured on becomes false silently when the hardware changes.
8. ~~**No `RUST_TEST_THREADS` in `ci.sh`**~~ **CLOSED 2026-08-16** — set
   stage-scoped on the CUDA stage. But the stated cause was **wrong**:
   `!Send`/`!Sync` is a compile-time property that *prevents* cross-thread
   sharing and cannot itself segfault a harness where each test builds its own
   backend, and `forward_graph.rs:167-172` already captures with
   `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`. `ForwardGraph::capture` is reachable
   only from `#[cfg(test)]`, so serialising is a fix for the symptom; **the
   cause is still unidentified** and `ci.sh` says so.
9. ~~**`aria_py` bypasses `omega-server` admission entirely** — no way for a
   client to ask what a circuit would cost.~~ **CLOSED 2026-09-26.** The
   component named here is the wrong one. `bindings/aria-py` is an in-process
   runtime: `python/aria_py/__init__.py` and `torch.py` never speak HTTP, and
   `tools/omega_client/omega_client.py` exists because of that. A cost formula
   in `aria_py` would be a second copy of `estimate_peak_bytes`.

   The gap was the server. `Governor::admit` ran only inside
   `/v1/quantum/execute`, `/expectation` and `/gradient`. `/health` publishes
   the budget, not the price of a circuit. There was no estimate route and no
   dry-run flag.

   `POST /v1/quantum/admission` calls the same `decide_shape` those routes use
   — `route` is `execute`, `expectation` or `gradient`, because the price is a
   property of the route — then drops the reservation before answering, so
   asking does not hold the budget. `admitted` is the decision, `peak_bytes`
   is the price, and `error` is the refusal text the real route returns.
   `OmegaClient.admission` is the caller. The test
   `admission_query_agrees_with_the_route_that_would_run_it` checks three
   circuits the routes refuse (40-qubit statevector execute, 34-qubit Auto
   expectation — the densify case — and a 40-qubit gradient), each with the
   same `error` as that route, and a 2-qubit shot-mode Bell that the query
   calls acceptable and `/execute` returns 200 for.
10. ~~**CPU statevector is single-threaded per circuit** and 2q-heavy circuits
    run sparse gates as dense 4×4 — the CPU is the only backend without a
    diagonal fast path.~~ **CLOSED — both clauses, verified 2026-08-26.**
    Checked clause by clause rather than as one entry, which is what surfaced
    that they had gone stale at different times:

    * *single-threaded* — false. `apply_1q`/`apply_2q` carry 18 rayon parallel
      sites behind a `PAR_MIN_DIM` threshold; `PLAN-SV-PERF.md` S2 landed.
    * *sparse gates as dense 4×4* — closed 2026-08-25. `CZ`/`CRz` take a
      diagonal path, `SWAP` a permutation, `CY`/`CU3` a controlled-U path that
      never reads the control-zero half, and `Rbs` a {|01>,|10>} subspace
      kernel. Detected from the MATRIX, not routed per call site, so a diagonal
      gate added later is covered with no dispatch table to keep in sync.
      Measured 1.27–1.57× at n=22 (`98403d4`, `0ef773d`, `88ed02e`).

    → `PLAN-SV-PERF.md` S1b, S2; `PLAN-OPEN-20260825.md` §2.1

    **Correction, 2026-08-16: CUDA did not have that fast path either.** It has
    a diagonal kernel, but `apply_cx`/`apply_cy`/`apply_cz`/`apply_swap`/
    `apply_crz` all built a dense `[Complex64; 16]` and dispatched the generic
    `apply_2q` — 4 loads + 4 stores and 160 real flops per quad, to apply a
    permutation or a diagonal. `diagonal_factor` never caught them because it
    carries a **1-qubit** factor only. Fixed for CX/SWAP/CZ/CRz (`31d53f5`,
    `4325a50`); ~~**CY is still on the dense path.**~~ **CLOSED — verified
    2026-08-26**: `apply_cy` (`statevector-cuda/src/lib.rs:788`) routes through
    `apply_quad_swap_phase`, falling back to dense only below
    `QUAD_KERNEL_MIN_QUBIT_2SLOT`.
11. ~~**MPS is ~210× slower than our own statevector on one 19-qubit chain.**
    Reproduced, undiagnosed.~~ **CLOSED 2026-08-20, and the attribution here
    was wrong.** It was not an unexplained slowdown: the original invocation
    (`--backend mps`, then-default bond 64) was **silently truncating**,
    discarding 34% of the norm, and now REFUSES in 0.194 s with the dirty
    certificate in the message. The sanctioned `mps:auto` run is exact
    (certificate 3.47e-24, fidelity 1.0) and slow for a diagnosed reason —
    `max_bond_reached` 511 is full rank at the 19q middle cut, so the CR's
    "strictly low entanglement" label was itself wrong.
    → `PLAN-CR-20260813.md` B4
12. ~~**The ppvm bridge refuses any noise request** — ppvm's tableau supports the
    channels including atom loss; the mapping is unwritten.~~ **CLOSED
    2026-09-30 (`6b5b46f`) for the Pauli channels;** atom loss stays
    unreachable — ppvm's parser rejects the erasure instructions, and the
    counts protocol has no slot for a lost atom (below).

    **Mapping written 2026-09-30.** The counts path now maps omega's
    `depolarizing`, `pauli`, `phase_damping` and symmetric
    `readout`/`readout_flip` onto Stim noise instructions
    (`python/stim_noise.py`, table in `docs/BRIDGES.md`) and refuses BY NAME
    what it cannot express exactly — `amplitude_damping` (non-Pauli),
    asymmetric `readout` (`p10 != p01`, no Stim instruction), unknown keys —
    still as `ppvm-noise-not-supported`. Two Stim spellings were wrong on
    first guess and are recorded in `stim_noise.py`: `DEPOLARIZE2(p)` on a
    2-qubit gate is a two-qubit channel, not two `DEPOLARIZE1(p)`; `M(p)` is
    a record-only flip, not `X_ERROR` before `M`. Empirically, the installed
    ppvm (0.1.0) parses `DEPOLARIZE1/2`, `X/Y/Z_ERROR`, `PAULI_CHANNEL_1/2`
    and `M(p)` and samples them correctly; `HERALDED_ERASE` and `E` are
    rejected at parse time — so the "atom loss" half of the headline is
    still nothing but prose, and the `bits_to_counts` blocker below stands.
    TVD ppvm-noisy vs `NoisyStatevectorBackend` at 20 000 shots is
    0.003–0.011 across Bell/GHZ3 under every mapped channel, with the
    noiseless-vs-noisy guard 0.06–0.36, so a silently dropped channel cannot
    pass. Tests: `crates/omega-bridges/tests/ppvm_noise.rs` (8, each
    hand-mutated red once, one of them checking the Python rate resolution
    against Rust `at_gate` on a 7-spec × 8-gate table), `bridge_smoke.rs`
    (mapped run + by-name refusal through the CLI); `ci.sh` now runs
    the `bridge-ppvm`-gated smoke tests under `ARIA_BRIDGE_XCHECK=1`, since
    the omega-cli arms otherwise compile with `bridge-qiskit` only.

    **Audited 2026-09-24. The headline is right about the counts path, wrong
    about "any", and the claim about ppvm is ours, not ppvm's.**

    * *Counts path — accurate.* `ppvm_runner.py:147-155` refuses loudly and
      with a typed error: `kind="ppvm-noise-not-supported"` routes through
      `runner.rs:577-585` to `BridgeError::CannotExpress`, deliberately
      distinct from "not installed" and "real defect". Good as it stands.
    * *Expectation path — it did NOT refuse, it dropped the flag silently.
      FIXED 2026-09-24.* `--bridge X --expectation Z0 --noise '{...}'` printed
      a noiseless number to stdout with nothing on stderr. The counts arm
      parses `--noise`; the expectation arm returned ~400 lines before the
      in-process guard that catches the same mistake for statevector/mps.
      `omega_bridges::expectation_qasm2` takes no noise argument at all, so
      this was never ppvm-specific — **no** bridge could express it, and
      qiskit and perceval leaked identically. Now refused before
      `Backend::parse`, so it fires with no bridge feature compiled in, and
      `bridge_smoke.rs` covers both the refusal and the counts path it must
      not disturb.
    * *"ppvm's tableau supports the channels including atom loss" is
      unverified repo prose*, from `TSIM-PPVM-NOTES.md:204-206`, never checked
      against the installed package. ppvm is an unpinned git dependency
      (`requirements-ppvm.txt`) with no vendored source and no local interface
      description; `qasm2_stim.py` emits no Stim noise instructions at all.
      Whoever writes the mapping must first establish empirically which Stim
      noise instructions ppvm actually accepts — the way SWAP/CCX/CCZ
      rejections were established (`docs/BRIDGES.md:63-66`), by handing raw
      text to the parser rather than assuming.
    * *Atom loss has a second, harder blocker than the missing mapping.*
      `qasm2_stim.bits_to_counts` (`qasm2_stim.py:582-591`) hard-errors on any
      outcome that is neither 0 nor 1, specifically so `LOST` cannot fold into
      the `1` bin. Even a reachable loss channel has nowhere to go in the
      current counts shape; that is a protocol change, not a mapping.
    * *No test asserts the refusal.* **Obsolete 2026-09-30** — see above;
      `ppvm_noise.rs` and `bridge_smoke.rs` assert both the by-name refusal
      and that a mapped channel actually moves the counts. `corpus.rs` still
      has no `noise` field.
    * *One more trap for whoever does this.* **Fixed in `73a00ac`** (the
      bridge arm now validates with `NoiseModel::from_json`). The CLI's bridge arm parses
      `--noise` with a bare `serde_json::from_str`, not `parse_noise_model`,
      so a key the in-process path rejects loudly reaches the bridge as an
      arbitrary dict. Either re-validate on the Python side or move the bridge
      arm onto the shared parser.
13. *(No item. The number was skipped when 12 and 14 were written — 14 first
    appears in `c12bf07` and no revision of this file has ever had a 13; checked
    2026-09-30 with `git log -p -S`. Kept vacant rather than renumbered because
    `§16` / `§5.16` are cited from this file, `PLAN-QUDIT.md` and the MPS
    sources, and a renumber would silently re-point every one of them.)*
14. **`hhl.aria` and `proofs/lean4/QuantumProofs/HHL.lean` need cleaning up —
    and the link between them is an annotation, not a check.** The off-by-one
    fixed on 2026-08-15 is the symptom; the structure around it is the item.
    Three separate things to settle. **Two are closed (2026-10-02): the
    example now implements what the proof proves, and the `A⁻¹b` claim is
    asserted against closed forms. The third — the harness being
    structurally blind — is unchanged, and so is the hand-maintained nature
    of the circuit↔theorem correspondence; see the end of this item.**
    - ~~**The example does not implement what the proof proves.**~~ **It does
      now — CLOSED 2026-10-02; the detail is at the end of this bullet.**
      `HHL.lean` is
      sorry-free and genuine — `hhl_solves_system` is a real matrix-vector
      identity `A · o = C · b` for `A = diagonal λ`, with an exact QPE and a
      controlled `RY(2·arcsin(C/λᵢ))`. The Aria example instead uses a
      *proxy* eigenvalue (`theta = 0.5 / (k + 1)`, commented "eigenvalue proxy
      λ = k + 1") and, at `hhl.aria` step 3, an "inverse QFT" that was **H on
      each qubit and nothing else**. ~~Whether the example should be brought up
      to the proof, or the header claim narrowed, is a decision, not a bug
      fix.~~ **The step-3 half was a bug, and worse than "incomplete" —
      CLOSED 2026-09-24.** Measured on an independent dense simulator: with the
      H-only version the probability of the CORRECT phase is **0.000000** at
      `n = 2` and at `n = 3`. Not imprecise, zero — and `n = 2` is the width
      both harnesses run. A second defect fell out beside it: **step 5 undid
      step 2 but never undid step 3**, so the counting register stayed
      entangled with the answer. The fraction of the `ancilla = |1⟩` branch with
      the register actually returned to `|0..0⟩` was 0.500 at `n = 1`, 0.097 at
      `n = 2`, 0.011 at `n = 3`. Both are fixed; all three are now 1.000000000,
      with `P(ancilla = 1)` width-independent at `sin²(0.25)`.
      ~~**The eigenvalue proxy is untouched and is still the open decision** —
      `theta` is keyed to the counting-qubit INDEX `k`, not to the eigenvalue
      the register holds, which is not what `HHL.lean` proves.~~
      **The proxy is gone — CLOSED 2026-10-02**, by bringing the example up to
      the proof rather than narrowing the header. Four changes, all in
      `hhl.aria`:
      1. **QPE now loads the eigenvalue, not a bare relative phase.** The
         evolution time is fixed at `t = 2π/2ⁿ`, so the eigenphase of `λⱼ` is
         exactly `λⱼ/2ⁿ` and the register reads back the INTEGER `λⱼ`. That
         needs a `P(2π·2ᵏ·λ₀/2ⁿ)` on the counting qubit alongside the `CP`:
         `U = diag(e^{iλ₀t}, e^{iλ₁t})` carries a `λ₀` factor on BOTH system
         states, and a global phase on a controlled sub-block is a RELATIVE
         phase — the same trap already settled for `GateKind::Sx` and the
         photonic `hwp`. Omitting it loads `λ₀ = 0`, and the `|0⟩`
         eigencomponent then has no inverse at all: measured, `P(ancilla = 1)`
         0.221566 → 0.037912 and the `|0⟩` amplitude of the answer goes to
         zero.
      2. **The rotation is keyed to the register's VALUE.** For every
         `m ∈ [1, 2ⁿ)` the circuit applies `RY(2·arcsin(C/m))` conditioned on
         the register reading exactly `m` — a full-width pattern match, which
         is `HHL.controlled_inv_rotation` verbatim. Aria has no multi-controlled
         RY, so each term is `RY(θ/2)·MCX·RY(-θ/2)·MCX` with the zero bits of
         `m` conjugated by `X`; the `MCX` is `CCX` at `n = 2` and a `C³X`
         through one clean work qubit at `n = 3`.
      3. **`b` is no longer an eigenvector** — `RY(0.8)|0⟩` instead of `X|0⟩`.
      4. **Widths outside 2..3 are refused at instantiation.** `λ_max = 2`
         needs two bits, and one work qubit only reaches a `C³X`. Outside that
         range every `when` arm of the rotation would miss and the circuit
         would run to completion returning an all-zero solution branch.
      **Why the proxy was worth this much work, measured.** Restoring it on
      the otherwise-fixed circuit gives a post-selected branch of
      `(0.227874, 0.048551)` where `C·A⁻¹b` is `(0.460530, 0.097355)` — every
      amplitude about half what it should be, yet the RATIO is 0.213059
      against a true 0.211397, **0.8% out**. An assertion on the *direction*
      of the answer would have passed it. The un-normalized amplitudes are
      what make it visible.
    - **`@prove "hhl_recovers_inverse"` is not verified against anything.** The
      annotation names a property; `ci.sh:656` separately checks the Lean
      theorems are sorry-free. Nothing connects the two — no check would notice
      if the circuit and the theorem drifted apart, and they have.
      **Partly addressed 2026-09-24**:
      `crates/apps/forward/tests/hhl_is_really_hhl.rs` asserts three properties
      the circuit must have to be HHL at all — the uncompute disentangles the
      counting register, the solution branch carries no spurious system
      component, and `P(ancilla = 1)` equals the closed form `sin²(0.25)` at
      every width. These are checked against closed forms, not against a second
      engine, so they are not blind the way the transport check is. Two of the
      three FAIL on the pre-fix circuit, which is the only evidence that a new
      test is worth having. ~~It does **not** yet verify the `A⁻¹b` claim
      itself; that needs the eigenvalue proxy settled first, and a `b` that is
      not an eigenvector (today `b = |1⟩` is one, so the 1/λ weighting never
      shows).~~ **The `A⁻¹b` claim is now asserted — 2026-10-02.** With the
      proxy gone and `b = RY(0.8)|0⟩`, the test is five assertions, each
      naming the `HHL.lean` theorem it is the circuit-level counterpart of:
      * `the_counting_register_holds_the_eigenvalue_after_qpe`
        (`hhl_qpe_eigenphase`) — cuts the IR at the first operation touching
        the ancilla and reads the QPE register. Measured
        **P(correct phase) = 1.000000000** at `n = 2` and `n = 3`, with the
        per-eigenvector weights equal to `|βᵢ|²`.
      * `the_postselected_branch_solves_the_linear_system`
        (`hhl_solves_system`) — the capstone, `A·o = C·b` componentwise.
        Measured residual `|A·o − C·b| ≤ 4.4e-16` at both widths.
      * `the_ancilla_probability_is_the_closed_form_and_width_independent`
        (`controlled_inv_rotation` / `success_prob`) — `P(ancilla = 1) =
        Σᵢ|βᵢ|²(C/λᵢ)² = 0.221566254001` at both widths.
      * `the_uncompute_disentangles_the_counting_register` — now also covers
        the `C³X` work qubit.
      * `widths_outside_the_supported_range_are_refused`.
      Only the global phase is divided out in the capstone, so every relative
      phase and every magnitude is under test. Honest limits, in the same
      place as the claim: the probability assertion is **implied** by the
      other two (uncompute + capstone ⟹ Born rule), it is kept because it is
      the cheapest statement of width-independence; and the capstone reads
      one lowered IR, so it cannot distinguish a defect from a matching defect
      in the lowering — that axis is what the differential gate covers.
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
    symptom of a house-wide misunderstanding. Second, ~~**`qft.aria` already
    contains the correct inverse QFT that `hhl.aria` step 3 is missing** — so
    reconciling the example needs no new physics, only reuse.~~ **That second
    claim was WRONG, and acting on it would have shipped another confident
    wrong answer — corrected 2026-09-24 by measurement.** `qft.aria`'s `IQFT`
    is a correct adjoint *of `qft.aria`'s own QFT*, which ends with its bit
    reversal, so the adjoint begins with one. `hhl.aria`'s QPE gives counting
    qubit `k` the phase `2^k`, making `q[0]` the LSB — the opposite convention.
    Pasted in verbatim it scores `P(correct phase) = 0.500000` at `n = 2` and
    `0.410533` at `n = 4`: better than H-only, still wrong, and wrong in the
    way that survives a casual look. Moving the bit reversal to the other end
    gives `1.000000` at every `n` tested. The reusable part was the phase
    ladder; the index convention had to be derived, and only measurement
    distinguished the two.

    The general lesson, since this entry is where it was learned twice: a
    same-IR differential harness cannot referee a convention question. Both
    sides agree on whichever convention the IR encodes.

    Scope of that audit, stated so nobody over-reads it: it covers loop bounds
    against register sizes. It does **not** cover gate choice, angle formulas,
    or eigenvalue proxies — the other three ways `hhl` was not what `HHL.lean`
    proves. All three are now closed for `hhl` (see above), and all three are
    still unaudited for the other 48 examples: nothing in the tree would catch
    a wrong angle formula in any of them.

    **The hand-kept correspondence is now checked (2026-10-06).**
    `hhl_is_really_hhl.rs::the_three_sources_state_one_instance` reads
    `lam2` out of `HHL.lean` and `lam0`/`lam1`/`cscale`/`b_angle` out of
    `hhl.aria` by parsing the source, and asserts they equal the test's
    `LAM`/`C`/`B_ANGLE`. `A` is linked across all three sources; `C` and `b`
    between the circuit and the test only, because the Lean numeric section's
    `C = 1`, `b = ![1, 1]` is a different, deliberately symmetric instance of
    a theorem general in both. The parse refuses rather than defaults: a
    definition found zero times or twice, a non-`![…]` vector, or a
    non-literal value fails naming the file and line. Mutations, each red:
    `lam2 := ![1, 3]` (message: "HHL.lean:167: the theorem's `lam2` =
    [1.0, 3.0] but this test's LAM = [1.0, 2.0]"); `lam1 = 3` in the
    circuit; `lam2` renamed (message: "expected exactly one line starting
    with `noncomputable def lam2 `, found 0"); `cscale = 0.25`. It is a
    text link, not an export: renaming the definition is caught, and moving
    the value into an expression is refused rather than evaluated.
15. ~~**The photonic backend has never been compared on *speed*, only on
    agreement.**~~ **MEASURED 2026-09-25 against Perceval. We lose on both
    stages that can be compared — the permanent gap was 17x of parallelism and
    only 1.2x of kernel, and `slos_full` was not SLOS at all. Both fixed; both
    now within ~1.2-1.4x of Perceval.** `bridge-perceval` exists and the DV/CV conventions are matched
    verbatim, so the correctness axis is covered — but Perceval's hot path is
    C++ (`quandelibc`) and piquasso's is NumPy/JAX, and neither has been timed
    against `omega-backend-photonics`. Deferred deliberately: the statevector
    lane (`PLAN-SV-PERF.md`) and the adjoint lane
    (`PLAN-ADJOINT-MEMORY.md`) are the measured bottlenecks today, and a
    photonic comparison wants the same discipline they got — a baseline before
    a claim, and per-stage timings rather than wall clock.

    **The baseline** (`benches/photonics_bench.rs`, this box): Reck decompose
    6 modes **1.047 us**, 6x6 permanent **403 ns**, SLOS n=4 m=6 **22.68 us**.

    **Against Perceval 1.3.0** (`exqalibur`, the C++ hot path that succeeded
    `quandelibc`), same 6x6 DFT unitary, same `|1,1,1,1,0,0>` input:

    | stage | ours | Perceval | |
    |---|---|---|---|
    | SLOS n=4 m=6, warm backend | 22.68 us | **5.73 us** | theirs 4.0x |
    | SLOS n=4 m=6, incl. circuit setup | 22.68 us | 14.76 us | theirs 1.5x |
    | permanent 16x16 | 1.368 ms | **453 us** | theirs 3.0x |
    | permanent 18x18 | 6.363 ms | **604 us** | theirs 10.5x |
    | permanent 20x20 | 28.996 ms | **1.731 ms** | theirs 16.8x |
    | permanent 22x22 | 132.41 ms | **6.297 ms** | theirs 21.0x |

    **The 6x6 permanent row is deliberately absent, and that is the
    methodological finding.** `exqalibur.permanent_cx` carries ~440 us of
    FIXED per-call cost — measured flat at 433-453 us from 2x2 through 16x16,
    where Ryser is O(n·2^n) and a 16x16 is ~4000x the arithmetic of a 2x2.
    Its kernel only surfaces from 18x18 (ratios 1.33x, then 2.87x, 3.64x per
    size step). Comparing at 6x6 measures THEIR BINDING against OUR
    ARITHMETIC and reports a ~1000x win for us that means nothing. Anyone
    quoting a small-matrix permanent comparison in either direction is
    quoting call overhead. The bench now carries n=12..22 for this reason.

    **Reck is not compared, on purpose.** Perceval's `Circuit.decomposition`
    is an optimiser that FITS parameters; `exqalibur.RectangularDecomposer` is
    analytic but Clements (rectangular mesh), not Reck (triangular). Neither
    is the same computation as `reck_decompose`, so any number would be a
    category error. Our 1.047 us stands as a baseline, not a comparison.

    **THE PERMANENT GAP IS ALMOST ENTIRELY PARALLELISM — measured, not
    inferred.** The fixed ~440 us per-call cost independent of problem size
    looked like per-call thread-team setup, so Perceval was pinned to a single
    core with `taskset -c 0` and re-timed:

    | n=22 | time | vs ours |
    |---|---|---|
    | ours, single-threaded | 132.41 ms | — |
    | Perceval, `taskset -c 0` | 109.08 ms | **1.21x** |
    | Perceval, 32 cores | 6.28 ms | 21.1x |

    n=20 says the same: ours 28.996 ms, Perceval 1-core 24.849 ms (**1.17x**),
    Perceval 32-core 1.661 ms. Perceval's own parallel speedup at n=22 is
    109.08 / 6.28 = **17.4x on 32 cores**.

    So their KERNEL is only 1.17-1.21x better than ours per core. The 21x is
    core count. `permanent()` is single-threaded and Ryser parallelises
    trivially — the subset sum is an embarrassingly parallel reduction over
    2^n Gray-code terms with no shared state. This is the same shape of finding
    as §5.16: the expensive-looking gap was untapped parallelism in our own
    code, not a better algorithm elsewhere, and the fix is local.

    **DONE 2026-09-25 — parallelised, 11-15x, and it closes most of the gap.**

    | n | before | after | |
    |---|---|---|---|
    | 12 | 61.58 us | 64.88 us | below the gate, unchanged path |
    | 16 | 1.368 ms | **119.6 us** | 11.4x |
    | 18 | 6.363 ms | **477.7 us** | 13.3x |
    | 20 | 28.996 ms | **2.071 ms** | 14.0x |
    | 22 | 132.41 ms | **9.036 ms** | 14.7x |

    Against Perceval on the same box, the 21.1x at n=22 becomes **1.44x**, and
    the 17.5x at n=20 becomes **1.25x**. At n=16 and n=18 we are now AHEAD
    (120 us vs 453 us, 478 us vs 604 us) — but say why honestly: Perceval is
    still overhead-dominated at those sizes, so that is our arithmetic against
    their BINDING, the same trap as the 6x6 row, pointing the other way. The
    residual 1.44x at n=22 decomposes as the ~1.2x kernel difference measured
    single-core times ~1.2x of parallel efficiency (our 14.7x on 32 cores
    against their 17.4x).

    **The mechanism, corrected.** An earlier draft of this entry said Ryser
    "parallelises trivially — an embarrassingly parallel reduction with no
    shared state". That is wrong about THIS implementation and the code says
    so: the Gray-code sweep carries `row_sums` from each iteration to the next,
    which is exactly what makes it O(2^n) instead of O(n·2^n). The carry is an
    optimisation, not a dependency — term `k` depends only on the subset
    `gray(k)` — so a chunk starting at `k0` reconstructs its own state from
    `gray(k0-1)` in O(n²) and then runs the identical incremental body. Without
    that reconstruction the only option is recomputing every row sum from
    scratch, which is n x the arithmetic and pays only above n cores.

    **Two properties it was built to have**, both from §5.16's bill:
    * *Bitwise thread-count invariance.* The chunk count is a function of `n`
      alone, never of the pool; `collect` preserves order and the partials fold
      in chunk order, so the summation tree is fixed. Asserted by
      `thread_count_does_not_change_the_bits` at 1 vs 8 workers, mirroring the
      SVD kernel's own test.
    * *A grain gate with the old path preserved below it.* n < 16 takes one
      unchunked sweep and allocates nothing extra. That matters because
      `slos_full` calls `permanent` once per output Fock state on submatrices
      the size of the PHOTON NUMBER — small. A first cut built the range vector
      unconditionally and cost 45 ns on the 6x6 and 4.7% of SLOS to speed up
      sizes SLOS never reaches.

    **Cost, stated rather than buried:** extracting the sweep into a function
    costs ~10% on the 6x6 permanent microbenchmark (403 -> 443 ns) and ~5% at
    n=12. SLOS — the real photonic hot path — is within run-to-run noise.
    `#[inline]` on the extracted function makes both WORSE (6x6 to 465 ns, SLOS
    +4.3%), so it is deliberately absent.

    **Reassociation is a real change and it is an IMPROVEMENT, checked against
    a float80 reference rather than assumed.** Chunked partials restart the
    incremental drift, so at n=16 the relative error is 9.0e-14 against the old
    sweep's 2.6e-13, and at n=18 2.7e-13 against 9.9e-13 — 2-4x more accurate.
    Below the gate the result is bit-identical to before. `GATE-EXACTNESS.md`
    lists photonics as not audited and no test asserts exact bits on a
    permanent, so nothing downstream pinned the old values.
    Two things to carry over from §5.16 when doing it: rayon's overhead is
    real at small n (that item's kernel LOST 1.31x at one worker from
    `par_iter` split-and-join alone), so this needs a grain gate keyed on n
    with the sequential path preserved below it; and the crossover must be
    measured on more than one core topology, because §5.16's conclusion
    inverted between a 10-core laptop and a 32-core box.

    **`slos_full` WAS NOT SLOS — found and fixed 2026-09-25.** The module doc
    said SLOS "computes all output amplitudes in O(n * M_n)" and is
    "exponentially faster than computing each permanent individually", and the
    implementation then looped over every output Fock state calling
    `permanent()` on a submatrix — which IS computing each permanent
    individually. Same class as §5.14's HHL header: a doc claiming the property
    the code lacks.

    Naive costs M_n·2^n against SLOS's n·M_n, so the penalty is 2^n/n and grows
    with photon number. That was predicted from the code and then measured
    against Perceval (a real SLOS), 6 modes, before the fix: 0.92x at n=2, then
    2.46x, 4.13x, 6.05x, 10.12x at n=6 — against 2^n/n = 2.0, 2.7, 4.0, 6.4,
    10.7. The fit is the evidence that the diagnosis was right.

    Replaced with the real dynamic program (photons added one input column at a
    time, carrying every reachable partial output state forward):

    | n | naive | DP | gain | Perceval | vs Perceval (was) |
    |---|---|---|---|---|---|
    | 2 | 1.42 us | **0.76 us** | 1.88x | 1.57 us | **0.48x** (0.92x) |
    | 3 | 7.40 us | **2.64 us** | 2.80x | 3.13 us | **0.84x** (2.46x) |
    | 4 | 24.24 us | **7.16 us** | 3.38x | 6.00 us | 1.19x (4.13x) |
    | 5 | 79.78 us | **16.86 us** | 4.73x | 12.67 us | 1.33x (6.05x) |
    | 6 | 282.26 us | **32.52 us** | 8.68x | 28.48 us | 1.14x (10.12x) |

    The ratio against Perceval was GROWING as 2^n/n; it is now flat at ~1.2x,
    and we are faster at n=2 and n=3. That flatness is the result — the
    exponential divergence was structural and is gone; what remains is a
    constant factor.

    **Three things the tests caught that would otherwise have shipped**, all
    worth knowing before touching this file again:
    * The differential test against the permanent route failed first on output
      ORDERING, not values. `slos_full` is public, so the sequence of returned
      states is observable; the DP's natural rank order ascends on mode 0 where
      `enumerate_fock_states` descends. Reordering every caller's output would
      have been silent.
    * The first working DP was **2-3x SLOWER** than the permanents it replaced.
      The asymptotics were right and entirely buried under `fock_rank`
      recomputing C(k+parts-1, parts-1) with a u128 multiply-divide loop on
      every probe. Precomputing that table is what made the algorithm change
      visible at all.
    * Flattening the partial-state storage (`states[si*m..]` rather than a Vec
      per state) was the rest of it — the boxed form paid ~460 small
      allocations per call at 6 photons in 6 modes.

    **Measurement hygiene, recorded because it nearly produced wrong numbers:**
    the first four post-change SLOS runs were taken while another session's job
    held ~28 of this box's 32 cores (load average 35). Identical runs disagreed
    by 86 vs 111 us. Every figure above was re-taken at load 1.66 with
    criterion intervals inside +-0.1%. Check `uptime` before believing a
    benchmark on a shared box.

    **`slos_masked` had the same defect one function down — also fixed
    2026-09-25.** Both entry points now share one DP; the mask is applied as a
    PRUNE during the sweep (a partial state is abandoned once it cannot satisfy
    the mask) rather than only as a filter at the end, so narrow masks do not
    pay for branches that cannot contribute. Held to the old enumerate-and-
    permanent route by `masked_dp_matches_the_masked_permanent_route`, which
    additionally asserts that at least one mask admits a STRICT, non-empty
    subset — without that, every case in the corpus compared empty lists
    against empty lists and the prune was never exercised at all.

    *Known limitation, stated rather than assumed away:* the level arrays are
    still sized `n_compositions(level, m)` whatever the mask says, so a mask
    saves arithmetic, not allocation. For a mask admitting a tiny fraction of a
    very large output space the old route could still win on memory. Compacting
    the indexing would fix it; no caller exercises that regime today.

    **The same pattern survives in `FockKet::evolve`** (`fockstate.rs`), and
    worse: it is a DOUBLE loop computing `fock_amplitude` — one permanent — per
    (input basis state, output basis state) pair, so M_n^2·2^n against a DP's
    n·M_n^2. Same 2^n/n factor, not yet done. This is three instances in one
    crate, so it is a house style rather than three bugs: reach for the shared
    dynamic program before writing another per-output permanent loop.

    **Perceval was not installed on this box at all** until 2026-09-25, which
    is why this had never been measured. Worse, `ci.sh` looks for
    `crates/omega-bridges/python/.venv-perceval` and, not finding it, let
    `test_perceval_conventions.py` call `pytest.importorskip("perceval")` and
    skip silently on every run — so the CONFORMANCE arm the standing-practice
    note reports as "Perceval 2/2" was not running here either. The venv now
    exists (`requirements-perceval.txt`, plus pytest) and those 27 tests pass.
16. **Metal MPS: the SVD still runs on the CPU — and it is 90-97% of MPS
    evolution, confirmed on two ISAs. Until 2026-09-24 it was also SERIAL
    while spending it; that part is fixed on the CPU.** *[2026-10-01: the
    90-97% is the Jacobi kernel's. Accelerate `zgesdd` behind the same SVD
    hook takes it to 58-74% on an M4, worth 4-8x on evolution at depth >= 12
    with the certificate bound intact — the CPU step this entry named as the
    cheaper one. DEFAULT macOS path since 2026-10-02, with the certificate
    naming its kernel; see the closing section.]* The θ-contraction in
    `omega-backend-mps-metal` (Steps 1+2 of `Mps::apply_2q`) is **not
    installed** on the production path; certificate soundness is why — the f32
    kernel under-reports `discarded_weight` (the measurement in item 3). Step
    3, the truncated SVD, falls back to
    `omega_backend_mps::svd::truncated_svd_flat`. The CUDA arm swaps the SVD
    via `SvdFlatFn` (`gesvdj.rs`); the Metal arm has no equivalent.

    **Measured 2026-09-23** (this box, 32 threads) with
    `crates/omega-backend-mps/examples/mps_stage_profile.rs`, which installs a
    timing shim through the production `SvdFlatFn` hook, so it times the real
    kernel on the real path with no library change:

    | q | depth | chi | evolve ms | svd ms | svd % | `execute` ms |
    |---|---|---|---|---|---|---|
    | 14 | 4 | 32 | 0.60 | 0.14 | 23.7% | 9.13 |
    | 14 | 4 | 128 | 0.62 | 0.14 | 22.4% | 4.85 |
    | 14 | 4 | 256 | 0.64 | 0.14 | 22.2% | 4.77 |
    | 14 | 12 | 32 | 97.37 | 90.60 | 93.1% | 169.67 |
    | 14 | 12 | 128 | 106.03 | 95.19 | 89.8% | 275.92 |
    | 14 | 12 | 256 | 105.33 | 94.88 | 90.1% | 278.53 |
    | 14 | 24 | 32 | 352.91 | 332.39 | 94.2% | 427.16 |
    | 14 | 24 | 128 | 865.26 | 824.65 | 95.3% | 1229.93 |
    | 14 | 24 | 256 | 862.85 | 822.49 | 95.3% | 1230.25 |
    | 20 | 12 | 32 | 254.58 | 241.56 | 94.9% | 10766.07 |
    | 20 | 12 | 128 | 698.82 | 671.77 | 96.1% | 29936.01 |
    | 20 | 12 | 256 | 700.47 | 673.11 | 96.1% | 29970.76 |
    | 20 | 16 | 32 | 548.61 | 523.93 | 95.5% | 11306.40 |
    | 20 | 16 | 128 | 4713.12 | 4564.17 | 96.8% | 155589.66 |
    | 20 | 16 | 256 | 6650.00 | 6315.55 | 95.0% | 214481.61 |

    Every depth-4 row sits at 22-24% and every depth >= 12 row at 89.8-96.8%,
    across both widths and all three caps — the split is depth, not chi. The
    24q rows were cut deliberately: at 2^24 the `execute` column alone runs
    about an hour per row to re-measure a reconstruction cost the 20q rows
    already price.

    **The 2026-05-12 deferral (`f9b477e`) rested on two claims, and only one
    survives.** Claim 1 — "the bench shape does not exercise SVD; wallclock is
    flat across chi at 14q x depth-4" — is true of that shape but measured the
    wrong quantity. `mps_chi_profile.rs` calls `execute` with `shots: None`,
    which ends in a dense 2^n `to_statevector`. At 14q x depth-4 the ENTIRE
    evolution is 0.60 ms inside a 4.8-9.1 ms `execute`, so ~90% of what that
    harness timed was reconstruction — which does not depend on chi, and is
    exactly the flatness it reported. At 20q the artefact is extreme: 254 ms
    of evolution inside 10.8 s of `execute`. Claim 2 — Apple's ~100 us
    dispatch floor against ~12 700 sub-sweep dispatches per SVD at n=256, and
    no f64 for the 1e-10 forward tolerance — is untouched by any of this and
    still rules out a Jacobi port.

    **What changes:** the size of the prize is now known rather than assumed.
    Amdahl caps an SVD accelerator at the share above, which at depth >= 12 is
    10-20x on evolution, not the "would not move the bench number" the shape
    argument implied. That is enough to justify the one re-open trigger in
    `f9b477e`'s own list that is matmul-shaped and therefore dispatch-light:
    randomized SVD or block Lanczos, trading the bit-for-bit Jacobi reference
    for a handful of large GEMM dispatches. Accelerate/LAPACK on the CPU side
    is the cheaper first step and keeps f64.

    ~~**Do the CUDA measurement first.** Confirm the post-swap share on a GB10
    before building anything Metal-specific; if `gesvdj` does not move it, the
    kernel is not the bottleneck anywhere and this whole item is mis-aimed.~~

    **DONE 2026-09-25, on an RTX PRO 6000 Blackwell IN THIS BOX. `gesvdj` does
    not move the share — so by this item's own stated criterion, a GPU SVD is
    not justified at these shapes.** (The prerequisite was never blocked on a
    GB10: nobody had checked what hardware akilles has. It has 97 GB of
    Blackwell, CUDA 12.9, sm_120, and all three CUDA backends build and test
    green here. Several "needs NVIDIA hardware" entries in this list are stale
    for the same reason.)

    `crates/omega-backend-mps-cuda/examples/cuda_svd_share.rs`, same circuit and
    same `expectation` path as the CPU profiler, 32 threads:

    | q | depth | chi | CPU Jacobi | gesvdj | |
    |---|---|---|---|---|---|
    | 14 | 4 | 32 | 0.61 ms | 13.49 ms | GPU **22x slower** |
    | 14 | 12 | 32 | 84.4 ms | 155.8 ms | 1.8x slower |
    | 14 | 12 | 128 | 96.2 ms | 263.6 ms | 2.7x slower |
    | 14 | 24 | 32 | 298.0 ms | 466.3 ms | 1.6x slower |
    | 20 | 12 | 128 | 464.6 ms | 860.3 ms | 1.9x slower |
    | 20 | 16 | 32 | 388.4 ms | 468.5 ms | 1.2x slower |
    | 20 | 16 | 128 | 4259.5 ms | **1507.4 ms** | GPU 2.8x faster |
    | 20 | 16 | 256 | 5769.9 ms | **4487.5 ms** | GPU 1.3x faster |

    The SVD share stays **89-97%** on the GPU rows — but for the opposite of
    the encouraging reason. It stays high because `gesvdj` is SLOWER, so the
    SVD keeps dominating a run that now takes longer. The only rows the GPU
    wins are 20q/depth-16 at chi >= 128, where the theta blocks reach 256-396
    columns. Below that the matrices are too small to repay dispatch and
    transfer.

    **What that settles.** CUDA is the BEST case for a GPU SVD here — native
    f64, sub-microsecond dispatch, a mature cuSOLVER Jacobi — and it loses on
    six of eight shapes. Metal starts from strictly worse ground: Apple's
    ~100 us dispatch floor and no f64 against a 1e-10 forward tolerance
    (`f9b477e`'s surviving claim). So a Metal SVD is not justified by anything
    measured here, and this item should stop being framed as "port the SVD".
    What IS justified is the CPU-side work: threading the Jacobi kernel was
    worth 1.17x here and 1.35-1.41x on an M4 (§ the gate change above), and an
    Accelerate/LAPACK comparison keeps f64 and has no dispatch floor at all.

    **TWO DEFECTS FOUND BY MEASURING, both in the shipped CUDA path:**

    * *`cuda_svd_flat` falls back to the CPU per call, silently.* On every GPU
      row, 20-45% of SVD calls returned `None` from the GPU context and went to
      `truncated_svd_flat` with nothing said (135 GPU / 93 CPU at 20q/12/chi128;
      31/21 at 14q/4). The harness counts both because of it — without
      counters those rows would have been published as pure-GPU numbers. Any
      future measurement through `cuda_svd_flat` has the same hazard.
    * *The `gesvdj` path is ~6 orders of magnitude less accurate UNDER
      TRUNCATION.* At 14 qubits the maximum middle-cut bond is 2^7 = 128, so
      chi >= 128 is exact and pins ground truth. There the two paths agree:
      `<Z>` = -4.0725e-15 (CPU, exact) against -4.1389e-15 (GPU, exact). At
      chi = 32, which truncates, the CPU holds at -4.1442e-15 while the GPU
      gives **-1.9921e-9**. Agreement when exact and divergence only when
      truncating points at `gesvdj`'s truncation/discarded-weight handling, not
      at its decomposition. That is exactly the value this item requires to be
      preserved ("whatever lands must return the same relative-discarded-
      weight"), so it needs diagnosing before the CUDA SVD is trusted on any
      truncating run — which is every run that reports a certificate.

    **CUDA HALF RESOLVED 2026-10-02 (akilles, RTX PRO 6000).** Both defects
    had one cause: `gesvdj.rs` never called `gesvdj`. It called
    `cusolverDnZgesvdaStridedBatched` — an APPROXIMATE SVD that reports
    `info != 0` on 20-45% of blocks (the silent CPU fallbacks) and whose
    discarded weight came from the Frobenius identity, which is rounding
    noise when little is dropped (χ=128: 1.26e-14 GPU against 1.07e-28 CPU).
    It now calls the exact `cusolverDnZgesvd`, takes the full thin SVD,
    truncates on the host with the CPU kernel's rule and reports the dropped
    Σσ² tail. Measured at 14q x depth 24, χ = 32: `<Z0>` 2.71e-15 GPU against
    -4.14e-15 CPU (was -1.99e-9), `discarded_weight` 8.002505e-7 on both
    (relative 6.9e-14), and **0 CPU fallbacks on all 28 GPU rows** of the
    share table. Every dispatch is now counted (`cuda_svd_dispatch()`), the
    CLI prints the split, writes it as `cuda_svd` in the JSON, and
    `--device cuda` REFUSES a run that fell back rather than label a CPU
    result a device run (`OMEGA_DEVICE=cuda`, not explicit, reports and
    runs). Tests: `omega-backend-mps-cuda/tests/cuda_svd_parity.rs` (value
    at 1e-12 and certificate at 1e-11 relative under truncation; exact
    regime; every split counted once) and
    `omega-cli/tests/cuda_svd_fallback_is_refused.rs` (a real forced
    fallback via `CUDA_VISIBLE_DEVICES=`). Mutations, each red: restoring
    `gesvda`; halving the dropped tail (GPU certificate 1.54e-13 against CPU
    3.08e-13, re-run 2026-10-02 at landing); removing the dispatch counter.
    **Speed:** the exact solver wins on the large truncating shapes
    (20q x depth 16: χ=128 1398 ms against CPU 4461, χ=256 1544 against
    5990, i.e. 3.2-3.9x) and loses everywhere else, including every 14q row.
    So the share question this item asked is answered for CUDA: the GPU
    moves it only where blocks are large enough to amortise dispatch.

    Whatever lands must return the same relative-discarded-weight so the
    truncation certificate is fed identically, and should be shaped for the
    lanes that will share the MPS core: the Majorana/fermionic track
    (`PLAN-OPEN-20260825.md` §3c) and the qudit track (§3d, where the physical
    index is `d > 2`, the two-site gate is `d^2 x d^2`, not `4x4`, and the
    theta block handed to the SVD is `(chi_l * d) x (d * chi_r)`, not
    `(2 chi_l) x (2 chi_r)`). [Wording corrected 2026-09-27: an earlier
    version of this sentence called the *theta block* `d^2 x d^2`; that is
    the gate's shape. The code is `mps.rs::apply_2q_with_svd_flat_reshaping`.]

    **Side finding, recorded because it will mislead someone again:** anyone
    benchmarking MPS through `execute(shots: None)` is benchmarking
    `to_statevector`, not MPS. Use `expectation`, or this harness.

    ---


    **CONFIRMED ON APPLE SILICON, and the item turned out to be pointed at
    the wrong machine — 2026-09-24.** Run on an M4 (4 P + 6 E cores, rayon
    threads=10, macOS 26.6.2, arm64) with the same harness: depth >= 12 lands
    at **91.2-97.5%** against 89.8-96.8% here. Across ISA, allocator, core
    count and thread count, the share generalises. The reconstruction artefact
    reproduced to three digits — 2.95% and 2.97% of `execute` against ~3% here
    — so that is a property of `execute(shots: None)` and not of either box,
    which is what makes it safe to say the 2026-05-12 shape evidence measured
    the wrong quantity.

    Two corrections to the table above, both from that run:

    * The chi >= 128 rows at 14q sit **outside MPS's useful regime** — the
      backend emits a NOTE on every one (at chi=256, 1,835,008 amplitudes
      against a dense 16,384, 112x larger than the state it encodes). The
      conclusion does not rest on them: the warning-free chi=32 rows read
      93.6-96.1%. State the result there.
    * The depth-4 rows are **sub-millisecond and should not be quoted as a
      band**. 0.83-0.97 ms total with 0.17-0.31 ms of SVD; the M4 saw
      20.6-34.3% where this box saw 22-24%. Directional only.

    **THE KERNEL WAS SERIAL WHILE SPENDING THAT 90-97%.** `svd.rs` gated
    parallelism on `cols >= 128`. The theta blocks reach 64 columns at chi=32
    and 120 at chi=128 on the common shapes — both under the gate — and
    because a deep circuit spends most of its gates at small chi while ramping
    up (`svd.rs:334-336`), even rows PEAKING at 256 columns were serial for
    most CALLS. Confirmed from outside: baseline SVD time is flat to under 1%
    from 1 to 32 threads. So "90-97% of evolution on 32 threads", as this
    entry previously implied, was wrong; it is 90-97% of evolution from a
    single-threaded kernel.

    **Lowering the gate to 64 is worth 1.17x at the default pool and 1.28x at
    its 4-worker optimum** (this box, 32 cores, chi=32, four shapes), with the
    truncation certificate bit-identical — verified by an order-independent
    XOR fingerprint over every `discarded_weight`, plus expectation values at
    17 significant digits, across 1/2/4/8/16/32 threads. Landed.

    The gate carries a thread-count term because bare `cols >= 64` measured
    **1.31x SLOWER at one worker** (`par_iter` pays split-and-join with
    nothing to split). Oversubscription still loses and is deliberately not
    detected: 32 rayon workers on a 10-core M4 measured 1.4x slower, while 32
    workers on 32 real cores measured 1.19x faster. That difference is the
    reason this was measured on both boxes before landing — the M4 run alone
    said "take it", and a 32-thread M4 run alone said "1.4x regression". Both
    were true of their host and neither generalised.

    **What this does to the Metal question.** The prize is real but it is
    smaller than the 90-97% share implies, and the cheapest part of it was not
    on a GPU at all. Remaining CPU headroom is bounded by block-pair count:
    64 columns is 4 blocks at `JACOBI_BLOCK = 16`, so ~2x of width, not the
    thread count. Going below 64 means a smaller block, and `svd.rs:280-292`
    records a measured regression from exactly that (a pair-level tournament
    went BACKWARDS from 4 to 16 threads, 22.2 s to 27.9 s with 129 s SYS) —
    and unlike the gate, block size CHANGES which column pairs meet in which
    round, so it changes the certificate's value rather than just its
    schedule. ~~Next step is unchanged: confirm the post-`gesvdj` share on a
    GB10 before building anything Metal-specific.~~ **Superseded — that
    measurement is DONE (see above, 2026-09-25, RTX PRO 6000), and it was never
    blocked on a GB10; no GB10 exists in the estate.** `gesvdj` does not move
    the share, so by this item's own stated criterion a GPU SVD is not
    justified at these shapes and there is nothing left to confirm before
    deciding. Two paragraphs of one entry disagreed about whether its own
    prerequisite had been met — the "DONE" was appended and this closing line
    was not updated, so a reader arriving at the end was sent to re-run work
    recorded as complete 130 lines earlier.

    **THE CPU STEP THIS ENTRY NAMED IS DONE, AND IT WORKS — Accelerate LAPACK
    `zgesdd` behind the existing `SvdFlatFn` hook, f64 throughout, measured on
    andromeda (M4, 10 rayon threads) 2026-10-01.** Unlike the GPU route this
    one moves the share: **89.2-95.0% down to 57.9-74.5%, and 4.1-7.8x on
    evolution at depth >= 12**. `crates/omega-backend-mps/src/accelerate.rs` (written in
    `omega-backend-mps-metal`, rehomed when it became the default),
    measured by `examples/accelerate_svd_share.rs` (same circuit, same
    `expectation` path and same rows as `mps_stage_profile` and
    `cuda_svd_share`, so all three tables compare).

    *Baseline first, on this box, because the table above is from a 32-thread
    Linux host and nothing says it transfers:*

    | q | depth | chi | recorded svd % (linux) | measured svd % (M4) |
    |---|---|---|---|---|
    | 14 | 4 | 32 | 23.7% | 24.8% |
    | 14 | 4 | 128 | 22.4% | 21.7% |
    | 14 | 4 | 256 | 22.2% | 18.7% |
    | 14 | 12 | 32 | 93.1% | 93.3% |
    | 14 | 12 | 128 | 89.8% | 90.9% |
    | 14 | 12 | 256 | 90.1% | 90.4% |
    | 14 | 24 | 32 | 94.2% | 94.5% |
    | 14 | 24 | 128 | 95.3% | 94.8% |
    | 14 | 24 | 256 | 95.3% | 94.8% |

    The depth split reproduces: depth-4 at 18.7-24.8%, depth >= 12 at
    90.4-94.8%. (20q x depth-16 and all 24q rows were not run — 16 GB box,
    three lanes, and those rows are minutes to an hour each.)

    *After, with the three arms in one process on one circuit — `jacobi` is
    the production kernel, `accelerate` is `zgesdd` on every block, `gated` is
    the shipped entry point (LAPACK above the size gate, Jacobi below):*

    | q | depth | chi | jacobi evolve / svd% | accelerate | gated | gated speedup |
    |---|---|---|---|---|---|---|
    | 14 | 4 | 32 | 0.77 ms / 30.4% | 0.86 ms / 39.8% | 0.81 ms / 26.8% | 0.95x |
    | 14 | 4 | 128 | 0.98 ms / 20.5% | 1.14 ms / 25.1% | 0.76 ms / 20.2% | 1.29x |
    | 14 | 12 | 32 | 79.66 ms / 92.9% | 14.88 ms / 61.2% | 14.39 ms / **60.8%** | **5.53x** |
    | 14 | 12 | 128 | 81.60 ms / 89.7% | 19.50 ms / 58.3% | 19.04 ms / **57.9%** | **4.28x** |
    | 14 | 12 | 256 | 80.52 ms / 89.2% | 19.53 ms / 58.2% | 18.87 ms / 58.6% | 4.27x |
    | 14 | 24 | 32 | 282.09 ms / 93.8% | 52.39 ms / 72.6% | 51.89 ms / **72.1%** | **5.44x** |
    | 14 | 24 | 128 | 595.74 ms / 93.9% | 130.17 ms / 74.6% | 129.29 ms / 74.4% | 4.61x |
    | 20 | 12 | 32 | 205.14 ms / 95.0% | 30.44 ms / 64.8% | 29.64 ms / **65.4%** | **6.92x** |
    | 20 | 12 | 128 | 395.28 ms / 94.5% | 51.42 ms / 62.8% | 50.89 ms / **62.8%** | **7.77x** |
    | 20 | 12 | 256 | 394.96 ms / 94.5% | 52.02 ms / 62.6% | 51.34 ms / 62.3% | 7.69x |

    Read the warning-free chi=32 rows for the headline, as the M4 correction
    above says: 92.9-95.0% becomes 60.8-72.1%, at 5.4-6.9x. The depth-4 rows
    are sub-millisecond and directional only, and the gate is what keeps them
    from being a regression — see below.

    **Why this wins where cuSOLVER lost, in one line:** it is not a better
    Jacobi, it is less work. One-sided Jacobi is O(sweeps · cols² · rows) with
    6-10 sweeps measured; a bidiagonalisation-based SVD is O(rows · cols²)
    once, in a vendor BLAS-3 kernel, with no dispatch and no transfer at all.
    The GPU rows lost to dispatch and PCIe on 64-120 column blocks; there is
    no dispatch here to lose to.

    **The gate is measured, not guessed** (`examples/accelerate_svd_crossover.rs`,
    the θ-block shapes, truncating to half the available rank):

    | shape | jacobi | zgesdd | |
    |---|---|---|---|
    | 4x4 | 1.29 us | 2.54 us | **0.51x — LAPACK loses** |
    | 8x8 | 8.29 us | 7.88 us | 1.05x — a wash |
    | 16x16 | 60.96 us | 29.21 us | 2.09x |
    | 32x32 | 504.54 us | 138.25 us | 3.65x |
    | 64x64 | 3008.50 us | 557.54 us | 5.40x |
    | 256x256 | 79018 us | 13291 us | 5.95x |

    So `MIN_DIM_FOR_LAPACK = 16` on `min(m, n)` — 16 and not 8 because 8 is
    break-even and there are only tens of microseconds between the two points
    to capture. A depth-4 brickwall tops out at 8-column blocks, so the gate
    hands that whole regime back to the Jacobi kernel bit-for-bit, which is
    the difference between the `accelerate` column's 0.87-0.98x on those rows
    and the `gated` column's 0.95-1.29x (both within the noise of a
    sub-millisecond row, which is the point: no regression). This is the same
    lesson `svd.rs`'s own `cols >= 64` gate records, one doubling down.

    **THE CERTIFICATE, which is the part that mattered.** `discarded_weight`
    is a BOUND, and the f32 θ-contraction is off the production path because
    it under-reports one. Three things, all asserted rather than argued:

    * **The reported weight is the max of two independent computations** — the
      dropped Σσ² tail (available here, unlike the CUDA `gesvda` arm, because
      `zgesdd` returns all min(m, n) singular values) and the Frobenius
      identity ‖A‖²_F − Σσ²_kept. The max cannot pick the smaller.
    * **A backward-error allowance of (m+n)·ε·‖A‖²_F on top**, so the weight is
      above the Jacobi reference's BY CONSTRUCTION and not by luck. This is
      load-bearing: removing it reddens the bound test (mutation M1 below), so
      `zgesdd` does land a few ulps under the reference on real inputs. The
      allowance is ~1e-13 relative at the 256-column shapes against discarded
      weights of 1e-3 to 1e-6 relative on any run that truncates.
    * **An untruncated split reports exactly 0.0, with no allowance.** `Mps`
      documents `discarded_weight == 0.0` as "every split was exact" and
      `fidelity_estimate == 1.0` as "the run was exact". An unconditional
      allowance retires both signals, and at 14 qubits with chi >= 128 that is
      EVERY split — measured, it turned those rows from 0.0 into ~1e-11 before
      the condition was added.
    * **Verified on the production path, not only on a corpus.**
      `OMEGA_ACC_VERIFY_BOUND=1` runs both kernels per call through the real
      hook and counts under-reports: **0 of 13 632 splits** across 14q x
      depth 4/12/24 and 20q x depth 12, at chi = 32 and 128, covering the
      ungated kernel and the shipped gated entry point. The harness refuses to
      report "0 undercuts" out of 0 comparisons.

    The forward tolerance is unchanged at 1e-10: UᴴU and VᴴV defect < 1e-10
    and reconstruction < 1e-9 on the same shapes `omega-backend-mps`'s own
    `jacobi_svd_is_unitary_tall_and_wide_at_bond_64_shapes` uses, and the
    singular values agree with the reference to 1e-10·σ_max.

    **AND IT DOES NOT HAVE `gesvdj`'S TRUNCATION DEFECT.** The two-defect note
    above records the CUDA path diverging by ~6 orders of magnitude under
    truncation (chi=32 at 14q: CPU -4.14e-15 against GPU -1.99e-9) while
    agreeing when exact. Here the truncating chi=32 row at 14q x depth-24
    reads `<Z>` = 1.96e-15 (jacobi) against 1.03e-15 (accelerate) — both at the
    same 1e-15 floor as the exact rows. Agreement holds under truncation,
    which is where a certificate is actually reported.

    **ON THE DEFAULT macOS PATH AS OF 2026-10-02 — the decision was taken, it
    went this way, and the certificate now names its kernel.** The measurements
    above are unchanged; what follows is what was decided, what it cost, and
    what is now asserted rather than described.

    *The gate it was behind was wrong by construction.* `omega-cli` reached
    `omega-backend-mps-metal` only through the `metal` feature, whose manifest
    comment exists to keep a stock `cargo build` free of `dep:metal`.
    Accelerate is in the base macOS SDK — linked with
    `#[link(name = "Accelerate", kind = "framework")]`, no optional crate, no
    `build.rs`, nothing to probe at runtime — so being gated beside a GPU crate
    withheld 4-8x from every default macOS build for a reason that did not apply
    to it. Verified rather than assumed, since the whole premise rested on it:
    `cargo tree -p omega-backend-mps --edges normal,build,dev` names no
    `metal`, `objc`, `cuda` or `cudarc` after the move.

    *Where it lives now.* `crates/omega-backend-mps/src/accelerate.rs` under
    `cfg(target_os = "macos")`, with the two harnesses
    (`examples/accelerate_svd_share.rs`, `examples/accelerate_svd_crossover.rs`)
    and the kernel's own tests following the code — the tests compare against
    `svd.rs`'s Jacobi kernel, which is in this crate, so they were never about
    Metal. `MpsBackend::new` and `NoisyMpsBackend::with_model` install
    `default_svd_kernel()`: `AccelerateZgesdd` on macOS, `Jacobi` on every other
    target, which keeps Linux bit-for-bit unchanged. `omega-backend-mps-metal`
    keeps the θ-contraction and loses its `dev-dependencies`, which existed only
    for the harness that left.

    *The second half of the decision, and why it was not optional.* After the
    promotion the same circuit yields certificates that differ between macOS and
    Linux — both sound, the macOS one ABOVE by at most ~1e-12 of the block norm.
    So `MpsRunStats` gained an `svd_kernel` field and the `mps_truncation` JSON
    block carries it (`"jacobi"`, `"accelerate-zgesdd"`, or `"custom"` for a
    hook installed through `with_svd_fn`, which is where the CUDA `gesvdj` arm
    lands because this crate cannot name an arbitrary function pointer). The
    rule is `omega-hostgate status`'s: when two sources can produce a number,
    the record says which one did. A real block from a stock release build with
    no `metal` feature, `examples/circuits/mps_chain_19q.qasm --backend mps:7`:

    ```json
    "mps_truncation":{"discarded_weight":1.226361525032711e-7,
      "fidelity_estimate":0.9999998773638523,
      "fidelity_estimate_is_a_bound":false,"max_bond_reached":7,
      "svd_kernel":"accelerate-zgesdd"}
    ```

    The label names the DISPATCH POLICY, not a count: the size gate hands blocks
    below `MIN_DIM_FOR_LAPACK` back to the Jacobi kernel, so that χ=7 run's
    14-column blocks never reached LAPACK and its certificate is bit-for-bit
    Jacobi's. Asserted, not left to be noticed —
    `below_the_gate_the_accelerate_label_still_carries_a_jacobi_certificate`.

    *What it cost, measured.* One tolerance in the tree had to widen, and it is
    worth naming because it is the exactness signal the kernel's own docs
    worried about. `accelerate.rs` reports exactly `0.0` when a split kept every
    singular value `zgesdd` returned, with no allowance, precisely so that
    "`discarded_weight == 0.0` means every split was exact" survives. But a
    RANK-DEFICIENT split — a CSum on a product pair — drops singular values that
    are numerically zero, and that counts as having dropped something: the
    allowance applies, because a bidiagonalisation cannot tell "I dropped exact
    zeros" from "I dropped something at my own error floor". The Jacobi kernel
    books the true ~1e-32 there. So `qudit_backend.rs`'s exact-regime
    certificate floor, 1e-15, is reachable only by Jacobi; on Accelerate the
    same runs certify **~1.5e-14**, measured. The floor is now read off
    `svd_kernel` — 1e-15 for Jacobi, 1e-12 for Accelerate — so the tight bound
    stays tight where it is reachable instead of being relaxed for every target.
    The amplitude comparison against quditsv at 1e-10, which is what "exact"
    means in that test, did not move.

    *What did NOT change.* The bound is still sound in the direction that
    matters: `accelerate_discarded_weight_never_undercuts_the_jacobi_reference`
    and the `backward_error_allowance` that makes it true moved intact, and
    zeroing that allowance still reddens the test in its new home (M1 below).
    `below_the_gate_the_result_is_the_jacobi_kernel_bit_for_bit` holds. The
    `SvdFlatFn` signature is untouched, so the CUDA arm swaps through it exactly
    as before — `with_svd_fn` gained only the `Custom` label beside it.

    *The acceptance criterion, which nothing proved before.*
    `crates/omega-backend-mps/tests/svd_kernel_on_the_default_path.rs` asserts
    that an unconfigured `MpsBackend` runs Accelerate on macOS, in a crate that
    has no `metal` feature to be passing because of. It checks the label AND the
    arithmetic only Accelerate produces — its bound sits strictly above the
    Jacobi reference's on a truncating 64-column block, by the allowance — so a
    cfg that quietly selected Jacobi would redden rather than cost 4-8x in
    silence (M3 below).

    **THE MUTATION LEDGER.** Three mutations, each one line, each restored and
    `cmp`-verified against a pre-mutation copy. Line numbers are post-rehome.

    | | mutation | reddens |
    |---|---|---|
    | **M1** | `accelerate.rs:140` — `backward_error_allowance` returns `0.0 * …` | `accelerate_discarded_weight_never_undercuts_the_jacobi_reference` ("UNSOUND BOUND at 64x64 ratio=0.5 rank=1: 0.33561807873973 below the reference's 0.3356180787397355"), `below_the_gate_the_result_is_the_jacobi_kernel_bit_for_bit`, and `the_default_macos_hook_adds_the_accelerate_backward_error_allowance`. 6 of 8 `accelerate` unit tests still pass: the allowance is ~1e-13 relative, so unitarity, the σ comparison and the `diag(4,3,2,1)` case cannot see it. This is the mutation line 1666 above refers to, and it still bites in the new home. |
    | **M2** | `sim.rs:373` — `MpsBackend::record_stats` stamps `default_svd_kernel()` instead of `self.svd_kernel` | 4 of 8 in `svd_kernel_on_the_default_path`. **The instructive one:** `a_default_backend_runs_the_target_default_kernel` stays GREEN, because a field hard-coded to the target default agrees with it. Only the two-sided tests catch it — `the_certificate_names_the_kernel_that_was_forced_onto_the_run` ("a run forced onto the Jacobi kernel must say so: left `accelerate-zgesdd`, right `jacobi`") and `an_unnameable_hook_is_reported_as_custom_rather_than_as_the_default`. A one-sided test of this field would have been worthless. |
    | **M3** | `mps.rs:171` — `default_svd_kernel`'s macOS arm returns `SvdKernel::Jacobi`, i.e. the promotion silently does not happen | 4 of 8 in the same file, and NOTHING else in `cargo test -p omega-backend-mps -p omega-backend-mps-metal -p omega-cli` — which is the pre-rehome state, and the reason this file had to exist. `the_default_macos_hook_adds_the_accelerate_backward_error_allowance` fails with the two bounds IDENTICAL to the last digit (1.970926818699761 both sides), so it catches the kernel and not merely the label. |

    **What is left of the Metal question after this.** Nothing has changed
    about Apple's dispatch floor or the absence of f64, and the CPU step the
    entry named as "cheaper first" turned out to be worth 4-8x on evolution
    with the bound intact. The remaining SVD share is 58-74%, so a further
    accelerator is capped at ~2.4-3.9x on evolution rather than the 10-20x
    Amdahl allowed before — on a kernel that is now a vendor BLAS-3 call
    instead of our rotation loop. Randomized SVD and block Lanczos are still
    the matmul-shaped candidates, and they are now competing against zgesdd
    rather than against Jacobi.

17. **No proper comparison against alternative emulators.** Owner requirement,
    2026-09-27. The external implementations in this tree are **correctness
    oracles, not competitors**: qiskit/qiskit-aer, Stim, PyMatching, piquasso,
    MQT and FQE are all wired to answer "is our number right?", and none to
    answer "is our emulator worth using?". The distinction matters because the
    project already learned it once — old item 15 read "the photonic backend
    has never been compared on *speed*, only on correctness", and closing it
    took a separate piece of work.

    What is missing, concretely:
    - **Whole families are absent.** Zero mentions anywhere in the tree of
      quimb, qulacs, cuQuantum or ITensor. For the MPS lane quimb
      and ITensor are the obvious yardsticks; for dense statevector, qsim and
      qulacs; for GPU, cuQuantum. Being unmeasured against any of them is not
      the same as being fast.

      **Correction, 2026-10-07 (EMULATOR-COMPARISON E0).** The clause above
      originally named **qsim/Cirq** among the absent families, and that was
      wrong: `tools/qec_cross_check/run.sh:24` already best-effort
      pip-installs `qsimcirq cirq` into that lane's venv, and both
      `check_qec.py`'s docstring and the README advertise qsim as an extra
      oracle for the encoded demos. The other four names are confirmed absent.
      This corrects the inventory and does **not** touch the ruling: qsim
      would be wired as an oracle, not a competitor, so the gap the entry
      describes is unchanged.

      **And the correction has a second half, which is worse than the first.**
      `check_qec.py` never imports `qsimcirq` or `cirq` and has no qsim code
      path at all — the only optional oracle it actually runs is `stim`
      (`_stim_grover`). So qsim is **installed and documented but never
      executed**: a dangling citation in the A4 sense, in prose that reads as
      coverage. Two consequences, both recorded rather than fixed here. First,
      `run.sh` spends install time on two packages nothing uses. Second, and
      the reason this belongs in item 17: qsim's standing in this tree is not
      "oracle" but "advertised and absent", which is exactly the shape the
      comparison work exists to stop producing. The qsim rows planned for the
      dense lane therefore start from nothing, and carry the f32 annotation
      (see the plan's finding 2) on top.
    - **No agreed workload set.** A comparison is only meaningful at fixed
      (circuit, qubits, depth, χ, shots, precision, hardware). Without that it
      is anecdote — and this repo's own A10 class exists because an
      accelerated path once never ran while the harness said it had.
    - **Capability, not only speed.** Several of the alternatives cannot
      express what we can (certificates as bounds, mid-circuit Reset policy,
      qudits once Track Q lands) and some do things we cannot. An honest
      comparison reports both directions, including where we lose.
    - **Named hardware.** andromeda (M4, 16 GB, Metal) and akilles (RTX PRO
      6000, CUDA 12.9, 123 GB). The estate has no GB10 — any inherited
      comparison plan naming one is stale.

    Planned 2026-10-02; **E0 landed 2026-10-07.** The plan states the
    yardstick per lane and why, pins the workload set before anything runs,
    and fixes the tie rule, precision rules and witness policy in advance. E0
    delivered the parts that everything else waits on:

    - `docs/EMULATOR-COMPARISON.md` — the results document, created empty with
      its rules stated up front: the tie rule (win/loss only at a min-of-N
      ratio >= 1.3x **and** disjoint [min, median] intervals; everything else
      a tie printed with its ratio), the precision rules, the speed-row vs
      capability-row distinction, that the numbers are dated measurements on
      named hardware and **not regression gates**, and that a loss gets
      published at the moment it is measured.
    - `tools/emu_compare/` — the pinned §3 workloads, committed: the HEA
      brickwall of `crates/omega-backend-mps/examples/mps_stage_profile.rs`
      **exported once to QASM2** so every arm ingests the identical file,
      RANDOM-1 at seed 0, the Clifford H/S/CX brickwall at 24q and 100q, and
      LUCJ-16 via the existing `lucj_gen.py`. One generator script, one
      manifest with a sha256 per artifact.
    - `crates/omega-emu-compare` — the row schema as one serde struct for
      every lane, plus a JSON-lines writer. **A row that cannot name its
      witnesses does not serialize**, as a type-level property rather than a
      convention, and a compile-time/worktree git-rev mismatch or a dirty tree
      voids the row. Speed rows and capability rows are different types, so a
      capability fact has nowhere to become an empty speed cell.

    Remaining: E1 (install audit, per-arm floors, the sm_120 go/no-go) and the
    lane phases E2-E7. Every lane phase depends only on E0 plus its own E1
    column, so any lane can land, slip or be dropped alone.

    **E2-E7 landed 2026-10-07/08; the results are `docs/EMULATOR-COMPARISON.md`,
    and that document, not this entry, is where the numbers live.** It
    publishes 26 speed rows, 9 capability or correctness rows and 14 void rows.
    It has a one-line verdict per lane and every published loss in one table.
    Losses: Stim on every stabilizer speed row (to 33,986× on sampling);
    qulacs on HEA (1.74-5.5×), with ties on RANDOM-1 at 24q/26q; qsim f32
    (9.6-13×); and ffsim on LUCJ-16 (8.26×, overhead-dominated). Win: exact
    pauliprop over monoprop on the 14q HEA (1.82×); matched accuracy is a tie.
    `crates/omega-emu-compare/tests/doc_rows.rs` fails if the doc cites a row
    no lane JSONL publishes, cites one whose witnesses are missing or
    unasserted, or leaves a published row uncited.

    **What is still open, stated so it does not read as finished:**
    - E5's GPU rows were never run, because the card is held.
    - E4's mac half is owed. Its §4.4a external-CPU census is Linux-only, so
      it needs a macOS equivalent or an explicit exemption first.
    - E2's Clifford rows gated on ⟨Z₀⟩ = 0.0 exactly, a vacuous gate. Fixed
      2026-10-08: `check()` refuses a constant degenerate gate (§4.3c), and
      all eight stabilizer rows were re-measured on andromeda under
      `[P, UPU†, −UPU†]`. Every classification is unchanged.
    - E1 never ran as a phase. Its competitor reporting-semantics reading was
      done in E7 for monoprop only (no error quantity), and the doc claims
      bound-vs-estimate only where the reporting was read.

18. ~~**`omega-hostgate` is admission-only; nothing in the estate caps a run
    at runtime, and the one thing that tried could not see the process
    tree.**~~ **CLOSED 2026-09-30 — `run --watch` landed and was verified on
    both platforms (`8019131`, corrected by `98adf11`).** The ledger is still
    an admission record by contract, and a run that omits `--watch` is
    unchanged; `--watch` is the opt-in that turns the declaration into a cap.

    What the entry found remains true and is why the shape of the fix is what
    it is. The Q4 tnsim leg reached ~49 GB resident under
    `--host-bytes 40G` on akilles, and andromeda's PID-level watchdog on the
    same script reported a 17 MB peak while the worker subprocess held
    3.7 GiB. Both are one failure: the memory is not in the process that was
    spawned, it is in something that process forked. So `--watch` polls the
    process **tree**, not the PID — Linux by the ppid chain reading `statm`
    field 2, with the cgroup v2 `memory.current` replacing the sum only when
    the child sits in an exclusive leaf; macOS by a `ps` child walk. Over the
    cap, the tree is SIGKILLed children-first and the run exits 5. Every
    watched run prints peak against declaration, including one that stayed
    inside it, so a compliant run leaves a receipt rather than silence.

    Two properties are stated rather than discovered later. **A poll is not a
    limit:** the default interval is 250 ms and an allocation that starts and
    ends inside one interval is not seen. **An unreadable census is not
    zero:** a census that did not run, complained, or cannot read an entry
    that might be in the tree fails closed — tree killed, exit 4 — because
    reading a failed probe as an empty tree reproduces the original 17 MB
    report exactly.

    Verified on akilles in a way that discriminated the two Linux branches
    instead of assuming them: in an exclusive delegated leaf, a 500 MB file
    write was killed at 514.4 M with the leaf's `memory.current` reading
    539361280 afterwards, so `memory.current` was the counter in play; the
    same job beside the watcher with the leaf not exclusive fell back to the
    `/proc` sum and read 5.1 M. A headline-shape run was killed at 361.6 M,
    exit 5, 1.05 s, tree empty 2 s later. That exposes an asymmetry worth
    knowing before an incident: `memory.current` charges page cache and
    `statm` resident does not, so a job in an exclusive leaf can be killed
    for cache it never held as RSS while the same job outside a leaf
    survives.

    The tests are the evidence, and the forking one is the point:
    `tests/watch_run.rs` asserts that a forked grandchild over the cap is
    killed and that the exit status is not the child's — a watcher measuring
    only the root must fail it — that a run inside its declaration is not
    killed and the receipt names both numbers, and, in both directions, that
    without `--watch` nothing is killed and nothing is claimed.
    `98adf11` also corrects `8019131`'s own commit message, which described
    a Linux path reading Pss from `smaps_rollup` with a VmRSS fallback. The
    crate contains no Pss, no `smaps_rollup` and no VmRSS; that description
    came from a reference implementation on another branch, written into a
    message about an implementation it did not describe. Recorded here
    because a commit message cannot be corrected without a rewrite, and
    because asserting a property instead of observing it is precisely what
    this option exists to stop elsewhere.

    **A second defect, found 2026-10-01 while checking andromeda's budget and
    fixed in the same crate.** The ledger records the caps it was created
    with, and every later process read those and ignored its own. On a box
    whose ledger already existed, `OMEGA_HOSTGATE_MAX_MEM=8G` therefore left
    the cap at 16 G — and `status` printed `env:OMEGA_HOSTGATE_MAX_MEM` in the
    "cap from" column beside the 16 G it had not produced, so the one place an
    operator checks reported the change as landed. Measured here: a fresh
    ledger under `MAX_MEM=8G` caps at 8.0 G and refuses a 12 G run; the same
    variable against this session's existing ledger showed 16.0 G with the new
    provenance. Fixed by adopting the configured caps exactly when nothing is
    held — the state an operator who just changed the budget is almost always
    in — keeping the old cap in force underneath a live grant, where lowering
    it would put `charged` above `cap`, and never printing a source that is
    not the one in force. `tests/cap_change.rs` covers both directions plus
    the display; the first version of its idle-box test was itself vacuous
    (the fixture never wrote a ledger, so it compared a new 8 G ledger against
    itself and stayed green with the fix removed) and the mutation is what
    caught that, not review.

    **Closed 2026-10-02 — the default host cap, and not the watch flag.**
    With no `OMEGA_HOSTGATE_MAX_MEM`, no `OMEGA_HOSTGATE_MEM_FRACTION` and no
    profile, the host cap is the ceiling minus a reserve rather than the
    ceiling. The reserve is one eighth of the ceiling, at least 4 GiB where
    the ceiling can spare it (and never more than a quarter of a small one,
    so a 2 GiB pod keeps 1.5 GiB), and never more than 8 GiB. A fraction that
    leaves 4 GiB on a 16 GiB machine leaves 32 GiB on a 128 GiB one; the
    bound is what stops that, and a 128 GiB box keeps 120 GiB. On this
    machine the ceiling is `hw.memsize` 17179869184, which `status` printed
    as 16.0 G from `detected`. The reserve makes it 12884901888, printed as
    12.0 G from `default`. Two lanes of 8 G no longer both fit: the second is
    refused. `OMEGA_HOSTGATE_MAX_MEM`, `OMEGA_HOSTGATE_MEM_FRACTION` and
    `OMEGA_HOSTGATE_PROFILE` still replace the reserve entirely, including an
    explicit request for the whole machine, and an absolute is clamped to the
    ceiling rather than to the reserve. `status` says `default` (or
    `default, of cgroup` when the ceiling was the container's), not
    `detected` beside a number detection did not return. Slots are unchanged.
    Above 80 GiB the reserve has stopped at 8 GiB, so the default is a larger
    share than `greedy` (90%): a profile is a share of the ceiling, not a
    request to exceed the default. On this 16 GiB machine `greedy` is still
    the larger of the two.

    Requiring `--watch` was left out of the crate on purpose. The ledger's
    contract is that a run which omits the flag is unchanged, a declaration
    nobody checks, and that is what every existing caller was written
    against. A default of on would break that. An
    `OMEGA_HOSTGATE_REQUIRE_WATCH` that refuses an unwatched `run` is the
    same omission moved sideways: it binds only a caller who remembers to
    set it, and that caller can put `--watch` on the command. Which
    invocations are watched belongs in the scripts that start them.
19. **`omega-run`'s CUDA statevector arm is f32 only, so there is no
    like-for-like double-precision GPU row against cuStateVec.** Found
    2026-10-06 building the akilles timing plumbing for §5 #17: the arm
    `omega-run --device cuda --backend statevector` reaches is `cuda-f32`
    (the `--timing-reps` line reports `"arm":"cuda-f32"`). A double forward
    path exists in `crates/omega-backend-statevector-cuda/src/f64_path.rs`,
    and nothing in `omega-run` reaches it. The yardstick it would be compared
    against runs double on this card: qiskit-aer-gpu 0.15.1 with
    `cuStateVec_enable=True, precision="double"` on the RTX PRO 6000 (sm_120,
    via the driver JIT-compiling the wheel's sm_90 PTX) reads 1−F = 7.3e-15
    against Aer CPU double on a 20-qubit random circuit, against 1.3e-7 at
    single. So a GPU speed row today compares our complex64 to their
    complex128 and must be labelled so (§17's precision rule), or published
    as a capability row. **Recorded, deliberately not wired:** reaching the
    f64 path from the CLI is a separate decision with its own test burden
    (an f64-vs-CPU differential at 1e-12 on the shapes the f32 arm is
    already checked on, and the timing arm label `cuda-f64`).

20. ~~**`omega-hostgate run --watch` can kill a healthy run on a transient
    unreadable census.**~~ **FIXED 2026-10-08.** Reported from akilles after a
    dense run died with "census did not include live pid …; process tree
    killed" *after* cargo had already printed `Finished`. Receipt:
    `/tmp/dense-run-censusfail.log` on akilles.

    The shape was a race, not a leak. `sample_tree` enumerates the process
    table and then reads each entry, so a process that exits between those two
    steps is absent from the listing; `fold_tree` returns
    `Observation::Exited` for a root it did not see, and the supervisor's
    `Reaction::Exited` arm cross-checked that against **a single**
    `child.try_wait()`. When the two disagreed — census says gone, `wait` says
    alive — it killed the tree immediately. One raced listing was enough.

    The fix is a bounded retry, not a relaxation. Failing closed on a census
    that cannot be read is unchanged and must stay: an unknown footprint is not
    a small one, and a missing pid cannot be read as zero. What changed is that
    a *transient* disagreement is re-sampled before the kill —
    `CENSUS_DISAGREEMENT_RETRIES = 3` at `CENSUS_DISAGREEMENT_DELAY = 50 ms` —
    and the kill message now names how many consecutive samples were taken, so
    the next reader of such a log can tell a race from a broken probe. A
    `FailClosed` or `OverCap` observed *during* a retry is acted on at once
    rather than finishing the budget, because a disagreement is not a licence
    to stop enforcing the cap.

    Three, and not thirty, for a stated reason: a run genuinely blowing past
    its declaration also produces unreadable censuses while the OOM killer
    works, so a patient retry loop turns the gate off at exactly the moment it
    is needed. Three samples bound the extra exposure to 150 ms.

    Tests in `watch.rs`, all three mutation-proven:
    `a_transient_census_miss_does_not_kill_a_healthy_run` (probe misses the
    root on its first call and sees it after; a real short-lived child, so a
    supervisor that killed it would report `CensusFailed`),
    `a_census_that_never_sees_a_live_root_still_fails_closed` (the half that
    must not regress — riding out a transient miss must not become ignoring a
    persistent one; pins the sample count at exactly retries + 1), and
    `a_retry_that_finds_the_run_over_its_cap_kills_it_without_waiting`.
    Mutations, each red: removing the retry budget (all three fail); a retry
    that waits but never re-reads the census (all three); ignoring an over-cap
    observation seen during a retry (the third).

    **Not addressed, and recorded rather than assumed:** the retry treats every
    disagreement alike. The sharper rule would distinguish "pid is gone"
    (exited; `wait` will catch up) from "pid is there and unreadable"
    (permission or memory pressure; kill now), which `/proc` can tell apart and
    `ps` on macOS cannot. That is a platform-split refinement on top of a fix
    that already removes the reported failure.

21. **§4.4a's external-CPU threshold ignores how many threads the row's own
    arm declares.** The census itself is now wired in every lane — it landed
    with E4 (dense, pauliprop, cv), the macOS backing and the pre-row half
    landed 2026-10-08 with the stabilizer lane, and the fermionic lane and
    `mps_quimb_compare` followed the same day onto the shared
    `wait_quiet_external` + `ExternalCpu::with_pre_row` path. What is still
    open is the threshold. It is one figure in cores — 2.0 on a small host,
    8.0 on a large one, reusing `void_above` — and it says nothing about the
    row it is applied to. A 1.5-core neighbour is noise for a single-threaded
    stabilizer row and real contention for a 32-thread dense row, and nothing
    in the schema connects the threshold to the arm's declared `threads`. The
    fix is not obvious: scaling the threshold by `cores - threads` assumes the
    arm actually uses the threads it declares, and a row whose arm declares 1
    thread but calls into a BLAS that opens 10 would then be held to a
    threshold it should not get. §4.3a's qulacs finding (1.37 s at 32 threads
    against 0.0015 s at 16, under load) is the same subject from the other
    side, and whatever rule lands should account for both or say which it
    covers.

22. **The results doc can cite a comparison phase that has no rows.**
    `crates/omega-emu-compare/tests/doc_rows.rs` (E7) refuses a row id the
    doc cites that no lane JSONL publishes, and a published row the doc never
    cites. It does not look at *phase* citations. PLAN-EMULATOR-COMPARISON
    §603 already had one: E3's deliverable says "as verified in E1", and E1
    never ran as a phase. That is the same dangling-citation defect one level
    up, and nothing would catch it in the doc.

    Not a generic rule. "Phase X is cited, so rows with phase X must exist"
    would fire on the doc's most honest sentences: "E5: no rows, the lane is
    written and has not been run" and "E3 published no speed row". Those
    state an absence on purpose. The check needs an **allow-list of declared
    absences**: a phase may be cited with no rows only next to an explicit
    marker the test recognises, for example a line naming the phase and the
    reason, kept in one table in the doc. Then a citation that is neither
    backed by rows nor declared absent fails. Mutation-prove it the way
    `doc_rows.rs` is: cite a phase with no rows and no marker, and delete a
    marker while its sentence stays.
