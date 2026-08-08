# tsim / ppvm bridges — integration notes

Both QuEra simulators are **working and validated against Qiskit**. This
file records the API investigation behind the two runners, the choices
made, and the one repository-level surprise that changed how the
validation harness had to be written.

Verified 2026-08-06 on macOS (Darwin 25.5.0, Apple Silicon), Python
3.12.13, `bloqade-tsim` 0.1.5, `ppvm` 0.1.0 @ `661fc66`.

---

## 1. The finding that shaped the harness: the 369-fixture corpus is not in this repo

`crates/omega-bridges/src/lib.rs` and `tests/cross_backend.rs` both refer
to a `verify-qiskit/fixtures/` tree of 369 QASM2 fixtures. **That
directory does not exist in this repository.** It is private and not
vendored — `crates/omega-cli/tests/bridge_smoke.rs` already hit the same
wall and left a comment saying so.

Consequences, all of which the new code handles explicitly:

- The pre-existing `perceval_matches_qiskit_within_threshold` and
  `bloqade_matches_qiskit_within_threshold` arms are **dormant here**.
  They exit early on the missing venv; if an operator builds those venvs
  in this checkout, they will fail at `assert!(!fixtures.is_empty())`,
  not pass. (Left as-is — outside this change's scope, but worth
  knowing.)
- The new arms therefore prefer `verify-qiskit/fixtures/` when it is
  present (walking it recursively, so the coverage count is over all
  369) and otherwise fall back to a **self-contained corpus vendored at
  `crates/omega-bridges/tests/fixtures/crosscheck/`** — 11 QASM2 files
  covering the Clifford core, the continuous rotations, the `u`-family,
  `sx`/`sxdg`, a T-ladder, a lowered QFT, partial measurement, a
  unitary-only circuit, and two files deliberately outside one or both
  gate sets so the filter itself is exercised.

**Fixture qualification, measured:**

| corpus | total | tsim | ppvm |
|---|---|---|---|
| `tests/fixtures/crosscheck` (vendored, this repo) | 11 | 10 | 9 |

- tsim drops 1: `11_controlled_rotation.qasm` (`crz`).
- ppvm drops 2: `11_controlled_rotation.qasm` (`crz`) and
  `10_swap_toffoli.qasm` (`swap`, `ccx`).

The per-backend count against the private 369-fixture corpus cannot be
stated from this checkout. The filter that produces it is implemented
and runs automatically the moment `verify-qiskit/fixtures/` is present;
the test prints `"N of M fixtures … are inside the <slug> gate set"` on
every run, so the number is reported rather than assumed.

---

## 2. tsim — API investigation and ingestion choice

`bloqade-tsim` (PyPI, Apache-2.0, 0.1.x beta) is a ZX-calculus
stabilizer-rank sampler built on `pyzx-param` + JAX, with `stim` as a
dependency.

```
tsim.Circuit(stim_program_text)          # wraps stim.Circuit
  .compile_sampler(strategy='cat5', seed=None) -> CompiledMeasurementSampler
      .sample(shots, batch_size=None)    -> np.ndarray[bool] (shots, n_measurements)
```

**Ingestion choice: lower QASM2 → extended-Stim text in-repo.** tsim has
no QASM front end. Its constructor takes Stim program text plus a
shorthand pre-pass (`tsim.circuit.shorthand_to_stim`) that expands the
non-Clifford sugar Stim itself cannot express.

The runner emits the **canonical** bracket form rather than tsim's
shorthand:

| QASM2 | emitted | why canonical |
|---|---|---|
| `t` / `tdg` | `S[T]` / `S_DAG[T]` | tsim's `T`→`S[T]` rewrite is guarded by a `(?<!\[)` lookbehind, so the canonical form passes through untouched |
| `rx/ry/rz(θ)` | `I[R_X(theta=<θ/π>*pi)]` | tsim's `R_X(<float>)` shorthand only fires on a bare float, so the named-parameter form passes through |
| `u/u3(θ,φ,λ)` | `I[U3(theta=…*pi, phi=…*pi, lambda=…*pi)]` | same |

The canonical form is what **ppvm's parser also accepts**, which is the
whole reason one converter serves both bridges (see §3).

**Gate-convention verification.** Checked numerically against
`tsim.Circuit.to_matrix()` before writing a line of the runner:

- `I[R_Z(theta=c*pi)]` = `diag(e^{-iθ/2}, e^{iθ/2})` — exactly QASM2's
  `rz`, no phase fudge.
- `I[R_X/R_Y]` match the standard `cos(θ/2) ∓ i sin(θ/2)` forms.
- `I[U3(θ,φ,λ)]` equals QASM2's `u3` matrix **exactly** (global phase
  1.0, not merely up to phase) over three random parameter triples.
- `S[T]` = `diag(1, e^{iπ/4})`.

Angles are in **half-turns** (`theta=<c>*pi`), and the coefficient is
written with `repr(float)` so it round-trips exactly through IEEE-754
across the text boundary.

**Seeding:** `compile_sampler(seed=…)` is honoured. tsim's own caveat
applies: the auto-chosen `batch_size` also affects the stream, so
reproducibility is per-machine.

**Cost:** 4M shots of a 3-qubit circuit in ~6.9 s. Runtime scales with
T-count, not qubit count.

---

## 3. ppvm — API investigation and ingestion choice

`ppvm` (GitHub, Apache-2.0, v0.1.0) is a Rust workspace
(`ppvm-pauli-sum`, `ppvm-tableau`, `ppvm-stim`, `stim-parser`,
`ppvm-cli`, …) with maturin-built Python bindings shipped from the
`ppvm-python/` subdirectory. **Not on PyPI** — git install only.

Two engines, and only one of them can answer this protocol:

1. `PauliSum` — Heisenberg-picture Pauli propagation. Produces
   **expectation values** (`overlap_with_zero()`), a single number. It
   cannot honour `shots` without misrepresenting what it computed, so
   it is not what the counts protocol wants.
2. `GeneralizedTableau` — stabilizer tableau extended with non-Clifford
   gates and measurement. **Forward sampling**, including noise and atom
   loss. This is the right engine.

**Ingestion choice: `ppvm.sample_stim`, not `ppvm-cli`.**

```
ppvm.StimProgram.parse(src) -> StimProgram
ppvm.sample_stim(prog, n_qubits=None, min_abs_coeff=1e-10,
                 num_shots=1, seed=None) -> list[list[MeasurementResult]]
```

One in-process call that samples all shots in parallel across CPU cores
with the GIL released. `ppvm-cli` would add a second subprocess hop, a
temp file, and a text-output parse for no capability we need. The
bindings are a first-class mixed Rust/Python wheel, so there is no
build-from-source cost beyond the one-time `pip install git+…` (which
does need a Rust toolchain — this repo already pins one).

**Dialect compatibility with tsim.** ppvm's `stim-parser` promotes the
*same* tag extensions into first-class AST nodes
(`crates/stim-parser/src/pipeline/lower.rs::interpret_identity_tag`):
`I[R_X|R_Y|R_Z(theta=<c>*pi)]` → `ExtendedInstruction::Rotation`,
`I[U3(theta=…, phi=…, lambda=…)]` → `::U3`, `S[T]`/`S_DAG[T]` → `::T`/
`::TDag`. Its printer emits the identical text. Hence one converter,
two backends.

**Gate-set divergence (verified by parsing each opcode against
`StimProgram.parse`).** ppvm's executor marks `SWAP`, `ISWAP`,
`ISWAP_DAG`, `SQRT_XX/YY/ZZ`, `CXSWAP`, `SWAPCX`, `XCX/XCY/XCZ`,
`YCX/YCY/YCZ`, `CXYZ`, `CZYX`, `HXY`, `HYZ` as
`unreachable!("… rejected by validate")`, and it has no CCX/CCZ sugar.
So `swap`, `ccx`, `ccz` are **tsim-only** in the shared converter, and
QASM2 using them comes back as `ppvm-unsupported-gate`. Everything else
in the subset (`h x y z s sdg t tdg sx sxdg rx ry rz p u1 u2 u3 u id u0
cx CX cy cz`, plus `measure`/`reset`/`barrier`) parses on both.

**Seeding:** `sample_stim(seed=…)` is honoured and reproducible.

**Cost:** 4M shots of a 3-qubit circuit in ~2.7 s.

**`MeasurementResult` has three members**, not two — `ZERO`, `ONE`, and
a loss/erasure outcome. `qasm2_stim.bits_to_counts` hard-errors on
anything that is not 0 or 1 rather than letting a third value coerce
into the `1` bin, which is the one place this integration could have
silently corrupted a distribution.

---

## 4. The shared converter: what it refuses and why

`crates/omega-bridges/python/qasm2_stim.py` is deliberately **total over
a small subset**. Anything outside it raises `UnsupportedGate`, which
the runners tag `tsim-unsupported-gate` / `ppvm-unsupported-gate` and
the Rust dispatcher surfaces as `BridgeError::Backend`.

Refused on purpose:

- **User `gate` / `opaque` definitions** — flatten upstream instead.
- **`if (…)` classical conditioning** — no lowering exists.
- **Controlled rotations (`crz`, `cu1`, `cp`, …)** — the converter
  folds `u1(λ)`/`p(λ)` onto `R_Z(λ)`, which differs by a global phase.
  That is exact for measurement statistics of an *uncontrolled* gate and
  **wrong** under a control. Refusing is the only correct answer, and
  it is why controlled rotations are excluded rather than "approximated
  well enough".
- **Everything not in the per-backend table.**

Also handled, because Qiskit's runner does and the comparison would
otherwise be apples-to-oranges:

- No `measure` in the source → synthesise a full-width classical
  register and measure every qubit in index order, mirroring
  `qiskit_runner.py`'s `circuit.measure_all()`.
- Partial measurement and multiple `creg`s → each measurement is placed
  at its **global classical-bit index**, bits nothing wrote stay `0`,
  and the output string is LSB-first (clbit 0 leftmost) — byte-identical
  in convention to `qiskit_runner.py`'s `key.replace(" ", "")[::-1]`.

**Noise is refused, not ignored.** Neither runner maps omega's opaque
noise dict onto its backend's channels yet, so a request carrying
`noise` returns `tsim-noise-not-supported` / `ppvm-noise-not-supported`.
Returning a noiseless distribution to a caller who asked for a noisy one
is exactly the silent wrongness the house doctrine forbids. (ppvm's
tableau *does* model depolarising / Pauli / atom-loss channels — wiring
them is a well-defined follow-up, not a blocker.)

**One source of truth for the gate sets.** The runners answer
`{"mode":"gates"}` with their supported QASM2 gate names, and
`tests/cross_backend.rs` queries that instead of hard-coding a Rust
copy. The fixture filter therefore cannot drift from the converter.

---

## 5. Why the new arms use 4M shots where Bloqade uses 1M

Bloqade's arm gets away with 1M because its side is **exact**
(probability × shots): the only binomial noise is Qiskit's, so
`Var(Δp) = p(1−p)/N`. tsim and ppvm are Monte-Carlo samplers, so *both*
sides carry that variance and `Var(Δp) = 2p(1−p)/N`.

The tightest case is a two-outcome fixture (Bell, GHZ) where
`L2 = √2·|Δp|` at `p = 1/2`, giving `sd(Δp) = √(0.5/N)`. The 0.0025
threshold then sits at `0.0025 / (√2 · √(0.5/N))` standard deviations:

| N | margin | false-failure rate per fixture |
|---|---|---|
| 1M | 2.5σ | ~1.2% — flaky, not a gate |
| 4M | 5.0σ | ~6e-7 — matches Bloqade's 1M safety |

This was not theoretical: at 20k shots every fixture "failed", at 1M the
worst observed L2 was 1.65e-3 against a 2.5e-3 threshold, i.e. inside a
1.5× margin. The dispatcher does not forward a seed (`RunnerRequest`
has no seed field), so these runs are genuinely random on every
invocation — the margin is what keeps the gate honest rather than lucky.

Cost at 4M shots on a 3-qubit fixture: qiskit 1.5 s, ppvm 2.7 s,
tsim 6.9 s. Whole-arm wall time: ppvm 32 s, tsim 73 s.

---

## 6. Measured results

Vendored corpus, 4M shots, threshold L2 ≤ 0.0025 (the Perceval /
Bloqade threshold from `backend.py:147`).

```
tsim_matches_qiskit_within_threshold: 10 of 11 fixtures
  01_single_qubit_basic.qasm               L2 = 5.4262e-4
  02_single_qubit_rotations.qasm           L2 = 6.0215e-4
  03_sqrt_x.qasm                           L2 = 7.1771e-5
  04_bell_phi_plus.qasm                    L2 = 1.2693e-4
  05_ghz_3.qasm                            L2 = 1.2339e-4
  06_qft_3.qasm                            L2 = 5.2206e-4
  07_clifford_t.qasm                       L2 = 7.9930e-4
  08_partial_measure.qasm                  L2 = 1.2869e-4
  09_unitary_only.qasm                     L2 = 5.5987e-4
  10_swap_toffoli.qasm                     L2 = 6.3356e-4

ppvm_matches_qiskit_within_threshold: 9 of 11 fixtures
  01_single_qubit_basic.qasm               L2 = 4.1478e-4
  02_single_qubit_rotations.qasm           L2 = 4.6094e-4
  03_sqrt_x.qasm                           L2 = 1.4213e-4
  04_bell_phi_plus.qasm                    L2 = 6.4170e-4
  05_ghz_3.qasm                            L2 = 3.6098e-4
  06_qft_3.qasm                            L2 = 5.8876e-4
  07_clifford_t.qasm                       L2 = 4.5232e-4
  08_partial_measure.qasm                  L2 = 2.7860e-4
  09_unitary_only.qasm                     L2 = 5.6483e-4
```

An additional out-of-tree sweep of 30 circuits (hand-written structural
cases plus 16 pseudo-random Clifford+T and rotation circuits on 2–4
qubits) ran clean at 1M shots against Qiskit with the same threshold:
worst tsim L2 1.65e-3, worst ppvm L2 1.13e-3, zero failures. That sweep
is what motivated the 4M shot count above.

---

## 7. Reproducing

```bash
# One-time venv setup (both need Python >= 3.10; macOS system python3 is 3.9)
make -C crates/omega-bridges/python qiskit-venv
make -C crates/omega-bridges/python tsim-venv PY=python3.12
make -C crates/omega-bridges/python ppvm-venv PY=python3.12   # builds Rust

# Per-runner smoke
make -C crates/omega-bridges/python check-tsim
make -C crates/omega-bridges/python check-ppvm

# The gate: two independent codebases must agree
cargo test -p omega-bridges --features bridge-qiskit,bridge-tsim \
  --test cross_backend tsim_matches -- --nocapture
cargo test -p omega-bridges --features bridge-qiskit,bridge-ppvm \
  --test cross_backend ppvm_matches -- --nocapture
```

---

## 8. Deliberately not done

- **Noise models.** Refused loudly on both bridges (see §4). ppvm's
  tableau supports the channels; the mapping from omega's noise dict is
  the follow-up.
- **`PauliSum` expectation values.** A different output shape than
  `Counts`; exposing it needs a `run_expectation`-style dispatch entry,
  not a counts backend.
- **Bloqade's `ahs` analog mode.** Untouched, still stubbed — the analog
  work lives elsewhere.
- **`omega-cli` feature forwarding for bloqade.** `bridge-tsim` /
  `bridge-ppvm` were added to `crates/omega-cli/Cargo.toml`;
  `bridge-bloqade` was already missing there before this change and was
  left alone rather than folded into an unrelated commit.
