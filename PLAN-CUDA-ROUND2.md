<!-- SPDX-License-Identifier: Apache-2.0 -->
# Plan — CUDA round 2 (2026-08-17), revision 2

Revision 1 was reviewed adversarially and **four of its seven "established
facts" were wrong**. The corrections are kept in place rather than edited out,
because the pattern matters: revision 1 was written from
`LINUX-CUDA-VERIFICATION.md`, `CUDA_TODO.md` and memory of round 1, and only
glancingly from the source. Its two most confident sections were its two worst.

Box: DGX Spark **GB10**, aarch64, sm_121, 48 SMs, 24 MiB L2, 121.7 GiB unified,
CUDA **13.0.88**, driver 580.159.03, 20 cores, Python 3.12.3.

Standing constraint unchanged: **must keep working on amd64/Xeon + RTX 6000 Pro
or H100, and on nvcc 12.x**, neither of which is here. Any item whose failure
mode is invisible on GB10 does not land.

---

## Facts, corrected

| revision 1 claimed | actually |
|---|---|
| CUDA has a collapse arm whose counts width is in question | **There is no collapse arm.** `lib.rs:917-924` refuses `MidCircuitMode::Collapse` outright so the CLI falls back to CPU. The question was mis-posed. |
| item 2 "needs a dependency on the CPU crate" | **Already a non-dev dependency** — `Cargo.toml:23`. The claim was copied from a stale comment at `lib.rs:1519`. |
| "S2 is implemented" | **Only the kernel half.** §3.4's dedicated rayon pool, §3.5's `with_threads`/`ARIA_THREADS`, and three of §3.6's tests are absent. `PLAN-SV-PERF.md:4` still says "S2–S5 open" and is right. |
| graph wiring is "a pooled variant per kernel" | **4 dense build sites, not 2**; `forward_graph` is `#[allow(dead_code)]`, test-only; and CRz's *derivative* is structurally impossible through `apply_quad_phase`. |
| `reset_is_deterministic_within` | correct — `sim.rs:546`, public, Metal already uses it |
| `torch 2.7.0+cu128` aarch64 wheel exists | correct, **and now proven end to end** — see item 11 |
| CY still dense | correct (`lib.rs:664`) |

---

## 1 — Collapse counts width: the question is mis-posed, not open

`LINUX-CUDA-VERIFICATION.md` §1 asks whether the CUDA collapse arm keys counts
on `circuit.num_qubits` or the creg width. **It keys on neither, because it does
not exist**: `lib.rs:917-924` returns `Unsupported` for any `Measure` under
`MidCircuitMode::Collapse`.

Revision 1 specified a test that runs a narrow-creg circuit under `Collapse` and
pins the width against `counts_outcome_width(&c, true)`. That test receives
`Err(Unsupported)` — there is no `Outcome` to pin — and the assertion, being
wrong, would go red. Revision 1 then pre-committed to *"the fix, not the
tolerance, is the response"*, i.e. to changing `lib.rs:1001` to the creg width.
`Outcome::from_u64` masks silently (`outcome.rs:94-98`), so that would truncate
full-register sampler keys to the creg width: **a confident wrong answer on any
circuit containing a measure.** This was the only item in the plan whose stated
procedure introduced a defect, and it was scheduled first.

The two width sites that *do* exist — `lib.rs:967` (reset trajectory) and
`lib.rs:1001` (shots) — key on `circuit.num_qubits`, and **both are right**:
`sample_counts_on_device` samples the full qubit register, so
`counts_outcome_width(circuit, false) == num_qubits`, which is what the CPU
computes for these shapes too.

**Do instead, both cheap:**
- pin the **refusal**: CUDA + `Collapse` + `Measure` ⇒ `Unsupported`. That is the
  real contract and nothing tests it;
- pin the existing width against `counts_outcome_width(&c, false)` on the
  shots/reset arms, so the two backends cannot drift.

Then close §1 as **answered: mis-posed**. No `ci.sh` change — `ci.sh:384`
already runs the crate's `tests/`.

Also record (not fix): CUDA never calls `check_counts_width` where the CPU does
(`sim.rs:87`). Unreachable today, same divergence family.

## 2 — Reset criterion: real, but it trades a safe refusal for a tolerance

Adopt `reset_is_deterministic_within` (`sim.rs:546`, public, already used by
Metal). No dependency work — the crate is already a dependency.

**The risk revision 1 missed: this replaces a criterion that is over-strict but
never wrong with one that can be wrong.** At `tol = 1e-4` a purity of 0.9999 is
admitted — a Schmidt weight ≈ 5e-5, amplitude error ≈ 7e-3, three to four orders
outside this repo's own gates (5e-7 f32, 1e-9 cross-check). f32 device noise on
purity is ~1e-6, so **1e-4 is ~100× looser than needed**; it was inherited from
Metal without re-deriving at CUDA's larger `MAX_QUBITS`. → use **1e-6**, and
justify it against measured purity noise at the largest n that runs here.

**The named verification was vacuous.** `reset_matches_cpu` (`lib.rs:2020`) uses
an *entangled* reset (`H; CX; Ry; Reset q0`) — refused by both backends before
and after, so it stays green whether the change works or not. Needs:
- a **new** test on `H q0; Reset q0`: now accepted, compared to CPU **on ⟨Z⟩,
  not amplitudes** — the CPU draws its branch from the RNG (`sim.rs:783`) while
  CUDA picks deterministically (`lib.rs:1536`), so on `|−⟩` the two differ by a
  global sign and an amplitude assertion would flake;
- a near-boundary **entangled** case that must still refuse.

**Cost, to be measured not assumed:** `reduced_purity` (`sim.rs:508-522`) is a
**serial host loop** over `dim`, and `read_state()` materialises 16 B/amp from an
8 B/amp device buffer — peak host memory doubles — **per Reset op**. At n=28 that
is ~4 GiB and 2.7e8 serial iterations each time. Record it in `LIMITATIONS.md`
rather than deleting the row and implying the cost is free.

Delete the stale "needs a dependency" claim at `lib.rs:1519` and
`LIMITATIONS.md:207`.

## 3 — CY, with the phase direction written out

Derived independently: `perm_2q_to_cuda` = `[0,2,1,3]` (`adjoint.rs:394`), so
the quad slot is `bit_qa + 2·bit_qb`. Applying it to `gates::cy()`:

```
out[1] = -i · in[3]        out[3] = +i · in[1]
```

**The phase attaches to the destination slot.** Written out because CY is
Hermitian and an involution, so a sign error survives `CY·CY = I` and would only
surface against Qiskit. Matches the existing dense matrix (`lib.rs:658-663`,
`u[1][3] = -i`, `u[3][1] = +i`).

**Do:** `apply_quad_swap_phase` — exchange two slots, each multiplied by its own
phase, through the shared `cmul` (operand order included).

**Exactness:** with `a = (0, ∓1)`, `cmul` computes `0·v.x − (∓1)·v.y` and
`0·v.y + (∓1)·v.x` — every product is `x·0` or `x·(±1)`, exact in IEEE-754, so
FMA contraction has nothing to round whichever product nvcc fuses. CRz's
contraction problem genuinely does not apply.

**But the bar is BOTH bars, not just bit-identity.** On the zeros-seeded state
the dense path's three `cmul((0,0), v)` accumulations canonicalise `−0.0 → +0.0`
and the specialised kernel does not — so CY takes `Bar::Identical` on
`fill_dense` and `Bar::ValueEqual` on `fill_with_zeros`, exactly like
CX/SWAP/CZ. Revision 1's unqualified "bit-identity is the bar" would have sent
someone hunting a nonexistent bug.

**Index-gate** as two-slot (`QUAD_KERNEL_MIN_QUBIT_2SLOT`), add to
`kernel_resource_report`'s list (it must show `local == 0`), and note CY stays
dense in the graph path.

**Ordering: 6a must land first** — a new params struct with `real` phase fields
is another f64 layout landmine of exactly the class 6a fixes.

## 4 — Graph wiring: **DESCOPED to coverage + an honest annotation**

Revision 1 called this "a pooled variant per kernel". The review found it is
3–5× that, and one part is not merely large but **impossible as specified**:

- **`apply_quad_phase` structurally cannot carry the CRz derivative.**
  `gates::dcrz` is `diag(0, 0, −i/2·e^{−iθ/2}, +i/2·e^{+iθ/2})` → CUDA-order
  `diag(0, x, 0, y)`. The kernel multiplies only the listed slots and **has no
  way to write zero into slots 0 and 2**. Routing a derivative through it yields
  a silently wrong gradient on the live QML training path.
- **4 dense build sites**, not 2: `forward_graph.rs:494`,
  `backward_graph.rs:1920`, `:2057` (dagger), `:2105` (deriv).
- **`forward_graph` is `#[allow(dead_code)]`**, callers are tests only
  (`lib.rs:2481, 2584, 2690`). Only `TrainStepGraph` is live.
- 6 new pools × (field + host mirror + alloc + memcpy + plan count), `KernelKind`
  matched in 7 places, and `DaggerBoth2q` fires exactly for CX/CZ/SWAP so
  preserving node-halving needs `*_pooled_dual` too — **4 new `.cu`, not 2**.
- Pooling turns 6a's dormant f32/f64 layout mismatch into a **stride error**:
  by value it is one bad launch; `pool[slot]` for `slot > 0` reads the wrong
  address.
- The index gate must be replicated into `build_plan`, and dense/quad pools then
  have diverging slot counters.

**And the gate revision 1 named does not cover this.**
`train_step_graph_matches_naive_backward` (`lib.rs:2730-2914`) contains **only
CX** among 2q gates — no CZ, SWAP, CRz, CY anywhere in the crate's tests. So
`deriv_2q_pool`, `PlanEntry::Deriv2q` and the unfused dagger arms are
**uncovered today**.

**Do:** take the severable option round 1 already annotated. Land the **missing
graph-vs-naive coverage on a CX+CZ+CRz+CU3 circuit** — worth doing on its own,
and it closes a pre-existing hole — and write the real numbers above into the
annotation so the next attempt starts from them. **Leave the graph dense.**

## 5 — Phase-kernel split: measure, keep both

Revision 1 proposed deleting `apply_quad_phase1` if it now matches. **That
violates this plan's own rule**: register allocation and spilling are properties
of compiler version and target arch, so a single GB10 / CUDA 13.0.88
measurement is exactly "invisible here, fatal on amd64".

Also, the experiment as written **is already the status quo**: `apply_cz` calls
`apply_quad_phase`, which redirects to `apply_quad_phase1` for a single entry
(`imp.rs:1713-1717`). Running it requires bypassing that redirect.

**Do:** bypass the redirect temporarily, record the number, **keep both kernels
regardless**. (`kernel_resource_report.rs:57` also hard-references
`apply_quad_phase1`.)

## 6a — f64 twins **(moves to first)**

`QuadPhase1Params`/`QuadPhaseParams` hardcode f32; under `-DOMEGA_REAL=double`
the device structs are 32 B / 96 B against 20 B / 60 B. Add `…F64` twins
following `Apply1qParamsF64`'s explicit `_pad`. Precedes items 3 and 4.

Worth noting while there: the Rust `#[repr(C)]` structs and the `struct`
re-declared in each `.cu` are hand-duplicated with **no size assertion
anywhere**. A `size_of` test is cheap.

## 6b — `--precision` flag: **DROPPED**

`f64_path` is `zero`/`to_host`/`apply_1q`/`apply_2q`/`expectation_z` — **no
`Backend` impl, no `CircuitIR` walker, no `GateKind` dispatch, no sampling, no
observables**. `omega-run` defaults to 1024 shots, so `--precision f64` would
refuse the default invocation, every `--expectation`, and every `--gradient`.
Combined with "not filling out `f64_path`", that is **a flag that always
errors**.

It also collides with an open defect: `Precision::bytes_per_amplitude` is unused
by the governor (`CUDA_TODO.md:255`), which prices device work at f32 8 B/amp —
so an f64 run would be admitted at half price and OOM.

**Keep only the useful half:** an audit, *written as a test*, that every
`GateKind` the CUDA backend accepts is reachable from `omega-run` in f32. Record
the f64 reachability gap in `BACKEND_FEATURE_PARITY.md` with the governor defect
named as its prerequisite.

## 8 — Topology by box type: TODO only, no code

Unchanged. Classify on **box type** — memory kind, `integrated`,
`pageableMemoryAccessUsesHostPageTables`, device-vs-host capacity — not a 25%
tolerance a GB10 cannot feed (`nvidia-smi` returns `[N/A]`). Record the measured
probe, the rule 2/rule 3 ordering trap, the `has_pool_for` consequence, and the
pre-existing RTX 6000 Pro misclassification. **Needs an amd64 box; lands
nothing here.**

## 9 — Scaling: correct the premise first

"S2 landed" is over-broad. What landed is the **kernel half**. Absent:
`PLAN-SV-PERF.md` §3.4's dedicated `OnceLock<ThreadPool>` (the code uses rayon's
**global** registry), §3.5's `with_threads`/`ARIA_THREADS`/`available_parallelism`
reporting, §3.6's `current_num_threads()==1` assertion and pool-construction
counter, and §3.2's `apply_ccx`/`apply_cswap` decomposition (both still serial,
`sim.rs:1062`, `:1108`).

`RAYON_NUM_THREADS` **is** respected — but only because §3.4 was skipped. The
sweep is banking on a violated design decision; say so rather than rely on it
silently.

**Bit-identity across T holds** and the sweep is safe: no `par_iter().sum()`,
`reduce` or `fold` anywhere in `sim.rs`; every float reduction is serial, and
chunk boundaries depend only on `dim` and the target qubit.

**Do:** sweep `T ∈ {1,2,4,8,16,20,40}` **on `--statevector`, not shots** —
`expectation_pauli` (`sim.rs:1190`) and `sample_counts` (`:1153`) are serial, so
a shots benchmark measures Amdahl and would report a flat curve and blame rayon.
Extend `tests/thread_count_invariance.rs` to T=40 (it uses a scoped
`ThreadPoolBuilder`, so this works).

**Two hazards to expect:** `sim.rs:278` feeds `rayon::current_num_threads()` into
`capacity::check_adjoint`, so **admission is thread-count dependent** — the same
batch can pass at T=1 and be refused at T=40. And
`thread_count_invariance.rs:114-127` is named `the_serial_and_parallel_paths_agree`
but **never compares the two paths**; worth fixing while there.

**Framing:** this is a 20-core **arm64** curve. It does not close the many-core
**x86** item.

## 11 — tch on GB10: **DONE, and it was never blocked**

Proven end to end on this box:

```
tch::Cuda::is_available()      = true
tch::Cuda::device_count()      = 1
tch::Cuda::cudnn_is_available()= true
GPU compute: sum(2*[1,2,3])    = 12
```

`torch 2.7.0+cu128` aarch64 + `tch 0.20` (`torch-sys` wants exactly `"2.7.0"`) —
**no PyTorch rebuild, no version bump.** sm_121 is reachable under cu128, which
was the one genuine unknown.

**Two real defects found and fixed** (neither previously known — both are
aarch64-specific, which is why the x86 verification missed them):

1. **rustc SIGSEGV.** `RUSTFLAGS` applies `--no-as-needed -ltorch_cuda` to every
   rustc invocation, including host **proc-macro** `.so`s. Confirmed by `ldd`:
   `libserde_derive.so` had `libtorch_cuda`, `libc10_cuda`, `libcudart`,
   `libcusparse`, `libcufft` in its `DT_NEEDED`. rustc `dlopen`s it, drags the
   CUDA stack into the compiler, and dies — on *unrelated* crates (`serde`,
   `zerocopy`, `omega-core`). Fixed by exporting `CARGO_BUILD_TARGET` = host
   triple, so cargo builds host artifacts without the target rustflags.
2. **`torch.libs` is not universal.** 2.7.0+cu128 aarch64 ships `libgfortran.so.5`
   and `libarm_compute.so` **inside `torch/lib/`** with no sibling directory, so
   the script's `-L native=…/torch.libs` and rpath pointed at a path that does
   not exist, and its WARN fired every run. Now conditional, and the WARN checks
   for the thing that actually matters (libgfortran).

Remaining: correct the stale `torch.libs`/line-range claims in
`INSTALL_LIBTORCH.md` and `CUDA_TODO.md`, and note that `ci.sh` sources
`tch-env.sh` so it inherits the fix.

## 13 — PauliProp timing hygiene **(moves BEFORE 12)**

Revision 1 had 12 → 13. **Backwards**: optimising against single-sample,
no-warm-up numbers is precisely what 13 exists to prevent, and the "host merge
dominates" claim behind item 12 is asserted three times in the repo and
**measured zero times** — there is no pauliprop bench, no per-phase timer.

**Do first:** warm-up, best-of-3, and **per-phase timers** (SoA build / upload /
launch+sync / download / merge). Annotate or drop the 0.11× row, which is
one-time CUDA context creation + module load rather than the effect it is cited
for. Only then decide whether item 12 is worth anything.

## 12 — PauliProp buffer reuse, conditional on 13's numbers

Corrections to revision 1:

- **Wrong test cited.** `parity.rs:115-120` is inside the `#[ignore]`d timing
  test, which `ci.sh:388` never runs. The real 1e-9 GPU-vs-CPU dropped-mass gate
  is `parity.rs:196-201`, in `gpu_branch_matches_cpu_with_max_freq`. And all
  three parity tests **pass vacuously with no device** — not the safety net
  claimed.
- `gpu_branch_matches_cpu_exact` **cannot** catch a stale `odropped` tail: with
  no `max_freq`, `odropped` is all-zero, so the stale tail is too. Only the
  `max_freq` variant can.
- **14** `alloc_zeros`, not ~15 — each is `cuMemAlloc` **plus** a memset.
- **The oversized-download regression revision 1 missed:** `clone_dtoh` sizes
  the destination to the **device buffer** length, so under reuse all seven
  downloads copy the oversized buffer. Only `odropped_h`'s `.iter().sum()` is a
  correctness bug (the merge loop is index-bounded), but the bandwidth
  regression is real and could cancel the win. Use sliced `memcpy_dtoh` into
  reused host `Vec`s.
- `&mut Ctx` is safe at the call site (no re-entrancy, no nested `borrow_mut`),
  but inside `run`, `let s = &ctx.stream` and `launch_builder(&ctx.func)` are
  live across `.arg(&mut ox)` — direct field projections compile, an
  `&mut self` helper does not. **Any growth must happen strictly before the
  launch**: reallocating between the async launch and `synchronize()` frees
  memory an in-flight kernel is writing.
- Free win: `min_terms()` does a `std::env::var` **and a String parse per branch
  call**, on both the GPU and declined-to-CPU paths.

**Expectation, stated up front:** the recorded rows are 0.45×–0.79×. Removing 14
malloc+memset pairs per gate is O(100 µs); it is **unlikely to reach a
crossover**, and if 13's per-phase numbers say the merge dominates, this stops
and gets written down instead.

## Smaller — test-time backend construction

Revision 1 said 10 calls; it is **7** (`quad_perm_bit_identity.rs:299,310,321,
350,364,371,378`). The larger cost is `new_backend()`/`new_state()`
(`lib.rs:1777`), used at **23** sites in the crate's test module, which revision
1 did not touch. There are **25** `.cu`, not 26.

`OnceLock` behind both, with the caveat that sharing the backend shares its
`Mutex<Option<TrainStepGraph>>` cache — safe only under `RUST_TEST_THREADS=1`,
which `ci.sh:379` sets.

---

## Order (corrected)

**6a → 1 → 2 → 3 → 5 → 13 → 12 → Smaller → 9 → 4 (coverage only) → 8 (TODO) →
docs.** Item 11 is **done**.

6a first because 3 and 4 both depend on it. 1 before 2 because 1 is a
reframing with no code risk. 13 before 12. 4 last and reduced to coverage.

## Standing gaps this plan does not close, stated because a skip is not a pass

- **The QEC cross-check is MANDATORY (`ci.sh:815`) and cannot run on this box** —
  `pymatching` has no aarch64 wheel and fails to build. Every `./ci.sh` in this
  plan runs with that gate **dark**. Round 1 measured this; it does not improve
  by being unmentioned.
- Every perf number is GB10-only; nvcc 12.x and amd64 remain untested.
- Item 9 yields an arm64 curve, not the x86 one.
- Item 8 lands no code; item 6b is dropped; item 4 leaves the graph dense.
- Items 7 and 10 from the original todo list were not selected by the user and
  are untouched.

Per commit: `./ci.sh` (plus `ARIA_CUDA=1 ARIA_QISKIT_XCHECK=1` for CUDA work)
with the **exit code captured, never piped to `tail`**; commit; **never push**;
`git format-patch -1 -o fixes/patches/` continuing from **0040** (0039 exists);
README row.
