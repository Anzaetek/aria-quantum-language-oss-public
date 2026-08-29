<!-- SPDX-License-Identifier: Apache-2.0 -->
# PLAN — CUDA↔Metal `Decompose` bit-for-bit, measured (closes TODO §3.2)

Status: rev 2, 2026-08-20 — rev 1 survived an adversarial review with no
kills and five repairs, all folded in below (marked "rev 2"). Protocol agreed
with the DGX side (their session ack'd 2026-08-20 and added the two cautions
in §6 below). Unblocked by the project owner 2026-08-20; file exchange is scp initiated
from the Mac.

## 1. The claim under test

Both `MultiControlMode` doc comments — `omega-core/src/executor.rs` and
`GATE-EXACTNESS.md` §2.3 / §2.3.1 — say the CUDA and Metal statevector
backends agree **bit-for-bit** at the `Decompose` default, because both run
the identical 15-gate Nielsen-Chuang sequence. Never measured; asserted from
"the sequences are identical by inspection". Each side agreeing with CPU f64
to 1e-6 does NOT imply they agree with each other in the last bits.

Both outcomes are useful: agreement confirms the promise; disagreement means
the doc comments soften to "the same decomposition, each validated against
CPU f64" — and nothing else changes, because CPU f64 already gates
correctness on both sides.

## 2. Hazards this plan must own (found 2026-08-20, this box)

**H1 — silent CPU fallback, in TWO layers (rev 2: the review found the
second).** Layer one: the CLI's device arms fall back to the CPU backend on
any `Unsupported`/`Backend` error (`main.rs`, cuda/metal/opencl arms). Layer
two, which no error interception can see: `DeviceKind::resolve`
(`omega-core/src/device.rs`) resolves a requested GPU to CPU with only a
stderr notice whenever the feature is not compiled in — and `metal`/`cuda`
are OFF by default in `omega-cli` — and it also consults `OMEGA_DEVICE` when
`--device` is absent, so an env var on either box can silently redirect even
the CPU arbiter runs onto a GPU. A dump produced after either fallback would
compare CPU-to-CPU and "confirm" the claim vacuously. → Repairs: (1) the
dump flag hard-errors unless the arm that actually executed equals the arm
requested — gated on the dispatch decision (`use_*_sv` true and the fallback
branch not taken), NOT on catching errors; (2) the artifact's `backend`
field and hex width derive from the EXECUTED arm; (3) the CPU-artifact run
commands pin `--device cpu` explicitly so `OMEGA_DEVICE` cannot reroute
them. (Safety net if all of that fails: a full fallback makes the Exact and
Decompose dumps bit-identical, so H2's gate catches it — but attributes it
wrongly, which is why the primary repair is at dispatch.)

**H2 — the mode flag has history.** `6c990aa` fixed `--multi-control` being
accepted, validated, then silently dropped. `cd5c403` added the exact
CCX/CSwap kernel as an OPT-IN switch — `Decompose` remains the default on
both GPU backends (rev 2: rev 1 claimed the default had changed; the
commit's own message says "default unchanged"). → Each side dumps the corpus
under BOTH modes;
the differ requires Exact ≠ Decompose bit-wise on at least one amplitude per
box (the technique `multi_control_modes.rs::the_two_modes_are_not_the_same_
computation` already uses). A box whose two dumps are bit-identical proves
its mode switch did not take effect, and its artifact is invalid. Build rev
is stamped in every artifact (the 0118 `--version` stamp).

**H3 — host libm is inside the comparison if we let it in.** Rotation and
phase matrices are computed on the HOST with `cos`/`sin`
(`gates.rs` `ei()`, `ry()`); macOS libm and glibc do not promise identical
rounding, and even `t()` is `ei(PI/4)`, not a stored constant. A cross-machine
bit difference caused by libm would be indistinguishable from a kernel
difference. Three consequences:

* The **strict corpus** uses only gates whose matrices are compile-time-exact
  constants: `h, x, y, z, s, sdg, sx, cx, cz, ccx, cswap`. Same bits into
  both machines, no libm anywhere in corpus *preparation*.
* The backends' own `Decompose` internals may still compute e^{±iπ/4} factors
  host-side — that is part of the implementation under test, not a confound.
  If it makes them differ, the claim really is false as stated.
* One **probe circuit** (`p90_ry_probe`) carries `ry` angles, EXCLUDED from
  the headline verdict. It measures the libm question directly: if the two
  machines' CPU **f64** dumps differ on the probe, that difference is host
  libm, quantified and attributed, and documented separately.

**Arbiter ordering:** compare the two machines' CPU f64 dumps FIRST. On the
strict corpus they must be bit-identical (integer/dyadic arithmetic plus the
`FRAC_1_SQRT_2` constant only — any difference there is a build/corpus bug,
stop and investigate). Only then are the GPU dumps compared.

**H4 — a corpus where every multi-control fires cannot distinguish "the
decomposition matches" from "the permutation matches".** At least two
circuits keep controls unset so the gate is an identity whose 15-gate chain
still churns the untriggered amplitudes — §3.2 calls this the entire cost.

**H5 (rev 2) — signed zeros are EXPECTED to differ, from a cause outside
the 15-gate sequence, and they still falsify.** CUDA's `apply_cx` dispatches
by qubit index (dense 4×4 matrix kernel below `QUAD_KERNEL_MIN_QUBIT_2SLOT`,
a dedicated permutation above); Metal's is always the dense 4×4. A dense CX
computes `b + 0·a`, which rewrites `-0.0` to `+0.0`; a permutation moves
`-0.0` intact — and H's `−1/√2` column manufactures negative zeros inside
the Decompose chain. TODO §3.2 predicted exactly this. Ruling: a headline
diff confined to zero signs still exits 2 — bit-for-bit is bit-for-bit — and
the differ's report attributes it to dense-vs-permutation kernel dispatch so
the doc softening does not blame "rounding".

**H6 (rev 2) — the verdict is scoped to a compiler pair, not timeless.**
Metal shaders compile at RUNTIME from MSL source with default options
(fast-math on), so FMA contraction belongs to whatever Metal compiler the OS
ships; a macOS update can flip a "confirmed" verdict with zero repo changes.
CUDA's `--fmad=true` default is the mirror. The artifact's `host` object
records the exact OS build, and whichever doc update results says "measured
on <os>/<rev>", never an unconditional promise.

## 3. Corpus — `tools/biteq/corpus/*.qasm`, committed

OpenQASM 2 (`qelib1.inc` names; the parser maps `ccx`/`cswap` at
`lower.rs:1454`). 3–6 qubits. Every file deterministic, no measurement, no
classical control. ~10 files:

| file | shape | why |
|---|---|---|
| `c01_ccx_basis_fire` | X on both controls, then `ccx` | permutation on a basis state, triggered |
| `c02_ccx_basis_nofire` | X on ONE control, then `ccx` | **untriggered** chain on a basis state (H4) |
| `c03_ccx_super` | h+s+sx layer on all, `ccx`, h layer | amplitude everywhere, triggered and untriggered mixed |
| `c04_ccx_super_nofire` | superpose only non-controls, controls left \|0⟩, `ccx` | untriggered with superposition around it (H4) |
| `c05_cswap_super` | as c03 with `cswap` | the other gate |
| `c06_ccx_repeat` | 4× `ccx` on rotating triples, interleaved cx/s | error accumulation over repeats |
| `c07_cswap_ccx_mix_6q` | 6q, both gates, sx/cz interleave | widest, both gates in one state |
| `c08_ccx_adjacent_vs_strided` | `ccx q0,q1,q2` then `ccx q0,q2,q5` (6q) | stride/slot-layout differences |
| `c09_cz_only_control` | no multi-control at all | negative control: two backends on plain Cliffords |
| `p90_ry_probe` | c03 shape + `ry(0.13)`/`ry(0.7)` | libm probe, excluded from verdict (H3) |

## 4. Artifact and differ

**Format: JSON with hex bit patterns** — TWO deliberate deltas from TODO
§3.2, both to be re-ack'd by the DGX side (rev 2 names the second). Delta 1:
hex `to_bits()` strings instead of raw little-endian binary — they satisfy
the actual requirement (bit patterns, never decimal), are endianness-proof
by construction, self-describing, and adjudicable by someone with neither
device; §3.2's real worry was decimal round-trips, not JSON. Delta 2: §3.2
wanted the CPU f64 reference "in the same file"; here it is a sibling
artifact in the same directory, and `diff.py` refuses to render a verdict
without it — same adjudicability, no format contortion.

One JSON file per (machine, device, mode):

```json
{ "format": "biteq-v1",
  "build": {"tool": "omega-run", "version": "…", "rev": "…"},
  "host": {"os": "…", "arch": "…"},
  "backend": "metal-f32 | cuda-f32 | cpu-f64",
  "multi_control": "decompose | exact",
  "ordering": "amp[i]: qubit 0 is the LOW bit of i",
  "circuits": [ {"file": "c01_…", "n": 3,
                 "amps": [["3f800000","00000000"], …] } ] }
```

`amps[i]` = `[re, im]` as lowercase hex of `f32::to_bits()` (8 digits) or
`f64::to_bits()` (16 digits) — width says which, and `-0.0` vs `+0.0` is
visible (`80000000` vs `00000000`).

**Differ: `tools/biteq/diff.py`** (stdlib-only, like `tools/omega_client`).
Takes artifact files, verifies in this order: (1) CPU-f64 cross-machine
bit-identity on the strict corpus; (2) per-box Exact ≠ Decompose (H2) — the
gate is PER BOX over the whole corpus, deliberately not per circuit: Metal's
own test proves Exact ≡ Decompose bitwise on every BASIS state, so the
basis-state circuits (c01, c02) and the no-multi-control circuit (c09)
legitimately match across modes, and a per-circuit gate would false-alarm on
them; (3) the headline CUDA-Decompose vs Metal-Decompose comparison — per
circuit, count differing amplitudes, max ULP distance, signed-zero cases
listed separately (H5: signed-zero-only still exits 2, attributed to kernel
dispatch); (4) rev 2, free extra gate: CUDA-Exact vs Metal-Exact — both are
pure permutations, so disagreement THERE localises a headline diff to the
non-CCX gates rather than the decomposition. Probe file reported under its
own heading. Exit 0 only if every gate passes AND the headline comparison is
bit-identical; a clean "claim is false" run exits 2 with the table (distinct
from exit 1 = harness/input error).

## 5. Implementation on this box

1. `--dump-state-bits FILE` in `omega-cli`: requires `--statevector`
   (analytic), refuses `--shots`/`--noise`, hard-errors on GPU fallback (H1).
   Serialises the final `ExecResult::Statevector` amplitudes: f32 bits when
   the executing backend computes in f32 (metal/cuda/opencl), f64 bits on
   CPU. Amplitudes come back from the GPU already promoted; f32→f64 is exact,
   so `(amp as f32).to_bits()` on the CLI side is faithful — noted in code.
2. Corpus files + `tools/biteq/README.md` (protocol, both cautions from the
   DGX side, the run commands for each box).
3. `diff.py` + a fixture-level self-test (two tiny hand-written artifacts:
   one equal pair, one differing in a signed zero — the differ must flag it).
4. CLI tests: dump+shots refused; dump on cpu backend produces a parseable
   artifact with 16-digit hex; mutation-check by breaking the hex width and
   watching the test fail. (Rev 2: a "dump+noise refused" test is VACUOUS —
   `--noise` with `--statevector` is already refused upstream of the dump
   flag, so such a test passes with no dump-side check at all. The dump flag
   cites the existing guard instead of duplicating it.)

Run here: metal decompose, metal exact, cpu f64 → 3 artifacts, with the CPU
run pinned `--device cpu` (H1 repair 3 — `OMEGA_DEVICE` must not be able to
reroute the arbiter). DGX runs the same commands with `--device cuda`
(serialised, per their E7 finding; ~50 s scale, far inside the 35 GiB
neighbour ceiling they flagged). Mac pulls the CUDA artifacts by scp,
`diff.py` renders the verdict, docs updated either way — scoped per H6 to
the measured OS/compiler pair.

## 6. What the DGX side flagged, kept on purpose

* Their `multi_control_modes.rs` tests already prove same-device correctness
  vs CPU and same-device mode divergence; this plan reuses the divergence
  technique (H2) and the 3–6q/CCX+CSwap corpus shape, and adds what those
  tests cannot: cross-device bit comparison through a transferred artifact.
* On any rev before `6c990aa` the mode flag silently no-ops — both boxes are
  past it (they hold 0148 + the two patches sent 2026-08-20), and the rev
  stamp in the artifact makes this checkable after the fact.
