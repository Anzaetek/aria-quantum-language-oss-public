<!-- SPDX-License-Identifier: Apache-2.0 -->
# Gate exactness ledger

**Standing rule: every gate that is applied approximately rather than exactly
must appear here, with its bound and the reason.** A gate whose implementation
introduces error and says so nowhere is the defect this file exists to prevent —
the error does not disappear because nobody wrote it down, it just stops being
attributable when a cross-check drifts.

"Exact" here means **bit-identical to the reference implementation named in the
row**, not mathematically exact in the abstract. A gate can be bit-identical to
an f32 reference and still be 1e-7 away from the f64 truth; both facts belong in
the table, in different columns.

Scope note, stated so this is not over-read: this ledger was written on
2026-08-17 and covers the **CUDA statevector** backend in full plus the
truncating backends' headline approximations. Metal, OpenCL, photonics and CV
are **not yet audited** — their rows say so rather than being left blank, which
would read as "exact".

---

## 1. The approximation everyone inherits: f32

The CUDA, Metal and OpenCL statevector backends compute in **f32** by default.
Against the f64 CPU statevector this caps agreement at **~5e-7** regardless of
how good any individual gate is, and that is six orders looser than the 1e-9
this project gates its cross-checks at.

This is why `f64_path.rs` exists (`≤1e-13` vs Qiskit, measured **1.388e-16** on
sm_121) and why the f32 GPU cannot enter its own verification gates unaided.

**Everything in §2 is a deviation measured against the f32 dense path**, i.e. on
top of this. Do not add the two and call it a total; they are different
references.

## 2. CUDA statevector — per gate

Reference for every row: the generic `apply_2q` / `apply_1q` dense matvec, in
f32, which is what each gate dispatched to before it was specialised.

| gate | implementation | vs the dense path | notes |
|---|---|---|---|
| `CX`, `SWAP` | permutation kernel | **exact** (bit-identical) | moves amplitudes; computes nothing |
| `CZ` | single-slot phase | **exact** | `x * -1.0` is exact in IEEE-754 |
| `CY` | phased permutation | **exact** | phases are `∓i`, so every product is `x·0` or `x·(±1)` |
| **`CRz`** | **two-slot phase** | **APPROXIMATE — ≤ `4·f32::EPSILON·\|amplitude\|`** | **see §2.1** |
| `Rz`, `U1`, `Z`, `S`, `T` (+daggers) | fused diagonal chain | **exact** | fusion saves launches, not multiplications — see §2.2 |
| `Sx`, `Sxdg` | exact 2×2 matrices | **exact** | deliberately **not** routed through `U3` — `sx = e^{iπ/4}·U3(...)`, and that global phase gives `\|sx − U3\| = 0.541`, invisible in counts but wrong in any amplitude comparison |
| **`CCX`** | 15-gate decomposition (default) **or exact octet permutation** | **APPROXIMATE by default**; `MultiControlMode::Exact` is exact | **see §2.3** |
| **`CSwap`** | `CX · CCX · CX` (default) **or exact octet permutation** | same | §2.3 |
| `CU3` | dense `apply_2q` | exact (is the reference) | |

> **"Exact" above means exact for every NON-ZERO amplitude. It does not mean
> bit-identical.** Added 2026-08-27 after the CPU statevector shipped the
> stronger claim three times and had to retract it.
>
> Every specialised row wins by NOT WRITING part of the state — a permutation
> leaves its fixed points alone, a single-slot phase leaves three slots alone.
> The dense reference writes all four: `1*a00 + 0*a01 + 0*a10 + 0*a11`. And
> `0.0 * a` is `-0.0` when `a` is negative, so the dense result carries sign
> information from amplitudes the specialised kernel never reads. A `-0.0`
> therefore SURVIVES on the fast path where the dense path canonicalises it to
> `+0.0`.
>
> This affects `CX`/`SWAP`, `CZ` and `CY` here — every row whose mechanism is
> "moves amplitudes" or "touches one slot". It is unobservable in anything
> these rows are used for (`-0.0 == +0.0`, identical `|z|^2`, identical
> sampling) and IS observable to `--dump-state-bits`, which is why
> `tools/biteq/diff.py` classifies signed-zero-only differences as dispatch
> artefacts rather than divergence.
>
> The note "moves amplitudes; computes nothing" is true and was the reasoning
> that produced the wrong claim: the difference never came from arithmetic in
> the region a kernel touches, but from the region it does not write.
> `kernels/apply_quad_perm.cu` §SIGNED ZERO has stated this for the CUDA
> permutation kernels all along; this ledger did not, and the CPU side was
> written from the ledger. Measured on the CPU equivalents by enumerating sign
> combinations — diagonal differs in 7 of 16, controlled-U in 3 of 16 — and
> pinned by `every_2q_gate_differs_from_the_dense_scan_only_in_the_sign_of_zero`
> in `omega-backend-statevector`.
>
> **Not re-verified on CUDA or Metal hardware**, which this box does not have.
> The mechanism is IEEE-754 and the `.cu` already documents it, but the row-level
> claims here are inferred for those backends rather than measured.
| `H`, `X`, `Y`, `Rx`, `Ry`, `U3` | generic `apply_1q` | exact (is the reference) | |

### 2.1 CRz — FMA contraction

Both the specialised kernel and the dense path call the *same* `cmul`, but the
dense path wraps it in a four-term accumulation. NVRTC contracts `p*q + r*s`
into `fma(p, q, r*s)` by default, and **which product keeps its intermediate
rounding depends on the surrounding expression**. For a complex phase the
imaginary component therefore differs in the last bits.

Bound: **`4 · f32::EPSILON · |input amplitude|`** — scaled to the operand, not
to the result, because CRz's two products can nearly cancel and a
ulp-of-result bound inflates for reasons that have nothing to do with the
kernel. Enforced by `tests/quad_perm_bit_identity.rs::crz_two_slot_phase_...`.

This was found the hard way: the gate asserted bit-identity and **passed**,
until an unrelated `#pragma unroll` changed the schedule and one amplitude moved
by exactly one f32 ulp. It had been passing by luck of one toolkit and one arch.
CX/SWAP/CZ/CY are genuinely exact and stay on the strict bar; CRz is not and
must not be put back on it.

### 2.2 Diagonal fusion — the same rounding count, in a different order

Consecutive diagonal gates are folded into **one kernel launch**, not into one
multiply. `apply_diagonal_product` converts each factor to f32 on the host and
passes the *list*; the kernel then loops `amp = amp * d_k` over the factors
(`apply_diagonal_product.cu:30-33`).

So the number of f32 multiplications per amplitude is unchanged — the saving is
launches and memory traffic, not arithmetic. **Each gate is individually exact
in the same sense as applying it alone, and a fused run is bit-identical to the
unfused one**, because the operations and their order are the same.

Recorded explicitly because the plausible-sounding alternative — pre-multiplying
the phases on the host in f64 and applying one diagonal — would have *fewer*
roundings and therefore would **not** be bit-identical. It is not what the code
does, and someone optimising this later should know that changing it changes the
numbers.

### 2.3 CCX / CSwap — the largest per-gate approximation

`CCX` is not applied as a permutation. It is decomposed into **15 gates**
(`H`, 6×`CX`, 4×`T`, 3×`Tdg`, `H`), each of which rounds in f32. `T` and `Tdg`
carry `e^{±iπ/4}`, which is irrational and not exactly representable, so the
error is not merely accumulation of exact operations.

`CSwap` = `CX · CCX · CX`, so it inherits all of it.

**LANDED 2026-08-17 as `MultiControlMode`, opt-in.** `CCX` is a permutation:
it swaps two amplitudes of an octet (`CSwap` two others). The exact subspace
kernel is one pass instead of fifteen:

```text
26 qubits, 20 gates:   ccx_decompose  1.2775 s
                       ccx_exact     41.68 ms      30.6x
accuracy vs the CPU f64 statevector (32 circuits, CCX + CSwap):
                       exact          4.150e-7
                       decompose      5.419e-7     1.6x more error
```

So it is both faster and more accurate. It stays **opt-in** anyway
(`--multi-control exact`, default `decompose`) because it CHANGES THE NUMBERS —
14 gates of f32 rounding disappear — and because the decomposition is
deliberately identical to Metal's, so cross-GPU agreement requires the two
devices to be at the *same* setting.

Gated against **Qiskit**, not against the decomposition: `ccx`/`cswap` are now
in `omega-xcheck`'s corpus (CPU ↔ Qiskit at 4.441e-16), and
`tests/multi_control_modes.rs` compares both CUDA modes against the CPU f64
statevector, which walks the CCX subspace directly. The chain is CUDA → CPU →
Qiskit with no link sharing an implementation. A test that the two modes are
*not* bit-identical guards against the switch silently becoming a no-op.

**NOT exposed in the job interface, and cannot be yet.** `omega-server` has no
CUDA dependency or feature — `exec_statevector` reaches CPU and OpenCL only — so
a `multi_control` field on `QuantumExecuteReq` would be inert, advertising a
capability no code path can honour. Prerequisite: give the server a CUDA
dispatch path. Until then the switch is CLI-only.

### 2.3.1 Metal — the same switch, same shape

**Done.** Metal carries the same `MultiControlMode`, reached the same way
(`MetalStatevectorBackend::with_multi_control`, `--multi-control` on the CLI),
backed by `shaders/apply_octet_swap.metal` — one pass over `dim/8` octets that
exchanges two slots.

Measured the same way as CUDA above, against the CPU f64 statevector over the
same 32-circuit corpus (`crates/omega-backend-statevector-metal/tests/multi_control_modes.rs`):

```
accuracy vs the CPU f64 statevector (32 circuits, CCX + CSwap):
                       exact          2.540e-7
                       decompose      4.150e-7     1.6x more error
```

Same 1.6× ratio CUDA saw, from an independent implementation on different
hardware — which is the expected shape, since the ratio comes from the 15-gate
chain's rounding and not from the device.

The unit side asserts something the probability comparison cannot: on a dense
input, `Exact` is **bit-identical** to the permutation applied to the f32
round-trip of that input (deviation exactly `0.0`), while the decomposed path
sits at `8.0e-8`. That is what pins the mode as reaching dispatch at all — an
earlier draft used a `1e-6` tolerance there and a mutant routing `Exact` back to
`apply_ccx` passed it.

**Measured 2026-08-20, both devices, and the claim was FALSE:** CUDA and
Metal `Decompose` are NOT bit-for-bit equal to each other. The two-machine
protocol (`tools/biteq/`, verdict tables in `tools/biteq/RESULTS.md`; rev
b9e06b9 on both sides, Darwin 25.5.0 Metal vs Linux 6.17 CUDA) found every
superposition circuit differing in its last bits — including a circuit with
**no multi-control gate at all** (16 amplitude parts), while Exact-vs-Exact
was bit-identical on basis-state circuits. So the permutation kernels agree
and the shared 15-gate sequence was never the issue: the elementary gate
kernels' f32 contraction differs between the two compilers (Metal runtime
fast-math vs CUDA `fmad`), and no identical gate ordering can cancel that.
Both hosts' CPU f64 references were bit-identical on the whole strict corpus,
which is what pins the attribution to the device kernels rather than host
math. Each GPU remains within its 1e-6 gate against CPU f64 — nothing here is
a correctness defect; the promise was simply stronger than the hardware
grants, and it is now scoped to what was measured.

<details><summary>Original TODO text</summary>

**TODO — exact CCX/CSwap kernel, as a SELECTABLE option.** `CCX` is a
permutation: it swaps two amplitudes of an octet. A dedicated subspace kernel is
both *faster* (one pass instead of 15) and *exact*. The CPU backend already
walks the CCX subspace directly (`2ed1eea`); CUDA and Metal do not.

It must land as an **opt switch, not a silent swap**, exposed in both the CLI
and the job interface:

- the exact kernel changes the numbers — it removes 14 gates' worth of f32
  rounding, so a caller comparing against a previously recorded run will see a
  difference and must be able to ask for the old behaviour;
- the decomposition is deliberately **identical to Metal's** so the two GPUs
  agree bit-for-bit; making CUDA exact breaks that cross-GPU agreement until
  Metal follows, and a run comparing the two needs to be able to keep them
  matched;
- "faster and exact" is the expected outcome, not a measured one. It must be
  measured against the decomposition on the same box before the default moves,
  and the switch is what makes that comparison possible at all.

Default stays on the decomposition until the exact path is measured **and**
gated against Qiskit — not against the decomposition it replaces, which shares
its conventions.

</details>

The decomposition is deliberately **identical to Metal's** — the same gate
sequence, so any systematic error shape is shared. What that does NOT buy,
measured 2026-08-20, is bit-for-bit agreement: identical sequences of f32
operations compile to different contraction patterns on the two platforms
(see the correction above), so cross-GPU agreement gates must compare within
tolerance against CPU f64, never bit-wise across devices. With §2.3.1 landed,
same-mode comparison remains necessary — mixing `Exact` with `Decompose`
adds a *further*, larger difference on top of the platform one.

## 3. Truncating backends — approximation is the method

These do not approximate a gate; they approximate the *state*, by construction.
Listed because a reader asking "where does error come from" needs them in the
same place.

| backend | what is truncated | reported as | tested |
|---|---|---|---|
| `mps` | Schmidt values beyond the bond dimension | `discarded_weight`, `fidelity_estimate` — **labelled an estimate** | vs Aer: worst TVD 0.0327 |
| `pauliprop` | Pauli terms below a coefficient threshold / above `max_freq` | `dropped_mass` — **a bound, not an estimate** | `dropped_mass_is_a_bound.rs`; GPU vs CPU at 1e-9 |

`stabilizer` is exact within its gate set (Clifford only) and refuses outside
it. `statevector` (CPU, f64) is the reference everything else is measured
against.

## 4. Not yet audited

Rows deliberately present rather than omitted, because an absent row reads as
"exact":

- **Metal** — shares CCX/CSwap's 15-gate decomposition (identical sequence, by
  design) and is f32, so §1 and §2.3 apply; §2.3.1 records the exact switch and
  its measurement. Its 2q gates have **not** been through the specialisation
  work CUDA had, so they still use the dense path; no per-gate exactness audit
  has been done.
- **OpenCL** — f32; not audited.
- **photonics**, **CV** — not audited. `STATUS.md` §15 notes these have been
  compared on agreement but never on speed; exactness is a separate axis and
  also open.
- **CUDA-graph path** — builds dense matrices for CX/CZ/SWAP/CRz/CY rather than
  using the specialised kernels, so the training path's rounding differs from
  the forward path's. Within f32 tolerance, but not bit-identical to it.

## 5. How to add a row

When a gate stops being exact — a decomposition, a fused pass, a
tolerance-gated criterion, a precision change — add it here **in the same commit
as the change**, with:

1. the bound, as a formula and not an adjective;
2. what the bound is measured *against* (which reference, which precision);
3. the test that enforces it, by name;
4. why the approximation is taken rather than the exact route.

If the answer to (4) is "it was faster", say so and say by how much — an
unmeasured approximation is not a trade, it is a guess.
