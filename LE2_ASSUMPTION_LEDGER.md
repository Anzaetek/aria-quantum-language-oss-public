# LE2 — emulator/machine assumption ledger (the trust boundary)

> `LEAN_ERROR_PLAN.md` step **LE2**. The honest framing of the QLSS
> end-to-end claim: we prove the **classical** kernels (LE1), then **assume**
> the omega emulator faithfully realizes ideal gate semantics, and pin that
> assumption with a **numeric conformance artifact** (`aria-verify`). This
> document is the ledger; `aria-verify all` is the checked artifact.

This is *not* a hardware proof. "End-to-end" means **modulo the assumptions
listed here**, each of which is either (a) discharged by a proof elsewhere, or
(b) conformance-checked numerically by `aria-verify`, or (c) explicitly carried
as an unproven assumption with its rationale.

---

## A. What is assumed about the omega emulator (the trust boundary)

| # | Assumption | Status |
|---|---|---|
| A1 | The statevector backend implements **exact ideal unitary semantics** over the declared gate set (H, X, Y, Z, S, T, RZ, RY, CX, CZ, CP, SWAP). | **conformance-checked** — `aria-verify` cross-checks omega output against an independent pure-Rust classical oracle per app (below). |
| A2 | Floating-point arithmetic is IEEE-754 binary64; results are reproducible up to a **stated tolerance**, not bit-identical across backends (shot sampling + summation order). | **assumed**, bounded — each app declares its `tol`; the verdict is `Δmax ≤ tol`. |
| A3 | No mid-circuit-measurement gaps for the circuits used — measurement is terminal (or, for Hadamard-test/ancilla apps, the ancilla protocol is the declared one). | **assumed**, scoped to the shipped circuits (all terminal-measure or declared-ancilla). |
| A4 | Sampling is from the Born-rule distribution of the ideal state; shot noise is `O(1/√N)` with the declared shot count. | **conformance-checked** — sampled estimators land within `4σ_MC` of the closed-form law (see the `noise` app anchors). |
| A5 | The Rust **circuit-builder** lowering (`CircuitBuilder` / `.aria` → omega IR) preserves gate order and operands. | **conformance-checked** — the same builder feeds both omega and the oracle; a lowering bug shows as an oracle mismatch. |

> A1/A5 together are the LE2 "L1 differential conformance" leg: *the Rust
> circuit-builder, executed on omega, matches an independent classical
> computation of the same algorithm.* A divergence is a real bug, not a
> tolerance artifact.

---

## B. The checked artifact — `aria-verify all`

`crates/aria-verify` runs every shipped `.aria` application through the omega
runtime (WASM guest, with a native fallback) and compares the quantum result to
a **pure-Rust classical oracle**, emitting `Δmax` vs a stated `tol` and a
`PASS/FAIL` verdict. CI asserts `14/14 passed`.

Latest run (`cargo run -p aria-verify -- all`):

```
qsvd PASS   qft PASS   vqe_ansatz PASS   grover3 PASS
bernstein_vazirani PASS   deutsch_jozsa PASS   swap_test PASS
teleport PASS   qaoa_maxcut PASS   qml_classifier PASS
qos PASS   circulant PASS   cqs PASS   noise PASS
14/14 passed
```

The three **QLSS reproductions named in the LE2 gate** all conform:

| app | quantum (omega) vs classical oracle | verdict |
|---|---|---|
| `circulant` | circulant solve via QFT diagonalization vs `solve_via_dft` | PASS |
| `qos` | scaling-exponent / fidelity vs least-squares oracle | PASS |
| `cqs` | overlap `Re⟨Z⟩` (Hadamard test, 8192 shots) vs `cos(π/3)` via `apply_pauli`: `Δmax = 4.9e-3` (tol `5.0e-2`) | PASS |

---

## C. Bonus — the `noise` app pre-stages LE3

The `noise` application already emits the **emulator-vs-analytic** agreement
that LE3's Kraus side needs (the closed-form channel laws cross-checked against
omega's trajectory noise model), with the Monte-Carlo error bar:

```
anchors (σ_MC = 0.0056, tol 4σ = 0.0224):
  (A) depolarizing  ⟨Z⟩ = (1 − 4p/3)·cosθ          worst|Δ| = 0.0091
  (B) amp damping   ⟨Z⟩ = ⟨Z⟩₀ + 2γ·sin²(θ/2)       worst|Δ| = 0.0011
  analytic ±0.05 depolarizing crossing (θ = π/3):  p* = 0.0750
reproductions under depolarizing noise p = [0, .01, .02, .05, .1, .2]:
  cqs Re⟨Z⟩        +0.500 +0.471 +0.448 +0.379 +0.292 +0.151
  circulant P(|1⟩) +1.000 +0.960 +0.924 +0.821 +0.673 +0.454
  qos fidelity     +1.000 +0.962 +0.926 +0.823 +0.680 +0.454
```

So LE3's **emulator ↔ analytic-law** leg (worst |Δ| ≤ 0.0091 ≤ 4σ) is already a
showable robustness certificate. What remains for LE3 is the **PRISM/GSPN
steady-state** cross-check and the **Lean/Kraus (D2)** fidelity proof — both
require infrastructure not yet built (a quantum IR + Kraus semantics in
leanlift; complex-matrix density-operator algebra in the Lean tree).

---

## D. What is NOT covered (carried honestly)

- **No hardware proof.** A1–A5 are about the *emulator*, not a QPU
  (`GPU_PLAN.md` Phase 8, hardware-gated).
- **LE3 Lean/Kraus (D2)** and **LE4 capstone** are gated on the quantum-algebra
  Lean library (complex matrices, density operators, fidelity/trace-distance)
  that does not exist in the proof tree yet.
- Tolerances in §B are per-app and **not** `1e-9` bit-exact — they bound shot
  noise + summation reordering (A2/A4), the honest tolerance for a sampled
  quantum estimator.
