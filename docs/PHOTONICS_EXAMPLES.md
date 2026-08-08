<!-- SPDX-License-Identifier: Apache-2.0 -->
# Photonic examples — run them with Aria, and with the reference

Every example here is runnable two ways: through Aria, and through the
independent implementation it is checked against. That pairing is the point.
Two Aria backends agreeing may only mean they share a convention; this project
has already shipped defects that every *internal* agreement gate missed, and one
of them (`bs_rx` mapped to the wrong Perceval component) was found while writing
this file.

**DV and CV are genuinely different programs, not two dialects of one.** DV here
is a mesh of modes — interferometers, permanents, `C(m+p−1, p)` output
configurations. CV is a single mode's Fock ladder under displacement, squeezing
and Kerr. They share the word "photonic" and almost nothing else, so they are
documented separately rather than forced into a parallel table.

| modality | Aria surface | reference | status |
|---|---|---|---|
| **DV** (discrete, mode mesh) | `OPTICQASM` + `omega-run --backend photonics` | **Perceval** (Quandela) | runnable end to end |
| **CV** (continuous, Fock ladder) | Rust API only — no language surface | **piquasso** | numerics only, see §3 |

---

## 1. DV — OPTICQASM examples

Install the reference once:

```console
$ cd crates/omega-bridges/python && make perceval-venv
```

Each example below gives the Aria command and the matching Perceval invocation.
The bridge takes JSON on stdin, so the recipe is the same shape every time:

```console
$ python3 -c "
import json,sys
print(json.dumps({'qasm':open(sys.argv[1]).read(),'shots':200000,'input_fock':[1,1,0]}))
" examples/circuits/boson_sampling.opticqasm \
  | crates/omega-bridges/python/omega-bridge-perceval-runner
```

### 1.1 Hong–Ou–Mandel dip — `hom_dip.opticqasm`

Two indistinguishable photons, one balanced beam splitter, coincidences vanish
exactly.

```console
$ omega-run examples/circuits/hom_dip.opticqasm --backend photonics --input 1,1
```

Expected: `|2,0>` 0.5, `|0,2>` 0.5, `|1,1>` **exactly 0** — self-verifying, since
the dip is an exact zero rather than a small number.

**What it cannot check.** The dip is a statement about `|amplitude|²`. A beam
splitter with the right magnitudes and the wrong phases reproduces it perfectly
— which is exactly how the `bs_rx` bridge defect survived for so long. Do not
treat a passing HOM test as evidence about conventions.

### 1.2 Mach–Zehnder interferometer — `mzi.opticqasm`

One photon, two balanced splitters, a phase in one arm. Deterministic in φ.

```console
$ omega-run examples/circuits/mzi.opticqasm --backend photonics --input 1,0
```

At φ = π/3: `|0,1>` = cos²(φ/2) = 0.75, `|1,0>` = 0.25.

### 1.3 Boson sampling — `boson_sampling.opticqasm`

3 modes, 2 photons, an **asymmetric** mesh with non-zero transverse phases. This
is the first example where the permanent genuinely carries the physics: HOM and
MZI are both solvable by hand, so an implementation can get the general
permanent badly wrong and still pass both.

```console
$ omega-run examples/circuits/boson_sampling.opticqasm \
    --backend photonics --input 1,1,0 --shots 200000
```

Measured agreement, aria vs Perceval at 200k shots each: **TVD 0.00108**.

| output | aria | Perceval |
|---|---|---|
| `2,0,0` | 0.42968 | 0.42931 |
| `0,1,1` | 0.26734 | 0.26802 |
| `0,0,2` | 0.16460 | 0.16390 |
| `0,2,0` | 0.06996 | 0.07025 |
| `1,1,0` | 0.04963 | 0.04976 |
| `1,0,1` | 0.01878 | 0.01877 |

Photon number is conserved, so the six probabilities must sum to 1 — a cheap
invariant that catches a mis-indexed submatrix immediately.

### 1.4 Grover, 4 elements, one query — `grover_polarization.opticqasm`

> P. G. Kwiat, J. R. Mitchell, P. D. D. Schwindt and A. G. White,
> *"Grover's search algorithm: an optical approach"*,
> **J. Mod. Opt. 47(2–3), 257–266 (2000)**, doi:10.1080/09500340008244040.

Quandela ship this as a Perceval notebook (`2-mode_Grover_algorithm`), so an
independent implementation already exists.

```console
$ omega-run examples/circuits/grover_polarization.opticqasm \
    --backend photonics --input 1,0,0,0 --shots 20000
```

Both stacks return `|0,0,1,0>` with probability 1 — the marked element `|10>`,
i.e. optical mode 2 = (spatial 1, H).

**Be careful what you conclude from that.** The output is deterministic, so
several *wrong* circuits also concentrate on one mode. The agreement shows the
two stacks compose the same way; it is not a sharp test of conventions. The
sharp tests are the matrix-level ones in §2.

It is also not a speedup: 4 modes encode 4 items, so the resource cost is
linear. Kwiat et al. say so plainly. The value is the interference, which is
what makes it a good correctness fixture.

---

## 2. The conventions, and why they are pinned in tests

Aria **matches Perceval verbatim**, including where Perceval departs from the
usual textbook statement. A third convention would put a fudge factor in every
cross-check, and a fudge factor is indistinguishable from a bug once the reason
is forgotten.

| element | Aria = Perceval | how that differs from the textbook |
|---|---|---|
| `hwp(θ)` | `i · [[cos2θ, sin2θ],[sin2θ, −cos2θ]]` | carries a global `i`; `det = +1`, not `−1` |
| `pbs` | **swaps H**, transmits V | usually stated as "transmits H, reflects V" |
| `bs_rx(θ,φ)` | `[[cosθ, −e^{iφ}sinθ],[e^{−iφ}sinθ, cosθ]]` = Perceval `BS.Ry` | the name says Rx; **the matrix is Ry** |

**The global `i` is not ignorable.** A wave plate acts on a *subset* of the
interferometer's modes, so a global factor on that 2×2 block is a **relative**
phase between interfering paths. Measured: dropping it moves 4-mode
single-photon output probabilities by up to **0.413**.

This was got wrong once already, in the plan for this very feature: the
decomposition was verified to 1e-16 against the **i-less** matrix while the same
document argued the `i` mattered. Determinants (`−1` vs `+1`) make the
contradiction unmissable in hindsight. Hence:

```console
# Aria's lowered unitary vs Perceval's matrices
$ cargo test -p omega-backend-photonics --test polarization_conventions

# Perceval's own matrices, pinned against drift
$ crates/omega-bridges/python/.venv-perceval/bin/python \
    crates/omega-bridges/python/tests/test_perceval_conventions.py
```

Both suites include tests that **guard the guard** — the i-less matrix must be
*rejected*, and the old `bs_rx` mapping must still *fail*. Without those, a
revert could leave the suite green.

### Polarization addressing

`photon q[N] pol;` declares **N spatial** modes = **2N optical** modes, ordered
`[q0_H, q0_V, q1_H, q1_V, ...]`. So `--input 1,0,0,0` is one horizontally
polarized photon in spatial mode 0. `hwp` and `pbs` take **spatial** mode
references; `ps` and `bs_rx` take optical ones.

Applying `hwp` or `pbs` to a register declared without `pol` is an **error**, not
a guess — silently treating `q[0]` as spatial would apply the plate to two
unrelated optical modes and return a plausible wrong answer.

---

## 3. CV — a different program, and a smaller surface

```console
$ cargo run -p omega-backend-cv --example cv_states
```

**CV has no language surface.** There is no `.cvqasm`, and `omega-backend-cv` is
a workspace member that nothing else depends on, so the Rust API is the entire
user interface today. That is a real gap, tracked as C3 — better stated here
than discovered after writing a file nothing parses.

The piquasso recipe for any state Aria can build:

```python
import piquasso as pq
with pq.Program() as p:
    pq.Q(0) | pq.Vacuum()
    pq.Q(0) | pq.Displacement(r=1.0, phi=0.0)   # Aria: FockState::coherent(1.0+0i)
    pq.Q(0) | pq.Kerr(xi=0.37)                  # Aria: state.kerr(0.37)
state = pq.PureFockSimulator(d=1, config=pq.Config(cutoff=20)).execute(p).state
print(state.fock_probabilities, state.state_vector)
```

**Convention note:** Aria's displacement is **Cartesian** `(re, im)` per K15,
piquasso's is **polar** `(r, φ)`. The conversion happens in exactly one place,
`tools/cv_cross_check/piquasso_ref.py`, so there is one line to audit rather
than a scattered assumption.

Cross-check:

```console
$ cargo test -p omega-backend-cv          # vs the committed piquasso fixture
$ ARIA_CV_XCHECK=1 ./ci.sh                # reruns piquasso live, checks for drift
```

### A CV variational circuit

```console
$ cargo run -p omega-backend-cv --example cv_variational
```

Alternating diagonal cost (`kerr` then `phase_shift`) and non-diagonal mixer
(`displace`) — the QAOA *shape*. Prepares `|2⟩` from vacuum:

| | `P(|2⟩)` |
|---|---|
| single displacement (baseline) | 0.270661 |
| analytic Poissonian optimum | 0.270671 |
| **3-layer ansatz** | **0.536636** |

The baseline is there so the result is self-checking: a coherent state's Fock
weights are Poissonian and cap `P(2)` at `2e⁻² ≈ 0.2707`, so roughly doubling it
had to come from interference between layers rather than from displacement
alone.

**It is NOT the CV QAOA of Verdon et al.** (arXiv:1902.00409), and calling it
that would overstate it. That algorithm evolves under `exp(-iγf(x̂))` for an
arbitrary objective `f` of the position quadrature; its photonic realisation is
Enomoto et al., **Phys. Rev. Research 5, 043005 (2023)** (arXiv:2206.07214).
This crate has no position-basis operator, so it offers a *quadratic-in-n* cost
— the same structural idea over a strictly smaller expressive class.

### What CV cannot express yet

* **`squeeze` is still prep-only.** `displace` now has an operator form, so
  *squeeze → phase → displace* works and is compared; but `squeezed_vacuum(r)`
  remains a constructor from vacuum, so squeezing a state that already has
  structure cannot be written. The cross-check pins the unexpressible set
  exactly, so a new gap cannot arrive unnoticed.
* **Single mode only.** No beamsplitter, no multimode state — so mode mixing,
  which is where independent CV implementations most often diverge on
  convention, is untested because it is unimplemented.

This is why there is **no CV QAOA example** here. CV QAOA is *gradient descent
in a potential* — see Verdon et al., **arXiv:1902.00409**, and the photonic
implementation of Enomoto et al., **Phys. Rev. Research 5, 043005 (2023)**
(arXiv:2206.07214) — and its mixer needs displacement as an **operator**. The
blocker is the missing operator, not the missing example.

---

## 4. Mesh-based algorithms — what is not here yet

A QFT over `m` modes is a multiport interferometer: take the DFT matrix and
decompose it into beam splitters and phase shifters.

> Reck, Zeilinger, Bernstein & Bertani, **Phys. Rev. Lett. 73, 58 (1994)** —
> triangular decomposition, `N(N−1)/2` splitters.
>
> Clements, Humphreys, Metcalf, Kolthammer & Walmsley,
> **Optica 3(12), 1460–1465 (2016)**, arXiv:1603.08788 — symmetric mesh, same
> count but **half the optical depth** and better loss tolerance.

**This now exists** — `examples/circuits/qft4.opticqasm`, the 4×4 DFT as 3 phase
shifters + 6 beam splitters, generated by `cargo run -p omega-backend-photonics
--example gen_qft` (recomposition 2.884e-16).

`clements_decompose` was **deleted**: it delegated to Reck while the module
header advertised Clements' loss tolerance, so a QFT example built on it would
have inherited a claim the code did not implement. Reck is correct and
sufficient; implementing Clements properly is separate work.

**Verify it with the Fourier suppression law**, not with uniform output:

> Tichy, Tiersch, de Melo, Mintert & Buchleitner, "Zero-transmission law for
> multiport beam splitters", **Phys. Rev. Lett. 104, 220405 (2010)**.

```console
$ omega-run examples/circuits/qft4.opticqasm --backend photonics \
    --input 1,1,1,1 --shots 200000
```

For a cyclic input, every output with `Σⱼ j·nⱼ ≢ 0 (mod m)` is **strictly
forbidden**. Measured: **25 of 35 configurations exactly zero, 0 violations**;
Perceval independently agrees (same 10 populated, TVD 0.00357).

Do **not** rely on "single-photon output is uniform 1/m". It only constrains
`|U_jk| = 1/√m`, so every complex Hadamard passes — measured at m=4, the real
Hadamard `H⊗H/2`, the continuous family `F₄(a=0.7)`, and a row-permuted DFT all
pass while being visibly not the DFT. It is also subsumed by recomposition.
Unlike either, the suppression law exercises the **permanent**.
