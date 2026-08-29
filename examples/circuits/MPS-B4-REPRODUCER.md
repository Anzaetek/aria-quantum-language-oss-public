<!-- SPDX-License-Identifier: Apache-2.0 -->
# B4 reproducer — MPS vs statevector, and where MPS stops being able to win

**Circuits authored 2026-08-19.** They are NOT the circuit the original change
request used. That one arrived with an external CR, was never committed, and no
copy exists in this tree — so a defect recorded as "REPRODUCED" had, for six
days, no reproducer anyone could run. These were written to close that, and the
distinction matters when reading what follows: a negative result on these does
not clear B4, it only says these shapes are not the shape.

Verified before authoring: no tracked `.qasm` in the repository reached 15
qubits. The widest was **4**. So B4 was not merely missing its own fixture — the
corpus had nothing wide enough to investigate any width-dependent behaviour at
all.

## The circuits

| file | qubits | ops | shape |
|---|---|---|---|
| `mps_chain_15q.qasm` | 15 | 159 | low-entanglement nearest-neighbour brick-wall |
| `mps_chain_19q.qasm` | 19 | 295 | same, at B4's width and op count (B4: 19q, 306 ops) |
| `mps_chain_22q.qasm` | 22 | 291 | same, wider |
| `mps_volume_19q.qasm` | 19 | 167 | **CONTROL** — long-range entanglers, high bond |

The control is the load-bearing one. Without it a slow MPS run says only "MPS is
slow at 19 qubits"; with it, the two shapes can be told apart.

## What was measured (2026-08-19, this box, release, 100 shots)

The low-entanglement chains do **not** reproduce B4. MPS runs them in under
10 ms against statevector's 0.15 s — MPS *wins*, which is the expected result
for a low-entanglement circuit and refutes the hypothesis B4 records first
("fixed-bond χ=64 doing full-rank work on a low-entanglement state"). If that
were the mechanism, these would be slow too.

The control reproduces it, and identifies the trigger:

| backend | time | outcome |
|---|---|---|
| `statevector` | **0.08 s** | — |
| `mps` (χ=64 default) | 0.26 s | REFUSES: truncation certificate 1.119e-1 exceeds the 1e-6 ceiling |
| `mps:256` | 1.59 s | REFUSES: 1.064e-1 exceeds the ceiling |
| `mps:512` | **158.09 s** | succeeds |
| `mps:1024` | 160.39 s | succeeds |

**~2000× slower than statevector, which is B4's own second figure.**

## The mechanism

It is not low entanglement, and it is not the bond dimension alone. It is the
*window between them*:

* below some χ the truncation guard fires, the discarded weight exceeds the
  ceiling, and the run refuses in under two seconds — fast, and correct;
* at χ large enough that the discarded weight falls under the ceiling, the guard
  does **not** fire, and the backend grinds through full-rank work to completion.

At n=19 that χ is 512, and 512 is exactly where MPS stops being able to win:

```
dense statevector, n=19 : 2^19       =    524,288 amplitudes
mps χ=64                : 19·64²·2   =    155,648 params  (smaller — MPS wins)
mps χ=256               : 19·256²·2  =  2,490,368 params  (4.7× LARGER than dense)
mps χ=512               : 19·512²·2  =  9,961,472 params  (19× LARGER than dense)
```

So the successful run spends 158 s manipulating a representation nineteen times
larger than the object it represents, to produce what the dense path produces in
0.08 s. χ=1024 costs the same as χ=512 (160.39 s) because the physical bond at
the centre of a 19-qubit chain cannot exceed 2^9 = 512 — the extra allowance
buys nothing, which is consistent with the explanation and would not be if the
cost were driven by the parameter rather than the state.

## What this says about B4

The remaining hypotheses in `PLAN-CR-20260813.md` B4 were: fixed-bond full-rank
work, a qubit-ordering effect, or a truncation threshold that never triggers.
**The third is correct**, and the first is correct only as a consequence of it —
full-rank work happens precisely when the threshold does not save you from it.

Whether the ORIGINAL circuit hit this window cannot be known without it. What is
now known is that this tree has a path that costs 158 s at 19 qubits, that it is
reachable from a committed fixture, and that nothing warns the user they have
asked for a representation larger than the state.
