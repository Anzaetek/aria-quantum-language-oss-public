<!-- SPDX-License-Identifier: Apache-2.0 -->
# quditsv against external qudit simulators

PLAN-QUDIT.md **Q2 leg (iii)**. `xcheck.py` builds the same circuits in
`omega-backend-quditsv` (through `omega-run --backend quditsv --statevector
--format json`) and in two external simulators, and compares statevectors at
`1e-9` with the global phase aligned.

| oracle | licence (Q0) | what it can say | built how |
|---|---|---|---|
| **mqt.qudits** 0.5.2 | MIT | mixed dimensions; the non-Clifford `rxy` | `QuantumCircuit(n, dims).r/h/x/z/csum` then `.simulate()` — **never** `from_qasm`, which drops `rxy` parameters (Q0 defect D2) |
| **QuickQudits** 1.0.1 | Apache-2.0 | one global `d`; Clifford gates `H` (Fourier), `X` (shift), `Z` (clock), `CNOT` (= `csum`) | `QuantumCircuit(n, d).H/X/Z/CNOT` then `.execute()` |

Both oracles index the state **big-endian** (qudit 0 most significant);
omega indexes wire 0 **least** significant, as every qubit engine here does.
The harness reindexes, and `--pin-order` proves it has to: on an asymmetric
circuit (`x` on wire 0 of a `[3,2]` register) the un-reindexed comparison
must FAIL. A reindexing that was accidentally the identity would pass every
symmetric circuit and prove nothing.

`ci.sh` runs the script under each Python that imports its oracle
(`make -C crates/omega-bridges/python mqt-venv` / `quickqudits-venv`, or
`OMEGA_MQT_PYTHON` / `OMEGA_QUICKQUDITS_PYTHON`) and registers a skip for
each that is absent.

Semantics pinned here and in `crates/omega-backend-quditsv/tests/anchors.rs`,
from the oracles' own output rather than their docs: `rxy(0,1,0.7,0.3)|0⟩ =
[0.939373, 0.101333 − 0.327583i, 0]`; `csum` on control `|2⟩`, target `|0⟩`
of qutrits gives target `|2⟩` (`t + c mod d`); `H` on `d = 3` is the Fourier
gate; `Z` puts `ω^k` on `|k⟩`.
