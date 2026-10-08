<!-- SPDX-License-Identifier: Apache-2.0 -->
# DITQASM fixtures, written by the reference

PLAN-QUDIT **Q0**. The files under
`crates/omega-parser/tests/fixtures/ditqasm/` are the byte output of
`mqt.qudits` (`QuantumCircuit.to_qasm()`, version in `PROVENANCE.json`). We
did not write them, which is the point: a Q1 grammar that diverges from MQT's
reading fails against bytes we did not author, instead of being validated by
its own reader (the QPY defect, `PLAN-OPEN-20260825.md:934-943`). Two files are
hand-written and say so in `PROVENANCE.json` — the bare `qreg` form, which
MQT never emits, and a corrected `ctl` line (D1 below).

* `gen_fixtures.py` — regenerates the directory. No-op on a clean tree.
* `reparse.py` — re-parses every fixture under `mqt.qudits`, pins the two
  upstream defects and the four §3e.1a answers. `ci.sh` runs it when a Python
  imports `mqt.qudits` (`make -C crates/omega-bridges/python mqt-venv`, or
  `OMEGA_MQT_PYTHON=…`) and registers a skip otherwise.

## What the fixtures settle (PLAN-OPEN §3e.1a's four unknowns)

| unknown | answer, from `mqt/qudits/quantum_circuit/qasm.py` | fixture |
|---|---|---|
| `qreg matter [2];` with no dimension group | **two** qudits, each `d = 2` — `parse_qreg`: `qdims = [2] * nq`. MQT's own emitter always writes the group: it re-emits this as `qreg matter [2][2,2];` | `bare_qreg_input` → `bare_qreg_reemitted` |
| `h` / `x` / `z` / `cx` on a qudit | `h` is the **Fourier** gate, `x` the cyclic shift `X_d`, `z` the clock `Z_d` — generalised. **`cx` is not**: it is a parametrised two-level embedded gate, emitted as `cx (0, 1, 1, 0.0) q[0], q[1]` even on qubits; the generalised controlled-X is `csum`, emitted without parameters | `qubits_default`, `mixed_325`, `two_regs` |
| level list after `ctl` | per control qudit, **positional** with the control list: `ctl q[0] q[2] [2, 1]` means q[0] at level 2, q[2] at level 1 | `ctl_hand_corrected` |
| version lock | the header regex is `(DITQASM\s+2.0;)\|(include\s+"qelib1.inc";)`. `DITQASM 3.0;`, `DITQASM 2.1;` and `OPENQASM 2.0;` are `SyntaxError`. `measure` and `barrier` lines are matched and ignored | (inline in `reparse.py`) |

## Two upstream defects, pinned so a fix is noticed

* **D1 — the `ctl` writer emits the target for every control.** For
  `h` on `q[1]` controlled by `[0, 2]` at levels `[2, 1]`, `to_qasm()` writes
  `h q[1] ctl q[1] q[1] [2, 1];`. MQT cannot re-parse its own output
  (`IndexError: Controls overlap with targets`). The committed
  `ctl_writer_defect.qasm` is what MQT writes; `ctl_hand_corrected.qasm` is
  what its parser accepts.
* **D2 — `from_qasm` drops `rxy`/`rz` parameters.** The parser hands
  parameters as a tuple; `R`/`Rz` keep a list and silently store `None` for a
  tuple, so re-emission writes `rxy q[0];`. `cx` keeps its tuple, which is why
  `qubits_default` round-trips and `mixed_325` does not.

`reparse.py` fails if either stops reproducing: upstream fixed it, this README
and `PROVENANCE.json` describe stale behaviour, and the pin in
`requirements-mqt.txt` should move deliberately.

## What this is not

No numbers, no certificate. Q1's own parser is not exercised here yet — that
leg is added when Q1 lands, reading the same files.
