#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate crates/omega-parser/tests/fixtures/ditqasm/ from mqt.qudits itself.

PLAN-FERMIONIC-QUDIT Q0. The fixtures are the test: every `.qasm` file here
except the two labelled `hand-written` in PROVENANCE.json is the byte output
of `mqt.qudits.QuantumCircuit.to_qasm()` — the reference implementation writes
them, we do not. A Q1 grammar that diverges from MQT's reading fails against
bytes we did not author, which is the whole point (the QPY lane's
writer-validated-by-its-own-reader defect, PLAN-OPEN-20260825.md:934-943).

Run from the repo root with a Python that imports mqt.qudits
(`make -C crates/omega-bridges/python mqt-venv`):

    crates/omega-bridges/python/.venv-mqt/bin/python tools/ditqasm_xcheck/gen_fixtures.py

Re-running on a clean tree must be a no-op. A diff means upstream changed its
emitter: bump the pin in requirements-mqt.txt deliberately, regenerate, and
re-read PROVENANCE.json's `defects` — never hand-edit a fixture.
"""

from __future__ import annotations

import importlib.metadata
import json
import pathlib
import sys

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components import QuantumRegister
from mqt.qudits.quantum_circuit.components.extensions.controls import ControlData

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "crates" / "omega-parser" / "tests" / "fixtures" / "ditqasm"


def emit(files: dict, name: str, c: QuantumCircuit, note: str) -> str:
    text = c.to_qasm()
    (OUT / f"{name}.qasm").write_text(text)
    d = QuantumCircuit()
    d.from_qasm(text)
    files[name] = {
        "origin": "mqt.qudits QuantumCircuit.to_qasm()",
        "dims": list(c.dimensions),
        "reparsed_dims": list(d.dimensions),
        "n_gates": len(c.instructions),
        "roundtrip_identical": d.to_qasm() == text,
        "note": note,
    }
    return text


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files: dict = {}

    c = QuantumCircuit(1, [3], 0)
    c.h(0)
    emit(files, "qutrit_h", c, "single qutrit, generalised Fourier H")

    c = QuantumCircuit(3, [3, 2, 5], 0)
    c.h(0)
    c.x(1)
    c.z(2)
    c.cx([0, 1])
    c.csum([2, 0])
    c.r(0, [0, 1, 1.2, 0.3])
    c.rz(2, [0, 2, 0.7])
    c.s(1)
    emit(
        files,
        "mixed_325",
        c,
        "mixed radix [3,2,5]; parametrised rxy/rz — NOT round-trip identical, see defects.D2",
    )

    c = QuantumCircuit()
    c.append(QuantumRegister("matter", 2, [3, 3]))
    c.append(QuantumRegister("field", 1, [7]))
    c.h(0)
    c.csum([1, 2])
    c.x(2)
    emit(files, "two_regs", c, "two named registers; gates across them; creg is one flat `meas`")

    c = QuantumCircuit(2, None, 0)
    c.h(0)
    c.cx([0, 1])
    emit(
        files,
        "qubits_default",
        c,
        "no dimensions given → every wire is 2; `cx` is emitted WITH four params (an embedded two-level gate)",
    )

    # Hand-written: the bare `qreg name [n];` form. MQT's emitter never writes
    # it (it always writes the dimension group), so the only way to learn what
    # the PARSER does with it is to feed it one.
    bare = "DITQASM 2.0;\nqreg matter [2];\ncreg meas[2];\nh matter[0];\nh matter[1];\n"
    (OUT / "bare_qreg_input.qasm").write_text(bare)
    e = QuantumCircuit()
    e.from_qasm(bare)
    reemitted = e.to_qasm()
    (OUT / "bare_qreg_reemitted.qasm").write_text(reemitted)
    files["bare_qreg_input"] = {
        "origin": "hand-written (MQT never emits the bare form)",
        "parsed_num_qudits": e.num_qudits,
        "parsed_dims": list(e.dimensions),
        "note": "`qreg matter [2];` = TWO qudits of dimension 2 (qasm.py parse_qreg: `qdims = [2] * nq`)",
    }
    files["bare_qreg_reemitted"] = {
        "origin": "mqt.qudits QuantumCircuit.to_qasm() of the parsed bare_qreg_input",
        "note": "what MQT writes back for the bare form: `[2][2,2]` plus its auto-appended measures",
    }

    # Controlled gate: the writer emits the TARGET index in place of each
    # control index (defects.D1), so this file is what MQT writes and does not
    # itself re-parse.
    c = QuantumCircuit(3, [3, 2, 5], 0)
    c.h(1, ControlData([0, 2], [2, 1]))
    text = c.to_qasm()
    (OUT / "ctl_writer_defect.qasm").write_text(text)
    try:
        d = QuantumCircuit()
        d.from_qasm(text)
        ctl_reparse = "parsed (upstream D1 FIXED — update PROVENANCE and README)"
    except Exception as ex:  # noqa: BLE001 — the exception type is the finding
        ctl_reparse = f"{type(ex).__name__}: {ex}"
    files["ctl_writer_defect"] = {
        "origin": "mqt.qudits QuantumCircuit.to_qasm()",
        "dims": [3, 2, 5],
        "controls_requested": {"indices": [0, 2], "ctrl_states": [2, 1]},
        "reparse": ctl_reparse,
        "note": "see defects.D1",
    }

    fixed = text.replace("h q[1] ctl q[1] q[1] [2, 1];", "h q[1] ctl q[0] q[2] [2, 1];")
    assert fixed != text, "the D1 line shape changed upstream; re-derive the hand correction"
    (OUT / "ctl_hand_corrected.qasm").write_text(fixed)
    d = QuantumCircuit()
    d.from_qasm(fixed)
    g = d.instructions[0]
    files["ctl_hand_corrected"] = {
        "origin": "hand-written (ctl_writer_defect with the control indices corrected by hand)",
        "parsed_controls": {"indices": list(g._controls_data.indices), "ctrl_states": list(g._controls_data.ctrl_states)},
        "note": "level list after `ctl` pairs POSITIONALLY with the control-qudit list: q[0]→level 2, q[2]→level 1",
    }

    prov = {
        "generator": "tools/ditqasm_xcheck/gen_fixtures.py",
        "library": "mqt.qudits",
        "version": importlib.metadata.version("mqt.qudits"),
        "python": ".".join(map(str, sys.version_info[:3])),
        "files": files,
        "defects": {
            "D1": "to_qasm() writes the TARGET index for every control: `h q[1] ctl q[1] q[1] [2, 1];` "
            "for controls [0, 2]. from_qasm() of its own output raises IndexError('Controls overlap with targets').",
            "D2": "from_qasm() drops the parameters of `rxy`/`rz`: the parser hands a tuple, the gate's "
            "validate_parameter accepts only a list, so `_params` is None and re-emission writes `rxy q[0];`. "
            "`cx` keeps its four params through the same path.",
        },
        "header_lock": "parse_ditqasm2_str header regex is `(DITQASM\\s+2.0;)|(include\\s+\"qelib1.inc\";)`: "
        "`DITQASM 3.0;`, `DITQASM 2.1;` and `OPENQASM 2.0;` are SyntaxError.",
    }
    (OUT / "PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n")
    for k, v in files.items():
        print(f"{k:22s} {v.get('note', '')[:90]}")
    print(f"wrote {len(files)} fixtures + PROVENANCE.json under {OUT.relative_to(ROOT)} (mqt.qudits {prov['version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
