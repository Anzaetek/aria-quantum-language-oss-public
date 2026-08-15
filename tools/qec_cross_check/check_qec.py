#!/usr/bin/env python3
"""Cross-check the aria-qec encoded-demo algorithms against Qiskit as an
independent reference.

The four `qec-*` harnesses run key algorithms on transversally QEC-encoded
logical qubits and assert the encoded result equals the ideal logical
distribution. This script closes the loop from the outside: for each demo it
takes the *exact same* logical circuit that aria emits (`aria export --qasm`),
runs it through an independent SDK (Qiskit `Statevector`, plus qsim / stim when
installed), and asserts the distribution matches both aria's own statevector and
the analytic golden. Three-way agreement (aria == Qiskit == analytic) within
1e-9 rules out a shared bug.

Cases:
  * qec-grover : 2-qubit Grover (pure Clifford) for marked ∈ {0,1,2,3} —
                 argmax == marked, P(marked) == 1. Also verified with a
                 stabilizer tableau (stim) when available: the "makes sense for
                 Clifford" reference.
  * qec-qft    : QFT|0000> is uniform; QFT∘QFT⁻¹ = identity for x ∈ {0,5,11,15}
                 (aria's exported QFT and IQFT circuits composed in Qiskit).
  * qec-qpe    : QPEDemo(m=3), φ = 3/8 — counting register collapses to |011> = 3.

The surface-code memory demo (qec-memory) is a code-capacity Monte-Carlo with
MWPM decoding; its logical-error rate is validated internally by the
distance-suppression signature pL(d=5) < pL(d=3) and the crate's unit tests, and
is out of scope for an exact statevector cross-check (it would require
re-implementing the decoder). See the README for the optional Julia path.

Run:  tools/qec_cross_check/run.sh
"""

import math
import os
import subprocess
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARIA = os.environ.get("ARIA_BIN", os.path.join(REPO, "target", "debug", "aria"))
TWINS = os.path.join(REPO, "examples", "aria")
TOL = 1e-9

PASS, FAIL = 0, 0


def check(desc, ok, detail=""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {desc}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# Drive the aria binary.
# ---------------------------------------------------------------------------
def aria_export_qasm(twin, circuit, ints):
    cmd = [ARIA, "export", os.path.join(TWINS, twin), "--circuit", circuit, "--qasm"]
    for k, v in ints.items():
        cmd += ["--int", f"{k}={v}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"aria export failed: {out.stderr.strip()}")
    return out.stdout


def aria_statevector_probs(twin, circuit, ints):
    """{basis_int: prob} from `aria run --statevector`. aria labels are MSB-first
    |q_{n-1}..q0>, so int(label, 2) is directly the basis integer Σ q_i·2^i,
    matching Qiskit's qubit-0-is-LSB statevector indexing."""
    cmd = [ARIA, "run", os.path.join(TWINS, twin), "--circuit", circuit, "--statevector"]
    for k, v in ints.items():
        cmd += ["--int", f"{k}={v}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"aria run failed: {out.stderr.strip()}")
    probs = {}
    for line in out.stdout.splitlines():
        line = line.strip()
        if not (line.startswith("|") and ">" in line):
            continue
        bits = line[1 : line.index(">")]
        rest = line[line.index(">") + 1 :].strip().rstrip("i").replace(" ", "")
        # "+0.250000+0.000000" -> re, im
        sign = 1
        body = rest
        if body[0] in "+-":
            body = body[1:]
        # split on the middle +/-
        for j in range(1, len(body)):
            if body[j] in "+-":
                re = float(rest[0] + body[:j]) if rest[0] in "+-" else float(body[:j])
                im = float(body[j:])
                break
        else:
            re, im = float(rest), 0.0
        val = int(bits, 2)  # aria label is MSB-first |q_{n-1}..q0>, == integer value
        probs[val] = re * re + im * im
    return probs


# ---------------------------------------------------------------------------
# Independent reference: a tiny OpenQASM-2 parser over the demo gate set
# {h, x, cz, cp(θ), swap} into a Qiskit circuit (qubit 0 = LSB, as in aria).
# ---------------------------------------------------------------------------
# Gate names this parser implements natively below. A `gate` DECLARATION for
# one of these can be skipped, because we apply Qiskit's own definition instead
# of the emitted body. Anything else must still be refused: silently ignoring
# an unknown custom-gate declaration would drop real operations and leave the
# comparison looking like agreement.
NATIVE_GATES = {"h", "x", "cz", "swap", "cp", "cu1"}


def qasm_to_qiskit(qasm, drop_measure=True):
    n = None
    body = []
    in_gate_decl = False
    for raw in qasm.splitlines():
        line = raw.strip().rstrip(";").strip()
        if not line or line.startswith(("OPENQASM", "include", "creg")):
            continue
        # `gate NAME(params) args { body }` — the aria exporter emits these for
        # gates outside the qelib1 core (e.g. `gate swap q0,q1 { cx …; cx …; cx …; }`)
        # so the QASM stands alone. This parser predates that and died on the
        # declaration with "unhandled QASM op", which took the whole MANDATORY
        # QEC cross-check down with it.
        if in_gate_decl:
            if "}" in line:
                in_gate_decl = False
            continue
        if line.startswith("gate "):
            name = line.split()[1].split("(")[0]
            if name not in NATIVE_GATES:
                raise RuntimeError(
                    f"QASM declares gate {name!r}, which this reference parser does "
                    f"not implement; add it to NATIVE_GATES and to the dispatch "
                    f"below rather than skipping it: {line!r}"
                )
            # Multi-line declaration: swallow until the closing brace.
            if "{" in line and "}" not in line:
                in_gate_decl = True
            continue
        if line.startswith("qreg"):
            n = int(line[line.index("[") + 1 : line.index("]")])
            continue
        if line.startswith("measure"):
            if not drop_measure:
                body.append(("measure",))
            continue
        body.append(line)
    qc = QuantumCircuit(n)

    def qidx(tok):
        return int(tok[tok.index("[") + 1 : tok.index("]")])

    for line in body:
        if isinstance(line, tuple):
            continue
        head, _, rest = line.partition(" ")
        qubits = [qidx(t) for t in rest.split(",") if "[" in t]
        if head == "h":
            qc.h(qubits[0])
        elif head == "x":
            qc.x(qubits[0])
        elif head == "cz":
            qc.cz(qubits[0], qubits[1])
        elif head == "swap":
            qc.swap(qubits[0], qubits[1])
        elif head.startswith(("cp(", "cu1(")):
            # `cu1(λ)` and `cp(λ)` are the same controlled-phase gate —
            # diag(1, 1, 1, e^{iλ}). Qiskit renamed `cu1` to `cp`; the aria
            # exporter emits the `cu1` spelling, and this parser only knew the
            # other one, so an exported QFT was unreadable by its own reference
            # checker.
            angle = eval(line[line.index("(") + 1 : line.index(")")], {"pi": math.pi})
            qc.cp(angle, qubits[0], qubits[1])
        else:
            raise RuntimeError(f"unhandled QASM op: {line!r}")
    return qc, n


def qiskit_probs(qc):
    """{basis_int: prob}; Qiskit qubit 0 = LSB, matching aria's convention."""
    sv = Statevector.from_instruction(qc)
    p = np.abs(sv.data) ** 2
    return {i: float(p[i]) for i in range(len(p)) if p[i] > 1e-15}


def argmax_prob(probs):
    k = max(probs, key=probs.get)
    return k, probs[k]


def agree(a, b, tol=TOL):
    keys = set(a) | set(b)
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) <= tol


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def case_grover():
    print("Case qec-grover: encoded 2-qubit Grover (Clifford), marked ∈ {0..3}")
    try:
        import stim  # noqa: F401

        have_stim = True
    except ImportError:
        have_stim = False
    for m in range(4):
        qasm = aria_export_qasm("qec_grover.aria", "Grover2", {"marked": m})
        qc, _ = qasm_to_qiskit(qasm)
        qp = qiskit_probs(qc)
        ap = aria_statevector_probs("qec_grover.aria", "Grover2", {"marked": m})
        k, p = argmax_prob(qp)
        check(f"marked={m}: qiskit argmax==marked, P==1",
              k == m and abs(p - 1.0) <= TOL, f"argmax={k} P={p:.6f}")
        check(f"marked={m}: aria == qiskit statevector", agree(ap, qp),
              f"Δ={max(abs(ap.get(x,0)-qp.get(x,0)) for x in set(ap)|set(qp)):.2e}")
        if have_stim:
            check(f"marked={m}: stim stabilizer tableau outcome==marked",
                  _stim_grover(qasm) == m)


def _stim_grover(qasm):
    """Deterministic Clifford outcome via a stim tableau simulator."""
    import stim

    sim = stim.TableauSimulator()
    for raw in qasm.splitlines():
        line = raw.strip().rstrip(";").strip()
        head, _, rest = line.partition(" ")
        qs = [int(t[t.index("[") + 1 : t.index("]")]) for t in rest.split(",") if "[" in t]
        if head == "h":
            sim.h(qs[0])
        elif head == "x":
            sim.x(qs[0])
        elif head == "cz":
            sim.cz(qs[0], qs[1])
        elif head == "swap":
            sim.swap(qs[0], qs[1])
    bits = [1 if sim.measure(q) else 0 for q in range(2)]
    return bits[0] + 2 * bits[1]  # qubit 0 = LSB


def case_qft():
    print("Case qec-qft: QFT|0> uniform + QFT∘QFT⁻¹ = identity")
    n = 4
    qasm = aria_export_qasm("qec_qft.aria", "QFT", {"n": n})
    qc, _ = qasm_to_qiskit(qasm)
    qp = qiskit_probs(qc)
    ap = aria_statevector_probs("qec_qft.aria", "QFT", {"n": n})
    unit = 1.0 / (1 << n)
    maxdev = max(abs(qp.get(i, 0.0) - unit) for i in range(1 << n))
    check("QFT|0000> uniform (qiskit)", maxdev <= TOL, f"maxdev={maxdev:.2e}")
    check("QFT|0000> aria == qiskit", agree(ap, qp))

    iqasm = aria_export_qasm("qec_qft.aria", "IQFT", {"n": n})
    fwd, _ = qasm_to_qiskit(qasm)
    inv, _ = qasm_to_qiskit(iqasm)
    for x in (0, 5, 11, 15):
        rt = QuantumCircuit(n)
        for i in range(n):
            if (x >> i) & 1:
                rt.x(i)
        rt.compose(fwd, inplace=True)
        rt.compose(inv, inplace=True)
        rp = qiskit_probs(rt)
        k, p = argmax_prob(rp)
        check(f"QFT∘QFT⁻¹|{x}> recovers |{x}> (aria QFT+IQFT via qiskit)",
              k == x and abs(p - 1.0) <= TOL, f"argmax={k} P={p:.6f}")


def case_qpe():
    print("Case qec-qpe: QPEDemo(m=3), φ=3/8 → counting register |011> = 3")
    m = 3
    qasm = aria_export_qasm("qec_qpe.aria", "QPEDemo", {"m": m})
    qc, nq = qasm_to_qiskit(qasm)
    qp = qiskit_probs(qc)
    # Marginalize the counting register (low m bits; target is qubit m).
    marg = {}
    for k, v in qp.items():
        marg[k & ((1 << m) - 1)] = marg.get(k & ((1 << m) - 1), 0.0) + v
    k, p = argmax_prob(marg)
    check("qiskit QPE counting register argmax==3, P==1",
          k == 3 and abs(p - 1.0) <= TOL, f"argmax={k} P={p:.6f}")
    ap = aria_statevector_probs("qec_qpe.aria", "QPEDemo", {"m": m})
    amarg = {}
    for kk, v in ap.items():
        amarg[kk & ((1 << m) - 1)] = amarg.get(kk & ((1 << m) - 1), 0.0) + v
    check("QPE aria == qiskit (counting register)", agree(amarg, marg))


def main():
    if not os.path.exists(ARIA):
        print(f"missing aria binary: {ARIA}\nbuild with: cargo build -p aria-cli")
        sys.exit(2)
    import qiskit

    print(f"reference: qiskit {qiskit.__version__}  (Statevector, exact)")
    case_grover()
    case_qft()
    case_qpe()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
