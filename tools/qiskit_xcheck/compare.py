import sys
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

worst = 0.0; n_ok = 0; n_bad = 0
for line in open(sys.argv[1]):
    if not line.startswith("C "):
        continue
    body = line[2:].rstrip("\n")
    left, right = body.split(" | ")
    parts = left.split()
    n = int(parts[0])
    qc = QuantumCircuit(n)
    for tok in parts[1:]:
        name, qs = tok.split(":")
        qs = [int(x) for x in qs.split(",")]
        # aria qubit 0 = LSB; Qiskit little-endian matches, so indices map 1:1.
        {"h": qc.h, "s": qc.s, "sdg": qc.sdg, "x": qc.x, "z": qc.z,
         "cx": lambda a, b=None: qc.cx(a, b)}[name](*qs)
    probs = Statevector.from_instruction(qc).probabilities()
    aria = [float(x) for x in right.split()]
    if len(aria) != len(probs):
        print(f"LENGTH MISMATCH n={n}: aria {len(aria)} vs qiskit {len(probs)}"); n_bad += 1; continue
    d = max(abs(a - b) for a, b in zip(aria, probs))
    worst = max(worst, d)
    if d > 1e-9:
        n_bad += 1
        if n_bad <= 3:
            print(f"DISAGREE n={n} maxΔ={d:.3e}\n  aria  {[round(x,6) for x in aria]}\n  qiskit{[round(x,6) for x in probs]}")
    else:
        n_ok += 1
print(f"\nQiskit cross-check: {n_ok} agree, {n_bad} disagree, worst |Δp| = {worst:.3e}")
