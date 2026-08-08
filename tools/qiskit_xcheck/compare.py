import sys
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

worst = 0.0; n_ok = 0; n_bad = 0
worst_tvd = 0.0; ff_ok = 0; ff_bad = 0
_sim = AerSimulator()


def check_feedforward(body):
    """Compare a mid-circuit-measurement circuit on DISTRIBUTIONS.

    The analytic corpus cannot reach these: it is Clifford-only with no
    conditions and MidCircuitMode::Skip. Three real defects lived exactly there
    while it reported 4.441e-16 over 60 cases — so this compares counts, not a
    statevector, because the defects were in the SAMPLING.
    """
    left, right = body.split(" | ")
    parts = left.split()
    n, shots = int(parts[0]), int(parts[1])
    qc = QuantumCircuit(n, n)
    for tok in parts[2:]:
        if tok.startswith("if:"):
            # if:<cbit>,<value>,<gate>:<qubit>
            _, rest = tok.split(":", 1)
            cbit, val, gate_q = rest.split(",", 2)
            gname, gq = gate_q.split(":")
            with qc.if_test((qc.clbits[int(cbit)], int(val))):
                {"x": qc.x, "z": qc.z}[gname](int(gq))
            continue
        name, qs = tok.split(":")
        if name == "measure":
            q, c = (int(x) for x in qs.split(","))
            qc.measure(q, c)
            continue
        idx = [int(x) for x in qs.split(",")]
        {"h": qc.h, "s": qc.s, "sdg": qc.sdg, "x": qc.x, "z": qc.z}[name](*idx)

    res = _sim.run(transpile(qc, _sim), shots=shots, seed_simulator=7).result()
    qk = {}
    for bits, v in res.get_counts().items():
        qk[int(bits.replace(" ", ""), 2)] = qk.get(int(bits.replace(" ", ""), 2), 0) + v

    aria = {}
    for tok in right.split():
        k, v = tok.split(":")
        aria[int(k)] = int(v)

    # Total variation distance between the two distributions.
    keys = set(aria) | set(qk)
    tvd = 0.5 * sum(abs(aria.get(k, 0) - qk.get(k, 0)) for k in keys) / shots
    return n, tvd, aria, qk


for line in open(sys.argv[1]):
    if line.startswith("M "):
        n, tvd, aria, qk = check_feedforward(line[2:].rstrip("\n"))
        worst_tvd = max(worst_tvd, tvd)
        # 5 sigma on 4000 draws is ~0.04; independent RNG streams, so compare
        # distributions statistically rather than bit-for-bit.
        if tvd > 0.05:
            ff_bad += 1
            if ff_bad <= 3:
                print(f"FEEDFORWARD DISAGREE n={n} TVD={tvd:.4f}\n  aria  {aria}\n  qiskit{qk}")
        else:
            ff_ok += 1
        continue
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
if ff_ok or ff_bad:
    print(f"Feedforward (mid-circuit measure + classical control): "
          f"{ff_ok} agree, {ff_bad} disagree, worst TVD = {worst_tvd:.4f}")
if n_bad or ff_bad:
    sys.exit(1)
