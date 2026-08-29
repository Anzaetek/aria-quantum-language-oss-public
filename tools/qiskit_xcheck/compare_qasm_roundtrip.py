# SPDX-License-Identifier: Apache-2.0
"""Aria NATIVE execution vs Qiskit running Aria's EXPORTED QASM.

`OPTIONAL_TESTS.md` gap #5, and the distinction that document insists on:

  > comparing *Aria native* against *Qiskit-on-Aria's-export*, not just both
  > engines on the same text, since a lossy export makes them agree.

The reference is Aria's execution of the circuit it held IN MEMORY. Qiskit runs
the exported text. So anything the exporter loses shows up as a distribution
difference — which is how the 2026-08-08 `if(c==V)` defect was found, and the
reason `qasm2_dialect.py` could not find it: that harness compares
`Operator(loaded)`, and `Operator()` raises on measurement and on classical
conditions, so it never looks at a circuit of this shape.

Consumes the records emitted by
`cargo run -p aria-runtime --bin qasm_roundtrip_xcheck`.

# The key layout is VERIFIED, not assumed

Aria emits one flat bit string, least-significant bit last. Qiskit emits
registers space-separated in reverse declaration order — `"00 0"` for
`creg m[1]; creg c[2]`, i.e. `c[1]c[0] m[0]`. Stripping the spaces gives Aria's
string, and that was checked empirically before this file was written.

It is also checked at RUNTIME below, against `qc.cregs`: if Qiskit ever changes
that ordering, this fails with a message naming the layout rather than reporting
a physics disagreement. A convention mismatch that presents as a numerical error
is the most expensive kind to debug.
"""
import sys

from qiskit.qasm2 import loads
from qiskit_aer import AerSimulator


def parse_records(text):
    recs, cur, mode = [], None, None
    for line in text.splitlines():
        if line.startswith("R "):
            parts = line.split()
            cur = {"id": parts[1], "note": None, "qasm": "", "counts": {}, "shots": 0}
            if "SHOTS" in parts:
                cur["shots"] = int(parts[parts.index("SHOTS") + 1])
            else:
                # EXPORT_REFUSED / LOWER_FAILED / VACUOUS_CORPUS — a result, not
                # a record to compare. Kept and reported rather than dropped.
                cur["note"] = " ".join(parts[2:])
            recs.append(cur)
        elif line == "QASM_BEGIN":
            mode = "qasm"
        elif line == "QASM_END":
            mode = None
        elif mode == "qasm" and cur is not None:
            cur["qasm"] += line + "\n"
        elif line.startswith("COUNTS ") and cur is not None:
            for tok in line[7:].split():
                k, v = tok.rsplit(":", 1)
                cur["counts"][k] = int(v)
    return recs


def canonical(counts, cregs):
    """Qiskit counts -> Aria's flat key, with the layout asserted."""
    sizes_expected = [r.size for r in reversed(cregs)]
    out = {}
    for key, v in counts.items():
        fields = key.split(" ")
        if [len(f) for f in fields] != sizes_expected:
            raise SystemExit(
                f"compare_qasm_roundtrip.py: Qiskit counts key {key!r} splits into "
                f"field widths {[len(f) for f in fields]}, but the circuit's cregs "
                f"in reverse declaration order are {sizes_expected}. The register "
                f"layout convention has changed; fix the mapping rather than the "
                f"tolerance."
            )
        out[key.replace(" ", "")] = out.get(key.replace(" ", ""), 0) + v
    return out


def tvd(a, b, shots):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / shots


def main():
    text = open(sys.argv[1]).read()
    recs = parse_records(text)
    if not recs:
        raise SystemExit("compare_qasm_roundtrip.py: no records — the generator emitted nothing")

    sim = AerSimulator()
    n_ok = n_bad = 0
    notes = []
    worst, worst_id = 0.0, None

    for r in recs:
        if r["note"]:
            notes.append(f"{r['id']}: {r['note']}")
            continue
        qc = loads(r["qasm"])
        res = sim.run(qc, shots=r["shots"], seed_simulator=1234).result()
        qk = canonical(res.get_counts(), qc.cregs)
        d = tvd(r["counts"], qk, r["shots"])
        if d > worst:
            worst, worst_id = d, r["id"]
        # 8000 shots: the 5-sigma sampling band on a two-outcome distribution is
        # ~0.028, so 0.05 separates sampling noise from a dropped construct,
        # which moves the distribution by ~0.5.
        if d > 0.05:
            n_bad += 1
            print(f"DISAGREE circuit {r['id']}: TVD {d:.4f}\n"
                  f"  aria native : {sorted(r['counts'].items())}\n"
                  f"  qiskit-on-export: {sorted(qk.items())}\n"
                  f"  exported text:\n{r['qasm']}")
        else:
            n_ok += 1

    for n in notes:
        print(f"NOTE {n}")
    print(f"QASM round-trip (aria native vs qiskit-on-aria's-export): "
          f"{n_ok} agree, {n_bad} disagree, worst TVD = {worst:.4f}"
          + (f" (circuit {worst_id})" if worst_id else ""))
    if notes:
        print(f"  {len(notes)} record(s) were not comparable — listed above. A "
              f"refusal or a vacuous circuit is a RESULT, not a pass.")
    # A run that compared NOTHING must not report success — every record
    # refused, or the corpus was empty.
    #
    # The condition is `n_ok + n_bad == 0`, not `n_ok == 0`. The first draft used
    # `n_ok == 0` and printed "no circuit was actually compared" on a mutation
    # run where all 12 circuits WERE compared and all 12 disagreed — a
    # misleading message on exactly the run the harness exists to produce.
    if n_ok + n_bad == 0:
        print("FAIL: no circuit was comparable — every record was refused or vacuous")
        return 1
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
