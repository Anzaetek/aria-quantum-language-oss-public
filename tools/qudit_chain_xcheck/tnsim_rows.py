#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PLAN-QUDIT Q4, the external comparison point: MQT Qudits' tensor-network
backend (`tnsim`) on the SAME qutrit chains the Rust harness measured.

Input is the `--json` file `qudit_chain_profile` wrote: every circuit as a
neutral gate list, and every MPS row with its certificate. This script builds
each circuit in mqt.qudits through its Python API — never `from_qasm`, which
drops `rxy` parameters (Q0 defect D2) — runs `tnsim`, times it, reads the
process peak RSS, and computes the same observable (`S_z` on the middle site)
from its statevector. Both sides are then compared:

  * against the exact dense reference the Rust side recorded (quditsv), where
    quditsv could hold the state — tnsim must agree at 1e-9 or the row FAILS
    (that is the oracle check; a disagreement means one of the two external
    engines is wrong about a gate, which is a finding, not a tolerance);
  * against every MPS row that was CERTIFIED (`discarded_weight <= 1e-12`,
    i.e. exact regime) at 1e-8 — same verdict;
  * MPS rows that were refused are printed with tnsim's value beside the
    UNCERTIFIED one, so the reader can see how far the uncertified number was
    from the truth — reported, not asserted, because the engine itself made no
    claim about it.

A10: the script asserts its own precondition — every circuit's dimensions are
all > 2 and tnsim's returned state is normalised — and exits non-zero rather
than print a row it cannot vouch for. Each tnsim run is a subprocess under an
address-space limit and a timeout, so a row that would exhaust the box is
published as `tnsim: REFUSED (limit)` and the table survives.

Both engines index the state big-endian (qudit 0 most significant) on the
MQT side and little-endian on the omega side; the observable is diagonal on a
single site, so only that site's digit needs reading, and it is read with the
big-endian stride mqt uses.

Usage:
  <python-with-mqt.qudits> tools/qudit_chain_xcheck/tnsim_rows.py ROWS.json \
      [--as-limit-gib 32|0] [--timeout-s 900]

`--as-limit-gib 0` sets no RLIMIT_AS (it counts virtual mappings, and a BLAS
process can fail a 6 GiB allocation under a 32 GiB cap). macOS cannot set
RLIMIT_AS at all; the worker says so and runs uncapped. Whatever external cap
you use must watch the process TREE, not this script's PID: the memory is in
the worker subprocess (parent 17 MB while the worker held 3.7 GiB).
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import time

TOL_EXACT = 1e-9
TOL_MPS_CERTIFIED = 1e-8
CERTIFIED_DW = 1e-12


def worker(argv: list[str]) -> int:
    """Run ONE circuit under tnsim; print a JSON line. Invoked as a subprocess."""
    spec = json.loads(argv[0])
    as_limit = int(argv[1])
    if as_limit > 0:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (as_limit, as_limit))
        except (ValueError, OSError) as e:
            # macOS accepts the constant and rejects every value
            # ("current limit exceeds maximum limit"): RLIMIT_AS is
            # effectively unsupported there. Say so and run uncapped rather
            # than turn the whole leg into a failure on the box the cap was
            # never protecting; the caller's own watchdog must cover it, and
            # it must watch the process TREE — this worker's memory is not
            # the parent's (measured 2026-09-30: parent 17 MB, worker 3.7 GiB).
            print(
                f"NOTE: RLIMIT_AS {as_limit >> 30} GiB not settable on this "
                f"platform ({e}); running without an address-space cap",
                file=sys.stderr,
            )
    import numpy as np
    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.simulation import MQTQuditProvider

    n, d = spec["n"], spec["d"]
    c = QuantumCircuit(n, [d] * n, 0)
    for g in spec["gates"]:
        name, w, p = g["name"], g["wires"], g["params"]
        if name == "rxy":
            c.r(w[0], [int(p[0]), int(p[1]), float(p[2]), float(p[3])])
        elif name == "csum":
            c.csum(list(w))
        else:
            raise ValueError(name)
    backend = MQTQuditProvider().get_backend("tnsim")
    t = time.perf_counter()
    sv = np.asarray(backend.run(c).result().get_state_vector()).reshape(-1)
    wall = time.perf_counter() - t
    norm = float(np.linalg.norm(sv))
    # S_z = diag((d-1)/2 - k) on site `site`. mqt is big-endian: qudit 0 is
    # the most significant digit, so site w's digit has stride d^(n-1-w).
    site = spec["observable"]["site"]
    stride = d ** (n - 1 - site)
    idx = np.arange(sv.size)
    digit = (idx // stride) % d
    weights = (d - 1) / 2 - digit
    val = float(np.sum(weights * np.abs(sv) ** 2) / (norm**2))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    print(json.dumps({"wall_s": wall, "norm": norm, "value": val, "peak_rss_bytes": rss_bytes}))
    return 0


def human(b: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PiB"


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        return worker(sys.argv[2:])
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    path = sys.argv[1]
    as_limit_gib = 32
    timeout_s = 900
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--as-limit-gib":
            as_limit_gib = int(args[i + 1])
        if a == "--timeout-s":
            timeout_s = int(args[i + 1])
    doc = json.load(open(path))
    try:
        import mqt.qudits  # noqa: F401
    except Exception as e:  # noqa: BLE001 — absence is the finding
        print(f"no mqt.qudits in this Python: {e}", file=sys.stderr)
        return 2
    import mqt.qudits as mq

    box = doc["box"]
    print(
        f"# Q4 external leg — MQT Qudits `tnsim` {getattr(mq, '__version__', '?')} on the "
        f"{len(doc['circuits'])} circuits `{box}` measured (profile `{doc['profile']}`, "
        f"d = {doc['d']}, seed {doc['seed']}); tnsim run on `{subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip() or 'unknown-host'}`\n"
    )
    print("| n | ops | tnsim wall ms | tnsim peak RSS | tnsim ⟨S_z⟩ | exact (quditsv) | \\|Δ exact\\| | MPS certified rows (χ: \\|Δ\\|) | MPS refused rows (χ: \\|Δ uncertified\\|) |")
    print("|---|---|---|---|---|---|---|---|---|")
    failures: list[str] = []
    compared = 0
    for circ in doc["circuits"]:
        n = circ["n"]
        dims_ok = all(dd == circ["d"] for dd in [circ["d"]]) and circ["d"] > 2
        if not dims_ok:
            print(f"circuit n={n}: d = {circ['d']} is not a qudit chain", file=sys.stderr)
            return 3
        spec = json.dumps({k: circ[k] for k in ("n", "d", "gates", "observable")})
        cmd = [sys.executable, __file__, "--worker", spec, str(as_limit_gib << 30)]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            print(f"| {n} | {len(circ['gates'])} | **tnsim: REFUSED** (timeout {timeout_s}s) | — | — | — | — | — | — |")
            continue
        if out.returncode != 0:
            why = (out.stderr.strip().splitlines() or ["?"])[-1][:80]
            limit = f"RLIMIT_AS {as_limit_gib} GiB" if as_limit_gib > 0 else "no rlimit; the host's cap"
            print(f"| {n} | {len(circ['gates'])} | **tnsim: REFUSED** ({limit}: {why}) | — | — | — | — | — | — |")
            continue
        r = json.loads(out.stdout.strip().splitlines()[-1])
        if abs(r["norm"] - 1.0) > 1e-9:
            failures.append(f"n={n}: tnsim state norm {r['norm']} is not 1")
        exact = circ.get("exact")
        d_exact = "—"
        if exact is not None:
            gap = abs(r["value"] - exact)
            d_exact = f"{gap:.2e}"
            compared += 1
            if gap > TOL_EXACT:
                failures.append(f"n={n}: tnsim {r['value']} vs quditsv {exact} (|Δ| {gap:.2e} > {TOL_EXACT})")
        cert_cells, ref_cells = [], []
        for row in doc["rows"]:
            if row["n"] != n:
                continue
            if row["certified"] is not None and row["discarded_weight"] <= CERTIFIED_DW:
                gap = abs(row["certified"] - r["value"])
                cert_cells.append(f"{row['chi']}: {gap:.2e}")
                compared += 1
                if gap > TOL_MPS_CERTIFIED:
                    failures.append(f"n={n} chi={row['chi']}: certified MPS {row['certified']} vs tnsim {r['value']} (|Δ| {gap:.2e})")
            elif row["refused"] is not None and row["uncertified"] is not None:
                ref_cells.append(f"{row['chi']}: {abs(row['uncertified'] - r['value']):.2e}")
            elif row["certified"] is not None:
                # Admitted under the ceiling but not exact: reported, not asserted.
                cert_cells.append(f"{row['chi']}: {abs(row['certified'] - r['value']):.2e} (dw {row['discarded_weight']:.1e})")
        ex = f"{exact:.10f}" if exact is not None else "no exact ref"
        print(
            f"| {n} | {len(circ['gates'])} | {r['wall_s']*1e3:.1f} | {human(r['peak_rss_bytes'])} | "
            f"{r['value']:.10f} | {ex} | {d_exact} | {'; '.join(cert_cells) or '—'} | {'; '.join(ref_cells) or '—'} |"
        )
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    if compared == 0:
        print("\nno comparison was possible (no exact reference and no certified MPS row)", file=sys.stderr)
        return 4
    print(f"\nOK: tnsim agrees on {compared} comparisons (exact at {TOL_EXACT}, certified MPS at {TOL_MPS_CERTIFIED})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
