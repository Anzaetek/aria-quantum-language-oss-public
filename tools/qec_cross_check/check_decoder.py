#!/usr/bin/env python3
"""Cross-check aria-qec's exact minimum-weight surface-code decoder against
**PyMatching** — the field-standard MWPM decoder (the one paired with stim in
the QEC community).

aria-qec ships its own decoder (`ecc/mwpm.rs::decode_mwpm_correction`, exact
min-weight by bounded enumeration) and unit-tests it on single errors. This
closes the loop against an *independent* decoder: the Rust
`dump_surface_code` example emits the rotated surface code (check matrices +
logical operators) and a seeded batch of code-capacity error trials, each
already decoded by aria-qec; here we re-decode the **identical** error samples
with PyMatching and assert the two agree.

Three assertions per distance d ∈ {3, 5}:
  1. Guaranteed-correctable: every error of weight ≤ ⌊(d−1)/2⌋ must leave no
     logical error under PyMatching (validates the exported check matrix +
     observable wiring against an independent decoder; aria-qec's own decoder is
     proven on this set by its Rust unit tests).
  2. Shot-for-shot logical-class agreement on the Monte-Carlo samples ≥ 99%
     (both are minimum-weight, so they may differ only on weight-ties).
  3. Logical-error-rate agreement: aria-qec's rate ≈ PyMatching's rate on the
     identical samples, |Δ| ≤ 3σ.

Both CSS sectors are checked (X errors via the Z-checks + logical-Z observable;
Z errors via the X-checks + logical-X observable).

Invoked by run.sh alongside check_qec.py.
"""

import itertools
import json
import math
import os
import subprocess
import sys

import numpy as np
import pymatching

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PASS, FAIL = 0, 0


def check(desc, ok, detail=""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    PASS, FAIL = (PASS + 1, FAIL) if ok else (PASS, FAIL + 1)
    print(f"  [{tag}] {desc}{('  ' + detail) if detail else ''}")


def dump(d, p, shots, seed):
    out = subprocess.run(
        ["cargo", "run", "-q", "-p", "aria-qec", "--example", "dump_surface_code",
         "--", str(d), str(p), str(shots), str(seed)],
        cwd=REPO, capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"dump_surface_code failed: {out.stderr.strip()}")
    lines = out.stdout.splitlines()
    header = json.loads(lines[0])
    trials = []
    for ln in lines[1:]:
        xe, ze, oxf, ozf = ln.split(";")
        trials.append((
            [int(t) for t in xe.split()],
            [int(t) for t in ze.split()],
            int(oxf), int(ozf),
        ))
    return header, trials


def check_matrix(checks, n_data):
    H = np.zeros((len(checks), n_data), dtype=np.uint8)
    for i, c in enumerate(checks):
        for q in c:
            H[i, q] = 1
    return H


def observable_row(support, n_data):
    L = np.zeros((1, n_data), dtype=np.uint8)
    for q in support:
        L[0, q] = 1
    return L


def err_vec(indices, n_data):
    e = np.zeros(n_data, dtype=np.uint8)
    for q in indices:
        e[q] = 1
    return e


class Sector:
    """One CSS sector: checks that detect this error type + the logical
    observable it can flip."""

    def __init__(self, checks, logical_support, n_data):
        self.H = check_matrix(checks, n_data)
        self.L = observable_row(logical_support, n_data)
        self.n = n_data
        self.m = pymatching.Matching.from_check_matrix(self.H, faults_matrix=self.L)

    def logical_error(self, err_indices):
        """Does PyMatching's correction leave a logical error on this error?"""
        e = err_vec(err_indices, self.n)
        syndrome = (self.H @ e) % 2
        pred = self.m.decode(syndrome)  # predicted observable flip(s)
        actual = int((self.L @ e)[0] % 2)
        return int(pred[0]) ^ actual


def guaranteed_correctable_ok(sector, n_data, t):
    """Every error of weight ≤ t must decode to no logical error."""
    for w in range(0, t + 1):
        for combo in itertools.combinations(range(n_data), w):
            if sector.logical_error(list(combo)):
                return False, list(combo)
    return True, None


def run_distance(d, p, shots, seed):
    print(f"Case d={d}: aria-qec exact MWPM vs PyMatching (p={p}, {shots} shots)")
    header, trials = dump(d, p, shots, seed)
    n = header["n_data"]
    t = (d - 1) // 2
    # X errors are detected by Z-checks; a residual X flips logical Z.
    xsec = Sector(header["z_checks"], header["logical_z"], n)
    # Z errors are detected by X-checks; a residual Z flips logical X.
    zsec = Sector(header["x_checks"], header["logical_x"], n)

    for label, sec in (("X-sector", xsec), ("Z-sector", zsec)):
        ok, bad = guaranteed_correctable_ok(sec, n, t)
        check(f"d={d} {label}: all weight≤{t} errors correctable (PyMatching)",
              ok, "" if ok else f"failed on {bad}")

    n_agree_x = n_agree_z = 0
    ours_x = ours_z = pm_x = pm_z = 0
    for xe, ze, oxf, ozf in trials:
        pmx = xsec.logical_error(xe)
        pmz = zsec.logical_error(ze)
        n_agree_x += (pmx == oxf)
        n_agree_z += (pmz == ozf)
        ours_x += oxf
        ours_z += ozf
        pm_x += pmx
        pm_z += pmz
    N = len(trials)

    for label, agree, ours, pm in (
        ("X-sector", n_agree_x, ours_x, pm_x),
        ("Z-sector", n_agree_z, ours_z, pm_z),
    ):
        frac = agree / N
        check(f"d={d} {label}: shot-for-shot logical-class agreement ≥ 99%",
              frac >= 0.99, f"{frac*100:.2f}% ({agree}/{N})")
        r_ours, r_pm = ours / N, pm / N
        rbar = max((r_ours + r_pm) / 2, 1e-9)
        sigma = math.sqrt(rbar * (1 - rbar) / N)
        tol = max(3 * sigma, 1e-4)
        check(f"d={d} {label}: logical rate aria≈PyMatching (≤3σ)",
              abs(r_ours - r_pm) <= tol,
              f"aria={r_ours:.4f} pymatching={r_pm:.4f} tol={tol:.4f}")


def main():
    import pymatching as _pm

    print(f"reference: PyMatching {getattr(_pm, '__version__', '?')} (MWPM)")
    # p=0.05 is well below the surface-code threshold, so trials are dominated by
    # low-weight (tie-free) errors; the two min-weight decoders should agree
    # essentially shot-for-shot. 20k shots resolves the ~1e-2 (d=3) rate.
    run_distance(3, 0.05, 20_000, 7)
    run_distance(5, 0.05, 20_000, 7)
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
