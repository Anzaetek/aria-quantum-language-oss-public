#!/usr/bin/env python3
"""Re-run piquasso live and check the COMMITTED fixture still matches it.

The committed fixture lets `cargo test` cross-check the CV backend with no
Python, no venv and no network — headless-CI clean. But a committed fixture has
one failure mode that the Rust test cannot see: if our conventions drifted and
somebody regenerated the fixture to match, the test goes green on wrong numbers.
The fixture would then be a record of our own opinion, not an independent one,
and the whole reason for using piquasso would be gone.

This script closes that: it regenerates from piquasso and compares to what is on
disk. Run it via `ARIA_CV_XCHECK=1 ./ci.sh`.

Compares NUMERICALLY, not by text diff — float repr varies across numpy versions
and a textual mismatch there would be noise, while a textual match would not
actually prove the numbers agree.

Exit 0 = fixture matches live piquasso. Exit 1 = drift (or piquasso missing,
which is reported distinctly: an absent tool must not read as a passing check).
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIXTURE = HERE / "piquasso_fixture.jsonl"
GENERATOR = HERE / "piquasso_ref.py"

# Interpreter discovery, in order. `.venv-piquasso` at the repo root is the
# pre-existing environment (FIXES_PLAN.md); prefer it over creating a second
# 400 MB copy of the same wheels. `sys.executable` last, so running this script
# with an interpreter that already has piquasso just works.
CANDIDATE_PYTHONS = [
    REPO / ".venv-piquasso" / "bin" / "python",
    HERE / ".venv" / "bin" / "python",
]


def pick_python():
    for cand in CANDIDATE_PYTHONS:
        if cand.is_file():
            return str(cand)
    return sys.executable

# Regeneration must be bit-comparable at this level. This is NOT the physics
# tolerance (the Rust side owns that, bounded by the backend's own leak metric);
# it is only guarding against a fixture that no longer reflects piquasso.
TOL = 1e-12


def load(lines):
    meta, cases = None, {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "error" in d:
            print(f"generator error: {d['error']}", file=sys.stderr)
            sys.exit(1)
        if "meta" in d:
            meta = d["meta"]
            continue
        cases[d["case"]] = d
    return meta, cases


def main():
    if not FIXTURE.exists():
        print(f"FAIL: no committed fixture at {FIXTURE}", file=sys.stderr)
        return 1

    try:
        proc = subprocess.run(
            [pick_python(), str(GENERATOR)],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        print("SKIP: python not available", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print("FAIL: piquasso generator did not run.", file=sys.stderr)
        print(exc.stderr or exc.stdout, file=sys.stderr)
        return 1

    live_meta, live = load(proc.stdout.splitlines())
    disk_meta, disk = load(FIXTURE.read_text().splitlines())

    if live_meta != disk_meta:
        # Not fatal by itself, but it must be said: a version change is the most
        # likely innocent explanation for any drift below, and the most likely
        # guilty explanation for drift that gets waved through.
        print(f"NOTE: fixture generated with {disk_meta}, running {live_meta}")

    missing = sorted(set(disk) - set(live))
    added = sorted(set(live) - set(disk))
    if missing:
        print(f"FAIL: cases in fixture but not regenerated: {missing}", file=sys.stderr)
    if added:
        print(f"FAIL: cases regenerated but absent from fixture: {added}", file=sys.stderr)
    if missing or added:
        return 1

    worst_name, worst = None, 0.0
    for name, d in sorted(disk.items()):
        lv = live[name]
        if d["ops"] != lv["ops"]:
            print(f"FAIL: {name}: the recipe itself changed", file=sys.stderr)
            return 1
        for field in ("probs", "mean_n"):
            a, b = d[field], lv[field]
            pairs = zip(a, b) if isinstance(a, list) else [(a, b)]
            for x, y in pairs:
                diff = abs(x - y)
                if diff > worst:
                    worst_name, worst = f"{name}.{field}", diff
        for (ar, ai), (br, bi) in zip(d["amps"], lv["amps"]):
            diff = max(abs(ar - br), abs(ai - bi))
            if diff > worst:
                worst_name, worst = f"{name}.amps", diff

    if worst > TOL:
        print(
            f"FAIL: committed fixture has DRIFTED from live piquasso — "
            f"worst {worst:.3e} at {worst_name} (tolerance {TOL:.1e}).\n"
            f"      Either piquasso changed, or the fixture was regenerated to "
            f"match a change in our own code. The second is the dangerous one.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: fixture matches live piquasso {live_meta.get('piquasso')} "
          f"across {len(disk)} cases (worst {worst:.3e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
