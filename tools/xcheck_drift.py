# SPDX-License-Identifier: Apache-2.0
"""The shared skeleton of every "re-run the oracle live and compare it to the
committed fixture" script.

Four scripts carried this ~45-line body each (`cv_cross_check/verify_fixture.py`,
`verify_multimode_fixture.py`, `verify_loss_fixture.py`,
`pp_cross_check/verify_fixture.py`) and had already diverged in small ways —
one looked for the venv relative to the CWD, one keyed `probs` by occupation,
one failed on the first bad case and the others at the end. What differs
between them is REAL and stays in each script: which interpreter can import
the oracle, which fields make up the recipe, and how one case is compared.
What is the same — discover the interpreter, run the generator, parse both
JSONL streams, refuse an `error` row, note a version change, diff the case
sets, fold the worst delta, report — lives here once.

A script supplies:

* `fixture`, `generator` — paths;
* `candidates` — interpreters to try, in order, before `sys.executable`;
* `oracle` — the name printed in messages ("piquasso", "monoprop");
* `recipe_fields` — the fields that must be EQUAL between fixture and live;
  if any differs the numbers are not comparable and the case FAILS outright;
* `compare_case(name, disk, live)` — returns an iterable of `(label, delta)`
  pairs, or a string describing a structural change (which FAILS outright).

Exit 0 = the fixture matches the live oracle within `tol`. Exit 1 = drift, a
recipe change, a case-set change, or the oracle not running — reported
distinctly, because an absent tool must never read as a passing check.
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def pick_python(candidates):
    """First executable candidate, else the interpreter running this script."""
    for cand in candidates:
        cand = Path(cand)
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return sys.executable


def load(lines):
    """Parse a generator's JSONL: one optional `meta` row, then one row per
    case keyed by `case`. An `error` row is the generator saying the oracle
    could not run — raised, so the caller reports it as FAIL, not as drift."""
    meta, cases = {}, {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "error" in row:
            raise RuntimeError(row["error"])
        if "meta" in row:
            meta = row["meta"]
            continue
        cases[row["case"]] = row
    return meta, cases


def run(*, fixture, generator, candidates, oracle, recipe_fields, compare_case,
        tol=1e-12, label=""):
    """The whole check. Returns the process exit code."""
    fixture, generator = Path(fixture), Path(generator)
    what = f"{label} fixture" if label else "fixture"
    if not fixture.exists():
        print(f"FAIL: no committed {what} at {fixture}", file=sys.stderr)
        return 1
    try:
        proc = subprocess.run(
            [pick_python(candidates), str(generator)],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        print("SKIP: python not available", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"FAIL: {label + ' ' if label else ''}{oracle} generator did not run.",
              file=sys.stderr)
        print(exc.stderr or exc.stdout, file=sys.stderr)
        return 1

    try:
        live_meta, live = load(proc.stdout.splitlines())
    except RuntimeError as exc:
        print(f"FAIL: {oracle} reported: {exc}", file=sys.stderr)
        return 1
    disk_meta, disk = load(fixture.read_text().splitlines())

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
        for field in recipe_fields:
            if d[field] != lv[field]:
                print(f"FAIL: {name}: recipe field {field!r} changed", file=sys.stderr)
                return 1
        result = compare_case(name, d, lv)
        if isinstance(result, str):
            print(f"FAIL: {name}: {result}", file=sys.stderr)
            return 1
        for where, diff in result:
            if diff > worst:
                worst_name, worst = where, diff

    if worst > tol:
        print(
            f"FAIL: committed {what} has DRIFTED from live {oracle} — "
            f"worst {worst:.3e} at {worst_name} (tolerance {tol:.1e}).\n"
            f"      Either {oracle} changed, or the fixture was regenerated to "
            f"match a change in our own code. The second is the dangerous one.",
            file=sys.stderr,
        )
        return 1

    version = live_meta.get(oracle)
    version = f" {version}" if version else ""
    print(f"OK: {what} matches live {oracle}{version} across {len(disk)} cases "
          f"(worst {worst:.3e}{' at ' + worst_name if worst_name else ''})")
    return 0
