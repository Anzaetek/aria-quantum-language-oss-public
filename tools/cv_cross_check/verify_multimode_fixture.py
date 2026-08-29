# SPDX-License-Identifier: Apache-2.0
"""Re-run the MULTI-MODE piquasso generator and diff it against the committed
fixture.

Sibling of `verify_fixture.py`, deliberately separate. That script's comparison
is shaped around the single-mode record (`probs` as a list, plus `mean_n` and
`amps`); this record keys `probs` by occupation string. Generalising one script
over both shapes would put the validated single-mode path through a rewrite to
gain nothing, so this duplicates ~40 lines instead.

What it is FOR is the failure the committed fixture cannot catch on its own: if
our conventions changed and someone regenerated the fixture to match, the Rust
test would go green on wrong numbers. Only re-running piquasso live can see
that. The two halves cover different failures and neither substitutes for the
other.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "piquasso_multimode_fixture.jsonl"
GENERATOR = HERE / "piquasso_multimode_ref.py"
TOL = 1e-12


def pick_python():
    for cand in (
        Path.cwd() / ".venv-piquasso/bin/python",
        HERE / ".venv/bin/python",
    ):
        if cand.exists():
            return str(cand)
    return sys.executable


def load(lines):
    meta, cases = {}, {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
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
        print("FAIL: multimode piquasso generator did not run.", file=sys.stderr)
        print(exc.stderr or exc.stdout, file=sys.stderr)
        return 1

    live_meta, live = load(proc.stdout.splitlines())
    disk_meta, disk = load(FIXTURE.read_text().splitlines())

    if live_meta != disk_meta:
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
        if d["ops"] != lv["ops"] or d["prep"] != lv["prep"]:
            print(f"FAIL: {name}: the recipe itself changed", file=sys.stderr)
            return 1
        # An occupation appearing on one side only is drift, not a rounding
        # difference — checked before the numeric comparison so it cannot be
        # reported as a tiny delta.
        if set(d["probs"]) != set(lv["probs"]):
            only_disk = sorted(set(d["probs"]) - set(lv["probs"]))
            only_live = sorted(set(lv["probs"]) - set(d["probs"]))
            print(
                f"FAIL: {name}: occupation set changed "
                f"(fixture-only {only_disk}, live-only {only_live})",
                file=sys.stderr,
            )
            return 1
        for occ, p in d["probs"].items():
            diff = abs(p - lv["probs"][occ])
            if diff > worst:
                worst_name, worst = f"{name}[{occ}]", diff
        # The mean_n column is a compared oracle too; a fixture whose means
        # were regenerated to match drifted code would pass the probs check
        # (probs and means can drift independently near the cutoff).
        for mode, m in enumerate(d.get("mean_n", [])):
            diff = abs(m - lv["mean_n"][mode])
            if diff > worst:
                worst_name, worst = f"{name}<n_{mode}>", diff

    if worst > TOL:
        print(
            f"FAIL: committed multimode fixture has DRIFTED from live piquasso — "
            f"worst {worst:.3e} at {worst_name} (tolerance {TOL:.1e}).\n"
            f"      Either piquasso changed, or the fixture was regenerated to "
            f"match a change in our own code. The second is the dangerous one.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: multimode fixture matches live piquasso "
          f"{live_meta.get('piquasso')} across {len(disk)} cases "
          f"(worst {worst:.3e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
