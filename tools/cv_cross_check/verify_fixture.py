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
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tools"))
from xcheck_drift import run  # noqa: E402

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

# Regeneration must be bit-comparable at this level. This is NOT the physics
# tolerance (the Rust side owns that, bounded by the backend's own leak metric);
# it is only guarding against a fixture that no longer reflects piquasso.
TOL = 1e-12


def compare_case(name, d, lv):
    """Single-mode record: `probs` and `mean_n` are lists or scalars, `amps`
    is a list of (re, im) pairs."""
    out = []
    for field in ("probs", "mean_n"):
        a, b = d[field], lv[field]
        pairs = zip(a, b) if isinstance(a, list) else [(a, b)]
        for x, y in pairs:
            out.append((f"{name}.{field}", abs(x - y)))
    for (ar, ai), (br, bi) in zip(d["amps"], lv["amps"]):
        out.append((f"{name}.amps", max(abs(ar - br), abs(ai - bi))))
    return out


if __name__ == "__main__":
    sys.exit(run(
        fixture=FIXTURE, generator=GENERATOR, candidates=CANDIDATE_PYTHONS,
        oracle="piquasso", recipe_fields=("ops",), compare_case=compare_case,
        tol=TOL,
    ))
