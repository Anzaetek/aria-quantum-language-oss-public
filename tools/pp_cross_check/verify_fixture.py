#!/usr/bin/env python3
"""Re-run monoprop live and check the COMMITTED fixture still matches it.

The committed fixture lets `cargo test -p omega-backend-pauliprop` cross-check
against monoprop with no Python, no venv and no network. But a committed
fixture has one failure mode the Rust test cannot see: if our conventions
drifted and somebody regenerated the fixture to match, the test goes green on
wrong numbers — the fixture would then record our own opinion, not an
independent one. This script closes that: it regenerates from monoprop and
compares to what is on disk. Run it via `ARIA_PP_XCHECK=1 ./ci.sh`.

Compares NUMERICALLY, not by text diff — float repr varies across numpy
versions. The recipe (`ops`, `observable`, truncation knobs) is compared
exactly: if the recipe changed, the numbers are not comparable at all.

Exit 0 = fixture matches live monoprop. Exit 1 = drift, or monoprop missing
(reported distinctly: an absent tool must not read as a passing check).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tools"))
from xcheck_drift import run  # noqa: E402

FIXTURE = HERE / "monoprop_fixture.jsonl"
GENERATOR = HERE / "monoprop_ref.py"
CANDIDATE_PYTHONS = [HERE / ".venv" / "bin" / "python"]
TOL = 1e-12


def compare_case(name, d, lv):
    return [(name, abs(d["expectation"] - lv["expectation"]))]


if __name__ == "__main__":
    sys.exit(run(
        fixture=FIXTURE, generator=GENERATOR, candidates=CANDIDATE_PYTHONS,
        oracle="monoprop",
        recipe_fields=("n_qubits", "ops", "observable", "cutoff", "lower_atol"),
        compare_case=compare_case, tol=TOL,
    ))
