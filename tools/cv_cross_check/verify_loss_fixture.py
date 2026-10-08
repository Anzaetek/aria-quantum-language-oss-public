# SPDX-License-Identifier: Apache-2.0
"""Re-run the LOSS-channel piquasso generator and diff it against the
committed fixture.

Third sibling of `verify_fixture.py` / `verify_multimode_fixture.py`,
deliberately separate for the same reason those two are: the loss record has
its own generator (density-matrix simulator) and its own schema. What it is
FOR is unchanged: only re-running piquasso live can catch a fixture
regenerated to match a drifted convention on our side.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tools"))
from xcheck_drift import run  # noqa: E402

FIXTURE = HERE / "piquasso_loss_fixture.jsonl"
GENERATOR = HERE / "piquasso_loss_ref.py"
CANDIDATE_PYTHONS = [
    REPO / ".venv-piquasso" / "bin" / "python",
    HERE / ".venv" / "bin" / "python",
]
TOL = 1e-12


def compare_case(name, d, lv):
    """`probs` keyed by occupation string, plus a `mean_n` column per mode."""
    # An occupation appearing on one side only is drift, not a rounding
    # difference — checked before the numeric comparison so it cannot be
    # reported as a tiny delta.
    if set(d["probs"]) != set(lv["probs"]):
        only_disk = sorted(set(d["probs"]) - set(lv["probs"]))
        only_live = sorted(set(lv["probs"]) - set(d["probs"]))
        return f"occupation set changed (fixture-only {only_disk}, live-only {only_live})"
    out = [(f"{name}[{occ}]", abs(p - lv["probs"][occ])) for occ, p in d["probs"].items()]
    # The mean_n column is a compared oracle too; a fixture whose means were
    # regenerated to match drifted code would pass the probs check (probs and
    # means can drift independently near the cutoff).
    out += [(f"{name}<n_{mode}>", abs(m - lv["mean_n"][mode]))
            for mode, m in enumerate(d.get("mean_n", []))]
    return out


if __name__ == "__main__":
    sys.exit(run(
        fixture=FIXTURE, generator=GENERATOR, candidates=CANDIDATE_PYTHONS,
        oracle="piquasso", recipe_fields=("ops", "prep"),
        compare_case=compare_case, tol=TOL, label="loss",
    ))
