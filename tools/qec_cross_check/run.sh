#!/usr/bin/env bash
# Cross-check the aria-qec encoded demos (grover/qft/qpe) against Qiskit as an
# independent reference. Builds the aria CLI, then runs check_qec.py under a
# Python (a venv, per repo policy) that has qiskit installed.
#
# Zero-config: if no QEC_PYTHON is set and no venv exists, this creates
# tools/qec_cross_check/.venv and pip-installs qiskit (+ optional qsimcirq/stim).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

# A Python with qiskit. Override with QEC_PYTHON=/path/to/python.
PY="${QEC_PYTHON:-$here/.venv/bin/python}"

if ! "$PY" -c "import qiskit" >/dev/null 2>&1; then
  if [ -z "${QEC_PYTHON:-}" ]; then
    echo "Creating venv at $here/.venv and installing qiskit (numpy)..."
    python3 -m venv "$here/.venv"
    PY="$here/.venv/bin/python"
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q qiskit numpy pymatching
    # Optional extra references (best effort; the script skips them if absent).
    "$PY" -m pip install -q stim qsimcirq cirq 2>/dev/null || true
  else
    echo "No qiskit for '$PY'. Set QEC_PYTHON to a venv with qiskit," >&2
    echo "or unset it to let this script create tools/qec_cross_check/.venv." >&2
    exit 2
  fi
fi

echo "Building aria CLI..."
( cd "$repo" && cargo build -q -p aria-cli )

echo "reference python: $("$PY" -c 'import qiskit;print("qiskit",qiskit.__version__)')"

# (1) Encoded-algorithm circuits vs Qiskit (+ stim). (2) The surface-code decoder
# vs PyMatching. Run both; fail if either does.
rc=0
"$PY" "$here/check_qec.py" || rc=$?
if "$PY" -c "import pymatching" >/dev/null 2>&1; then
  echo
  "$PY" "$here/check_decoder.py" || rc=$?
else
  echo
  echo "  (skipping decoder cross-check — pymatching not installed)"
fi
exit "$rc"
