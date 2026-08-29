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
  # LOUD, and it says which half was lost.
  #
  # This used to read "(skipping decoder cross-check — pymatching not
  # installed)" in parentheses, and then exit 0. So on a host where the install
  # had silently failed, HALF this stage stopped running and the line reporting
  # it was quieter than the lines reporting success. The decoder half is the one
  # that cannot be replaced by anything internal: a wrong decoder still returns
  # corrections and still looks green, and only PyMatching shows the logical
  # error rate is off.
  #
  # Still exit 0 rather than fail: a platform whose pymatching wheel does not
  # build must be able to run CI (K13), and ci.sh classifies that case as
  # INCAPABLE by making the same import check itself. What is fixed here is that
  # the narrowing is now impossible to read past.
  echo
  echo "  ############################################################"
  echo "  # HALF OF THIS STAGE DID NOT RUN"
  echo "  #   the encoded-algorithm cross-check ABOVE ran;"
  echo "  #   the surface-code DECODER cross-check did NOT."
  echo "  # reason: pymatching does not import for $PY"
  echo "  # This is the only differential check on the decoder. Without it a"
  echo "  # decoder that is subtly wrong still decodes and still looks green."
  echo "  ############################################################"
fi
exit "$rc"
