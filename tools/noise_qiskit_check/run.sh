#!/usr/bin/env bash
# Cross-check Aria's --noise against Qiskit Aer (density-matrix, exact) as ground
# truth. Builds the Aria CLIs, then runs check_noise.py under a Python that has
# qiskit-aer installed.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

# A Python with qiskit + qiskit-aer. Override with AER_PYTHON=/path/to/python.
# Defaults to a sibling venv that already has a compiled Aer.
PY="${AER_PYTHON:-/Users/renaudbechade/work/initial-tests/.venv/bin/python}"
if ! "$PY" -c "import qiskit_aer" >/dev/null 2>&1; then
  echo "No qiskit-aer for '$PY'. Set AER_PYTHON to a venv with qiskit + qiskit-aer" >&2
  echo "(e.g. python -m venv .venv && .venv/bin/pip install qiskit qiskit-aer)." >&2
  exit 2
fi

echo "Building Aria CLIs..."
( cd "$repo" && cargo build -q -p aria-cli -p omega-cli )

echo "Aer: $("$PY" -c 'import qiskit,qiskit_aer;print("qiskit",qiskit.__version__,"aer",qiskit_aer.__version__)')"
exec "$PY" "$here/check_noise.py"
