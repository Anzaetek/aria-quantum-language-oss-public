#!/usr/bin/env bash
# Dense GPU lane (E5), akilles only. Builds both binaries (the lane and our
# arm, `--features cuda`) at a 16G cap, then measures under
# EMU_LANE_HOST_BYTES, the lane's HOST cap: --host-bytes bounds resident host
# memory only (the oracle's f64 state, the arms' staging, the .npy files);
# device memory is the device census's business, not the ledger's. A dirty
# tree is refused: the row would be void, and voiding it after the run is how
# a number gets lost.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMEGA_HOSTGATE="${OMEGA_HOSTGATE:-/tmp/akilles-hostgate.ledger}"
export OMEGA_HOSTGATE_MODE=enforce
hg="${OMEGA_HOSTGATE_BIN:?set OMEGA_HOSTGATE_BIN to the omega-hostgate binary}"
: "${EMU_GPU_CUSV_PY:?set EMU_GPU_CUSV_PY to the venv python built from requirements-compare-gpu-cusv.txt}"
: "${EMU_GPU_AER_PY:?set EMU_GPU_AER_PY to the venv python built from requirements-compare-gpu-aer.txt}"
: "${EMU_GPU_ORACLE_PY:?set EMU_GPU_ORACLE_PY to the dense lane venv python (qiskit 2.5.2, qiskit-aer 0.17.2)}"
cap="${EMU_LANE_HOST_BYTES:?set EMU_LANE_HOST_BYTES: measured peak RSS at the largest shape, with margin}"
export OMEGA_EMU_HOSTGATE_CAP="$cap"
if [ -n "$(git status --porcelain)" ]; then
  echo "dirty tree: a row from this tree would be void (§4.4)" >&2
  exit 1
fi
export OMEGA_EMU_GIT_REV="$(git rev-parse HEAD)"
export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-120}"
"$hg" run --watch --host-bytes 16G -- cargo build -p emu-compare-gpu --features cuda --release
log=$(mktemp)
set +e
"$hg" run --watch --host-bytes "$cap" -- target/release/emu-compare-gpu measure --out tools/emu_compare/gpu/rows.jsonl "$@" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e
if [ "$rc" -eq 5 ]; then
  cell=$(grep '^BEGIN ' "$log" | tail -1 | awk '{print $2}')
  row="gpu-${cell}-o-expectation-vs-custatevec"
  if [ -n "$cell" ] && ! grep -q "^ROW ${row}" "$log"; then
    target/release/emu-compare-gpu void --out tools/emu_compare/gpu/rows.jsonl \
      --row-id "$row" --reason "exceeds declared budget at this shape"
  fi
fi
rm -f "$log"
exit "$rc"
