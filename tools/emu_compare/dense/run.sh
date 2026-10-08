#!/usr/bin/env bash
# Dense CPU lane (E4), akilles. Builds the release binary at a 16G cap (a
# build, not a row) and measures at EMU_LANE_HOST_BYTES (default 6G, the
# lane's cap: a 26q f64 state is 1 GiB per arm). A dirty tree is refused: the
# row would be void, and voiding it after the run is how a number gets lost.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMEGA_HOSTGATE="${OMEGA_HOSTGATE:-/tmp/akilles-hostgate.ledger}"
export OMEGA_HOSTGATE_MODE=enforce
hg="${OMEGA_HOSTGATE_BIN:?set OMEGA_HOSTGATE_BIN to the omega-hostgate binary}"
: "${EMU_DENSE_PY:?set EMU_DENSE_PY to the venv python built from tools/emu_compare/dense/requirements-compare-dense.txt}"
# The one place the measurement cap is written: the same value goes to
# --host-bytes and to the row's hostgate_cap.
cap="${EMU_LANE_HOST_BYTES:-6G}"
export OMEGA_EMU_HOSTGATE_CAP="$cap"
if [ -n "$(git status --porcelain)" ]; then
  echo "dirty tree: a row from this tree would be void (§4.4)" >&2
  exit 1
fi
export OMEGA_EMU_GIT_REV="$(git rev-parse HEAD)"
"$hg" run --watch --host-bytes 16G -- cargo build -p emu-compare-dense --release
log=$(mktemp)
set +e
"$hg" run --watch --host-bytes "$cap" -- target/release/emu-compare-dense measure --out tools/emu_compare/dense/rows.jsonl "$@" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e
if [ "$rc" -eq 5 ]; then
  cell=$(grep '^BEGIN ' "$log" | tail -1 | awk '{print $2}')
  if [ -n "$cell" ]; then
    for comp in qulacs qsim; do
      row="dense-${cell}-o-expectation-vs-${comp}"
      if ! grep -q "^ROW ${row}" "$log"; then
        target/release/emu-compare-dense void --out tools/emu_compare/dense/rows.jsonl \
          --row-id "$row" --reason "exceeds declared budget at this shape"
      fi
    done
  fi
fi
rm -f "$log"
exit "$rc"
