#!/usr/bin/env bash
# Stabilizer lane. Builds the release binary at a 4G cap (a build, not a
# row) and measures at EMU_LANE_HOST_BYTES (default 2G, the lane's cap).
# A dirty tree is refused: the row would be void, and voiding it after the
# run is how a number gets lost.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMEGA_HOSTGATE="${OMEGA_HOSTGATE:-/tmp/claude-501/hostgate.ledger}"
export OMEGA_HOSTGATE_MODE=enforce
hg="${OMEGA_HOSTGATE_BIN:-$HOME/work/aria-quantum-language-oss/target/release/omega-hostgate}"
# The one place the measurement cap is written: the same value goes to
# --host-bytes and to the row's hostgate_cap, so a 40G akilles run cannot
# publish andromeda's 2G.
cap="${EMU_LANE_HOST_BYTES:-2G}"
export OMEGA_EMU_HOSTGATE_CAP="$cap"
if [ -n "$(git status --porcelain)" ]; then
  echo "dirty tree: a row from this tree would be void (§4.4)" >&2
  exit 1
fi
export OMEGA_EMU_GIT_REV="$(git rev-parse HEAD)"
"$hg" run --watch --host-bytes 4G -- cargo build -p emu-compare-stim --release
log=$(mktemp)
set +e
"$hg" run --watch --host-bytes "$cap" -- target/release/emu-compare-stim measure --out tools/emu_compare/stim/rows.jsonl 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e
if [ "$rc" -eq 5 ]; then
  row=$(grep '^BEGIN ' "$log" | tail -1 | awk '{print $2}')
  if [ -n "$row" ] && ! grep -q "^ROW ${row}" "$log" && ! grep -q "^CAP ${row}" "$log"; then
    target/release/emu-compare-stim void --out tools/emu_compare/stim/rows.jsonl \
      --row-id "$row" \
      --reason "exceeds declared budget at this shape"
  fi
fi
rm -f "$log"
exit "$rc"
