#!/usr/bin/env bash
# Pauliprop lane (E7), akilles. Release build at 16G (a build, not a row);
# measure at EMU_LANE_HOST_BYTES (default 8G). A dirty tree is refused.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMEGA_HOSTGATE="${OMEGA_HOSTGATE:-/tmp/akilles-hostgate.ledger}"
export OMEGA_HOSTGATE_MODE=enforce
hg="${OMEGA_HOSTGATE_BIN:?set OMEGA_HOSTGATE_BIN to the omega-hostgate binary}"
: "${EMU_PAULIPROP_PY:?set EMU_PAULIPROP_PY to the venv python built from tools/emu_compare/pauliprop/requirements-compare-pauliprop.txt}"
cap="${EMU_LANE_HOST_BYTES:-8G}"
export OMEGA_EMU_HOSTGATE_CAP="$cap"
if [ -n "$(git status --porcelain)" ]; then
  echo "dirty tree: a row from this tree would be void (§4.4)" >&2
  exit 1
fi
export OMEGA_EMU_GIT_REV="$(git rev-parse HEAD)"
"$hg" run --watch --host-bytes 16G -- cargo build -p emu-compare-pauliprop --release
"$hg" run --watch --host-bytes "$cap" -- target/release/emu-compare-pauliprop measure --out tools/emu_compare/pauliprop/rows.jsonl "$@"
