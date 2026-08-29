#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# One side of the CUDA<->Metal bit-equality protocol (PLAN-BITEQ-CUDA-METAL.md).
#
#   tools/biteq/run_side.sh <metal|cuda> <outdir>
#
# Runs every corpus circuit three ways — GPU decompose, GPU exact, CPU f64 —
# and writes biteq-v1 artifacts into <outdir>. Build the featured binary
# FIRST, e.g.:
#   cargo build --release -p omega-cli --features metal   # Mac side
#   cargo build --release -p omega-cli --features cuda    # DGX side
# Override the binary with OMEGA_RUN=<path> if it lives elsewhere.
#
# Serialised on purpose (one run at a time): the CUDA side flaked under
# parallel test load (E7), and a measurement artifact is not the place to
# rediscover that.
set -eu

DEVICE="${1:?usage: run_side.sh <metal|cuda> <outdir>}"
OUT="${2:?usage: run_side.sh <metal|cuda> <outdir>}"
case "$DEVICE" in metal|cuda) ;; *) echo "device must be metal or cuda" >&2; exit 1;; esac

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BIN="${OMEGA_RUN:-$ROOT/target/release/omega-run}"
[ -x "$BIN" ] || { echo "no binary at $BIN — build it (see header) or set OMEGA_RUN" >&2; exit 1; }

mkdir -p "$OUT"
"$BIN" --version

for f in "$HERE"/corpus/*.qasm; do
  stem="$(basename "$f" .qasm)"
  for mode in decompose exact; do
    "$BIN" "$f" --statevector --device "$DEVICE" --multi-control "$mode" \
      --dump-state-bits "$OUT/$stem.$DEVICE.$mode.json" >/dev/null
  done
  # --device cpu is EXPLICIT so OMEGA_DEVICE cannot reroute the arbiter (H1).
  "$BIN" "$f" --statevector --device cpu \
    --dump-state-bits "$OUT/$stem.cpu.json" >/dev/null
done

echo "wrote $(ls "$OUT" | wc -l | tr -d ' ') artifacts to $OUT"
