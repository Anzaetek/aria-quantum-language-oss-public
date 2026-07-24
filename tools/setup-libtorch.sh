#!/usr/bin/env bash
# setup-libtorch.sh — install libtorch and build/verify the optional `tch`
# backend. The default Aria build needs none of this (it is pure Rust); this is
# only for exercising `--features tch` / `--backend tch`. See INSTALL_LIBTORCH.md.
#
# Installs the prebuilt libtorch that matches the `tch` crate pin, exports the
# environment, and works around the Apple-clang-≥21 build failure (libtorch
# 2.7's c10/util/strong_type.h specializes std::is_arithmetic, which newer
# libc++ forbids) via -Wno-invalid-specialization.
#
# Usage:
#   tools/setup-libtorch.sh                 # install (if needed) + build + verify
#   tools/setup-libtorch.sh --no-verify     # install + build only
#   LIBTORCH=/path/to/libtorch tools/setup-libtorch.sh   # reuse an existing dist
#
# For later builds in a fresh shell, source the generated env file instead of
# re-running:  source ./tch-env.sh  &&  cargo build -p aria-runtime --features tch
set -euo pipefail

LIBTORCH_VERSION="2.7.0"   # keep in sync with INSTALL_LIBTORCH.md and the tch pin
LIBTORCH_DIR="${LIBTORCH:-$HOME/libtorch}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_DIR/tch-env.sh"
DO_VERIFY=1
[ "${1:-}" = "--no-verify" ] && DO_VERIFY=0

# Only the macOS arm64 URL is auto-detected; other platforms: grab the matching
# CPU dist from https://pytorch.org/get-started/locally/ and pass LIBTORCH=...
uname_s="$(uname -s)"; uname_m="$(uname -m)"
if [ "$uname_s" = "Darwin" ] && [ "$uname_m" = "arm64" ]; then
  LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip"
else
  LIBTORCH_URL=""
fi

echo "==> libtorch dir : $LIBTORCH_DIR"
echo "==> repo         : $REPO_DIR"

# ---- 1. install libtorch (idempotent) ----
if [ -f "$LIBTORCH_DIR/build-version" ] && \
   [ "$(cat "$LIBTORCH_DIR/build-version")" = "$LIBTORCH_VERSION" ]; then
  echo "==> libtorch $LIBTORCH_VERSION already present — skipping download"
elif [ -n "$LIBTORCH_URL" ]; then
  echo "==> downloading libtorch $LIBTORCH_VERSION (~67 MB)..."
  tmp="$(mktemp -d)"
  curl -fL --retry 3 --max-time 600 -o "$tmp/libtorch.zip" "$LIBTORCH_URL"
  echo "==> unzipping..."
  rm -rf "$LIBTORCH_DIR"; mkdir -p "$(dirname "$LIBTORCH_DIR")"
  unzip -q "$tmp/libtorch.zip" -d "$(dirname "$LIBTORCH_DIR")"
  rm -rf "$tmp"
  got="$(cat "$LIBTORCH_DIR/build-version" 2>/dev/null || echo '?')"
  [ "$got" = "$LIBTORCH_VERSION" ] || { echo "ERROR: got '$got', want $LIBTORCH_VERSION" >&2; exit 1; }
  echo "==> installed libtorch $got at $LIBTORCH_DIR"
else
  echo "ERROR: no auto-download URL for $uname_s/$uname_m." >&2
  echo "  Download libtorch $LIBTORCH_VERSION (CPU) from https://pytorch.org/get-started/locally/" >&2
  echo "  unzip it, then re-run with:  LIBTORCH=/path/to/libtorch $0" >&2
  exit 1
fi

# ---- 2. write the env file to source later ----
cat > "$ENV_FILE" <<EOF
# Source before building the Aria tch backend:  source ./tch-env.sh
export LIBTORCH="$LIBTORCH_DIR"
export DYLD_LIBRARY_PATH="\$LIBTORCH/lib:\${DYLD_LIBRARY_PATH:-}"  # macOS
export LD_LIBRARY_PATH="\$LIBTORCH/lib:\${LD_LIBRARY_PATH:-}"      # Linux
# Apple clang >=21 forbids libtorch 2.7's std::is_arithmetic specialization;
# demote that diagnostic so torch-sys compiles (see INSTALL_LIBTORCH.md).
export CXXFLAGS="-std=gnu++17 -Wno-invalid-specialization -Wno-error=invalid-specialization"
unset LIBTORCH_USE_PYTORCH   # any value makes torch-sys hunt for a pip torch
EOF
echo "==> wrote env file: $ENV_FILE  (source it in future shells)"

# ---- 3. set the env for this script's own builds ----
export LIBTORCH="$LIBTORCH_DIR"
export DYLD_LIBRARY_PATH="$LIBTORCH/lib:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
export CXXFLAGS="-std=gnu++17 -Wno-invalid-specialization -Wno-error=invalid-specialization"
unset LIBTORCH_USE_PYTORCH || true

# ---- 4. build ----
cd "$REPO_DIR"
echo "==> building aria-runtime --features tch ..."
cargo build -p aria-runtime --features tch

# ---- 5. verify ----
if [ "$DO_VERIFY" = 1 ]; then
  echo "==> numeric gate: tch backend == CPU statevector ..."
  cargo test -p aria-runtime --features tch --test run_examples tch_backend -- --test-threads=1
  echo "==> end-to-end: VQE train on tch (expect final <O> ~ -1.851199) ..."
  H2="-0.4804*I0+0.3435*Z0+-0.4347*Z1+0.5716*Z0Z1+0.0910*X0X1+0.0910*Y0Y1"
  cargo run -q -p aria-cli --features tch -- train examples/aria/vqe_ansatz.aria \
    --circuit VQEAnsatz --int n_layers=2 --observable "$H2" \
    --backend tch --steps 600 --lr 0.1 --seed 7 | grep -iE 'backend|final|improvement'
fi

echo
echo "==> DONE. Future shells:  source ./tch-env.sh  then cargo ... --features tch"
