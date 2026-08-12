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

# Auto-detected CPU dists. Anything else: grab the matching CPU build from
# https://pytorch.org/get-started/locally/ and pass LIBTORCH=...
uname_s="$(uname -s)"; uname_m="$(uname -m)"
case "$uname_s/$uname_m" in
  Darwin/arm64)
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip"
    ;;
  Linux/x86_64)
    # The cxx11-ABI shared-with-deps build is the one torch-sys links against.
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"
    ;;
  *)
    LIBTORCH_URL=""
    ;;
esac

# CUDA-enabled libtorch is opt-in: `ARIA_TCH_CUDA=1` (default is CPU, unchanged).
# The CUDA minor must be one pytorch actually publishes for 2.7.0 — cu118 /
# cu126 / cu128 — via `ARIA_TCH_CUDA_VER` (default cu128). torch-sys auto-detects
# CUDA from the dist contents (no TORCH_CUDA_VERSION needed), and the `+cuXXX`
# local tag is stripped by the build-version check exactly like `+cpu`. A CUDA
# libtorch makes `--backend tch` run on the GPU via TchBackend::cuda_or_cpu().
# NOTE: the correct cuXXX for a given box (esp. the DGX Spark / GB10) is not
# verified here — set ARIA_TCH_CUDA_VER to match the target's driver.
# WANT_LOCAL is the build-version local tag an already-present install MUST have
# for the idempotency shortcut to reuse it. Empty (default/CPU) accepts any
# matching-base dist; `cuXXX` (ARIA_TCH_CUDA=1) forces a re-provision when the
# box currently holds a `+cpu` dist — otherwise a CPU->CUDA switch silently keeps
# CPU, since the base version (2.7.0) matches either way.
TORCH_PIP_INDEX=""
WANT_LOCAL=""
if [ "${ARIA_TCH_CUDA:-0}" = "1" ]; then
  CUDA_VER="${ARIA_TCH_CUDA_VER:-cu128}"
  WANT_LOCAL="$CUDA_VER"
  case "$uname_s/$uname_m" in
    Linux/x86_64)
      LIBTORCH_URL="https://download.pytorch.org/libtorch/${CUDA_VER}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${CUDA_VER}.zip"
      echo "==> ARIA_TCH_CUDA=1: fetching CUDA libtorch ($CUDA_VER, -with-deps bundles the CUDA runtime)"
      ;;
    Linux/aarch64)
      # The default PyPI aarch64 torch wheel is CPU-only; the CUDA sbsa wheel
      # lives on the pytorch cuXXX index.
      TORCH_PIP_INDEX="--index-url https://download.pytorch.org/whl/${CUDA_VER}"
      echo "==> ARIA_TCH_CUDA=1: aarch64 CUDA wheel from the $CUDA_VER index"
      ;;
    *)
      echo "WARN: ARIA_TCH_CUDA=1 ignored on $uname_s/$uname_m (no CUDA libtorch there)" >&2
      ;;
  esac
fi

# ---- 0b. Linux/aarch64: take libtorch from the pip wheel ----
#
# pytorch.org publishes no prebuilt C++ dist for Linux/aarch64, but the pip
# `torch` wheel for this arch SHIPS the same libtorch — `lib/` and `include/`
# in the package directory — at the same version. So an ARM Linux box (Grace,
# GB10, Jetson, Ampere) does not need to build libtorch from source: install
# the pinned wheel into a throwaway venv and point LIBTORCH at it.
#
# Two wrinkles the plain layout does not have, both handled here:
#   * the wheel has no `build-version`; we write one, so the idempotency check
#     above and ci.sh's `[ -f "$LIBTORCH/build-version" ]` both work,
#   * the bundled OpenBLAS lives in a SIBLING `torch.libs/` and needs its
#     `libgfortran` from there, which the linker will not find on its own —
#     hence the extra -L and -rpath in the env file below. Without them the
#     link fails with `undefined reference to _gfortran_concat_string`.
LIBTORCH_PIP_SITE=""
if [ -z "$LIBTORCH_URL" ] && [ "$uname_s/$uname_m" = "Linux/aarch64" ] && \
   [ ! -f "$LIBTORCH_DIR/build-version" ]; then
  TORCH_VENV="${ARIA_TORCH_VENV:-$REPO_DIR/.venv-libtorch}"
  echo "==> Linux/aarch64: no C++ dist upstream; using the pip torch wheel"
  if [ ! -x "$TORCH_VENV/bin/python" ]; then
    # uv resolves aarch64 wheels directly; plain pip may try source builds.
    # $TORCH_PIP_INDEX (unquoted, may be empty) points at the CUDA sbsa index
    # under ARIA_TCH_CUDA=1; empty → the default PyPI (CPU) wheel.
    if command -v uv >/dev/null 2>&1; then
      uv venv "$TORCH_VENV"
      # shellcheck disable=SC2086
      uv pip install --python "$TORCH_VENV/bin/python" $TORCH_PIP_INDEX "torch==$LIBTORCH_VERSION"
    else
      python3 -m venv "$TORCH_VENV"
      "$TORCH_VENV/bin/pip" install -q --upgrade pip
      # shellcheck disable=SC2086
      "$TORCH_VENV/bin/pip" install -q $TORCH_PIP_INDEX "torch==$LIBTORCH_VERSION"
    fi
  fi
  # Probe the wheel's dir AND real version in one call. Capture stderr: a failed
  # `import torch` (missing deps, glibc too old) must be shown, not swallowed —
  # otherwise a half-provisioned venv loops forever with a generic message.
  probe_err="$(mktemp)"
  pip_info="$("$TORCH_VENV/bin/python" -c \
    'import torch,os;print(os.path.dirname(torch.__file__));print(torch.__version__)' \
    2>"$probe_err" || true)"
  pip_torch="$(printf '%s\n' "$pip_info" | sed -n 1p)"
  pip_ver="$(printf '%s\n' "$pip_info" | sed -n 2p)"
  if [ -z "$pip_torch" ] || [ ! -d "$pip_torch/lib" ] || [ ! -d "$pip_torch/include" ]; then
    echo "ERROR: pip torch==$LIBTORCH_VERSION did not yield a usable libtorch." >&2
    echo "  looked in: $TORCH_VENV" >&2
    [ -s "$probe_err" ] && { echo "  import torch said:" >&2; sed 's/^/    /' "$probe_err" >&2; }
    echo "  then remove the venv and retry:  rm -rf \"$TORCH_VENV\"" >&2
    rm -f "$probe_err"; exit 1
  fi
  rm -f "$probe_err"
  # A leftover or user-supplied venv can carry a different torch; the `[ ! -x
  # bin/python ]` gate above would skip the pinned install and we'd trust it
  # blindly. Verify the REAL version against the pin, and stamp what is actually
  # installed (e.g. 2.7.0+cpu / +cuXXX) rather than the pin string.
  if [ "${pip_ver%%+*}" != "$LIBTORCH_VERSION" ]; then
    echo "ERROR: $TORCH_VENV has torch $pip_ver, need $LIBTORCH_VERSION." >&2
    echo "  remove the venv and retry:  rm -rf \"$TORCH_VENV\"" >&2
    exit 1
  fi
  # Variant guard (CPU<->CUDA switch on aarch64): the venv-exists gate above
  # skips reinstall, so an existing CPU venv would be trusted under
  # ARIA_TCH_CUDA=1. Fail loudly rather than silently keep CPU; removing the venv
  # forces a reinstall from the requested cuXXX index. Not auto-removed — the
  # venv may be a user-supplied ARIA_TORCH_VENV.
  pip_local="${pip_ver#*+}"; [ "$pip_local" = "$pip_ver" ] && pip_local=""
  if [ -n "$WANT_LOCAL" ] && [ "$pip_local" != "$WANT_LOCAL" ]; then
    echo "ERROR: $TORCH_VENV has torch '$pip_ver' but ARIA_TCH_CUDA wants '$WANT_LOCAL'." >&2
    echo "  remove the venv and retry:  rm -rf \"$TORCH_VENV\"" >&2
    exit 1
  fi
  LIBTORCH_DIR="$pip_torch"
  LIBTORCH_PIP_SITE="$(dirname "$pip_torch")"
  printf '%s' "$pip_ver" > "$LIBTORCH_DIR/build-version"
  # The OpenBLAS/libgfortran the link needs live in a sibling torch.libs/ (the
  # auditwheel layout). Warn if it is absent — a CUDA sbsa wheel may stage those
  # under nvidia/*/lib instead, and the link would then fail on gfortran.
  [ -d "$LIBTORCH_PIP_SITE/torch.libs" ] || \
    echo "WARN: $LIBTORCH_PIP_SITE/torch.libs not found — link may fail on _gfortran_concat_string" >&2
  echo "==> libtorch $pip_ver from the wheel at $LIBTORCH_DIR"
fi

echo "==> libtorch dir : $LIBTORCH_DIR"
echo "==> repo         : $REPO_DIR"

# ---- 1. install libtorch (idempotent) ----
# The CPU dist stamps build-version with a local-version tag (`2.7.0+cpu`), so
# compare only the part before `+` — otherwise the exact-match check re-downloads
# forever and the post-download gate below hard-exits on a correct install.
have_ver="$(cat "$LIBTORCH_DIR/build-version" 2>/dev/null || echo '')"
# The local-version tag of the present install ("" if none, else cpu / cuXXX).
have_local="${have_ver#*+}"; [ "$have_local" = "$have_ver" ] && have_local=""
# Reuse only when the base matches AND (no specific variant wanted, OR the
# present variant is the wanted one). WANT_LOCAL=cuXXX therefore does NOT reuse a
# `+cpu` install, so control falls to the download branch (which rm's and
# re-fetches the CUDA dist) — the CPU->CUDA switch that used to silently no-op.
if [ -n "$WANT_LOCAL" ] && [ -n "$have_ver" ] && [ "$have_local" != "$WANT_LOCAL" ]; then
  echo "==> present libtorch is '$have_ver' but '$WANT_LOCAL' requested — re-provisioning"
fi
if [ -f "$LIBTORCH_DIR/build-version" ] && \
   [ "${have_ver%%+*}" = "$LIBTORCH_VERSION" ] && \
   { [ -z "$WANT_LOCAL" ] || [ "$have_local" = "$WANT_LOCAL" ]; }; then
  echo "==> libtorch $have_ver already present — skipping download"
elif [ -n "$LIBTORCH_URL" ]; then
  echo "==> downloading libtorch $LIBTORCH_VERSION (~67 MB)..."
  tmp="$(mktemp -d)"
  curl -fL --retry 3 --max-time 600 -o "$tmp/libtorch.zip" "$LIBTORCH_URL"
  echo "==> unzipping..."
  rm -rf "$LIBTORCH_DIR"; mkdir -p "$(dirname "$LIBTORCH_DIR")"
  unzip -q "$tmp/libtorch.zip" -d "$(dirname "$LIBTORCH_DIR")"
  rm -rf "$tmp"
  got="$(cat "$LIBTORCH_DIR/build-version" 2>/dev/null || echo '?')"
  [ "${got%%+*}" = "$LIBTORCH_VERSION" ] || { echo "ERROR: got '$got', want $LIBTORCH_VERSION" >&2; exit 1; }
  echo "==> installed libtorch $got at $LIBTORCH_DIR"
elif [ "$uname_s/$uname_m" = "Linux/aarch64" ]; then
  # Reached only when a stale $LIBTORCH_DIR/build-version made the 0b pip route
  # skip (its gate is `[ ! -f build-version ]`). pytorch.org has no aarch64 C++
  # dist, so point at the pip route rather than a nonexistent download.
  echo "ERROR: $LIBTORCH_DIR holds '$have_ver' (want $LIBTORCH_VERSION${WANT_LOCAL:++$WANT_LOCAL}), and" >&2
  echo "  no aarch64 C++ dist exists upstream. Clear it so the pip-wheel route runs:" >&2
  echo "    rm -rf \"$LIBTORCH_DIR\" \"${ARIA_TORCH_VENV:-$REPO_DIR/.venv-libtorch}\"" >&2
  echo "  then re-run${WANT_LOCAL:+ with ARIA_TCH_CUDA=1 ARIA_TCH_CUDA_VER=$WANT_LOCAL}:  $0" >&2
  exit 1
else
  echo "ERROR: no auto-download URL for $uname_s/$uname_m." >&2
  echo "  Download libtorch $LIBTORCH_VERSION (CPU) from https://pytorch.org/get-started/locally/" >&2
  echo "  unzip it, then re-run with:  LIBTORCH=/path/to/libtorch $0" >&2
  exit 1
fi

# Apple clang >= 21 forbids libtorch 2.7's std::is_arithmetic specialization
# (c10/util/strong_type.h), so that diagnostic has to be demoted — but ONLY on
# macOS. GCC has no `-Winvalid-specialization`, and it does not ignore the
# unknown flag: `cc1plus: error: '-Wno-error=invalid-specialization': no option
# '-Winvalid-specialization'` is a hard error, so exporting these
# unconditionally makes torch-sys fail to build on every GCC host.
# Kept non-empty on Linux so ci.sh's `${CXXFLAGS:-<apple flags>}` fallback does
# not reintroduce them.
case "$uname_s" in
  Darwin) TCH_CXXFLAGS="-std=gnu++17 -Wno-invalid-specialization -Wno-error=invalid-specialization" ;;
  *)      TCH_CXXFLAGS="-std=gnu++17" ;;
esac

# The wheel route needs the sibling `torch.libs/` on both the link path and the
# rpath (see 0b). Empty for a normal C++ dist, so the env file is unchanged there.
PIP_LINK_FLAGS=""
if [ -n "$LIBTORCH_PIP_SITE" ]; then
  PIP_LINK_FLAGS="-L native=$LIBTORCH_PIP_SITE/torch.libs \
-C link-arg=-Wl,-rpath,$LIBTORCH_DIR/lib \
-C link-arg=-Wl,-rpath,$LIBTORCH_PIP_SITE/torch.libs"
fi

# ---- CUDA link retention ----
# torch-sys emits `-ltorch_cuda` but the linker's default --as-needed DROPS it,
# because no Rust symbol references it directly — so libtorch_cuda.so never lands
# in the binary's DT_NEEDED and tch::Cuda::is_available() is false even with a
# CUDA dist and a working driver. Force-retain it: --no-as-needed followed (in
# link order) by an explicit re-link of torch_cuda + c10_cuda. Keyed on the .so
# actually being present, so this covers the x86_64 dist AND the aarch64 wheel
# (same lib names) AND a user-supplied CUDA LIBTORCH — not just ARIA_TCH_CUDA.
CUDA_LINK_FLAGS=""
if [ -f "$LIBTORCH_DIR/lib/libtorch_cuda.so" ]; then
  CUDA_LINK_FLAGS="-C link-arg=-Wl,--no-as-needed -C link-arg=-L$LIBTORCH_DIR/lib \
-C link-arg=-ltorch_cuda -C link-arg=-lc10_cuda"
  echo "==> CUDA libtorch detected — forcing torch_cuda into the link (--no-as-needed)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "    driver: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
  else
    echo "    WARN: no nvidia-smi/driver found — the build links CUDA but is_available() will be false at runtime" >&2
  fi
  command -v nvcc >/dev/null 2>&1 && echo "    nvcc:   $(nvcc --version | tail -1)"
fi

# Everything that belongs in RUSTFLAGS, in one place.
LINK_FLAGS="$(printf '%s %s' "$PIP_LINK_FLAGS" "$CUDA_LINK_FLAGS" | sed -e 's/^ *//' -e 's/ *$//')"

# ---- 2. write the env file to source later ----
cat > "$ENV_FILE" <<EOF
# Source before building the Aria tch backend:  source ./tch-env.sh
export LIBTORCH="$LIBTORCH_DIR"
export DYLD_LIBRARY_PATH="\$LIBTORCH/lib:\${DYLD_LIBRARY_PATH:-}"  # macOS
export LD_LIBRARY_PATH="\$LIBTORCH/lib:${LIBTORCH_PIP_SITE:+$LIBTORCH_PIP_SITE/torch.libs:}\${LD_LIBRARY_PATH:-}"      # Linux
export CXXFLAGS="$TCH_CXXFLAGS"
unset LIBTORCH_USE_PYTORCH   # any value makes torch-sys hunt for a pip torch
EOF
# The RUSTFLAGS line is appended OUTSIDE the heredoc: writing shell via
# `${LINK_FLAGS:+export RUSTFLAGS=...${RUSTFLAGS:-}}` inside a heredoc mis-nests
# the braces and emits an unbalanced quote, which aborts every shell that
# sources the file. Non-empty for the aarch64 wheel and/or a CUDA dist.
if [ -n "$LINK_FLAGS" ]; then
  printf 'export RUSTFLAGS="%s ${RUSTFLAGS:-}"\n' "$LINK_FLAGS" >> "$ENV_FILE"
fi
echo "==> wrote env file: $ENV_FILE  (source it in future shells)"

# ---- 3. set the env for this script's own builds ----
export LIBTORCH="$LIBTORCH_DIR"
export DYLD_LIBRARY_PATH="$LIBTORCH/lib:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LIBTORCH_PIP_SITE:+$LIBTORCH_PIP_SITE/torch.libs:}${LD_LIBRARY_PATH:-}"
[ -n "$LINK_FLAGS" ] && export RUSTFLAGS="$LINK_FLAGS ${RUSTFLAGS:-}"
export CXXFLAGS="$TCH_CXXFLAGS"
unset LIBTORCH_USE_PYTORCH || true

# ---- 4. build ----
cd "$REPO_DIR"
echo "==> building aria-runtime --features tch ..."
cargo build -p aria-runtime --features tch

# ---- 5. verify ----
if [ "$DO_VERIFY" = 1 ]; then
  # With a CUDA dist, prove the GPU is actually reached — the numeric gate below
  # agrees with CPU whether it ran on GPU or silently fell back, so it cannot
  # catch a dropped libtorch_cuda. gpu_probe asserts is_available() + a real GPU
  # op. (Excluded crate → its own target dir, so this rebuilds torch-sys once.)
  if [ -n "$CUDA_LINK_FLAGS" ]; then
    echo "==> GPU gate: tch::Cuda::is_available() + a device op ..."
    ARIA_EXPECT_CUDA=1 cargo test --manifest-path crates/aria-backend-tch/Cargo.toml \
      --test gpu_probe -- --nocapture --test-threads=1
  fi
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
