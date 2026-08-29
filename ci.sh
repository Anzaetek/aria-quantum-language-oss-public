#!/usr/bin/env bash
# Local CI for the Aria Quantum Language. Single source of truth — no GitHub.
# Every check is numeric / pass-fail; no GUI. Run from the repo root: ./ci.sh
set -euo pipefail

# Per-example application crates (one project each under crates/apps/<name>).
APP_CRATES=(-p aria-app-qsvd -p aria-app-qft -p aria-app-vqe-ansatz \
  -p aria-app-grover3 -p aria-app-bernstein-vazirani -p aria-app-deutsch-jozsa \
  -p aria-app-swap-test -p aria-app-teleport -p aria-app-qaoa-maxcut \
  -p aria-app-qml-classifier -p aria-app-qml-tune -p aria-app-butterfly-qnn -p aria-app-jl-sketch-digits \
  -p aria-app-spectra \
  -p aria-app-qos -p aria-app-circulant -p aria-app-cqs -p aria-app-noise \
  -p aria-app-bell -p aria-app-ghz -p aria-app-superdense -p aria-app-simon -p aria-app-qpe \
  -p aria-app-qsp -p aria-app-forward \
  -p aria-app-qec-grover -p aria-app-qec-qft -p aria-app-qec-qpe -p aria-app-qec-memory)
ARIA_CRATES=(-p aria-core -p aria-runtime -p aria-cli -p aria-verify-core \
  -p aria-qec -p aria-verify -p aria-tune "${APP_CRATES[@]}")
# Crates we keep rustfmt-clean (aria-core is ported verbatim — left as-is).
FMT_CRATES=(-p aria-runtime -p aria-cli -p aria-verify-core -p aria-qec -p aria-verify \
  -p aria-tune "${APP_CRATES[@]}")
# Pure-Rust omega crates the default Aria build links against.
#
# This list is a HAND-MAINTAINED GATE, and it had holes. `omega-backend-pauli`
# was absent, so only its `reset_channel` target ever ran (invoked by name
# below) and its whole remaining suite -- including the creg-keying regression
# added with the N-way matrix -- never executed in CI. Same for
# `omega-backend-photonics` and `omega-backend-cv`: full suites, never run.
#
# The general problem is worth stating because it will recur: a crate is
# covered here only if someone remembered to type it. `cargo test --workspace`
# would have no holes, but it is currently RED on a clean checkout
# (`omega-wasm-cli` reads `examples/circuits/vqe_ansatz_2q.qasm`, which has
# never been tracked in git), so switching to it is its own piece of work --
# tracked in FIXES_PLAN.md K9 rather than done silently here.
OMEGA_CORE=(-p omega-core -p omega-backend-statevector -p omega-backend-mps \
  -p omega-backend-pauliprop -p omega-parser -p omega-backend-refplugin \
  -p omega-backend-pauli -p omega-backend-photonics -p omega-backend-cv \
  -p omega-tensor)
# WASM guests loaded into omega-wasm-runtime by the application harnesses.
WASM_GUESTS=(vqe omega_app)

step() { printf '\n\033[1;34m== %s ==\033[0m\n' "$1"; }

# Stages that did NOT run, collected for the final line.
#
# An inline "(skipping ...)" scrolls away in a 3000-line log, and the last line
# is what a human — or an agent reading the tail — actually acts on. A green
# summary that does not say what was skipped is how coverage quietly erodes:
# every run looks identical whether 12 stages ran or 6.
SKIPPED_STAGES=()
skipped() { SKIPPED_STAGES+=("$1"); echo "  (skipping $1)"; }

step "1/9  Format check (Aria crates)"
cargo fmt "${FMT_CRATES[@]}" -- --check

step "2/9  Clippy -D warnings (WHOLE WORKSPACE, all targets)"
# `--workspace --all-targets`, NOT a typed crate list, and NOT lib targets only.
#
# What the old `cargo clippy "${ARIA_CRATES[@]}" -- -D warnings` did and did not
# cover, MEASURED rather than assumed (a first draft of this change asserted the
# opposite and was disproved by canary):
#
#   * It DID lint the omega backends. `cargo clippy -p X` sets
#     RUSTC_WORKSPACE_WRAPPER, which runs clippy-driver over every workspace
#     member in X's dependency graph with `-D warnings` applied to each.
#     `aria-runtime` pulls in `omega-backend-pauliprop`, so that crate — and
#     omega-core, omega-parser, statevector, mps, pauli, photonics — were
#     already gated. A dropped `Result` in pauliprop's lib DOES fail the old
#     stage 2; verified by appending one and watching it exit 101.
#
#   * It did NOT lint any TEST target, anywhere. 9 of the 13 diagnostics this
#     change fixed were in test files.
#
#   * It did NOT reach the 18 workspace members outside that dependency
#     closure: omega-cli, omega-server, omega-ffi, omega-client, omega-bridges,
#     omega-tensor, omega-backend-cv, omega-backend-refplugin,
#     omega-plugin-conformance, omega-wasm-cli, omega-xcheck, and the seven
#     GPU/OpenCL backends. The remaining 4 diagnostics lived there, including a
#     CLI flag that had been accepted and silently discarded for its whole life.
#
# Cost: 24.5 s from a cold target dir, measured. `--all-targets` includes
# `--benches`, and the eight benches/ declare `harness = false`, so stage 5's
# `cargo test --workspace` does not build them — this is genuinely new compile
# surface, not a re-run.
#
# STILL NOT COVERED, so nobody reads "workspace" as "everything": the aria-py
# bindings (separate cargo project, see its stage below), crates/aria-backend-tch
# and examples/wasm-guests/* (workspace `exclude`). Those are hand-gated, which
# is the same shape of hole this change closes here.
cargo clippy --workspace --all-targets -- -D warnings

step "3/9  Build pure-Rust omega backends"
cargo build "${OMEGA_CORE[@]}"

step "4/9  Build Aria crates"
cargo build "${ARIA_CRATES[@]}"

step "5/9  Test the WHOLE WORKSPACE (numeric gates)"
# `--workspace`, NOT a typed crate list.
#
# This used to be `cargo test "${ARIA_CRATES[@]}"` + `cargo test
# "${OMEGA_CORE[@]}"`, and the list had holes that hid real work: the
# creg-keying regression in omega-backend-pauli was inert because only that
# crate's `reset_channel` target was named, and omega-backend-photonics /
# omega-backend-cv / omega-tensor had full suites that never ran. A crate was
# covered only if someone remembered to type it.
#
# `--workspace` has no holes by construction. It was RED until 2026-08-12 --
# omega-wasm-cli reads four fixtures under examples/circuits/ that had never
# been tracked in git, so 8 tests failed on every clean checkout (FIXES_PLAN.md
# K9) -- which is why the typed list survived as long as it did. With those
# fixtures restored the workspace is green: 227 targets, 0 failed.
#
# The crate arrays above are still used for `cargo build`, fmt and clippy,
# where a curated set is the point rather than a gap.
#
# ARIA_TEST_RELEASE=1 runs this in release for iteration speed, and it is NOT
# equivalent. Release drops every `debug_assert` in the tree — 34 of them — and
# several are load-bearing: `creg_to_u64`'s set-bit-above-64 check,
# `counts_from_u64`'s width check, and the `<< 64` guard in
# `condition_satisfied`. `wide_creg_is_not_an_overflow.rs` says so in its own
# header: "These tests must run in a debug build to mean anything ...
# `cargo test --release` passes on the broken code."
#
# So it registers as a skip. A faster run that checks less must SAY it checked
# less, or the speed is bought by quietly lowering the bar.
if [ "${ARIA_TEST_RELEASE:-0}" = "1" ]; then
  cargo test --workspace --release
  skipped "34 debug_assert checks — ARIA_TEST_RELEASE=1 built without them \
(see wide_creg_is_not_an_overflow.rs; a debug run is required to gate)"
else
  cargo test --workspace
fi

# --- Reset-channel regression gates (every CPU backend) ---------------------
# `reset q` is the non-unitary CHANNEL rho -> |0><0|_q (x) Tr_q(rho): the qubit
# is discarded and any entanglement it had is DESTROYED, not transferred. Three
# backends each shipped a DIFFERENT wrong version of it — statevector and MPS
# folded the amplitudes ([[1,1],[0,0]]), the stabilizer post-selected on
# outcome 0, and pauliprop/tch/aria-verify skipped it entirely — and each is
# wrong in a different basis, so every pairwise "backends agree" check landed
# on one where two of them happened to coincide. Run these as a NAMED stage so
# a regression is obvious instead of buried in a 100-test summary. Expected
# values are pinned to Qiskit Aer's exact DensityMatrix (qiskit 2.4.1 /
# aer 0.17.2); see the module docs in tests/reset_channel.rs.
cargo test -p omega-backend-statevector --test reset_channel
cargo test -p omega-backend-mps --test reset_channel
cargo test -p omega-backend-pauli --test reset_channel
# pauliprop (and the tch plugin / aria-verify reference sim) cannot represent
# the channel at all, so their contract is a clean refusal rather than a value.
cargo test -p omega-backend-pauliprop --test reset_refused
echo "  OK: Reset channel == Aer on statevector + MPS + stabilizer"

step "6/9  Numeric smoke against the built 'aria' binary"
ARIA=$(cargo build -p aria-cli --message-format=json 2>/dev/null \
  | sed -n 's/.*"executable":"\([^"]*aria\)".*/\1/p' | head -1)
[ -x "$ARIA" ] || ARIA="target/debug/aria"
# Bell <Z0 Z1> must be exactly 1.0.
zz=$("$ARIA" run examples/aria/bell.aria --circuit Bell --expectation "Z0 Z1")
echo "  $zz"
case "$zz" in
  *"= 1.000000000000") echo "  OK: Bell <Z0 Z1> = 1" ;;
  *) echo "  FAIL: expected <Z0 Z1> = 1, got: $zz"; exit 1 ;;
esac

# `aria tune` smoke: a seeded TPE study over the fixed qml_tune template must
# beat chance on the synthetic set and dump one CSV row per trial.
TUNE_DATA=$(mktemp "${TMPDIR:-/tmp}/aria_tune_data.XXXXXX")
python3 - "$TUNE_DATA" <<'PYEOF'
import sys
st = 20260729
def nxt():
    global st
    st = (st + 0x9E3779B97F4A7C15) % (1 << 64)
    z = st
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) % (1 << 64)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) % (1 << 64)
    return ((z ^ (z >> 31)) >> 11) / float(1 << 53)
w = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, -0.05]
with open(sys.argv[1], "w") as f:
    for _ in range(80):
        r = [2 * nxt() - 1 for _ in range(8)]
        s = sum(a * b for a, b in zip(r, w))
        f.write(",".join(f"{v:.6f}" for v in r + [1.0 if s >= 0 else 0.0]) + "\n")
PYEOF
TUNE_CSV=$(mktemp "${TMPDIR:-/tmp}/aria_tune_csv.XXXXXX")
TUNE_OUT=$("$ARIA" tune examples/aria/qml_tune.aria --circuit QmlTune \
  --observable Z0 --data "$TUNE_DATA" \
  --space "n=4..8:2,L=1..3,lr=log:1e-3..3e-1,opt=gd|adam" \
  --trials 8 --steps 12 --seed 7 --csv "$TUNE_CSV" 2>/dev/null)
TUNE_BEST=$(echo "$TUNE_OUT" | sed -n 's/^best_score : \(.*\)$/\1/p')
TUNE_ROWS=$(( $(wc -l < "$TUNE_CSV") - 1 ))
echo "  aria tune best_score=$TUNE_BEST csv_rows=$TUNE_ROWS"
if awk -v b="$TUNE_BEST" 'BEGIN{exit !(b >= 0.70)}' && [ "$TUNE_ROWS" -eq 8 ]; then
  echo "  OK: aria tune best_score >= 0.70 and one CSV row per trial"
else
  echo "  FAIL: aria tune best_score=$TUNE_BEST (floor 0.70), csv_rows=$TUNE_ROWS (want 8)"
  rm -f "$TUNE_DATA" "$TUNE_CSV"; exit 1
fi
rm -f "$TUNE_DATA" "$TUNE_CSV"

step "6b/9  Backend plugin ABI: load the reference cdylib and run through it"
# Builds the reference plugin, drops it in a dir, and runs a Bell circuit
# through omega-run's plugin loader. Exercises the full FFI round-trip
# (ABI-version handshake, vtable, execute, free_result) that the in-process
# unit tests can't reach. All-CPU, headless.
OMEGA_RUN=$(cargo build -p omega-cli --message-format=json 2>/dev/null \
  | sed -n 's/.*"executable":"\([^"]*omega-run\)".*/\1/p' | head -1)
[ -x "$OMEGA_RUN" ] || OMEGA_RUN="target/debug/omega-run"
cargo build -p omega-backend-refplugin >/dev/null 2>&1
PLUGIN_DIR=$(mktemp -d)
# The cdylib file extension is platform-specific (.dylib on macOS, .so on Linux,
# .dll on Windows). Copy only the shared library — NOT cargo's `.d` dep-info
# file, which shares the basename and would fail to dlopen.
for ext in dylib so dll; do
  cp "target/debug/libomega_backend_refplugin.$ext" "$PLUGIN_DIR"/ 2>/dev/null || true
done
"$ARIA" export examples/aria/bell.aria --circuit Bell --qasm > "$PLUGIN_DIR/bell.qasm"
plugin_out=$("$OMEGA_RUN" "$PLUGIN_DIR/bell.qasm" --backend refplugin \
  --backend-dir "$PLUGIN_DIR" --shots 512 2>/dev/null)
echo "$plugin_out" | grep -qE '\|00>|\|11>' \
  && ! echo "$plugin_out" | grep -qE '\|01>|\|10>' \
  && echo "  OK: refplugin ran Bell to correlated counts" \
  || { echo "  FAIL: refplugin output unexpected:"; echo "$plugin_out"; exit 1; }
# Conformance kit: the corpus (bell/ghz3/uniform/rotation) vs the statevector
# oracle. Exit 0 iff every case is within tolerance.
CONF=$(cargo build -p omega-plugin-conformance --message-format=json 2>/dev/null \
  | sed -n 's/.*"executable":"\([^"]*omega-plugin-conformance\)".*/\1/p' | head -1)
[ -x "$CONF" ] || CONF="target/debug/omega-plugin-conformance"
# Pick the shared library by extension (never cargo's `.d` dep-info file).
# A plain `ls a b c` would exit non-zero for the absent extensions and, under
# `set -euo pipefail`, abort the script — so test each candidate instead.
REFPLUGIN_DYLIB=""
for ext in dylib so dll; do
  cand="$PLUGIN_DIR/libomega_backend_refplugin.$ext"
  [ -f "$cand" ] && REFPLUGIN_DYLIB="$cand" && break
done
if "$CONF" "$REFPLUGIN_DYLIB"; then
  echo "  OK: refplugin passed the conformance corpus"
else
  echo "  FAIL: refplugin failed conformance"; exit 1
fi
rm -rf "$PLUGIN_DIR"

step "6c/9  C embedder: link the .dylib from OUTSIDE the workspace"
# The acceptance test for embedding as a native shared object. Deliberately a C
# program, not a Rust test: a Rust test in this workspace proves the linker
# works, and proves nothing about the CONTRACT an embedder depends on — that
# `include/omega.h` compiles under a C compiler, that ownership rules hold, and
# that every error path behaves. There was no `.c` file in the tree at all
# before this, so none of that had ever been exercised from outside.
#
# `cc` is present on every platform this repo builds on, so this is not
# optional and does not skip.
cargo build -p omega-ffi --release >/dev/null 2>&1
FFI_LIB_DIR="target/release"
# A per-run directory, not a fixed path: a fixed-path fixture is what made
# `flags_refuse_rather_than_panic` race ~1 run in 6 until it was fixed.
EMBED_DIR=$(mktemp -d)
if cc -I include examples/c-embed/embed.c -o "$EMBED_DIR/embed" \
     -L "$FFI_LIB_DIR" -lomega_ffi -lm; then
  echo "  header compiles under a C compiler"
else
  echo "  FAIL: include/omega.h does not compile as C, or the dylib will not link"
  rm -rf "$EMBED_DIR"; exit 1
fi
if DYLD_LIBRARY_PATH="$FFI_LIB_DIR" LD_LIBRARY_PATH="$FFI_LIB_DIR" \
     "$EMBED_DIR/embed"; then
  echo "  OK: C embedder passed every check"
else
  echo "  FAIL: C embedder reported failures"; rm -rf "$EMBED_DIR"; exit 1
fi
rm -rf "$EMBED_DIR"

step "7/9  Build WASM guests (wasm32-wasip1) for the application harnesses"
if rustup target list --installed 2>/dev/null | grep -q wasm32-wasip1; then
  for g in "${WASM_GUESTS[@]}"; do
    echo "  building examples/wasm-guests/$g"
    ( cd "examples/wasm-guests/$g" && cargo build --target wasm32-wasip1 --release >/dev/null )
  done
  echo "  OK: ${#WASM_GUESTS[@]} guests built"
else
  echo "  NOTE: wasm32-wasip1 not installed (rustup target add wasm32-wasip1)."
  echo "        aria-verify will fall back to the native transport below."
fi

if [ "${ARIA_DEV:-0}" = "1" ]; then
  # ITERATION ONLY. This stage runs every shipped example through the WASM
  # runtime against its classical oracle, and it is ~33 of the ~40 minutes a
  # full run takes — while having no bearing at all on a parser, an emitter or a
  # C ABI. Skipping it turns the edit/verify loop from 40 minutes into about 8.
  #
  # It registers through `skipped()` so the final summary NAMES it, exactly as
  # CUDA and Lean are named. That is the whole point: a fast run must not be
  # able to look like a gate run. The tch stage is the cautionary tale — it once
  # printed SKIP inline and then vanished from the summary, so a run with no
  # libtorch ended "All CI stages that ran passed" listing only Metal/OpenCL/Lean.
  #
  # Never set this in automation, and never report a run with it set as green.
  echo
  skipped "Application harnesses (aria-verify all) — ARIA_DEV=1, ITERATION ONLY"
else
step "8/9  Application harnesses: quantum vs classical (aria-verify all)"
# Runs every shipped example through the omega WASM runtime (in-process) and
# asserts each matches its pure-Rust classical oracle within tolerance.
# (For local iteration only: `ARIA_QML_QUICK=1 ./ci.sh` skips the minutes-long
# QML search/training harnesses with a printed notice. CI runs the full set —
# do not export the flag in automation.)
# `all` skips the DEEP harnesses (e.g. spectra_noise, ~15 min of exact noisy
# simulation) by default; set ARIA_DEEP=1 to fold them in, or run them by name.
cargo run -q -p aria-verify -- all
if [ "${ARIA_DEEP:-0}" = "1" ]; then
  step "8b/9  Deep harnesses (ARIA_DEEP=1)"
  ARIA_DEEP=1 cargo run -q -p aria-verify -- all
fi
fi

step "9/9  Socket transport (omega-server over HTTP) — best effort"
# Sends the same Aria package to a live omega-server and cross-checks counts.
# Skipped (not failed) if the server can't start, so CI stays green offline.
if [ "${ARIA_SKIP_SOCKET:-0}" = "1" ]; then
  echo "  skipped (ARIA_SKIP_SOCKET=1)"
else
  TOK=$(mktemp)
  DB=$(mktemp -u)
  OMEGA_PORT=8899 OMEGA_DB_PATH="$DB" \
    cargo run -q -p omega-server -- --auth bearer-only --save-token-to "$TOK" \
    >/tmp/omega-server-ci.log 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  ok=0
  for _ in $(seq 1 60); do [ -s "$TOK" ] && { ok=1; break; }; sleep 1; done
  if [ "$ok" = "1" ]; then
    cargo run -q -p aria-verify --features remote -- \
      socket --url http://127.0.0.1:8899 --token "$(cat "$TOK")"
    echo "  OK: socket transport verified against omega-server"
    # The P5 template routes, end to end through the Python client. Reuses the
    # server already running above rather than starting a second one.
    #
    # Stdlib only (`urllib.request`), so this needs no venv and no install —
    # which is why it can be a plain `python3` here. It caught a real defect a
    # mock could not have: the client encoded `OmegaParam` in the
    # externally-tagged form and the server, which uses `#[serde(untagged)]`,
    # answered 422.
    python3 tools/omega_client/live_test.py \
      --url http://127.0.0.1:8899 --token "$(cat "$TOK")"
    echo "  OK: template routes verified through the client"
  else
    echo "  SKIP: omega-server did not come up in time (see /tmp/omega-server-ci.log)"
  fi
  kill $SRV 2>/dev/null || true
  trap - EXIT
fi

# libtorch (tch) backend — ON by default. If libtorch isn't configured this
# stage FETCHES it rather than telling you to go install it: the pinned CPU dist
# is one download away and tools/setup-libtorch.sh already knows how.
# Resolution order:
#   1. an existing $LIBTORCH
#   2. ./tch-env.sh (written by a previous setup run)
#   3. tools/setup-libtorch.sh --no-verify (downloads ~67 MB, idempotent — it
#      reuses an existing install, so this is a one-time cost per machine)
# `ARIA_TCH=0 ./ci.sh` skips the stage.
if [ "${ARIA_TCH:-1}" = "1" ]; then
  step "+   libtorch (tch) backend"
  # Source tch-env.sh when LIBTORCH is unset OR already points at the same dir it
  # configured — otherwise a user who exports LIBTORCH loses the RUSTFLAGS it
  # carries (the --no-as-needed/torch_cuda retention and the aarch64 torch.libs
  # rpath), and the CUDA link silently drops libtorch_cuda. A LIBTORCH pointing
  # somewhere else is left untouched.
  if [ -f ./tch-env.sh ]; then
    env_libtorch="$(sed -n 's/^export LIBTORCH="\(.*\)"$/\1/p' ./tch-env.sh)"
    if [ -z "${LIBTORCH:-}" ] || [ "${LIBTORCH:-}" = "$env_libtorch" ]; then
      # shellcheck disable=SC1091
      . ./tch-env.sh
    fi
  fi
  if [ -z "${LIBTORCH:-}" ] || [ ! -f "${LIBTORCH}/build-version" ]; then
    echo "  libtorch not configured — fetching the pinned CPU dist..."
    if tools/setup-libtorch.sh --no-verify; then
      # shellcheck disable=SC1091
      . ./tch-env.sh
    else
      echo "  SKIP: could not install libtorch automatically on this platform."
      echo "        Grab the 2.7.0 CPU dist per INSTALL_LIBTORCH.md and re-run"
      echo "        with LIBTORCH=/path/to/libtorch."
    fi
  fi
  # `tch-env.sh` EXPORTS DYLD_LIBRARY_PATH / LD_LIBRARY_PATH, and everything
  # sourcing it inherits those for the rest of the run. That leak is what broke
  # this script: the aria-py stage's venv python then loaded the standalone
  # libtorch 2.7.0 C++ libraries off DYLD_LIBRARY_PATH instead of the ones
  # bundled inside its own pip `torch`, and the ABI mismatch surfaced as
  #
  #   ImportError: dlopen(.../torch/_C.cpython-313-darwin.so):
  #     symbol not found in flat namespace '__PyObject_NextNotImplemented'
  #
  # which reads like a broken wheel and is not one — `import torch` succeeds in
  # that venv with a clean environment.
  #
  # The export was never doing useful work here anyway. macOS SIP strips DYLD_*
  # when exec'ing a protected binary, which is exactly why the cargo invocation
  # below passes it INLINE. A venv python is not SIP-protected, so it is the one
  # process that does inherit it — the variable is stripped where it was wanted
  # and honoured where it was not.
  #
  # LIBTORCH and RUSTFLAGS must survive (the inline invocation derives from
  # them); the loader paths must not.
  unset DYLD_LIBRARY_PATH LD_LIBRARY_PATH || true

  if [ -n "${LIBTORCH:-}" ] && [ -f "${LIBTORCH}/build-version" ]; then
    # Apple clang >= 21 rejects libtorch 2.7's std::is_arithmetic specialization
    # (c10/util/strong_type.h); demote it or torch-sys will not compile. macOS
    # ONLY: GCC has no `-Winvalid-specialization` and hard-errors on the
    # `-Wno-error=` form, so applying it everywhere breaks every GCC host.
    if [ "$(uname -s)" = "Darwin" ]; then
      export CXXFLAGS="${CXXFLAGS:--std=gnu++17 -Wno-invalid-specialization -Wno-error=invalid-specialization}"
    fi
    # Any value of LIBTORCH_USE_PYTORCH makes torch-sys hunt for a pip torch.
    unset LIBTORCH_USE_PYTORCH || true
    # macOS SIP strips DYLD_* when exec'ing a protected binary, and this script
    # runs under /bin/bash — so an EXPORTED DYLD_LIBRARY_PATH is gone before
    # cargo ever starts, and the test binary aborts with
    #   dyld: Library not loaded: @rpath/libtorch_cpu.dylib
    # Setting it INLINE on the cargo invocation hands it straight to cargo's env
    # (cargo is not a protected binary), which is the only form that survives.
    # LD_LIBRARY_PATH is the Linux equivalent and is harmless on macOS.
    #
    # tch uses a process-global RNG, so the backend tests run single-threaded.
    DYLD_LIBRARY_PATH="$LIBTORCH/lib:${DYLD_LIBRARY_PATH:-}" \
    LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}" \
      cargo test -p aria-runtime --features tch --test run_examples tch_backend \
        -- --test-threads=1
    echo "  OK: tch statevector matches CPU (libtorch $(cat "$LIBTORCH/build-version"))"
  else
    # Register the skip. Every other optional stage reports itself in the final
    # summary; this one used to print SKIP inline and then vanish from it, so a
    # run with no libtorch ended "All CI stages that ran passed" listing only
    # Metal/OpenCL/Lean — the exact erosion the summary exists to prevent.
    skipped "tch backend — no usable libtorch (see INSTALL_LIBTORCH.md)"
  fi
else
  echo
  skipped "tch backend — ARIA_TCH=0"
fi

# --- Mac GPU stages default ON -----------------------------------------------
# Every Apple Silicon Mac has a GPU, so on macOS there is nothing to opt into:
# `ARIA_METAL` and `ARIA_OPENCL` default to 1 and the stages just run. The
# reason is concrete — the Metal RBS/Reset work landed verified on a CUDA box
# with the Metal mirrors "deferred to the Mac box" (GPU_BACKEND_PLAN.md) and
# shipped with two failing Metal tests, because no Mac contributor's default
# `./ci.sh` ever ran them. Now it does.
#
# `ARIA_METAL=0 ./ci.sh` (or `ARIA_OPENCL=0`) still forces a stage off.
case "$(uname -s)" in
  Darwin)
    : "${ARIA_METAL:=1}"
    : "${ARIA_OPENCL:=1}"
    ;;
esac

# Optional: CUDA GPU backends (opt-in; needs an NVIDIA GPU + CUDA toolkit).
# GPU paths are always optional with a CPU fallback, so the default CI above
# stays green on machines without a GPU. Set ARIA_CUDA=1 on a CUDA box to
# assert the GPU statevector / MPS-SVD / pauliprop paths numerically match CPU.
if [ "${ARIA_CUDA:-0}" = "1" ]; then
  step "+   Optional: CUDA GPU backends"
  # Serialise the CUDA tests. STAGE-SCOPED deliberately: exporting this would
  # serialise every CPU stage above too, which is both slow and would mask an
  # unrelated CPU-side race.
  #
  # WHAT THIS IS AND IS NOT. It is a workaround, not a diagnosis. The received
  # explanation — "CudaStatevectorBackend is neither Send nor Sync by
  # construction" — does not survive contact with the code: !Send/!Sync is a
  # compile-time property that PREVENTS cross-thread sharing, so it cannot
  # itself segfault a harness where each #[test] builds its own backend, and
  # forward_graph.rs already captures with CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
  # precisely so other threads' default-stream work is not swept up.
  #
  # What IS true: ForwardGraph::capture is reachable only from #[cfg(test)]
  # code, so the capture/execute interaction is test-only and serialising the
  # tests is a legitimate fix for the symptom.
  #
  # MEASURED 2026-08-19 on a GB10, unserialised. The cause is still not
  # identified, but it is no longer unobserved — `cargo test --lib` fails
  # 4/10 runs with, on the same run:
  #   forward_graph_replay_matches_naive_forward
  #     end_capture: CUDA_ERROR_STREAM_CAPTURE_INVALIDATED
  #     ("operation failed due to a previous error during capture")
  #   reset_channel_matches_aer_ground_truth
  #     curand fill_with_uniform: CURAND_STATUS_LAUNCH_FAILURE
  # Thread sweep: 1, 2, 4 threads clean (0/6 each); 8 threads fails (1/6). A
  # THRESHOLD, not a two-test collision.
  #
  # The two guesses previously recorded here are both now unlikely, and are
  # kept only so nobody re-derives them: "multiple primary contexts under
  # parallel NVRTC compiles" — NVRTC runs at backend construction, while these
  # failures land at end_capture and at a curand launch, well after; and
  # "concurrent large allocations colliding with the oom_fallback test" —
  # oom_fallback.rs holds BOUNDED max-qubit allocations that test refusal
  # CLASSIFICATION, not pool exhaustion.
  #
  # STRONGEST REMAINING HYPOTHESIS, UNTESTED: the THREAD_LOCAL defence above is
  # insufficient. CU_STREAM_CAPTURE_MODE_THREAD_LOCAL stops another thread's
  # LEGACY DEFAULT-STREAM work being swept into a capture; it does NOT protect
  # a stream under capture from concurrent submission to THAT SAME stream. And
  # 132f8fa established the enabling fact: "every call goes through a single
  # Arc<CudaStream>". Confirming this needs instrumenting whether each backend
  # instance owns a distinct stream — not done, so do not cite it as known.
  #
  # A test-hygiene race on the process-global READ_STATE_CALL_COUNT was found
  # and fixed separately (see qml_no_host_syncs.rs). It was NOT this, it was in
  # a different binary, and fixing it does not make this line removable.
  # Do not close that question on the strength of this line.
  export RUST_TEST_THREADS=1
  # Full statevector-CUDA unit suite: apply_2q / adjoint / execute, the
  # deterministic mid-circuit Reset ≡ CPU gate, and the odd-Y Pauli-expectation
  # regression (pauli_expectation now uses the correct (-i)^|Y| prefactor).
  # Mirrors the Metal stage, which already runs its full statevector suite.
  cargo test -p omega-backend-statevector-cuda --features cuda
  # MPS cuSOLVER gesvdj ≡ CPU Jacobi SVD (native f64: reconstruction + GPU-taken).
  cargo test -p omega-backend-mps-cuda --features cuda
  # PauliProp GPU branch expansion ≡ CPU branch (exact + max_freq, incl. budget).
  cargo test -p omega-backend-pauliprop-cuda --features cuda
  # Statevector GPU ≡ CPU statevector (f32 kernels, tol 1e-5), and the WIRED
  # `--backend mps` GPU-SVD path ≡ exact CPU statevector (native f64, tol 1e-10).
  # (cargo test takes one positional filter, so run the two gates separately.)
  cargo test -p aria-runtime --features cuda --test run_examples gpu_cuda_agrees_with_sim_on_qft
  cargo test -p aria-runtime --features cuda --test run_examples gpu_mps_cuda_agrees_with_sim
  # RBS (Givens) statevector forward ≡ CPU (f32, tol 1e-5) AND the RBS adjoint
  # gradient ≡ CPU adjoint (tol 1e-4). The `rbs` filter runs both gates.
  cargo test -p aria-runtime --features cuda --test run_examples rbs
  # The statevector-CUDA suite above includes the Reset-channel gates:
  # `reset_matches_cpu` (GPU vs CPU shot distributions + both refusing the
  # ill-defined analytic case) and `reset_channel_matches_aer_ground_truth`
  # (the three discriminating circuits pinned to Aer). CUDA samples the Born
  # outcome ON DEVICE via the fused Pauli-expectation reduction.
  # The SERVER's cuda feature. Added because it is a new CUDA surface and
  # nothing above reaches it: every stage so far builds backend crates or the
  # CLI, and `cargo test --workspace` builds omega-server WITHOUT features. A
  # feature-gated path that no CI command compiles is this repository's most
  # repeated defect — it is how the Outcome migration survived in four backends
  # at once, and how the CUDA statevector shipped with two casts that had never
  # seen a compiler.
  #
  # Both halves matter: `--features cuda` proves the CUDA dispatch, the
  # availability memo and the MPS/PauliProp hooks compile and pass; the default
  # build proves the feature is genuinely optional.
  cargo test -p omega-server --features cuda
  cargo clippy -p omega-server --features cuda --all-targets -- -D warnings
  echo "  OK: omega-server --features cuda (statevector dispatch, MPS gesvdj + pauliprop hooks)"

  # Clippy the FEATURED CUDA surface. Stage 2's `--workspace --all-targets`
  # cannot reach any of this: cfg'd-out code is not linted, so every line behind
  # `#[cfg(feature = "cuda")]` is invisible there no matter how wide the crate
  # list gets. Until this line existed, `omega-cli`, `aria-runtime` and the three
  # CUDA backends under `--features cuda` were linted by NOTHING.
  #
  # That is not hypothetical: `QUAD_KERNEL_MIN_QUBIT_2SLOT` needed its cfg
  # corrected to match its four use sites, and a cfg fix is checkable only by an
  # invocation that actually enables the feature. Without this it would have
  # shipped unverified.
  cargo clippy -p omega-backend-statevector-cuda -p omega-backend-mps-cuda \
    -p omega-backend-pauliprop-cuda -p omega-cli -p aria-runtime \
    --features omega-backend-statevector-cuda/cuda,omega-backend-mps-cuda/cuda,omega-backend-pauliprop-cuda/cuda,omega-cli/cuda,aria-runtime/cuda \
    --all-targets -- -D warnings
  echo "  OK: clippy --features cuda (cli, runtime, statevector/mps/pauliprop CUDA backends)"
  echo "  OK: CUDA GPU statevector + MPS(gesvdj) + pauliprop(branch) + RBS match CPU"
  echo "  OK: CUDA Reset channel (on-device sampling) == Aer"
  # Unset so nothing below this stage inherits the serialisation.
  unset RUST_TEST_THREADS
else
  echo
  skipped "CUDA backends — set ARIA_CUDA=1 on a CUDA box"
fi

# Bridge cross-checks (opt-in; needs the per-backend venvs).
#
# The omega-bridges cross-backend arms — perceval, bloqade, and now tsim/ppvm —
# assert L2 vs Qiskit under 0.0025 on a shared QASM2 corpus. NOTHING INVOKED
# THEM: ci.sh had no bridges stage at all, so those arms had only ever run by
# hand. An arm that skips silently in a CI that never calls it is not a gate,
# and a fresh checkout reported green having compared nothing — the same failure
# this script fixes one layer up for the Qiskit stage.
#
# Each arm auto-skips when its venv is absent and carries a vacuous-pass guard
# (every case Unavailable => fail, not pass), so enabling this cannot turn a
# missing toolchain into a false green.
# aria-py bindings. A SEPARATE cargo project (own Cargo.lock), so
# `cargo test --workspace` does not reach it -- the same coverage hole as the
# typed crate list, one directory over. Builds always; the python tests need a
# venv with the extension built, so they skip cleanly when it is absent.
# ---------------------------------------------------------------------------
# WHICH PYTHON THIS STAGE BUILDS AGAINST — read this before "fixing" a failure
# ---------------------------------------------------------------------------
# pyo3 links against a REAL interpreter and REFUSES to build against one newer
# than it supports. pyo3 0.23.5 (see bindings/aria-py/Cargo.lock) caps at
# CPython 3.13, so on a host whose `python3` is newer — Homebrew moved to 3.14
# — a bare `cargo build` here dies with:
#
#     error: the configured Python interpreter version (3.14) is newer than
#            PyO3's maximum supported version (3.13)
#
# That is an ENVIRONMENT mismatch, not a defect in this repo, and it is worth
# saying so loudly: on 2026-08-23 it aborted a full run AFTER every numbered
# stage had already passed, which reads like a real failure and is not one.
#
# Resolution order, first hit wins. Set nothing and this auto-detects:
#
#   1. $PYO3_PYTHON     — already exported: used verbatim and NOT version
#                         checked. You asked for it, you own it.
#   2. $ARIA_PY_PYTHON  — this repo's knob. Path to, or name of, an interpreter.
#   3. bindings/aria-py/.venv/bin/python, then ./.venv/bin/python
#   4. python3.13 … python3.10 from PATH, newest first
#   5. plain `python3`, but ONLY if it is <= 3.13
#
# If none yields a compatible interpreter the stage SKIPS with a named remedy
# instead of failing: it is an optional `+` stage, and an absent toolchain must
# not read as a defect. It still lands in SKIPPED_STAGES, so the final summary
# says out loud that it did not run — silence here is the thing to avoid.
#
# DELIBERATELY NOT USED: PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1. It suppresses
# the check and builds against the stable ABI anyway, trading a loud
# environment error for an untested build. Wrong direction for a CI script.
#
# TO RAISE THE CAP: bump pyo3 in bindings/aria-py/Cargo.toml, then update
# PYO3_MAX_MINOR here. The two must agree or this stage lies about what it
# accepts.
PYO3_MAX_MINOR=13

# Minor version of interpreter $1, or empty if it will not run.
py_minor() { "$1" -c 'import sys; print(sys.version_info[1])' 2>/dev/null; }
# Can pyo3 build against $1?
py_ok() {
  [ -n "${1:-}" ] || return 1
  local m
  m="$(py_minor "$1")" || return 1
  [ -n "$m" ] && [ "$m" -le "$PYO3_MAX_MINOR" ]
}

# An interpreter named EXPLICITLY (either knob) is a promise, not a hint. If it
# cannot be used this stage STOPS — it does not quietly auto-detect a different
# one. Falling back would build against a python the caller did not choose and
# then report success, which is the "refuse rather than substitute" rule this
# repo already applies to its emitters, one layer out.
ARIA_PY_INTERP=""; ARIA_PY_WHY=""; ARIA_PY_EXPLICIT=0
if [ -n "${PYO3_PYTHON:-}" ]; then
  # Verbatim and NOT version-checked: PYO3_PYTHON is pyo3's own knob, so an
  # operator setting it outranks our cap and owns the outcome.
  ARIA_PY_INTERP="$PYO3_PYTHON"; ARIA_PY_WHY="\$PYO3_PYTHON (exported; not version-checked)"
  ARIA_PY_EXPLICIT=1
elif [ -n "${ARIA_PY_PYTHON:-}" ]; then
  if ! py_ok "$ARIA_PY_PYTHON"; then
    echo "  FAIL: ARIA_PY_PYTHON='$ARIA_PY_PYTHON' is not a usable CPython <= 3.$PYO3_MAX_MINOR"
    echo "        (got: 3.$(py_minor "$ARIA_PY_PYTHON" || echo '<would not run>'))."
    echo "        Refusing to auto-detect past an interpreter you named."
    exit 1
  fi
  ARIA_PY_INTERP="$ARIA_PY_PYTHON"; ARIA_PY_WHY="\$ARIA_PY_PYTHON"; ARIA_PY_EXPLICIT=1
else
  for cand in bindings/aria-py/.venv/bin/python ./.venv/bin/python \
              python3.13 python3.12 python3.11 python3.10 python3; do
    if py_ok "$cand"; then
      ARIA_PY_INTERP="$cand"; ARIA_PY_WHY="auto-detected"; break
    fi
  done
fi
# pyo3 runs from inside bindings/aria-py, so a relative path would not resolve
# there. Make it absolute, or resolve a bare command name through PATH.
if [ -n "$ARIA_PY_INTERP" ]; then
  if [ -x "$ARIA_PY_INTERP" ]; then
    case "$ARIA_PY_INTERP" in /*) ;; *) ARIA_PY_INTERP="$PWD/${ARIA_PY_INTERP#./}" ;; esac
  else
    ARIA_PY_RESOLVED="$(command -v "$ARIA_PY_INTERP" || true)"
    if [ -z "$ARIA_PY_RESOLVED" ] && [ "$ARIA_PY_EXPLICIT" = 1 ]; then
      echo "  FAIL: the interpreter you named ('$ARIA_PY_INTERP') is not executable and"
      echo "        is not on PATH. Stopping rather than skipping: a knob that is set"
      echo "        and then silently ignored is worse than a stopped run."
      exit 1
    fi
    ARIA_PY_INTERP="$ARIA_PY_RESOLVED"
  fi
fi

step "+   aria-py bindings"
if [ -z "$ARIA_PY_INTERP" ]; then
  echo "  SKIP: no CPython <= 3.$PYO3_MAX_MINOR on this host (pyo3 cannot build against"
  echo "        anything newer). Found python3 = 3.$(py_minor python3 || echo '?')."
  echo "        Remedy — either is fine:"
  echo "          python3.13 -m venv bindings/aria-py/.venv     # auto-detected next run"
  echo "          ARIA_PY_PYTHON=/path/to/python3.13 ./ci.sh    # or point at one"
  skipped "aria-py bindings — no CPython <= 3.$PYO3_MAX_MINOR (see the note above this stage in ci.sh)"
else
  echo "  python: $ARIA_PY_INTERP (3.$(py_minor "$ARIA_PY_INTERP" || echo '?')) — $ARIA_PY_WHY"
  ( cd bindings/aria-py && PYO3_PYTHON="$ARIA_PY_INTERP" cargo build )
  ARIA_PY_VENV="bindings/aria-py/.venv/bin/python"
  ARIA_PY_MATURIN="bindings/aria-py/.venv/bin/maturin"
  if [ -x "$ARIA_PY_VENV" ] && "$ARIA_PY_VENV" -c "import aria_py, pytest" 2>/dev/null; then
    # REBUILD BEFORE TESTING. `cargo build` above proves the crate compiles; it
    # does NOT update the extension module the venv imports, which is whatever
    # `maturin develop` last installed. So without this the stage happily tests a
    # STALE .so and reports green for source it never ran -- the same
    # green-when-absent shape this file guards against elsewhere.
    #
    # It went unnoticed while every test here asserted numerics that rarely
    # change. `tests/test_gil_release.py` asserts a property of the BINARY (does
    # the extension release the GIL), so a stale module now shows up as a
    # baffling failure rather than a silent pass.
    if [ -x "$ARIA_PY_MATURIN" ]; then
      ( cd bindings/aria-py && .venv/bin/maturin develop --release -q )
    else
      echo "  NOTE: no maturin in the venv — the python tests below run against"
      echo "        the LAST INSTALLED extension, which may predate src/lib.rs."
    fi
    "$ARIA_PY_VENV" -m pytest bindings/aria-py/tests -q
    echo "  OK: aria-py builds and its python tests pass"
  else
    echo "  OK: aria-py builds (python tests skipped — no venv with aria_py + pytest)"
    skipped "aria-py python tests — build bindings/aria-py/.venv and \`maturin develop\`"
  fi
fi

# Python-side unit tests for the runners. These need no simulator install
# beyond a venv with pytest, and they gate two things nothing else does: the
# stdout-protocol guard (`runner_io`) and the Perceval convention pins.
#
# ci.sh ran NO python tests at all until this stage existed, so
# `tests/test_perceval_conventions.py` — written to pin the hwp/pbs/bs_rx
# conventions — had never executed in CI either. Same hand-maintained-gate
# hole as FIXES_PLAN.md K9.
# Run under EVERY venv that has pytest, not the first one found.
#
# This used to `break` at the first candidate, which is always `.venv-qiskit`
# — and that venv has no perceval, so `test_perceval_conventions.py` called
# `pytest.importorskip("perceval")` and skipped on every run this stage has
# ever had. Those are the six tests pinning the `bs_rx` / `hwp` / `pbs` matrix
# conventions, and that file's own header records the `bs_rx` mapping being
# wrong for two years with an error of 0.798-1.0 while a phase-insensitive HOM
# check called it fine. The stage said "Perceval conventions are pinned" and
# they were not: 6 passed, 1 skipped, and the skip was the whole point.
#
# `test_runner_io.py` is stdlib-only so it runs under any venv; the perceval
# venv therefore covers strictly more than the qiskit one. Running all of them
# costs a second and removes the question.
PYTEST_RAN=0
for cand in crates/omega-bridges/python/.venv-qiskit/bin/python \
            crates/omega-bridges/python/.venv-perceval/bin/python; do
  if [ -x "$cand" ] && "$cand" -c "import pytest" 2>/dev/null; then
    if [ "$PYTEST_RAN" -eq 0 ]; then
      step "+   Bridge runner python tests (protocol guard + conventions)"
    fi
    echo "  -- under $cand"
    "$cand" -m pytest crates/omega-bridges/python/tests -q
    PYTEST_RAN=$((PYTEST_RAN + 1))
  fi
done
if [ "$PYTEST_RAN" -gt 0 ]; then
  echo "  OK: stdout protocol guard holds and Perceval conventions are pinned ($PYTEST_RAN venv(s))"
else
  skipped "bridge runner python tests — no venv with pytest installed"
fi

if [ "${ARIA_BRIDGE_XCHECK:-0}" = "1" ]; then
  step "+   Bridge cross-checks (perceval / bloqade / tsim / ppvm)"
  # bridge-perceval was MISSING from this list, so both arms that call
  # `curated_fixtures()` were cfg'd out of every CI run — which is how that
  # helper kept a hard-wired path to a private corpus and a dead
  # `perceval.converters` import. A test CI cannot compile is not a gate.
  # NOTE two things this line gets wrong when it is written the obvious way.
  #
  # 1. `bridge-bloqade` was missing, so `cross_backend.rs`'s bloqade arm —
  #    `#[cfg(all(feature = "bridge-qiskit", feature = "bridge-bloqade"))]` —
  #    had NEVER been compiled by CI. That is verbatim the bug the comment
  #    above records fixing for perceval, still live one backend over. It hid a
  #    real wrong answer: bloqade's pyqrack interpreter implements `sx`/`sxdg`
  #    incorrectly, and `03_sqrt_x.qasm` came back at L2 = 1.0 against a gate
  #    of 1.3e-2.
  # 2. `--test cross_backend` builds ONLY that target, so every other
  #    feature-gated test binary in the crate is skipped — including
  #    `qiskit_expectation.rs`, whose own header calls it "the anchor for the
  #    N-way expectation lane" whose "conventions have to be pinned harder than
  #    usual". Six targets were in that position. Dropping the flag runs them.
  cargo test -p omega-bridges \
    --features bridge-qiskit,bridge-perceval,bridge-bloqade,bridge-tsim,bridge-ppvm \
    -- --nocapture
  echo "  OK: bridge arms agree with Qiskit within L2 0.0025 (or skipped with a reason)"
else
  skipped "bridge cross-checks — set ARIA_BRIDGE_XCHECK=1 with the backend venvs"
fi

# N-way counts matrix (FIXES_PLAN.md Part K step 3). One QASM2 corpus through
# every in-tree engine that can sample it, anchored on Qiskit.
#
# Distinct from the stage above: that one compares BRIDGE to bridge, this one
# compares OUR ENGINES to an independent implementation. The distinction is the
# whole point — the first defect this matrix found (MpsBackend replaying one
# trajectory for every shot, reporting a fair coin as certainty) was invisible
# to every internal comparison, because the broken backend agreed with itself
# perfectly and the correct MPS backend was never put beside it on a
# conditional circuit.
#
# The test auto-skips out loud when the Qiskit venv is absent — the anchor is
# not optional, and a lane without it degrades to our engines agreeing with
# each other. It also carries a vacuous-pass guard: zero cells compared is a
# failure, not a green.
#
# NOTE the harness-only tests in the same file (bit order, gate scaling, key
# width) need no venv and no feature, so they already run under the plain
# `cargo test --workspace` stage above. This stage adds the matrix itself.
if [ "${ARIA_NWAY:-0}" = "1" ]; then
  step "+   N-way counts matrix (statevector / mps / noisy-mps / pauli vs Qiskit)"
  cargo test -p omega-cli --features bridge-qiskit \
    --test nway_counts -- --nocapture
  # Expectation lane (Part K step 4). Separate target, separate quantity: this
  # one is ANALYTIC (no shots), so it gates at 1e-12 rather than a derived
  # sigma. K3 said this lane could not have an independent anchor without new
  # protocol work; the protocol work is the `expectation` mode in
  # qiskit_runner.py, so it does.
  cargo test -p omega-cli --features bridge-qiskit \
    --test nway_expectation -- --nocapture
  # `dropped_mass` as a BOUND, not a printed number: |E_trunc - E_exact| must
  # be <= the reported budget, swept across coeff_min. Needs the Qiskit anchor
  # for E_exact, and needs a TRUNCATED backend — PauliPropBackend::new() has
  # truncation off, so the assertions would otherwise hold vacuously.
  cargo test -p omega-cli --features bridge-qiskit \
    --test pauliprop_truncation_bound -- --nocapture
  # ppvm's PauliSum as the SAME-ALGORITHM anchor for pauliprop. Qiskit catches
  # arithmetic errors; ppvm catches a shared misunderstanding of the method.
  cargo test -p omega-bridges --features bridge-qiskit,bridge-ppvm \
    --test ppvm_expectation -- --nocapture
  # Stim's tableau as the same-algorithm anchor for the stabilizer backend.
  # Clifford-only, EXACT (integer expectation values, no float slack).
  cargo test -p omega-bridges --features bridge-qiskit,bridge-tsim \
    --test stim_expectation -- --nocapture
  echo "  OK: counts, expectations (1e-12), truncation bound, ppvm + stim same-algorithm anchors"
else
  skipped "N-way counts matrix — set ARIA_NWAY=1 with the qiskit venv"
fi

# Qiskit differential cross-check. MANDATORY in intent: it is the only
# INDEPENDENT implementation available, and this project has already shipped a
# defect that every internal cross-backend agreement gate missed (each pair of
# backends coincided in the basis being checked).
#
# It SKIPS rather than fails when the venv is absent — a machine without it must
# be able to run CI (K13) — but the skip is LOUD. A silent skip is how a check
# quietly stops running: the run stays green and nobody notices the strongest
# evidence was never gathered. So an absent venv prints a banner and the final
# summary repeats it.
#
# RECOMMENDED, not mandatory, by decision 2026-08-18 — see the POLICY block at
# the top of OPTIONAL_TESTS.md for the cadence and the trade being accepted.
# The loud skip is what makes 'recommended' mean something rather than
# decaying into 'never'.
QISKIT_XCHECK_SKIPPED=""
if [ "${ARIA_QISKIT_XCHECK:-0}" = "1" ]; then
    echo "== Qiskit differential cross-check =="
    QK_PY="${ARIA_QISKIT_PY:-./.venv-qiskit/bin/python}"
    XCHECK_FEATS=""
    [ "$(uname -s)" = "Darwin" ] && XCHECK_FEATS="--features metal"
    if [ ! -x "$QK_PY" ]; then
        QISKIT_XCHECK_SKIPPED="no venv at $QK_PY"
    else
        cargo run -q --release -p omega-xcheck --bin omega-xcheck $XCHECK_FEATS -- 60 > /tmp/aria_xcheck.txt \
            && "$QK_PY" tools/qiskit_xcheck/compare.py /tmp/aria_xcheck.txt \
            && echo "  qiskit cross-check OK" \
            || { echo "  QISKIT CROSS-CHECK FAILED"; exit 1; }

        # QASM ROUND-TRIP: aria's NATIVE execution vs Qiskit running aria's
        # EXPORTED text. OPTIONAL_TESTS.md gap #5.
        #
        # A separate stage from the comparison above, because the reference is
        # different in a way that matters. `compare.py` builds the circuit in
        # Qiskit from a gate list, so both sides construct independently. This
        # one puts our EXPORTER in the path and keeps aria's in-memory execution
        # as the reference — so a lossy export shows up, where feeding the same
        # text to both engines would let them agree on what the export dropped.
        #
        # It is also the only export check that can see measurement and classical
        # conditions at all: `qasm2_dialect.py` compares `Operator(loaded)`, and
        # `Operator()` raises on both, so every circuit of this shape is excluded
        # from it. That is precisely the shape of the 2026-08-08 defect.
        cargo run -q --release -p aria-runtime --bin qasm_roundtrip_xcheck -- 12 \
            > /tmp/aria_qasm_roundtrip.txt \
            && "$QK_PY" tools/qiskit_xcheck/compare_qasm_roundtrip.py /tmp/aria_qasm_roundtrip.txt \
            && echo "  qasm round-trip OK" \
            || { echo "  QASM ROUND-TRIP FAILED"; exit 1; }

        # PauliProp against Qiskit. The harness above drives STATEVECTOR-shaped
        # backends and compares probabilities; PauliProp returns expectations and
        # never entered it, so until now its only checks were our own CPU
        # statevector, the same-algorithm ppvm anchor, and GPU-vs-CPU — all
        # sharing this project's conventions. Two implementations that share a
        # convention agree on a shared mistake, which is exactly how the Reset
        # channel shipped wrong in three backends at once.
        #
        # The corpus is deliberately NON-Clifford (rz/rx/ry/t at angles off the
        # π/2 lattice): the Clifford-only corpus above never reaches `branch`,
        # the tree-expansion step that IS the engine, so a Clifford cross-check
        # of PauliProp would be green and vacuous.
        cargo run -q --release -p omega-xcheck --bin pauliprop_xcheck -- 40 \
                > /tmp/aria_pp_xcheck.txt \
            && "$QK_PY" tools/qiskit_xcheck/compare_pauliprop.py /tmp/aria_pp_xcheck.txt \
            && echo "  pauliprop vs qiskit OK" \
            || { echo "  PAULIPROP QISKIT CROSS-CHECK FAILED"; exit 1; }

        # Is our QASM2 the same DIALECT qiskit speaks? The check above compares
        # simulation results; this one compares the interchange format, which is
        # a different failure surface — an export qiskit cannot load, or loads as
        # a different circuit, never reaches a simulator to disagree.
        #
        # Both halves are needed. The Rust test writes the corpus and asserts
        # what needs no python; the python half asserts the claim that does —
        # that qiskit's Operator for OUR file equals its Operator for its OWN
        # native gate. Loading proves nothing on its own: emitting `crz` where
        # `cp` was meant loads perfectly and is a different circuit (measured,
        # max|Δ| = 3.482e-01).
        cargo test -q -p aria-core --test qasm2_qiskit_dialect \
            && "$QK_PY" tools/qiskit_xcheck/qasm2_dialect.py target/qasm2_dialect_corpus.txt \
            || { echo "  QASM2 DIALECT CHECK FAILED"; exit 1; }

        # And the same question for OpenQASM 3, which the 2.0 check cannot
        # answer. The QASM2 lane validates against a reader that accepts names
        # absent from `stdgates.inc` (`cu1` among them), so it would happily
        # certify a QASM3 file a strict consumer rejects. This one parses with
        # `qiskit.qasm3.loads` and compares operators. Self-skips with a note
        # when `qiskit-qasm3-import` is not installed, rather than failing.
        cargo test -q -p aria-core --test emitters_refuse_rather_than_substitute \
            && "$QK_PY" tools/qiskit_xcheck/qasm3_dialect.py target/qasm3_dialect_corpus.txt \
            || { echo "  QASM3 DIALECT CHECK FAILED"; exit 1; }
    fi
else
    QISKIT_XCHECK_SKIPPED="ARIA_QISKIT_XCHECK not set"
fi

if [ -n "$QISKIT_XCHECK_SKIPPED" ]; then
    printf '\n\033[1;33m'
    echo "!! NOTE — the RECOMMENDED Qiskit differential cross-check did not run"
    echo "!!   reason: $QISKIT_XCHECK_SKIPPED"
    echo "!!"
    echo "!! Not required on every run, BY DECISION (2026-08-18, OPTIONAL_TESTS.md):"
    echo "!! it costs a venv and minutes, and the edit/verify loop pays that for no"
    echo "!! signal on most changes. Run it PERIODICALLY instead — before anything"
    echo "!! touching gate semantics, a parser/emitter lane or a backend's numerics,"
    echo "!! and at least once before a release or hand-off."
    echo "!!"
    echo "!! Worth the cadence because it is the only INDEPENDENT implementation"
    echo "!! checked against. Two Aria backends agreeing may only mean they share a"
    echo "!! convention, and this project has already shipped a defect that every"
    echo "!! internal agreement gate missed. A green run without it is WEAKER"
    echo "!! EVIDENCE than it looks — accepted knowingly, not enforced."
    echo "!!"
    echo "!!   python3 -m venv .venv-qiskit"
    echo "!!   ./.venv-qiskit/bin/pip install qiskit qiskit-aer"
    echo "!!   ARIA_QISKIT_XCHECK=1 ARIA_NWAY=1 ./ci.sh"
    printf '\033[0m\n'
fi

# Metal GPU backends. Default ON on macOS (see the Mac block above), skipped
# elsewhere — off-Mac there is no Metal to test, so the default CI stays green.
# Asserts the GPU statevector / MPS-θ-contraction / pauliprop-branch / RBS paths
# numerically match CPU. Uses --release: the Metal QML/statevector suites are
# slow in a debug build.
if [ "${ARIA_METAL:-0}" = "1" ]; then
  step "+   Optional: Metal GPU backends"
  # PauliProp GPU branch: integer symplectic on the GPU, f64 coefficients on the
  # CPU (Apple has no native f64) → exact (exact + max_freq, incl. dropped-mass).
  cargo test --release -p omega-backend-pauliprop-metal --features metal
  # MPS Metal θ-contraction ≡ CPU (f32 kernel; SVD stays on CPU per GPU_BACKEND_PLAN).
  cargo test --release -p omega-backend-mps-metal --features metal
  # Statevector Metal full suite (f32 kernels + adjoint/QML training parity).
  cargo test --release -p omega-backend-statevector-metal --features metal
  # WIRED end-to-end: `--backend gpu` (statevector, tol 1e-6), `--backend mps`
  # (θ-contraction, f32), and `--backend pauliprop` (branch, exact) all match the
  # CPU. (cargo test takes one positional filter, so run each gate separately.)
  cargo test --release -p aria-runtime --features metal --test run_examples gpu_metal_agrees_with_sim_on_qft
  cargo test --release -p aria-runtime --features metal --test run_examples gpu_mps_metal_agrees_with_sim
  cargo test --release -p aria-runtime --features metal --test run_examples gpu_pauliprop_metal_agrees_with_sim
  # RBS (Givens) statevector forward ≡ CPU (f32, tol 1e-6) AND the RBS adjoint
  # gradient ≡ CPU adjoint (tol 1e-5). The `rbs` filter runs both gates.
  cargo test --release -p aria-runtime --features metal --test run_examples rbs
  # Featured clippy for the Metal half — the mirror of the CUDA arm above, and
  # necessary for the same reason: `--workspace --all-targets` in stage 2 CANNOT
  # lint cfg'd-out code, so everything behind `feature = "metal"` was linted by
  # nothing. Measured when the widened gate landed: stage 2 was clean, and
  # `--features metal` immediately found two `int_plus_one` errors in
  # `statevector-metal` that no run had ever seen.
  #
  # This is the half a CUDA host can never check, exactly as the CUDA arm is the
  # half this host can never check. Both must exist or the widened gate is only
  # widened on whichever machine happens to run it.
  cargo clippy --workspace --all-targets --features metal -- -D warnings
  echo "  OK: clippy --features metal (whole workspace, all targets)"
  echo "  OK: Metal GPU statevector + MPS(θ-contraction) + pauliprop(branch) + RBS match CPU"
else
  echo
  skipped "Metal backends — not macOS"
fi

# OpenCL GPU statevector backend. Default ON on macOS (Apple ships
# OpenCL.framework, so there is nothing to opt into — see the Mac block above);
# elsewhere opt-in via ARIA_OPENCL=1, since a Linux/Windows host may have no ICD
# and no device. Asserts the OpenCL kernels numerically match CPU.
#
# Off-Mac, linking also needs the ICD *loader* dev symlink `libOpenCL.so` (not
# just the runtime `libOpenCL.so.1`): `cl-sys` emits `-lOpenCL`, which the linker
# can only resolve against the `.so`. Most hosts get it from `ocl-icd-opencl-dev`.
# A CUDA-only box (nvidia.icd present, no ocl-icd-opencl-dev) has the loader at
# $CUDA/targets/<arch>/lib/libOpenCL.so but not on the default link path — point
# the linker at it, e.g.:
#   RUSTFLAGS="-L native=/usr/local/cuda/targets/x86_64-linux/lib" ARIA_OPENCL=1 ./ci.sh
if [ "${ARIA_OPENCL:-0}" = "1" ]; then
  step "+   Optional: OpenCL GPU statevector backend"
  # Full OpenCL suite: the per-kernel smokes (apply_1q / apply_diagonal /
  # apply_diagonal_product / inner_product), buffer-pool semantics, shot-
  # sampling TVD, the adjoint-vs-CPU gradient parity, and pauli_expectation —
  # including `pauli_expectation_matches_host_on_random_14q`, whose X·Y·Z
  # string has an ODD Y count and so pins the (-i)^|Y| prefactor in
  # `pauli_masks`. That gate is why this stage exists: the odd-Y sign bug was
  # fixed in all three GPU backends, but OpenCL had no CI stage to prove it.
  #
  # ARIA_OPENCL_REQUIRE_DEVICE=1 turns the crate's "no device → silently
  # return" test guard into a hard failure (tests/device_present.rs). Without
  # it this stage would report OK on a host where not one kernel ever ran.
  ARIA_OPENCL_REQUIRE_DEVICE=1 \
    cargo test -p omega-backend-statevector-opencl --features opencl
  echo "  OK: OpenCL statevector kernels + adjoint + pauli(odd-Y) match CPU"
else
  echo
  skipped "OpenCL backend — set ARIA_OPENCL=1 on a host with an OpenCL ICD"
fi

# Optional: Lean 4 proof tree (opt-in; needs a warm mathlib cache via elan/lake).
# Makes `aria export --lean` self-contained and ships the proven circulant
# correspondence + noise-deviation theorems.
if [ "${ARIA_LEAN:-0}" = "1" ]; then
  step "+   Optional: Lean 4 proof tree (mathlib)"
  # Assert a `#print axioms` batch is BOTH sorry-free AND actually ran.
  #
  # `grep -q sorryAx` alone cannot tell "sorry-free" from "never checked": if a
  # theorem is renamed, an import breaks, or the module is dropped, `lean
  # --stdin` prints an error, emits no `sorryAx`, and the check reports OK. All
  # nine batches below were that shape, so a rename would have silently retired
  # the proof obligation while CI stayed green. Verified by hand: feeding a
  # deliberately misspelled theorem name produced `error(lean.unknownIdentifier)`
  # and the old guard still concluded "OK: sorry-free".
  #
  # This is the same vacuous-pass the bridges stage already guards against
  # ("every case Unavailable => fail, not pass"). The knowledge existed in this
  # file and had not reached these checks.
  #
  # So also require the EXACT number of `depends on axioms` reports.
  # $1 = label, $2 = expected theorem count, $3 = the Lean source.
  #
  # TWO BASH TRAPS, both hit while writing this, both invisible to a
  # passing run and caught only by a deliberate negative test:
  #
  # 1. The source is passed as $3, NOT piped in. A function on the right of a
  #    `|` runs in a SUBSHELL, where its `exit 1` kills only that subshell —
  #    the stage printed FAIL and still exited 0.
  # 2. `out=$(...)` needs `|| true`. lean exits 1 on exactly the errors this
  #    guard exists to catch (measured: unknown constant, broken import and a
  #    failed proof all exit 1), so under `set -euo pipefail` the assignment
  #    aborted the whole script AT THAT LINE — before the count check, and with
  #    every byte of lean's output swallowed by the substitution. CI still went
  #    red, so the gate was safe, but it died silently and the diagnostic below
  #    was unreachable dead code for its primary trigger.
  lean_axioms() {
    local label="$1" want="$2" src="$3" out got
    # `|| true`: see trap 2 above. The count check is what turns a lean failure
    # into a REPORTED failure rather than a bare abort.
    out=$(cd proofs/lean4 && printf '%b' "$src" | lake env lean --stdin 2>&1) || true
    if printf '%s\n' "$out" | grep -q sorryAx; then
      echo "  FAIL: $label depends on sorryAx"; exit 1
    fi
    # `grep -c` exits 1 on zero matches; `|| true` keeps `set -e`/pipefail from
    # turning "checked nothing" into a silent stage abort instead of a FAIL.
    got=$(printf '%s\n' "$out" | grep -c 'depends on axioms' || true)
    if [ "$got" -ne "$want" ]; then
      echo "  FAIL: $label — expected $want axiom report(s), got $got."
      echo "        A theorem was renamed/removed, an import broke, lean errored,"
      echo "        or a theorem became fully axiom-free (that prints 'does not"
      echo "        depend on any axioms', which deliberately does not match)."
      printf '%s\n' "$out" | head -20
      exit 1
    fi
  }
  if ! command -v lake >/dev/null 2>&1; then
    echo "  SKIP: 'lake' not found (install via elan)"
  else
    if ( cd proofs/lean4 && lake exe cache get >/dev/null 2>&1 && \
         lake build QuantumProofs >/dev/null 2>&1 ); then
      echo "  OK: QuantumProofs proof tree builds (Bell prep + circulant correspondence + noise)"
    else
      echo "  FAIL: Lean proof tree failed to build"; exit 1
    fi
    # Enforce sorry-free on the shipped correspondence theorems (`lake build`
    # does not error on `sorry`).
    lean_axioms "circulant theorems" 5 \
      'import QuantumProofs.CirculantSolveGeneral\nopen QuantumProofs.CirculantSolveGeneral\n#print axioms dft_diagonalizes_circulant\n#print axioms qft_diagonalizes_circulant\n#print axioms circulant_solve_operator\n#print axioms circulant_solve_noise_deviation\n#print axioms qft_diagonalizes_solve_error\n'
    echo "  OK: general-n circulant diagonalize + solve op = C⁻¹ + noisy-solve deviation axiom-clean (sorry-free)"
    # Noise-channel library: the formal backing for the `noise` app's laws (A)/(B).
    # The CPTP Kraus maps + closed-form fidelity/relaxation/coherence laws must be sorry-free.
    lean_axioms "noise-channel theorems" 10 \
      'import QuantumProofs.Noise\nopen QuantumProofs.Noise\n#print axioms KrausMap.apply_isDensity\n#print axioms depolarizing_apply\n#print axioms amplitudeDamping_expZ\n#print axioms phaseDamping_coherence\n#print axioms depolarizing_iterate_fidelity\n#print axioms circulant_cyclicshift_fidelity\n#print axioms kraus_tensor_complete\n#print axioms threeQubitDepolarizing_fidelity\n#print axioms globalDepol_fidelity\n#print axioms globalDepol_circulant_fidelity\n'
    echo "  OK: noise channels (depolarizing + amp/phase damping + depth-G + tensor width + global entangled) CPTP + laws sorry-free"
    # Quantum-linear-algebra capstones (HHL + QSVT inversion): the formal
    # backing for the `hhl`/`qsvt_invert` Aria examples and the certified
    # Neumann 1/x inverter. Must be sorry-free.
    lean_axioms "HHL/QSVT theorems" 7 \
      'import QuantumProofs.HHL\nimport QuantumProofs.QSVT\nopen QuantumProofs.HHL QuantumProofs.QSVT\n#print axioms hhl_solves_system\n#print axioms controlled_inv_rotation\n#print axioms hhl_success_prob_lower\n#print axioms inv_poly_approx\n#print axioms qsvt_invert_correct\n#print axioms qsvt_residual_exact\n#print axioms qsvt_solves_system_approx\n'
    echo "  OK: HHL (solves A·x=C·b + RY rotation + success bound) + QSVT (1/x poly + A⁻¹ approx + exact residual) sorry-free"
    # QSP fundamental theorem: the formal backing for QSVT angle-finding. Forward
    # (any phase list implements a degree/parity polynomial transform) AND converse
    # (every admissible polynomial pair is realized by some phase list, up to a
    # global phase — the SL₂ obstruction forces the qualifier). Must be sorry-free.
    lean_axioms "QSP theorems" 4 \
      'import QuantumProofs.QSP\nopen QuantumProofs.QSP\n#print axioms qsp_implements_poly\n#print axioms qsp_implements_poly_degree\n#print axioms qsp_gram_diag\n#print axioms qsp_converse\n'
    echo "  OK: QSP fundamental theorem (forward implements-poly + degree + Gram + converse up-to-global-phase) sorry-free"
    # Gate-model export obligation: the `aria export --gate-model` artefact for
    # Bell must build sorry-free (closed by QuantumProofs.BellPrep theorems).
    if ( cd proofs/lean4 && lake build QuantumProofs.Generated.GateModel.Bell_Spec >/dev/null 2>&1 ); then
      lean_axioms "gate-model Bell_Spec" 2 \
        'import QuantumProofs.Generated.GateModel.Bell_Spec\nopen Exported.GateModel.Bell\n#print axioms bell_correct\n#print axioms circuit_unitary\n'
      echo "  OK: gate-model export (aria export --gate-model) builds sorry-free"
    else
      echo "  FAIL: gate-model Bell_Spec.lean failed to build"; exit 1
    fi
    # GHZ — the second recognized circuit, closed by GHZPrep.ghz_state_prep_correct
    # (state-prep) and GHZPrep.ghz_unitary (the `@assert unitary`, now closed
    # sorry-free via compositional unitarity — formerly dropped).
    if ( cd proofs/lean4 && lake build QuantumProofs.Generated.GateModel.GHZ_Spec >/dev/null 2>&1 ); then
      lean_axioms "gate-model GHZ_Spec" 2 \
        'import QuantumProofs.Generated.GateModel.GHZ_Spec\nopen Exported.GateModel.GHZ\n#print axioms ghz_correct\n#print axioms circuit_unitary\n'
      echo "  OK: gate-model GHZ export builds sorry-free (state-prep + unitary)"
    else
      echo "  FAIL: gate-model GHZ_Spec.lean failed to build"; exit 1
    fi
    # QFT: the EQUIV obligation (denote = dft_matrix n) — recognized circuit is
    # the exporter-lowered QFT(n), closed sorry-free by QFTExport.qftLowered_correct.
    if ( cd proofs/lean4 && lake build QuantumProofs.Generated.GateModel.QFT_Spec >/dev/null 2>&1 ); then
      lean_axioms "gate-model QFT_Spec" 1 \
        'import QuantumProofs.Generated.GateModel.QFT_Spec\nopen Exported.GateModel.QFT\n#print axioms qft_equals_dft\n'
      echo "  OK: gate-model QFT export builds sorry-free (equiv denote=DFT)"
    else
      echo "  FAIL: gate-model QFT_Spec.lean failed to build"; exit 1
    fi
    # QPE: the FAITHFULNESS obligation — measuring the counting register of the
    # actual (n+1)-qubit QPE circuit yields the phase m with probability 1,
    # closed sorry-free by QPEFaithful.qpe_faithful (no matrix-adjoint caveat).
    if ( cd proofs/lean4 && lake build QuantumProofs.Generated.GateModel.QPE_Spec >/dev/null 2>&1 ); then
      lean_axioms "gate-model QPE_Spec" 1 \
        'import QuantumProofs.Generated.GateModel.QPE_Spec\nopen Exported.GateModel.QPE\n#print axioms qpe_recovers_phase\n'
      echo "  OK: gate-model QPE export builds sorry-free (faithful counting-reg measure)"
    else
      echo "  FAIL: gate-model QPE_Spec.lean failed to build"; exit 1
    fi
    # Grover: the MEASUREMENT obligation — running the actual 3-qubit Grover
    # circuit (uniform H^⊗3 then `optimal_iterations 3` oracle·diffusion rounds)
    # on the uniform state and measuring yields the marked item |111⟩ with
    # probability ≥ 1 − 1/8, closed sorry-free by
    # GroverCircuit.grover_gate_optimal_success.
    if ( cd proofs/lean4 && lake build QuantumProofs.Generated.GateModel.Grover_Spec >/dev/null 2>&1 ); then
      lean_axioms "gate-model Grover_Spec" 1 \
        'import QuantumProofs.Generated.GateModel.Grover_Spec\nopen Exported.GateModel.Grover\n#print axioms grover_finds_marked\n'
      echo "  OK: gate-model Grover export builds sorry-free (measurement ≥ 1−1/N)"
    else
      echo "  FAIL: gate-model Grover_Spec.lean failed to build"; exit 1
    fi
  fi
else
  echo
  skipped "Lean proof tree — set ARIA_LEAN=1 (needs lake + mathlib cache)"
fi

# QEC encoded-demo cross-check against Qiskit + stim + PyMatching.
#
# POLICY (decided 2026-08-18): **MANDATORY on any platform that can run it.**
# Deliberately a stronger rule than the Qiskit cross-check above, which is
# RECOMMENDED — see the POLICY block in OPTIONAL_TESTS.md for that one. The two
# differ because this stage checks a DECODER against an independent
# implementation of the same algorithm, and a decoder that is subtly wrong still
# decodes: it returns corrections, the run stays green, and only a differential
# comparison shows the logical error rate is off.
#
# "Can run it" is not a matter of opinion, so it is measured rather than
# assumed. Three outcomes, kept distinct because collapsing them is how a
# mandatory check quietly becomes optional:
#
#   RAN        — the venv imports qiskit AND pymatching. Failure is fatal.
#   INCAPABLE  — the venv exists but pymatching does not import. That is the
#                real state of at least one host in this project: pymatching's
#                C++ extension does not build wheels on aarch64 Linux there. A
#                clean skip, because the platform genuinely cannot, and K13 says
#                such a machine must still be able to run CI.
#   NOT RUN    — no venv, so capability is UNKNOWN. Reported LOUDLY as a
#                mandatory check that did not run, because "I did not build the
#                venv" must never read the same as "this box cannot".
#
# The distinction is the whole point of the policy. Without it, the machine that
# CAN run the check and the machine that CANNOT produce identical output, and
# the mandatory one silently degrades to the optional one.
QEC_PY="tools/qec_cross_check/.venv/bin/python"
if [ "${ARIA_QEC_XCHECK:-0}" = "1" ]; then
  step "+   QEC demo cross-check vs Qiskit (MANDATORY where capable)"
  if command -v python3 >/dev/null 2>&1; then
    bash tools/qec_cross_check/run.sh
    echo "  OK: encoded grover/qft/qpe match Qiskit (+ stim); surface decoder matches PyMatching"
  else
    echo "  SKIP: python3 not found (needed to build the qiskit venv)"
    skipped "QEC cross-check — no python3 on this host (INCAPABLE)"
  fi
elif [ -x "$QEC_PY" ] && ! "$QEC_PY" -c "import pymatching" >/dev/null 2>&1; then
  # Venv built, pymatching absent from it: the platform tried and could not.
  echo
  skipped "QEC cross-check — pymatching does not import in $QEC_PY (INCAPABLE platform, clean skip)"
else
  # Either the venv imports pymatching (capable, and the operator chose not to
  # run it) or there is no venv at all (unknown). Both are a mandatory check
  # that did not run, and both get the banner.
  echo
  if [ -x "$QEC_PY" ]; then
    QEC_WHY="this host CAN run it — the venv imports pymatching"
  else
    QEC_WHY="no venv at $QEC_PY, so capability is unverified"
  fi
  printf '\n\033[1;33m'
  echo "!! WARNING — the MANDATORY QEC cross-check did NOT run"
  echo "!!   $QEC_WHY"
  echo "!!"
  echo "!! MANDATORY on any platform that can run it (decision 2026-08-18)."
  echo "!! It is the only differential check on the DECODER: a decoder that is"
  echo "!! subtly wrong still returns corrections and still looks green. Only"
  echo "!! comparison against PyMatching shows the logical error rate is off."
  echo "!!"
  echo "!!   ARIA_QEC_XCHECK=1 ./ci.sh"
  printf '\033[0m\n'
  skipped "QEC cross-check (MANDATORY where capable) — $QEC_WHY"
fi

# Optional: CV backend vs piquasso, LIVE.
#
# The committed fixture (tools/cv_cross_check/piquasso_fixture.jsonl) is already
# compared on every `cargo test -p omega-backend-cv`, with no Python needed.
# This stage covers the one failure that check structurally cannot see: a
# fixture regenerated to match a change in OUR conventions would still be green,
# and the independence that justifies using piquasso at all would be gone.
if [ "${ARIA_CV_XCHECK:-0}" = "1" ]; then
  step "+   Optional: CV backend vs piquasso (fixture drift)"
  CV_PY=""
  for cand in ./.venv-piquasso/bin/python ./tools/cv_cross_check/.venv/bin/python; do
    [ -x "$cand" ] && CV_PY="$cand" && break
  done
  if [ -n "$CV_PY" ]; then
    "$CV_PY" tools/cv_cross_check/verify_fixture.py
    # The multi-mode fixture is a SEPARATE artifact with its own generator, and
    # it must drift-check too. Without this line the beamsplitter corpus would
    # be protected only by the committed file, which cannot catch a fixture
    # regenerated to match a change in our own code.
    "$CV_PY" tools/cv_cross_check/verify_multimode_fixture.py
    # And the loss fixture, third sibling — density-matrix simulator, own
    # generator, same reasoning.
    "$CV_PY" tools/cv_cross_check/verify_loss_fixture.py
  else
    echo "  SKIP: no piquasso venv (see PREREQUISITES.md)"
  fi
else
  skipped "CV/piquasso drift check — set ARIA_CV_XCHECK=1 with the piquasso venv"
fi

if [ -n "${QISKIT_XCHECK_SKIPPED:-}" ]; then
    SKIPPED_STAGES+=("Qiskit cross-check (MANDATORY) — $QISKIT_XCHECK_SKIPPED")
fi

if [ ${#SKIPPED_STAGES[@]} -eq 0 ]; then
    printf '\n\033[1;32mAll CI stages passed — nothing skipped.\033[0m\n'
else
    printf '\n\033[1;32mAll CI stages that ran passed\033[0m'
    printf '\033[1;33m — but %d stage(s) did NOT run:\033[0m\n' "${#SKIPPED_STAGES[@]}"
    for sk in "${SKIPPED_STAGES[@]}"; do
        case "$sk" in
            *MANDATORY*) printf '\033[1;31m  !! %s\033[0m\n' "$sk" ;;
            *)           printf '\033[1;33m   - %s\033[0m\n' "$sk" ;;
        esac
    done
    printf '\033[1;33m   A green run is only as strong as what actually ran.\033[0m\n'
fi
