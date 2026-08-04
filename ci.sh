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
OMEGA_CORE=(-p omega-core -p omega-backend-statevector -p omega-backend-mps \
  -p omega-backend-pauliprop -p omega-parser -p omega-backend-refplugin)
# WASM guests loaded into omega-wasm-runtime by the application harnesses.
WASM_GUESTS=(vqe omega_app)

step() { printf '\n\033[1;34m== %s ==\033[0m\n' "$1"; }

step "1/9  Format check (Aria crates)"
cargo fmt "${FMT_CRATES[@]}" -- --check

step "2/9  Clippy -D warnings (Aria crates)"
cargo clippy "${ARIA_CRATES[@]}" -- -D warnings

step "3/9  Build pure-Rust omega backends"
cargo build "${OMEGA_CORE[@]}"

step "4/9  Build Aria crates"
cargo build "${ARIA_CRATES[@]}"

step "5/9  Test Aria crates (numeric gates)"
cargo test "${ARIA_CRATES[@]}"
# The pure-Rust omega backends carry their own numeric gates (e.g. the MPS
# SVD-unitarity + deep-circuit norm/truncation regressions). Step 3 only builds
# them, so run their tests here or those gates never execute in CI.
cargo test "${OMEGA_CORE[@]}"

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
  else
    echo "  SKIP: omega-server did not come up in time (see /tmp/omega-server-ci.log)"
  fi
  kill $SRV 2>/dev/null || true
  trap - EXIT
fi

# Optional: libtorch backend, only when LIBTORCH is configured.
if [ -n "${LIBTORCH:-}" ]; then
  step "+   Optional: libtorch (tch) backend"
  cargo test -p aria-runtime --features tch --test run_examples tch_backend
else
  echo
  echo "  (skipping tch backend — set LIBTORCH to enable; see INSTALL_LIBTORCH.md)"
fi

# Optional: CUDA GPU backends (opt-in; needs an NVIDIA GPU + CUDA toolkit).
# GPU paths are always optional with a CPU fallback, so the default CI above
# stays green on machines without a GPU. Set ARIA_CUDA=1 on a CUDA box to
# assert the GPU statevector / MPS-SVD / pauliprop paths numerically match CPU.
if [ "${ARIA_CUDA:-0}" = "1" ]; then
  step "+   Optional: CUDA GPU backends"
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
  echo "  OK: CUDA GPU statevector + MPS(gesvdj) + pauliprop(branch) + RBS match CPU"
else
  echo
  echo "  (skipping CUDA backends — set ARIA_CUDA=1 on a CUDA box to enable)"
fi

# Optional: Metal GPU backends (opt-in; needs an Apple Silicon Mac). Mirrors the
# ARIA_CUDA stage: GPU paths are always optional with a CPU fallback, so the
# default CI above stays green off-Mac. Set ARIA_METAL=1 on a Mac to assert the
# GPU statevector / MPS-θ-contraction / pauliprop-branch paths numerically match
# CPU. Uses --release: the Metal QML/statevector suites are slow in a debug build.
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
  echo "  OK: Metal GPU statevector + MPS(θ-contraction) + pauliprop(branch) + RBS match CPU"
else
  echo
  echo "  (skipping Metal backends — set ARIA_METAL=1 on an Apple Silicon Mac to enable)"
fi

# Optional: OpenCL GPU statevector backend (opt-in; needs an OpenCL ICD and a
# device — Apple's OpenCL.framework, the Intel/AMD/NVIDIA runtimes, or POCL).
# Mirrors the ARIA_CUDA / ARIA_METAL stages: the OpenCL path is optional with a
# CPU fallback, so the default CI stays green on hosts with no ICD. Set
# ARIA_OPENCL=1 to assert the OpenCL kernels numerically match CPU.
#
# Linking also needs the ICD *loader* dev symlink `libOpenCL.so` (not just the
# runtime `libOpenCL.so.1`): `cl-sys` emits `-lOpenCL`, which the linker can
# only resolve against the `.so`. Most hosts get it from `ocl-icd-opencl-dev`.
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
  echo "  (skipping OpenCL backend — set ARIA_OPENCL=1 on a host with an OpenCL ICD to enable)"
fi

# Optional: Lean 4 proof tree (opt-in; needs a warm mathlib cache via elan/lake).
# Makes `aria export --lean` self-contained and ships the proven circulant
# correspondence + noise-deviation theorems.
if [ "${ARIA_LEAN:-0}" = "1" ]; then
  step "+   Optional: Lean 4 proof tree (mathlib)"
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
    ax=$(cd proofs/lean4 && printf 'import QuantumProofs.CirculantSolveGeneral\nopen QuantumProofs.CirculantSolveGeneral\n#print axioms dft_diagonalizes_circulant\n#print axioms qft_diagonalizes_circulant\n#print axioms circulant_solve_operator\n#print axioms circulant_solve_noise_deviation\n#print axioms qft_diagonalizes_solve_error\n' \
      | lake env lean --stdin 2>&1)
    if printf '%s' "$ax" | grep -q sorryAx; then
      echo "  FAIL: circulant theorems depend on sorryAx"; exit 1
    else
      echo "  OK: general-n circulant diagonalize + solve op = C⁻¹ + noisy-solve deviation axiom-clean (sorry-free)"
    fi
    # Noise-channel library: the formal backing for the `noise` app's laws (A)/(B).
    # The CPTP Kraus maps + closed-form fidelity/relaxation/coherence laws must be sorry-free.
    nx=$(cd proofs/lean4 && printf 'import QuantumProofs.Noise\nopen QuantumProofs.Noise\n#print axioms KrausMap.apply_isDensity\n#print axioms depolarizing_apply\n#print axioms amplitudeDamping_expZ\n#print axioms phaseDamping_coherence\n#print axioms depolarizing_iterate_fidelity\n#print axioms circulant_cyclicshift_fidelity\n#print axioms kraus_tensor_complete\n#print axioms threeQubitDepolarizing_fidelity\n#print axioms globalDepol_fidelity\n#print axioms globalDepol_circulant_fidelity\n' \
      | lake env lean --stdin 2>&1)
    if printf '%s' "$nx" | grep -q sorryAx; then
      echo "  FAIL: noise-channel theorems depend on sorryAx"; exit 1
    else
      echo "  OK: noise channels (depolarizing + amp/phase damping + depth-G + tensor width + global entangled) CPTP + laws sorry-free"
    fi
    # Quantum-linear-algebra capstones (HHL + QSVT inversion): the formal
    # backing for the `hhl`/`qsvt_invert` Aria examples and the certified
    # Neumann 1/x inverter. Must be sorry-free.
    la=$(cd proofs/lean4 && printf 'import QuantumProofs.HHL\nimport QuantumProofs.QSVT\nopen QuantumProofs.HHL QuantumProofs.QSVT\n#print axioms hhl_solves_system\n#print axioms controlled_inv_rotation\n#print axioms hhl_success_prob_lower\n#print axioms inv_poly_approx\n#print axioms qsvt_invert_correct\n#print axioms qsvt_residual_exact\n#print axioms qsvt_solves_system_approx\n' \
      | lake env lean --stdin 2>&1)
    if printf '%s' "$la" | grep -q sorryAx; then
      echo "  FAIL: HHL/QSVT theorems depend on sorryAx"; exit 1
    else
      echo "  OK: HHL (solves A·x=C·b + RY rotation + success bound) + QSVT (1/x poly + A⁻¹ approx + exact residual) sorry-free"
    fi
    # QSP fundamental theorem: the formal backing for QSVT angle-finding. Forward
    # (any phase list implements a degree/parity polynomial transform) AND converse
    # (every admissible polynomial pair is realized by some phase list, up to a
    # global phase — the SL₂ obstruction forces the qualifier). Must be sorry-free.
    qs=$(cd proofs/lean4 && printf 'import QuantumProofs.QSP\nopen QuantumProofs.QSP\n#print axioms qsp_implements_poly\n#print axioms qsp_implements_poly_degree\n#print axioms qsp_gram_diag\n#print axioms qsp_converse\n' \
      | lake env lean --stdin 2>&1)
    if printf '%s' "$qs" | grep -q sorryAx; then
      echo "  FAIL: QSP theorems depend on sorryAx"; exit 1
    else
      echo "  OK: QSP fundamental theorem (forward implements-poly + degree + Gram + converse up-to-global-phase) sorry-free"
    fi
    # Gate-model export obligation: the `aria export --gate-model` artefact for
    # Bell must build sorry-free (closed by QuantumProofs.BellPrep theorems).
    if ( cd proofs/lean4 && lake build QuantumProofs.Generated.GateModel.Bell_Spec >/dev/null 2>&1 ); then
      gm=$(cd proofs/lean4 && printf 'import QuantumProofs.Generated.GateModel.Bell_Spec\nopen Exported.GateModel.Bell\n#print axioms bell_correct\n#print axioms circuit_unitary\n' \
        | lake env lean --stdin 2>&1)
      if printf '%s' "$gm" | grep -q sorryAx; then
        echo "  FAIL: gate-model Bell_Spec depends on sorryAx"; exit 1
      else
        echo "  OK: gate-model export (aria export --gate-model) builds sorry-free"
      fi
    else
      echo "  FAIL: gate-model Bell_Spec.lean failed to build"; exit 1
    fi
    # GHZ — the second recognized circuit, closed by GHZPrep.ghz_state_prep_correct
    # (state-prep) and GHZPrep.ghz_unitary (the `@assert unitary`, now closed
    # sorry-free via compositional unitarity — formerly dropped).
    if ( cd proofs/lean4 && lake build QuantumProofs.Generated.GateModel.GHZ_Spec >/dev/null 2>&1 ); then
      gz=$(cd proofs/lean4 && printf 'import QuantumProofs.Generated.GateModel.GHZ_Spec\nopen Exported.GateModel.GHZ\n#print axioms ghz_correct\n#print axioms circuit_unitary\n' \
        | lake env lean --stdin 2>&1)
      if printf '%s' "$gz" | grep -q sorryAx; then
        echo "  FAIL: gate-model GHZ_Spec depends on sorryAx"; exit 1
      else
        echo "  OK: gate-model GHZ export builds sorry-free (state-prep + unitary)"
      fi
    else
      echo "  FAIL: gate-model GHZ_Spec.lean failed to build"; exit 1
    fi
    # QFT: the EQUIV obligation (denote = dft_matrix n) — recognized circuit is
    # the exporter-lowered QFT(n), closed sorry-free by QFTExport.qftLowered_correct.
    if ( cd proofs/lean4 && lake build QuantumProofs.Generated.GateModel.QFT_Spec >/dev/null 2>&1 ); then
      qf=$(cd proofs/lean4 && printf 'import QuantumProofs.Generated.GateModel.QFT_Spec\nopen Exported.GateModel.QFT\n#print axioms qft_equals_dft\n' \
        | lake env lean --stdin 2>&1)
      if printf '%s' "$qf" | grep -q sorryAx; then
        echo "  FAIL: gate-model QFT_Spec depends on sorryAx"; exit 1
      else
        echo "  OK: gate-model QFT export builds sorry-free (equiv denote=DFT)"
      fi
    else
      echo "  FAIL: gate-model QFT_Spec.lean failed to build"; exit 1
    fi
    # QPE: the FAITHFULNESS obligation — measuring the counting register of the
    # actual (n+1)-qubit QPE circuit yields the phase m with probability 1,
    # closed sorry-free by QPEFaithful.qpe_faithful (no matrix-adjoint caveat).
    if ( cd proofs/lean4 && lake build QuantumProofs.Generated.GateModel.QPE_Spec >/dev/null 2>&1 ); then
      qp=$(cd proofs/lean4 && printf 'import QuantumProofs.Generated.GateModel.QPE_Spec\nopen Exported.GateModel.QPE\n#print axioms qpe_recovers_phase\n' \
        | lake env lean --stdin 2>&1)
      if printf '%s' "$qp" | grep -q sorryAx; then
        echo "  FAIL: gate-model QPE_Spec depends on sorryAx"; exit 1
      else
        echo "  OK: gate-model QPE export builds sorry-free (faithful counting-reg measure)"
      fi
    else
      echo "  FAIL: gate-model QPE_Spec.lean failed to build"; exit 1
    fi
    # Grover: the MEASUREMENT obligation — running the actual 3-qubit Grover
    # circuit (uniform H^⊗3 then `optimal_iterations 3` oracle·diffusion rounds)
    # on the uniform state and measuring yields the marked item |111⟩ with
    # probability ≥ 1 − 1/8, closed sorry-free by
    # GroverCircuit.grover_gate_optimal_success.
    if ( cd proofs/lean4 && lake build QuantumProofs.Generated.GateModel.Grover_Spec >/dev/null 2>&1 ); then
      gv=$(cd proofs/lean4 && printf 'import QuantumProofs.Generated.GateModel.Grover_Spec\nopen Exported.GateModel.Grover\n#print axioms grover_finds_marked\n' \
        | lake env lean --stdin 2>&1)
      if printf '%s' "$gv" | grep -q sorryAx; then
        echo "  FAIL: gate-model Grover_Spec depends on sorryAx"; exit 1
      else
        echo "  OK: gate-model Grover export builds sorry-free (measurement ≥ 1−1/N)"
      fi
    else
      echo "  FAIL: gate-model Grover_Spec.lean failed to build"; exit 1
    fi
  fi
else
  echo
  echo "  (skipping Lean proof tree — set ARIA_LEAN=1 to enable; needs lake + mathlib cache)"
fi

# Optional: QEC encoded-demo cross-check against Qiskit (opt-in; needs a Python
# venv with qiskit). Mirrors the GPU/Lean stages: default CI stays green without
# qiskit. Set ARIA_QEC_XCHECK=1 to export the aria QEC demo circuits (grover/
# qft/qpe) and assert an independent SDK (Qiskit Statevector, + stim stabilizer)
# reproduces aria's distributions exactly (aria == qiskit == analytic, ≤ 1e-9).
if [ "${ARIA_QEC_XCHECK:-0}" = "1" ]; then
  step "+   Optional: QEC demo cross-check vs Qiskit"
  if command -v python3 >/dev/null 2>&1; then
    bash tools/qec_cross_check/run.sh
    echo "  OK: encoded grover/qft/qpe match Qiskit (+ stim); surface decoder matches PyMatching"
  else
    echo "  SKIP: python3 not found (needed to build the qiskit venv)"
  fi
else
  echo
  echo "  (skipping QEC cross-check — set ARIA_QEC_XCHECK=1 with a qiskit venv to enable)"
fi

printf '\n\033[1;32mAll CI stages passed.\033[0m\n'
