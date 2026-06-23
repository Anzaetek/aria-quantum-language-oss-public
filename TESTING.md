# TESTING — Aria Quantum Language (numeric manual)

Copy-pasteable shell. **Numbers only**: every step states an expected value and a
tolerance. No GUI, no "looks right". Run from the repo root.

Build the binary once:

```console
$ cargo build -p aria-cli
$ ARIA=target/debug/aria
```

---

## 1. Bell state — statevector is exact (tol ≤ 1e-6)

```console
$ $ARIA run examples/aria/bell.aria --circuit Bell --statevector
```

Expected exactly two non-zero amplitudes:

```
|00>  +0.707107+0.000000i
|11>  +0.707107+0.000000i
```

Check: amplitude `0.707107 = 1/√2` (±1e-6); `|01>` and `|10>` absent.

## 2. Bell — ⟨Z0 Z1⟩ = 1 (tol ≤ 1e-10)

```console
$ $ARIA run examples/aria/bell.aria --circuit Bell --expectation "Z0 Z1"
<Z0 Z1> = 1.000000000000
```

Check: value `= 1.000000000000` (±1e-10).

## 3. Bell — sampled counts balanced & correlated (tol ±5%)

```console
$ $ARIA run examples/aria/bell.aria --circuit Bell --shots 8192 --seed 7
```

Check: only `|00>` and `|11>` appear (no `|01>`/`|10>`); each probability in
`[0.45, 0.55]`. Cross-check the MPS backend agrees:

```console
$ $ARIA run examples/aria/bell.aria --circuit Bell --shots 8192 --seed 7 --backend mps
```

## 4. QFT(n=3) on |000⟩ is uniform (tol ≤ 1e-6)

```console
$ $ARIA run examples/aria/qft.aria --circuit QFT --int n=3 --statevector
```

Check: all 8 amplitudes equal `+0.353553+0.000000i` (`= 1/√8`, ±1e-6). This
exercises the controlled-phase gate (`CP → CU3(0,0,λ)`).

## 5. Controlled-phase is phase-exact (tol ≤ 1e-6)

```console
$ printf 'circuit C() {\n  qreg q[2]\n  apply X on q[0]\n  apply X on q[1]\n  apply CP(pi/2) on q[0], q[1]\n}\n' > /tmp/cp.aria
$ $ARIA run /tmp/cp.aria --circuit C --statevector
|11>  +0.000000+1.000000i
```

Check: `|11>` coefficient `= 0 + 1.0i` (= e^{iπ/2}; ±1e-6).

## 6. QML training — single qubit reaches ⟨Z⟩ = −1 (tol ≤ 1e-3)

```console
$ printf 'circuit Ry() {\n  qreg q[1]\n  let t = symbolic[1]\n  apply RY(t[0]) on q[0]\n}\n' > /tmp/ry.aria
$ $ARIA train /tmp/ry.aria --circuit Ry --observable "Z0" --steps 300 --lr 0.2 --seed 3
```

Check: `final   <O>: -1.000000000000` (≤ −0.999); `trained parameters: t_0 = -3.141593`
(= −π, ±1e-3).

## 7. VQE — H₂ ground-state energy (tol ≤ 1e-3)

```console
$ H2="-0.4804*I0+0.3435*Z0+-0.4347*Z1+0.5716*Z0Z1+0.0910*X0X1+0.0910*Y0Y1"
$ $ARIA train examples/aria/vqe_ansatz.aria --circuit VQEAnsatz --int n_layers=2 \
      --observable "$H2" --steps 600 --lr 0.1 --seed 7
```

Check: `final   <O>: -1.851199...` — equals the exact minimum eigenvalue
`-1.851199` (±1e-3).

## 8. Export round-trips (structural + numeric)

```console
$ $ARIA export examples/aria/bell.aria --circuit Bell --qasm | grep -c '^cx q\[0\], q\[1\];'
1
$ $ARIA export examples/aria/qft.aria --circuit QFT --int n=3 --json | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["instructions"]))'
7
```

Check: QASM contains exactly one `cx q[0], q[1];`; QFT(n=3) JSON has 7 instructions.

## 9. GPU backend agrees with CPU (tol ≤ 1e-6, macOS/Metal)

```console
$ cargo test -p aria-runtime --features metal --test run_examples gpu_metal
```

Check: `gpu_metal_agrees_with_sim_on_qft ... ok` — the Metal statevector matches
the CPU statevector on QFT(n=3) to ≤ 1e-6. (Build `aria` with `--features metal`
and use `--backend gpu` to run on the GPU.)

## 10. Remote backend via omega-server (tol ±5%)

```console
# Terminal 1: start a local omega-server (bearer-only auth).
$ OMEGA_PORT=8899 OMEGA_DB_PATH=/tmp/aria.db \
    cargo run -p omega-server -- --auth bearer-only --save-token-to /tmp/aria.tok

# Terminal 2: run Bell remotely.
$ aria=$(cargo build -p aria-cli --features remote --message-format=json \
      | sed -n 's/.*"executable":"\([^"]*aria\)".*/\1/p' | head -1)
$ $aria run examples/aria/bell.aria --circuit Bell --backend remote \
      --url http://127.0.0.1:8899 --token "$(cat /tmp/aria.tok)" --shots 4096 --seed 7
```

Check: only `|00>`/`|11>` appear, each probability in `[0.45, 0.55]` — the same
physics as the local `sim` backend, executed over HTTP.

## 11. libtorch (tch) backend trains VQE (tol ≤ 1e-3)

Needs `LIBTORCH` (libtorch 2.7.0, see `INSTALL_LIBTORCH.md`). Do **not** set
`LIBTORCH_USE_PYTORCH` — any value makes `torch-sys` look for a pip `torch`:

```console
$ export LIBTORCH=/path/to/libtorch
$ export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH   # macOS
$ cargo test -p aria-runtime --features tch --test run_examples tch_backend -- --test-threads=1
$ H2="-0.4804*I0+0.3435*Z0+-0.4347*Z1+0.5716*Z0Z1+0.0910*X0X1+0.0910*Y0Y1"
$ cargo run -p aria-cli --features tch -- train examples/aria/vqe_ansatz.aria \
      --circuit VQEAnsatz --int n_layers=2 --observable "$H2" --backend tch --steps 600 --lr 0.1 --seed 7
```

Check: `tch_backend_agrees_with_sim_on_qft ... ok` (tch statevector == CPU ≤ 1e-9);
VQE `final <O>: -1.851199...` — exact H₂ ground state −1.851199 (±1e-3), trained on
libtorch.

## 12. Full example corpus parses + instantiates

```console
$ cargo test -p aria-core --test aria_examples
```

Check: `test result: ok` — all 30 `.aria` examples parse and instantiate to a
non-empty circuit, and every `.aria` file on disk is covered by the table.

## 13. Application harnesses — quantum vs classical (`aria-verify`)

Each shipped example is run as a real application: the `.aria` source is
lowered and executed **through the omega WASM runtime** (a guest loaded
in-process — the no-socket all-in-one case), and the result is cross-checked
against a pure-Rust **classical oracle**. Every harness prints exactly WHAT it
computes and a PASS/FAIL verdict within a stated tolerance.

Each example is its own crate under `crates/apps/<name>` (the matching `.aria`
file points at it in its header). Build the WASM guests once (so the wasm
transport is used, not the native fallback), then run the whole suite:

```console
$ ( cd examples/wasm-guests/vqe       && cargo build --target wasm32-wasip1 --release )
$ ( cd examples/wasm-guests/omega_app && cargo build --target wasm32-wasip1 --release )
$ cargo run -q -p aria-verify -- all      # whole suite (what CI asserts)
$ cargo run -q -p aria-app-qsvd           # a single example, standalone crate
```

Check: `14/14 passed` (exit code 0). Per-example numeric goldens:

| example              | computed quantity                         | classical oracle              | golden / tol |
|----------------------|-------------------------------------------|-------------------------------|--------------|
| `qsvd`               | singular values of M=[[2,1],[1,3]]        | Jacobi SVD                    | σ=[3.618034, 1.381966], Δ≤1e-3 |
| `qft`                | QFT\|x=5⟩ amplitudes, n=3                  | DFT matrix · e_x              | Δ≤1e-6 (≈9e-16) |
| `vqe_ansatz`         | H₂ ground-state energy                     | exact min eigenvalue (4×4)    | E₀=−1.851199, \|Δ\|≤1e-3 |
| `grover3`            | P(marked=5), 8192 shots                    | sin²((2k+1)θ)=0.9453          | Δ≤0.05; argmax==5 |
| `bernstein_vazirani` | recovered hidden string                    | a=5 (input)                   | exact (=0101) |
| `deutsch_jozsa`      | balanced/constant decision                 | truth table ⇒ 2ⁿ−1=7          | exact (=111) |
| `swap_test`          | P(anc=0) for orthogonal \|+⟩/\|−⟩ states   | ½+½\|⟨ψ\|φ⟩\|²=0.5            | Δ≤0.02 |
| `teleport`           | Bob's qubit over 8 trajectories            | exact teleportation           | fraction==1.0 |
| `qaoa_maxcut`        | QAOA(p=2) cut, triangle                     | brute-force MaxCut=2          | Δ≤0.5 |
| `qml_classifier`     | confusion matrix, synthetic y=sign(x)      | ground-truth labels           | accuracy ≥ 0.9 (=1.0) |
| `qos`                | exact phase oracle (qos_oracle.aria via omega) + sketch convergence | `qos::target_state`; analytic exponent −2 | oracle infidelity ≤1e-6 (≈0); fitted exponent ≈−2.000 |
| `circulant`          | cyclic-shift generator Q (circulant.aria, 3q) via omega | exact permutation σ(x)=(x+1) mod 8 | 0 mis-mapped basis states; purity ≥0.99 |
| `cqs`                | Hadamard-test overlap Re⟨ψ\|Z\|ψ⟩ (cqs.aria) via omega | cos(π/3)=0.5 via `cqs::apply_pauli` | \|Δ\| ≤ 0.05 (8192 shots, ≈0.005) |
| `noise`              | depolarizing law anchor + real cqs.aria Hadamard test under noise | (1−4p/3)cosθ; noiseless overlap 0.5 | worst \|Δ\| ≤ 4σ≈0.022 (≈0.009); overlap degrades 0.5→0.15; p*=0.075 |

Run a single example: `cargo run -q -p aria-verify -- qsvd`. Force the native
fallback (no wasm guest): add `--native`.

## 14. Socket transport — same package over omega-server (numeric)

The same Aria package can be driven over a socket: the lowered circuit is POSTed
to a running `omega-server`, executed remotely, and the returned counts are
cross-checked against the same classical oracle.

```console
# Terminal 1: start a local omega-server (bearer-only auth).
$ OMEGA_PORT=8899 OMEGA_DB_PATH=/tmp/aria.db \
    cargo run -p omega-server -- --auth bearer-only --save-token-to /tmp/aria.tok

# Terminal 2: run the counts-based examples over the socket.
$ cargo run -q -p aria-verify --features remote -- \
    socket --url http://127.0.0.1:8899 --token "$(cat /tmp/aria.tok)"
```

Check: both examples PASS — `grover3` P(marked) within 0.05 of 0.9453, and
`bernstein_vazirani` recovers `0101`. (`ci.sh` automates this as a best-effort
stage; the in-process transport in §13 is the canonical path.)
