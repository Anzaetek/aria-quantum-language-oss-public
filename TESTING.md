# TESTING — Aria Quantum Language (numeric manual)

Copy-pasteable shell. **Numbers only**: every step states an expected value and a
tolerance. No GUI, no "looks right". Run from the repo root.

Build the binary once:

```console
$ cargo build -p aria-cli
$ ARIA=target/debug/aria
```

> **Use a release build beyond ~20 qubits.** The debug binary is limited by
> speed (not memory): statevector runs that finish in seconds under
> `--release` appear to hang around 28 qubits in a debug build. For anything
> wide, build with:
>
> ```console
> $ cargo build --release -p aria-cli
> $ ARIA=target/release/aria
> ```

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

Check: value `= 1.000000000000` (±1e-10). Cross-check the Pauli-propagation
backend agrees (it evolves the observable instead of the state; exact on
Clifford circuits at any width):

```console
$ $ARIA run examples/aria/bell.aria --circuit Bell --expectation "Z0 Z1" --backend pauliprop
<Z0 Z1> = 1.000000000000
```

Note: `pauliprop` computes expectation values only — `--shots` /
`--statevector` on it fail with an explicit "expectation-value backend"
error rather than mis-sampling.

**Truncation knobs (deep non-Clifford, PauliPropagation.jl-style).** For circuits
where the Pauli tree explodes, cap it on any of three axes and read the certified
error budget. On a Clifford circuit nothing is dropped, so the budget is 0:

```console
$ $ARIA run examples/aria/bell.aria --circuit Bell --expectation "Z0 Z1" \
      --backend pauliprop --truncate 1e-3 --max-weight 4 --max-freq 6
<Z0 Z1> = 1.000000000000  (dropped-mass budget 0.000e0)
```

Check: value `= 1.000000000000` (±1e-10) and `dropped-mass budget 0.000e0`. The
budget is a hard bound: `|approx − exact| ≤ budget` always (numerically gated in
`omega-backend-pauliprop`'s `max_freq_truncation_stays_within_budget_and_converges`).

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

## 9. Metal GPU backends agree with CPU (Apple Silicon; **on by default**)

On an Apple Silicon Mac, three GPU paths are numerically gated against the CPU —
the same three the CUDA arm wires (§9a). **`./ci.sh` runs this stage
automatically on macOS**: every Apple Silicon Mac has a GPU, so there is nothing
to opt into. Off-Mac the stage is skipped (there is no Metal to test), so the
default CI stays green everywhere. `ARIA_METAL=0 ./ci.sh` forces it off.

This default is not cosmetic: the RBS/Reset work landed verified on a CUDA box
with the Metal mirrors deferred, and shipped with two failing Metal tests
precisely because no contributor's default `./ci.sh` ran them.

The stage runs exactly this command list:

```console
$ cargo test --release -p omega-backend-pauliprop-metal --features metal
$ cargo test --release -p omega-backend-mps-metal --features metal
$ cargo test --release -p omega-backend-statevector-metal --features metal
$ cargo test --release -p aria-runtime --features metal --test run_examples gpu_metal_agrees_with_sim_on_qft
$ cargo test --release -p aria-runtime --features metal --test run_examples gpu_mps_metal_agrees_with_sim
$ cargo test --release -p aria-runtime --features metal --test run_examples gpu_pauliprop_metal_agrees_with_sim
$ cargo test --release -p aria-runtime --features metal --test run_examples rbs
```

Checks (each prints `... ok`):
- The `omega-backend-statevector-metal` suite is the per-kernel layer under the
  three end-to-end gates below: `apply_1q` / `apply_2q` / the diagonal fast
  paths, the adjoint and its QML-training parity, `reset_matches_cpu` (the
  deterministic mid-circuit Reset ≡ the CPU gate), and
  `expectation_matches_cpu_pauli_string` (an odd-Y observable, pinning the
  `(-i)^|Y|` prefactor in `pauli_masks` — see §9c).
- `gpu_metal_agrees_with_sim_on_rbs` / `gpu_metal_rbs_gradient_agrees_with_sim`
  (the `rbs` filter runs both) — RBS (Givens) forward ≤ 1e-6 and its adjoint
  gradient ≤ 1e-5 vs the CPU f64 path. RBS is sign-antisymmetric, so an
  amplitude match pins the gate's sign convention and qubit order, not just its
  magnitude.
- `gpu_metal_agrees_with_sim_on_qft` — Metal statevector == CPU statevector on
  QFT(n=3), ≤ 1e-6 (f32 kernels). Use `--backend gpu` to run on the GPU.
- `gpu_mps_metal_agrees_with_sim` — `--backend mps` with the Metal two-site
  θ-contraction (the SVD stays on CPU) == exact CPU statevector on a 12-qubit
  entangling brickwork, ≤ 1e-3 (f32 contraction). Apple GPUs have no native f64,
  so on-GPU Jacobi SVD is deferred; the contraction is the GPU-friendly half.
- `gpu_pauliprop_metal_agrees_with_sim` / `gpu_branch_matches_cpu_*` — the Metal
  Pauli-propagation branch == the CPU branch, ≤ 1e-9. The GPU runs only the
  integer symplectic work (anticommute test + child key); the f64 coefficient
  arithmetic stays on the CPU, so the result is **exact**, not approximate.

End-to-end (build with `--features metal`):

```console
$ cargo run --release -p aria-cli --features metal -- run examples/aria/qft.aria \
      --circuit QFT --int n=4 --backend gpu --statevector      # == --backend sim
$ cargo run --release -p aria-cli --features metal -- run examples/aria/bell.aria \
      --circuit Bell --expectation "Z0 Z1" --backend pauliprop  # branch on GPU
<Z0 Z1> = 1.000000000000
```

### 9a. CUDA GPU backends agree with CPU (NVIDIA; opt-in `ARIA_CUDA=1`)

On a CUDA host (Linux/Windows + an NVIDIA GPU), three GPU paths are numerically
gated against the CPU. All are optional and fall back to the CPU when the feature
is off or no device is present, so `./ci.sh` stays green without a GPU; set
`ARIA_CUDA=1 ./ci.sh` to run them, or invoke directly:

```console
$ cargo test -p omega-backend-statevector-cuda --features cuda
$ cargo test -p omega-backend-mps-cuda --features cuda
$ cargo test -p omega-backend-pauliprop-cuda --features cuda
$ cargo test -p aria-runtime --features cuda --test run_examples gpu_cuda_agrees_with_sim_on_qft
$ cargo test -p aria-runtime --features cuda --test run_examples gpu_mps_cuda_agrees_with_sim
$ cargo test -p aria-runtime --features cuda --test run_examples rbs
```

Checks (each test prints `... ok`):
- The `omega-backend-statevector-cuda` suite — `apply_2q` / adjoint / execute,
  `reset_matches_cpu` (deterministic mid-circuit Reset ≡ the CPU gate), and
  `expectation_matches_cpu_pauli_string` (odd-Y, pinning `(-i)^|Y|`; see §9c).
- `gpu_cuda_agrees_with_sim_on_rbs` / `gpu_cuda_rbs_gradient_agrees_with_sim`
  (the `rbs` filter runs both) — RBS forward ≤ 1e-5 and its adjoint gradient
  ≤ 1e-4 vs CPU.
- `gpu_cuda_agrees_with_sim_on_qft` — CUDA statevector == CPU statevector on
  QFT(n=4), ≤ 1e-5 (f32 kernels).
- `gpu_mps_cuda_agrees_with_sim` — `--backend mps` with the cuSOLVER `gesvdj`
  bond-compression SVD == exact CPU statevector on QFT(n=5), ≤ 1e-10 (native f64).
- `gpu_branch_matches_cpu_exact` / `gpu_branch_matches_cpu_with_max_freq` — the
  GPU Pauli-propagation branch (value **and** certified dropped-mass budget) ==
  the CPU branch, ≤ 1e-9, on a deep non-Clifford Trotter circuit.

End-to-end (build with `--features cuda`); the GPU result equals the CPU-only
build bit-for-bit:

```console
$ cargo run -p aria-cli --features cuda -- run examples/aria/qft.aria \
      --circuit QFT --int n=4 --backend gpu --statevector      # == --backend sim
$ cargo run -p aria-cli --features cuda -- run examples/aria/bell.aria \
      --circuit Bell --expectation "Z0 Z1" --backend pauliprop  # branch on GPU
<Z0 Z1> = 1.000000000000
```

Metal (Apple Silicon) wires all three arms too — statevector, the MPS two-site
θ-contraction (SVD stays on CPU), and the pauliprop branch — verified under
`ARIA_METAL=1` (§9). The one piece deferred on Metal is on-GPU Jacobi SVD, which
Apple's lack of native f64 rules out (see `GPU_BACKEND_PLAN.md`).

### 9b. OpenCL GPU statevector agrees with CPU (cross-vendor; on by default on macOS)

The OpenCL statevector backend is the cross-vendor arm (Apple's
`OpenCL.framework`, the Intel/AMD/NVIDIA runtimes, or POCL). Only the
statevector is wired — there is no OpenCL MPS or pauliprop arm — and RBS,
photonic, and 3q gates surface a clean *"unsupported gate"* error so the CLI
falls back to CPU.

**On macOS this stage runs by default** (Apple ships `OpenCL.framework`).
Elsewhere it is opt-in with `ARIA_OPENCL=1`, since a Linux/Windows host may have
no ICD installed. `ARIA_OPENCL=0` forces it off. Invoke directly with:

```console
$ ARIA_OPENCL_REQUIRE_DEVICE=1 cargo test -p omega-backend-statevector-opencl --features opencl
```

> **Linking needs the ICD loader *dev* symlink.** `cl-sys` emits `-lOpenCL`,
> which the linker can only resolve against `libOpenCL.so` — the bare runtime
> `libOpenCL.so.1` that ships with a driver is not enough, so the command above
> fails with `rust-lld: error: unable to find library -lOpenCL` on an otherwise
> working host. Most distros provide it in `ocl-icd-opencl-dev`
> (`sudo apt install ocl-icd-opencl-dev`). A CUDA box already has one under the
> toolkit but not on the default link path, so without root:
>
> ```console
> $ RUSTFLAGS="-L native=/usr/local/cuda/targets/x86_64-linux/lib" \
>     ARIA_OPENCL_REQUIRE_DEVICE=1 \
>     cargo test -p omega-backend-statevector-opencl --features opencl
> ```
>
> Both verified on the Linux/CUDA box (NVIDIA ICD, 43 tests, including
> `device_present_when_required` — i.e. a real device was used, not a silent
> skip). With `ocl-icd-opencl-dev` installed the plain command above links and
> `ARIA_OPENCL=1 ./ci.sh` is green with **no** `RUSTFLAGS`; the `RUSTFLAGS` form
> is only the no-root fallback. See also the note above that stage in `ci.sh`.

Checks: the per-kernel smokes (`apply_1q`, `apply_diagonal`,
`apply_diagonal_product`, `inner_product`), the end-to-end `execute` smoke,
buffer-pool semantics, shot-sampling TVD, the adjoint gradient vs the CPU
adjoint, and `pauli_expectation` — of which
`pauli_expectation_matches_host_on_random_14q` is the load-bearing one: its
X·Y·Z string has an **odd** Y count, so it pins the `(-i)^|Y|` prefactor in
`pauli_masks` against a host oracle. (An even-Y or Y-free string cannot see that
sign, and `⟨+|Y|+⟩ = 0` cannot either.)

`ARIA_OPENCL_REQUIRE_DEVICE=1` is what makes the stage honest. Every other test
in the crate skips itself when the constructor can't reach a device, and
`is_available()` reports only compile-time feature presence — so without the
env var, `cargo test --features opencl` reports "ok" on a host where not one
kernel ran. With it, `tests/device_present.rs` turns that silent skip into a
loud failure. Unset, that test is a no-op, so `cargo test --all-features`
elsewhere is unaffected.

### 9c. The Pauli-expectation sign convention (all backends)

One convention is shared by every backend and is the single easiest thing to get
silently wrong, so it is stated once here. For a Pauli string `P`, the fused
kernels and the CPU loop both accumulate over basis indices `i` with
`j = i XOR x_mask`:

```
⟨ψ|P|ψ⟩ = Σ_i conj(ψ[j]) · coeff(i) · ψ[i]
```

where `coeff(i)` is the **ket-side** coefficient of `P|i⟩ = coeff(i)·|j⟩`. The
coefficient is keyed on the bits of `i`, so it belongs with `ψ[i]`, not `ψ[j]`.

Pairing it the other way — `conj(ψ[i]) · coeff(i) · ψ[j]` — negates every string
with an **odd** number of Y factors, and nothing else. X and Z are symmetric
matrices, so they are unaffected; Y is antisymmetric (`Yᵀ = −Y`), so transposing
the element flips its sign. Equivalently, on the GPU side the global prefactor
is `(-i)^|Y|`, not `i^|Y|`: the kernel forms the matrix element `P[i, i^x]`,
which for a Y qubit is `(-i)·(-1)^bit_i`, and the `(-1)^bit` half is already
carried by `sign_mask`.

The failure mode is silent — a wrong-signed observable, not a crash — and two
such errors cancel, so a test oracle that makes the same mistake will agree with
a broken kernel. The gates that pin it:

| layer | test |
|---|---|
| CPU | `omega-backend-statevector/tests/pauli_y_expectation.rs` |
| Metal | `expectation_matches_cpu_pauli_string`, `pauli_expectation_matches_cpu_oracle` |
| CUDA | `expectation_matches_cpu_pauli_string` |
| OpenCL | `pauli_expectation_matches_host_on_random_14q` |

Each uses a string with an odd Y count. An even-Y string, a Y-free string, or
`⟨+|Y|+⟩ = 0` cannot see the sign at all.

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

## 11. libtorch (tch) backend trains VQE (tol ≤ 1e-3; **on by default**)

`./ci.sh` runs this stage automatically and **fetches libtorch itself** if it
isn't already configured — it resolves `$LIBTORCH`, then `./tch-env.sh`, then
falls back to `tools/setup-libtorch.sh`, which downloads the pinned 2.7.0 CPU
dist (~67 MB, one-time per machine; the script is idempotent and reuses an
existing install). Auto-download covers macOS arm64 and Linux x86_64; on other
platforms the stage prints a SKIP with the manual steps rather than failing.
`ARIA_TCH=0 ./ci.sh` skips it.

To drive it by hand, see `INSTALL_LIBTORCH.md`. Two requirements are easy to
miss and the CI stage applies both: the Apple-clang ≥ 21 workaround (libtorch
2.7 specializes `std::is_arithmetic`, which newer libc++ forbids), and
`--test-threads=1` (`tch` uses a process-global RNG). Do **not** set
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

Check: `test result: ok` — all 44 `.aria` examples parse and instantiate to a
non-empty circuit, and every `.aria` file on disk is covered by the table. The
second assertion is the one that keeps this number honest: adding a `.aria` file
without a table entry fails the test rather than silently going untested.

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

Check: `49/49 passed` (exit code 0). A representative set of the per-example
numeric goldens:

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
| `qec_grover`         | encoded 2-qubit Grover on Steane [[7,1,3]] (transversal, pauliprop) | ⟨Z̄ᵢ⟩=(−1)^bit(marked); statevector | Δ≤1e-6 (≈2.2e-16); all marked∈{0..3} + pauliprop==statevector |
| `qec_qft`            | logical QFT: QFT\|0⟩ uniform + QFT∘QFT⁻¹ roundtrip | uniform 1/16; recovered=1 | Δ≤1e-9 (≈4.4e-16) |
| `qec_qpe`            | logical QPE: φ=j/2ᵐ (m=3) recovered as a clean delta | φ̂ = true φ (all 8 phases) | Δ≤1e-6 (=0) |
| `qec_memory`         | surface-code memory under neutral-atom (ZZ-biased) noise, 8k shots | pL(d=5)<pL(d=3); p_lz>p_lx | both inequalities hold (violation margins =0, tol 1e-9) |

The four `qec_*` demos run key algorithms on **transversally QEC-encoded logical
qubits** via the `aria-qec` crate (Steane [[7,1,3]] + rotated surface code) and
are native-only (no wasm guest). `aria-verify -- all` reports `49/49 passed`
(the harness registry has grown well beyond the goldens tabled above — QML,
SPECTRA, HHL/QSVT, and GPU/kernel families among them; the table samples the
load-bearing ones).

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

## 15. QEC encoded demos cross-checked against Qiskit (opt-in `ARIA_QEC_XCHECK=1`)

The three algorithmic QEC demos (`qec-grover`, `qec-qft`, `qec-qpe`) are
cross-validated against an **independent SDK**. The tool exports the *exact* aria
logical circuit (`aria export --qasm` on the `examples/aria/qec_*.aria` twins),
runs it through Qiskit `Statevector` (exact) — plus a **stim** stabilizer tableau
for the Clifford Grover — and asserts **aria == Qiskit == analytic**, ≤ 1e-9.
Python runs in a venv (per repo policy); `run.sh` auto-creates one with qiskit.

```console
$ tools/qec_cross_check/run.sh          # auto-builds .venv + installs qiskit/stim
# or point at an existing venv:
$ QEC_PYTHON=/path/to/venv/bin/python tools/qec_cross_check/run.sh
# or as an opt-in ci.sh stage:
$ ARIA_QEC_XCHECK=1 ./ci.sh
```

`run.sh` runs two independent cross-checks; both must pass.

**(a) Encoded algorithms vs Qiskit** — `check_qec.py`, `20 passed, 0 failed`:

| case | independent reference | golden / tol |
|------|-----------------------|--------------|
| `qec-grover` marked∈{0,1,2,3} | Qiskit `Statevector` + stim stabilizer tableau | argmax==marked, P(marked)==1; aria==qiskit (Δ≈8.9e-16) |
| `qec-qft` | Qiskit `Statevector` (aria's `QFT`+`IQFT` circuits) | QFT\|0000⟩ uniform (maxdev≤1e-9); QFT∘QFT⁻¹\|x⟩=\|x⟩, x∈{0,5,11,15} |
| `qec-qpe` | Qiskit `Statevector` (aria's `QPEDemo(m=3)`, φ=3/8) | counting register →\|011⟩=3, P==1; aria==qiskit |

**(b) Surface-code decoder vs PyMatching** — `check_decoder.py`, `12 passed, 0
failed`. `qec-memory`'s engine is aria-qec's exact minimum-weight decoder
(`ecc/mwpm.rs`); it is cross-checked against **PyMatching** (the MWPM decoder the
QEC community pairs with stim) on identical error samples, for d ∈ {3,5} × both
CSS sectors:

| assertion | result |
|-----------|--------|
| all weight ≤ ⌊(d−1)/2⌋ errors correctable | holds (PyMatching) |
| shot-for-shot logical-class agreement ≥ 99% | **100.00%** (20k shots each) |
| logical-rate agreement aria ≈ PyMatching (≤3σ) | e.g. d=3 X: 0.0367 vs 0.0367; d=5 X: 0.0251 vs 0.0251 |

Both decoders independently reproduce the distance suppression `pL(d=5) <
pL(d=3)` the `qec-memory` demo asserts in §13. See
`tools/qec_cross_check/README.md`.
