# Installing libtorch for the `tch` backend

The `tch` backend (`--features tch`, `--backend tch`) links **libtorch**, the
C++ runtime behind PyTorch, via the [`tch`](https://crates.io/crates/tch) crate.
It is an **optional accelerator** — the default Aria training path is pure Rust
and needs none of this (`cargo build` and `cargo test` pass with no libtorch
installed). You do not have to follow this file to run CI: `./ci.sh` fetches
libtorch on its own (see the bottom of this page). Read on only if you want to
install it by hand or something went wrong.

## Quick start (macOS arm64)

```console
$ tools/setup-libtorch.sh          # download libtorch, build, and verify the tch backend
$ source ./tch-env.sh              # in later shells, before `cargo ... --features tch`
```

The script is idempotent, applies the clang workaround below automatically, and
writes `tch-env.sh` for reuse. The rest of this file is the manual walkthrough.

## Required version

| Component | Version |
| --- | --- |
| `tch` crate | `0.20` (see `crates/aria-backend-tch/Cargo.toml`) |
| libtorch | **2.7.0** (CPU build is sufficient) |

The libtorch version **must** match what the `tch` crate expects. With a
mismatched libtorch the `torch-sys` build script either fails or links against
ABI-incompatible symbols.

## 1. Get libtorch 2.7.0

Download the prebuilt C++ distribution from <https://pytorch.org/get-started/locally/>
(pick "LibTorch", your OS, and the **CPU** compute platform), then unzip it. You
want the directory that contains `lib/`, `include/`, and `share/`. Use the
**prebuilt** distribution — do not build libtorch from source.

Apple Silicon (arm64) direct link:

```console
$ curl -fL -o /tmp/libtorch.zip \
    https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.0.zip
$ unzip -q /tmp/libtorch.zip -d ~/work/            # → ~/work/libtorch
$ ls ~/work/libtorch
build-version  include  lib  share
$ cat ~/work/libtorch/build-version
2.7.0
```

## 2. Point the toolchain at it

```console
$ export LIBTORCH=/path/to/libtorch
$ export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH   # macOS
$ export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH       # Linux
```

### Gotchas

- **Recent clang / libc++ (Apple clang ≥ 21):** `torch-sys` fails to compile
  with `error: 'is_arithmetic' cannot be specialized [-Winvalid-specialization]`
  — libtorch 2.7's vendored `c10/util/strong_type.h` specializes
  `std::is_arithmetic`, which newer libc++ marks `[[no_specializations]]`. Demote
  that diagnostic when building:

  ```console
  $ export CXXFLAGS="-std=gnu++17 -Wno-invalid-specialization"
  ```

  This is a header-compat issue, not a missing include — `-include type_traits`
  does **not** fix it. If a future clang removes the flag, bump the tch/libtorch
  pin to a release whose `strong_type` no longer specializes `is_arithmetic`.
- **Do _not_ set `LIBTORCH_USE_PYTORCH`.** The `torch-sys` build script treats
  *any* value (including `0`) as "discover libtorch through an installed pip
  `torch`". If you have no pip `torch`, the build fails with
  `ModuleNotFoundError: No module named 'torch'`. Leave it unset and rely on
  `LIBTORCH` instead.
- **macOS SIP strips `DYLD_*` from protected processes.** Exporting
  `DYLD_LIBRARY_PATH` works for `cargo`-spawned binaries; if you wrap execution
  in a SIP-protected interpreter the variable is dropped. Run the `cargo`
  commands directly.
- **Global RNG.** `tch` uses a process-global RNG, so the backend tests run
  single-threaded (`-- --test-threads=1`).

## 3. Verify

```console
# Numeric gate: tch statevector == CPU statevector (≤ 1e-9)
$ cargo test -p aria-runtime --features tch --test run_examples tch_backend \
      -- --test-threads=1
#   tch_backend_agrees_with_sim_on_qft ... ok

# End-to-end: train the VQE ansatz on libtorch
$ H2="-0.4804*I0+0.3435*Z0+-0.4347*Z1+0.5716*Z0Z1+0.0910*X0X1+0.0910*Y0Y1"
$ cargo run -p aria-cli --features tch -- train examples/aria/vqe_ansatz.aria \
      --circuit VQEAnsatz --int n_layers=2 --observable "$H2" \
      --backend tch --steps 600 --lr 0.1 --seed 7
#   final   <O>: -1.851199...   (exact H₂ ground state -1.851199, ±1e-3)
```

`./ci.sh` runs the numeric gate **by default** and installs libtorch itself if
you haven't: it resolves `$LIBTORCH`, then `./tch-env.sh`, then falls back to
`tools/setup-libtorch.sh`, which downloads the pinned CPU dist (macOS arm64 and
Linux x86_64 are auto-detected; other platforms print a SKIP with these manual
steps rather than failing the run). So none of the above is a prerequisite for
`./ci.sh` — it is the manual path. `ARIA_TCH=0 ./ci.sh` skips the stage.

See `TESTING.md` §11 for the same steps in the manual testing manual.

## 4. GPU (CUDA) — run `--backend tch` on an NVIDIA device

CPU is the default. To run tch on the GPU, install a **CUDA** libtorch and let
the runtime pick the device:

```console
$ ARIA_TCH_CUDA=1 ARIA_TCH_CUDA_VER=cu128 tools/setup-libtorch.sh
```

- `ARIA_TCH_CUDA_VER` selects the CUDA build pytorch publishes for the pin
  (`cu118` / `cu126` / `cu128`); it must match your driver. **Blackwell (sm_120,
  e.g. RTX PRO 6000) and Hopper (H100) → `cu128`.** x86_64 downloads the
  `+cuXXX -with-deps` dist (bundles the CUDA runtime); aarch64 (Grace/GB10) pulls
  the CUDA `torch` wheel from `download.pytorch.org/whl/cuXXX`.
- `crates/aria-runtime/src/run.rs` `make_tch()` calls `TchBackend::cuda_or_cpu()`
  — GPU if `tch::Cuda::is_available()`, else CPU (so a CPU-only libtorch is
  unchanged). CUDA keeps `Kind::Double`, so numerics match the CPU path.
  `ARIA_TCH_CPU=1` forces CPU on a CUDA libtorch.

### Gotcha: the linker drops `libtorch_cuda` (the reason "CUDA present but unused")

`torch-sys` emits `-ltorch_cuda`, but the linker's default `--as-needed` **drops
it** because no Rust symbol references it directly — so `libtorch_cuda.so` never
lands in the binary's `DT_NEEDED`, the CUDA backend never registers, and
`tch::Cuda::is_available()` is **false even with a CUDA dist and a working
driver**. `tools/setup-libtorch.sh` fixes this automatically (when it sees
`libtorch_cuda.so` in the dist) by adding to `RUSTFLAGS`:

```
-C link-arg=-Wl,--no-as-needed -C link-arg=-L$LIBTORCH/lib \
-C link-arg=-ltorch_cuda -C link-arg=-lc10_cuda
```

appended **after** the flag so link order retains both libs. The recipe's CUDA
verify gate runs `crates/aria-backend-tch/tests/gpu_probe.rs` with
`ARIA_EXPECT_CUDA=1`, so a dropped `libtorch_cuda` is a hard failure, not a
silent CPU fallback.

### Tested matrix (build + `libtorch_cuda` retained in `NEEDED`; ✱ = real GPU op)

| Environment | GCC | Result |
|---|---|---|
| Host: Ubuntu 24.04, RTX PRO 6000 Blackwell, `cu128`, driver 580 | 13.3 | GPU gate + VQE ✱ |
| `ubuntu:22.04` container | 11 | build + link ✓ |
| `ubuntu:24.04` container, GPU device-bind | 13 | `is_available`=true, GPU op ✱ |
| `ubuntu:26.04` container | 15.2 | build + link ✓ |

GPU-in-container passthrough on this host uses **device-bind, not CDI** (no
`/etc/cdi` here): `--device /dev/nvidia0 --device /dev/nvidiactl --device
/dev/nvidia-uvm --device /dev/nvidia-uvm-tools --security-opt=label=disable`,
plus bind-mounting the host driver libs (`libcuda.so.<drv>`,
`libnvidia-ptxjitcompiler.so.<drv>`) and recreating their `.so.1` symlinks. The
CUDA *runtime* (cudart/cuDNN/cuBLAS) rides along inside the libtorch dist.
