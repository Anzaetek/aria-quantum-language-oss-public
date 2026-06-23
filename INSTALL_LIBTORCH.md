# Installing libtorch for the `tch` backend

The `tch` backend (`--features tch`, `--backend tch`) links **libtorch**, the
C++ runtime behind PyTorch, via the [`tch`](https://crates.io/crates/tch) crate.
It is an **optional accelerator** — the default Aria training path is pure Rust
and needs none of this (`cargo build`, `cargo test`, and `./ci.sh` all pass with
no libtorch installed). Install libtorch only if you want to exercise
`--backend tch`.

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
want the directory that contains `lib/`, `include/`, and `share/`.

```console
$ ls /path/to/libtorch
build-version  include  lib  share
$ cat /path/to/libtorch/build-version
2.7.0
```

> On this mac dev box a matching build already lives at
> `…/work/quantum/libtorch` (libtorch 2.7.0, arm64) and can be reused directly —
> point `LIBTORCH` at it.

## 2. Point the toolchain at it

```console
$ export LIBTORCH=/path/to/libtorch
$ export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH   # macOS
$ export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH       # Linux
```

### Gotchas

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

`./ci.sh` runs the numeric gate automatically when `LIBTORCH` is set, and skips
it (with a pointer to this file) otherwise. See `TESTING.md` §11 for the same
steps in the manual testing manual.
