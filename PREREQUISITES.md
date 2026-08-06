<!-- SPDX-License-Identifier: Apache-2.0 -->
# Prerequisites — what you actually need to install

Audited on this machine (Apple Silicon, macOS) on 2026-08-06. The short answer:

> **For the default `./ci.sh`: nothing. You already have everything.**
>
> **One optional stage needs an install:** the Qiskit cross-check wants a Python
> venv. Everything else is either present or fetched automatically.

## Already installed here — no action

| tool | why it is needed | found at |
|---|---|---|
| `cargo` / `rustup` | everything | `~/.cargo/bin` |
| `wasm32-wasip1` target | CI step 7 builds the WASM guests | installed |
| `python3` | the `aria tune` smoke in CI step 6 | `~/miniconda3/bin/python3` |
| `curl`, `unzip` | fetching libtorch | system |
| `elan` / `lake` | Lean 4 proof tree (`ARIA_LEAN=1`) | `~/.elan/bin`, `.lake` cache warm |
| `openjdk@17` | TLA+ model checking | `/opt/homebrew/opt/openjdk@17` |

## Fetched automatically — no action

- **libtorch 2.7.0** — the tch stage downloads the pinned CPU distribution on
  first run (~67 MB, idempotent). You do **not** need to install PyTorch, and
  you should not set `LIBTORCH_USE_PYTORCH`.
- **TLC (`tla2tools.jar`)** — 2.2 MB, project-local and gitignored. Fetch with:
  ```console
  $ curl -fsSL -o tools/tla/tla2tools.jar \
      https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
  ```
  then `tools/tla/check.sh`. The script skips cleanly if either the jar or a JDK
  is missing.

## The one thing you might want to install

**Qiskit cross-check** (`ARIA_QISKIT_XCHECK=1`, `ARIA_QEC_XCHECK=1`) — an
independent implementation to differential-test against, so agreement is
evidence rather than a shared convention. It is **not** installed here:

```console
$ python3 -m venv .venv-qiskit
$ ./.venv-qiskit/bin/pip install qiskit qiskit-aer
$ ARIA_QISKIT_XCHECK=1 ./ci.sh
```

Use a venv, never system Python — `tools/qiskit_xcheck/README.md` has the
details. Both stages skip cleanly without it, which is why the default run is
green on a bare machine.

## Nothing to install for GPUs on this Mac

- **Metal** — part of macOS. The stage runs by default.
- **OpenCL** — Apple ships `OpenCL.framework`. Also default-on.
- **CUDA** — not applicable (no NVIDIA hardware; `nvidia-smi` absent, correctly).
  On a CUDA box it is `ARIA_CUDA=1`.

## If you are setting up a Linux box

Not needed here, recorded so the Mac-centric list above does not mislead:

- **OpenCL** needs the ICD *loader dev symlink* `libOpenCL.so`, not just the
  runtime `libOpenCL.so.1` — `cl-sys` emits `-lOpenCL`, which the linker can
  only resolve against the `.so`. Usually `apt install ocl-icd-opencl-dev`.
  A CUDA-only box has the loader under
  `$CUDA/targets/<arch>/lib/libOpenCL.so`, off the default link path.
- **A JDK** for TLA+: `apt install openjdk-17-jdk`.

## Summary

```console
# Optional, and the only genuine install:
python3 -m venv .venv-qiskit && ./.venv-qiskit/bin/pip install qiskit qiskit-aer

# Optional, if you want to run the TLA+ models:
curl -fsSL -o tools/tla/tla2tools.jar \
  https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
```

No `brew install` is required on this machine. `openjdk@17` — the one thing that
would normally need brew — is already present.
