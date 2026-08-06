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

## Qiskit — MANDATORY, and now installed

The differential cross-checks are **not optional** (`OPTIONAL_TESTS.md`). Both
are installed here, **each in its own venv** — never system Python:

| venv | used by | contents | size |
|---|---|---|---|
| `.venv-qiskit` | `ARIA_QISKIT_XCHECK=1` | qiskit 2.5.1, qiskit-aer 0.17.2, numpy 2.5.1, scipy 1.18.0 | 193 MB |
| `tools/qec_cross_check/.venv` | `ARIA_QEC_XCHECK=1` | qiskit 2.5.1, PyMatching 2.4.0, stim 1.16.0 | 566 MB |

Two separate environments is deliberate, not an accident: the QEC script
**self-provisions** its own (`tools/qec_cross_check/run.sh` creates it and
installs qiskit + stim + PyMatching on first run), so it stays reproducible on a
fresh machine without anyone remembering an extra step. Both are gitignored via
`**/.venv*`.

Recreate either from scratch:

```console
$ python3 -m venv .venv-qiskit
$ ./.venv-qiskit/bin/pip install qiskit qiskit-aer
$ ARIA_QISKIT_XCHECK=1 ./ci.sh        # QEC venv builds itself on first run
$ ARIA_QEC_XCHECK=1 ./ci.sh
```

No C++ toolchain is needed: `qiskit-aer`, `stim` and `PyMatching` all ship
prebuilt wheels for Apple Silicon. Nothing compiles from source.

Measured here 2026-08-06, both `CI_EXIT=0`:

- Qiskit: **60 agree, 0 disagree, worst |Δp| = 4.441e-16**
- QEC vs PyMatching: **100.00% (20000/20000)** shot-for-shot logical-class
  agreement at d=3 and d=5; logical rates within 3σ

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

## The optional test matrix

`OPTIONAL_TESTS.md` is the durable record of every opt-in stage — what it
needs, what it buys, and what has actually been run on this box. Consult it
before a release: a stage that has not run recently should be visible rather
than quietly forgotten.

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
