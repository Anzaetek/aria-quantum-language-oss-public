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
| `.venv-piquasso` | `ARIA_CV_XCHECK=1` | piquasso 8.0.1, numpy 2.4.6 | 400 MB |
| `bindings/aria-py/.venv` | aria-py python tests | maturin, pytest, the built `aria_py` wheel. Build it with a **CPython &le; 3.13** (`python3.13 -m venv bindings/aria-py/.venv`) — pyo3 0.23 refuses anything newer, and ci.sh then auto-detects this venv. See `ARIA_PY_PYTHON` in `OPTIONAL_TESTS.md` | 120 MB |
| `crates/omega-bridges/python/.venv-qiskit` | bridge runner python tests | qiskit, qiskit-aer, **pytest** | 200 MB |
| `crates/omega-bridges/python/.venv-{perceval,bloqade,tsim,ppvm}` | `ARIA_BRIDGE_XCHECK=1` | see `requirements-*.txt` | ~3 GB total |

### Which arms can this machine run?

```sh
$ make -C crates/omega-bridges/python check-env
ARM        VENV      PYTHON   STATUS
qiskit     built     3.12     ready
perceval   built     3.12     ready
...
```

It probes each venv by importing the package, not by testing for a directory —
a venv can exist and be unusable, and one of the two ways that happens is the
trap below. It also prints which `ARIA_*` flag each arm gates, so "why did that
stage skip?" has a one-command answer instead of a trail through `ci.sh`, the
Makefile and five requirements files.

### The bridge venvs need Python ≥ 3.10 — including perceval

`crates/omega-bridges/python/Makefile` defaults to `PY ?= python3`, which on
macOS is often **3.9**. The `tsim` and `ppvm` requirement files already say they
need ≥ 3.10 and fail loudly on 3.9. **Perceval does not fail loudly**, and that
is the trap: on 3.9, pip silently resolves `perceval-quandela` back to **1.0.1**,
whose `Sampler.samples()` returns `None` where 1.2.x returns a dict. The runner
then dies with `AttributeError: 'NoneType' object has no attribute 'get'`, the
cross-backend harness records it as "skipping", and the stage reports success
having compared **nothing** on that arm — the same shape as the anchor that
reported "7 passed" while comparing zero cells.

Build all four with an explicit interpreter:

```sh
$ P=$(uv python find 3.12)          # or any python3.12 on PATH
$ make -C crates/omega-bridges/python perceval-venv bloqade-venv tsim-venv ppvm-venv PY="$P"
```

Verified on 3.12: perceval-quandela 1.2.4, bloqade-circuit 0.14.4,
bloqade-tsim 0.1.5, ppvm 0.1.0 (git + maturin build, needs the Rust toolchain).
With those the cross-backend arms genuinely compare — perceval 3 fixtures,
ppvm 10 of 14, tsim 11 of 14, the rest refused with typed reasons.

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

No C++ toolchain is needed: `qiskit-aer`, `stim`, `PyMatching` and `piquasso`
all ship prebuilt wheels for Apple Silicon. Nothing compiles from source.

The CV venv is **not** needed to run the CV cross-check: its piquasso fixture is
committed, so `cargo test -p omega-backend-cv` compares against it with no Python
at all. The venv is only for `ARIA_CV_XCHECK=1`, which reruns piquasso live and
checks the committed fixture has not drifted — the one failure the Rust test
cannot see, since a fixture regenerated to match our own change would still be
green. Create it with:

```console
$ python3 -m venv .venv-piquasso
$ ./.venv-piquasso/bin/pip install piquasso numpy
```

`verify_fixture.py` finds it at `.venv-piquasso` first, then
`tools/cv_cross_check/.venv`, then falls back to whatever interpreter runs it.

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
