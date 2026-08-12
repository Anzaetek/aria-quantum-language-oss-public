<!-- SPDX-License-Identifier: Apache-2.0 -->
# aria-py — Python bindings for the Aria runtime

Opt-in [pyo3](https://pyo3.rs) bindings that expose the pure-Rust Aria runtime to
Python: lower an Aria circuit once, then evaluate expectations and **exact
adjoint gradients** over parameter vectors — including a PyTorch
`autograd.Function` and an `nn.Module` layer. No libtorch inside the circuit; the
extension links the statevector / MPS backends directly.

This crate is **excluded from the default cargo workspace** — `cargo build` and
`ci.sh` never touch it. Build it with [maturin](https://www.maturin.rs).

## Build & install

```console
$ python3 -m venv bindings/aria-py/.venv && source bindings/aria-py/.venv/bin/activate
$ pip install maturin numpy
$ cd bindings/aria-py && maturin develop --release        # unset CONDA_PREFIX if using conda
$ python -m pytest tests/                                  # 5 binding tests
$ pip install '.[torch]' && python -m pytest tests/       # + 3 torch tests
```

## Core API (numpy, no torch)

```python
import aria_py, math

m = aria_py.load_source(
    "circuit C() { qreg q[1]  let t = symbolic[1]  apply RY(t[0]) on q[0] }", "C")
# or: aria_py.load("model.aria", "Circuit", [("L", 3)])

m.symbols                              # ['t_0']  — ascending SymbolId order
m.expectation([0.7], "Z0")            # ⟨Z0⟩ = cos(0.7)
m.gradient([0.7], "Z0")               # ∂⟨Z0⟩/∂t = [-sin(0.7)]  (adjoint AD)
m.expectation_batch([[0.0], [1.0]], "Z0")            # one value per row (parallel)
m.gradient_batch([[0.0], [1.0]], "Z0")               # (rows × params)
m.expectation([0.7], "Z0", backend="mps:8")          # see Backends below
```

**Parameter vectors align to `m.symbols`** (ascending SymbolId order) — the same
binding-order contract documented in `docs/LIBRARY.md`. Look symbols up by name;
don't assume positions from source order.

## Backends — and how this API evolved

```python
aria_py.accelerator()      # 'cuda' | 'metal' | 'opencl' | None  (what this wheel has)
aria_py.backends()         # specs this wheel accepts

m.expectation([0.7], "Z0", "sv")          # statevector, CPU (the default)
m.expectation([0.7], "Z0", "mps:8")       # MPS, bond dimension 8
m.expectation([0.7], "Z0", "pauliprop")   # Pauli propagation
m.expectation([0.7], "Z0", "gpu")         # accelerated statevector

be = aria_py.Backend("gpu")               # build the device handle ONCE
for _ in range(steps):                    # and reuse it
    g = m.gradient_batch(rows, "Z0", be)
```

Two things changed here, both for reasons worth stating.

**1. A backend can now be built once and reused.** Every method used to
construct its backend internally, per call. On CPU that is free; on a GPU it is
a device handle, and the cost was not marginal — measured on a GB10, one
`gradient_batch` at n=6, L=3, batch 32:

| how the backend is passed | median |
|---|---|
| `"gpu"` — spec string, fresh backend per call | 429.7 ms / 603.7 ms |
| `aria_py.Backend("gpu")` — built once, reused | **51.1 ms / 52.4 ms** |
| `"sv"`, either way | 0.3 ms |

Two independent runs, hence two figures: **8.4x and 11.5x**. The per-call side
moves with machine load (both runs shared the GPU with another job) while the
reused side does not, which is what you would expect if the difference is
construction rather than work.

Measured directly, `aria_py.Backend("gpu")` construction is **316 ms** (median
of 12; min 313, max 348) against **0.000 ms** for `"sv"`. That is the number to
quote. An earlier version of this note described a "flat ~390 ms floor
independent of circuit size", inferred from batch-1 cells sitting at 379-409 ms
from n=4 to n=16; the direct measurement is the better evidence, and the reused
path does scale with `n` because it is doing the actual work. Results are
bit-identical between the two paths (max |Δ| = 0.0, checked both runs). Spec
strings still work exactly as before, so this is additive; the default `sv`
still builds per call because there it costs nothing.

A `Backend` **belongs to the thread that built it.** The CUDA backend holds a
captured `CudaGraph` (a raw `*mut CUgraph_st`), so it is neither `Send` nor
`Sync`, and CUDA stream capture really is invalidated by concurrent work in the
same context. Using one from another thread raises rather than corrupting a
handle. **For parallel work, build one backend per thread or per process.**

**2. Spec strings name the engine, never the vendor.** An earlier version of
these bindings exposed `sv:cuda`, `mps:cuda[:<chi>]` and `pauliprop:cuda`. That
was a mistake: it put the accelerator vendor into user code, so the same script
needed editing to run on an Apple machine even though the Metal backends
provide the same capability. The canonical specs are now exactly the ones
`aria_runtime::run::BackendSel` accepts — `sv`, `mps[:<chi>]`, `pauliprop`,
`gpu` — and `gpu` resolves to CUDA, Metal or OpenCL by the same build-time
priority the CLI uses. The old vendor-named spellings still work as
**deprecated aliases**.

`mps` and `pauliprop` have no GPU variant on purpose: their accelerator (a
bond-compression SVD, a branch expansion) is transparent and falls back per
operation, so the answer is identical either way. `gpu` instead **errors** when
the device is unusable — a silent CPU fallback there would make a "GPU"
measurement quietly report the CPU. Pin an arm only to benchmark it
deliberately (`gpu:cuda`, `gpu:metal`, `gpu:opencl`); a pin this wheel lacks is
an error naming what it does have, never a downgrade.

## PyTorch bridge (`pip install '.[torch]'`)

```python
import torch, aria_py
from aria_py.torch import AriaLayer

m = aria_py.load_source(SRC, "C")          # feature symbols x_0.., weights theta_*
layer = AriaLayer(m, "Z0", feature_prefix="x")   # weights are an nn.Parameter
y = layer(torch.randn(32, layer.num_features))    # (32,) — ⟨Z0⟩ per sample, differentiable

# use like any nn.Module:
opt = torch.optim.Adam(layer.parameters(), lr=0.1)
loss = torch.nn.functional.binary_cross_entropy_with_logits(y, labels)
loss.backward(); opt.step()
```

`AriaFunction.backward` returns the Rust runtime's exact adjoint gradients for
both the weights and (if they require grad) the input features — verified by
`torch.autograd.gradcheck` in `tests/test_torch.py`.

## Benchmark (`pip install '.[bench]'`)

```console
$ python bench/bench_pennylane.py --qubits 6 --layers 3 --rows 128
```

Measures batched forward and gradient wall-clock against PennyLane's
`default.qubit` (backprop) and `lightning.qubit` (adjoint) on the same circuit.
On an Apple-silicon dev box (6 qubits, 3 layers, 128 rows) aria_py ran the
forward in ~0.36 ms and the gradient in ~0.71 ms; PennyLane took hundreds of ms
per call. **Read the numbers, not a single ratio** — much of the gap is
PennyLane's per-row Python/QNode overhead (it has no native parameter-row
batching), so this measures how the two *APIs* perform at scoring/training a
dataset, not a bare simulator-kernel comparison.
