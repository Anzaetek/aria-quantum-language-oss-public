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
m.expectation([0.7], "Z0", backend="mps:8")          # sv | mps | mps:<chi>
```

**Parameter vectors align to `m.symbols`** (ascending SymbolId order) — the same
binding-order contract documented in `docs/LIBRARY.md`. Look symbols up by name;
don't assume positions from source order.

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
