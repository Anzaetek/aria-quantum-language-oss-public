# SPDX-License-Identifier: Apache-2.0
"""PyTorch bridge for Aria circuits (``pip install .[torch]``).

``AriaFunction`` is a ``torch.autograd.Function`` whose forward is the circuit
readout ``⟨O⟩`` over a batch of parameter rows and whose backward returns exact
**adjoint** gradients computed by the Rust runtime — no libtorch inside the
circuit, no parameter-shift shot noise. ``AriaLayer`` wraps it as an
``nn.Module`` usable like ``nn.Linear`` inside a hybrid model: it holds the
circuit weights as an ``nn.Parameter`` and maps a batch of feature vectors to
one ``⟨O⟩`` per sample.

    import aria_py
    from aria_py.torch import AriaLayer
    m = aria_py.load_source(SRC, "C")           # features x_0.., weights theta_*
    layer = AriaLayer(m, "Z0", feature_prefix="x")
    y = layer(torch.randn(32, layer.num_features))   # (32,) — differentiable
"""

from __future__ import annotations

import numpy as np
import torch

from . import Model


def _feature_weight_columns(symbols, feature_prefix):
    """Split ascending-order symbol positions into (feature_cols, weight_cols).

    ``feature_cols[j]`` is the param-vector position of feature symbol
    ``{feature_prefix}_{j}`` (so column j of the input maps there); every other
    symbol is a trainable weight.
    """
    pre = feature_prefix + "_"
    feat = {}
    weight_cols = []
    for pos, name in enumerate(symbols):
        if name.startswith(pre):
            try:
                j = int(name[len(pre):])
                feat[j] = pos
                continue
            except ValueError:
                pass
        weight_cols.append(pos)
    if feat:
        n = max(feat) + 1
        if set(feat) != set(range(n)):
            missing = sorted(set(range(n)) - set(feat))
            raise ValueError(f"missing feature symbols {feature_prefix}_{missing}")
        feature_cols = [feat[j] for j in range(n)]
    else:
        feature_cols = []
    return feature_cols, weight_cols


def _assemble_rows(weights_np, x_np, n_symbols, feature_cols, weight_cols):
    """Build the (batch, n_symbols) parameter matrix from features + weights."""
    batch = x_np.shape[0] if x_np is not None and x_np.size else (1 if not feature_cols else 0)
    if not feature_cols:
        batch = 1
    rows = np.zeros((batch, n_symbols), dtype=np.float64)
    for k, col in enumerate(weight_cols):
        rows[:, col] = weights_np[k]
    for j, col in enumerate(feature_cols):
        rows[:, col] = x_np[:, j]
    return rows


class AriaFunction(torch.autograd.Function):
    """autograd.Function: forward = ⟨O⟩ per row; backward = adjoint gradients."""

    @staticmethod
    def forward(ctx, weights, x, model, observable, backend, feature_cols, weight_cols):
        w_np = weights.detach().cpu().double().numpy()
        x_np = None if x is None else x.detach().cpu().double().numpy()
        rows = _assemble_rows(w_np, x_np, model.num_symbols, feature_cols, weight_cols)
        vals = np.asarray(model.expectation_batch(rows.tolist(), observable, backend))

        ctx.model = model
        ctx.observable = observable
        ctx.backend = backend
        ctx.feature_cols = feature_cols
        ctx.weight_cols = weight_cols
        ctx.save_for_backward(weights, x if x is not None else torch.empty(0))
        ctx.rows = rows
        ctx.has_x = x is not None
        return torch.as_tensor(vals, dtype=weights.dtype, device=weights.device)

    @staticmethod
    def backward(ctx, grad_out):
        g = grad_out.detach().cpu().double().numpy()  # (batch,)
        # (batch, n_symbols) adjoint gradients d⟨O⟩/dparam for every row.
        grads = np.asarray(
            ctx.model.gradient_batch(ctx.rows.tolist(), ctx.observable, ctx.backend)
        )
        weights, x = ctx.saved_tensors

        # dL/dθ = Σ_row grad_out_row · d⟨O⟩_row/dθ  (weights are shared).
        wcols = np.asarray(ctx.weight_cols, dtype=int)
        grad_w_np = (g[:, None] * grads[:, wcols]).sum(axis=0)
        grad_w = torch.as_tensor(grad_w_np, dtype=weights.dtype, device=weights.device)

        grad_x = None
        if ctx.has_x and ctx.feature_cols:
            fcols = np.asarray(ctx.feature_cols, dtype=int)
            grad_x_np = g[:, None] * grads[:, fcols]  # (batch, n_features)
            grad_x = torch.as_tensor(grad_x_np, dtype=x.dtype, device=x.device)

        # One grad per forward input; non-tensors get None.
        return grad_w, grad_x, None, None, None, None, None


class AriaLayer(torch.nn.Module):
    """An Aria circuit as an ``nn.Module`` mapping features → ⟨O⟩ per sample."""

    def __init__(
        self,
        model: Model,
        observable: str,
        feature_prefix: str = "x",
        backend: str = "sv",
        weight_init: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.observable = observable
        self.backend = backend
        self.feature_cols, self.weight_cols = _feature_weight_columns(
            model.symbols, feature_prefix
        )
        gen = None if seed is None else torch.Generator().manual_seed(seed)
        w0 = torch.randn(len(self.weight_cols), generator=gen) * weight_init
        self.weights = torch.nn.Parameter(w0.double())

    @property
    def num_features(self) -> int:
        return len(self.feature_cols)

    @property
    def num_weights(self) -> int:
        return len(self.weight_cols)

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        if self.num_features and x is None:
            raise ValueError(f"layer has {self.num_features} features; forward needs x")
        return AriaFunction.apply(
            self.weights,
            x,
            self.model,
            self.observable,
            self.backend,
            self.feature_cols,
            self.weight_cols,
        )
