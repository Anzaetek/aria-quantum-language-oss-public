# SPDX-License-Identifier: Apache-2.0
"""Tests for the PyTorch bridge (skipped if torch isn't installed)."""
import pytest

torch = pytest.importorskip("torch")
import aria_py  # noqa: E402
from aria_py.torch import AriaFunction, AriaLayer  # noqa: E402

SRC = ("circuit C() { qreg q[1]\n  let theta = symbolic[3]\n  let x = symbolic[1]\n"
       "  apply RY(x[0]) on q[0]\n  apply RZ(theta[0]) on q[0]\n"
       "  apply RY(theta[1]) on q[0]\n  apply RZ(theta[2]) on q[0] }")


def _layer():
    return AriaLayer(aria_py.load_source(SRC, "C"), "Z0", feature_prefix="x", seed=0)


def test_layer_shapes():
    layer = _layer()
    assert layer.num_features == 1
    assert layer.num_weights == 3
    y = layer(torch.zeros(5, 1, dtype=torch.float64))
    assert y.shape == (5,)


def test_gradcheck_weights_and_features():
    layer = _layer()
    m = layer.model
    x = torch.randn(6, 1, dtype=torch.float64, requires_grad=True)
    w = layer.weights.detach().clone().requires_grad_(True)

    def f(weights, xin):
        return AriaFunction.apply(weights, xin, m, "Z0", "sv",
                                  layer.feature_cols, layer.weight_cols)

    assert torch.autograd.gradcheck(f, (w, x), eps=1e-6, atol=1e-5)


def test_trains_in_a_hybrid_model():
    torch.manual_seed(0)
    layer = _layer()
    s = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float64))
    b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    opt = torch.optim.Adam(list(layer.parameters()) + [s, b], lr=0.1)
    xs = torch.linspace(-1, 1, 60, dtype=torch.float64).unsqueeze(1) * 1.5
    ys = (xs.squeeze() >= 0).double()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(80):
        opt.zero_grad()
        loss = loss_fn(s * layer(xs) + b, ys)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (((s * layer(xs) + b) > 0).double() == ys).double().mean().item()
    assert acc > 0.95, acc
