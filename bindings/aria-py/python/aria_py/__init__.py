# SPDX-License-Identifier: Apache-2.0
"""aria_py — Python bindings for the Aria Quantum Language runtime.

Lower an Aria circuit once, then evaluate expectations and adjoint gradients
over parameter vectors (aligned to ``Model.symbols``, ascending-SymbolId order):

    import aria_py, numpy as np
    m = aria_py.load_source(
        "circuit C() { qreg q[1]  let t = symbolic[1]  apply RY(t[0]) on q[0] }",
        "C",
    )
    m.symbols                      # ['t_0']
    m.expectation([0.7], "Z0")     # cos(0.7)
    m.gradient([0.7], "Z0")        # [-sin(0.7)]
    m.expectation_batch([[0.0], [1.0]], "Z0")   # [1.0, cos(1.0)]

The optional PyTorch bridge lives in ``aria_py.torch`` (``pip install .[torch]``):
``AriaFunction`` (autograd) and ``AriaLayer`` (nn.Module).
"""

from ._aria_py import Model, load, load_source, __version__  # noqa: F401

__all__ = ["Model", "load", "load_source", "__version__"]
