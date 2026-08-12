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

Backends
--------
Each method's ``backend`` argument takes a spec string or a reusable
:class:`Backend`. **The spec names the engine, never the vendor** — the same
names ``aria`` itself accepts::

    "sv"          statevector (CPU, the default)
    "mps"         matrix product state, "mps:<chi>" to set the bond dimension
    "pauliprop"   Pauli propagation
    "gpu"         accelerated statevector — CUDA, Metal or OpenCL, whichever
                  this wheel was built with

So a script does not change when it moves between an NVIDIA box and an Apple
one. ``aria_py.accelerator()`` reports which one is compiled in (``"cuda"``,
``"metal"``, ``"opencl"`` or ``None``); ``aria_py.backends()`` lists the specs
this wheel accepts. Pin an arm only to benchmark it deliberately: ``"gpu:cuda"``,
``"gpu:metal"``, ``"gpu:opencl"`` — a pin the wheel lacks is an error, never a
silent downgrade to CPU.

``mps`` and ``pauliprop`` have no GPU variant on purpose: their accelerator (a
bond-compression SVD, a branch expansion) is transparent, engages when it pays,
and falls back per operation, so the result is identical either way. ``gpu``
instead *errors* when the device is unusable — a silent fallback there would
make a "GPU" measurement quietly report the CPU.

**Reuse a backend when it owns a device.** Construction is not free: building a
CUDA backend per call put a flat ~390 ms floor under every GPU call, unchanged
from n=4 to n=16::

    be = aria_py.Backend("gpu")            # one device handle...
    for _ in range(steps):
        g = m.gradient_batch(rows, "Z0", be)   # ...reused every step

A ``Backend`` belongs to the thread that built it — the CUDA handle is neither
``Send`` nor ``Sync``, and stream capture is invalidated by concurrent work in
the same context — so using one from another thread raises. For parallel work,
build one per thread or per process.

The optional PyTorch bridge lives in ``aria_py.torch`` (``pip install .[torch]``):
``AriaFunction`` (autograd) and ``AriaLayer`` (nn.Module).
"""

from ._aria_py import (  # noqa: F401
    Backend,
    Model,
    accelerator,
    backends,
    load,
    load_source,
    __version__,
)

__all__ = [
    "Backend",
    "Model",
    "accelerator",
    "backends",
    "load",
    "load_source",
    "__version__",
]
