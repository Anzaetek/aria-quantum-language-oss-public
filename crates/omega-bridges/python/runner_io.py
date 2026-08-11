# SPDX-License-Identifier: Apache-2.0
"""Protocol I/O shared by every omega-bridge runner.

**STDOUT IS THE WIRE.** The Rust side reads exactly one JSON object from a
runner's stdout (`crates/omega-bridges/src/runner.rs`). Anything else that
reaches fd 1 — a library banner, a deprecation notice, a progress bar —
corrupts the response, and the Rust side reports it as

    invalid JSON from runner: expected value at line 1 column 2

which reads as a broken backend rather than as a backend being talked over.

This is not hypothetical. Perceval logs a DeprecationWarning while the Qiskit
converter builds a Processor, and it broke the whole Perceval arm. The
diagnosis cost real time because the symptom names the wrong culprit.

**Why a shared module rather than four copies.** Each runner had its own
`_emit`, and the fix initially went into one of them. That leaves the property
depending on whether the next person remembers — and `tsim_runner.py` pulls in
JAX/XLA, which is famously chatty on first use. Importing the guard makes it
structural: a new runner gets it by construction.

**Why fd-level and not `sys.stdout = sys.stderr`.** Measured: the Python-level
rebind was not enough. Perceval's logger writes to file descriptor 1 directly,
so only `dup2` intercepts it. That also covers native/C extensions, which no
Python-level trick can reach.

Import this FIRST, before any heavy third-party import, so nothing can print to
the real stdout before the swap.
"""

from __future__ import annotations

import json
import os
import sys

# Duplicate the real stdout to a private descriptor, then point fd 1 at stderr.
# From here on, *anything* written to stdout by any code — Python or native —
# lands on stderr as operator-visible diagnostics instead of corrupting the
# protocol.
_PROTOCOL_FD = os.dup(1)
os.dup2(2, 1)
_PROTOCOL_STDOUT = os.fdopen(_PROTOCOL_FD, "w")

# Belt and braces: rebind the Python-level handle too, so code that captured
# `sys.stdout` rather than writing to fd 1 also lands on stderr.
sys.stdout = sys.stderr


def emit(payload: dict) -> None:
    """Write the single JSON response line to the protocol descriptor."""
    _PROTOCOL_STDOUT.write(json.dumps(payload))
    _PROTOCOL_STDOUT.write("\n")
    _PROTOCOL_STDOUT.flush()


def err(msg: str, kind: str = "execute") -> None:
    """Emit a typed failure.

    `kind` is a CONTRACT, not a label: `runner.rs::classify_failure` maps the
    **suffix** onto a typed `BridgeError`. See `docs/BRIDGES.md`.

      *-not-installed                                  -> Unavailable
      *-unsupported-gate / -not-supported / -not-implemented -> CannotExpress
      anything else                                    -> Backend (a defect)
    """
    emit({"ok": False, "error": msg, "kind": kind})
