# SPDX-License-Identifier: Apache-2.0
"""A client for the omega-server template routes.

`PLAN-SIX-PROGRAMMES.md` P5's fourth item. The two template routes exist
(`0132`, `0133`) but a route nobody can call is a route nobody uses, and the
payload argument for the shape is only realised by a caller that actually sends
the compact form.

# Python, and stdlib only — both deliberate

**Python** because the plan says so and the reasoning holds: `bindings/aria-py`
has no HTTP client of any kind — no `reqwest`, no `hyper`, no `tokio` — and
adding an async runtime to a workspace that has none, for a client, is a large
dependency for a small job. A Python client also never holds the GIL during I/O,
so the question P3 raised does not arise here at all.

**Stdlib only** because `requests` is not installed in any of this repo's venvs
(checked: `.venv-qiskit`, `.venv-piquasso`, `bindings/aria-py/.venv`) and K13
says a headless machine must be able to run everything. `urllib.request` is
always there, so this imports in bare `python3` with nothing installed.

# What it does NOT do

No retries, no connection pooling, no async. Those belong to whatever the caller
already uses for the rest of its HTTP, and inventing a policy here would be
guessing. A 429 is surfaced as a typed error carrying the server's message so the
caller can implement its own backoff — see `Busy`.

Refused scope per the plan and absent here: per-row cancellation, durable
batches, cluster manager.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Sequence


class OmegaError(RuntimeError):
    """Any non-2xx response, carrying the server's own message.

    The server always states numbers in its refusals ("needs 32 GB, 12 GB
    free"), so the message is forwarded verbatim rather than summarised — a
    caller can act on the former and not on the latter.
    """

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class Busy(OmegaError):
    """429 — the request would fit but not right now.

    Its own class because the correct response DIFFERS: a 429 invites a retry
    and a 413 never will. Collapsing them is how a retry loop either gives up on
    recoverable work or spins for ever on impossible work.
    """


class TooLarge(OmegaError):
    """413 — the request cannot run on this host at any time."""


class OmegaClient:
    """Talks to one omega-server."""

    def __init__(self, url: str, token: str, *, timeout: float = 300.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    # -- transport ---------------------------------------------------------
    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                msg = json.loads(raw).get("error", raw)
            except json.JSONDecodeError:
                msg = raw
            if e.code == 429:
                raise Busy(e.code, msg) from None
            if e.code == 413:
                raise TooLarge(e.code, msg) from None
            raise OmegaError(e.code, msg) from None

    # -- payload construction ---------------------------------------------
    @staticmethod
    def _template_body(
        circuit: dict,
        observable: str,
        params: Sequence[str],
        rows: Sequence[Sequence[float]],
    ) -> dict:
        """Validate CLIENT-SIDE before spending a round trip.

        The server checks all of this and refuses properly; doing it here too
        turns a network round trip into an immediate exception with the same
        information. The checks are the server's, restated — a ragged matrix
        would bind different rows to different meanings, and a name that is not
        a free symbol is a caller error either way.
        """
        if not rows:
            raise ValueError("rows is empty — nothing to evaluate")
        width = len(params)
        if width == 0:
            raise ValueError("params is empty — a template needs free symbols")
        for i, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(
                    f"rows[{i}] has {len(row)} values but params names {width} — "
                    f"a ragged matrix would bind different rows to different meanings"
                )
        return {
            "circuit": circuit,
            "observable": observable,
            "params": list(params),
            "rows": [list(r) for r in rows],
        }

    # -- routes ------------------------------------------------------------
    def expectation_template(
        self,
        circuit: dict,
        observable: str,
        params: Sequence[str],
        rows: Sequence[Sequence[float]],
    ) -> list[float]:
        """One circuit, an `N × P` matrix, `N` expectation values in row order."""
        body = self._template_body(circuit, observable, params, rows)
        out = self._post("/v1/quantum/expectation_template", body)
        values = out["values"]
        if len(values) != len(rows):
            raise OmegaError(
                200,
                f"server returned {len(values)} values for {len(rows)} rows — "
                f"the row/result correspondence is the contract of this route",
            )
        return values

    def gradient_template(
        self,
        circuit: dict,
        observable: str,
        params: Sequence[str],
        rows: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """`N × P` gradient matrix, columns aligned to `params`.

        The server echoes `params` once; this asserts it came back unchanged. A
        server that reordered the columns would otherwise be indistinguishable
        from one that did not, and the numbers alone cannot show it — which is
        exactly the failure the route's own alignment test was written for.
        """
        body = self._template_body(circuit, observable, params, rows)
        out = self._post("/v1/quantum/gradient_template", body)
        echoed = out.get("params")
        if echoed is not None and list(echoed) != list(params):
            raise OmegaError(
                200,
                f"server echoed params {echoed} but the request declared "
                f"{list(params)} — the gradient columns cannot be trusted",
            )
        grads = out["gradients"]
        if len(grads) != len(rows):
            raise OmegaError(
                200, f"server returned {len(grads)} gradient rows for {len(rows)} rows"
            )
        for i, g in enumerate(grads):
            if len(g) != len(params):
                raise OmegaError(
                    200,
                    f"gradients[{i}] has {len(g)} columns but params names "
                    f"{len(params)}",
                )
        return grads


def ry_ansatz(n: int, layers: int) -> tuple[dict, list[str]]:
    """A hardware-efficient `Ry` + `CX`-ring ansatz, and its parameter names.

    Here so the client is usable from a REPL without hand-writing an IR, and so
    the payload measurement and the live test describe the same circuit.
    """
    ops: list[dict[str, Any]] = []
    names: list[str] = []
    for l in range(layers):
        for q in range(n):
            name = f"t{l}_{q}"
            names.append(name)
            # `OmegaParam` is `#[serde(untagged)]`: a concrete value is a BARE
            # number and a symbol is `{"symbol": name}` — not `{"Symbol": {...}}`.
            # Getting this wrong is a 422 from the server, which is how it was
            # found; the live test caught what a mock could not have.
            ops.append({"gate": "Ry", "qubits": [q],
                        "params": [{"symbol": name}],
                        "classical_bit": None, "condition": None})
        for q in range(n):
            ops.append({"gate": "CX", "qubits": [q, (q + 1) % n], "params": [],
                        "classical_bit": None, "condition": None})
    circuit = {"num_qubits": n, "num_classical_bits": 0, "is_photonic": False,
               "mid_circuit_mode": "Skip", "backend": "Statevector", "ops": ops}
    return circuit, names
