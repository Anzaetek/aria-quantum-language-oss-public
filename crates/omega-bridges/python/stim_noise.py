#!/usr/bin/env python3
"""omega's `--noise` dict → Stim noise instructions, for the ppvm bridge.

This module reproduces **omega's** noise semantics, not Stim's defaults. The
reference is the per-gate channel both noisy in-process engines apply —
`omega-backend-statevector/src/noise.rs::apply_channel` and
`omega-backend-mps/src/sim.rs::apply_mps_channel` — and the data model in
`omega-core/src/noise.rs`. What those do, and what this emits for it:

====================  ===========================================  ==========================
omega key             omega applies                                emitted here
====================  ===========================================  ==========================
``depolarizing``      after EVERY gate op (incl. ``id``/``reset``),  ``DEPOLARIZE1(p_q) q`` per
                      on EACH qubit ``q`` it touched,                qubit, right after the gate
                      independently: with prob ``p_q`` one of
                      X/Y/Z uniformly. ``p_q = at_gate(q, gate)``
``pauli``             same place, per qubit: X/Y/Z w.p. px/py/pz    ``PAULI_CHANNEL_1(px,py,pz) q``
``phase_damping``     same place, per qubit: Z w.p. ``λ/2``         ``Z_ERROR(λ/2) q``
``readout`` /         at measurement: the RECORDED bit flips w.p.   ``M(p_q) q`` (record-only
``readout_flip``      ``p10``/``p01``; the post-measurement state    flip; state untouched)
                      is NOT flipped
``amplitude_damping`` quantum jump, non-unital                      **refused**
asymmetric readout    ``p10 != p01``                                **refused**
====================  ===========================================  ==========================

Two places where the obvious Stim spelling would be WRONG:

* **Not ``DEPOLARIZE2`` after a two-qubit gate.** omega kicks each qubit of a
  2q gate with its own independent one-qubit depolarizing draw (a product of
  two 1q channels, 9 of 15 non-identity Paulis reachable, and each qubit may
  carry a different per-qubit rate). ``DEPOLARIZE2(p)`` is the uniform
  15-Pauli two-qubit channel — a different channel with different marginals.
* **Not ``X_ERROR`` before ``M``.** That flips the qubit itself, so a later
  gate or measurement on the same qubit sees the flip. omega flips only the
  record (`maybe_flip_readout` after `projective_measure`). Stim's ``M(p)``
  is exactly the record-only flip, and ppvm honours it (``M(0.3) 0; M 0`` on
  ``|0>`` gives P(1) = 0.300 then 0.000, verified 2026-09-30). For a
  terminal measurement the two agree; for a repeated one they do not.

Rates resolve exactly as `omega_core::noise` does — see `rate_at` and
`Depolarizing.at_gate`, which mirror `Rate::at` and `Depolarizing::at_gate`
line for line and are pinned against the Rust in
`crates/omega-bridges/tests/ppvm_noise.rs`.

Empirical ground (ppvm 0.1.0 from git, 2026-09-30): `StimProgram.parse` +
`sample_stim` accept DEPOLARIZE1, DEPOLARIZE2, X_ERROR, Y_ERROR, Z_ERROR,
PAULI_CHANNEL_1, PAULI_CHANNEL_2 and ``M(p)``; they reject HERALDED_ERASE
("unsupported instruction") and E ("unknown instruction") at parse time.
"""

from __future__ import annotations

#: Keys this mapping expresses exactly. The runner's `capabilities` handshake
#: reports this tuple, so the two cannot drift.
SUPPORTED_KEYS = ("depolarizing", "pauli", "phase_damping", "readout", "readout_flip")

#: Keys omega recognises that the mapping refuses, with the reason. A key in
#: neither table is refused as unrecognised.
REFUSED_KEYS = {
    "amplitude_damping": (
        "amplitude damping is not a Pauli channel (its Kraus operators "
        "|0><1| and diag(1, sqrt(1-g)) are non-unital), and ppvm's generalized "
        "tableau only carries Pauli noise, so there is no Stim instruction "
        "that expresses it — a Pauli-twirled substitute would be a different "
        "channel"
    ),
}


class NoiseRefused(Exception):
    """A well-formed noise request the mapping cannot express exactly.

    Runners tag it `<backend>-noise-not-supported`, which
    `runner.rs::classify_failure` routes to `BridgeError::CannotExpress`.
    """


class NoiseInvalid(Exception):
    """A malformed noise request (wrong shape or type)."""


def _num(v, where: str) -> float:
    # `bool` is an `int` subclass; omega's serde `as_f64` rejects it.
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise NoiseInvalid(f"{where} must be a number, got {v!r}")
    return float(v)


def _rate(v, where: str):
    """A `Rate`: a float (uniform) or a list of floats (per qubit)."""
    if isinstance(v, list):
        return [_num(e, f"{where}[{i}]") for i, e in enumerate(v)]
    return _num(v, where)


def rate_at(rate, q: int) -> float:
    """`Rate::at`: uniform value, or the per-qubit entry (0.0 past the end)."""
    if isinstance(rate, list):
        return rate[q] if q < len(rate) else 0.0
    return rate


def _rate_is_zero(rate) -> bool:
    return all(v == 0.0 for v in rate) if isinstance(rate, list) else rate == 0.0


class Depolarizing:
    """Mirror of `omega_core::noise::Depolarizing`."""

    def __init__(self, v) -> None:
        self.pairs: dict[tuple[int, int], float] | None = None
        if isinstance(v, dict):
            for k in v:
                if k not in ("1q", "2q"):
                    raise NoiseInvalid(f"unknown depolarizing key {k!r} (recognised: 1q, 2q)")
            self.one_q = _rate(v["1q"], "depolarizing.1q") if "1q" in v else 0.0
            self.two_q = 0.0
            two = v.get("2q")
            if isinstance(two, dict):
                pairs = {}
                for k, val in two.items():
                    if k == "default":
                        self.two_q = _rate(val, "depolarizing.2q.default")
                        continue
                    a, sep, b = k.partition(",")
                    try:
                        if not sep:
                            raise ValueError
                        ia, ib = int(a.strip()), int(b.strip())
                        if ia < 0 or ib < 0:
                            raise ValueError
                    except ValueError:
                        raise NoiseInvalid(
                            f"depolarizing 2q pair key {k!r} must be \"i,j\""
                        ) from None
                    pairs[(min(ia, ib), max(ia, ib))] = _num(val, f"depolarizing.2q[{k!r}]")
                self.pairs = pairs or None
            elif two is not None:
                self.two_q = _rate(two, "depolarizing.2q")
        else:
            self.one_q = self.two_q = _rate(v, "depolarizing")

    def at_gate(self, q: int, qubits) -> float:
        """`Depolarizing::at_gate`: 1q → `one_q`; a 2q gate prefers its
        (min,max) pair entry; everything else (incl. arity 3) → `two_q`."""
        if len(qubits) <= 1:
            return rate_at(self.one_q, q)
        if len(qubits) == 2 and self.pairs is not None:
            key = (min(qubits[0], qubits[1]), max(qubits[0], qubits[1]))
            if key in self.pairs:
                return self.pairs[key]
        return rate_at(self.two_q, q)


def _fmt(p: float) -> str:
    # `repr` round-trips an IEEE double exactly through the text boundary.
    return repr(float(p))


class StimNoise:
    """A parsed omega noise dict, ready to emit Stim noise instructions.

    Construction validates and refuses; `after_gate` / `measure` only emit.
    """

    def __init__(self, noise) -> None:
        if not isinstance(noise, dict):
            raise NoiseInvalid(f"noise must be a JSON object, got {noise!r}")
        for k in noise:
            if k in REFUSED_KEYS:
                if not _rate_is_zero(_rate(noise[k], k)):
                    raise NoiseRefused(
                        f"the ppvm bridge refuses noise key `{k}`: {REFUSED_KEYS[k]}. "
                        f"Mapped keys: {', '.join(SUPPORTED_KEYS)}. Rerun without "
                        f"`{k}`, or use an in-process noisy backend "
                        "(`--backend statevector`/`mps` with --noise)"
                    )
            elif k not in SUPPORTED_KEYS:
                raise NoiseRefused(
                    f"the ppvm bridge does not map noise key `{k}` (request carried "
                    f"noise={noise!r}); mapped keys: {', '.join(SUPPORTED_KEYS)}"
                )
        if "readout" in noise and "readout_flip" in noise:
            raise NoiseInvalid('give either "readout" or "readout_flip", not both')

        self.depol = Depolarizing(noise["depolarizing"]) if "depolarizing" in noise else None

        self.pauli = None
        if "pauli" in noise:
            p = noise["pauli"]
            if not isinstance(p, dict):
                raise NoiseInvalid('"pauli" must be an object with X/Y/Z rates')
            for k in p:
                if k not in ("I", "X", "Y", "Z"):
                    raise NoiseInvalid(f"unknown pauli key {k!r} (recognised: I, X, Y, Z)")
            self.pauli = tuple(
                _rate(p[k], f"pauli.{k}") if k in p else 0.0 for k in ("X", "Y", "Z")
            )

        self.phase = _rate(noise["phase_damping"], "phase_damping") if "phase_damping" in noise else 0.0

        # Readout: symmetric only. `readout_flip` and a scalar / plain-array
        # `readout` are symmetric by construction; an array of {p10,p01}
        # objects is symmetric only if every entry has p10 == p01.
        self.readout = 0.0
        if "readout_flip" in noise:
            self.readout = _rate(noise["readout_flip"], "readout_flip")
        elif "readout" in noise:
            r = noise["readout"]
            if isinstance(r, list) and any(isinstance(e, dict) for e in r):
                sym = []
                for i, e in enumerate(r):
                    if not isinstance(e, dict):
                        raise NoiseInvalid("readout array must contain {p10, p01} objects")
                    for k in e:
                        if k not in ("p10", "p01"):
                            raise NoiseInvalid(f"unknown readout key {k!r} (recognised: p10, p01)")
                    p10 = _num(e["p10"], f"readout[{i}].p10") if "p10" in e else 0.0
                    p01 = _num(e["p01"], f"readout[{i}].p01") if "p01" in e else 0.0
                    if p10 != p01:
                        raise NoiseRefused(
                            f"the ppvm bridge refuses asymmetric `readout` (qubit {i}: "
                            f"p10={p10!r} != p01={p01!r}): Stim's record flip `M(p)` "
                            "flips with one probability whatever the true bit, so a "
                            "bit-dependent flip has no Stim instruction. Symmetric "
                            "readout (p10 == p01, or `readout_flip`) is mapped"
                        )
                    sym.append(p10)
                self.readout = sym
            else:
                self.readout = _rate(r, "readout")

    def after_gate(self, qubits) -> list[str]:
        """Noise lines for ONE gate application on `qubits`, in omega's
        per-qubit order (depolarizing, Pauli, phase damping)."""
        qubits = tuple(qubits)
        out: list[str] = []
        for q in qubits:
            if self.depol is not None:
                p = self.depol.at_gate(q, qubits)
                if p > 0.0:
                    out.append(f"DEPOLARIZE1({_fmt(p)}) {q}")
            if self.pauli is not None:
                px, py, pz = (rate_at(r, q) for r in self.pauli)
                if px + py + pz > 0.0:
                    out.append(f"PAULI_CHANNEL_1({_fmt(px)},{_fmt(py)},{_fmt(pz)}) {q}")
            lam = rate_at(self.phase, q)
            if lam > 0.0:
                out.append(f"Z_ERROR({_fmt(0.5 * lam)}) {q}")
        return out

    def measure(self, qubits) -> list[str]:
        """Measurement lines for `qubits`, in record order, with the
        record-only readout flip folded into `M(p)`."""
        out: list[str] = []
        for q in qubits:
            p = rate_at(self.readout, q)
            out.append(f"M({_fmt(p)}) {q}" if p > 0.0 else f"M {q}")
        return out
