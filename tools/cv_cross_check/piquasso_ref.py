#!/usr/bin/env python3
"""Emit piquasso reference values for the CV backend's cross-check.

piquasso is an INDEPENDENT implementation of truncated-Fock CV simulation. Two
Aria backends agreeing may only mean they share a convention; agreement with an
implementation nobody here wrote is evidence. Same reasoning as the mandatory
Qiskit and PyMatching cross-checks.

## Shape of the output

One JSON object per line. A leading meta line stamps the versions, so a
regenerated fixture is attributable to what produced it:

    {"meta": {"piquasso": "8.0.1", "numpy": "2.4.6"}}
    {"case": ..., "cutoff": 20, "ops": [...], "probs": [...], "mean_n": f}

`ops` is the circuit as **structured data**, not prose:

    {"op": "vacuum"}
    {"op": "squeeze",     "r": 0.3}
    {"op": "displace",    "re": 0.7071, "im": 0.7071}   # CARTESIAN, per K15
    {"op": "kerr",        "chi": 0.37}
    {"op": "phase_shift", "phi": 0.7}

The Rust side interprets `ops` and never parses the case name. That matters:
if the two sides derive the circuit from different sources, they can silently
drift apart and still agree — the failure this whole file exists to prevent.
Displacement is emitted Cartesian because that is what the Aria surface fixed on
(K15), and converted to piquasso's polar form here, in ONE place.

The full Fock probability vector is emitted, not just `<n>`. A scalar `<n>`
collides: two visibly different states can share it, so agreeing on `<n>` is weak
evidence that the *state* is right. The vector is what catches a gate applied
with the wrong sign, the wrong scaling, or to the wrong ladder rung.

Both `probs` and `amps` are emitted. The amplitudes are the load-bearing ones:
Kerr and the phase shifter are diagonal, so they move phases and nothing else,
and against a probability vector a no-op implementation of either is
indistinguishable from a correct one. `state_vector` is public API on piquasso's
`PureFockState` (verified on 8.0.1), so this needs no private attribute.

## What this comparison can and cannot see

* **Global phase is quotiented out**, as it must be — it is unobservable, and
  the two libraries have no reason to agree on it. Relative phase between Fock
  levels IS compared, which is what makes the diagonal gates real tests.
* **Single mode (d=1) only.** Not a choice — `omega-backend-cv`'s `FockState`
  IS single-mode: no beamsplitter, no multimode state. Mode mixing is exactly
  where independent CV implementations diverge on convention, so the most
  valuable comparison is the one that cannot be written yet. Recorded here
  rather than quietly omitted.
"""
import json
import math
import sys

import numpy as np
import piquasso as pq

CUTOFF = 20


def apply_ops(ops):
    """Translate the structured recipe into piquasso instructions."""
    for o in ops:
        kind = o["op"]
        if kind == "vacuum":
            pq.Q(0) | pq.Vacuum()
        elif kind == "squeeze":
            pq.Q(0) | pq.Squeezing(r=o["r"])
        elif kind == "displace":
            # Cartesian -> polar happens HERE and nowhere else.
            re, im = o["re"], o["im"]
            pq.Q(0) | pq.Displacement(r=math.hypot(re, im),
                                      phi=math.atan2(im, re))
        elif kind == "kerr":
            pq.Q(0) | pq.Kerr(xi=o["chi"])
        elif kind == "phase_shift":
            pq.Q(0) | pq.Phaseshifter(phi=o["phi"])
        else:
            raise ValueError(f"unknown op {kind!r}")


def case(name, ops, out, cutoff=CUTOFF):
    with pq.Program() as program:
        apply_ops(ops)
    simulator = pq.PureFockSimulator(d=1, config=pq.Config(cutoff=cutoff))
    state = simulator.execute(program).state

    probs = np.asarray(state.fock_probabilities, dtype=float)
    # piquasso may return fewer entries than the cutoff when trailing levels are
    # unpopulated. Pad so the Rust side compares index-for-index without having
    # to guess whether a short vector means "zero" or "truncated differently".
    if len(probs) < cutoff:
        probs = np.concatenate([probs, np.zeros(cutoff - len(probs))])
    probs = probs[:cutoff]

    # AMPLITUDES, not just probabilities. `state_vector` is public API on
    # piquasso's PureFockState (checked against 8.0.1), so this costs no
    # private-attribute access.
    #
    # This is what makes the diagonal gates testable at all. Kerr and the phase
    # shifter are exp(i*chi*n^2) and exp(i*phi*n): they move ONLY phases, so
    # against a probability vector they are invisible and a no-op implementation
    # of either would sail through. Comparing amplitudes (up to one global
    # phase, which is unobservable and must be quotiented out) is what turns
    # those cases from decoration into a real check.
    amps = np.asarray(state.state_vector, dtype=complex)
    if len(amps) < cutoff:
        amps = np.concatenate([amps, np.zeros(cutoff - len(amps), dtype=complex)])
    amps = amps[:cutoff]

    # NORMALISATION — the single most important line in this file.
    #
    # piquasso returns RAW truncated probabilities: at r=0.8, cutoff 20,
    # `sum(probs)` is 0.999936664825, not 1. Aria renormalises by the represented
    # mass. Comparing the two conventions directly produces a spurious
    # disagreement of exactly (1 - sum) * p_max — 4.736e-05 at r=0.8, 3.434e-08
    # at r=0.5.
    #
    # Those two numbers were previously reported by the cross-check as
    # "reconciled by the truncation budget" and explained as a genuine physical
    # difference ("we take closed-form amplitudes and cut, piquasso applies an
    # already-truncated operator"). That explanation was WRONG. The underlying
    # amplitudes agree to 1e-16; the entire gap was this normalisation.
    #
    # So both sides are put on Aria's convention HERE, once, and the comparison
    # tightens from a leak-budget-sized tolerance (~1e-4) to a
    # floating-point-sized one (~1e-14) — about ten orders of magnitude more
    # discriminating.
    total = probs.sum()
    if total > 0:
        probs = probs / total
        amps = amps / np.sqrt(total)

    ks = np.arange(len(probs))
    mean_n = float((ks * probs).sum()) if total > 0 else 0.0

    out.append({
        "case": name,
        "cutoff": cutoff,
        "ops": ops,
        "probs": [float(p) for p in probs],
        "amps": [[float(a.real), float(a.imag)] for a in amps],
        "mean_n": mean_n,
    })


VAC = {"op": "vacuum"}


def main():
    out = []

    # --- anchors: closed-form states, no gate machinery involved -------------
    case("vacuum", [VAC], out)

    for r in (0.1, 0.3, 0.5, 0.8):
        case(f"squeezed_r={r}", [VAC, {"op": "squeeze", "r": r}], out)

    for a in (0.5, 1.0, 1.2):
        case(f"coherent_re={a}",
             [VAC, {"op": "displace", "re": a, "im": 0.0}], out)

    # Complex alpha. The Aria surface is CARTESIAN (K15) and piquasso is polar,
    # so this is where a re/im-vs-r/phi mix-up would show. Be precise about WHY
    # it shows: not because the phase is observable — `fock_probabilities` is
    # phase-blind — but because a mix-up changes |alpha|, and |alpha| is what
    # the probability vector depends on. A phase-ONLY convention difference
    # stays invisible here; see the module docstring.
    case("coherent_re=0.7_im=0.7",
         [VAC, {"op": "displace", "re": 0.7, "im": 0.7}], out)
    case("coherent_re=0.4_im=-0.9",
         [VAC, {"op": "displace", "re": 0.4, "im": -0.9}], out)

    # --- diagonal gates: exact on the truncated space -----------------------
    # Kerr and the phase shifter are both diagonal in Fock, so they must leave
    # the probability vector UNCHANGED. That makes them a sharp test: any
    # movement at all is a bug, with no truncation excuse available. It also
    # means these cases cannot distinguish a correct chi from a wrong one —
    # only that neither side leaks probability where it must not.
    case("coherent_re=1.0_then_kerr",
         [VAC, {"op": "displace", "re": 1.0, "im": 0.0},
          {"op": "kerr", "chi": 0.37}], out)
    case("squeezed_r=0.5_then_kerr",
         [VAC, {"op": "squeeze", "r": 0.5}, {"op": "kerr", "chi": 0.9}], out)

    for phi in (0.7, -1.3):
        case(f"coherent_re=1.0_then_phase={phi}",
             [VAC, {"op": "displace", "re": 1.0, "im": 0.0},
              {"op": "phase_shift", "phi": phi}], out)

    case("squeezed_r=0.5_then_phase=0.4",
         [VAC, {"op": "squeeze", "r": 0.5},
          {"op": "phase_shift", "phi": 0.4}], out)

    # Displacement AFTER a phase shift is the one composition here where the
    # phase is not merely cosmetic: it rotates the state the displacement then
    # acts on, so |alpha_effective| changes and the probability vector moves.
    # Without a case like this, every phase-shift row above would agree even if
    # phase_shift were a no-op.
    case("squeezed_r=0.5_phase_then_displace",
         [VAC, {"op": "squeeze", "r": 0.5},
          {"op": "phase_shift", "phi": 0.9},
          {"op": "displace", "re": 0.6, "im": 0.2}], out)
    case("coherent_re=1.2_then_phase_then_kerr",
         [VAC, {"op": "displace", "re": 1.2, "im": 0.0},
          {"op": "phase_shift", "phi": 0.55},
          {"op": "kerr", "chi": 0.21}], out)

    print(json.dumps({"meta": {"piquasso": pq.__version__,
                               "numpy": np.__version__}}))
    for row in out:
        print(json.dumps(row))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
