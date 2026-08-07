#!/usr/bin/env python3
"""Emit piquasso reference values for the CV backend's cross-check.

piquasso is an INDEPENDENT implementation of truncated-Fock CV simulation. Two
Aria backends agreeing may only mean they share a convention; agreement with an
implementation nobody here wrote is evidence. Same reasoning as the mandatory
Qiskit and PyMatching cross-checks.

Prints one JSON object per line: {"case", "cutoff", "mean_n"}.
"""
import json
import sys

import numpy as np
import piquasso as pq


def mean_n(prep, cutoff):
    """<n> on mode 0 after `prep` applies its instructions."""
    with pq.Program() as program:
        prep()
    simulator = pq.PureFockSimulator(
        d=1, config=pq.Config(cutoff=cutoff)
    )
    state = simulator.execute(program).state
    # Fock probabilities -> <n> = sum_k k p_k. Reading the density/probability
    # vector keeps this independent of piquasso's own expectation helpers.
    probs = np.asarray(state.fock_probabilities)
    ks = np.arange(len(probs))
    total = probs.sum()
    return float((ks * probs).sum() / total)


def main():
    cutoff = 20
    out = []

    for r in (0.1, 0.3, 0.5, 0.8):
        def prep(r=r):
            pq.Q(0) | pq.Vacuum()
            pq.Q(0) | pq.Squeezing(r=r)
        out.append({"case": f"squeezed_r={r}", "cutoff": cutoff,
                    "mean_n": mean_n(prep, cutoff)})

    for a in (0.5, 1.0, 1.2):
        def prep(a=a):
            pq.Q(0) | pq.Vacuum()
            pq.Q(0) | pq.Displacement(r=a, phi=0.0)
        out.append({"case": f"coherent_alpha={a}", "cutoff": cutoff,
                    "mean_n": mean_n(prep, cutoff)})

    # Kerr must not move <n>: n commutes with n^2.
    def prep_kerr():
        pq.Q(0) | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=0.0)
        pq.Q(0) | pq.Kerr(xi=0.37)
    out.append({"case": "coherent_alpha=1.0_then_kerr", "cutoff": cutoff,
                "mean_n": mean_n(prep_kerr, cutoff)})

    for row in out:
        print(json.dumps(row))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
