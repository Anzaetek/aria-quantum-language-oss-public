# SPDX-License-Identifier: Apache-2.0
"""Generate the LOSS-channel piquasso reference fixture.

A third file beside the single-mode and multimode ones, for the same reason
those two are separate: loss cases need piquasso's DENSITY-MATRIX simulator
(`FockSimulator` — `PureFockSimulator` refuses `Attenuator`, measured), while
every existing case runs pure. Widening either existing schema to carry a
simulator switch would put validated rows through a change to gain nothing.

The comparison surface is the MARGINAL distribution over the system modes and
the per-mode means:

* piquasso applies `Attenuator` natively on a density matrix;
* the Rust side dilates — one vacuum ancilla per loss, beamsplitter with
  `theta = acos(sqrt(eta))`, then `marginal_probs` over the original modes.

Those are two different representations of the same channel, which is what
makes the agreement worth measuring. `probs` is keyed by occupation string
over the SYSTEM modes only; the ancillas are the Rust side's business.

Preparation is `Vacuum` + `Create`^n per mode (FockSimulator has no
NumberState; Create is unnormalised, hence the explicit `normalize()`).
"""
import json

import numpy as np
import piquasso as pq


def run(cutoff, n_modes, prep, ops):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()
        for mode, n in enumerate(prep):
            for _ in range(n):
                pq.Q(mode) | pq.Create()
        for o in ops:
            if o["op"] == "beamsplitter":
                pq.Q(o["a"], o["b"]) | pq.Beamsplitter(theta=o["theta"], phi=o["phi"])
            elif o["op"] == "phase_shift":
                pq.Q(o["mode"]) | pq.Phaseshifter(phi=o["phi"])
            elif o["op"] == "loss":
                pq.Q(o["mode"]) | pq.Attenuator(
                    theta=float(np.arccos(np.sqrt(o["eta"])))
                )
            else:
                raise SystemExit(f"unknown op {o['op']}")
    sim = pq.FockSimulator(d=n_modes, config=pq.Config(cutoff=cutoff))
    state = sim.execute(program).state
    state.normalize()

    probs = {}
    for occ, pr in state.fock_probabilities_map.items():
        occ = tuple(int(x) for x in occ)
        pr = float(pr)
        if pr > 1e-15:
            probs[",".join(map(str, occ))] = pr

    mean_n = []
    for mode in range(n_modes):
        reduced = state.reduced(modes=(mode,))
        mean_n.append(float(sum(
            occ[0] * float(pr)
            for occ, pr in reduced.fock_probabilities_map.items()
        )))
    return probs, mean_n


def case(name, cutoff, n_modes, prep, ops, out):
    probs, mean_n = run(cutoff, n_modes, prep, ops)
    out.append({
        "case": name,
        "cutoff": cutoff,
        "n_modes": n_modes,
        "prep": prep,
        "ops": ops,
        "probs": probs,
        "mean_n": mean_n,
    })


def bs(a, b, theta, phi=0.0):
    return {"op": "beamsplitter", "a": a, "b": b, "theta": theta, "phi": phi}


def loss(mode, eta):
    return {"op": "loss", "mode": mode, "eta": eta}


def main():
    out = []
    HALF = np.pi / 4

    # The Bernoulli anchor: the one row whose answer is known from physics.
    case("loss_single_photon", 6, 2, [1, 0], [loss(0, 0.7)], out)
    # Two photons: binomial thinning, eta^2 / 2eta(1-eta) / (1-eta)^2.
    case("loss_two_photons", 6, 2, [2, 0], [loss(0, 0.5)], out)
    # Loss on ONE ARM of an entangled state — the case a product-state test
    # cannot see, and the one that catches a marginal walking wrong strides.
    case("loss_after_hom", 6, 2, [1, 1], [bs(0, 1, HALF), loss(0, 0.6)], out)
    # Both modes lossy, different rates.
    case("loss_both_modes", 6, 2, [1, 1], [loss(0, 0.8), loss(1, 0.3)], out)
    # Channel then unitary: the beamsplitter acts on a MIXED state, which the
    # dilation represents purely with the ancilla riding along.
    case("loss_then_bs", 6, 2, [1, 0], [loss(0, 0.5), bs(0, 1, 0.6, 0.2)], out)
    # A spectator mode that must not move.
    case("loss_three_mode_spectator", 5, 3, [1, 0, 2], [loss(0, 0.4)], out)

    print(json.dumps({"meta": {"piquasso": pq.__version__,
                               "numpy": np.__version__}}))
    for row in out:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
