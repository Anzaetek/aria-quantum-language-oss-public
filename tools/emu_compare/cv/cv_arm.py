# SPDX-License-Identifier: Apache-2.0
"""piquasso arm of the CV row (E7). piquasso 8.0.1, PureFockSimulator.

Its hot path is NumPy plus numba JIT, not a tuned CV simulator, so a timing
against it measures our Rust against NumPy/numba at a two-mode cutoff-14
size. The row publishes the timings and their floors and refuses a speed
classification. Timed call: `PureFockSimulator(...).execute(program)` and
the per-mode <n> through piquasso's own reduction, as the fixture generator
(tools/cv_cross_check/piquasso_multimode_ref.py) does it.
"""

import json
import sys
import time

import numpy
import piquasso as pq

PINNED = "8.0.1"


def program(prep, ops):
    with pq.Program() as p:
        pq.Q() | pq.NumberState(prep)
        for o in ops:
            if o["op"] == "beamsplitter":
                pq.Q(o["a"], o["b"]) | pq.Beamsplitter(theta=o["theta"], phi=o["phi"])
            elif o["op"] == "squeezing2":
                pq.Q(o["a"], o["b"]) | pq.Squeezing2(r=o["r"], phi=o["phi"])
            elif o["op"] == "phase_shift":
                pq.Q(o["mode"]) | pq.Phaseshifter(phi=o["phi"])
            else:
                raise ValueError(f"op {o['op']}")
    return p


def run(cutoff, n_modes, prep, ops):
    prog = program(prep, ops)
    t = time.perf_counter()
    state = pq.PureFockSimulator(d=n_modes, config=pq.Config(cutoff=cutoff)).execute(prog).state
    mean_n = []
    for mode in range(n_modes):
        red = state.reduced(modes=(mode,))
        mean_n.append(float(sum(occ[0] * float(pr) for occ, pr in red.fock_probabilities_map.items())))
    return mean_n, time.perf_counter() - t


def main():
    for line in sys.stdin:
        req = json.loads(line)
        try:
            if req["op"] == "hello":
                if pq.__version__ != PINNED:
                    raise RuntimeError(f"piquasso {pq.__version__}, pinned {PINNED}")
                out = {"versions": {"piquasso": pq.__version__, "numpy": numpy.__version__},
                       "simulator": "PureFockSimulator"}
            elif req["op"] == "run":
                m, s = run(req["cutoff"], req["n_modes"], req["prep"], req["ops"])
                out = {"mean_n": m, "seconds": s}
            elif req["op"] == "floor":
                m, s = run(2, 1, [0], [])
                if abs(m[0]) > 1e-12:
                    raise RuntimeError(f"floor <n> of vacuum is {m[0]}")
                out = {"seconds": s}
            elif req["op"] == "quit":
                break
            else:
                raise ValueError(req["op"])
            sys.stdout.write(json.dumps({"ok": True, **out}) + "\n")
        except Exception as e:
            sys.stdout.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
