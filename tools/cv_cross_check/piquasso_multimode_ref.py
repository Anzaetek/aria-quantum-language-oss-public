# SPDX-License-Identifier: Apache-2.0
"""Generate the MULTI-MODE piquasso reference fixture.

Separate from `piquasso_ref.py` on purpose. That file's 22 single-mode rows are
validated and stable, and widening its schema to carry mode indices would put
every one of them through a format change to add coverage that does not touch
them. A second file costs a little duplication and risks nothing.

Emits one JSON object per line:

    {"meta": {...}}
    {"case": ..., "cutoff": C, "n_modes": M, "prep": [n0, n1, ...],
     "ops": [...], "probs": {"n0,n1": p, ...}}

`prep` is a Fock (number) basis state, which is what a photon-counting
experiment actually prepares and what Hong-Ou-Mandel needs. `probs` is keyed by
occupation string rather than by a flat index, so the two sides never have to
agree on an index ordering — the one convention most likely to differ silently
between two implementations, and the one that would make a mismatch look like a
physics bug.

Only occupations representable on BOTH sides are emitted: every mode strictly
below the cutoff.
"""
import json

import numpy as np
import piquasso as pq


def run(cutoff, n_modes, prep, ops):
    with pq.Program() as program:
        pq.Q() | pq.NumberState(prep)
        for o in ops:
            if o["op"] == "beamsplitter":
                pq.Q(o["a"], o["b"]) | pq.Beamsplitter(theta=o["theta"], phi=o["phi"])
            elif o["op"] == "phase_shift":
                pq.Q(o["mode"]) | pq.Phaseshifter(phi=o["phi"])
            elif o["op"] == "squeezing2":
                pq.Q(o["a"], o["b"]) | pq.Squeezing2(r=o["r"], phi=o["phi"])
            else:
                raise SystemExit(f"unknown op {o['op']}")
    sim = pq.PureFockSimulator(d=n_modes, config=pq.Config(cutoff=cutoff))
    state = sim.execute(program).state

    # `fock_probabilities_map` is keyed by OCCUPATION TUPLE, so neither side
    # ever has to agree on a flat index ordering. That is the single convention
    # most likely to differ silently between two implementations, and a
    # disagreement there would surface as a physics-looking mismatch. Taking the
    # map instead of the vector removes the question rather than answering it.
    probs = {}
    for occ, pr in state.fock_probabilities_map.items():
        occ = tuple(int(x) for x in occ)
        pr = float(pr)
        if pr > 1e-15:
            probs[",".join(map(str, occ))] = pr

    # Per-mode mean photon number through piquasso's OWN reduction machinery
    # (density-matrix trace-out), not through the probs map above. That keeps
    # this column an independent oracle for `expect_n(mode)` on ENTANGLED
    # states: the Rust side marginalises its own amplitudes, piquasso traces
    # out a density matrix, and the only thing the two share is the definition
    # of a mean.
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


def s2(a, b, r, phi=0.0):
    return {"op": "squeezing2", "a": a, "b": b, "r": r, "phi": phi}


def main():
    out = []
    HALF = np.pi / 4

    # --- Hong-Ou-Mandel, the canonical two-mode interference -----------------
    # P(1,1) = 0 at 50:50 by cancellation. This row is the one whose answer is
    # known from the physics rather than from either implementation.
    case("hom_50_50", 6, 2, [1, 1], [bs(0, 1, HALF)], out)
    # Away from balance the dip fills in as cos^2(2 theta) — so the corpus
    # cannot be satisfied by anything that merely empties |1,1>.
    for th in (0.15, 0.5, 1.2):
        case(f"two_photon_theta={th}", 6, 2, [1, 1], [bs(0, 1, th)], out)

    # --- single photon: the splitting ratio ---------------------------------
    for th in (0.0, 0.3, HALF, 1.1):
        case(f"one_photon_theta={th}", 6, 2, [1, 0], [bs(0, 1, th)], out)

    # --- non-zero phase, which is where conventions diverge ------------------
    # The single-mode corpus cannot test this: a phase on a beamsplitter is
    # observable in the OUTPUT PROBABILITIES only through interference between
    # two modes, so nothing before this file could have caught a phase-sign
    # error here.
    for ph in (0.7, -1.3):
        case(f"hom_phi={ph}", 6, 2, [1, 1], [bs(0, 1, HALF, ph)], out)
    case("two_photon_theta=0.9_phi=0.55", 6, 2, [1, 1], [bs(0, 1, 0.9, 0.55)], out)

    # --- a phase shift BETWEEN two beamsplitters: a Mach-Zehnder -------------
    # The composition where an interior phase is not cosmetic: it sets the
    # interference at the second splitter, so the output moves with it. Without
    # a case like this every phase_shift row would agree even if the phase
    # shifter were a no-op.
    for ph in (0.0, 0.6, np.pi / 2, np.pi):
        case(f"mach_zehnder_phi={ph}", 6, 2, [1, 0],
             [bs(0, 1, HALF), {"op": "phase_shift", "mode": 0, "phi": ph},
              bs(0, 1, HALF)], out)
    case("mach_zehnder_two_photon_phi=0.8", 6, 2, [1, 1],
         [bs(0, 1, HALF), {"op": "phase_shift", "mode": 0, "phi": 0.8},
          bs(0, 1, HALF)], out)

    # --- three modes: a chain, and a SPECTATOR that must not move ------------
    # Mode 2 is never named by any operation. If the fiber walk is wrong, a
    # spectator picks up amplitude, and only a >2-mode case can show it.
    case("three_mode_chain", 5, 3, [1, 1, 0],
         [bs(0, 1, HALF), bs(1, 2, 0.4)], out)
    case("three_mode_spectator_holds_photons", 5, 3, [1, 1, 2],
         [bs(0, 1, 0.7)], out)
    # Non-adjacent pair, skipping the middle mode entirely.
    case("three_mode_non_adjacent", 5, 3, [1, 0, 1],
         [bs(0, 2, 0.55, 0.3)], out)

    # --- two-mode squeezing --------------------------------------------------
    # r is kept SMALL and the cutoff HIGH on purpose: S2 populates |j,j> with
    # weight tanh^{2j}(r), and piquasso truncates by TOTAL photon number, so the
    # conservation gate (compared mass ~ 1 within 1e-9) bounds tanh(r)^{cutoff}.
    # A phase on a bare TMS vacuum is invisible in probabilities — hence the
    # composition cases, where a beamsplitter turns the phase into interference.
    # piquasso builds its S2 matrix truncated at the cutoff, so ITS boundary
    # levels carry ~1e-10 truncation error against the exact operator; the
    # 1e-12 comparison needs tanh(r)^cutoff well below 1e-12, hence r <= 0.1
    # at cutoff 14 (measured: r=0.15 at cutoff 12 fails by exactly that error).
    case("tms_vacuum", 14, 2, [0, 0], [s2(0, 1, 0.1)], out)
    case("tms_on_one_photon", 14, 2, [1, 0], [s2(0, 1, 0.1, -0.4)], out)
    case("tms_then_bs_phi=0.8", 14, 2, [0, 0],
         [s2(0, 1, 0.1, 0.8), bs(0, 1, HALF)], out)
    case("bs_then_tms", 14, 2, [1, 1],
         [bs(0, 1, 0.6, 0.2), s2(0, 1, 0.05, 0.6)], out)
    case("tms_three_mode_spectator", 12, 3, [0, 0, 1],
         [s2(0, 1, 0.08, 0.3)], out)
    # The RELATIVE phase between S2's raising and lowering halves. Two mutants
    # were measured surviving a first version of this corpus: (a) phi -> -phi
    # on the raise side only — on VACUUM the lowering half never acts, so every
    # vacuum-prep case degenerates to S2(r, -phi), whose state is the complex
    # conjugate, invisible to probabilities through real downstream ops and
    # (measured) even through a complex beamsplitter for symmetric |j,j>
    # content. The mutant only separates from every true S2 when the lowering
    # half has photons to act on AND the phase is non-zero — hence prep |1,1>
    # routed through a phased bs into a PHASED s2 (bs_then_tms above carries
    # phi=0.6 for exactly this reason), plus this direct case.
    case("tms_phased_on_photons", 14, 2, [1, 1],
         [s2(0, 1, 0.05, 0.7), bs(0, 1, HALF, 0.9)], out)

    print(json.dumps({"meta": {"piquasso": pq.__version__,
                               "numpy": np.__version__}}))
    for row in out:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
