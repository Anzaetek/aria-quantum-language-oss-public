#!/usr/bin/env python3
"""Pin the omega <-> Perceval convention mapping at the MATRIX level.

Run:
    ./.venv-perceval/bin/python -m pytest tests/test_perceval_conventions.py -q
or standalone:
    ./.venv-perceval/bin/python tests/test_perceval_conventions.py

## Why this file exists

The `bs_rx` -> Perceval mapping was wrong for two years' worth of commits and
nothing caught it. The justification in the code read "gives the same
transmission/reflection split" and "confirmed by HOM smoke test in
tests/cross_backend.rs" — that is, it was checked on MAGNITUDES, by a test that
is structurally phase-insensitive. The Hong-Ou-Mandel dip is a statement about
|amplitude|^2; a beam splitter with the right magnitudes and the wrong phases
reproduces it exactly.

Measured error of the old mapping against omega's own matrix: **0.798** at
theta=0.6, **1.0** at 50/50.

It stayed invisible downstream because for meshes of plain beam splitters there
is a mode-local gauge (a phase per coupled pair) under which all Fock
distributions still agree. Polarization support destroys that gauge — a
polarizing beam splitter demands an exact, ungauged swap on the same mode pairs
a spatial splitter couples — so this had to be fixed before any polarized
cross-check could mean anything.

The lesson encoded here: compare MATRICES, not just distributions. A
distribution test can be satisfied by a whole gauge orbit of wrong matrices.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# K13: an external tool that is not installed must SKIP CLEANLY, never error.
# A bare `import perceval` made pytest fail at COLLECTION time, which takes the
# whole test session down — including tests that need no Perceval at all. That
# turned "Perceval is not installed here" into "the python test stage is red",
# which is the opposite of the clean-skip contract.
pcvl = pytest.importorskip("perceval", reason="perceval-quandela not installed")
_components = pytest.importorskip(
    "perceval.components", reason="perceval-quandela not installed"
)
BS, PS, HWP, PBS = (
    _components.BS,
    _components.PS,
    _components.HWP,
    _components.PBS,
)

TOL = 1e-14


def omega_bs_rx(theta, phi):
    """omega's `bs_rx(theta, phi)`.

    Source of truth: omega-backend-photonics/src/components.rs, the
    `apply_beam_splitter_rx` doc block.
    """
    return np.array([
        [np.cos(theta), -np.exp(1j * phi) * np.sin(theta)],
        [np.exp(-1j * phi) * np.sin(theta), np.cos(theta)],
    ])


def perceval_bs_rx(theta, phi):
    """The mapping perceval_runner.py uses. Keep these two in lockstep."""
    c = pcvl.Circuit(2)
    if phi != 0.0:
        c.add(0, PS(phi=-phi))
    c.add((0, 1), BS.Ry(theta=2.0 * theta))
    if phi != 0.0:
        c.add(0, PS(phi=phi))
    return np.array(c.compute_unitary(False))


def test_bs_rx_matches_across_theta_and_phi():
    worst = 0.0
    for theta in (0.0, 0.35, 0.6, np.pi / 4, 1.2, np.pi / 2):
        for phi in (0.0, 0.4, -0.7, 1.1, -2.3, np.pi):
            err = np.abs(omega_bs_rx(theta, phi) - perceval_bs_rx(theta, phi)).max()
            worst = max(worst, err)
            assert err < TOL, f"bs_rx({theta}, {phi}) mismatch: {err:.3e}"
    print(f"  bs_rx: worst {worst:.3e} over 36 (theta, phi) pairs")


def test_the_old_mapping_would_fail_this():
    """Guard the guard.

    A pin that passes against the BROKEN mapping would be worthless, so assert
    the old convention is actually rejected. Without this, someone could revert
    the fix and the suite above might still be green for a subtler reason.
    """
    bad = np.array(BS(theta=2 * 0.6, phi_tr=0.0).compute_unitary(False))
    err = np.abs(omega_bs_rx(0.6, 0.0) - bad).max()
    assert err > 0.5, (
        "the old BS(Rx) mapping now AGREES with omega's matrix — either "
        "Perceval changed its conventions or this test has stopped testing "
        f"anything (err {err:.3e})"
    )
    print(f"  old mapping correctly rejected: err {err:.3e}")


def test_phi_tr_is_not_a_leg_phase():
    """The specific mistake, pinned.

    Passing phi as `phi_tr` is exact at phi=0 and wrong everywhere else, which
    is precisely why the bug survived: every fixture used phi=0.
    """
    for phi in (0.4, -0.7, 1.1):
        naive = np.array(BS.Ry(theta=2 * 0.6, phi_tr=phi).compute_unitary(False))
        err = np.abs(omega_bs_rx(0.6, phi) - naive).max()
        assert err > 0.1, f"phi_tr={phi} unexpectedly matches ({err:.3e})"
    # ...and exact at zero, which is the trap.
    zero = np.array(BS.Ry(theta=2 * 0.6, phi_tr=0.0).compute_unitary(False))
    assert np.abs(omega_bs_rx(0.6, 0.0) - zero).max() < TOL
    print("  phi_tr trap pinned: exact at phi=0, wrong for phi != 0")


def test_hwp_convention_including_global_phase():
    """Perceval's HWP carries a global `i`, and it is NOT ignorable.

    A polarization element acts on a SUBSET of a larger interferometer's modes,
    so a global factor on that block is a RELATIVE phase between paths. Aria
    adopts Perceval's convention verbatim (FIXES_PLAN.md I1b) rather than the
    textbook i-less form, so that cross-checks compare raw matrices with no
    fudge factor in between.
    """
    for th in (0.0, np.pi / 8, 0.37, np.pi / 4):
        c2, s2 = np.cos(2 * th), np.sin(2 * th)
        textbook = np.array([[c2, s2], [s2, -c2]], dtype=complex)
        perc = np.array(HWP(th).compute_unitary(False))

        assert np.abs(1j * textbook - perc).max() < TOL, f"HWP({th}) not i*textbook"
        # And the i is not a no-op: determinants differ in sign.
        assert abs(np.linalg.det(textbook) - np.linalg.det(perc)) > 1.5, (
            "textbook and Perceval HWP have the same determinant — the global "
            "phase claim in FIXES_PLAN.md I0 would then be wrong"
        )
    print("  HWP: Perceval = i * textbook, det differs by sign (i is real)")


def test_pbs_swaps_H_and_transmits_V():
    """Pinned because it is the opposite of how it is usually stated.

    Ordering is interleaved [a_H, a_V, b_H, b_V].
    """
    u = np.array(PBS().compute_unitary(use_polarization=True))
    expected = np.zeros((4, 4), dtype=complex)
    expected[0, 2] = 1.0  # out a_H <- in b_H
    expected[2, 0] = 1.0  # out b_H <- in a_H
    expected[1, 1] = 1.0  # out a_V <- in a_V
    expected[3, 3] = 1.0  # out b_V <- in b_V
    err = np.abs(u - expected).max()
    assert err < TOL, f"PBS is not the expected H-swap permutation: {err:.3e}\n{u}"
    print("  PBS: swaps H between spatial modes, transmits V (NOT the textbook phrasing)")


def test_pbs_swap_block_is_expressible_in_omega_ops():
    """PS(pi) . BSrx(pi/2, .) is an exact 2x2 swap.

    det(swap) = -1 while det(BSrx) = +1, so the phase shifter is not cosmetic —
    it is what supplies the sign. The first draft of the plan asserted "PBS is a
    permutation" with no measurement at all; this is that measurement.
    """
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    found = None
    for phi_bs in (0.0, np.pi):
        for ps_mode in (0, 1):
            d = [1.0, 1.0]
            d[ps_mode] = np.exp(1j * np.pi)
            cand = np.diag(d) @ omega_bs_rx(np.pi / 2, phi_bs)
            if np.abs(cand - swap).max() < TOL:
                found = (phi_bs, ps_mode)
    assert found is not None, "no PS(pi)+BSrx(pi/2) combination yields an exact swap"
    print(f"  PBS swap block = PS(pi) on mode {found[1]} . bs_rx(pi/2, {found[0]:.3f})")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Perceval {pcvl.__version__} convention pins\n")
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
            failed += 1
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
