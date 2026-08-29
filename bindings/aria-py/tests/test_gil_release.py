# SPDX-License-Identifier: Apache-2.0
"""**Does a simulation actually release the GIL?**

A measurement, not an assertion about the source. `Python::allow_threads` can be
present and still buy nothing — if the release was added to a wrapper rather
than to the call that spends the time, or if the boxed backend quietly loses its
`Send + Sync` bound and someone "fixes" it by removing the release.

# Why the test is a Python-side counter and NOT a wall-clock speedup

The obvious test — same work on two threads vs one, expect ~2x — does not work
here, and the first draft of this file shipped it and was wrong. The CPU
statevector backend is **already rayon-parallel internally**: `sim.rs` applies
every gate through `par_chunks_mut` and batches rows through `par_iter`. Two
Python threads therefore contend for one rayon pool, and the measured "speedup"
was 1.49x, then 1.15x on the identical build — noise of the same size as the
effect. It reported 1.16x for a deliberately GIL-HOLDING mutant and 1.15x for
the real thing, so it separated nothing.

What the GIL release actually buys on this backend is not more cores for Aria —
rayon already took those. It is that **the rest of the Python program keeps
running** during a simulation: other threads doing I/O, feeding a data loader,
serving a request, or driving a progress bar. That is a binary property, so it
is measured as one.

A pure-Python thread increments a counter. If the GIL is held across the
simulation that thread cannot execute a single bytecode for the whole call and
the counter is frozen; if it is released the counter climbs. The gap between
"zero" and "thousands" is not a threshold anyone has to tune.
"""
import math
import threading
import time

import aria_py

# Big enough that one call takes tens of milliseconds — far longer than the
# ~5 ms interpreter switch interval, so a running spinner gets many turns.
SRC = """circuit W() {
  qreg q[14]
  let t = symbolic[14]
  apply RY(t[0]) on q[0]
  apply RY(t[1]) on q[1]
  apply RY(t[2]) on q[2]
  apply RY(t[3]) on q[3]
  apply RY(t[4]) on q[4]
  apply RY(t[5]) on q[5]
  apply RY(t[6]) on q[6]
  apply RY(t[7]) on q[7]
  apply RY(t[8]) on q[8]
  apply RY(t[9]) on q[9]
  apply RY(t[10]) on q[10]
  apply RY(t[11]) on q[11]
  apply RY(t[12]) on q[12]
  apply RY(t[13]) on q[13]
  apply CX on q[0], q[1]
  apply CX on q[1], q[2]
  apply CX on q[2], q[3]
  apply CX on q[3], q[4]
  apply CX on q[4], q[5]
  apply CX on q[5], q[6]
  apply CX on q[6], q[7]
  apply CX on q[7], q[8]
  apply CX on q[8], q[9]
  apply CX on q[9], q[10]
  apply CX on q[10], q[11]
  apply CX on q[11], q[12]
  apply CX on q[12], q[13]
}"""

ROWS = [[0.1 * (i + j) for i in range(14)] for j in range(16)]


class Spinner:
    """A pure-Python thread whose only job is to prove it got scheduled."""

    def __init__(self):
        self.count = 0
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop:
            self.count += 1

    def __enter__(self):
        self._t.start()
        # Let it get going, so a zero later means blocked rather than not-started.
        time.sleep(0.05)
        assert self.count > 0, "spinner never ran even before the call"
        return self

    def __exit__(self, *exc):
        self._stop = True
        self._t.join(timeout=2.0)
        return False


def _measure(call):
    """Bytecodes the spinner executed during `call`, and how long it took."""
    with Spinner() as sp:
        before = sp.count
        t0 = time.perf_counter()
        call()
        elapsed = time.perf_counter() - t0
        during = sp.count - before
    return during, elapsed


def test_each_compute_method_releases_the_gil():
    m = aria_py.load_source(SRC, "W")
    be = aria_py.Backend("sv")
    params = ROWS[0]

    cases = {
        # ONE call per case, deliberately. An earlier draft looped
        # `expectation` 40x in Python, and the loop itself released the GIL
        # between calls — a GIL-holding mutant still scored 31.5% and PASSED.
        # With a single call the interpreter cannot run at all while the
        # extension holds the lock.
        "expectation": lambda: m.expectation(params, "Z0 Z13", backend=be),
        "gradient": lambda: m.gradient(params, "Z0 Z13", backend=be),
        "expectation_batch": lambda: m.expectation_batch(ROWS, "Z0 Z13", backend=be),
        "gradient_batch": lambda: m.gradient_batch(ROWS, "Z0 Z13", backend=be),
    }

    # A reference for what "the GIL was available" looks like on this machine:
    # sleep releases it by definition.
    ref_ticks, _ = _measure(lambda: time.sleep(0.05))
    per_sec = ref_ticks / 0.05
    print(f"\n  reference: {per_sec:,.0f} python ops/sec while the GIL is free")

    failures = []
    for name, call in cases.items():
        ticks, elapsed = _measure(call)
        # What the spinner *could* have done in that window if fully free.
        budget = per_sec * elapsed
        share = ticks / budget if budget > 0 else 0.0
        print(f"  {name:<18} {elapsed*1000:7.1f} ms  spinner ran {ticks:>9,} "
              f"({share:5.1%} of a free GIL)")
        # 50% sits in a MEASURED gap, not a guessed one. Mutants that hold the
        # GIL scored 11.7% and 31.5% on this machine; the real thing scored
        # 70.6%-82.3% across runs. The remainder is not the GIL — it is the
        # spinner losing its share of a machine whose cores rayon is also
        # using, which is why the bar is not near 100%.
        if share < 0.50:
            failures.append(
                f"{name}: the spinner executed {ticks:,} bytecodes during a "
                f"{elapsed*1000:.1f} ms call ({share:.1%} of what a free GIL "
                f"allows). The GIL is being HELD across this call — check that "
                f"`py.allow_threads` still wraps the backend call in "
                f"`Model::{name}` and that the boxed backend is still "
                f"`dyn Backend + Send + Sync`."
            )
    assert not failures, "\n".join(failures)


def test_results_are_unchanged_by_the_release():
    """The release must not change a single number.

    Worth its own test: `allow_threads` moves where the work happens relative to
    the GIL, and a mistake there surfaces as a wrong value or a panic, not as a
    slow run. Checked against the closed form, so the reference is not our own
    output.
    """
    src = "circuit C() { qreg q[1]\n  let t = symbolic[1]\n  apply RY(t[0]) on q[0] }"
    m = aria_py.load_source(src, "C")
    for theta in (0.0, 0.7, 1.9, math.pi, -2.4):
        assert abs(m.expectation([theta], "Z0") - math.cos(theta)) < 1e-12
        assert abs(m.gradient([theta], "Z0")[0] + math.sin(theta)) < 1e-9

    rows = [[0.0], [0.7], [math.pi], [-0.5]]
    zb = m.expectation_batch(rows, "Z0")
    gb = m.gradient_batch(rows, "Z0")
    for i, r in enumerate(rows):
        assert abs(zb[i] - math.cos(r[0])) < 1e-12
        assert abs(gb[i][0] + math.sin(r[0])) < 1e-9


def test_concurrent_calls_do_not_corrupt_each_other():
    """Four threads, four backends, genuinely overlapping work.

    With the GIL released the simulations really do interleave, so anything
    shared and unsynchronised between them can now race. Before the release that
    was unreachable, which is why this test is new rather than pre-existing.

    One `Backend` per thread, per the `unsendable` contract on that class.
    """
    src = "circuit C() { qreg q[1]\n  let t = symbolic[1]\n  apply RY(t[0]) on q[0] }"
    m = aria_py.load_source(src, "C")
    errors = []

    def run(offset):
        try:
            be = aria_py.Backend("sv")
            for k in range(200):
                th = offset + 0.01 * k
                got = m.expectation([th], "Z0", backend=be)
                if abs(got - math.cos(th)) > 1e-12:
                    errors.append(f"thread {offset}: {got} != cos({th})")
                    return
        except Exception as e:  # noqa: BLE001 — reported, not swallowed
            errors.append(f"thread {offset} raised: {e!r}")

    threads = [threading.Thread(target=run, args=(o,)) for o in (0.0, 1.0, 2.0, 3.0)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
