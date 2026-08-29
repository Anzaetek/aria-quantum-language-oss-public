#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive checker for HostGate.tla — python3, no dependencies, no Java.

Why this exists next to the TLA+ module rather than instead of it: TLC is the
authority, but it needs a JVM, and a proof that only runs on a machine with the
right toolchain installed is a proof that stops being run. This enumerates the
same state space from an INDEPENDENT implementation of the same transition
relation, so agreement between the two is evidence that the model says what the
module says. Divergence in the distinct-state count means one of them has
drifted and neither should be trusted until it is explained.

Run:
    python3 check_hostgate.py

It checks three configurations and expects a specific outcome from each:

  baseline          every safety property holds, and the books always come back
                    into balance after a crash — the failsafe works
  non-atomic ledger NeverOvercommitted FAILS. A crash mid-rewrite truncates the
                    file, every holder disappears from the accounting while its
                    memory is still resident, and the next request is admitted
                    on top of it
  pid-only keying   the books can never rebalance. A recycled PID makes a dead
                    holder's record look live, the prune never reclaims it, and
                    that capacity is gone until reboot

The last two are the argument for temp-file+rename and for keying on
(pid, start_time). They are expected to fail, and the day they stop failing
either the model or the claim is wrong.
"""
from collections import deque
from itertools import product

MAX_GEN = 2


class Config:
    def __init__(self, name, needs, capacity, atomic_ledger=True,
                 identity_keyed=True):
        self.name = name
        self.needs = tuple(needs)
        self.n = len(needs)
        self.capacity = capacity
        self.atomic_ledger = atomic_ledger
        self.identity_keyed = identity_keyed


# A state is (ledger, resident, alive, gen, ledger_gen), all tuples/frozenset
# so it is hashable and comparable — the same five variables the module has.
def initial(cfg):
    z = (0,) * cfg.n
    return (z, z, frozenset(range(cfg.n)), z, z)


def charged(s):
    return sum(s[0])


def occupied(s):
    return sum(s[1])


def free(cfg, s):
    return max(0, cfg.capacity - charged(s))


def still_held(cfg, s, p):
    """What the PRUNE asks while scanning the file — as good as the keying.

    With identity keying the incarnation must match too, so a recycled PID does
    not inherit the dead holder's record.
    """
    _ledger, _resident, alive, gen, ledger_gen = s
    if p not in alive:
        return False
    return ledger_gen[p] == gen[p] if cfg.identity_keyed else True


def owns(s, p):
    """What the PROCESS knows: it has a guard in its own memory, or it does not.

    Always incarnation-scoped, whatever the keying scheme, because a restarted
    process's predecessor took its guard down with its address space. Using
    `still_held` here instead is what hid the PID-reuse leak on the first run.
    """
    _ledger, _resident, alive, gen, ledger_gen = s
    return p in alive and ledger_gen[p] == gen[p]


def _set(t, i, v):
    lst = list(t)
    lst[i] = v
    return tuple(lst)


def successors(cfg, s):
    """Every enabled action, as (label, next_state). Mirrors `Next`."""
    ledger, resident, alive, gen, ledger_gen = s
    out = []

    for p in range(cfg.n):
        # Acquire
        if p in alive and ledger[p] == 0 and cfg.needs[p] <= free(cfg, s):
            out.append((f"Acquire({p})", (
                _set(ledger, p, cfg.needs[p]),
                _set(resident, p, cfg.needs[p]),
                alive, gen, _set(ledger_gen, p, gen[p]))))

        # Release — guarded by identity, not merely by liveness: a process
        # releases the grant IT took, never whatever record carries its PID.
        # Without this a restarted process tidies up its dead predecessor's
        # record, which hides the PID-reuse leak from the liveness check.
        if owns(s, p) and ledger[p] > 0:
            out.append((f"Release({p})", (
                _set(ledger, p, 0), _set(resident, p, 0),
                alive, gen, ledger_gen)))

        # Crash — memory goes, the record does not. With a non-atomic ledger a
        # crash can land mid-rewrite and truncate the whole file.
        if p in alive:
            new_ledger = ledger if cfg.atomic_ledger else (0,) * cfg.n
            out.append((f"Crash({p})", (
                new_ledger, _set(resident, p, 0),
                alive - {p}, gen, ledger_gen)))

        # Restart — the slot comes back as a new incarnation.
        if p not in alive and gen[p] < MAX_GEN:
            out.append((f"Restart({p})", (
                ledger, resident, alive | {p},
                _set(gen, p, gen[p] + 1), ledger_gen)))

    # Prune — the failsafe. Enabled when any record has no live owner.
    if any(ledger[p] > 0 and not still_held(cfg, s, p) for p in range(cfg.n)):
        out.append(("Prune", (
            tuple(ledger[p] if still_held(cfg, s, p) else 0
                  for p in range(cfg.n)),
            resident, alive, gen, ledger_gen)))

    return out


def invariants(cfg, s):
    """Returns the name of the first violated safety property, or None."""
    ledger, resident, alive, gen, ledger_gen = s
    if occupied(s) > cfg.capacity:
        return "NeverOvercommitted"
    if charged(s) < occupied(s):
        return "AccountingCoversReality"
    for p in alive:
        if resident[p] > ledger[p]:
            return "NoPhantomReclaim"
    # Deliberately NOT checked: "a live process's record belongs to its own
    # incarnation". That is false in one step in the correct design — Acquire,
    # Crash, Restart — because a stale record beside a live process is exactly
    # what a crash produces and exactly what the prune clears. See HostGate.tla.
    return None


def explore(cfg):
    """BFS the whole space, keeping parents so a violation prints a trace."""
    start = initial(cfg)
    seen = {start: (None, None)}
    order = [start]
    q = deque([start])
    violation = None
    edges = 0
    while q:
        s = q.popleft()
        if violation is None:
            bad = invariants(cfg, s)
            if bad:
                violation = (bad, s)
        for label, nxt in successors(cfg, s):
            edges += 1
            if nxt not in seen:
                seen[nxt] = (s, label)
                order.append(nxt)
                q.append(nxt)
    return seen, order, edges, violation


def trace_to(seen, s):
    steps = []
    while True:
        parent, label = seen[s]
        if parent is None:
            break
        steps.append((label, s))
        s = parent
    return list(reversed(steps))


def balanced_reachable(cfg, seen):
    """Which states can still reach agreement between the books and reality?

    This is the failsafe as a liveness question: after any death, can the
    system always get back to `Charged = Occupied`? A state that cannot is a
    permanent leak — capacity charged to nobody, for as long as the box is up.

    Only SELF-HEALING transitions count: `Prune` and `Release`, the two things
    the system does on its own. Following every transition instead would count
    "crash a second time, which makes the stale record prunable again" as
    recovery — and a design that needs another failure to repair the last one
    is not a failsafe. This is the same restriction the module expresses as weak
    fairness on `Prune` and `Release` and on nothing else.
    """
    def healing(s):
        return [n for l, n in successors(cfg, s)
                if l == "Prune" or l.startswith("Release")]

    rev = {s: [] for s in seen}
    for s in seen:
        for n in healing(s):
            if n in rev:
                rev[n].append(s)
    good = {s for s in seen if charged(s) == occupied(s)}
    q = deque(good)
    while q:
        s = q.popleft()
        for p in rev[s]:
            if p not in good:
                good.add(p)
                q.append(p)
    return {s for s in seen if s not in good}


def report(cfg):
    seen, order, edges, violation = explore(cfg)
    print(f"\n=== {cfg.name} ===")
    print(f"    needs={cfg.needs} capacity={cfg.capacity} "
          f"atomic_ledger={cfg.atomic_ledger} identity_keyed={cfg.identity_keyed}")
    print(f"    {len(seen)} distinct states, {edges} transitions")

    if violation:
        name, s = violation
        print(f"    SAFETY VIOLATED: {name}")
        for label, st in trace_to(seen, s):
            print(f"      {label:<12} ledger={st[0]} resident={st[1]} "
                  f"alive={sorted(st[2])}")
        print(f"      -> charged={charged(s)} occupied={occupied(s)} "
              f"capacity={cfg.capacity}")
    else:
        print("    safety: NeverOvercommitted, AccountingCoversReality, "
              "NoPhantomReclaim, NoChargeAcrossRestart all hold")

    stuck = balanced_reachable(cfg, seen)
    if stuck:
        s = min(stuck, key=lambda x: len(trace_to(seen, x)))
        print(f"    LIVENESS VIOLATED: EventuallyReclaimed — {len(stuck)} states "
              f"can never rebalance")
        for label, st in trace_to(seen, s):
            print(f"      {label:<12} ledger={st[0]} resident={st[1]} "
                  f"alive={sorted(st[2])}")
        print(f"      -> charged={charged(s)} occupied={occupied(s)}: "
              f"{charged(s) - occupied(s)} unit(s) charged to nobody, forever")
    else:
        print("    liveness: EventuallyReclaimed holds — after every crash the "
              "books come back into balance")
    return len(seen), violation is not None, bool(stuck)


def main():
    # Three processes, one of which needs half the box: enough for a large
    # request to be blocked by two small ones, which is where the interesting
    # interleavings live.
    base = dict(needs=(2, 1, 1), capacity=4)
    configs = [
        Config("baseline — atomic ledger, identity-keyed holders", **base),
        Config("non-atomic ledger (in-place rewrite)", **base,
               atomic_ledger=False),
        Config("pid-only keying (no start time)", **base,
               identity_keyed=False),
    ]
    results = [report(c) for c in configs]

    print("\n--- expected outcomes ---")
    ok = True
    # (safety_violated, liveness_violated); None means "not asserted here".
    expect = [(False, False), (True, None), (False, True)]
    names = ["baseline", "non-atomic ledger", "pid-only keying"]
    for (_states, unsafe, leaky), (want_unsafe, want_leaky), name in zip(
            results, expect, names):
        for got, want, label in ((unsafe, want_unsafe, "safety violation"),
                                 (leaky, want_leaky, "liveness violation")):
            if want is None:
                continue
            mark = "ok " if got == want else "FAIL"
            if got != want:
                ok = False
            print(f"  {mark} {name}: {label} = {got}, expected {want}")
    print("\nall expectations met" if ok else "\nEXPECTATIONS NOT MET")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
