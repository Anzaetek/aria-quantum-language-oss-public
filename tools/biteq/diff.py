# SPDX-License-Identifier: Apache-2.0
"""Verdict renderer for the CUDA<->Metal bit-equality protocol.

Usage:
    python3 tools/biteq/diff.py <side_a_dir> <side_b_dir>

Each directory holds one side's artifacts, written by tools/biteq/run_side.sh:
    <circuit>.<gpu>.decompose.json
    <circuit>.<gpu>.exact.json
    <circuit>.cpu.json

Gates, in order (PLAN-BITEQ-CUDA-METAL.md section 4):
  1. CPU f64 cross-machine bit identity on the strict corpus. If the two
     machines' CPUs disagree, the difference is host math (libm), and NOTHING
     downstream may be attributed to the GPU kernels. Hard stop.
  2. Per box, over the whole corpus: Exact != Decompose somewhere. A box whose
     two dumps are bit-identical everywhere has a mode switch that did not
     take effect, and its artifacts are invalid (exit 1, not 2 — that is a
     harness failure, not a measurement).
     The gate is PER BOX, never per circuit: Exact == Decompose is CORRECT on
     basis-state circuits (the Metal test suite proves it per basis state)
     and on circuits with no multi-control gate.
  3. The headline: gpuA-Decompose vs gpuB-Decompose, bit for bit. Signed-zero
     differences are listed separately and STILL falsify (they come from
     dense-vs-permutation CX dispatch, a real implementation difference).
  4. Free localiser: gpuA-Exact vs gpuB-Exact. Both are pure permutations; a
     difference here means the non-CCX gates already disagree, so a headline
     diff is not about the decomposition.

The p90 libm probe is reported under its own heading and NEVER contributes to
the verdict.

Exit codes: 0 = every gate passed and the headline is bit-identical.
            2 = valid measurement, claim is FALSE (differences found).
            1 = harness/input error (missing files, gate 1 or 2 failure).
"""
import json
import struct
import sys
from pathlib import Path

PROBE_PREFIX = "p"  # p90_ry_probe and any future probes: excluded from verdict


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_dir(d):
    arts = {}
    for f in sorted(Path(d).glob("*.json")):
        doc = json.loads(f.read_text())
        if doc.get("format") != "biteq-v1":
            die(f"{f}: not a biteq-v1 artifact")
        key = (doc["circuit"], doc["backend"], doc["multi_control"])
        arts[key] = doc
    if not arts:
        die(f"{d}: no artifacts")
    return arts


def gpu_of(arts):
    gpus = {b for (_, b, _) in arts if b != "cpu-f64"}
    if len(gpus) != 1:
        die(f"expected exactly one GPU backend per side, found {sorted(gpus)}")
    return gpus.pop()


def f_of_hex(h):
    if len(h) == 8:
        return struct.unpack(">f", bytes.fromhex(h))[0]
    if len(h) == 16:
        return struct.unpack(">d", bytes.fromhex(h))[0]
    die(f"bad hex width {len(h)}: {h!r}")


def ulp_distance(ha, hb):
    """Distance in representation order for same-width bit patterns."""
    ia, ib = int(ha, 16), int(hb, 16)
    bits = len(ha) * 4
    sign = 1 << (bits - 1)

    # Monotone key over IEEE bit patterns: positives map to themselves,
    # negatives to sign - i (so -0.0 and +0.0 both map to 0, and the order
    # decreases as negative magnitude grows). Signed zeros are classified
    # separately by the caller and never reach this as their own case.
    def key(i):
        return i if not i & sign else sign - i

    return abs(key(ia) - key(ib))


def compare(doc_a, doc_b, label):
    """Return (n_diff, max_ulp, signed_zero_only_diffs, first_diff_desc)."""
    a, b = doc_a["amps"], doc_b["amps"]
    if len(a) != len(b):
        die(f"{label}: amplitude count differs ({len(a)} vs {len(b)})")
    n_diff, max_ulp, zeros, first = 0, 0, 0, None
    for i, (pa, pb) in enumerate(zip(a, b)):
        for part, (ha, hb) in enumerate(zip(pa, pb)):
            if ha == hb:
                continue
            n_diff += 1
            va, vb = f_of_hex(ha), f_of_hex(hb)
            if va == 0.0 and vb == 0.0:
                zeros += 1  # -0.0 vs +0.0: equal as floats, different bits
            else:
                max_ulp = max(max_ulp, ulp_distance(ha, hb))
            if first is None:
                first = f"amp[{i}].{'re' if part == 0 else 'im'}: {ha} vs {hb}"
    return n_diff, max_ulp, zeros, first


def main():
    if len(sys.argv) != 3:
        die(__doc__)
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    A, B = load_dir(a_dir), load_dir(b_dir)
    gpu_a, gpu_b = gpu_of(A), gpu_of(B)
    for side, arts in (("A", A), ("B", B)):
        revs = {d["build"]["rev"] for d in arts.values()}
        oses = {d["host"].get("os_version", "?") for d in arts.values()}
        print(f"side {side}: gpu={gpu_of(arts)} rev={sorted(revs)} os={sorted(oses)}")

    circuits = sorted({c for (c, bk, _) in A if bk == "cpu-f64"})
    strict = [c for c in circuits if not c.startswith(PROBE_PREFIX)]
    probes = [c for c in circuits if c.startswith(PROBE_PREFIX)]

    # Gate 1: CPU f64 arbiter.
    print("\n== gate 1: CPU f64 cross-machine bit identity (strict corpus) ==")
    for c in strict:
        ka = (c, "cpu-f64", "decompose")
        if ka not in A or ka not in B:
            die(f"missing cpu artifact for {c}")
        n, _, _, first = compare(A[ka], B[ka], f"cpu:{c}")
        state = "identical" if n == 0 else f"{n} DIFFER ({first})"
        print(f"  {c}: {state}")
        if n:
            die(
                "the two machines' CPU f64 states differ on the strict corpus; "
                "that is host math, not kernels — stop and investigate the "
                "build/corpus before blaming any GPU"
            )

    # Gate 2: per box, the mode switch took effect.
    print("\n== gate 2: per box, Exact != Decompose somewhere ==")
    for side, arts, gpu in (("A", A, gpu_a), ("B", B, gpu_b)):
        total = 0
        for c in strict:
            kd, ke = (c, gpu, "decompose"), (c, gpu, "exact")
            if kd not in arts or ke not in arts:
                die(f"side {side}: missing {gpu} artifacts for {c}")
            n, _, _, _ = compare(arts[kd], arts[ke], f"{side}:{c}")
            total += n
        print(f"  side {side} ({gpu}): {total} differing amplitude parts across corpus")
        if total == 0:
            die(
                f"side {side}: Exact and Decompose are bit-identical on the whole "
                "corpus — the mode switch did not take effect; artifacts invalid"
            )

    # Headline.
    print(f"\n== headline: {gpu_a}-Decompose vs {gpu_b}-Decompose ==")
    claim_true = True
    for c in strict:
        n, ulp, zeros, first = compare(
            A[(c, gpu_a, "decompose")], B[(c, gpu_b, "decompose")], c
        )
        if n == 0:
            print(f"  {c}: bit-identical")
        else:
            claim_true = False
            # Signed-zero-only differences come from a SPARSE DISPATCH, not
            # from arithmetic: a fast path that does not write an amplitude
            # leaves whatever sign was there, where the dense 4x4 would write
            # `1*a + 0 + 0 + 0` and normalise `-0.0` to `+0.0`. Originally seen
            # from the CX permutation; as of 2026-08-25 the CPU also dispatches
            # CZ/CRz (diagonal), SWAP (permutation) and CY/CU3 (controlled-U),
            # and the controlled-U path leaves the whole control-zero half
            # untouched — so this is now expected from more arms than CX and
            # the attribution should not name only CX.
            attr = (
                " [signed zeros only: sparse-vs-dense dispatch, not arithmetic]"
                if n == zeros
                else ""
            )
            print(
                f"  {c}: {n} parts differ (max {ulp} ULP, {zeros} signed-zero) "
                f"first {first}{attr}"
            )

    # Localiser.
    print(f"\n== localiser: {gpu_a}-Exact vs {gpu_b}-Exact ==")
    for c in strict:
        n, ulp, zeros, first = compare(A[(c, gpu_a, "exact")], B[(c, gpu_b, "exact")], c)
        note = "bit-identical" if n == 0 else f"{n} parts differ (max {ulp} ULP) — non-CCX gates already disagree"
        print(f"  {c}: {note}")

    # Probe, outside the verdict.
    if probes:
        print("\n== libm probe (EXCLUDED from verdict) ==")
        for c in probes:
            ka = (c, "cpu-f64", "decompose")
            if ka in A and ka in B:
                n, ulp, _, first = compare(A[ka], B[ka], f"probe:{c}")
                msg = "cpu f64 identical — host libm agrees on this corpus" if n == 0 else (
                    f"cpu f64 differ: {n} parts, max {ulp} ULP ({first}) — host libm diverges; "
                    "rotation gates are outside any cross-machine bit claim"
                )
                print(f"  {c}: {msg}")

    if claim_true:
        print("\nVERDICT: bit-for-bit CONFIRMED on this corpus, for the OS/compiler pairs above.")
        sys.exit(0)
    print(
        "\nVERDICT: the bit-for-bit claim is FALSE as stated. Soften the doc "
        "comments to 'the same decomposition, each validated against CPU f64', "
        "scoped to the measured pairs above."
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
