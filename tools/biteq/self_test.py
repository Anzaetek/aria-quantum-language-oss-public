# SPDX-License-Identifier: Apache-2.0
"""Self-test for diff.py's comparison kernel on synthetic artifacts.

Three cases the real corpus cannot be trusted to produce on demand:
an equal pair, a 1-ULP difference, and -0.0 vs +0.0 — the last must count as
a DIFFERENCE (they are == as floats) and be classed as a signed zero.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diff import compare, ulp_distance  # noqa: E402


def art(amps):
    return {"amps": amps}


def main():
    ok = True

    # Equal pair.
    a = art([["3f800000", "00000000"], ["00000000", "80000000"]])
    n, ulp, zeros, _ = compare(a, art(list(a["amps"])), "equal")
    if (n, ulp, zeros) != (0, 0, 0):
        ok = False
        print(f"FAIL equal: {(n, ulp, zeros)}")

    # 1 ULP apart (f32 1.0 vs nextafter).
    b = art([["3f800000", "00000000"]])
    c = art([["3f800001", "00000000"]])
    n, ulp, zeros, first = compare(b, c, "ulp")
    if (n, ulp, zeros) != (1, 1, 0):
        ok = False
        print(f"FAIL 1-ulp: {(n, ulp, zeros)} {first}")

    # Signed zero: equal as floats, different bits — must be a diff, classed
    # as a zero, and contribute nothing to max ULP.
    d = art([["00000000", "00000000"]])
    e = art([["80000000", "00000000"]])
    n, ulp, zeros, first = compare(d, e, "signed-zero")
    if (n, zeros) != (1, 1) or ulp != 0:
        ok = False
        print(f"FAIL signed-zero: n={n} zeros={zeros} ulp={ulp} {first}")

    # f64 width, 1 ULP.
    f = art([["3ff0000000000000", "0000000000000000"]])
    g = art([["3ff0000000000001", "0000000000000000"]])
    n, ulp, zeros, _ = compare(f, g, "f64-ulp")
    if (n, ulp) != (1, 1):
        ok = False
        print(f"FAIL f64 1-ulp: {(n, ulp)}")

    # ULP distance crosses the sign boundary sanely: -0.0 to smallest
    # positive subnormal is 1 in representation order.
    if ulp_distance("80000000", "00000001") != 1:
        ok = False
        print("FAIL ulp across zero")

    print("self_test:", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
