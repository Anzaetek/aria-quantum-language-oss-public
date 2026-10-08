#!/usr/bin/env python3
"""Audit A9 — verification scope narrower than change scope.

PLAN-OPEN-20260825.md §3f. Given the crates a change touched, print the set
that must actually be tested: the *reverse* dependencies, transitively.

RECEIPT. `cargo test -p <the crate I edited>` came back green twice while full
CI exited 101 — once on omega-cli's integration tests, once on the Metal parity
crate. Both were downstream of a change in a different crate. Testing the crate
you edited is the check that does not catch this.

WHY NOT `cargo tree -i`, WHICH IS THE OBVIOUS ANSWER
----------------------------------------------------
Three holes, each found by running it rather than by reading the docs:

1. `cargo tree --invert -p <crate>` returns only the crate itself. Inside a
   workspace, `-i` needs `--workspace` to look outside the current member, and
   `-p` then means a subtree filter instead of the target. Written
   confidently, silently useless.

2. Even spelled correctly, `cargo tree -i <crate> --workspace` returns NOTHING
   for omega-backend-mps-cuda, because every dependent takes it as an
   `optional` dependency and the feature is off. Three real dependents,
   invisible. An empty result reads as "nothing depends on this".

3. `cargo metadata --no-deps` fixes (2) — it reports declared dependencies
   including optional ones — but it is scoped to workspace MEMBERS, and this
   repo `exclude`s crates that are real dependents. `bindings/aria-py` takes
   omega-backend-mps-cuda as an optional path dependency and is excluded so the
   default build stays pyo3-free. Both cargo answers miss it.

So this tool enumerates every Cargo.toml in the tree, not just the members, and
reads declared dependencies rather than the resolved feature graph.

WHY NOT GREP, WHICH IS THE OTHER OBVIOUS ANSWER
-----------------------------------------------
`grep -rln omega-backend-mps-cuda --include=Cargo.toml .` is complete — it is
the only one of the three that finds bindings/aria-py — but it is not exact.
It returns seven files for three real dependents: the workspace root, the crate
itself, and crates/omega-backend-pauliprop-cuda, whose only match is the word
inside a COMMENT. A dependent list padded with false positives gets trimmed by
hand, and hand-trimming is how the real one gets dropped.

Run --audit to print the divergence between this tool and the grep.

NOT CLAIMED: that a green reverse-dependency run means the change is safe. It
means the blast radius was tested, which is the specific thing that was not.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Overridable so the tool can be run against a synthetic fixture. A tool that
# can only be pointed at the repo it audits cannot be fault-injected without
# editing that repo, and "I read it and it looks right" is what audit A2 is
# about.
REPO = Path(os.environ.get("ARIA_RDEPS_REPO")
            or Path(__file__).resolve().parents[2]).resolve()


def _cargo_metadata(manifest: Path) -> dict:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps",
         "--manifest-path", str(manifest)],
        cwd=REPO, capture_output=True, text=True,
    )
    if out.returncode != 0:
        return {}
    return json.loads(out.stdout)


def all_packages() -> tuple[dict, set]:
    """Every package in the tree, member or not.

    Returns (name -> package, set of names that are NOT workspace members).
    `cargo test -p X` from the root cannot reach the second set; it needs
    --manifest-path, which is why they are tracked separately rather than
    merged into one list that would be wrong for half its entries.
    """
    root = _cargo_metadata(REPO / "Cargo.toml")
    if not root:
        sys.exit("FAIL: cargo metadata failed on the workspace root")
    pkgs = {p["name"]: p for p in root["packages"]}
    members = set(pkgs)

    seen_manifests = {Path(p["manifest_path"]).resolve() for p in root["packages"]}
    for mf in sorted(REPO.rglob("Cargo.toml")):
        if "target" in mf.parts or ".git" in mf.parts:
            continue
        if mf.resolve() in seen_manifests:
            continue
        outside = _cargo_metadata(mf)
        for p in outside.get("packages", []):
            if Path(p["manifest_path"]).resolve() == mf.resolve():
                pkgs.setdefault(p["name"], p)
    return pkgs, set(pkgs) - members


def reverse_edges(pkgs: dict) -> dict:
    """name -> {dependent_name: kind}. Declared deps, optional included."""
    rev: dict[str, dict[str, str]] = {}
    for name, p in pkgs.items():
        for d in p.get("dependencies", []):
            kind = d.get("kind") or "normal"
            flags = [kind]
            if d.get("optional"):
                flags.append("optional")
            rev.setdefault(d["name"], {})[name] = "+".join(flags)
    return rev


def closure(targets, rev, direct=False) -> dict:
    """Transitive reverse-dependency closure.

    Transitive by default: a break in X reaches Y through an intermediate just
    as well as directly, and CI's 101 came from exactly that shape.
    """
    out: dict[str, str] = {}
    frontier = list(targets)
    seen = set(targets)
    depth = 0
    while frontier:
        nxt = []
        for t in frontier:
            for dep, kind in rev.get(t, {}).items():
                label = kind if depth == 0 else f"{kind} (via {t})"
                out.setdefault(dep, label)
                if dep not in seen:
                    seen.add(dep)
                    nxt.append(dep)
        depth += 1
        if direct:
            break
        frontier = nxt
    for t in targets:
        out.pop(t, None)
    return out


def changed_crates(pkgs: dict, base: str) -> list:
    """Crates owning the files this branch touched, committed or not."""
    files = set()
    for cmd in (["git", "diff", "--name-only", base, "--"],
                ["git", "diff", "--name-only", "HEAD", "--"],
                ["git", "ls-files", "--others", "--exclude-standard"]):
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if r.returncode == 0:
            files.update(x for x in r.stdout.split("\n") if x.strip())

    # Longest-prefix wins: crates/apps/foo must beat crates/.
    roots = sorted(
        ((Path(p["manifest_path"]).resolve().parent, n) for n, p in pkgs.items()),
        key=lambda kv: len(str(kv[0])), reverse=True,
    )
    hit, orphans = [], []
    for f in sorted(files):
        fp = (REPO / f).resolve()
        for root, name in roots:
            if root == fp or root in fp.parents:
                if name not in hit:
                    hit.append(name)
                break
        else:
            orphans.append(f)
    return hit, orphans


def grep_hits(target: str) -> set:
    r = subprocess.run(
        ["grep", "-rl", "--include=Cargo.toml", target, "."],
        cwd=REPO, capture_output=True, text=True,
    )
    out = set()
    for line in r.stdout.split("\n"):
        line = line.strip()
        if line and "target/" not in line:
            out.add(line.lstrip("./"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Reverse dependencies to test (audit A9)")
    ap.add_argument("crates", nargs="*", help="crates that changed")
    ap.add_argument("--changed", metavar="BASE", nargs="?", const="origin/main",
                    help="derive changed crates from git (default base origin/main)")
    ap.add_argument("--direct", action="store_true", help="direct dependents only")
    ap.add_argument("--audit", action="store_true",
                    help="also print the divergence against grep")
    args = ap.parse_args()

    pkgs, outside = all_packages()
    rev = reverse_edges(pkgs)

    targets = list(args.crates)
    orphans: list[str] = []
    if args.changed is not None:
        found, orphans = changed_crates(pkgs, args.changed)
        targets += [c for c in found if c not in targets]

    # "No crates changed" and "files changed that no crate owns" are different
    # facts, and collapsing them is audit A8 — a skip that reads as a pass —
    # inside the tool for A9. The first version printed the same line for both,
    # on a change to ci.sh and tools/, which is exactly the case where a
    # reader would wrongly conclude nothing needed testing.
    if orphans:
        print(f"{len(orphans)} changed file(s) belong to no crate — this tool")
        print("says nothing about them; they need whatever check owns them:")
        for f in orphans[:20]:
            print(f"   {f}")
        if len(orphans) > 20:
            print(f"   ... and {len(orphans) - 20} more")
        print()

    if not targets:
        print("No changed CRATES. Nothing for a reverse-dependency run to cover.")
        return 0

    unknown = [t for t in targets if t not in pkgs]
    if unknown:
        print(f"FAIL: not packages in this tree: {', '.join(unknown)}", file=sys.stderr)
        return 2

    print(f"Changed: {' '.join(targets)}")
    deps = closure(targets, rev, direct=args.direct)
    if not deps:
        print("\nNo reverse dependencies. Testing the changed crates is the whole check.")
    else:
        scope = "direct dependents" if args.direct else "reverse-dependency closure"
        print(f"\n{len(deps)} crate(s) in the {scope}:")
        for name in sorted(deps):
            mark = "  [OUTSIDE WORKSPACE]" if name in outside else ""
            print(f"   {name:34s} {deps[name]}{mark}")

    inside = sorted(n for n in list(deps) + targets if n not in outside)
    if inside:
        print("\nRun:")
        print("   cargo test " + " ".join(f"-p {n}" for n in inside))

    excluded = sorted(n for n in list(deps) + targets if n in outside)
    if excluded:
        print("\nNOT reachable by `-p` from the root — excluded from the workspace,")
        print("so a green workspace run says nothing about them:")
        for n in excluded:
            mf = os.path.relpath(pkgs[n]["manifest_path"], REPO)
            print(f"   cargo check --manifest-path {mf}")

    if args.audit:
        for t in targets:
            g = grep_hits(t)
            mine = {os.path.relpath(pkgs[n]["manifest_path"], REPO)
                    for n in closure([t], rev, direct=True)}
            print(f"\n-- audit: {t} --")
            print(f"   this tool : {len(mine)} manifest(s)")
            print(f"   grep      : {len(g)} manifest(s)")
            for extra in sorted(g - mine):
                print(f"   grep-only : {extra}   (root/self/comment — a false positive)")
            for extra in sorted(mine - g):
                print(f"   MISSED BY GREP: {extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
