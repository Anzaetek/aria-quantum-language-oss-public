#!/usr/bin/env python3
"""Audit A1 — computed but never consulted (PLAN-OPEN-20260825.md §3f).

RECEIPT. `dropped_mass` was accumulated, plumbed to the `Backend` result,
documented as "the CERTIFIED error bound", printed by the CLI, and read by
nothing that could change an outcome. `omega-backend-mps/src/sim.rs:528`
records fixing the identical defect in MPS, as a lesson, in prose — and the
defect was sitting one directory away the whole time. Hence a tool.

WHAT COUNTS AS CONSUMING
------------------------
Printing is not consuming. A field that reaches the user only through
`format!`/`Debug`/serialisation cannot change what the program does, and the
question this audit asks is whether a bound can.

An assertion in a `#[cfg(test)]` module or a `tests/` file is not consuming
EITHER, and this is the distinction that makes the tool worth running: the
original `dropped_mass` had test assertions on it. Tests pin a value; they do
not let it change a result. A field whose only non-formatting reads are in
tests is reported.

Any other read in production code counts — including plain arithmetic with no
comparison in it. An early version of this check looked for comparison
operators and flagged `MagicStateProtocol.input_infidelity`, which is read as
`let mut p = self.input_infidelity;` — the seed of the distillation recurrence,
and about as load-bearing as a field gets. Requiring a comparison operator
tests the shape of the read, not whether anything depends on it.

NOT CLAIMED: that a flagged field is a bug. A public library field may have a
downstream consumer outside this tree. What the tool asserts is narrower and
checkable: nothing IN THIS WORKSPACE would behave differently if the field were
wrong. For a bound, a budget, or an error estimate, that is the defect.
"""
import argparse
import os
import pathlib
import re
import subprocess
import sys

# Names that make a field a claim about accuracy, cost or health — the ones
# where "nobody reads it" is a defect rather than a spare accessor.
CLAIM = re.compile(
    r"(mass|weight|fidelity|infidelity|error|bound|budget|discarded|dropped"
    r"|certified|certificate|residual|overhead|cost|rounds|estimate)", re.I)

FIELD = re.compile(r"^\s*pub\s+([a-z_][a-z0-9_]*)\s*:\s*(.+?),\s*$")
STRUCT = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+(\w+)")
FN = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+\w+")
FORMATTING = re.compile(
    r"format!|println!|print!|eprintln!|eprint!|write!|writeln!|json!"
    r"|\.to_string\(\)|serde_json::")


def source_files(root: pathlib.Path):
    dirs = [d for d in ("crates", "bindings") if (root / d).is_dir()]
    if not dirs:
        dirs = ["."]
    out = subprocess.run(
        ["find", *dirs, "-name", "*.rs", "-not", "-path", "*/target/*"],
        cwd=root, capture_output=True, text=True)
    return [root / p for p in out.stdout.split() if p]


def test_lines(lines):
    """Indices inside a `#[cfg(test)]` item, by brace balance."""
    out, i = set(), 0
    while i < len(lines):
        if "#[cfg(test)]" in lines[i]:
            j, depth, started = i, 0, False
            while j < len(lines):
                depth += lines[j].count("{") - lines[j].count("}")
                if "{" in lines[j]:
                    started = True
                out.add(j)
                if started and depth <= 0:
                    break
                j += 1
            i = j
        i += 1
    return out


def collect(root):
    """(fields, sources) — declared claim-fields, and every file's lines."""
    fields, sources = [], {}
    for path in sorted(source_files(root)):
        rel = str(path.relative_to(root))
        lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
        sources[rel] = (lines, test_lines(lines))
        cur = None
        for i, line in enumerate(lines):
            m = STRUCT.match(line)
            if m:
                cur = m.group(1)
            if FN.match(line):
                cur = None          # fn bodies are not struct bodies
            fm = FIELD.match(line)
            if fm and cur and CLAIM.search(fm.group(1)):
                fields.append((rel, i + 1, cur, fm.group(1), fm.group(2)))
    return fields, sources


def reads(sources, field):
    """Classify every mention of `.field` outside its own declaration."""
    prod, test, fmt = [], [], []
    # `(?!\s*\()` keeps a METHOD of the same name out of the field's read set.
    # Without it `CqsResult.residual` looked consumed, on the strength of
    # `circulant.rs`'s `fn residual(&self, x, b)` — a different type entirely.
    # A field and a method can share a name and share nothing else.
    pat = re.compile(rf"\.{re.escape(field)}\b(?!\s*\()")
    assign = re.compile(rf"\.{re.escape(field)}\s*(=[^=]|[+\-*/]=)")
    for rel, (lines, tl) in sources.items():
        in_test_file = "/tests/" in rel or rel.endswith("_test.rs")
        for i, line in enumerate(lines):
            if not pat.search(line):
                continue
            s = line.strip()
            if s.startswith("//") or s.startswith("*") or s.startswith("///"):
                continue
            if assign.search(line):
                continue                      # a write is not a read
            site = (rel, i + 1, s[:100])
            if in_test_file or i in tl:
                test.append(site)
            elif FORMATTING.search(line):
                fmt.append(site)
            else:
                prod.append(site)
    return prod, test, fmt


def main():
    ap = argparse.ArgumentParser(description="Fields computed but never consulted (audit A1)")
    ap.add_argument("--all", action="store_true", help="list consumed fields too")
    ap.add_argument("--allow", default="tools/audit/computed_not_consulted.allow",
                    help="file of `Struct.field  # reason` lines to exempt")
    args = ap.parse_args()

    # Overridable so the tool can be fault-injected against a fixture instead
    # of by editing the repo it audits.
    root = pathlib.Path(os.environ.get("ARIA_AUDIT_ROOT")
                        or pathlib.Path(__file__).resolve().parents[2]).resolve()
    allow = {}
    ap_path = root / args.allow
    if ap_path.exists():
        for line in ap_path.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, why = line.partition("#")
            allow[key.strip()] = why.strip()

    fields, sources = collect(root)
    findings, consumed = [], []
    for rel, ln, struct, name, ty in fields:
        prod, test, fmt = reads(sources, name)
        (findings if not prod else consumed).append(
            (f"{struct}.{name}", rel, ln, ty, len(prod), len(test), len(fmt), test, fmt))

    print(f"{len(fields)} pub field(s) whose NAME makes an accuracy/cost claim\n")
    unexcused = [f for f in findings if f[0] not in allow]
    for key, rel, ln, ty, np_, nt, nf, test, fmt in sorted(findings):
        tag = "EXEMPT" if key in allow else "FINDING"
        print(f"[{tag}] {key}: {ty}")
        print(f"          declared {rel}:{ln}")
        print(f"          production reads: {np_}   test-only: {nt}   formatting-only: {nf}")
        if key in allow:
            print(f"          reason: {allow[key]}")
        for r, n, s in (test + fmt)[:3]:
            print(f"            {r}:{n}  {s}")
        print()

    if args.all:
        print("--- consumed ---")
        for key, rel, ln, ty, np_, nt, nf, _, _ in sorted(consumed):
            print(f"  {key:44s} production reads: {np_}")

    if unexcused:
        print(f"FAIL: {len(unexcused)} field(s) nothing in this workspace consults.")
        print("      Either wire one to a decision, or add it to")
        print(f"      {args.allow} with the reason it is output-only.")
        return 1
    print(f"OK: every accuracy/cost field has a production consumer "
          f"({len(allow)} exempted by name).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
