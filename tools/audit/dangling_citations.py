#!/usr/bin/env python3
"""Audit A4 — a claim surviving on a surface where it is not true.

PLAN-OPEN-20260825.md §3f. A doc comment that cites a test BY NAME is making a
checkable promise: go look, and you will find the guard. This finds citations
whose referent does not exist.

RECEIPT (found by running it). `omega-backend-statevector/src/sim.rs:1454`
justifies a signed-zero guard with

    MEASURED: removing it fails
    `controlled_1q_fast_path_matches_the_dense_scan_bit_for_bit`
    on the signed-zero fixture at idx 0.

There is no such test. The citation is the evidence that the guard is
load-bearing, so a reader who checks it finds nothing and is left choosing
between distrusting the comment and re-doing the measurement — and a reader who
does not check may delete the guard. Same shape at
`omega-backend-mps/src/lib.rs:33`, which says a constant tie is *"pinned by
`both_front_ends_use_the_same_auto_epsilon`"*; the tie IS pinned, by
`both_front_ends_use_the_same_defaults`, so only the name is wrong — but the
name is what a verifier greps.

WHAT COUNTS AS EXISTING, AND WHY THE LIST IS LONG
-------------------------------------------------
The first version looked only for `fn <name>` in Rust and reported 112
citations. Almost all were correct usage, and each class had to be added:

  * an integration test is a FILE — `multi_control_is_not_silently_dropped`
    is `crates/omega-cli/tests/<name>.rs`, with no fn of that name;
  * GPU kernels live in `.cu` / `.metal` / `.cl`;
  * proofs live in `.lean`;
  * struct fields, host imports and external APIs (`par_chunks_mut`,
    `matrix_product_state`, clippy's `await_holding_lock`) are named on
    purpose and are not ours to define;
  * a citation inside "renamed from X" / "this replaces X" is ABOUT the
    absence. Flagging it would invert the check.

Reporting the raw 112 would have been 107 fabricated findings. The tool is
therefore deliberately conservative: it reports a citation only when the name
is absent everywhere AND the sentence around it presents it as evidence.
"""
import argparse
import os
import pathlib
import re
import subprocess
import sys

# Verbs that turn a mention into a promise. "see also X" is a pointer;
# "pinned by X" is a claim that X exists and does a job.
EVIDENCE = re.compile(
    r"\b(pinned|caught|proves?|proven|guarded|enforced|asserted|covered|"
    r"verified|exercised?|checked|regression|MEASURED|fails|shown)\b", re.I)

# Phrases whose whole point is that the name is gone.
#
# `was \`` is the bare-predicate form — "This test was `old_name` and asserted
# only ...". `was named` did not cover it, so a comment whose entire subject is
# a rename was reported as a dangling citation. Requiring the backtick keeps it
# narrow: "was `identifier`" is a renaming announcement, where a bare "was"
# would silence ordinary prose.
RETIRED = re.compile(
    r"\b(renamed|replaces|replaced|used to|was named|was\s+`|removed|superseded|"
    r"formerly|no longer|deleted|dropped in)\b", re.I)

CITE = re.compile(r"`([a-z][a-z0-9_]*(?:_[a-z0-9]+){2,})`")

# A findings register REPORTS citations; it does not make them.
#
# PLAN-OPEN §3f records every dangling name this tool has found, by name and
# on purpose. Scanning markdown turned all seven of those write-ups into fresh
# findings — the tool objecting to its own results being written down. This is
# the same defect already fixed for the Lean pass and NOT generalised at the
# time, which is the §A3 failure ("the correct thing exists elsewhere and did
# not travel") committed by the person who had just written the §A3 section.
#
# Deliberately vocabulary of REPORTING, not of denial: "cited" in a sentence
# means someone else did the citing.
REPORTING = re.compile(
    r"\b(receipts?|cite[sd]?|citations?|dangling|the real test is|no referent|"
    r"retracted|reported as|stale name|says the|says that|none of which|"
    r"do not resolve|does not resolve)\b", re.I)

# Phrases whose point is that the referent never existed at all — a stronger
# claim than RETIRED's "it moved". A retraction must be able to name the thing
# it retracts without the audit counting that as a fresh citation.
NEVER_EXISTED = re.compile(
    r"\b(does not exist|do not exist|did not exist|never existed|has never "
    r"existed|not machine-checked|no such file|zero matches|was not true|"
    r"there is no|there was no|not yet written|yet to be written|"
    r"proposed:|planned:|none of that was true)\b", re.I)


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout


def known_names(root):
    """Every identifier that exists anywhere OUTSIDE a comment.

    One rule instead of a per-language list. The first version enumerated the
    places a name could live — fn, file stem, .lean, .cu, .metal — and kept
    being incomplete: it had no entry for `.pest` grammar rules, none for test
    fixtures named by their file, and none for `gpu_cuda_agrees_with_sim`,
    which is the shared PREFIX of `..._on_qft` and `..._on_rbs` and names a
    family rather than a function. Enumerating hiding places is a losing game;
    asking "does this token occur anywhere that is not prose" is not.

    Rust comment lines are stripped, because a name that appears ONLY in the
    comment citing it is exactly the defect being looked for.
    """
    tokens = set()
    files = [p for p in run(["find", ".", "-type", "f",
                             "-not", "-path", "*/target/*",
                             "-not", "-path", "./.git/*",
                             "-not", "-path", "./fixes/*",
                             "-not", "-path", "*/__pycache__/*",
                             "-not", "-path", "./.codegraph/*"], root).split() if p]
    for rel in files:
        path = root / rel
        if path.suffix in (".md", ".patch", ".lock", ".png", ".jpg", ".pdf"):
            continue
        # THE TOOL MUST NOT READ ITSELF, or the audit directory generally.
        # This file's own docstring quotes both receipts by name, so on the
        # first run the tool declared them to exist and reported neither. A
        # tool that records its findings in prose will suppress exactly those
        # findings unless its own prose is out of scope.
        if "tools/audit" in rel:
            continue
        try:
            txt = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if path.suffix == ".rs":
            txt = "\n".join(l for l in txt.split("\n")
                             if not l.strip().startswith(("//", "///", "//!", "*")))
        tokens |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", txt))
        tokens.add(path.stem)
    # A citation may name a FAMILY: `gpu_cuda_agrees_with_sim` for
    # `gpu_cuda_agrees_with_sim_on_qft` and `..._on_rbs`.
    prefixes = set()
    for t in tokens:
        parts = t.split("_")
        for k in range(3, len(parts)):
            prefixes.add("_".join(parts[:k]))
    return tokens | prefixes


# Two Lean trees, and for a long time this only knew about one.
#
# `verification/` is the specification-first tree (mostly stated targets).
# `proofs/lean4/` is the tree `ci.sh` actually builds, and where the real
# sorry-free proofs live. Matching only `Verification/` meant the audit
# guarded the nearly-empty tree and ignored every citation of the load-bearing
# one — 11 of them, none checked. Found 2026-09-04 when the quantum-core
# session named the mirror-image false positive on their side.
LEAN_CITE = re.compile(
    r"((?:Verification|proofs/lean4)/[A-Za-z0-9_/{},*]+\.lean)(?:::([A-Za-z0-9_]+))?")


def _lean_base(root, path):
    """Filesystem root the citation path is relative to."""
    return root if path.startswith("proofs/lean4/") else root / "verification"


def _glob_matches(path, present):
    """A citation may name a FAMILY with a wildcard: `Dir/ParamShift*.lean`.

    Until 2026-09-04 `*` was not in the path character class, so these were not
    matched and therefore never checked — a silent false negative, and the
    worst kind: the tool reported a clean Lean pass while a wildcard citation
    of a directory that does not exist sat in the tree. Found because the
    quantum-core session pointed its own checker at this repo and matched a
    form mine could not see.

    Returns every present file matching the glob. A glob resolves iff that
    list is non-empty — the family reading (any member satisfies the claim) —
    and a named theorem must be found in at least one MEMBER, which is why
    this returns the list rather than a bool. The first cut returned a bool
    and then read the literal `Dir/Foo*.lean` path to check the theorem,
    which cannot exist: a crash, latent in production only because the one
    wildcard citation in the tree sat inside a retraction and was suppressed
    before reaching it. The self-test fixture found it immediately.
    """
    import fnmatch
    return sorted(q for q in present if fnmatch.fnmatch(q, path))


def _expand_braces(path):
    """`Dir/{A,B}.lean` -> [`Dir/A.lean`, `Dir/B.lean`].

    Brace shorthand is how this repo cites a pair of sibling proofs, and taken
    literally it names a file that cannot exist — two real proofs reported as
    dangling. The quantum-core session flagged the mirror image of this class
    (emit-target `.lean` names that are outputs, not citations) before it bit
    here, which is the only reason it was checked rather than reported.
    """
    if "{" not in path:
        return [path]
    pre, rest = path.split("{", 1)
    inner, post = rest.split("}", 1)
    return [pre + alt.strip() + post for alt in inner.split(",") if alt.strip()]


def lean_citations(root):
    """`Verification/X.lean::theorem` citations that do not resolve.

    Exact, unlike the identifier pass: a path either names a file in this
    repository or it does not, and a theorem either appears in that file or it
    does not. No heuristic, no false positives.

    This is the strongest form of the A4 defect because the claim is
    "verified in Lean" — a reader who believes it stops checking.
    """
    present = {str(p.relative_to(root / "verification"))
               for p in (root / "verification").rglob("*.lean")} \
        if (root / "verification").is_dir() else set()
    present |= {str(p.relative_to(root))
                for p in (root / "proofs" / "lean4").rglob("*.lean")} \
        if (root / "proofs" / "lean4").is_dir() else set()
    bad, muted = [], []
    files = [p for p in run(["find", ".", "-name", "*.rs", "-o", "-name", "*.md"],
                            root).split() if p]
    for rel in files:
        if "/target/" in rel or rel.startswith("./fixes/") or "tools/audit" in rel:
            continue
        try:
            lines = (root / rel).read_text(encoding="utf-8", errors="replace").split("\n")
        except OSError:
            continue
        for i, line in enumerate(lines):
            # A retraction has to be able to NAME the path it is retracting.
            # The identifier pass has skipped retired names since it was
            # written; this pass did not, so documenting "there is no
            # Verification/Adjoint/Foo.lean" was itself reported as a dangling
            # citation — the tool objecting to its own finding being written
            # down. Window is the line plus its two neighbours, because these
            # retractions wrap.
            hits_on_line = list(LEAN_CITE.finditer(line))
            if not hits_on_line:
                continue
            ctx = " ".join(lines[max(0, i - 2):i + 3])
            if (RETIRED.search(ctx) or NEVER_EXISTED.search(ctx)
                    or REPORTING.search(ctx)):
                # Counted, not dropped. Every suppression here is a pattern
                # match on prose, and a pattern that over-matches deletes a
                # real finding with no trace — verifying that required
                # hand-disabling the filter, which is not a standing signal.
                # A count that MOVES is the signal to look. Adapted from the
                # quantum-core session's "resolved and reported, never failed
                # on" design for planning docs.
                #
                # The first cut counted every LINE whose context matched,
                # citation or not, and reported 3950 suppressions. Absurdity
                # was the tell, exactly as that session described: a plausible
                # number would have been believed.
                muted.extend((rel, i + 1, "lean") for _ in hits_on_line)
                continue
            for m in hits_on_line:
                raw, thm = m.group(1), m.group(2)
                for path in _expand_braces(raw):
                    hits = (_glob_matches(path, present) if "*" in path
                            else ([path] if path in present else []))
                    if not hits:
                        bad.append((rel, i + 1, path, thm, "file absent"))
                        continue
                    if not thm:
                        continue
                    pat = rf"\b(theorem|lemma|def|axiom)\s+{re.escape(thm)}\b"
                    if not any(re.search(pat, (_lean_base(root, h) / h).read_text(
                            encoding="utf-8", errors="replace")) for h in hits):
                        bad.append((rel, i + 1, path, thm, "theorem absent"))
    return bad, present, muted


def prose_blocks(lines):
    """Paragraphs — the markdown analogue of a comment block.

    Markdown has no comment marker: the whole file is prose, so the claim unit
    is a paragraph (a run of non-blank lines). Fenced code blocks are SKIPPED,
    which is load-bearing rather than tidy — the quantum-core session found
    that its only two unresolved doc claims sat inside fences showing what a
    generator *emits*, `:= by sorry` and all. A fenced sample is an
    illustration, not an assertion, and reading one as a citation manufactures
    a finding out of correct documentation.
    """
    blocks, start, fenced = [], None, False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            if start is not None:
                blocks.append((start, i))
                start = None
            continue
        if fenced:
            continue
        if line.strip():
            if start is None:
                start = i
        elif start is not None:
            blocks.append((start, i))
            start = None
    if start is not None:
        blocks.append((start, len(lines)))
    return blocks


def comment_blocks(lines):
    """Contiguous runs of comment lines — the natural unit for a claim.

    Replaces an arbitrary +/-N line window, which bled both ways: a
    "renamed from X" line suppressed a real citation four lines below it, and
    an unrelated neighbour's verb promoted a passing mention to "evidence".
    Doc comments are separated by code, so a block is exactly one claim.
    """
    blocks, i, n = [], 0, len(lines)
    while i < n:
        if lines[i].strip().startswith(("//", "/*", "*")):
            j = i
            while j < n and lines[j].strip().startswith(("//", "/*", "*")):
                j += 1
            blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks


def sentence_around(block, name):
    """The sentence containing `name`, for the retirement test."""
    for part in re.split(r"(?<=[.;])\s+", block):
        if f"`{name}`" in part:
            return part
    return block


def main():
    ap = argparse.ArgumentParser(description="Doc citations with no referent (audit A4)")
    ap.add_argument("--all", action="store_true",
                    help="report every dangling citation, not just evidence claims")
    ap.add_argument("--muted", action="store_true",
                    help="list the citations suppressed as retraction / "
                         "findings-register prose, rather than only counting them")
    ap.add_argument("--allow", default="tools/audit/dangling_citations.allow",
                    help="file of `name  # reason` lines for referents that are "
                         "real but structurally invisible here (external ABI "
                         "symbols, fixtures in gitignored trees)")
    args = ap.parse_args()
    root = pathlib.Path(os.environ.get("ARIA_AUDIT_ROOT")
                        or pathlib.Path(__file__).resolve().parents[2]).resolve()

    # Names whose referent is real but cannot be seen from this tree: WASI ABI
    # symbols, benchmark fixtures under gitignored `fixes/`. Exempting these is
    # NOT the same as excusing a dangling citation — an entry must name why the
    # referent is invisible, and anything editorial (a proof this repo does not
    # ship, a test in the private monorepo) stays a finding.
    allow = {}
    allow_path = root / args.allow
    if allow_path.exists():
        for line in allow_path.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, why = line.partition("#")
            allow[key.strip()] = why.strip()

    lean_bad, lean_present, lean_muted = lean_citations(root)
    names = known_names(root)
    # Markdown was excluded until 2026-09-04, and that was the wrong half to
    # skip: code comments are read by contributors, but the docs are where the
    # repo makes its claims to everyone else. The quantum-core session found
    # exactly this blind spot on its own tree — same tool, same day, opposite
    # file type — and its public proof claims lived in the markdown it had
    # never audited.
    rs = [p for p in run(["find", ".",
                          "(", "-name", "*.rs", "-o", "-name", "*.md", ")",
                          "-not", "-path", "./target/*",
                          "-not", "-path", "./.git/*",
                          "-not", "-path", "./fixes/*",
                          "-not", "-path", "*/tools/audit/*"], root).split() if p]

    findings, weak, muted = [], [], []
    for rel in sorted(rs):
        lines = (root / rel).read_text(encoding="utf-8", errors="replace").split("\n")
        blocks = prose_blocks(lines) if rel.endswith(".md") else comment_blocks(lines)
        for start, end in blocks:
            block = " ".join(lines[k].strip().lstrip("/*! ") for k in range(start, end))
            evidence = bool(EVIDENCE.search(block))
            for i in range(start, end):
                for m in CITE.finditer(lines[i]):
                    name = m.group(1)
                    if name in names or name in allow:
                        continue
                    # Retirement is a property of the SENTENCE the name sits
                    # in — "renamed from `X`" — not of the paragraph. Scoping
                    # it to the block let one retired name silence unrelated
                    # citations beside it.
                    sent = sentence_around(block, name)
                    if (RETIRED.search(sent) or NEVER_EXISTED.search(sent)
                            or REPORTING.search(sent)):
                        muted.append((rel, i + 1, name))
                        continue
                    (findings if evidence else weak).append(
                        (rel, i + 1, name, lines[i].strip()[:110]))

    print(f"{len(findings)} citation(s) presented as EVIDENCE with no referent\n")
    for rel, ln, name, s in findings:
        print(f"  {rel}:{ln}")
        print(f"      cites `{name}` — not a fn, file, field, kernel or theorem")
        print(f"      {s}\n")
    print(f"({len(weak)} further dangling citations are not framed as evidence; "
          f"--all to list)")

    # Suppression is a judgement the tool makes silently, so it is reported.
    # These are citations whose surrounding prose says the referent is gone or
    # is being reported rather than claimed. The number is expected to be
    # non-zero and stable; a JUMP means a pattern started over-matching and is
    # now eating real findings. Never a failure on its own — the point is that
    # it is visible without hand-disabling the filter.
    total_muted = len(muted) + len(lean_muted)
    if total_muted:
        print(f"\n{total_muted} citation(s) suppressed as retraction or "
              f"findings-register prose ({len(muted)} identifier, "
              f"{len(lean_muted)} Lean) — not failures; "
              f"a CHANGE in this count is the signal. --muted to list")
        if args.muted:
            for rel, ln, what in sorted(muted + lean_muted):
                print(f"      {rel}:{ln}  {what}")

    print(f"\n{len(lean_bad)} Lean citation(s) that do not resolve "
          f"({len(lean_present)} .lean file(s) present across verification/ "
          f"and proofs/lean4/)")
    by_dir = {}
    for rel, ln, path, thm, why in lean_bad:
        by_dir.setdefault(path.rsplit("/", 1)[0], []).append((rel, ln, thm, why))
    for d, v in sorted(by_dir.items()):
        print(f"\n  {d}/ — {len(v)} citation(s)")
        for rel, ln, thm, why in v:
            print(f"      {rel}:{ln}  ::{thm or '(no theorem named)'}  [{why}]")
    # WHAT A GREEN RUN DOES NOT MEAN.
    #
    # This checks REFERENTIAL INTEGRITY only: the named thing exists. It says
    # nothing about whether the claim ABOUT that thing is true. A cited test
    # can exist and assert nothing; a cited .lean file can exist while the
    # theorem count or property described beside it has drifted; a cited
    # constant can exist with a different value. Those are the expensive half
    # and none of it is built.
    #
    # Stated in the OUTPUT rather than only in this docstring, because the
    # failure mode is a reader taking a green gate for more coverage than it
    # has — which is the same defect this audit found in `adjoint.rs`, where a
    # CI gate that did not exist was cited as enforcing proofs that did not
    # exist. A gate overstating its own scope is that defect one level up.
    print("\n  scope: referential integrity only — that each cited name EXISTS. "
          "Whether\n         the claim made about it is true is not checked "
          "by this or any gate here.")

    if args.all:
        for rel, ln, name, s in weak:
            print(f"  {rel}:{ln}  `{name}`")
    return 1 if (findings or lean_bad) else 0


if __name__ == "__main__":
    sys.exit(main())
