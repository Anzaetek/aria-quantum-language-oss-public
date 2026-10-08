#!/usr/bin/env bash
# Self-test for tools/audit/dangling_citations.py (audit A4).
#
# The fixture carries one citation per way this check has actually been wrong.
# Every case below is a real false positive or false negative the tool produced
# on this repo before it was fixed:
#
#   real_dangling_citation        — cited as evidence, exists nowhere.        REPORT.
#   a_real_test_function      — exists as `fn`.                           silent.
#   integration_test_by_file — exists only as tests/<name>.rs.           silent.
#   kernel_side_device_name    — exists only in a .cu file.                silent.
#   grammar_rule_pest_name   — exists only in a .pest file.              silent.
#   family_prefix_shared       — PREFIX of family_prefix_shared_shared_on_qft.           silent.
#   retired_old_name        — cited as "renamed from"; absence is the
#                         point of the sentence.                    silent.
#   mentioned_but_not_cited — dangling but not framed as evidence.      not a finding.
#
# Each case sits on its OWN function: with them stacked, the +/-3 line context
# of the 'Renamed from' case bled onto its neighbours and silently suppressed
# two of them. That is the same bleeding the tool itself was fixed for.
#
# Plus the three contaminations that made the tool go quiet on findings it had
# already produced: reading its own source, reading __pycache__, and reading a
# NESTED target/ whose .rmeta artifacts embed every doc comment in the tree.
set -euo pipefail

TOOL="$(cd "$(dirname "$0")" && pwd)/dangling_citations.py"
FIX="$(mktemp -d)"
trap 'rm -rf "$FIX"' EXIT
mkdir -p "$FIX/crates/f/src" "$FIX/crates/f/tests" "$FIX/crates/f/cuda" \
         "$FIX/crates/f/grammar" "$FIX/verification/Verification/Backend"

cat > "$FIX/crates/f/src/lib.rs" <<'RS'
/// MEASURED: removing this fails `real_dangling_citation` on the fixture.
pub fn a() {}

/// Pinned by `a_real_test_function`, which asserts it.
pub fn b() {}

/// Covered by `integration_test_by_file` end to end.
pub fn c() {}

/// Proven against the `kernel_side_device_name` device path.
pub fn d() {}

/// The parser proves `grammar_rule_pest_name` matches first.
pub fn e() {}

/// Exercised by `family_prefix_shared` in both shapes.
pub fn f() {}

/// Renamed from `retired_old_name`, which promised more than it checked.
pub fn g() {}

/// See also `mentioned_but_not_cited` for background.
pub fn h() {}

/// Verified in Lean: `Verification/Backend/Glob*.lean` resolves — family present.
/// Verified in Lean: `Verification/Absent/Glob*.lean` must NOT resolve.
/// Verified in Lean: `Verification/Backend/Present.lean::real_theorem`
/// Verified in Lean: `Verification/Backend/Present.lean::absent_theorem`
/// Verified in Lean: `Verification/Absent/Missing.lean::whatever`
pub fn i() {}

fn a_real_test_function() {}
fn family_prefix_shared_on_qft() {}
RS
echo "fn placeholder() {}" > "$FIX/crates/f/tests/integration_test_by_file.rs"
echo "__global__ void kernel_side_device_name() {}" > "$FIX/crates/f/cuda/k.cu"
echo "grammar_rule_pest_name = { \"x\" }" > "$FIX/crates/f/grammar/g.pest"
cat > "$FIX/verification/Verification/Backend/Present.lean" <<'LEAN'
theorem real_theorem : True := trivial
LEAN
# A wildcard citation names a FAMILY. `*` was absent from the path character
# class until 2026-09-04, so these were never matched and never checked — the
# tool reported a clean Lean pass while a wildcard citation of a missing
# directory sat in the tree. Both polarities, or a broken glob resolver looks
# identical to a clean run.
echo 'theorem glob_family_member : True := trivial' \
  > "$FIX/verification/Verification/Backend/GlobOne.lean"

# --- markdown arm ---------------------------------------------------------
# Added 2026-09-04 with the markdown pass. It shipped with NO self-test
# coverage: the suite passed and said nothing about the new code path, which
# is the precise trap the quantum-core session named — an extractor bug and a
# real finding are indistinguishable from the output alone. A gate over a
# surface has to be tested against a known-good AND a known-bad name on that
# surface, or it only proves the tool still runs.
cat > "$FIX/DOCS.md" <<'MD'
Pinned by `md_dangling_citation`, which asserts it end to end.

Pinned by `a_real_test_function`, which is a real fn.

The register below reports a dead name rather than claiming it:
that comment cited `md_reported_not_claimed` as its evidence.

A fenced sample is an illustration, not an assertion:

```
theorem md_inside_a_fence_only : True := trivial
Pinned by `md_fenced_name_never_real`.
```
MD

run_tool() { ARIA_AUDIT_ROOT="$FIX" "$TOOL" 2>&1 || true; }
out="$(run_tool)"
fail=0
want()  { grep -q "$1" <<<"$out" || { echo "  FAIL: expected to see $1"; fail=1; }; }
deny()  { grep -q "$1" <<<"$out" && { echo "  FAIL: must NOT report $1"; fail=1; } || true; }

want '`real_dangling_citation`'
deny '`a_real_test_function`'
deny '`integration_test_by_file`'
deny '`kernel_side_device_name`'
deny '`grammar_rule_pest_name`'
deny '`family_prefix_shared`'
deny '`retired_old_name`'
grep -q "cites \`mentioned_but_not_cited\`" <<<"$out" && {
  echo "  FAIL: a non-evidence mention must not be a finding"; fail=1; } || true

# Lean pass: exact, so all three verdicts must be exactly right.
want 'Verification/Absent/'
# The report groups by DIRECTORY and prints rel:ln, so the glob's filename
# never appears in the output — assert the count instead. Without glob
# support this is 1 (Missing.lean only); with it, 2.
want 'Verification/Absent/ — 2 citation'
grep -q 'Backend/Glob\*' <<<"$out" && {
  echo "  FAIL: a glob matching a present family was reported"; fail=1; } || true
want 'absent_theorem'
grep -q 'real_theorem' <<<"$out" && { echo "  FAIL: a resolving theorem was reported"; fail=1; } || true

# Suppression must be COUNTED, not silent. A pattern that over-matches
# deletes a real finding with no trace; the count is the standing signal that
# replaces hand-disabling the filter. The fixture has exactly one retired
# citation and one findings-register mention, so it must report suppressions.
want 'suppressed as retraction'
grep -qE '[1-9][0-9]* citation\(s\) suppressed' <<<"$out" || {
  echo "  FAIL: suppressions must be counted, not silently dropped"; fail=1; }

# Markdown: the docs are a claim surface, so the same rules must hold there.
want '`md_dangling_citation`'
deny '`md_fenced_name_never_real`'   # inside a fence: an illustration, not a claim
grep -q 'cites `md_reported_not_claimed`' <<<"$out" && {
  echo "  FAIL: a findings register REPORTS a dead name; it does not cite it"; fail=1; } || true
grep -q 'DOCS.md' <<<"$out" || { echo "  FAIL: markdown was not scanned at all"; fail=1; }

# --- contamination: each of these made the tool go silent on a real finding ---
contaminate() {
  local label="$1" path="$2"
  mkdir -p "$(dirname "$FIX/$path")"
  printf 'real_dangling_citation\n' > "$FIX/$path"
  local o; o="$(run_tool)"
  rm -f "$FIX/$path"
  grep -q '`real_dangling_citation`' <<<"$o" || { echo "  FAIL: $label suppressed a real finding"; fail=1; }
}
contaminate "a nested target/ artifact"    "crates/f/target/debug/deps/lib.rmeta"
contaminate "a __pycache__ artifact"       "tools/x/__pycache__/m.pyc"
contaminate "the audit tools' own prose"   "tools/audit/notes.txt"

[ "$fail" -eq 0 ] || exit 1
echo "  OK: A4 resolves fn/file/kernel/grammar/prefix/retired across .rs AND .md,"
echo "      honours fences and findings-register prose, ignores build + audit artifacts"
