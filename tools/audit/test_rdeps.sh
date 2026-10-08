#!/usr/bin/env bash
# Self-test for tools/audit/rdeps.py (audit A9).
#
# Builds a synthetic workspace containing, deliberately, one instance of each
# hole that made the three obvious answers wrong on the real repo:
#
#   plain      — an ordinary dependent. `cargo tree -i` finds this one.
#   opt        — an OPTIONAL dependent. `cargo tree -i --workspace` returns
#                nothing for a crate whose every dependent is optional.
#   excluded   — a dependent EXCLUDED from the workspace. Both cargo answers
#                miss it; only grep finds it.
#   commenter  — names the target in a COMMENT and does not depend on it.
#                grep reports this one; it is a false positive.
#   indirect   — depends on `plain`, so it is in the transitive closure but
#                not the direct one.
#
# The tool is correct iff it finds the first three and the fifth, and does not
# find the fourth. Asserting all five in one fixture is the point: each of the
# three tools this replaces passes some of them.
set -euo pipefail

RDEPS="$(cd "$(dirname "$0")" && pwd)/rdeps.py"
FIX="$(mktemp -d)"
trap 'rm -rf "$FIX"' EXIT

mk() { # mk <dir> <name> <cargo-toml-body>
  mkdir -p "$FIX/$1/src"
  echo "pub fn f() {}" > "$FIX/$1/src/lib.rs"
  { echo '[package]'; echo "name = \"$2\""; echo 'version = "0.1.0"';
    echo 'edition = "2021"'; echo; printf '%s\n' "$3"; } > "$FIX/$1/Cargo.toml"
}

cat > "$FIX/Cargo.toml" <<'TOML'
[workspace]
resolver = "2"
members = ["target-crate", "plain", "opt", "commenter", "indirect"]
exclude = ["excluded"]
TOML

mk target-crate target-crate ''
mk plain     plain     '[dependencies]
target-crate = { path = "../target-crate" }'
mk opt       opt       '[dependencies]
target-crate = { path = "../target-crate", optional = true }

[features]
gpu = ["dep:target-crate"]'
mk commenter commenter '# mirrors target-crate, but does not depend on it
[dependencies]'
mk indirect  indirect  '[dependencies]
plain = { path = "../plain" }'
mk excluded  excluded  '[dependencies]
target-crate = { path = "../target-crate", optional = true }'

fail=0
expect_in()   { grep -q "^   $1 " <<<"$2" || { echo "  FAIL: expected '$1' in:"; echo "$2"; fail=1; }; }
expect_out()  { grep -q "^   $1 " <<<"$2" && { echo "  FAIL: '$1' should NOT be listed"; fail=1; } || true; }

direct="$(ARIA_RDEPS_REPO="$FIX" "$RDEPS" target-crate --direct)"
expect_in  plain     "$direct"
expect_in  opt       "$direct"     # cargo tree -i misses this
expect_in  excluded  "$direct"     # cargo metadata --no-deps misses this
expect_out commenter "$direct"     # grep false-positives on this
expect_out indirect  "$direct"     # not a DIRECT dependent

grep -q 'excluded .*\[OUTSIDE WORKSPACE\]' <<<"$direct" || {
  echo "  FAIL: 'excluded' must be flagged as unreachable by \`-p\`"; fail=1; }

trans="$(ARIA_RDEPS_REPO="$FIX" "$RDEPS" target-crate)"
expect_in  indirect  "$trans"      # transitive closure must reach it
expect_out commenter "$trans"

# A crate nothing depends on must say so, not print an empty list that reads
# like an answer.
leaf="$(ARIA_RDEPS_REPO="$FIX" "$RDEPS" indirect)"
grep -q 'No reverse dependencies' <<<"$leaf" || {
  echo "  FAIL: a leaf crate must say it has no reverse dependencies"; fail=1; }

if [ "$fail" -ne 0 ]; then exit 1; fi
echo "  OK: rdeps finds optional + workspace-excluded dependents, rejects comment-only matches"
