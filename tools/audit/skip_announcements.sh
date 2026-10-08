#!/usr/bin/env bash
# Audit A8 — "Skips that read as passes" (PLAN-OPEN-20260825.md §3f).
#
# RECEIPT. A green ci.sh run once ended "All CI stages passed — nothing
# skipped." on a host where FOUR stages had not run: the socket stage under
# ARIA_SKIP_SOCKET=1, the socket client test when the server did not come up,
# the Lean proof tree when `lake` was absent under ARIA_LEAN=1, and the
# CV/piquasso drift check when the venv was absent under ARIA_CV_XCHECK=1.
# Each printed its own inline "SKIP:" line, which scrolls away in a 3000-line
# log, and none of them reached SKIPPED_STAGES, which is the line a reader —
# or an agent reading the tail — actually acts on.
#
# Two of the four were in the *enabled* arm, which is the worse half: the
# operator had explicitly asked for that check and was told it passed.
#
# THE INVARIANT. There is exactly ONE way to announce a skip: the `skipped`
# helper, which registers into SKIPPED_STAGES and prints. An `echo` that says
# "SKIP:" is an announcement that bypasses the register, so it is banned
# outright rather than paired-up by static analysis: pairing would need to
# match `if`/`else`/`fi` nesting in bash, and a check that is fragile in the
# same place the defect lives is not a check.
#
# Explanatory prose next to a *registered* skip is fine and stays — it just
# must not use the "SKIP:" announce token, which is reserved for the thing
# that does not register. Those sites say "note:".
#
# NOT CLAIMED: that this catches every silent skip. It catches the shape that
# has actually recurred here four times. A stage that quietly does nothing
# without saying anything at all is invisible to this and to every grep.
set -euo pipefail

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=("ci.sh")

fail=0

for t in "${targets[@]}"; do
  if [ ! -f "$t" ]; then
    echo "  FAIL: $t does not exist" >&2
    fail=1
    continue
  fi

  # Comment lines are exempt: this file's own receipt, and ci.sh's comments
  # recording past fixes, both have to be able to name the token.
  unregistered=$(awk '
    /^[[:space:]]*#/ { next }
    /echo .*SKIP:/   { printf "  %s:%d: %s\n", FILENAME, FNR, $0 }
  ' "$t" || true)

  if [ -n "$unregistered" ]; then
    echo "  FAIL: $t announces a skip without registering it in SKIPPED_STAGES." >&2
    printf '%s\n' "$unregistered" >&2
    echo "        Use \`skipped \"<stage> — <why>\"\` instead of \`echo \"  SKIP: ...\"\`." >&2
    echo "        If the line is prose beside a skip that IS registered, say \"note:\"." >&2
    fail=1
  fi

  # The summary is only load-bearing if the COUNT is in it: "some stages were
  # skipped" reads the same whether it is 1 or 12, and the erosion this audit
  # exists to catch is gradual. Guard the count itself, not just the list.
  #
  # It must be in a PRINTING position. The first version of this check grepped
  # for ${#SKIPPED_STAGES[@]} anywhere in the file, and passed on a ci.sh whose
  # summary had been reduced to "some stage(s) did NOT run" — because
  # `if [ ${#SKIPPED_STAGES[@]} -eq 0 ]` two lines above still matched. An
  # assertion true of the thing it was written to exclude, which is audit A2,
  # inside the tool for A8. Caught by fault injection, not by reading it.
  if ! grep -qE '(printf|echo).*\$\{#SKIPPED_STAGES\[@\]\}' "$t"; then
    echo "  FAIL: $t never PRINTS \${#SKIPPED_STAGES[@]} — the skip count is" >&2
    echo "        what makes the final line comparable between runs. Testing the" >&2
    echo "        count does not count; it has to reach the reader." >&2
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  exit 1
fi

echo "  OK: every skip announcement in ${targets[*]} registers in SKIPPED_STAGES"
