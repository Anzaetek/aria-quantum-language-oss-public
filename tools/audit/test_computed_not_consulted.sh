#!/usr/bin/env bash
# Self-test for tools/audit/computed_not_consulted.py (audit A1).
#
# The fixture carries one field per way the check can be fooled:
#
#   gated       — read in an `if` in production. Consumed.
#   arithmetic  — read as `let p = self.arithmetic_bound;` with no comparison
#                 anywhere. Consumed. An early version demanded a comparison
#                 operator and reported MagicStateProtocol.input_infidelity,
#                 which seeds the whole distillation recurrence.
#   printed     — reaches the user only through `format!`. NOT consumed:
#                 printing cannot change what the program does, which is the
#                 entire content of this audit.
#   tested      — asserted on in `#[cfg(test)]` and nowhere else. NOT consumed:
#                 the dropped_mass defect had test assertions the whole time.
#   shadowed    — a field whose name is also a METHOD on another type. The
#                 method's call sites must not be credited to the field.
#
# The tool is correct iff it reports exactly {printed, tested, shadowed}.
set -euo pipefail

TOOL="$(cd "$(dirname "$0")" && pwd)/computed_not_consulted.py"
FIX="$(mktemp -d)"
trap 'rm -rf "$FIX"' EXIT
mkdir -p "$FIX/crates/f/src"

cat > "$FIX/crates/f/src/lib.rs" <<'RS'
pub struct Report {
    pub gated_bound: f64,
    pub arithmetic_bound: f64,
    pub printed_error: f64,
    pub tested_fidelity: f64,
    pub shadowed_residual: f64,
}

pub struct Other;
impl Other {
    pub fn shadowed_residual(&self, x: f64) -> f64 { x }
}

pub fn decide(r: &Report) -> bool {
    if r.gated_bound > 1e-6 {
        return false;
    }
    let mut p = r.arithmetic_bound;
    p *= 0.5;
    p < 1.0
}

pub fn show(r: &Report) -> String {
    format!("error {}", r.printed_error)
}

pub fn use_the_method(o: &Other) -> f64 {
    o.shadowed_residual(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn t() {
        let r = Report { gated_bound: 0.0, arithmetic_bound: 0.0, printed_error: 0.0,
                         tested_fidelity: 1.0, shadowed_residual: 0.0 };
        assert!(r.tested_fidelity > 0.5);
        assert!(r.shadowed_residual < 1.0);
    }
}
RS

out="$(ARIA_AUDIT_ROOT="$FIX" "$TOOL" --allow /nonexistent 2>&1 || true)"
fail=0
want_finding() { grep -q "FINDING] Report.$1" <<<"$out" || { echo "  FAIL: $1 should be reported"; fail=1; }; }
want_clean()   { grep -q "FINDING] Report.$1" <<<"$out" && { echo "  FAIL: $1 must NOT be reported"; fail=1; } || true; }

want_clean   gated_bound
want_clean   arithmetic_bound       # no comparison operator, still load-bearing
want_finding printed_error          # format! is not a consumer
want_finding tested_fidelity        # a test assertion is not a consumer
want_finding shadowed_residual      # the same-named method must not excuse it

# The allowlist has to actually suppress, or nobody will use it and the check
# will be disabled wholesale instead.
echo "Report.printed_error  # deliberately output-only" > "$FIX/allow.txt"
out2="$(ARIA_AUDIT_ROOT="$FIX" "$TOOL" --allow "$FIX/allow.txt" 2>&1 || true)"
grep -q "EXEMPT] Report.printed_error" <<<"$out2" || { echo "  FAIL: allowlist did not exempt"; fail=1; }
grep -q "FINDING] Report.tested_fidelity" <<<"$out2" || { echo "  FAIL: allowlist over-suppressed"; fail=1; }

[ "$fail" -eq 0 ] || exit 1
echo "  OK: A1 separates production reads from printing, tests, and same-named methods"
