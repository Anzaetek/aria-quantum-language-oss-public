#!/usr/bin/env bash
# Run the TLA+ models with a project-local TLC.
#
# Opt-in: ./ci.sh must not acquire a JVM dependency (K13), so this is invoked
# by hand or by an ARIA_TLA=1 stage. Skips cleanly when the tooling is absent
# rather than failing, matching the Qiskit cross-check's contract.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
specs="$here/../../proofs/tla"
jar="$here/tla2tools.jar"

# macOS ships a `java` stub that only offers to install a JDK, so probe for a
# real one rather than trusting `command -v java`.
java_bin=""
for cand in "${JAVA_HOME:-}/bin/java" \
            /opt/homebrew/opt/openjdk@17/bin/java \
            /opt/homebrew/opt/openjdk/bin/java; do
  [ -x "$cand" ] && { java_bin="$cand"; break; }
done
if [ -z "$java_bin" ] && command -v java >/dev/null 2>&1 \
   && java -version >/dev/null 2>&1; then
  java_bin="java"
fi

if [ -z "$java_bin" ]; then
  echo "  SKIP: no JDK found (brew install openjdk@17, or set JAVA_HOME)"
  exit 0
fi
if [ ! -f "$jar" ]; then
  echo "  SKIP: $jar missing — fetch it with:"
  echo "    curl -fsSL -o $jar \\"
  echo "      https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar"
  exit 0
fi

run() { # <cfg> <module> <description>
  echo "==> $3"
  ( cd "$specs" && "$java_bin" -XX:+UseParallelGC -jar "$jar" \
      -nowarning -config "$1" "$2" ) 2>&1 | tail -"${TAIL:-6}"
}

# Safety must hold. A failure here is a real defect.
run MCSafety.cfg MCGovernor.tla "Governor: safety (capacity, no double-accounting)"

# Liveness is EXPECTED to fail until a bounded queue exists: the counterexample
# is the documented finding, not a regression. See proofs/tla/README.md.
echo
echo "==> Governor: liveness (expected to FAIL — starvation under try_acquire)"
( cd "$specs" && "$java_bin" -XX:+UseParallelGC -jar "$jar" \
    -nowarning -config MCGovernor.cfg MCGovernor.tla ) 2>&1 \
  | grep -aE 'Temporal properties were violated|No error has been found' || true
