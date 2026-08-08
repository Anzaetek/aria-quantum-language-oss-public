// Mid-circuit reset on a qubit in superposition.
//
// The discriminating mechanism, verified rather than asserted:
//
//   WITH reset:    H |0> -> superposition, reset -> |0>, H -> superposition.
//                  q0 measures 50/50, so all FOUR outcomes appear.
//   no-op reset:   the two H's compose to the identity, so q0 is
//                  deterministically 0 and only TWO outcomes appear.
//
// Measured on Qiskit Aer, 20000 shots:
//   with reset:    {"00":5057,"01":4952,"10":4899,"11":5092}
//   reset deleted: {"00":9927,"01":10073}
//
// So a backend that silently ignores reset halves the support. That is an
// unmissable difference, not a small numeric one — which is the property worth
// having in a fixture.
//
// NOTE ON SKIPPING: `reset` is handled STRUCTURALLY by the corpus gate scan
// (it is not reported as a gate name), so unlike `if` this fixture is NOT
// auto-skipped by a runner that cannot do it. A runner that silently ignores
// reset will therefore produce a wrong number rather than a skip. That is a
// gap in the scan, recorded here so the next reader does not assume the skip
// machinery protects this case.
//
// Aria's Reset acceptance also diverges across backends (five distinct
// policies, ledger A6), so this fixture is where that shows up.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
h q[1];
reset q[0];
h q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
