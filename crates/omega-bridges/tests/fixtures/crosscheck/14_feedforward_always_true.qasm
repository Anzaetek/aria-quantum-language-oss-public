// Classically-conditioned gate whose condition ALWAYS holds — the CONTROL
// case for 12_feedforward_sometimes_false.qasm.
//
// X q0 makes the measurement deterministic, so c==1 on every shot and the
// guarded gate always fires. The output is a single bitstring.
//
// This fixture on its own proves almost nothing about feedforward. MEASURED on
// Qiskit Aer, 20000 shots:
//   guard honoured:  {"1": 20000}
//   guard dropped:   {"1": 20000}   <- IDENTICAL
// An engine that ignores the condition entirely passes this fixture perfectly.
//
// That is the point of keeping it. Paired with fixture 12 (which separates the
// two cases 50/50 vs deterministic), a failure pattern of "12 disagrees, 14
// agrees" localises the defect to guard HANDLING rather than to gate
// application — and it is a standing demonstration that adding conditionals to
// a corpus is not the same as testing them.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[1];
x q[0];
measure q[0] -> c[0];
if (c==1) x q[1];
measure q[1] -> c[0];
