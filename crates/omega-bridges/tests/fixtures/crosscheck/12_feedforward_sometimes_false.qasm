// Classically-conditioned gate whose condition is SOMETIMES FALSE.
//
// This is the discriminating feedforward fixture. q0 is put in an equal
// superposition and measured, so c==1 roughly half the time and the guarded
// X on q1 fires on only half the shots. The resulting distribution has
// weight on |00> and |11> and none on |01>/|10>.
//
// Contrast 14_feedforward_always_true.qasm, whose condition ALWAYS holds: an
// engine that drops the guard entirely and applies the gate unconditionally
// reproduces that fixture exactly, and only disagrees here. A corpus with
// only always-true conditions cannot distinguish "honours the guard" from
// "ignores the guard", which is why both are present.
//
// MEASURED on Qiskit Aer, 20000 shots, rather than argued:
//   guard honoured:  {"1": 10105, "0": 9895}   (~50/50)
//   guard dropped:   {"1": 20000}              (deterministic)
// A ~50/50 split versus a single bitstring is unmissable.
//
// Aria fixed exactly this defect class in 11888a9 / ae6da5c, and the shared
// corpus could not reach it until this file existed.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[1];
h q[0];
measure q[0] -> c[0];
if (c==1) x q[1];
measure q[1] -> c[0];
