// crz is outside BOTH gate sets — neither backend may claim this
// fixture, and the filter must drop it rather than the runner
// approximating it.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
h q[1];
crz(0.7) q[0],q[1];
measure q -> c;
