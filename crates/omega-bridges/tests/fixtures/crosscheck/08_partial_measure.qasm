// Partial measurement into a narrower creg: pins the classical-bit
// mapping (unmeasured clbits must stay 0 on both sides).
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[2];
h q[0];
cx q[0],q[1];
h q[2];
barrier q;
measure q[0] -> c[0];
measure q[1] -> c[1];
