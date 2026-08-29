// Amplitude everywhere before the CCX, computation after, so a wrong bit
// cannot hide in a zero and an error propagates.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
h q[3];
s q[1];
sx q[2];
cx q[0],q[3];
ccx q[0],q[1],q[2];
h q[0];
h q[1];
h q[2];
h q[3];
