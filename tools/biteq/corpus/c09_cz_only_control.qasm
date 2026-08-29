// Negative control: no multi-control gate at all — two backends on plain
// Cliffords. A diff here is not about the decomposition.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
sx q[2];
cz q[0],q[1];
cx q[1],q[2];
cz q[2],q[3];
h q[1];
h q[3];
