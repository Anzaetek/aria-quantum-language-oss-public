// Adjacent triple then strided triple: slot-layout and stride differences.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
h q[0];
h q[1];
h q[2];
h q[5];
ccx q[0],q[1],q[2];
ccx q[0],q[2],q[5];
h q[0];
h q[2];
h q[5];
