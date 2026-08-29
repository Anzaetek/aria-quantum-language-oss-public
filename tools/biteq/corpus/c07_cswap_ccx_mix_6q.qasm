// Widest case: both gates in one state, cz/sx interleave, 6 qubits.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
ccx q[0],q[2],q[4];
sx q[1];
cswap q[1],q[3],q[5];
cz q[0],q[5];
ccx q[5],q[3],q[1];
h q[0];
h q[3];
h q[5];
