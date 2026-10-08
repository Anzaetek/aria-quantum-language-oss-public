OPENQASM 2.0;
include "qelib1.inc";
// Kitaev sweet spot, N = 8, then T on modes 1 and 5 (a local kick of the chemical potential).
qreg q[8];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
t q[1];
t q[5];
