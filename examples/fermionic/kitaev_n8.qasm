OPENQASM 2.0;
include "qelib1.inc";
// Kitaev sweet spot, N = 8. |+> on every mode.
qreg q[8];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
