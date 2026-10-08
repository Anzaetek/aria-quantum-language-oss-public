OPENQASM 2.0;
include "qelib1.inc";
// Kitaev sweet spot, N = 2. |+> on every mode.
qreg q[2];
h q[0];
h q[1];
