OPENQASM 2.0;
include "qelib1.inc";
// Kitaev sweet spot, N = 2. |-> on every mode (global parity of |+>^N).
qreg q[2];
h q[0];
h q[1];
z q[0];
z q[1];
