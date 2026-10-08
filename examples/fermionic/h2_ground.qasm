OPENQASM 2.0;
include "qelib1.inc";
// Hartree-Fock plus exp(-i theta X0 X1 X2 Y3).
// theta = -0.11306813351297684. One parameter is the FCI state for this molecule.
// X to Z is h; Y to Z is sdg then h; the Z string is a CX ladder and rz(2*theta).
qreg q[4];
x q[0];
x q[1];
sdg q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.22613626702595369) q[3];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[1];
h q[2];
h q[3];
s q[3];
