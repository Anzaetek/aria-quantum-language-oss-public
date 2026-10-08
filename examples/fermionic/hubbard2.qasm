OPENQASM 2.0;
include "qelib1.inc";
// 2-site Fermi-Hubbard, t = 1, U = 4, two electrons.
// Neel determinant |1001>, then exp(-i (pi/4) X0 X1 X2 X3) and s on
// q[0], which prepares the covalent singlet, then rbs(pi/8) on the
// spin-down modes q[1], q[3]. Read with --qasm-dialect lenient.
// The rbs angle is the closed form for this t and U, not a fit.
qreg q[4];
x q[0];
x q[3];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
rz(1.5707963267948966) q[3];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[1];
h q[2];
h q[3];
s q[0];
rbs(0.39269908169872414) q[1],q[3];
