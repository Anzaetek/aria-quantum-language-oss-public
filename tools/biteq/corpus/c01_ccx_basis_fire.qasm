// Triggered CCX on a basis state: the permutation itself.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
x q[0];
x q[1];
ccx q[0],q[1],q[2];
