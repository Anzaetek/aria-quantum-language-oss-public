// No creg, no measure — the runner must synthesise a full-width
// measurement exactly as the Qiskit runner's measure_all() does.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
ry(0.9) q[0];
cx q[0],q[1];
t q[1];
h q[2];
cx q[1],q[2];
