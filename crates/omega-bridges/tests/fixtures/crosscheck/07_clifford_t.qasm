// Non-Clifford T ladder — the reason tsim/ppvm exist rather than Stim.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
t q[0];
h q[1];
tdg q[1];
cx q[0],q[1];
t q[1];
h q[1];
h q[2];
t q[2];
cz q[0],q[2];
h q[2];
cy q[1],q[2];
measure q -> c;
