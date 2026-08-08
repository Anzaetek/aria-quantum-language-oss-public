// SWAP + CCX: inside tsim's gate set, OUTSIDE ppvm's (its executor
// rejects SWAP and has no CCX sugar). Exercises the per-backend
// fixture filter.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
h q[1];
ccx q[0],q[1],q[2];
swap q[0],q[2];
measure q -> c;
