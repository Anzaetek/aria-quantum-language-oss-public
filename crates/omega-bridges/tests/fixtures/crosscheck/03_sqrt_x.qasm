// SX / SXDG (Stim SQRT_X / SQRT_X_DAG).
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
sx q[0];
sx q[1];
sxdg q[1];
cx q[0],q[1];
measure q -> c;
