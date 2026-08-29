// The other multi-control gate, same superposition shape as c03.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
h q[3];
sdg q[2];
sx q[1];
cx q[3],q[0];
cswap q[0],q[1],q[2];
h q[0];
h q[1];
h q[2];
h q[3];
