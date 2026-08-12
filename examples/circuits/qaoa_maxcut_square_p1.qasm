// Depth-1 QAOA for MaxCut on a 4-cycle (C4), 4 qubits, 2 free parameters.
//
// Same structure and the same zero-parameter identity property as
// qaoa_maxcut_triangle_p1.qasm; reconstructed 2026-08-12 for the same reason.
// Edges of C4: (0,1), (1,2), (2,3), (0,3) — note this graph is bipartite, so
// its MaxCut is the full 4 edges, unlike the triangle where it is 2.

OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];

h q[0];
h q[1];
h q[2];
h q[3];

// cost layer, gamma
cx q[0],q[1];
rz(2*gamma) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(2*gamma) q[2];
cx q[1],q[2];
cx q[2],q[3];
rz(2*gamma) q[3];
cx q[2],q[3];
cx q[0],q[3];
rz(2*gamma) q[3];
cx q[0],q[3];

// mixer layer, beta
rx(2*beta) q[0];
rx(2*beta) q[1];
rx(2*beta) q[2];
rx(2*beta) q[3];
