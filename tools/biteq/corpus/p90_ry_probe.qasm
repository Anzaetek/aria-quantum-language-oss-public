// LIBM PROBE — excluded from the headline verdict (plan H3). ry matrices
// are host cos/sin; if the two machines' CPU f64 dumps differ HERE and
// nowhere else, the difference is libm, quantified and attributed.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
h q[3];
ry(0.13) q[0];
ry(0.7) q[2];
ccx q[0],q[1],q[2];
h q[0];
h q[2];
