// Continuous one-qubit rotations: rx/ry/rz plus the qelib1 u-family.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
ry(pi/3) q[0];
rz(2*pi/5) q[0];
rx(0.7853981633974483) q[0];
u2(0.3,-1.1) q[1];
u1(pi/4) q[1];
h q[1];
u3(1.0471975511965976,0.5235987755982988,-0.7853981633974483) q[2];
p(0.9) q[2];
h q[2];
measure q -> c;
