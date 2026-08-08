// 3-qubit QFT lowered to cx + rz (no controlled rotations left),
// applied to a non-uniform input so the output distribution is
// genuinely spread — QFT of the uniform state is |000> and would
// make the L2 comparison vacuous.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
x q[0];
h q[1];
h q[2];
cx q[2],q[1];
rz(-0.7853981633974483) q[1];
cx q[2],q[1];
rz(0.7853981633974483) q[1];
h q[1];
cx q[1],q[0];
rz(-1.5707963267948966) q[0];
cx q[1],q[0];
rz(1.5707963267948966) q[0];
h q[0];
measure q -> c;
