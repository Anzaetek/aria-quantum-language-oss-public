// Controls held at |0>, superposition around them: the untriggered chain
// with non-trivial spectator amplitudes (H4).
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[2];
sx q[3];
cx q[2],q[3];
ccx q[0],q[1],q[2];
h q[2];
h q[3];
