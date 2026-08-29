// UNTRIGGERED CCX on a basis state (one control set): the identity whose
// 15-gate decomposition still churns every untriggered amplitude (H4).
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
x q[0];
ccx q[0],q[1],q[2];
