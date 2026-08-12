// Four-qubit, FOUR-PARAMETER VQE ansatz.
//
// Referenced by `native_path_handles_four_param_vqe_ansatz`, whose whole point
// is that the native optimizer path is NOT capped at the vqe.wasm guest's
// NUM_PARAMS=2 — so this file must declare exactly 4 distinct free symbols and
// the optimizer must receive all four rather than a silently truncated vector.
//
// Reconstructed 2026-08-12; see examples/circuits/vqe_ansatz_2q.qasm.
//
// The test's observable is 0.5*Z0 + 0.3*Z1Z2 + 0.2*X2X3, so the ansatz has to
// (a) move Z0, (b) correlate q1 with q2, and (c) put weight on X2X3 — hence a
// ry on each qubit plus an entangling chain. An ansatz touching only q0 would
// leave two of the three terms at their initial values and the test would
// measure almost nothing.

OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];

ry(theta0) q[0];
ry(theta1) q[1];
ry(theta2) q[2];
ry(theta3) q[3];

cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];

ry(theta0) q[1];
ry(theta2) q[3];
