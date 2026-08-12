// Depth-1 QAOA for MaxCut on a triangle (K3), 3 qubits, 2 free parameters.
//
// Referenced by crates/omega-wasm-cli/src/main.rs. Reconstructed 2026-08-12 —
// see examples/circuits/vqe_ansatz_2q.qasm for why these files were missing.
//
// Structure is forced by `sample_circuit_at_zero_params_recovers_uniform`,
// which asserts that at gamma = beta = 0 the circuit is exactly H^3|000>, so
// Z-basis sampling is uniform over all 8 bitstrings. That pins the layout:
//
//   * the initial layer must be H on every qubit,
//   * every parameterised gate must reduce to the IDENTITY at parameter 0.
//
// `rz(0)` and `rx(0)` are both the identity, so the cost and mixer layers
// vanish together and the state is |+++>. A fixture with, say, a fixed
// non-identity gate after the Hadamards would break that test — which is what
// makes it a real check on this file rather than a formality.
//
// Cost layer: exp(-i*gamma*Z_i*Z_j) per edge, written as CX-RZ(2*gamma)-CX
// because the parser has no native rzz. Mixer: RX(2*beta) per qubit.
// Edges of K3: (0,1), (1,2), (0,2).

OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];

h q[0];
h q[1];
h q[2];

// cost layer, gamma
cx q[0],q[1];
rz(2*gamma) q[1];
cx q[0],q[1];
cx q[1],q[2];
rz(2*gamma) q[2];
cx q[1],q[2];
cx q[0],q[2];
rz(2*gamma) q[2];
cx q[0],q[2];

// mixer layer, beta
rx(2*beta) q[0];
rx(2*beta) q[1];
rx(2*beta) q[2];
