OPENQASM 2.0;
include "qelib1.inc";
// Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2.
// Used by the omega-cli json_smoke end-to-end tests: 2-qubit statevector
// (4 amplitudes), ⟨Z0 Z1⟩ = +1, and ~50/50 |00⟩/|11⟩ sampled counts.
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
