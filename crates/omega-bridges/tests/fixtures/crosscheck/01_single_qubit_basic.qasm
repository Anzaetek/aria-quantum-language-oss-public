// Clifford one-qubit surface: H/X/Y/Z/S/SDG on two independent qubits.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
s q[0];
h q[0];
x q[1];
h q[1];
sdg q[1];
h q[1];
y q[0];
z q[1];
measure q -> c;
