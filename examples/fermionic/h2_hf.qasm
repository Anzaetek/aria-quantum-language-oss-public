OPENQASM 2.0;
include "qelib1.inc";
// Hartree-Fock for H2/STO-3G: both electrons in 1 sigma_g.
// |1⟩ on modes 0 and 1, |0⟩ on modes 2 and 3.
qreg q[4];
x q[0];
x q[1];
