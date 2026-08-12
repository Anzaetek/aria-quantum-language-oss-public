// Two-qubit hardware-efficient VQE ansatz for the H2 Hamiltonian.
//
// Referenced by crates/omega-wasm-cli/src/main.rs, which requires exactly
// 2 qubits and exactly 2 free parameters (the vqe.wasm guest hard-codes
// NUM_PARAMS=2).
//
// RECONSTRUCTED 2026-08-12. This file, and the other three under
// examples/circuits/ that omega-wasm-cli's tests read, had NEVER been tracked
// in git — `git log` on the path was empty and they were not gitignored — so
// `cargo test --workspace` was red on every clean checkout and had been for
// some time (FIXES_PLAN.md K9). CI could not see it because ci.sh tests a
// hand-typed crate list that excludes omega-wasm-cli.
//
// The reconstruction is only legitimate because the tests still DISCRIMINATE
// with it, which was checked rather than assumed. Against
//   H = 0.3979 Z0 - 0.3979 Z1 - 0.0112 Z0Z1 + 0.1809 X0X1
// (the identity term is dropped in h2_hamiltonian; it shifts every energy
// equally and cannot change an argmin):
//
//   exact ground energy         = -0.804902
//   E at the test's start point = +0.014975   <- ABOVE the -0.18 threshold
//   min over this ansatz        = -0.804899   <- essentially exact
//
// So the optimizer must genuinely travel from +0.015 to below -0.18: a broken
// optimizer that returns its input, or one that truncates the parameter
// vector, fails. Had the start already been below threshold the tests would
// have passed without the optimizer doing anything, which is the failure mode
// this whole repository keeps cataloguing.

OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];

ry(theta0) q[0];
ry(theta1) q[1];
cx q[0],q[1];
