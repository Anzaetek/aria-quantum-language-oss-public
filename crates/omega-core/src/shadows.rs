//! Classical shadows protocol for efficient observable estimation.
//!
//! Huang, Kueng, Preskill (2020): "Predicting Many Properties of a Quantum State
//! from Very Few Measurements."
//!
//! Protocol:
//! 1. Apply a random Clifford unitary U_i (single-qubit random Paulis for efficiency)
//! 2. Measure in computational basis → bitstring b_i
//! 3. Store shadow snapshot ρ̂_i = U_i† |b_i⟩⟨b_i| U_i (classically reconstructed)
//! 4. Estimate ⟨O⟩ = median-of-means of Tr(O · ρ̂_i)
//!
//! For random single-qubit Cliffords: the inverse channel is M^{-1}(ρ) = 3ρ - I.
//! So each qubit's shadow is: (3|b_q⟩⟨b_q| - I) in the Pauli basis.

use rand::RngExt;
use rand::SeedableRng;

use crate::circuit::*;
use crate::executor::PauliOp;
use smallvec::smallvec;
use std::collections::HashMap;

/// A single shadow snapshot: random Pauli basis per qubit + measurement outcome.
#[derive(Clone, Debug)]
pub struct ShadowSnapshot {
    /// Random Pauli basis for each qubit: 0=X, 1=Y, 2=Z.
    pub bases: Vec<u8>,
    /// Measurement outcome per qubit: 0 or 1.
    pub outcomes: Vec<u8>,
}

/// Collection of classical shadow snapshots.
pub struct ClassicalShadow {
    pub n: usize,
    pub snapshots: Vec<ShadowSnapshot>,
}

/// Generate random measurement circuits for classical shadows.
///
/// Returns `num_snapshots` circuits, each with random single-qubit Pauli rotations
/// before measurement in the computational basis.
pub fn shadow_circuits(n: usize, num_snapshots: usize, seed: u64) -> Vec<(CircuitIR, Vec<u8>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut circuits = Vec::with_capacity(num_snapshots);

    for _ in 0..num_snapshots {
        let mut ops = Vec::new();
        let mut bases = Vec::with_capacity(n);

        for q in 0..n {
            let basis = rng.random_range(0u8..3);
            bases.push(basis);

            // Rotate from chosen basis to Z basis (computational basis measurement)
            match basis {
                0 => {
                    // X basis: Hy = H to rotate X eigenstates to Z eigenstates
                    ops.push(GateOp {
                        gate: GateKind::H,
                        qubits: smallvec![Qubit(q as u32)],
                        params: smallvec![],
                        classical_bit: None,
                        condition: None,
                    });
                }
                1 => {
                    // Y basis: S†H to rotate Y eigenstates to Z eigenstates
                    ops.push(GateOp {
                        gate: GateKind::Sdg,
                        qubits: smallvec![Qubit(q as u32)],
                        params: smallvec![],
                        classical_bit: None,
                        condition: None,
                    });
                    ops.push(GateOp {
                        gate: GateKind::H,
                        qubits: smallvec![Qubit(q as u32)],
                        params: smallvec![],
                        classical_bit: None,
                        condition: None,
                    });
                }
                2 => {
                    // Z basis: no rotation needed
                }
                _ => unreachable!(),
            }
        }

        let circuit = CircuitIR {
            num_qubits: n as u32,
            num_classical_bits: 0,
            ops,
            circuit_type: CircuitType::GateBased,
            symbols: HashMap::new(),
            custom_gates: HashMap::new(),
        };

        circuits.push((circuit, bases));
    }

    circuits
}

impl ClassicalShadow {
    /// Construct a classical shadow from measurement results.
    pub fn new(n: usize, snapshots: Vec<ShadowSnapshot>) -> Self {
        Self { n, snapshots }
    }

    /// Estimate the expectation value of a Pauli observable using classical shadows.
    ///
    /// For a k-local Pauli operator P = P_1 ⊗ P_2 ⊗ ... ⊗ P_k:
    /// Each snapshot contributes iff the random basis matches the Pauli on all non-identity qubits.
    /// When it matches: the estimator is Π_q (3 * (-1)^{b_q} delta_{basis_q, P_q}).
    pub fn estimate_pauli(&self, pauli_terms: &[(u32, PauliOp)]) -> f64 {
        if pauli_terms.is_empty() {
            return 1.0; // Identity operator
        }

        let mut total = 0.0;
        let num_snapshots = self.snapshots.len();

        if num_snapshots == 0 {
            return 0.0;
        }

        for snap in &self.snapshots {
            // Check if this snapshot's random bases match the Pauli operator
            let mut matches = true;
            let mut value = 1.0;

            for &(q, ref op) in pauli_terms {
                let q = q as usize;
                let required_basis = match op {
                    PauliOp::I => continue,
                    PauliOp::X => 0,
                    PauliOp::Y => 1,
                    PauliOp::Z => 2,
                };

                if snap.bases[q] != required_basis {
                    matches = false;
                    break;
                }

                // Factor of 3 per matched qubit (from inverse channel M^{-1} = 3ρ - I)
                // Sign from measurement outcome: (-1)^b_q
                let sign = if snap.outcomes[q] == 0 { 1.0 } else { -1.0 };
                value *= 3.0 * sign;
            }

            // Non-matching snapshots contribute 0
            if matches {
                total += value;
            }
        }

        total / num_snapshots as f64
    }

    /// Estimate the expectation value of a general observable (sum of Pauli strings).
    pub fn estimate_observable(&self, terms: &[(f64, Vec<(u32, PauliOp)>)]) -> f64 {
        let mut total = 0.0;
        for (coeff, pauli_string) in terms {
            total += coeff * self.estimate_pauli(pauli_string);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_circuits_generation() {
        let circuits = shadow_circuits(3, 10, 42);
        assert_eq!(circuits.len(), 10);

        for (circuit, bases) in &circuits {
            assert_eq!(circuit.num_qubits, 3);
            assert_eq!(bases.len(), 3);
            assert!(bases.iter().all(|&b| b <= 2));
        }
    }

    #[test]
    fn test_shadow_estimate_z_on_zero_state() {
        // |0⟩ state: ⟨Z⟩ = 1
        // Simulate by creating snapshots where Z-basis measurements always give 0
        let n = 1;
        let snapshots: Vec<ShadowSnapshot> = (0..1000)
            .map(|i| {
                let basis = (i % 3) as u8; // cycle through X, Y, Z
                let outcome = if basis == 2 { 0 } else { (i % 2) as u8 }; // Z always gives 0
                ShadowSnapshot {
                    bases: vec![basis],
                    outcomes: vec![outcome],
                }
            })
            .collect();

        let shadow = ClassicalShadow::new(n, snapshots);
        let val = shadow.estimate_pauli(&[(0, PauliOp::Z)]);

        // Should be approximately 1.0 (Z basis snapshots dominate)
        // With 1/3 of snapshots matching (Z basis), and all giving outcome 0 (sign +1):
        // estimate = 3 * 1 = 3, averaged over matching snapshots... wait.
        // Actually the factor of 3 accounts for the 1/3 probability of choosing Z basis.
        // So for |0⟩ with Z measurements: each matching snapshot gives 3*1 = 3,
        // and 1/3 of snapshots match, so average = 3 * (1/3-fraction-times-count) / count = 1. ✓
        assert!(
            (val - 1.0).abs() < 0.1,
            "expected ~1.0 for ⟨Z⟩ on |0⟩, got {val}"
        );
    }

    #[test]
    fn test_shadow_identity_observable() {
        let shadow = ClassicalShadow::new(1, vec![]);
        assert_eq!(shadow.estimate_pauli(&[]), 1.0);
    }
}
