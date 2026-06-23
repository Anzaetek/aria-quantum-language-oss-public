//! QAOA (Quantum Approximate Optimization Algorithm) circuit generator.
//!
//! Generates parameterised CircuitIR from an Ising model:
//!   |ψ(γ,β)⟩ = Π_{k=1..p} [U_M(β_k) · U_C(γ_k)] · |+⟩^n
//!
//! where:
//!   U_C(γ) = exp(-iγH_C) — cost unitary (ZZ + Z rotations)
//!   U_M(β) = exp(-iβH_M) — mixer unitary (X rotations)

use smallvec::smallvec;
use std::collections::HashMap;

use crate::circuit::*;
use crate::qubo::IsingModel;

/// Generate a QAOA ansatz circuit for the given Ising model.
///
/// `p` is the number of QAOA rounds (depth).
/// Returns a parameterised circuit with 2p free parameters: γ_1,..,γ_p, β_1,..,β_p.
pub fn qaoa_circuit(ising: &IsingModel, p: usize) -> CircuitIR {
    let n = ising.n;
    let mut ops = Vec::new();
    let mut symbols = HashMap::new();
    let mut next_sym = 0u32;

    let mut make_sym = |name: &str| -> (SymbolId, ParamExpr) {
        let id = next_sym;
        next_sym += 1;
        symbols.insert(id, name.to_string());
        (id, ParamExpr::Symbol(id))
    };

    // Initial state: |+⟩^n = H^n |0⟩^n
    for q in 0..n {
        ops.push(GateOp {
            gate: GateKind::H,
            qubits: smallvec![Qubit(q as u32)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }

    // p rounds of cost + mixer
    for k in 0..p {
        let (_, gamma_expr) = make_sym(&format!("gamma_{k}"));
        let (_, beta_expr) = make_sym(&format!("beta_{k}"));

        // Cost unitary U_C(γ): exp(-iγ H_C)
        // For ZZ terms: exp(-iγ J_{ij} Z_i Z_j) = CX(i,j) · Rz(2γJ_{ij}, j) · CX(i,j)
        for (&(i, j), &jval) in &ising.couplings {
            if jval.abs() < 1e-15 {
                continue;
            }
            // Rz angle = 2 * gamma * J_ij
            let angle = ParamExpr::Mul(
                Box::new(ParamExpr::Concrete(2.0 * jval)),
                Box::new(gamma_expr.clone()),
            );

            ops.push(GateOp {
                gate: GateKind::CX,
                qubits: smallvec![Qubit(i as u32), Qubit(j as u32)],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
            ops.push(GateOp {
                gate: GateKind::Rz,
                qubits: smallvec![Qubit(j as u32)],
                params: smallvec![angle],
                classical_bit: None,
                condition: None,
            });
            ops.push(GateOp {
                gate: GateKind::CX,
                qubits: smallvec![Qubit(i as u32), Qubit(j as u32)],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }

        // For Z terms: exp(-iγ h_i Z_i) = Rz(2γ h_i, i)
        for (i, &h) in ising.fields.iter().enumerate() {
            if h.abs() < 1e-15 {
                continue;
            }
            let angle = ParamExpr::Mul(
                Box::new(ParamExpr::Concrete(2.0 * h)),
                Box::new(gamma_expr.clone()),
            );

            ops.push(GateOp {
                gate: GateKind::Rz,
                qubits: smallvec![Qubit(i as u32)],
                params: smallvec![angle],
                classical_bit: None,
                condition: None,
            });
        }

        // Mixer unitary U_M(β): exp(-iβ Σ X_i) = Π_i Rx(2β, i)
        for q in 0..n {
            let angle = ParamExpr::Mul(
                Box::new(ParamExpr::Concrete(2.0)),
                Box::new(beta_expr.clone()),
            );

            ops.push(GateOp {
                gate: GateKind::Rx,
                qubits: smallvec![Qubit(q as u32)],
                params: smallvec![angle],
                classical_bit: None,
                condition: None,
            });
        }
    }

    CircuitIR {
        num_qubits: n as u32,
        num_classical_bits: 0,
        ops,
        circuit_type: CircuitType::GateBased,
        symbols,
        custom_gates: HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ParameterBinding;
    use crate::qubo::Qubo;

    #[test]
    fn test_qaoa_circuit_structure() {
        let mut q = Qubo::new(2);
        q.set(0, 0, -1.0);
        q.set(0, 1, 2.0);
        let ising = q.to_ising();

        let circuit = qaoa_circuit(&ising, 1);

        assert_eq!(circuit.num_qubits, 2);
        // Should have: 2 H's + (3 ops for ZZ) + (Rz for Z field) + 2 Rx's = ~8 ops
        assert!(circuit.ops.len() >= 5);
        // Should have 2 symbols: gamma_0, beta_0
        assert_eq!(circuit.symbols.len(), 2);

        // First ops should be H gates
        assert_eq!(circuit.ops[0].gate, GateKind::H);
        assert_eq!(circuit.ops[1].gate, GateKind::H);
    }

    #[test]
    fn test_qaoa_circuit_executes() {
        let mut q = Qubo::new(2);
        q.set(0, 0, -1.0);
        q.set(1, 1, -1.0);
        q.set(0, 1, 2.0);
        let ising = q.to_ising();

        let circuit = qaoa_circuit(&ising, 1);

        // Bind parameters
        let mut params = ParameterBinding::new();
        for &id in circuit.symbols.keys() {
            params.bind(id, 0.5);
        }

        // Verify all parameters resolve
        for op in &circuit.ops {
            for p in &op.params {
                assert!(params.resolve(p).is_ok());
            }
        }
    }

    #[test]
    fn test_qaoa_multiple_rounds() {
        let mut q = Qubo::new(3);
        q.set(0, 1, 1.0);
        q.set(1, 2, 1.0);
        let ising = q.to_ising();

        let circuit = qaoa_circuit(&ising, 3);
        // 3 rounds → 6 symbols (gamma_0..2, beta_0..2)
        assert_eq!(circuit.symbols.len(), 6);
    }
}
