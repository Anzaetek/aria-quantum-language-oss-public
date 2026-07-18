use super::codes::QECCode;
use aria_core::ast::nodes::*;
use aria_core::ast::{Circuit, CircuitBuilder};

/// Transform a logical circuit into an error-corrected physical circuit.
///
/// Wraps with: encoding -> transversal operations -> syndrome measurement.
pub fn add_error_correction(logical_circuit: &Circuit, code: &dyn QECCode) -> Circuit {
    let n_logical = logical_circuit.n_qubits();
    let n_physical_per = code.n_physical();
    let n_physical_total = n_logical * n_physical_per;
    let n_syndrome_ancilla = (n_physical_per - 1) * n_logical;
    let n_total = n_physical_total + n_syndrome_ancilla;

    let mut b = CircuitBuilder::new(
        &format!("{}_ecc", logical_circuit.name),
        n_total,
        logical_circuit.n_clbits() + n_syndrome_ancilla,
    );

    // Phase 1: Encoding
    let enc = code.encoding_circuit();
    for logical_q in 0..n_logical {
        let offset = logical_q * n_physical_per;
        for inst in &enc.instructions {
            let qidxs: Vec<usize> = inst.qubits.iter().map(|q| offset + q.index).collect();
            apply_gate_to_builder(&mut b, &inst.gate, &qidxs);
        }
    }

    b.barrier_all();

    // Phase 2: Transversal logical operations
    let mut all_logical_qubits = Vec::new();
    for reg in &logical_circuit.registers {
        if reg.kind == RegisterKind::Quantum {
            all_logical_qubits.extend(reg.qubits());
        }
    }
    let logical_qubit_index: std::collections::HashMap<_, _> = all_logical_qubits
        .iter()
        .enumerate()
        .map(|(i, q)| (q.clone(), i))
        .collect();

    for inst in &logical_circuit.instructions {
        if inst.gate.kind.is_meta() {
            continue;
        }

        if inst.gate.n_qubits() == 1 {
            if let Some(&idx) = logical_qubit_index.get(&inst.qubits[0]) {
                for phys in 0..n_physical_per {
                    let target = idx * n_physical_per + phys;
                    apply_gate_to_builder(&mut b, &inst.gate, &[target]);
                }
            }
        } else if inst.gate.n_qubits() == 2 {
            if let (Some(&idx0), Some(&idx1)) = (
                logical_qubit_index.get(&inst.qubits[0]),
                logical_qubit_index.get(&inst.qubits[1]),
            ) {
                for phys in 0..n_physical_per {
                    let t0 = idx0 * n_physical_per + phys;
                    let t1 = idx1 * n_physical_per + phys;
                    apply_gate_to_builder(&mut b, &inst.gate, &[t0, t1]);
                }
            }
        }
    }

    b.barrier_all();

    // Phase 3: Syndrome measurement
    let syn = code.syndrome_circuit();
    for logical_q in 0..n_logical {
        let data_offset = logical_q * n_physical_per;
        let anc_offset = n_physical_total + logical_q * (n_physical_per - 1);
        let clbit_offset = logical_circuit.n_clbits() + logical_q * (n_physical_per - 1);

        // Remap a single-code syndrome-circuit qubit index into this patch's
        // physical layout: data qubits (index < n_physical_per) sit at the
        // patch's data offset, ancilla (index >= n_physical_per) after all data.
        let remap = |qi: usize| {
            if qi < n_physical_per {
                data_offset + qi
            } else {
                anc_offset + (qi - n_physical_per)
            }
        };
        for inst in &syn.instructions {
            if inst.gate.kind == GateKind::Measure {
                let anc_idx = inst.qubits[0].index;
                if anc_idx >= n_physical_per {
                    let phys_anc = anc_offset + (anc_idx - n_physical_per);
                    b.measure(phys_anc, clbit_offset + (anc_idx - n_physical_per));
                }
            } else {
                // Every other gate (crucially the basis-change Hadamards that
                // wrap X-type stabilizer measurements — dropping them would read
                // the X-check syndrome in the wrong basis) is re-emitted on the
                // remapped physical qubits.
                let qidxs: Vec<usize> = inst.qubits.iter().map(|q| remap(q.index)).collect();
                apply_gate_to_builder(&mut b, &inst.gate, &qidxs);
            }
        }
    }

    let mut result = b.build();
    result.metadata.insert(
        "ecc_code".to_string(),
        std::any::type_name_of_val(code).to_string(),
    );
    result
}

fn apply_gate_to_builder(b: &mut CircuitBuilder, gate: &GateDef, qubits: &[usize]) {
    match gate.kind {
        GateKind::H => {
            b.h(qubits[0]);
        }
        GateKind::X => {
            b.x(qubits[0]);
        }
        GateKind::Y => {
            b.y(qubits[0]);
        }
        GateKind::Z => {
            b.z(qubits[0]);
        }
        GateKind::S => {
            b.s(qubits[0]);
        }
        GateKind::T => {
            b.t(qubits[0]);
        }
        GateKind::RX => {
            b.rx(qubits[0], gate.params[0].try_as_f64().unwrap());
        }
        GateKind::RY => {
            b.ry(qubits[0], gate.params[0].try_as_f64().unwrap());
        }
        GateKind::RZ => {
            b.rz(qubits[0], gate.params[0].try_as_f64().unwrap());
        }
        GateKind::CX => {
            b.cx(qubits[0], qubits[1]);
        }
        GateKind::CZ => {
            b.cz(qubits[0], qubits[1]);
        }
        GateKind::SWAP => {
            b.swap(qubits[0], qubits[1]);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecc::codes::SteaneCode;

    #[test]
    fn syndrome_extraction_keeps_ancilla_hadamards() {
        // Regression: Phase-3 syndrome extraction used to re-emit only Measure/CX,
        // silently dropping the Hadamards that wrap Steane's X-type stabilizer
        // measurements — which would read the X-check syndrome in the wrong basis.
        // A bare 1-logical-qubit circuit must now carry every syndrome Hadamard:
        // 3 from the Steane encoder (h on qubits 1,2,3) + 6 from the three X-stabs
        // (an h before and after each ancilla) = 9.
        let logical = CircuitBuilder::new("id", 1, 0).build();
        let ecc = add_error_correction(&logical, &SteaneCode);
        let h_count = ecc
            .instructions
            .iter()
            .filter(|i| i.gate.kind == GateKind::H)
            .count();
        assert_eq!(
            h_count, 9,
            "encoder (3) + X-stabilizer basis-change Hadamards (6)"
        );
    }
}
