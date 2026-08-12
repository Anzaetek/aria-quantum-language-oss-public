//! Emit a [`Circuit`] as `.aria` source.
//!
//! The inverse of the QASM importer + `parse_aria`: it renders a concrete
//! circuit as an Aria `circuit` block so an imported QASM program can become a
//! readable `.aria` source. Gate names match `gate_from_name` in the parser;
//! `parse_aria(&to_aria_source(c, name))` reproduces an equivalent circuit for
//! the gate set QASM import can produce.

use super::nodes::*;

/// The `.aria` gate token for a kind (matches the parser's `gate_from_name`).
fn gate_to_aria(kind: GateKind) -> Option<&'static str> {
    Some(match kind {
        GateKind::I => "I",
        GateKind::X => "X",
        GateKind::Y => "Y",
        GateKind::Z => "Z",
        GateKind::H => "H",
        GateKind::S => "S",
        GateKind::Sdg => "SDG",
        GateKind::T => "T",
        GateKind::Tdg => "TDG",
        GateKind::SX => "SX",
        GateKind::RX => "RX",
        GateKind::RY => "RY",
        GateKind::RZ => "RZ",
        GateKind::P => "P",
        GateKind::U => "U",
        GateKind::CX => "CX",
        GateKind::CY => "CY",
        GateKind::CZ => "CZ",
        GateKind::SWAP => "SWAP",
        GateKind::CP => "CP",
        GateKind::CRz => "CRZ",
        GateKind::CCX => "CCX",
        GateKind::CSWAP => "CSWAP",
        GateKind::RXX => "RXX",
        GateKind::RYY => "RYY",
        GateKind::RZZ => "RZZ",
        GateKind::RBS => "RBS",
        _ => return None,
    })
}

/// Render a [`Circuit`] as `.aria` source with the given circuit name.
///
/// Statements that have no `.aria` surface syntax — `barrier`, `reset` — are
/// emitted as `--` comments rather than silently dropped or rendered as invalid
/// source. An unsupported gate becomes a `-- unsupported gate:` comment for the
/// same reason.
pub fn to_aria_source(circuit: &Circuit, name: &str) -> String {
    let mut lines = vec![format!("circuit {name} {{")];

    for reg in &circuit.registers {
        let kw = match reg.kind {
            RegisterKind::Quantum => "qreg",
            RegisterKind::Classical => "creg",
        };
        lines.push(format!("    {kw} {}[{}]", reg.name, reg.size));
    }
    if !circuit.registers.is_empty() {
        lines.push(String::new());
    }

    for inst in &circuit.instructions {
        // Where this instruction's output starts, so a classical guard can be
        // wrapped around everything it emitted.
        //
        // The Aria emitter had the SAME defect as `to_qasm` did: zero
        // references to `condition`, so `when m[0] == 1 { apply X on q[0] }`
        // round-tripped as a bare `apply X on q[0]` and the guard vanished.
        // Aria -> Aria is the one export where a reader is most likely to
        // assume fidelity, which makes the silence worse rather than better.
        let emitted_from = lines.len();

        match inst.gate.kind {
            GateKind::Measure => {
                lines.push(format!(
                    "    measure {} -> {}",
                    inst.qubits[0], inst.clbits[0]
                ));
            }
            GateKind::Barrier => {
                let qrefs: Vec<String> = inst.qubits.iter().map(|q| q.to_string()).collect();
                lines.push(format!("    -- barrier {}", qrefs.join(", ")));
            }
            GateKind::Reset => {
                lines.push(format!("    -- reset {}", inst.qubits[0]));
            }
            kind => {
                let Some(gate_name) = gate_to_aria(kind) else {
                    lines.push(format!("    -- unsupported gate: {kind:?}"));
                    continue;
                };
                let params = if inst.gate.params.is_empty() {
                    String::new()
                } else {
                    let ps: Vec<String> = inst
                        .gate
                        .params
                        .iter()
                        .map(|p| format!("{}", p.try_as_f64().unwrap_or(0.0)))
                        .collect();
                    format!("({})", ps.join(", "))
                };
                let targets: Vec<String> = inst.qubits.iter().map(|q| q.to_string()).collect();
                lines.push(format!(
                    "    apply {gate_name}{params} on {}",
                    targets.join(", ")
                ));
            }
        }

        // Wrap whatever this instruction produced in the `when reg[i] == v { … }`
        // form the parser lowers back to the same per-instruction condition.
        //
        // Unlike the QASM 2.0 path there is nothing to refuse: Aria's guard
        // addresses a single classical bit directly, which is exactly what the
        // condition holds. Lines already emitted as `--` comments (barrier,
        // reset, unsupported) are wrapped too — the comment is what round-trips,
        // and dropping the guard from it would lose information the reader can
        // see is there.
        if let Some((clbit, value)) = &inst.condition {
            for line in lines.iter_mut().skip(emitted_from) {
                let body = line.trim_start();
                *line = format!(
                    "    when {}[{}] == {} {{ {} }}",
                    clbit.register, clbit.index, value, body
                );
            }
        }
    }

    lines.push("}".to_string());
    lines.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{from_qasm, parse_aria, CircuitBuilder};
    use std::f64::consts::PI;

    #[test]
    fn emits_and_reparses_a_builder_circuit() {
        let circ = CircuitBuilder::new("Demo", 2, 2)
            .h(0)
            .ry(1, PI / 4.0)
            .cx(0, 1)
            .build();
        let src = to_aria_source(&circ, "Demo");
        assert!(src.contains("circuit Demo {"));
        assert!(src.contains("qreg q[2]"));
        assert!(src.contains("apply H on q[0]"));
        assert!(src.contains("apply CX on q[0], q[1]"));
        assert!(src.contains("apply RY("));

        // Round-trip: the emitted source parses and instantiates.
        let prog = parse_aria(&src).expect("emitted .aria must parse");
        let reparsed = prog.instantiate("Demo", &[]).expect("must instantiate");
        // Gate count is preserved (H, RY, CX = 3).
        assert_eq!(reparsed.gate_count(), circ.gate_count());
    }

    #[test]
    fn round_trips_qasm_import_to_aria() {
        let qasm = "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\n";
        let circ = from_qasm(qasm).unwrap();
        let src = to_aria_source(&circ, "Imported");
        let prog = parse_aria(&src).expect("imported-then-emitted .aria must parse");
        let reparsed = prog.instantiate("Imported", &[]).unwrap();
        assert_eq!(reparsed.n_qubits(), 2);
        assert_eq!(reparsed.gate_count(), 2);
    }
}
