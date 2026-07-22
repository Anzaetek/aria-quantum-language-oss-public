use super::nodes::*;
use regex::Regex;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Export: Circuit -> QASM string
// ---------------------------------------------------------------------------

fn gate_to_qasm(kind: GateKind) -> Option<&'static str> {
    match kind {
        GateKind::I => Some("id"),
        GateKind::X => Some("x"),
        GateKind::Y => Some("y"),
        GateKind::Z => Some("z"),
        GateKind::H => Some("h"),
        GateKind::S => Some("s"),
        GateKind::Sdg => Some("sdg"),
        GateKind::T => Some("t"),
        GateKind::Tdg => Some("tdg"),
        GateKind::SX => Some("sx"),
        GateKind::RX => Some("rx"),
        GateKind::RY => Some("ry"),
        GateKind::RZ => Some("rz"),
        GateKind::P => Some("p"),
        GateKind::U => Some("u3"),
        GateKind::CX => Some("cx"),
        GateKind::CY => Some("cy"),
        GateKind::CZ => Some("cz"),
        GateKind::SWAP => Some("swap"),
        GateKind::CP => Some("cp"),
        GateKind::CCX => Some("ccx"),
        GateKind::CSWAP => Some("cswap"),
        GateKind::RXX => Some("rxx"),
        GateKind::RYY => Some("ryy"),
        GateKind::RZZ => Some("rzz"),
        _ => None,
    }
}

fn format_param(p: f64) -> String {
    if p == 0.0 {
        return "0".to_string();
    }
    let ratio = p / PI;
    if (ratio - ratio.round()).abs() < 1e-10 {
        let n = ratio.round() as i64;
        return match n {
            1 => "pi".to_string(),
            -1 => "-pi".to_string(),
            _ => format!("{n}*pi"),
        };
    }
    if (ratio * 2.0 - (ratio * 2.0).round()).abs() < 1e-10 {
        let n = (ratio * 2.0).round() as i64;
        return format!("{n}*pi/2");
    }
    if (ratio * 4.0 - (ratio * 4.0).round()).abs() < 1e-10 {
        let n = (ratio * 4.0).round() as i64;
        return format!("{n}*pi/4");
    }
    format!("{p:.10}")
}

/// Convert a Circuit AST to an OpenQASM 2.0 string.
pub fn to_qasm(circuit: &Circuit) -> String {
    let mut lines = vec![
        "OPENQASM 2.0;".to_string(),
        "include \"qelib1.inc\";".to_string(),
        String::new(),
    ];

    for reg in &circuit.registers {
        let kw = match reg.kind {
            RegisterKind::Quantum => "qreg",
            RegisterKind::Classical => "creg",
        };
        lines.push(format!("{kw} {}[{}];", reg.name, reg.size));
    }
    lines.push(String::new());

    for inst in &circuit.instructions {
        match inst.gate.kind {
            GateKind::Barrier => {
                let qrefs: Vec<String> = inst.qubits.iter().map(|q| q.to_string()).collect();
                lines.push(format!("barrier {};", qrefs.join(", ")));
            }
            GateKind::Measure => {
                lines.push(format!("measure {} -> {};", inst.qubits[0], inst.clbits[0]));
            }
            GateKind::Reset => {
                lines.push(format!("reset {};", inst.qubits[0]));
            }
            GateKind::RBS => {
                // No RBS primitive in QASM 2.0 / qelib1 — emit the exact
                // decomposition RBS(θ) = (H⊗H)·CZ·(Ry(−θ)⊗Ry(θ))·CZ·(H⊗H)
                // (full angle θ, verified against exp(−iθ/2(YX−XY)) in
                // `omega-backend-statevector` tests).
                let theta = inst
                    .gate
                    .params
                    .first()
                    .and_then(|p| p.try_as_f64())
                    .unwrap_or(0.0);
                let (a, b) = (&inst.qubits[0], &inst.qubits[1]);
                lines.push(format!("// rbs({}) {}, {}", format_param(theta), a, b));
                lines.push(format!("h {a};"));
                lines.push(format!("h {b};"));
                lines.push(format!("cz {a}, {b};"));
                lines.push(format!("ry({}) {a};", format_param(-theta)));
                lines.push(format!("ry({}) {b};", format_param(theta)));
                lines.push(format!("cz {a}, {b};"));
                lines.push(format!("h {a};"));
                lines.push(format!("h {b};"));
            }
            _ => {
                if let Some(qasm_name) = gate_to_qasm(inst.gate.kind) {
                    let params = if inst.gate.params.is_empty() {
                        String::new()
                    } else {
                        let ps: Vec<String> = inst
                            .gate
                            .params
                            .iter()
                            .map(|p| format_param(p.try_as_f64().unwrap_or(0.0)))
                            .collect();
                        format!("({})", ps.join(", "))
                    };
                    let qrefs: Vec<String> = inst.qubits.iter().map(|q| q.to_string()).collect();
                    lines.push(format!("{qasm_name}{params} {};", qrefs.join(", ")));
                } else {
                    lines.push(format!("// unsupported gate: {:?}", inst.gate.kind));
                }
            }
        }
    }

    lines.join("\n") + "\n"
}

// ---------------------------------------------------------------------------
// Import: QASM string -> Circuit
// ---------------------------------------------------------------------------

fn qasm_to_gate(name: &str) -> Option<GateKind> {
    match name {
        "id" => Some(GateKind::I),
        "x" => Some(GateKind::X),
        "y" => Some(GateKind::Y),
        "z" => Some(GateKind::Z),
        "h" => Some(GateKind::H),
        "s" => Some(GateKind::S),
        "sdg" => Some(GateKind::Sdg),
        "t" => Some(GateKind::T),
        "tdg" => Some(GateKind::Tdg),
        "sx" => Some(GateKind::SX),
        "rx" => Some(GateKind::RX),
        "ry" => Some(GateKind::RY),
        "rz" => Some(GateKind::RZ),
        "p" | "u1" => Some(GateKind::P),
        "u3" | "u" => Some(GateKind::U),
        "cx" | "cnot" => Some(GateKind::CX),
        "cy" => Some(GateKind::CY),
        "cz" => Some(GateKind::CZ),
        "swap" => Some(GateKind::SWAP),
        "cp" => Some(GateKind::CP),
        "ccx" => Some(GateKind::CCX),
        "cswap" => Some(GateKind::CSWAP),
        "rxx" => Some(GateKind::RXX),
        "ryy" => Some(GateKind::RYY),
        "rzz" => Some(GateKind::RZZ),
        _ => None,
    }
}

fn parse_param(s: &str) -> f64 {
    let s = s.trim();
    // Handle pi expressions
    let s = s.replace("pi", &format!("{PI}"));
    // Evaluate simple arithmetic
    // Support: number, number*number, number/number, -number
    let s = s.trim();
    if let Some((a, b)) = s.split_once('/') {
        let a: f64 = a.trim().parse().unwrap_or(0.0);
        let b: f64 = b.trim().parse().unwrap_or(1.0);
        return a / b;
    }
    if let Some((a, b)) = s.split_once('*') {
        let a: f64 = a.trim().parse().unwrap_or(0.0);
        let b: f64 = b.trim().parse().unwrap_or(1.0);
        return a * b;
    }
    s.parse().unwrap_or(0.0)
}

/// Parse an OpenQASM 2.0 string into a Circuit AST.
pub fn from_qasm(qasm_str: &str) -> Result<Circuit, String> {
    let mut circuit = Circuit::new("imported");
    let reg_re = Regex::new(r"(qreg|creg)\s+(\w+)\[(\d+)\];").unwrap();
    let gate_re = Regex::new(r"(\w+)(?:\(([^)]*)\))?\s+(.+);").unwrap();
    let bit_re = Regex::new(r"(\w+)\[(\d+)\]").unwrap();

    for line in qasm_str.lines() {
        let line = line.trim();
        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with("OPENQASM")
            || line.starts_with("include")
        {
            continue;
        }

        // Register declaration
        if let Some(caps) = reg_re.captures(line) {
            let kind_str = &caps[1];
            let name = &caps[2];
            let size: usize = caps[3].parse().unwrap();
            match kind_str {
                "qreg" => {
                    circuit.qreg(name, size);
                }
                "creg" => {
                    circuit.creg(name, size);
                }
                _ => {}
            }
            continue;
        }

        // Barrier
        if line.starts_with("barrier") {
            let qubits: Vec<Qubit> = bit_re
                .captures_iter(line)
                .map(|c| Qubit::new(&c[1], c[2].parse().unwrap()))
                .collect();
            circuit.barrier(&qubits);
            continue;
        }

        // Measure
        if line.starts_with("measure") {
            let refs: Vec<(String, usize)> = bit_re
                .captures_iter(line)
                .map(|c| (c[1].to_string(), c[2].parse().unwrap()))
                .collect();
            if refs.len() >= 2 {
                let q = Qubit::new(&refs[0].0, refs[0].1);
                let c = Clbit::new(&refs[1].0, refs[1].1);
                circuit.measure(&q, &c);
            }
            continue;
        }

        // Reset
        if line.starts_with("reset") {
            if let Some(caps) = bit_re.captures(line) {
                let q = Qubit::new(&caps[1], caps[2].parse().unwrap());
                circuit.reset_qubit(&q);
            }
            continue;
        }

        // Gate instruction
        if let Some(caps) = gate_re.captures(line) {
            let gate_name = &caps[1];
            let params_str = caps.get(2).map(|m| m.as_str());
            let _operands_str = &caps[3];

            if let Some(kind) = qasm_to_gate(gate_name) {
                let params: Vec<f64> = match params_str {
                    Some(s) if !s.is_empty() => s.split(',').map(parse_param).collect(),
                    _ => vec![],
                };
                let qubits: Vec<Qubit> = bit_re
                    .captures_iter(&caps[3])
                    .map(|c| Qubit::new(&c[1], c[2].parse().unwrap()))
                    .collect();
                let gate = GateDef::with_params(kind, params);
                circuit.apply(gate, qubits);
            }
        }
    }

    Ok(circuit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::CircuitBuilder;

    #[test]
    fn test_qasm_roundtrip() {
        let original = CircuitBuilder::new("test", 2, 2)
            .h(0)
            .cx(0, 1)
            .rz(1, PI / 4.0)
            .measure_all()
            .build();

        let qasm_str = to_qasm(&original);
        assert!(qasm_str.contains("OPENQASM 2.0;"));
        assert!(qasm_str.contains("qreg q[2];"));
        assert!(qasm_str.contains("h q[0];"));
        assert!(qasm_str.contains("cx q[0], q[1];"));

        let reimported = from_qasm(&qasm_str).unwrap();
        assert_eq!(reimported.n_qubits(), original.n_qubits());
        assert_eq!(reimported.n_clbits(), original.n_clbits());
        assert_eq!(reimported.gate_count(), original.gate_count());
        assert_eq!(reimported.instructions.len(), original.instructions.len());
    }

    #[test]
    fn test_qasm_parametric_gates() {
        let circ = CircuitBuilder::new("param", 1, 0)
            .rx(0, PI)
            .ry(0, PI / 2.0)
            .rz(0, PI / 4.0)
            .build();
        let qasm_str = to_qasm(&circ);
        assert!(qasm_str.contains("rx(pi)"));
        assert!(qasm_str.contains("ry(1*pi/2)"));
        assert!(qasm_str.contains("rz(1*pi/4)"));
    }
}
