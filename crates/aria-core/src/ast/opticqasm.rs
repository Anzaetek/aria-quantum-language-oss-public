//! OPTICQASM 1.0 import/export.
//!
//! Format for photonic quantum circuits, compatible with omega-functions'
//! photonic backend. Uses `photon` registers and `ps`/`bs_rx` gates.
//!
//! Example:
//! ```text
//! OPTICQASM 1.0;
//! photon q[4];
//! ps(0.5) q[0];
//! bs_rx(1.2, 0.3) q[0], q[1];
//! ```

use super::nodes::*;
use regex::Regex;

/// Convert a photonic Circuit to OPTICQASM 1.0 string.
pub fn to_opticqasm(circuit: &Circuit) -> String {
    let mut lines = vec!["OPTICQASM 1.0;".to_string()];

    // Photon register declarations
    for reg in &circuit.registers {
        if reg.kind == RegisterKind::Quantum {
            lines.push(format!("photon {}[{}];", reg.name, reg.size));
        }
    }

    // Gate applications
    for inst in &circuit.instructions {
        let mode_refs: Vec<String> = inst
            .qubits
            .iter()
            .map(|q| format!("{}[{}]", q.register, q.index))
            .collect();

        match inst.gate.kind {
            GateKind::PhaseShifter => {
                let param = inst
                    .gate
                    .params
                    .first()
                    .and_then(|p| p.try_as_f64())
                    .unwrap_or(0.0);
                lines.push(format!("ps({}) {};", param, mode_refs.join(", ")));
            }
            GateKind::BeamSplitter => {
                let theta = inst
                    .gate
                    .params
                    .first()
                    .and_then(|p| p.try_as_f64())
                    .unwrap_or(0.0);
                let phi = inst
                    .gate
                    .params
                    .get(1)
                    .and_then(|p| p.try_as_f64())
                    .unwrap_or(0.0);
                lines.push(format!(
                    "bs_rx({}, {}) {};",
                    theta,
                    phi,
                    mode_refs.join(", ")
                ));
            }
            GateKind::Squeezing => {
                let r = inst
                    .gate
                    .params
                    .first()
                    .and_then(|p| p.try_as_f64())
                    .unwrap_or(0.0);
                let phi = inst
                    .gate
                    .params
                    .get(1)
                    .and_then(|p| p.try_as_f64())
                    .unwrap_or(0.0);
                lines.push(format!("squeeze({}, {}) {};", r, phi, mode_refs.join(", ")));
            }
            GateKind::Displacement => {
                let alpha_r = inst
                    .gate
                    .params
                    .first()
                    .and_then(|p| p.try_as_f64())
                    .unwrap_or(0.0);
                let alpha_i = inst
                    .gate
                    .params
                    .get(1)
                    .and_then(|p| p.try_as_f64())
                    .unwrap_or(0.0);
                lines.push(format!(
                    "displace({}, {}) {};",
                    alpha_r,
                    alpha_i,
                    mode_refs.join(", ")
                ));
            }
            GateKind::Kerr => {
                let chi = inst
                    .gate
                    .params
                    .first()
                    .and_then(|p| p.try_as_f64())
                    .unwrap_or(0.0);
                lines.push(format!("kerr({}) {};", chi, mode_refs.join(", ")));
            }
            GateKind::Barrier => {
                lines.push(format!("// barrier {};", mode_refs.join(", ")));
            }
            _ => {
                lines.push(format!(
                    "// unsupported: {:?} {};",
                    inst.gate.kind,
                    mode_refs.join(", ")
                ));
            }
        }
    }

    lines.join("\n") + "\n"
}

/// Parse an OPTICQASM 1.0 string into a Circuit.
pub fn from_opticqasm(src: &str) -> Result<Circuit, String> {
    let mut circuit = Circuit::new("photonic");
    let mode_re = Regex::new(r"(\w+)\[(\d+)\]").unwrap();
    let gate_re = Regex::new(r"(\w+)\(([^)]*)\)\s+(.+);").unwrap();
    let photon_re = Regex::new(r"photon\s+(\w+)\[(\d+)\];").unwrap();

    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("OPTICQASM") {
            continue;
        }

        // Photon declaration
        if let Some(caps) = photon_re.captures(line) {
            let name = &caps[1];
            let size: usize = caps[2].parse().unwrap();
            circuit.qreg(name, size);
            continue;
        }

        // Gate application
        if let Some(caps) = gate_re.captures(line) {
            let gate_name = &caps[1];
            let params_str = &caps[2];
            let modes_str = &caps[3];

            let params: Vec<f64> = params_str
                .split(',')
                .map(|s| s.trim().parse().unwrap_or(0.0))
                .collect();

            let qubits: Vec<Qubit> = mode_re
                .captures_iter(modes_str)
                .map(|c| Qubit::new(&c[1], c[2].parse().unwrap()))
                .collect();

            let gate = match gate_name {
                "ps" => GateDef::with_params(GateKind::PhaseShifter, params),
                "bs_rx" | "bs" => GateDef::with_params(GateKind::BeamSplitter, params),
                "squeeze" => GateDef::with_params(GateKind::Squeezing, params),
                "displace" => GateDef::with_params(GateKind::Displacement, params),
                "kerr" => GateDef::with_params(GateKind::Kerr, params),
                // Never `continue`: skipping an unknown gate parses the file
                // "successfully" into a circuit missing an operation, which
                // then executes and returns confident wrong numbers. Same
                // defect class as the lowering drop in
                // `backends::omega::try_to_omega_ir`. `omega-parser`'s
                // OPTICQASM front end already refuses unknown gates; this one
                // now matches it.
                other => {
                    return Err(format!(
                        "unknown photonic gate '{other}' in OPTICQASM input \
                         (supported: ps, bs_rx/bs, squeeze, displace, kerr)"
                    ))
                }
            };

            circuit.apply(gate, qubits);
        }
    }

    Ok(circuit)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_opticqasm_roundtrip() {
        let mut circ = Circuit::new("photonic");
        let modes = circ.qreg("q", 4);
        circ.apply(
            GateDef::with_params(GateKind::PhaseShifter, vec![0.5]),
            vec![modes[0].clone()],
        );
        circ.apply(
            GateDef::with_params(GateKind::PhaseShifter, vec![1.0]),
            vec![modes[1].clone()],
        );
        circ.apply(
            GateDef::with_params(GateKind::BeamSplitter, vec![1.2, 0.3]),
            vec![modes[0].clone(), modes[1].clone()],
        );

        let opticqasm = to_opticqasm(&circ);
        assert!(opticqasm.contains("OPTICQASM 1.0;"));
        assert!(opticqasm.contains("photon q[4];"));
        assert!(opticqasm.contains("ps(0.5) q[0];"));
        assert!(opticqasm.contains("bs_rx(1.2, 0.3) q[0], q[1];"));

        let reimported = from_opticqasm(&opticqasm).unwrap();
        assert_eq!(reimported.n_qubits(), 4);
        assert_eq!(reimported.gate_count(), 3);
    }

    #[test]
    fn test_parse_clements_mesh() {
        let src = r#"OPTICQASM 1.0;
photon q[4];
ps(0.5) q[0];
ps(0.7) q[1];
ps(0.3) q[2];
ps(0.9) q[3];
bs_rx(1.2, 0.1) q[0], q[1];
bs_rx(0.8, 0.2) q[2], q[3];
bs_rx(0.5, 0.4) q[1], q[2];
"#;
        let circ = from_opticqasm(src).unwrap();
        assert_eq!(circ.n_qubits(), 4);
        assert_eq!(circ.gate_count(), 7); // 4 ps + 3 bs_rx
    }
}
