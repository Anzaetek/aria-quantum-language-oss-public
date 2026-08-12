//! OPTICQASM 1.0 import/export.
//!
//! Format for photonic quantum circuits, compatible with omega-functions'
//! photonic backend. Uses `photon` registers, the discrete-variable gates
//! `ps` / `bs_rx`, and the continuous-variable gates `squeeze` / `displace` /
//! `kerr`.
//!
//! Example:
//! ```text
//! OPTICQASM 1.0;
//! photon q[4];
//! ps(0.5) q[0];
//! bs_rx(1.2, 0.3) q[0], q[1];
//! ```
//!
//! # Both directions refuse rather than approximate
//!
//! Every function here returns `Result`. That is not decoration: before this,
//! the emitter wrote a comment for any gate it did not recognise and
//! substituted `0.0` for any parameter it could not evaluate, and the reader
//! skipped any line its regexes did not match. All three produced a
//! *successful* result describing a **different circuit** than the input.
//!
//! Measured, on the code this replaces:
//!
//! | input | old output | old re-import |
//! |---|---|---|
//! | `H q[0]` | `// unsupported: H q[0];` | `Ok`, **0 operations** |
//! | `ps(theta) q[0]` (symbolic) | `ps(0) q[0];` | `Ok`, angle is now **0** |
//! | `photon q[2] pol; pbs q[0],q[1];` | — | `Ok`, **0 registers, 0 operations** |
//!
//! The third is the worst: a complete, grammatically valid polarization
//! circuit parsed "successfully" into nothing at all. See
//! `PLAN-OPTICQASM-INTEGRITY.md` D4/D5/D7.

use super::nodes::*;
use regex::Regex;

/// Read one parameter as a concrete number, or refuse.
///
/// The old code was `params.get(i).and_then(|p| p.try_as_f64()).unwrap_or(0.0)`,
/// which turns both "the parameter is missing" and "the parameter is symbolic"
/// into a hard zero. OPTICQASM 1.0 has no syntax for a symbolic parameter — the
/// grammar's `symbol` rule (`$name`) exists on the reader side but nothing
/// binds it — so a symbolic angle genuinely cannot be written here. That is a
/// real limitation of the format, and it must be *stated*, never silently
/// resolved to 0.
fn num(params: &[super::expr::ParamExpr], i: usize, gate: &str, arity: usize) -> Result<f64, String> {
    let p = params.get(i).ok_or_else(|| {
        format!(
            "OPTICQASM `{gate}` needs {arity} parameter(s), got {}",
            params.len()
        )
    })?;
    p.try_as_f64().ok_or_else(|| {
        format!(
            "OPTICQASM cannot express the symbolic parameter `{p:?}` (argument \
             {i} of `{gate}`): the format has no syntax for one. Bind the \
             parameter to a concrete value before exporting."
        )
    })
}

/// Convert a photonic Circuit to OPTICQASM 1.0.
///
/// Refuses any gate outside the photonic set rather than emitting a comment —
/// OPTICQASM is a photonic dialect, and a qubit gate in one is not a formatting
///problem to be commented away, it is a circuit that cannot be represented.
pub fn to_opticqasm(circuit: &Circuit) -> Result<String, String> {
    let mut lines = vec!["OPTICQASM 1.0;".to_string()];

    for reg in &circuit.registers {
        if reg.kind == RegisterKind::Quantum {
            lines.push(format!("photon {}[{}];", reg.name, reg.size));
        }
    }

    for inst in &circuit.instructions {
        let mode_refs: Vec<String> = inst
            .qubits
            .iter()
            .map(|q| format!("{}[{}]", q.register, q.index))
            .collect();
        let modes = mode_refs.join(", ");
        let ps = &inst.gate.params;

        let line = match inst.gate.kind {
            GateKind::PhaseShifter => format!("ps({}) {};", num(ps, 0, "ps", 1)?, modes),
            GateKind::BeamSplitter => format!(
                "bs_rx({}, {}) {};",
                num(ps, 0, "bs_rx", 2)?,
                num(ps, 1, "bs_rx", 2)?,
                modes
            ),
            GateKind::Squeezing => format!(
                "squeeze({}, {}) {};",
                num(ps, 0, "squeeze", 2)?,
                num(ps, 1, "squeeze", 2)?,
                modes
            ),
            GateKind::Displacement => format!(
                "displace({}, {}) {};",
                num(ps, 0, "displace", 2)?,
                num(ps, 1, "displace", 2)?,
                modes
            ),
            GateKind::Kerr => format!("kerr({}) {};", num(ps, 0, "kerr", 1)?, modes),
            // A barrier has no operational meaning to lose, so dropping it to a
            // comment costs nothing. Every other gate does have one.
            GateKind::Barrier => format!("// barrier {};", modes),
            other => {
                return Err(format!(
                    "OPTICQASM is a photonic dialect and cannot represent `{other:?}` \
                     (on {modes}). Supported: PhaseShifter, BeamSplitter, Squeezing, \
                     Displacement, Kerr, Barrier. Previously this was emitted as a \
                     `// unsupported:` comment, which re-imported as a circuit \
                     silently missing the operation."
                ))
            }
        };
        lines.push(line);
    }

    Ok(lines.join("\n") + "\n")
}

/// Parse an OPTICQASM 1.0 string into a Circuit.
///
/// Every non-empty line must be understood. A line that matches nothing is an
/// error, never a skip — see the module header for what skipping produced.
pub fn from_opticqasm(src: &str) -> Result<Circuit, String> {
    let mut circuit = Circuit::new("photonic");
    let mode_re = Regex::new(r"(\w+)\[(\d+)\]").unwrap();
    // The parameter list is OPTIONAL: the grammar allows `pbs q[0], q[1];`
    // with no parentheses, and requiring them meant every parameter-less gate
    // fell through to the silent skip below.
    let gate_re = Regex::new(r"^(\w+)\s*(?:\(([^)]*)\))?\s+(.+);$").unwrap();
    // `pol` marks a polarization register: N SPATIAL modes each carrying H and
    // V, so 2N optical modes. Requiring `];` immediately meant `photon q[2] pol;`
    // was not recognised as a declaration at all.
    let photon_re = Regex::new(r"^photon\s+(\w+)\[(\d+)\]\s*(pol)?\s*;$").unwrap();

    for (lineno, raw) in src.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("OPTICQASM") {
            continue;
        }

        if let Some(caps) = photon_re.captures(line) {
            let name = &caps[1];
            let size: usize = caps[2]
                .parse()
                .map_err(|e| format!("line {}: bad register size: {e}", lineno + 1))?;
            if caps.get(3).is_some() {
                // Representing polarization in the aria-core AST needs register
                // metadata the AST does not carry yet (PLAN-OPTICQASM-INTEGRITY
                // O4). Refusing is the honest interim: accepting would silently
                // reinterpret every mode index, since `pol` means 2N optical
                // modes and a plain declaration means N.
                return Err(format!(
                    "line {}: `photon {name}[{size}] pol;` — polarization registers are \
                     read by `omega-parser` but the aria-core AST cannot yet carry the \
                     H/V mode mapping, and accepting this would silently reinterpret \
                     every mode index (pol means {} optical modes, not {size}).",
                    lineno + 1,
                    2 * size
                ));
            }
            circuit.qreg(name, size);
            continue;
        }

        let caps = gate_re.captures(line).ok_or_else(|| {
            format!(
                "line {}: cannot parse `{line}` as an OPTICQASM declaration or gate \
                 application. Unrecognised lines used to be skipped, which parsed \
                 whole circuits into empty ones.",
                lineno + 1
            )
        })?;

        let gate_name = &caps[1];
        let params_str = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let modes_str = &caps[3];

        let params: Vec<f64> = if params_str.trim().is_empty() {
            Vec::new()
        } else {
            params_str
                .split(',')
                .map(|s| {
                    let s = s.trim();
                    s.parse::<f64>().map_err(|_| {
                        format!(
                            "line {}: `{s}` is not a number. OPTICQASM 1.0 has no \
                             symbolic parameters; bind it before importing.",
                            lineno + 1
                        )
                    })
                })
                .collect::<Result<_, _>>()?
        };

        let qubits: Vec<Qubit> = mode_re
            .captures_iter(modes_str)
            .map(|c| {
                c[2].parse()
                    .map(|i| Qubit::new(&c[1], i))
                    .map_err(|e| format!("line {}: bad mode index: {e}", lineno + 1))
            })
            .collect::<Result<_, _>>()?;
        if qubits.is_empty() {
            return Err(format!(
                "line {}: `{gate_name}` names no modes",
                lineno + 1
            ));
        }

        // Arity is checked here, not left to the backend: `ps(0.5, 0.2)` is a
        // typo, and accepting it drops the second parameter silently.
        let need = |n: usize| -> Result<(), String> {
            if params.len() == n {
                Ok(())
            } else {
                Err(format!(
                    "line {}: `{gate_name}` takes {n} parameter(s), got {}",
                    lineno + 1,
                    params.len()
                ))
            }
        };

        let gate = match gate_name {
            "ps" => {
                need(1)?;
                GateDef::with_params(GateKind::PhaseShifter, params)
            }
            "bs_rx" | "bs" => {
                need(2)?;
                GateDef::with_params(GateKind::BeamSplitter, params)
            }
            "squeeze" => {
                need(2)?;
                GateDef::with_params(GateKind::Squeezing, params)
            }
            "displace" => {
                need(2)?;
                GateDef::with_params(GateKind::Displacement, params)
            }
            "kerr" => {
                need(1)?;
                GateDef::with_params(GateKind::Kerr, params)
            }
            // `hwp` / `pbs` are read by `omega-parser` and executed by the
            // perceval bridge, but the aria-core AST has no GateKind for
            // either. Naming them explicitly beats "unknown gate", which would
            // suggest the operation does not exist.
            "hwp" | "pbs" => {
                return Err(format!(
                    "line {}: `{gate_name}` is a polarization element that `omega-parser` \
                     reads and the perceval bridge executes, but the aria-core AST has no \
                     GateKind for it yet (PLAN-OPTICQASM-INTEGRITY O4).",
                    lineno + 1
                ))
            }
            // Never `continue`: skipping an unknown gate parses the file
            // "successfully" into a circuit missing an operation, which then
            // executes and returns confident wrong numbers.
            other => {
                return Err(format!(
                    "line {}: unknown photonic gate `{other}` \
                     (supported: ps, bs_rx/bs, squeeze, displace, kerr)",
                    lineno + 1
                ))
            }
        };

        circuit.apply(gate, qubits);
    }

    Ok(circuit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::expr::ParamExpr;

    /// A fixture that touches BOTH profiles.
    ///
    /// The old fixture used `ps` and `bs_rx` only — the two gates that already
    /// worked — so it was invariant under every defect this module was
    /// rewritten to fix. Both profiles are non-negotiable here: `squeeze` /
    /// `displace` / `kerr` are exactly what `omega-parser` could not read.
    fn both_profiles() -> Circuit {
        let mut c = Circuit::new("photonic");
        let m = c.qreg("q", 4);
        c.apply(
            GateDef::with_params(GateKind::PhaseShifter, vec![0.5]),
            vec![m[0].clone()],
        );
        c.apply(
            GateDef::with_params(GateKind::BeamSplitter, vec![1.2, 0.3]),
            vec![m[0].clone(), m[1].clone()],
        );
        c.apply(
            GateDef::with_params(GateKind::Squeezing, vec![0.4, 0.2]),
            vec![m[2].clone()],
        );
        c.apply(
            GateDef::with_params(GateKind::Displacement, vec![0.7, -0.1]),
            vec![m[2].clone()],
        );
        c.apply(
            GateDef::with_params(GateKind::Kerr, vec![0.15]),
            vec![m[3].clone()],
        );
        c
    }

    /// Round trip asserted on the RE-PARSED CIRCUIT, not on substrings.
    ///
    /// The predecessor asserted `contains("ps(0.5) q[0];")`. A substring check
    /// cannot see a dropped gate, a zeroed parameter, or a register that never
    /// appeared — which is why it passed while all three were happening.
    #[test]
    fn round_trip_preserves_every_gate_and_parameter() {
        let circ = both_profiles();
        let text = to_opticqasm(&circ).expect("both profiles are representable");
        let back = from_opticqasm(&text).expect("our own output must re-import");

        assert_eq!(back.n_qubits(), circ.n_qubits(), "register lost");
        assert_eq!(
            back.gate_count(),
            circ.gate_count(),
            "gate count changed across the round trip:\n{text}"
        );
        for (before, after) in circ.instructions.iter().zip(back.instructions.iter()) {
            assert_eq!(before.gate.kind, after.gate.kind, "gate kind changed");
            assert_eq!(
                before.gate.params.len(),
                after.gate.params.len(),
                "parameter count changed for {:?}",
                before.gate.kind
            );
            for (b, a) in before.gate.params.iter().zip(after.gate.params.iter()) {
                let (b, a) = (b.try_as_f64().unwrap(), a.try_as_f64().unwrap());
                assert!((b - a).abs() < 1e-12, "parameter changed: {b} -> {a}");
            }
            let bq: Vec<_> = before.qubits.iter().map(|q| (&q.register, q.index)).collect();
            let aq: Vec<_> = after.qubits.iter().map(|q| (&q.register, q.index)).collect();
            assert_eq!(bq, aq, "modes changed for {:?}", before.gate.kind);
        }
    }

    /// D5: a symbolic parameter used to be emitted as `ps(0) q[0];` — a
    /// well-formed file that parses cleanly and computes the wrong thing.
    #[test]
    fn symbolic_parameter_is_refused_not_zeroed() {
        let mut c = Circuit::new("photonic");
        let m = c.qreg("q", 1);
        c.apply(
            GateDef {
                kind: GateKind::PhaseShifter,
                params: vec![ParamExpr::Symbol("theta".into())],
                label: None,
            },
            vec![m[0].clone()],
        );
        let err = to_opticqasm(&c).expect_err("a symbolic angle is not representable");
        assert!(
            err.contains("symbolic"),
            "the error must say WHY, so a caller can bind the parameter: {err}"
        );
    }

    /// D4: `H` used to become `// unsupported: H q[0];`, which re-imported as
    /// an empty circuit — a successful round trip to a different computation.
    #[test]
    fn non_photonic_gate_is_refused_not_commented_out() {
        let mut c = Circuit::new("photonic");
        let m = c.qreg("q", 1);
        c.apply(GateDef::new(GateKind::H), vec![m[0].clone()]);
        let err = to_opticqasm(&c).expect_err("H is not a photonic gate");
        assert!(err.contains('H'), "the error must name the gate: {err}");
    }

    /// D7: this exact input returned `Ok` with ZERO registers and ZERO
    /// operations — the worst defect in the file, because nothing downstream
    /// had any way to notice.
    #[test]
    fn polarization_declaration_is_never_silently_dropped() {
        let src = "OPTICQASM 1.0;\nphoton q[2] pol;\npbs q[0], q[1];\n";
        match from_opticqasm(src) {
            Ok(c) => panic!(
                "accepted a polarization circuit as {} registers / {} ops — \
                 the old code returned Ok(0, 0) here",
                c.registers.len(),
                c.instructions.len()
            ),
            Err(e) => assert!(
                e.contains("pol"),
                "must refuse because of the pol marker specifically: {e}"
            ),
        }
    }

    /// D7, second half: a parameter-less gate application is grammatical
    /// (`gate_app` makes the parameter list optional), and the old regex
    /// required parentheses, so the line vanished.
    #[test]
    fn unparseable_line_is_an_error_not_a_skip() {
        let err = from_opticqasm("OPTICQASM 1.0;\nphoton q[2];\nnonsense here\n")
            .expect_err("a line matching nothing must not be skipped");
        assert!(err.contains("line 3"), "the error must locate the line: {err}");
    }

    /// Arity: `ps(0.5, 0.2)` is a typo, and accepting it drops a parameter.
    #[test]
    fn wrong_arity_is_refused_on_import() {
        let err = from_opticqasm("OPTICQASM 1.0;\nphoton q[1];\nps(0.5, 0.2) q[0];\n")
            .expect_err("ps takes exactly one parameter");
        assert!(err.contains("1 parameter"), "{err}");
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
