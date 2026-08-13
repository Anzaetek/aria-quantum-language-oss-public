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

/// A polarization gate is meaningless on a register that does not carry H/V.
///
/// Refusing here is what stops the emitter producing the file described above:
/// `hwp` under a plain `photon q[N];` declaration. There is no way for a reader
/// to detect that, because both forms are grammatical.
fn polarized_check(circuit: &Circuit, inst: &Instruction, gate: &str) -> Result<(), String> {
    for q in &inst.qubits {
        let reg = circuit
            .registers
            .iter()
            .find(|r| r.name == q.register)
            .ok_or_else(|| format!("`{gate}` names undeclared register `{}`", q.register))?;
        if !reg.polarized {
            return Err(format!(
                "`{gate}` is a polarization element but `{}` is not a polarized register. \
                 Declare it with `Circuit::qreg_polarized` so it emits as \
                 `photon {}[{}] pol;` — under a plain declaration the same file would \
                 parse and every mode index would mean something else ({} optical modes \
                 instead of {}).",
                reg.name,
                reg.name,
                reg.size,
                2 * reg.size,
                reg.size
            ));
        }
    }
    Ok(())
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
            // The `pol` marker is NOT decoration, and it is why O4 could not be
            // "just emit hwp/pbs": `photon q[N] pol;` declares N SPATIAL modes
            // each carrying H and V — 2N optical modes indexed 2s+p — while
            // `photon q[N];` declares N optical modes. Emitting a polarization
            // gate under a plain declaration yields a file that parses, refuses
            // nothing, and means something different in every mode index. The
            // marker and the gates land together or not at all.
            let marker = if reg.polarized { " pol" } else { "" };
            lines.push(format!("photon {}[{}]{marker};", reg.name, reg.size));
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
            // Polarization elements. `omega-parser` reads both and expands them
            // into phase shifters and beam splitters on the H/V sub-modes; the
            // expansion means the spelling cannot survive a round trip through
            // the IR, so this AST is the only layer that can re-emit `hwp` as
            // `hwp` rather than as four ops.
            GateKind::HalfWavePlate => {
                polarized_check(circuit, inst, "hwp")?;
                format!("hwp({}) {};", num(ps, 0, "hwp", 1)?, modes)
            }
            // `pbs` takes no parameters, unlike every other gate here — the
            // grammar makes the parameter list optional precisely for it.
            GateKind::PolarizingBeamSplitter => {
                polarized_check(circuit, inst, "pbs")?;
                if !ps.is_empty() {
                    return Err(format!(
                        "OPTICQASM `pbs` takes no parameters, got {}",
                        ps.len()
                    ));
                }
                format!("pbs {};", modes)
            }
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
/// # The unit is a STATEMENT, not a line
///
/// The first version of this function iterated over `src.lines()`. That is the
/// wrong unit and it re-committed the defect the module header describes, one
/// level down. The grammar's `WHITESPACE` rule includes `\n`, so a newline is
/// no more significant than a space and a statement may share a line with any
/// other:
///
/// | input | old result |
/// |---|---|
/// | `OPTICQASM 1.0; photon q[2]; ps(0.5) q[0];` (one line) | **`Ok`, 0 registers, 0 operations** |
/// | `ps(0.5) q[0]; ps(0.7) q[1];` (one line) | **`Ok`, ONE gate, param 0.5, on modes [0, 1]** |
///
/// The first is `Ok(0, 0)` again — a whole circuit parsed into nothing, because
/// `line.starts_with("OPTICQASM")` skipped the entire line rather than the
/// header token. The second is worse than a drop: the two gates were *merged*
/// into one with a different parameter and different modes, because the
/// gate regex was anchored to the line and greedily swallowed the tail.
///
/// So this splits on `;` after stripping comments, which is what the grammar
/// does. Every statement must be understood; none may be skipped.
///
/// # Validation matches `omega-parser`
///
/// Undefined registers, out-of-range mode indices, and wrong parameter or mode
/// counts are refused here exactly as they are in
/// `omega_parser::lower::lower_opticqasm`. Two readers of the same dialect
/// disagreeing about what is valid is the same class of defect as one of them
/// being unable to read the other's output — and this side was the silent one:
/// `ps(0.5) zz[7];` with no `zz` register declared used to return `Ok`.
///
/// `opticqasm_reader_agreement.rs` pins the two against each other so they
/// cannot drift apart again.
pub fn from_opticqasm(src: &str) -> Result<Circuit, String> {
    let mut circuit = Circuit::new("photonic");
    let mode_re = Regex::new(r"^(\w+)\[(\d+)\]$").unwrap();
    let gate_re = Regex::new(r"^(\w+)\s*(?:\(([^)]*)\))?\s+(.+)$").unwrap();
    // `pol` marks a polarization register: N SPATIAL modes each carrying H and
    // V, so 2N optical modes.
    let photon_re = Regex::new(r"^photon\s+(\w+)\[(\d+)\]\s*(pol)?$").unwrap();

    // Declared registers, so mode references can be validated instead of
    // invented.
    let mut regs: std::collections::HashMap<String, (usize, bool)> =
        std::collections::HashMap::new();
    let mut seen_header = false;

    for (stmt, line) in statements(src) {
        if !seen_header {
            let version = stmt
                .strip_prefix("OPTICQASM")
                .map(str::trim)
                .ok_or_else(|| {
                    format!("line {line}: expected an `OPTICQASM <version>;` header, got `{stmt}`")
                })?;
            if version.is_empty() {
                return Err(format!("line {line}: the OPTICQASM header names no version"));
            }
            seen_header = true;
            continue;
        }

        if let Some(caps) = photon_re.captures(stmt) {
            let name = caps[1].to_string();
            let size: usize = caps[2]
                .parse()
                .map_err(|e| format!("line {line}: bad register size: {e}"))?;
            let polarized = caps.get(3).is_some();
            if regs.insert(name.clone(), (size, polarized)).is_some() {
                return Err(format!("line {line}: register `{name}` declared twice"));
            }
            // `size` stays SPATIAL modes on both sides, matching the
            // declaration; the doubling to optical modes happens in
            // `omega-parser`'s lowering and nowhere else, so the two agree.
            if polarized {
                circuit.qreg_polarized(&name, size);
            } else {
                circuit.qreg(&name, size);
            }
            continue;
        }

        let caps = gate_re.captures(stmt).ok_or_else(|| {
            format!(
                "line {line}: cannot parse `{stmt}` as an OPTICQASM declaration or gate \
                 application. Unrecognised statements used to be skipped, which parsed \
                 whole circuits into empty ones."
            )
        })?;
        let gate_name = caps[1].to_string();
        let params_str = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let modes_str = caps[3].to_string();

        let params: Vec<f64> = if params_str.trim().is_empty() {
            Vec::new()
        } else {
            params_str
                .split(',')
                .map(|s| {
                    let s = s.trim();
                    // `f64::from_str` accepts `1e-5`, `+0.5`, `inf` and `NaN`,
                    // none of which the grammar's `number` rule admits. Accepting
                    // them here would produce a Circuit that re-exports to a file
                    // `omega-parser` cannot read — the round-trip gap again.
                    if !NUMBER_RE.is_match(s) {
                        return Err(format!(
                            "line {line}: `{s}` is not an OPTICQASM number. The grammar \
                             admits an optional sign, digits, and an optional fractional \
                             part — no exponent, no `inf`, no `NaN`, and no symbolic \
                             parameters."
                        ));
                    }
                    s.parse::<f64>()
                        .map_err(|e| format!("line {line}: `{s}`: {e}"))
                })
                .collect::<Result<_, _>>()?
        };

        // Each mode reference is matched WHOLE. The previous version scanned the
        // tail for anything shaped `name[index]`, so `kerr(0.1) garbage q[0];`
        // parsed happily by ignoring the garbage.
        let mut qubits: Vec<Qubit> = Vec::new();
        for raw in modes_str.split(',') {
            let raw = raw.trim();
            let m = mode_re.captures(raw).ok_or_else(|| {
                format!("line {line}: `{raw}` is not a mode reference (expected `name[index]`)")
            })?;
            let reg = &m[1];
            let index: usize = m[2]
                .parse()
                .map_err(|e| format!("line {line}: bad mode index: {e}"))?;
            let (size, _pol) = regs.get(reg).ok_or_else(|| {
                format!("line {line}: undefined photon register `{reg}`")
            })?;
            if index >= *size {
                return Err(format!(
                    "line {line}: mode index {index} out of range for `{reg}[{size}]`"
                ));
            }
            qubits.push(Qubit::new(reg, index));
        }
        if qubits.is_empty() {
            return Err(format!("line {line}: `{gate_name}` names no modes"));
        }

        // Arity is checked on BOTH parameters and modes. Checking only
        // parameters is what let `bs_rx(1.2, 0.3) q[0];` through as a
        // one-mode beam splitter.
        let check = |np: usize, nm: usize| -> Result<(), String> {
            if params.len() != np {
                return Err(format!(
                    "line {line}: `{gate_name}` takes {np} parameter(s), got {}",
                    params.len()
                ));
            }
            if qubits.len() != nm {
                return Err(format!(
                    "line {line}: `{gate_name}` acts on {nm} mode(s), got {}",
                    qubits.len()
                ));
            }
            Ok(())
        };

        let gate = match gate_name.as_str() {
            "ps" => {
                check(1, 1)?;
                GateDef::with_params(GateKind::PhaseShifter, params)
            }
            "bs_rx" | "bs" => {
                check(2, 2)?;
                GateDef::with_params(GateKind::BeamSplitter, params)
            }
            "squeeze" => {
                check(2, 1)?;
                GateDef::with_params(GateKind::Squeezing, params)
            }
            "displace" => {
                check(2, 1)?;
                GateDef::with_params(GateKind::Displacement, params)
            }
            "kerr" => {
                check(1, 1)?;
                GateDef::with_params(GateKind::Kerr, params)
            }
            // `hwp` / `pbs` are read by `omega-parser` and executed by the
            // perceval bridge, but the aria-core AST has no GateKind for
            // either. Naming them explicitly beats "unknown gate", which would
            // suggest the operation does not exist.
            "hwp" => {
                check(1, 1)?;
                polarized_regs_only(&regs, &qubits, "hwp", line)?;
                GateDef::with_params(GateKind::HalfWavePlate, params)
            }
            "pbs" => {
                check(0, 2)?;
                polarized_regs_only(&regs, &qubits, "pbs", line)?;
                GateDef::new(GateKind::PolarizingBeamSplitter)
            }
            // Never skip: skipping an unknown gate parses the file
            // "successfully" into a circuit missing an operation, which then
            // executes and returns confident wrong numbers.
            other => {
                return Err(format!(
                    "line {line}: unknown photonic gate `{other}` \
                     (supported: ps, bs_rx/bs, squeeze, displace, kerr)"
                ))
            }
        };

        circuit.apply(gate, qubits);
    }

    if !seen_header {
        return Err("empty input: an OPTICQASM file must open with `OPTICQASM <version>;`".into());
    }
    Ok(circuit)
}

/// A polarization gate on a non-polarized register is refused on IMPORT too.
///
/// Symmetry with the emitter is the point: if only one side checked, the two
/// readers of this dialect would disagree about validity — the defect
/// `opticqasm_reader_agreement.rs` exists to prevent.
fn polarized_regs_only(
    regs: &std::collections::HashMap<String, (usize, bool)>,
    qubits: &[Qubit],
    gate: &str,
    line: usize,
) -> Result<(), String> {
    for q in qubits {
        match regs.get(&q.register) {
            Some((_, true)) => {}
            _ => {
                return Err(format!(
                    "line {line}: `{gate}` is a polarization element but `{}` was not \
                     declared `pol`. Under a plain declaration every mode index means \
                     something different, and nothing downstream can detect it.",
                    q.register
                ))
            }
        }
    }
    Ok(())
}

/// The grammar's `number` rule: `"-"? ~ ASCII_DIGIT+ ~ ("." ~ ASCII_DIGIT+)?`.
static NUMBER_RE: std::sync::LazyLock<Regex> =
    std::sync::LazyLock::new(|| Regex::new(r"^-?\d+(\.\d+)?$").unwrap());

/// Split into `;`-terminated statements, stripping comments, and pair each with
/// the line it started on so diagnostics stay locatable.
///
/// A trailing fragment with no `;` is yielded as a statement so it is reported
/// as unparseable rather than dropped — an unterminated statement is a
/// truncated file, which is precisely when silence is most costly.
fn statements(src: &str) -> Vec<(&str, usize)> {
    let mut out = Vec::new();
    let mut line = 1usize;
    let mut start = 0usize;
    let bytes = src.as_bytes();
    let mut i = 0usize;
    let mut stmt_line = 1usize;
    let mut pending = false;

    while i < bytes.len() {
        // `//` to end of line, exactly as the grammar's COMMENT rule.
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            let nl = src[i..].find('\n').map(|k| i + k).unwrap_or(bytes.len());
            if pending {
                // Keep the text before the comment; the comment itself is not
                // part of the statement.
                out.push((src[start..i].trim(), stmt_line));
                pending = false;
            }
            i = nl;
            start = i;
            continue;
        }
        if bytes[i] == b';' {
            let text = src[start..i].trim();
            if !text.is_empty() {
                out.push((text, stmt_line));
            }
            pending = false;
            i += 1;
            start = i;
            continue;
        }
        if bytes[i] == b'\n' {
            line += 1;
        }
        if !pending && !bytes[i].is_ascii_whitespace() {
            pending = true;
            stmt_line = line;
        }
        i += 1;
    }
    let tail = src[start..].trim();
    if !tail.is_empty() {
        out.push((tail, stmt_line));
    }
    out
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
    ///
    /// The assertion is deliberately on the CONTENT, not on acceptance. Before
    /// O4 this input was refused, and the test asserted the refusal; when O4
    /// landed and the input became legitimately readable, that assertion failed
    /// and had to be rewritten — which is right, but it means "it was refused"
    /// was never the real invariant. The invariant is that a grammatical
    /// polarization circuit is either refused with a reason or parsed IN FULL,
    /// and never silently reduced to nothing.
    #[test]
    fn polarization_circuit_is_parsed_in_full_never_silently_emptied() {
        let src = "OPTICQASM 1.0;\nphoton q[2] pol;\npbs q[0], q[1];\n";
        let c = from_opticqasm(src).expect("O4: polarization is readable");
        assert_eq!(c.registers.len(), 1, "register lost:\n{src}");
        assert_eq!(c.instructions.len(), 1, "the pbs was dropped:\n{src}");
        assert_eq!(c.instructions[0].gate.kind, GateKind::PolarizingBeamSplitter);
        assert_eq!(c.instructions[0].qubits.len(), 2, "pbs spans two spatial modes");
        assert!(
            c.registers[0].polarized,
            "the `pol` marker was dropped — the register would re-emit as \
             `photon q[2];` and every mode index would change meaning"
        );
        // `size` counts SPATIAL modes on both sides of the boundary; the
        // doubling to optical modes happens in omega-parser's lowering and
        // nowhere else.
        assert_eq!(c.registers[0].size, 2, "size must stay spatial, not doubled here");
    }

    /// A polarization gate on a NON-polarized register is refused. Accepting it
    /// is undetectable downstream: both declarations are grammatical, and the
    /// only difference is what every mode index means.
    #[test]
    fn polarization_gate_needs_a_polarized_register() {
        let err = from_opticqasm("OPTICQASM 1.0;\nphoton q[2];\npbs q[0], q[1];\n")
            .expect_err("pbs on a plain register must be refused");
        assert!(err.contains("pol"), "the error must name the marker: {err}");
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
