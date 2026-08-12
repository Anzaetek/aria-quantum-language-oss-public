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
        GateKind::CRz => Some("crz"),
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

/// Convert a Circuit AST to an OpenQASM 2.0 string, or refuse if QASM 2.0
/// cannot express the circuit.
///
/// # Why this is fallible (FIXES_PLAN.md Part H)
///
/// It used to return `String` and emit a classically-conditioned gate
/// **unconditionally** — the guard vanished with no diagnostic on either side.
/// Measured on `H q0; measure q0 -> c0; when c0 == 1 { X q1 }`: Qiskit running
/// Aria's own export returned `{"11": 2002, "10": 1998}` where the true
/// distribution is `{"00": 1995, "11": 2005}`. The correlation is destroyed,
/// the file is valid QASM, and the importer has no way to know a guard was
/// lost. A lossy export is the worst case for silence, because both ends look
/// healthy.
///
/// # Why it cannot simply emit `if`
///
/// Aria's condition is a **single classical bit** (`Option<(Clbit, u64)>`).
/// QASM 2.0's `if` compares a **whole classical register**: `if (c == V)`. On
/// a register of size 1 the two coincide. On a wider one they do not —
/// `if (c == 1)` asserts the *entire register* equals 1, a different
/// predicate. Emitting it anyway would trade a silent drop for a silent
/// **change of meaning**, which is worse: the output would look right and be
/// wrong.
///
/// So a size-1 register gets the `if`; anything wider is refused, naming the
/// register and bit and pointing at [`to_qasm3`], whose `if (c[i] == V)` can
/// address a single bit.
pub fn to_qasm(circuit: &Circuit) -> Result<String, String> {
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
        // Remember where this instruction's output starts. Some kinds expand
        // to SEVERAL statements (RBS becomes ~7), and QASM 2.0's `if` guards
        // exactly one statement — so a conditioned multi-line gate needs the
        // guard on every line it produced, not just the first. Prefixing by
        // line range handles single- and multi-line expansions identically;
        // special-casing the single-line arm would leave a conditioned RBS
        // emitting six unguarded statements and one guarded one — a NEW silent
        // defect of exactly the kind this change removes.
        let emitted_from = lines.len();

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

        if let Some((clbit, value)) = &inst.condition {
            // QASM 2.0: `if (creg == N) qop;` where a *qop* is a gate
            // application, a measurement, or a reset. `barrier` is NOT one.
            //
            // Measured against Qiskit 2.5.1: `if (c==1) barrier q;` is
            // REJECTED with "needed a gate application, measurement or reset",
            // while guarded gate / reset / measure are all accepted. Our own
            // pest grammar happens to admit it — `barrier q` parses as a gate
            // application — so a round-trip through omega-parser alone would
            // NOT have caught this. The export exists for interchange, so the
            // consumer that matters is the other toolchain.
            if inst.gate.kind == GateKind::Barrier {
                return Err(format!(
                    "cannot export a conditioned `barrier` to OpenQASM 2.0: `if` \
                     takes a gate application, measurement or reset, and a barrier \
                     is none of those (Qiskit rejects `if ({} == {}) barrier ...;`). \
                     A barrier is a scheduling hint with no classical semantics — \
                     drop the guard, or express the intent with a guarded gate.",
                    clbit.register, value
                ));
            }

            let reg = circuit
                .registers
                .iter()
                .find(|r| r.name == clbit.register && r.kind == RegisterKind::Classical)
                .ok_or_else(|| {
                    format!(
                        "conditioned gate {:?} references classical register `{}`, \
                         which the circuit does not declare",
                        inst.gate.kind, clbit.register
                    )
                })?;

            if reg.size != 1 {
                return Err(format!(
                    "cannot export `when {}[{}] == {}` to OpenQASM 2.0: Aria conditions \
                     on a SINGLE classical bit, but QASM 2.0's `if` compares the WHOLE \
                     register, and `{}` has size {}. Emitting `if ({} == {})` would \
                     assert that the entire register equals {}, a different predicate — \
                     a silent change of meaning. Use `to_qasm3` (OpenQASM 3's \
                     `if ({}[{}] == {})` addresses a single bit), or restructure the \
                     circuit so the guard reads a size-1 register.",
                    clbit.register, clbit.index, value,
                    clbit.register, reg.size,
                    clbit.register, value, value,
                    clbit.register, clbit.index, value,
                ));
            }

            // Size 1: `c[0] == V` and `c == V` are the same predicate.
            for line in lines.iter_mut().skip(emitted_from) {
                if line.starts_with("//") || line.is_empty() {
                    continue;
                }
                *line = format!("if ({} == {}) {}", clbit.register, value, line);
            }
        }
    }

    Ok(lines.join("\n") + "\n")
}

/// QASM 3 gate name for a kind. Identical to the 2.0 table except `U` is the
/// OpenQASM 3 built-in gate (2.0 spells it `u3`).
fn gate_to_qasm3(kind: GateKind) -> Option<&'static str> {
    match kind {
        GateKind::U => Some("U"),
        // `stdgates.inc` defines no rxx/ryy/rzz — emitting those names produces
        // a file a strict OpenQASM 3 consumer rejects. Fall through to `None`
        // so they become `// unsupported gate:` comments like any other
        // kind without a stdgates spelling (the 2.0 path, pinned, still emits
        // them for backward compatibility).
        GateKind::RXX | GateKind::RYY | GateKind::RZZ => None,
        other => gate_to_qasm(other),
    }
}

/// Format a parameter as a plain shortest-round-trip decimal. QASM 3 export
/// deliberately keeps `format_param`'s `pi`-fraction pretty-printing **out** of
/// the interchange path — a decimal parses back to the same f64 without the
/// formatter-drift class of bug that `parse_param` had to absorb on the 2.0
/// side.
fn format_param3(p: f64) -> String {
    format!("{p}")
}

/// Convert a Circuit AST to an OpenQASM **3.0** string.
///
/// Additive to [`to_qasm`] (which stays OpenQASM 2.0): same gate decompositions
/// — including the exact RBS expansion — but a `OPENQASM 3.0;` +
/// `stdgates.inc` header, `qubit[n]` / `bit[n]` declarations, the
/// `c = measure q;` assignment form, and plain-decimal parameters.
///
/// Profile: gate model only. Emitted constructs are register declarations,
/// unitary gate applications (unsupported kinds become `// unsupported gate:`
/// comments, as in 2.0), `barrier`, `reset`, and single-bit
/// `creg[i] = measure qreg[i];` measurements. No pulse blocks, no classical
/// control flow (`if`), no gate/subroutine definitions are emitted.
pub fn to_qasm3(circuit: &Circuit) -> String {
    let mut lines = vec![
        "OPENQASM 3.0;".to_string(),
        "include \"stdgates.inc\";".to_string(),
        String::new(),
    ];

    for reg in &circuit.registers {
        match reg.kind {
            RegisterKind::Quantum => lines.push(format!("qubit[{}] {};", reg.size, reg.name)),
            RegisterKind::Classical => lines.push(format!("bit[{}] {};", reg.size, reg.name)),
        }
    }
    lines.push(String::new());

    for inst in &circuit.instructions {
        // Where this instruction's output starts, so a classical guard can wrap
        // everything it produced. Same line-range approach as `to_qasm`, and
        // for the same reason: RBS expands to several statements, and guarding
        // only the first would leave the rest unguarded.
        let emitted_from = lines.len();

        match inst.gate.kind {
            GateKind::Barrier => {
                let qrefs: Vec<String> = inst.qubits.iter().map(|q| q.to_string()).collect();
                lines.push(format!("barrier {};", qrefs.join(", ")));
            }
            GateKind::Measure => {
                // QASM 3 assignment form: `c[i] = measure q[i];`.
                lines.push(format!("{} = measure {};", inst.clbits[0], inst.qubits[0]));
            }
            GateKind::Reset => {
                lines.push(format!("reset {};", inst.qubits[0]));
            }
            GateKind::RBS => {
                // Same exact decomposition as the 2.0 path (stdgates has no
                // RBS): RBS(θ) = (H⊗H)·CZ·(Ry(−θ)⊗Ry(θ))·CZ·(H⊗H).
                let theta = inst
                    .gate
                    .params
                    .first()
                    .and_then(|p| p.try_as_f64())
                    .unwrap_or(0.0);
                let (a, b) = (&inst.qubits[0], &inst.qubits[1]);
                lines.push(format!("// rbs({}) {}, {}", format_param3(theta), a, b));
                lines.push(format!("h {a};"));
                lines.push(format!("h {b};"));
                lines.push(format!("cz {a}, {b};"));
                lines.push(format!("ry({}) {a};", format_param3(-theta)));
                lines.push(format!("ry({}) {b};", format_param3(theta)));
                lines.push(format!("cz {a}, {b};"));
                lines.push(format!("h {a};"));
                lines.push(format!("h {b};"));
            }
            _ => {
                if let Some(qasm_name) = gate_to_qasm3(inst.gate.kind) {
                    let params = if inst.gate.params.is_empty() {
                        String::new()
                    } else {
                        let ps: Vec<String> = inst
                            .gate
                            .params
                            .iter()
                            .map(|p| format_param3(p.try_as_f64().unwrap_or(0.0)))
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

        // OpenQASM 3's `if (c[i] == V)` addresses a SINGLE BIT — exactly
        // what an Aria condition is — so unlike the 2.0 path there is nothing
        // to refuse and no register-width restriction.
        //
        // This gap made `to_qasm`'s refusal message HARMFUL: it told users to
        // come here for single-bit guards, and this function had zero
        // references to `inst.condition`, emitting every guarded instruction
        // bare. Following our own advice produced the silently wrong circuit
        // Part H measured as {"11": 2002, "10": 1998} against a true
        // {"00": 1995, "11": 2005}. Misdirection is worse than silence.
        //
        // Comment lines are left unwrapped: prefixing `// rbs(...)` with a
        // guard reads as though the condition applies to a statement that is
        // not there.
        if let Some((clbit, value)) = &inst.condition {
            for line in lines.iter_mut().skip(emitted_from) {
                if line.starts_with("//") || line.is_empty() {
                    continue;
                }
                *line = format!(
                    "if ({}[{}] == {}) {}",
                    clbit.register, clbit.index, value, line
                );
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
        // `U` and `CX` are the OpenQASM 2.0 *builtin* gates (qelib1 adds the
        // lowercase `u3`/`cx` spellings); accept both so a spec-valid file that
        // doesn't include qelib1 still imports.
        "u3" | "u" | "U" => Some(GateKind::U),
        "cx" | "cnot" | "CX" => Some(GateKind::CX),
        "cy" => Some(GateKind::CY),
        "cz" => Some(GateKind::CZ),
        "swap" => Some(GateKind::SWAP),
        "cp" => Some(GateKind::CP),
        "crz" => Some(GateKind::CRz),
        "ccx" => Some(GateKind::CCX),
        "cswap" => Some(GateKind::CSWAP),
        "rxx" => Some(GateKind::RXX),
        "ryy" => Some(GateKind::RYY),
        "rzz" => Some(GateKind::RZZ),
        _ => None,
    }
}

/// Parse a single QASM parameter atom: a decimal literal (incl. scientific
/// notation) or the constant `pi` (optionally sign-prefixed).
fn parse_atom(atom: &str) -> Result<f64, String> {
    let atom = atom.trim();
    match atom {
        "pi" => Ok(PI),
        "-pi" => Ok(-PI),
        "+pi" => Ok(PI),
        _ => atom
            .parse::<f64>()
            .map_err(|_| format!("cannot parse QASM parameter atom '{atom}'")),
    }
}

/// Evaluate a QASM parameter expression: a sequence of `pi`/decimal atoms
/// joined by `*` and `/`, applied **left-to-right at equal precedence**.
///
/// This is sufficient for everything `format_param` emits (`pi`, `-pi`,
/// `n*pi`, `n*pi/2`, `n*pi/4`) and for common external QASM (`2*pi`, `pi/2`,
/// `3*pi/4`, decimals, scientific notation).
///
/// It returns an `Err` on anything it cannot evaluate rather than silently
/// defaulting to `0.0`. The previous implementation split on `/` *before*
/// `*`, so `1*pi/4` parsed as `parse("1*3.14159…") = 0.0` divided by 4 — every
/// `n*pi/2` / `n*pi/4` angle silently became 0.0 on re-import.
fn parse_param(s: &str) -> Result<f64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty QASM parameter expression".to_string());
    }
    let mut result = 1.0_f64;
    let mut op = '*';
    let mut atom = String::new();
    for c in s.chars() {
        if c == '*' || c == '/' {
            let a = parse_atom(&atom)?;
            if op == '*' {
                result *= a;
            } else {
                result /= a;
            }
            op = c;
            atom.clear();
        } else {
            atom.push(c);
        }
    }
    let a = parse_atom(&atom)?;
    if op == '*' {
        result *= a;
    } else {
        result /= a;
    }
    // A non-finite angle (e.g. `pi/0`, or an `inf`/`nan` atom that `f64::parse`
    // accepts) is never a valid gate parameter — reject rather than import it.
    if !result.is_finite() {
        return Err(format!("QASM parameter '{s}' is not a finite value"));
    }
    Ok(result)
}

/// Parse an OpenQASM 2.0 string into a Circuit AST.
///
/// This importer reads the OpenQASM **2.0** subset only. It fails loudly
/// rather than silently dropping content: an `OPENQASM` header declaring a
/// version outside `2.x` is rejected, and any non-empty statement that is not
/// a recognized declaration/gate/measure/reset/barrier returns an `Err`. (The
/// fuller pest-based pipeline that accepts a QASM 3 minimal subset lives in the
/// `omega-parser` crate and is unaffected by this function.)
pub fn from_qasm(qasm_str: &str) -> Result<Circuit, String> {
    // Parse a `\d+` capture as a non-negative index. The regexes below accept
    // arbitrarily long digit runs, so an oversized literal (`q[99999999999999999999]`)
    // must fail loudly here rather than panic on `.parse().unwrap()` — the same
    // fail-loud contract every other malformed input in this importer honors.
    fn parse_idx(raw: &str, lineno: usize) -> Result<usize, String> {
        raw.parse::<usize>()
            .map_err(|_| format!("'{raw}' at line {lineno} is not a valid non-negative integer"))
    }
    // Reject a bit reference to an undeclared register or an out-of-range index,
    // so `x q[5]` on `qreg q[1]` fails at import instead of producing a circuit
    // that only misbehaves (or panics) downstream.
    fn check_ref(
        sizes: &std::collections::HashMap<String, usize>,
        kind: &str,
        name: &str,
        index: usize,
        lineno: usize,
    ) -> Result<(), String> {
        match sizes.get(name) {
            None => Err(format!(
                "undeclared {kind} register '{name}' at line {lineno}"
            )),
            Some(&size) if index >= size => Err(format!(
                "{kind} index {name}[{index}] out of range at line {lineno} \
                 (register '{name}' has size {size})"
            )),
            Some(_) => Ok(()),
        }
    }

    let mut circuit = Circuit::new("imported");
    let mut qreg_sizes: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut creg_sizes: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let version_re = Regex::new(r"^OPENQASM\s+(\d+)(?:\.(\d+))?\s*;").unwrap();
    let reg_re = Regex::new(r"(qreg|creg)\s+(\w+)\[(\d+)\];").unwrap();
    let gate_re = Regex::new(r"(\w+)(?:\(([^)]*)\))?\s+(.+);").unwrap();
    let bit_re = Regex::new(r"(\w+)\[(\d+)\]").unwrap();

    for (idx, raw_line) in qasm_str.lines().enumerate() {
        let lineno = idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("include") {
            continue;
        }

        // Version header — accept 2.x, reject everything else loudly rather
        // than silently reading a 3.x program as a mangled 2.0 fragment.
        if line.starts_with("OPENQASM") {
            let caps = version_re.captures(line).ok_or_else(|| {
                format!("malformed OPENQASM version header at line {lineno}: '{line}'")
            })?;
            let major: u32 = caps[1].parse().unwrap_or(0);
            if major != 2 {
                let minor = caps.get(2).map(|m| m.as_str()).unwrap_or("0");
                return Err(format!(
                    "unsupported OPENQASM version {major}.{minor} at line {lineno}: \
                     this importer reads OpenQASM 2.0"
                ));
            }
            continue;
        }

        // Reject multiple statements on one physical line — the line-oriented
        // arms below would mis-merge them (e.g. `h q[0]; x q[1];` would apply H
        // to both qubits and drop the X). QASM is newline-insensitive, so this
        // is a real limitation, surfaced loudly rather than silently mis-parsed.
        let code = line.split("//").next().unwrap_or(line);
        if code.matches(';').count() > 1 {
            return Err(format!(
                "multiple statements on line {lineno}: '{line}' — put each on its own line"
            ));
        }

        // Register declaration
        if let Some(caps) = reg_re.captures(line) {
            let kind_str = &caps[1];
            let name = &caps[2];
            let size: usize = parse_idx(&caps[3], lineno)?;
            match kind_str {
                "qreg" => {
                    circuit.qreg(name, size);
                    qreg_sizes.insert(name.to_string(), size);
                }
                "creg" => {
                    circuit.creg(name, size);
                    creg_sizes.insert(name.to_string(), size);
                }
                _ => {}
            }
            continue;
        }

        // Barrier
        if line.starts_with("barrier") {
            let qubits: Vec<Qubit> = bit_re
                .captures_iter(line)
                .map(|c| {
                    let i = parse_idx(&c[2], lineno)?;
                    check_ref(&qreg_sizes, "qubit", &c[1], i, lineno)?;
                    Ok(Qubit::new(&c[1], i))
                })
                .collect::<Result<Vec<_>, String>>()?;
            circuit.barrier(&qubits);
            continue;
        }

        // Measure
        if line.starts_with("measure") {
            let refs: Vec<(String, usize)> = bit_re
                .captures_iter(line)
                .map(|c| Ok((c[1].to_string(), parse_idx(&c[2], lineno)?)))
                .collect::<Result<Vec<_>, String>>()?;
            if refs.len() < 2 {
                return Err(format!(
                    "unsupported measure at line {lineno}: '{line}' — only the indexed \
                     form `measure q[i] -> c[j];` is supported (register-broadcast \
                     `measure q -> c;` is not)"
                ));
            }
            check_ref(&qreg_sizes, "qubit", &refs[0].0, refs[0].1, lineno)?;
            check_ref(&creg_sizes, "clbit", &refs[1].0, refs[1].1, lineno)?;
            let q = Qubit::new(&refs[0].0, refs[0].1);
            let c = Clbit::new(&refs[1].0, refs[1].1);
            circuit.measure(&q, &c);
            continue;
        }

        // Reset
        if line.starts_with("reset") {
            let Some(caps) = bit_re.captures(line) else {
                return Err(format!(
                    "unsupported reset at line {lineno}: '{line}' — only the indexed \
                     form `reset q[i];` is supported"
                ));
            };
            let i = parse_idx(&caps[2], lineno)?;
            check_ref(&qreg_sizes, "qubit", &caps[1], i, lineno)?;
            let q = Qubit::new(&caps[1], i);
            circuit.reset_qubit(&q);
            continue;
        }

        // Gate instruction
        if let Some(caps) = gate_re.captures(line) {
            let gate_name = &caps[1];
            let params_str = caps.get(2).map(|m| m.as_str());
            let _operands_str = &caps[3];

            let Some(kind) = qasm_to_gate(gate_name) else {
                return Err(format!(
                    "unsupported gate '{gate_name}' at line {lineno}: '{line}'"
                ));
            };
            let params: Vec<f64> = match params_str {
                Some(s) if !s.is_empty() => s
                    .split(',')
                    .map(parse_param)
                    .collect::<Result<Vec<f64>, String>>()?,
                _ => vec![],
            };
            let qubits: Vec<Qubit> = bit_re
                .captures_iter(&caps[3])
                .map(|c| {
                    let i = parse_idx(&c[2], lineno)?;
                    check_ref(&qreg_sizes, "qubit", &c[1], i, lineno)?;
                    Ok(Qubit::new(&c[1], i))
                })
                .collect::<Result<Vec<_>, String>>()?;
            if qubits.is_empty() {
                return Err(format!(
                    "gate '{gate_name}' at line {lineno} has no indexed qubit operands: \
                     '{line}' (register-broadcast forms like `h q;` are not supported)"
                ));
            }
            let gate = GateDef::with_params(kind, params);
            circuit.apply(gate, qubits);
            continue;
        }

        // Nothing matched — a non-empty statement we do not understand. Fail
        // loudly instead of dropping it (the failure mode this importer had).
        return Err(format!("unparsed statement at line {lineno}: '{line}'"));
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

        let qasm_str = to_qasm(&original).expect("unconditioned circuit exports");
        assert!(qasm_str.contains("OPENQASM 2.0;"));
        assert!(qasm_str.contains("qreg q[2];"));
        assert!(qasm_str.contains("h q[0];"));
        assert!(qasm_str.contains("cx q[0], q[1];"));

        let reimported = from_qasm(&qasm_str).unwrap();
        assert_eq!(reimported.n_qubits(), original.n_qubits());
        assert_eq!(reimported.n_clbits(), original.n_clbits());
        assert_eq!(reimported.gate_count(), original.gate_count());
        assert_eq!(reimported.instructions.len(), original.instructions.len());

        // Parameters must survive the round-trip, not just gate counts. The
        // RZ(pi/4) is exactly the shape that used to re-import as 0.0.
        for (orig, back) in original
            .instructions
            .iter()
            .zip(reimported.instructions.iter())
        {
            assert_eq!(orig.gate.params.len(), back.gate.params.len());
            for (po, pb) in orig.gate.params.iter().zip(back.gate.params.iter()) {
                let (vo, vb) = (po.try_as_f64().unwrap(), pb.try_as_f64().unwrap());
                assert!(
                    (vo - vb).abs() <= 1e-12,
                    "parameter changed on round-trip: {vo} -> {vb}"
                );
            }
        }
    }

    #[test]
    fn qasm_param_roundtrip_is_exact() {
        // Every angle format_param can emit, plus decimals and scientific
        // notation, must survive to_qasm -> from_qasm to 1e-12.
        for &theta in &[
            PI / 4.0,
            3.0 * PI / 4.0,
            PI / 2.0,
            PI,
            -PI / 4.0,
            1e-5,
            0.123_456_789,
        ] {
            let circ = CircuitBuilder::new("p", 2, 0)
                .rx(0, theta)
                .ry(1, theta)
                .rz(0, theta)
                .build();
            let back = from_qasm(&to_qasm(&circ).expect("unconditioned circuit exports")).unwrap();
            for (orig, imported) in circ.instructions.iter().zip(back.instructions.iter()) {
                let vo = orig.gate.params[0].try_as_f64().unwrap();
                let vb = imported.gate.params[0].try_as_f64().unwrap();
                assert!(
                    (vo - vb).abs() <= 1e-12,
                    "theta {theta}: {vo} -> {vb} not exact"
                );
            }
        }
    }

    #[test]
    fn to_qasm3_emits_v3_profile() {
        let circ = CircuitBuilder::new("test", 2, 2)
            .h(0)
            .cx(0, 1)
            .rz(1, PI / 4.0)
            .measure_all()
            .build();
        let out = to_qasm3(&circ);

        assert!(out.contains("OPENQASM 3.0;"));
        assert!(out.contains("include \"stdgates.inc\";"));
        assert!(!out.contains("OPENQASM 2.0"));
        // QASM 3 register-declaration form.
        assert!(out.contains("qubit[2] q;"));
        assert!(out.contains("bit[2] c;"));
        assert!(out.contains("h q[0];"));
        assert!(out.contains("cx q[0], q[1];"));
        // Plain-decimal parameter, not a pi-fraction — this is the exact
        // string the omega-parser round-trip test consumes verbatim.
        assert!(out.contains("rz(0.7853981633974483) q[1];"));
        assert!(!out.contains("pi"));
        // QASM 3 assignment measurement form.
        assert!(out.contains("c[0] = measure q[0];"));
    }

    #[test]
    fn from_qasm_rejects_qasm3_header() {
        // The Defect-A reproducer: a QASM3 program must not be accepted as a
        // mangled 2.0 fragment.
        let qasm3 = "OPENQASM 3.0;\n\
                     include \"stdgates.inc\";\n\
                     qubit[2] q;\n\
                     bit[2] c;\n\
                     h q[0];\n\
                     ry(0.25) q[0];\n\
                     c = measure q;\n";
        let err = from_qasm(qasm3).unwrap_err();
        assert!(err.contains("3.0"), "error should name the version: {err}");
        assert!(err.contains("2.0"), "error should mention 2.0: {err}");
    }

    #[test]
    fn from_qasm_errors_on_unknown_gate_and_garbage() {
        let unknown = "OPENQASM 2.0;\nqreg q[1];\nnotagate q[0];\n";
        assert!(from_qasm(unknown).unwrap_err().contains("notagate"));

        let garbage = "OPENQASM 2.0;\nqreg q[1];\nthis is not qasm\n";
        assert!(from_qasm(garbage).unwrap_err().contains("unparsed"));
    }

    #[test]
    fn from_qasm_fails_loud_on_silent_drop_forms() {
        // Register-broadcast measure / reset and gate-on-a-register were
        // previously dropped silently; they must now error.
        assert!(from_qasm("qreg q[2];\ncreg c[2];\nmeasure q -> c;\n")
            .unwrap_err()
            .contains("measure"));
        assert!(from_qasm("qreg q[1];\nreset q;\n")
            .unwrap_err()
            .contains("reset"));
        assert!(from_qasm("qreg q[2];\nh q;\n")
            .unwrap_err()
            .contains("no indexed qubit"));
    }

    #[test]
    fn from_qasm_fails_loud_on_oversized_and_out_of_range_indices() {
        // An index literal too large for usize must return Err, not panic on
        // `.parse().unwrap()` — the same fail-loud contract as every other
        // malformed input.
        let err = from_qasm("qreg q[1];\nh q[99999999999999999999];\n").unwrap_err();
        assert!(err.contains("valid non-negative integer"), "{err}");
        // An in-range integer that exceeds the declared register size is
        // rejected rather than silently imported.
        let err = from_qasm("qreg q[1];\nx q[5];\n").unwrap_err();
        assert!(err.contains("out of range"), "{err}");
        // A reference to a register that was never declared is rejected.
        let err = from_qasm("qreg q[1];\nh r[0];\n").unwrap_err();
        assert!(err.contains("undeclared"), "{err}");
        // The measure clbit is range-checked against its creg too.
        let err = from_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[3];\n").unwrap_err();
        assert!(err.contains("out of range"), "{err}");
        // A valid indexed circuit still imports.
        assert!(from_qasm("qreg q[2];\ncreg c[2];\nh q[0];\nmeasure q[0] -> c[0];\n").is_ok());
    }

    #[test]
    fn from_qasm_rejects_multiple_statements_per_line() {
        let err = from_qasm("qreg q[2];\nh q[0]; x q[1];\n").unwrap_err();
        assert!(err.contains("multiple statements"), "{err}");
    }

    #[test]
    #[allow(non_snake_case)]
    fn from_qasm_accepts_qasm2_builtin_U_and_CX() {
        let circ = from_qasm("qreg q[2];\nU(pi/2, 0, pi) q[0];\nCX q[0], q[1];\n").unwrap();
        assert_eq!(circ.gate_count(), 2);
    }

    #[test]
    fn parse_param_rejects_non_finite() {
        assert!(parse_param("pi/0").is_err());
        assert!(parse_param("inf").is_err());
        assert!(parse_param("nan").is_err());
    }

    #[test]
    fn from_qasm_accepts_headerless_2_0() {
        // Header-less input stays accepted (existing embedder inputs).
        let circ = from_qasm("qreg q[1];\nh q[0];\n").unwrap();
        assert_eq!(circ.n_qubits(), 1);
        assert_eq!(circ.gate_count(), 1);
    }

    #[test]
    fn parse_param_evaluates_left_to_right() {
        assert!((parse_param("1*pi/4").unwrap() - PI / 4.0).abs() < 1e-12);
        assert!((parse_param("-3*pi/4").unwrap() + 3.0 * PI / 4.0).abs() < 1e-12);
        assert!((parse_param("pi/2").unwrap() - PI / 2.0).abs() < 1e-12);
        assert!((parse_param("2*pi").unwrap() - 2.0 * PI).abs() < 1e-12);
        assert!((parse_param("-pi").unwrap() + PI).abs() < 1e-12);
        assert!((parse_param("1.5e-3").unwrap() - 1.5e-3).abs() < 1e-15);
        assert!((parse_param("0").unwrap()).abs() < 1e-15);
        // Unparseable expressions error rather than defaulting to 0.0.
        assert!(parse_param("pi+1").is_err());
        assert!(parse_param("bogus").is_err());
        assert!(parse_param("").is_err());
    }

    #[test]
    fn test_qasm_parametric_gates() {
        let circ = CircuitBuilder::new("param", 1, 0)
            .rx(0, PI)
            .ry(0, PI / 2.0)
            .rz(0, PI / 4.0)
            .build();
        let qasm_str = to_qasm(&circ).expect("unconditioned circuit exports");
        assert!(qasm_str.contains("rx(pi)"));
        assert!(qasm_str.contains("ry(1*pi/2)"));
        assert!(qasm_str.contains("rz(1*pi/4)"));
    }
}
