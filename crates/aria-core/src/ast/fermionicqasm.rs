//! FERMIONICQASM 1.0 import/export.
//!
//! In-house format. It carries no authority of a standard: the name has no
//! prior art, and this module does not interpret gate signs. Those are pinned
//! in `omega-parser`'s `conventions.json` and checked by the ffsim gate.
//! Writing a circuit and reading it back shows the emitter and this reader
//! agree with each other. It is not evidence the gates mean the right thing.
//!
//! The text is written from this AST, not from `CircuitIR`. `tunnel` lowers
//! to three ops; the IR cannot be folded back into the spelling.

use super::expr::ParamExpr;
use super::nodes::*;
use regex::Regex;
use std::collections::{BTreeSet, HashMap};

/// Read one FermionicQASM parameter: a concrete number, or `$name`.
///
/// An expression (`pi`, `theta/2`) has no spelling here. Substituting a
/// number for it would emit a file that parses and computes something else.
fn fmt_param(p: &ParamExpr, gate: &str) -> Result<String, String> {
    match p {
        ParamExpr::Concrete(v) => fmt_number(*v),
        ParamExpr::Symbol(name) => {
            if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(format!(
                    "FermionicQASM `$` names are ASCII alphanumeric or `_`, not `{name}` \
                     (parameter of `{gate}`)"
                ));
            }
            Ok(format!("${name}"))
        }
        other => Err(format!(
            "FermionicQASM `{gate}` parameters are a number or a `$name`, not `{other}`. \
             Bind the expression before exporting — there is no spelling for it, and \
             substituting a number would be a different circuit."
        )),
    }
}

fn fmt_number(v: f64) -> Result<String, String> {
    if v == 0.0 {
        return Ok("0".to_string());
    }
    if !v.is_finite() {
        return Err(format!(
            "FermionicQASM cannot express non-finite parameter {v}"
        ));
    }
    let rendered = format!("{v}");
    if number_spelling(&rendered, v) {
        return Ok(rendered);
    }
    for digits in 1..=16 {
        let raw = format!("{v:.digits$}");
        let trimmed = trim_fraction_zeros(&raw);
        if number_spelling(&trimmed, v) {
            return Ok(trimmed);
        }
    }
    Err(format!(
        "FermionicQASM has no decimal spelling of {v} that parses back to the same value. \
         The grammar is digits with an optional fraction, not scientific notation."
    ))
}

fn trim_fraction_zeros(s: &str) -> String {
    if let Some((head, frac)) = s.split_once('.') {
        let frac = frac.trim_end_matches('0');
        if frac.is_empty() {
            head.to_string()
        } else {
            format!("{head}.{frac}")
        }
    } else {
        s.to_string()
    }
}

fn number_spelling(s: &str, v: f64) -> bool {
    static RE: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"^-?[0-9]+(\.[0-9]+)?$").unwrap());
    if !RE.is_match(s) {
        return false;
    }
    match s.parse::<f64>() {
        Ok(back) => back.to_bits() == v.to_bits(),
        Err(_) => false,
    }
}

struct Placed {
    start: u32,
    spatial: u32,
    wires: u32,
    spin: bool,
}

fn layout(circuit: &Circuit) -> Result<HashMap<String, Placed>, String> {
    let mut map = HashMap::new();
    let mut cursor: u32 = 0;
    for reg in &circuit.registers {
        if reg.kind != RegisterKind::Quantum {
            continue;
        }
        if reg.polarized {
            return Err(format!(
                "`{}` is a polarized photonic register. FermionicQASM modes are not optical \
                 modes — `to_fermionicqasm` will not emit it as `mode`.",
                reg.name
            ));
        }
        let spatial = u32::try_from(reg.size).map_err(|_| {
            format!(
                "mode {}[{}]: size does not fit in a wire index",
                reg.name, reg.size
            )
        })?;
        let wires = if reg.spin {
            spatial.checked_mul(2).ok_or_else(|| {
                format!(
                    "mode {}[{spatial}] spin: 2 * {spatial} does not fit in a wire index",
                    reg.name
                )
            })?
        } else {
            spatial
        };
        if map.contains_key(&reg.name) {
            return Err(format!("duplicate mode register: {}", reg.name));
        }
        map.insert(
            reg.name.clone(),
            Placed {
                start: cursor,
                spatial,
                wires,
                spin: reg.spin,
            },
        );
        cursor = cursor.checked_add(wires).ok_or_else(|| {
            format!(
                "mode {}: adding {wires} wires overflows the register (start {})",
                reg.name, cursor
            )
        })?;
    }
    Ok(map)
}

fn abs_wire(map: &HashMap<String, Placed>, q: &Qubit) -> Result<u32, String> {
    let rec = map.get(&q.register).ok_or_else(|| {
        format!(
            "undefined mode register: {} (in {}[{}])",
            q.register, q.register, q.index
        )
    })?;
    let index = u32::try_from(q.index).map_err(|_| {
        format!(
            "mode index {} out of range for {} ({} wires)",
            q.index, q.register, rec.wires
        )
    })?;
    if index >= rec.wires {
        return Err(format!(
            "mode index {index} out of range for {} ({} wires{})",
            q.register,
            rec.wires,
            if rec.spin {
                "; under spin the index is a wire, 0 .. 2N"
            } else {
                ""
            }
        ));
    }
    Ok(rec.start + index)
}

/// `None` on a spinless wire. `Some(0)` alpha, `Some(1)` beta, on a spin block.
fn species(map: &HashMap<String, Placed>, wire: u32) -> Option<u32> {
    for rec in map.values() {
        if wire >= rec.start && wire < rec.start + rec.wires {
            return if rec.spin && rec.spatial > 0 {
                Some((wire - rec.start) / rec.spatial)
            } else {
                None
            };
        }
    }
    None
}

/// Same two refusals as the lowering, checked before any text is produced.
///
/// Cross-spin is tested first. The adjacent pair across a spin-block boundary
/// has `|p-q| == 1`, so an adjacency check alone would mis-file it as a
/// Z-string refusal. The lowering refuses that pair as unpinned.
fn refuse_hop(gate: &str, a: u32, b: u32, map: &HashMap<String, Placed>) -> Result<(), String> {
    let sa = species(map, a);
    let sb = species(map, b);
    if sa != sb {
        return Err(format!(
            "{gate} on wires {a} and {b}: cross-spin {gate} is unpinned. \
             conventions.json (`unpinned.cross_spin_givens_and_tunnel`) records that \
             ffsim's calls take one Spin and stay inside that species, so no matrix was \
             pinned for a pair in different spin blocks — including the adjacent pair \
             across the block boundary. Emitting the text would hand the lowering a \
             statement it refuses."
        ));
    }
    if a.abs_diff(b) != 1 {
        return Err(format!(
            "{gate} on wires {a} and {b}: a fermionic {gate} between non-adjacent \
             Jordan–Wigner modes needs a Z string over the modes between them, which \
             this adjacent-mode gate does not carry. Emitting it would produce text \
             the lowering refuses."
        ));
    }
    Ok(())
}

fn modes_of(inst: &Instruction) -> String {
    inst.qubits
        .iter()
        .map(|q| format!("{}[{}]", q.register, q.index))
        .collect::<Vec<_>>()
        .join(", ")
}

fn one_param(inst: &Instruction, gate: &str) -> Result<String, String> {
    if inst.gate.params.len() != 1 {
        return Err(format!(
            "`{gate}` takes 1 parameter, got {}",
            inst.gate.params.len()
        ));
    }
    fmt_param(&inst.gate.params[0], gate)
}

fn require_modes(inst: &Instruction, gate: &str, n: usize) -> Result<(), String> {
    if inst.qubits.len() != n {
        return Err(format!(
            "`{gate}` acts on {n} mode(s), got {}",
            inst.qubits.len()
        ));
    }
    Ok(())
}

fn no_condition(inst: &Instruction) -> Result<(), String> {
    if inst.condition.is_some() {
        return Err(
            "FermionicQASM 1.0 has no classical condition. Emitting the gate without \
             it would be a different circuit."
                .to_string(),
        );
    }
    Ok(())
}

/// Convert a Circuit to FERMIONICQASM 1.0.
///
/// Refuses what the lowering refuses — cross-spin `givens`/`tunnel`,
/// non-adjacent `givens`/`tunnel`, `orbrot` (there is no spelling), a `load`
/// after a gate, a doubled `load` — rather than emitting text the reader
/// accepts and the lowering then rejects. A qubit gate is refused, not
/// commented out.
pub fn to_fermionicqasm(circuit: &Circuit) -> Result<String, String> {
    let map = layout(circuit)?;
    let mut lines = vec!["FERMIONICQASM 1.0;".to_string()];

    for reg in &circuit.registers {
        match reg.kind {
            RegisterKind::Quantum => {
                if reg.polarized {
                    return Err(format!(
                        "`{}` is a polarized photonic register, not a fermionic mode",
                        reg.name
                    ));
                }
                let marker = if reg.spin { " spin" } else { "" };
                lines.push(format!("mode {}[{}]{marker};", reg.name, reg.size));
            }
            RegisterKind::Classical => {
                lines.push(format!("creg {}[{}];", reg.name, reg.size));
            }
        }
    }

    let mut seen_gate = false;
    let mut loaded: BTreeSet<(String, usize)> = BTreeSet::new();

    for inst in &circuit.instructions {
        no_condition(inst)?;
        let modes = modes_of(inst);
        let line = match inst.gate.kind {
            GateKind::Load => {
                if inst.qubits.is_empty() {
                    return Err("`load` names no modes".to_string());
                }
                if !inst.gate.params.is_empty() {
                    return Err(format!(
                        "`load` takes no parameters, got {}",
                        inst.gate.params.len()
                    ));
                }
                if seen_gate {
                    return Err(format!(
                        "load {modes} after a gate: load prepares the occupation and must \
                         precede every gate"
                    ));
                }
                for q in &inst.qubits {
                    let _ = abs_wire(&map, q)?;
                    if !loaded.insert((q.register.clone(), q.index)) {
                        return Err(format!(
                            "load {}[{}]: that mode is already loaded",
                            q.register, q.index
                        ));
                    }
                }
                format!("load {modes};")
            }
            GateKind::P => {
                require_modes(inst, "num", 1)?;
                let _ = abs_wire(&map, &inst.qubits[0])?;
                seen_gate = true;
                format!("num({}) {modes};", one_param(inst, "num")?)
            }
            GateKind::CP => {
                require_modes(inst, "numnum", 2)?;
                let w0 = abs_wire(&map, &inst.qubits[0])?;
                let w1 = abs_wire(&map, &inst.qubits[1])?;
                if w0 == w1 {
                    return Err(format!("numnum {modes}: the two modes are the same wire"));
                }
                seen_gate = true;
                format!("numnum({}) {modes};", one_param(inst, "numnum")?)
            }
            GateKind::RBS => {
                require_modes(inst, "givens", 2)?;
                let w0 = abs_wire(&map, &inst.qubits[0])?;
                let w1 = abs_wire(&map, &inst.qubits[1])?;
                refuse_hop("givens", w0, w1, &map)?;
                seen_gate = true;
                format!("givens({}) {modes};", one_param(inst, "givens")?)
            }
            GateKind::Tunnel => {
                require_modes(inst, "tunnel", 2)?;
                let w0 = abs_wire(&map, &inst.qubits[0])?;
                let w1 = abs_wire(&map, &inst.qubits[1])?;
                refuse_hop("tunnel", w0, w1, &map)?;
                seen_gate = true;
                format!("tunnel({}) {modes};", one_param(inst, "tunnel")?)
            }
            GateKind::Measure => {
                if inst.qubits.len() != 1 || inst.clbits.len() != 1 {
                    return Err(format!(
                        "`measure` takes one mode and one classical bit, got {} mode(s) and {} bit(s)",
                        inst.qubits.len(),
                        inst.clbits.len()
                    ));
                }
                let _ = abs_wire(&map, &inst.qubits[0])?;
                let bit = &inst.clbits[0];
                let creg = circuit
                    .registers
                    .iter()
                    .find(|r| r.name == bit.register && r.kind == RegisterKind::Classical)
                    .ok_or_else(|| format!("undefined creg: {}", bit.register))?;
                if bit.index >= creg.size {
                    return Err(format!(
                        "cbit index {} out of range for {} (size {})",
                        bit.index, bit.register, creg.size
                    ));
                }
                format!("measure {modes} -> {}[{}];", bit.register, bit.index)
            }
            other => {
                return Err(format!(
                    "FermionicQASM 1.0 cannot represent `{other:?}` (on {modes}). \
                     Supported: num (P), numnum (CP), givens (RBS), tunnel, load, measure. \
                     `orbrot` is not in 1.0; decompose the orbital rotation to givens before \
                     emission. Previously an unspelled gate would have had nowhere to go \
                     except a comment, which re-imports as a circuit missing the operation."
                ));
            }
        };
        lines.push(line);
    }

    Ok(lines.join("\n") + "\n")
}

/// Parse FERMIONICQASM 1.0 into a Circuit.
///
/// Syntax only. Cross-spin and non-adjacent `givens`/`tunnel` parse — they
/// are grammatical — and [`to_fermionicqasm`] refuses to write them back,
/// which is the same split as the pest grammar and the lowering.
pub fn from_fermionicqasm(src: &str) -> Result<Circuit, String> {
    let stmts = split_statements(src)?;
    let mut circuit = Circuit::new("fermionic");
    let mut seen_header = false;
    // name -> wire count, so an index past the register is refused here
    // rather than stored and re-emitted.
    let mut modes: HashMap<String, usize> = HashMap::new();
    let mut cregs: HashMap<String, usize> = HashMap::new();

    for (text, line) in stmts {
        let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if !seen_header {
            let version = header_version(&flat).ok_or_else(|| {
                format!("line {line}: a FermionicQASM file must open with `FERMIONICQASM 1.0;`, got `{flat}`")
            })?;
            if version != "1.0" {
                return Err(format!(
                    "line {line}: unsupported FERMIONICQASM version {version}; this reader accepts FERMIONICQASM 1.0 only"
                ));
            }
            seen_header = true;
            continue;
        }
        if header_version(&flat).is_some() {
            return Err(format!("line {line}: a second FERMIONICQASM header"));
        }
        if let Some(rest) = flat.strip_prefix("mode ") {
            parse_mode(&mut circuit, &mut modes, rest, line)?;
        } else if let Some(rest) = flat.strip_prefix("creg ") {
            parse_creg(&mut circuit, &mut cregs, rest, line)?;
        } else if let Some(rest) = flat.strip_prefix("load ") {
            let qs = parse_mode_list(&modes, rest, line)?;
            circuit.apply(GateDef::new(GateKind::Load), qs);
        } else if let Some(rest) = flat.strip_prefix("measure ") {
            parse_measure(&mut circuit, &modes, &cregs, rest, line)?;
        } else {
            parse_gate(&mut circuit, &modes, &flat, line)?;
        }
    }
    if !seen_header {
        return Err("empty input: a FermionicQASM file must open with `FERMIONICQASM 1.0;`".into());
    }
    Ok(circuit)
}

fn header_version(flat: &str) -> Option<&str> {
    let rest = flat.strip_prefix("FERMIONICQASM ")?;
    if rest.is_empty() {
        return None;
    }
    Some(rest)
}

fn parse_mode(
    circuit: &mut Circuit,
    modes: &mut HashMap<String, usize>,
    rest: &str,
    line: usize,
) -> Result<(), String> {
    static RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(r"^([A-Za-z_][A-Za-z0-9_]*)\[([0-9]+)\]( spin)?$").unwrap()
    });
    let caps = RE.captures(rest).ok_or_else(|| {
        format!("line {line}: expected `mode name[N];` or `mode name[N] spin;`, got `{rest}`")
    })?;
    let name = caps.get(1).unwrap().as_str();
    let spatial: usize = caps
        .get(2)
        .unwrap()
        .as_str()
        .parse()
        .map_err(|_| format!("line {line}: mode size does not fit"))?;
    let spin = caps.get(3).is_some();
    if modes.contains_key(name) || circuit.registers.iter().any(|r| r.name == name) {
        return Err(format!("line {line}: duplicate mode register: {name}"));
    }
    let wires = if spin {
        spatial.checked_mul(2).ok_or_else(|| {
            format!("line {line}: mode {name}[{spatial}] spin: 2 * {spatial} does not fit")
        })?
    } else {
        spatial
    };
    modes.insert(name.to_string(), wires);
    circuit.mode_reg(name, spatial, spin);
    Ok(())
}

fn parse_creg(
    circuit: &mut Circuit,
    cregs: &mut HashMap<String, usize>,
    rest: &str,
    line: usize,
) -> Result<(), String> {
    static RE: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"^([A-Za-z_][A-Za-z0-9_]*)\[([0-9]+)\]$").unwrap());
    let caps = RE
        .captures(rest)
        .ok_or_else(|| format!("line {line}: expected `creg name[N];`, got `{rest}`"))?;
    let name = caps.get(1).unwrap().as_str();
    let size: usize = caps
        .get(2)
        .unwrap()
        .as_str()
        .parse()
        .map_err(|_| format!("line {line}: creg size does not fit"))?;
    if cregs.contains_key(name) || circuit.registers.iter().any(|r| r.name == name) {
        return Err(format!("line {line}: duplicate creg: {name}"));
    }
    cregs.insert(name.to_string(), size);
    circuit.creg(name, size);
    Ok(())
}

fn parse_mode_list(
    modes: &HashMap<String, usize>,
    rest: &str,
    line: usize,
) -> Result<Vec<Qubit>, String> {
    if rest.is_empty() {
        return Err(format!("line {line}: mode list is empty"));
    }
    let mut out = Vec::new();
    for part in rest.split(',') {
        let part = part.trim();
        let (name, index) = mode_ref(part)
            .ok_or_else(|| format!("line {line}: expected `name[index]`, got `{part}`"))?;
        let wires = modes
            .get(name)
            .ok_or_else(|| format!("line {line}: undefined mode register: {name}"))?;
        if index >= *wires {
            return Err(format!(
                "line {line}: mode index {index} out of range for {name} ({wires} wires)"
            ));
        }
        out.push(Qubit::new(name, index));
    }
    Ok(out)
}

fn mode_ref(part: &str) -> Option<(&str, usize)> {
    let (name, rest) = part.split_once('[')?;
    let index = rest.strip_suffix(']')?;
    if name.is_empty() || !index.chars().all(|c| c.is_ascii_digit()) || index.is_empty() {
        return None;
    }
    if !name
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
    {
        return None;
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    Some((name, index.parse().ok()?))
}

fn parse_measure(
    circuit: &mut Circuit,
    modes: &HashMap<String, usize>,
    cregs: &HashMap<String, usize>,
    rest: &str,
    line: usize,
) -> Result<(), String> {
    let (left, right) = rest
        .split_once("->")
        .ok_or_else(|| format!("line {line}: expected `measure m[i] -> c[j];`, got `{rest}`"))?;
    let qs = parse_mode_list(modes, left.trim(), line)?;
    if qs.len() != 1 {
        return Err(format!(
            "line {line}: `measure` takes one mode, got {}",
            qs.len()
        ));
    }
    let (cname, cindex) = mode_ref(right.trim()).ok_or_else(|| {
        format!(
            "line {line}: expected a classical bit `name[index]`, got `{}`",
            right.trim()
        )
    })?;
    let size = cregs
        .get(cname)
        .ok_or_else(|| format!("line {line}: undefined creg: {cname}"))?;
    if cindex >= *size {
        return Err(format!(
            "line {line}: cbit index {cindex} out of range for {cname} (size {size})"
        ));
    }
    circuit.measure(&qs[0], &Clbit::new(cname, cindex));
    Ok(())
}

fn parse_gate(
    circuit: &mut Circuit,
    modes: &HashMap<String, usize>,
    flat: &str,
    line: usize,
) -> Result<(), String> {
    static RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\(([^)]*)\))?\s+(.+)$").unwrap()
    });
    let caps = RE
        .captures(flat)
        .ok_or_else(|| format!("line {line}: not a FermionicQASM statement: `{flat}`"))?;
    let name = caps.get(1).unwrap().as_str();
    if name == "orbrot" {
        return Err(
            "orbrot is not in FermionicQASM 1.0; decompose the orbital rotation \
             to givens before emission"
                .to_string(),
        );
    }
    let params = match caps.get(2) {
        None => Vec::new(),
        Some(m) => parse_params(m.as_str(), line)?,
    };
    let qs = parse_mode_list(modes, caps.get(3).unwrap().as_str().trim(), line)?;
    let kind = match name {
        "num" => GateKind::P,
        "numnum" => GateKind::CP,
        "givens" => GateKind::RBS,
        "tunnel" => GateKind::Tunnel,
        other => return Err(format!("line {line}: unknown fermionic gate: {other}")),
    };
    let need = match kind {
        GateKind::P => 1,
        GateKind::CP | GateKind::RBS | GateKind::Tunnel => 2,
        _ => unreachable!(),
    };
    if qs.len() != need {
        return Err(format!(
            "line {line}: `{name}` acts on {need} mode(s), got {}",
            qs.len()
        ));
    }
    if params.len() != 1 {
        return Err(format!(
            "line {line}: `{name}` takes 1 parameter, got {}",
            params.len()
        ));
    }
    circuit.apply(GateDef::with_exprs(kind, params), qs);
    Ok(())
}

fn parse_params(list: &str, line: usize) -> Result<Vec<ParamExpr>, String> {
    let list = list.trim();
    if list.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for part in list.split(',') {
        let part = part.trim();
        if let Some(name) = part.strip_prefix('$') {
            if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(format!("line {line}: bad symbol `{part}`"));
            }
            out.push(ParamExpr::symbol(name));
        } else if number_token(part) {
            let v: f64 = part
                .parse()
                .map_err(|_| format!("line {line}: bad number `{part}`"))?;
            out.push(ParamExpr::Concrete(v));
        } else {
            return Err(format!(
                "line {line}: parameter `{part}` is not a number or a `$name`"
            ));
        }
    }
    Ok(out)
}

fn number_token(s: &str) -> bool {
    static RE: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"^-?[0-9]+(\.[0-9]+)?$").unwrap());
    RE.is_match(s)
}

/// Statements split on `;`, with `//` comments removed. A trailing fragment
/// with no semicolon is an error: dropping it would read a truncated file
/// as a shorter one.
fn split_statements(src: &str) -> Result<Vec<(String, usize)>, String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut line = 1usize;
    let mut stmt_line = 1usize;
    let mut in_stmt = false;
    let mut chars = src.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '/' && chars.peek() == Some(&'/') {
            chars.next();
            for d in chars.by_ref() {
                if d == '\n' {
                    line += 1;
                    break;
                }
            }
            continue;
        }
        if c == '\n' {
            line += 1;
            if in_stmt {
                buf.push('\n');
            }
            continue;
        }
        if c == ';' {
            let text = buf.trim().to_string();
            if !text.is_empty() {
                out.push((text, stmt_line));
            }
            buf.clear();
            in_stmt = false;
            continue;
        }
        if !in_stmt && c.is_whitespace() {
            continue;
        }
        if !in_stmt {
            in_stmt = true;
            stmt_line = line;
        }
        buf.push(c);
    }
    if !buf.trim().is_empty() {
        return Err(format!(
            "line {stmt_line}: statement has no trailing `;`: `{}`",
            buf.trim()
        ));
    }
    Ok(out)
}
