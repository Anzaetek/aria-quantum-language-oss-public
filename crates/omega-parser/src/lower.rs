use std::collections::HashMap;

use omega_core::circuit::*;

use crate::ast::*;

/// Lower a QASM2 AST to CircuitIR.
pub fn lower_qasm2(prog: &Qasm2Program) -> Result<CircuitIR, String> {
    let mut ctx = LowerCtx::new(CircuitType::GateBased);

    for stmt in &prog.statements {
        ctx.lower_qasm2_stmt(stmt)?;
    }

    Ok(ctx.finish())
}

/// Lower an OPTICQASM AST to CircuitIR.
pub fn lower_opticqasm(prog: &OpticQasmProgram) -> Result<CircuitIR, String> {
    let mut ctx = LowerCtx::new(CircuitType::Photonic);

    for stmt in &prog.statements {
        ctx.lower_opticqasm_stmt(stmt)?;
    }

    Ok(ctx.finish())
}

/// Convenience: parse + lower in one step.
pub fn lower_to_ir(source: &str) -> Result<CircuitIR, String> {
    // Skip leading comments and whitespace
    let trimmed = source
        .lines()
        .find(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with("//")
        })
        .unwrap_or("")
        .trim();
    if trimmed.starts_with("OPENQASM") {
        let ast = crate::qasm2::parse_qasm2(source)?;
        lower_qasm2(&ast)
    } else if trimmed.starts_with("OPTICQASM") {
        let ast = crate::opticqasm::parse_opticqasm(source)?;
        lower_opticqasm(&ast)
    } else {
        Err("unknown circuit format: expected OPENQASM or OPTICQASM header".to_string())
    }
}

struct LowerCtx {
    circuit_type: CircuitType,
    /// qreg name -> (start_qubit_index, size)
    qregs: HashMap<String, (u32, u32)>,
    /// creg name -> (start_bit_index, size)
    cregs: HashMap<String, (u32, u32)>,
    /// Symbol name -> SymbolId
    symbols: HashMap<String, SymbolId>,
    next_symbol_id: SymbolId,
    /// All symbol id -> name
    symbol_names: HashMap<SymbolId, String>,
    num_qubits: u32,
    num_classical_bits: u32,
    ops: Vec<GateOp>,
    /// Gate definitions from QASM (name -> params, qubit_names, body)
    gate_defs: HashMap<String, GateDef>,
    /// OPTICQASM registers declared `pol`. Mode refs into these name a
    /// **spatial** mode, which expands to optical modes `2s` (H) and `2s+1` (V).
    polarized_regs: std::collections::HashSet<String>,
}

impl LowerCtx {
    fn new(circuit_type: CircuitType) -> Self {
        Self {
            circuit_type,
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            symbols: HashMap::new(),
            next_symbol_id: 0,
            symbol_names: HashMap::new(),
            polarized_regs: std::collections::HashSet::new(),
            num_qubits: 0,
            num_classical_bits: 0,
            ops: Vec::new(),
            gate_defs: HashMap::new(),
        }
    }

    fn finish(self) -> CircuitIR {
        CircuitIR {
            num_qubits: self.num_qubits,
            num_classical_bits: self.num_classical_bits,
            ops: self.ops,
            circuit_type: self.circuit_type.clone(),
            symbols: self.symbol_names,
            custom_gates: HashMap::new(),
        }
    }

    fn get_or_create_symbol(&mut self, name: &str) -> SymbolId {
        if let Some(&id) = self.symbols.get(name) {
            return id;
        }
        let id = self.next_symbol_id;
        self.next_symbol_id += 1;
        self.symbols.insert(name.to_string(), id);
        self.symbol_names.insert(id, name.to_string());
        id
    }

    fn resolve_qubit(&self, qref: &QubitRef) -> Result<Vec<Qubit>, String> {
        match qref {
            QubitRef::Indexed { reg, index } => {
                let (start, size) = self
                    .qregs
                    .get(reg)
                    .ok_or_else(|| format!("undefined qreg: {}", reg))?;
                if *index >= *size {
                    return Err(format!(
                        "qubit index {} out of range for qreg {} (size {})",
                        index, reg, size
                    ));
                }
                Ok(vec![Qubit(start + index)])
            }
            QubitRef::Register(reg) => {
                let (start, size) = self
                    .qregs
                    .get(reg)
                    .ok_or_else(|| format!("undefined qreg: {}", reg))?;
                Ok((0..*size).map(|i| Qubit(start + i)).collect())
            }
        }
    }

    fn resolve_cbit(&self, cref: &CbitRef) -> Result<Vec<u32>, String> {
        match cref {
            CbitRef::Indexed { reg, index } => {
                let (start, size) = self
                    .cregs
                    .get(reg)
                    .ok_or_else(|| format!("undefined creg: {}", reg))?;
                if *index >= *size {
                    return Err(format!("cbit index {} out of range", index));
                }
                Ok(vec![start + index])
            }
            CbitRef::Register(reg) => {
                let (start, size) = self
                    .cregs
                    .get(reg)
                    .ok_or_else(|| format!("undefined creg: {}", reg))?;
                Ok((0..*size).map(|i| start + i).collect())
            }
        }
    }

    fn lower_expr(
        &mut self,
        expr: &Expr,
        local_params: &HashMap<String, usize>,
        param_values: &[ParamExpr],
    ) -> Result<ParamExpr, String> {
        match expr {
            Expr::Num(v) => Ok(ParamExpr::Concrete(*v)),
            Expr::Pi => Ok(ParamExpr::Concrete(std::f64::consts::PI)),
            Expr::Ident(name) => {
                // Check if it's a local parameter from a gate definition
                if let Some(&idx) = local_params.get(name) {
                    return Ok(param_values[idx].clone());
                }
                // Otherwise it's a free symbol
                let id = self.get_or_create_symbol(name);
                Ok(ParamExpr::Symbol(id))
            }
            Expr::Neg(inner) => {
                let e = self.lower_expr(inner, local_params, param_values)?;
                if let ParamExpr::Concrete(v) = e {
                    Ok(ParamExpr::Concrete(-v))
                } else {
                    Ok(ParamExpr::Negate(Box::new(e)))
                }
            }
            Expr::BinOp(lhs, op, rhs) => {
                let l = self.lower_expr(lhs, local_params, param_values)?;
                let r = self.lower_expr(rhs, local_params, param_values)?;
                match op {
                    BinOp::Add => match (&l, &r) {
                        (ParamExpr::Concrete(a), ParamExpr::Concrete(b)) => {
                            Ok(ParamExpr::Concrete(a + b))
                        }
                        _ => Ok(ParamExpr::Add(Box::new(l), Box::new(r))),
                    },
                    BinOp::Sub => match (&l, &r) {
                        (ParamExpr::Concrete(a), ParamExpr::Concrete(b)) => {
                            Ok(ParamExpr::Concrete(a - b))
                        }
                        _ => Ok(ParamExpr::Add(
                            Box::new(l),
                            Box::new(ParamExpr::Negate(Box::new(r))),
                        )),
                    },
                    BinOp::Mul => match (&l, &r) {
                        (ParamExpr::Concrete(a), ParamExpr::Concrete(b)) => {
                            Ok(ParamExpr::Concrete(a * b))
                        }
                        _ => Ok(ParamExpr::Mul(Box::new(l), Box::new(r))),
                    },
                    BinOp::Div => match (&l, &r) {
                        (ParamExpr::Concrete(a), ParamExpr::Concrete(b)) => {
                            Ok(ParamExpr::Concrete(a / b))
                        }
                        _ => Err("non-constant division not yet supported".to_string()),
                    },
                }
            }
            Expr::FnCall(fname, arg) => {
                let a = self.lower_expr(arg, local_params, param_values)?;
                // Evaluate eagerly if concrete
                match a {
                    ParamExpr::Concrete(v) => {
                        let result = match fname.as_str() {
                            "sin" => v.sin(),
                            "cos" => v.cos(),
                            "tan" => v.tan(),
                            "exp" => v.exp(),
                            "ln" => v.ln(),
                            "sqrt" => v.sqrt(),
                            _ => return Err(format!("unknown function: {}", fname)),
                        };
                        Ok(ParamExpr::Concrete(result))
                    }
                    _ => Err(format!(
                        "function {} on symbolic args not yet supported",
                        fname
                    )),
                }
            }
        }
    }

    fn lower_qasm2_stmt(&mut self, stmt: &Qasm2Stmt) -> Result<(), String> {
        match stmt {
            Qasm2Stmt::Include(_) => {
                // We handle qelib1.inc gates natively
                Ok(())
            }
            Qasm2Stmt::QregDecl { name, size } => {
                let start = self.num_qubits;
                self.qregs.insert(name.clone(), (start, *size));
                self.num_qubits += size;
                Ok(())
            }
            Qasm2Stmt::CregDecl { name, size } => {
                let start = self.num_classical_bits;
                self.cregs.insert(name.clone(), (start, *size));
                self.num_classical_bits += size;
                Ok(())
            }
            Qasm2Stmt::GateDef(def) => {
                self.gate_defs.insert(def.name.clone(), def.clone());
                Ok(())
            }
            Qasm2Stmt::GateApp(app) => self.lower_gate_app(app, &HashMap::new(), &[]),
            Qasm2Stmt::Measure { qubit, cbit } => {
                let qubits = self.resolve_qubit(qubit)?;
                let cbits = self.resolve_cbit(cbit)?;
                for (q, c) in qubits.into_iter().zip(cbits) {
                    self.ops.push(GateOp {
                        gate: GateKind::Measure,
                        qubits: smallvec::smallvec![q],
                        params: smallvec::smallvec![],
                        classical_bit: Some(c),
                        condition: None,
                    });
                }
                Ok(())
            }
            Qasm2Stmt::Barrier(qrefs) => {
                let mut qubits = smallvec::SmallVec::new();
                for qr in qrefs {
                    for q in self.resolve_qubit(qr)? {
                        qubits.push(q);
                    }
                }
                self.ops.push(GateOp {
                    gate: GateKind::Barrier,
                    qubits,
                    params: smallvec::smallvec![],
                    classical_bit: None,
                    condition: None,
                });
                Ok(())
            }
            Qasm2Stmt::If { creg, value, then } => {
                // Emit the inner gate(s) and patch each one with a classical
                // condition keyed on the creg's LSB. The executor checks
                // `classical_bits[cbit] != expected`, so this is exact for
                // single-bit cregs (the common case) and best-effort for
                // multi-bit cregs (only the LSB is consulted — see
                // verify-qiskit/KNOWN_GAPS.md).
                let &(start_bit, size) = self
                    .cregs
                    .get(creg)
                    .ok_or_else(|| format!("undefined creg in if: {}", creg))?;
                let before = self.ops.len();
                if let Qasm2Stmt::GateApp(app) = then.as_ref() {
                    self.lower_gate_app(app, &HashMap::new(), &[])?;
                }
                for op in &mut self.ops[before..] {
                    op.condition = Some((start_bit, size, *value));
                }
                Ok(())
            }
            Qasm2Stmt::Reset(qr) => {
                let qubits = self.resolve_qubit(qr)?;
                for q in qubits {
                    self.ops.push(GateOp {
                        gate: GateKind::Reset,
                        qubits: smallvec::SmallVec::from_slice(&[q]),
                        params: smallvec::SmallVec::new(),
                        classical_bit: None,
                        condition: None,
                    });
                }
                Ok(())
            }
        }
    }

    fn lower_gate_app(
        &mut self,
        app: &GateApp,
        local_params: &HashMap<String, usize>,
        param_values: &[ParamExpr],
    ) -> Result<(), String> {
        // QASM 3 modifiers: collapse to (inv_count_parity, total_pow).
        // Modifier list is outermost-first so we iterate forwards but the
        // semantic is "apply right-to-left". For pow we multiply
        // exponents; for inv we toggle a parity bit. Negative total
        // exponent means "apply inv |n| times".
        let mut inv_parity = false;
        let mut total_pow: i32 = 1;
        for m in &app.modifiers {
            match m {
                GateModifier::Inv => inv_parity = !inv_parity,
                GateModifier::Pow(n) => {
                    if *n == 0 {
                        // pow(0) @ U = I, so the entire application is a no-op.
                        return Ok(());
                    }
                    if *n < 0 {
                        inv_parity = !inv_parity;
                    }
                    total_pow = total_pow.saturating_mul(n.unsigned_abs() as i32);
                }
            }
        }
        // Lower parameter expressions
        let mut params: smallvec::SmallVec<[ParamExpr; 3]> = smallvec::SmallVec::new();
        for p in &app.params {
            params.push(self.lower_expr(p, local_params, param_values)?);
        }

        // Modifiers on user-defined gates would require expanding the
        // body, then applying inv (reverse + per-op inverse) / pow
        // (repeat) — not yet wired. Fail-fast with a clear error so
        // operators don't get silently-wrong IR when they reach for
        // `inv @ mygate(...)`.
        if (inv_parity || total_pow != 1) && self.gate_defs.contains_key(&app.name) {
            return Err(format!(
                "inv @ / pow @ modifiers on user-defined gate '{}' not yet supported; \
                 either expand the gate inline or define an explicit inverse",
                app.name
            ));
        }

        // Check if it's a user-defined gate
        if let Some(def) = self.gate_defs.get(&app.name).cloned() {
            // Build local param map for the gate body
            let mut inner_params = HashMap::new();
            for (i, pname) in def.params.iter().enumerate() {
                inner_params.insert(pname.clone(), i);
            }

            // Resolve qubit arguments
            let mut qubit_args = Vec::new();
            for qr in &app.qubits {
                let qs = self.resolve_qubit(qr)?;
                qubit_args.push(qs);
            }

            // Map gate's qubit names to actual qubits
            let mut qubit_map: HashMap<String, Qubit> = HashMap::new();
            for (i, qname) in def.qubits.iter().enumerate() {
                if i < qubit_args.len() && !qubit_args[i].is_empty() {
                    qubit_map.insert(qname.clone(), qubit_args[i][0]);
                }
            }

            // Expand body
            for body_app in &def.body {
                let mut body_qubits = Vec::new();
                for qr in &body_app.qubits {
                    match qr {
                        QubitRef::Register(name) | QubitRef::Indexed { reg: name, .. } => {
                            if let Some(&q) = qubit_map.get(name) {
                                body_qubits.push(q);
                            } else {
                                return Err(format!("undefined qubit in gate body: {}", name));
                            }
                        }
                    }
                }

                let mut body_params = smallvec::SmallVec::new();
                for p in &body_app.params {
                    body_params.push(self.lower_expr(p, &inner_params, &params)?);
                }

                // The `cp`/`cu1` widening and the arity check both live at
                // the top-level path below, and this path used to reach neither.
                // `gate mycp(t) a,b { cp(t) a,b; }` therefore produced a CU3 with
                // ONE parameter, and the statevector backend indexed
                // `resolved[2]` and PANICKED. Measured, along with plain
                // `cu3(0.7) a,b;` and `u3(0.7) a;` at top level — three ways for
                // a malformed QASM2 file to crash the process instead of being
                // refused.
                let gate = name_to_gate(&body_app.name)?;
                let body_params = widen_cp_params(&body_app.name, body_params)?;
                let body_qubits: smallvec::SmallVec<[Qubit; 3]> =
                    body_qubits.into_iter().collect();
                check_gate_arity(&body_app.name, &gate, body_params.len(), body_qubits.len())?;
                self.ops.push(GateOp {
                    gate,
                    qubits: body_qubits,
                    params: body_params,
                    classical_bit: None,
                    condition: None,
                });
            }
            return Ok(());
        }

        // Standard gate
        let gate = name_to_gate(&app.name)?;

        // `cp(λ)` / `cu1(λ)` are qelib1's controlled-phase gate, and omega has
        // no CP variant — but `CU3(0, 0, λ)` IS it, exactly: `U3(0, 0, λ) =
        // diag(1, e^{iλ}) = U1(λ) = P(λ)`, with no phase slack. So the name
        // resolves to CU3 and the single angle is widened to three here.
        //
        // Without this, `aria-core` exported `cp` (its own emitter's spelling
        // for `GateKind::CP`) and this parser rejected it with "unknown gate:
        // cp" — a circuit whose QASM is valid qelib1, which Qiskit accepts,
        // and which this repository could not read back. The round trip was
        // broken while both ends looked healthy.
        //
        // Synthesised BEFORE the `inv @` handling below so the inverse path
        // sees an ordinary CU3: `U3(θ,φ,λ)† = U3(−θ,−λ,−φ)`, which at
        // `(0,0,λ)` gives `U3(0,−λ,0) = diag(1, e^{−iλ})` — correct.
        let params = widen_cp_params(&app.name, params)?;

        // Resolve qubits
        let mut qubits: smallvec::SmallVec<[Qubit; 3]> = smallvec::SmallVec::new();
        for qr in &app.qubits {
            let qs = self.resolve_qubit(qr)?;
            for q in qs {
                qubits.push(q);
            }
        }

        // `rxx(θ)` / `rzz(θ)` are DECOMPOSED here into ops that already exist
        // rather than becoming new `GateKind` variants.
        //
        // Why decompose: these are interchange spellings, not new capabilities.
        // The goal is that a QASM2 file round-trips, not that the engine gains a
        // primitive. New variants would cost seven backends and re-open the CUDA
        // non-exhaustive-match problem for a gate nobody asked to accelerate.
        // The parser already decomposes this way for polarization
        // (`lower_half_wave_plate`), so there is precedent and no new dispatch
        // surface.
        //
        // Verified against Qiskit 2.5.1's own gate matrices via `Operator(qc).data`
        // at θ = 0.7:
        //
        //   RZZ(θ) = cx q0,q1 ; rz(θ) q1 ; cx q0,q1        max|Δ| = 0.000e+00
        //   RXX(θ) = h⊗h ; RZZ(θ) ; h⊗h                    max|Δ| = 3.331e-16
        //
        // Before this, `aria-core` emitted `rxx`/`rzz` and this parser answered
        // "unknown gate" — a file Qiskit's legacy loader reads and we could not.
        //
        // `ryy` is deliberately NOT handled. Measured on qiskit 2.5.1: the
        // strict `qasm2.loads` rejects all of rxx/ryy/rzz/cp, and the LEGACY
        // `from_qasm_str` accepts rxx/rzz/cp but rejects `ryy` — "'ryy' is not
        // defined in this scope". So no Qiskit loader reads `ryy`. Teaching this
        // parser to read it would make the round trip work for us alone while
        // every other toolchain still could not load the file, which is worse
        // than the status quo because it would look fixed. What to do about
        // EMITTING it is a separate decision (see PLAN-EXPORT-INTEGRITY.md).
        //
        // θ appears EXACTLY ONCE in each decomposition (the `h` conjugators are
        // constants), so a symbolic angle differentiates correctly: `adjoint.rs`
        // accumulates per-symbol contributions with `+=` and there is nothing
        // here to double-count.
        if matches!(app.name.as_str(), "rxx" | "rzz") {
            if params.len() != 1 {
                return Err(format!(
                    "{} expects 1 parameter, got {}",
                    app.name,
                    params.len()
                ));
            }
            if qubits.len() != 2 {
                return Err(format!(
                    "{} acts on 2 qubits, got {}",
                    app.name,
                    qubits.len()
                ));
            }
            // `inv @ rzz(θ) == rzz(−θ)`: the decomposition is parameterised by θ,
            // so negating the ANGLE is equivalent to reversing the sequence.
            let theta = if inv_parity {
                ParamExpr::Negate(Box::new(params[0].clone()))
            } else {
                params[0].clone()
            };
            let (a, b) = (qubits[0].clone(), qubits[1].clone());
            let mut push = |gate: GateKind, qs: &[Qubit], ps: &[ParamExpr]| {
                self.ops.push(GateOp {
                    gate,
                    qubits: qs.iter().cloned().collect(),
                    params: ps.iter().cloned().collect(),
                    classical_bit: None,
                    condition: None,
                });
            };
            for _ in 0..total_pow {
                if app.name == "rxx" {
                    push(GateKind::H, &[a.clone()], &[]);
                    push(GateKind::H, &[b.clone()], &[]);
                }
                push(GateKind::CX, &[a.clone(), b.clone()], &[]);
                push(GateKind::Rz, &[b.clone()], std::slice::from_ref(&theta));
                push(GateKind::CX, &[a.clone(), b.clone()], &[]);
                if app.name == "rxx" {
                    push(GateKind::H, &[a.clone()], &[]);
                    push(GateKind::H, &[b.clone()], &[]);
                }
            }
            return Ok(());
        }

        check_gate_arity(&app.name, &gate, params.len(), qubits.len())?;

        // Resolve modifiers. With no modifiers this is the original
        // single-op emit (total_pow=1, inv_parity=false).
        let (final_gate, final_params) = if inv_parity {
            invert_gate_with_params(gate, &params)?
        } else {
            (gate, params.clone())
        };
        for _ in 0..total_pow {
            self.ops.push(GateOp {
                gate: final_gate.clone(),
                qubits: qubits.clone(),
                params: final_params.clone(),
                classical_bit: None,
                condition: None,
            });
        }

        Ok(())
    }

    /// Optical mode indices `(H, V)` for a **spatial** mode reference.
    ///
    /// Refuses on a non-`pol` register rather than guessing: silently treating
    /// `q[0]` as a spatial mode in an unpolarized register would apply a wave
    /// plate to two unrelated optical modes and produce a plausible wrong
    /// answer, which is the worst available outcome.
    fn polarization_submodes(&self, m: &ModeRef, gate: &str) -> Result<(u32, u32), String> {
        if !self.polarized_regs.contains(&m.reg) {
            return Err(format!(
                "`{gate}` needs a polarized register: declare `photon {}[N] pol;` \
                 (a polarization element has no meaning on modes that carry no \
                 polarization)",
                m.reg
            ));
        }
        let (start, optical) = self
            .qregs
            .get(&m.reg)
            .ok_or_else(|| format!("undefined photon register: {}", m.reg))?;
        let spatial = optical / 2;
        if m.index >= spatial {
            return Err(format!(
                "spatial mode index {} out of range ({} declared)",
                m.index, spatial
            ));
        }
        Ok((start + 2 * m.index, start + 2 * m.index + 1))
    }

    fn push_photonic(&mut self, gate: GateKind, qubits: &[u32], params: &[f64]) {
        self.ops.push(GateOp {
            gate,
            qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
            params: params.iter().map(|p| ParamExpr::Concrete(*p)).collect(),
            classical_bit: None,
            condition: None,
        });
    }

    /// Half-wave plate on one spatial mode's `(H, V)` pair.
    ///
    /// Aria adopts **Perceval's** convention verbatim (FIXES_PLAN.md I1b):
    ///
    /// ```text
    ///   HWP(θ) = i · BSrx(2θ, 0) · PS(π on V)
    /// ```
    ///
    /// The global `i` is deliberate and is NOT droppable. A wave plate acts on
    /// a *subset* of a larger interferometer's modes, so a global factor on
    /// that 2×2 block becomes a **relative** phase between interfering paths.
    /// Dropping it changes single-photon output probabilities by up to 0.413 in
    /// a 4-mode MZI — measured, not estimated. The textbook i-less matrix has
    /// `det = −1` against Perceval's `+1`, so the two are not the same operator
    /// and no amount of "it's only a global phase" makes them agree here.
    ///
    /// Ops compose as `U -> Op·U`, so the FIRST op pushed is the RIGHTMOST
    /// factor: `PS(π on V)` goes first.
    fn lower_half_wave_plate(&mut self, app: &OpticGateApp) -> Result<(), String> {
        if app.modes.len() != 1 {
            return Err(format!(
                "hwp acts on 1 spatial mode; got {}",
                app.modes.len()
            ));
        }
        let theta = match app.params.as_slice() {
            [OpticParam::Num(t)] => *t,
            [OpticParam::Symbol(_)] => {
                return Err(
                    "hwp does not accept a symbolic angle yet: the expansion into \
                     phase shifters is built at lowering time and would need the \
                     bound value"
                        .to_string(),
                )
            }
            other => return Err(format!("hwp takes (theta); got {} params", other.len())),
        };

        let (h, v) = self.polarization_submodes(&app.modes[0], "hwp")?;

        self.push_photonic(GateKind::PhaseShifter, &[v], &[std::f64::consts::PI]);
        self.push_photonic(GateKind::BeamSplitterRx, &[h, v], &[2.0 * theta, 0.0]);
        // The global `i`, as a phase of π/2 on BOTH sub-modes.
        self.push_photonic(GateKind::PhaseShifter, &[h], &[std::f64::consts::FRAC_PI_2]);
        self.push_photonic(GateKind::PhaseShifter, &[v], &[std::f64::consts::FRAC_PI_2]);
        Ok(())
    }

    /// Polarizing beam splitter across two spatial modes.
    ///
    /// **Swaps H between the two spatial modes and transmits V** — Perceval's
    /// convention, pinned in `test_perceval_conventions.py`. Note this is the
    /// opposite of the common "transmits H, reflects V" phrasing, which is why
    /// it is pinned against a read of the actual matrix rather than described
    /// in a comment.
    ///
    /// The swap of two modes is `PS(π) · BSrx(π/2, π)`. The phase shifter is
    /// not cosmetic: `det(swap) = −1` while `det(BSrx) = +1`, so it supplies
    /// the sign. V is untouched, hence no ops on the V sub-modes at all.
    fn lower_polarizing_beam_splitter(&mut self, app: &OpticGateApp) -> Result<(), String> {
        if app.modes.len() != 2 {
            return Err(format!(
                "pbs acts on 2 spatial modes; got {}",
                app.modes.len()
            ));
        }
        if !app.params.is_empty() {
            return Err(format!(
                "pbs takes no parameters; got {}",
                app.params.len()
            ));
        }

        let (a_h, _a_v) = self.polarization_submodes(&app.modes[0], "pbs")?;
        let (b_h, _b_v) = self.polarization_submodes(&app.modes[1], "pbs")?;

        self.push_photonic(
            GateKind::BeamSplitterRx,
            &[a_h, b_h],
            &[std::f64::consts::FRAC_PI_2, std::f64::consts::PI],
        );
        self.push_photonic(GateKind::PhaseShifter, &[b_h], &[std::f64::consts::PI]);
        Ok(())
    }

    fn lower_opticqasm_stmt(&mut self, stmt: &OpticQasmStmt) -> Result<(), String> {
        match stmt {
            OpticQasmStmt::PhotonDecl {
                name,
                size,
                polarized,
            } => {
                let start = self.num_qubits;
                // A polarized register occupies TWO optical modes per spatial
                // mode. The doubling happens here and nowhere else: the
                // governor prices photonic admission from `num_qubits`, so
                // doubling at lowering keeps admission correct with no change
                // to the governor. See FIXES_PLAN.md I1.
                let optical = if *polarized { size * 2 } else { *size };
                self.qregs.insert(name.clone(), (start, optical));
                if *polarized {
                    self.polarized_regs.insert(name.clone());
                }
                self.num_qubits += optical;
                Ok(())
            }
            OpticQasmStmt::GateApp(app) => {
                // Polarization elements are not new primitives: they expand
                // into the phase shifters and beam splitters already
                // implemented, acting on polarization sub-modes. Handled before
                // the ordinary path because they emit MULTIPLE ops.
                match app.name.as_str() {
                    "hwp" => return self.lower_half_wave_plate(app),
                    "pbs" => return self.lower_polarizing_beam_splitter(app),
                    _ => {}
                }

                let gate = match app.name.as_str() {
                    "ps" => GateKind::PhaseShifter,
                    // `bs` is an accepted alias in `aria-core::from_opticqasm`
                    // and is named by the grammar's `gate_name` rule, so it
                    // parsed here and then died at lowering — a spelling two of
                    // the three tables knew.
                    "bs_rx" | "bs" => GateKind::BeamSplitterRx,
                    // The continuous-variable profile. Saying "unknown" here was
                    // FALSE and it is what made `aria-core`'s own OPTICQASM
                    // export unreadable: piquasso implements all three, and so
                    // does this workspace's `omega-backend-cv`. What is true is
                    // that the discrete-variable IR cannot express a Fock-space
                    // operator. Wording deliberately parallels
                    // `aria-core/src/backends/omega.rs`, which refuses the same
                    // three on the execution path.
                    cv @ ("squeeze" | "displace" | "kerr") => {
                        return Err(format!(
                            "`{cv}` is a continuous-variable gate; the discrete-variable \
                             omega IR cannot express it. Import the CV profile with \
                             `omega_parser::lower_opticqasm_cv` and run it on \
                             `omega-backend-cv` (piquasso reads the same operations)."
                        ))
                    }
                    // Named by the grammar's `gate_name` rule but implemented
                    // nowhere. It cannot simply be deleted from the grammar:
                    // `gate_name` is an ordered choice over an atomic rule, so
                    // with `bs_ry` gone the earlier `"bs"` alternative matches
                    // its prefix and leaves `_ry` dangling — measured, the clear
                    // message below degrades into `--> 3:3` from pest. So the
                    // spelling stays tokenizable and the refusal says what is
                    // actually true.
                    "bs_ry" => {
                        return Err(
                            "`bs_ry` (Ry-convention beam splitter) is named by the OPTICQASM \
                             grammar but has no implementation: the IR carries only \
                             `BeamSplitterRx`. Use `bs_rx`, which Perceval's `BS` and \
                             piquasso's `Beamsplitter` both match."
                                .to_string(),
                        )
                    }
                    other => return Err(format!("unknown photonic gate: {}", other)),
                };

                // Arity, on BOTH parameters and modes.
                //
                // Neither was checked. Measured before this: `bs_rx(1.2) q[0], q[1];`,
                // `ps(0.5, 0.7) q[0];` and `ps(0.5) q[0], q[1];` all lowered to
                // `Ok(1 op)` here while the CV profile refused all three — two
                // readers of one dialect disagreeing about what is valid, with
                // this one silent. A one-mode beam splitter then reaches a
                // backend that indexes `qubits[1]`.
                let (want_params, want_modes) = match gate {
                    GateKind::PhaseShifter => (1usize, 1usize),
                    GateKind::BeamSplitterRx => (2, 2),
                    _ => unreachable!("gate table above yields only these two"),
                };
                if app.params.len() != want_params {
                    return Err(format!(
                        "`{}` takes {want_params} parameter(s), got {}",
                        app.name,
                        app.params.len()
                    ));
                }
                if app.modes.len() != want_modes {
                    return Err(format!(
                        "`{}` acts on {want_modes} mode(s), got {}",
                        app.name,
                        app.modes.len()
                    ));
                }

                let mut params = smallvec::SmallVec::new();
                for p in &app.params {
                    match p {
                        OpticParam::Symbol(name) => {
                            let id = self.get_or_create_symbol(name);
                            params.push(ParamExpr::Symbol(id));
                        }
                        OpticParam::Num(v) => {
                            params.push(ParamExpr::Concrete(*v));
                        }
                    }
                }

                let mut qubits = smallvec::SmallVec::new();
                for m in &app.modes {
                    let (start, size) = self
                        .qregs
                        .get(&m.reg)
                        .ok_or_else(|| format!("undefined photon register: {}", m.reg))?;
                    if m.index >= *size {
                        return Err(format!("mode index {} out of range", m.index));
                    }
                    qubits.push(Qubit(start + m.index));
                }

                self.ops.push(GateOp {
                    gate,
                    qubits,
                    params,
                    classical_bit: None,
                    condition: None,
                });

                Ok(())
            }
        }
    }
}

/// Apply the `inv @ G` modifier to a gate-kind + parameter list, per
/// the QASM 3 spec. Returns the inverted (gate, params) pair.
///
/// - Self-inverse gates (X, Y, Z, H, CX, CY, CZ, CCX, CSwap, Swap, Id):
///   unchanged.
/// - Single-qubit Cliffords with a named inverse: S ↔ Sdg, T ↔ Tdg.
/// - Rotation gates with negatable parameter: Rx/Ry/Rz/U1 → same gate
///   with parameter negated. CRz follows the same rule.
/// - U3(θ, φ, λ)⁻¹ = U3(-θ, -λ, -φ). CU3 follows the same shape.
/// - U2 is intentionally omitted — its inverse isn't a U2; callers
///   needing it should rewrite to U3 first. We surface a clear error.
/// - Non-unitary ops (Measure, Reset, Barrier, BeamSplitter,
///   PhaseShifter, Custom) cannot be inverted at this layer.
fn invert_gate_with_params(
    gate: GateKind,
    params: &smallvec::SmallVec<[ParamExpr; 3]>,
) -> Result<(GateKind, smallvec::SmallVec<[ParamExpr; 3]>), String> {
    let negate = |p: &ParamExpr| ParamExpr::Negate(Box::new(p.clone()));
    let same = |g: GateKind| Ok((g, params.clone()));
    match gate {
        GateKind::X | GateKind::Y | GateKind::Z | GateKind::H | GateKind::Id => same(gate),
        GateKind::CX | GateKind::CY | GateKind::CZ => same(gate),
        GateKind::Swap | GateKind::CCX | GateKind::CSwap => same(gate),
        GateKind::S => same(GateKind::Sdg),
        GateKind::Sdg => same(GateKind::S),
        GateKind::T => same(GateKind::Tdg),
        GateKind::Tdg => same(GateKind::T),
        // sxdg = sx^dagger, verified `sx @ sxdg == I` to 0.000e+00.
        GateKind::Sx => same(GateKind::Sxdg),
        GateKind::Sxdg => same(GateKind::Sx),
        // RBS(θ)⁻¹ = RBS(−θ), same negatable-parameter shape as the
        // rotation gates.
        GateKind::Rx
        | GateKind::Ry
        | GateKind::Rz
        | GateKind::U1
        | GateKind::CRz
        | GateKind::Rbs => {
            let mut out = smallvec::SmallVec::with_capacity(params.len());
            if let Some(p) = params.first() {
                out.push(negate(p));
            }
            Ok((gate, out))
        }
        GateKind::U3 => {
            if params.len() != 3 {
                return Err(format!("inv @ u3 expects 3 params, got {}", params.len()));
            }
            // U3(-θ, -λ, -φ): swap last two and negate all three.
            let mut out = smallvec::SmallVec::with_capacity(3);
            out.push(negate(&params[0]));
            out.push(negate(&params[2]));
            out.push(negate(&params[1]));
            Ok((gate, out))
        }
        GateKind::CU3 => {
            if params.len() != 3 {
                return Err(format!("inv @ cu3 expects 3 params, got {}", params.len()));
            }
            let mut out = smallvec::SmallVec::with_capacity(3);
            out.push(negate(&params[0]));
            out.push(negate(&params[2]));
            out.push(negate(&params[1]));
            Ok((gate, out))
        }
        GateKind::U2 => {
            Err("inv @ u2 not supported; rewrite to u3 first (u2(φ,λ) ≡ u3(π/2,φ,λ))".into())
        }
        GateKind::Measure | GateKind::Barrier | GateKind::Reset => {
            Err(format!("inv @ {:?} is undefined (non-unitary op)", gate))
        }
        GateKind::PhaseShifter | GateKind::BeamSplitterRx => Err(format!(
            "inv @ {:?} not yet supported on the photonic surface",
            gate
        )),
        GateKind::Custom(_) => Err(
            "inv @ <custom gate> not supported; expand to primitives or define an explicit inverse"
                .into(),
        ),
    }
}
/// `cp(λ)` / `cu1(λ)` widened to the `CU3(0, 0, λ)` that omega actually has.
///
/// Extracted so BOTH the top-level path and the user-defined-gate body path use
/// it. The body path reached neither this nor the arity check, so
/// `gate mycp(t) a,b { cp(t) a,b; }` produced a CU3 carrying one parameter and
/// the statevector backend panicked indexing `resolved[2]`.
fn widen_cp_params(
    name: &str,
    params: smallvec::SmallVec<[ParamExpr; 3]>,
) -> Result<smallvec::SmallVec<[ParamExpr; 3]>, String> {
    if !matches!(name, "cp" | "cu1") {
        return Ok(params);
    }
    if params.len() != 1 {
        return Err(format!("{name} expects 1 parameter, got {}", params.len()));
    }
    let mut widened: smallvec::SmallVec<[ParamExpr; 3]> = smallvec::SmallVec::with_capacity(3);
    widened.push(ParamExpr::Concrete(0.0));
    widened.push(ParamExpr::Concrete(0.0));
    widened.push(params[0].clone());
    Ok(widened)
}

/// Refuse a wrong parameter or qubit count at LOWERING.
///
/// There was no arity validation anywhere between the parser and the backend's
/// array index, so a malformed but grammatical file crashed the process.
/// Measured before this, all three panicking in `omega-backend-statevector`:
///
/// ```text
///   gate mycp(t) a,b { cp(t) a,b; }  ->  CU3 with 1 param  -> PANIC
///   cu3(0.7) q[0], q[1];             ->  CU3 with 1 param  -> PANIC
///   u3(0.7) q[0];                    ->  U3  with 1 param  -> PANIC
/// ```
///
/// A wrong arity is a malformed input, and malformed input must be an error,
/// never a panic — a parser that crashes on bad input cannot be pointed at
/// anything untrusted.
///
/// Photonic gates are validated on their own path (`lower_opticqasm_stmt`) and
/// `Barrier`/`Measure`/`Reset` are not gate applications, so both are skipped
/// here rather than given wrong expectations.
fn check_gate_arity(
    name: &str,
    gate: &GateKind,
    n_params: usize,
    n_qubits: usize,
) -> Result<(), String> {
    use GateKind::*;
    // (params, qubits). `None` means "not checked here", with the reason above.
    let want: Option<(usize, usize)> = match gate {
        Id | X | Y | Z | H | S | Sdg | T | Tdg | Sx | Sxdg => Some((0, 1)),
        Rx | Ry | Rz | U1 => Some((1, 1)),
        U2 => Some((2, 1)),
        U3 => Some((3, 1)),
        CX | CY | CZ | Swap => Some((0, 2)),
        CU3 => Some((3, 2)),
        CRz => Some((1, 2)),
        Rbs => Some((1, 2)),
        CCX | CSwap => Some((0, 3)),
        Barrier | Measure | Reset => None,
        PhaseShifter | BeamSplitterRx => None,
        // A user-supplied opaque gate: the parser knows nothing about its
        // signature, so there is nothing to check against. Named rather than
        // swept into a `_` arm, so a NEW GateKind still fails to compile here.
        Custom(_) => None,
    };
    let Some((wp, wq)) = want else {
        return Ok(());
    };
    if n_params != wp {
        return Err(format!(
            "`{name}` takes {wp} parameter(s), got {n_params}"
        ));
    }
    if n_qubits != wq {
        return Err(format!("`{name}` acts on {wq} qubit(s), got {n_qubits}"));
    }
    Ok(())
}


/// Map a gate name string to a GateKind.
fn name_to_gate(name: &str) -> Result<GateKind, String> {
    match name {
        "h" | "H" => Ok(GateKind::H),
        "x" | "X" => Ok(GateKind::X),
        "y" | "Y" => Ok(GateKind::Y),
        "z" | "Z" => Ok(GateKind::Z),
        "s" | "S" => Ok(GateKind::S),
        "sdg" => Ok(GateKind::Sdg),
        "t" | "T" => Ok(GateKind::T),
        "tdg" => Ok(GateKind::Tdg),
        // Native, NOT aliased to U3 — see GateKind::Sx. Both tsim's and
        // ppvm's gate sets accept these, so before this the in-tree lowering
        // was the only thing in the pipeline that could not.
        "sx" => Ok(GateKind::Sx),
        "sxdg" => Ok(GateKind::Sxdg),
        "id" => Ok(GateKind::Id),
        "rx" => Ok(GateKind::Rx),
        "ry" => Ok(GateKind::Ry),
        "rz" => Ok(GateKind::Rz),
        "u3" | "U" => Ok(GateKind::U3),
        "u2" => Ok(GateKind::U2),
        // `p(λ)` is qelib1's phase gate. Qiskit's `PhaseGate(λ)` and
        // `U1Gate(λ)` are **bit-identical** — `diag(1, e^{iλ})`, verified at
        // max|Δ| = 0.0 — so this is a pure alias with no global-phase caveat,
        // unlike `sx` (see `NOT_LOWERABLE` in
        // `crates/omega-cli/tests/nway_counts.rs`). Missing until the N-way
        // matrix ran `02_single_qubit_rotations.qasm` through every in-tree
        // engine and all four refused it identically.
        "u1" | "p" => Ok(GateKind::U1),
        "cx" | "CX" | "cnot" => Ok(GateKind::CX),
        "cy" => Ok(GateKind::CY),
        "cz" => Ok(GateKind::CZ),
        "swap" => Ok(GateKind::Swap),
        "crz" => Ok(GateKind::CRz),
        // qelib1's controlled-phase. CP(λ) == CU3(0, 0, λ) exactly; the
        // 1 -> 3 parameter widening happens in `lower_gate_app`.
        "cu3" | "cp" | "cu1" => Ok(GateKind::CU3),
        // Placeholder only: `rxx`/`rzz` never reach the single-op push —
        // `lower_gate_app` intercepts them above and emits a decomposition. The
        // entry exists so `name_to_gate` does not reject the name first.
        // `ryy` is absent on purpose: no Qiskit loader reads it (measured), so
        // reading it here would fix the round trip for us alone.
        "rxx" | "rzz" => Ok(GateKind::CX),
        "ccx" | "toffoli" => Ok(GateKind::CCX),
        "cswap" | "fredkin" => Ok(GateKind::CSwap),
        "measure" => Ok(GateKind::Measure),
        "barrier" => Ok(GateKind::Barrier),
        _ => Err(format!("unknown gate: {}", name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omega_core::params::ParameterBinding;

    #[test]
    fn test_lower_bell_state() {
        let src = r#"OPENQASM 2.0;
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#;
        let ir = lower_to_ir(src).unwrap();
        assert_eq!(ir.num_qubits, 2);
        assert_eq!(ir.num_classical_bits, 2);
        assert_eq!(ir.circuit_type, CircuitType::GateBased);
        // h, cx, measure, measure = 4 ops
        assert_eq!(ir.ops.len(), 4);
        assert_eq!(ir.ops[0].gate, GateKind::H);
        assert_eq!(ir.ops[1].gate, GateKind::CX);
    }

    /// QASM 3 minimal subset (#2): `OPENQASM 3.0;`,
    /// `include "stdgates.inc";`, `qubit[n] q;`, `bit[n] c;` plus the
    /// QASM 2 statement set. The pest grammar accepts both register
    /// forms; the lowering is identical.
    #[test]
    fn test_lower_qasm3_minimal_subset() {
        let src = r#"OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#;
        let ir = lower_to_ir(src).unwrap();
        assert_eq!(ir.num_qubits, 2);
        assert_eq!(ir.num_classical_bits, 2);
        assert_eq!(ir.circuit_type, CircuitType::GateBased);
        assert_eq!(ir.ops.len(), 4);
        assert_eq!(ir.ops[0].gate, GateKind::H);
        assert_eq!(ir.ops[1].gate, GateKind::CX);
    }

    /// QASM 3 assignment measurement `c[i] = measure q[i];` lowers to the
    /// same `Measure` op as the QASM 2 arrow form. This is the form the
    /// `to_qasm3` exporter emits, so accepting it keeps the export round-
    /// trippable through this parser.
    #[test]
    fn test_lower_qasm3_measure_assignment() {
        let src = r#"OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
rz(0.7853981633974483) q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"#;
        let ir = lower_to_ir(src).unwrap();
        assert_eq!(ir.num_qubits, 2);
        assert_eq!(ir.num_classical_bits, 2);
        assert_eq!(ir.circuit_type, CircuitType::GateBased);
        // h, cx, rz, measure, measure = 5 ops
        assert_eq!(ir.ops.len(), 5);
        let measures = ir
            .ops
            .iter()
            .filter(|o| o.gate == GateKind::Measure)
            .count();
        assert_eq!(measures, 2, "both v3-form measurements must lower");
        // The decimal parameter must survive (defect-B class check on the v3 path).
        match &ir.ops[2].params[0] {
            ParamExpr::Concrete(v) => {
                assert!((v - std::f64::consts::FRAC_PI_4).abs() < 1e-12)
            }
            other => panic!("expected concrete param, got {other:?}"),
        }
    }

    /// QASM 3 mixed-form: a single circuit can use either v2 or v3
    /// register-decl syntax interchangeably, in any order.
    #[test]
    fn test_lower_qasm3_mixed_with_qasm2_decls() {
        let src = r#"OPENQASM 3.0;
qreg q[1];
bit[1] c;
x q[0];
measure q[0] -> c[0];
"#;
        let ir = lower_to_ir(src).unwrap();
        assert_eq!(ir.num_qubits, 1);
        assert_eq!(ir.num_classical_bits, 1);
        assert_eq!(ir.ops.len(), 2);
        assert_eq!(ir.ops[0].gate, GateKind::X);
    }

    /// QASM 3 unsupported constructs (`while`/`for`/real classical
    /// variables, `let`) should fail at parse-time with a clear pest
    /// error rather than silently produce wrong IR. `inv @ U` and
    /// `pow(n) @ U` modifiers are now supported (see
    /// `test_lower_qasm3_inv_modifier_*` / `_pow_modifier_*` below).
    #[test]
    fn test_lower_qasm3_unsupported_constructs_fail_fast() {
        // `let` declaration — not in the grammar.
        let src_let = r#"OPENQASM 3.0;
qubit[1] q;
let alpha = 0.5;
rx(alpha) q[0];
"#;
        assert!(
            lower_to_ir(src_let).is_err(),
            "QASM3 `let` is not in the supported subset; should error"
        );

        // `for` loop — not in the grammar.
        let src_for = r#"OPENQASM 3.0;
qubit[2] q;
for i in [0:1] { x q[i]; }
"#;
        assert!(
            lower_to_ir(src_for).is_err(),
            "QASM3 `for` is not in the supported subset; should error"
        );
    }

    #[test]
    fn test_lower_qasm3_inv_modifier_self_inverse() {
        // `inv @ x q[0]` lowers to a single X (self-inverse).
        let src = r#"OPENQASM 3.0;
qubit[1] q;
inv @ x q[0];
"#;
        let ir = lower_to_ir(src).unwrap();
        assert_eq!(ir.ops.len(), 1);
        assert_eq!(ir.ops[0].gate, GateKind::X);
    }

    #[test]
    fn test_lower_qasm3_inv_modifier_named_inverse() {
        // `inv @ s q[0]` → `sdg`. `inv @ sdg q[0]` → `s`.
        let src = r#"OPENQASM 3.0;
qubit[1] q;
inv @ s q[0];
inv @ sdg q[0];
"#;
        let ir = lower_to_ir(src).unwrap();
        assert_eq!(ir.ops.len(), 2);
        assert_eq!(ir.ops[0].gate, GateKind::Sdg);
        assert_eq!(ir.ops[1].gate, GateKind::S);
    }

    #[test]
    fn test_lower_qasm3_inv_modifier_rotation_param_negated() {
        // `inv @ rx(0.5) q[0]` → `rx(-0.5)`. We check the IR by
        // re-evaluating the negation: the lowered ParamExpr should
        // produce -0.5 when bound.
        let src = r#"OPENQASM 3.0;
qubit[1] q;
inv @ rx(0.5) q[0];
"#;
        let ir = lower_to_ir(src).unwrap();
        assert_eq!(ir.ops.len(), 1);
        assert_eq!(ir.ops[0].gate, GateKind::Rx);
        let binding = ParameterBinding::new();
        let v = binding.resolve(&ir.ops[0].params[0]).unwrap();
        assert!((v - (-0.5)).abs() < 1e-12, "rx inverse param was {v}");
    }

    #[test]
    fn test_lower_qasm3_pow_modifier_repeats_ops() {
        // `pow(3) @ x q[0]` emits three X gates.
        let src = r#"OPENQASM 3.0;
qubit[1] q;
pow(3) @ x q[0];
"#;
        let ir = lower_to_ir(src).unwrap();
        assert_eq!(ir.ops.len(), 3);
        for op in &ir.ops {
            assert_eq!(op.gate, GateKind::X);
        }
    }

    #[test]
    fn test_lower_qasm3_pow_zero_is_noop() {
        // `pow(0) @ x q[0]` lowers to nothing.
        let src = r#"OPENQASM 3.0;
qubit[1] q;
pow(0) @ x q[0];
h q[0];
"#;
        let ir = lower_to_ir(src).unwrap();
        assert_eq!(ir.ops.len(), 1, "pow(0) must elide; only the H remains");
        assert_eq!(ir.ops[0].gate, GateKind::H);
    }

    #[test]
    fn test_lower_qasm3_inv_chained_with_pow_on_rotation() {
        // `inv @ pow(2) @ rz(0.3) q[0]` should emit two Rz(-0.3) ops:
        // inv toggles parity, pow(2) multiplies repetition by 2.
        let src = r#"OPENQASM 3.0;
qubit[1] q;
inv @ pow(2) @ rz(0.3) q[0];
"#;
        let ir = lower_to_ir(src).unwrap();
        assert_eq!(ir.ops.len(), 2);
        let binding = ParameterBinding::new();
        for op in &ir.ops {
            assert_eq!(op.gate, GateKind::Rz);
            let v = binding.resolve(&op.params[0]).unwrap();
            assert!((v - (-0.3)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_lower_qasm3_inv_modifier_on_user_gate_rejected() {
        // Until inv-on-user-gate is implemented, the lowerer must
        // fail-fast so operators don't get silently-wrong IR.
        let src = r#"OPENQASM 3.0;
qubit[1] q;
gate mygate(a) p { rz(a) p; }
inv @ mygate(0.5) q[0];
"#;
        assert!(
            lower_to_ir(src).is_err(),
            "inv on user-defined gate must error until we expand bodies"
        );
    }

    #[test]
    fn test_lower_opticqasm() {
        let src = r#"OPTICQASM 1.0;
photon q[4];
ps($phi0) q[0];
bs_rx($theta0, $phi_tr0) q[0], q[1];
"#;
        let ir = lower_to_ir(src).unwrap();
        assert_eq!(ir.num_qubits, 4);
        assert_eq!(ir.circuit_type, CircuitType::Photonic);
        assert_eq!(ir.ops.len(), 2);
        assert_eq!(ir.ops[0].gate, GateKind::PhaseShifter);
        assert_eq!(ir.ops[1].gate, GateKind::BeamSplitterRx);
        // Should have 3 free symbols: phi0, theta0, phi_tr0
        assert_eq!(ir.symbols.len(), 3);
    }

    #[test]
    fn test_lower_reset() {
        let src = r#"OPENQASM 2.0;
qreg q[2];
h q[0];
cx q[0], q[1];
reset q[0];
"#;
        let ir = lower_to_ir(src).unwrap();
        assert_eq!(ir.num_qubits, 2);
        // h, cx, reset = 3 ops
        assert_eq!(ir.ops.len(), 3);
        assert_eq!(ir.ops[2].gate, GateKind::Reset);
        assert_eq!(ir.ops[2].qubits[0], Qubit(0));
    }

    #[test]
    fn test_lower_reset_register() {
        // reset on entire register should expand to per-qubit resets
        let src = r#"OPENQASM 2.0;
qreg q[3];
reset q;
"#;
        let ir = lower_to_ir(src).unwrap();
        assert_eq!(ir.ops.len(), 3);
        for (i, op) in ir.ops.iter().enumerate() {
            assert_eq!(op.gate, GateKind::Reset);
            assert_eq!(op.qubits[0], Qubit(i as u32));
        }
    }

    #[test]
    fn test_lower_if_classical_condition() {
        // `if(c==1) x q[1];` must lower the inner gate WITH a classical
        // condition keyed on the creg's first bit. Previously this was
        // dropped, and the conditional gate fired unconditionally.
        let src = r#"OPENQASM 2.0;
qreg q[2];
creg c[1];
h q[0];
measure q[0] -> c[0];
if(c==1) x q[1];
measure q[1] -> c[0];
"#;
        let ir = lower_to_ir(src).unwrap();
        // Locate the X gate.
        let x_op = ir
            .ops
            .iter()
            .find(|op| op.gate == GateKind::X)
            .expect("conditional X should be present in IR");
        assert_eq!(
            x_op.condition,
            Some((0, 1, 1)),
            "conditional X must carry (start_bit=0, num_bits=1, expected=1); got {:?}",
            x_op.condition
        );
        assert_eq!(x_op.qubits[0], Qubit(1));
    }

    #[test]
    fn test_lower_angle_constant_folding() {
        // Six expressions previously rejected with `non-constant division
        // not yet supported` because Mul/Neg didn't fold their concrete
        // children. KNOWN_GAPS.md → "Constant-folding of mixed `*` and `/`".
        let cases: &[(&str, f64)] = &[
            ("2*pi/3", 2.0 * std::f64::consts::PI / 3.0),
            ("(2*pi)/3", 2.0 * std::f64::consts::PI / 3.0),
            ("pi*2/3", std::f64::consts::PI * 2.0 / 3.0),
            ("-pi/2", -std::f64::consts::PI / 2.0),
            ("(-pi)/2", -std::f64::consts::PI / 2.0),
            ("-1*pi/2", -std::f64::consts::PI / 2.0),
        ];
        for (expr, expected) in cases {
            let src = format!("OPENQASM 2.0;\nqreg q[1];\nrx({}) q[0];\n", expr);
            let ir = lower_to_ir(&src)
                .unwrap_or_else(|e| panic!("expected `{}` to lower; got {}", expr, e));
            let op = &ir.ops[0];
            assert_eq!(op.gate, GateKind::Rx);
            match &op.params[0] {
                ParamExpr::Concrete(v) => assert!(
                    (v - expected).abs() < 1e-12,
                    "`{}` lowered to {} but expected {}",
                    expr,
                    v,
                    expected
                ),
                other => panic!(
                    "`{}` should constant-fold to Concrete; got {:?}",
                    expr, other
                ),
            }
        }
    }

    #[test]
    fn test_lower_if_zero_value() {
        // `if(c==0) ...` must use expected=0, not be dropped.
        let src = r#"OPENQASM 2.0;
qreg q[1];
creg c[1];
measure q[0] -> c[0];
if(c==0) x q[0];
"#;
        let ir = lower_to_ir(src).unwrap();
        let x_op = ir
            .ops
            .iter()
            .find(|op| op.gate == GateKind::X)
            .expect("conditional X should be present in IR");
        assert_eq!(x_op.condition, Some((0, 1, 0)));
    }

    #[test]
    fn test_lower_if_multi_bit_creg_value() {
        // `creg c[2]; if(c == 2) x q[0];` — closes the
        // KNOWN_GAPS.md "multi-bit creg only checks LSB" gap.
        // The condition must record `num_bits = 2` so the executor
        // assembles c[0..2] into a 2-bit value before comparing.
        let src = r#"OPENQASM 2.0;
qreg q[1];
creg c[2];
if(c==2) x q[0];
"#;
        let ir = lower_to_ir(src).unwrap();
        let x_op = ir
            .ops
            .iter()
            .find(|op| op.gate == GateKind::X)
            .expect("conditional X should be present in IR");
        // c starts at classical bit 0, has 2 bits, expects value 2
        // (binary 10 → c[0]=0, c[1]=1).
        assert_eq!(x_op.condition, Some((0, 2, 2)));
    }

    #[test]
    fn test_lower_if_multi_bit_creg_offset() {
        // Multi-bit creg that doesn't start at offset 0 — the
        // condition must record the correct start bit.
        let src = r#"OPENQASM 2.0;
qreg q[1];
creg a[1];
creg b[3];
if(b==5) x q[0];
"#;
        let ir = lower_to_ir(src).unwrap();
        let x_op = ir
            .ops
            .iter()
            .find(|op| op.gate == GateKind::X)
            .expect("conditional X should be present in IR");
        // a has 1 bit at offset 0; b starts at offset 1, has 3
        // bits, expected value 5 (binary 101 → b[0]=1, b[1]=0,
        // b[2]=1).
        assert_eq!(x_op.condition, Some((1, 3, 5)));
    }
}
