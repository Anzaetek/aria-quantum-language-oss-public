/// Source format of the parsed circuit.
#[derive(Clone, Debug, PartialEq)]
pub enum SourceFormat {
    Qasm2,
    OpticQasm,
}

/// AST for QASM 2.0
#[derive(Clone, Debug)]
pub struct Qasm2Program {
    pub version: String,
    pub statements: Vec<Qasm2Stmt>,
}

#[derive(Clone, Debug)]
pub enum Qasm2Stmt {
    Include(String),
    QregDecl {
        name: String,
        size: u32,
    },
    CregDecl {
        name: String,
        size: u32,
    },
    GateDef(GateDef),
    GateApp(GateApp),
    Measure {
        qubit: QubitRef,
        cbit: CbitRef,
    },
    Barrier(Vec<QubitRef>),
    If {
        creg: String,
        value: u64,
        then: Box<Qasm2Stmt>,
    },
    Reset(QubitRef),
}

#[derive(Clone, Debug)]
pub struct GateDef {
    pub name: String,
    pub params: Vec<String>,
    pub qubits: Vec<String>,
    pub body: Vec<GateApp>,
}

#[derive(Clone, Debug)]
pub struct GateApp {
    pub name: String,
    pub params: Vec<Expr>,
    pub qubits: Vec<QubitRef>,
    /// QASM 3 modifiers, outermost-first. `inv @ pow(3) @ x` parses as
    /// `[GateModifier::Inv, GateModifier::Pow(3)]` and lowers as
    /// `inv (pow(3) (x))`.
    pub modifiers: Vec<GateModifier>,
}

#[derive(Clone, Debug)]
pub enum GateModifier {
    Inv,
    Pow(i32),
}

#[derive(Clone, Debug)]
pub enum QubitRef {
    /// Single qubit: qreg[index]
    Indexed { reg: String, index: u32 },
    /// Whole register
    Register(String),
}

#[derive(Clone, Debug)]
pub enum CbitRef {
    Indexed { reg: String, index: u32 },
    Register(String),
}

/// Arithmetic expressions for gate parameters.
#[derive(Clone, Debug)]
pub enum Expr {
    Num(f64),
    Pi,
    Ident(String),
    Neg(Box<Expr>),
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    FnCall(String, Box<Expr>),
}

#[derive(Clone, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// AST for OPTICQASM 1.0
#[derive(Clone, Debug)]
pub struct OpticQasmProgram {
    pub version: String,
    pub statements: Vec<OpticQasmStmt>,
}

#[derive(Clone, Debug)]
pub enum OpticQasmStmt {
    PhotonDecl {
        name: String,
        /// Declared size. When `polarized`, this counts **spatial** modes and
        /// the register occupies `2 * size` optical modes.
        size: u32,
        /// `photon q[N] pol;` — each spatial mode carries H and V, indexed
        /// `(s, p) -> 2s + p` with `p = 0` meaning H.
        ///
        /// Kept as a flag on the declaration rather than a separate statement
        /// kind so that the mode-doubling happens in exactly one place
        /// (`lower_opticqasm_stmt`). That matters beyond tidiness: the resource
        /// governor prices photonic jobs from `ir.num_qubits`, so as long as
        /// lowering doubles, admission is automatically correct and no change
        /// is needed in the governor. Doubling anywhere downstream of the IR
        /// would under-price by a binomial factor — see FIXES_PLAN.md I1.
        polarized: bool,
    },
    GateApp(OpticGateApp),
}

#[derive(Clone, Debug)]
pub struct OpticGateApp {
    pub name: String,
    pub params: Vec<OpticParam>,
    pub modes: Vec<ModeRef>,
}

#[derive(Clone, Debug)]
pub enum OpticParam {
    Symbol(String),
    Num(f64),
}

#[derive(Clone, Debug)]
pub struct ModeRef {
    pub reg: String,
    pub index: u32,
}
