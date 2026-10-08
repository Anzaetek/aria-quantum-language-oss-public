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
    /// `true` when the header token was `DITQASM` (mqt.qudits). The only
    /// grammar it unlocks is the `qreg` dimension group; under an `OPENQASM`
    /// header, or none, that group is refused in lowering rather than read
    /// as a fourth dialect no other tool writes.
    pub ditqasm: bool,
    pub statements: Vec<Qasm2Stmt>,
}

#[derive(Clone, Debug)]
pub enum Qasm2Stmt {
    Include(String),
    QregDecl {
        name: String,
        size: u32,
        /// DITQASM dimension group, `qreg q [3][3,2,5];` — one entry per
        /// wire. `None` for QASM 2/3 declarations and for the bare DITQASM
        /// form `qreg q [2];`, which `mqt.qudits` reads as all-qubit
        /// (`tools/ditqasm_xcheck/README.md`, Q0).
        dims: Option<Vec<u32>>,
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
        /// `Some(i)` when the guard addresses a SINGLE BIT (`c[i] == true`,
        /// OpenQASM 3's `bit == const bool`); `None` when it compares the whole
        /// register (`c == 1`). The two are different predicates — the reason
        /// `to_qasm` refuses a single-bit guard on a wide register rather than
        /// widening it — so they cannot share a representation.
        bit: Option<u32>,
        /// The guarded body. A `Vec` because OpenQASM 3 allows a braced block
        /// (`if (c==1) { x q[0]; y q[1]; }`) as well as a single bare
        /// statement; the bare form is simply a body of length one, so both
        /// spellings share one lowering path and cannot diverge.
        then: Vec<Qasm2Stmt>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModeRef {
    pub reg: String,
    pub index: u32,
}

/// AST for FERMIONICQASM 1.0.
///
/// In-house format. It carries no authority of a standard.
#[derive(Clone, Debug, PartialEq)]
pub struct FermionicQasmProgram {
    pub version: String,
    pub statements: Vec<FermionicQasmStmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FermionicQasmStmt {
    /// `mode m[N];` or `mode m[N] spin;`.
    ModeDecl {
        name: String,
        /// Declared size. When `spin`, this counts **spatial** modes and the
        /// register occupies `2 * size` wires in blocks: `0..size` spin-up
        /// (alpha), then `size..2*size` spin-down (beta). Not `pol`'s
        /// interleaved `(s, p) -> 2s + p` map — the lowering is not shared.
        size: u32,
        spin: bool,
    },
    /// `creg c[N];`. A statement of its own: under ordered choice a `creg`
    /// line is otherwise a gate application named `creg`.
    CregDecl {
        name: String,
        size: u32,
    },
    /// `load m[i], ...;` — which modes start occupied.
    ///
    /// A `load` after a gate, or two loads of one mode, is refused in
    /// lowering. This phase only records the statement.
    Load {
        modes: Vec<ModeRef>,
    },
    /// `measure m[i] -> c[j];`.
    Measure {
        mode: ModeRef,
        /// Classical bit. Same spelling as a mode reference; not a mode.
        cbit: ModeRef,
    },
    GateApp(FermionicGateApp),
}

#[derive(Clone, Debug, PartialEq)]
pub struct FermionicGateApp {
    pub name: String,
    pub params: Vec<FermionicParam>,
    pub modes: Vec<ModeRef>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FermionicParam {
    Symbol(String),
    Num(f64),
}
