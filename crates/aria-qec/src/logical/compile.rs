//! Logical circuits and their compilation to physical circuits.
//!
//! A [`LogicalCircuit`] is a sequence of [`LogicalOp`]s over `K` logical patches.
//! [`compile_physical`] expands it into a single flat physical [`Circuit`] by
//! emitting each op's *transversal gadget* (see [`super::transversal`]) at the
//! patch's data-qubit offsets. The result plus its [`PatchLayout`] and a
//! [`ResourceReport`] is a [`PhysicalProgram`], ready to run on a backend.
//!
//! Phase 1 covers the Clifford gate set + state preparation; non-Clifford
//! (`T`/`Rz`) and hardware noise arrive in later phases.

use aria_core::ast::nodes::Circuit;
use aria_core::ast::CircuitBuilder;

use super::distill::MagicStateProtocol;
use super::patch::{PatchLayout, PauliBasis};
use super::transversal::{NonCliffordMode, TransversalCode};

/// A single logical operation over the patch register. Phase-1 gate set:
/// state prep + Clifford. `patch` indices are `0..n_patches`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LogicalOp {
    /// Prepare logical |0⟩ on `patch` (runs the code's encoder).
    PrepZero(usize),
    /// Prepare logical |+⟩ on `patch`.
    PrepPlus(usize),
    H(usize),
    S(usize),
    Sdg(usize),
    X(usize),
    Z(usize),
    /// Transversal logical CNOT (qubit-wise physical CX, `ctrl` → `tgt`).
    Cx { ctrl: usize, tgt: usize },
    /// Transversal logical CZ.
    Cz { a: usize, b: usize },
    /// Non-Clifford logical T = Rz(π/4).
    T(usize),
    /// Non-Clifford logical Rz(θ).
    Rz { patch: usize, theta: f64 },
}

/// A logical circuit: `n_patches` logical qubits + an op list.
#[derive(Clone, Debug)]
pub struct LogicalCircuit {
    pub n_patches: usize,
    pub ops: Vec<LogicalOp>,
}

impl LogicalCircuit {
    pub fn new(n_patches: usize) -> Self {
        Self {
            n_patches,
            ops: Vec::new(),
        }
    }
    pub fn prep_zero(&mut self, p: usize) -> &mut Self {
        self.ops.push(LogicalOp::PrepZero(p));
        self
    }
    pub fn prep_plus(&mut self, p: usize) -> &mut Self {
        self.ops.push(LogicalOp::PrepPlus(p));
        self
    }
    pub fn h(&mut self, p: usize) -> &mut Self {
        self.ops.push(LogicalOp::H(p));
        self
    }
    pub fn s(&mut self, p: usize) -> &mut Self {
        self.ops.push(LogicalOp::S(p));
        self
    }
    pub fn sdg(&mut self, p: usize) -> &mut Self {
        self.ops.push(LogicalOp::Sdg(p));
        self
    }
    pub fn x(&mut self, p: usize) -> &mut Self {
        self.ops.push(LogicalOp::X(p));
        self
    }
    pub fn z(&mut self, p: usize) -> &mut Self {
        self.ops.push(LogicalOp::Z(p));
        self
    }
    pub fn cx(&mut self, ctrl: usize, tgt: usize) -> &mut Self {
        self.ops.push(LogicalOp::Cx { ctrl, tgt });
        self
    }
    pub fn cz(&mut self, a: usize, b: usize) -> &mut Self {
        self.ops.push(LogicalOp::Cz { a, b });
        self
    }
    pub fn t(&mut self, p: usize) -> &mut Self {
        self.ops.push(LogicalOp::T(p));
        self
    }
    pub fn rz(&mut self, p: usize, theta: f64) -> &mut Self {
        self.ops.push(LogicalOp::Rz { patch: p, theta });
        self
    }
}

/// Compilation options controlling how non-Clifford gates are realized.
#[derive(Clone, Debug)]
pub struct CompileOptions {
    pub noncliff: NonCliffordMode,
    /// Magic-state protocol for faithful T (code-switch); `None` ⇒ small-angle
    /// transversal injection.
    pub magic: Option<MagicStateProtocol>,
    /// Physical error rate feeding the small-angle injection model.
    pub p_ph: f64,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            noncliff: NonCliffordMode::IdealLogical,
            magic: None,
            p_ph: 1e-3,
        }
    }
}

/// Accounting for a compiled physical program.
#[derive(Clone, Debug, Default)]
pub struct ResourceReport {
    /// Total physical data qubits (`K * phys_per_patch`).
    pub data_qubits: usize,
    /// Syndrome-extraction rounds the gadget sequence would cost on hardware.
    pub syndrome_rounds: usize,
    /// Magic states consumed (non-Clifford; 0 in the Clifford phase).
    pub magic_states: usize,
    /// Total residual logical error injected by non-Clifford gadgets
    /// (0 in ideal-logical mode).
    pub injected_logical_error: f64,
}

/// A logical circuit expanded to a physical circuit under a chosen code.
#[derive(Clone, Debug)]
pub struct PhysicalProgram {
    /// Flat physical circuit over one `q` register of `data_qubits` qubits.
    pub circuit: Circuit,
    /// The patch address map.
    pub layout: PatchLayout,
    /// Per-patch logical-Z observable as a physical Pauli string.
    pub logical_z: Vec<Vec<(usize, PauliBasis)>>,
    /// Per-patch logical-X observable as a physical Pauli string.
    pub logical_x: Vec<Vec<(usize, PauliBasis)>>,
    /// Resource accounting.
    pub resource: ResourceReport,
}

/// Compile a logical circuit into a physical circuit (ideal-logical
/// non-Clifford, default options).
pub fn compile_physical(circ: &LogicalCircuit, code: &dyn TransversalCode) -> PhysicalProgram {
    compile_physical_opts(circ, code, &CompileOptions::default())
}

/// Compile a logical circuit into a physical circuit by emitting the transversal
/// gadget of every op at its patch offsets, honoring the non-Clifford `opts`.
pub fn compile_physical_opts(
    circ: &LogicalCircuit,
    code: &dyn TransversalCode,
    opts: &CompileOptions,
) -> PhysicalProgram {
    let layout = PatchLayout::new(code.code(), circ.n_patches);
    let mut b = CircuitBuilder::new("logical_phys", layout.total_data_qubits(), 0);
    let mut rounds = 0usize;
    let mut magic_states = 0usize;
    let mut injected = 0.0f64;

    for op in &circ.ops {
        match *op {
            LogicalOp::PrepZero(p) => code.emit_prepare_zero(&mut b, &layout, p),
            LogicalOp::PrepPlus(p) => code.emit_prepare_plus(&mut b, &layout, p),
            LogicalOp::H(p) => rounds += code.emit_h(&mut b, &layout, p),
            LogicalOp::S(p) => rounds += code.emit_s(&mut b, &layout, p),
            LogicalOp::Sdg(p) => rounds += code.emit_sdg(&mut b, &layout, p),
            LogicalOp::X(p) => rounds += code.emit_x(&mut b, &layout, p),
            LogicalOp::Z(p) => rounds += code.emit_z(&mut b, &layout, p),
            LogicalOp::Cx { ctrl, tgt } => rounds += code.emit_cx(&mut b, &layout, ctrl, tgt),
            LogicalOp::Cz { a, b: bp } => rounds += code.emit_cz(&mut b, &layout, a, bp),
            LogicalOp::T(p) => {
                let c = code.emit_t(&mut b, &layout, p, opts.noncliff, opts.magic.as_ref(), opts.p_ph);
                rounds += c.syndrome_rounds;
                magic_states += c.magic_states;
                injected += c.injected_pl;
            }
            LogicalOp::Rz { patch, theta } => {
                let c = code.emit_rz(
                    &mut b, &layout, patch, theta, opts.noncliff, opts.magic.as_ref(), opts.p_ph,
                );
                rounds += c.syndrome_rounds;
                magic_states += c.magic_states;
                injected += c.injected_pl;
            }
        }
    }

    let sc = code.code();
    let logical_z = (0..circ.n_patches)
        .map(|p| layout.logical_z_string(p, sc))
        .collect();
    let logical_x = (0..circ.n_patches)
        .map(|p| layout.logical_x_string(p, sc))
        .collect();

    PhysicalProgram {
        resource: ResourceReport {
            data_qubits: layout.total_data_qubits(),
            syndrome_rounds: rounds,
            magic_states,
            injected_logical_error: injected,
        },
        circuit: b.build(),
        layout,
        logical_z,
        logical_x,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical::transversal::SteaneTransversal;

    #[test]
    fn compile_two_patch_cnot_shapes() {
        let code = SteaneTransversal::new();
        let mut lc = LogicalCircuit::new(2);
        lc.prep_zero(0).prep_zero(1).h(0).cx(0, 1);
        let prog = compile_physical(&lc, &code);
        // 2 patches × 7 data qubits.
        assert_eq!(prog.resource.data_qubits, 14);
        assert_eq!(prog.circuit.n_qubits(), 14);
        // H (1 round) + CX (1 round) = 2 syndrome rounds; prep is free.
        assert_eq!(prog.resource.syndrome_rounds, 2);
        assert_eq!(prog.logical_z.len(), 2);
        // Logical-Z of patch 1 lives on qubits 7..14.
        assert!(prog.logical_z[1].iter().all(|&(q, _)| (7..14).contains(&q)));
    }
}
