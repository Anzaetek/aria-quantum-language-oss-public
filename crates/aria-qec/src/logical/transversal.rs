//! Transversal gadgets: emit the *physical* circuit that realizes a logical gate.
//!
//! A [`TransversalCode`] wraps a [`StabilizerCode`] and knows how to emit the
//! physical operations that implement each logical gate transversally — the key
//! property of the low-overhead architectures (QuEra Transversal STAR,
//! Quantinuum trapped-ion small codes): a logical gate is a *parallel* pattern
//! of physical gates across a patch's data qubits, distance-preserving and
//! costing ~1 syndrome round.
//!
//! Phase 1 implements the Clifford gate set for [`SteaneTransversal`]. Steane is
//! the ideal first vehicle: it is a self-dual CSS code whose Clifford group is
//! *bare* transversal (physical H/S/CNOT qubit-for-qubit), and it already ships
//! a real logical-|0⟩ encoder in [`crate::ecc::codes::SteaneCode`]. Surface-code
//! and non-Clifford gadgets arrive in later phases.

use std::f64::consts::PI;

use crate::ecc::codes::{QECCode, SteaneCode};
use aria_core::ast::CircuitBuilder;

use super::distill::MagicStateProtocol;
use super::patch::{PatchLayout, StabilizerCode};

/// How a non-Clifford logical gate (T / small-angle Rz) is realized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonCliffordMode {
    /// Faithful: small-angle transversal multi-rotation injection (surface) or a
    /// distilled magic state consumed by code switching (ion). The gadget's
    /// residual logical error is reported in [`GadgetCost::injected_pl`].
    Faithful,
    /// Ideal: apply the exact noiseless logical rotation (isolates code
    /// performance from the magic-state cost). No injected error.
    IdealLogical,
}

/// Resource cost of one non-Clifford gadget.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GadgetCost {
    /// Syndrome-extraction rounds the gadget costs.
    pub syndrome_rounds: usize,
    /// Distilled magic states consumed.
    pub magic_states: usize,
    /// Residual logical error injected by the gadget (0 in ideal mode).
    pub injected_pl: f64,
}

/// Small-angle transversal injection logical error, `pL(θ) ≈ α·|θ|·p_ph` with
/// `α ≈ 0.2·d` for a distance-`d` surface code (QuEra Transversal STAR,
/// arXiv:2509.18294). Linear in the rotation angle.
pub fn small_angle_injection_error(theta: f64, d: usize, p_ph: f64) -> f64 {
    0.2 * d as f64 * theta.abs() * p_ph
}

/// Append the exact multi-qubit Z-rotation `exp(-i θ/2 · Z^⊗support)` via a CX
/// parity ladder + single-qubit Rz. On a codeword this is the logical Rz(θ)
/// because the logical-Z string commutes with every stabilizer.
pub fn emit_pauli_z_rotation(b: &mut CircuitBuilder, support: &[usize], theta: f64) {
    let n = support.len();
    if n == 0 {
        return;
    }
    for i in 0..n - 1 {
        b.cx(support[i], support[i + 1]);
    }
    b.rz(support[n - 1], theta);
    for i in (0..n - 1).rev() {
        b.cx(support[i], support[i + 1]);
    }
}

/// Emits physical gadgets for logical operations on a specific code + layout.
/// Each Clifford gadget returns the number of syndrome rounds it costs (1 for a
/// gate, 0 for a Pauli frame update / state prep) for resource accounting.
pub trait TransversalCode {
    /// The underlying stabilizer code (geometry + logical operators).
    fn code(&self) -> &dyn StabilizerCode;

    /// Prepare logical |0⟩ on `patch` (append the encoder at the patch offset).
    fn emit_prepare_zero(&self, b: &mut CircuitBuilder, l: &PatchLayout, patch: usize);
    /// Prepare logical |+⟩ on `patch`.
    fn emit_prepare_plus(&self, b: &mut CircuitBuilder, l: &PatchLayout, patch: usize);

    fn emit_h(&self, b: &mut CircuitBuilder, l: &PatchLayout, patch: usize) -> usize;
    fn emit_s(&self, b: &mut CircuitBuilder, l: &PatchLayout, patch: usize) -> usize;
    fn emit_sdg(&self, b: &mut CircuitBuilder, l: &PatchLayout, patch: usize) -> usize;
    fn emit_x(&self, b: &mut CircuitBuilder, l: &PatchLayout, patch: usize) -> usize;
    fn emit_z(&self, b: &mut CircuitBuilder, l: &PatchLayout, patch: usize) -> usize;
    fn emit_cx(&self, b: &mut CircuitBuilder, l: &PatchLayout, ctrl: usize, tgt: usize) -> usize;
    fn emit_cz(&self, b: &mut CircuitBuilder, l: &PatchLayout, a: usize, bp: usize) -> usize;

    /// Logical Rz(θ) on `patch`. Applies the exact logical rotation gadget and
    /// reports the gadget cost: in `Faithful` mode a residual logical error
    /// (small-angle injection, or a distilled magic state's infidelity when
    /// `magic` is supplied); in `IdealLogical` mode none.
    #[allow(clippy::too_many_arguments)]
    fn emit_rz(
        &self,
        b: &mut CircuitBuilder,
        l: &PatchLayout,
        patch: usize,
        theta: f64,
        mode: NonCliffordMode,
        magic: Option<&MagicStateProtocol>,
        p_ph: f64,
    ) -> GadgetCost;

    /// Logical T = Rz(π/4).
    fn emit_t(
        &self,
        b: &mut CircuitBuilder,
        l: &PatchLayout,
        patch: usize,
        mode: NonCliffordMode,
        magic: Option<&MagicStateProtocol>,
        p_ph: f64,
    ) -> GadgetCost {
        self.emit_rz(b, l, patch, PI / 4.0, mode, magic, p_ph)
    }
}

/// Steane [[7,1,3]] with bare-transversal Clifford gates.
pub struct SteaneTransversal {
    code: SteaneCode,
}

impl Default for SteaneTransversal {
    fn default() -> Self {
        Self::new()
    }
}

impl SteaneTransversal {
    pub fn new() -> Self {
        Self { code: SteaneCode }
    }

    /// Number of data qubits (7).
    fn n(&self) -> usize {
        7
    }
}

impl TransversalCode for SteaneTransversal {
    fn code(&self) -> &dyn StabilizerCode {
        &self.code
    }

    fn emit_prepare_zero(&self, b: &mut CircuitBuilder, l: &PatchLayout, patch: usize) {
        // Logical |0⟩_L = (1/√8) Σ_{g ∈ ⟨X-stabs⟩} g |0…0⟩, built consistently
        // with the X_STABS/Z_STABS generators used by SteaneCode: qubits 0,1,3
        // each lie in exactly one X-stabilizer (Sx1=X₀₂₄₆, Sx2=X₁₂₅₆,
        // Sx3=X₃₄₅₆), so seed them in |+⟩ and CNOT along each stabilizer's
        // support. The result is a +1 eigenstate of every stabilizer and of
        // Z̄ = Z^⊗7 / X̄ = X^⊗7 (verified in the run-layer tests). NB: this is a
        // different (self-consistent) convention from
        // SteaneCode::encoding_circuit, whose generators are not cross-checked.
        let g = |q: usize| l.data_qubit(patch, q);
        b.h(g(0)).h(g(1)).h(g(3));
        // Sx1 = X on {0,2,4,6}; seed 0.
        b.cx(g(0), g(2)).cx(g(0), g(4)).cx(g(0), g(6));
        // Sx2 = X on {1,2,5,6}; seed 1.
        b.cx(g(1), g(2)).cx(g(1), g(5)).cx(g(1), g(6));
        // Sx3 = X on {3,4,5,6}; seed 3.
        b.cx(g(3), g(4)).cx(g(3), g(5)).cx(g(3), g(6));
    }

    fn emit_prepare_plus(&self, b: &mut CircuitBuilder, l: &PatchLayout, patch: usize) {
        // |+⟩_L = H_L |0⟩_L.
        self.emit_prepare_zero(b, l, patch);
        self.emit_h(b, l, patch);
    }

    fn emit_h(&self, b: &mut CircuitBuilder, l: &PatchLayout, patch: usize) -> usize {
        // Self-dual CSS ⇒ transversal H maps X-stabilizers ↔ Z-stabilizers (the
        // same set) and X̄ ↔ Z̄. Bare physical H on every data qubit.
        for q in 0..self.n() {
            b.h(l.data_qubit(patch, q));
        }
        1
    }

    fn emit_s(&self, b: &mut CircuitBuilder, l: &PatchLayout, patch: usize) -> usize {
        // Transversal S^⊗7 realizes logical S† on Steane; emit Sdg^⊗7 for logical
        // S (verified numerically in the run-layer tests).
        for q in 0..self.n() {
            b.sdg(l.data_qubit(patch, q));
        }
        1
    }

    fn emit_sdg(&self, b: &mut CircuitBuilder, l: &PatchLayout, patch: usize) -> usize {
        for q in 0..self.n() {
            b.s(l.data_qubit(patch, q));
        }
        1
    }

    fn emit_x(&self, b: &mut CircuitBuilder, l: &PatchLayout, patch: usize) -> usize {
        // Logical X = X^⊗7 (Pauli frame update; no syndrome round).
        for q in 0..self.n() {
            b.x(l.data_qubit(patch, q));
        }
        0
    }

    fn emit_z(&self, b: &mut CircuitBuilder, l: &PatchLayout, patch: usize) -> usize {
        for q in 0..self.n() {
            b.z(l.data_qubit(patch, q));
        }
        0
    }

    fn emit_cx(&self, b: &mut CircuitBuilder, l: &PatchLayout, ctrl: usize, tgt: usize) -> usize {
        // Transversal CNOT: qubit-wise physical CX between the two patches.
        for q in 0..self.n() {
            b.cx(l.data_qubit(ctrl, q), l.data_qubit(tgt, q));
        }
        1
    }

    fn emit_cz(&self, b: &mut CircuitBuilder, l: &PatchLayout, a: usize, bp: usize) -> usize {
        for q in 0..self.n() {
            b.cz(l.data_qubit(a, q), l.data_qubit(bp, q));
        }
        1
    }

    fn emit_rz(
        &self,
        b: &mut CircuitBuilder,
        l: &PatchLayout,
        patch: usize,
        theta: f64,
        mode: NonCliffordMode,
        magic: Option<&MagicStateProtocol>,
        p_ph: f64,
    ) -> GadgetCost {
        // Logical Z̄ string for this patch = the code's logical-Z support at the
        // patch offset. The exact rotation exp(-i θ/2 Z̄) is logical Rz(θ).
        let support: Vec<usize> = self
            .code
            .logical_z()
            .into_iter()
            .map(|q| l.data_qubit(patch, q))
            .collect();
        emit_pauli_z_rotation(b, &support, theta);

        match mode {
            NonCliffordMode::IdealLogical => GadgetCost {
                syndrome_rounds: 1,
                magic_states: 0,
                injected_pl: 0.0,
            },
            NonCliffordMode::Faithful => match magic {
                // Distilled magic state (code switch): residual = its infidelity.
                Some(m) => GadgetCost {
                    syndrome_rounds: 1,
                    magic_states: 1,
                    injected_pl: m.output_infidelity(),
                },
                // Direct small-angle transversal injection.
                None => GadgetCost {
                    syndrome_rounds: 1,
                    magic_states: 0,
                    injected_pl: small_angle_injection_error(theta, self.code.distance(), p_ph),
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steane_cx_emits_seven_physical_cx() {
        let code = SteaneTransversal::new();
        let layout = PatchLayout::new(code.code(), 2);
        let mut b = CircuitBuilder::new("t", layout.total_data_qubits(), 0);
        let rounds = code.emit_cx(&mut b, &layout, 0, 1);
        assert_eq!(rounds, 1);
        assert_eq!(b.build().gate_count(), 7);
    }

    #[test]
    fn small_angle_injection_is_linear_with_expected_slope() {
        // pL(θ) = α·θ·p_ph, α = 0.2·d. Slope in θ must equal α·p_ph exactly.
        let d = 5usize;
        let p_ph = 1e-3;
        let alpha = 0.2 * d as f64;
        let e1 = small_angle_injection_error(0.1, d, p_ph);
        let e2 = small_angle_injection_error(0.3, d, p_ph);
        let slope = (e2 - e1) / (0.3 - 0.1);
        assert!((slope - alpha * p_ph).abs() < 1e-15, "slope = {slope}");
        // At the T angle π/4 the injected error is α·(π/4)·p_ph.
        let et = small_angle_injection_error(PI / 4.0, d, p_ph);
        assert!((et - alpha * (PI / 4.0) * p_ph).abs() < 1e-15);
    }
}
