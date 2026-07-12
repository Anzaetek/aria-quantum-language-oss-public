//! Multi-patch geometry for the transversal logical layer.
//!
//! The single-patch [`QECCode`] codes in [`crate::ecc::codes`] describe *one*
//! logical qubit. Transversal logical circuits act on *many* logical patches at
//! once (a transversal CNOT couples two patches qubit-for-qubit), so this module
//! adds two things on top of the existing codes without changing them:
//!
//! * [`StabilizerCode`] — a read-only accessor trait that unifies the geometry
//!   of [`SurfaceCode`] and [`SteaneCode`] (check supports + logical operators)
//!   so the compiler can treat every CSS code the same way.
//! * [`PatchLayout`] — a flat address map that places `K` identical patches into
//!   a single `q` register (the AST has no multi-register concept), turning a
//!   `(patch, local)` pair into a global data-qubit index and a logical
//!   observable into a physical Pauli string.

use crate::ecc::codes::{QECCode, SteaneCode, SurfaceCode};

/// A single-qubit Pauli basis, used to describe logical observables without
/// pulling in the `omega-sim`-gated backend `PauliOp`. The lowering to
/// `omega_core::executor::PauliOp` lives in [`super::run`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauliBasis {
    X,
    Y,
    Z,
}

/// Read-only geometry of a CSS stabilizer code, unified across the concrete
/// codes so the transversal layer is code-agnostic. This *wraps* the existing
/// [`QECCode`] impls — it does not replace them.
pub trait StabilizerCode: QECCode {
    /// X-type check supports (each detects Z errors on its data qubits).
    fn x_check_supports(&self) -> Vec<Vec<usize>>;
    /// Z-type check supports (each detects X errors on its data qubits).
    fn z_check_supports(&self) -> Vec<Vec<usize>>;
    /// Data-qubit support of a logical-X representative.
    fn logical_x_support(&self) -> Vec<usize>;
    /// Data-qubit support of a logical-Z representative.
    fn logical_z_support(&self) -> Vec<usize>;
    /// Whether the code is CSS (X/Z checks separable). Both current codes are.
    fn is_css(&self) -> bool {
        true
    }
}

impl StabilizerCode for SurfaceCode {
    fn x_check_supports(&self) -> Vec<Vec<usize>> {
        self.x_checks().to_vec()
    }
    fn z_check_supports(&self) -> Vec<Vec<usize>> {
        self.z_checks().to_vec()
    }
    fn logical_x_support(&self) -> Vec<usize> {
        self.logical_x()
    }
    fn logical_z_support(&self) -> Vec<usize> {
        self.logical_z()
    }
}

impl StabilizerCode for SteaneCode {
    fn x_check_supports(&self) -> Vec<Vec<usize>> {
        self.x_checks()
    }
    fn z_check_supports(&self) -> Vec<Vec<usize>> {
        self.z_checks()
    }
    fn logical_x_support(&self) -> Vec<usize> {
        self.logical_x()
    }
    fn logical_z_support(&self) -> Vec<usize> {
        self.logical_z()
    }
}

/// Flat physical-address map for `K` identical logical patches sharing one `q`
/// register. Patch `p`'s local data qubit `l` lives at global index
/// `p * phys_per_patch + l`; the layout owns nothing but that arithmetic plus
/// the logical-observable → Pauli-string mapping.
#[derive(Clone, Debug)]
pub struct PatchLayout {
    n_patches: usize,
    phys_per_patch: usize,
    bases: Vec<usize>,
}

impl PatchLayout {
    /// Place `n_patches` copies of `code`'s data register back-to-back.
    pub fn new(code: &dyn StabilizerCode, n_patches: usize) -> Self {
        let phys_per_patch = code.n_physical();
        let bases = (0..n_patches).map(|p| p * phys_per_patch).collect();
        Self {
            n_patches,
            phys_per_patch,
            bases,
        }
    }

    /// Number of logical patches.
    pub fn n_patches(&self) -> usize {
        self.n_patches
    }

    /// Physical data qubits per patch (`= code.n_physical()`).
    pub fn phys_per_patch(&self) -> usize {
        self.phys_per_patch
    }

    /// Global base (data-qubit index of local qubit 0) of `patch`.
    pub fn base(&self, patch: usize) -> usize {
        self.bases[patch]
    }

    /// Global data-qubit index of local qubit `local` within `patch`.
    pub fn data_qubit(&self, patch: usize, local: usize) -> usize {
        assert!(patch < self.n_patches, "patch {patch} out of range");
        assert!(
            local < self.phys_per_patch,
            "local qubit {local} out of range"
        );
        self.bases[patch] + local
    }

    /// Total data qubits across all patches (`K * phys_per_patch`).
    pub fn total_data_qubits(&self) -> usize {
        self.n_patches * self.phys_per_patch
    }

    /// Logical-Z observable of `patch` as a physical Pauli string.
    pub fn logical_z_string(
        &self,
        patch: usize,
        code: &dyn StabilizerCode,
    ) -> Vec<(usize, PauliBasis)> {
        let base = self.bases[patch];
        code.logical_z_support()
            .into_iter()
            .map(|q| (base + q, PauliBasis::Z))
            .collect()
    }

    /// Logical-X observable of `patch` as a physical Pauli string.
    pub fn logical_x_string(
        &self,
        patch: usize,
        code: &dyn StabilizerCode,
    ) -> Vec<(usize, PauliBasis)> {
        let base = self.bases[patch];
        code.logical_x_support()
            .into_iter()
            .map(|q| (base + q, PauliBasis::X))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every X-type and Z-type check of a CSS code must overlap in an even
    /// number of qubits (they commute). Checked over both codes.
    fn assert_checks_commute(code: &dyn StabilizerCode) {
        for xc in code.x_check_supports() {
            let xs: BTreeSet<usize> = xc.into_iter().collect();
            for zc in code.z_check_supports() {
                let overlap = zc.iter().filter(|q| xs.contains(q)).count();
                assert_eq!(overlap % 2, 0, "non-commuting checks");
            }
        }
    }

    /// Logical X and Z must commute with every stabilizer of the opposite type
    /// and anticommute with each other (odd overlap).
    fn assert_logical_valid(code: &dyn StabilizerCode) {
        let lx: BTreeSet<usize> = code.logical_x_support().into_iter().collect();
        let lz: BTreeSet<usize> = code.logical_z_support().into_iter().collect();
        // Logical Z commutes with every X-check; logical X with every Z-check.
        for xc in code.x_check_supports() {
            assert_eq!(xc.iter().filter(|q| lz.contains(q)).count() % 2, 0);
        }
        for zc in code.z_check_supports() {
            assert_eq!(zc.iter().filter(|q| lx.contains(q)).count() % 2, 0);
        }
        // X̄ and Z̄ anticommute.
        assert_eq!(lx.iter().filter(|q| lz.contains(q)).count() % 2, 1);
    }

    #[test]
    fn surface_geometry_unified() {
        for d in [3usize, 5] {
            let code = SurfaceCode::new(d);
            assert_checks_commute(&code);
            assert_logical_valid(&code);
            assert_eq!(code.logical_z_support().len(), d);
        }
    }

    #[test]
    fn steane_geometry_unified() {
        let code = SteaneCode;
        assert_checks_commute(&code);
        assert_logical_valid(&code);
        assert_eq!(code.x_check_supports().len(), 3);
        assert_eq!(code.z_check_supports().len(), 3);
    }

    #[test]
    fn patch_layout_is_bijective() {
        let code = SurfaceCode::new(3);
        let k = 3;
        let layout = PatchLayout::new(&code, k);
        assert_eq!(layout.total_data_qubits(), k * 9);
        // Every (patch, local) maps to a distinct global index covering 0..27.
        let mut seen = BTreeSet::new();
        for p in 0..k {
            for l in 0..layout.phys_per_patch() {
                let g = layout.data_qubit(p, l);
                assert!(seen.insert(g), "collision at global {g}");
                assert!(g < layout.total_data_qubits());
            }
        }
        assert_eq!(seen.len(), k * 9);
    }

    #[test]
    fn logical_strings_are_offset_correctly() {
        let code = SurfaceCode::new(3);
        let layout = PatchLayout::new(&code, 2);
        let z0 = layout.logical_z_string(0, &code);
        let z1 = layout.logical_z_string(1, &code);
        assert_eq!(z0.len(), 3);
        assert_eq!(z1.len(), 3);
        // Patch 1's logical support is patch 0's shifted by phys_per_patch (9).
        for ((q0, _), (q1, _)) in z0.iter().zip(z1.iter()) {
            assert_eq!(q1 - q0, 9);
        }
        assert!(z0.iter().all(|&(_, b)| b == PauliBasis::Z));
    }
}
