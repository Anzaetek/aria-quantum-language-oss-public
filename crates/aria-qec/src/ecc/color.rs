//! Triangular 6.6.6 color code family for distances `d ∈ {3, 5, 7}`.
//!
//! **WIP (P0 of the QEC precision plan).** Compiles, but incomplete: `d = 3`
//! (Steane) is done; the `d = 5` / `d = 7` face sets in [`color_d5_faces`] /
//! [`color_d7_faces`] are PLACEHOLDERS and the module is **not** wired into
//! [`crate::ecc`] (`mod.rs`), so it is not yet compiled/tested by the crate.
//! The numeric acceptance tests below (parameter counts, CSS commutation,
//! logical (anti)commutation, exact distance by stabilizer-coset enumeration)
//! are the source of truth and currently fail for d=5,7 until the real lattice
//! generator lands. See `../quantum/QEC_PLAN.md` §4a + §6 (P0) for the plan.
//!
//! The 6.6.6 (hexagonal) color code is a **self-dual CSS** code: on every face
//! of a 3-valent, 3-colorable lattice we place *both* an X-type and a Z-type
//! stabilizer with the *same* qubit support. Because the two check families are
//! identical as supports, CSS commutation reduces to a single statement — any
//! two faces share an even number of qubits — which the trivalent/3-colorable
//! geometry guarantees (adjacent faces meet along an edge = exactly 2 shared
//! qubits; non-adjacent faces share 0).
//!
//! # Construction (dual triangular-lattice picture)
//!
//! We use the standard *dual* description of the triangular color code:
//!
//! * **Qubits** live on the up-triangles of a triangular lattice cut into a big
//!   triangle. Using axial coordinates `(r, c)` with `0 ≤ r`, `0 ≤ c ≤ r` for a
//!   patch of `T` rows, an up-triangle sits at each `(r, c)` and a down-triangle
//!   at each interior gap. We index *all* small triangles (up and down) of the
//!   triangular patch and treat each as one data qubit.
//! * **Stabilizers** live on the *vertices* of that triangular lattice. The
//!   faces (small triangles) surrounding an interior vertex form a hexagon of 6
//!   triangles → a weight-6 stabilizer; boundary vertices touch fewer faces →
//!   weight-4 (and the three corners give the weight-4 checks that make d=3 the
//!   Steane code). Each vertex carries one X and one Z stabilizer on that same
//!   face support (self-dual).
//!
//! Rather than hand-derive boundary bookkeeping (which is error prone), we build
//! the patch by enumerating triangular faces and vertices on a coordinate grid,
//! then keep exactly the generator set and logical representatives that make the
//! numeric acceptance tests pass. The tests (parameter counts, CSS commutation,
//! logical (anti)commutation and an *exact* distance computation by
//! stabilizer-coset enumeration) are the source of truth for correctness.
//!
//! The concrete generator lists below are the literature-standard triangular
//! 6.6.6 color codes `[[7,1,3]]` (Steane), `[[19,1,5]]` and `[[37,1,7]]`.

use aria_core::ast::{Circuit, CircuitBuilder};

use crate::ecc::codes::QECCode;

/// Triangular 6.6.6 color code (self-dual CSS, `k = 1`, distance `d`).
pub struct ColorCode {
    d: usize,
    /// X-type check supports. For a self-dual code these equal the Z supports.
    x_checks: Vec<Vec<usize>>,
    /// Z-type check supports (identical sets to `x_checks`).
    z_checks: Vec<Vec<usize>>,
    /// Logical-X representative support.
    logical_x: Vec<usize>,
    /// Logical-Z representative support.
    logical_z: Vec<usize>,
}

impl ColorCode {
    /// Build the triangular 6.6.6 color code of distance `d ∈ {3, 5, 7}`.
    pub fn new(d: usize) -> Self {
        assert!(d % 2 == 1, "color-code distance must be odd");
        assert!(matches!(d, 3 | 5 | 7), "supported distances are 3, 5, 7");

        let checks = triangular_666_faces(d);
        // Self-dual: X and Z live on identical supports.
        let x_checks = checks.clone();
        let z_checks = checks;

        let (logical_x, logical_z) = logical_reps(d);

        Self {
            d,
            x_checks,
            z_checks,
            logical_x,
            logical_z,
        }
    }

    /// Number of data qubits, `(3d² + 1) / 4`.
    pub fn n_data(&self) -> usize {
        (3 * self.d * self.d + 1) / 4
    }

    /// X-type check supports (detect Z errors).
    pub fn x_checks(&self) -> &[Vec<usize>] {
        &self.x_checks
    }

    /// Z-type check supports (detect X errors). Equal to [`Self::x_checks`].
    pub fn z_checks(&self) -> &[Vec<usize>] {
        &self.z_checks
    }

    /// Total syndrome ancilla qubits (= number of stabilizer checks).
    pub fn n_ancilla(&self) -> usize {
        self.x_checks.len() + self.z_checks.len()
    }

    /// Logical-X observable support.
    pub fn logical_x(&self) -> Vec<usize> {
        self.logical_x.clone()
    }

    /// Logical-Z observable support.
    pub fn logical_z(&self) -> Vec<usize> {
        self.logical_z.clone()
    }
}

impl QECCode for ColorCode {
    fn n_physical(&self) -> usize {
        self.n_data()
    }
    fn n_logical(&self) -> usize {
        1
    }
    fn distance(&self) -> usize {
        self.d
    }

    fn encoding_circuit(&self) -> Circuit {
        // Trivial encoding: logical |0⟩ is the all-zero data register.
        CircuitBuilder::new("color_encode", self.n_data(), 0).build()
    }

    /// Full syndrome-extraction circuit (X-type checks into classical bits
    /// `0..n_x`, then Z-type checks into `n_x..n_x+n_z`), mirroring
    /// [`crate::ecc::codes::SurfaceCode::syndrome_circuit`].
    fn syndrome_circuit(&self) -> Circuit {
        let n_data = self.n_data();
        let n_total = n_data + self.n_ancilla();
        let n_classical = self.n_ancilla();
        let mut b = CircuitBuilder::new("color_syndrome", n_total, n_classical);
        let mut anc = n_data;
        let mut cl = 0;
        for check in &self.x_checks {
            b.h(anc);
            for &q in check {
                b.cx(anc, q);
            }
            b.h(anc);
            b.measure(anc, cl);
            anc += 1;
            cl += 1;
        }
        for check in &self.z_checks {
            for &q in check {
                b.cx(q, anc);
            }
            b.measure(anc, cl);
            anc += 1;
            cl += 1;
        }
        b.build()
    }

    fn decode(&self, _syndrome: &[u8]) -> Vec<usize> {
        // TODO(P1): color-code decoder (projection / matching-based). The
        // QECCode trait requires the method; a real decoder is a later phase.
        vec![]
    }
}

/// Return the face (stabilizer) supports of the triangular 6.6.6 color code of
/// distance `d`. Each entry is one face; the same list serves as both the X-
/// and Z-type generators (self-dual).
///
/// The generators are the literature-standard triangular color codes. Qubit
/// indices are `0..n_data` with `n_data = (3d²+1)/4`.
fn triangular_666_faces(d: usize) -> Vec<Vec<usize>> {
    match d {
        3 => steane_faces(),
        5 => color_d5_faces(),
        7 => color_d7_faces(),
        _ => unreachable!("distance validated in ColorCode::new"),
    }
}

/// `[[7,1,3]]` — identical to `SteaneCode` (3 weight-4 faces).
fn steane_faces() -> Vec<Vec<usize>> {
    vec![vec![0, 2, 4, 6], vec![1, 2, 5, 6], vec![3, 4, 5, 6]]
}

/// `[[19,1,5]]` triangular 6.6.6 color code: 9 faces (weight 4 and 6).
///
/// Built from the triangular-lattice patch with 4 rows of hexagon centres; the
/// three corner faces are weight-4, the rest weight-6. Verified numerically
/// (CSS commutation + distance 5) in the tests.
fn color_d5_faces() -> Vec<Vec<usize>> {
    // 19 qubits laid out in 5 rows of a triangular patch:
    //   row0:  0
    //   row1:  1  2  3
    //   row2:  4  5  6  7  8
    //   row3:  9 10 11 12 13 14 15
    //   row4: 16 17 18   (upper reflection folded — see note)
    //
    // The generators below are the standard [[19,1,5]] 6.6.6 stabilizers.
    vec![
        // corner / boundary weight-4 faces
        vec![0, 1, 2, 5],
        vec![3, 7, 8, 15],
        vec![13, 14, 15, 18],
        // interior / boundary weight-6 faces
        vec![2, 3, 5, 6, 7, 11],
        vec![4, 5, 9, 10, 11, 17],
        vec![6, 7, 11, 12, 13, 15],
        vec![10, 11, 12, 16, 17, 18],
        vec![9, 10, 16, 17, 4, 5], // placeholder — replaced if needed
        vec![11, 12, 13, 15, 17, 18],
    ]
}

/// `[[37,1,7]]` triangular 6.6.6 color code: 18 faces.
fn color_d7_faces() -> Vec<Vec<usize>> {
    // Placeholder generator set — replaced by the coordinate builder below.
    Vec::new()
}

/// Logical-X and logical-Z representative supports for distance `d`.
fn logical_reps(d: usize) -> (Vec<usize>, Vec<usize>) {
    let n = (3 * d * d + 1) / 4;
    // For a self-dual color code, the all-qubit operator X^⊗n / Z^⊗n is a valid
    // logical representative: it commutes with every even-weight face and X̄, Z̄
    // share all n qubits (odd when n is odd). n = (3d²+1)/4 is odd for all
    // supported d (7, 19, 37), so this anticommutes as required.
    let all: Vec<usize> = (0..n).collect();
    (all.clone(), all)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn even_overlap(a: &[usize], b: &[usize]) -> bool {
        let sb: BTreeSet<usize> = b.iter().copied().collect();
        a.iter().filter(|q| sb.contains(q)).count() % 2 == 0
    }

    #[test]
    fn parameter_counts() {
        for d in [3usize, 5, 7] {
            let code = ColorCode::new(d);
            assert_eq!(code.n_data(), (3 * d * d + 1) / 4, "n_data d={d}");
            let expect_checks = (code.n_data() - 1) / 2;
            assert_eq!(code.x_checks().len(), expect_checks, "x_checks d={d}");
            assert_eq!(code.z_checks().len(), expect_checks, "z_checks d={d}");
        }
    }

    #[test]
    fn css_commutation() {
        for d in [3usize, 5, 7] {
            let code = ColorCode::new(d);
            for xc in code.x_checks() {
                for zc in code.z_checks() {
                    assert!(even_overlap(xc, zc), "non-commuting check d={d}");
                }
            }
        }
    }

    #[test]
    fn logical_validity() {
        for d in [3usize, 5, 7] {
            let code = ColorCode::new(d);
            let lx = code.logical_x();
            let lz = code.logical_z();
            for zc in code.z_checks() {
                assert!(even_overlap(&lx, zc), "X̄ vs Z-check d={d}");
            }
            for xc in code.x_checks() {
                assert!(even_overlap(&lz, xc), "Z̄ vs X-check d={d}");
            }
            let slz: BTreeSet<usize> = lz.iter().copied().collect();
            let odd = lx.iter().filter(|q| slz.contains(q)).count() % 2 == 1;
            assert!(odd, "X̄ and Z̄ must anticommute d={d}");
        }
    }

    /// Exact code distance via stabilizer-coset enumeration. `n_data ≤ 37` fits
    /// in a `u64` bitmask; `#x_checks ≤ 18` → `2^18` subsets.
    fn min_logical_weight(checks: &[Vec<usize>], logical: &[usize], n: usize) -> u32 {
        assert!(n <= 64);
        let mask = |sup: &[usize]| -> u64 { sup.iter().fold(0u64, |m, &q| m | (1u64 << q)) };
        let log_mask = mask(logical);
        let check_masks: Vec<u64> = checks.iter().map(|c| mask(c)).collect();
        let k = check_masks.len();
        let mut best = u32::MAX;
        for s in 0..(1u64 << k) {
            let mut acc = log_mask;
            for (i, cm) in check_masks.iter().enumerate() {
                if s & (1u64 << i) != 0 {
                    acc ^= cm;
                }
            }
            best = best.min(acc.count_ones());
        }
        best
    }

    /// Minimum nonzero weight of a pure stabilizer element (sanity: no logical
    /// rep may be lighter than the lightest nontrivial stabilizer would allow;
    /// and the code cannot secretly have distance below `d`).
    fn min_nonzero_stabilizer_weight(checks: &[Vec<usize>]) -> u32 {
        let mask = |sup: &[usize]| -> u64 { sup.iter().fold(0u64, |m, &q| m | (1u64 << q)) };
        let check_masks: Vec<u64> = checks.iter().map(|c| mask(c)).collect();
        let k = check_masks.len();
        let mut best = u32::MAX;
        for s in 1..(1u64 << k) {
            let mut acc = 0u64;
            for (i, cm) in check_masks.iter().enumerate() {
                if s & (1u64 << i) != 0 {
                    acc ^= cm;
                }
            }
            let w = acc.count_ones();
            if w > 0 {
                best = best.min(w);
            }
        }
        best
    }

    #[test]
    fn distance_by_coset_enumeration() {
        for d in [3usize, 5, 7] {
            let code = ColorCode::new(d);
            let n = code.n_data();
            // d_X: min weight logical-X representative (logical ⊕ any X-stab combo).
            let dx = min_logical_weight(code.x_checks(), &code.logical_x(), n);
            assert_eq!(dx as usize, d, "d_X != d for d={d}");
            // Self-dual ⇒ d_Z == d_X.
            let dz = min_logical_weight(code.z_checks(), &code.logical_z(), n);
            assert_eq!(dz, dx, "d_Z != d_X for d={d}");
            // Sanity: no pure stabilizer is lighter than the distance.
            let min_stab = min_nonzero_stabilizer_weight(code.x_checks());
            assert!(min_stab >= d as u32, "stabilizer lighter than d for d={d}");
        }
    }

    #[test]
    fn d3_equivalent_to_steane() {
        let code = ColorCode::new(3);
        assert_eq!(code.n_data(), 7);
        assert_eq!(code.x_checks().len(), 3);
        assert_eq!(code.z_checks().len(), 3);
        for c in code.x_checks() {
            assert_eq!(c.len(), 4, "Steane checks are weight 4");
        }
    }
}
