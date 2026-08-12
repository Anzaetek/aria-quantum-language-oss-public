//! Stabilizer tableau for efficient Clifford circuit simulation.
//!
//! Uses the Aaronson-Gottesman (2004) formalism where each Pauli row is:
//!   (-1)^sign * Π_q i^{x_q·z_q} X_q^{x_q} Z_q^{z_q}
//! giving {I, X, Y, Z} per qubit with a global ±1 phase tracked by `sign`.

use rand::{Rng, RngExt};

/// A single Pauli string in binary symplectic form.
#[derive(Clone, Debug)]
pub struct PauliRow {
    pub n: usize,
    pub x: Vec<bool>,
    pub z: Vec<bool>,
    /// Sign bit: true means an overall factor of -1.
    pub sign: bool,
}

impl PauliRow {
    pub fn identity(n: usize) -> Self {
        Self {
            n,
            x: vec![false; n],
            z: vec![false; n],
            sign: false,
        }
    }

    /// Check if this Pauli row anti-commutes with another.
    pub fn anticommutes(&self, other: &PauliRow) -> bool {
        let mut count = 0u32;
        for q in 0..self.n {
            // Symplectic inner product
            if (self.x[q] && other.z[q]) ^ (self.z[q] && other.x[q]) {
                count += 1;
            }
        }
        count % 2 == 1
    }
}

/// The Aaronson–Gottesman `g` function: the power of `i` picked up when the
/// per-qubit Pauli `(x1,z1)` is multiplied by `(x2,z2)`, in the
/// `i^{xz} X^x Z^z` convention the tableau rows use. Returned mod 4, so
/// `3 ≡ -1`.
///
/// Derived from `P(x,z) = i^{xz} X^x Z^z`:
/// `g = x1·z1 + x2·z2 + 2·z1·x2 − (x1⊕x2)(z1⊕z2)  (mod 4)`.
///
/// **This is the ONLY copy.** There were two: this one, driving the
/// measurement path (`rowmult` / `row_product` / `measure` / `measure_prob`),
/// and a free function in `sim.rs` driving expectations. Both had the `X·Z`
/// and `Z·X` rows inverted; commit 1c3ef82 fixed only the `sim.rs` copy, so
/// expectations became correct while **measurement sampling stayed broken** —
/// a 3-qubit Clifford circuit put 1000/1000 shots on zero-probability
/// bitstrings. `sim.rs` now calls this one. Do not reintroduce a second table.
pub(crate) fn pauli_mult_phase(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
    match ((x1, z1), (x2, z2)) {
        ((false, false), _) | (_, (false, false)) => 0,
        ((true, false), (false, true)) => 3, // X·Z  => g = -1
        ((false, true), (true, false)) => 1, // Z·X  => g = +1
        ((true, false), (true, true)) => 1,  // X·(XZ) = Z
        ((true, true), (true, false)) => 3,  // (XZ)·X
        ((false, true), (true, true)) => 3,  // Z·(XZ)
        ((true, true), (false, true)) => 1,  // (XZ)·Z
        _ => 0,                              // same: P² = I
    }
}

/// Stabilizer tableau for n qubits.
///
/// Contains 2n rows: rows 0..n are destabilizers, rows n..2n are stabilizers.
/// The state |ψ⟩ is uniquely defined by the n stabilizer generators.
pub struct StabilizerTableau {
    pub n: usize,
    pub rows: Vec<PauliRow>,
}

impl StabilizerTableau {
    /// Initialize |0...0⟩: stabilizers are Z_q, destabilizers are X_q.
    pub fn zero_state(n: usize) -> Self {
        let mut rows = Vec::with_capacity(2 * n);
        // Destabilizers: X_0, X_1, ..., X_{n-1}
        for q in 0..n {
            let mut row = PauliRow::identity(n);
            row.x[q] = true;
            rows.push(row);
        }
        // Stabilizers: Z_0, Z_1, ..., Z_{n-1}
        for q in 0..n {
            let mut row = PauliRow::identity(n);
            row.z[q] = true;
            rows.push(row);
        }
        Self { n, rows }
    }

    /// Apply Hadamard gate on qubit q.
    /// X↔Z, Y→-Y.
    pub fn h(&mut self, q: usize) {
        for row in &mut self.rows {
            // Phase flips when qubit was Y (both x and z set)
            row.sign ^= row.x[q] && row.z[q];
            // Swap x and z
            std::mem::swap(&mut row.x[q], &mut row.z[q]);
        }
    }

    /// Apply S gate on qubit q.
    /// X→Y, Y→-X, Z→Z.
    pub fn s(&mut self, q: usize) {
        for row in &mut self.rows {
            // Phase flips when qubit was Y (x=1, z=1 → becomes X with -1)
            row.sign ^= row.x[q] && row.z[q];
            // z' = z XOR x
            row.z[q] ^= row.x[q];
        }
    }

    /// Apply S† gate on qubit q.
    /// X→-Y, Y→X, Z→Z.
    pub fn sdg(&mut self, q: usize) {
        // S† = S^3 = S·S·S. Or: apply S, then flip sign when qubit was X.
        // Direct rule: phase flips when x=1 and z=0 (X→-Y)
        // Equivalently: S† transforms x→x, z→z XOR x, sign changes for X (not Y).
        for row in &mut self.rows {
            // For S†: sign flips when x=1 and z=0 (X→-Y, needs -1 factor)
            // But not when x=1,z=1 (Y→X, no sign change for S†)
            row.sign ^= row.x[q] && !row.z[q];
            row.z[q] ^= row.x[q];
        }
    }

    /// Apply `√X` (Stim's `SQRT_X`) on qubit q.
    /// `X → +X`, `Y → +Z`, `Z → −Y`.
    ///
    /// Closed form `x' = x XOR z, z' = z, sign ^= z AND NOT x`, derived from
    /// that action and checked against ALL FOUR Pauli inputs (I/X/Y/Z), not
    /// by analogy with `s()`. Stim's `Tableau::from_named_gate("SQRT_X")`
    /// gives the same `X → +X, Z → −Y`.
    pub fn sx(&mut self, q: usize) {
        for row in &mut self.rows {
            row.sign ^= row.z[q] && !row.x[q];
            row.x[q] ^= row.z[q];
        }
    }

    /// Apply `√X†` (Stim's `SQRT_X_DAG`) on qubit q.
    /// `X → +X`, `Y → −Z`, `Z → +Y`.
    ///
    /// Same closed form with the sign condition on `z AND x` instead — note
    /// this is NOT the `sx` rule with a blanket sign flip, which is why it is
    /// derived rather than copied.
    pub fn sxdg(&mut self, q: usize) {
        for row in &mut self.rows {
            row.sign ^= row.z[q] && row.x[q];
            row.x[q] ^= row.z[q];
        }
    }

    /// Apply CNOT gate (control a, target b).
    pub fn cx(&mut self, ctrl: usize, tgt: usize) {
        for row in &mut self.rows {
            // Phase update from Aaronson-Gottesman
            row.sign ^= row.x[ctrl] && row.z[tgt] && (row.x[tgt] ^ row.z[ctrl] ^ true);
            // Propagate X from control to target
            row.x[tgt] ^= row.x[ctrl];
            // Propagate Z from target to control
            row.z[ctrl] ^= row.z[tgt];
        }
    }

    /// Apply X gate on qubit q. X commutes with X, anti-commutes with Z and Y.
    pub fn x(&mut self, q: usize) {
        for row in &mut self.rows {
            // X flips sign when qubit has Z component (Z or Y)
            row.sign ^= row.z[q];
        }
    }

    /// Apply Y gate on qubit q.
    pub fn y(&mut self, q: usize) {
        for row in &mut self.rows {
            // Y flips sign when qubit has X or Z but not both (X or Z, not Y or I)
            row.sign ^= row.x[q] ^ row.z[q];
        }
    }

    /// Apply Z gate on qubit q.
    pub fn z(&mut self, q: usize) {
        for row in &mut self.rows {
            // Z flips sign when qubit has X component (X or Y)
            row.sign ^= row.x[q];
        }
    }

    /// Measure qubit q in the computational (Z) basis.
    /// Returns the measurement outcome (false=0, true=1).
    pub fn measure(&mut self, q: usize, rng: &mut impl Rng) -> bool {
        // Find a stabilizer that anti-commutes with Z_q
        let mut z_q = PauliRow::identity(self.n);
        z_q.z[q] = true;

        // Check stabilizer rows (indices n..2n)
        let anticommuting = (self.n..2 * self.n).find(|&i| self.rows[i].anticommutes(&z_q));

        match anticommuting {
            Some(p) => {
                // Random outcome: measurement is non-deterministic
                let outcome: bool = rng.random();

                // Update tableau following Aaronson-Gottesman Algorithm 2:
                // 1. For all rows i != p that anti-commute with Z_q, multiply by row p
                for i in 0..2 * self.n {
                    if i != p && self.rows[i].anticommutes(&z_q) {
                        self.rowmult(i, p);
                    }
                }
                // 2. Set destabilizer[p-n] = old stabilizer[p]
                let destab_idx = p - self.n;
                self.rows[destab_idx] = self.rows[p].clone();
                // 3. Set stabilizer[p] = ±Z_q (based on outcome)
                self.rows[p] = PauliRow::identity(self.n);
                self.rows[p].z[q] = true;
                self.rows[p].sign = outcome;

                outcome
            }
            None => {
                // Deterministic outcome: compute from stabilizers
                // The outcome is determined by whether Z_q can be expressed as
                // a product of stabilizers, with the sign giving ±1 eigenvalue.
                // Use the "scratch row" approach: multiply destabilizers that anti-commute.
                let mut scratch = PauliRow::identity(self.n);
                for i in 0..self.n {
                    if self.rows[i].anticommutes(&z_q) {
                        // Destabilizer i anti-commutes → need stabilizer i in the product
                        scratch = self.row_product(&scratch, &self.rows[self.n + i]);
                    }
                }
                scratch.sign
            }
        }
    }

    /// Multiply row i by row j (row i ← row i × row j).
    pub(crate) fn rowmult(&mut self, i: usize, j: usize) {
        let rj = self.rows[j].clone();
        let ri = &mut self.rows[i];
        // Phase from Pauli multiplication
        let mut phase_count = 0i32;
        for q in 0..self.n {
            // Count phase from multiplying single-qubit Paulis
            phase_count += Self::pauli_mult_phase(ri.x[q], ri.z[q], rj.x[q], rj.z[q]);
            ri.x[q] ^= rj.x[q];
            ri.z[q] ^= rj.z[q];
        }
        // Total sign: original signs XOR, plus phase contributions
        // phase_count mod 4: 0→+1, 2→-1 (sign flip)
        let extra_sign = ((phase_count % 4 + 4) % 4) == 2;
        ri.sign ^= rj.sign ^ extra_sign;
    }

    /// Compute product of two PauliRows (without modifying either).
    pub(crate) fn row_product(&self, a: &PauliRow, b: &PauliRow) -> PauliRow {
        let mut result = a.clone();
        let mut phase_count = 0i32;
        for q in 0..self.n {
            phase_count += Self::pauli_mult_phase(a.x[q], a.z[q], b.x[q], b.z[q]);
            result.x[q] ^= b.x[q];
            result.z[q] ^= b.z[q];
        }
        let extra_sign = ((phase_count % 4 + 4) % 4) == 2;
        result.sign ^= b.sign ^ extra_sign;
        result
    }

    /// Phase contribution from multiplying two single-qubit Paulis.
    /// Returns 0, 1, 2, or 3 (power of i).
    fn pauli_mult_phase(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
        pauli_mult_phase(x1, z1, x2, z2)
    }

    /// Get stabilizer generator k (index 0..n-1).
    pub fn stabilizer(&self, k: usize) -> &PauliRow {
        &self.rows[self.n + k]
    }

    /// Non-destructive Z-basis measurement on qubit `q`. Returns the
    /// `(p0, p1)` distribution without modifying the tableau.
    #[allow(dead_code)]
    // public API, exercised by tests; future callers
    // include the QML inference path's "check
    // determinism before collapsing" optimisation.
    ///
    /// For a stabilizer state, ⟨Z_q⟩ is either ±1 (deterministic, when
    /// `Z_q` is in the stabilizer group) or 0 (random over {0, 1} when
    /// some stabilizer anti-commutes with `Z_q`). The return values
    /// are therefore always one of `(1.0, 0.0)`, `(0.0, 1.0)`, or
    /// `(0.5, 0.5)` — anything else would be a tableau-invariant
    /// violation. Mirrors the read half of `measure` without the
    /// stabilizer-update / collapse half. Useful for interleaving the
    /// destructive `measure` with stabilizer-aware control logic (e.g.
    /// "skip a conditional X when the read-out is already pinned to
    /// 0").
    pub fn measure_prob(&self, q: usize) -> (f64, f64) {
        let mut z_q = PauliRow::identity(self.n);
        z_q.z[q] = true;

        let anticommuting = (self.n..2 * self.n).any(|i| self.rows[i].anticommutes(&z_q));
        if anticommuting {
            return (0.5, 0.5);
        }

        // Deterministic — the same algorithm `measure` uses on its
        // `None` branch, just read-only.
        let mut scratch = PauliRow::identity(self.n);
        for i in 0..self.n {
            if self.rows[i].anticommutes(&z_q) {
                scratch = self.row_product(&scratch, &self.rows[self.n + i]);
            }
        }
        if scratch.sign {
            (0.0, 1.0)
        } else {
            (1.0, 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_state_stabilizers() {
        let tab = StabilizerTableau::zero_state(3);
        // Stabilizers should be Z_0, Z_1, Z_2 with positive sign
        for q in 0..3 {
            let stab = tab.stabilizer(q);
            assert!(stab.z[q]);
            assert!(!stab.x[q]);
            assert!(!stab.sign);
        }
    }

    #[test]
    fn test_h_transforms_z_to_x() {
        let mut tab = StabilizerTableau::zero_state(1);
        // Stabilizer is Z_0
        assert!(tab.stabilizer(0).z[0]);
        assert!(!tab.stabilizer(0).x[0]);

        tab.h(0);
        // After H, stabilizer should be X_0
        assert!(!tab.stabilizer(0).z[0]);
        assert!(tab.stabilizer(0).x[0]);
        assert!(!tab.stabilizer(0).sign);
    }

    #[test]
    fn test_bell_state_measurement() {
        let mut tab = StabilizerTableau::zero_state(2);
        // Create Bell state: H(0), CX(0,1)
        tab.h(0);
        tab.cx(0, 1);

        // Stabilizers should be X_0 X_1 and Z_0 Z_1
        // After H(0): stabilizer 0 = X_0, stabilizer 1 = Z_1
        // After CX(0,1): X_0 → X_0 X_1, Z_1 → Z_0 Z_1
        let s0 = tab.stabilizer(0);
        assert!(s0.x[0] && s0.x[1] && !s0.z[0] && !s0.z[1]); // XX
        let s1 = tab.stabilizer(1);
        assert!(!s1.x[0] && !s1.x[1] && s1.z[0] && s1.z[1]); // ZZ

        // Measurement: outcomes should be correlated
        let mut rng = rand::rng();
        for _ in 0..20 {
            let mut tab2 = StabilizerTableau::zero_state(2);
            tab2.h(0);
            tab2.cx(0, 1);
            let m0 = tab2.measure(0, &mut rng);
            let m1 = tab2.measure(1, &mut rng);
            assert_eq!(m0, m1, "Bell state measurements should be correlated");
        }
    }

    #[test]
    fn test_x_gate_flips_state() {
        let mut tab = StabilizerTableau::zero_state(1);
        tab.x(0);
        // |1⟩ is stabilized by -Z
        let s = tab.stabilizer(0);
        assert!(s.z[0] && !s.x[0] && s.sign); // -Z
    }

    #[test]
    fn test_measure_prob_zero_state_deterministic_zero() {
        // |0⟩ is stabilized by +Z, so Z-measurement is pinned to 0.
        let tab = StabilizerTableau::zero_state(2);
        assert_eq!(tab.measure_prob(0), (1.0, 0.0));
        assert_eq!(tab.measure_prob(1), (1.0, 0.0));
    }

    #[test]
    fn test_measure_prob_after_x_deterministic_one() {
        // X|0⟩ = |1⟩, stabilizer is -Z, measurement pinned to 1.
        let mut tab = StabilizerTableau::zero_state(1);
        tab.x(0);
        assert_eq!(tab.measure_prob(0), (0.0, 1.0));
    }

    #[test]
    fn test_measure_prob_plus_state_uniform() {
        // H|0⟩ = |+⟩: stabilizer becomes X, which anti-commutes with Z,
        // so Z-measurement is uniform over {0, 1}.
        let mut tab = StabilizerTableau::zero_state(1);
        tab.h(0);
        assert_eq!(tab.measure_prob(0), (0.5, 0.5));
    }

    #[test]
    fn test_measure_prob_non_destructive() {
        // Calling measure_prob must not alter the tableau, so a
        // subsequent destructive `measure` should still see the
        // original distribution. Bell state on (q0,q1): measuring
        // q0 collapses both; if measure_prob mutated the state,
        // the Bell correlation would be lost.
        let mut tab = StabilizerTableau::zero_state(2);
        tab.h(0);
        tab.cx(0, 1);
        let probs_before = tab.measure_prob(0);
        let probs_after = tab.measure_prob(0);
        assert_eq!(probs_before, probs_after, "measure_prob must be idempotent");
        // Now a real measurement should still collapse Bell with the
        // correlation intact.
        let mut rng = rand::rng();
        let m0 = tab.measure(0, &mut rng);
        let m1 = tab.measure(1, &mut rng);
        assert_eq!(
            m0, m1,
            "Bell correlation must survive prior measure_prob calls"
        );
    }

    #[test]
    fn test_ghz_state() {
        let mut tab = StabilizerTableau::zero_state(3);
        tab.h(0);
        tab.cx(0, 1);
        tab.cx(1, 2);

        // GHZ stabilizers: XXX, ZZI, IZZ (up to signs)
        // More precisely: X₀X₁X₂, Z₀Z₁, Z₁Z₂
        let s0 = tab.stabilizer(0);
        assert!(s0.x[0] && s0.x[1] && s0.x[2]); // XXX

        // Measurements should be all-0 or all-1
        let mut rng = rand::rng();
        for _ in 0..20 {
            let mut tab2 = StabilizerTableau::zero_state(3);
            tab2.h(0);
            tab2.cx(0, 1);
            tab2.cx(1, 2);
            let m0 = tab2.measure(0, &mut rng);
            let m1 = tab2.measure(1, &mut rng);
            let m2 = tab2.measure(2, &mut rng);
            assert_eq!(m0, m1);
            assert_eq!(m1, m2);
        }
    }
}
