use aria_core::ast::{Circuit, CircuitBuilder};

/// Result of a syndrome measurement.
#[derive(Debug, Clone)]
pub struct SyndromeResult {
    pub syndrome: Vec<u8>,
    pub error_detected: bool,
    pub correction: Option<Vec<usize>>,
}

/// Trait for quantum error correcting codes.
pub trait QECCode {
    fn n_physical(&self) -> usize;
    fn n_logical(&self) -> usize;
    fn distance(&self) -> usize;
    fn encoding_circuit(&self) -> Circuit;
    fn syndrome_circuit(&self) -> Circuit;
    fn decode(&self, syndrome: &[u8]) -> Vec<usize>;
}

/// Bit-flip repetition code [[n, 1, n]].
pub struct RepetitionCode {
    n: usize,
}

impl RepetitionCode {
    pub fn new(n: usize) -> Self {
        assert!(n >= 3 && n % 2 == 1, "n must be odd and >= 3");
        Self { n }
    }
}

impl QECCode for RepetitionCode {
    fn n_physical(&self) -> usize {
        self.n
    }
    fn n_logical(&self) -> usize {
        1
    }
    fn distance(&self) -> usize {
        self.n
    }

    fn encoding_circuit(&self) -> Circuit {
        let mut b = CircuitBuilder::new("rep_encode", self.n, 0);
        for i in 1..self.n {
            b.cx(0, i);
        }
        b.build()
    }

    fn syndrome_circuit(&self) -> Circuit {
        let n_total = self.n + (self.n - 1);
        let n_classical = self.n - 1;
        let mut b = CircuitBuilder::new("rep_syndrome", n_total, n_classical);
        for i in 0..(self.n - 1) {
            let anc = self.n + i;
            b.cx(i, anc).cx(i + 1, anc).measure(anc, i);
        }
        b.build()
    }

    fn decode(&self, syndrome: &[u8]) -> Vec<usize> {
        let errors: Vec<usize> = syndrome
            .iter()
            .enumerate()
            .filter(|&(_, s)| *s == 1)
            .map(|(i, _)| i)
            .collect();

        if errors.is_empty() {
            return vec![];
        }
        if errors.len() == 1 {
            return if errors[0] == 0 {
                vec![0]
            } else {
                vec![errors[0]]
            };
        }
        if errors.len() == 2 && errors[1] == errors[0] + 1 {
            return vec![errors[0] + 1];
        }
        vec![errors[0]]
    }
}

/// Steane [[7, 1, 3]] code.
pub struct SteaneCode;

const X_STABS: [[usize; 4]; 3] = [[0, 2, 4, 6], [1, 2, 5, 6], [3, 4, 5, 6]];
const Z_STABS: [[usize; 4]; 3] = [[0, 2, 4, 6], [1, 2, 5, 6], [3, 4, 5, 6]];

impl SteaneCode {
    /// X-type stabilizer supports (detect Z errors).
    pub fn x_checks(&self) -> Vec<Vec<usize>> {
        X_STABS.iter().map(|s| s.to_vec()).collect()
    }
    /// Z-type stabilizer supports (detect X errors).
    pub fn z_checks(&self) -> Vec<Vec<usize>> {
        Z_STABS.iter().map(|s| s.to_vec()).collect()
    }
    /// Logical-X observable support. The all-qubit representative `X^⊗7` is a
    /// valid logical X: it overlaps every weight-4 stabilizer evenly (commutes)
    /// and the logical-Z representative oddly (anticommutes). Any minimal
    /// weight-3 representative differs from it by a stabilizer, so it reads the
    /// same ±1 on a codespace state.
    pub fn logical_x(&self) -> Vec<usize> {
        (0..7).collect()
    }
    /// Logical-Z observable support (`Z^⊗7`; see [`Self::logical_x`]).
    pub fn logical_z(&self) -> Vec<usize> {
        (0..7).collect()
    }
}

impl QECCode for SteaneCode {
    fn n_physical(&self) -> usize {
        7
    }
    fn n_logical(&self) -> usize {
        1
    }
    fn distance(&self) -> usize {
        3
    }

    fn encoding_circuit(&self) -> Circuit {
        let mut b = CircuitBuilder::new("steane_encode", 7, 0);
        b.h(1).h(2).h(3);
        b.cx(0, 4).cx(0, 5).cx(0, 6);
        b.cx(1, 4).cx(1, 5);
        b.cx(2, 4).cx(2, 6);
        b.cx(3, 5).cx(3, 6);
        b.build()
    }

    fn syndrome_circuit(&self) -> Circuit {
        let mut b = CircuitBuilder::new("steane_syndrome", 13, 6); // 7 data + 6 ancilla
                                                                   // X stabilizer measurements (ancilla 7, 8, 9)
        for (s_idx, stab) in X_STABS.iter().enumerate() {
            let anc = 7 + s_idx;
            b.h(anc);
            for &data_q in stab {
                b.cx(anc, data_q);
            }
            b.h(anc);
            b.measure(anc, s_idx);
        }
        // Z stabilizer measurements (ancilla 10, 11, 12)
        for (s_idx, stab) in Z_STABS.iter().enumerate() {
            let anc = 10 + s_idx;
            for &data_q in stab {
                b.cx(data_q, anc);
            }
            b.measure(anc, 3 + s_idx);
        }
        b.build()
    }

    fn decode(&self, syndrome: &[u8]) -> Vec<usize> {
        if syndrome.len() != 6 {
            return vec![];
        }
        let sx = &syndrome[..3];
        let sz = &syndrome[3..];

        fn hamming_decode(s: &[u8]) -> Option<usize> {
            let idx = s[0] as usize + 2 * s[1] as usize + 4 * s[2] as usize;
            if idx == 0 {
                return None;
            }
            let map = [0, 0, 1, 2, 3, 4, 5, 6]; // idx -> qubit
            Some(map[idx])
        }

        let mut corrections = vec![];
        if let Some(z_err) = hamming_decode(sx) {
            corrections.push(z_err);
        }
        if let Some(x_err) = hamming_decode(sz) {
            if !corrections.contains(&x_err) {
                corrections.push(x_err);
            }
        }
        corrections
    }
}

/// Build the X-type and Z-type check supports of a rotated surface code of
/// odd distance `d`. Data qubits sit on a d×d grid, index `row*d + col`.
///
/// Interior weight-4 plaquettes are checkerboard-colored (Z on even faces,
/// X on odd); weight-2 boundary checks close the lattice — Z on the top/bottom
/// (rough) boundaries, X on the left/right (smooth) boundaries. The result is
/// a genuine `[[d², 1, d]]` CSS code: X/Z checks pairwise commute, a vertical
/// column is a logical Z, a horizontal row a logical X, and the minimum
/// logical weight is `d`. All of this is asserted in the unit tests.
fn build_checks(d: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let idx = |r: usize, c: usize| r * d + c;
    let di = d as i64;
    let mut x_checks: Vec<Vec<usize>> = Vec::new();
    let mut z_checks: Vec<Vec<usize>> = Vec::new();

    // Interior weight-4 plaquettes, checkerboard-colored.
    for i in 0..d - 1 {
        for j in 0..d - 1 {
            let sup = vec![idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)];
            if (i + j) % 2 == 0 {
                z_checks.push(sup);
            } else {
                x_checks.push(sup);
            }
        }
    }
    // Weight-2 boundary checks. Top/bottom → Z; left/right → X.
    for j in 0..d - 1 {
        if (-1 + j as i64).rem_euclid(2) == 0 {
            z_checks.push(vec![idx(0, j), idx(0, j + 1)]);
        }
        if ((di - 1) + j as i64).rem_euclid(2) == 0 {
            z_checks.push(vec![idx(d - 1, j), idx(d - 1, j + 1)]);
        }
    }
    for i in 0..d - 1 {
        if (i as i64 - 1).rem_euclid(2) != 0 {
            x_checks.push(vec![idx(i, 0), idx(i + 1, 0)]);
        }
        if (i as i64 + (di - 1)).rem_euclid(2) != 0 {
            x_checks.push(vec![idx(i, d - 1), idx(i + 1, d - 1)]);
        }
    }
    (x_checks, z_checks)
}

/// Rotated surface code `[[d², 1, d]]` (CSS).
///
/// The smallest non-trivial instance is d=3 → `[[9,1,3]]` (the "surface-17"
/// code: 9 data + 8 ancilla). Data qubits live on a d×d grid, index
/// `row*d + col`. See [`build_checks`] for the stabilizer geometry.
///
/// Syndrome extraction is split into two deterministic sectors so that every
/// simulator backend agrees on the outcome bit-for-bit:
/// * **bit-flip sector** — data in |0…0⟩, measure the Z-type checks; detects X
///   (bit-flip) errors. Z-checks are diagonal so the syndrome is deterministic.
/// * **phase-flip sector** — data in |+…+⟩, measure the X-type checks; detects
///   Z (phase-flip) errors (the Hadamard dual of the above).
pub struct SurfaceCode {
    d: usize,
    x_checks: Vec<Vec<usize>>,
    z_checks: Vec<Vec<usize>>,
}

impl SurfaceCode {
    pub fn new(d: usize) -> Self {
        assert!(d >= 3 && d % 2 == 1, "d must be odd and >= 3");
        let (x_checks, z_checks) = build_checks(d);
        Self {
            d,
            x_checks,
            z_checks,
        }
    }

    /// Number of data qubits (`d²`).
    pub fn n_data(&self) -> usize {
        self.d * self.d
    }

    /// X-type check supports (detect Z errors).
    pub fn x_checks(&self) -> &[Vec<usize>] {
        &self.x_checks
    }

    /// Z-type check supports (detect X errors).
    pub fn z_checks(&self) -> &[Vec<usize>] {
        &self.z_checks
    }

    /// Total syndrome ancilla qubits (= number of stabilizer checks).
    pub fn n_ancilla(&self) -> usize {
        self.x_checks.len() + self.z_checks.len()
    }

    /// Logical-Z observable support: the leftmost column.
    pub fn logical_z(&self) -> Vec<usize> {
        (0..self.d).map(|r| r * self.d).collect()
    }

    /// Logical-X observable support: the top row.
    pub fn logical_x(&self) -> Vec<usize> {
        (0..self.d).collect()
    }
}

impl QECCode for SurfaceCode {
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
        // Logical |0⟩ is the +1 eigenstate of every Z-type check, which for the
        // rotated code is simply |0…0⟩ — the data register needs no gates.
        CircuitBuilder::new("surface_encode", self.n_data(), 0).build()
    }

    /// Full syndrome-extraction circuit (X-type checks into classical bits
    /// `0..n_x`, then Z-type checks into `n_x..n_x+n_z`). Provided for
    /// inspection (`quantum info`) and for the decoder's bit ordering; the
    /// *deterministic* per-sector extraction the simulator uses lives in
    /// `ecc::run`.
    fn syndrome_circuit(&self) -> Circuit {
        let n_data = self.n_data();
        let n_total = n_data + self.n_ancilla();
        let n_classical = self.n_ancilla();
        let mut b = CircuitBuilder::new("surface_syndrome", n_total, n_classical);
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

    /// Combined min-weight correction for a full syndrome vector (X-check bits
    /// then Z-check bits). Delegates to the MWPM-equivalent exact decoder.
    fn decode(&self, syndrome: &[u8]) -> Vec<usize> {
        let corr = super::mwpm::decode_mwpm_correction(self, syndrome);
        let mut all = corr.x_flips;
        all.extend(corr.z_flips);
        all.sort_unstable();
        all.dedup();
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_code() {
        let code = RepetitionCode::new(3);
        assert_eq!(code.n_physical(), 3);
        assert_eq!(code.n_logical(), 1);
        assert_eq!(code.distance(), 3);
        assert_eq!(code.encoding_circuit().gate_count(), 2);
        assert_eq!(code.syndrome_circuit().n_qubits(), 5);
        assert!(code.decode(&[0, 0]).is_empty());
        assert_eq!(code.decode(&[1, 0]).len(), 1);
    }

    #[test]
    fn test_steane_code() {
        let code = SteaneCode;
        assert_eq!(code.n_physical(), 7);
        assert_eq!(code.distance(), 3);
        assert_eq!(code.encoding_circuit().n_qubits(), 7);
        assert_eq!(code.syndrome_circuit().n_qubits(), 13);
        assert!(code.decode(&[0, 0, 0, 0, 0, 0]).is_empty());
    }

    #[test]
    fn test_surface_code() {
        let code = SurfaceCode::new(3);
        assert_eq!(code.n_physical(), 9);
        assert_eq!(code.n_logical(), 1);
        assert_eq!(code.distance(), 3);

        // A real [[9,1,3]] has 4 X-checks + 4 Z-checks.
        assert_eq!(code.x_checks().len(), 4);
        assert_eq!(code.z_checks().len(), 4);
        assert_eq!(code.n_ancilla(), 8);
        // n_data - n_checks = 1 logical qubit.
        assert_eq!(code.n_data() - code.n_ancilla(), 1);

        let enc = code.encoding_circuit();
        assert_eq!(enc.n_qubits(), 9);

        let syn = code.syndrome_circuit();
        assert_eq!(syn.n_qubits(), 17); // 9 data + 8 ancilla
        assert!(syn.gate_count() > 0);

        // No-error syndrome decodes to no correction.
        let no_err = vec![0u8; code.n_ancilla()];
        assert!(code.decode(&no_err).is_empty());
    }

    #[test]
    fn surface_code_is_valid_css() {
        // X-type and Z-type checks must pairwise commute (even overlap).
        // d=7 is included so the "PauliProp reaches d ≥ 7" claim (see
        // `ecc::run::SimBackend::PauliProp`) rests on a code that is actually
        // asserted to be a valid `[[d²,1,d]]` CSS code, not just constructed.
        for d in [3usize, 5, 7] {
            let code = SurfaceCode::new(d);
            for xc in code.x_checks() {
                let xs: std::collections::BTreeSet<usize> = xc.iter().copied().collect();
                for zc in code.z_checks() {
                    let overlap = zc.iter().filter(|q| xs.contains(q)).count();
                    assert_eq!(overlap % 2, 0, "non-commuting checks at d={d}");
                }
            }
            // Stabilizer count = d² - 1 (one logical qubit).
            assert_eq!(code.x_checks().len() + code.z_checks().len(), d * d - 1);
            // Logical operators have weight d and anticommute (overlap 1).
            let lx: std::collections::BTreeSet<usize> = code.logical_x().into_iter().collect();
            let lz: std::collections::BTreeSet<usize> = code.logical_z().into_iter().collect();
            assert_eq!(lx.len(), d);
            assert_eq!(lz.len(), d);
            assert_eq!(lx.iter().filter(|q| lz.contains(q)).count(), 1);
            // Logical Z commutes with every X-check; logical X with every Z-check.
            for xc in code.x_checks() {
                assert_eq!(xc.iter().filter(|q| lz.contains(q)).count() % 2, 0);
            }
            for zc in code.z_checks() {
                assert_eq!(zc.iter().filter(|q| lx.contains(q)).count() % 2, 0);
            }
        }
    }
}
