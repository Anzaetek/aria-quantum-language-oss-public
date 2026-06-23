//! QUBO (Quadratic Unconstrained Binary Optimization) and Ising model.
//!
//! QUBO: minimize f(x) = x^T Q x, where x ∈ {0,1}^n.
//! Ising: minimize H(s) = Σ_{i<j} J_{ij} s_i s_j + Σ_i h_i s_i, where s ∈ {-1,+1}^n.
//!
//! Conversion: x = (1 - s) / 2, mapping QUBO ↔ Ising.

use std::collections::HashMap;

use crate::executor::{Observable, PauliOp};

/// A QUBO problem: minimize x^T Q x for binary x ∈ {0,1}^n.
///
/// Q is stored as a sparse upper-triangular map: (i, j) → coefficient.
/// Diagonal entries Q_{ii} are linear terms, off-diagonal Q_{ij} (i<j) are quadratic.
#[derive(Clone, Debug)]
pub struct Qubo {
    /// Number of binary variables.
    pub n: usize,
    /// Sparse upper-triangular Q matrix: (i, j) → coefficient, with i <= j.
    pub entries: HashMap<(usize, usize), f64>,
}

/// Ising model: H = Σ_{i<j} J_{ij} Z_i Z_j + Σ_i h_i Z_i + constant.
#[derive(Clone, Debug)]
pub struct IsingModel {
    pub n: usize,
    /// Coupling terms J_{ij} for i < j.
    pub couplings: HashMap<(usize, usize), f64>,
    /// Local field terms h_i.
    pub fields: Vec<f64>,
    /// Constant energy offset from the QUBO → Ising conversion.
    pub offset: f64,
}

impl Qubo {
    /// Create an empty QUBO problem on n variables.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            entries: HashMap::new(),
        }
    }

    /// Set Q_{i,j}. Automatically normalises to i <= j.
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        if value.abs() < 1e-15 {
            self.entries.remove(&(a, b));
        } else {
            self.entries.insert((a, b), value);
        }
    }

    /// Add to Q_{i,j}. Automatically normalises to i <= j.
    pub fn add(&mut self, i: usize, j: usize, value: f64) {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        let entry = self.entries.entry((a, b)).or_insert(0.0);
        *entry += value;
    }

    /// Evaluate f(x) = x^T Q x for a given binary assignment.
    pub fn evaluate(&self, x: &[bool]) -> f64 {
        let mut total = 0.0;
        for (&(i, j), &coeff) in &self.entries {
            if x[i] && x[j] {
                total += coeff;
            }
        }
        total
    }

    /// Convert QUBO to Ising model.
    ///
    /// Using x_i = (1 - s_i) / 2:
    ///   Q_{ii} x_i = Q_{ii} (1 - s_i)/2
    ///   Q_{ij} x_i x_j = Q_{ij} (1 - s_i)(1 - s_j)/4
    pub fn to_ising(&self) -> IsingModel {
        let mut couplings: HashMap<(usize, usize), f64> = HashMap::new();
        let mut fields = vec![0.0; self.n];
        let mut offset = 0.0;

        for (&(i, j), &q) in &self.entries {
            if i == j {
                // Diagonal: Q_{ii} * x_i = Q_{ii} * (1 - s_i)/2
                //         = Q_{ii}/2 - Q_{ii}/2 * s_i
                offset += q / 2.0;
                fields[i] -= q / 2.0;
            } else {
                // Off-diagonal: Q_{ij} * x_i * x_j = Q_{ij} * (1-s_i)(1-s_j)/4
                //             = Q_{ij}/4 * (1 - s_i - s_j + s_i*s_j)
                //             = Q_{ij}/4 + Q_{ij}/4 * s_i*s_j - Q_{ij}/4 * s_i - Q_{ij}/4 * s_j
                offset += q / 4.0;
                *couplings.entry((i, j)).or_insert(0.0) += q / 4.0;
                fields[i] -= q / 4.0;
                fields[j] -= q / 4.0;
            }
        }

        IsingModel {
            n: self.n,
            couplings,
            fields,
            offset,
        }
    }

    /// Brute-force solve for small problems (n <= 20).
    /// Returns (best_assignment, best_value).
    pub fn brute_force(&self) -> (Vec<bool>, f64) {
        assert!(self.n <= 20, "brute-force limited to n <= 20");
        let dim = 1u64 << self.n;
        let mut best_val = f64::INFINITY;
        let mut best_bits = 0u64;

        for bits in 0..dim {
            let x: Vec<bool> = (0..self.n).map(|q| (bits >> q) & 1 == 1).collect();
            let val = self.evaluate(&x);
            if val < best_val {
                best_val = val;
                best_bits = bits;
            }
        }

        let best_x: Vec<bool> = (0..self.n).map(|q| (best_bits >> q) & 1 == 1).collect();
        (best_x, best_val)
    }

    /// Parse the on-wire JSON form `{"n": N, "Q": [[i, j, c], ...]}`.
    ///
    /// Each `Q` row is a sparse `(i, j, coefficient)` triple. Diagonal
    /// (`i == j`) rows are linear; off-diagonal rows are quadratic. Indices
    /// are normalised to `i <= j` automatically by `Qubo::add`. Multiple
    /// rows with the same `(i, j)` are summed.
    pub fn from_json(s: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(s).map_err(|e| format!("parse JSON: {e}"))?;
        let n = v.get("n").and_then(|x| x.as_u64()).ok_or("missing 'n'")? as usize;
        let q_arr = v
            .get("Q")
            .and_then(|x| x.as_array())
            .ok_or("missing 'Q' array")?;
        let mut q = Qubo::new(n);
        for entry in q_arr {
            let arr = entry.as_array().ok_or("Q entry not an array")?;
            if arr.len() != 3 {
                return Err("Q entry must be [i, j, c]".into());
            }
            let i = arr[0].as_u64().ok_or("Q[0] not int")? as usize;
            let j = arr[1].as_u64().ok_or("Q[1] not int")? as usize;
            let c = arr[2].as_f64().ok_or("Q[2] not float")?;
            if i >= n || j >= n {
                return Err(format!("index ({i},{j}) out of range for n={n}"));
            }
            q.add(i, j, c);
        }
        Ok(q)
    }
}

impl IsingModel {
    /// Convert the Ising Hamiltonian to a Pauli Observable for quantum simulation.
    ///
    /// H = Σ J_{ij} Z_i Z_j + Σ h_i Z_i + offset·I
    pub fn to_observable(&self) -> Observable {
        let mut terms = Vec::new();

        // Constant / identity term
        if self.offset.abs() > 1e-15 {
            terms.push((self.offset, vec![]));
        }

        // Local field terms: h_i Z_i
        for (i, &h) in self.fields.iter().enumerate() {
            if h.abs() > 1e-15 {
                terms.push((h, vec![(i as u32, PauliOp::Z)]));
            }
        }

        // Coupling terms: J_{ij} Z_i Z_j
        for (&(i, j), &jval) in &self.couplings {
            if jval.abs() > 1e-15 {
                terms.push((jval, vec![(i as u32, PauliOp::Z), (j as u32, PauliOp::Z)]));
            }
        }

        Observable { terms }
    }

    /// Evaluate H(s) for a given spin assignment (true = +1, false = -1).
    pub fn evaluate(&self, spins: &[bool]) -> f64 {
        let s = |i: usize| -> f64 {
            if spins[i] {
                1.0
            } else {
                -1.0
            }
        };

        let mut total = self.offset;
        for (&(i, j), &jval) in &self.couplings {
            total += jval * s(i) * s(j);
        }
        for (i, &h) in self.fields.iter().enumerate() {
            total += h * s(i);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qubo_evaluate() {
        // f(x) = -x_0 - x_1 + 2*x_0*x_1
        // Minimum at (1,0) or (0,1) with value -1
        let mut q = Qubo::new(2);
        q.set(0, 0, -1.0); // -x_0
        q.set(1, 1, -1.0); // -x_1
        q.set(0, 1, 2.0); // 2*x_0*x_1

        assert_eq!(q.evaluate(&[false, false]), 0.0);
        assert_eq!(q.evaluate(&[true, false]), -1.0);
        assert_eq!(q.evaluate(&[false, true]), -1.0);
        assert_eq!(q.evaluate(&[true, true]), 0.0);
    }

    #[test]
    fn test_qubo_brute_force() {
        let mut q = Qubo::new(3);
        q.set(0, 0, -1.0);
        q.set(1, 1, -1.0);
        q.set(2, 2, -1.0);
        q.set(0, 1, 2.0);

        let (best, val) = q.brute_force();
        // Best: x_2=1 and one of (x_0,x_1) = (1,0) or (0,1): value = -2
        assert_eq!(val, -2.0);
        assert!(best[2]);
    }

    #[test]
    fn test_qubo_to_ising_roundtrip() {
        let mut q = Qubo::new(2);
        q.set(0, 0, -1.0);
        q.set(1, 1, -2.0);
        q.set(0, 1, 3.0);

        let ising = q.to_ising();

        // Check that Ising evaluation matches QUBO for all 4 assignments
        for bits in 0..4u64 {
            let x: Vec<bool> = (0..2).map(|i| (bits >> i) & 1 == 1).collect();
            let qubo_val = q.evaluate(&x);

            // Convert x → s: s_i = 1 - 2*x_i
            let spins: Vec<bool> = x.iter().map(|&xi| !xi).collect();
            let ising_val = ising.evaluate(&spins);

            assert!(
                (qubo_val - ising_val).abs() < 1e-10,
                "QUBO({:?})={qubo_val} != Ising({:?})={ising_val}",
                x,
                spins
            );
        }
    }

    #[test]
    fn test_ising_to_observable() {
        let mut q = Qubo::new(2);
        q.set(0, 0, -1.0);
        q.set(0, 1, 2.0);

        let ising = q.to_ising();
        let obs = ising.to_observable();

        // Observable should have ZZ and Z terms
        assert!(!obs.terms.is_empty());
    }
}
