//! Matrix Product State representation and operations.
//!
//! An n-qubit state is represented as a chain of tensors:
//!   A[0] (1 x 2 x d1), A[1] (d1 x 2 x d2), ..., A[n-1] (d_{n-1} x 2 x 1)
//! where d_i is the bond dimension between sites i and i+1.

use num_complex::Complex64;
use rand::RngExt;

use crate::svd::{truncated_svd_flat, SvdResultFlat};

/// A single MPS tensor: shape (bond_left, physical=2, bond_right).
/// Stored as a flat array in row-major order.
#[derive(Clone, Debug)]
pub struct MpsTensor {
    pub bond_left: usize,
    pub bond_right: usize,
    /// Data indexed as [left * 2 * bond_right + phys * bond_right + right]
    pub data: Vec<Complex64>,
}

impl MpsTensor {
    pub fn new(bond_left: usize, bond_right: usize) -> Self {
        Self {
            bond_left,
            bond_right,
            data: vec![Complex64::new(0.0, 0.0); bond_left * 2 * bond_right],
        }
    }

    pub fn get(&self, left: usize, phys: usize, right: usize) -> Complex64 {
        self.data[left * 2 * self.bond_right + phys * self.bond_right + right]
    }

    pub fn set(&mut self, left: usize, phys: usize, right: usize, val: Complex64) {
        self.data[left * 2 * self.bond_right + phys * self.bond_right + right] = val;
    }
}

/// Matrix Product State for n qubits.
pub struct Mps {
    pub tensors: Vec<MpsTensor>,
    pub n: usize,
    pub max_bond_dim: usize,
}

impl Mps {
    /// Create |00...0> state as MPS with bond dimension 1.
    pub fn zero_state(n: usize, max_bond_dim: usize) -> Self {
        let mut tensors = Vec::with_capacity(n);
        for _ in 0..n {
            let mut t = MpsTensor::new(1, 1);
            t.set(0, 0, 0, Complex64::new(1.0, 0.0)); // |0>
            tensors.push(t);
        }
        Self {
            tensors,
            n,
            max_bond_dim,
        }
    }

    /// Apply a single-qubit gate (2x2 matrix) to qubit q.
    pub fn apply_1q(&mut self, q: usize, gate: &[Complex64; 4]) {
        let t = &self.tensors[q];
        let bl = t.bond_left;
        let br = t.bond_right;
        let mut new_data = vec![Complex64::new(0.0, 0.0); bl * 2 * br];

        for l in 0..bl {
            for r in 0..br {
                let a0 = t.get(l, 0, r);
                let a1 = t.get(l, 1, r);
                // new[l,s',r] = sum_s gate[s',s] * old[l,s,r]
                new_data[l * 2 * br + r] = gate[0] * a0 + gate[1] * a1;
                new_data[l * 2 * br + br + r] = gate[2] * a0 + gate[3] * a1;
            }
        }

        self.tensors[q].data = new_data;
    }

    /// Apply a two-qubit gate (4x4 matrix) to adjacent qubits (q, q+1).
    /// Uses SVD to split the result back into two tensors, truncating to max_bond_dim.
    pub fn apply_2q(&mut self, q: usize, gate: &[Complex64; 16]) {
        self.apply_2q_with_svd_flat(q, gate, |m, m_dim, n, stride, max_rank, threshold| {
            truncated_svd_flat(m, m_dim, n, stride, max_rank, threshold)
        });
    }

    /// Apply a two-qubit gate with a caller-supplied flat-buffer SVD
    /// function. The closure receives the contracted-and-gated `Θ'`
    /// reshaped as a flat row-major matrix of shape `(bl*2) × (2*br)`
    /// — no nested `Vec<Vec<_>>` allocation. The CUDA backend wires
    /// `CudaSvdContext::truncated_svd_flat` in through this hook.
    ///
    /// Closure args: `(matrix, m, n, stride, max_rank, threshold)`,
    /// where `matrix[i * stride + j]` is element `(i, j)`.
    pub fn apply_2q_with_svd_flat<F>(&mut self, q: usize, gate: &[Complex64; 16], svd_fn: F)
    where
        F: FnOnce(&[Complex64], usize, usize, usize, usize, f64) -> SvdResultFlat,
    {
        assert!(q + 1 < self.n, "q+1 out of range");

        let tl = &self.tensors[q];
        let tr = &self.tensors[q + 1];
        let bl = tl.bond_left;
        let bm = tl.bond_right; // = tr.bond_left
        let br = tr.bond_right;

        // The matrix layout the SVD consumes is `(bl*2) × (2*br)`
        // row-major. Equivalently — and this is the key for the
        // flat-buffer refactor — that's the *same* memory layout as
        // `Theta'[l, s0, s1, r]` with strides `(4*br, 2*br, br, 1)`:
        //
        //   Theta'[l, s0, s1, r]
        //     = matrix[l*2 + s0][s1*br + r]
        //     = matrix_flat[(l*2 + s0) * (2*br) + s1*br + r]
        //     = matrix_flat[l*4*br + s0*2*br + s1*br + r]
        //
        // So we build Theta' directly as the SVD's input buffer and
        // skip the host-side reshape that the old API needed.

        // Step 1: Contract two sites into Theta[l, s0, s1, r]
        let mut theta = vec![Complex64::new(0.0, 0.0); bl * 4 * br];
        for l in 0..bl {
            for s0 in 0..2 {
                for s1 in 0..2 {
                    for r in 0..br {
                        let mut val = Complex64::new(0.0, 0.0);
                        for m in 0..bm {
                            val += tl.get(l, s0, m) * tr.get(m, s1, r);
                        }
                        theta[l * 4 * br + s0 * 2 * br + s1 * br + r] = val;
                    }
                }
            }
        }

        // Step 2: Apply gate to physical indices. We write directly
        // into the row-major matrix layout consumed by Step 3 (SVD).
        // Note that `Theta'[l, s_out, r] = matrix_flat[(l*2+s0_out)*(2*br) + s1_out*br + r]`
        // is the same flat layout as `Theta[l, s_in, r]` indexed by
        // `(l*2+s0_in)*(2*br) + s1_in*br + r`, so the two share an
        // identical stride pattern — only the gate row index differs.
        let cols = 2 * br;
        let mut matrix_flat = vec![Complex64::new(0.0, 0.0); bl * 4 * br];
        for l in 0..bl {
            let theta_base = l * 4 * br;
            let mat_base = l * 2 * cols;
            for r in 0..br {
                for ss_out in 0..4 {
                    let mut val = Complex64::new(0.0, 0.0);
                    for ss_in in 0..4 {
                        val += gate[ss_out * 4 + ss_in] * theta[theta_base + ss_in * br + r];
                    }
                    let s0_out = ss_out >> 1;
                    let s1_out = ss_out & 1;
                    matrix_flat[mat_base + s0_out * cols + s1_out * br + r] = val;
                }
            }
        }
        drop(theta);

        // Step 3: SVD and truncate (CPU or GPU via the injected fn).
        let rows = bl * 2;
        let svd = svd_fn(&matrix_flat, rows, cols, cols, self.max_bond_dim, 1e-14);
        let new_bm = svd.s.len();

        // Step 4: Build new tensors directly from flat U / Vt buffers.
        // A[l, s0, m] = U[(l*2+s0) * new_bm + m] * sqrt(S[m])
        // B[m, s1, r] = sqrt(S[m]) * Vt[m * cols + s1*br + r]
        let mut new_tl = MpsTensor::new(bl, new_bm);
        let mut new_tr = MpsTensor::new(new_bm, br);

        for l in 0..bl {
            for s0 in 0..2 {
                for m in 0..new_bm {
                    let sqrt_s = svd.s[m].sqrt();
                    new_tl.set(l, s0, m, svd.u[(l * 2 + s0) * new_bm + m] * sqrt_s);
                }
            }
        }

        for m in 0..new_bm {
            let sqrt_s = svd.s[m].sqrt();
            for s1 in 0..2 {
                for r in 0..br {
                    new_tr.set(m, s1, r, sqrt_s * svd.vt[m * cols + s1 * br + r]);
                }
            }
        }

        self.tensors[q] = new_tl;
        self.tensors[q + 1] = new_tr;
    }

    /// Apply a two-qubit gate to non-adjacent qubits by SWAPping to make them adjacent.
    pub fn apply_2q_distant(&mut self, q0: usize, q1: usize, gate: &[Complex64; 16]) {
        let (qa, qb) = if q0 < q1 { (q0, q1) } else { (q1, q0) };

        if qb - qa == 1 {
            // Already adjacent
            if q0 < q1 {
                self.apply_2q(qa, gate);
            } else {
                // Need to swap qubit indices in the gate
                let mut swapped = *gate;
                // Swap rows 1,2 and cols 1,2
                for col in 0..4 {
                    swapped.swap(4 + col, 2 * 4 + col);
                }
                for row in 0..4 {
                    swapped.swap(row * 4 + 1, row * 4 + 2);
                }
                self.apply_2q(qa, &swapped);
            }
            return;
        }

        // SWAP chain: move qa next to qb
        let swap_gate = swap_matrix();
        for i in qa..qb - 1 {
            self.apply_2q(i, &swap_gate);
        }
        // Apply gate to (qb-1, qb)
        if q0 < q1 {
            self.apply_2q(qb - 1, gate);
        } else {
            let mut swapped = *gate;
            for col in 0..4 {
                swapped.swap(4 + col, 2 * 4 + col);
            }
            for row in 0..4 {
                swapped.swap(row * 4 + 1, row * 4 + 2);
            }
            self.apply_2q(qb - 1, &swapped);
        }
        // SWAP back
        for i in (qa..qb - 1).rev() {
            self.apply_2q(i, &swap_gate);
        }
    }

    /// Compute the full statevector by contracting all tensors.
    /// Only feasible for small n (testing).
    pub fn to_statevector(&self) -> Vec<Complex64> {
        let dim = 1usize << self.n;
        let mut sv = vec![Complex64::new(0.0, 0.0); dim];

        for basis in 0..dim {
            // For each tensor, the physical index is the bit of `basis`
            let mut left_vec = vec![Complex64::new(1.0, 0.0)]; // 1x1 initial

            for q in 0..self.n {
                let bit = (basis >> q) & 1;
                let t = &self.tensors[q];
                let mut new_left = vec![Complex64::new(0.0, 0.0); t.bond_right];
                for r in 0..t.bond_right {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for l in 0..t.bond_left {
                        sum += left_vec[l] * t.get(l, bit, r);
                    }
                    new_left[r] = sum;
                }
                left_vec = new_left;
            }

            sv[basis] = left_vec[0];
        }

        sv
    }

    /// Sample a measurement outcome.
    pub fn sample(&self, rng: &mut impl rand::Rng) -> u64 {
        // Sequential sampling: measure each qubit from left to right
        let mut result = 0u64;
        let mut left_vec = vec![Complex64::new(1.0, 0.0)];

        for q in 0..self.n {
            let t = &self.tensors[q];
            // Compute probability of qubit q being 0
            let mut p0 = 0.0;
            let mut p1 = 0.0;

            for bit in 0..2 {
                let mut prob = 0.0;
                for r in 0..t.bond_right {
                    let mut val = Complex64::new(0.0, 0.0);
                    for l in 0..t.bond_left {
                        val += left_vec[l] * t.get(l, bit, r);
                    }
                    prob += val.norm_sqr();
                }
                if bit == 0 {
                    p0 = prob;
                } else {
                    p1 = prob;
                }
            }

            let total = p0 + p1;
            let bit = if rng.random::<f64>() < p0 / total {
                0
            } else {
                1
            };
            result |= (bit as u64) << q;

            // Update left_vec conditioned on measurement outcome
            let norm = if bit == 0 { p0.sqrt() } else { p1.sqrt() };
            let mut new_left = vec![Complex64::new(0.0, 0.0); t.bond_right];
            for r in 0..t.bond_right {
                let mut val = Complex64::new(0.0, 0.0);
                for l in 0..t.bond_left {
                    val += left_vec[l] * t.get(l, bit as usize, r);
                }
                new_left[r] = val / norm;
            }
            left_vec = new_left;
        }

        result
    }

    /// Mid-circuit projective measurement of a single qubit.
    /// Computes P(q=0), samples outcome, projects the local tensor,
    /// and renormalizes. Returns the measurement outcome (0 or 1).
    pub fn measure_site(&mut self, q: usize, rng: &mut impl rand::Rng) -> u8 {
        let t = &self.tensors[q];
        let bl = t.bond_left;
        let br = t.bond_right;

        // Compute norms for physical dim 0 and 1
        let mut norm_sq = [0.0_f64; 2];
        for phys in 0..2 {
            for l in 0..bl {
                for r in 0..br {
                    norm_sq[phys] += t.get(l, phys, r).norm_sqr();
                }
            }
        }

        let total = norm_sq[0] + norm_sq[1];
        let p0 = if total > 0.0 { norm_sq[0] / total } else { 0.5 };
        let outcome: u8 = if rng.random::<f64>() < p0 { 0 } else { 1 };

        // Project: zero out the other physical dimension and renormalize
        let norm = norm_sq[outcome as usize].sqrt();
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 1.0 };

        let new_data: Vec<Complex64> = (0..bl * 2 * br)
            .map(|idx| {
                let phys = (idx / br) % 2;
                if phys == outcome as usize {
                    self.tensors[q].data[idx] * inv_norm
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();
        self.tensors[q].data = new_data;

        outcome
    }
}

fn swap_matrix() -> [Complex64; 16] {
    let o = Complex64::new(0.0, 0.0);
    let i = Complex64::new(1.0, 0.0);
    [
        i, o, o, o, // |00> -> |00>
        o, o, i, o, // |01> -> |10>
        o, i, o, o, // |10> -> |01>
        o, o, o, i, // |11> -> |11>
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_state() {
        let mps = Mps::zero_state(3, 16);
        let sv = mps.to_statevector();
        assert!((sv[0].re - 1.0).abs() < 1e-10);
        for i in 1..8 {
            assert!(sv[i].norm() < 1e-10);
        }
    }

    #[test]
    fn test_x_gate() {
        let mut mps = Mps::zero_state(2, 16);
        let x = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        mps.apply_1q(0, &x);
        let sv = mps.to_statevector();
        // |10> = basis 1 (qubit 0 is bit 0)
        assert!((sv[1].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state() {
        let mut mps = Mps::zero_state(2, 16);
        let isq2 = 1.0 / 2.0_f64.sqrt();
        let h = [
            Complex64::new(isq2, 0.0),
            Complex64::new(isq2, 0.0),
            Complex64::new(isq2, 0.0),
            Complex64::new(-isq2, 0.0),
        ];
        let o = Complex64::new(0.0, 0.0);
        let i = Complex64::new(1.0, 0.0);
        let cx = [i, o, o, o, o, i, o, o, o, o, o, i, o, o, i, o];

        mps.apply_1q(0, &h);
        mps.apply_2q(0, &cx);

        let sv = mps.to_statevector();
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - expected).abs() < 1e-8, "|00> = {}", sv[0]);
        assert!(sv[1].norm() < 1e-8);
        assert!(sv[2].norm() < 1e-8);
        assert!((sv[3].re - expected).abs() < 1e-8, "|11> = {}", sv[3]);
    }

    #[test]
    fn test_ghz_state() {
        let mut mps = Mps::zero_state(3, 16);
        let isq2 = 1.0 / 2.0_f64.sqrt();
        let h = [
            Complex64::new(isq2, 0.0),
            Complex64::new(isq2, 0.0),
            Complex64::new(isq2, 0.0),
            Complex64::new(-isq2, 0.0),
        ];
        let o = Complex64::new(0.0, 0.0);
        let i = Complex64::new(1.0, 0.0);
        let cx = [i, o, o, o, o, i, o, o, o, o, o, i, o, o, i, o];

        mps.apply_1q(0, &h);
        mps.apply_2q(0, &cx);
        mps.apply_2q(1, &cx);

        let sv = mps.to_statevector();
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - expected).abs() < 1e-8);
        assert!((sv[7].re - expected).abs() < 1e-8);
        for i in 1..7 {
            assert!(sv[i].norm() < 1e-8, "|{:03b}> = {}", i, sv[i]);
        }
    }
}
