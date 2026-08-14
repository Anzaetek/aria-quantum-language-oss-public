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

/// A truncated-SVD provider in the flat-buffer calling convention used by
/// [`Mps::apply_2q`] (`(matrix, m, n, stride, max_rank, threshold)`). A plain
/// `fn` pointer (not a boxed closure) so [`Mps`] stays `Clone` and `Copy`-free
/// GPU contexts live behind the pointer (e.g. a thread-local cuSOLVER handle in
/// `omega-backend-mps-cuda`). Default is the CPU Jacobi SVD
/// [`truncated_svd_flat`]; the CUDA backend injects its `gesvdj` path here.
///
/// CONTRACT: the returned `s` MUST be sorted DESCENDING (largest first). The
/// adaptive-truncation path reads `s[0]` as σ_max and `take_while(> ε·σ_max)`,
/// so an unsorted provider would silently mis-truncate. The CPU kernel sorts;
/// cuSOLVER returns descending order.
pub type SvdFlatFn = fn(&[Complex64], usize, usize, usize, usize, f64) -> SvdResultFlat;

/// A two-site-gate accelerator in the calling convention of [`Mps::apply_2q`]:
/// given the adjacent `(left, right)` tensors, the 4×4 `gate`, the bond cap, and
/// the SVD threshold, it does the whole contract-gate-SVD-split on an
/// accelerator and returns the post-truncation `(left, right)`. It returns
/// `Some` only when the accelerator actually ran; `None` means "not handled —
/// use the built-in path", so [`Mps::apply_2q`] transparently falls back and
/// the result never depends on whether the hook was installed. This is the
/// Metal arm's injection point (contraction on GPU, SVD on CPU — see
/// `omega-backend-mps-metal`); the CUDA arm instead swaps the SVD via
/// [`SvdFlatFn`]. A plain `fn` pointer so [`Mps`] stays `Clone`.
/// Returns the two new site tensors and the *relative* discarded singular-value
/// weight of the split (Σσ²_dropped / Σσ²_total, 0.0 when nothing truncated), so
/// an accelerated path feeds the same truncation certificate as the CPU path.
pub type Contract2qFn =
    fn(&MpsTensor, &MpsTensor, &[Complex64; 16], usize, f64) -> Option<(MpsTensor, MpsTensor, f64)>;

/// Matrix Product State for n qubits.
#[derive(Clone)]
pub struct Mps {
    pub tensors: Vec<MpsTensor>,
    pub n: usize,
    pub max_bond_dim: usize,
    /// Accumulated relative truncation error across every two-site split in the
    /// run so far: Σ over splits of (dropped Σσ² / total Σσ²). 0.0 means every
    /// split was exact (e.g. χ ≥ 2^(n/2)); a growing value is the honest signal
    /// that the bond dimension is under-provisioned. The standard MPS fidelity
    /// proxy — read via [`MpsBackend::last_run_stats`].
    pub discarded_weight: f64,
    /// The largest right-bond dimension reached after truncation across the run
    /// — how close the state came to saturating `max_bond_dim`.
    pub max_bond_reached: usize,
    /// Adaptive truncation: when `Some(ε)`, a split keeps only the singular
    /// values above `ε · σ_max` (relative threshold) rather than always filling
    /// `max_bond_dim`, so the bond grows with the actual entanglement and stays
    /// small when the state is nearly product — with `max_bond_dim` as the hard
    /// ceiling. `None` = fixed-rank truncation at `max_bond_dim` (the default).
    pub adaptive_eps: Option<f64>,
    /// Bond-compression SVD. Every two-qubit gate (including the SWAP network
    /// behind `apply_2q_distant`) routes its truncation through this, so
    /// swapping in a GPU SVD accelerates the whole circuit. Defaults to CPU.
    svd_fn: SvdFlatFn,
    /// Optional two-site-gate accelerator (see [`Contract2qFn`]). `None` = the
    /// built-in contract+SVD path. When set (e.g. the Metal contraction), each
    /// adjacent two-qubit gate is offered to it first, with a transparent
    /// fall-through to the built-in path on `None`.
    contract_fn: Option<Contract2qFn>,
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
            discarded_weight: 0.0,
            max_bond_reached: 1,
            adaptive_eps: None,
            svd_fn: truncated_svd_flat,
            contract_fn: None,
        }
    }

    /// Enable adaptive truncation with relative singular-value threshold `eps`
    /// (see [`Mps::adaptive_eps`]). `max_bond_dim` stays the hard ceiling.
    pub fn set_adaptive_eps(&mut self, eps: f64) {
        self.adaptive_eps = Some(eps);
    }

    /// Route bond-compression SVDs through `f` (e.g. a GPU `gesvdj`). Applies to
    /// every subsequent two-qubit gate, distant swaps included.
    pub fn set_svd_fn(&mut self, f: SvdFlatFn) {
        self.svd_fn = f;
    }

    /// Route each adjacent two-qubit gate through the accelerator `f` (e.g. the
    /// Metal θ-contraction) before falling back to the built-in path. Applies to
    /// every subsequent two-qubit gate, distant swaps included.
    pub fn set_contract_fn(&mut self, f: Contract2qFn) {
        self.contract_fn = Some(f);
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
        assert!(q + 1 < self.n, "q+1 out of range");

        // Offer the whole contract-gate-SVD to an accelerator first (e.g. Metal
        // θ-contraction). It returns `Some` only when it actually ran; `None`
        // falls through to the built-in path below, so the result is identical
        // whether or not the hook is installed. The accelerator truncates at a
        // FIXED rank and knows nothing about `adaptive_eps`, so skip it in
        // adaptive mode — otherwise `mps:auto` would silently fill the bond to
        // the ceiling on a GPU build. Correctness over GPU speed for auto mode.
        if self.adaptive_eps.is_none() {
            if let Some(cf) = self.contract_fn {
                if let Some((nl, nr, rel_discarded)) = cf(
                    &self.tensors[q],
                    &self.tensors[q + 1],
                    gate,
                    self.max_bond_dim,
                    1e-14,
                ) {
                    self.max_bond_reached = self.max_bond_reached.max(nl.bond_right);
                    self.discarded_weight += rel_discarded;
                    self.tensors[q] = nl;
                    self.tensors[q + 1] = nr;
                    return;
                }
            }
        }

        // Route through the configured SVD provider (CPU by default, GPU
        // `gesvdj` when set via `set_svd_fn`). Read the fn pointer out first so
        // the `&mut self` borrow in `apply_2q_with_svd_flat` doesn't conflict.
        let svd_fn = self.svd_fn;
        self.apply_2q_with_svd_flat(q, gate, |m, m_dim, n, stride, max_rank, threshold| {
            svd_fn(m, m_dim, n, stride, max_rank, threshold)
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

        // Step 3: SVD and truncate (CPU or GPU via the injected fn). The
        // total two-site weight is Σσ²_kept + discarded_weight — with the
        // orthonormal-by-construction kernel that also equals ‖Θ'‖²_F, the
        // pre-SVD Frobenius norm.
        let rows = bl * 2;
        let total: f64 = matrix_flat.iter().map(|z| z.norm_sqr()).sum();
        let mut svd = svd_fn(&matrix_flat, rows, cols, cols, self.max_bond_dim, 1e-14);
        // Adaptive truncation: the kernel returns the kept σ sorted descending
        // (capped at the ceiling max_bond_dim). Drop the tail below ε·σ_max so
        // the bond tracks the actual entanglement, folding the extra-dropped
        // weight into the certificate and repacking the k-strided U / Vt buffers
        // to the new column count. `u` is rows×k row-major (element (i,m) at
        // `u[i*k+m]`); `vt` is k×cols row-major (first `keep` rows survive).
        if let Some(eps) = self.adaptive_eps {
            let old_k = svd.s.len();
            let smax = svd.s.first().copied().unwrap_or(0.0);
            let keep = svd
                .s
                .iter()
                .take_while(|&&sv| sv > eps * smax)
                .count()
                .max(1);
            if keep < old_k {
                let extra: f64 = svd.s[keep..].iter().map(|&sv| sv * sv).sum();
                svd.discarded_weight += extra;
                let mut u2 = vec![Complex64::new(0.0, 0.0); rows * keep];
                for i in 0..rows {
                    for m in 0..keep {
                        u2[i * keep + m] = svd.u[i * old_k + m];
                    }
                }
                svd.u = u2;
                svd.vt.truncate(keep * cols);
                svd.s.truncate(keep);
            }
        }
        let new_bm = svd.s.len();
        let kept: f64 = svd.s.iter().map(|&sv| sv * sv).sum();
        // Free unitarity certificate: with orthonormal U/V the kept + dropped
        // weight must equal the input norm. A drift here is exactly the
        // non-unitary-split bug the kernel rewrite fixes (which drove the norm
        // ~20× off — vastly outside this bound). The 1e-7 tolerance leaves ample
        // headroom over the naive-summation error of `total`/`kept` even at the
        // mps:auto ceiling (χ=1024 ⇒ ~4M-term sums), so it never spuriously
        // fires while still catching any real non-unitarity.
        debug_assert!(
            (kept + svd.discarded_weight - total).abs() <= 1e-7 * total.max(1e-300),
            "MPS split not norm-preserving: kept {kept} + dropped {} vs ‖Θ'‖² {total}",
            svd.discarded_weight
        );

        // Truncation certificate — the discarded weight relative to the block
        // norm, accumulated across the run. We deliberately do NOT rescale the
        // kept singular values here: this MPS is not held in canonical form, so
        // a per-split "local" renormalization does not restore the GLOBAL state
        // norm (it can drift it far from 1). Instead the readout paths
        // (`to_statevector`, expectation, sampling) divide by the true ‖ψ‖,
        // which is exact and gauge-independent. Under truncation the raw state
        // norm is < 1 by exactly the lost weight — the honest signal.
        let rel_discarded = if total > 1e-300 {
            svd.discarded_weight / total
        } else {
            0.0
        };
        self.discarded_weight += rel_discarded;
        self.max_bond_reached = self.max_bond_reached.max(new_bm);

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

        // Readout normalization. Truncating splits are unitary but drop weight,
        // so the raw contracted state has ‖ψ‖ ≤ 1 (exactly 1 when no truncation
        // occurred). Divide by the true norm here — gauge-independent and exact
        // — rather than rescaling per split, which would not restore the global
        // norm on this non-canonical MPS. `discarded_weight` remains the
        // separate honest signal of how much was lost.
        let norm_sq: f64 = sv.iter().map(|z| z.norm_sqr()).sum();
        if norm_sq > 1e-300 {
            let inv = 1.0 / norm_sq.sqrt();
            for z in &mut sv {
                *z *= inv;
            }
        }

        sv
    }

    /// The RAW state norm ⟨ψ|ψ⟩, contracted directly from the tensors WITHOUT
    /// the readout normalization `to_statevector` applies. Exactly 1 when no
    /// split truncated; below 1 by the accumulated truncation loss otherwise.
    /// This is the quantity a non-unitary split corrupts — the honest unitarity
    /// probe (`to_statevector`'s output is always normalized, so it cannot see
    /// the drift). Uses the right-environment contraction: `envs[0]` is the 1×1
    /// ⟨ψ|ψ⟩ (bond_left(0) = 1).
    pub fn norm_sqr(&self) -> f64 {
        self.right_environments()[0][0].re
    }

    /// Right-environment matrices for exact sequential sampling.
    ///
    /// `envs[q]` is the Hermitian `bond_left(q) × bond_left(q)` contraction
    /// of sites `q..n` over their physical indices (row-major;
    /// `envs[n]` is the 1×1 identity):
    ///
    ///   E_n = [1],   E_q[l,l'] = Σ_s Σ_{r,r'} A_q[l,s,r] E_{q+1}[r,r'] Ā_q[l',s,r']
    ///
    /// Sampling site q must weight each branch by `w E_{q+1} w†`, not by
    /// `|w|²`: the latter assumes the chain right of q contracts to the
    /// identity (right-canonical form), which gate application via plain
    /// `U√S / √S V†` SVD splits does not maintain — that assumption is what
    /// biased sampled counts while `to_statevector()` stayed exact.
    pub fn right_environments(&self) -> Vec<Vec<Complex64>> {
        let zero = Complex64::new(0.0, 0.0);
        let mut envs: Vec<Vec<Complex64>> = vec![Vec::new(); self.n + 1];
        envs[self.n] = vec![Complex64::new(1.0, 0.0)];
        for q in (0..self.n).rev() {
            let t = &self.tensors[q];
            let (bl, br) = (t.bond_left, t.bond_right);
            let e_next = &envs[q + 1];
            let mut e = vec![zero; bl * bl];
            for s in 0..2 {
                // tmp[l, r'] = Σ_r A[l,s,r] · E_{q+1}[r,r']
                let mut tmp = vec![zero; bl * br];
                for l in 0..bl {
                    for r in 0..br {
                        let a = t.get(l, s, r);
                        for rp in 0..br {
                            tmp[l * br + rp] += a * e_next[r * br + rp];
                        }
                    }
                }
                // E_q[l, l'] += Σ_{r'} tmp[l, r'] · conj(A[l',s,r'])
                for l in 0..bl {
                    for lp in 0..bl {
                        let mut sum = zero;
                        for rp in 0..br {
                            sum += tmp[l * br + rp] * t.get(lp, s, rp).conj();
                        }
                        e[l * bl + lp] += sum;
                    }
                }
            }
            envs[q] = e;
        }
        envs
    }

    /// Sample a measurement outcome.
    ///
    /// Convenience wrapper that recomputes the right environments on every
    /// call; when drawing many shots from the same state, compute
    /// [`Mps::right_environments`] once and use [`Mps::sample_with_envs`].
    pub fn sample(&self, rng: &mut impl rand::Rng) -> u64 {
        self.sample_with_envs(&self.right_environments(), rng)
    }

    /// Sample a measurement outcome using precomputed right environments
    /// (see [`Mps::right_environments`]).
    /// Sample a shot, packing **only the measured qubits** into the key.
    ///
    /// `cbit_of[q]` is `Some(c)` when qubit `q` is measured into classical bit
    /// `c`, and `None` otherwise. The walk over the chain is identical to
    /// [`Self::sample_with_envs`] — every qubit is still sampled, because each
    /// outcome conditions the ones after it — but the returned key carries the
    /// measured bits at their classical positions.
    ///
    /// # Why this exists
    ///
    /// `sample_with_envs` builds its key with `result |= bit << q` over all `n`
    /// qubits, which cannot represent more than 64 and shifts out of range above
    /// that. So a 1024-qubit circuit measuring two qubits was refused, though
    /// its outcome is two bits wide. Keying on the classical register is also
    /// what qiskit reports over, which is what makes the two comparable.
    pub fn sample_with_envs_projected(
        &self,
        envs: &[Vec<Complex64>],
        rng: &mut impl rand::Rng,
        cbit_of: &[Option<u32>],
    ) -> u64 {
        self.sample_inner(envs, rng, Some(cbit_of))
    }

    pub fn sample_with_envs(&self, envs: &[Vec<Complex64>], rng: &mut impl rand::Rng) -> u64 {
        self.sample_inner(envs, rng, None)
    }

    fn sample_inner(
        &self,
        envs: &[Vec<Complex64>],
        rng: &mut impl rand::Rng,
        cbit_of: Option<&[Option<u32>]>,
    ) -> u64 {
        let mut bits = Vec::new();
        self.sample_bits_with_envs_into(envs, rng, &mut bits);
        Self::pack_bits(&bits, cbit_of)
    }

    /// Pack per-site bits into a counts key.
    ///
    /// `cbit_of` is `Some` when the key is the CLASSICAL register: each site
    /// lands at its classical index, and an unmeasured site is dropped. `None`
    /// keys on the qubit register, which is only representable below 64 sites.
    pub fn pack_bits(bits: &[u8], cbit_of: Option<&[Option<u32>]>) -> u64 {
        let mut result = 0u64;
        for (q, bit) in bits.iter().enumerate() {
            match cbit_of {
                Some(map) => {
                    if let Some(Some(c)) = map.get(q) {
                        result |= ((*bit & 1) as u64) << *c;
                    }
                }
                None => result |= ((*bit & 1) as u64) << q,
            }
        }
        result
    }

    /// Sample a shot, writing **one bit per site** into `out`.
    ///
    /// Every site must be drawn even when it is not reported, because each
    /// outcome conditions the ones after it — so the projection can only narrow
    /// the KEY, never the work.
    ///
    /// Exposed unpacked because a caller may need to transform a bit before it
    /// is placed: `NoisyMpsBackend` applies readout error per qubit, which has
    /// to happen on the qubit index, while the key is built on the classical
    /// index. Packing first would make readout flips land on creg positions.
    ///
    /// `out` is reused across shots; the sampling loop is hot and this keeps it
    /// to one allocation per run rather than one per shot.
    pub fn sample_bits_with_envs_into(
        &self,
        envs: &[Vec<Complex64>],
        rng: &mut impl rand::Rng,
        out: &mut Vec<u8>,
    ) {
        let zero = Complex64::new(0.0, 0.0);
        out.clear();
        out.resize(self.n, 0);
        let mut left_vec = vec![Complex64::new(1.0, 0.0)];
        // Scratch buffers recycled across sites (and, via the swap below,
        // with left_vec): this loop runs shots × n times, so per-site
        // allocations dominate otherwise.
        let mut w = [Vec::new(), Vec::new()];

        for q in 0..self.n {
            let t = &self.tensors[q];
            let (bl, br) = (t.bond_left, t.bond_right);
            let e_next = &envs[q + 1];

            // w_s[r] = Σ_l v[l] A[l,s,r];  p_s = Re(w_s E_{q+1} w_s†) ≥ 0.
            let mut p = [0.0f64; 2];
            for (bit, (w_bit, p_bit)) in w.iter_mut().zip(p.iter_mut()).enumerate() {
                w_bit.clear();
                w_bit.resize(br, zero);
                for (r, w_r) in w_bit.iter_mut().enumerate() {
                    for l in 0..bl {
                        *w_r += left_vec[l] * t.get(l, bit, r);
                    }
                }
                let mut prob = zero;
                for r in 0..br {
                    for rp in 0..br {
                        prob += w_bit[r] * e_next[r * br + rp] * w_bit[rp].conj();
                    }
                }
                *p_bit = prob.re.max(0.0);
            }

            let total = p[0] + p[1];
            let bit = if total > 0.0 && rng.random::<f64>() < p[0] / total {
                0usize
            } else {
                1usize
            };
            out[q] = bit as u8;

            // Condition on the outcome; rescale so the running prefix
            // stays O(1) (only probability *ratios* matter downstream).
            let norm = p[bit].sqrt();
            let inv = if norm > 0.0 { 1.0 / norm } else { 1.0 };
            std::mem::swap(&mut left_vec, &mut w[bit]);
            for v in &mut left_vec {
                *v *= inv;
            }
        }
    }

    /// Left-environment matrix of sites `0..q` (row-major
    /// `bond_left(q) × bond_left(q)`; the 1×1 identity for q = 0) — the
    /// mirror image of [`Mps::right_environments`]:
    ///
    ///   L_0 = [1],   L_{k+1}[r,r'] = Σ_s Σ_{l,l'} L_k[l,l'] A_k[l,s,r] Ā_k[l',s,r']
    fn left_environment(&self, q: usize) -> Vec<Complex64> {
        let zero = Complex64::new(0.0, 0.0);
        let mut env = vec![Complex64::new(1.0, 0.0)];
        for k in 0..q {
            let t = &self.tensors[k];
            let (bl, br) = (t.bond_left, t.bond_right);
            let mut next = vec![zero; br * br];
            for s in 0..2 {
                // u[l', r] = Σ_l L[l,l'] · A[l,s,r]
                let mut u = vec![zero; bl * br];
                for l in 0..bl {
                    for lp in 0..bl {
                        let e = env[l * bl + lp];
                        for r in 0..br {
                            u[lp * br + r] += e * t.get(l, s, r);
                        }
                    }
                }
                // L'[r, r'] += Σ_{l'} u[l', r] · conj(A[l',s,r'])
                for lp in 0..bl {
                    for r in 0..br {
                        let ur = u[lp * br + r];
                        for rp in 0..br {
                            next[r * br + rp] += ur * t.get(lp, s, rp).conj();
                        }
                    }
                }
            }
            env = next;
        }
        env
    }

    /// Mid-circuit projective measurement of a single qubit.
    /// Computes P(q=0), samples outcome, projects the local tensor,
    /// and renormalizes. Returns the measurement outcome (0 or 1).
    ///
    /// The outcome probabilities contract the full state — left environment,
    /// local tensor, right environment — because local tensor norms are only
    /// the true marginals when the chain is canonical around site q, which
    /// plain-SVD gate splits do not maintain (same mechanism as the sampling
    /// fix in [`Mps::sample_with_envs`]).
    pub fn measure_site(&mut self, q: usize, rng: &mut impl rand::Rng) -> u8 {
        let left = self.left_environment(q);
        let right = self.right_environments();
        let e_next = &right[q + 1];
        let t = &self.tensors[q];
        let bl = t.bond_left;
        let br = t.bond_right;
        let zero = Complex64::new(0.0, 0.0);

        // p(b) = Σ_{l,l',r,r'} L[l,l'] A[l,b,r] E[r,r'] conj(A[l',b,r'])
        let mut norm_sq = [0.0_f64; 2];
        for (phys, p_bit) in norm_sq.iter_mut().enumerate() {
            // m[l, r'] = Σ_r A[l,b,r] · E[r,r']
            let mut m = vec![zero; bl * br];
            for l in 0..bl {
                for r in 0..br {
                    let a = t.get(l, phys, r);
                    for rp in 0..br {
                        m[l * br + rp] += a * e_next[r * br + rp];
                    }
                }
            }
            let mut p = zero;
            for l in 0..bl {
                for lp in 0..bl {
                    let mut g = zero; // g = Σ_{r'} m[l,r'] conj(A[l',b,r'])
                    for rp in 0..br {
                        g += m[l * br + rp] * t.get(lp, phys, rp).conj();
                    }
                    p += left[l * bl + lp] * g;
                }
            }
            *p_bit = p.re.max(0.0);
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
    fn sampling_matches_statevector_probabilities() {
        // Regression: sequential sampling used to assume a right-canonical
        // chain (right environment = identity), which plain-SVD gate splits
        // do not produce — sampled frequencies were biased ~10σ while the
        // contracted statevector stayed exact. State: H q0; RY(0.7) q1;
        // CX q0,q3 (distant → SWAP chain); CX q1,q2. Bond dimension 2, so
        // truncation cannot explain any deviation.
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let isq2 = 1.0 / 2.0_f64.sqrt();
        let h = [
            Complex64::new(isq2, 0.0),
            Complex64::new(isq2, 0.0),
            Complex64::new(isq2, 0.0),
            Complex64::new(-isq2, 0.0),
        ];
        let (c, s) = ((0.35f64).cos(), (0.35f64).sin());
        let ry = [
            Complex64::new(c, 0.0),
            Complex64::new(-s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(c, 0.0),
        ];
        let o = Complex64::new(0.0, 0.0);
        let i = Complex64::new(1.0, 0.0);
        let cx = [i, o, o, o, o, i, o, o, o, o, o, i, o, o, i, o];

        let mut mps = Mps::zero_state(4, 16);
        mps.apply_1q(0, &h);
        mps.apply_1q(1, &ry);
        mps.apply_2q_distant(0, 3, &cx);
        mps.apply_2q_distant(1, 2, &cx);

        let exact: Vec<f64> = mps.to_statevector().iter().map(|a| a.norm_sqr()).collect();

        let shots = 20000usize;
        let mut rng = StdRng::seed_from_u64(7);
        let envs = mps.right_environments();
        let mut freq = [0.0f64; 16];
        for _ in 0..shots {
            freq[mps.sample_with_envs(&envs, &mut rng) as usize] += 1.0 / shots as f64;
        }

        // SE ≈ sqrt(p(1-p)/shots) ≤ 0.0035; 0.015 is > 4σ for every outcome.
        for (basis, (&p, &f)) in exact.iter().zip(freq.iter()).enumerate() {
            assert!(
                (p - f).abs() < 0.015,
                "|{basis:04b}>: exact p = {p:.4}, sampled f = {f:.4}"
            );
        }
    }

    #[test]
    fn measure_site_matches_statevector_marginals() {
        // Regression: measure_site used local tensor norms for the outcome
        // probability — the same environment-=-identity assumption that
        // biased terminal sampling. Measuring every site sequentially must
        // reproduce the exact joint distribution on a non-canonical chain.
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let isq2 = 1.0 / 2.0_f64.sqrt();
        let h = [
            Complex64::new(isq2, 0.0),
            Complex64::new(isq2, 0.0),
            Complex64::new(isq2, 0.0),
            Complex64::new(-isq2, 0.0),
        ];
        let (c, s) = ((0.35f64).cos(), (0.35f64).sin());
        let ry = [
            Complex64::new(c, 0.0),
            Complex64::new(-s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(c, 0.0),
        ];
        let o = Complex64::new(0.0, 0.0);
        let i = Complex64::new(1.0, 0.0);
        let cx = [i, o, o, o, o, i, o, o, o, o, o, i, o, o, i, o];

        let mut mps = Mps::zero_state(4, 16);
        mps.apply_1q(0, &h);
        mps.apply_1q(1, &ry);
        mps.apply_2q_distant(0, 3, &cx);
        mps.apply_2q_distant(1, 2, &cx);

        let exact: Vec<f64> = mps.to_statevector().iter().map(|a| a.norm_sqr()).collect();

        let trials = 20000usize;
        let mut rng = StdRng::seed_from_u64(11);
        let mut freq = [0.0f64; 16];
        for _ in 0..trials {
            let mut state = mps.clone();
            let mut outcome = 0usize;
            for q in 0..4 {
                outcome |= (state.measure_site(q, &mut rng) as usize) << q;
            }
            freq[outcome] += 1.0 / trials as f64;
        }

        for (basis, (&p, &f)) in exact.iter().zip(freq.iter()).enumerate() {
            assert!(
                (p - f).abs() < 0.015,
                "|{basis:04b}>: exact p = {p:.4}, measured f = {f:.4}"
            );
        }
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
