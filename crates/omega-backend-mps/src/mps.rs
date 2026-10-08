//! Matrix Product State representation and operations.
//!
//! An n-site state is represented as a chain of tensors:
//!   A[0] (1 x d_0 x b1), A[1] (b1 x d_1 x b2), ..., A[n-1] (b_{n-1} x d_{n-1} x 1)
//! where b_i is the bond dimension between sites i and i+1 and d_q the
//! physical dimension of site q (2 for a qubit; per-site, mixed radix allowed).
//! Everything here is dimension-generic: the gate kernels, the SWAP network,
//! `to_statevector`, sampling (one DIGIT `< d` per site), mid-circuit
//! measurement and the product-operator expectation. The one place `d = 2` is
//! still required is packing digits into a BIT-string key
//! ([`Mps::pack_bits`] / [`Mps::pack_outcome`]), which refuses a digit `>= 2`
//! by assertion rather than writing it into a bit.

use num_complex::Complex64;
use rand::RngExt;

use crate::svd::{truncated_svd_flat, SvdResultFlat};

/// A single MPS tensor: shape (bond_left, phys, bond_right), `phys = 2` for a
/// qubit site. Stored as a flat array in row-major order.
#[derive(Clone, Debug)]
pub struct MpsTensor {
    pub bond_left: usize,
    /// Physical (local Hilbert space) dimension of this site.
    pub phys: usize,
    pub bond_right: usize,
    /// Data indexed as [left * phys * bond_right + p * bond_right + right]
    pub data: Vec<Complex64>,
}

impl MpsTensor {
    /// A zeroed qubit tensor (`phys = 2`).
    pub fn new(bond_left: usize, bond_right: usize) -> Self {
        Self::with_phys(bond_left, 2, bond_right)
    }

    /// A zeroed tensor of physical dimension `phys`.
    pub fn with_phys(bond_left: usize, phys: usize, bond_right: usize) -> Self {
        Self {
            bond_left,
            phys,
            bond_right,
            data: vec![Complex64::new(0.0, 0.0); bond_left * phys * bond_right],
        }
    }

    pub fn get(&self, left: usize, p: usize, right: usize) -> Complex64 {
        self.data[left * self.phys * self.bond_right + p * self.bond_right + right]
    }

    pub fn set(&mut self, left: usize, p: usize, right: usize, val: Complex64) {
        self.data[left * self.phys * self.bond_right + p * self.bond_right + right] = val;
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

/// Which truncated-SVD kernel a run's bond compressions went through.
///
/// # Why a run has to say this
///
/// Two kernels now produce the production certificate: the one-sided Jacobi
/// kernel on every target, and Accelerate `zgesdd` by default on macOS. Both
/// are sound and they do not agree in the last digits — the Accelerate path's
/// `discarded_weight` sits ABOVE the Jacobi reference's by up to ~1e-12 of the
/// block norm (`accelerate::backward_error_allowance`). So the same circuit
/// yields two certificates that differ far below any threshold and are not
/// bit-identical, and nothing in the record said which kernel produced which
/// until this existed. A reader who diffs a macOS run against a Linux one and
/// has no field to look at spends a day on it.
///
/// The rule is the one `omega-hostgate status` follows about budget sources:
/// when two sources can produce a number, the record names the one that did.
///
/// # The label and the code come from one place
///
/// [`Self::svd_flat_fn`] maps each variant to the function that implements it,
/// so the name a certificate carries and the kernel that ran are the same match
/// arm. A label set independently of the hook is a label that can lie, which is
/// how [`crate::MpsBackend::with_svd_fn`] is handled: an arbitrary function
/// pointer cannot be named, so it reports [`Self::Custom`] rather than
/// inheriting whatever the default was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SvdKernel {
    /// [`truncated_svd_flat`], the threaded one-sided Jacobi kernel. The
    /// reference the Accelerate bound is held above, the kernel on every
    /// non-macOS target, and — through the size gate — the kernel macOS still
    /// uses on blocks below `accelerate::MIN_DIM_FOR_LAPACK`.
    Jacobi,
    /// `accelerate::accelerate_svd_flat`: Accelerate LAPACK `zgesdd` in f64
    /// above the size gate, delegating to [`Self::Jacobi`] below it. macOS
    /// only, and the default there — see `STATUS.md` §5 item 16.
    ///
    /// **This names the ENTRY POINT the run went through, which is not the
    /// same as "every block was solved by LAPACK".** The gate
    /// (`accelerate::MIN_DIM_FOR_LAPACK` = 16 on `min(m, n)`) exists because
    /// LAPACK measurably loses on small blocks, so a shallow or small-χ run can
    /// report `accelerate-zgesdd` and carry a certificate that is bit-for-bit
    /// the Jacobi kernel's — a χ=7 chain produces 14-column θ blocks and never
    /// reaches LAPACK at all. That is the gate working, and it is why this
    /// variant does not promise a number: what it promises is the dispatch
    /// POLICY, which is what a reader comparing two platforms' certificates
    /// needs to know. Per-call dispatch counts are a measurement harness's job
    /// (`examples/accelerate_svd_share.rs` counts both outcomes); putting them
    /// on the hook would change [`SvdFlatFn`]'s contract, which the CUDA arm
    /// shares.
    AccelerateZgesdd,
    /// A kernel installed through [`crate::MpsBackend::with_svd_fn`] /
    /// [`Mps::set_svd_fn`] that this crate cannot name: the CUDA `gesvdj` arm
    /// in `omega-backend-mps-cuda`, or a measurement shim such as
    /// `examples/mps_stage_profile.rs`'s timing wrapper.
    ///
    /// Deliberately not resolved by comparing function addresses. Rust does not
    /// guarantee distinct `fn` items have distinct addresses, so an identity
    /// test could silently report one kernel's name for another's output — the
    /// exact failure this field exists to prevent.
    Custom,
}

impl SvdKernel {
    /// The spelling that goes into a certificate, human or JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Jacobi => "jacobi",
            Self::AccelerateZgesdd => "accelerate-zgesdd",
            Self::Custom => "custom",
        }
    }

    /// The hook function this variant names, or `None` when the variant does
    /// not name one that this target can build ([`Self::AccelerateZgesdd`] off
    /// macOS) or that this crate owns ([`Self::Custom`]).
    pub fn svd_flat_fn(self) -> Option<SvdFlatFn> {
        match self {
            Self::Jacobi => Some(truncated_svd_flat),
            #[cfg(target_os = "macos")]
            Self::AccelerateZgesdd => Some(crate::accelerate::accelerate_svd_flat),
            #[cfg(not(target_os = "macos"))]
            Self::AccelerateZgesdd => None,
            Self::Custom => None,
        }
    }

    /// Is this kernel available on the target being built?
    pub fn is_available(self) -> bool {
        self.svd_flat_fn().is_some()
    }
}

/// The kernel the MPS backends install when the caller does not ask for one —
/// Accelerate `zgesdd` on macOS, the Jacobi kernel everywhere else.
///
/// This `cfg` is the whole promotion. Accelerate is in the base macOS SDK, so
/// there is no optional dependency to probe and no feature to opt into; a
/// target check is the only availability question there is. Every non-macOS
/// target keeps the Jacobi kernel bit for bit.
pub const fn default_svd_kernel() -> SvdKernel {
    #[cfg(target_os = "macos")]
    {
        SvdKernel::AccelerateZgesdd
    }
    #[cfg(not(target_os = "macos"))]
    {
        SvdKernel::Jacobi
    }
}

/// The hook the MPS backends install by default.
///
/// Panics rather than falling back if [`default_svd_kernel`] ever names a
/// kernel this target cannot build. A fallback here would turn "the promotion
/// did not happen" into a silent 4-8x loss that still reports a sound
/// certificate, which is the one failure mode nothing downstream can see.
pub fn default_svd_flat_fn() -> SvdFlatFn {
    default_svd_kernel()
        .svd_flat_fn()
        .expect("default_svd_kernel named a kernel this target cannot build")
}

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

/// Matrix Product State for n sites (qubits, or qudits of per-site dimension).
#[derive(Clone)]
pub struct Mps {
    pub tensors: Vec<MpsTensor>,
    pub n: usize,
    /// Per-site physical dimension (`dims[q] == tensors[q].phys`). All 2 for
    /// a qubit chain; mixed radix is allowed. A SWAP moves a site's dimension
    /// along with its state, so the SWAP network keeps this in step.
    pub dims: Vec<usize>,
    pub max_bond_dim: usize,
    /// Accumulated relative truncation error across every two-site split in the
    /// run so far: Σ over splits of (dropped Σσ² / total Σσ²). 0.0 means every
    /// split was exact (e.g. χ ≥ 2^(n/2)); a growing value is the honest signal
    /// that the bond dimension is under-provisioned. The standard MPS fidelity
    /// proxy — read via [`MpsBackend::last_run_stats`].
    pub discarded_weight: f64,
    /// **Fidelity estimate**: Π over splits of `(1 − εᵢ)`, starting at 1.0.
    ///
    /// The same quantity, and the same shape, that a commercial MPS simulator
    /// reports per run: a number in [0, 1] where **1.0 means the run was
    /// exact**. It is reported alongside `discarded_weight` rather than instead
    /// of it, because the two answer different questions — "how good is this
    /// state" versus "how much was thrown away" — and the sum is what the
    /// refusal ceiling is expressed in.
    ///
    /// # Not a bound here, and why
    ///
    /// In **canonical gauge** each εᵢ is the true local discarded probability
    /// weight and this product is a genuine LOWER bound on the state fidelity.
    /// This MPS is deliberately not canonical (see the truncation comment in
    /// [`Self::apply_2q_with_svd_flat`]), so εᵢ is measured against the block norm rather than
    /// the global one and the product is an ESTIMATE.
    ///
    /// Measured across 108 non-Clifford circuits with non-flat Schmidt spectra
    /// (`tests/fidelity_estimate.rs`): it never exceeded the true fidelity, and
    /// `F_true / F_est` spanned **[1.0067, 142.7]**. Tight under mild
    /// truncation; up to ~140× pessimistic when the bond is badly starved. Read
    /// it as "is this run usable", not as "how good is this bad run".
    ///
    /// Nothing proves it cannot understate on a circuit outside that corpus,
    /// and that is the difference between this number and one you may rely on.
    /// A separate measurement on one trajectory put the canonical certificate
    /// at 0.664 against our 6.402 — ours overstates the loss ~10× there too.
    pub fidelity_estimate: f64,
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

/// Divide a raw contraction by the state norm, with the degenerate-state floor.
///
/// One place, so `expectation_product` and the batched `expectation_from_mps`
/// cannot drift on what happens at `⟨ψ|ψ⟩ ≈ 0`.
pub(crate) fn normalize_expectation(value: Complex64, norm_sq: f64) -> f64 {
    if norm_sq > 1e-300 {
        value.re / norm_sq
    } else {
        0.0
    }
}

impl Mps {
    /// Create |00...0> state as MPS with bond dimension 1 (all sites qubits).
    pub fn zero_state(n: usize, max_bond_dim: usize) -> Self {
        Self::zero_state_dims(&vec![2; n], max_bond_dim)
    }

    /// Create |0...0> over sites of the given physical dimensions (mixed radix
    /// allowed), bond dimension 1. Every dimension must be >= 2.
    pub fn zero_state_dims(dims: &[usize], max_bond_dim: usize) -> Self {
        for (q, &d) in dims.iter().enumerate() {
            assert!(
                d >= 2,
                "site {q} has physical dimension {d}; every site needs d >= 2"
            );
        }
        let n = dims.len();
        let mut tensors = Vec::with_capacity(n);
        for &d in dims {
            let mut t = MpsTensor::with_phys(1, d, 1);
            t.set(0, 0, 0, Complex64::new(1.0, 0.0)); // |0>
            tensors.push(t);
        }
        Self {
            tensors,
            n,
            dims: dims.to_vec(),
            max_bond_dim,
            discarded_weight: 0.0,
            fidelity_estimate: 1.0,
            max_bond_reached: 1,
            adaptive_eps: None,
            svd_fn: truncated_svd_flat,
            contract_fn: None,
        }
    }

    /// True when every site is a qubit — the precondition for reading a
    /// sampled outcome as a bit string (see [`Self::pack_bits`]).
    pub fn is_qubit_only(&self) -> bool {
        self.dims.iter().all(|&d| d == 2)
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
    ///
    /// Qubit wrapper over [`Self::apply_1`]; refuses a site whose dimension is
    /// not 2 instead of silently reading only its first two levels.
    pub fn apply_1q(&mut self, q: usize, gate: &[Complex64; 4]) {
        assert_eq!(
            self.dims[q], 2,
            "apply_1q on site {q} of dimension {}: use apply_1",
            self.dims[q]
        );
        self.apply_1(q, gate);
    }

    /// Apply a single-site gate to site `q`. `gate` is `d×d` row-major with
    /// `d = self.dims[q]`: `new[l,s',r] = Σ_s gate[s'·d + s] · old[l,s,r]`.
    pub fn apply_1(&mut self, q: usize, gate: &[Complex64]) {
        let d = self.dims[q];
        assert_eq!(
            gate.len(),
            d * d,
            "apply_1: site {q} has dimension {d}, so the gate must be {d}x{d}"
        );
        let zero = Complex64::new(0.0, 0.0);
        let t = &self.tensors[q];
        debug_assert_eq!(t.phys, d, "site {q}: tensor phys disagrees with dims");
        let bl = t.bond_left;
        let br = t.bond_right;
        let mut new_data = vec![zero; bl * d * br];
        let mut a = vec![zero; d];

        for l in 0..bl {
            for r in 0..br {
                for (s, a_s) in a.iter_mut().enumerate() {
                    *a_s = t.get(l, s, r);
                }
                // new[l,s',r] = sum_s gate[s',s] * old[l,s,r], accumulated over
                // s ascending and opening on the s = 0 product: for d = 2 this
                // is exactly the qubit kernel's `g0*a0 + g1*a1`, bit for bit.
                for out in 0..d {
                    let row = &gate[out * d..(out + 1) * d];
                    let mut v = row[0] * a[0];
                    for (g, x) in row.iter().zip(a.iter()).skip(1) {
                        v += *g * *x;
                    }
                    new_data[l * d * br + out * br + r] = v;
                }
            }
        }

        self.tensors[q].data = new_data;
    }

    /// Apply a two-qubit gate (4x4 matrix) to adjacent qubits (q, q+1).
    /// Uses SVD to split the result back into two tensors, truncating to max_bond_dim.
    ///
    /// Qubit wrapper over [`Self::apply_2`]; refuses a non-qubit pair.
    pub fn apply_2q(&mut self, q: usize, gate: &[Complex64; 16]) {
        assert!(q + 1 < self.n, "q+1 out of range");
        self.assert_qubit_pair(q, q + 1, "apply_2q");
        self.apply_2(q, gate);
    }

    fn assert_qubit_pair(&self, a: usize, b: usize, what: &str) {
        assert!(
            self.dims[a] == 2 && self.dims[b] == 2,
            "{what} on sites ({a}, {b}) of dimensions ({}, {}): use the slice API",
            self.dims[a],
            self.dims[b]
        );
    }

    /// Apply a two-site gate to adjacent sites (q, q+1). `gate` is
    /// `(d0·d1)×(d0·d1)` row-major, `d0 = dims[q]`, `d1 = dims[q+1]`, with the
    /// combined index `s0·d1 + s1` (for qubits: `ss = s0*2 + s1`). Uses SVD to
    /// split the result back into two tensors, truncating to max_bond_dim.
    pub fn apply_2(&mut self, q: usize, gate: &[Complex64]) {
        assert!(q + 1 < self.n, "q+1 out of range");
        let out = (self.dims[q], self.dims[q + 1]);
        self.apply_2_reshaping(q, gate, out);
    }

    /// [`Self::apply_2`] where the gate's OUTPUT pair has dimensions
    /// `out = (e0, e1)` (with `e0·e1 = d0·d1`) — only the SWAP between sites of
    /// different dimension needs `out ≠ (d0, d1)`. `self.dims` is updated.
    fn apply_2_reshaping(&mut self, q: usize, gate: &[Complex64], out: (usize, usize)) {
        assert!(q + 1 < self.n, "q+1 out of range");
        let (d0, d1) = (self.dims[q], self.dims[q + 1]);
        assert_eq!(
            gate.len(),
            (d0 * d1) * (d0 * d1),
            "apply_2: sites ({q}, {}) have dimensions ({d0}, {d1}), so the gate must be \
             {n}x{n}",
            q + 1,
            n = d0 * d1
        );

        // Offer the whole contract-gate-SVD to an accelerator first (e.g. Metal
        // θ-contraction). It returns `Some` only when it actually ran; `None`
        // falls through to the built-in path below, so the result is identical
        // whether or not the hook is installed. The accelerator truncates at a
        // FIXED rank and knows nothing about `adaptive_eps`, so skip it in
        // adaptive mode — otherwise `mps:auto` would silently fill the bond to
        // the ceiling on a GPU build. Correctness over GPU speed for auto mode.
        // The hook is a qubit kernel (`&[Complex64; 16]`): consulted only for a
        // qubit pair that stays a qubit pair.
        if d0 == 2 && d1 == 2 && out == (2, 2) && self.adaptive_eps.is_none() {
            if let Some(cf) = self.contract_fn {
                let gate16: &[Complex64; 16] = gate
                    .try_into()
                    .expect("a qubit-pair gate has 16 entries (checked above)");
                if let Some((nl, nr, rel_discarded)) = cf(
                    &self.tensors[q],
                    &self.tensors[q + 1],
                    gate16,
                    self.max_bond_dim,
                    1e-14,
                ) {
                    self.max_bond_reached = self.max_bond_reached.max(nl.bond_right);
                    self.discarded_weight += rel_discarded;
                    self.fidelity_estimate *= (1.0 - rel_discarded).max(0.0);
                    self.tensors[q] = nl;
                    self.tensors[q + 1] = nr;
                    return;
                }
            }
        }

        // Route through the configured SVD provider (CPU by default, GPU
        // `gesvdj` when set via `set_svd_fn`). Read the fn pointer out first so
        // the `&mut self` borrow in `apply_2_with_svd_flat` doesn't conflict.
        let svd_fn = self.svd_fn;
        self.apply_2_with_svd_flat_reshaping(
            q,
            gate,
            out,
            |m, m_dim, n, stride, max_rank, threshold| {
                svd_fn(m, m_dim, n, stride, max_rank, threshold)
            },
        );
    }

    /// Apply a two-qubit gate with a caller-supplied flat-buffer SVD
    /// function. Qubit wrapper over [`Self::apply_2_with_svd_flat`].
    pub fn apply_2q_with_svd_flat<F>(&mut self, q: usize, gate: &[Complex64; 16], svd_fn: F)
    where
        F: FnOnce(&[Complex64], usize, usize, usize, usize, f64) -> SvdResultFlat,
    {
        assert!(q + 1 < self.n, "q+1 out of range");
        self.assert_qubit_pair(q, q + 1, "apply_2q_with_svd_flat");
        self.apply_2_with_svd_flat(q, gate, svd_fn);
    }

    /// Apply a two-site gate with a caller-supplied flat-buffer SVD
    /// function. The closure receives the contracted-and-gated `Θ'`
    /// reshaped as a flat row-major matrix of shape `(bl*d0) × (d1*br)`
    /// — no nested `Vec<Vec<_>>` allocation. The CUDA backend wires
    /// `CudaSvdContext::truncated_svd_flat` in through this hook.
    ///
    /// Closure args: `(matrix, m, n, stride, max_rank, threshold)`,
    /// where `matrix[i * stride + j]` is element `(i, j)`.
    pub fn apply_2_with_svd_flat<F>(&mut self, q: usize, gate: &[Complex64], svd_fn: F)
    where
        F: FnOnce(&[Complex64], usize, usize, usize, usize, f64) -> SvdResultFlat,
    {
        assert!(q + 1 < self.n, "q+1 out of range");
        let out = (self.dims[q], self.dims[q + 1]);
        self.apply_2_with_svd_flat_reshaping(q, gate, out, svd_fn);
    }

    fn apply_2_with_svd_flat_reshaping<F>(
        &mut self,
        q: usize,
        gate: &[Complex64],
        out: (usize, usize),
        svd_fn: F,
    ) where
        F: FnOnce(&[Complex64], usize, usize, usize, usize, f64) -> SvdResultFlat,
    {
        assert!(q + 1 < self.n, "q+1 out of range");
        let (d0, d1) = (self.dims[q], self.dims[q + 1]);
        let (e0, e1) = out;
        let din = d0 * d1;
        assert_eq!(
            e0 * e1,
            din,
            "output pair ({e0}, {e1}) cannot hold ({d0}, {d1})"
        );
        assert_eq!(gate.len(), din * din, "two-site gate must be {din}x{din}");

        let tl = &self.tensors[q];
        let tr = &self.tensors[q + 1];
        debug_assert_eq!(tl.phys, d0, "site {q}: tensor phys disagrees with dims");
        debug_assert_eq!(
            tr.phys,
            d1,
            "site {}: tensor phys disagrees with dims",
            q + 1
        );
        let bl = tl.bond_left;
        let bm = tl.bond_right; // = tr.bond_left
        let br = tr.bond_right;

        // The matrix layout the SVD consumes is `(bl*e0) × (e1*br)`
        // row-major. Equivalently — and this is the key for the
        // flat-buffer refactor — that's the *same* memory layout as
        // `Theta'[l, s0, s1, r]` with strides `(e0*e1*br, e1*br, br, 1)`:
        //
        //   Theta'[l, s0, s1, r]
        //     = matrix[l*e0 + s0][s1*br + r]
        //     = matrix_flat[(l*e0 + s0) * (e1*br) + s1*br + r]
        //
        // So we build Theta' directly as the SVD's input buffer and
        // skip the host-side reshape that the old API needed. (For qubits
        // e0 = e1 = d0 = d1 = 2 and every index below is the old one.)

        // Step 1: Contract two sites into Theta[l, s0, s1, r].
        //
        // Parallel over `l` (one output block per left-bond value), inner sum
        // untouched — each element's summation order is exactly the serial
        // loop's, so the result is BIT-IDENTICAL at any thread count. O(d²χ³)
        // work; secondary to the SVD but no longer negligible once the SVD is
        // parallel too (Amdahl).
        use rayon::prelude::*;
        let mut theta = vec![Complex64::new(0.0, 0.0); bl * din * br];
        theta
            .par_chunks_mut(din * br)
            .enumerate()
            .for_each(|(l, block)| {
                for s0 in 0..d0 {
                    for s1 in 0..d1 {
                        for r in 0..br {
                            let mut val = Complex64::new(0.0, 0.0);
                            for m in 0..bm {
                                val += tl.get(l, s0, m) * tr.get(m, s1, r);
                            }
                            block[s0 * d1 * br + s1 * br + r] = val;
                        }
                    }
                }
            });

        // Step 2: Apply gate to physical indices. We write directly
        // into the row-major matrix layout consumed by Step 3 (SVD).
        // `Theta[l, ss_in, r]` sits at `l*din*br + ss_in*br + r` with
        // `ss_in = s0*d1 + s1`; the output index splits as
        // `ss_out = s0_out*e1 + s1_out`.
        let cols = e1 * br;
        let mut matrix_flat = vec![Complex64::new(0.0, 0.0); bl * din * br];
        for l in 0..bl {
            let theta_base = l * din * br;
            let mat_base = l * e0 * cols;
            for r in 0..br {
                for ss_out in 0..din {
                    let mut val = Complex64::new(0.0, 0.0);
                    for ss_in in 0..din {
                        val += gate[ss_out * din + ss_in] * theta[theta_base + ss_in * br + r];
                    }
                    let s0_out = ss_out / e1;
                    let s1_out = ss_out % e1;
                    matrix_flat[mat_base + s0_out * cols + s1_out * br + r] = val;
                }
            }
        }
        drop(theta);

        // Step 3: SVD and truncate (CPU or GPU via the injected fn). The
        // total two-site weight is Σσ²_kept + discarded_weight — with the
        // orthonormal-by-construction kernel that also equals ‖Θ'‖²_F, the
        // pre-SVD Frobenius norm. Nothing below mentions the physical
        // dimension: the certificate is the same quantity for any d.
        let rows = bl * e0;
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
        // Π(1 − εᵢ). Clamped at 0: a single split cannot discard more than the
        // whole block, but floating point at ε ≈ 1 could produce a small
        // negative and a NEGATIVE fidelity would be nonsense to report.
        self.fidelity_estimate *= (1.0 - rel_discarded).max(0.0);
        self.max_bond_reached = self.max_bond_reached.max(new_bm);

        // Step 4: Build new tensors directly from flat U / Vt buffers.
        // A[l, s0, m] = U[(l*e0+s0) * new_bm + m] * sqrt(S[m])
        // B[m, s1, r] = sqrt(S[m]) * Vt[m * cols + s1*br + r]
        let mut new_tl = MpsTensor::with_phys(bl, e0, new_bm);
        let mut new_tr = MpsTensor::with_phys(new_bm, e1, br);

        for l in 0..bl {
            for s0 in 0..e0 {
                for m in 0..new_bm {
                    let sqrt_s = svd.s[m].sqrt();
                    new_tl.set(l, s0, m, svd.u[(l * e0 + s0) * new_bm + m] * sqrt_s);
                }
            }
        }

        for m in 0..new_bm {
            let sqrt_s = svd.s[m].sqrt();
            for s1 in 0..e1 {
                for r in 0..br {
                    new_tr.set(m, s1, r, sqrt_s * svd.vt[m * cols + s1 * br + r]);
                }
            }
        }

        self.tensors[q] = new_tl;
        self.tensors[q + 1] = new_tr;
        self.dims[q] = e0;
        self.dims[q + 1] = e1;
    }

    /// Apply a two-qubit gate to non-adjacent qubits by SWAPping to make them
    /// adjacent. Qubit wrapper over [`Self::apply_2_distant`].
    pub fn apply_2q_distant(&mut self, q0: usize, q1: usize, gate: &[Complex64; 16]) {
        self.assert_qubit_pair(q0, q1, "apply_2q_distant");
        self.apply_2_distant(q0, q1, gate);
    }

    /// Apply a two-site gate to sites `(q0, q1)` in any order and at any
    /// distance, SWAPping `min(q0,q1)` next to `max(q0,q1)` and back. `gate` is
    /// `(d0·d1)×(d0·d1)` with `d0 = dims[q0]`, `d1 = dims[q1]` and combined
    /// index `s0·d1 + s1` where `s0` is `q0`'s level. The SWAPs carry each
    /// site's dimension with its state, and the reverse SWAPs restore `dims`.
    pub fn apply_2_distant(&mut self, q0: usize, q1: usize, gate: &[Complex64]) {
        assert_ne!(q0, q1, "a two-site gate needs two distinct sites");
        let (d0, d1) = (self.dims[q0], self.dims[q1]);
        assert_eq!(
            gate.len(),
            (d0 * d1) * (d0 * d1),
            "apply_2_distant: sites ({q0}, {q1}) have dimensions ({d0}, {d1}), so the gate \
             must be {n}x{n}",
            n = d0 * d1
        );
        let (qa, qb) = if q0 < q1 { (q0, q1) } else { (q1, q0) };

        // In chain order the pair is (qa, qb); re-index the gate when the
        // caller named them the other way round.
        let reoriented;
        let oriented: &[Complex64] = if q0 < q1 {
            gate
        } else {
            reoriented = reorient_pair_gate(gate, d0, d1);
            &reoriented
        };

        if qb - qa == 1 {
            // Already adjacent
            self.apply_2(qa, oriented);
            return;
        }

        // SWAP chain: move qa next to qb
        for i in qa..qb - 1 {
            self.swap_adjacent(i);
        }
        // Apply gate to (qb-1, qb)
        self.apply_2(qb - 1, oriented);
        // SWAP back
        for i in (qa..qb - 1).rev() {
            self.swap_adjacent(i);
        }
    }

    /// SWAP the states of adjacent sites `(i, i+1)`, their dimensions with
    /// them. A qubit pair goes through the exact qubit SWAP (and so through the
    /// accelerator hook, as before).
    fn swap_adjacent(&mut self, i: usize) {
        let (da, db) = (self.dims[i], self.dims[i + 1]);
        if da == 2 && db == 2 {
            let s = swap_matrix();
            self.apply_2(i, &s);
        } else {
            let s = swap_matrix_dims(da, db);
            self.apply_2_reshaping(i, &s, (db, da));
        }
    }

    /// Compute the full statevector by contracting all tensors.
    /// Only feasible for small n (testing).
    pub fn to_statevector(&self) -> Vec<Complex64> {
        // Mixed radix, site 0 the least-significant digit: for an all-qubit
        // chain `digit(q) = (basis >> q) & 1` exactly as before.
        let dim: usize = self.dims.iter().product();
        let mut sv = vec![Complex64::new(0.0, 0.0); dim];

        for basis in 0..dim {
            // For each tensor, the physical index is the q-th digit of `basis`
            let mut left_vec = vec![Complex64::new(1.0, 0.0)]; // 1x1 initial
            let mut rem = basis;

            for q in 0..self.n {
                let d = self.dims[q];
                let digit = rem % d;
                rem /= d;
                let t = &self.tensors[q];
                let mut new_left = vec![Complex64::new(0.0, 0.0); t.bond_right];
                for r in 0..t.bond_right {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for l in 0..t.bond_left {
                        sum += left_vec[l] * t.get(l, digit, r);
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

    /// ⟨ψ|O|ψ⟩ for a **product operator** `O = ⊗_q O_q`, contracted straight
    /// through the tensors — never through a dense statevector.
    ///
    /// `site_ops[q]` is site `q`'s `d×d` operator (`d = self.dims[q]`),
    /// row-major — for a qubit `[m00, m01, m10, m11]` — in the same physical
    /// basis [`Self::to_statevector`] uses: digit `q` of the (mixed-radix)
    /// basis index is site `q`'s physical index. Any slice-like container
    /// works (`[Complex64; 4]` for a qubit chain, `Vec<Complex64>` for a
    /// mixed one).
    ///
    /// # Identity sites must still be passed
    ///
    /// `site_ops` has length `self.n`, and a site outside the operator's
    /// support is passed as an explicit identity (`[1,0,0,1]` on a qubit, the
    /// `d×d` identity on a qudit) rather than skipped. That
    /// is not a convenience of the signature — it is required. Skipping the
    /// sites outside the support is valid **only in canonical gauge**, where
    /// the untouched chain contracts to the identity. This MPS is deliberately
    /// not canonical (see [`Self::fidelity_estimate`] and the truncation note
    /// in [`Self::apply_2q_with_svd_flat`]), so every site contributes its gauge and the sweep
    /// must cross all of them. This is the same assumption whose violation
    /// biased sampled counts until [`Self::right_environments`] replaced it.
    ///
    /// # Cost
    ///
    /// `O(n · χ³)` time, `O(χ²)` live memory: one left-to-right sweep carrying
    /// `L[l,l']` between the ket bond `l` and the bra bond `l'`. Contrast the
    /// dense route — [`Self::to_statevector`] and then index the `2^n` vector —
    /// which is `O(2^n)` in **both**, and which the bond dimension does not
    /// bound at all: a χ=2 GHZ chain at n=31 needs 32 GiB there.
    ///
    /// Memory is `O(χ²)` and not `O(n·χ²)` because only one environment is live
    /// at a time; [`Self::right_environments`]keeps all `n+1`, which is why
    /// the norm is taken from this sweep (see [`Self::expectation_product`])
    /// rather than from [`Self::norm_sqr`].
    ///
    /// The returned value is **unnormalized** — see
    /// [`Self::expectation_product`].
    pub fn contract_product_operator<O: AsRef<[Complex64]>>(&self, site_ops: &[O]) -> Complex64 {
        assert_eq!(
            site_ops.len(),
            self.n,
            "contract_product_operator needs one d x d operator per site, \
             identities included"
        );
        let ops: Vec<&[Complex64]> = site_ops.iter().map(|o| o.as_ref()).collect();
        self.contract_site_operators(&ops)
    }

    /// The sweep behind [`Self::contract_product_operator`]: `site_ops[q]` is
    /// site `q`'s `d×d` operator (`d = self.dims[q]`), row-major. Same loop
    /// nest and summation order as the qubit sweep, with `0..2` replaced by
    /// `0..d`, so an all-qubit chain gives the identical value.
    fn contract_site_operators(&self, site_ops: &[&[Complex64]]) -> Complex64 {
        assert_eq!(site_ops.len(), self.n, "one operator per site");
        let zero = Complex64::new(0.0, 0.0);
        // L[l, l'] over (ket bond, bra bond). Site 0 has bond_left == 1, so the
        // sweep opens on the 1x1 scalar 1.
        let mut left = vec![Complex64::new(1.0, 0.0)];
        for q in 0..self.n {
            let t = &self.tensors[q];
            let (bl, br, d) = (t.bond_left, t.bond_right, t.phys);
            let o = site_ops[q];
            assert_eq!(o.len(), d * d, "site {q}: operator must be {d}x{d}");

            // tmp[l', s, r] = Σ_l L[l,l'] · A[l,s,r]      — O(χ³)
            let mut tmp = vec![zero; bl * d * br];
            for l in 0..bl {
                for lp in 0..bl {
                    let c = left[l * bl + lp];
                    for s in 0..d {
                        let base = (lp * d + s) * br;
                        for r in 0..br {
                            tmp[base + r] += c * t.get(l, s, r);
                        }
                    }
                }
            }

            // u[l', s', r] = Σ_s O[s',s] · tmp[l', s, r]  — O(χ²), and written
            // densely over all entries on purpose: branching past a zero
            // operator entry would make the result depend on the sign of a zero
            // amplitude for some inputs and not others.
            let mut u = vec![zero; bl * d * br];
            for lp in 0..bl {
                for sp in 0..d {
                    for s in 0..d {
                        let coeff = o[sp * d + s];
                        let src = (lp * d + s) * br;
                        let dst = (lp * d + sp) * br;
                        for r in 0..br {
                            u[dst + r] += coeff * tmp[src + r];
                        }
                    }
                }
            }

            // L'[r, r'] = Σ_{l',s'} u[l',s',r] · conj(A[l',s',r'])  — O(χ³)
            let mut next = vec![zero; br * br];
            for lp in 0..bl {
                for sp in 0..d {
                    let src = (lp * d + sp) * br;
                    for r in 0..br {
                        let uv = u[src + r];
                        for rp in 0..br {
                            next[r * br + rp] += uv * t.get(lp, sp, rp).conj();
                        }
                    }
                }
            }
            left = next;
        }
        // bond_right(n-1) == 1, so the sweep closes on a 1x1.
        left[0]
    }

    /// ⟨ψ|O|ψ⟩ / ⟨ψ|ψ⟩ for a product operator — the normalized expectation, and
    /// the number a caller actually wants.
    ///
    /// The normalization is what [`Self::to_statevector`] applies at readout, so
    /// this agrees with a dense readout to floating point.
    ///
    /// # Batching
    ///
    /// This recomputes the state norm on every call. The norm is a property of
    /// the STATE, not of the operator, so a caller evaluating many operators
    /// against one chain — every term of a Hamiltonian, every observable of a
    /// `expectation_multi` — must hoist it: take
    /// [`Self::state_norm_sqr_contracted`] once and divide
    /// [`Self::contract_product_operator`] by it per term. Not doing so is the
    /// same per-observable rework that made the dense readout this replaced
    /// worth complaining about, just one level down. `expectation_from_mps`
    /// hoists; this entry point is the single-shot convenience.
    ///
    /// # Degenerate states
    ///
    /// Returns `0.0` when `⟨ψ|ψ⟩ ≤ 1e-300`, rather than dividing and yielding
    /// `NaN` or `inf`. Note this does NOT match `to_statevector`, which at that
    /// floor SKIPS its normalization and returns the raw tiny vector, so a
    /// dense readout of such a state would give some unnormalized number. The
    /// two disagree there deliberately: a state of that norm carries no
    /// information, and `0.0` is the honest answer where the dense path's is an
    /// artefact of the amplitudes that happened to survive.
    ///
    /// Only the real part is returned: a Pauli string (or any Hermitian
    /// product) has a real expectation, so the imaginary part is zero up to
    /// rounding.
    pub fn expectation_product<O: AsRef<[Complex64]>>(&self, site_ops: &[O]) -> f64 {
        normalize_expectation(
            self.contract_product_operator(site_ops),
            self.state_norm_sqr_contracted(),
        )
    }

    /// `⟨ψ|ψ⟩` via the same sweep [`Self::contract_product_operator`] uses, with
    /// identity operators on every site.
    ///
    /// Equal to [`Self::norm_sqr`] by construction — pinned by
    /// `the_identity_contraction_is_the_state_norm` — and preferred on the
    /// expectation path because `norm_sqr` routes through
    /// [`Self::right_environments`], which holds all `n+1` environments live at
    /// once (`O(n · χ²)`). This sweep keeps one, so an expectation stays
    /// `O(χ²)` in memory end to end, which is the point of the whole item.
    pub fn state_norm_sqr_contracted(&self) -> f64 {
        // Per-site d×d identity. For an all-qubit chain these are exactly the
        // `[1,0,0,1]` operators the qubit sweep always used, so the value is
        // unchanged bit for bit.
        let idents: Vec<Vec<Complex64>> = self
            .dims
            .iter()
            .map(|&d| {
                let mut m = vec![Complex64::new(0.0, 0.0); d * d];
                for s in 0..d {
                    m[s * d + s] = Complex64::new(1.0, 0.0);
                }
                m
            })
            .collect();
        let ops: Vec<&[Complex64]> = idents.iter().map(|m| m.as_slice()).collect();
        self.contract_site_operators(&ops).re
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
            for s in 0..t.phys {
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

    /// Pack per-site bits into an [`Outcome`] of `width` bits.
    ///
    /// This is the wide-capable form: unlike [`Self::pack_bits`], it has no
    /// 64-site ceiling, which is the point of the whole exercise — a 1024-qubit
    /// chain has a perfectly well-defined shot outcome and only the key type
    /// ever prevented reporting it.
    ///
    /// # A bit packer, not a digit packer
    ///
    /// [`Self::sample_bits_with_envs_into`] writes one DIGIT per site, and on
    /// a qudit site that digit can be `>= 2`. Every digit that lands in the
    /// key must be a bit: a `2` packed into a bit string is a silent lie
    /// (it would be read as `0` or as `1` depending on the masking), so it is
    /// refused by assertion. A site the projection drops never enters the key
    /// and is not checked.
    pub fn pack_outcome(
        bits: &[u8],
        cbit_of: Option<&[Option<u32>]>,
        width: u32,
    ) -> omega_core::outcome::Outcome {
        match cbit_of {
            None => {
                let packed = &bits[..width as usize];
                assert_bits(packed, "pack_outcome");
                omega_core::outcome::Outcome::from_bits(packed)
            }
            Some(map) => {
                let mut o = omega_core::outcome::Outcome::zeros(width);
                for (q, bit) in bits.iter().enumerate() {
                    if let Some(Some(c)) = map.get(q) {
                        assert_bits(std::slice::from_ref(bit), "pack_outcome");
                        o.set_bit(*c, *bit);
                    }
                }
                o
            }
        }
    }

    /// Pack per-site bits into a counts key.
    ///
    /// `cbit_of` is `Some` when the key is the CLASSICAL register: each site
    /// lands at its classical index, and an unmeasured site is dropped. `None`
    /// keys on the qubit register, which is only representable below 64 sites.
    ///
    /// Like [`Self::pack_outcome`], a BIT packer: every digit that lands in
    /// the key must be `< 2`, and a qudit digit `>= 2` is refused by assertion
    /// rather than masked into a bit.
    pub fn pack_bits(bits: &[u8], cbit_of: Option<&[Option<u32>]>) -> u64 {
        let mut result = 0u64;
        for (q, bit) in bits.iter().enumerate() {
            match cbit_of {
                Some(map) => {
                    if let Some(Some(c)) = map.get(q) {
                        assert_bits(std::slice::from_ref(bit), "pack_bits");
                        result |= ((*bit & 1) as u64) << *c;
                    }
                }
                None => {
                    assert_bits(std::slice::from_ref(bit), "pack_bits");
                    result |= ((*bit & 1) as u64) << q;
                }
            }
        }
        result
    }

    /// Sample a shot, writing **one digit per site** into `out`: `out[q]` is
    /// site `q`'s level, `0 <= out[q] < dims[q]` (a bit on a qubit site).
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
    ///
    /// # The draw, and why `d = 2` is unchanged bit for bit
    ///
    /// One uniform `u` per site, and only when the branch weights are not all
    /// zero; the level is the first `k` with `u < (p_0 + … + p_k) / total`,
    /// the last level otherwise. At `d = 2` that is exactly the old
    /// `u < p_0 / total ? 0 : 1` — the same comparison on the same operands,
    /// the same RNG consumption — so a qubit chain draws the same stream.
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
        // allocations dominate otherwise. One branch vector per level.
        let dmax = self.dims.iter().copied().max().unwrap_or(2);
        let mut w: Vec<Vec<Complex64>> = vec![Vec::new(); dmax];
        let mut p = vec![0.0f64; dmax];

        for q in 0..self.n {
            let t = &self.tensors[q];
            let (bl, br, d) = (t.bond_left, t.bond_right, t.phys);
            let e_next = &envs[q + 1];

            // w_s[r] = Σ_l v[l] A[l,s,r];  p_s = Re(w_s E_{q+1} w_s†) ≥ 0.
            for (level, (w_s, p_s)) in w.iter_mut().zip(p.iter_mut()).take(d).enumerate() {
                w_s.clear();
                w_s.resize(br, zero);
                for (r, w_r) in w_s.iter_mut().enumerate() {
                    for l in 0..bl {
                        *w_r += left_vec[l] * t.get(l, level, r);
                    }
                }
                let mut prob = zero;
                for r in 0..br {
                    for rp in 0..br {
                        prob += w_s[r] * e_next[r * br + rp] * w_s[rp].conj();
                    }
                }
                *p_s = prob.re.max(0.0);
            }

            // Summed left to right: for d = 2 this is `p[0] + p[1]`.
            let mut total = p[0];
            for &pk in &p[1..d] {
                total += pk;
            }
            let digit = if total > 0.0 {
                pick_level(rng.random::<f64>(), &p[..d], total)
            } else {
                d - 1
            };
            out[q] = digit as u8;

            // Condition on the outcome; rescale so the running prefix
            // stays O(1) (only probability *ratios* matter downstream).
            let norm = p[digit].sqrt();
            let inv = if norm > 0.0 { 1.0 / norm } else { 1.0 };
            std::mem::swap(&mut left_vec, &mut w[digit]);
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
            for s in 0..t.phys {
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

    /// Mid-circuit projective measurement of a single site.
    /// Computes the outcome probabilities, samples a level, projects the local
    /// tensor, and renormalizes. Returns the measured DIGIT (`0..dims[q]`; 0
    /// or 1 on a qubit).
    ///
    /// The outcome probabilities contract the full state — left environment,
    /// local tensor, right environment — because local tensor norms are only
    /// the true marginals when the chain is canonical around site q, which
    /// plain-SVD gate splits do not maintain (same mechanism as the sampling
    /// fix in [`Mps::sample_with_envs`]).
    ///
    /// At `d = 2` the draw is the old one exactly: one uniform `u`, always
    /// consumed; outcome 0 iff `u < p_0 / total` (or `u < 0.5` on a zero-norm
    /// site, since `(0 + 1) / 2 == 0.5`).
    pub fn measure_site(&mut self, q: usize, rng: &mut impl rand::Rng) -> u8 {
        let left = self.left_environment(q);
        let right = self.right_environments();
        let e_next = &right[q + 1];
        let t = &self.tensors[q];
        let bl = t.bond_left;
        let br = t.bond_right;
        let d = t.phys;
        let zero = Complex64::new(0.0, 0.0);

        // p(b) = Σ_{l,l',r,r'} L[l,l'] A[l,b,r] E[r,r'] conj(A[l',b,r'])
        let mut norm_sq = vec![0.0_f64; d];
        for (phys, p_level) in norm_sq.iter_mut().enumerate() {
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
            *p_level = p.re.max(0.0);
        }

        // Summed left to right: for d = 2 this is `norm_sq[0] + norm_sq[1]`.
        let mut total = norm_sq[0];
        for &pk in &norm_sq[1..] {
            total += pk;
        }
        let u = rng.random::<f64>();
        let outcome = if total > 0.0 {
            pick_level(u, &norm_sq, total)
        } else {
            // Degenerate site: uniform over the levels (0.5 at d = 2).
            (0..d)
                .find(|&k| u < (k + 1) as f64 / d as f64)
                .unwrap_or(d - 1)
        };

        // Project: zero out the other levels and renormalize
        let norm = norm_sq[outcome].sqrt();
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 1.0 };

        let new_data: Vec<Complex64> = (0..bl * d * br)
            .map(|idx| {
                let phys = (idx / br) % d;
                if phys == outcome {
                    self.tensors[q].data[idx] * inv_norm
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();
        self.tensors[q].data = new_data;

        outcome as u8
    }
}

/// The level a uniform draw `u` selects from unnormalised weights `p` summing
/// to `total > 0`: the first `k` with `u < (p_0 + … + p_k) / total`, the last
/// level if rounding leaves `u` above every partial sum. At `d = 2` the only
/// comparison is `u < p_0 / total`, the old qubit draw verbatim.
fn pick_level(u: f64, p: &[f64], total: f64) -> usize {
    let d = p.len();
    let mut cum = 0.0;
    for (k, &pk) in p.iter().enumerate().take(d - 1) {
        cum = if k == 0 { pk } else { cum + pk };
        if u < cum / total {
            return k;
        }
    }
    d - 1
}

/// Refuse a digit `>= 2` on its way into a bit-string key.
fn assert_bits(digits: &[u8], what: &str) {
    if let Some(bad) = digits.iter().find(|&&b| b >= 2) {
        panic!(
            "{what}: digit {bad} is not a bit — a qudit outcome packed into a bit-string \
             key would be a silent lie (d = 2 sites only)"
        );
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

/// The `(da·db)²` SWAP between adjacent sites of dimensions `(da, db)`:
/// `|a, b⟩ → |b, a⟩`, row-major, input combined index `a·db + b` and output
/// combined index `b·da + a` (the output pair has dimensions `(db, da)`). For
/// `da = db = 2` it is [`swap_matrix`] entry for entry.
fn swap_matrix_dims(da: usize, db: usize) -> Vec<Complex64> {
    let n = da * db;
    let mut m = vec![Complex64::new(0.0, 0.0); n * n];
    for a in 0..da {
        for b in 0..db {
            m[(b * da + a) * n + (a * db + b)] = Complex64::new(1.0, 0.0);
        }
    }
    m
}

/// Re-orient a two-site gate from the pair `(q0, q1)` to `(q1, q0)`:
/// `G'[(s1·d0+s0), (t1·d0+t0)] = G[(s0·d1+s1), (t0·d1+t1)]`, `d0 = dim(q0)`,
/// `d1 = dim(q1)`. Pure data movement; for `d0 = d1 = 2` it is exactly the old
/// qubit kernel's "swap rows 1,2 and cols 1,2".
fn reorient_pair_gate(gate: &[Complex64], d0: usize, d1: usize) -> Vec<Complex64> {
    let n = d0 * d1;
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for s0 in 0..d0 {
        for s1 in 0..d1 {
            for t0 in 0..d0 {
                for t1 in 0..d1 {
                    out[(s1 * d0 + s0) * n + (t1 * d0 + t0)] =
                        gate[(s0 * d1 + s1) * n + (t0 * d1 + t1)];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- 3b.1: contracting an operator through the tensors ----------------

    /// Deterministic xorshift. The corpus below is a *class* assertion, so it
    /// has to be the same class on every machine and every run — a randomized
    /// oracle test that fails one time in fifty teaches nothing.
    struct Xs(u64);
    impl Xs {
        fn f(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn u3(theta: f64, phi: f64, lam: f64) -> [Complex64; 4] {
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        [
            Complex64::new(c, 0.0),
            -Complex64::from_polar(s, lam),
            Complex64::from_polar(s, phi),
            Complex64::from_polar(c, phi + lam),
        ]
    }

    fn cx_gate() -> [Complex64; 16] {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        // |00>,|01>,|10>,|11> with qubit0 = LOW bit, control = first argument
        // of apply_2q. Whatever this is exactly, both sides of the comparison
        // see the same state, which is all the oracle test needs.
        [
            o, z, z, z, //
            z, o, z, z, //
            z, z, z, o, //
            z, z, o, z,
        ]
    }

    fn pauli_ops() -> Vec<(&'static str, [Complex64; 4])> {
        let z = Complex64::new(0.0, 0.0);
        let p1 = Complex64::new(1.0, 0.0);
        let m1 = Complex64::new(-1.0, 0.0);
        let pi = Complex64::new(0.0, 1.0);
        let mi = Complex64::new(0.0, -1.0);
        vec![
            ("I", [p1, z, z, p1]),
            ("X", [z, p1, p1, z]),
            ("Y", [z, mi, pi, z]),
            ("Z", [p1, z, z, m1]),
        ]
    }

    /// ⟨ψ|O|ψ⟩ over the DENSE statevector, summing every basis PAIR.
    ///
    /// The independent second implementation: a different algorithm (enumerate
    /// `4^n` basis pairs and factor the operator per site) reaching the number
    /// the environment sweep reaches. `sv` comes from `to_statevector`, which is
    /// already normalized, so this returns the normalized expectation.
    fn dense_product_expectation(sv: &[Complex64], n: usize, ops: &[[Complex64; 4]]) -> f64 {
        let dim = 1usize << n;
        let mut acc = Complex64::new(0.0, 0.0);
        for b in 0..dim {
            for bp in 0..dim {
                let mut m = Complex64::new(1.0, 0.0);
                for q in 0..n {
                    let s = (b >> q) & 1;
                    let sp = (bp >> q) & 1;
                    m *= ops[q][sp * 2 + s];
                }
                acc += sv[bp].conj() * m * sv[b];
            }
        }
        acc.re
    }

    /// Random circuits at a range of bond caps. The starved ones are the point:
    /// truncation is what leaves the chain non-canonical, and a canonical-form
    /// assumption in the contraction would survive every exact case and fail
    /// only here.
    fn corpus(n: usize) -> Vec<(String, Mps)> {
        let mut out = Vec::new();
        for (ci, chi) in [1usize, 2, 4, 32].into_iter().enumerate() {
            for depth in [1usize, 3, 6] {
                let mut rng = Xs(0x2026_0827 + (ci as u64) * 7 + depth as u64);
                let mut mps = Mps::zero_state(n, chi);
                for _ in 0..depth {
                    for q in 0..n {
                        let g = u3(
                            rng.f() * std::f64::consts::PI,
                            rng.f() * std::f64::consts::TAU,
                            rng.f() * std::f64::consts::TAU,
                        );
                        mps.apply_1q(q, &g);
                    }
                    for q in 0..n - 1 {
                        mps.apply_2q(q, &cx_gate());
                    }
                }
                out.push((format!("chi={chi} depth={depth}"), mps));
            }
        }
        out
    }

    /// THE load-bearing test for 3b.1: the tensor contraction and the dense
    /// route agree, on every state in the corpus and every single-site Pauli,
    /// plus a two-site and a full-width string.
    #[test]
    fn the_tensor_contraction_matches_the_dense_operator_on_every_corpus_state() {
        const N: usize = 5;
        let paulis = pauli_ops();
        let mut checked = 0usize;
        let mut truncated_states = 0usize;
        for (label, mps) in corpus(N) {
            let sv = mps.to_statevector();
            if mps.discarded_weight > 1e-12 {
                truncated_states += 1;
            }
            // Every single-site placement of every Pauli.
            for (pname, pm) in &paulis {
                for q in 0..N {
                    let mut ops = vec![paulis[0].1; N];
                    ops[q] = *pm;
                    let got = mps.expectation_product(&ops);
                    let want = dense_product_expectation(&sv, N, &ops);
                    assert!(
                        (got - want).abs() < 1e-9,
                        "{label}: {pname}{q} contraction {got} vs dense {want}"
                    );
                    checked += 1;
                }
            }
            // A two-site string (adjacent and distant) and a full-width one —
            // the odd-Y count varies across these, which is what exercises the
            // Y phase convention rather than assuming it.
            let combos: Vec<Vec<usize>> = vec![
                vec![1, 2, 0, 0, 0],
                vec![2, 0, 0, 3, 0],
                vec![1, 2, 3, 2, 1],
                vec![2, 2, 2, 0, 0],
            ];
            for combo in combos {
                let ops: Vec<[Complex64; 4]> = combo.iter().map(|&i| paulis[i].1).collect();
                let got = mps.expectation_product(&ops);
                let want = dense_product_expectation(&sv, N, &ops);
                assert!(
                    (got - want).abs() < 1e-9,
                    "{label}: string {combo:?} contraction {got} vs dense {want}"
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 200,
            "corpus too thin to mean anything: {checked}"
        );
        assert!(
            truncated_states >= 3,
            "the corpus must contain genuinely TRUNCATED states or it never \
             exercises the non-canonical gauge — got {truncated_states}"
        );
    }

    /// Pins the claim `expectation_product` relies on: contracting the identity
    /// is the state norm, so the normalization can be taken from the same sweep
    /// instead of from `norm_sqr`'s all-environments-live route.
    #[test]
    fn the_identity_contraction_is_the_state_norm() {
        for (label, mps) in corpus(5) {
            let ident = [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ];
            let via_sweep = mps.contract_product_operator(&vec![ident; mps.n]).re;
            let via_envs = mps.norm_sqr();
            assert!(
                (via_sweep - via_envs).abs() < 1e-12 * via_envs.abs().max(1.0),
                "{label}: identity sweep {via_sweep} vs norm_sqr {via_envs}"
            );
        }
    }

    /// The doc on `contract_product_operator` says identity sites cannot be
    /// skipped because this MPS is not canonical. That is a claim, so it is
    /// asserted rather than trusted: the canonical shortcut — contract site `q`
    /// alone and treat both chains as the identity — is computed here and shown
    /// to DISAGREE. Without this test the caveat is folklore, and the next
    /// person to want a faster path deletes it.
    #[test]
    fn skipping_identity_sites_would_be_wrong_on_this_non_canonical_mps() {
        const N: usize = 5;
        let z_op = pauli_ops()[3].1;
        let ident = pauli_ops()[0].1;
        let mut disagreements = 0usize;
        let mut compared = 0usize;
        for (_label, mps) in corpus(N) {
            for q in 0..N {
                let mut ops = vec![ident; N];
                ops[q] = z_op;
                let truth = mps.expectation_product(&ops);

                // The shortcut: site q's local reduced expectation, assuming
                // the chains either side contract to the identity.
                let t = &mps.tensors[q];
                let (mut num, mut den) = (0.0f64, 0.0f64);
                for l in 0..t.bond_left {
                    for r in 0..t.bond_right {
                        for s in 0..2 {
                            let a = t.get(l, s, r);
                            let w = a.norm_sqr();
                            den += w;
                            num += if s == 1 { -w } else { w };
                        }
                    }
                }
                let shortcut = if den > 1e-300 { num / den } else { 0.0 };
                compared += 1;
                if (truth - shortcut).abs() > 1e-6 {
                    disagreements += 1;
                }
            }
        }
        assert!(
            disagreements > 0,
            "the canonical shortcut agreed on all {compared} cases, which would \
             mean the gauge warning on `contract_product_operator` is wrong — \
             check whether the MPS became canonical before relaxing the doc"
        );
    }

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
