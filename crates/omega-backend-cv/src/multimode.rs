// SPDX-License-Identifier: Apache-2.0
//! A **multi-mode** truncated-Fock state, and the only API by which a padded
//! single-mode operator may touch it.
//!
//! `n` modes at `cutoff` levels is a `cutoff^n` product basis. The indexing is
//! mixed-radix with **mode 0 most significant**, so mode `m` has stride
//! `cutoff^(n-1-m)`.
//!
//! # The API exists to make one mistake unwriteable
//!
//! Truncated displacement and squeezing are not unitary, so the single-mode code
//! computes in a padded space (`cutoff + 96`) and cuts back. Lifting a padded
//! operator into the product basis costs `(cutoff+96)^n` — **18.41 TiB** at
//! `cutoff = 8, n = 6`, against 4 MiB for the state itself. See
//! [`crate::capacity`], which pins both numbers.
//!
//! [`MultiFockState::apply_single_mode`] is therefore the *only* way to apply a
//! single-mode operator here, and its signature makes the padded dimension
//! structurally incapable of entering the product: the callback is handed one
//! **fiber** — the `cutoff` amplitudes along one mode's axis at fixed occupation
//! of every other mode — and writes `cutoff` amplitudes back. Whatever padding
//! it uses internally is its own scratch, bounded by `padded_dim` and *not* by
//! anything raised to the power of `n`.
//!
//! Peak extra memory is therefore `O(padded_dim)` for the streaming form used
//! here, which is even better than the `cutoff^(n-1) * padded_dim` bound
//! [`crate::capacity::mode_local_temp_amplitudes`] reports — that figure is the
//! conservative bound for a batched implementation that materialises every fiber
//! at once. Stated rather than left as a discrepancy between two numbers.
//!
//! Total arithmetic is `cutoff^n` amplitudes touched per gate, which is optimal:
//! every amplitude is read and written once.

use num_complex::Complex64;

use crate::{capacity, CvError};

/// `n_modes` CV modes, each truncated at `cutoff` Fock levels.
#[derive(Clone, Debug)]
pub struct MultiFockState {
    cutoff: usize,
    n_modes: usize,
    amps: Vec<Complex64>,
    /// Probability the cutoff has discarded, summed over every fiber of every
    /// operation. NOT the single-mode scalar: a multi-mode operation spills on
    /// each fiber independently, so the total is a sum and reusing a per-fiber
    /// figure would under-report.
    lost_norm: f64,
    /// Photon-number-weighted discarded mass, `Σ k·p_k` over the discarded tail.
    lost_n_weight: f64,
}

impl MultiFockState {
    /// The joint vacuum `|0,0,..,0>`.
    ///
    /// Refuses through [`capacity::check`] before allocating, so an impossible
    /// shape is an error rather than an allocation attempt. `budget_bytes` is
    /// the caller's memory allowance — see the note in [`crate::capacity`] on
    /// why this crate does not probe the host itself.
    pub fn vacuum(
        cutoff: usize,
        n_modes: usize,
        budget_bytes: Option<usize>,
    ) -> Result<Self, CvError> {
        capacity::check(cutoff, n_modes, budget_bytes)?;
        if n_modes == 0 {
            return Err(CvError::Unrepresentable {
                what: "a state with zero modes has no amplitudes",
            });
        }
        let dim = capacity::product_dim(cutoff, n_modes).ok_or(CvError::Unrepresentable {
            what: "cutoff^n_modes overflows a usize",
        })?;
        let mut amps = vec![Complex64::new(0.0, 0.0); dim];
        amps[0] = Complex64::new(1.0, 0.0);
        Ok(Self {
            cutoff,
            n_modes,
            amps,
            lost_norm: 0.0,
            lost_n_weight: 0.0,
        })
    }

    pub fn cutoff(&self) -> usize {
        self.cutoff
    }

    pub fn n_modes(&self) -> usize {
        self.n_modes
    }

    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amps
    }

    /// Stride of `mode`'s axis: mode 0 is most significant.
    fn stride(&self, mode: usize) -> usize {
        self.cutoff.pow((self.n_modes - 1 - mode) as u32)
    }

    /// Flat index for an occupation vector, or `None` if any entry is at or
    /// above the cutoff.
    pub fn index_of(&self, occupation: &[usize]) -> Option<usize> {
        if occupation.len() != self.n_modes {
            return None;
        }
        let mut idx = 0usize;
        for (m, &n) in occupation.iter().enumerate() {
            if n >= self.cutoff {
                return None;
            }
            idx += n * self.stride(m);
        }
        Some(idx)
    }

    /// Occupation vector for a flat index. Inverse of [`Self::index_of`].
    pub fn occupation_of(&self, mut index: usize) -> Vec<usize> {
        let mut occ = vec![0usize; self.n_modes];
        for (m, slot) in occ.iter_mut().enumerate() {
            let s = self.stride(m);
            *slot = index / s;
            index %= s;
        }
        occ
    }

    /// **Apply a single-mode operator, mode-locally.**
    ///
    /// `op` is called once per fiber with `(input, output)`, each of length
    /// `cutoff`: the amplitudes along `mode`'s axis at one fixed occupation of
    /// every other mode. It must write all `cutoff` outputs.
    ///
    /// This is the shape that makes the padded-product blow-up unwriteable — see
    /// the module docs. A caller needing padding allocates it *inside* `op`,
    /// where it is bounded by `padded_dim` and cannot become a factor of the
    /// product basis.
    pub fn apply_single_mode<F>(&mut self, mode: usize, mut op: F) -> Result<(), CvError>
    where
        F: FnMut(&[Complex64], &mut [Complex64]),
    {
        if mode >= self.n_modes {
            return Err(CvError::Unrepresentable {
                what: "mode index is out of range for this state",
            });
        }
        let c = self.cutoff;
        let stride = self.stride(mode);
        // Fibers are enumerated by (block, offset): the flat index of a fiber's
        // k-th element is `block * stride * c + offset + k * stride`. That walks
        // every fiber exactly once with no allocation per fiber beyond the two
        // scratch vectors below.
        let blocks = self.amps.len() / (stride * c);
        let mut fin = vec![Complex64::new(0.0, 0.0); c];
        let mut fout = vec![Complex64::new(0.0, 0.0); c];
        for block in 0..blocks {
            for offset in 0..stride {
                let base = block * stride * c + offset;
                for (k, slot) in fin.iter_mut().enumerate() {
                    *slot = self.amps[base + k * stride];
                }
                for v in fout.iter_mut() {
                    *v = Complex64::new(0.0, 0.0);
                }
                op(&fin, &mut fout);
                for (k, v) in fout.iter().enumerate() {
                    self.amps[base + k * stride] = *v;
                }
            }
        }
        Ok(())
    }

    /// Displace one mode by `alpha`, **mode-locally**.
    ///
    /// Uses the same [`crate::displace_fiber`] the single-mode path uses — one
    /// implementation of the Miatto–Quesada recurrence, not two. The padded
    /// scratch lives inside the callback, so it is `cutoff + PAD` amplitudes
    /// once, never `(cutoff + PAD)^n_modes`.
    ///
    /// Truncation is accounted for across **all** fibers: the probability that
    /// spilled past the cutoff is summed, because a per-fiber figure would
    /// under-report the state's total loss. The single-mode scalar cannot be
    /// reused unchanged for this reason.
    /// The Fock (number) basis state `|n₀, n₁, …⟩`.
    ///
    /// The natural preparation for a photon-counting experiment, and the one
    /// Hong–Ou–Mandel needs: `|1,1⟩` is not reachable from
    /// [`Self::vacuum`] by any operation this type has, because displacement
    /// makes coherent states and nothing here creates a definite photon number.
    ///
    /// Mirrors piquasso's `NumberState`, so a cross-check can prepare the same
    /// input on both sides without a conversion step in between.
    ///
    /// Exact and lossless: a basis state IS representable, so unlike the
    /// coherent and squeezed constructors there is no tail to account for.
    /// Refuses an occupation that does not fit rather than clamping it —
    /// clamping would silently prepare a different experiment.
    pub fn number_state(
        cutoff: usize,
        n_modes: usize,
        occupation: &[usize],
        budget_bytes: Option<usize>,
    ) -> Result<Self, CvError> {
        let mut st = Self::vacuum(cutoff, n_modes, budget_bytes)?;
        if occupation.len() != n_modes {
            return Err(CvError::Unrepresentable {
                what: "occupation length does not match the mode count",
            });
        }
        let idx = st.index_of(occupation).ok_or(CvError::Unrepresentable {
            what: "occupation has a mode at or above the cutoff",
        })?;
        for v in st.amps.iter_mut() {
            *v = Complex64::new(0.0, 0.0);
        }
        st.amps[idx] = Complex64::new(1.0, 0.0);
        Ok(st)
    }

    /// Beamsplitter on the mode pair `(a, b)`:
    /// `BS(θ, φ) = exp[θ(e^{iφ} a†b − e^{−iφ} a b†)]`.
    ///
    /// **The first operation in this type that couples two modes.** Until now a
    /// `MultiFockState` was a product basis with no way to entangle across it,
    /// so every multi-mode state was just independent modes and Hong–Ou–Mandel
    /// was unreachable — HOM *is* a beamsplitter.
    ///
    /// # No `PAD`, and that is a property of the physics rather than a shortcut
    ///
    /// [`crate::PAD`] exists because displacement and squeezing do not conserve
    /// photon number: they push amplitude up an unbounded ladder, so the exact
    /// operator has no finite matrix and the padding buys accuracy.
    ///
    /// A beamsplitter **conserves total photon number**. It only moves photons
    /// *between* the two modes, so it block-diagonalises by `N = n_a + n_b`, and
    /// within a block the exact operator is a finite `(N+1)×(N+1)` matrix. There
    /// is nothing to pad: this is the whole operator, not a truncation of it.
    ///
    /// That matters for cost as much as accuracy. Padding two modes of a product
    /// basis would cost `(cutoff + PAD)²` per fiber — the direction that reaches
    /// 18.41 TiB in [`crate::capacity`]. Blocks cap at `2·cutoff − 1`.
    ///
    /// # Where truncation still bites
    ///
    /// The block is exact; the *representable* part of it is not. A state
    /// `|k, N−k⟩` needs `k < cutoff` **and** `N − k < cutoff`, so for
    /// `N ≥ cutoff` the block extends beyond what this basis can hold. The
    /// operator is applied over the full block and the amplitude landing outside
    /// is accumulated into [`Self::lost_norm`], per fiber, the same way every
    /// other operation here reports its spill.
    ///
    /// So a beamsplitter is exact whenever `n_a + n_b < cutoff` for every
    /// occupied basis state, and lossy above it — and it says which by reporting.
    ///
    /// # Convention
    ///
    /// `(θ, φ)` matches piquasso's `Beamsplitter(theta, phi)`. Verified before
    /// this was written rather than after: the block generator was prototyped
    /// against piquasso 8.0.1 over 25 combinations of input occupation, `θ` and
    /// **non-zero `φ`**, worst `|Δp|` 4.163e-16.
    ///
    /// The phase is supported here where [`crate::FockState::squeeze`] refuses
    /// it, and the difference is not taste: squeezing had no oracle covering
    /// `φ`, so its convention would have rested on my reading of the literature.
    /// This one is measured.
    pub fn beamsplitter(
        &mut self,
        mode_a: usize,
        mode_b: usize,
        theta: f64,
        phi: f64,
    ) -> Result<(), CvError> {
        if mode_a >= self.n_modes || mode_b >= self.n_modes {
            return Err(CvError::Unrepresentable {
                what: "beamsplitter mode index is out of range for this state",
            });
        }
        if mode_a == mode_b {
            return Err(CvError::Unrepresentable {
                what: "a beamsplitter needs two DISTINCT modes",
            });
        }
        if !theta.is_finite() || !phi.is_finite() {
            return Err(CvError::Unrepresentable {
                what: "beamsplitter angle is not finite",
            });
        }

        let c = self.cutoff;
        let sa = self.stride(mode_a);
        let sb = self.stride(mode_b);
        let max_n = 2 * (c - 1);

        // Block matrices depend only on (theta, phi, N) — never on the fiber —
        // so they are built once and reused across every fiber.
        let blocks: Vec<Vec<Complex64>> = (0..=max_n)
            .map(|n| beamsplitter_block(n, theta, phi))
            .collect();

        let mut spill_mass = 0.0f64;
        let mut spill_weight = 0.0f64;
        let mut vin: Vec<Complex64> = Vec::with_capacity(max_n + 1);
        let mut vout: Vec<Complex64> = Vec::with_capacity(max_n + 1);

        // A fiber is one assignment of the OTHER modes. Its base index is any
        // flat index whose digits for both `mode_a` and `mode_b` are zero.
        for base in 0..self.amps.len() {
            if !(base / sa).is_multiple_of(c) || !(base / sb).is_multiple_of(c) {
                continue;
            }
            for (n, u) in blocks.iter().enumerate() {
                let dim = n + 1;
                vin.clear();
                vin.resize(dim, Complex64::new(0.0, 0.0));
                let mut any = false;
                for (k, slot) in vin.iter_mut().enumerate() {
                    // |k, n-k>: representable only if BOTH fit under the cutoff.
                    let other = n - k;
                    if k < c && other < c {
                        let v = self.amps[base + k * sa + other * sb];
                        *slot = v;
                        if v != Complex64::new(0.0, 0.0) {
                            any = true;
                        }
                    }
                }
                if !any {
                    continue;
                }
                vout.clear();
                vout.resize(dim, Complex64::new(0.0, 0.0));
                for (row, out) in vout.iter_mut().enumerate() {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for (col, x) in vin.iter().enumerate() {
                        acc += u[row * dim + col] * x;
                    }
                    *out = acc;
                }
                for (k, v) in vout.iter().enumerate() {
                    let other = n - k;
                    if k < c && other < c {
                        self.amps[base + k * sa + other * sb] = *v;
                    } else {
                        // Outside the representable corner of this block.
                        let p = v.norm_sqr();
                        spill_mass += p;
                        spill_weight += (n as f64) * p;
                    }
                }
            }
        }

        self.lost_norm = (self.lost_norm + spill_mass).min(1.0);
        self.lost_n_weight += spill_weight;
        Ok(())
    }

    /// Two-mode squeezing on `(a, b)`:
    /// `S₂(r, φ) = exp[r(e^{iφ} a†b† − e^{−iφ} a b)]`, piquasso's
    /// `Squeezing2(r, phi)` convention.
    ///
    /// # `PAD` is back, and the chain structure says why
    ///
    /// A beamsplitter conserves the photon **total**, so its blocks are finite
    /// and there is nothing to pad. Two-mode squeezing creates photons in
    /// PAIRS — one in each mode — so what it conserves is the **difference**
    /// `d = n_a − n_b`. The state space splits into chains
    /// `|d+j, j⟩, j = 0, 1, …`, each chain an unbounded ladder exactly like
    /// single-mode squeezing's even ladder. So each chain is applied into a
    /// `cutoff + PAD` scratch and the mass that lands outside the
    /// representable corner (`d+j < cutoff` AND `j < cutoff`) is reported,
    /// additively, matching this type's other operations (NOT the single-mode
    /// squeeze's `√ε` fold — that convention belongs to `FockState` and mixing
    /// the two inside one type would make `lost_norm` mean two things).
    ///
    /// # The operator, per chain
    ///
    /// The disentangled (normal-ordered) form
    ///
    /// ```text
    ///   S₂ = exp(λ e^{iφ} a†b†) · sech(r)^{n_a + n_b + 1} · exp(−λ e^{−iφ} a b),
    ///   λ = tanh r
    /// ```
    ///
    /// applied as three passes along the chain: a triangular lowering sum, the
    /// diagonal `sech^{d+2j+1}`, a triangular raising sum. Each pass's
    /// incremental factor at step `k` is `√((d+j+k)(j+k))` (lowering) and
    /// `√((d+j−k+1)(j−k+1))` (raising) — the second is NOT the first with a
    /// sign flipped, and getting it wrong reproduces the vacuum column
    /// perfectly while corrupting every excited input (measured: `|Δp|` up to
    /// 0.36 on `|1,1⟩` with only the vacuum cases passing).
    ///
    /// # Convention verified BEFORE this was written
    ///
    /// Like the beamsplitter above and for the same reason: the three-pass
    /// form was prototyped against piquasso 8.0.1 over six input occupations
    /// including non-zero `φ` both signs, compared far from the truncation
    /// boundary, worst `|Δp|` 4.4e-15. Column `j=0` of the `d=0` chain is the
    /// textbook two-mode squeezed vacuum `sech r · (e^{iφ} tanh r)^j |j,j⟩`.
    pub fn squeezing2(
        &mut self,
        mode_a: usize,
        mode_b: usize,
        r: f64,
        phi: f64,
    ) -> Result<(), CvError> {
        if mode_a >= self.n_modes || mode_b >= self.n_modes {
            return Err(CvError::Unrepresentable {
                what: "squeezing2 mode index is out of range for this state",
            });
        }
        if mode_a == mode_b {
            return Err(CvError::Unrepresentable {
                what: "two-mode squeezing needs two DISTINCT modes",
            });
        }
        if !r.is_finite() || !phi.is_finite() {
            return Err(CvError::Unrepresentable {
                what: "squeezing2 parameter is not finite",
            });
        }

        let c = self.cutoff;
        let sa = self.stride(mode_a);
        let sb = self.stride(mode_b);
        let l = c + crate::PAD;
        let lam = r.tanh();
        let sech = 1.0 / r.cosh();
        let zero = Complex64::new(0.0, 0.0);
        let lower = Complex64::from_polar(-lam, -phi);
        let raise = Complex64::from_polar(lam, phi);

        let mut x = vec![zero; l];
        let mut y = vec![zero; l];
        let mut out = vec![zero; l];
        let mut spill_mass = 0.0f64;
        let mut spill_weight = 0.0f64;

        for base in 0..self.amps.len() {
            if !(base / sa).is_multiple_of(c) || !(base / sb).is_multiple_of(c) {
                continue;
            }
            // Chains by difference d: |d+j⟩ on the `sh` mode, |j⟩ on `sl`.
            // The operator is symmetric under a↔b (a†b† is), so the d > 0
            // chains of the swapped orientation cover n_b > n_a, and d = 0
            // exists once.
            for d in 0..c {
                for &(sh, sl) in &[(sa, sb), (sb, sa)] {
                    if d == 0 && sh == sb {
                        continue;
                    }
                    let mut any = false;
                    for (j, slot) in x.iter_mut().enumerate() {
                        *slot = if d + j < c && j < c {
                            let v = self.amps[base + (d + j) * sh + j * sl];
                            if v != zero {
                                any = true;
                            }
                            v
                        } else {
                            zero
                        };
                    }
                    if !any {
                        continue;
                    }
                    // Lowering pass, then the diagonal folded into the same
                    // loop: y[j] = sech^{d+2j+1} · Σ_k (−λe^{−iφ})^k/k! ·
                    // Π√((d+j+i)(j+i)) · x[j+k]
                    for j in 0..l {
                        let mut acc = zero;
                        let mut term = Complex64::new(1.0, 0.0);
                        for k in 0..(l - j) {
                            if k > 0 {
                                term *=
                                    lower / (k as f64) * (((d + j + k) * (j + k)) as f64).sqrt();
                            }
                            acc += term * x[j + k];
                        }
                        y[j] = acc * sech.powi((d + 2 * j + 1) as i32);
                    }
                    // Raising pass.
                    for j in 0..l {
                        let mut acc = zero;
                        let mut term = Complex64::new(1.0, 0.0);
                        for k in 0..=j {
                            if k > 0 {
                                term *= raise / (k as f64)
                                    * (((d + j - k + 1) * (j - k + 1)) as f64).sqrt();
                            }
                            acc += term * y[j - k];
                        }
                        out[j] = acc;
                    }
                    for (j, v) in out.iter().enumerate() {
                        if d + j < c && j < c {
                            self.amps[base + (d + j) * sh + j * sl] = *v;
                        } else {
                            let p = v.norm_sqr();
                            spill_mass += p;
                            spill_weight += ((d + 2 * j) as f64) * p;
                        }
                    }
                }
            }
        }

        self.lost_norm = (self.lost_norm + spill_mass).min(1.0);
        self.lost_n_weight += spill_weight;
        Ok(())
    }

    /// Loss (attenuation) on one mode, by EXPLICIT pure-state dilation.
    ///
    /// A loss channel is a CP map, and this type holds a pure state, so the
    /// channel cannot be applied in place — a "lossy pure state" would be a
    /// lie about what the object is. Stinespring instead, visibly: the state
    /// grows one ancilla mode (appended LAST, prepared in vacuum), and the
    /// lossy mode is coupled to it by a beamsplitter with
    /// `θ = acos(√η)`, so a single photon survives with probability `η` —
    /// piquasso's `Attenuator(theta)` convention with transmissivity
    /// `cos²θ`, verified against its density-matrix simulator before this
    /// was written (|2⟩ at η = 0.7 → 0.49 / 0.42 / 0.09, exact).
    ///
    /// The caller keeps the ancilla — that is the point, not a leak. Reading
    /// out the lossy distribution is [`Self::marginal_probs`] over the
    /// original modes, which is exactly the partial trace; asking for joint
    /// amplitudes after loss has no meaning and therefore no API. Each loss
    /// multiplies the amplitude count by `cutoff`, so the dilation
    /// re-validates [`capacity::check`] with the caller's budget rather than
    /// assuming construction-time headroom still covers it.
    pub fn loss(
        &mut self,
        mode: usize,
        eta: f64,
        budget_bytes: Option<usize>,
    ) -> Result<(), CvError> {
        if mode >= self.n_modes {
            return Err(CvError::Unrepresentable {
                what: "loss mode index is out of range for this state",
            });
        }
        if !(0.0..=1.0).contains(&eta) {
            return Err(CvError::Unrepresentable {
                what: "loss transmissivity must be in [0, 1]",
            });
        }
        capacity::check(self.cutoff, self.n_modes + 1, budget_bytes)?;
        let dim = capacity::product_dim(self.cutoff, self.n_modes + 1).ok_or(
            CvError::Unrepresentable {
                what: "cutoff^(n_modes+1) overflows a usize",
            },
        )?;
        // Ancilla appended as the LAST (least significant, stride 1) mode in
        // vacuum: old amplitude i becomes new amplitude i·cutoff + 0.
        let mut amps = vec![Complex64::new(0.0, 0.0); dim];
        for (i, a) in self.amps.iter().enumerate() {
            amps[i * self.cutoff] = *a;
        }
        self.amps = amps;
        self.n_modes += 1;
        let ancilla = self.n_modes - 1;
        self.beamsplitter(mode, ancilla, eta.sqrt().acos(), 0.0)
    }

    /// Marginal probability distribution over `keep`, normalised by the
    /// represented mass — the partial trace a photon counter on those modes
    /// would see. The readout for [`Self::loss`], and meaningful on any
    /// entangled state.
    ///
    /// Returned sorted by occupation so the output is deterministic; entries
    /// below 1e-15 are dropped, matching the cross-check fixtures.
    pub fn marginal_probs(&self, keep: &[usize]) -> Result<Vec<(Vec<usize>, f64)>, CvError> {
        for &m in keep {
            if m >= self.n_modes {
                return Err(CvError::Unrepresentable {
                    what: "marginal mode index is out of range for this state",
                });
            }
        }
        let norm = self.norm_sqr();
        if norm <= 0.0 {
            return Err(CvError::Unrepresentable {
                what: "the state has no represented probability mass",
            });
        }
        let mut map: std::collections::BTreeMap<Vec<usize>, f64> =
            std::collections::BTreeMap::new();
        for (i, a) in self.amps.iter().enumerate() {
            let p = a.norm_sqr();
            if p == 0.0 {
                continue;
            }
            let occ = self.occupation_of(i);
            let key: Vec<usize> = keep.iter().map(|&m| occ[m]).collect();
            *map.entry(key).or_insert(0.0) += p / norm;
        }
        Ok(map.into_iter().filter(|(_, p)| *p > 1e-15).collect())
    }

    pub fn displace(&mut self, mode: usize, alpha: Complex64) -> Result<(), CvError> {
        if !alpha.re.is_finite() || !alpha.im.is_finite() {
            return Err(CvError::Unrepresentable {
                what: "displacement alpha is not finite",
            });
        }
        let c = self.cutoff;
        let dim = c + crate::PAD;
        let mut scratch = vec![Complex64::new(0.0, 0.0); dim];
        let mut spill = 0.0f64;
        let mut spill_weight = 0.0f64;
        self.apply_single_mode(mode, |fin, fout| {
            crate::displace_fiber(fin, alpha, &mut scratch);
            for (i, a) in scratch[c..].iter().enumerate() {
                let p = a.norm_sqr();
                spill += p;
                spill_weight += (c + i) as f64 * p;
            }
            fout.copy_from_slice(&scratch[..c]);
        })?;
        self.lost_norm += spill;
        self.lost_n_weight += spill_weight;
        Ok(())
    }

    /// Probability mass the cutoff has discarded so far, summed over every
    /// fiber of every operation applied.
    pub fn lost_norm(&self) -> f64 {
        self.lost_norm
    }

    /// Photon-number-weighted discarded mass, `Σ k·p_k` over the discarded tail.
    pub fn lost_n_weight(&self) -> f64 {
        self.lost_n_weight
    }

    /// Squared norm of the represented (in-cutoff) part.
    pub fn norm_sqr(&self) -> f64 {
        self.amps.iter().map(|a| a.norm_sqr()).sum()
    }

    /// `<n_mode>` — mean photon number in one mode.
    ///
    /// Normalised by the represented mass, matching `FockState::expect_n`'s
    /// convention so single- and multi-mode readouts are comparable.
    pub fn expect_n(&self, mode: usize) -> Result<f64, CvError> {
        if mode >= self.n_modes {
            return Err(CvError::Unrepresentable {
                what: "mode index is out of range for this state",
            });
        }
        let norm = self.norm_sqr();
        if norm <= 0.0 {
            return Err(CvError::Unrepresentable {
                what: "the state has no represented probability mass",
            });
        }
        let stride = self.stride(mode);
        let mut acc = 0.0;
        for (i, a) in self.amps.iter().enumerate() {
            let n = (i / stride) % self.cutoff;
            acc += (n as f64) * a.norm_sqr();
        }
        Ok(acc / norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(cutoff: usize, n: usize) -> MultiFockState {
        MultiFockState::vacuum(cutoff, n, None).expect("vacuum")
    }

    #[test]
    fn the_vacuum_is_normalised_and_all_in_the_ground_state() {
        let st = s(4, 3);
        assert_eq!(st.amplitudes().len(), 64);
        assert!((st.norm_sqr() - 1.0).abs() < 1e-15);
        for m in 0..3 {
            assert!(st.expect_n(m).unwrap().abs() < 1e-15);
        }
    }

    /// Index and occupation must be exact inverses over the whole basis, or
    /// every mode-local operation touches the wrong amplitudes.
    #[test]
    fn indexing_round_trips_over_the_entire_basis() {
        let st = s(3, 4);
        for i in 0..st.amplitudes().len() {
            let occ = st.occupation_of(i);
            assert_eq!(occ.len(), 4);
            assert_eq!(st.index_of(&occ), Some(i), "index_of(occupation_of({i}))");
        }
        assert_eq!(st.index_of(&[3, 0, 0, 0]), None, "3 is at the cutoff");
        assert_eq!(st.index_of(&[0, 0, 0]), None, "wrong arity");
    }

    /// Mode 0 is most significant, which fixes the stride convention. Asserted
    /// on a concrete case so a silent convention flip fails here rather than
    /// showing up as a wrong expectation three layers away.
    #[test]
    fn mode_zero_is_most_significant() {
        let st = s(5, 3);
        assert_eq!(st.index_of(&[1, 0, 0]), Some(25));
        assert_eq!(st.index_of(&[0, 1, 0]), Some(5));
        assert_eq!(st.index_of(&[0, 0, 1]), Some(1));
    }

    /// **The discriminating test.** A single-mode operator on mode `m` must
    /// change mode `m` and leave every other mode untouched.
    ///
    /// A product state cannot detect a stride error on its own — an operator
    /// applied to the wrong axis of a symmetric product state gives the same
    /// answer. So the modes are given DIFFERENT states first, and then the
    /// operator moves one of them.
    #[test]
    fn a_single_mode_operator_touches_only_its_own_mode() {
        let mut st = s(4, 3);
        // Put distinct, asymmetric content in each mode: mode0 -> |1>,
        // mode1 -> |2>, mode2 -> |0>. Built by raising each mode separately,
        // which also exercises `apply_single_mode` itself.
        let raise = |k: usize| {
            move |fin: &[Complex64], fout: &mut [Complex64]| {
                for i in 0..fin.len() {
                    if i >= k {
                        fout[i] = fin[i - k];
                    }
                }
            }
        };
        st.apply_single_mode(0, raise(1)).unwrap();
        st.apply_single_mode(1, raise(2)).unwrap();
        assert!((st.expect_n(0).unwrap() - 1.0).abs() < 1e-12);
        assert!((st.expect_n(1).unwrap() - 2.0).abs() < 1e-12);
        assert!(st.expect_n(2).unwrap().abs() < 1e-12);

        // Now move ONLY mode 2 and check the others are unchanged.
        let before = (st.expect_n(0).unwrap(), st.expect_n(1).unwrap());
        st.apply_single_mode(2, raise(1)).unwrap();
        assert!(
            (st.expect_n(2).unwrap() - 1.0).abs() < 1e-12,
            "mode 2 moved"
        );
        assert!(
            (st.expect_n(0).unwrap() - before.0).abs() < 1e-15
                && (st.expect_n(1).unwrap() - before.1).abs() < 1e-15,
            "modes 0 and 1 must be untouched — a stride error shows up here"
        );
    }

    /// Every fiber must be visited exactly once. A stride bug that skips or
    /// double-counts fibers is invisible to a norm check, so count the calls.
    #[test]
    fn every_fiber_is_visited_exactly_once() {
        let (cutoff, n) = (3usize, 4usize);
        let mut st = s(cutoff, n);
        for mode in 0..n {
            let mut calls = 0usize;
            st.apply_single_mode(mode, |fin, fout| {
                calls += 1;
                fout.copy_from_slice(fin);
            })
            .unwrap();
            assert_eq!(
                calls,
                cutoff.pow((n - 1) as u32),
                "mode {mode}: expected cutoff^(n-1) fibers"
            );
        }
    }

    /// The identity through the primitive must be the identity — this catches a
    /// gather/scatter mismatch that a norm-preserving permutation would hide.
    #[test]
    fn the_identity_operator_is_the_identity() {
        let mut st = s(4, 3);
        st.apply_single_mode(0, |fin, fout| {
            for i in 0..fin.len() {
                fout[i] = fin[i] * Complex64::new(0.5, 0.25);
            }
        })
        .unwrap();
        let snapshot = st.amplitudes().to_vec();
        for mode in 0..3 {
            st.apply_single_mode(mode, |fin, fout| fout.copy_from_slice(fin))
                .unwrap();
        }
        for (a, b) in st.amplitudes().iter().zip(&snapshot) {
            assert!((a - b).norm() < 1e-15, "identity changed an amplitude");
        }
    }

    /// A shape the guard refuses never reaches an allocation.
    #[test]
    fn an_over_budget_shape_is_refused_before_allocating() {
        assert!(MultiFockState::vacuum(8, 6, Some(1024)).is_err());
        assert!(MultiFockState::vacuum(64, 6, None).is_err(), "hard ceiling");
        assert!(MultiFockState::vacuum(0, 3, None).is_err(), "zero cutoff");
        assert!(MultiFockState::vacuum(4, 0, None).is_err(), "zero modes");
    }

    /// An out-of-range mode is an error, not a panic or a silent no-op.
    #[test]
    fn an_out_of_range_mode_is_refused() {
        let mut st = s(4, 2);
        assert!(st.apply_single_mode(2, |_, _| {}).is_err());
        assert!(st.expect_n(2).is_err());
    }
}

#[cfg(test)]
mod displace_tests {
    use super::*;
    use crate::FockState;

    /// **The analytic anchor, and mode-locality in the same assertion.**
    ///
    /// A displaced vacuum has `<n> = |alpha|^2`. Displacing mode 1 of a 3-mode
    /// vacuum must put exactly that in mode 1 and leave modes 0 and 2 at zero.
    /// One test, two independent ways to fail: a wrong operator misses the
    /// analytic value, a wrong stride puts it in the wrong mode.
    #[test]
    fn a_displaced_mode_has_the_analytic_photon_number_and_the_others_are_empty() {
        let alpha = Complex64::new(0.4, -0.3);
        let mut st = MultiFockState::vacuum(12, 3, None).expect("vacuum");
        st.displace(1, alpha).expect("displace");
        let want = alpha.norm_sqr();
        assert!(
            (st.expect_n(1).unwrap() - want).abs() < 1e-9,
            "<n_1> should be |alpha|^2 = {want}, got {}",
            st.expect_n(1).unwrap()
        );
        assert!(
            st.expect_n(0).unwrap().abs() < 1e-12,
            "mode 0 must be empty"
        );
        assert!(
            st.expect_n(2).unwrap().abs() < 1e-12,
            "mode 2 must be empty"
        );
    }

    /// **The multi-mode path and the single-mode path must agree EXACTLY.**
    ///
    /// They share `displace_fiber`, so on a one-mode state the amplitudes should
    /// be bit-identical — not merely close. Anything else means the shared
    /// implementation is not actually shared, which is the whole reason it was
    /// extracted.
    #[test]
    fn one_mode_multimode_is_bit_identical_to_the_single_mode_path() {
        let alpha = Complex64::new(-0.7, 0.25);
        let cutoff = 16;

        let mut single = FockState::vacuum(cutoff).expect("single vacuum");
        single.displace(alpha).expect("single displace");

        let mut multi = MultiFockState::vacuum(cutoff, 1, None).expect("multi vacuum");
        multi.displace(0, alpha).expect("multi displace");

        assert_eq!(multi.amplitudes().len(), single.amplitudes().len());
        for (i, (a, b)) in multi
            .amplitudes()
            .iter()
            .zip(single.amplitudes())
            .enumerate()
        {
            assert_eq!(
                (a.re.to_bits(), a.im.to_bits()),
                (b.re.to_bits(), b.im.to_bits()),
                "amplitude {i} differs — the two paths are not sharing displace_fiber"
            );
        }
    }

    /// Truncation loss is summed across fibers, not reported per fiber.
    ///
    /// A displacement large enough to spill, applied to a 2-mode state, must
    /// report loss from every fiber it touched. `cutoff^(n-1)` fibers each spill,
    /// so a per-fiber figure would under-report by that factor.
    #[test]
    fn truncation_loss_accumulates_over_every_fiber() {
        let big = Complex64::new(2.5, 0.0); // spills hard at cutoff 6
        let mut one = MultiFockState::vacuum(6, 1, None).unwrap();
        one.displace(0, big).unwrap();
        let mut two = MultiFockState::vacuum(6, 2, None).unwrap();
        two.displace(0, big).unwrap();

        assert!(one.lost_norm() > 0.0, "this displacement must spill");
        assert!(
            two.lost_norm() >= one.lost_norm(),
            "a 2-mode state has {} fibers, so its total loss cannot be smaller \
             than the 1-mode case: got {} vs {}",
            6,
            two.lost_norm(),
            one.lost_norm()
        );
        assert!(
            two.lost_n_weight() > 0.0,
            "weighted loss must be recorded too"
        );
    }

    /// Displacing twice by ±alpha returns to the vacuum, which no wrong stride
    /// or wrong sign convention reproduces.
    #[test]
    fn displacing_by_alpha_then_minus_alpha_returns_to_vacuum() {
        let alpha = Complex64::new(0.3, 0.2);
        let mut st = MultiFockState::vacuum(24, 2, None).unwrap();
        st.displace(1, alpha).unwrap();
        st.displace(1, -alpha).unwrap();
        assert!(
            st.expect_n(1).unwrap().abs() < 1e-9,
            "D(-a)D(a) = I up to truncation, got <n_1> = {}",
            st.expect_n(1).unwrap()
        );
    }

    /// A non-finite alpha is refused rather than producing NaN amplitudes.
    #[test]
    fn a_non_finite_alpha_is_refused() {
        let mut st = MultiFockState::vacuum(8, 2, None).unwrap();
        assert!(st.displace(0, Complex64::new(f64::NAN, 0.0)).is_err());
        assert!(st.displace(0, Complex64::new(0.0, f64::INFINITY)).is_err());
    }

    /// `S₂(r)|0,0⟩ = sech r · Σ (e^{iφ} tanh r)^j |j,j⟩` — the textbook
    /// two-mode squeezed vacuum, asserted analytically per level.
    #[test]
    fn two_mode_squeezed_vacuum_matches_the_analytic_series() {
        let (r, phi) = (0.4f64, 0.3f64);
        let mut st = MultiFockState::vacuum(10, 2, None).unwrap();
        st.squeezing2(0, 1, r, phi).unwrap();
        let sech = 1.0 / r.cosh();
        let tanh = r.tanh();
        for j in 0..6usize {
            let idx = st.index_of(&[j, j]).unwrap();
            let want = Complex64::from_polar(sech * tanh.powi(j as i32), phi * j as f64);
            let got = st.amplitudes()[idx];
            assert!(
                (got - want).norm() < 1e-12,
                "level |{j},{j}>: got {got}, analytic {want}"
            );
        }
        // Off-chain amplitudes stay exactly zero: photons come in pairs.
        assert_eq!(
            st.amplitudes()[st.index_of(&[1, 0]).unwrap()].norm_sqr(),
            0.0
        );
        assert_eq!(
            st.amplitudes()[st.index_of(&[2, 1]).unwrap()].norm_sqr(),
            0.0
        );
    }

    /// The operator is symmetric under a↔b, so the ARGUMENT order must not
    /// matter — a wrong chain orientation would break exactly this.
    #[test]
    fn squeezing2_is_symmetric_in_its_mode_arguments() {
        let mut ab = MultiFockState::number_state(8, 2, &[2, 0], None).unwrap();
        let mut ba = ab.clone();
        ab.squeezing2(0, 1, 0.35, 0.6).unwrap();
        ba.squeezing2(1, 0, 0.35, 0.6).unwrap();
        for (x, y) in ab.amplitudes().iter().zip(ba.amplitudes()) {
            assert!((x - y).norm() < 1e-14, "S2(a,b) != S2(b,a): {x} vs {y}");
        }
    }

    /// A spectator mode must hold its photons; only a >2-mode case can see a
    /// wrong fiber walk (same reasoning as the beamsplitter corpus).
    #[test]
    fn squeezing2_leaves_a_spectator_mode_alone() {
        let mut st = MultiFockState::number_state(6, 3, &[0, 0, 2], None).unwrap();
        st.squeezing2(0, 1, 0.3, 0.0).unwrap();
        let n2 = st.expect_n(2).unwrap();
        assert!(
            (n2 - 2.0).abs() < 1e-12,
            "the spectator's photon number moved: {n2}"
        );
    }

    /// `S₂(−r)` undoes `S₂(r)` (φ = 0), up to the mass PAD could not hold.
    #[test]
    fn squeezing2_inverts_with_negated_r() {
        let mut st = MultiFockState::number_state(12, 2, &[1, 0], None).unwrap();
        st.squeezing2(0, 1, 0.3, 0.0).unwrap();
        st.squeezing2(0, 1, -0.3, 0.0).unwrap();
        let idx = st.index_of(&[1, 0]).unwrap();
        assert!(
            (st.amplitudes()[idx].re - 1.0).abs() < 1e-9,
            "S2(-r)S2(r)|1,0> should be |1,0> again, got {}",
            st.amplitudes()[idx]
        );
    }

    /// Truncation is REPORTED: squeezing hard at a small cutoff must leave a
    /// visible `lost_norm`, not a silently sub-normalised state.
    #[test]
    fn squeezing2_reports_its_spill() {
        let mut st = MultiFockState::vacuum(3, 2, None).unwrap();
        st.squeezing2(0, 1, 1.2, 0.0).unwrap();
        assert!(
            st.lost_norm() > 1e-3,
            "r=1.2 at cutoff 3 must spill measurably, got {}",
            st.lost_norm()
        );
        assert!(st.lost_n_weight() > st.lost_norm(), "spill sits at high n");
    }

    /// `|1⟩` through `η = 0.7` is a Bernoulli coin: the photon survives with
    /// probability exactly η. The dilation adds a mode; the ORIGINAL mode
    /// indices keep working, and the marginal over them is the channel output.
    #[test]
    fn loss_on_a_single_photon_is_a_bernoulli_in_eta() {
        let mut st = MultiFockState::number_state(4, 2, &[1, 0], None).unwrap();
        st.loss(0, 0.7, None).unwrap();
        assert_eq!(st.n_modes(), 3, "the dilation is explicit: one new mode");
        let m = st.marginal_probs(&[0, 1]).unwrap();
        assert_eq!(m.len(), 2);
        let p = |occ: &[usize]| {
            m.iter()
                .find(|(k, _)| k == occ)
                .map(|(_, p)| *p)
                .unwrap_or(0.0)
        };
        assert!((p(&[1, 0]) - 0.7).abs() < 1e-14, "survival = eta");
        assert!((p(&[0, 0]) - 0.3).abs() < 1e-14, "loss = 1 - eta");
        assert!(
            (st.expect_n(0).unwrap() - 0.7).abs() < 1e-14,
            "<n> scales by eta"
        );
    }

    /// η = 1 must be the identity on the kept modes; η = 0 must empty them.
    #[test]
    fn loss_edge_transmissivities_are_identity_and_annihilation() {
        let mut keep = MultiFockState::number_state(5, 2, &[2, 1], None).unwrap();
        keep.loss(0, 1.0, None).unwrap();
        assert!((keep.expect_n(0).unwrap() - 2.0).abs() < 1e-12);

        let mut drop = MultiFockState::number_state(5, 2, &[2, 1], None).unwrap();
        drop.loss(0, 0.0, None).unwrap();
        assert!(drop.expect_n(0).unwrap().abs() < 1e-12);
        // The spectator mode is untouched either way.
        assert!((drop.expect_n(1).unwrap() - 1.0).abs() < 1e-12);
    }

    /// The dilation multiplies the amplitude count by `cutoff`, so it must
    /// re-validate the budget rather than trust construction-time headroom.
    #[test]
    fn loss_respects_the_caller_budget() {
        let budget = Some(6 * 6 * 16 + 64); // fits 2 modes at cutoff 6, not 3
        let mut st = MultiFockState::number_state(6, 2, &[1, 0], budget).unwrap();
        assert!(
            st.loss(0, 0.5, budget).is_err(),
            "dilated shape must refuse"
        );
        assert_eq!(st.n_modes(), 2, "a refused loss must not mutate the state");
    }

    #[test]
    fn loss_refuses_bad_arguments() {
        let mut st = MultiFockState::vacuum(4, 2, None).unwrap();
        assert!(st.loss(2, 0.5, None).is_err(), "out of range");
        assert!(st.loss(0, 1.5, None).is_err(), "eta > 1");
        assert!(st.loss(0, -0.1, None).is_err(), "eta < 0");
        assert!(st.loss(0, f64::NAN, None).is_err(), "NaN eta");
    }

    /// The marginal walks the right strides: distinct occupations per mode,
    /// then every single-mode marginal must name its own mode's occupation.
    #[test]
    fn marginal_probs_reads_each_mode_not_a_permutation_of_them() {
        let st = MultiFockState::number_state(4, 3, &[1, 2, 3], None).unwrap();
        for (mode, want) in [(0usize, 1usize), (1, 2), (2, 3)] {
            let m = st.marginal_probs(&[mode]).unwrap();
            assert_eq!(m, vec![(vec![want], 1.0)], "mode {mode}");
        }
        // A two-mode marginal in REVERSED order follows the request order.
        let m = st.marginal_probs(&[2, 0]).unwrap();
        assert_eq!(m, vec![(vec![3, 1], 1.0)]);
    }

    #[test]
    fn squeezing2_refuses_bad_arguments() {
        let mut st = MultiFockState::vacuum(4, 2, None).unwrap();
        assert!(st.squeezing2(0, 0, 0.1, 0.0).is_err(), "same mode");
        assert!(st.squeezing2(0, 2, 0.1, 0.0).is_err(), "out of range");
        assert!(st.squeezing2(0, 1, f64::NAN, 0.0).is_err(), "NaN r");
        assert!(st.squeezing2(0, 1, 0.1, f64::INFINITY).is_err(), "inf phi");
    }
}

/// `exp[θ(e^{iφ} a†b − e^{−iφ} a b†)]` restricted to the total-photon-number
/// block `N`, as a row-major `(N+1)×(N+1)` matrix indexed by `k` = photons in
/// the FIRST mode, i.e. basis `|k, N−k⟩`.
///
/// The generator is **tridiagonal**: `a†b` moves exactly one photon from the
/// second mode to the first, so it connects `k` only to `k±1`. It is also
/// anti-Hermitian, so the exponential is unitary on the block — and unlike the
/// truncated single-mode operators, that unitarity is exact rather than
/// approximate, because the block IS the whole operator.
///
/// Matrix elements, from `a†b|k, N−k⟩ = √(k+1)·√(N−k)·|k+1, N−k−1⟩`.
fn beamsplitter_block(n: usize, theta: f64, phi: f64) -> Vec<Complex64> {
    let dim = n + 1;
    let mut g = vec![Complex64::new(0.0, 0.0); dim * dim];
    let e_ip = Complex64::from_polar(1.0, phi);
    let e_mip = Complex64::from_polar(1.0, -phi);
    for k in 0..n {
        let amp = ((k + 1) as f64).sqrt() * ((n - k) as f64).sqrt() * theta;
        g[(k + 1) * dim + k] += e_ip * amp;
        g[k * dim + (k + 1)] -= e_mip * amp;
    }
    expm_small(&g, dim)
}

/// `exp(M)` for a small dense complex matrix, by scaling-and-squaring with a
/// Taylor series.
///
/// Small means `dim ≤ 2·cutoff − 1`, so an O(dim³) squaring is nothing and a
/// general eigensolver would be a dependency bought for no benefit.
///
/// Scaling first is what makes the Taylor series trustworthy: the series
/// converges for any argument but LOSES ACCURACY when the norm is large,
/// because it must cancel big alternating terms. Halving until the norm is
/// below 1/2 keeps every term small and the cancellation mild; the squarings
/// then rebuild the full argument exactly.
fn expm_small(m: &[Complex64], dim: usize) -> Vec<Complex64> {
    // Max absolute row sum — an upper bound on the spectral norm, and cheap.
    let norm = (0..dim)
        .map(|r| (0..dim).map(|c| m[r * dim + c].norm()).sum::<f64>())
        .fold(0.0f64, f64::max);
    let squarings = if norm > 0.5 {
        (norm / 0.5).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    let scale = 1.0 / (2f64.powi(squarings as i32));

    let mut term: Vec<Complex64> = (0..dim * dim)
        .map(|i| {
            if i % dim == i / dim {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();
    let mut acc = term.clone();
    let scaled: Vec<Complex64> = m.iter().map(|v| v * scale).collect();

    // 24 terms is far past convergence for ‖M‖ ≤ 1/2 — the tail is bounded by
    // (1/2)^24/24!, which is below f64 epsilon by a wide margin. Fixed rather
    // than adaptive so the cost does not depend on the input.
    for k in 1..=24u32 {
        term = matmul(&term, &scaled, dim);
        for v in term.iter_mut() {
            *v /= k as f64;
        }
        for (a, t) in acc.iter_mut().zip(term.iter()) {
            *a += t;
        }
    }
    for _ in 0..squarings {
        acc = matmul(&acc, &acc, dim);
    }
    acc
}

fn matmul(a: &[Complex64], b: &[Complex64], dim: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for k in 0..dim {
            let aik = a[i * dim + k];
            if aik == Complex64::new(0.0, 0.0) {
                continue;
            }
            for j in 0..dim {
                out[i * dim + j] += aik * b[k * dim + j];
            }
        }
    }
    out
}
