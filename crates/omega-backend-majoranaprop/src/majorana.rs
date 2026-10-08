// SPDX-License-Identifier: Apache-2.0
//! The Majorana monomial algebra.
//!
//! Conventions (fixed here, verified against dense matrices in
//! `tests/majorana_oracle.rs`, and relied on by everything above):
//!
//! * `n` fermionic modes ↔ `n` qubits under Jordan–Wigner. Mode `j` has two
//!   Majoranas, `γ_{2j} = Z_0…Z_{j−1} X_j` and `γ_{2j+1} = Z_0…Z_{j−1} Y_j`,
//!   so there are `2n` Majorana indices `k ∈ [0, 2n)`.
//! * A **monomial** is labelled by a bitset `b ⊆ [0, 2n)` and defined as
//!   `M_b = i^{r_b} · γ_{k_1} γ_{k_2} ⋯ γ_{k_m}` with `k_1 < k_2 < ⋯ < k_m`
//!   the set bits in ascending order and `r_b = m(m−1)/2 mod 4`. That phase
//!   makes every `M_b` **Hermitian with `M_b² = 1`**, which is what turns the
//!   dropped L1 mass into a bound: `|⟨M_b⟩| ≤ 1` in every state.
//! * The **length** of `M_b` is `m = |b|`. Truncation happens on it.
//!
//! Products: for two labels `c, b`, `M_c · M_b = i^e · M_{c⊕b}` with
//! `e = r_c + r_b − r_{c⊕b} + 2·t(c,b)  (mod 4)` where `t(c,b)` counts the
//! pairs `(i ∈ c, j ∈ b)` with `i > j` — the anticommutations needed to sort
//! the concatenated product back into ascending order (equal indices meet
//! and square to `1` without a sign). Two monomials commute iff
//! `|c|·|b| − |c ∩ b|` is even.
//!
//! Rotation rule (the whole engine): with `U = exp(−iφ M_c / 2)`,
//! `U† M_b U = M_b` if they commute, else `cos φ · M_b + sin φ · (i M_c M_b)`;
//! `i M_c M_b` is itself `±M_{c⊕b}` because anticommuting monomials have odd
//! `e`. So a rotation splits a term into at most two, both of length
//! `≤ |b| + |c| − 2|c ∩ b|`, and for a Givens generator (`|c| = 2`, sharing
//! one index with `b`) the length is **unchanged** — the point of the basis.
//!
//! Fock readout: `⟨n|M_b|n⟩` is nonzero only when `b` pairs every mode it
//! touches (bits `2j` and `2j+1` both set or both clear); then it equals
//! `∏_{paired j} (2 n_j − 1)`, i.e. `−1` per empty paired mode and `+1` per
//! occupied one. Unpaired monomials change particle number and read `0`.

use smallvec::{smallvec, SmallVec};

/// Label of a Hermitian Majorana monomial on `n` modes: a `2n`-bit set.
///
/// Bits above `2n` in the final word are always zero (every constructor
/// masks); two keys that differed only in padding would hash apart and split
/// one operator into two entries of a sum.
///
/// `Ord` is lexicographic on the words, then `n_modes` — no physical meaning,
/// it only gives [`MajoranaSum::l1_norm`](crate::engine::MajoranaSum::l1_norm)
/// a fixed summation order so the certificate's `observable_range` is the
/// same bits on every run.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct MajoranaKey {
    /// `2n` bits, little-endian across words. Inline for `n ≤ 64`.
    words: SmallVec<[u64; 2]>,
    n_modes: u32,
}

#[inline]
fn words_for(n_modes: usize) -> usize {
    (2 * n_modes).div_ceil(64)
}

#[inline]
fn tail_mask(n_modes: usize) -> u64 {
    let r = (2 * n_modes) % 64;
    if r == 0 {
        !0
    } else {
        (1u64 << r) - 1
    }
}

/// `m(m−1)/2 mod 4` — the Hermitian-making phase exponent for length `m`.
#[inline]
fn r_of(m: u32) -> u32 {
    ((m as u64 * (m as u64).wrapping_sub(1)) / 2 % 4) as u32
}

impl MajoranaKey {
    /// The identity monomial (empty set) on `n` modes.
    pub fn identity(n_modes: usize) -> Self {
        Self {
            words: smallvec![0u64; words_for(n_modes)],
            n_modes: n_modes as u32,
        }
    }

    /// Monomial on the given Majorana indices (any order, duplicates refused).
    ///
    /// # Panics
    /// If an index is `≥ 2n` or repeated — both are caller bugs, not data.
    pub fn from_indices(n_modes: usize, idx: &[usize]) -> Self {
        let mut k = Self::identity(n_modes);
        for &i in idx {
            assert!(
                i < 2 * n_modes,
                "Majorana index {i} out of range for {n_modes} modes"
            );
            assert!(!k.bit(i), "Majorana index {i} repeated");
            k.words[i / 64] |= 1u64 << (i % 64);
        }
        k
    }

    /// Build from packed words (the device / kernel path). Masks the tail.
    pub fn from_words(words: &[u64], n_modes: usize) -> Self {
        let w = words_for(n_modes);
        assert_eq!(words.len(), w);
        let mut v: SmallVec<[u64; 2]> = SmallVec::from_slice(words);
        if w > 0 {
            v[w - 1] &= tail_mask(n_modes);
        }
        Self {
            words: v,
            n_modes: n_modes as u32,
        }
    }

    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    #[inline]
    pub fn n_modes(&self) -> usize {
        self.n_modes as usize
    }

    #[inline]
    pub fn bit(&self, k: usize) -> bool {
        (self.words[k / 64] >> (k % 64)) & 1 == 1
    }

    /// Number of Majoranas in the monomial — what truncation cuts on.
    #[inline]
    pub fn length(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    #[inline]
    pub fn is_identity(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Set Majorana indices, ascending.
    pub fn indices(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.length() as usize);
        for (wi, &w) in self.words.iter().enumerate() {
            let mut x = w;
            while x != 0 {
                let t = x.trailing_zeros() as usize;
                out.push(wi * 64 + t);
                x &= x - 1;
            }
        }
        out
    }

    /// Whether `M_self` and `M_other` commute: iff `|c|·|b| − |c∩b|` is even.
    #[inline]
    pub fn commutes(&self, other: &Self) -> bool {
        debug_assert_eq!(self.n_modes, other.n_modes);
        let lc = self.length() as u64;
        let lb = other.length() as u64;
        let shared: u64 = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(&a, &b)| (a & b).count_ones() as u64)
            .sum();
        (lc * lb - shared).is_multiple_of(2)
    }

    /// Parity of `t(c,b) = #{(i ∈ c, j ∈ b) : i > j}`.
    fn inversions_parity(&self, other: &Self) -> u32 {
        // For each set bit j of `other`, count bits of `self` strictly above j.
        let w = self.words.len();
        // suffix[k] = popcount of self's words with index >= k
        let mut above_words = 0u64;
        let mut suffix: SmallVec<[u64; 2]> = smallvec![0u64; w + 1];
        for k in (0..w).rev() {
            above_words += self.words[k].count_ones() as u64;
            suffix[k] = above_words;
        }
        let mut t = 0u64;
        for (wi, &bw) in other.words.iter().enumerate() {
            let mut x = bw;
            while x != 0 {
                let j = x.trailing_zeros();
                // bits of self in words strictly above wi
                t += suffix[wi + 1];
                // bits of self in this word strictly above j
                let above_in_word = if j == 63 {
                    0
                } else {
                    self.words[wi] >> (j + 1)
                };
                t += above_in_word.count_ones() as u64;
                x &= x - 1;
            }
        }
        (t & 1) as u32
    }

    /// `M_self · M_other = i^e · M_{self ⊕ other}`; returns `(key, e mod 4)`.
    pub fn mul(&self, other: &Self) -> (Self, u32) {
        debug_assert_eq!(self.n_modes, other.n_modes);
        let mut words: SmallVec<[u64; 2]> = self.words.clone();
        for (a, &b) in words.iter_mut().zip(other.words.iter()) {
            *a ^= b;
        }
        let out = Self {
            words,
            n_modes: self.n_modes,
        };
        let rc = r_of(self.length());
        let rb = r_of(other.length());
        let rcb = r_of(out.length());
        let t = self.inversions_parity(other);
        let e = (rc + rb + 4 - rcb + 2 * t) % 4;
        (out, e)
    }

    /// The Heisenberg image of `M_self` under `U = exp(−iφ M_c/2)` when the
    /// two anticommute: `U† M_b U = cos φ · M_b + sin φ · s · M_{c⊕b}` with
    /// `s = ±1`. Returns `(c ⊕ b, s)`. Caller checks `!commutes` first; for
    /// commuting pairs the image is `M_b` itself.
    ///
    /// `s` is real because `i M_c M_b` is Hermitian for anticommuting
    /// monomials (`e` is odd, so `i · i^e = ±1`).
    pub fn rotate_by(&self, c: &Self) -> (Self, f64) {
        let (k, e) = c.mul(self);
        // i * i^e: e odd => (e+1) mod 4 ∈ {0, 2} => +1 or −1.
        debug_assert!(e % 2 == 1, "rotate_by on commuting monomials");
        let s = if (e + 1) % 4 == 0 { 1.0 } else { -1.0 };
        (k, s)
    }

    /// `⟨n|M_b|n⟩` for the Fock state with occupation bits `occ` (bit `j` of
    /// `occ[j/64]` is `n_j`). Zero unless every touched mode is paired;
    /// otherwise `∏_{paired j} (2 n_j − 1)`.
    pub fn fock_expectation(&self, occ: &[u64]) -> f64 {
        const EVEN: u64 = 0x5555_5555_5555_5555;
        const ODD: u64 = 0xAAAA_AAAA_AAAA_AAAA;
        let mut sign = 1.0f64;
        for (wi, &w) in self.words.iter().enumerate() {
            if ((w & EVEN) << 1) != (w & ODD) {
                return 0.0;
            }
            let mut x = w & EVEN;
            while x != 0 {
                let k = x.trailing_zeros() as usize; // = 2j within the word
                let j = wi * 32 + k / 2;
                let nj = (occ.get(j / 64).copied().unwrap_or(0) >> (j % 64)) & 1;
                if nj == 0 {
                    sign = -sign;
                }
                x &= x - 1;
            }
        }
        sign
    }

    /// `⟨0|M_b|0⟩` — the vacuum, i.e. every paired mode empty.
    pub fn vacuum_expectation(&self) -> f64 {
        self.fock_expectation(&[])
    }

    /// The Jordan–Wigner image of a Pauli string as `P = i^e · M_b`; returns
    /// `(b, e mod 4)`. `x[j]`/`z[j]` are the symplectic bits of qubit `j`
    /// (`Y` = both). The phase is `±1` whenever `P` is Hermitian, which every
    /// Pauli string is, so `e ∈ {0, 2}` always — asserted.
    ///
    /// Derivation, site by site in ascending order with `Z_k = −i γ_{2k}γ_{2k+1}`:
    /// `X_j = (−i)^j (∏_{k<j} γ_{2k}γ_{2k+1}) γ_{2j}`, `Y_j` likewise with
    /// `γ_{2j+1}`, `Z_j = −i γ_{2j} γ_{2j+1}` — each an ascending product with
    /// a known phase, folded in with [`Self::mul`].
    pub fn from_pauli(x: &[bool], z: &[bool]) -> (Self, u32) {
        let (key, e, _) = Self::from_pauli_counted(x, z);
        (key, e)
    }

    /// [`from_pauli`](Self::from_pauli) plus the number of monomial products
    /// it performed — one per non-identity site — so a caller can count what
    /// re-condensing a Pauli string into the Majorana basis costs. Observed,
    /// not computed: `tests/fermionic_seed.rs` (iii) compares this against
    /// the direct fermionic seed, which never builds the string at all.
    pub fn from_pauli_counted(x: &[bool], z: &[bool]) -> (Self, u32, usize) {
        assert_eq!(x.len(), z.len());
        let n = x.len();
        let mut acc = Self::identity(n);
        let mut e = 0u32; // accumulated exponent of i
        let mut steps = 0usize;
        for j in 0..n {
            let (bits, ph): (Vec<usize>, u32) = match (x[j], z[j]) {
                (false, false) => continue,
                (true, false) => (
                    (0..2 * j).chain([2 * j]).collect(),
                    (4 - (j as u32 % 4)) % 4,
                ),
                (true, true) => (
                    (0..2 * j).chain([2 * j + 1]).collect(),
                    (4 - (j as u32 % 4)) % 4,
                ),
                (false, true) => (vec![2 * j, 2 * j + 1], 3),
            };
            // Ascending product ∏γ = i^{−r} M_S.
            let s = Self::from_indices(n, &bits);
            let r = r_of(s.length());
            let factor_e = (ph + 4 - r) % 4;
            let (next, me) = acc.mul(&s);
            acc = next;
            e = (e + factor_e + me) % 4;
            steps += 1;
        }
        assert!(
            e.is_multiple_of(2),
            "Pauli string mapped to a non-Hermitian phase"
        );
        (acc, e, steps)
    }
}
