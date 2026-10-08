// SPDX-License-Identifier: Apache-2.0
//! The fixed-Hamming-weight basis: every `n`-bit mask with exactly `k` set
//! bits, in increasing numeric order, with O(k) ranking through the
//! combinatorial number system.
//!
//! Increasing numeric order of masks is colexicographic order of the
//! k-subsets, whose rank has the closed form
//! `Σ_j C(pos_j, j + 1)` over the set-bit positions `pos_0 < pos_1 < …`.
//! That is what lets a gate find the partner of a basis state (`mask ^ bits`)
//! without a hash table, and it is pinned by `rank_inverts_enumeration`.

/// One `(n, k)` sector.
#[derive(Clone, Debug)]
pub struct Sector {
    n: u32,
    k: u32,
    /// Basis masks, index = rank.
    states: Vec<u64>,
    /// `binom[i][j] = C(i, j)`, `i ≤ 64`, `j ≤ k`.
    binom: Vec<Vec<u64>>,
}

/// `C(n, k)` exactly, or `None` past `u128`. `n` and `k` are unbounded here on
/// purpose: the caller decides what is too large, this only refuses to lie.
pub fn binomial(n: u32, k: u32) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        // acc * (n - i) / (i + 1) is exact at every step.
        acc = acc.checked_mul((n - i) as u128)? / (i as u128 + 1);
    }
    Some(acc)
}

impl Sector {
    /// Enumerate the sector. `n ≤ 64`, `k ≤ n`, and the caller has already
    /// decided `C(n, k)` fits in memory.
    pub fn new(n: u32, k: u32) -> Self {
        assert!(n <= 64, "sector basis is u64-keyed; n = {n}");
        assert!(k <= n, "k = {k} > n = {n}");
        let mut binom = vec![vec![0u64; k as usize + 2]; 65];
        for i in 0..=64usize {
            binom[i][0] = 1;
            for j in 1..=(k as usize + 1) {
                binom[i][j] = if i == 0 {
                    0
                } else {
                    binom[i - 1][j - 1] + binom[i - 1][j]
                };
            }
        }
        let dim = binomial(n, k).expect("caller bounded the sector") as usize;
        let mut states = Vec::with_capacity(dim);
        if k == 0 {
            states.push(0);
        } else {
            // Colex successor on the sorted position vector: bump the lowest
            // position that has room, reset everything below it to 0..j.
            let mut pos: Vec<u32> = (0..k).collect();
            loop {
                states.push(pos.iter().fold(0u64, |m, &p| m | (1u64 << p)));
                let mut j = 0usize;
                loop {
                    if j == pos.len() {
                        break;
                    }
                    let limit = if j + 1 == pos.len() { n } else { pos[j + 1] };
                    if pos[j] + 1 < limit {
                        pos[j] += 1;
                        for (r, p) in pos.iter_mut().enumerate().take(j) {
                            *p = r as u32;
                        }
                        break;
                    }
                    j += 1;
                }
                if j == pos.len() {
                    break;
                }
            }
        }
        debug_assert_eq!(states.len(), dim);
        Self {
            n,
            k,
            states,
            binom,
        }
    }

    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn k(&self) -> u32 {
        self.k
    }
    pub fn dim(&self) -> usize {
        self.states.len()
    }
    pub fn states(&self) -> &[u64] {
        &self.states
    }
    pub fn mask(&self, rank: usize) -> u64 {
        self.states[rank]
    }

    /// Rank of a mask that IS in this sector (`popcount == k`). Not checked
    /// in release builds: every caller has just built the mask by flipping an
    /// equal number of bits on and off.
    pub fn rank(&self, mask: u64) -> usize {
        debug_assert_eq!(
            mask.count_ones(),
            self.k,
            "mask {mask:#b} is not in the sector"
        );
        let mut r = 0u64;
        let mut m = mask;
        let mut j = 1usize;
        while m != 0 {
            let p = m.trailing_zeros() as usize;
            r += self.binom[p][j];
            m &= m - 1;
            j += 1;
        }
        r as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binomial_is_exact() {
        assert_eq!(binomial(0, 0), Some(1));
        assert_eq!(binomial(5, 7), Some(0));
        assert_eq!(binomial(32, 16), Some(601_080_390));
        assert_eq!(binomial(64, 32), Some(1_832_624_140_942_590_534));
        assert_eq!(binomial(64, 4), Some(635_376));
        // The §3c.0b table rows are spinful: C(N, Nα) · C(N, Nβ).
        assert_eq!(binomial(16, 8).unwrap().pow(2), 165_636_900);
        assert_eq!(binomial(32, 4).unwrap().pow(2), 1_293_121_600);
    }

    #[test]
    fn rank_inverts_enumeration() {
        for n in 0..=10u32 {
            for k in 0..=n {
                let s = Sector::new(n, k);
                assert_eq!(s.dim() as u128, binomial(n, k).unwrap(), "n={n} k={k}");
                let mut prev: Option<u64> = None;
                for (i, &m) in s.states().iter().enumerate() {
                    assert_eq!(m.count_ones(), k);
                    assert!(m < (1u64 << n) || n == 64);
                    if let Some(p) = prev {
                        assert!(m > p, "not increasing at n={n} k={k} i={i}");
                    }
                    prev = Some(m);
                    assert_eq!(s.rank(m), i, "rank of {m:#b} at n={n} k={k}");
                }
            }
        }
    }

    #[test]
    fn full_width_sectors_enumerate() {
        // n = 64 is the u64 edge: no `1 << 64` anywhere in the enumeration.
        let s = Sector::new(64, 1);
        assert_eq!(s.dim(), 64);
        assert_eq!(s.mask(63), 1u64 << 63);
        assert_eq!(s.rank(1u64 << 63), 63);
        let s = Sector::new(64, 63);
        assert_eq!(s.dim(), 64);
        assert_eq!(s.mask(0), u64::MAX >> 1);
        let s = Sector::new(64, 64);
        assert_eq!(s.dim(), 1);
        assert_eq!(s.mask(0), u64::MAX);
        assert_eq!(s.rank(u64::MAX), 0);
    }
}
