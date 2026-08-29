//! Pauli algebra for the propagation engine.
//!
//! A Pauli string is stored in **raw symplectic form**: two bit vectors
//! `(x, z)` with the operator meaning
//!
//! ```text
//!   raw(x, z) = ∏_q  X_q^{x_q} Z_q^{z_q}        (q in increasing order)
//! ```
//!
//! The raw product of two such strings only ever differs by a **sign**
//! `(-1)^{Σ z₁·x₂}` (no factor of `i`), so every `i` — the one in `Y = i·XZ`,
//! the one from a Pauli rotation's `i sinθ` branch — is folded into a
//! `Complex64` **coefficient** carried alongside the string. That keeps the
//! string itself a clean hashable key, and a sum of Paulis a plain map.

use num_complex::Complex64;
use smallvec::{smallvec, SmallVec};

thread_local! {
    /// Every [`PauliSum::add_weighted`] call on this thread since
    /// [`reset_inserts`].
    ///
    /// # Why this exists
    ///
    /// The out-of-support skip's load-bearing evidence is that the insert count
    /// goes **flat across register width** — 8,737,669 at nq = 10, 14, 20 and 30
    /// on the corpus in `PLAN-PAULISUM-MAP.md`. Cost tracks the observable's
    /// light cone, not the register, which is the property pauliprop's
    /// width-unboundedness claim rests on.
    ///
    /// A count is immune to machine load in a way no wall-clock is, which makes
    /// it the better evidence — but until now there was **no insert counter in
    /// the tree**. That number came from an ad-hoc patch that was never
    /// committed, so the claim was load-immune and yet unreproducible by anyone,
    /// including on the box that produced it. A bench measures time and can
    /// never re-derive a count.
    ///
    /// Thread-local, not a global `AtomicU64`, for the reason written out at
    /// `sim::GATES_SKIPPED`: cargo runs tests in parallel threads and a
    /// process-wide counter reports other tests' work as your own. That mistake
    /// has already been made once here.
    static INSERTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Terms inserted or merged **on the calling thread** since [`reset_inserts`].
pub fn inserts() -> u64 {
    INSERTS.with(|c| c.get())
}

/// Reset the per-thread insert count.
pub fn reset_inserts() {
    INSERTS.with(|c| c.set(0));
}

/// A raw Pauli string `∏_q X_q^{x_q} Z_q^{z_q}` (no phase — phase lives in the
/// coefficient of whatever sum holds it). Hashable, so it keys a [`PauliSum`].
///
/// # Storage: packed `u64` words, not `Vec<bool>`
///
/// This used to be two `Vec<bool>`, i.e. **two heap allocations and one byte per
/// qubit**, for a structure whose entire job is to be hashed and compared inside
/// the propagation map. Profiling the GPU branch hook made the cost concrete —
/// see `PAULIPROP_GPU_PROFILE=1`:
///
/// * the merge (`add_weighted`, i.e. hash + compare + insert) was **54-60%** of
///   the whole branch step, and the CPU path pays the same cost;
/// * converting these vectors into packed words for the device (`pack_bits`) was
///   another **37-40%** — pure overhead the CPU path never pays.
///
/// Packed, a 100-qubit key hashes **16 bytes instead of 200**, compares two
/// words instead of two hundred, allocates **nothing** at or below 128 qubits
/// (`SmallVec` inline capacity), and hands the GPU its words with no conversion
/// at all.
///
/// ## The canonical-form invariant, which `Eq`/`Hash` depend on
///
/// Bits above `n` in the final word **must be zero**. Two keys that differ only
/// in padding would hash differently and compare unequal while representing the
/// same operator — silently splitting one term into two in the sum. Every
/// mutator here masks; `mul_raw` preserves it because XOR of two canonical
/// values is canonical.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PauliKey {
    /// `x` words followed by `z` words, `w = n.div_ceil(64)` each.
    /// Inline for `n <= 128`, which covers every circuit this engine is used on.
    words: SmallVec<[u64; 4]>,
    n: u32,
}

#[inline]
fn words_for(n: usize) -> usize {
    n.div_ceil(64)
}

/// Mask of the live bits in the final word, `!0` when `n` is a multiple of 64.
#[inline]
fn tail_mask(n: usize) -> u64 {
    let r = n % 64;
    if r == 0 {
        !0
    } else {
        (1u64 << r) - 1
    }
}

impl PauliKey {
    pub fn identity(n: usize) -> Self {
        Self {
            words: smallvec![0u64; 2 * words_for(n)],
            n: n as u32,
        }
    }

    /// Build from unpacked bit vectors. The bridge for callers that still think
    /// in `bool`s; prefer the packed accessors on any hot path.
    pub fn from_bits(x: &[bool], z: &[bool]) -> Self {
        debug_assert_eq!(x.len(), z.len());
        let n = x.len();
        let w = words_for(n);
        let mut words = smallvec![0u64; 2 * w];
        for (q, (&bx, &bz)) in x.iter().zip(z.iter()).enumerate() {
            if bx {
                words[q / 64] |= 1u64 << (q % 64);
            }
            if bz {
                words[w + q / 64] |= 1u64 << (q % 64);
            }
        }
        Self { words, n: n as u32 }
    }

    /// Build directly from packed words — the device path, with no conversion.
    ///
    /// Masks the tail defensively. XOR of canonical values is canonical, so
    /// words coming back from a kernel that only XORed should already satisfy
    /// the invariant; masking here means a kernel that ever violates it
    /// produces a wrong ANSWER rather than a silently split term, which is far
    /// easier to notice.
    pub fn from_words(x: &[u64], z: &[u64], n: usize) -> Self {
        let w = words_for(n);
        debug_assert!(x.len() >= w && z.len() >= w);
        let mut words: SmallVec<[u64; 4]> = smallvec![0u64; 2 * w];
        words[..w].copy_from_slice(&x[..w]);
        words[w..].copy_from_slice(&z[..w]);
        if w > 0 {
            let m = tail_mask(n);
            words[w - 1] &= m;
            words[2 * w - 1] &= m;
        }
        Self { words, n: n as u32 }
    }

    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.n as usize
    }

    /// The packed words, `x` half followed by `z` half.
    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    #[inline]
    fn w(&self) -> usize {
        words_for(self.n as usize)
    }

    /// Packed `x` words — handed straight to a device with no conversion.
    #[inline]
    pub fn x_words(&self) -> &[u64] {
        &self.words[..self.w()]
    }

    /// Packed `z` words.
    #[inline]
    pub fn z_words(&self) -> &[u64] {
        &self.words[self.w()..]
    }

    #[inline]
    pub fn x(&self, q: usize) -> bool {
        self.words[q / 64] >> (q % 64) & 1 == 1
    }

    #[inline]
    pub fn z(&self, q: usize) -> bool {
        let w = self.w();
        self.words[w + q / 64] >> (q % 64) & 1 == 1
    }

    #[inline]
    pub fn set_x(&mut self, q: usize, v: bool) {
        let b = 1u64 << (q % 64);
        if v {
            self.words[q / 64] |= b;
        } else {
            self.words[q / 64] &= !b;
        }
    }

    #[inline]
    pub fn set_z(&mut self, q: usize, v: bool) {
        let w = self.w();
        let b = 1u64 << (q % 64);
        if v {
            self.words[w + q / 64] |= b;
        } else {
            self.words[w + q / 64] &= !b;
        }
    }

    /// Number of qubits on which this string is not the identity.
    pub fn weight(&self) -> usize {
        let w = self.w();
        (0..w)
            .map(|i| (self.words[i] | self.words[w + i]).count_ones() as usize)
            .sum()
    }

    /// Does this string act as `I` or `Z` on every qubit (so `⟨0…0|·|0…0⟩ ≠ 0`)?
    pub fn is_all_iz(&self) -> bool {
        self.x_words().iter().all(|&v| v == 0)
    }

    /// Symplectic inner product parity with `(gx, gz)`: `true` iff the two
    /// strings ANTICOMMUTE. This is the hot test in the branch step — one
    /// popcount per word instead of a loop over qubits.
    pub fn anticommutes_with(&self, gx: &[u64], gz: &[u64]) -> bool {
        let w = self.w();
        let mut acc = 0u32;
        for i in 0..w {
            acc += (self.words[i] & gz[i]).count_ones();
            acc += (self.words[w + i] & gx[i]).count_ones();
        }
        acc & 1 == 1
    }

    /// Ordering key for `sorted_terms` — the packed words, which order
    /// deterministically.
    fn cmp_words(&self, other: &Self) -> std::cmp::Ordering {
        self.words.cmp(&other.words)
    }
}

/// Raw product `(x1,z1)·(x2,z2)` → `(x,z)` plus the `±1` sign from commuting
/// the `Z₁` factors past the `X₂` factors: `Z^{z₁}X^{x₂} = (-1)^{z₁·x₂}X^{x₂}Z^{z₁}`.
/// Packed form: `(gx, gz)` are the generator's words, `key` the term.
/// XOR of two canonical values is canonical, so the padding invariant survives.
pub fn mul_raw_packed(gx: &[u64], gz: &[u64], key: &PauliKey) -> (PauliKey, f64) {
    let w = key.w();
    let mut out = key.clone();
    let mut neg = 0u32;
    for i in 0..w {
        // sign = (-1)^{z1 . x2} with (1) = the generator on the LEFT.
        neg += (gz[i] & key.words[i]).count_ones();
        out.words[i] = gx[i] ^ key.words[i];
        out.words[w + i] = gz[i] ^ key.words[w + i];
    }
    let sign = if neg.is_multiple_of(2) { 1.0 } else { -1.0 };
    (out, sign)
}

/// A term's coefficient plus its **split frequency**: the number of sin-branches
/// (non-Clifford splits) the cheapest path to this Pauli has taken. This is the
/// axis PauliPropagation.jl's `max_freq` truncation acts on. When two paths reach
/// the same Pauli, the coefficients add and the frequency is the **minimum** —
/// the term is reachable within the smaller budget, so that's the honest bound.
#[derive(Clone, Copy, Debug)]
pub struct Weighted {
    pub coeff: Complex64,
    pub freq: u32,
}

/// A weighted sum of Pauli strings `Σ cₚ · P`, the observable as it propagates.
///
/// # The default hasher stays — an FxHash-style one was tried and was SLOWER
///
/// `terms` is a `HashMap` with std's SipHash-1-3, and the merge into it
/// (hash + compare + insert) is **60-89% of the branch step**, measured with
/// `PAULIPROP_GPU_PROFILE=1`. SipHash's DoS resistance buys nothing here — the
/// keys are Pauli strings this engine generates itself — so swapping it for a
/// cheap multiply-xor hash looks like free money. It is not:
///
/// ```text
///   qubits          12      40      72     100
///   FxHash vs std  -20%     -9%    +19%    +19%
/// ```
///
/// Reproduced across two runs. Narrow cases improve; **wide cases get
/// consistently worse**, and the mechanism is the interesting part: packed
/// Pauli keys are SPARSE and highly structured — mostly-zero words differing in
/// a handful of bits — which is exactly the input a weak multiply-xor hash
/// handles worst. The extra collisions and probing cost more than the cheaper
/// hash saves, and they cost most where the strings are longest.
///
/// So if the merge is to get faster it will not be by changing the hash
/// function. The map itself (open addressing over the packed words, or a
/// sorted-vec merge) is the thing to attack.
#[derive(Clone, Debug, Default)]
pub struct PauliSum {
    /// `pub(crate)`, NOT `pub`, and not fully private either.
    ///
    /// Public, this field let any crate do `sum.terms.insert(..)` and bypass the
    /// `support` maintenance in [`PauliSum::add_weighted`], leaving the mask an
    /// under-approximation and the out-of-support skip unsound. Both GPU hooks
    /// only ever READ it (`.len()` and iteration, six sites across
    /// `pauliprop-cuda` and `pauliprop-metal`), so closing it cost them the
    /// accessors below and no logic — a much smaller blast radius than it looks,
    /// and worth taking now at six sites rather than later at sixty.
    ///
    /// Fully private would mean visible only inside this module, and `sim.rs` is
    /// a sibling that legitimately needs `drain()` on the hot rebuild path.
    ///
    /// **Be honest about the scope: this stops bypasses from OUTSIDE the crate.
    /// It does not stop one from inside.** That is what
    /// [`PauliSum::debug_assert_support_is_superset`] is for, and why the guard
    /// landed first rather than being made redundant by this.
    pub(crate) terms: std::collections::HashMap<PauliKey, Weighted>,
    /// L1 mass of coefficients dropped by truncation so far (an error budget).
    pub dropped_mass: f64,
    /// Conservative SUPERSET of the qubits any term has support on: the OR of
    /// `x | z` over every key inserted through [`PauliSum::add_weighted`].
    ///
    /// This exists so a gate acting entirely OUTSIDE the support can be skipped
    /// instead of rebuilding the whole map into a bit-identical copy. With a
    /// local observable that is most of the circuit — the reason pauliprop can
    /// run at 100+ qubits at all is that the observable's light cone, not the
    /// register, bounds the work.
    ///
    /// **It may over-approximate, and that is the safety property.** Terms
    /// removed by truncation are not cleared from it, so a bit can stay set
    /// after the last term supporting it is gone. An over-approximation only
    /// makes the skip fire LESS often; it can never make it fire wrongly.
    /// Under-approximating would silently drop real work, so the mask is only
    /// ever OR-ed into, never cleared.
    ///
    /// Maintained by `add_weighted`. Every construction path in this workspace
    /// goes through it (there are no direct `terms.insert` callers anywhere);
    /// a future one that bypasses it would under-approximate, so don't.
    support: smallvec::SmallVec<[u64; 2]>,
}

impl PauliSum {
    pub fn new() -> Self {
        Self {
            terms: std::collections::HashMap::new(),
            dropped_mass: 0.0,
            support: smallvec::SmallVec::new(),
        }
    }

    /// Empty, but sized for `cap` terms up front.
    ///
    /// Every gate rebuilds the sum into a FRESH map (`map_one`, `map_two`,
    /// `branch` all start empty and insert), so an unsized map grows from zero
    /// and rehashes ~log2(cap) times per gate — rehashing being a full
    /// re-insert of everything inserted so far.
    ///
    /// **Size to the incoming term count, not to a worst-case fan-out.** Every
    /// caller here produces at least one entry per input term, so `n` never
    /// over-allocates. Sizing `branch` at its true 2n worst case was measured
    /// and is a PESSIMISATION — slower than not pre-sizing at all — because an
    /// over-sized table is a sparser one, and the lost cache locality costs
    /// more than the rehashes it avoids. See `PauliPropBackend::branch`.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            terms: std::collections::HashMap::with_capacity(cap),
            dropped_mass: 0.0,
            support: smallvec::SmallVec::new(),
        }
    }

    /// Number of distinct Pauli strings in the sum.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Clippy requires this beside `len`; the GPU hooks' `< min_terms()` checks
    /// are the real callers of the pair.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Iterate `(&key, &weighted)` in the map's arbitrary order.
    ///
    /// Read-only by construction: handing out `&mut` here would re-open exactly
    /// the bypass that closing [`PauliSum::terms`] shuts. Use
    /// [`PauliSum::sorted_terms`] when the ORDER has to be deterministic — it
    /// allocates and sorts, so it is for output, not for a device pack loop.
    pub fn iter(&self) -> impl Iterator<Item = (&PauliKey, &Weighted)> {
        self.terms.iter()
    }

    /// Add `coeff · P` at split-frequency 0 (Clifford / seed terms).
    pub fn add(&mut self, key: PauliKey, coeff: Complex64) {
        self.add_weighted(key, coeff, 0);
    }

    /// Add `coeff · P` carrying split-frequency `freq`, merging onto any existing
    /// entry: coefficients sum, frequency takes the minimum.
    pub fn add_weighted(&mut self, key: PauliKey, coeff: Complex64, freq: u32) {
        INSERTS.with(|c| c.set(c.get() + 1));
        self.note_support(&key);
        self.terms
            .entry(key)
            .and_modify(|w| {
                w.coeff += coeff;
                w.freq = w.freq.min(freq);
            })
            .or_insert(Weighted { coeff, freq });
    }

    /// OR a key's support into the mask. One OR per 64 qubits.
    #[inline]
    fn note_support(&mut self, key: &PauliKey) {
        let w = words_for(key.num_qubits());
        let ws = key.words();
        if self.support.len() < w {
            self.support.resize(w, 0);
        }
        for i in 0..w {
            self.support[i] |= ws[i] | ws[w + i];
        }
    }

    /// Panic in debug builds if `support` UNDER-approximates the live terms.
    ///
    /// # What this catches, and what it cannot
    ///
    /// The skip in `map_single` / `map_two` / `branch` is sound only while the
    /// mask is a genuine SUPERSET. `terms` is reachable as `pub(crate)`, so a
    /// direct `terms.insert` inside this crate would leave the mask short and
    /// the skip would then drop a gate it must not — a silently wrong
    /// expectation value, with no crash and no failing value test.
    ///
    /// **This is a smoke detector, not a proof.** The recomputation ORs over
    /// every live term, so a bypassed insert is INVISIBLE whenever its support
    /// is already covered by another surviving term. In a light-cone-local sum —
    /// the regime this engine exists for — terms share support, so a bypass is
    /// more likely to slip past this than to trip it. Do not read a green run as
    /// proof that nothing bypasses `add_weighted`.
    ///
    /// # Direction
    ///
    /// Superset, NOT equality. `truncate` removes terms via `retain` without
    /// clearing the mask, so a stored bit with no live term is legal and
    /// documented. An equality assert fires on correct code at the first
    /// truncation — verified, it trips
    /// `truncation_error_curve_is_certified_and_converges` with
    /// `recomputed 0x1e, stored 0x1f`.
    ///
    /// # Where it is called
    ///
    /// Immediately before each of the three mask CONSUMERS, not once per gate.
    /// Once per gate looks cheaper and is nearly useless: every non-skipped gate
    /// rebuilds the sum into a fresh `PauliSum` through `add_weighted` and
    /// assigns it over the old one, so the mask is regenerated and a bypass is
    /// washed away before the next gate's check runs. Measured on an injected
    /// bypass: three value mismatches, zero assert firings.
    pub(crate) fn debug_assert_support_is_superset(&self) {
        debug_assert!(
            self.support_superset_violation().is_none(),
            "PauliSum::support UNDER-approximates the live terms at {:?} \
             (word index, recomputed, stored). Something inserted into `terms` \
             without going through `add_weighted`, so an out-of-support skip is \
             about to drop a gate it must not. See PLAN-PAULISUM-INVARIANT.md.",
            self.support_superset_violation()
        );
    }

    /// First word where the stored mask misses a bit the live terms carry.
    ///
    /// Not `cfg`'d out: `debug_assert!` expands to `if cfg!(debug_assertions)
    /// { .. }`, so the call survives compilation in release and the function
    /// must exist. The branch is dead there and optimises away.
    fn support_superset_violation(&self) -> Option<(usize, u64, u64)> {
        let mut recomputed: smallvec::SmallVec<[u64; 2]> = smallvec::SmallVec::new();
        for key in self.terms.keys() {
            let w = words_for(key.num_qubits());
            let ws = key.words();
            if recomputed.len() < w {
                recomputed.resize(w, 0);
            }
            for i in 0..w {
                recomputed[i] |= ws[i] | ws[w + i];
            }
        }
        for (i, r) in recomputed.iter().enumerate() {
            let stored = self.support.get(i).copied().unwrap_or(0);
            if r & !stored != 0 {
                return Some((i, *r, stored));
            }
        }
        None
    }

    /// Does any term have support on qubit `q`? Conservative: `true` is always
    /// safe, `false` means provably no term touches `q`.
    #[inline]
    pub fn touches_qubit(&self, q: usize) -> bool {
        self.support
            .get(q / 64)
            .is_some_and(|word| (word >> (q % 64)) & 1 == 1)
    }

    /// Does any term overlap the generator `(gx, gz)`?
    ///
    /// `false` means every term COMMUTES with it: anticommutation is
    /// `popcount(x & gz) + popcount(z & gx)` odd, and both terms are zero when
    /// the supports are disjoint.
    #[inline]
    pub fn overlaps(&self, gx: &[u64], gz: &[u64]) -> bool {
        self.support.iter().enumerate().any(|(i, s)| {
            let g = gx.get(i).copied().unwrap_or(0) | gz.get(i).copied().unwrap_or(0);
            s & g != 0
        })
    }

    /// Iterate terms in a canonical (symplectic-key-sorted) order. The
    /// propagation `HashMap` is order-nondeterministic; callers that need
    /// reproducibility (GPU batch layout, top-K decisions) go through this.
    pub fn sorted_terms(&self) -> Vec<(&PauliKey, &Weighted)> {
        let mut v: Vec<_> = self.terms.iter().collect();
        v.sort_by(|(a, _), (b, _)| a.cmp_words(b));
        v
    }

    /// Drop terms below `coeff_min` (by magnitude), above `max_weight` (Pauli
    /// weight), or above `max_freq` (split frequency); accumulate the dropped
    /// magnitude into `dropped_mass`.
    pub fn truncate(&mut self, coeff_min: f64, max_weight: Option<usize>, max_freq: Option<u32>) {
        let mut dropped = 0.0;
        self.terms.retain(|k, w| {
            let too_small = w.coeff.norm() < coeff_min;
            let too_heavy = max_weight.is_some_and(|m| k.weight() > m);
            let too_deep = max_freq.is_some_and(|m| w.freq > m);
            if too_small || too_heavy || too_deep {
                dropped += w.coeff.norm();
                false
            } else {
                true
            }
        });
        self.dropped_mass += dropped;
    }
}

#[cfg(test)]
mod tests {
    //! The crate's first unit-test module, and it has to be one.
    //!
    //! The bypass these tests perform — writing straight into `terms` — is
    //! exactly what `terms` being non-`pub` prevents, so it cannot be written
    //! from `tests/`, which is a separate crate. An integration test could only
    //! assert that the guard stays quiet on correct input, which is the half
    //! that proves nothing.

    use super::*;

    fn zkey(n: usize, z_on: &[usize]) -> PauliKey {
        let mut k = PauliKey::identity(n);
        for q in z_on {
            k.set_z(*q, true);
        }
        k
    }

    fn one() -> Complex64 {
        Complex64::new(1.0, 0.0)
    }

    /// The maintained path must never trip the guard, at any width.
    #[test]
    fn a_sum_built_through_add_weighted_never_violates() {
        for n in [1usize, 8, 64, 65, 130] {
            let mut s = PauliSum::new();
            s.add(zkey(n, &[0]), one());
            s.add(zkey(n, &[n - 1]), one());
            s.add_weighted(zkey(n, &[0, n - 1]), one(), 2);
            assert!(
                s.support_superset_violation().is_none(),
                "n={n}: add_weighted maintains the mask by construction"
            );
        }
    }

    /// The guard's reason to exist: a direct `terms.insert` leaves the mask
    /// short, and the next skip decision is then unsound.
    ///
    /// The inserted key touches qubit 5, which **no other term carries** — that
    /// shape is required, see the sibling test below.
    #[test]
    fn a_direct_insert_carrying_a_new_support_bit_is_caught() {
        let mut s = PauliSum::new();
        s.add(zkey(8, &[0]), one());
        s.terms.insert(
            zkey(8, &[5]),
            Weighted {
                coeff: one(),
                freq: 0,
            },
        );
        let v = s
            .support_superset_violation()
            .expect("a bypassed insert widening the support must be detected");
        assert_eq!(v.0, 0, "the violation is in word 0");
        assert_eq!(
            v.1 & !v.2,
            1 << 5,
            "the missing bit is exactly qubit 5's: recomputed {:#x}, stored {:#x}",
            v.1,
            v.2
        );
    }

    /// **The limitation, pinned as a test rather than left in prose.**
    ///
    /// A bypassed insert whose support is already covered by a surviving term is
    /// INVISIBLE to the guard — the recomputation ORs over all terms, so it sees
    /// nothing new. This is the common case in a light-cone-local sum, where
    /// terms share support.
    ///
    /// This test asserts the guard STAYS QUIET on a real bypass. It exists so
    /// that a future reader who assumes a green suite proves no bypass exists
    /// finds the counter-example in the same file.
    #[test]
    fn a_direct_insert_hidden_under_existing_support_is_not_caught() {
        let mut s = PauliSum::new();
        s.add(zkey(8, &[0, 5]), one());
        s.terms.insert(
            zkey(8, &[5]),
            Weighted {
                coeff: one(),
                freq: 0,
            },
        );
        assert!(
            s.support_superset_violation().is_none(),
            "documented blind spot: qubit 5 was already in the mask, so the \
             bypass adds no bit and the guard cannot see it. If this ever starts \
             failing the guard got STRONGER — update the docs, do not 'fix' it."
        );
    }

    /// The over-eager direction. Truncation removes terms without clearing the
    /// mask, so the mask legally over-approximates afterwards. An equality
    /// check here would fire on correct code.
    #[test]
    fn truncation_leaves_stale_bits_and_that_is_legal() {
        let mut s = PauliSum::new();
        s.add(zkey(8, &[0]), one());
        s.add(zkey(8, &[5]), Complex64::new(1e-12, 0.0));
        s.truncate(1e-6, None, None);

        assert_eq!(s.terms.len(), 1, "the tiny term is gone");
        assert!(
            s.touches_qubit(5),
            "and its support bit is deliberately NOT cleared"
        );
        assert!(
            s.support_superset_violation().is_none(),
            "a stale bit is an over-approximation, which is the safety property"
        );
    }
}
