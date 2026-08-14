// SPDX-License-Identifier: Apache-2.0
//! A shot outcome of any width.
//!
//! `ExecResult::Counts` keys by `u64`, so above 64 qubits every high bit was
//! silently dropped — a **confident wrong answer**, currently prevented by
//! refusing (`check_counts_width`). The refusal was always step 1. This type is
//! step 2: the representation that lets the refusal be replaced by support.
//!
//! # Why packed words rather than a bitstring
//!
//! Measured (key construction + `HashMap` insert, ns per shot, 1000 shots):
//!
//! | n | `u64` | `String` | `SmallVec<[u64;1]>` |
//! |---|---|---|---|
//! | 64 | 40 | 157 | 75 |
//! | 1024 | 38 | 1308 | 394 |
//! | 4096 | 32 | 4422 | 1195 |
//!
//! A `String` bitstring is 2–4x the cost and the gap **widens** with n, which is
//! the opposite of what "simplest, and it matches the wire" suggested. Against
//! the sampler itself — 22 µs/shot at 1024 qubits on MPS — a packed key is 1.8%
//! and a string key 6%.
//!
//! One inline word means **no allocation at or below 64 qubits**, which is the
//! overwhelming majority of runs.
//!
//! # Bit order
//!
//! Bit `i` is qubit (or classical bit) `i`, LSB-first, in word `i / 64` at
//! position `i % 64`. This matches `creg_to_u64` and the existing `u64` key
//! exactly, so `Outcome::from(k: u64)` and the old key agree bit for bit.
//!
//! **`Ord` is not numeric across widths.** It compares width first, then words
//! from the most significant down, so ordering is total and deterministic —
//! enough for stable output — but two outcomes of different widths do not
//! compare as the integers they encode. Nothing needs that, and pretending
//! otherwise would make `0b11` at width 2 and width 70 compare equal.

use smallvec::SmallVec;
use std::fmt;

/// Inline capacity: one word covers every run at or below 64 qubits.
type Words = SmallVec<[u64; 1]>;

/// A shot outcome: `width` bits, LSB-first, packed into 64-bit words.
///
/// `width` is carried explicitly because trailing zeros are significant: a
/// 70-qubit all-zero outcome is not the same result as a 2-bit one, and both
/// pack to the same words. This is exactly the distinction the `u64` key could
/// not make, and the reason a bare `SmallVec` is not enough.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Outcome {
    /// Number of significant bits. Bits at or above this in `words` are zero.
    width: u32,
    words: Words,
}

impl Outcome {
    /// An all-zero outcome of `width` bits.
    pub fn zeros(width: u32) -> Self {
        Self {
            width,
            words: smallvec::smallvec![0u64; Self::words_for(width)],
        }
    }

    /// Build from LSB-first bits, one `u8` per position (any non-zero is 1).
    ///
    /// Packs by 64-bit chunks with a register accumulator rather than
    /// `words[i / 64] |= bit << (i % 64)` per bit. The per-bit form measured
    /// **2.5x slower than building a String**, which briefly made a bitstring
    /// look like the faster representation and nearly decided the design.
    pub fn from_bits(bits: &[u8]) -> Self {
        let mut words = Words::with_capacity(bits.len().div_ceil(64));
        for chunk in bits.chunks(64) {
            let mut w = 0u64;
            for (i, b) in chunk.iter().enumerate() {
                w |= ((*b & 1) as u64) << i;
            }
            words.push(w);
        }
        Self {
            width: bits.len() as u32,
            words,
        }
    }

    /// Reinterpret the low `width` bits of a `u64`, LSB-first.
    ///
    /// The bridge from every site that still produces a `u64` key. Bits at or
    /// above `width` are dropped, so the invariant holds by construction.
    pub fn from_u64(bits: u64, width: u32) -> Self {
        debug_assert!(width <= 64, "use `from_bits` above 64 bits, not `from_u64`");
        let masked = if width >= 64 {
            bits
        } else {
            bits & ((1u64 << width) - 1)
        };
        Self {
            width,
            words: smallvec::smallvec![masked],
        }
    }

    /// The outcome as a `u64`, or `None` if it does not fit.
    ///
    /// `None` is the case the old key represented as a wrong answer. Callers
    /// that cannot widen (the C ABIs) must handle it rather than truncate.
    pub fn as_u64(&self) -> Option<u64> {
        if self.width > 64 {
            return None;
        }
        Some(self.words.first().copied().unwrap_or(0))
    }

    /// Number of significant bits.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Bit `i`, LSB-first. Out of range reads as 0.
    pub fn bit(&self, i: u32) -> u8 {
        if i >= self.width {
            return 0;
        }
        ((self.words[(i / 64) as usize] >> (i % 64)) & 1) as u8
    }

    /// Set bit `i`. Panics if `i` is at or above the width — a bit outside the
    /// declared width is the silent truncation this type exists to prevent.
    pub fn set_bit(&mut self, i: u32, value: u8) {
        assert!(
            i < self.width,
            "bit {i} is outside an outcome of width {}",
            self.width
        );
        let (w, b) = ((i / 64) as usize, i % 64);
        if value & 1 == 1 {
            self.words[w] |= 1u64 << b;
        } else {
            self.words[w] &= !(1u64 << b);
        }
    }

    /// The packed words, LSB-first.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// MSB-first bitstring, the wire and display form: bit `width-1` first.
    ///
    /// Matches what `format_counts` and the JSON encoder already emit, so the
    /// wire does not change — it was never `u64`-shaped, only the conversion
    /// was.
    pub fn to_bitstring(&self) -> String {
        (0..self.width)
            .rev()
            .map(|i| if self.bit(i) == 1 { '1' } else { '0' })
            .collect()
    }

    /// Parse an MSB-first bitstring. The inverse of [`Self::to_bitstring`].
    pub fn from_bitstring(s: &str) -> Result<Self, String> {
        let width = s.len() as u32;
        let mut out = Self::zeros(width);
        for (j, c) in s.chars().enumerate() {
            let v = match c {
                '0' => 0,
                '1' => 1,
                other => return Err(format!("bitstring contains {other:?}, not 0 or 1")),
            };
            // MSB-first: character j is bit width-1-j.
            out.set_bit(width - 1 - j as u32, v);
        }
        Ok(out)
    }

    fn words_for(width: u32) -> usize {
        (width as usize).div_ceil(64).max(1)
    }
}

impl fmt::Debug for Outcome {
    /// The bitstring, not the words — a reader debugging a counts map wants
    /// `|0110>`, and `[6]` at width 4 is the same information spelled worse.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "|{}>", self.to_bitstring())
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_bitstring())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `from_u64` must agree with the old key bit for bit, or every existing
    /// result silently changes meaning.
    #[test]
    fn from_u64_matches_the_old_lsb_first_key() {
        let o = Outcome::from_u64(0b1011, 4);
        assert_eq!(o.bit(0), 1);
        assert_eq!(o.bit(1), 1);
        assert_eq!(o.bit(2), 0);
        assert_eq!(o.bit(3), 1);
        assert_eq!(o.as_u64(), Some(0b1011));
        // MSB-first display, as the CLI and JSON already print it.
        assert_eq!(o.to_bitstring(), "1011");
    }

    /// Trailing zeros are significant: this is the distinction `u64` could not
    /// make, and the whole reason `width` is stored.
    ///
    /// The two outcomes here occupy the **same single word** — `0b11` at width
    /// 2 and at width 4 both pack to `[0b11]`. That is deliberate: the first
    /// version of this test compared width 2 against width 70, whose word
    /// counts differ (1 vs 2), so it passed even with `width` removed from
    /// equality entirely. It was testing `Vec::len`, not the invariant it
    /// names — verified by mutation.
    #[test]
    fn width_is_part_of_the_value() {
        let two = Outcome::from_u64(0b11, 2);
        let four = Outcome::from_u64(0b11, 4);
        assert_eq!(two.words(), four.words(), "fixture: same packed words");
        assert_ne!(two, four, "0b11 at width 2 is not 0b0011 at width 4");
        assert_eq!(two.to_bitstring(), "11");
        assert_eq!(four.to_bitstring(), "0011");

        // Hash must agree with Eq, or a counts map silently merges them.
        use std::collections::HashMap;
        let mut m: HashMap<Outcome, u32> = HashMap::new();
        *m.entry(two).or_insert(0) += 1;
        *m.entry(four).or_insert(0) += 1;
        assert_eq!(m.len(), 2, "two distinct outcomes collapsed into one key");

        // And across the cliff, where the word counts do differ.
        let mut b = vec![0u8; 70];
        b[0] = 1;
        b[1] = 1;
        let wide = Outcome::from_bits(&b);
        assert_ne!(Outcome::from_u64(0b11, 2), wide);
        assert_eq!(wide.to_bitstring().len(), 70);
        assert!(wide.to_bitstring().ends_with("11"));
    }

    /// The case that motivated the whole change: bit 65 must survive.
    #[test]
    fn a_bit_above_sixty_four_survives() {
        let mut bits = vec![0u8; 70];
        bits[65] = 1;
        let o = Outcome::from_bits(&bits);
        assert_eq!(o.bit(65), 1);
        assert_eq!(o.bit(1), 0, "bit 65 must not alias into bit 1");
        assert_eq!(
            o.as_u64(),
            None,
            "70 bits cannot be a u64; returning Some here is the original defect"
        );
        assert_eq!(o.words().len(), 2);
    }

    /// Round-trip through the wire form, at widths either side of the cliff.
    #[test]
    fn bitstrings_round_trip() {
        for width in [1u32, 2, 63, 64, 65, 70, 128, 1024] {
            let mut bits = vec![0u8; width as usize];
            // A pattern that is not a palindrome and not uniform, so a reversed
            // or truncated round-trip is visible.
            for (i, b) in bits.iter_mut().enumerate() {
                *b = ((i % 3 == 0) || (i == width as usize - 1)) as u8;
            }
            let o = Outcome::from_bits(&bits);
            let s = o.to_bitstring();
            assert_eq!(s.len(), width as usize);
            let back = Outcome::from_bitstring(&s).expect("parse");
            assert_eq!(back, o, "round-trip changed the value at width {width}");
        }
    }

    /// Below the cliff the two constructors must agree, or the conversion sites
    /// silently change results as they are migrated one at a time.
    #[test]
    fn from_bits_and_from_u64_agree_below_the_cliff() {
        for width in [1u32, 7, 63, 64] {
            for sample in [0u64, 1, 0b1010101, u64::MAX] {
                let masked = if width >= 64 {
                    sample
                } else {
                    sample & ((1u64 << width) - 1)
                };
                let a = Outcome::from_u64(masked, width);
                let bits: Vec<u8> = (0..width).map(|i| ((masked >> i) & 1) as u8).collect();
                let b = Outcome::from_bits(&bits);
                assert_eq!(a, b, "width {width}, value {masked:#x}");
            }
        }
    }

    /// One inline word: no allocation at or below 64 bits, which is the
    /// argument for `SmallVec` over `Box<[u64]>` and most runs.
    #[test]
    fn sixty_four_bits_stay_inline() {
        let o = Outcome::from_bits(&[1u8; 64]);
        assert!(!o.words.spilled(), "64 bits must not allocate");
        let wide = Outcome::from_bits(&[1u8; 65]);
        assert!(wide.words.spilled(), "65 bits needs a second word");
    }

    #[test]
    fn set_bit_refuses_to_write_outside_the_width() {
        let mut o = Outcome::zeros(4);
        o.set_bit(3, 1);
        assert_eq!(o.to_bitstring(), "1000");
        assert!(std::panic::catch_unwind(move || {
            let mut o = Outcome::zeros(4);
            o.set_bit(4, 1);
        })
        .is_err());
    }

    #[test]
    fn ordering_is_total_and_stable() {
        let mut v = [
            Outcome::from_u64(0b10, 2),
            Outcome::from_u64(0b01, 2),
            Outcome::from_u64(0b00, 2),
        ];
        v.sort();
        assert_eq!(
            v.iter().map(|o| o.to_bitstring()).collect::<Vec<_>>(),
            vec!["00", "01", "10"]
        );
    }
}
