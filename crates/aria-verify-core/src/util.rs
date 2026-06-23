// SPDX-License-Identifier: Apache-2.0
//! Small decode/format helpers shared by several harnesses.

use crate::Complex64;

/// Reverse the low `n` bits of `k` (little/big-endian index conversion).
pub fn bitrev(k: usize, n: u32) -> usize {
    let mut r = 0usize;
    for b in 0..n {
        if (k >> b) & 1 == 1 {
            r |= 1 << (n - 1 - b);
        }
    }
    r
}

/// Flatten complex amplitudes to interleaved `(re, im)` for vector comparison.
pub fn interleave(sv: &[Complex64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(sv.len() * 2);
    for a in sv {
        out.push(a.re);
        out.push(a.im);
    }
    out
}

/// Decode the `omega_app` counts payload `[bits0, count0, bits1, count1, ...]`.
pub fn counts_from_payload(payload: &[f64]) -> Vec<(u64, u64)> {
    payload
        .chunks_exact(2)
        .map(|c| (c[0] as u64, c[1] as u64))
        .collect()
}

/// Total shot count across all outcomes.
pub fn total_shots(c: &[(u64, u64)]) -> u64 {
    c.iter().map(|(_, n)| n).sum()
}

/// Probability of a specific outcome `key`.
pub fn prob_of(c: &[(u64, u64)], key: u64) -> f64 {
    let t = total_shots(c) as f64;
    if t == 0.0 {
        return 0.0;
    }
    c.iter()
        .find(|(k, _)| *k == key)
        .map(|(_, n)| *n as f64)
        .unwrap_or(0.0)
        / t
}

/// Most-frequent outcome.
pub fn argmax_outcome(c: &[(u64, u64)]) -> u64 {
    c.iter()
        .max_by_key(|(_, n)| *n)
        .map(|(k, _)| *k)
        .unwrap_or(0)
}
