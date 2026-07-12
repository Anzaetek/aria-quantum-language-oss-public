//! Exact minimum-weight decoder for the rotated surface code.
//!
//! For the small codes the toolkit exercises (d ≤ 5), we decode each CSS
//! sector independently by finding the **minimum-weight** Pauli-error support
//! whose check syndrome matches the observed one. This is precisely the
//! quantity graph-MWPM approximates, computed exactly here by bounded
//! enumeration — for d=3 the search is over ≤ C(9,2)=36 candidates per sector.
//!
//! * Z-type checks detect X (bit-flip) errors → correction returned in
//!   [`Correction::x_flips`] (data qubits to flip with X).
//! * X-type checks detect Z (phase-flip) errors → correction in
//!   [`Correction::z_flips`] (data qubits to flip with Z).
//!
//! Two different minimum-weight errors with the same syndrome differ by a
//! stabilizer (their product has trivial syndrome and weight < d, so it cannot
//! be a logical operator) — i.e. they are correction-equivalent, so picking any
//! minimum-weight representative is a valid decode.

use super::codes::{QECCode, SurfaceCode};

/// Full correction set for a surface-code syndrome: which data-qubit indices to
/// flip to undo the estimated error chain.
#[derive(Clone, Debug, Default)]
pub struct Correction {
    /// Data qubits to flip with X (corrects X errors flagged by Z-type checks).
    pub x_flips: Vec<usize>,
    /// Data qubits to flip with Z (corrects Z errors flagged by X-type checks).
    pub z_flips: Vec<usize>,
    /// Total correction weight `|x_flips| + |z_flips|`.
    pub weight: usize,
}

/// Syndrome of an error support against a set of checks: bit `k` is the parity
/// of the overlap between `error` and `checks[k]`.
fn syndrome_of(error: &[usize], checks: &[Vec<usize>]) -> Vec<u8> {
    let eset: std::collections::BTreeSet<usize> = error.iter().copied().collect();
    checks
        .iter()
        .map(|c| (c.iter().filter(|q| eset.contains(q)).count() % 2) as u8)
        .collect()
}

/// Invoke `f` on every `k`-subset of `0..n` (indices), stopping early if `f`
/// returns `true`. Returns the matching subset if found.
fn combinations<F: FnMut(&[usize]) -> bool>(n: usize, k: usize, mut f: F) -> Option<Vec<usize>> {
    let mut idx: Vec<usize> = (0..k).collect();
    if k > n {
        return None;
    }
    loop {
        if f(&idx) {
            return Some(idx);
        }
        // Advance to the next combination in lexicographic order.
        let mut i = k as isize - 1;
        while i >= 0 && idx[i as usize] == n - k + i as usize {
            i -= 1;
        }
        if i < 0 {
            return None;
        }
        idx[i as usize] += 1;
        for j in (i as usize + 1)..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// Minimum-weight error over `n_data` data qubits whose syndrome against
/// `checks` equals `target`. Enumerates by increasing weight up to `max_w`,
/// returning the first match (weight 0 = no error). Falls back to the empty set
/// if nothing is found within the bound (only reachable for uncorrectable,
/// high-weight syndromes outside the demo regime).
fn min_weight_error(
    n_data: usize,
    checks: &[Vec<usize>],
    target: &[u8],
    max_w: usize,
) -> Vec<usize> {
    if target.iter().all(|&b| b == 0) {
        return Vec::new();
    }
    for w in 1..=max_w.min(n_data) {
        if let Some(found) = combinations(n_data, w, |combo| syndrome_of(combo, checks) == target) {
            return found;
        }
    }
    Vec::new()
}

/// Decode a full surface-code syndrome (X-check bits `0..n_x`, then Z-check
/// bits `n_x..n_x+n_z`) into a minimum-weight [`Correction`].
pub fn decode_mwpm_correction(code: &SurfaceCode, syndrome: &[u8]) -> Correction {
    let n_data = code.n_physical();
    let n_x = code.x_checks().len();
    let n_z = code.z_checks().len();
    // Cap the search a little above the code distance: correctable errors are
    // well within this, and it bounds the enumeration for d=5.
    let max_w = code.distance() + 1;

    let x_bits = &syndrome[..n_x.min(syndrome.len())];
    let z_bits = if syndrome.len() >= n_x + n_z {
        &syndrome[n_x..n_x + n_z]
    } else {
        &[][..]
    };

    // Z-type checks flag X errors; X-type checks flag Z errors.
    let x_flips = min_weight_error(n_data, code.z_checks(), z_bits, max_w);
    let z_flips = min_weight_error(n_data, code.x_checks(), x_bits, max_w);
    let weight = x_flips.len() + z_flips.len();
    Correction {
        x_flips,
        z_flips,
        weight,
    }
}

/// Scalar minimum correction weight for a syndrome — thin wrapper retained for
/// callers that only need the weight, not the qubit-flip set.
pub fn decode_mwpm(code: &SurfaceCode, syndrome: &[u8]) -> usize {
    decode_mwpm_correction(code, syndrome).weight
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a full syndrome vector from separate X-check and Z-check bits.
    fn full_syndrome(code: &SurfaceCode, x_bits: &[u8], z_bits: &[u8]) -> Vec<u8> {
        let mut s = x_bits.to_vec();
        s.extend_from_slice(z_bits);
        let _ = code;
        s
    }

    #[test]
    fn clean_syndrome_decodes_to_nothing() {
        let code = SurfaceCode::new(3);
        let s = vec![0u8; code.n_ancilla()];
        let corr = decode_mwpm_correction(&code, &s);
        assert_eq!(corr.weight, 0);
        assert!(corr.x_flips.is_empty() && corr.z_flips.is_empty());
    }

    #[test]
    fn every_single_x_error_is_corrected() {
        // For each single X error, the Z-check syndrome must decode to an
        // X-correction that reproduces the same syndrome (residual trivial).
        let code = SurfaceCode::new(3);
        for q in 0..code.n_physical() {
            let z_bits = syndrome_of(&[q], code.z_checks());
            let s = full_syndrome(&code, &vec![0u8; code.x_checks().len()], &z_bits);
            let corr = decode_mwpm_correction(&code, &s);
            // residual = injected XOR correction; its syndrome must be zero.
            let mut resid: std::collections::BTreeSet<usize> = [q].into_iter().collect();
            for &c in &corr.x_flips {
                if !resid.remove(&c) {
                    resid.insert(c);
                }
            }
            let resid: Vec<usize> = resid.into_iter().collect();
            assert_eq!(
                syndrome_of(&resid, code.z_checks()),
                vec![0u8; code.z_checks().len()],
                "single X error on qubit {q} not corrected"
            );
            // ... and the correction must be weight ≤ 1 (minimum weight).
            assert!(corr.x_flips.len() <= 1);
        }
    }

    #[test]
    fn every_single_z_error_is_corrected() {
        let code = SurfaceCode::new(3);
        for q in 0..code.n_physical() {
            let x_bits = syndrome_of(&[q], code.x_checks());
            let s = full_syndrome(&code, &x_bits, &vec![0u8; code.z_checks().len()]);
            let corr = decode_mwpm_correction(&code, &s);
            let mut resid: std::collections::BTreeSet<usize> = [q].into_iter().collect();
            for &c in &corr.z_flips {
                if !resid.remove(&c) {
                    resid.insert(c);
                }
            }
            let resid: Vec<usize> = resid.into_iter().collect();
            assert_eq!(
                syndrome_of(&resid, code.x_checks()),
                vec![0u8; code.x_checks().len()]
            );
        }
    }

    #[test]
    fn decode_weight_matches_low_weight_errors() {
        let code = SurfaceCode::new(3);
        // A weight-2 X error (two qubits) → correction weight ≤ 2.
        let err = [0usize, 4];
        let z_bits = syndrome_of(&err, code.z_checks());
        let s = full_syndrome(&code, &vec![0u8; code.x_checks().len()], &z_bits);
        let corr = decode_mwpm_correction(&code, &s);
        assert!(corr.x_flips.len() <= 2);
    }
}
