//! Strong Linear Optical Simulation (SLOS).
//!
//! Implements SLOS_full: computes the full output Fock-state distribution
//! for a given input Fock state and unitary transfer matrix.
//!
//! SLOS computes all output amplitudes in O(n * M_n) where M_n = C(n+m-1, m-1)
//! is the number of ways to distribute n photons into m modes.
//! This is exponentially faster than computing each permanent individually.

use num_complex::Complex64;

use crate::permanent::permanent;

/// A Fock state: vector of photon numbers per mode.
/// E.g., [2, 0, 1] means 2 photons in mode 0, 0 in mode 1, 1 in mode 2.
pub type FockState = Vec<u32>;

/// Total photon number.
pub fn total_photons(state: &FockState) -> u32 {
    state.iter().sum()
}

/// Compute the output probability distribution for a photonic circuit.
///
/// Given:
/// - `unitary`: m×m unitary transfer matrix
/// - `input`: input Fock state (m modes)
///
/// Returns: Vec of (output_fock_state, probability)
///
/// Uses SLOS for efficiency: iterates over all output states with the
/// same total photon number, computing each amplitude via the permanent
/// of the appropriate submatrix with repetitions.
pub fn slos_full(unitary: &[Vec<Complex64>], input: &FockState) -> Vec<(FockState, f64)> {
    let m = unitary.len();
    let n = total_photons(input) as usize;

    if n == 0 {
        // Vacuum in -> vacuum out with probability 1
        return vec![(vec![0; m], 1.0)];
    }

    // Build input column indices with repetitions:
    // If input = [2, 0, 1], columns = [0, 0, 2]
    let input_cols = fock_to_mode_list(input);

    // Enumerate all output Fock states with n total photons in m modes
    let output_states = enumerate_fock_states(m, n as u32);

    let mut results = Vec::with_capacity(output_states.len());

    // Normalization factor: product of factorials of input photon numbers
    let input_norm: f64 = input
        .iter()
        .map(|&ni| factorial(ni as usize) as f64)
        .product();

    for output in &output_states {
        // Build output row indices with repetitions
        let output_rows = fock_to_mode_list(output);

        // Extract the n×n submatrix U[output_rows, input_cols]
        let sub = extract_submatrix(unitary, &output_rows, &input_cols);

        // Amplitude = perm(sub) / sqrt(prod(n_i!) * prod(m_j!))
        let perm = permanent(&sub);

        let output_norm: f64 = output
            .iter()
            .map(|&nj| factorial(nj as usize) as f64)
            .product();

        let prob = perm.norm_sqr() / (input_norm * output_norm);

        if prob > 1e-15 {
            results.push((output.clone(), prob));
        }
    }

    results
}

/// Compute the amplitude for a specific output Fock state.
pub fn fock_amplitude(
    unitary: &[Vec<Complex64>],
    input: &FockState,
    output: &FockState,
) -> Complex64 {
    let input_cols = fock_to_mode_list(input);
    let output_rows = fock_to_mode_list(output);

    if input_cols.len() != output_rows.len() {
        return Complex64::new(0.0, 0.0); // Different photon numbers
    }

    let sub = extract_submatrix(unitary, &output_rows, &input_cols);
    let perm = permanent(&sub);

    let input_norm: f64 = input
        .iter()
        .map(|&ni| factorial(ni as usize) as f64)
        .product();
    let output_norm: f64 = output
        .iter()
        .map(|&nj| factorial(nj as usize) as f64)
        .product();

    perm / Complex64::new((input_norm * output_norm).sqrt(), 0.0)
}

/// Convert a Fock state to a list of mode indices with repetitions.
/// [2, 0, 1] -> [0, 0, 2]
fn fock_to_mode_list(state: &FockState) -> Vec<usize> {
    let mut modes = Vec::new();
    for (mode, &count) in state.iter().enumerate() {
        for _ in 0..count {
            modes.push(mode);
        }
    }
    modes
}

/// Extract submatrix with given row and column indices (may repeat).
fn extract_submatrix(u: &[Vec<Complex64>], rows: &[usize], cols: &[usize]) -> Vec<Vec<Complex64>> {
    rows.iter()
        .map(|&r| cols.iter().map(|&c| u[r][c]).collect())
        .collect()
}

/// Enumerate all Fock states with exactly `n` total photons in `m` modes.
/// Returns them in lexicographic order.
pub fn enumerate_fock_states(m: usize, n: u32) -> Vec<FockState> {
    let mut results = Vec::new();
    let mut current = vec![0u32; m];
    enumerate_helper(&mut results, &mut current, 0, m, n);
    results
}

fn enumerate_helper(
    results: &mut Vec<FockState>,
    current: &mut FockState,
    mode: usize,
    m: usize,
    remaining: u32,
) {
    if mode == m - 1 {
        current[mode] = remaining;
        results.push(current.clone());
        current[mode] = 0;
        return;
    }
    for k in (0..=remaining).rev() {
        current[mode] = k;
        enumerate_helper(results, current, mode + 1, m, remaining - k);
    }
    current[mode] = 0;
}

/// Compute n!
fn factorial(n: usize) -> u64 {
    (1..=n as u64).product()
}

/// SLOS with output masking (SLOS_gen): only compute amplitudes for
/// output states that match the given mask.
///
/// `mask`: for each mode, (min_photons, max_photons). Only output states
/// where mode i has between min and max photons are computed.
pub fn slos_masked(
    unitary: &[Vec<Complex64>],
    input: &FockState,
    mask: &[(u32, u32)],
) -> Vec<(FockState, f64)> {
    let m = unitary.len();
    let n = total_photons(input);

    if n == 0 {
        let vacuum = vec![0u32; m];
        let valid = mask
            .iter()
            .enumerate()
            .all(|(i, &(lo, hi))| vacuum[i] >= lo && vacuum[i] <= hi);
        if valid {
            return vec![(vacuum, 1.0)];
        } else {
            return vec![];
        }
    }

    let input_cols = fock_to_mode_list(input);
    let input_norm: f64 = input
        .iter()
        .map(|&ni| factorial(ni as usize) as f64)
        .product();

    let output_states = enumerate_fock_states_masked(m, n, mask);

    let mut results = Vec::new();
    for output in &output_states {
        let output_rows = fock_to_mode_list(output);
        let sub = extract_submatrix(unitary, &output_rows, &input_cols);
        let perm = permanent(&sub);
        let output_norm: f64 = output
            .iter()
            .map(|&nj| factorial(nj as usize) as f64)
            .product();
        let prob = perm.norm_sqr() / (input_norm * output_norm);
        if prob > 1e-15 {
            results.push((output.clone(), prob));
        }
    }

    results
}

/// Enumerate Fock states with masking constraints.
fn enumerate_fock_states_masked(m: usize, n: u32, mask: &[(u32, u32)]) -> Vec<FockState> {
    let mut results = Vec::new();
    let mut current = vec![0u32; m];
    enumerate_masked_helper(&mut results, &mut current, 0, m, n, mask);
    results
}

fn enumerate_masked_helper(
    results: &mut Vec<FockState>,
    current: &mut FockState,
    mode: usize,
    m: usize,
    remaining: u32,
    mask: &[(u32, u32)],
) {
    let (lo, hi) = if mode < mask.len() {
        mask[mode]
    } else {
        (0, remaining)
    };

    if mode == m - 1 {
        if remaining >= lo && remaining <= hi {
            current[mode] = remaining;
            results.push(current.clone());
            current[mode] = 0;
        }
        return;
    }

    let max_here = remaining.min(hi);
    for k in (lo..=max_here).rev() {
        current[mode] = k;
        enumerate_masked_helper(results, current, mode + 1, m, remaining - k, mask);
    }
    current[mode] = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components;

    #[test]
    fn test_enumerate_fock_states() {
        // 2 photons in 2 modes: |20>, |11>, |02>
        let states = enumerate_fock_states(2, 2);
        assert_eq!(states.len(), 3);
        assert_eq!(states[0], vec![2, 0]);
        assert_eq!(states[1], vec![1, 1]);
        assert_eq!(states[2], vec![0, 2]);
    }

    #[test]
    fn test_enumerate_3_photons_3_modes() {
        // C(3+3-1, 3-1) = C(5,2) = 10
        let states = enumerate_fock_states(3, 3);
        assert_eq!(states.len(), 10);
    }

    #[test]
    fn test_vacuum_through_anything() {
        let u = components::identity(3);
        let input = vec![0, 0, 0];
        let result = slos_full(&u, &input);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, vec![0, 0, 0]);
        assert!((result[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_photon_identity() {
        // |1,0> through identity -> |1,0> with prob 1
        let u = components::identity(2);
        let input = vec![1, 0];
        let result = slos_full(&u, &input);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, vec![1, 0]);
        assert!((result[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_photon_through_50_50_bs() {
        // |1,0> through 50:50 BS -> equal prob |1,0> and |0,1>
        let ops = vec![components::PhotonicOp::BeamSplitterRx {
            mode0: 0,
            mode1: 1,
            theta: std::f64::consts::FRAC_PI_4,
            phi: 0.0,
        }];
        let u = components::build_unitary(2, &ops);
        let input = vec![1, 0];
        let result = slos_full(&u, &input);

        let total_prob: f64 = result.iter().map(|(_, p)| p).sum();
        assert!(
            (total_prob - 1.0).abs() < 1e-10,
            "total prob = {}",
            total_prob
        );

        // Should have |1,0> and |0,1> each with prob 0.5
        for (state, prob) in &result {
            assert!(
                (*prob - 0.5).abs() < 1e-10,
                "state {:?} has prob {} (expected 0.5)",
                state,
                prob
            );
        }
    }

    #[test]
    fn test_hong_ou_mandel() {
        // Hong-Ou-Mandel effect: |1,1> into 50:50 BS
        // Output: (|2,0> + |0,2>)/sqrt(2), probability 0.5 each
        // |1,1> -> 0 probability (destructive interference)
        let ops = vec![components::PhotonicOp::BeamSplitterRx {
            mode0: 0,
            mode1: 1,
            theta: std::f64::consts::FRAC_PI_4,
            phi: 0.0,
        }];
        let u = components::build_unitary(2, &ops);
        let input = vec![1, 1];
        let result = slos_full(&u, &input);

        let total_prob: f64 = result.iter().map(|(_, p)| p).sum();
        assert!(
            (total_prob - 1.0).abs() < 1e-10,
            "total prob = {}",
            total_prob
        );

        // |1,1> should have 0 probability (HOM dip)
        let p11 = result
            .iter()
            .find(|(s, _)| *s == vec![1, 1])
            .map(|(_, p)| *p)
            .unwrap_or(0.0);
        assert!(p11 < 1e-10, "HOM: |1,1> prob should be ~0, got {}", p11);

        // |2,0> and |0,2> should each have 0.5
        let p20 = result
            .iter()
            .find(|(s, _)| *s == vec![2, 0])
            .map(|(_, p)| *p)
            .unwrap_or(0.0);
        let p02 = result
            .iter()
            .find(|(s, _)| *s == vec![0, 2])
            .map(|(_, p)| *p)
            .unwrap_or(0.0);
        assert!((p20 - 0.5).abs() < 1e-10, "HOM: |2,0> prob = {}", p20);
        assert!((p02 - 0.5).abs() < 1e-10, "HOM: |0,2> prob = {}", p02);
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        // 2 photons through a non-trivial 3-mode circuit
        let ops = vec![
            components::PhotonicOp::PhaseShifter { mode: 0, phi: 0.5 },
            components::PhotonicOp::BeamSplitterRx {
                mode0: 0,
                mode1: 1,
                theta: 0.7,
                phi: 0.3,
            },
            components::PhotonicOp::BeamSplitterRx {
                mode0: 1,
                mode1: 2,
                theta: 0.4,
                phi: 0.6,
            },
        ];
        let u = components::build_unitary(3, &ops);
        let input = vec![1, 1, 0];
        let result = slos_full(&u, &input);

        let total: f64 = result.iter().map(|(_, p)| p).sum();
        assert!(
            (total - 1.0).abs() < 1e-8,
            "probabilities sum to {} (expected 1.0)",
            total
        );
    }

    #[test]
    fn test_slos_masked() {
        // Only get outputs where mode 0 has exactly 1 photon
        let u = components::identity(2);
        let input = vec![1, 1];
        let mask = vec![(1, 1), (0, 1)]; // mode 0: exactly 1, mode 1: 0 or 1
        let result = slos_masked(&u, &input, &mask);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, vec![1, 1]);
    }
}
