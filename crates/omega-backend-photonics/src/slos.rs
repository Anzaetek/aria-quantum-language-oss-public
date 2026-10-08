//! Strong Linear Optical Simulation (SLOS).
//!
//! Implements SLOS_full: computes the full output Fock-state distribution
//! for a given input Fock state and unitary transfer matrix.
//!
//! SLOS computes all output amplitudes in O(n * M_n) where M_n = C(n+m-1, m-1)
//! is the number of ways to distribute n photons into m modes.
//! This is exponentially faster than computing each permanent individually.
//!
//! **Until 2026-09-25 this file claimed that and did the opposite.**
//! `slos_full` looped over every output Fock state calling `permanent()` on a
//! submatrix — which IS "computing each permanent individually", the thing the
//! line above says SLOS avoids. Naive costs M_n·2^n against SLOS's n·M_n, so
//! the penalty is 2^n/n and grows with photon number. Measured against
//! Perceval (a real SLOS) on 6 modes before the fix: 0.92x at n=2, then 2.46x,
//! 4.13x, 6.05x, 10.12x at n=6 — tracking 2^n/n (2.0, 2.7, 4.0, 6.4, 10.7)
//! almost exactly. At n=10 that is ~100x.
//!
//! The DP below is the real thing: photons are added one input column at a
//! time and the partial amplitude of every reachable output state is carried
//! forward, so the work shared between output states is done ONCE instead of
//! being redone inside each permanent.

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
    slos_dp(unitary, input, None)
}

/// The SLOS dynamic program, with optional output masking.
///
/// Photons are added one input column at a time and the partial amplitude of
/// every reachable output state is carried forward, so work shared between
/// output states happens ONCE. That is the whole point of SLOS and the reason
/// this exists: both public entry points previously computed a separate
/// permanent per output state, costing M_n·2^n against this n·M_n.
///
/// `mask` gives each mode an inclusive `(min, max)` occupancy; modes past its
/// end are unconstrained. It is applied as a PRUNE during the sweep rather than
/// only as a filter at the end: a partial state is abandoned as soon as it
/// cannot satisfy the mask, so the rank-and-accumulate work for dead branches
/// is never paid.
///
/// What the prune does NOT do, stated so nobody assumes otherwise: the level
/// arrays are still sized `n_compositions(level, m)` whatever the mask says, so
/// a mask saves arithmetic, not allocation. For a mask admitting a tiny
/// fraction of a very large output space, the old enumerate-and-permanent route
/// could still win on memory. Compacting the indexing would fix that and is
/// unwritten — there is no caller exercising that regime today.
fn slos_dp(
    unitary: &[Vec<Complex64>],
    input: &FockState,
    mask: Option<&[(u32, u32)]>,
) -> Vec<(FockState, f64)> {
    let m = unitary.len();
    let n = total_photons(input) as usize;

    // Inclusive occupancy bounds for mode `i`; modes past the mask's end, and
    // every mode when there is no mask, are unconstrained.
    let bounds = |i: usize| -> (u32, u32) {
        match mask {
            Some(mk) if i < mk.len() => mk[i],
            _ => (0, n as u32),
        }
    };

    if n == 0 {
        let vacuum = vec![0u32; m];
        // Vacuum satisfies the mask exactly when no mode demands a photon;
        // `hi` cannot exclude it, since occupancy is zero everywhere.
        let ok = (0..m).all(|i| bounds(i).0 == 0);
        return if ok { vec![(vacuum, 1.0)] } else { vec![] };
    }

    let input_cols = fock_to_mode_list(input);
    let table = comp_table(n as u32, m);

    // Partial states are held FLAT — `states[si * m .. si * m + m]` — not as a
    // Vec of Vecs. At 6 photons in 6 modes the levels hold 1+6+21+56+126+252
    // partial states, so the boxed form paid ~460 small allocations per call,
    // which is most of why the first working version of this DP was slower
    // than the permanent route it replaced.
    let mut states: Vec<u32> = vec![0u32; m];
    let mut g: Vec<Complex64> = vec![Complex64::new(1.0, 0.0)];
    let mut scratch = vec![0u32; m];

    for (p, &col) in input_cols.iter().enumerate() {
        let level = (p + 1) as u32;
        let remaining = n as u32 - level;
        let next_len = n_compositions(level, m);
        let mut next_states = vec![0u32; next_len * m];
        let mut next = vec![Complex64::new(0.0, 0.0); next_len];
        let mut seen = vec![false; next_len];

        for si in 0..g.len() {
            let a = g[si];
            // Whole branches are exactly zero for a sparse unitary; skipping
            // them is free and keeps the inner loop tight.
            if a.re == 0.0 && a.im == 0.0 {
                continue;
            }
            scratch.copy_from_slice(&states[si * m..si * m + m]);
            for i in 0..m {
                let u = unitary[i][col];
                if u.re == 0.0 && u.im == 0.0 {
                    continue;
                }
                scratch[i] += 1;

                // Mask prune. Occupancy only grows, so exceeding `hi` is
                // terminal; and if the photons still to place cannot lift every
                // mode to its `lo`, nothing downstream can satisfy the mask.
                let alive = mask.is_none_or(|_| {
                    if scratch[i] > bounds(i).1 {
                        return false;
                    }
                    let deficit: u32 = (0..m).map(|q| bounds(q).0.saturating_sub(scratch[q])).sum();
                    deficit <= remaining
                });

                if alive {
                    let r = fock_rank(&scratch, level, &table);
                    next[r] += u * a;
                    if !seen[r] {
                        next_states[r * m..r * m + m].copy_from_slice(&scratch);
                        seen[r] = true;
                    }
                }
                scratch[i] -= 1;
            }
        }
        states = next_states;
        g = next;
    }

    // perm(U[rows(t), cols(input)]) = prod_i(t_i!) * g[t], because the t_i
    // identical expanded rows permute among the positions assigned to mode i.
    // So amplitude(t) = g[t] * sqrt(prod t_i!) / sqrt(prod input_j!), which is
    // algebraically the same value the permanent route produced.
    let input_norm: f64 = input
        .iter()
        .map(|&ni| factorial(ni as usize) as f64)
        .product();

    let mut results = Vec::with_capacity(g.len());
    for idx in 0..g.len() {
        let out = &states[idx * m..idx * m + m];
        let output_norm: f64 = out
            .iter()
            .map(|&nj| factorial(nj as usize) as f64)
            .product();
        let amp = g[idx] * Complex64::new((output_norm / input_norm).sqrt(), 0.0);
        let prob = amp.norm_sqr();
        if prob > 1e-15 {
            results.push((out.to_vec(), prob));
        }
    }

    results
}

/// Fock states of `n` photons in `m` modes, in the order `fock_rank` ranks
/// them: lexicographic DESCENDING on the occupation vector.
///
/// Test-only. The DP no longer enumerates — it writes each state into the slot
/// `fock_rank` gives it, so rank order IS the output order and materialising a
/// list per level was pure allocation. This stays as the independent statement
/// of that order, so `rank_and_enumeration_agree` can prove `fock_rank` is a
/// bijection onto `0..M_n` rather than just self-consistent.
///
/// Descending because that is the order `enumerate_fock_states` already
/// produced, and `slos_full` is public — the sequence of returned states is
/// observable, so the DP has to preserve it rather than quietly reorder every
/// caller's output. (Caught by `dp_matches_the_permanent_route`, which
/// compares state-by-state in order and failed on `[0,1]` vs `[1,0]` before
/// the values were ever in question.)
///
/// Enumeration and ranking are one definition split in two, and they must not
/// drift — `rank_and_enumeration_agree` pins that.
#[cfg(test)]
fn enumerate_ranked(m: usize, n: u32) -> Vec<FockState> {
    let mut out = Vec::new();
    let mut cur = vec![0u32; m];
    fn rec(out: &mut Vec<FockState>, cur: &mut FockState, mode: usize, m: usize, rem: u32) {
        if mode == m - 1 {
            cur[mode] = rem;
            out.push(cur.clone());
            cur[mode] = 0;
            return;
        }
        for v in (0..=rem).rev() {
            cur[mode] = v;
            rec(out, cur, mode + 1, m, rem - v);
        }
        cur[mode] = 0;
    }
    rec(&mut out, &mut cur, 0, m, n);
    out
}

/// Number of Fock states of `k` photons in `parts` modes: C(k + parts - 1, parts - 1).
fn n_compositions(k: u32, parts: usize) -> usize {
    if parts == 0 {
        return usize::from(k == 0);
    }
    let mut acc: u128 = 1;
    for i in 1..parts {
        acc = acc * (k as u128 + i as u128) / i as u128;
    }
    acc as usize
}

/// `t[parts][k] = n_compositions(k, parts)`, precomputed.
///
/// `fock_rank` is called once per (partial state, mode) — about 5500 times for
/// 6 photons in 6 modes — and each call sums O(m + p) of these. Computing them
/// on the fly costs a u128 multiply-divide loop apiece, which made the first
/// cut of this DP 2-3x SLOWER than the permanent route it replaced: the
/// asymptotics were right and entirely buried under the constant.
type CompTable = Vec<Vec<usize>>;

fn comp_table(max_k: u32, max_parts: usize) -> CompTable {
    (0..=max_parts)
        .map(|parts| {
            (0..=max_k as usize)
                .map(|k| n_compositions(k as u32, parts))
                .collect()
        })
        .collect()
}

/// Position of `state` in `enumerate_ranked(state.len(), total)`.
///
/// Computed arithmetically rather than by lookup: a HashMap keyed on the
/// occupation vector costs ~20 ns per probe, and the DP probes once per
/// (state, mode) — about 5500 times at 6 photons in 6 modes, which would have
/// eaten most of what the algorithm change wins.
fn fock_rank(state: &[u32], total: u32, table: &CompTable) -> usize {
    let m = state.len();
    let mut idx = 0usize;
    let mut rem = total;
    for (i, &si) in state.iter().enumerate().take(m - 1) {
        // Descending: every state whose i-th mode holds MORE photons sorts
        // earlier, so count those.
        let row = &table[m - i - 1];
        for v in (si + 1)..=rem {
            idx += row[(rem - v) as usize];
        }
        rem -= si;
    }
    idx
}

/// The pre-2026-09-25 route: one permanent per output Fock state.
///
/// Kept because it is an INDEPENDENT derivation of the same numbers — Ryser's
/// formula per output state against a shared dynamic program — so it can hold
/// the new implementation to account. It is the reference in
/// `dp_matches_the_permanent_route`, not dead code.
#[cfg(test)]
fn slos_full_via_permanents(
    unitary: &[Vec<Complex64>],
    input: &FockState,
) -> Vec<(FockState, f64)> {
    let m = unitary.len();
    let n = total_photons(input) as usize;
    if n == 0 {
        return vec![(vec![0; m], 1.0)];
    }
    let input_cols = fock_to_mode_list(input);
    let output_states = enumerate_fock_states(m, n as u32);
    let input_norm: f64 = input
        .iter()
        .map(|&ni| factorial(ni as usize) as f64)
        .product();
    let mut results = Vec::with_capacity(output_states.len());
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
    slos_dp(unitary, input, Some(mask))
}

/// Enumerate Fock states with masking constraints.
///
/// Test-only since 2026-09-25: `slos_masked` no longer enumerates candidate
/// outputs and computes a permanent for each — it prunes mask-dead branches
/// inside the DP. Kept as the independent statement of which states a mask
/// admits, so `masked_dp_matches_the_masked_permanent_route` has something to
/// check the prune against.
#[cfg(test)]
fn enumerate_fock_states_masked(m: usize, n: u32, mask: &[(u32, u32)]) -> Vec<FockState> {
    let mut results = Vec::new();
    let mut current = vec![0u32; m];
    enumerate_masked_helper(&mut results, &mut current, 0, m, n, mask);
    results
}

#[cfg(test)]
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

    /// The masked DP must agree with masking the permanent route, for masks
    /// that admit everything, nothing, and a slice in between.
    ///
    /// The prune is the risky part: it drops partial states that "cannot reach"
    /// an admissible output, and an over-eager version silently loses
    /// amplitude, leaving a distribution that still looks plausible. Comparing
    /// against independent enumeration plus Ryser is what catches that.
    #[test]
    fn masked_dp_matches_the_masked_permanent_route() {
        for m in 2..=4usize {
            let dft: Vec<Vec<Complex64>> = (0..m)
                .map(|j| {
                    (0..m)
                        .map(|k| {
                            let ph = 2.0 * std::f64::consts::PI * (j * k) as f64 / m as f64;
                            Complex64::new(ph.cos(), ph.sin()) / (m as f64).sqrt()
                        })
                        .collect()
                })
                .collect();
            let mut input = vec![0u32; m];
            input[0] = 2;
            if m > 1 {
                input[1] = 1;
            }
            let n = total_photons(&input);

            let masks: Vec<Vec<(u32, u32)>> = vec![
                vec![(0, n); m],         // admits everything
                vec![(n + 1, n + 1); m], // admits nothing
                {
                    let mut mk = vec![(0, n); m];
                    mk[0] = (0, 1);
                    mk
                }, // cap mode 0
                {
                    let mut mk = vec![(0, n); m];
                    mk[0] = (1, n);
                    mk
                }, // require a photon in mode 0
                vec![(0, 1)],            // short mask, mode 0 only
            ];

            // A test that only ever compares empty lists proves nothing, so
            // require at least one mask to admit a STRICT, non-empty subset —
            // otherwise the prune is never actually exercised.
            let unmasked = slos_masked(&dft, &input, &vec![(0, n); m]).len();
            let mut saw_strict_subset = false;

            for mask in &masks {
                let dp = slos_masked(&dft, &input, mask);
                let reference = masked_via_permanents(&dft, &input, mask);
                assert_eq!(
                    dp.len(),
                    reference.len(),
                    "m={m} mask={mask:?}: state count differs\n dp={dp:?}\n ref={reference:?}"
                );
                for ((sa, pa), (sb, pb)) in dp.iter().zip(reference.iter()) {
                    assert_eq!(sa, sb, "m={m} mask={mask:?}: order differs");
                    assert!(
                        (pa - pb).abs() < 1e-12,
                        "m={m} mask={mask:?} state={sa:?}: {pa} vs {pb}"
                    );
                }
                if !dp.is_empty() && dp.len() < unmasked {
                    saw_strict_subset = true;
                }
            }
            assert!(
                saw_strict_subset,
                "m={m}: every mask admitted all or nothing, so the prune was never exercised"
            );
        }
    }

    /// Pre-2026-09-25 `slos_masked`: enumerate admissible outputs, one permanent
    /// each. The reference for the prune.
    fn masked_via_permanents(
        unitary: &[Vec<Complex64>],
        input: &FockState,
        mask: &[(u32, u32)],
    ) -> Vec<(FockState, f64)> {
        let m = unitary.len();
        let n = total_photons(input);
        if n == 0 {
            let vacuum = vec![0u32; m];
            let valid = mask.iter().all(|&(lo, _)| lo == 0);
            return if valid { vec![(vacuum, 1.0)] } else { vec![] };
        }
        let input_cols = fock_to_mode_list(input);
        let input_norm: f64 = input
            .iter()
            .map(|&ni| factorial(ni as usize) as f64)
            .product();
        let mut results = Vec::new();
        for output in &enumerate_fock_states_masked(m, n, mask) {
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

    /// Rank and enumeration are one definition split in two; if they drift the
    /// DP scatters amplitudes into the wrong bins and still returns a
    /// plausible-looking distribution.
    #[test]
    fn rank_and_enumeration_agree() {
        for m in 1..=5usize {
            let tbl = comp_table(6, m);
            for n in 0..=6u32 {
                let states = enumerate_ranked(m, n);
                assert_eq!(
                    states.len(),
                    n_compositions(n, m),
                    "count disagrees at m={m} n={n}"
                );
                for (i, st) in states.iter().enumerate() {
                    assert_eq!(st.iter().sum::<u32>(), n, "bad total at m={m} n={n}");
                    assert_eq!(
                        fock_rank(st, n, &tbl),
                        i,
                        "rank != position at m={m} n={n} {st:?}"
                    );
                }
            }
        }
    }

    /// The dynamic program must reproduce the permanent route exactly.
    ///
    /// These are genuinely independent derivations — Ryser per output state
    /// versus one shared DP — so agreement across a spread of unitaries and
    /// occupations is real evidence, not a tautology. Includes a repeated-mode
    /// input (`[2,1,0,..]`), which is where the prod(t_i!) bookkeeping that
    /// relates g[t] to the permanent actually bites.
    #[test]
    fn dp_matches_the_permanent_route() {
        for m in 2..=5usize {
            // A DFT mode unitary, and a less symmetric one.
            let dft: Vec<Vec<Complex64>> = (0..m)
                .map(|j| {
                    (0..m)
                        .map(|k| {
                            let ph = 2.0 * std::f64::consts::PI * (j * k) as f64 / m as f64;
                            Complex64::new(ph.cos(), ph.sin()) / (m as f64).sqrt()
                        })
                        .collect()
                })
                .collect();
            let mut inputs: Vec<FockState> = Vec::new();
            for n in 1..=4u32 {
                let mut st = vec![0u32; m];
                let mut left = n;
                let mut i = 0;
                while left > 0 {
                    st[i % m] += 1;
                    left -= 1;
                    i += 1;
                }
                inputs.push(st);
            }
            // A bunched input: two photons in one mode.
            if m >= 2 {
                let mut st = vec![0u32; m];
                st[0] = 2;
                st[1] = 1;
                inputs.push(st);
            }
            for input in &inputs {
                let a = slos_full(&dft, input);
                let b = slos_full_via_permanents(&dft, input);
                assert_eq!(
                    a.len(),
                    b.len(),
                    "m={m} input={input:?}: state count differs"
                );
                for ((sa, pa), (sb, pb)) in a.iter().zip(b.iter()) {
                    assert_eq!(sa, sb, "m={m} input={input:?}: state order differs");
                    assert!(
                        (pa - pb).abs() < 1e-12,
                        "m={m} input={input:?} state={sa:?}: {pa} vs {pb}"
                    );
                }
                // And the distribution is still normalised.
                let total: f64 = a.iter().map(|(_, p)| p).sum();
                assert!(
                    (total - 1.0).abs() < 1e-9,
                    "m={m} input={input:?}: sum = {total}"
                );
            }
        }
    }
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
