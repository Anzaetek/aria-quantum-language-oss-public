// SPDX-License-Identifier: Apache-2.0
//! Reck decomposition against **independently generated** unitaries.
//!
//! ## Why this file exists separately from the in-module tests
//!
//! `decompose.rs`'s own tests build their inputs with `random_unitary`, which
//! composes `build_unitary` over the **same** `BeamSplitterRx` primitive that
//! `reck_decompose` emits and that recomposition then re-applies. A review
//! flagged that as circular — "a convention change cancels on both sides, so
//! the test cannot see it".
//!
//! **Measured, and the claim is weaker than that.** Two convention mutations
//! were applied to `components.rs` and both suites were run:
//!
//! | mutation | in-module reck tests | this file |
//! |---|---|---|
//! | phase-shifter sign flip | 5 failed | 3 failed |
//! | beam-splitter `e^{iφ}`/`e^{-iφ}` swap | 5 failed | 2 failed |
//!
//! The in-module tests **do** catch both. The reason the circularity is not
//! total: `reck_decompose` derives its rotation angles **analytically** from the
//! target matrix rather than by calling `build_unitary`, so the decomposition
//! sits between input-generation and reconstruction and a convention change does
//! not cleanly cancel through it.
//!
//! So this file is not a rescue of a broken suite. It is defence in depth
//! against the mutations that *would* cancel, and it is the only place the
//! decomposition meets matrices it did not construct itself:
//!
//! * the DFT, written from its closed form — the matrix a QFT example depends
//!   on, so it is worth testing directly rather than by proxy;
//! * unitaries from modified Gram–Schmidt on a deterministic PRNG, sharing no
//!   code with the photonic primitives;
//! * permutations, which are all-or-nothing — an amplitude either lands in the
//!   right slot or it does not, so an error cannot hide in a small residual.
//!
//! The honest summary: the existing tests were better than the review credited,
//! and these add independent inputs rather than fixing a hole.

use num_complex::Complex64;
use omega_backend_photonics::components::build_unitary;
use omega_backend_photonics::decompose::reck_decompose;

const TOL: f64 = 1e-12;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// The `m × m` DFT, `U[j][k] = ω^{jk} / √m` with `ω = e^{2πi/m}`.
fn dft(m: usize) -> Vec<Vec<Complex64>> {
    let norm = 1.0 / (m as f64).sqrt();
    (0..m)
        .map(|j| {
            (0..m)
                .map(|k| {
                    let ang = 2.0 * std::f64::consts::PI * (j * k) as f64 / m as f64;
                    c(ang.cos() * norm, ang.sin() * norm)
                })
                .collect()
        })
        .collect()
}

/// A unitary from modified Gram–Schmidt over a deterministic PRNG.
///
/// Shares no code with `components.rs`, which is the whole point.
fn gram_schmidt_unitary(m: usize, seed: u64) -> Vec<Vec<Complex64>> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    };

    let mut cols: Vec<Vec<Complex64>> = (0..m)
        .map(|_| (0..m).map(|_| c(next(), next())).collect())
        .collect();

    for i in 0..m {
        for j in 0..i {
            // proj = <col_j, col_i>
            let mut dot = c(0.0, 0.0);
            for (cj, ci) in cols[j].iter().zip(cols[i].iter()) {
                dot += cj.conj() * ci;
            }
            let sub: Vec<Complex64> = (0..m).map(|k| dot * cols[j][k]).collect();
            for k in 0..m {
                cols[i][k] -= sub[k];
            }
        }
        let n: f64 = cols[i].iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        assert!(n > 1e-9, "degenerate random column; change the seed");
        for v in cols[i].iter_mut() {
            *v /= n;
        }
    }

    // rows[j][i] = cols[i][j]
    (0..m)
        .map(|j| (0..m).map(|i| cols[i][j]).collect())
        .collect()
}

fn max_diff(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| (x - y).norm()))
        .fold(0.0f64, f64::max)
}

fn assert_unitary(u: &[Vec<Complex64>], what: &str) {
    let m = u.len();
    for i in 0..m {
        for j in 0..m {
            let mut dot = c(0.0, 0.0);
            for row in u.iter().take(m) {
                dot += row[i].conj() * row[j];
            }
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - c(want, 0.0)).norm() < 1e-10,
                "{what}: generated matrix is not unitary at ({i},{j}): {dot}"
            );
        }
    }
}

#[test]
fn reck_recomposes_the_dft() {
    for m in 2..=8 {
        let u = dft(m);
        assert_unitary(&u, "dft");
        let ops = reck_decompose(&u);
        let back = build_unitary(m, &ops);
        let d = max_diff(&u, &back);
        assert!(d < TOL, "DFT m={m} recomposition differs by {d:.3e}");
    }
}

#[test]
fn reck_recomposes_independently_generated_unitaries() {
    for m in 2..=8 {
        for seed in [1u64, 42, 9001] {
            let u = gram_schmidt_unitary(m, seed);
            assert_unitary(&u, "gram-schmidt");
            let ops = reck_decompose(&u);
            let back = build_unitary(m, &ops);
            let d = max_diff(&u, &back);
            assert!(d < TOL, "m={m} seed={seed} recomposition differs by {d:.3e}");
        }
    }
}

#[test]
fn reck_recomposes_permutations() {
    // Permutations are all-or-nothing: an amplitude either lands in the right
    // slot or it does not, so a convention error cannot hide in a small residual.
    for m in 2..=6 {
        let mut u = vec![vec![c(0.0, 0.0); m]; m];
        for i in 0..m {
            u[(i + 1) % m][i] = c(1.0, 0.0);
        }
        let ops = reck_decompose(&u);
        let back = build_unitary(m, &ops);
        let d = max_diff(&u, &back);
        assert!(d < TOL, "cyclic permutation m={m} differs by {d:.3e}");
    }
}

/// A 1×1 unitary is a pure phase. `reck_decompose` used to early-return an empty
/// op list for `m <= 1`, silently recomposing `[e^{iφ}]` to the identity.
#[test]
fn reck_keeps_the_phase_of_a_one_by_one_unitary() {
    for phi in [0.0, 0.7, -2.1, std::f64::consts::PI] {
        let u = vec![vec![Complex64::from_polar(1.0, phi)]];
        let ops = reck_decompose(&u);
        let back = build_unitary(1, &ops);
        let d = max_diff(&u, &back);
        assert!(d < TOL, "1x1 phase {phi} was dropped: differs by {d:.3e}");
    }
}
