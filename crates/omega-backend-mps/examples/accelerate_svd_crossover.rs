//! Where Accelerate `zgesdd` starts beating the one-sided Jacobi kernel, by
//! θ-block size — the measurement that sets [`accelerate_svd_flat`]'s gate.
//!
//! `accelerate_svd_share` shows the two ends and not the middle: at depth 4 the
//! blocks are 8 columns wide and LAPACK LOSES (0.76-1.07x on evolution), while
//! at depth >= 12 they reach 64-256 and it wins 4.3-7.6x. A kernel installed
//! without a gate would therefore make shallow circuits slower to make deep
//! ones faster. `omega-backend-mps`'s own Jacobi parallelism gate was set the
//! same way and its comment records the cost of setting it by one doubling too
//! high, so this sweeps the kernel directly rather than inferring a threshold
//! from two circuit rows.
//!
//! Shapes are the ones `Mps::apply_2q` actually produces: `(bl·2) × (2·br)`
//! with bl, br bounded by χ, so both dimensions are even and the matrix is
//! square when the bond is balanced and 2:1 either way while it ramps.

#[cfg(target_os = "macos")]
fn main() {
    use num_complex::Complex64;
    use omega_backend_mps::accelerate;
    use omega_backend_mps::svd::truncated_svd_flat as jacobi;
    use std::time::Instant;

    fn seeded_matrix(m: usize, n: usize, seed: u64) -> Vec<Complex64> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let z = z ^ (z >> 31);
            (z as f64) / (u64::MAX as f64) * 2.0 - 1.0
        };
        (0..m * n).map(|_| Complex64::new(next(), next())).collect()
    }

    /// Minimum over repeats, not mean: this box runs three lanes and the mean
    /// measures the neighbours.
    fn best_us(mut f: impl FnMut(), iters: usize) -> f64 {
        f();
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64() * 1e6);
        }
        best
    }

    println!("rayon threads={}", rayon::current_num_threads());
    println!(
        "\n{:>6} {:>6} {:>7} {:>12} {:>12} {:>10}",
        "rows", "cols", "rank", "jacobi us", "zgesdd us", "speedup"
    );

    for &(rows, cols) in &[
        (4usize, 4usize),
        (8, 8),
        (16, 16),
        (16, 32),
        (32, 16),
        (24, 24),
        (32, 32),
        (48, 48),
        (64, 64),
        (64, 128),
        (128, 64),
        (120, 120),
        (128, 128),
        (256, 256),
    ] {
        let a = seeded_matrix(rows, cols, 0x51D);
        let k = rows.min(cols);
        // The bond cap as the hot path passes it: truncating to half the
        // available rank, which is what a saturated split does.
        let max_rank = (k / 2).max(1);
        let iters = if k >= 128 { 20 } else { 200 };
        let j = best_us(
            || {
                std::hint::black_box(jacobi(&a, rows, cols, cols, max_rank, 1e-14));
            },
            iters,
        );
        let l = best_us(
            || {
                std::hint::black_box(accelerate::truncated_svd_flat(
                    &a, rows, cols, cols, max_rank, 1e-14,
                ));
            },
            iters,
        );
        println!(
            "{:>6} {:>6} {:>7} {:>12.2} {:>12.2} {:>9.2}x",
            rows,
            cols,
            max_rank,
            j,
            l,
            j / l
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("Accelerate is macOS-only.");
}
