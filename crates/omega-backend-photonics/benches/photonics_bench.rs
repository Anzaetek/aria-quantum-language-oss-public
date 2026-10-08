//! Photonic backend benchmarks: Reck decomposition, permanent, SLOS propagation.

use criterion::{criterion_group, criterion_main, Criterion};
use num_complex::Complex64;

use omega_backend_photonics::decompose::reck_decompose;
use omega_backend_photonics::permanent::permanent;
use omega_backend_photonics::slos::slos_full;

fn random_unitary_6() -> Vec<Vec<Complex64>> {
    // Simple Hadamard-like 6x6 DFT matrix: unitary by construction.
    // U_{jk} = (1/√m) exp(2πi jk / m)
    let m = 6;
    let two_pi = 2.0 * std::f64::consts::PI;
    let scale = 1.0 / (m as f64).sqrt();
    (0..m)
        .map(|j| {
            (0..m)
                .map(|k| {
                    let phase = two_pi * (j * k) as f64 / m as f64;
                    Complex64::new(scale * phase.cos(), scale * phase.sin())
                })
                .collect()
        })
        .collect()
}

fn bench_photonics(c: &mut Criterion) {
    let u = random_unitary_6();

    c.bench_function("photonics_6mode_reck_decompose", |b| {
        b.iter(|| {
            let ops = reck_decompose(&u);
            std::hint::black_box(ops);
        });
    });

    c.bench_function("photonics_6x6_permanent", |b| {
        b.iter(|| {
            let p = permanent(&u);
            std::hint::black_box(p);
        });
    });

    // SLOS propagation of |1,1,1,1,0,0⟩ (n=4 photons in 6 modes).
    let input: Vec<u32> = vec![1, 1, 1, 1, 0, 0];
    c.bench_function("photonics_slos_n4_m6", |b| {
        b.iter(|| {
            let res = slos_full(&u, &input);
            std::hint::black_box(res);
        });
    });

    // Permanent SCALING, added 2026-09-25 for the §5.15 speed comparison
    // against Perceval, and load-bearing for it.
    //
    // Perceval's C++ permanent (`exqalibur.permanent_cx`) carries ~440 µs of
    // FIXED per-call cost: measured flat at 433-453 µs from 2x2 all the way to
    // 16x16, where Ryser is O(n·2ⁿ) and a 16x16 is ~4000x the arithmetic of a
    // 2x2. Its kernel only starts to show from 18x18 (1.33x per size step,
    // then 2.87x and 3.64x). So comparing at the 6x6 above measures their
    // binding against our arithmetic and reports a meaningless ~1000x in our
    // favour. These sizes are where the two kernels can actually be compared.
    // SLOS SCALING in photon number, added 2026-09-25.
    //
    // `slos.rs`'s module doc claims SLOS "computes all output amplitudes in
    // O(n * M_n)" and is "exponentially faster than computing each permanent
    // individually" — but `slos_full` loops over output Fock states calling
    // `permanent()` on each, which IS computing each permanent individually.
    // Naive costs M_n·2^n against real SLOS's n·M_n, so the ratio is 2^n/n and
    // the gap against a real implementation must GROW with photon number. This
    // group is here to show that growth rather than assert it.
    let mut slos_group = c.benchmark_group("photonics_slos_scaling");
    for n_photons in 2usize..=6 {
        let input: Vec<u32> = (0..6).map(|i| if i < n_photons { 1 } else { 0 }).collect();
        slos_group.bench_function(format!("n{n_photons}_m6"), |b| {
            b.iter(|| std::hint::black_box(slos_full(&u, &input)));
        });
    }
    slos_group.finish();

    let mut group = c.benchmark_group("photonics_permanent_scaling");
    group.sample_size(10);
    for n in [12usize, 16, 18, 20, 22] {
        let big: Vec<Vec<Complex64>> = (0..n)
            .map(|j| {
                (0..n)
                    .map(|k| {
                        // Deterministic, non-degenerate, and NOT unitary: a
                        // unitary DFT has a near-zero permanent, which can
                        // flatter an implementation that cancels early.
                        let a = ((j * 7 + k * 13 + 1) as f64).sin();
                        let b = ((j * 11 + k * 5 + 3) as f64).cos();
                        Complex64::new(a, b)
                    })
                    .collect()
            })
            .collect();
        group.bench_function(format!("n{n}"), |b| {
            b.iter(|| std::hint::black_box(permanent(&big)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_photonics);
criterion_main!(benches);
