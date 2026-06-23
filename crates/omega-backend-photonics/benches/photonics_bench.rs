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
}

criterion_group!(benches, bench_photonics);
criterion_main!(benches);
