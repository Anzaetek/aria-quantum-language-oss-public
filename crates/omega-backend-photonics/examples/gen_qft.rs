// SPDX-License-Identifier: Apache-2.0
//! Generate `examples/circuits/qft4.opticqasm` from the Reck decomposition.
//!
//! ```console
//! $ cargo run -p omega-backend-photonics --example gen_qft
//! ```
//!
//! Kept in the tree so the example file is reproducible rather than a set of
//! magic constants nobody can regenerate. It prints the recomposition residual
//! to stderr (2.884e-16 at the time of writing) and the OPTICQASM body to
//! stdout, so a regenerated file can be diffed against the committed one.

fn main() {
    use num_complex::Complex64;
    use omega_backend_photonics::components::{build_unitary, PhotonicOp};
    use omega_backend_photonics::decompose::reck_decompose;
    let m = 4usize;
    let norm = 1.0 / (m as f64).sqrt();
    let u: Vec<Vec<Complex64>> = (0..m).map(|j| (0..m).map(|k| {
        let a = 2.0 * std::f64::consts::PI * (j*k) as f64 / m as f64;
        Complex64::new(a.cos()*norm, a.sin()*norm)
    }).collect()).collect();
    let ops = reck_decompose(&u);
    let back = build_unitary(m, &ops);
    let d: f64 = u.iter().zip(&back).flat_map(|(a,b)| a.iter().zip(b).map(|(x,y)| (x-y).norm())).fold(0.0, f64::max);
    eprintln!("recomposition max diff = {d:.3e}   ops = {}", ops.len());
    for op in &ops {
        match op {
            PhotonicOp::PhaseShifter { mode, phi } => println!("ps({phi:.15}) q[{mode}];"),
            PhotonicOp::BeamSplitterRx { mode0, mode1, theta, phi } =>
                println!("bs_rx({theta:.15}, {phi:.15}) q[{mode0}], q[{mode1}];"),
        }
    }
}
