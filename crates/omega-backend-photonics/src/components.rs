//! Optical components: phase shifters and beam splitters.
//!
//! Each component maps to a unitary transformation on the modes it acts on.
//! The full circuit unitary is built by composing these components.

use num_complex::Complex64;

/// Build the full m×m unitary matrix from a sequence of optical components.
pub fn build_unitary(num_modes: usize, ops: &[PhotonicOp]) -> Vec<Vec<Complex64>> {
    let mut u = identity(num_modes);

    for op in ops {
        match op {
            PhotonicOp::PhaseShifter { mode, phi } => {
                apply_phase_shifter(&mut u, num_modes, *mode, *phi);
            }
            PhotonicOp::BeamSplitterRx {
                mode0,
                mode1,
                theta,
                phi,
            } => {
                apply_beam_splitter_rx(&mut u, num_modes, *mode0, *mode1, *theta, *phi);
            }
        }
    }

    u
}

/// An optical component operation.
#[derive(Clone, Debug)]
pub enum PhotonicOp {
    /// Phase shifter on a single mode: applies e^{i*phi} to mode k.
    PhaseShifter { mode: usize, phi: f64 },
    /// Beam splitter (Rx convention) on any two modes.
    ///
    /// NOT restricted to adjacent modes — this doc previously said "adjacent",
    /// which is wrong and actively misleading: polarization lowering emits
    /// beam splitters on non-adjacent pairs by construction (a PBS acts on the
    /// H sub-modes `2a` and `2b`). `apply_beam_splitter_rx` operates on
    /// arbitrary row pairs. Note Perceval's `Circuit.add` DOES require
    /// consecutive ports, which is why its bridge routes with PERM.
    /// Matrix: [[cos(theta), -e^{i*phi}*sin(theta)],
    ///          [e^{-i*phi}*sin(theta), cos(theta)]]
    BeamSplitterRx {
        mode0: usize,
        mode1: usize,
        theta: f64,
        phi: f64,
    },
}

/// Create an m×m identity matrix.
pub fn identity(m: usize) -> Vec<Vec<Complex64>> {
    let mut u = vec![vec![Complex64::new(0.0, 0.0); m]; m];
    for i in 0..m {
        u[i][i] = Complex64::new(1.0, 0.0);
    }
    u
}

/// Apply a phase shifter to the unitary: U -> P * U
/// where P is identity except P[mode][mode] = e^{i*phi}.
fn apply_phase_shifter(u: &mut [Vec<Complex64>], _m: usize, mode: usize, phi: f64) {
    let phase = Complex64::new(phi.cos(), phi.sin());
    for val in u[mode].iter_mut() {
        *val *= phase;
    }
}

/// Apply a beam splitter (Rx convention) to the unitary: U -> BS * U
/// BS acts on (mode0, mode1).
fn apply_beam_splitter_rx(
    u: &mut [Vec<Complex64>],
    m: usize,
    mode0: usize,
    mode1: usize,
    theta: f64,
    phi: f64,
) {
    let ct = theta.cos();
    let st = theta.sin();
    let eip = Complex64::new(phi.cos(), phi.sin());
    let eim = Complex64::new(phi.cos(), -phi.sin());

    // BS matrix: [[ct, -eip*st], [eim*st, ct]]
    // Row transformation: new_row0 = ct*row0 - eip*st*row1
    //                     new_row1 = eim*st*row0 + ct*row1
    for j in 0..m {
        let a = u[mode0][j];
        let b = u[mode1][j];
        u[mode0][j] = Complex64::new(ct, 0.0) * a - eip * st * b;
        u[mode1][j] = eim * st * a + Complex64::new(ct, 0.0) * b;
    }
}

/// Multiply two square matrices: C = A * B.
pub fn mat_mul(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let m = a.len();
    let mut c = vec![vec![Complex64::new(0.0, 0.0); m]; m];
    for i in 0..m {
        for k in 0..m {
            let aik = a[i][k];
            if aik.norm_sqr() < 1e-30 {
                continue;
            }
            for j in 0..m {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

/// Extract a submatrix from U given row indices and column indices (with repetitions).
/// For Fock-space evolution: if input state has n_i photons in mode i,
/// column i is repeated n_i times. Similarly for output.
pub fn submatrix_with_repetitions(
    u: &[Vec<Complex64>],
    output_modes: &[usize],
    input_modes: &[usize],
) -> Vec<Vec<Complex64>> {
    let n = output_modes.len();
    let mut sub = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for (i, &row) in output_modes.iter().enumerate() {
        for (j, &col) in input_modes.iter().enumerate() {
            sub[i][j] = u[row][col];
        }
    }
    sub
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_unitary(u: &[Vec<Complex64>], m: usize) -> bool {
        // Check U * U† = I
        let mut uh = vec![vec![Complex64::new(0.0, 0.0); m]; m];
        for i in 0..m {
            for j in 0..m {
                uh[i][j] = u[j][i].conj();
            }
        }
        let prod = mat_mul(u, &uh);
        for i in 0..m {
            for j in 0..m {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (prod[i][j].re - expected).abs() > 1e-10 || prod[i][j].im.abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_identity() {
        let u = identity(4);
        assert!(is_unitary(&u, 4));
    }

    #[test]
    fn test_phase_shifter() {
        let ops = vec![PhotonicOp::PhaseShifter {
            mode: 0,
            phi: std::f64::consts::FRAC_PI_2,
        }];
        let u = build_unitary(2, &ops);
        assert!(is_unitary(&u, 2));
        // Mode 0 should have phase i
        assert!((u[0][0].re).abs() < 1e-10);
        assert!((u[0][0].im - 1.0).abs() < 1e-10);
        // Mode 1 unchanged
        assert!((u[1][1].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_beam_splitter_unitary() {
        let ops = vec![PhotonicOp::BeamSplitterRx {
            mode0: 0,
            mode1: 1,
            theta: std::f64::consts::FRAC_PI_4,
            phi: 0.0,
        }];
        let u = build_unitary(2, &ops);
        assert!(is_unitary(&u, 2));
    }

    #[test]
    fn test_balanced_bs_50_50() {
        // theta = pi/4, phi = 0 -> 50:50 beam splitter
        let ops = vec![PhotonicOp::BeamSplitterRx {
            mode0: 0,
            mode1: 1,
            theta: std::f64::consts::FRAC_PI_4,
            phi: 0.0,
        }];
        let u = build_unitary(2, &ops);
        // |u[0][0]|^2 = |u[0][1]|^2 = 0.5
        assert!((u[0][0].norm_sqr() - 0.5).abs() < 1e-10);
        assert!((u[0][1].norm_sqr() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_6mode_mesh_is_unitary() {
        use std::f64::consts::FRAC_PI_4;
        // Build a 6-mode mesh similar to the spec example
        let ops = vec![
            PhotonicOp::PhaseShifter { mode: 0, phi: 0.3 },
            PhotonicOp::PhaseShifter { mode: 1, phi: 0.7 },
            PhotonicOp::PhaseShifter { mode: 2, phi: 1.1 },
            PhotonicOp::PhaseShifter { mode: 3, phi: 0.2 },
            PhotonicOp::PhaseShifter { mode: 4, phi: 0.9 },
            PhotonicOp::PhaseShifter { mode: 5, phi: 1.5 },
            PhotonicOp::BeamSplitterRx {
                mode0: 0,
                mode1: 1,
                theta: FRAC_PI_4,
                phi: 0.1,
            },
            PhotonicOp::BeamSplitterRx {
                mode0: 2,
                mode1: 3,
                theta: 0.6,
                phi: 0.2,
            },
            PhotonicOp::BeamSplitterRx {
                mode0: 4,
                mode1: 5,
                theta: 0.8,
                phi: 0.3,
            },
            PhotonicOp::BeamSplitterRx {
                mode0: 1,
                mode1: 2,
                theta: 0.5,
                phi: 0.4,
            },
            PhotonicOp::BeamSplitterRx {
                mode0: 3,
                mode1: 4,
                theta: 0.7,
                phi: 0.5,
            },
        ];
        let u = build_unitary(6, &ops);
        assert!(is_unitary(&u, 6));
    }
}
