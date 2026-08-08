//! Decomposition of arbitrary unitaries into optical components.
//!
//! Implements the **Reck** decomposition (triangular): `N(N-1)/2` beam splitters
//! plus `N` phase shifters.
//!
//! > Reck, Zeilinger, Bernstein & Bertani, "Experimental realization of any
//! > discrete unitary operator", Phys. Rev. Lett. 73, 58 (1994).
//!
//! **There is deliberately no Clements decomposition here.** This module used to
//! export `clements_decompose`, documented as "the Clements (symmetric) scheme"
//! under a header advertising "same count but better loss tolerance", whose body
//! was `reck_decompose(unitary)`. It had zero callers and advertised a property
//! it did not have, so it is gone rather than left as a trap for the next reader.
//!
//! Implementing Clements properly is worthwhile — it achieves **half the optical
//! depth** and is more loss-tolerant (Clements, Humphreys, Metcalf, Kolthammer &
//! Walmsley, Optica 3(12), 1460-1465 (2016), arXiv:1603.08788) — but it is a
//! real algorithm that deserves its own change and its own verification,
//! including an assertion that its depth is genuinely lower than Reck's. Without
//! that assertion a second silent delegation could reappear and nothing would
//! notice, which is exactly how the first one survived.

use num_complex::Complex64;

use crate::components::PhotonicOp;

/// Decompose an m×m unitary into a sequence of phase shifters and beam splitters
/// using the Reck (triangular) decomposition.
pub fn reck_decompose(unitary: &[Vec<Complex64>]) -> Vec<PhotonicOp> {
    let m = unitary.len();
    if m == 0 {
        return vec![];
    }
    if m == 1 {
        // A 1x1 unitary is a pure phase [e^{i*phi}]. Returning an empty op list
        // here used to silently recompose it to the identity, dropping the
        // phase. There are no beam splitters to apply, but the phase shifter is
        // still real.
        let (re, im) = (unitary[0][0].re, unitary[0][0].im);
        let phi = im.atan2(re);
        return if phi.abs() < 1e-15 {
            vec![]
        } else {
            vec![PhotonicOp::PhaseShifter { mode: 0, phi }]
        };
    }

    let mut u: Vec<Vec<Complex64>> = unitary.to_vec();
    // Store the inverse transformations directly as PhotonicOps
    let mut inverse_ops: Vec<PhotonicOp> = Vec::new();

    // Zero below-diagonal elements column by column, LEFT to RIGHT.
    // Within each column, zero from BOTTOM to TOP using adjacent-row rotations.
    // This order ensures that later rotations don't disturb previously zeroed entries:
    // - Zeroing u[i][j] via rows (i-1, i) only affects rows i-1 and i
    // - Elements in column j' < j at row i were already zeroed
    // - Since rows (i-1, i) rotation preserves zeros in columns < j
    //   (because u[i][j']=0 and u[i-1][j']=0 for j'<j when processed bottom-up)
    for j in 0..m {
        for i in ((j + 1)..m).rev() {
            let a = u[i - 1][j];
            let b = u[i][j];

            if b.norm() < 1e-15 {
                continue;
            }

            // Find (theta, phi) such that BS_rx(theta, phi) applied to rows (i-1, i)
            // zeros u[i][j].
            //
            // BS_rx: [[cos(t), -e^{ip}*sin(t)], [e^{-ip}*sin(t), cos(t)]]
            //
            // new_u[i][j] = e^{-ip}*sin(t)*a + cos(t)*b = 0
            // => e^{-ip}*a/b = -cos(t)/sin(t) = -cot(t)
            //
            // For real t: e^{-ip}*a/b must be real (and negative for t in (0, pi/2))
            // => phi = arg(a) - arg(b)
            // => e^{-ip}*a/b = |a/b| (real positive)
            // => cot(t) = |a|/|b|
            // => t = atan2(|b|, |a|)
            // theta is negative to satisfy the zeroing equation
            let theta = -(b.norm().atan2(a.norm()));
            let phi = if a.norm() < 1e-15 {
                -b.arg()
            } else {
                a.arg() - b.arg()
            };

            // Apply BS_rx(theta, phi) to rows (i-1, i) of u
            let ct = theta.cos();
            let st = theta.sin();
            let eip = Complex64::new(phi.cos(), phi.sin());
            let eim = eip.conj();

            for k in 0..m {
                let ai = u[i - 1][k];
                let bi = u[i][k];
                u[i - 1][k] = Complex64::new(ct, 0.0) * ai - eip * st * bi;
                u[i][k] = eim * st * ai + Complex64::new(ct, 0.0) * bi;
            }

            debug_assert!(
                u[i][j].norm() < 1e-10,
                "failed to zero u[{}][{}]: {}",
                i,
                j,
                u[i][j]
            );

            // The inverse of BS_rx(t, p) is BS_rx(t, p)^†.
            // BS_rx(t,p)^† = [[cos(t), e^{ip}*sin(t)], [-e^{-ip}*sin(t), cos(t)]]
            //
            // We need to express this as a sequence of our primitive ops.
            // Note: BS_rx(-t, p) = [[cos(t), e^{ip}*sin(t)], [-e^{-ip}*sin(t), cos(t)]]
            // which is exactly BS_rx(t,p)^†!
            inverse_ops.push(PhotonicOp::BeamSplitterRx {
                mode0: i - 1,
                mode1: i,
                theta: -theta,
                phi,
            });
        }
    }

    // Now: BS_k * ... * BS_1 * U = D (upper triangular, actually diagonal since U is unitary)
    // Therefore: U = BS_1^{-1} * ... * BS_k^{-1} * D
    //
    // build_unitary applies ops as left-multiplications:
    // build_unitary([op1, op2, ..., opN]) = opN * ... * op2 * op1
    //
    // We need the ops list [D, BS_k^{-1}, ..., BS_1^{-1}]
    // inverse_ops currently stores [BS_1^{-1}, BS_2^{-1}, ..., BS_k^{-1}]

    let mut ops: Vec<PhotonicOp> = Vec::new();

    // Diagonal phases first
    for i in 0..m {
        let phase = u[i][i].arg();
        if phase.abs() > 1e-12 {
            ops.push(PhotonicOp::PhaseShifter {
                mode: i,
                phi: phase,
            });
        }
    }

    // Inverse BS ops in reverse order
    for op in inverse_ops.into_iter().rev() {
        ops.push(op);
    }

    ops
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{build_unitary, identity, mat_mul};

    fn random_unitary(m: usize, seed: u64) -> Vec<Vec<Complex64>> {
        let mut u = identity(m);
        let mut s = seed;

        for _ in 0..(m * m * 2) {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let theta = ((s >> 32) as f64 / u32::MAX as f64) * std::f64::consts::PI / 2.0;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let phi = ((s >> 32) as f64 / u32::MAX as f64) * 2.0 * std::f64::consts::PI;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mode = ((s >> 32) as usize) % m.max(1);

            if mode + 1 < m {
                let ops = vec![PhotonicOp::BeamSplitterRx {
                    mode0: mode,
                    mode1: mode + 1,
                    theta,
                    phi,
                }];
                let bs = build_unitary(m, &ops);
                u = mat_mul(&bs, &u);
            }
        }

        for i in 0..m {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let phase = ((s >> 32) as f64 / u32::MAX as f64) * 2.0 * std::f64::consts::PI;
            let p = Complex64::new(phase.cos(), phase.sin());
            for j in 0..m {
                u[i][j] *= p;
            }
        }

        u
    }

    fn max_error(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> f64 {
        let m = a.len();
        let mut max_err = 0.0f64;
        for i in 0..m {
            for j in 0..m {
                let err = (a[i][j] - b[i][j]).norm();
                max_err = max_err.max(err);
            }
        }
        max_err
    }

    #[test]
    fn test_bs_inverse() {
        // Verify that BS_rx(-t, p) is the inverse of BS_rx(t, p)
        let t = 0.7;
        let p = 1.3;
        let fwd = build_unitary(
            2,
            &[PhotonicOp::BeamSplitterRx {
                mode0: 0,
                mode1: 1,
                theta: t,
                phi: p,
            }],
        );
        let inv = build_unitary(
            2,
            &[PhotonicOp::BeamSplitterRx {
                mode0: 0,
                mode1: 1,
                theta: -t,
                phi: p,
            }],
        );
        let prod = mat_mul(&fwd, &inv);
        let err = max_error(&prod, &identity(2));
        assert!(err < 1e-10, "BS * BS_inv error: {}", err);
    }

    #[test]
    fn test_reck_2x2() {
        let u = random_unitary(2, 42);
        let ops = reck_decompose(&u);
        let reconstructed = build_unitary(2, &ops);
        let err = max_error(&u, &reconstructed);
        assert!(err < 1e-10, "Reck 2x2 error: {}", err);
    }

    #[test]
    fn test_reck_3x3_simple() {
        // Build a known 3x3 unitary from simple BS ops
        let ops = vec![
            PhotonicOp::BeamSplitterRx {
                mode0: 0,
                mode1: 1,
                theta: 0.5,
                phi: 0.3,
            },
            PhotonicOp::BeamSplitterRx {
                mode0: 1,
                mode1: 2,
                theta: 0.7,
                phi: 0.1,
            },
            PhotonicOp::BeamSplitterRx {
                mode0: 0,
                mode1: 1,
                theta: 0.2,
                phi: 0.8,
            },
        ];
        let u = build_unitary(3, &ops);
        let decomposed = reck_decompose(&u);
        let recon = build_unitary(3, &decomposed);
        let err = max_error(&u, &recon);
        assert!(err < 1e-8, "Reck 3x3 simple error: {}", err);
    }

    #[test]
    fn test_reck_3x3() {
        let u = random_unitary(3, 123);
        let ops = reck_decompose(&u);
        let reconstructed = build_unitary(3, &ops);
        let err = max_error(&u, &reconstructed);
        assert!(err < 1e-8, "Reck 3x3 error: {}", err);
    }

    #[test]
    fn test_reck_4x4() {
        let u = random_unitary(4, 999);
        let ops = reck_decompose(&u);
        let reconstructed = build_unitary(4, &ops);
        let err = max_error(&u, &reconstructed);
        assert!(err < 1e-8, "Reck 4x4 error: {}", err);
    }

    #[test]
    fn test_reck_5x5() {
        let u = random_unitary(5, 314);
        let ops = reck_decompose(&u);
        let reconstructed = build_unitary(5, &ops);
        let err = max_error(&u, &reconstructed);
        assert!(err < 1e-6, "Reck 5x5 error: {}", err);
    }

    #[test]
    fn test_reck_identity() {
        let u = identity(3);
        let ops = reck_decompose(&u);
        let reconstructed = build_unitary(3, &ops);
        let err = max_error(&u, &reconstructed);
        assert!(err < 1e-10, "Reck identity error: {}", err);
    }

    #[test]
    fn test_reck_gate_count() {
        for m in 2..=5 {
            let u = random_unitary(m, m as u64 * 7);
            let ops = reck_decompose(&u);
            let num_bs = ops
                .iter()
                .filter(|op| matches!(op, PhotonicOp::BeamSplitterRx { .. }))
                .count();
            let num_ps = ops
                .iter()
                .filter(|op| matches!(op, PhotonicOp::PhaseShifter { .. }))
                .count();
            assert!(
                num_bs <= m * (m - 1) / 2,
                "m={}: {} BS (max {})",
                m,
                num_bs,
                m * (m - 1) / 2
            );
            assert!(num_ps <= m, "m={}: {} PS (max {})", m, num_ps, m);
        }
    }

    #[test]
    fn test_reck_roundtrip_preserves_photon_statistics() {
        use crate::slos;
        let u_orig = random_unitary(3, 777);
        let ops = reck_decompose(&u_orig);
        let u_recon = build_unitary(3, &ops);

        let input = vec![1, 1, 0];
        let dist_orig = slos::slos_full(&u_orig, &input);
        let dist_recon = slos::slos_full(&u_recon, &input);

        for (state, p_orig) in &dist_orig {
            let p_recon = dist_recon
                .iter()
                .find(|(s, _)| s == state)
                .map(|(_, p)| *p)
                .unwrap_or(0.0);
            assert!(
                (p_orig - p_recon).abs() < 1e-8,
                "state {:?}: orig={}, recon={}",
                state,
                p_orig,
                p_recon
            );
        }
    }
}
