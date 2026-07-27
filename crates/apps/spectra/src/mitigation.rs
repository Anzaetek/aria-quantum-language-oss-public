// SPDX-License-Identifier: Apache-2.0
//! Error-mitigation primitives for the noise scans: higher-order zero-noise
//! extrapolation (ZNE) and probabilistic error cancellation (PEC) overhead.
//!
//! These are the honest counterweight to the noise sweeps. `spectra_noise`
//! shows the certified advantage is destroyed by a per-gate rate ~0.02–0.08;
//! the obvious rebuttal is "mitigate it back". This module implements the two
//! standard mitigators exactly enough to show WHY that does not rescue a deep,
//! scrambling circuit cheaply:
//!
//! * **ZNE** amplifies the noise to scales {1,2,…}×base and extrapolates the
//!   observable to zero noise. Two model families are provided — polynomial
//!   (Richardson) and exponential. On EXACT expectations of a smooth decay the
//!   exponential model and higher-order Richardson recover well; the catch is
//!   the Lagrange one-norm ‖w‖₁, which grows fast with the node count and
//!   multiplies any STATISTICAL (finite-shot) error — so the higher-order gain
//!   evaporates once the observable is sampled rather than computed exactly.
//!   The harness reports the exact-case recovery alongside ‖w‖₁ so both facts
//!   are visible.
//!
//! * **PEC** is unbiased — it writes the inverse noise channel as a
//!   quasi-probability mix of implementable operations and reweights samples by
//!   the sign product — but the variance blows up by γ² per shot, where the
//!   one-norm γ multiplies over every noisy gate. So the sampling overhead is
//!   γ_gate^(2·#channels): exponential in the gate count, hence EXTENSIVE in n.
//!   This module computes γ exactly for the depolarizing channel.

/// Lagrange extrapolation of the points `(nodes[i], values[i])` to node 0 — the
/// zero-noise limit when `nodes` are noise scale factors. Exact for any
/// polynomial of degree < nodes.len(). The sum of the returned weights is 1, so
/// it preserves constants; for the classic scales {1,2,3} the weights are the
/// familiar {3, −3, 1}. Returns `None` on fewer than two nodes or a repeated
/// node.
pub fn richardson_extrapolate(nodes: &[f64], values: &[f64]) -> Option<f64> {
    let n = nodes.len();
    if n < 2 || values.len() != n {
        return None;
    }
    let mut acc = 0.0;
    for i in 0..n {
        let mut li = 1.0;
        for j in 0..n {
            if j == i {
                continue;
            }
            let d = nodes[i] - nodes[j];
            if d.abs() < 1e-15 {
                return None; // repeated node
            }
            li *= (0.0 - nodes[j]) / d;
        }
        acc += values[i] * li;
    }
    Some(acc)
}

/// The L1 norm of the Lagrange weights at node 0 — the factor by which ZNE
/// amplifies *statistical* (finite-shot) error in the inputs. Grows with the
/// number of nodes, so high-order Richardson is fine on exact expectations but
/// fragile under sampling. Independent of the data — it quantifies the
/// *method's* conditioning, not its bias.
pub fn extrapolation_norm(nodes: &[f64]) -> Option<f64> {
    let n = nodes.len();
    if n < 2 {
        return None;
    }
    let mut norm = 0.0;
    for i in 0..n {
        let mut li = 1.0;
        for j in 0..n {
            if j == i {
                continue;
            }
            let d = nodes[i] - nodes[j];
            if d.abs() < 1e-15 {
                return None;
            }
            li *= (0.0 - nodes[j]) / d;
        }
        norm += li.abs();
    }
    Some(norm)
}

/// Three-point exponential extrapolation to zero noise: fit y = a + b·e^{−c·x}
/// through the values at scales {1,2,3} and return y(0) = a + b. The standard
/// exponential ZNE model — closer to a decoherence observable than a polynomial,
/// but only valid when the increments decay monotonically with the same sign
/// (else the fitted rate is undefined); returns `None` otherwise.
pub fn exp3_extrapolate(y1: f64, y2: f64, y3: f64) -> Option<f64> {
    let d1 = y2 - y1;
    let d2 = y3 - y2;
    if d1.abs() < 1e-15 {
        return None;
    }
    let r = d2 / d1; // = e^{−c}
                     // Need 0 < r < ∞ and r ≠ 1 for a real, finite decay constant.
    if r <= 0.0 || (r - 1.0).abs() < 1e-9 {
        return None;
    }
    let b = d1 / (r * (r - 1.0));
    let a = y1 - b * r;
    Some(a + b)
}

/// The PEC quasi-probability one-norm γ for a single-qubit depolarizing channel
/// with total error probability `p` (Pauli-twirled form: each of X,Y,Z applied
/// with probability p/3, so in the Pauli picture X,Y,Z are scaled by 1−4p/3).
///
/// The inverse channel is the Pauli channel with weights q_I = (3μ+1)/4,
/// q_{X,Y,Z} = (1−μ)/4 where μ = 1/(1−4p/3); its one-norm is
/// γ = |q_I| + 3|q_P| = (1 + 2p/3)/(1 − 4p/3). γ(0) = 1 and γ grows with p; the
/// sampling overhead to cancel this channel is γ². Panics-free for p < 3/4.
pub fn pec_gamma_depolarizing(p: f64) -> f64 {
    (1.0 + 2.0 * p / 3.0) / (1.0 - 4.0 * p / 3.0)
}

/// log₁₀ of the PEC sampling overhead to cancel `n_channels` independent
/// depolarizing channels each at rate `p`: the shot count for fixed variance
/// scales as γ^{2·n_channels}, so this returns 2·n_channels·log₁₀(γ). Reported
/// in the log domain because the raw overhead overflows f64 for realistic
/// circuits — which is exactly the point.
pub fn pec_log10_sampling_overhead(p: f64, n_channels: usize) -> f64 {
    2.0 * n_channels as f64 * pec_gamma_depolarizing(p).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn richardson_reproduces_the_classic_three_point_weights() {
        // Scales {1,2,3} → weights {3, −3, 1}.
        assert!(
            (richardson_extrapolate(&[1.0, 2.0, 3.0], &[1.0, 0.0, 0.0]).unwrap() - 3.0).abs()
                < 1e-12
        );
        assert!(
            (richardson_extrapolate(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.0]).unwrap() + 3.0).abs()
                < 1e-12
        );
        assert!(
            (richardson_extrapolate(&[1.0, 2.0, 3.0], &[0.0, 0.0, 1.0]).unwrap() - 1.0).abs()
                < 1e-12
        );
    }

    #[test]
    fn richardson_is_exact_for_a_polynomial_of_degree_below_the_node_count() {
        // A quadratic f(x) = 0.3 − 0.2x + 0.05x² sampled at {1,2,3,4} must
        // extrapolate to f(0) = 0.3 exactly (degree 2 < 4 nodes).
        let f = |x: f64| 0.3 - 0.2 * x + 0.05 * x * x;
        let nodes = [1.0, 2.0, 3.0, 4.0];
        let vals: Vec<f64> = nodes.iter().map(|&x| f(x)).collect();
        assert!((richardson_extrapolate(&nodes, &vals).unwrap() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn extrapolation_norm_grows_with_order() {
        // ‖w‖₁ is 1 for exact constants only in the limit; for real node sets it
        // increases with the number of scales — the high-order instability.
        let n3 = extrapolation_norm(&[1.0, 2.0, 3.0]).unwrap();
        let n5 = extrapolation_norm(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(n3 >= 1.0 && n5 > n3, "expected ‖w‖₁ to grow: {n3} → {n5}");
        // 3-point: |3|+|−3|+|1| = 7.
        assert!((n3 - 7.0).abs() < 1e-12);
    }

    #[test]
    fn exp3_recovers_the_zero_noise_value_of_an_exponential() {
        // y = a + b·e^{−c·x}; y(0) = a + b.
        let (a, b, c) = (0.1, 0.5, 0.7);
        let y = |x: f64| a + b * (-c * x).exp();
        let got = exp3_extrapolate(y(1.0), y(2.0), y(3.0)).unwrap();
        assert!(
            (got - (a + b)).abs() < 1e-9,
            "expected {}, got {got}",
            a + b
        );
    }

    #[test]
    fn exp3_refuses_a_non_monotone_triple() {
        // Increments of opposite sign → no real decay constant.
        assert!(exp3_extrapolate(0.5, 0.2, 0.4).is_none());
    }

    #[test]
    fn pec_gamma_is_one_at_zero_and_grows() {
        assert!((pec_gamma_depolarizing(0.0) - 1.0).abs() < 1e-15);
        // Closed form at p = 0.02: (1 + 0.013333)/(1 − 0.026667) = 1.04110…
        assert!((pec_gamma_depolarizing(0.02) - 1.041096).abs() < 1e-5);
        assert!(pec_gamma_depolarizing(0.05) > pec_gamma_depolarizing(0.02));
    }

    #[test]
    fn pec_sampling_overhead_is_extensive_in_channel_count() {
        // Doubling the channels doubles log₁₀(overhead) — exponential in gates.
        let a = pec_log10_sampling_overhead(0.02, 100);
        let b = pec_log10_sampling_overhead(0.02, 200);
        assert!((b - 2.0 * a).abs() < 1e-9);
        assert!(a > 0.0);
    }
}
