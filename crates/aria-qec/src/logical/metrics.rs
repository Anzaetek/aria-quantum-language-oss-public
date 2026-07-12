//! Numeric metrics for the logical layer: logical-error curves + (in later
//! phases) algorithm-fidelity measures.
//!
//! Every acceptance check in this module is a single number compared to a golden
//! within a stated tolerance, per the repo's numeric-only testing discipline.

/// A logical-error-rate curve: `(physical_rate, distance, logical_rate)` points.
#[derive(Clone, Debug, Default)]
pub struct LogicalErrorCurve {
    pub points: Vec<(f64, usize, f64)>,
}

impl LogicalErrorCurve {
    pub fn push(&mut self, p: f64, d: usize, logical_rate: f64) {
        self.points.push((p, d, logical_rate));
    }

    /// Points for a fixed distance, sorted by physical rate.
    pub fn for_distance(&self, d: usize) -> Vec<(f64, f64)> {
        let mut pts: Vec<(f64, f64)> = self
            .points
            .iter()
            .filter(|&&(_, dd, _)| dd == d)
            .map(|&(p, _, pl)| (p, pl))
            .collect();
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        pts
    }
}

/// Least-squares slope of `log(pL)` vs `log(p)` for a fixed distance — the
/// sub-threshold scaling exponent (expected ≈ (d+1)/2). Returns `None` if there
/// are fewer than two usable (strictly positive) points.
pub fn sub_threshold_slope(curve: &LogicalErrorCurve, d: usize) -> Option<f64> {
    let pts: Vec<(f64, f64)> = curve
        .for_distance(d)
        .into_iter()
        .filter(|&(p, pl)| p > 0.0 && pl > 0.0)
        .map(|(p, pl)| (p.ln(), pl.ln()))
        .collect();
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let sx: f64 = pts.iter().map(|&(x, _)| x).sum();
    let sy: f64 = pts.iter().map(|&(_, y)| y).sum();
    let sxx: f64 = pts.iter().map(|&(x, _)| x * x).sum();
    let sxy: f64 = pts.iter().map(|&(x, y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-18 {
        return None;
    }
    Some((n * sxy - sx * sy) / denom)
}

/// Total variation distance ½Σ|p−q| between two outcome distributions keyed by
/// integer bitstring. Missing keys are treated as zero probability.
pub fn tvd(
    a: &std::collections::HashMap<u64, f64>,
    b: &std::collections::HashMap<u64, f64>,
) -> f64 {
    let mut keys: std::collections::BTreeSet<u64> = a.keys().copied().collect();
    keys.extend(b.keys().copied());
    let mut s = 0.0;
    for k in keys {
        let pa = a.get(&k).copied().unwrap_or(0.0);
        let pb = b.get(&k).copied().unwrap_or(0.0);
        s += (pa - pb).abs();
    }
    0.5 * s
}

/// Grover success probability: the mass on the marked bitstring.
pub fn grover_success_prob(dist: &std::collections::HashMap<u64, f64>, marked: u64) -> f64 {
    dist.get(&marked).copied().unwrap_or(0.0)
}

/// QPE phase error: circular distance between the most-likely `n_counting`-bit
/// estimate `argmax/2^n` and the true phase (both in [0,1)).
pub fn qpe_phase_error(
    dist: &std::collections::HashMap<u64, f64>,
    n_counting: usize,
    true_phase: f64,
) -> f64 {
    let best = dist
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(&k, _)| k)
        .unwrap_or(0);
    let est = best as f64 / (1u64 << n_counting) as f64;
    let d = (est - true_phase).abs().fract();
    d.min(1.0 - d)
}

/// Pass a distribution over `n` bits through an independent per-bit classical
/// bit-flip channel of probability `b` — the readout action of an effective
/// logical bit-flip (X/Y) error. Exact convolution (no sampling).
pub fn readout_bitflip(
    dist: &std::collections::HashMap<u64, f64>,
    n: usize,
    b: f64,
) -> std::collections::HashMap<u64, f64> {
    let mut out: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    for (&x, &px) in dist {
        // Distribute px over all y by flipping each bit w.p. b.
        for y in 0u64..(1u64 << n) {
            let mut w = px;
            for i in 0..n {
                let xi = (x >> i) & 1;
                let yi = (y >> i) & 1;
                w *= if xi == yi { 1.0 - b } else { b };
            }
            if w > 0.0 {
                *out.entry(y).or_insert(0.0) += w;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn slope_of_power_law_is_the_exponent() {
        // pL = c * p^2 ⇒ log-log slope exactly 2.
        let mut curve = LogicalErrorCurve::default();
        for &p in &[0.01_f64, 0.02, 0.04, 0.08] {
            curve.push(p, 3, 5.0 * p * p);
        }
        let slope = sub_threshold_slope(&curve, 3).unwrap();
        assert!((slope - 2.0).abs() < 1e-9, "slope = {slope}");
    }

    #[test]
    fn tvd_bounds() {
        let mut a = HashMap::new();
        a.insert(0u64, 0.5);
        a.insert(1u64, 0.5);
        let mut b = HashMap::new();
        b.insert(0u64, 1.0);
        assert!((tvd(&a, &a)).abs() < 1e-12);
        assert!((tvd(&a, &b) - 0.5).abs() < 1e-12);
    }
}
