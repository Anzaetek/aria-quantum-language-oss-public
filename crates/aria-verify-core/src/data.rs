// SPDX-License-Identifier: Apache-2.0
//! Loaders + tiny numeric helpers for the vendored open datasets in
//! `examples/data/` (see its README for sources and licenses).
//!
//! Pure Rust, no CSV crate: the files are plain comma-separated numerics
//! with a single header row; missing values are `?` (UCI convention).

use std::path::PathBuf;

/// Absolute path to a vendored dataset, e.g. `heart_cleveland.csv`.
pub fn data_path(file: &str) -> PathBuf {
    crate::harness::repo_root().join("examples/data").join(file)
}

/// Load a headered CSV of numerics. `?` and empty cells become `None`.
pub fn load_csv(file: &str) -> Result<Vec<Vec<Option<f64>>>, String> {
    let path = data_path(file);
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 || line.trim().is_empty() {
            continue; // header / trailing blank
        }
        let row: Vec<Option<f64>> = line
            .split(',')
            .map(|cell| {
                let cell = cell.trim();
                if cell.is_empty() || cell == "?" {
                    Ok(None)
                } else {
                    cell.parse::<f64>()
                        .map(Some)
                        .map_err(|_| format!("{file}:{}: bad cell '{cell}'", lineno + 1))
                }
            })
            .collect::<Result<_, String>>()?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("{file}: no data rows"));
    }
    Ok(rows)
}

/// Column-wise z-score standardization over the rows where the column is
/// present. Returns `(means, stds)`; zero-variance columns get std 1 so
/// division is safe.
pub fn column_stats(rows: &[Vec<Option<f64>>]) -> (Vec<f64>, Vec<f64>) {
    let ncols = rows[0].len();
    let mut means = vec![0.0; ncols];
    let mut stds = vec![1.0; ncols];
    for c in 0..ncols {
        let vals: Vec<f64> = rows.iter().filter_map(|r| r[c]).collect();
        if vals.is_empty() {
            continue;
        }
        let m = vals.iter().sum::<f64>() / vals.len() as f64;
        let v = vals.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / vals.len() as f64;
        means[c] = m;
        stds[c] = if v.sqrt() > 1e-12 { v.sqrt() } else { 1.0 };
    }
    (means, stds)
}

/// Deterministic SplitMix64 — the same generator the trainers use, so
/// dataset shuffles/masks are reproducible from a printed seed.
pub struct SplitMix64(pub u64);
impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Ordinary least squares with a small ridge term (λ·I), solved by
/// Gauss–Jordan elimination — the classical baseline for the imputation
/// demo. `x` rows are feature vectors (no intercept column; one is
/// appended internally). Returns the weight vector `[w..., b]`.
pub fn ridge_regression(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Result<Vec<f64>, String> {
    let n = x.len();
    if n == 0 || n != y.len() {
        return Err("ridge: empty or mismatched inputs".into());
    }
    let d = x[0].len() + 1; // + intercept
                            // Normal equations A = XᵀX + λI, b = Xᵀy.
    let mut a = vec![vec![0.0; d]; d];
    let mut b = vec![0.0; d];
    for (row, &yi) in x.iter().zip(y.iter()) {
        let mut xi = row.clone();
        xi.push(1.0);
        for i in 0..d {
            b[i] += xi[i] * yi;
            for j in 0..d {
                a[i][j] += xi[i] * xi[j];
            }
        }
    }
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += lambda;
    }
    // Gauss–Jordan with partial pivoting.
    let mut aug: Vec<Vec<f64>> = a
        .into_iter()
        .zip(b)
        .map(|(mut row, bi)| {
            row.push(bi);
            row
        })
        .collect();
    for col in 0..d {
        let pivot = (col..d)
            .max_by(|&i, &j| {
                aug[i][col]
                    .abs()
                    .partial_cmp(&aug[j][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        if aug[pivot][col].abs() < 1e-12 {
            return Err("ridge: singular normal matrix".into());
        }
        aug.swap(col, pivot);
        let p = aug[col][col];
        for v in aug[col].iter_mut() {
            *v /= p;
        }
        let pivot_row = aug[col].clone();
        for (row, r) in aug.iter_mut().enumerate() {
            if row != col {
                let f = r[col];
                for (v, pv) in r.iter_mut().zip(&pivot_row) {
                    *v -= f * pv;
                }
            }
        }
    }
    Ok(aug.into_iter().map(|row| row[d]).collect())
}

/// Predict with a `ridge_regression` weight vector.
pub fn ridge_predict(w: &[f64], x: &[f64]) -> f64 {
    x.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f64>() + w[w.len() - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_recovers_a_linear_map() {
        // y = 2·x0 − 3·x1 + 0.5, tiny ridge.
        let x: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![(i as f64) / 7.0, ((3 * i % 11) as f64) / 5.0])
            .collect();
        let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] - 3.0 * r[1] + 0.5).collect();
        let w = ridge_regression(&x, &y, 1e-9).unwrap();
        assert!((w[0] - 2.0).abs() < 1e-6, "w0 = {}", w[0]);
        assert!((w[1] + 3.0).abs() < 1e-6, "w1 = {}", w[1]);
        assert!((w[2] - 0.5).abs() < 1e-6, "b = {}", w[2]);
    }

    #[test]
    fn vendored_datasets_load_and_have_expected_shapes() {
        let heart = load_csv("heart_cleveland.csv").unwrap();
        assert_eq!(heart.len(), 303);
        assert_eq!(heart[0].len(), 14);
        // Missing cells exist (ca/thal columns) and parse as None.
        assert!(heart.iter().any(|r| r.iter().any(|c| c.is_none())));

        let train = load_csv("optdigits_train.csv").unwrap();
        let test = load_csv("optdigits_test.csv").unwrap();
        assert_eq!(train[0].len(), 65);
        assert_eq!(test[0].len(), 65);
        assert!(train.len() >= 1000 && test.len() >= 500);
        // Labels are digits 0..9.
        for r in train.iter().chain(test.iter()) {
            let d = r[64].unwrap();
            assert!((0.0..=9.0).contains(&d) && d.fract() == 0.0);
        }
    }
}
