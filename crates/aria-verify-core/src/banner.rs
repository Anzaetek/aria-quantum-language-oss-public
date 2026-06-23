// SPDX-License-Identifier: Apache-2.0
//! Make it obvious WHAT is computed, and prove quantum == classical numerically.
//!
//! Every harness prints a fixed-shape banner: the example name, a one-line
//! plain-English statement of the computed quantity, the quantum result, the
//! classical-oracle result, and a PASS/FAIL verdict with the max abs diff vs a
//! stated tolerance. The verdict is what CI asserts on.

/// Outcome of one numeric comparison. Fields are retained for diagnostics and
/// potential aggregation even when only `ok()` is consulted.
#[allow(dead_code)]
pub struct Verdict {
    pub name: String,
    pub pass: bool,
    pub max_abs_diff: f64,
    pub tol: f64,
}

impl Verdict {
    pub fn ok(&self) -> bool {
        self.pass
    }
}

fn fmt_vec(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x:+.7}")).collect();
    format!("[{}]", parts.join(", "))
}

/// Print the banner header common to every harness.
pub fn header(name: &str, computes: &str, transport: &str) {
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  example : {name}");
    println!("  COMPUTES: {computes}");
    println!("  transport: {transport}");
    println!("───────────────────────────────────────────────────────────────");
}

/// Compare two equal-length real vectors elementwise; print and return verdict.
pub fn report_values(
    name: &str,
    quantum_label: &str,
    quantum: &[f64],
    classical_label: &str,
    classical: &[f64],
    tol: f64,
) -> Verdict {
    assert_eq!(
        quantum.len(),
        classical.len(),
        "quantum/classical vector length mismatch"
    );
    let max_abs_diff = quantum
        .iter()
        .zip(classical)
        .map(|(q, c)| (q - c).abs())
        .fold(0.0_f64, f64::max);
    let pass = max_abs_diff <= tol;
    println!("  QUANTUM   ({quantum_label}): {}", fmt_vec(quantum));
    println!("  CLASSICAL ({classical_label}): {}", fmt_vec(classical));
    print_verdict(name, max_abs_diff, tol, pass)
}

/// Compare two scalars; print and return verdict.
pub fn report_scalar(
    name: &str,
    quantum_label: &str,
    quantum: f64,
    classical_label: &str,
    classical: f64,
    tol: f64,
) -> Verdict {
    let max_abs_diff = (quantum - classical).abs();
    let pass = max_abs_diff <= tol;
    println!("  QUANTUM   ({quantum_label}): {quantum:+.10}");
    println!("  CLASSICAL ({classical_label}): {classical:+.10}");
    print_verdict(name, max_abs_diff, tol, pass)
}

/// Exact-match verdict for integer/bitstring quantities (tolerance is 0).
pub fn report_exact_u64(
    name: &str,
    quantum_label: &str,
    quantum: u64,
    classical_label: &str,
    classical: u64,
    width: usize,
) -> Verdict {
    let pass = quantum == classical;
    println!(
        "  QUANTUM   ({quantum_label}): {quantum:0width$b} (={quantum})",
        width = width
    );
    println!(
        "  CLASSICAL ({classical_label}): {classical:0width$b} (={classical})",
        width = width
    );
    print_verdict(name, if pass { 0.0 } else { 1.0 }, 0.0, pass)
}

/// Print a 2-class confusion matrix and verdict on accuracy. Rows = actual
/// class (+1, −1), columns = predicted. PASS if accuracy ≥ `min_accuracy`.
/// `[[tp, fn_], [fp, tn]]` where class +1 is "positive".
pub fn report_confusion(
    name: &str,
    quantum_label: &str,
    tp: u32,
    fn_: u32,
    fp: u32,
    tn: u32,
    min_accuracy: f64,
) -> Verdict {
    let total = (tp + fn_ + fp + tn).max(1);
    let accuracy = (tp + tn) as f64 / total as f64;
    println!("  QUANTUM   ({quantum_label}) vs CLASSICAL (ground-truth labels):");
    println!("              pred +1   pred −1");
    println!("  actual +1     {tp:>4}      {fn_:>4}");
    println!("  actual −1     {fp:>4}      {tn:>4}");
    println!("  accuracy = {accuracy:.4} ({}/{} correct)", tp + tn, total);
    let pass = accuracy >= min_accuracy;
    let tag = if pass { "PASS" } else { "FAIL" };
    println!("  {tag} (accuracy ≥ {min_accuracy:.2})");
    println!("═══════════════════════════════════════════════════════════════");
    Verdict {
        name: name.to_string(),
        pass,
        max_abs_diff: 1.0 - accuracy,
        tol: 1.0 - min_accuracy,
    }
}

fn print_verdict(name: &str, max_abs_diff: f64, tol: f64, pass: bool) -> Verdict {
    let tag = if pass { "PASS" } else { "FAIL" };
    println!("  Δmax = {max_abs_diff:.3e}   {tag} (tol {tol:.1e})");
    println!("═══════════════════════════════════════════════════════════════");
    Verdict {
        name: name.to_string(),
        pass,
        max_abs_diff,
        tol,
    }
}
