// SPDX-License-Identifier: Apache-2.0
//! Refuse an MPS whose tensors cannot fit, before allocating them.
//!
//! The fourth and last of the unbounded allocators in this workspace. The other
//! three — the dense statevector, the adjoint checkpoint tape, and the Pauli
//! sum — are bounded in their own crates; this is the MPS one.
//!
//! # What is actually large here
//!
//! A site tensor is `bond_left × 2 × bond_right` complex amplitudes, so at full
//! bond dimension it is `32χ²` bytes — **33 MB per site at χ = 1024**, which is
//! `mps:auto`'s default ceiling. A hundred sites at that bond is 3 GB of
//! tensors before any working space.
//!
//! # Why the bound is not simply `n · 32χ²`
//!
//! Because that would refuse work that cannot possibly reach it. The bond at
//! site `i` can never exceed `2^min(i, n−i)` — the Schmidt rank of a cut that
//! only has that many qubits on one side — regardless of what χ permits. A
//! 10-qubit circuit at χ = 1024 has a true worst case of a few MB, and a guard
//! that charged it `10 × 33 MB` would reject a run that is never going to
//! allocate anything of the sort.
//!
//! A guard that refuses ordinary work is worse than the exhaustion it prevents,
//! so the estimate uses the real per-site cap `min(χ, 2^min(i, n−i))`.
//!
//! # Why this is a refusal and not a truncation
//!
//! MPS already has a truncation story: bond dimension is the knob, and the run
//! reports a certificate saying how much weight it discarded. That is the right
//! answer to "this state is too entangled". It is not an answer to "these
//! tensors do not fit in RAM", because the allocation happens before any
//! truncation decision can be made about it.

use omega_core::error::{OmegaError, Result};
use omega_core::hostmem;

/// `Complex64` — two `f64`.
const BYTES_PER_AMPLITUDE: u128 = 16;

/// Env var that skips the check, mirroring the statevector guard's.
pub const OVERSUBSCRIBE_VAR: &str = "OMEGA_MPS_ALLOW_OVERSUBSCRIBE";

/// Largest bond the cut left of site `i` can carry, for an `n`-site chain.
///
/// `2^min(i, n-i)`, saturating: a cut with `k` qubits on one side has Schmidt
/// rank at most `2^k`, and past 63 the shift would overflow long before any
/// real χ is reached.
fn max_bond_at(i: usize, n: usize) -> u128 {
    let k = i.min(n - i);
    if k >= 63 {
        u128::MAX
    } else {
        1u128 << k
    }
}

/// Worst-case bytes the site tensors can occupy at bond ceiling `chi`.
///
/// Counts the tensors only. Working space for a two-site split is charged
/// separately by [`split_workspace_bytes`], because it is transient and scales
/// with χ alone rather than with `n`.
pub fn tensor_bytes(n: usize, chi: usize) -> u128 {
    if n == 0 {
        return 0;
    }
    let chi = chi as u128;
    let mut total: u128 = 0;
    for i in 0..n {
        let left = max_bond_at(i, n).min(chi);
        let right = max_bond_at(i + 1, n).min(chi);
        total = total.saturating_add(
            left.saturating_mul(2)
                .saturating_mul(right)
                .saturating_mul(BYTES_PER_AMPLITUDE),
        );
    }
    total
}

/// Transient working space for one two-site split at bond ceiling `chi`.
///
/// The split matrix is `2χ × 2χ`, and the one-sided Jacobi SVD keeps a working
/// copy plus the accumulated rotations, so charge three of them. This is the
/// term that makes a large χ expensive even on a short chain.
pub fn split_workspace_bytes(chi: usize) -> u128 {
    let chi = chi as u128;
    let matrix = chi
        .saturating_mul(2)
        .saturating_mul(chi.saturating_mul(2))
        .saturating_mul(BYTES_PER_AMPLITUDE);
    matrix.saturating_mul(3)
}

/// Refuse an MPS run this host cannot hold.
pub fn check(n: usize, chi: usize) -> Result<()> {
    if std::env::var(OVERSUBSCRIBE_VAR).is_ok_and(|v| v == "1") {
        return Ok(());
    }
    let Some(available) = hostmem::available_bytes() else {
        return Ok(());
    };
    let need = tensor_bytes(n, chi).saturating_add(split_workspace_bytes(chi));
    if need > available as u128 {
        return Err(OmegaError::Unsupported(format!(
            "an MPS of {n} sites at bond dimension {chi} needs up to {} \
             (tensors + one split's working space) but only {} is available on \
             this host — refusing rather than driving it into swap. A site \
             tensor is 32*chi^2 bytes, so halving chi quarters this. Options: \
             pin a smaller bond with `--backend mps:<chi>`, let `mps:auto` \
             grow only as far as it needs, or set {OVERSUBSCRIBE_VAR}=1 if you \
             know this host can take it. Note that a smaller chi is an \
             approximation, and the run will report the discarded weight it \
             cost you.",
            hostmem::human_bytes(need),
            hostmem::human_bytes(available as u128),
        )));
    }
    Ok(())
}

/// Hard ceiling for a *dense* materialisation, matching the statevector
/// backend's: past this the index space stops fitting in `usize`.
pub const MAX_DENSE_QUBITS: u32 = 64;

/// Refuse a dense `to_statevector()` this host cannot hold.
///
/// # Why an MPS backend needs a dense guard at all
///
/// Because `execute` materialises one. With `shots: None` — which is what
/// `--statevector` and every `expectation` call use — the MPS is contracted
/// into a full `2^n` vector and the observable is read off that. So the
/// backend chosen specifically to *avoid* an exponential allocation performs
/// one anyway on its analytic path, and the bond dimension does nothing to
/// bound it.
///
/// Measured before this guard existed: `--backend mps --expectation Z0` on a
/// 40-qubit circuit was **killed by the OOM killer (exit 137)**, having asked
/// for 16 TiB. The circuit itself is trivial — one `h` — and its MPS
/// representation is a few kilobytes.
///
/// This does not make the analytic path cheap; it makes it honest. Contracting
/// the observable through the tensor network directly would remove the
/// Is a DENSE statevector strictly cheaper than an MPS at this `(n, chi)`?
///
/// Returns the advisory text when it is, `None` otherwise.
///
/// # Why this exists
///
/// An MPS stores `n` site tensors of at most `chi x 2 x chi` complex amplitudes;
/// a dense statevector stores `2^n`. Above a certain `chi` the MPS is the LARGER
/// object, and every contraction and SVD is then work spent to represent
/// something the dense path holds outright.
///
/// Nothing warned about that, and the cost is not subtle. Measured 2026-08-19 on
/// `examples/circuits/mps_volume_19q.qasm` (19 qubits):
///
/// ```text
///   statevector   0.08 s
///   mps:512     158.09 s      ~2000x, and it SUCCEEDS
/// ```
///
/// `19 * 512^2 * 2 = 9,961,472` amplitudes against `2^19 = 524,288`: the MPS
/// representation is **19x larger than the state it encodes**. See
/// `examples/circuits/MPS-B4-REPRODUCER.md` and `PLAN-CR-20260813.md` B4.
///
/// # An advisory, NOT a refusal
///
/// Deliberate. A large `chi` on a small circuit is a legitimate thing to ask
/// for — testing the MPS path itself, measuring the crossover, or reproducing
/// B4 — and refusing it would break the reproducer committed in this very
/// repository. The caller gets told; the caller decides.
///
/// # Why the guard already present does not cover this
///
/// The truncation certificate refuses when the DISCARDED WEIGHT exceeds its
/// ceiling. That fires when `chi` is too SMALL. This case is the opposite: `chi`
/// is so large that nothing is discarded, the certificate is clean, and the run
/// proceeds at maximum cost. The two guards bracket the useful range from
/// opposite sides, and B4 lived in the gap above the upper one.
pub fn dense_is_cheaper(n: usize, chi: usize) -> Option<String> {
    // `1 << n` overflows past 63 and the dense path is unreachable there
    // anyway — the statevector backend refuses above 64 qubits on the same
    // arithmetic. No advisory to give.
    if n >= 63 {
        return None;
    }
    let dense = 1u128 << n;
    // Upper bound on MPS amplitudes: `n` tensors of `chi x 2 x chi`. An upper
    // bound is the right side to err on for an ADVISORY — it is the size the
    // caller has authorised, whether or not the bond grows to meet it.
    let mps = (n as u128)
        .saturating_mul(chi as u128)
        .saturating_mul(chi as u128)
        .saturating_mul(2);
    if mps <= dense {
        return None;
    }
    let ratio = mps as f64 / dense as f64;
    Some(format!(
        "mps chi={chi} at {n} qubits allocates up to {mps} amplitudes against \
         the dense statevector's {dense} — {ratio:.1}x LARGER than the state it \
         encodes, so `--backend statevector` will be faster and exact. MPS pays \
         off when the bond stays small; above chi ~= 2^(n/2) it cannot. \
         Proceeding anyway (this is advice, not a refusal)."
    ))
}

/// exponential, and that is a real improvement worth making. Until then the
/// limit is stated rather than discovered by the kernel.
pub fn check_dense(n: u32) -> Result<()> {
    if n >= MAX_DENSE_QUBITS {
        return Err(OmegaError::Unsupported(format!(
            "an MPS analytic run contracts to a dense {n}-qubit statevector, \
             and 2^{n} indices do not fit in a 64-bit usize. Use shots \
             (`--shots N`) so the run samples from the tensors instead of \
             materialising them."
        )));
    }
    if std::env::var(OVERSUBSCRIBE_VAR).is_ok_and(|v| v == "1") {
        return Ok(());
    }
    let Some(available) = hostmem::available_bytes() else {
        return Ok(());
    };
    let need = (1u128 << n) * BYTES_PER_AMPLITUDE;
    if need > available as u128 {
        return Err(OmegaError::Unsupported(format!(
            "an MPS analytic run (no shots) contracts the tensors into a DENSE \
             {n}-qubit statevector needing {}, but only {} is available on this \
             host — refusing rather than being killed part-way through. The \
             bond dimension does not bound this: the dense form is 2^n \
             regardless of how little entanglement the state has. Use \
             `--shots N` to sample from the tensors instead, which stays \
             proportional to the bond dimension.",
            hostmem::human_bytes(need),
            hostmem::human_bytes(available as u128),
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The headline figure the CLI help quotes: ~33 MB per site at χ = 1024.
    #[test]
    fn a_full_bond_site_is_thirty_three_megabytes() {
        // A single interior site with both bonds at 1024: 1024*2*1024*16.
        let one_site = 1024u128 * 2 * 1024 * 16;
        assert_eq!(one_site, 33_554_432);
    }

    /// **The per-site cap must bite, or the guard over-refuses.** A 10-qubit
    /// chain at χ = 1024 cannot reach that bond anywhere — the widest cut has
    /// 5 qubits on a side, so 32 is the real ceiling.
    #[test]
    fn a_narrow_chain_is_not_charged_for_an_unreachable_bond() {
        let naive = 10u128 * 1024 * 2 * 1024 * 16; // what a flat n*32chi^2 would say
        let real = tensor_bytes(10, 1024);
        assert!(
            real * 100 < naive,
            "a 10-site chain must cost far less than the flat estimate: \
             real {real}, flat {naive}"
        );
        // And it must be admitted on any ordinary host.
        check(10, 1024).expect("a 10-qubit mps:auto run must not be refused");
    }

    /// The cap must not *under*-charge a chain wide enough to reach χ.
    #[test]
    fn a_wide_chain_is_charged_the_full_bond() {
        // 60 sites, χ = 64: interior cuts far exceed 2^6, so most sites sit at
        // the χ cap of 64 → 64*2*64*16 = 131072 bytes each.
        let bytes = tensor_bytes(60, 64);
        assert!(
            bytes > 40 * 131_072,
            "most of a 60-site chain should be at the chi cap; got {bytes}"
        );
    }

    /// Ordinary runs stay admitted — the failure mode that would matter most.
    #[test]
    fn ordinary_runs_are_admitted() {
        for (n, chi) in [(2, 64), (24, 64), (128, 64), (20, 1024)] {
            check(n, chi).unwrap_or_else(|e| panic!("{n} sites at chi={chi} must run: {e}"));
        }
    }

    /// A genuinely impossible request is refused, and the message says which
    /// knob moves it.
    #[test]
    fn an_impossible_bond_is_refused_and_names_the_knob() {
        let err = check(1000, 1 << 20).expect_err("must be refused");
        let msg = format!("{err}");
        assert!(msg.contains("mps:"), "must name the bond knob: {msg}");
        assert!(
            msg.contains("available"),
            "must say what was available: {msg}"
        );
    }

    /// Size arithmetic must not wrap on absurd input — the check has to survive
    /// the values it exists to reject.
    #[test]
    fn the_estimate_saturates_rather_than_wrapping() {
        let huge = tensor_bytes(1_000, usize::MAX / 4);
        assert!(huge > 0, "estimate wrapped to zero");
        assert!(split_workspace_bytes(usize::MAX / 4) > 0);
    }

    #[test]
    fn an_empty_chain_costs_nothing() {
        assert_eq!(tensor_bytes(0, 64), 0);
    }

    /// **The case that was killed by the OOM killer.** A one-gate 40-qubit
    /// circuit whose MPS is kilobytes, contracted to a 16 TiB dense vector.
    /// Measured exit 137 before this guard.
    #[test]
    fn a_wide_analytic_run_is_refused_rather_than_oom_killed() {
        let err = check_dense(40).expect_err("40 dense qubits must be refused");
        let msg = format!("{err}");
        assert!(
            msg.contains("shots"),
            "the refusal must point at the sampling path, which does not \
             materialise the state; got: {msg}"
        );
    }

    /// Past 64 it is arithmetic, not memory, and must say so.
    #[test]
    fn the_dense_ceiling_is_arithmetic_past_sixty_four() {
        let err = check_dense(64).expect_err("must be refused");
        assert!(format!("{err}").contains("usize"), "{err}");
    }

    /// Ordinary analytic MPS runs must still work — this is the common case.
    #[test]
    fn ordinary_analytic_widths_are_admitted() {
        for n in [1u32, 8, 16, 20] {
            check_dense(n).unwrap_or_else(|e| panic!("{n} qubits analytic must run: {e}"));
        }
    }
}

#[cfg(test)]
mod dense_crossover_tests {
    use super::*;

    /// The advisory fires exactly where MPS stops being able to win, and the
    /// boundary is checked from BOTH sides at each width.
    ///
    /// One-sided checks are how a threshold test passes while sitting in the
    /// wrong place: asserting only that `chi=512` warns at n=19 would pass for
    /// a function that warned unconditionally.
    #[test]
    fn the_advisory_brackets_the_crossover() {
        // (n, chi_that_must_stay_quiet, chi_that_must_warn)
        // Crossover is at n*chi^2*2 == 2^n.
        for (n, quiet, loud) in [(19usize, 64usize, 256usize), (24, 256, 2048), (12, 8, 64)] {
            assert!(
                dense_is_cheaper(n, quiet).is_none(),
                "n={n} chi={quiet}: warned, but the MPS is smaller than the dense state"
            );
            assert!(
                dense_is_cheaper(n, loud).is_some(),
                "n={n} chi={loud}: silent, but the MPS is larger than the dense state"
            );
        }
    }

    /// The B4 case, by its numbers.
    #[test]
    fn the_b4_configuration_is_flagged_with_its_ratio() {
        let msg = dense_is_cheaper(19, 512).expect("mps:512 at 19 qubits must warn");
        assert!(msg.contains("9961472"), "should quote the MPS size: {msg}");
        assert!(msg.contains("524288"), "should quote the dense size: {msg}");
        assert!(msg.contains("19.0x"), "should quote the ratio: {msg}");
        assert!(
            msg.contains("not a refusal"),
            "must say it is advisory: {msg}"
        );
    }

    /// No panic and no advice above the dense path's own ceiling — `1 << n`
    /// would overflow, and the statevector backend refuses there anyway, so
    /// there is no cheaper alternative to point at.
    #[test]
    fn very_wide_circuits_get_no_advice_instead_of_an_overflow() {
        for n in [62usize, 63, 64, 100, 1024] {
            let _ = dense_is_cheaper(n, 1 << 20);
        }
        assert!(dense_is_cheaper(63, 1 << 20).is_none());
        assert!(dense_is_cheaper(1024, 4096).is_none());
    }
}
