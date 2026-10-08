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

/// Saturating product of local dimensions — the Hilbert-space size of a
/// sub-chain, which is also the Schmidt-rank ceiling of any cut that leaves
/// that sub-chain on one side.
fn hilbert_dim(dims: &[u32]) -> u128 {
    dims.iter()
        .fold(1u128, |acc, &d| acc.saturating_mul(d as u128))
}

/// Largest bond the cut left of site `i` can carry, for a chain whose sites
/// have local dimensions `dims`.
///
/// `min(∏_{j<i} d_j, ∏_{j>=i} d_j)`, saturating: a cut is bounded by the
/// smaller Hilbert space on either side of it. For qubits that is the old
/// `2^min(i, n−i)`; for a d = 3 register it is `3^min(i, n−i)`, and a guard
/// that kept the `2` would under-charge every qutrit chain by `(3/2)^k`.
fn max_bond_at(i: usize, dims: &[u32]) -> u128 {
    hilbert_dim(&dims[..i]).min(hilbert_dim(&dims[i..]))
}

/// Worst-case bytes the site tensors can occupy at bond ceiling `chi`, for a
/// chain whose sites have local dimensions `dims` (PLAN-QUDIT.md Q3: the MPS
/// core carries a per-site physical dimension, so the guard must too).
///
/// Counts the tensors only. Working space for a two-site split is charged
/// separately by [`split_workspace_bytes_dims`], because it is transient and
/// scales with χ alone rather than with `n`.
pub fn tensor_bytes_dims(dims: &[u32], chi: usize) -> u128 {
    let n = dims.len();
    if n == 0 {
        return 0;
    }
    let chi = chi as u128;
    let mut total: u128 = 0;
    for (i, &d) in dims.iter().enumerate() {
        let left = max_bond_at(i, dims).min(chi);
        let right = max_bond_at(i + 1, dims).min(chi);
        total = total.saturating_add(
            left.saturating_mul(d as u128)
                .saturating_mul(right)
                .saturating_mul(BYTES_PER_AMPLITUDE),
        );
    }
    total
}

/// [`tensor_bytes_dims`] for an all-qubit chain of `n` sites.
pub fn tensor_bytes(n: usize, chi: usize) -> u128 {
    tensor_bytes_dims(&vec![2; n], chi)
}

/// Transient working space for one two-site split at bond ceiling `chi` on a
/// chain whose largest local dimension is `d_max`.
///
/// The split matrix is `(d·χ) × (d·χ)` for a two-site gate on `d`-dimensional
/// sites — `2χ × 2χ` for qubits, `3χ × 3χ` for qutrits — and the one-sided
/// Jacobi SVD keeps a working copy plus the accumulated rotations, so charge
/// three of them. This is the term that makes a large χ expensive even on a
/// short chain.
pub fn split_workspace_bytes_dims(dims: &[u32], chi: usize) -> u128 {
    let d_max = dims.iter().copied().max().unwrap_or(2) as u128;
    let chi = chi as u128;
    let side = chi.saturating_mul(d_max);
    let matrix = side
        .saturating_mul(side)
        .saturating_mul(BYTES_PER_AMPLITUDE);
    matrix.saturating_mul(3)
}

/// [`split_workspace_bytes_dims`] for a qubit chain.
pub fn split_workspace_bytes(chi: usize) -> u128 {
    split_workspace_bytes_dims(&[2], chi)
}

/// Refuse an MPS run this host cannot hold, for a chain whose sites have
/// local dimensions `dims` (one entry per wire, `2` for a qubit).
pub fn check_dims(dims: &[u32], chi: usize) -> Result<()> {
    if std::env::var(OVERSUBSCRIBE_VAR).is_ok_and(|v| v == "1") {
        return Ok(());
    }
    let Some(available) = hostmem::available_bytes() else {
        return Ok(());
    };
    let n = dims.len();
    let d_max = dims.iter().copied().max().unwrap_or(2);
    let need = tensor_bytes_dims(dims, chi).saturating_add(split_workspace_bytes_dims(dims, chi));
    if need > available as u128 {
        return Err(OmegaError::Unsupported(format!(
            "an MPS of {n} sites (local dimension up to {d_max}) at bond \
             dimension {chi} needs up to {} (tensors + one split's working \
             space) but only {} is available on this host — refusing rather \
             than driving it into swap. A site tensor is 16*d*chi^2 bytes \
             (32*chi^2 for a qubit), so halving chi quarters this. Options: \
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

/// [`check_dims`] for an all-qubit chain of `n` sites.
pub fn check(n: usize, chi: usize) -> Result<()> {
    check_dims(&vec![2; n], chi)
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
pub fn dense_is_cheaper_dims(dims: &[u32], chi: usize) -> Option<String> {
    let n = dims.len();
    // `∏ d_i` past `2^63` is unreachable for the dense path anyway — the
    // statevector backend refuses above 64 qubits on the same arithmetic, and
    // a qudit chain hits the same ceiling sooner. No advisory to give.
    let dense = dims
        .iter()
        .try_fold(1u128, |acc, &d| acc.checked_mul(d as u128))
        .filter(|&dense| dense < (1u128 << 63))?;
    // Upper bound on MPS amplitudes: `n` tensors of `chi x d_i x chi`. An
    // upper bound is the right side to err on for an ADVISORY — it is the size
    // the caller has authorised, whether or not the bond grows to meet it.
    let chi2 = (chi as u128).saturating_mul(chi as u128);
    let mps = dims.iter().fold(0u128, |acc, &d| {
        acc.saturating_add(chi2.saturating_mul(d as u128))
    });
    if mps <= dense {
        return None;
    }
    let ratio = mps as f64 / dense as f64;
    let d_max = dims.iter().copied().max().unwrap_or(2);
    let what = if d_max == 2 { "qubits" } else { "sites" };
    Some(format!(
        "mps chi={chi} at {n} {what} allocates up to {mps} amplitudes against \
         the dense statevector's {dense} — {ratio:.1}x LARGER than the state it \
         encodes, so `--backend statevector` will be faster and exact. MPS pays \
         off when the bond stays small; above chi ~= d^(n/2) it cannot. \
         Proceeding anyway (this is advice, not a refusal)."
    ))
}

/// [`dense_is_cheaper_dims`] for an all-qubit chain of `n` sites.
pub fn dense_is_cheaper(n: usize, chi: usize) -> Option<String> {
    dense_is_cheaper_dims(&vec![2; n], chi)
}

/// exponential, and that is a real improvement worth making. Until then the
/// limit is stated rather than discovered by the kernel.
pub fn check_dense_dims(dims: &[u32]) -> Result<()> {
    let n = dims.len();
    // `∏ d_i` must fit the `usize` index space the dense vector is addressed
    // with; for qubits that is the `n < 64` ceiling the statevector backend
    // shares, and a qudit chain reaches it at a smaller `n`.
    let amplitudes = dims
        .iter()
        .try_fold(1u128, |acc, &d| acc.checked_mul(d as u128))
        .filter(|&a| a < (1u128 << MAX_DENSE_QUBITS));
    let Some(amplitudes) = amplitudes else {
        return Err(OmegaError::Unsupported(format!(
            "an MPS analytic run contracts to a dense statevector over {n} \
             sites, and its index space (the product of the local dimensions) \
             does not fit in a 64-bit usize. Use shots (`--shots N`) so the run \
             samples from the tensors instead of materialising them."
        )));
    };
    if std::env::var(OVERSUBSCRIBE_VAR).is_ok_and(|v| v == "1") {
        return Ok(());
    }
    let Some(available) = hostmem::available_bytes() else {
        return Ok(());
    };
    let need = amplitudes.saturating_mul(BYTES_PER_AMPLITUDE);
    if need > available as u128 {
        return Err(OmegaError::Unsupported(format!(
            "an MPS analytic run (no shots) contracts the tensors into a DENSE \
             {n}-site statevector needing {}, but only {} is available on this \
             host — refusing rather than being killed part-way through. The \
             bond dimension does not bound this: the dense form is the full \
             product of the local dimensions (2^n for qubits) regardless of \
             how little entanglement the state has. Use `--shots N` to sample \
             from the tensors instead, which stays proportional to the bond \
             dimension.",
            hostmem::human_bytes(need),
            hostmem::human_bytes(available as u128),
        )));
    }
    Ok(())
}

/// [`check_dense_dims`] for an all-qubit chain of `n` sites.
pub fn check_dense(n: u32) -> Result<()> {
    check_dense_dims(&vec![2; n as usize])
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
        assert_eq!(tensor_bytes_dims(&[], 64), 0);
    }

    /// The qubit wrappers must be the `d = 2` row of the dims arithmetic —
    /// not a separate formula that can drift.
    #[test]
    fn the_qubit_wrappers_are_the_all_two_row() {
        for (n, chi) in [(1usize, 4usize), (10, 1024), (60, 64), (3, 1)] {
            let dims = vec![2u32; n];
            assert_eq!(
                tensor_bytes(n, chi),
                tensor_bytes_dims(&dims, chi),
                "n={n} chi={chi}"
            );
            assert_eq!(
                split_workspace_bytes(chi),
                split_workspace_bytes_dims(&dims, chi)
            );
            assert_eq!(dense_is_cheaper(n, chi), dense_is_cheaper_dims(&dims, chi));
        }
    }

    /// PLAN-QUDIT.md Q3: the bond ceiling at a cut is the smaller Hilbert
    /// space beside it, so a qutrit chain is charged `3^k`, not `2^k`, where
    /// the chain is too short to reach χ. Checked from both sides: the exact
    /// per-site figure, and that a `2^k` guard would have under-charged it.
    #[test]
    fn a_qutrit_chain_is_charged_its_own_schmidt_ceiling() {
        // Three qutrits at χ = 1024: cuts carry 1, 3, 3, 1 (never near χ), so
        // site tensors are 1·3·3, 3·3·3, 3·3·1 amplitudes = 45 × 16 bytes.
        assert_eq!(
            tensor_bytes_dims(&[3, 3, 3], 1024),
            45 * BYTES_PER_AMPLITUDE
        );
        // Three qubits on the same arithmetic: 1·2·2 + 2·2·2 + 2·2·1 = 16.
        assert_eq!(
            tensor_bytes_dims(&[2, 2, 2], 1024),
            16 * BYTES_PER_AMPLITUDE
        );
        // A wide qutrit chain at the χ cap costs 3/2 of the qubit chain: the
        // per-site tensor is χ·d·χ.
        let qubit = tensor_bytes_dims(&vec![2; 60], 64);
        let qutrit = tensor_bytes_dims(&vec![3; 60], 64);
        assert!(
            qutrit > qubit,
            "a qutrit chain must cost more than a qubit chain"
        );
        assert!(
            qutrit < qubit * 2,
            "at the chi cap a qutrit site is 3/2 of a qubit site, not more: {qutrit} vs {qubit}"
        );
        // Mixed radix: the largest local dimension sizes the split workspace.
        assert_eq!(
            split_workspace_bytes_dims(&[2, 5, 3], 8),
            3 * (5 * 8) * (5 * 8) * BYTES_PER_AMPLITUDE
        );
    }

    /// The dense guard must count `∏ d_i`, not `2^n`: 41 qutrits is `3^41`
    /// ≈ 3.6 × 10¹⁹ amplitudes, past the `usize` index space (`2^64` ≈
    /// 1.8 × 10¹⁹) where 41 qubits would have been a mere 32 TiB; 30 qutrits
    /// is `3^30` ≈ 2 × 10¹⁴ × 16 B, which fits `usize` but no host — a `2^30`
    /// guard would have admitted it at 16 GiB.
    ///
    /// The admitted qubit row is derived from what the host actually has, not
    /// pinned at 30 qubits: 2^30 × 16 B is 16 GiB, which a 16 GB laptop can
    /// never have free, and a test that asserts one host's memory is red on
    /// every other. Without a memory probe the guard admits everything below
    /// `usize` by design, so the host rows are a registered skip there.
    #[test]
    fn the_dense_guard_counts_the_product_of_local_dimensions() {
        // Index-space rows: the same on every host.
        let err = check_dense_dims(&[3; 41]).expect_err("3^41 must be refused");
        assert!(format!("{err}").contains("usize"), "{err}");
        // The qubit chain of the same length sits inside the index space:
        // 2^41 is 32 TiB, so a host refuses it, but never for the usize reason.
        if let Err(err) = check_dense(41) {
            let msg = format!("{err}");
            assert!(
                !msg.contains("usize"),
                "2^41 fits usize; refused for the wrong reason: {msg}"
            );
            assert!(msg.contains("available on this host"), "{msg}");
        }

        // Host rows: need the probe.
        let Some(available) = hostmem::available_bytes() else {
            eprintln!(
                "SKIP: no host-memory probe on this platform; the host rows of \
                 the dense guard are not exercised here"
            );
            return;
        };
        let err = check_dense_dims(&[3; 30]).expect_err("3^30 amplitudes (3.3 TB) must be refused");
        assert!(format!("{err}").contains("shots"), "{err}");
        // The largest 2^k that fits half of what is available now: k ≈ 28 on
        // a 16 GB laptop, ≈ 31 on a 123 GB workstation. Half, so the row
        // survives the probe drifting between this line and the guard's own.
        let budget = (available as u128 / BYTES_PER_AMPLITUDE).max(2);
        let k = (budget / 2).ilog2();
        check_dense(k).expect("the qubit row within this host's memory stays admitted");
        // One doubling past the whole budget is refused, and for the host
        // reason, not the index-space one.
        let over = budget.ilog2() + 1;
        let err =
            check_dense(over).expect_err("one doubling past the host's memory must be refused");
        assert!(format!("{err}").contains("available on this host"), "{err}");
    }

    /// The crossover advisory for qutrits sits where `n·d·χ² > d^n`.
    #[test]
    fn the_dense_advisory_uses_the_qudit_hilbert_space() {
        // 10 qutrits: dense = 3^10 = 59049; mps at χ = 8 is 10·3·64 = 1920 (quiet),
        // at χ = 64 it is 10·3·4096 = 122880 (loud).
        assert!(dense_is_cheaper_dims(&[3; 10], 8).is_none());
        let msg = dense_is_cheaper_dims(&[3; 10], 64).expect("must warn");
        assert!(
            msg.contains("59049"),
            "should quote the qutrit dense size: {msg}"
        );
        assert!(
            msg.contains("sites"),
            "should not call qutrits qubits: {msg}"
        );
        // Past the index-space ceiling there is no dense path to point at.
        assert!(dense_is_cheaper_dims(&[3; 41], 1 << 20).is_none());
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
