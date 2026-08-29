// SPDX-License-Identifier: Apache-2.0
//! Refuse a statevector that cannot fit, before allocating it.
//!
//! # What went wrong without this
//!
//! `evolve` and `evolve_once` both opened with
//!
//! ```text
//! let dim = 1usize << n;
//! let mut state = vec![Complex64::new(0.0, 0.0); dim];
//! ```
//!
//! with no check on `n` whatsoever, which fails in two different ways:
//!
//! * **`n >= 64`** — `1usize << n` is an overflowing shift. A plain 64-qubit
//!   GHZ file panicked the process (`attempt to shift left with overflow`).
//!   In release mode the shift wraps instead, which is worse: `n == 64` yields
//!   `dim == 1`, so the run *succeeds* and returns a one-amplitude "statevector"
//!   for a 64-qubit circuit.
//! * **`n` merely large** — 34 qubits is 256 GiB. The allocation is not refused,
//!   it is attempted, and the machine goes to swap or the OOM killer takes the
//!   process (or something else) down. The user's input, not a config knob,
//!   picks that number, and it doubles every qubit.
//!
//! # The rule
//!
//! Two gates, in order:
//!
//! 1. **A hard ceiling at 64 qubits**, independent of the machine. Beyond it the
//!    index space itself stops fitting in `usize`, so no amount of RAM helps and
//!    the refusal is a statement about arithmetic rather than about resources.
//! 2. **A free-memory sanity check** below that. The state is `2^n * 16` bytes.
//!    If the host will say what is available and the requirement exceeds it,
//!    refuse and name both numbers.
//!
//!    Sampling used to double this — the sampler materialised two more `2^n`
//!    f64 vectors — so a sampled run was refused a whole qubit earlier than an
//!    analytic one. It no longer does; the sampler sorts its draws and walks
//!    the state once.
//!
//! When the platform will not report memory, gate 2 is skipped rather than
//! guessed. A fabricated figure would refuse valid work on one machine and admit
//! fatal work on another; the hard ceiling still applies.
//!
//! # The escape hatch
//!
//! `OMEGA_SV_ALLOW_OVERSUBSCRIBE=1` skips gate 2 only. Someone who knows their
//! box better than `vm_stat` does — a big swap file, a container whose limits
//! the probe cannot see — should not be blocked by an estimate. Gate 1 has no
//! override because it is not a resource question.

use omega_core::error::{OmegaError, Result};
use omega_core::hostmem;

/// Bytes per stored amplitude: `Complex64` is two `f64`.
const BYTES_PER_AMPLITUDE: u128 = 16;

/// Beyond this many qubits `1usize << n` no longer fits, on any machine.
pub const MAX_QUBITS: u32 = 64;

/// Env var that skips the memory check (never the hard ceiling).
pub const OVERSUBSCRIBE_VAR: &str = "OMEGA_SV_ALLOW_OVERSUBSCRIBE";

/// Bytes a dense statevector of `n` qubits occupies.
///
/// `u128` so the arithmetic itself cannot overflow while we are in the middle of
/// deciding whether something is too big — computing the size must not be
/// subject to the failure it is checking for.
pub fn required_bytes(n: u32, sampling: bool) -> u128 {
    let amplitudes: u128 = 1u128 << n;
    let state = amplitudes * BYTES_PER_AMPLITUDE;
    // `sampling` no longer changes the footprint.
    //
    // It used to double it: `sample_counts` materialised two `2^n` f64 vectors
    // — the probabilities and their running sum — beside the live state, which
    // took a 28-qubit run from 4099 MiB to 8195 MiB and made a sampled circuit
    // hit this guard a whole qubit earlier than an analytic one. The sampler
    // now sorts its `shots` draws and walks the state once, so its extra
    // memory is kilobytes.
    //
    // The parameter is kept rather than removed: it is part of the question
    // callers are asking ("will this run fit"), and if a future sampler
    // reintroduces a per-amplitude buffer this is where it must be charged.
    let _ = sampling;
    state
}

/// Bytes the adjoint gradient's checkpoint tape occupies.
///
/// The tape is the largest allocation in this workspace, and it is *not* the
/// statevector: `adjoint_gradient` stores one full copy of the state per
/// unitary gate, so the cost is `gates × 2^n × 16` rather than `2^n × 16`. At
/// 16 qubits and ~660 gates that is 8.3 GB — measured — while the state itself
/// is 1 MB. Reading the width alone and concluding "that is small" is exactly
/// how this one gets missed.
///
/// `rows` is the number of parameter bindings evaluated concurrently: the batch
/// path gives every rayon row its own tape, so peak scales with it.
pub fn adjoint_tape_bytes(n: u32, unitary_gates: usize, rows: usize) -> u128 {
    let state = (1u128 << n) * BYTES_PER_AMPLITUDE;
    // +1: the initial checkpoint, pushed before the first gate. +1 again for
    // the live state and lambda, which coexist with the tape.
    let per_row = state * (unitary_gates as u128 + 2);
    per_row * rows.max(1) as u128
}

/// Refuse an adjoint gradient whose tape this host cannot hold.
///
/// Separate from [`check`] because the failure is a different shape: the width
/// can be perfectly ordinary — 16 or 20 qubits — and the *depth* is what
/// exhausts memory. A guard that only looks at `n` waves this straight through.
pub fn check_adjoint(n: u32, unitary_gates: usize, rows: usize) -> Result<()> {
    // The state itself must still be addressable and affordable.
    check(n, false)?;

    if std::env::var(OVERSUBSCRIBE_VAR).is_ok_and(|v| v == "1") {
        return Ok(());
    }
    let Some(available) = hostmem::available_bytes() else {
        return Ok(());
    };
    let need = adjoint_tape_bytes(n, unitary_gates, rows);
    if need > available as u128 {
        return Err(OmegaError::Unsupported(format!(
            "the adjoint gradient of a {n}-qubit circuit with {unitary_gates} \
             unitary gates needs {} of checkpoint tape{} but only {} is \
             available on this host — refusing rather than driving it into \
             swap. The tape stores one full state per gate, so it grows with \
             DEPTH, not just width. Options: fewer gates or qubits, a smaller \
             batch, or set {OVERSUBSCRIBE_VAR}=1 if you know this host can \
             take it.",
            hostmem::human_bytes(need),
            if rows > 1 {
                format!(" across {rows} concurrent parameter rows")
            } else {
                String::new()
            },
            hostmem::human_bytes(available as u128),
        )));
    }
    Ok(())
}

/// Refuse a circuit this host cannot hold, before a single amplitude is
/// allocated.
///
/// `sampling` is retained in the signature so callers keep declaring their
/// intent, but it no longer changes the answer — see [`required_bytes`].
pub fn check(n: u32, sampling: bool) -> Result<()> {
    // Gate 1 — arithmetic, not resources. No override.
    if n >= MAX_QUBITS {
        return Err(OmegaError::Unsupported(format!(
            "a dense statevector of {n} qubits cannot be addressed on this \
             platform: 2^{n} indices do not fit in a 64-bit usize, so the hard \
             ceiling is {MAX_QUBITS} qubits. Memory runs out long before that — \
             40 qubits is already {}. Use a backend whose cost is not \
             exponential in the qubit count: `mps` for low-entanglement \
             circuits, `pauli` for Clifford ones, `pauliprop` for expectation \
             values.",
            hostmem::human_bytes(required_bytes(40, false)),
        )));
    }

    if std::env::var(OVERSUBSCRIBE_VAR).is_ok_and(|v| v == "1") {
        return Ok(());
    }

    // Gate 2 — resources. Skipped, not guessed, when the platform is silent.
    let Some(available) = hostmem::available_bytes() else {
        return Ok(());
    };
    let need = required_bytes(n, sampling);
    if need > available as u128 {
        let what = if sampling {
            "state + sampler working set"
        } else {
            "state"
        };
        return Err(OmegaError::Unsupported(format!(
            "a {n}-qubit statevector needs {} ({what}) but only {} is available \
             on this host — refusing rather than driving it into swap. Options: \
             a backend that is not exponential in the qubit count (`mps`, \
             `pauli`, `pauliprop`), fewer qubits, or set \
             {OVERSUBSCRIBE_VAR}=1 to proceed anyway if you know this host can \
             take it.",
            hostmem::human_bytes(need),
            hostmem::human_bytes(available as u128),
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact input that panicked the process before this guard existed.
    #[test]
    fn sixty_four_qubits_is_refused_not_panicked() {
        let err = check(64, false).expect_err("64 qubits must be refused");
        let msg = format!("{err}");
        assert!(msg.contains("64"), "{msg}");
        assert!(
            msg.contains("usize"),
            "must say why it is arithmetic: {msg}"
        );
    }

    /// In release mode the old shift would WRAP rather than panic, making a
    /// 64-qubit run return a 1-amplitude state. Refusal must not depend on
    /// debug assertions being on.
    #[test]
    fn the_ceiling_holds_far_beyond_the_wrap_point() {
        for n in [64u32, 65, 100, 128, 1024] {
            assert!(check(n, false).is_err(), "{n} qubits must be refused");
        }
    }

    /// Small circuits must stay unaffected — a guard that refuses ordinary work
    /// is worse than the panic it replaces.
    #[test]
    fn ordinary_widths_are_admitted() {
        for n in [0u32, 1, 2, 10, 20] {
            check(n, true).unwrap_or_else(|e| panic!("{n} qubits must be admitted: {e}"));
        }
    }

    /// **Sampling no longer costs extra.** It used to double the footprint,
    /// because the sampler built two `2^n` f64 vectors; it now sorts its draws
    /// and walks the state once. This is the assertion that would have to
    /// change if a per-amplitude buffer ever came back — which is why it is
    /// stated as an equality rather than deleted.
    #[test]
    fn sampling_no_longer_costs_extra_memory() {
        for n in [1u32, 10, 30] {
            assert_eq!(required_bytes(n, true), required_bytes(n, false));
        }
    }

    /// Pinned against hand arithmetic: 2^28 amplitudes x 16 B = 4 GiB, the
    /// figure measured as the evolution RSS of a 28-qubit run. Sampling used to
    /// add another 4 GiB on top; it no longer does.
    #[test]
    fn required_bytes_matches_measured_footprint() {
        assert_eq!(required_bytes(28, false), 4 * (1u128 << 30));
        assert_eq!(required_bytes(28, true), 4 * (1u128 << 30));
        assert_eq!(required_bytes(0, false), 16);
    }

    /// **The tape grows with DEPTH, not width.** This is the case the
    /// statevector guard cannot see: at 16 qubits the state is 1 MiB, so
    /// `check` waves it straight through, while ~660 gates of checkpoints is
    /// 8.3 GB — the figure actually measured before this guard existed.
    #[test]
    fn an_ordinary_width_can_still_have_an_enormous_tape() {
        let state_only = required_bytes(16, false);
        assert!(
            state_only < (2 << 20),
            "a 16-qubit state is ~1 MiB; if this changes the premise below moves"
        );
        // One row of 660 gates is ~692 MB — already 660x the state.
        let one_row = adjoint_tape_bytes(16, 660, 1);
        assert!(
            (600_000_000..800_000_000).contains(&(one_row as u64)),
            "660 gates at 16 qubits should be ~692 MB, got {one_row}"
        );
        // The measured 8.27 GB figure is the BATCH path: the same circuit on 12
        // concurrent rows. The row multiplier, not the width and not even the
        // depth alone, is what actually exhausted the machine.
        let batched = adjoint_tape_bytes(16, 660, 12);
        assert!(
            batched > 8_000_000_000,
            "660 gates x 12 rows should reproduce the measured ~8.3 GB, got {batched}"
        );
        // And the width-only guard is blind to it, which is the whole point.
        assert!(check(16, false).is_ok());
    }

    /// Row concurrency multiplies the tape. Charging for one row would admit a
    /// batch `threads` times too large — the multiplier is invisible in the
    /// circuit, so the guard has to supply it.
    #[test]
    fn concurrent_rows_multiply_the_tape() {
        let one = adjoint_tape_bytes(10, 50, 1);
        assert_eq!(adjoint_tape_bytes(10, 50, 12), one * 12);
        // `rows = 0` must not zero the estimate.
        assert_eq!(adjoint_tape_bytes(10, 50, 0), one);
    }

    /// A gateless circuit still holds the state, the initial checkpoint and
    /// lambda — the estimate must not read zero.
    #[test]
    fn an_empty_circuit_is_not_free() {
        assert!(adjoint_tape_bytes(4, 0, 1) >= required_bytes(4, false));
    }

    /// Ordinary gradients must not be refused.
    #[test]
    fn a_modest_gradient_is_admitted() {
        check_adjoint(10, 40, 1).expect("a 10-qubit, 40-gate gradient must run");
    }

    /// A tape no machine has must be refused, and the message must say that
    /// depth is the cause — otherwise the user shrinks the wrong dimension.
    ///
    /// The width here is deliberately modest (16 qubits, a 1 MiB state) so the
    /// statevector gate passes and the refusal can only be coming from the
    /// tape. Picking a wide circuit would prove nothing: it would trip the
    /// width check first and never reach this code.
    #[test]
    fn an_impossible_tape_is_refused_and_says_why() {
        assert!(
            check(16, false).is_ok(),
            "premise: 16 qubits must pass the width gate"
        );
        let err = check_adjoint(16, 100_000, 12).expect_err("must be refused");
        let msg = format!("{err}");
        assert!(msg.contains("DEPTH"), "must name depth as the cause: {msg}");
        assert!(msg.contains("100000"), "must name the gate count: {msg}");
        assert!(msg.contains("12"), "must name the row concurrency: {msg}");
    }

    /// Computing the size must not itself overflow at the ceiling — the check
    /// has to survive the input it exists to reject.
    #[test]
    fn size_arithmetic_survives_the_ceiling() {
        // 2^63 amplitudes * 16 B = 2^67 bytes. Representable in u128 and
        // nowhere near it in usize — which is the point.
        assert_eq!(required_bytes(MAX_QUBITS - 1, true), 1u128 << 67);
    }
}
