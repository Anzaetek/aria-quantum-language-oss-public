// SPDX-License-Identifier: Apache-2.0
//! The seam that lets a WASM guest be priced by the server's governor.
//!
//! # The gap this closes, stated precisely
//!
//! The WASM route does **not** bypass the capacity guard — `capacity::check`
//! runs inside the backend on both the `execute` and `expectation` paths, so a
//! guest registering a 40-qubit circuit is refused today. What it bypasses is
//! the governor's **global reservation ledger**: the WASM path takes no
//! reservation, so it does not participate in fair-share accounting with the
//! HTTP routes. Two large jobs, one through each door, can each individually
//! pass the capacity check and collectively oversubscribe the machine.
//!
//! That is **accounting, not memory safety**, and this trait is deliberately
//! scoped to it.
//!
//! # Why a trait here rather than moving the governor
//!
//! `omega-server` owns the `Governor` and already **depends on**
//! `omega-wasm-runtime`, so the dependency points the wrong way and the runtime
//! cannot call the governor directly. The two options were to extract
//! `Governor`/`JobShape`/`CostKind` into a new crate, or to invert the call
//! through a trait both sides can see.
//!
//! This is the second. `omega-core` is already a dependency of both, the trait
//! is three lines, and no admission POLICY moves — the governor keeps owning
//! pricing, pools and the ledger. Moving those types would have been a large
//! mechanical change whose only purpose was to make one call reachable.
//!
//! # Priced at registration, and on the worst case
//!
//! Admission happens when a circuit is REGISTERED, not at the eight simulating
//! host functions. Registration is two sites instead of eight, and it prices
//! before anything is lowered or allocated.
//!
//! The trade is that the guest has not yet said whether it will run analytically
//! or in shot mode, and only the analytic path materialises `2^n` amplitudes. So
//! the price is the DENSE worst case. That errs toward refusing a job that might
//! have been cheap, which is the correct direction for admission — the opposite
//! error admits a job that then cannot fit, which is the oversubscription this
//! exists to prevent.

use std::fmt;

/// Why a circuit was refused admission, in terms a guest can act on.
#[derive(Clone, Debug)]
pub struct Refusal {
    /// Human-readable, and expected to NAME the limit — a guest that is told
    /// only "refused" cannot decide whether to retry smaller or give up.
    pub reason: String,
    /// `true` when waiting could help (the machine is busy), `false` when the
    /// request can never fit. The distinction is the difference between a
    /// retry loop that terminates and one that does not.
    pub retryable: bool,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({})",
            self.reason,
            if self.retryable {
                "retryable"
            } else {
                "permanent"
            }
        )
    }
}

/// Admission control, as seen from a host that is not the server.
///
/// Implemented by `omega-server`'s `Governor`. `None` in `HostState` means no
/// governor is present — the standalone `omega-wasm-cli` case — and the runtime
/// must keep working without one, so the absence is a normal state rather than
/// a misconfiguration.
/// A held admission ticket: the capacity it priced stays CHARGED in the
/// implementor's ledger for exactly as long as the value is alive, and is
/// released when it drops. Opaque to the runtime on purpose — the ledger's
/// currency (permits, MiB, slots) is the governor's business.
pub type RunTicket = Box<dyn std::any::Any + Send>;

pub trait Admission: Send + Sync {
    /// Price a circuit about to be registered. `Ok(())` admits it.
    ///
    /// `num_qubits` is all the caller has at registration time, and the
    /// implementation should price the DENSE worst case for it — see the module
    /// note on why erring toward refusal is the right direction here.
    ///
    /// This is an admissibility QUESTION, not a charge: nothing stays
    /// reserved afterwards. Holding capacity from registration was
    /// considered and refused — a guest may register circuits it never runs,
    /// and a ledger charged for work that never happens is wrong in the
    /// other direction. The charge happens per-execution via
    /// [`Admission::admit_run`].
    fn admit_circuit(&self, num_qubits: u32) -> Result<(), Refusal>;

    /// Admit ONE execution and hold its capacity until the returned ticket
    /// drops. This is the charging half `admit_circuit` deliberately is not:
    /// the caller takes a ticket immediately before backend work and drops
    /// it when the call returns, so the governor's ledger sees WASM-route
    /// work for exactly the interval it occupies the machine — no longer
    /// (the registration-ticket mistake) and no shorter (the gap that let
    /// admission race execution).
    ///
    /// `analytic` says whether this call materialises `2^n` amplitudes.
    /// Today's WASM backend is the dense statevector either way, so an
    /// implementation may price both cases identically; the parameter exists
    /// so a future shot-mode backend that does NOT densify can be priced
    /// honestly without changing this seam again.
    fn admit_run(&self, num_qubits: u32, analytic: bool) -> Result<RunTicket, Refusal>;
}
