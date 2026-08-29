// SPDX-License-Identifier: Apache-2.0
//! The seam into omega: [`HostGate`] as an [`Admission`] implementation.
//!
//! `omega-core`'s trait was written so a WASM guest could be priced by the
//! server's governor without the runtime depending on the server. It turns out
//! to be exactly the right shape for the opposite problem too — pricing a
//! *host*, across processes the server cannot see — and the fit is not a
//! coincidence: both are "somebody who is not the governor needs to ask the
//! governor's question".
//!
//! Two details of that trait make this clean rather than approximate.
//!
//! **`RunTicket` is opaque and its lifetime is the charge.** The doc is explicit
//! that "the capacity it priced stays CHARGED in the implementor's ledger for
//! exactly as long as the value is alive". A [`Grant`] is precisely that value,
//! so the ticket is the grant, boxed. Nothing has to remember to release: the
//! caller drops the ticket when the backend call returns, and the ledger sees
//! the work for exactly the interval it occupied the machine.
//!
//! **`admit_circuit` asks, `admit_run` charges.** That distinction already
//! exists upstream for a good reason — a guest may register circuits it never
//! runs, and a ledger charged for work that never happens is wrong in the other
//! direction. It maps onto this crate without strain: the question is answered
//! against the cap alone and touches no file, while the charge takes the lock.
//!
//! # What is priced
//!
//! The dense worst case, `2^n × 16` bytes, on both paths. That is the same
//! formula the in-process governor uses, and using a different one here would
//! mean two components disagreeing about what the same circuit costs — which is
//! worse than either number being slightly wrong.
//!
//! `analytic` is accepted and deliberately ignored, exactly as the trait's own
//! documentation anticipates: today's backends densify either way, and the
//! parameter exists so a future shot-mode backend can be priced honestly without
//! changing the seam again. Pricing it lower *now* would be pricing a promise.
//!
//! # This does not replace the in-process governor
//!
//! It sits beside it. The governor answers "will this job kill the box?" against
//! what this process is doing; this answers it against what the whole machine is
//! doing. Both must say yes, and the cheap local one should be asked first — a
//! refusal that needs no lock is better than one that does.

use omega_core::admission::{Admission, Refusal as CoreRefusal, RunTicket};

use crate::{Axis, HostGate, Refusal, Request};

/// Bytes a dense statevector of `n` qubits occupies: `2^n` amplitudes at 16
/// bytes each. Saturating, because a width that overflows a `u64` is refused on
/// the strength of being enormous rather than by wrapping to something small —
/// wrapping here would admit the largest possible job.
pub fn dense_bytes(num_qubits: u32) -> u64 {
    if num_qubits >= 60 {
        return u64::MAX;
    }
    (1u64 << num_qubits).saturating_mul(16)
}

impl From<Refusal> for CoreRefusal {
    fn from(r: Refusal) -> Self {
        CoreRefusal {
            retryable: r.retryable(),
            // The trait asks that the reason NAME the limit, because a guest
            // told only "refused" cannot decide whether to retry smaller or give
            // up. Our Display already does that, with the numbers.
            reason: format!("host budget: {r}"),
        }
    }
}

impl Admission for HostGate {
    fn admit_circuit(&self, num_qubits: u32) -> Result<(), CoreRefusal> {
        let need = dense_bytes(num_qubits);
        // An admissibility QUESTION: answered against the cap, charging nothing
        // and taking no lock. Whether it would fit *right now* is deliberately
        // not asked — that is `admit_run`'s job, and answering it here would
        // make registration fail for a reason that has nothing to do with the
        // circuit.
        // An unreadable ledger propagates as a refusal rather than an empty
        // iteration. This used to fail OPEN — see `HostGate::capacity`.
        for report in self.capacity().map_err(CoreRefusal::from)? {
            if report.axis == Axis::HostBytes && need > report.cap {
                return Err(Refusal::TooLarge {
                    axis: Axis::HostBytes,
                    required: need,
                    cap: report.cap,
                }
                .into());
            }
        }
        Ok(())
    }

    fn admit_run(&self, num_qubits: u32, _analytic: bool) -> Result<RunTicket, CoreRefusal> {
        let need = dense_bytes(num_qubits);
        let req = Request::new(format!("omega:{num_qubits}q")).want(Axis::HostBytes, need);
        let grant = self.try_acquire(&req)?;
        // The ticket IS the grant. Dropping it releases; there is no second
        // bookkeeping path that could get out of step with this one.
        Ok(Box::new(grant))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Amounts, Mode};
    use std::sync::atomic::{AtomicU32, Ordering};

    static N: AtomicU32 = AtomicU32::new(0);

    fn gate(host_bytes: u64) -> HostGate {
        let d = std::env::temp_dir().join(format!(
            "omega-hostgate-adm-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&d).unwrap();
        let mut caps = Amounts::new();
        caps.insert(Axis::HostBytes, host_bytes);
        HostGate::new(d.join("gate.json"), "t", caps, Mode::Enforce)
    }

    #[test]
    fn a_statevector_is_priced_at_sixteen_bytes_an_amplitude() {
        assert_eq!(dense_bytes(0), 16);
        assert_eq!(dense_bytes(10), 1024 * 16);
        assert_eq!(dense_bytes(30), (1u64 << 30) * 16);
    }

    #[test]
    fn an_absurd_width_saturates_rather_than_wrapping_to_something_small() {
        // Wrapping here would admit the largest possible job, which is the one
        // direction this must never fail in.
        assert_eq!(dense_bytes(64), u64::MAX);
        assert_eq!(dense_bytes(60), u64::MAX);
        assert!(dense_bytes(59) > dense_bytes(58));
    }

    #[test]
    fn registration_asks_but_does_not_charge() {
        let g = gate(1 << 30);
        // 20 qubits = 16 MiB, comfortably inside a 1 GiB cap.
        g.admit_circuit(20).expect("must be admissible");
        let charged: u64 = g
            .capacity()
            .unwrap()
            .into_iter()
            .filter(|r| r.axis == Axis::HostBytes)
            .map(|r| r.charged)
            .sum();
        assert_eq!(charged, 0, "asking must not charge");
    }

    #[test]
    fn a_circuit_wider_than_the_whole_budget_is_refused_permanently() {
        let g = gate(1 << 30);
        let e = g.admit_circuit(40).expect_err("16 TiB cannot fit in 1 GiB");
        assert!(!e.retryable, "{e}");
        assert!(e.reason.contains("host_bytes"), "must name the limit: {e}");
    }

    #[test]
    fn a_run_holds_its_capacity_for_exactly_the_life_of_the_ticket() {
        let g = gate(1 << 30);
        let charged = |g: &HostGate| -> u64 {
            g.capacity()
                .unwrap()
                .into_iter()
                .filter(|r| r.axis == Axis::HostBytes)
                .map(|r| r.charged)
                .sum()
        };
        {
            let _ticket = g.admit_run(20, true).expect("must fit");
            assert_eq!(charged(&g), 16 << 20, "the ticket must hold its bytes");
        }
        assert_eq!(charged(&g), 0, "dropping the ticket must release them");
    }

    #[test]
    fn a_second_run_is_refused_while_the_first_holds_and_admitted_after() {
        // 26 qubits = 1 GiB exactly, so the second cannot fit beside the first.
        let g = gate(1 << 30);
        let first = g.admit_run(26, true).expect("the first must fit");
        let e = g
            .admit_run(26, true)
            .expect_err("the second must not fit beside it");
        assert!(
            e.retryable,
            "it fits in an empty box, so waiting helps: {e}"
        );
        drop(first);
        drop(
            g.admit_run(26, true)
                .expect("it must fit once the first is done"),
        );
    }

    #[test]
    fn an_unreadable_ledger_refuses_registration_rather_than_admitting_it() {
        // This crate shipped the opposite. `capacity()` swallowed the error and
        // returned an empty Vec, so this loop iterated over nothing and returned
        // Ok — an unreadable ledger made admission PERMISSIVE, in a crate whose
        // entire stance is to fail closed. A 40-qubit circuit is 16 TiB.
        let d = std::env::temp_dir().join(format!(
            "omega-hostgate-adm-corrupt-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("gate.json");
        std::fs::write(&p, "").unwrap(); // a writer died mid-rewrite
        let mut caps = Amounts::new();
        caps.insert(Axis::HostBytes, 1 << 30);
        let g = HostGate::new(&p, "t", caps, Mode::Enforce);

        let e = g
            .admit_circuit(40)
            .expect_err("an unreadable budget must refuse, not admit 16 TiB");
        assert!(
            e.reason.contains("could not answer") || e.reason.contains("unreadable"),
            "the refusal must say the gate could not answer: {e}"
        );
        assert!(g.admit_run(40, true).is_err(), "and the charging path too");
    }

    #[test]
    fn an_unconfigured_gate_admits_everything_through_the_trait_too() {
        // The adoption contract: linking this must change nothing until an
        // operator asks for it.
        let g = HostGate::off();
        g.admit_circuit(45).expect("off must not refuse");
        let _t = g.admit_run(45, true).expect("off must not refuse");
    }
}
