// SPDX-License-Identifier: Apache-2.0
//! **A guest's own circuit registration is priced, and an ordinary guest still
//! runs.**
//!
//! `PLAN-SIX-PROGRAMMES.md` P6. The WASM route never bypassed the capacity guard
//! — that runs inside the backend on both the execute and expectation paths, so a
//! 40-qubit guest circuit was already refused. What it bypassed was the
//! governor's global RESERVATION LEDGER: the route took no reservation, so two
//! large jobs arriving through different doors could each pass capacity
//! individually and collectively oversubscribe the machine.
//!
//! Accounting, not memory safety. These tests are about the accounting.
//!
//! # Both directions, deliberately
//!
//! A test that only checks the refusal would pass for an implementation that
//! refuses everything, which is the easier bug to write — and it would break the
//! standalone `omega-wasm-cli`, which has no governor at all. So the
//! no-admission path is asserted too, and it is the DEFAULT.

use std::sync::Arc;

use omega_core::admission::{Admission, Refusal};
use omega_core::circuit::{CircuitIR, CircuitType};
use omega_wasm_runtime::host::HostState;

/// Refuses anything above `limit` qubits, naming the limit — which is the
/// property the message is required to have, since a guest told only "refused"
/// cannot decide whether to retry smaller.
struct CapAt {
    limit: u32,
    retryable: bool,
}

impl Admission for CapAt {
    fn admit_circuit(&self, num_qubits: u32) -> Result<(), Refusal> {
        if num_qubits > self.limit {
            Err(Refusal {
                reason: format!(
                    "circuit needs {num_qubits} qubits but the budget allows {}",
                    self.limit
                ),
                retryable: self.retryable,
            })
        } else {
            Ok(())
        }
    }
    fn admit_run(
        &self,
        num_qubits: u32,
        _analytic: bool,
    ) -> Result<omega_core::admission::RunTicket, Refusal> {
        self.admit_circuit(num_qubits)?;
        Ok(Box::new(()))
    }
}

/// A ledger that counts LIVE tickets, so a test can see whether an execution
/// held its charge for the call and released it after — the property the
/// registration-only mock above cannot observe.
struct LiveLedger {
    live: Arc<std::sync::atomic::AtomicI32>,
    peak: Arc<std::sync::atomic::AtomicI32>,
}

/// The ticket: decrements the live count when dropped.
struct LedgerTicket(Arc<std::sync::atomic::AtomicI32>);
impl Drop for LedgerTicket {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Admission for LiveLedger {
    fn admit_circuit(&self, _num_qubits: u32) -> Result<(), Refusal> {
        Ok(())
    }
    fn admit_run(
        &self,
        _num_qubits: u32,
        _analytic: bool,
    ) -> Result<omega_core::admission::RunTicket, Refusal> {
        use std::sync::atomic::Ordering::SeqCst;
        let now = self.live.fetch_add(1, SeqCst) + 1;
        self.peak.fetch_max(now, SeqCst);
        Ok(Box::new(LedgerTicket(Arc::clone(&self.live))))
    }
}

fn circuit(n: u32) -> CircuitIR {
    CircuitIR::new(n, CircuitType::GateBased)
}

#[test]
fn no_governor_means_every_registration_is_admitted() {
    // The standalone `omega-wasm-cli` shape. `None` is a normal state, not a
    // misconfiguration, and this is the test that keeps it that way.
    let mut host = HostState::new();
    assert!(host.admission.is_none(), "default must have no governor");
    for n in [1u32, 8, 30] {
        assert!(
            host.register_circuit_admitted(circuit(n)).is_ok(),
            "n={n} refused with no governor attached"
        );
    }
}

#[test]
fn an_over_budget_guest_is_refused_and_the_limit_is_named() {
    let mut host = HostState::new().with_admission(Arc::new(CapAt {
        limit: 20,
        retryable: false,
    }));
    let err = host
        .register_circuit_admitted(circuit(28))
        .expect_err("28 qubits must be refused against a 20-qubit budget");
    assert!(err.reason.contains("28"), "must name the request: {err}");
    assert!(err.reason.contains("20"), "must name the limit: {err}");
    assert!(!err.retryable, "a budget ceiling is permanent, not busy");
    // And the refusal must not leave the circuit half-registered.
    assert!(
        host.circuits.is_empty(),
        "a refused circuit must not be registered anyway"
    );
}

#[test]
fn an_ordinary_guest_still_runs_under_a_governor() {
    let mut host = HostState::new().with_admission(Arc::new(CapAt {
        limit: 20,
        retryable: false,
    }));
    let id = host
        .register_circuit_admitted(circuit(6))
        .expect("6 qubits is well inside a 20-qubit budget");
    assert_eq!(id, 1);
    assert_eq!(host.circuits.len(), 1);
}

/// Retryable and permanent must be distinguishable.
///
/// The distinction is the difference between a guest retry loop that terminates
/// and one that spins for ever: "the machine is busy" invites a retry, "wider
/// than the budget" never will be. Collapsing them into one flag is the failure
/// this asserts against.
#[test]
fn retryable_and_permanent_refusals_are_distinguishable() {
    for retryable in [true, false] {
        let mut host = HostState::new().with_admission(Arc::new(CapAt {
            limit: 4,
            retryable,
        }));
        let err = host.register_circuit_admitted(circuit(10)).unwrap_err();
        assert_eq!(err.retryable, retryable);
        // Display carries it too, so a log line is actionable without the field.
        let shown = format!("{err}");
        assert!(
            shown.contains(if retryable { "retryable" } else { "permanent" }),
            "Display must say which: {shown}"
        );
    }
}

/// The unadmitted entry point still exists and still bypasses admission.
///
/// Not an oversight — `omega-server` uses it to PRE-register the circuit a
/// lambda was invoked with, which the governor already priced on the HTTP path.
/// Routing that through admission again would charge one job twice and refuse
/// work that was legitimately admitted. This pins that the two entry points mean
/// different things.
#[test]
fn the_unadmitted_entry_point_is_still_a_bypass_on_purpose() {
    let mut host = HostState::new().with_admission(Arc::new(CapAt {
        limit: 2,
        retryable: false,
    }));
    // Would be refused through the admitted door.
    assert!(host.register_circuit_admitted(circuit(16)).is_err());
    // The server's pre-registration door does not ask.
    let id = host.register_circuit(circuit(16));
    assert_eq!(id, 1, "pre-registration must not consult admission");
}

/// A run CHARGES the ledger for exactly the call's duration: one live ticket
/// while the backend works, zero after it returns. Registration alone charges
/// nothing — the exact split the P6 "still open" note asked for.
#[test]
fn a_run_holds_one_ticket_and_releases_it_after() {
    use std::sync::atomic::{AtomicI32, Ordering::SeqCst};
    let live = Arc::new(AtomicI32::new(0));
    let peak = Arc::new(AtomicI32::new(0));
    let mut host = HostState::new().with_admission(Arc::new(LiveLedger {
        live: Arc::clone(&live),
        peak: Arc::clone(&peak),
    }));

    let mut c = circuit(2);
    {
        use omega_core::circuit::{GateKind, GateOp, Qubit};
        c.ops.push(GateOp {
            gate: GateKind::H,
            qubits: [Qubit(0)].into_iter().collect(),
            params: Default::default(),
            classical_bit: None,
            condition: None,
        });
    }
    let cid = host.register_circuit_admitted(c).expect("admitted");
    assert_eq!(
        peak.load(SeqCst),
        0,
        "registration must not charge a run ticket"
    );

    let counts = host
        .execute_with_shots(cid, &[], 50, Some(1))
        .expect("run admitted");
    assert_eq!(counts.values().sum::<u32>(), 50);
    assert_eq!(
        peak.load(SeqCst),
        1,
        "the run must have held exactly one ticket"
    );
    assert_eq!(
        live.load(SeqCst),
        0,
        "the ticket must be released after the call"
    );

    // A batch over several observables is ONE execution and one ticket.
    let oid = host.register_observable(omega_wasm_runtime::host::h2_hamiltonian());
    let _ = host
        .execute_multi(cid, &[], &[oid, oid])
        .expect("multi runs");
    assert_eq!(
        peak.load(SeqCst),
        1,
        "execute_multi must charge once, not per observable"
    );
    assert_eq!(live.load(SeqCst), 0);
}

/// A refused run reaches the guest as an ERROR naming admission control, and
/// no backend work happens behind the refusal (the ledger shows no ticket).
#[test]
fn a_refused_run_is_an_error_that_names_admission() {
    let mut host = HostState::new().with_admission(Arc::new(CapAt {
        limit: 1,
        retryable: true,
    }));
    // Registration at 1 qubit passes a 1-qubit cap...
    let cid = host
        .register_circuit_admitted(circuit(1))
        .expect("admitted");
    // ...then tighten the world: swap in a governor that refuses everything,
    // the moral equivalent of the machine filling up between registration and
    // execution — the race this feature closes.
    host.admission = Some(Arc::new(CapAt {
        limit: 0,
        retryable: true,
    }));
    let err = host
        .execute_with_shots(cid, &[], 10, Some(1))
        .expect_err("the run must be refused by the new ledger state");
    let msg = format!("{err}");
    assert!(
        msg.contains("admission control"),
        "the guest must learn WHO refused: {msg}"
    );
    assert!(
        msg.contains("retryable"),
        "and whether waiting helps: {msg}"
    );
}
