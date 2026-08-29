// SPDX-License-Identifier: Apache-2.0
//! The failsafe, against real processes that are really killed.
//!
//! `proofs/tla/HostGate.tla` shows that a crashed holder's tokens come back and
//! that a live holder's never do. That is a statement about a model. These tests
//! are the same two statements about this machine: they spawn actual child
//! processes, record grants against their real PIDs and real start times, send a
//! real `SIGKILL`, and check what the next transaction does.
//!
//! The distinction matters because everything interesting here lives in the
//! liveness probe, and a probe is exactly the part a model has to assume. The
//! estate's prior art assumed it correctly on Linux and got it wrong on macOS,
//! where every scan pruned every live holder and each writer then persisted a
//! ledger containing only its own record — a gate that did not merely fail open
//! but actively erased other processes' grants. No model would have caught that.
//! Running it would.
//!
//! The catastrophic direction is `a_live_holders_grant_is_never_reclaimed`. A
//! leak wastes budget; a phantom reclaim hands a running process's memory to the
//! next request.

use std::collections::BTreeMap;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU32, Ordering};

use omega_hostgate::gate::HostGate;
use omega_hostgate::identity::{self, Liveness};
use omega_hostgate::ledger::{Ledger, Record, Store};
use omega_hostgate::{Amounts, Axis, Mode, Request};

static N: AtomicU32 = AtomicU32::new(0);

fn workdir() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "omega-hostgate-mp-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn caps(host_bytes: u64) -> Amounts {
    let mut c = Amounts::new();
    c.insert(Axis::HostBytes, host_bytes);
    c
}

/// A child that will outlive the test unless something kills it.
fn spawn_sleeper() -> Child {
    Command::new("/bin/sh")
        .args(["-c", "sleep 300"])
        .spawn()
        .expect("could not spawn a child process")
}

/// Write a grant for someone else's PID, stamped with that process's real start
/// time — exactly the record that process would have left behind.
fn record_for(path: &std::path::Path, caps: &Amounts, pid: u32, amount: u64) {
    let start = match identity::probe(&[pid]).get(&pid).copied() {
        Some(Liveness::Alive { start }) => start,
        other => panic!("expected the child {pid} to be alive, saw {other:?}"),
    };
    let store = Store::new(path);
    store
        .with_lock("test", caps, |ledger: &mut Ledger| {
            let mut amounts = BTreeMap::new();
            amounts.insert(Axis::HostBytes.to_string(), amount);
            ledger.holders.push(Record {
                pid,
                start,
                tag: format!("child-{pid}"),
                amounts,
                extra: BTreeMap::new(),
            });
            ((), true)
        })
        .expect("could not write the child's grant");
}

fn charged(gate: &HostGate) -> u64 {
    gate.capacity()
        .unwrap()
        .into_iter()
        .find(|r| r.axis == Axis::HostBytes)
        .map(|r| r.charged)
        .unwrap_or(0)
}

#[test]
fn a_live_holders_grant_is_never_reclaimed() {
    // The catastrophic direction. If this fails, the gate hands a running
    // process's memory to the next request.
    let d = workdir();
    let p = d.join("gate.json");
    let c = caps(100);
    let gate = HostGate::new(&p, "test", c.clone(), Mode::Enforce);

    let mut child = spawn_sleeper();
    record_for(&p, &c, child.id(), 90);

    // Several transactions, each of which runs a prune first.
    for _ in 0..3 {
        assert_eq!(
            charged(&gate),
            90,
            "a live child's grant was reclaimed — the gate would now admit \
             work on top of memory that is still in use"
        );
    }
    // And the budget really is spent: 90 of 100 charged leaves no room for 20.
    assert!(gate
        .try_acquire(&Request::new("mine").want(Axis::HostBytes, 20))
        .is_err());

    child.kill().ok();
    child.wait().ok();
}

#[test]
fn a_sigkilled_holder_returns_its_tokens_with_nobody_watching() {
    // The failsafe itself. No destructor runs in the child, nothing observes
    // its death, and the budget still repairs itself on the next transaction.
    let d = workdir();
    let p = d.join("gate.json");
    let c = caps(100);
    let gate = HostGate::new(&p, "test", c.clone(), Mode::Enforce);

    let mut child = spawn_sleeper();
    let pid = child.id();
    record_for(&p, &c, pid, 90);
    assert_eq!(charged(&gate), 90);
    assert!(
        gate.try_acquire(&Request::new("blocked").want(Axis::HostBytes, 20))
            .is_err(),
        "the child's grant must be honoured while it lives"
    );

    // SIGKILL: no unwinding, no Drop, no chance to release.
    child.kill().expect("kill failed");
    child.wait().expect("wait failed"); // reap, so the PID is fully gone

    assert_eq!(
        charged(&gate),
        0,
        "the dead child's tokens were not reclaimed — every crash would cost \
         the fleet that much budget until reboot"
    );
    let held = gate
        .try_acquire(&Request::new("after").want(Axis::HostBytes, 100))
        .expect("the whole budget must be available again");
    drop(held);
}

#[test]
fn a_zombie_holder_is_dead_even_though_its_pid_still_resolves() {
    // Killed but not reaped. The process table entry survives, so a probe that
    // only asks "does this PID exist" says yes — while the memory is already
    // gone. Both platforms must see through it.
    let d = workdir();
    let p = d.join("gate.json");
    let c = caps(100);
    let gate = HostGate::new(&p, "test", c.clone(), Mode::Enforce);

    let mut child = spawn_sleeper();
    let pid = child.id();
    record_for(&p, &c, pid, 90);

    child.kill().expect("kill failed");
    // Deliberately NOT reaped: dropping a Child does not wait on Unix, so the
    // entry lingers as a zombie for as long as this test process lives.
    std::mem::forget(child);

    // Give the kernel a moment to move it to Z; poll rather than sleep blindly.
    let mut seen = Liveness::Unknown;
    for _ in 0..200 {
        seen = identity::probe(&[pid])
            .get(&pid)
            .copied()
            .unwrap_or(Liveness::Unknown);
        if seen == Liveness::Dead {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(
        seen,
        Liveness::Dead,
        "a zombie must read as dead: its PID resolves but its memory does not"
    );
    assert_eq!(charged(&gate), 0, "a zombie's tokens must be reclaimed");
}

#[test]
fn two_processes_share_one_budget_and_the_second_is_refused() {
    // The whole point of the crate, in one test: two independent gate instances
    // over one file, as two separate programs on a box would be.
    let d = workdir();
    let p = d.join("gate.json");
    let c = caps(100);
    let engine = HostGate::new(&p, "test", c.clone(), Mode::Enforce);
    let sweep = HostGate::new(&p, "test", c.clone(), Mode::Enforce);

    let held = engine
        .try_acquire(&Request::new("engine").want(Axis::HostBytes, 70))
        .expect("the first must fit");

    let err = sweep
        .try_acquire(&Request::new("sweep").want(Axis::HostBytes, 50))
        .expect_err("the second must not fit alongside the first");
    assert!(
        err.retryable(),
        "it fits in an empty box, so it is retryable: {err}"
    );

    drop(held);
    let after = sweep
        .try_acquire(&Request::new("sweep again").want(Axis::HostBytes, 50))
        .expect("it must fit once the first releases");
    drop(after);
}

#[test]
fn a_transferred_grant_survives_the_wrapper_and_dies_with_the_child() {
    // The `omega-hostgate run -- <cmd>` shape. A wrapper acquires and then hands
    // the charge to the process that will actually hold the memory.
    //
    // Holding it in the wrapper instead is the tempting simplification and it is
    // wrong: kill the wrapper and the prune reclaims its tokens while the
    // orphaned child keeps running and keeps its memory, so the gate then admits
    // fresh work on top of it. That is this crate's own failure mode,
    // reintroduced by the component whose job is to extend the budget to
    // processes that cannot call it.
    let d = workdir();
    let p = d.join("gate.json");
    let c = caps(100);
    let gate = HostGate::new(&p, "test", c.clone(), Mode::Enforce);

    let mut child = spawn_sleeper();
    let pid = child.id();

    let grant = gate
        .try_acquire(&Request::new("wrapper").want(Axis::HostBytes, 90))
        .expect("must fit");
    grant.transfer_to(pid).expect("transfer failed");

    // The wrapper's guard is gone, and the charge is still standing.
    assert_eq!(
        charged(&gate),
        90,
        "the transfer must not have released the charge — the child holds the \
         memory and must be accounted for it"
    );
    assert!(gate
        .try_acquire(&Request::new("other").want(Axis::HostBytes, 20))
        .is_err());

    // And it is the CHILD that is charged: kill it, and the prune reclaims.
    child.kill().expect("kill failed");
    child.wait().expect("wait failed");
    assert_eq!(
        charged(&gate),
        0,
        "the child's death must return the transferred tokens"
    );
}

#[test]
fn transferring_to_a_process_that_is_already_gone_releases_rather_than_leaks() {
    let d = workdir();
    let p = d.join("gate.json");
    let c = caps(100);
    let gate = HostGate::new(&p, "test", c.clone(), Mode::Enforce);

    let mut child = spawn_sleeper();
    let pid = child.id();
    child.kill().expect("kill failed");
    child.wait().expect("wait failed");

    let grant = gate
        .try_acquire(&Request::new("wrapper").want(Axis::HostBytes, 90))
        .expect("must fit");
    grant
        .transfer_to(pid)
        .expect("transfer to a dead pid must not error");

    assert_eq!(
        charged(&gate),
        0,
        "there is no identity left to charge, so the grant must be released \
         rather than parked on a pid that will never be pruned"
    );
}
