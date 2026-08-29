// SPDX-License-Identifier: Apache-2.0
//! Acquire, release, and the failsafe that runs when neither happens.
//!
//! This module is the implementation of `proofs/tla/HostGate.tla`, and the
//! correspondence is deliberate enough to be worth reading side by side:
//!
//! | model | here |
//! |---|---|
//! | `Acquire(p)` | [`HostGate::try_acquire`] |
//! | `Release(p)`, guarded by `Owns` | [`Grant::drop`] |
//! | `Crash(p)` | nothing — that is the point |
//! | `Prune`, guarded by `StillHeld` | [`prune`], run at the head of every transaction |
//!
//! # Why there is no `Crash` handler
//!
//! Because there cannot be one. A process that is `SIGKILL`ed, OOM-killed, or
//! aborted runs no destructor, so nothing on its way out can return its tokens.
//! The recovery is therefore not an action the dying process takes; it is
//! something every *other* process does on its behalf, every time it takes the
//! lock. That is the whole failsafe: `prune` runs first, unconditionally, in
//! every transaction, so the budget repairs itself as a side effect of being
//! used. Nothing detects a death and nothing needs to.
//!
//! # Two guards that look alike and are not
//!
//! The model earned this distinction the hard way and the code inherits it.
//!
//! * **`StillHeld`** is what [`prune`] asks of a *record in the file*, and it is
//!   only ever as good as the identity we recorded. It compares
//!   `(pid, start_time)`, so a recycled PID does not inherit a dead holder's
//!   record.
//! * **`Owns`** is what a *process* knows about its own [`Grant`]: it has the
//!   guard object in its own memory, or it does not. A restarted process never
//!   has one, whatever the file says.
//!
//! Using the first where the second belongs lets a restarted process release
//! its predecessor's record — which in the model repaired the exact leak the
//! model existed to expose. Here it would mean a process returning tokens it
//! never took, and the next request being admitted against them.
//!
//! # Nested acquisition, for now
//!
//! Two grants taken by one process are charged **twice**. That is the safe
//! direction and it is deliberate: the alternative — a refcount that lets a
//! nested request pass free when it "fits inside" what the process already
//! holds — cannot distinguish re-entrancy (the same bytes, on one call stack)
//! from concurrency (a sibling thread, genuinely new bytes), and charges twelve
//! parallel workers once. Making the cheap case free needs a thread-scoped
//! sub-allocator, and it is not in this step.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::identity::{self, HolderId};
use crate::ledger::{Ledger, Record, Store};
use crate::{Amounts, Axis, Mode, Refusal, Request};

/// One axis as an operator sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AxisReport {
    pub axis: Axis,
    pub cap: u64,
    pub charged: u64,
    pub holders: usize,
}

struct Inner {
    store: Option<Store>,
    instance: String,
    caps: Amounts,
    mode: Mode,
}

/// The gate. Cheap to clone; every clone shares one budget.
#[derive(Clone)]
pub struct HostGate(Arc<Inner>);

impl HostGate {
    /// A gate that grants everything and touches nothing.
    ///
    /// This is what an unconfigured process gets, and it performs **no lock, no
    /// read and no syscall** — a linked-but-unconfigured build must behave
    /// byte-for-byte as it did before, or nobody can adopt this incrementally.
    pub fn off() -> Self {
        HostGate(Arc::new(Inner {
            store: None,
            instance: String::new(),
            caps: Amounts::new(),
            mode: Mode::Off,
        }))
    }

    pub fn new(
        path: impl Into<std::path::PathBuf>,
        instance: impl Into<String>,
        caps: Amounts,
        mode: Mode,
    ) -> Self {
        if mode == Mode::Off {
            return Self::off();
        }
        HostGate(Arc::new(Inner {
            store: Some(Store::new(path)),
            instance: instance.into(),
            caps,
            mode,
        }))
    }

    pub fn mode(&self) -> Mode {
        self.0.mode
    }

    /// Price a request and charge it, or say why not.
    ///
    /// Non-blocking by construction: it fits now or it is refused now. A caller
    /// that wants to wait retries, because a gate that queues on the caller's
    /// behalf turns an admission problem into an unbounded-queue problem — and
    /// the queue would live in a file, which is the worst available place for
    /// one.
    pub fn try_acquire(&self, req: &Request) -> Result<Grant, Refusal> {
        let inner = &self.0;
        let Some(store) = inner.store.as_ref() else {
            return Ok(Grant::inert());
        };
        let advisory = inner.mode == Mode::Advisory;

        let Some(start) = identity::self_start() else {
            // We cannot stamp an identity, so a record we wrote could never be
            // pruned correctly — it would leak until reboot. Refusing is the
            // lesser failure, and in advisory mode we simply do not record.
            return if advisory {
                Ok(Grant::inert())
            } else {
                Err(Refusal::Unavailable {
                    why: "cannot determine this process's own start time, so a \
                          grant could not be reclaimed if we died"
                        .into(),
                })
            };
        };
        let me = HolderId {
            pid: std::process::id(),
            start,
        };

        let outcome = store.with_lock(&inner.instance, &inner.caps, |ledger| {
            // The failsafe, first and unconditionally.
            prune(ledger);

            if !advisory {
                if let Some(refusal) = price(ledger, req) {
                    // Refused: the prune may still have reclaimed something, so
                    // that repair is worth keeping even though the request is not.
                    return (Err(refusal), true);
                }
            }
            charge(ledger, &me, req, advisory);
            (Ok(()), true)
        });

        match outcome {
            Ok(Ok(())) => Ok(Grant {
                gate: Some(Arc::clone(&self.0)),
                holder: me,
                amounts: req.amounts.clone(),
                tag: req.tag.clone(),
            }),
            Ok(Err(refusal)) => Err(refusal),
            Err(e) => {
                // Fail closed: an unreadable budget is not an absent one.
                if advisory {
                    Ok(Grant::inert())
                } else {
                    Err(Refusal::Unavailable { why: e.to_string() })
                }
            }
        }
    }

    /// Every axis, with what is charged against it right now. Prunes first, so
    /// what it reports is what a request would actually meet.
    ///
    /// **Returns a `Result` because an empty list is a legitimate answer.** A
    /// gate that is off has no axes, and so does one whose ledger could not be
    /// read — and those must not look alike. When this swallowed the error and
    /// returned an empty `Vec`, `Admission::admit_circuit` iterated over nothing
    /// and returned `Ok`, so an unreadable ledger made admission PERMISSIVE in a
    /// crate whose whole stance is to fail closed. `status` printed a clean
    /// empty table for a broken gate.
    pub fn capacity(&self) -> Result<Vec<AxisReport>, Refusal> {
        let inner = &self.0;
        let Some(store) = inner.store.as_ref() else {
            return Ok(Vec::new());
        };
        let snapshot = store.with_lock(&inner.instance, &inner.caps, |ledger| {
            let before = ledger.holders.len();
            prune(ledger);
            // Rewrite only when the prune actually reclaimed something: an
            // inspection must not be a write load on the file every acquire on
            // the box has to lock.
            let changed = ledger.holders.len() != before;
            let report: Vec<AxisReport> = inner
                .caps
                .keys()
                .map(|axis| AxisReport {
                    axis: axis.clone(),
                    cap: ledger.cap(axis),
                    charged: charged_enforcing(ledger, axis),
                    holders: ledger.holders.iter().filter(|r| r.amount(axis) > 0).count(),
                })
                .collect();
            (report, changed)
        });
        snapshot.map_err(|e| Refusal::Unavailable { why: e.to_string() })
    }
}

/// A held claim. Releases when it drops, including while unwinding a panic.
///
/// An inert grant (`Mode::Off`, or a gate that could not record) drops to
/// nothing, so callers never branch on whether the gate is configured.
pub struct Grant {
    gate: Option<Arc<Inner>>,
    holder: HolderId,
    amounts: Amounts,
    tag: String,
}

impl std::fmt::Debug for Grant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grant")
            .field("held", &self.gate.is_some())
            .field("pid", &self.holder.pid)
            .field("amounts", &self.amounts.len())
            .finish()
    }
}

impl Grant {
    fn inert() -> Self {
        Grant {
            gate: None,
            holder: HolderId { pid: 0, start: 0 },
            amounts: Amounts::new(),
            tag: String::new(),
        }
    }

    /// What this grant is charged, per axis.
    pub fn amounts(&self) -> &Amounts {
        &self.amounts
    }

    /// Hand this grant to another process, by identity.
    ///
    /// # Why a wrapper must transfer rather than hold
    ///
    /// `omega-hostgate run -- <cmd>` acquires, then spawns. If the grant stayed
    /// recorded against the WRAPPER and the wrapper were killed, the prune would
    /// reclaim its tokens while the orphaned child kept running and kept its
    /// memory — and the gate would then admit fresh work on top of it. That is
    /// the failure this whole crate exists to prevent, reintroduced by the one
    /// component whose job is to extend the budget to processes that cannot
    /// call it.
    ///
    /// Recording it against the CHILD makes both deaths behave: kill the
    /// wrapper and the child keeps its charge; kill the child and its identity
    /// stops matching, so the next prune reclaims it. Nothing has to notice
    /// either event.
    ///
    /// A child that is already gone is released rather than transferred — there
    /// is no identity left to charge, and holding it would leak.
    pub fn transfer_to(mut self, pid: u32) -> Result<(), Refusal> {
        // Borrowed, not taken: the grant stays armed until the transaction has
        // actually committed. Taking it up front meant a FAILED transfer left
        // the charge on this process's record with the guard already disarmed,
        // so `Drop` would not release it — the wrapper silently degraded from
        // transfer semantics to hold semantics, which is the orphaned-child
        // failure `transfer_to` exists to prevent, reached by the error path.
        let Some(inner) = self.gate.clone() else {
            return Ok(());
        };
        let Some(store) = inner.store.as_ref() else {
            return Ok(());
        };
        let from = self.holder;
        let mine = self.amounts.clone();
        let tag = self.tag.clone();

        let to = match identity::probe(&[pid]).get(&pid).copied() {
            Some(crate::Liveness::Alive { start }) => Some(HolderId { pid, start }),
            // Unknown means alive-but-unreadable everywhere else in this crate,
            // and it must here too: refusing to transfer would leave the charge
            // on a wrapper that is about to exit.
            Some(crate::Liveness::Unknown) => Some(HolderId { pid, start: 0 }),
            _ => None,
        };

        match store.with_lock(&inner.instance, &inner.caps, |ledger| {
            discharge(ledger, &from, &mine);
            if let Some(to) = to {
                charge_holder(ledger, &to, &mine, &tag, false);
            }
            prune(ledger);
            ((), true)
        }) {
            Ok(()) => {
                // Committed: only now is this guard no longer responsible.
                self.gate = None;
                self.amounts.clear();
                Ok(())
            }
            Err(e) => Err(Refusal::Unavailable { why: e.to_string() }),
        }
    }

    /// Release now and report failure, rather than losing it in `drop`.
    ///
    /// A release that fails silently is a slow leak: the record stays, the
    /// process stays alive so the prune never fires, and the budget shrinks by
    /// that much for the lifetime of the process. `Drop` cannot return an error,
    /// so a caller that cares is given a way to ask.
    pub fn release(mut self) -> Result<(), Refusal> {
        self.release_inner()
    }

    fn release_inner(&mut self) -> Result<(), Refusal> {
        let Some(inner) = self.gate.take() else {
            return Ok(());
        };
        let Some(store) = inner.store.as_ref() else {
            return Ok(());
        };
        let me = self.holder;
        let mine = std::mem::take(&mut self.amounts);
        store
            .with_lock(&inner.instance, &inner.caps, |ledger| {
                discharge(ledger, &me, &mine);
                prune(ledger);
                ((), true)
            })
            .map_err(|e| Refusal::Unavailable { why: e.to_string() })
    }
}

impl Drop for Grant {
    fn drop(&mut self) {
        // `Drop` cannot return an error, but it can refuse to be silent. A
        // failed release leaves the record standing while this process lives
        // on, so the prune will not touch it — the budget simply shrinks by
        // that much for the rest of the process's life, and nothing anywhere
        // says why. The backstop is real but distant: the tokens come back when
        // this process eventually dies.
        if let Err(e) = self.release_inner() {
            eprintln!(
                "[hostgate] could not return a grant ({e}); the budget stays \
                 charged for it until this process exits"
            );
        }
    }
}

// ---------------------------------------------------------------- internals

/// The failsafe. Drop every record whose owner is no longer there.
///
/// `Liveness::Unknown` counts as held — an unreadable probe must never evict a
/// live holder, because handing its memory to the next request is strictly
/// worse than leaving a stale record for one more scan.
fn prune(ledger: &mut Ledger) {
    let pids: Vec<u32> = ledger.holders.iter().map(|r| r.pid).collect();
    if pids.is_empty() {
        return;
    }
    let seen = identity::probe(&pids);
    ledger.holders.retain(|r| {
        let live = seen
            .get(&r.pid)
            .copied()
            .unwrap_or(crate::Liveness::Unknown);
        r.holder().matches(live)
    });
}

/// What the enforcing budget counts. Advisory records are excluded: a process
/// that is not honouring the budget must not be able to spend it on behalf of
/// the processes that are.
fn charged_enforcing(ledger: &Ledger, axis: &Axis) -> u64 {
    ledger
        .holders
        .iter()
        .filter(|r| !is_advisory(r))
        .map(|r| r.amount(axis))
        .sum()
}

fn is_advisory(r: &Record) -> bool {
    r.extra
        .get("advisory")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// `None` admits. Caps are checked across **every** axis before any headroom
/// is, so a request that can never fit is told so rather than being told to
/// come back later — the difference between a retry loop that terminates and
/// one that does not.
fn price(ledger: &Ledger, req: &Request) -> Option<Refusal> {
    for (axis, want) in &req.amounts {
        let cap = ledger.cap(axis);
        if *want > cap {
            return Some(Refusal::TooLarge {
                axis: axis.clone(),
                required: *want,
                cap,
            });
        }
    }
    for (axis, want) in &req.amounts {
        let free = ledger
            .cap(axis)
            .saturating_sub(charged_enforcing(ledger, axis));
        if *want > free {
            return Some(Refusal::Busy {
                axis: axis.clone(),
                required: *want,
                available: free,
            });
        }
    }
    None
}

/// Add this request to the holder's record, creating it if needed.
fn charge(ledger: &mut Ledger, me: &HolderId, req: &Request, advisory: bool) {
    charge_holder(ledger, me, &req.amounts, &req.tag, advisory);
}

/// Add `amounts` to `who`'s record, creating it if needed.
fn charge_holder(ledger: &mut Ledger, me: &HolderId, amounts: &Amounts, tag: &str, advisory: bool) {
    let idx = ledger
        .holders
        .iter()
        .position(|r| r.pid == me.pid && r.start == me.start);
    let idx = match idx {
        Some(i) => i,
        None => {
            let mut extra = BTreeMap::new();
            if advisory {
                extra.insert("advisory".to_string(), serde_json::Value::Bool(true));
            }
            ledger.holders.push(Record {
                pid: me.pid,
                start: me.start,
                tag: tag.to_string(),
                amounts: BTreeMap::new(),
                extra,
            });
            ledger.holders.len() - 1
        }
    };
    for (axis, want) in amounts {
        let slot = ledger.holders[idx]
            .amounts
            .entry(axis.to_string())
            .or_insert(0);
        *slot = slot.saturating_add(*want);
    }
}

/// Subtract exactly what this grant took.
///
/// Saturating, and the record is removed once it reaches zero on every axis.
/// An over-release cannot make the total go negative and re-inflate the budget
/// — which, in a shared file, would hand capacity to everyone at once.
fn discharge(ledger: &mut Ledger, me: &HolderId, mine: &Amounts) {
    let Some(idx) = ledger
        .holders
        .iter()
        .position(|r| r.pid == me.pid && r.start == me.start)
    else {
        return;
    };
    for (axis, amount) in mine {
        if let Some(slot) = ledger.holders[idx].amounts.get_mut(&axis.to_string()) {
            *slot = slot.saturating_sub(*amount);
        }
    }
    if ledger.holders[idx].amounts.values().all(|v| *v == 0) {
        ledger.holders.remove(idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static N: AtomicU32 = AtomicU32::new(0);

    fn gate(mode: Mode, host_bytes: u64, slots: u64) -> (HostGate, std::path::PathBuf) {
        let d = std::env::temp_dir().join(format!(
            "omega-hostgate-gate-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("gate.json");
        let mut caps = Amounts::new();
        caps.insert(Axis::HostBytes, host_bytes);
        caps.insert(Axis::Slots, slots);
        (HostGate::new(&p, "inst", caps, mode), p)
    }

    #[test]
    fn off_grants_everything_and_never_creates_a_file() {
        let g = HostGate::off();
        let r = Request::new("anything").want(Axis::HostBytes, u64::MAX);
        assert!(g.try_acquire(&r).is_ok());
        assert!(g.capacity().unwrap().is_empty());
    }

    #[test]
    fn a_grant_is_charged_and_returns_on_drop() {
        let (g, _p) = gate(Mode::Enforce, 100, 4);
        {
            let _held = g
                .try_acquire(&Request::new("job").want(Axis::HostBytes, 60))
                .expect("must fit");
            let rep = g.capacity().unwrap();
            let hb = rep.iter().find(|r| r.axis == Axis::HostBytes).unwrap();
            assert_eq!(hb.charged, 60);
            assert_eq!(hb.holders, 1);
        }
        let hb = g
            .capacity()
            .unwrap()
            .into_iter()
            .find(|r| r.axis == Axis::HostBytes)
            .unwrap();
        assert_eq!(hb.charged, 0, "dropping the guard must return the tokens");
        assert_eq!(hb.holders, 0);
    }

    #[test]
    fn bigger_than_the_whole_budget_is_never_retryable() {
        let (g, _p) = gate(Mode::Enforce, 100, 4);
        let err = g
            .try_acquire(&Request::new("huge").want(Axis::HostBytes, 101))
            .unwrap_err();
        assert!(!err.retryable(), "{err}");
        match err {
            Refusal::TooLarge { required, cap, .. } => {
                assert_eq!((required, cap), (101, 100));
            }
            other => panic!("expected TooLarge, got {other}"),
        }
    }

    #[test]
    fn fits_but_not_now_is_retryable_and_says_what_is_free() {
        let (g, _p) = gate(Mode::Enforce, 100, 4);
        let _held = g
            .try_acquire(&Request::new("first").want(Axis::HostBytes, 70))
            .unwrap();
        let err = g
            .try_acquire(&Request::new("second").want(Axis::HostBytes, 40))
            .unwrap_err();
        assert!(err.retryable(), "{err}");
        match err {
            Refusal::Busy {
                required,
                available,
                ..
            } => {
                assert_eq!((required, available), (40, 30));
            }
            other => panic!("expected Busy, got {other}"),
        }
    }

    #[test]
    fn a_multi_axis_request_is_all_or_nothing() {
        let (g, _p) = gate(Mode::Enforce, 100, 4);
        let _held = g
            .try_acquire(&Request::new("cores").want(Axis::Slots, 4))
            .unwrap();
        // Memory is free; slots are not. The whole request must fail, and it
        // must not have charged the memory half on its way out.
        let err = g
            .try_acquire(
                &Request::new("mixed")
                    .want(Axis::HostBytes, 10)
                    .want(Axis::Slots, 1),
            )
            .unwrap_err();
        assert!(matches!(err, Refusal::Busy { .. }), "{err}");
        let hb = g
            .capacity()
            .unwrap()
            .into_iter()
            .find(|r| r.axis == Axis::HostBytes)
            .unwrap();
        assert_eq!(hb.charged, 0, "a refused request must charge nothing");
    }

    #[test]
    fn a_dead_holders_tokens_come_back_without_anyone_noticing_it_died() {
        // The failsafe, and the one property no destructor can provide: write a
        // record for a PID that cannot exist, exactly as a SIGKILLed holder
        // would have left behind, and watch the next transaction reclaim it.
        let (g, p) = gate(Mode::Enforce, 100, 4);
        let _real = g
            .try_acquire(&Request::new("live").want(Axis::HostBytes, 10))
            .unwrap();
        let raw = std::fs::read_to_string(&p).unwrap();
        let mut led: Ledger = serde_json::from_str(&raw).unwrap();
        let mut amounts = BTreeMap::new();
        amounts.insert(Axis::HostBytes.to_string(), 80u64);
        led.holders.push(Record {
            pid: 0, // never a live process
            start: 1,
            tag: "ghost".into(),
            amounts,
            extra: BTreeMap::new(),
        });
        std::fs::write(&p, serde_json::to_string(&led).unwrap()).unwrap();

        // 80 + 10 charged; without the prune this request cannot fit.
        let held = g
            .try_acquire(&Request::new("after").want(Axis::HostBytes, 85))
            .expect("the ghost's tokens must have been reclaimed");
        drop(held);
        let hb = g
            .capacity()
            .unwrap()
            .into_iter()
            .find(|r| r.axis == Axis::HostBytes)
            .unwrap();
        assert_eq!(hb.charged, 10, "only the live holder should remain");
    }

    #[test]
    fn a_recycled_pid_does_not_keep_the_dead_holders_record_alive() {
        // Our own PID, but a start time that is not ours: the record looks
        // live to a PID-only check and must still be reclaimed.
        let (g, p) = gate(Mode::Enforce, 100, 4);
        let raw_caps = std::fs::read_to_string(&p);
        assert!(raw_caps.is_err(), "no file until something is charged");

        let _seed = g
            .try_acquire(&Request::new("seed").want(Axis::HostBytes, 1))
            .unwrap();
        let mut led: Ledger = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        let mut amounts = BTreeMap::new();
        amounts.insert(Axis::HostBytes.to_string(), 90u64);
        led.holders.push(Record {
            pid: std::process::id(),
            start: 0xDEAD_BEEF, // not this incarnation
            tag: "predecessor".into(),
            amounts,
            extra: BTreeMap::new(),
        });
        std::fs::write(&p, serde_json::to_string(&led).unwrap()).unwrap();

        let held = g
            .try_acquire(&Request::new("now").want(Axis::HostBytes, 90))
            .expect("the predecessor's record must be reclaimed");
        drop(held);
    }

    #[test]
    fn advisory_records_but_never_refuses_and_never_spends_the_enforcing_budget() {
        let (g, _p) = gate(Mode::Advisory, 100, 4);
        let _over = g
            .try_acquire(&Request::new("way over").want(Axis::HostBytes, 10_000))
            .expect("advisory never refuses");
        let hb = g
            .capacity()
            .unwrap()
            .into_iter()
            .find(|r| r.axis == Axis::HostBytes)
            .unwrap();
        assert_eq!(
            hb.charged, 0,
            "an advisory holder must not consume the budget that enforcing \
             processes respect"
        );
    }

    #[test]
    fn an_unreadable_ledger_refuses_rather_than_granting() {
        let (g, p) = gate(Mode::Enforce, 100, 4);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, "").unwrap(); // a writer died mid-rewrite
        let err = g
            .try_acquire(&Request::new("after the crash").want(Axis::HostBytes, 1))
            .unwrap_err();
        assert!(matches!(err, Refusal::Unavailable { .. }), "{err}");
    }

    #[test]
    fn two_grants_from_one_process_are_charged_twice() {
        // The safe direction, and deliberate: a refcount that let the second
        // pass free could not tell a nested call from a sibling thread.
        let (g, _p) = gate(Mode::Enforce, 100, 4);
        let _a = g
            .try_acquire(&Request::new("a").want(Axis::HostBytes, 30))
            .unwrap();
        let _b = g
            .try_acquire(&Request::new("b").want(Axis::HostBytes, 30))
            .unwrap();
        let hb = g
            .capacity()
            .unwrap()
            .into_iter()
            .find(|r| r.axis == Axis::HostBytes)
            .unwrap();
        assert_eq!(hb.charged, 60);
        assert_eq!(hb.holders, 1, "one process is one record");
    }

    #[test]
    fn releasing_more_than_was_taken_cannot_inflate_the_budget() {
        let (g, _p) = gate(Mode::Enforce, 100, 4);
        let a = g
            .try_acquire(&Request::new("a").want(Axis::HostBytes, 10))
            .unwrap();
        let b = g
            .try_acquire(&Request::new("b").want(Axis::HostBytes, 10))
            .unwrap();
        a.release().unwrap();
        b.release().unwrap();
        let hb = g
            .capacity()
            .unwrap()
            .into_iter()
            .find(|r| r.axis == Axis::HostBytes)
            .unwrap();
        assert_eq!(hb.charged, 0);
        // And the budget is exactly what it started as, not more.
        let _all = g
            .try_acquire(&Request::new("whole").want(Axis::HostBytes, 100))
            .expect("the full cap must be available, and no more");
        assert!(g
            .try_acquire(&Request::new("one more").want(Axis::HostBytes, 1))
            .is_err());
    }
}
