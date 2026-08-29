// SPDX-License-Identifier: Apache-2.0
//! One resource budget per machine, shared by every process on it.
//!
//! # What went wrong without this
//!
//! The governor in `omega-server` prices a job correctly and then cannot see the
//! rest of the box. `worker.rs` says so in its own words: a job is priced
//! against "a job's dataset, optimizer state, or *other processes on this
//! host*" — and the only defence against those other processes is
//! `Rejection::MachinePressure`, which reads free memory *after* the fact. That
//! is an observation, not coordination.
//!
//! On a machine running an engine, an emulator and a search sweep at once, three
//! governors that are each individually right will overcommit the machine. Two
//! recorded outcomes: a whole overnight campaign lost to the OOM killer because
//! nothing bounded the *sum* of concurrent workers, and a second incident where
//! two sweeps each concluded independently that they owned the whole box.
//!
//! # The rule
//!
//! One state file per budget. A lock around each transaction. Holders keyed by
//! an identity that survives PID reuse, pruned when the owner is gone. **No
//! daemon** — there is nothing to supervise, nothing to restart, and nothing
//! whose death either frees the machine for everyone or blocks it for everyone.
//! A `SIGKILL`ed holder returns its tokens on the next scan because its identity
//! stops matching, not because anything noticed it die.
//!
//! # Off unless asked
//!
//! Nothing here runs unless an operator names a state file. Absence is a normal
//! state, exactly as it is for [`omega_core::admission::Admission`]: an
//! unconfigured process performs no lock, no read and no syscall, and behaves
//! byte-for-byte as it did before this crate existed. That is deliberate — a
//! resource manager that changes behaviour by being linked is one nobody can
//! adopt incrementally.
//!
//! # What it does not do
//!
//! No queue. A caller that queues instead of failing turns an admission problem
//! into an unbounded-queue problem, and this crate keeps `omega-server`'s
//! existing answer: refuse, with the numbers, and let the caller decide. Bounded
//! queueing is specified elsewhere in this repository and belongs there, not in
//! a file on disk.
//!
//! No cross-machine budget, no cgroup enforcement, no preemption, no priority
//! classes. The threat model is **cooperative processes on one box**: this is
//! accounting between programs that want to co-exist, not a sandbox.
//!
//! # The contract lives in `PROTOCOL.md`
//!
//! This crate is the reference implementation, not the definition. A second
//! implementation is expected in Python — the workers that most need throttling
//! are usually the ones that cannot import a crate — and the two are held
//! together by the conformance list in `PROTOCOL.md` §9 rather than by anyone
//! reading this file and copying its structure. If the two disagree, the model
//! in `proofs/tla/HostGate.tla` decides.

#[cfg(feature = "admission")]
pub mod admission;
pub mod config;
pub mod gate;
pub mod identity;
pub mod ledger;

use std::collections::BTreeMap;
use std::fmt;

pub use gate::{AxisReport, Grant, HostGate};
pub use identity::{HolderId, Liveness};

/// A named resource. There is no single budgeted quantity on a real machine,
/// and pretending otherwise is what produces a gate that guards the wrong thing:
/// a dense statevector engine dies on bytes, a single-threaded emulator dies on
/// cores while memory looks fine, and a metered vendor dies on quota.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum Axis {
    /// Host memory, in bytes.
    HostBytes,
    /// One GPU's memory, in bytes, priced against *that* device.
    DeviceBytes(u32),
    /// Schedulable cores. A job that will use eight threads asks for eight.
    Slots,
    /// A named external quota — a vendor's shot budget, an API rate.
    Vendor(String),
}

impl fmt::Display for Axis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Axis::HostBytes => write!(f, "host_bytes"),
            Axis::DeviceBytes(i) => write!(f, "device_bytes[{i}]"),
            Axis::Slots => write!(f, "slots"),
            Axis::Vendor(v) => write!(f, "vendor[{v}]"),
        }
    }
}

/// An amount per axis. Integers only: a budget expressed in floats has to define
/// a canonical form before two languages can share the file, and the estate has
/// already paid once for a float-repr mismatch between a Rust writer and a
/// Python reader.
pub type Amounts = BTreeMap<Axis, u64>;

/// What a request asks for, and who is asking.
#[derive(Clone, Debug)]
pub struct Request {
    pub amounts: Amounts,
    /// Free text, for `status` and for a human reading the ledger. Never parsed.
    pub tag: String,
}

impl Request {
    pub fn new(tag: impl Into<String>) -> Self {
        Request {
            amounts: Amounts::new(),
            tag: tag.into(),
        }
    }

    pub fn want(mut self, axis: Axis, amount: u64) -> Self {
        if amount > 0 {
            let slot = self.amounts.entry(axis).or_insert(0);
            // Saturating, like every other total in this crate. A plain `+=`
            // wraps in release builds, and a wrapped request is a SMALL one —
            // so an overflow would be admitted rather than refused, which is
            // the one direction a resource request must never fail in. The
            // ledger already saturates in `charge_holder` and `discharge`;
            // this was the sibling that did not.
            *slot = slot.saturating_add(amount);
        }
        self
    }
}

/// Why a request was not granted.
///
/// Three outcomes, never a boolean. A limiter that answers yes/no cannot express
/// the difference between "come back later" and "this can never fit", and
/// collapsing them is what turns a clear refusal into a hammering retry loop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Larger than the whole budget for this axis. Waiting cannot help.
    /// Maps to `413`, and must never be retried.
    TooLarge { axis: Axis, required: u64, cap: u64 },
    /// It fits, but not alongside what is running now. Maps to `429`.
    Busy {
        axis: Axis,
        required: u64,
        available: u64,
    },
    /// The gate itself could not answer. Fails closed in enforcing mode.
    Unavailable { why: String },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::TooLarge {
                axis,
                required,
                cap,
            } => write!(
                f,
                "{axis}: need {required}, but the whole budget is {cap} — this \
                 cannot fit, and retrying will not change that"
            ),
            Refusal::Busy {
                axis,
                required,
                available,
            } => write!(
                f,
                "{axis}: need {required}, {available} free right now — it fits, \
                 but not alongside what is already running"
            ),
            Refusal::Unavailable { why } => {
                write!(f, "the host gate could not answer: {why}")
            }
        }
    }
}

impl Refusal {
    /// Whether waiting could help. The caller uses this to decide between a
    /// retry and giving up, so it is the one bit that must never be guessed.
    pub fn retryable(&self) -> bool {
        matches!(self, Refusal::Busy { .. })
    }
}

/// How the gate behaves when it is configured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Mode {
    /// Grant everything, touch nothing. The default, and what an unconfigured
    /// process does.
    #[default]
    Off,
    /// Record, report, but never refuse. For measuring a box before enforcing
    /// on it — observations are kept apart from the enforcing ledger so an
    /// advisory process cannot spend a budget it is not honouring.
    Advisory,
    /// Refuse when the budget is spent.
    Enforce,
}

impl Mode {
    /// Parse an operator's value. A malformed setting is an error rather than a
    /// silent fallback: a throttle that quietly turns itself off looks exactly
    /// like one that is working.
    pub fn parse(s: &str) -> Result<Mode, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "" => Ok(Mode::Off),
            "advisory" => Ok(Mode::Advisory),
            "enforce" => Ok(Mode::Enforce),
            other => Err(format!(
                "unrecognised mode {other:?} — expected one of: off, advisory, enforce"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refusal_says_which_limit_and_whether_waiting_helps() {
        let too_big = Refusal::TooLarge {
            axis: Axis::HostBytes,
            required: 64 << 30,
            cap: 32 << 30,
        };
        assert!(!too_big.retryable());
        let s = too_big.to_string();
        assert!(s.contains("host_bytes"), "{s}");
        assert!(
            s.contains("68719476736"),
            "the number needed must be in it: {s}"
        );

        let busy = Refusal::Busy {
            axis: Axis::Slots,
            required: 8,
            available: 2,
        };
        assert!(busy.retryable());
        assert!(busy.to_string().contains("slots"));
    }

    #[test]
    fn a_malformed_mode_is_an_error_not_a_silent_off() {
        assert_eq!(Mode::parse("enforce"), Ok(Mode::Enforce));
        assert_eq!(Mode::parse(" Advisory "), Ok(Mode::Advisory));
        assert_eq!(Mode::parse(""), Ok(Mode::Off));
        let err = Mode::parse("enforcing").unwrap_err();
        assert!(
            err.contains("enforcing"),
            "the bad value must be quoted: {err}"
        );
        assert!(err.contains("off, advisory, enforce"), "{err}");
    }

    #[test]
    fn requests_accumulate_per_axis_and_ignore_zero() {
        let r = Request::new("sweep")
            .want(Axis::HostBytes, 1 << 30)
            .want(Axis::HostBytes, 1 << 30)
            .want(Axis::Slots, 4)
            .want(Axis::Slots, 0);
        assert_eq!(r.amounts.get(&Axis::HostBytes), Some(&(2 << 30)));
        assert_eq!(r.amounts.get(&Axis::Slots), Some(&4));
        assert_eq!(r.amounts.len(), 2);
    }

    #[test]
    fn an_overflowing_request_saturates_rather_than_wrapping_to_something_small() {
        // The hazard is not that it wraps, it is WHERE it lands. MAX + MAX
        // wraps to MAX-1, which is still refused; MAX + 2 wraps to 1, which is
        // granted instantly. So test the pair that lands small — a request for
        // an absurd amount must never become a request for one byte.
        let r = Request::new("absurd")
            .want(Axis::HostBytes, u64::MAX)
            .want(Axis::HostBytes, 2);
        assert_eq!(
            r.amounts.get(&Axis::HostBytes),
            Some(&u64::MAX),
            "an overflowing request wrapped to a small one, and a small one is \
             admitted without a second thought"
        );
    }

    #[test]
    fn axes_render_the_way_an_operator_would_grep_for_them() {
        assert_eq!(Axis::HostBytes.to_string(), "host_bytes");
        assert_eq!(Axis::DeviceBytes(1).to_string(), "device_bytes[1]");
        assert_eq!(Axis::Slots.to_string(), "slots");
        assert_eq!(Axis::Vendor("mimiq".into()).to_string(), "vendor[mimiq]");
    }
}
