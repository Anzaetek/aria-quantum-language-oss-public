// SPDX-License-Identifier: Apache-2.0
//! Who holds a grant, and whether they are still alive.
//!
//! # Why a PID is not an identity
//!
//! A ledger keyed by PID alone has two failure modes, and both of them hand a
//! live process's budget to somebody else.
//!
//! **Recycled PIDs.** A holder dies, the kernel reissues its number, and a scan
//! that asks only "does some process have this number" concludes the dead
//! holder is alive and keeps its tokens charged forever. On this laptop
//! `kern.maxproc` is 9000, so the wrap is hours, not months.
//!
//! **Reboots.** A state file that outlives a reboot describes holders that
//! cannot exist, and every one of them is charged.
//!
//! The first closes by keying on `(pid, start_time)`: a recycled PID has a
//! different start time, and the start time is free on both platforms because it
//! is already in the row we have to read.
//!
//! **The second does NOT close that way on Linux, and an earlier version of this
//! comment claimed it did.** Field 22 of `/proc/<pid>/stat` is clock ticks
//! *since boot*, so a process started five seconds into one boot and another
//! started five seconds into the next both read about 500 at 100 Hz — and
//! early-boot daemons draw low, repeatable PIDs besides. `(pid, start_time)` can
//! therefore alias across a reboot, silently, in the worst direction: a stale
//! record matches a live unrelated process, so it is never pruned and its tokens
//! stay charged forever.
//!
//! What actually closes it is [`crate::config::boot_id`], recorded in the
//! ledger's `instance` field: a ledger stamped with a different boot is known to
//! describe processes that cannot exist, and its holders are dropped wholesale.
//! macOS gets this right by accident — `lstart` is a wall-clock date, so it does
//! differ across boots — but relying on that would be relying on an accident of
//! one platform.
//!
//! # "I cannot tell" is not "dead"
//!
//! This is the rule that matters most, and the prior art in this estate gets it
//! wrong in a way that destroys other processes' state.
//!
//! A gate that fails closed must never let an unreadable probe mean "dead",
//! because pruning a *live* holder does not merely lose accounting — it hands
//! that holder's memory to the next request, on a machine that is already at its
//! limit. So [`Liveness::Unknown`] is treated as alive everywhere in this crate:
//! `/proc/<pid>` present but its `stat` unreadable (hardened `hidepid`, or
//! another user's process), a `ps` that fails to run, a platform we do not
//! recognise. The cost of being wrong that way is a budget that is too small
//! for a while; the cost of the other way is the machine.
//!
//! Zombies are the one case that IS dead: the process table entry survives but
//! the memory is gone, so a zombie holder's tokens must come back.
//!
//! # Batched on purpose
//!
//! `probe` takes every PID at once because the macOS path forks, and a fork per
//! holder per scan would put a process spawn on the hot path of a gate whose
//! entire job is to be cheaper than the work it governs. One `ps` covers the
//! whole ledger. Linux needs no fork at all.

use std::collections::HashMap;
#[cfg(target_os = "linux")]
use std::path::Path;
#[cfg(target_os = "macos")]
use std::process::Command;

/// What a scan concluded about one holder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Liveness {
    /// Running, with the start time that pins its identity.
    Alive { start: u64 },
    /// Gone, or a zombie — either way its memory is not held any more.
    Dead,
    /// Could not be determined. **Treated as alive.** See the module note.
    Unknown,
}

impl Liveness {
    /// The only question callers should ask. `Unknown` is alive.
    pub fn holds(self) -> bool {
        !matches!(self, Liveness::Dead)
    }
}

/// A holder's identity: the PID plus the start time that makes it unambiguous.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HolderId {
    pub pid: u32,
    pub start: u64,
}

impl HolderId {
    /// True when `probe` says this exact process — not merely this PID — is
    /// still there. A different start time is a different process that happens
    /// to have inherited the number.
    pub fn matches(&self, live: Liveness) -> bool {
        match live {
            Liveness::Alive { start } => start == self.start,
            Liveness::Unknown => true,
            Liveness::Dead => false,
        }
    }
}

/// Probe every PID in one pass.
pub fn probe(pids: &[u32]) -> HashMap<u32, Liveness> {
    if pids.is_empty() {
        return HashMap::new();
    }
    #[cfg(target_os = "linux")]
    {
        probe_linux(pids)
    }
    #[cfg(target_os = "macos")]
    {
        probe_macos(pids)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        // Unrecognised platform: we cannot tell, so nothing is ever pruned and
        // the gate degrades to a budget that only grows. The caller refuses to
        // run in enforcing mode on such a platform rather than pretending.
        pids.iter().map(|p| (*p, Liveness::Unknown)).collect()
    }
}

/// This process's own start time, for stamping a grant we are about to write.
pub fn self_start() -> Option<u64> {
    let me = std::process::id();
    match probe(&[me]).get(&me) {
        Some(Liveness::Alive { start }) => Some(*start),
        _ => None,
    }
}

// ---------------------------------------------------------------- linux

/// Field 22 of `/proc/<pid>/stat` is the start time in clock ticks since boot.
///
/// **Not platform-gated, deliberately.** The file read is Linux-only; this is a
/// string going in and a `Liveness` coming out, and the parsing is where the
/// bugs are — the same split `omega_core::hostmem::parse_mem_available` already
/// makes for `/proc/meminfo`.
///
/// It was gated once, and the gate cost exactly what it looks like it would: the
/// test below never COMPILED on the machine this crate was written on, so a
/// fixture that was four fields short survived a full green suite, a review, and
/// a cross-machine hand-off before a Linux box ran it for the first time. A
/// platform-gated test is not a weak test — it is absent code that reads as
/// present.
///
/// **The parse is the whole thing.** Field 2 is `comm`, it is unquoted, and it
/// may contain spaces and parentheses — a process really can be named
/// `") Z ("`. Splitting the line on whitespace therefore mis-indexes every
/// later field, and the result is a plausible, stable, wrong number that no
/// test notices. So: split once on the LAST `)`, and count from there. After
/// that split the remaining fields begin at field 3, which puts field 22 at
/// index 19.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_stat(stat: &str) -> Liveness {
    let Some((_, rest)) = stat.rsplit_once(')') else {
        return Liveness::Unknown;
    };
    let mut it = rest.split_whitespace();
    // Field 3 is the run state; 'Z' is a zombie — the entry is there, the
    // memory is not.
    match it.next() {
        Some("Z") => return Liveness::Dead,
        Some(_) => {}
        None => return Liveness::Unknown,
    }
    // field 22 = index 19 of `rest`, and we have consumed index 0 already.
    match it.nth(18).and_then(|v| v.parse::<u64>().ok()) {
        Some(start) => Liveness::Alive { start },
        None => Liveness::Unknown,
    }
}

#[cfg(target_os = "linux")]
fn probe_linux(pids: &[u32]) -> HashMap<u32, Liveness> {
    pids.iter()
        .map(|&pid| {
            let dir = format!("/proc/{pid}");
            let v = if !Path::new(&dir).exists() {
                Liveness::Dead
            } else {
                match std::fs::read_to_string(format!("{dir}/stat")) {
                    Ok(s) => parse_stat(&s),
                    // It existed a moment ago and does not now: it exited
                    // between the two calls. Reporting Unknown here would be
                    // conservative in the wrong place — the tokens would sit
                    // charged until something else happened to prune them, and
                    // a leak whose trigger is a race is the hard kind to
                    // reproduce later.
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Liveness::Dead,
                    // Present but unreadable — hidepid, another user's process.
                    // That is the case that must never prune.
                    Err(_) => Liveness::Unknown,
                }
            };
            (pid, v)
        })
        .collect()
}

// ---------------------------------------------------------------- macos

/// One `ps` for the whole ledger.
///
/// `ps -o pid=,lstart=` reports every user's processes, which is what a
/// host-wide budget needs — a per-user view would prune the neighbours.
/// `lstart` is a wall-clock date string; it is used only for EQUALITY against
/// what was recorded, never as a clock, so an NTP step cannot make two
/// different processes compare equal. A process's own `lstart` does not change.
///
/// Zombies: `ps` reports state `Z` for them, but `lstart=` alone does not carry
/// state, so `stat=` is requested too.
#[cfg(target_os = "macos")]
fn probe_macos(pids: &[u32]) -> HashMap<u32, Liveness> {
    let (out, ok) = ps_batch(pids);
    if ok {
        return out;
    }
    // `ps` refused the batch, so its silence means nothing. Falling through to
    // "absent from the listing = dead" here would mark EVERY holder dead at
    // once, which is the mass-erasure failure this crate exists to prevent.
    //
    // Returning all-Unknown instead would be safe but would never make
    // progress: one malformed record would blind every future scan and the
    // budget would fill until it refused everything. So ask about each PID on
    // its own, where a bad one can only spoil its own answer.
    eprintln!(
        "[hostgate] batched process probe was refused; falling back to \
         one-at-a-time for {} holder(s)",
        pids.len()
    );
    pids.iter()
        .map(|&pid| {
            let (one, ok) = ps_batch(&[pid]);
            let v = one.get(&pid).copied().unwrap_or(Liveness::Unknown);
            // A PID `ps` will not even accept as an argument — out of range,
            // malformed — cannot name a live process on this machine, and the
            // ledger's boot stamp guarantees it is this machine's ledger.
            (pid, if ok { v } else { Liveness::Dead })
        })
        .collect()
}

/// One `ps` for a set of PIDs. The bool is whether the answer can be trusted.
///
/// # The trap this exists to avoid
///
/// `ps -p` validates its whole argument list before printing anything. Hand it
/// one PID it will not accept — out of range is the reachable case — and it
/// exits non-zero having printed **nothing at all**, with the complaint on
/// stderr. `Command::output()` still returns `Ok`, because the command ran.
///
/// So "no lines" has two meanings that look identical: *nobody you asked about
/// is alive*, and *I refused to answer*. Reading the second as the first prunes
/// every live holder in one scan.
///
/// Measured on macOS 15, because the distinction is not in any manual page:
///
/// ```text
/// ps -p <live>,<reaped>          rc=0  "<live> Ss"        absent PID simply omitted
/// ps -p <zombie>,<reaped>        rc=0  "<zombie> Z"       zombies still reported
/// ps -p <reaped>                 rc=1  ""                 nothing found, and that is the answer
/// ps -p <live>,999999            rc=1  ""   stderr: "process id too large"
/// ```
///
/// A merely absent PID does **not** poison the batch. Only an unacceptable one
/// does, and stderr is what tells the two apart.
#[cfg(target_os = "macos")]
fn ps_batch(pids: &[u32]) -> (HashMap<u32, Liveness>, bool) {
    let mut out: HashMap<u32, Liveness> = HashMap::new();
    let list = pids
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let res = Command::new("/bin/ps")
        .args(["-o", "pid=,stat=,lstart=", "-p", &list])
        .output();
    let Ok(res) = res else {
        // ps did not run at all. We cannot tell, so nobody is pruned.
        return (pids.iter().map(|p| (*p, Liveness::Unknown)).collect(), true);
    };
    if !res.stderr.is_empty() {
        // It complained, so the empty listing is a refusal and not a census.
        return (HashMap::new(), false);
    }
    for line in String::from_utf8_lossy(&res.stdout).lines() {
        let line = line.trim();
        let mut it = line.split_whitespace();
        let (Some(pid), Some(state)) = (it.next(), it.next()) else {
            continue;
        };
        let Ok(pid) = pid.parse::<u32>() else {
            continue;
        };
        let rest: String = it.collect::<Vec<_>>().join(" ");
        if state.starts_with('Z') {
            out.insert(pid, Liveness::Dead);
        } else if rest.is_empty() {
            out.insert(pid, Liveness::Unknown);
        } else {
            out.insert(
                pid,
                Liveness::Alive {
                    start: hash_start(&rest),
                },
            );
        }
    }
    // ps answered without complaint, so anything it did not mention is gone.
    // This is determinate evidence and the entire basis of reclamation.
    for &pid in pids {
        out.entry(pid).or_insert(Liveness::Dead);
    }
    (out, true)
}

/// Fold a start-time string into the u64 the ledger stores.
///
/// FNV-1a: stable across runs and processes, which is the only property
/// required — this is an equality tag, never an ordering or a time.
///
/// Ungated for the same reason as `parse_stat`: pure logic tested on one
/// platform only is logic tested nowhere the day that platform is not the one
/// you are on.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn hash_start(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // 0 is reserved for "no start time recorded", so never return it.
    if h == 0 {
        1
    } else {
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_is_treated_as_alive_because_pruning_a_live_holder_is_the_worse_error() {
        assert!(Liveness::Unknown.holds());
        assert!(Liveness::Alive { start: 7 }.holds());
        assert!(!Liveness::Dead.holds());
    }

    #[test]
    fn a_recycled_pid_does_not_inherit_the_dead_holders_grant() {
        let held = HolderId {
            pid: 4242,
            start: 111,
        };
        // Same number, different process.
        assert!(!held.matches(Liveness::Alive { start: 222 }));
        assert!(held.matches(Liveness::Alive { start: 111 }));
        assert!(!held.matches(Liveness::Dead));
        // And an unreadable probe never evicts.
        assert!(held.matches(Liveness::Unknown));
    }

    #[test]
    fn this_process_is_alive_and_has_a_start_time() {
        let me = std::process::id();
        let seen = probe(&[me]);
        let v = seen
            .get(&me)
            .copied()
            .expect("our own pid must be reported");
        assert!(v.holds(), "this process must not read as dead: {v:?}");
        assert!(
            matches!(v, Liveness::Alive { .. }),
            "expected a start time for our own pid, got {v:?}"
        );
        assert!(self_start().is_some());
    }

    #[test]
    fn a_probe_that_runs_and_does_not_find_the_pid_means_dead() {
        // PROTOCOL.md 5.1: determinate absence is the entire basis of
        // reclamation, and it is the branch an over-cautious implementer gets
        // wrong in the opposite direction to the one this estate already got
        // wrong — reading it as indeterminate produces a gate that only fills.
        // 0 is never a user process on either platform.
        let seen = probe(&[0]);
        assert_eq!(seen.get(&0).copied(), Some(Liveness::Dead));
    }

    #[test]
    fn the_start_tag_matches_the_published_vector() {
        // PROTOCOL.md 5.3 names this exact construction because the value is
        // compared BETWEEN implementations sharing one ledger. If two of them
        // hash differently, every foreign record looks like a recycled PID and
        // the prune reclaims all of them — mass erasure through the front door,
        // by two implementations that each passed their own tests.
        //
        // So this is a published vector, not a self-consistency check: anyone
        // reimplementing must reproduce these bytes.
        assert_eq!(
            hash_start("Thu Aug 21 09:00:00 2026"),
            0x36a9_c87d_7b2c_b6cf,
            "the canonical start tag changed — every existing ledger's identities \
             are now unrecognisable, and any other implementation disagrees"
        );
        assert_eq!(hash_start(""), 0xcbf2_9ce4_8422_2325, "bare offset basis");
        assert_ne!(
            hash_start("Thu Aug 21 09:00:00 2026"),
            hash_start("Thu Aug 21 09:00:01 2026")
        );
        // 0 means "no start time recorded" everywhere else in the crate, so the
        // hash must never produce it — a holder stamped 0 is unverifiable.
        assert_ne!(hash_start(""), 0);
    }

    #[test]
    fn one_unacceptable_pid_does_not_mark_every_other_holder_dead() {
        // The regression for a defect this crate shipped: `ps -p` validates its
        // whole argument list first, so a single out-of-range PID makes it exit
        // non-zero having printed NOTHING. Reading that silence as a census
        // marks every live holder dead in one scan — the exact mass erasure
        // this crate criticises in its predecessors.
        //
        // 999999 is past macOS's PID_MAX, and a record carrying it is reachable
        // from a hand-edited or corrupted ledger.
        let me = std::process::id();
        let seen = probe(&[me, 999_999]);
        assert!(
            seen.get(&me).copied().unwrap_or(Liveness::Dead).holds(),
            "a live holder was pruned because another record was malformed"
        );
        assert!(
            matches!(seen.get(&me).copied(), Some(Liveness::Alive { .. })),
            "and it must still carry a usable start time, not degrade to Unknown"
        );
        // The unacceptable one cannot name a live process on this machine.
        assert_eq!(seen.get(&999_999).copied(), Some(Liveness::Dead));
    }

    #[test]
    fn probing_nothing_costs_nothing() {
        assert!(probe(&[]).is_empty());
    }

    #[test]
    fn comm_containing_parens_and_spaces_does_not_shift_the_field_index() {
        // A process really can be named this, and splitting on whitespace
        // would take "Z" from the name rather than the state field.
        //
        // Fields 3..22 in order: state ppid pgrp session tty_nr tpgid flags
        // minflt cminflt majflt cmajflt utime stime cutime cstime priority nice
        // num_threads itrealvalue starttime. An earlier version of this fixture
        // omitted utime/stime/cutime/cstime, which put 987654 at field 18 and
        // asserted against a line no kernel produces. It survived because the
        // test was Linux-gated and never compiled here.
        let line = "1234 () Z (weird) S 1 1234 1234 0 -1 4194304 100 0 0 0 \
                    0 0 0 0 20 0 1 0 987654 1 2 3 4 5";
        match parse_stat(line) {
            Liveness::Alive { start } => assert_eq!(start, 987654),
            other => panic!("expected a start time, got {other:?}"),
        }
    }

    #[test]
    fn a_zombie_is_dead_even_though_its_pid_still_resolves() {
        let line = "1234 (worker) Z 1 1234 1234 0 -1 4194304 100 0 0 0 \
                    0 0 0 0 20 0 1 0 987654 1 2 3 4 5";
        assert_eq!(parse_stat(line), Liveness::Dead);
    }
}
