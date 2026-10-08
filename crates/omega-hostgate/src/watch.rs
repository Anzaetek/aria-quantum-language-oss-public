// SPDX-License-Identifier: Apache-2.0
//! Opt-in resident-memory watch for `omega-hostgate run --watch`.
//!
//! The ledger records a declaration. It does not read what the child
//! allocates, and that is the contract for every caller that does not pass
//! `--watch`. This module is the opt-in that turns the declaration into a
//! cap: after the child has been spawned, poll its **process tree** and
//! SIGKILL the tree when the summed resident set exceeds the declared
//! `--host-bytes`.
//!
//! # A poll is not a limit
//!
//! The sample interval is the hole. A process that allocates past the cap
//! and is gone again before the next sample — touch, free, or exit, all
//! inside one interval — is not seen. The default interval is 250 ms.
//! Shared pages are counted in every process that has them mapped, so the
//! sum can sit above the number of unique resident bytes; that error is
//! toward killing early, which is the direction a cap may be wrong in.
//!
//! # An unreadable census is not zero
//!
//! The failure this exists to catch is a watchdog that reported 17 MB while
//! the worker, a child of the process it was watching, held 3.7 GiB. The
//! same shape happens if the probe fails and the failure is read as an
//! empty tree: the log says the declaration held, and the machine is the
//! one that finds out it did not.
//!
//! So a census that did not run, that complained, or that cannot read an
//! entry which might be in the tree is **fail closed**. The tree is
//! SIGKILLed and the run exits [`EXIT_CENSUS_FAILED`] (4), the code the
//! gate already uses when it cannot answer. It is not warn-and-continue.
//! Continuing is the declaration-only behaviour `--watch` was added to
//! leave behind, with a log line the caller may not be reading, for as
//! long as the probe stays broken — and a probe that fails under memory
//! pressure fails at the moment the cap is the only thing left.
//!
//! That is the opposite of the liveness rule in [`crate::identity`], on
//! purpose. There, "I cannot tell" must not mean "dead", because the
//! destructive act (reclaiming a grant) hands a live process's memory to
//! the next request. Here the destructive act in the other direction is
//! the OOM the flag exists to prevent, and the caller asked for a cap.
//! Determinate absence is still not this case: a probe that ran, did not
//! complain, and does not list a process `wait` says has exited is an
//! exit, and the child's own status is returned.
//!
//! # Platforms
//!
//! * **macOS** — one `ps -axo pid=,ppid=,rss=,stat=` for the whole table,
//!   then the parent chain. The same two failures [`crate::identity`]
//!   already separates: `ps` did not run at all, and `ps` complained. Neither
//!   is a census.
//! * **Linux** — walk `/proc/<pid>/stat` by ppid and read RSS from
//!   `statm`. When the child already sits in a cgroup v2 leaf whose
//!   `cgroup.procs` is entirely inside that tree, `memory.current` replaces
//!   the sum: it does not race with a fork between two reads. A cgroup is
//!   never created and `memory.max` is never written.
//!
//!   **Verified on akilles (Linux 6.x, systemd, cgroup v2) at 8019131**, with
//!   the two branches discriminated rather than assumed: under a delegated
//!   `Delegate=yes` scope with the job in its own exclusive leaf, a 500 MB
//!   file write was killed at a reported peak of 514.4 M and the leaf's
//!   `memory.current` read 539361280 afterwards, so the counter in play was
//!   `memory.current` and not the `statm` sum; with the same job left beside
//!   the watcher so the leaf is NOT exclusive, the reading fell back to the
//!   `/proc` sum and reported 5.1 M. The headline shape (parent 2 MB,
//!   grandchild ballooning, 300 M declared) was killed at 361.6 M, exit 5,
//!   1.05 s from launch, with no member of the tree alive 2 s later.
//!
//!   **The two branches do not measure the same thing, and that asymmetry is
//!   deliberate but surprising.** `memory.current` charges page cache;
//!   `statm` resident does not. So a job in an exclusive leaf that merely
//!   *writes or reads a large file* can be killed for cache it never held as
//!   RSS — that is exactly what the 500 MB test above did — while the same
//!   job outside an exclusive leaf would not be. Callers who want RSS
//!   semantics should not place the job in its own leaf; callers who want
//!   cgroup-charged memory should. Stated here because the same command
//!   behaving differently by cgroup placement is the kind of thing that
//!   otherwise gets discovered in an incident.
//!
//! # Kill order
//!
//! The tree is snapshotted first, while the root is still alive, so ppids
//! stay put for the duration of the walk. Then every descendant is
//! SIGKILLed, then the root. Killing the root first would reparent the
//! children mid-walk and the snapshot would miss them; a parent left alive
//! until the end of the snapshot cannot be reaped out from under the walk.
//! A fork in the gap between the snapshot and the root's death can still
//! escape that one pass. The child is spawned as its own process-group
//! leader so a final group signal catches a grandchild that inherited the
//! group. A grandchild that has called `setsid` is reached only if it was
//! in the snapshot.

use std::collections::{HashMap, HashSet};
use std::io::ErrorKind;
use std::process::{Child, ExitStatus};
use std::time::Duration;

/// Default gap between samples. A spike that fits inside this window is not
/// seen; that limit is documented on the flag, not papered over.
pub const DEFAULT_INTERVAL: Duration = Duration::from_millis(250);

/// `run --watch` killed the tree because resident memory exceeded the
/// declared `--host-bytes`. Not the command's own status. Documented next
/// to exit 3 (refused) and exit 4 (fail closed).
pub const EXIT_KILLED: u8 = 5;

/// The census could not be taken. Same code as the gate's fail-closed exit:
/// we could not answer, so the run is not reported as having stayed inside
/// the declaration. The tree is SIGKILLed first.
pub const EXIT_CENSUS_FAILED: u8 = 4;

/// Why a sample is not a measurement. Display text is the contract the
/// tests pin: it must say that this is not a reading of zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CensusFailure {
    /// `ps` (or the platform probe) was not executed. Distinct from a probe
    /// that ran and refused.
    DidNotRun { why: String },
    /// The probe ran and complained, or its listing is not a census. An
    /// empty table in this state is a refusal, not "nothing is resident".
    Refused { why: String },
    /// A process entry exists but cannot be read. It might be the child
    /// holding the memory.
    Unreadable { pid: u32, why: String },
}

impl std::fmt::Display for CensusFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CensusFailure::DidNotRun { why } => write!(
                f,
                "ps did not run at all ({why}); this is not a census and not a reading of zero"
            ),
            CensusFailure::Refused { why } => write!(
                f,
                "ps complained ({why}); the listing is a refusal and not a census, not a reading of zero"
            ),
            CensusFailure::Unreadable { pid, why } => write!(
                f,
                "/proc/{pid} is unreadable ({why}); this is not a census and not a reading of zero"
            ),
        }
    }
}

/// One row of a process table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcSnap {
    pub pid: u32,
    pub ppid: u32,
    pub rss_bytes: u64,
    pub zombie: bool,
}

/// A process we could see the directory of, but not read. Carried separately
/// so a vanished pid (determinate absence) is not the same event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnreadableProc {
    pub pid: u32,
    pub why: String,
}

/// What one sample concluded about `root`'s tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Observation {
    /// The probe ran and did not list `root`. Only means "exited" when
    /// `wait` agrees; see [`supervise`].
    Exited,
    Alive {
        rss_bytes: u64,
        /// Root included. Descendants follow. This is the kill list.
        pids: Vec<u32>,
    },
}

/// Sum `root` and every descendant in `rows`.
///
/// `unreadable` fails the sample even when the listed rows would have been
/// under the cap. An entry we cannot read might be the grandchild the
/// listed rows do not name; dropping it on the floor is the 17 MB reading.
///
/// A pid that has already vanished is simply not in `rows`. That is
/// determinate absence, the same distinction [`crate::identity`] draws for
/// `/proc/<pid>` disappearing between the directory existing and the read.
pub fn fold_tree(
    root: u32,
    rows: &[ProcSnap],
    unreadable: &[UnreadableProc],
) -> Result<Observation, CensusFailure> {
    if let Some(bad) = unreadable.first() {
        return Err(CensusFailure::Unreadable {
            pid: bad.pid,
            why: bad.why.clone(),
        });
    }

    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut rss: HashMap<u32, u64> = HashMap::new();
    let mut zombie = HashSet::new();
    let mut saw_root = false;
    for row in rows {
        if row.pid == root {
            saw_root = true;
            if row.zombie {
                // The command has exited. Its children have been, or are
                // about to be, reparented; `wait` is the authority.
                return Ok(Observation::Exited);
            }
        }
        children.entry(row.ppid).or_default().push(row.pid);
        if row.zombie {
            zombie.insert(row.pid);
        } else {
            rss.insert(row.pid, row.rss_bytes);
        }
    }
    if !saw_root {
        return Ok(Observation::Exited);
    }

    // Descendants, not just `root`. Stopping here — pushing nothing — is
    // the measurement that reports the parent's RSS and misses the worker.
    let mut stack = vec![root];
    let mut seen = HashSet::new();
    let mut pids = Vec::new();
    let mut total = 0u64;
    while let Some(pid) = stack.pop() {
        if !seen.insert(pid) {
            continue;
        }
        pids.push(pid);
        if !zombie.contains(&pid) {
            total = total.saturating_add(rss.get(&pid).copied().unwrap_or(0));
        }
        if let Some(kids) = children.get(&pid) {
            for &kid in kids {
                stack.push(kid);
            }
        }
    }
    Ok(Observation::Alive {
        rss_bytes: total,
        pids,
    })
}

/// What the supervisor should do with one sample. A [`CensusFailure`] never
/// becomes [`Reaction::Continue`] with a zero reading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reaction {
    Continue { rss_bytes: u64 },
    OverCap { rss_bytes: u64, pids: Vec<u32> },
    Exited,
    FailClosed { why: String },
}

pub fn react(sample: Result<Observation, CensusFailure>, limit_bytes: u64) -> Reaction {
    match sample {
        Err(e) => Reaction::FailClosed { why: e.to_string() },
        Ok(Observation::Exited) => Reaction::Exited,
        Ok(Observation::Alive { rss_bytes, pids }) => {
            if rss_bytes > limit_bytes {
                Reaction::OverCap { rss_bytes, pids }
            } else {
                Reaction::Continue { rss_bytes }
            }
        }
    }
}

/// How a watched run ended. The receipt is printed for every variant.
#[derive(Clone, Debug)]
pub enum WatchEnd {
    /// The child exited on its own. `status` is the child's.
    Finished {
        status: ExitStatus,
        peak_bytes: u64,
        samples: u32,
    },
    /// Resident memory went over the declaration. The tree was SIGKILLed.
    Exceeded { peak_bytes: u64 },
    /// The census failed. The tree was SIGKILLed. Not a zero reading.
    CensusFailed { peak_bytes: u64, why: String },
}

impl WatchEnd {
    pub fn peak_bytes(&self) -> u64 {
        match self {
            WatchEnd::Finished { peak_bytes, .. }
            | WatchEnd::Exceeded { peak_bytes }
            | WatchEnd::CensusFailed { peak_bytes, .. } => *peak_bytes,
        }
    }
}

pub fn parse_interval_ms(s: &str) -> Result<Duration, String> {
    let n: u64 = s
        .parse()
        .map_err(|_| format!("{s:?} is not a whole number of milliseconds"))?;
    if n == 0 {
        return Err("watch interval must be at least 1 ms — 0 would busy-loop the sampler".into());
    }
    Ok(Duration::from_millis(n))
}

/// Poll `child` until it exits or the watch stops it.
///
/// `own_process_group` is true when the child was spawned with
/// `process_group(0)`, so its pid is the process-group id. The final
/// signal goes to that group. Passing true for a child that shares the
/// caller's group would signal the caller.
/// How many extra samples to take when the census does not list a root that
/// `wait` reports alive.
///
/// Three, not one and not thirty. One is what the supervisor did before and it
/// kills a healthy run on a single raced listing. A long retry loop is worse
/// than either: a run that is genuinely blowing past its declaration also
/// produces unreadable censuses while the OOM killer works, so a patient
/// retry turns the cap off at exactly the moment it is needed. Three samples
/// at [`CENSUS_DISAGREEMENT_DELAY`] bound the extra exposure to a fraction of
/// a second while covering a race that lasts one probe.
pub const CENSUS_DISAGREEMENT_RETRIES: u32 = 3;

/// Gap between those samples. Long enough that `wait` can catch up with a
/// child that really did exit, short enough that a process over its cap is
/// not left running for long.
pub const CENSUS_DISAGREEMENT_DELAY: Duration = Duration::from_millis(50);

pub fn supervise(
    child: &mut Child,
    limit_bytes: u64,
    interval: Duration,
    own_process_group: bool,
) -> std::io::Result<WatchEnd> {
    supervise_with(child, limit_bytes, interval, own_process_group, sample_tree)
}

fn supervise_with(
    child: &mut Child,
    limit_bytes: u64,
    interval: Duration,
    own_process_group: bool,
    mut sample: impl FnMut(u32) -> Result<Observation, CensusFailure>,
) -> std::io::Result<WatchEnd> {
    let root = child.id();
    let mut peak = 0u64;
    let mut samples = 0u32;
    'watch: loop {
        match react(sample(root), limit_bytes) {
            Reaction::Continue { rss_bytes } => {
                peak = peak.max(rss_bytes);
                samples = samples.saturating_add(1);
            }
            Reaction::OverCap { rss_bytes, pids } => {
                peak = peak.max(rss_bytes);
                signal_kill_tree(root, &pids, own_process_group);
                reap(child)?;
                return Ok(WatchEnd::Exceeded { peak_bytes: peak });
            }
            Reaction::FailClosed { why } => {
                // No pid list: the census is exactly what we do not have.
                // Kill the root, and the group if the child leads one, so
                // a grandchild is not left running under a blind watch.
                signal_kill_tree(root, &[root], own_process_group);
                reap(child)?;
                return Ok(WatchEnd::CensusFailed {
                    peak_bytes: peak,
                    why,
                });
            }
            Reaction::Exited => {
                if let Some(status) = child.try_wait()? {
                    return Ok(WatchEnd::Finished {
                        status,
                        peak_bytes: peak,
                        samples,
                    });
                }
                // `wait` says the child is alive and the census does not
                // contain it. That is not an exit and not a zero reading --
                // but it is also not, on its own, a broken census. It is two
                // observations of one fact disagreeing, and the probe has a
                // race that produces exactly this: it enumerates the process
                // table, then reads each entry, and a process that exits
                // between those two steps is absent from the listing while
                // `wait` has not yet caught up. A dense run on akilles was
                // killed this way *after* cargo had printed `Finished`
                // (receipt: /tmp/dense-run-censusfail.log).
                //
                // So re-sample before killing. Failing closed on a census
                // that cannot be read stays exactly as it was -- an unknown
                // footprint is not a small one -- and a `FailClosed` or
                // `OverCap` seen during a retry is acted on immediately. What
                // changes is that a *transient* disagreement no longer kills
                // a healthy run, and the kill message says how many samples
                // were taken before it gave up, so the next reader of that
                // log can tell a race from a broken probe.
                for _ in 0..CENSUS_DISAGREEMENT_RETRIES {
                    std::thread::sleep(CENSUS_DISAGREEMENT_DELAY);
                    if let Some(status) = child.try_wait()? {
                        return Ok(WatchEnd::Finished {
                            status,
                            peak_bytes: peak,
                            samples,
                        });
                    }
                    match react(sample(root), limit_bytes) {
                        Reaction::Continue { rss_bytes } => {
                            peak = peak.max(rss_bytes);
                            samples = samples.saturating_add(1);
                            continue 'watch;
                        }
                        Reaction::OverCap { rss_bytes, pids } => {
                            peak = peak.max(rss_bytes);
                            signal_kill_tree(root, &pids, own_process_group);
                            reap(child)?;
                            return Ok(WatchEnd::Exceeded { peak_bytes: peak });
                        }
                        Reaction::FailClosed { why } => {
                            signal_kill_tree(root, &[root], own_process_group);
                            reap(child)?;
                            return Ok(WatchEnd::CensusFailed {
                                peak_bytes: peak,
                                why,
                            });
                        }
                        // Still absent. Try again, or fall through below.
                        Reaction::Exited => {}
                    }
                }
                let why = format!(
                    "census did not include live pid {root} on {} consecutive samples over {:?}; the process is still running, so absence is not an exit and not a reading of zero",
                    CENSUS_DISAGREEMENT_RETRIES + 1,
                    CENSUS_DISAGREEMENT_DELAY * CENSUS_DISAGREEMENT_RETRIES
                );
                signal_kill_tree(root, &[root], own_process_group);
                reap(child)?;
                return Ok(WatchEnd::CensusFailed {
                    peak_bytes: peak,
                    why,
                });
            }
        }
        if let Some(status) = child.try_wait()? {
            return Ok(WatchEnd::Finished {
                status,
                peak_bytes: peak,
                samples,
            });
        }
        std::thread::sleep(interval);
    }
}

fn reap(child: &mut Child) -> std::io::Result<()> {
    match child.wait() {
        Ok(_) => Ok(()),
        // Already reaped by a try_wait that raced the kill, or the child
        // exited in the gap. Nothing left to collect.
        Err(e) if e.kind() == ErrorKind::InvalidInput => Ok(()),
        Err(e) => Err(e),
    }
}

/// SIGKILL descendants, then the root, then the process group when the
/// child leads one.
pub fn signal_kill_tree(root: u32, pids: &[u32], own_process_group: bool) {
    for &pid in pids {
        if pid != root {
            sigkill(pid);
        }
    }
    sigkill(root);
    if own_process_group {
        sigkill_group(root);
    }
}

fn sigkill(pid: u32) {
    // pid 0 is "every process in this group" and pid 1 is init. A bug in
    // the walk must not become a bug that takes the machine down.
    if pid <= 1 {
        return;
    }
    unsafe {
        // SAFETY: `pid` is a positive i32-sized process id, signal 9 is
        // SIGKILL. The call does not dereference memory. ESRCH (already
        // gone) is the success case of a second pass and is ignored.
        kill(pid as i32, SIGKILL);
    }
}

fn sigkill_group(pgid: u32) {
    if pgid <= 1 {
        return;
    }
    unsafe {
        // SAFETY: a negative pid is POSIX for "the process group". `pgid`
        // is the child's own pid, and only when that child was placed in
        // a new group at spawn. We never pass the caller's pgid.
        kill(-(pgid as i32), SIGKILL);
    }
}

const SIGKILL: i32 = 9;

unsafe extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}

/// One sample of `root`'s tree on this platform.
pub fn sample_tree(root: u32) -> Result<Observation, CensusFailure> {
    #[cfg(target_os = "macos")]
    {
        sample_macos(root)
    }
    #[cfg(target_os = "linux")]
    {
        sample_linux(root)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = root;
        Err(CensusFailure::DidNotRun {
            why: "no process-tree census on this platform".into(),
        })
    }
}

// ---------------------------------------------------------------- receipt

pub fn receipt_line(peak_bytes: u64, declared_bytes: u64, end: &WatchEnd) -> String {
    let peak_h = human_bytes(peak_bytes);
    let declared_h = human_bytes(declared_bytes);
    match end {
        WatchEnd::Finished { samples, .. } if *samples == 0 => format!(
            "omega-hostgate: watch: no sample was taken before the command exited \
             (peak unknown, not a reading of zero), declared {declared_bytes} bytes \
             ({declared_h}) — a spike inside the first interval would not have been seen"
        ),
        WatchEnd::Finished { .. } => format!(
            "omega-hostgate: watch: peak {peak_bytes} bytes ({peak_h}) resident, \
             declared {declared_bytes} bytes ({declared_h}) — stayed inside the declaration"
        ),
        WatchEnd::Exceeded { .. } => format!(
            "omega-hostgate: watch: peak {peak_bytes} bytes ({peak_h}) resident, \
             declared {declared_bytes} bytes ({declared_h}) — exceeded the declaration; \
             process tree killed"
        ),
        WatchEnd::CensusFailed { why, .. } => format!(
            "omega-hostgate: watch: peak {peak_bytes} bytes ({peak_h}) resident, \
             declared {declared_bytes} bytes ({declared_h}) — census failed ({}); \
             not a reading of zero; process tree killed",
            one_line(why)
        ),
    }
}

fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn human_bytes(v: u64) -> String {
    const UNITS: [&str; 5] = ["B", "K", "M", "G", "T"];
    let mut f = v as f64;
    let mut u = 0;
    while f >= 1024.0 && u < UNITS.len() - 1 {
        f /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{v} {unit}", unit = UNITS[u])
    } else {
        format!("{f:.1} {}", UNITS[u])
    }
}

// ---------------------------------------------------------------- parsers
//
// Ungated, like `identity::parse_stat`. A platform-gated parser is absent
// code on the machine that does not run that platform, which is how a
// short fixture once survived every review this crate had.

/// Field 4 of `/proc/<pid>/stat` is ppid. Field 2 (`comm`) may contain
/// spaces and parentheses, so the split is on the last `)`, same as
/// [`crate::identity`].
pub fn ppid_from_stat(stat: &str) -> Option<u32> {
    let (_, rest) = stat.rsplit_once(')')?;
    let mut it = rest.split_whitespace();
    let _state = it.next()?;
    it.next()?.parse().ok()
}

pub fn state_is_zombie(stat: &str) -> bool {
    stat.rsplit_once(')')
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .is_some_and(|s| s == "Z")
}

/// `statm` field 2 is resident pages.
pub fn rss_from_statm(statm: &str, page_size: u64) -> Option<u64> {
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages.saturating_mul(page_size))
}

/// `memory.current` is a byte count. `max` and anything else is "no reading",
/// not zero.
pub fn parse_memory_current(text: &str) -> Option<u64> {
    let t = text.trim();
    if t.is_empty() || t == "max" {
        return None;
    }
    t.parse().ok()
}

pub fn parse_cgroup_procs(text: &str) -> Vec<u32> {
    text.split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect()
}

/// The cgroup v2 path from `/proc/<pid>/cgroup` (`0::/path`). Hybrid mounts
/// also carry legacy `N:controller:/path` lines; those are not v2.
pub fn parse_cgroup_v2_rel(text: &str) -> Option<&str> {
    text.lines().find_map(|l| l.trim().strip_prefix("0::"))
}

/// Use `memory.current` only when every member of the cgroup is inside the
/// watched tree. A shared cgroup's counter includes the rest of the box,
/// and preferring it would kill the job for someone else's memory.
pub fn exclusive_leaf_bytes(
    memory_current: u64,
    cgroup_pids: &[u32],
    tree_pids: &[u32],
) -> Option<u64> {
    if cgroup_pids.is_empty() || memory_current == 0 {
        return None;
    }
    if cgroup_pids.iter().all(|p| tree_pids.contains(p)) {
        Some(memory_current)
    } else {
        None
    }
}

// macOS-only at runtime; compiled and unit-tested on every platform, the
// same split `identity.rs` makes, so the parse is never absent code.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn parse_ps_line(line: &str) -> Option<ProcSnap> {
    let mut it = line.split_whitespace();
    let pid: u32 = it.next()?.parse().ok()?;
    let ppid: u32 = it.next()?.parse().ok()?;
    let rss_kib: u64 = it.next()?.parse().ok()?;
    let state = it.next()?;
    Some(ProcSnap {
        pid,
        ppid,
        // `ps` rss on macOS is 1024-byte units. See `man ps`.
        rss_bytes: rss_kib.saturating_mul(1024),
        zombie: state.starts_with('Z'),
    })
}

// macOS-only at runtime; compiled and unit-tested on every platform, the
// same split `identity.rs` makes, so the parse is never absent code.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct PsRaw {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Interpret one `ps` invocation the way [`crate::identity`] does.
///
/// Stderr non-empty: the probe complained, so an empty listing is a
/// refusal and not a census. Stdout empty with a quiet probe: also not a
/// census — a machine with no processes is not a reading this crate will
/// treat as "the child has no resident memory".
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn census_from_ps(root: u32, raw: &PsRaw) -> Result<Observation, CensusFailure> {
    if !raw.stderr.is_empty() {
        let why = String::from_utf8_lossy(&raw.stderr).trim().to_string();
        return Err(CensusFailure::Refused {
            why: if why.is_empty() {
                "ps wrote to stderr".into()
            } else {
                why
            },
        });
    }
    if raw.stdout.iter().all(|b| b.is_ascii_whitespace()) {
        return Err(CensusFailure::Refused {
            why: "ps produced no rows; an empty listing is not a census of this machine".into(),
        });
    }
    let text = String::from_utf8_lossy(&raw.stdout);
    let mut rows = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_ps_line(line) {
            Some(row) => rows.push(row),
            None => {
                return Err(CensusFailure::Refused {
                    why: format!("unparseable ps line {line:?}; the listing is not a census"),
                });
            }
        }
    }
    fold_tree(root, &rows, &[])
}

// ---------------------------------------------------------------- macos

#[cfg(target_os = "macos")]
fn sample_macos(root: u32) -> Result<Observation, CensusFailure> {
    let res = std::process::Command::new("/bin/ps")
        .args(["-axo", "pid=,ppid=,rss=,stat="])
        .output();
    let raw = match res {
        Ok(out) => PsRaw {
            stdout: out.stdout,
            stderr: out.stderr,
        },
        // `ps` did not run at all. Distinct from a complaint on stderr.
        Err(e) => return Err(CensusFailure::DidNotRun { why: e.to_string() }),
    };
    census_from_ps(root, &raw)
}

// ---------------------------------------------------------------- linux
//
// UNVERIFIED on the machine that added `--watch` (macOS, 16 GB). The
// parsers above are executed by the unit tests there. This walk was not
// run against a Linux `/proc` or a cgroup mount.

#[cfg(target_os = "linux")]
fn sample_linux(root: u32) -> Result<Observation, CensusFailure> {
    let mut obs = sample_proc(root)?;
    if let Observation::Alive { rss_bytes, pids } = &mut obs {
        // Prefer the cgroup counter when it describes this tree and only
        // this tree. A zero `memory.current` next to a positive RSS sum
        // means the controller is not charging the leaf; substituting it
        // would publish the zero the rest of this module refuses to invent.
        // `exclusive_leaf_bytes` already rejects zero.
        if let Some(bytes) = try_exclusive_cgroup(root, pids) {
            *rss_bytes = bytes;
        }
    }
    Ok(obs)
}

#[cfg(target_os = "linux")]
fn sample_proc(root: u32) -> Result<Observation, CensusFailure> {
    let dir = std::fs::read_dir("/proc").map_err(|e| CensusFailure::DidNotRun {
        why: format!("cannot list /proc: {e}"),
    })?;
    let mut rows = Vec::new();
    let mut unreadable = Vec::new();
    for ent in dir {
        let ent = ent.map_err(|e| CensusFailure::DidNotRun {
            why: format!("cannot list /proc: {e}"),
        })?;
        let Some(pid) = ent.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        match read_proc_row(pid) {
            Ok(Some(row)) => rows.push(row),
            Ok(None) => {}
            Err(bad) => unreadable.push(bad),
        }
    }
    fold_tree(root, &rows, &unreadable)
}

#[cfg(target_os = "linux")]
fn read_proc_row(pid: u32) -> Result<Option<ProcSnap>, UnreadableProc> {
    let stat = match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(s) => s,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(UnreadableProc {
                pid,
                why: e.to_string(),
            })
        }
    };
    let Some(ppid) = ppid_from_stat(&stat) else {
        return Err(UnreadableProc {
            pid,
            why: "stat did not yield a ppid".into(),
        });
    };
    let zombie = state_is_zombie(&stat);
    let statm = match std::fs::read_to_string(format!("/proc/{pid}/statm")) {
        Ok(s) => s,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(UnreadableProc {
                pid,
                why: e.to_string(),
            })
        }
    };
    let rss_bytes = rss_from_statm(&statm, page_size()).unwrap_or(0);
    Ok(Some(ProcSnap {
        pid,
        ppid,
        rss_bytes,
        zombie,
    }))
}

#[cfg(target_os = "linux")]
fn page_size() -> u64 {
    static PAGES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *PAGES.get_or_init(|| {
        // Linux `_SC_PAGESIZE` (bits/confname.h). Not the macOS value.
        const SC_PAGESIZE: i32 = 30;
        unsafe extern "C" {
            fn sysconf(name: i32) -> i64;
        }
        // SAFETY: `sysconf` with `_SC_PAGESIZE` reads a kernel constant and
        // does not dereference caller memory. A non-positive return is
        // "unknown", and the caller substitutes 4096.
        let n = unsafe { sysconf(SC_PAGESIZE) };
        if n > 0 {
            n as u64
        } else {
            4096
        }
    })
}

#[cfg(target_os = "linux")]
fn try_exclusive_cgroup(root: u32, tree_pids: &[u32]) -> Option<u64> {
    let cg = std::fs::read_to_string(format!("/proc/{root}/cgroup")).ok()?;
    let rel = parse_cgroup_v2_rel(&cg)?.trim();
    // The root cgroup is the whole machine. Its `memory.current` is not
    // this job's, even if a membership check were somehow to pass.
    if rel.is_empty() || rel == "/" {
        return None;
    }
    if !rel.starts_with('/') {
        return None;
    }
    let dir = format!("/sys/fs/cgroup{rel}");
    let current = std::fs::read_to_string(format!("{dir}/memory.current")).ok()?;
    let bytes = parse_memory_current(&current)?;
    let procs = std::fs::read_to_string(format!("{dir}/cgroup.procs")).ok()?;
    let members = parse_cgroup_procs(&procs);
    exclusive_leaf_bytes(bytes, &members, tree_pids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(pid: u32, ppid: u32, rss: u64) -> ProcSnap {
        ProcSnap {
            pid,
            ppid,
            rss_bytes: rss,
            zombie: false,
        }
    }

    #[test]
    fn a_grandchild_is_in_the_resident_sum_and_the_root_alone_is_not() {
        // The shape that beat both real guards: the parent is small, the
        // memory is in a child. 2 MB under a 64 MB cap must not be the
        // answer the fold returns.
        let rows = vec![snap(10, 1, 2_000_000), snap(11, 10, 400_000_000)];
        match fold_tree(10, &rows, &[]).unwrap() {
            Observation::Alive { rss_bytes, pids } => {
                assert!(
                    rss_bytes > 64 * 1024 * 1024,
                    "root alone is 2 MB, which is under a 64 MB cap; the \
                     grandchild's 400 MB has to be in the sum, got {rss_bytes}"
                );
                assert!(pids.contains(&10) && pids.contains(&11), "{pids:?}");
            }
            other => panic!("expected a live tree, got {other:?}"),
        }
    }

    #[test]
    fn an_unreadable_proc_entry_is_not_a_zero_reading() {
        let rows = vec![snap(10, 1, 1000)];
        let bad = vec![UnreadableProc {
            pid: 11,
            why: "permission denied".into(),
        }];
        let sample = fold_tree(10, &rows, &bad);
        let err = sample.expect_err("an unreadable entry must not become a sum");
        let msg = err.to_string();
        assert!(
            msg.contains("not a reading of zero"),
            "the failure must say it is not zero: {msg}"
        );
        assert!(
            msg.contains("unreadable"),
            "and which failure it was: {msg}"
        );
        assert!(
            matches!(react(Err(err), 1 << 30), Reaction::FailClosed { .. }),
            "fail closed, not continue-as-zero"
        );
    }

    #[test]
    fn a_ps_that_complained_is_a_refusal_not_an_empty_tree() {
        let raw = PsRaw {
            stdout: Vec::new(),
            stderr: b"ps: process id too large\n".to_vec(),
        };
        let sample = census_from_ps(10, &raw);
        let err = sample.expect_err("stderr means the listing was refused");
        assert!(
            matches!(err, CensusFailure::Refused { .. }),
            "complaining is the refusal case, not 'did not run': {err}"
        );
        let msg = err.to_string();
        assert!(msg.contains("complained"), "{msg}");
        assert!(msg.contains("not a census"), "{msg}");
        assert!(msg.contains("not a reading of zero"), "{msg}");
        assert!(matches!(
            react(Err(err), 64 * 1024 * 1024),
            Reaction::FailClosed { .. }
        ));
    }

    #[test]
    fn a_ps_that_did_not_run_is_not_the_same_failure_as_a_complaint() {
        // identity.rs keeps these apart: one is "we cannot tell", the other
        // is "it refused". Both must fail closed here, and the text must
        // not collapse them — a later reader debugging a blind watch needs
        // to know which one happened.
        let err = CensusFailure::DidNotRun {
            why: "No such file or directory".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("did not run at all"), "{msg}");
        assert!(msg.contains("not a reading of zero"), "{msg}");
        assert!(!msg.contains("complained"), "{msg}");
        assert!(matches!(react(Err(err), 1), Reaction::FailClosed { .. }));
    }

    #[test]
    fn an_empty_ps_listing_is_not_a_census() {
        let raw = PsRaw {
            stdout: b"\n".to_vec(),
            stderr: Vec::new(),
        };
        let err = census_from_ps(10, &raw).unwrap_err();
        assert!(
            matches!(err, CensusFailure::Refused { .. }),
            "silence from ps is not 'the child uses nothing': {err}"
        );
    }

    #[test]
    fn ps_rows_sum_the_grandchild_and_skip_zombies() {
        let raw = PsRaw {
            stdout: b"  10  1  1024 S\n  11  10  204800 Z\n  12  11  307200 S\n".to_vec(),
            stderr: Vec::new(),
        };
        // 1024 KiB + 307200 KiB. The zombie contributes nothing.
        match census_from_ps(10, &raw).unwrap() {
            Observation::Alive { rss_bytes, pids } => {
                assert_eq!(rss_bytes, (1024 + 307_200) * 1024);
                assert!(
                    pids.contains(&12),
                    "the grandchild past the zombie: {pids:?}"
                );
                assert!(
                    pids.contains(&11),
                    "the walk has to pass through the zombie to reach its child: {pids:?}"
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn equal_to_the_cap_is_inside_and_one_byte_over_is_not() {
        let under = react(
            Ok(Observation::Alive {
                rss_bytes: 100,
                pids: vec![1],
            }),
            100,
        );
        assert!(matches!(under, Reaction::Continue { rss_bytes: 100 }));
        let over = react(
            Ok(Observation::Alive {
                rss_bytes: 101,
                pids: vec![4, 5],
            }),
            100,
        );
        match over {
            Reaction::OverCap { rss_bytes, pids } => {
                assert_eq!(rss_bytes, 101);
                assert_eq!(pids, vec![4, 5]);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn comm_with_parens_does_not_shift_ppid_or_hide_a_zombie() {
        let line = "1234 () Z (weird) S 42 1234 1234 0 -1 0 1 0 0 0 0 0 0 0 20 0 1 0 9";
        assert_eq!(ppid_from_stat(line), Some(42));
        assert!(!state_is_zombie(line));
        let zombie = "1234 (worker) Z 7 1 1 0 -1 0 1 0 0 0 0 0 0 0 20 0 1 0 9";
        assert!(state_is_zombie(zombie));
        assert_eq!(ppid_from_stat(zombie), Some(7));
        assert_eq!(rss_from_statm("100 25 10 5 0 15 0", 4096), Some(25 * 4096));
        assert_eq!(parse_memory_current("12345\n"), Some(12345));
        assert_eq!(parse_memory_current("max\n"), None);
        assert_eq!(parse_memory_current("nope"), None);
    }

    #[test]
    fn memory_current_is_used_only_for_an_exclusive_leaf() {
        let tree = vec![10, 11];
        assert_eq!(exclusive_leaf_bytes(5000, &[10, 11], &tree), Some(5000));
        assert_eq!(
            exclusive_leaf_bytes(5000, &[10, 11, 99], &tree),
            None,
            "a shared cgroup counts the neighbours"
        );
        assert_eq!(
            exclusive_leaf_bytes(0, &[10], &tree),
            None,
            "a zero counter must not replace a real sum"
        );
        assert_eq!(exclusive_leaf_bytes(5000, &[], &tree), None);
        let hybrid = "12:memory:/foo\n0::/user.slice/session.scope\n";
        assert_eq!(
            parse_cgroup_v2_rel(hybrid),
            Some("/user.slice/session.scope")
        );
        assert_eq!(parse_cgroup_procs("10\n11\n"), vec![10, 11]);
    }

    #[test]
    fn a_zero_interval_is_rejected() {
        assert!(parse_interval_ms("0").is_err());
        assert_eq!(
            parse_interval_ms("250").unwrap(),
            Duration::from_millis(250)
        );
        assert!(parse_interval_ms("fast").is_err());
    }

    #[test]
    fn receipts_name_the_peak_the_declaration_and_which_way_it_went() {
        let stayed = receipt_line(
            8_000_000,
            256 * 1024 * 1024,
            &WatchEnd::Finished {
                status: finished_status(),
                peak_bytes: 8_000_000,
                samples: 3,
            },
        );
        assert!(stayed.contains("8000000 bytes"), "{stayed}");
        assert!(stayed.contains("268435456 bytes"), "{stayed}");
        assert!(stayed.contains("stayed inside the declaration"), "{stayed}");

        let exceeded = receipt_line(
            300_000_000,
            64 * 1024 * 1024,
            &WatchEnd::Exceeded {
                peak_bytes: 300_000_000,
            },
        );
        assert!(exceeded.contains("300000000 bytes"), "{exceeded}");
        assert!(exceeded.contains("67108864 bytes"), "{exceeded}");
        assert!(exceeded.contains("exceeded the declaration"), "{exceeded}");
        assert!(exceeded.contains("process tree killed"), "{exceeded}");

        let failed = receipt_line(
            0,
            64 * 1024 * 1024,
            &WatchEnd::CensusFailed {
                peak_bytes: 0,
                why: "ps did not run at all (no); this is not a census and not a reading of zero"
                    .into(),
            },
        );
        assert!(failed.contains("not a reading of zero"), "{failed}");
        assert!(
            !failed.contains("stayed inside"),
            "a failed census must not read as a clean bill: {failed}"
        );

        let unseen = receipt_line(
            0,
            100,
            &WatchEnd::Finished {
                status: finished_status(),
                peak_bytes: 0,
                samples: 0,
            },
        );
        assert!(unseen.contains("peak unknown"), "{unseen}");
        assert!(!unseen.contains("stayed inside"), "{unseen}");
    }

    fn finished_status() -> ExitStatus {
        std::process::Command::new("/bin/sleep")
            .arg("0")
            .status()
            .expect("sleep 0")
    }

    #[test]
    fn a_transient_census_miss_does_not_kill_a_healthy_run() {
        // STATUS §5 #20. The probe enumerates the process table and then reads
        // each entry; a process exiting between those two steps is absent from
        // the listing while `wait` has not caught up. Before the retry, that
        // single raced sample killed the tree — a dense run on akilles died
        // this way after cargo had already printed `Finished`.
        //
        // The injected probe misses the root on its first call and sees it
        // afterwards, which is the race. The child is real and short-lived, so
        // a supervisor that killed it would report CensusFailed and a
        // supervisor that rides out the miss reports Finished.
        use std::os::unix::process::CommandExt;
        use std::sync::atomic::{AtomicU32, Ordering};
        let mut cmd = std::process::Command::new("/bin/sleep");
        cmd.arg("1");
        cmd.process_group(0);
        let mut child = cmd.spawn().expect("spawn sleep");
        let calls = AtomicU32::new(0);
        let end = supervise_with(
            &mut child,
            64 * 1024 * 1024,
            Duration::from_millis(20),
            true,
            |root| {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    Ok(Observation::Exited)
                } else {
                    Ok(Observation::Alive {
                        rss_bytes: 1024,
                        pids: vec![root],
                    })
                }
            },
        )
        .expect("supervise");
        match end {
            WatchEnd::Finished { status, .. } => {
                assert!(status.success(), "sleep 1 should exit 0: {status:?}")
            }
            other => panic!(
                "one raced sample must not kill a healthy run; got {other:?}. \
                 Samples taken: {}",
                calls.load(Ordering::SeqCst)
            ),
        }
        assert!(
            calls.load(Ordering::SeqCst) >= 2,
            "the retry never happened, so this test proves nothing"
        );
    }

    #[test]
    fn a_census_that_never_sees_a_live_root_still_fails_closed() {
        // The other half, and the one that must not regress: riding out a
        // transient miss must not become ignoring a persistent one. A probe
        // that never lists the root is a probe whose reading we do not have,
        // and an unknown footprint is not a small one. The child outlives the
        // retries, so `wait` never rescues the supervisor.
        use std::os::unix::process::CommandExt;
        use std::sync::atomic::{AtomicU32, Ordering};
        let mut cmd = std::process::Command::new("/bin/sleep");
        cmd.arg("30");
        cmd.process_group(0);
        let mut child = cmd.spawn().expect("spawn sleep");
        let calls = AtomicU32::new(0);
        let end = supervise_with(
            &mut child,
            64 * 1024 * 1024,
            Duration::from_millis(20),
            true,
            |_root| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(Observation::Exited)
            },
        )
        .expect("supervise");
        match end {
            WatchEnd::CensusFailed { why, peak_bytes } => {
                assert!(why.contains("consecutive samples"), "{why}");
                assert!(why.contains("not a reading of zero"), "{why}");
                assert_eq!(peak_bytes, 0);
            }
            other => panic!("a persistent census miss must fail closed; got {other:?}"),
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            CENSUS_DISAGREEMENT_RETRIES + 1,
            "the supervisor must take exactly one sample plus its retries before killing"
        );
    }

    #[test]
    fn a_retry_that_finds_the_run_over_its_cap_kills_it_without_waiting() {
        // A disagreement is not a licence to stop enforcing. If a retry sees
        // the root and it is over the declaration, that is the cap's own case
        // and it fires immediately rather than finishing the retry budget.
        use std::os::unix::process::CommandExt;
        use std::sync::atomic::{AtomicU32, Ordering};
        let mut cmd = std::process::Command::new("/bin/sleep");
        cmd.arg("30");
        cmd.process_group(0);
        let mut child = cmd.spawn().expect("spawn sleep");
        let calls = AtomicU32::new(0);
        let end = supervise_with(&mut child, 1024, Duration::from_millis(20), true, |root| {
            if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(Observation::Exited)
            } else {
                Ok(Observation::Alive {
                    rss_bytes: 8 * 1024,
                    pids: vec![root],
                })
            }
        })
        .expect("supervise");
        match end {
            WatchEnd::Exceeded { peak_bytes } => assert_eq!(peak_bytes, 8 * 1024),
            other => panic!("an over-cap retry must kill; got {other:?}"),
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "it should not have kept retrying"
        );
    }

    #[test]
    fn a_failing_census_kills_the_child_instead_of_treating_it_as_idle() {
        // The probe is injected so this does not depend on making `/bin/ps`
        // fail on a machine where ps works. The child is real: if the
        // supervisor treated the error as zero RSS it would keep sleeping
        // next to it, and this test would still be waiting.
        use std::os::unix::process::CommandExt;
        let mut cmd = std::process::Command::new("/bin/sleep");
        cmd.arg("30");
        cmd.process_group(0);
        let mut child = cmd.spawn().expect("spawn sleep");
        let pid = child.id();
        let end = supervise_with(
            &mut child,
            64 * 1024 * 1024,
            Duration::from_millis(50),
            true,
            |_root| {
                Err(CensusFailure::DidNotRun {
                    why: "simulated: probe not executed".into(),
                })
            },
        )
        .expect("supervise");
        match end {
            WatchEnd::CensusFailed { why, peak_bytes } => {
                assert!(why.contains("did not run at all"), "{why}");
                assert!(why.contains("not a reading of zero"), "{why}");
                assert_eq!(
                    peak_bytes, 0,
                    "a failed sample must not be recorded as a measured zero that then counts as the peak"
                );
                let line = receipt_line(
                    peak_bytes,
                    64 * 1024 * 1024,
                    &WatchEnd::CensusFailed { peak_bytes, why },
                );
                assert!(line.contains("not a reading of zero"), "{line}");
                assert!(!line.contains("stayed inside"), "{line}");
            }
            other => panic!("census failure must fail closed, got {other:?}"),
        }
        assert!(
            !pid_alive(pid),
            "the child must have been SIGKILLed; a warn-and-continue would leave it sleeping"
        );
    }

    fn pid_alive(pid: u32) -> bool {
        let rc = unsafe { kill(pid as i32, 0) };
        rc == 0
    }

    #[test]
    fn this_process_has_a_nonzero_resident_set() {
        // The platform probe, for real, on whatever this test is running
        // on. A zero here is the silent-census bug, not a small process.
        let me = std::process::id();
        match sample_tree(me).expect("census of our own pid") {
            Observation::Alive { rss_bytes, pids } => {
                assert!(pids.contains(&me), "{pids:?}");
                assert!(rss_bytes > 0, "our own RSS read as zero");
            }
            other => panic!("we are alive: {other:?}"),
        }
    }
}
