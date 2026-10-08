// SPDX-License-Identifier: Apache-2.0
//! §4.4a: how much CPU work that is NOT the lane's ran during a row.
//!
//! §4.4 voided a row whose 1-minute load average crossed the host threshold.
//! The hazard it guards is contention from someone else's work — item 15's
//! numbers moved 29% while another session held 28 of 32 cores — but the load
//! average cannot tell that apart from the lane's own threads. On akilles a
//! 16–32-thread dense row at 24 qubits lifts load1 past 8.0 by itself, so the
//! old rule voided correctly-run rows for running.
//!
//! The census integrates over the row instead of sampling two instants:
//! `/proc/stat`'s aggregate busy time (user + nice + system + irq + softirq +
//! steal; exited processes included, because the kernel never forgets their
//! ticks) minus the busy time of the lane's own pids (`/proc/<pid>/stat`
//! utime + stime, all threads), divided by wall time.
//!
//! # macOS
//!
//! There is no `/proc` and no Mach binding here (this crate carries no
//! `libc`, the same decision `omega-hostgate` documents), so the macOS path
//! samples `ps -axo pid=,time=` at the two ends of the row instead of
//! integrating. It is a worse measurement and it is published as one — the
//! `method` field says which path ran — but it is not optional, because the
//! quantity it replaces is worse still. Measured on andromeda a minute apart
//! with nothing but the user's desktop running:
//!
//! | source                   | reading                          |
//! |--------------------------|----------------------------------|
//! | `uptime` load1           | 3.28, then 1.92                  |
//! | this census, 10 s window | 1.36 cores                       |
//! | `top -l 2`               | 15.96% of 10 cores = 1.60 cores  |
//!
//! The census and `top` agree within a quarter of a core; load1 disagrees with
//! both by a factor of two and moves 70% in a minute, because the Mach
//! run-queue count is not a count of running threads. Against a 2.0 threshold
//! that number refuses to publish on a box that is 85% idle, which is why the
//! stabilizer lane could produce no row on andromeda at all: `wait_quiet`
//! blocked for five minutes and then voided everything.
//!
//! The two paths have **different and opposite** error directions, which is
//! why `method` is in the row rather than in this comment:
//!
//! * `ps` cannot see a process that exited inside the window, so the system
//!   total is short and external CPU is **under**-stated. `/proc/stat` has no
//!   such gap — the kernel keeps an exited process's ticks.
//! * Both paths count only the listed pids as the lane's own, so a child
//!   process the lane spawned and did not declare is counted as external, and
//!   external CPU is **over**-stated. That one is shared, and it is the safe
//!   direction: it voids rather than admits.
//!
//! So a macOS row's figure is not a bound in either direction, and the method
//! string says so. It is a measurement of CPU contention, which is what the
//! rule asks for and what the load average on this platform does not provide.

use std::time::Instant;

use crate::row::ExternalCpu;

/// Printed into a macOS row. Names the two error directions, because they
/// point opposite ways and a reader cannot correct for one without the other.
pub const EXTERNAL_CPU_METHOD_PS: &str =
    "sample: (Σ ps cpu time over all pids, after − before) − (same for the lane's own \
     pids), over the row's wall time; macOS, no /proc. Two-instant sample, not an \
     integral: a process that exited inside the window is lost, which understates \
     external CPU, while an undeclared child of the lane counts as external, which \
     overstates it";

/// Printed into every Linux row that carries an [`ExternalCpu`].
pub const EXTERNAL_CPU_METHOD: &str =
    "integral: Δ(/proc/stat busy ticks) − Δ(utime+stime of the lane's own pids), \
     over the row's wall time, in USER_HZ ticks; exited processes count (kernel aggregate)";

/// Which measurement the box supports. Chosen by whether `/proc/stat` can be
/// read, not by `cfg!`: a Linux container with `/proc` masked should fall back
/// rather than fail, and the fallback is a real measurement.
enum Backing {
    /// `/proc/stat` and `/proc/<pid>/stat`, integrated. Ticks.
    ProcStat {
        system0: u64,
        own0: u64,
        hz: f64,
        /// Children of the own tree alive when the census opened, with the
        /// CPU they had already used (see [`live_children`]).
        children0: Vec<ChildAtStart>,
    },
    /// `ps -axo pid=,time=`, sampled. Seconds.
    Ps { system0: f64, own0: f64 },
}

/// A census started at the beginning of a row.
pub struct CpuCensus {
    own: Vec<u32>,
    t0: Instant,
    backing: Backing,
}

impl CpuCensus {
    /// Start counting. `own` is the lane's tree: its own pid and every arm
    /// process it launched. Anything else on the box is external.
    pub fn start(own: &[u32]) -> Result<Self, String> {
        if own.is_empty() {
            return Err("an empty own tree would count the lane as external".into());
        }
        let backing = if std::path::Path::new("/proc/stat").exists() {
            Backing::ProcStat {
                system0: system_busy()?,
                own0: tree_ticks(own)?,
                hz: user_hz()?,
                children0: live_children(own)?,
            }
        } else {
            let table = ps_table()?;
            Backing::Ps {
                system0: ps_total_cpu_s(&table),
                own0: ps_own_cpu_s(&table, own),
            }
        };
        Ok(Self {
            own: own.to_vec(),
            t0: Instant::now(),
            backing,
        })
    }

    /// Stop counting and report.
    ///
    /// **Waits, if the row was faster than the counters can resolve.** Both
    /// paths read a cumulative per-process CPU counter — `/proc`'s at
    /// `USER_HZ` (10 ms at 100 Hz), `ps`'s at a centisecond on this macOS and
    /// coarser on others — so a census over a short row quantises to a
    /// multiple of one tick divided by the window, and the most likely
    /// multiple is **zero**. A 13 ms window reporting `0.0` external cores is
    /// not a quiet box; it is a window too short to hold a tick. That number
    /// was published on a real row before this floor existed:
    /// `stab-clifford-24q-d100-z0-expectation` at `window_s 0.013668` and
    /// `external_cores_during_row 0.0`, on a box the pre-row census had just
    /// measured at 0.607 cores.
    ///
    /// Refusing a short row instead would void every fast row in the
    /// comparison, and the fast rows are most of it. So the window is padded
    /// to [`CENSUS_MIN_WINDOW_S`] and the padding is counted: external work
    /// during the padding is external work next to the row, which is the
    /// hazard being tested for. Padding cannot manufacture quiet — it can only
    /// fail to resolve a spike shorter than the floor — and it is outside the
    /// timed region, so it costs the measurement nothing but wall clock.
    pub fn finish(self) -> Result<ExternalCpu, String> {
        let short_by = CENSUS_MIN_WINDOW_S - self.t0.elapsed().as_secs_f64();
        if short_by > 0.0 {
            std::thread::sleep(std::time::Duration::from_secs_f64(short_by));
        }
        match self.backing {
            Backing::ProcStat {
                system0,
                own0,
                hz,
                children0,
            } => {
                let system1 = system_busy()?;
                // Read the reaped set BEFORE the own tree's counters: a child
                // reaped between the two reads would otherwise have its
                // cutime credited without its pre-window ticks taken back.
                let pre_window = pre_window_ticks_of_reaped(&children0);
                let own1 = tree_ticks(&self.own)?.saturating_sub(pre_window);
                let wall = self.t0.elapsed().as_secs_f64();
                Ok(external_cpu_since(
                    system1.saturating_sub(system0),
                    own1.saturating_sub(own0),
                    hz,
                    wall,
                    &self.own,
                ))
            }
            Backing::Ps { system0, own0 } => {
                let table = ps_table()?;
                let wall = self.t0.elapsed().as_secs_f64();
                Ok(external_cpu_sampled(
                    ps_total_cpu_s(&table) - system0,
                    ps_own_cpu_s(&table, &self.own) - own0,
                    wall,
                    &self.own,
                ))
            }
        }
    }
}

/// The arithmetic, separated so it is testable without a box.
pub fn external_cpu_since(
    system_ticks: u64,
    own_ticks: u64,
    hz: f64,
    wall_s: f64,
    own: &[u32],
) -> ExternalCpu {
    let external = system_ticks.saturating_sub(own_ticks) as f64 / hz;
    ExternalCpu {
        external_cores_during_row: if wall_s > 0.0 {
            external / wall_s
        } else {
            f64::INFINITY
        },
        window_s: wall_s,
        method: EXTERNAL_CPU_METHOD.to_string(),
        own_pids: own.to_vec(),
        external_cores_before: None,
        before_window_s: None,
    }
}

/// The macOS arithmetic, separated so it is testable without a box.
///
/// Clamps at zero in both numerator terms. `ps` losing one of the lane's own
/// children makes `own_s` overshoot, and reporting the result as negative
/// external CPU would read as positive evidence that the box was quiet, which
/// is the worst available answer.
pub fn external_cpu_sampled(system_s: f64, own_s: f64, wall_s: f64, own: &[u32]) -> ExternalCpu {
    let external = (system_s.max(0.0) - own_s.max(0.0)).max(0.0);
    ExternalCpu {
        external_cores_during_row: if wall_s > 0.0 {
            external / wall_s
        } else {
            f64::INFINITY
        },
        window_s: wall_s,
        method: EXTERNAL_CPU_METHOD_PS.to_string(),
        own_pids: own.to_vec(),
        external_cores_before: None,
        before_window_s: None,
    }
}

/// One row of `ps -axo pid=,time=`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsRow {
    /// The process.
    pub pid: u32,
    /// CPU-seconds it has consumed, all threads.
    pub cpu_s: f64,
}

fn ps_table() -> Result<Vec<PsRow>, String> {
    let out = std::process::Command::new("ps")
        .args(["-axo", "pid=,time="])
        .output()
        .map_err(|e| format!("ps: {e}"))?;
    if !out.status.success() {
        return Err(format!("ps exited {:?}", out.status.code()));
    }
    let rows = parse_ps_table(&String::from_utf8_lossy(&out.stdout));
    if rows.is_empty() {
        return Err("ps returned no parseable rows; a census of nothing is not a census".into());
    }
    Ok(rows)
}

/// Parse the whole `ps` table, skipping rows that do not parse rather than
/// failing: `ps` races process exit and can print a truncated line, and
/// voiding a row over a cosmetic one would be the load-average mistake again.
pub fn parse_ps_table(text: &str) -> Vec<PsRow> {
    let mut out = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(pid), Some(time)) = (it.next(), it.next()) else {
            continue;
        };
        let (Ok(pid), Some(cpu_s)) = (pid.parse::<u32>(), parse_ps_time(time)) else {
            continue;
        };
        out.push(PsRow { pid, cpu_s });
    }
    out
}

/// `ps`'s cumulative CPU time → seconds.
///
/// Accepts every shape the platforms print: `0:00.01` (mm:ss.cc),
/// `284:25.07` (macOS past an hour of CPU — still mm:ss, the minutes grow),
/// `00:00:12` (hh:mm:ss) and `2-03:04:05` (days).
///
/// Read right-to-left, which is what makes one function cover all four: the
/// last field is always seconds, the one before it minutes, then hours, then
/// days. Left-to-right would need to know which shape it was looking at, and
/// guessing wrong on launchd's `284:25.07` turns 4.7 hours into 11.9 days — a
/// 60x overstatement of a process that is in every census.
pub fn parse_ps_time(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let parts: Vec<&str> = s.split(['-', ':']).collect();
    if parts.len() > 4 {
        return None;
    }
    let mut total = 0.0;
    for (i, part) in parts.iter().rev().enumerate() {
        let v: f64 = part.parse().ok()?;
        if v < 0.0 {
            return None;
        }
        total += v * match i {
            0 => 1.0,
            1 => 60.0,
            2 => 3600.0,
            _ => 86400.0,
        };
    }
    Some(total)
}

/// Every process's CPU time, summed. The macOS analogue of `/proc/stat`'s
/// aggregate line, and short by exactly the processes that have exited.
pub fn ps_total_cpu_s(rows: &[PsRow]) -> f64 {
    rows.iter().map(|r| r.cpu_s).sum()
}

/// The declared pids' CPU time.
///
/// Only the declared pids, matching the Linux path's `/proc/<pid>/stat` read:
/// both cover every thread of a listed process and neither follows into a
/// child the lane did not declare. Keeping the two the same matters more than
/// making one of them cleverer, because a row's number has to mean the same
/// thing on both boxes.
pub fn ps_own_cpu_s(rows: &[PsRow], own: &[u32]) -> f64 {
    rows.iter()
        .filter(|r| own.contains(&r.pid))
        .map(|r| r.cpu_s)
        .sum()
}

fn system_busy() -> Result<u64, String> {
    let s = std::fs::read_to_string("/proc/stat")
        .map_err(|e| format!("/proc/stat: {e} (the external-CPU census is Linux-only)"))?;
    let line = s.lines().next().ok_or("/proc/stat is empty")?;
    let f: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .map(|x| x.parse().map_err(|e| format!("/proc/stat field {x}: {e}")))
        .collect::<Result<_, _>>()?;
    if f.len() < 8 {
        return Err(format!("/proc/stat cpu line has {} fields", f.len()));
    }
    // user nice system idle iowait irq softirq steal: busy excludes idle, iowait.
    Ok(f[0] + f[1] + f[2] + f[5] + f[6] + f[7])
}

/// A child of the own tree that was alive when a census opened.
#[derive(Debug, Clone, Copy)]
struct ChildAtStart {
    pid: u32,
    /// Field 22, so a recycled pid is not mistaken for the same child.
    starttime: u64,
    /// utime + stime + cutime + cstime at the start: what the parent's
    /// cutime/cstime will gain when it reaps this child, minus whatever the
    /// child does inside the window.
    ticks: u64,
}

/// `(ppid, starttime, utime+stime+cutime+cstime)` of one `/proc/<pid>/stat`.
fn stat_fields(pid: u32) -> Option<(u32, u64, u64, char)> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rest = s.rsplit_once(')')?.1;
    let f: Vec<&str> = rest.split_whitespace().collect();
    // rest starts at field 3 (state): ppid is field 4, utime..cstime 14..17,
    // starttime 22.
    let state = f.first()?.chars().next()?;
    let ppid = f.get(1)?.parse().ok()?;
    let sum: u64 = (11..=14)
        .map(|k| f.get(k)?.parse::<u64>().ok())
        .sum::<Option<u64>>()?;
    let starttime = f.get(19)?.parse().ok()?;
    Some((ppid, starttime, sum, state))
}

/// Children of the own tree alive now. **Why this exists:** a parent's
/// cutime/cstime grow when it *reaps* a child, by everything the child ever
/// used, including CPU it burned before the census opened. Counting cutime as
/// the lane's own (needed so an arm the lane starts and reaps inside the row
/// is not read as somebody else's work) would then credit the lane with
/// pre-window work, and since external is `all − own` clamped at zero, that
/// over-statement reads as a quiet box — the answer that admits a row. So the
/// census records each live child's ticks at the start and takes them back if
/// the child is reaped inside the window.
fn live_children(own: &[u32]) -> Result<Vec<ChildAtStart>, String> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir("/proc").map_err(|e| format!("/proc: {e}"))? {
        let Ok(entry) = entry else { continue };
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<u32>().ok())
        else {
            continue;
        };
        if let Some((ppid, starttime, ticks, _)) = stat_fields(pid) {
            if own.contains(&ppid) {
                out.push(ChildAtStart {
                    pid,
                    starttime,
                    ticks,
                });
            }
        }
    }
    Ok(out)
}

/// Pre-window ticks of the children that have been reaped since the start
/// (gone, or the pid now names a different process). A child that exited but
/// is not yet reaped (state `Z`) has not been credited to its parent, so it is
/// not taken back.
fn pre_window_ticks_of_reaped(children0: &[ChildAtStart]) -> u64 {
    children0
        .iter()
        .filter(|c| match stat_fields(c.pid) {
            None => true,
            Some((_, starttime, _, _)) => starttime != c.starttime,
        })
        .map(|c| c.ticks)
        .sum()
}

fn tree_ticks(own: &[u32]) -> Result<u64, String> {
    let mut total = 0u64;
    for pid in own {
        let s = std::fs::read_to_string(format!("/proc/{pid}/stat"))
            .map_err(|e| format!("/proc/{pid}/stat: {e} (an own-tree process exited mid-row)"))?;
        // Field 2 (comm) may contain spaces; parse after the last ')'.
        let rest = s.rsplit_once(')').ok_or("malformed /proc/pid/stat")?.1;
        let f: Vec<&str> = rest.split_whitespace().collect();
        // rest starts at field 3 (state); utime is field 14, stime 15.
        let utime: u64 = f
            .get(11)
            .ok_or("no utime")?
            .parse()
            .map_err(|e| format!("utime: {e}"))?;
        let stime: u64 = f
            .get(12)
            .ok_or("no stime")?
            .parse()
            .map_err(|e| format!("stime: {e}"))?;
        // cutime/cstime (fields 16, 17): children the process has waited
        // for. An arm the lane starts and reaps inside the row is the lane's
        // own tree; without these it would read as somebody else's work.
        let cutime: u64 = f
            .get(13)
            .ok_or("no cutime")?
            .parse()
            .map_err(|e| format!("cutime: {e}"))?;
        let cstime: u64 = f
            .get(14)
            .ok_or("no cstime")?
            .parse()
            .map_err(|e| format!("cstime: {e}"))?;
        total += utime + stime + cutime + cstime;
    }
    Ok(total)
}

fn user_hz() -> Result<f64, String> {
    let out = std::process::Command::new("getconf")
        .arg("CLK_TCK")
        .output()
        .map_err(|e| format!("getconf CLK_TCK: {e}"))?;
    let hz: f64 = String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .map_err(|e| format!("CLK_TCK: {e}"))?;
    if hz <= 0.0 {
        return Err(format!("CLK_TCK {hz}"));
    }
    Ok(hz)
}

/// What the pre-row quiet check measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuietProbe {
    /// Cores held by work outside the lane during the probe.
    pub cores: f64,
    /// How long the probe ran, seconds.
    pub window_s: f64,
    /// The 1-minute load average when the probe passed. Recorded as
    /// `load1_before` for continuity, and on macOS it is context only — see
    /// the module docs for why it is not the gate.
    pub load1: f64,
}

/// The shortest window a census is allowed to speak about, seconds.
///
/// Set by the counters' resolution, not by taste: at `USER_HZ` = 100 a tick is
/// 10 ms, so a 2 s window resolves external CPU to 0.005 cores — two orders
/// of magnitude under the smallest threshold (2.0). At 0.2 s it would resolve
/// to 0.05 cores, which is still fine, and at the 13 ms a real row produced it
/// resolves to 0.77 cores, which is not. Two seconds leaves the margin on the
/// right side of the smallest interesting quantity.
pub const CENSUS_MIN_WINDOW_S: f64 = 2.0;

/// The shortest pre-row probe that is worth reporting. Below a second the
/// macOS path's centisecond resolution and the scheduler's own jitter are a
/// large fraction of the threshold.
pub const PROBE_WINDOW_S: f64 = 2.0;

/// Wait until work outside the lane is under `void_above` cores, and report
/// the census that said so.
///
/// This replaces a `wait_quiet` that looped on the 1-minute load average.
/// Waiting on load1 while *publishing* on external CPU blocks a lane on a
/// condition its own rows do not require — and on macOS it blocks it
/// permanently, because load1 there is not a count of running threads. A lane
/// should wait on the quantity it will be judged by.
pub fn wait_quiet_external(
    void_above: f64,
    own: &[u32],
    load1: impl Fn() -> Result<f64, String>,
) -> Result<QuietProbe, String> {
    let start = Instant::now();
    loop {
        let census = CpuCensus::start(own)?;
        std::thread::sleep(std::time::Duration::from_secs_f64(PROBE_WINDOW_S));
        let probe = census.finish()?;
        let cores = probe.external_cores_during_row;
        if !crate::load::void_at_load(cores, void_above) {
            return Ok(QuietProbe {
                cores,
                window_s: probe.window_s,
                load1: load1()?,
            });
        }
        if start.elapsed() > std::time::Duration::from_secs(300) {
            return Err(format!(
                "work outside the lane held {cores:.2} cores (threshold {void_above:.2}) for \
                 5 minutes"
            ));
        }
        eprintln!("outside the lane: {cores:.2} cores > {void_above:.2}, waiting");
        std::thread::sleep(std::time::Duration::from_secs(20));
    }
}

impl ExternalCpu {
    /// Attach the pre-row census, so `check` tests it instead of
    /// `load1_before`.
    pub fn with_pre_row(mut self, probe: &QuietProbe) -> Self {
        self.external_cores_before = Some(probe.cores);
        self.before_window_s = Some(probe.window_s);
        self
    }
}

/// §4.4a for a row with no `LoadRecord` (a capability row), or any row a
/// lane must refuse before writing: `Some(reason)` when work outside the lane held
/// more than `void_above` cores before the row or during it. A row without a
/// pre-row census is judged on the during-row integral alone.
pub fn external_void_reason(
    row_id: &str,
    external: &ExternalCpu,
    void_above: f64,
) -> Option<String> {
    let during = external.external_cores_during_row;
    let before = external.external_cores_before;
    let hot_before = before.is_some_and(|b| crate::load::void_at_load(b, void_above));
    if hot_before || crate::load::void_at_load(during, void_above) || !during.is_finite() {
        Some(format!(
            "{row_id}: {:.2} cores outside the lane before and {during:.2} during, above {void_above}; not publishing",
            before.unwrap_or(f64::NAN)
        ))
    } else {
        None
    }
}
