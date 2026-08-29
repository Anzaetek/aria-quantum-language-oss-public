// SPDX-License-Identifier: Apache-2.0
//! What the host will actually give us, asked of the host rather than assumed.
//!
//! A dense statevector is the one allocation in this workspace whose size is
//! set by the *input* rather than by a configuration knob: `2^n` amplitudes,
//! doubling with every qubit. At 30 qubits that is 16 GiB and at 34 it is 256
//! GiB, so the difference between "runs" and "takes the machine down" is four
//! qubits of user input. A backend that allocates it without asking is not
//! making a performance trade-off, it is gambling with the whole box.
//!
//! This module is the measurement half of the refusal. It answers only what the
//! platform is willing to state, and returns `None` otherwise — a caller must
//! then fall back on its own ceiling rather than invent a number, because a
//! guessed memory figure is worse than no check: it refuses valid work on one
//! machine and admits fatal work on another.
//!
//! The probes live here, in the crate every backend already depends on, so the
//! server's admission control and a bare `omega-run` reach the same numbers by
//! the same route. `omega-server`'s topology module delegates here.

/// Physical RAM, in bytes. `None` if the platform will not say.
pub fn total_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        parse_mem_total(&meminfo)
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// How many times the platform has actually been probed.
///
/// Exists so a test can assert the probe is NOT on a hot path, which is a
/// property no timing assertion can state without becoming a benchmark. See
/// `omega-backend-statevector/tests/capacity_probe_is_not_per_shot.rs`.
pub fn probes() -> u64 {
    PROBES.load(std::sync::atomic::Ordering::Relaxed)
}

static PROBES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How long a reading is reused before the platform is asked again.
///
/// Short enough that a guard still reacts to real memory pressure within a
/// human-noticeable interval, long enough that a per-shot loop asks once rather
/// than once per shot.
const CACHE_TTL: std::time::Duration = std::time::Duration::from_millis(250);

#[allow(clippy::type_complexity)]
static CACHE: std::sync::Mutex<Option<(std::time::Instant, Option<u64>)>> =
    std::sync::Mutex::new(None);

/// What a new allocation could plausibly get *right now*, in bytes.
///
/// Deliberately not `total_bytes()`: the question a capacity guard asks is not
/// how much RAM was installed but how much is obtainable without pushing the
/// machine into swap.
///
/// `None` when the platform will not say.
///
/// # Cached for [`CACHE_TTL`], and that is a correctness-relevant compromise
///
/// **The probe is expensive on both supported platforms and this is called from
/// a per-trajectory guard.** On macOS it forks a `vm_stat` SUBPROCESS; on Linux
/// it reads and parses `/proc/meminfo`. `capacity::check` runs once per
/// trajectory, so a 1000-shot mid-circuit-measurement run forked `vm_stat` one
/// thousand times: measured at **1.05 s for 1000 shots, and identical for a
/// 2-qubit/4-op circuit and a 5-qubit/28-op one** — the cost was entirely the
/// fork and had nothing to do with the simulation. Setting the oversubscribe
/// escape hatch, which returns before this call, took the same run to 0.00 s.
///
/// So "right now" in the sentence above was already a fiction in the only way
/// that mattered: the guard asked a thousand times and every answer was the
/// same, because nothing in a shot loop moves the host's memory.
///
/// The compromise, stated plainly rather than buried: a reading up to 250 ms
/// stale can admit an allocation that no longer fits, or refuse one that now
/// would. That window is far shorter than the time to actually exhaust memory
/// by any route this guard defends against, and the guard was never a
/// substitute for the allocator failing.
pub fn available_bytes() -> Option<u64> {
    let mut cache = match CACHE.lock() {
        Ok(c) => c,
        // A poisoned lock means another thread panicked mid-probe. Fall through
        // to a direct probe rather than propagate a panic out of a guard.
        Err(poisoned) => poisoned.into_inner(),
    };
    if let Some((taken, value)) = *cache {
        if taken.elapsed() < CACHE_TTL {
            return value;
        }
    }
    let fresh = probe_available_bytes();
    *cache = Some((std::time::Instant::now(), fresh));
    fresh
}

/// The uncached platform probe. Separate so [`available_bytes`] is only the
/// caching policy and this is only the measurement.
fn probe_available_bytes() -> Option<u64> {
    PROBES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    #[cfg(target_os = "linux")]
    {
        // `MemAvailable` is the kernel's own estimate of what a new allocation
        // can get without swapping — strictly better than `MemFree`, which
        // ignores reclaimable page cache and reads far too low.
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        parse_mem_available(&meminfo)
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("vm_stat").output().ok()?;
        parse_vm_stat(&String::from_utf8_lossy(&out.stdout))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// `MemTotal:` from `/proc/meminfo`, in bytes. Split out so it is testable on
/// any platform — the parsing is where the bugs are, not the file read.
pub fn parse_mem_total(meminfo: &str) -> Option<u64> {
    parse_meminfo_field(meminfo, "MemTotal:")
}

/// `MemAvailable:` from `/proc/meminfo`, in bytes.
pub fn parse_mem_available(meminfo: &str) -> Option<u64> {
    parse_meminfo_field(meminfo, "MemAvailable:")
}

fn parse_meminfo_field(meminfo: &str, field: &str) -> Option<u64> {
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
            return kb.checked_mul(1024);
        }
    }
    None
}

/// Free + inactive pages from `vm_stat`, in bytes.
///
/// Inactive pages count as available: macOS reclaims them under pressure. Free
/// alone reads catastrophically low on a perfectly healthy Mac — a machine with
/// 20 GiB reclaimable routinely reports a few hundred MiB free — and a guard
/// built on it would refuse everything.
pub fn parse_vm_stat(out: &str) -> Option<u64> {
    let mut page_size: u64 = 4096;
    if let Some(first) = out.lines().next() {
        // "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
        if let Some(i) = first.find("page size of ") {
            let rest = &first[i + 13..];
            if let Some(n) = rest.split_whitespace().next() {
                if let Ok(v) = n.parse::<u64>() {
                    page_size = v;
                }
            }
        }
    }
    let pages = |label: &str| -> u64 {
        out.lines()
            .find(|l| l.starts_with(label))
            .and_then(|l| l.split(':').nth(1))
            .map(|v| v.trim().trim_end_matches('.').replace(',', ""))
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
    };
    let free = pages("Pages free");
    let inactive = pages("Pages inactive");
    if free == 0 && inactive == 0 {
        return None;
    }
    free.checked_add(inactive)?.checked_mul(page_size)
}

/// Render a byte count the way a refusal should state it.
///
/// Scales all the way up because a statevector refusal routinely quotes figures
/// no machine has: printing "137438953472.0 GiB" instead of "128.0 EiB" makes
/// the message look broken rather than emphatic.
pub fn human_bytes(bytes: u128) -> String {
    const UNITS: &[(&str, u32)] = &[
        ("EiB", 60),
        ("PiB", 50),
        ("TiB", 40),
        ("GiB", 30),
        ("MiB", 20),
    ];
    for (label, shift) in UNITS {
        let scale = 1u128 << shift;
        if bytes >= scale {
            return format!("{:.1} {label}", bytes as f64 / scale as f64);
        }
    }
    format!("{bytes} B")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_mem_available_parses() {
        let mi = "MemTotal:       131072000 kB\nMemFree:         1000000 kB\n\
                  MemAvailable:    64000000 kB\n";
        assert_eq!(parse_mem_available(mi), Some(64_000_000 * 1024));
        assert_eq!(parse_mem_total(mi), Some(131_072_000 * 1024));
    }

    /// A kernel that does not publish `MemAvailable` must yield `None`, not a
    /// silent fallback to `MemFree` — the two mean different things and
    /// substituting one for the other would refuse valid work.
    #[test]
    fn missing_mem_available_is_none_not_memfree() {
        let mi = "MemTotal:       131072000 kB\nMemFree:         1000000 kB\n";
        assert_eq!(parse_mem_available(mi), None);
    }

    #[test]
    fn macos_vm_stat_parses_with_its_page_size() {
        let out = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
                   Pages free:                               10,000.\n\
                   Pages active:                            500,000.\n\
                   Pages inactive:                           20,000.\n";
        // (10_000 + 20_000) pages * 16384 B
        assert_eq!(parse_vm_stat(out), Some(30_000 * 16_384));
    }

    /// The page size is read from the header, not assumed to be 4 KiB — Apple
    /// silicon uses 16 KiB, so assuming 4096 would under-report by 4x.
    #[test]
    fn vm_stat_page_size_is_read_not_assumed() {
        let four_k = "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n\
                      Pages free:                               10,000.\n\
                      Pages inactive:                           20,000.\n";
        assert_eq!(parse_vm_stat(four_k), Some(30_000 * 4_096));
    }

    #[test]
    fn vm_stat_without_usable_counters_is_none() {
        let out = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
                   Pages active:                            500,000.\n";
        assert_eq!(parse_vm_stat(out), None);
    }

    #[test]
    fn human_bytes_reads_as_a_refusal_should() {
        assert_eq!(human_bytes(16 << 30), "16.0 GiB");
        assert_eq!(human_bytes(512 << 20), "512.0 MiB");
        assert_eq!(human_bytes(1024), "1024 B");
    }

    /// A statevector refusal quotes sizes no machine has. They must still read
    /// as a size — "137438953472.0 GiB" makes the message look broken.
    #[test]
    fn absurd_sizes_still_read_as_sizes() {
        assert_eq!(human_bytes(1u128 << 44), "16.0 TiB"); // 40 qubits
        assert_eq!(human_bytes(1u128 << 67), "128.0 EiB"); // 63 qubits
    }
}
