// SPDX-License-Identifier: Apache-2.0
//! Turning an operator's environment into a budget, and saying where it came from.
//!
//! The conventions here are `omega-server`'s `limits.rs`, deliberately, because
//! an operator who has learned one set of rules should not have to learn a
//! second. That module cannot be imported — it lives in a binary-only crate —
//! so the rules are restated rather than reused, and the duplication is the
//! honest cost of not extracting a shared crate for it yet.
//!
//! The three rules, unchanged:
//!
//! 1. **Absolute beats fraction**, and the resolution is reported rather than
//!    silently applied.
//! 2. **Caps compose by `min`, never `max`.** Every cap is a ceiling, and a
//!    generous one must never widen a strict one. Getting this backwards fails
//!    *silently*, as over-admission.
//! 3. **A malformed value is an error, never a fallback.** `limits.rs` records
//!    why: `OMEGA_MAX_MEM=48G` once parsed as garbage and left a 4 GiB budget,
//!    which is worse than an error because it looks like it worked. The
//!    dangerous direction here is the same one — a throttle that quietly turns
//!    itself off is indistinguishable from one that is working.
//!
//! And one rule of this module's own:
//!
//! 4. **In a container, the budget is the container's.** `/sys/fs/cgroup` is
//!    consulted and the smaller of it and the host's memory wins. Without this a
//!    process on a 64 GB node inside a 2 GB pod believes it has 64 GB, admits a
//!    job that fits the node and not the pod, and the kernel OOM-kills it — the
//!    exact inversion of this crate's contract, which is to refuse.

use std::path::{Path, PathBuf};

use crate::{Amounts, Axis, Mode};

/// Where a resolved limit came from, so an operator never has to read source to
/// answer "why was my job refused?".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Provenance {
    Env(&'static str),
    Profile,
    Detected,
    Cgroup,
    Default,
}

impl Provenance {
    pub fn label(&self) -> String {
        match self {
            Provenance::Env(n) => format!("env:{n}"),
            Provenance::Profile => "profile".into(),
            Provenance::Detected => "detected".into(),
            Provenance::Cgroup => "cgroup".into(),
            Provenance::Default => "default".into(),
        }
    }
}

/// A resolved cap and its provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limit {
    pub value: u64,
    pub from: Provenance,
}

/// The operator's one knob, when they do not want five.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Gentle,
    Balanced,
    Greedy,
}

impl Profile {
    /// The same fractions `omega-server` and the neutral-atom server already
    /// use. Divergent numbers across one estate would be a trap.
    pub fn fraction(self) -> f64 {
        match self {
            Profile::Gentle => 0.25,
            Profile::Balanced => 0.50,
            Profile::Greedy => 0.90,
        }
    }

    pub fn parse(s: &str) -> Result<Profile, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "gentle" => Ok(Profile::Gentle),
            "balanced" => Ok(Profile::Balanced),
            "greedy" => Ok(Profile::Greedy),
            other => Err(format!(
                "unrecognised resource profile {other:?} — expected one of: \
                 gentle, balanced, greedy"
            )),
        }
    }
}

pub const ENV_PATH: &str = "OMEGA_HOSTGATE";
pub const ENV_MODE: &str = "OMEGA_HOSTGATE_MODE";
pub const ENV_PROFILE: &str = "OMEGA_HOSTGATE_PROFILE";
pub const ENV_MAX_MEM: &str = "OMEGA_HOSTGATE_MAX_MEM";
pub const ENV_MEM_FRACTION: &str = "OMEGA_HOSTGATE_MEM_FRACTION";
pub const ENV_SLOTS: &str = "OMEGA_HOSTGATE_SLOTS";
pub const ENV_CPU_FRACTION: &str = "OMEGA_HOSTGATE_CPU_FRACTION";

/// What the machine says about itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Detected {
    pub host_bytes: u64,
    /// A cgroup memory ceiling, when one is in force and is lower than the host.
    pub cgroup_bytes: Option<u64>,
    /// Schedulable cores, with **0 meaning detection failed** — the same
    /// sentinel `host_bytes` uses, because zero of either is impossible.
    ///
    /// This field previously fell back to 1 on failure, and 1 is a legitimate
    /// core count, so "unknown" was indistinguishable from "single core" — and
    /// an explicit `OMEGA_HOSTGATE_SLOTS=8` was then clamped by `min` to 1. That
    /// is the byte-side defect fixed one commit earlier, left standing in the
    /// neighbouring field by the very commit that fixed it. A guard applied to
    /// one field and not its sibling is harder to find than no guard at all,
    /// because its presence reads as evidence the class was handled.
    pub cores: u64,
    /// Whether [`boot_id`] found a real boot identity. When it did not, the
    /// ledger has no way to know it survived a reboot, and §5.2 of `PROTOCOL.md`
    /// then requires the file to live somewhere that cannot.
    pub boot_identified: bool,
}

/// A resolved configuration.
#[derive(Clone, Debug)]
pub struct Config {
    pub path: Option<PathBuf>,
    pub mode: Mode,
    pub caps: Amounts,
    pub provenance: Vec<(Axis, Provenance)>,
    /// False when the boot could not be named and the ledger is relying on its
    /// filesystem being cleared per boot instead. Surfaced by `status`, because
    /// an operator should be able to SEE a degraded gate rather than infer it.
    pub boot_identified: bool,
}

impl Config {
    /// Resolve from an arbitrary lookup, so this is testable without mutating
    /// the process environment — which is racy across threads and, since Rust
    /// 1.90, unsafe for good reason.
    pub fn resolve(
        get: impl Fn(&str) -> Option<String>,
        detected: Detected,
    ) -> Result<Config, String> {
        let Some(path) = get(ENV_PATH).filter(|s| !s.trim().is_empty()) else {
            // Unset means off, and off means this crate does nothing at all.
            return Ok(Config {
                path: None,
                mode: Mode::Off,
                caps: Amounts::new(),
                provenance: Vec::new(),
                boot_identified: true,
            });
        };

        // Both reboot protections off at once is the original bug reachable by
        // another route, so it is refused rather than documented.
        //
        // The reasoning is the one this crate applies everywhere else: a gate
        // that silently degrades looks exactly like one that is working. We know
        // both facts right here — whether the boot could be named, and where the
        // ledger lives — so the condition can be made impossible instead of
        // written down and hoped for.
        let path_buf = PathBuf::from(&path);
        if !detected.boot_identified && !is_per_boot_path(&path_buf, &get) {
            return Err(format!(
                "{ENV_PATH}: this platform cannot identify its boot (no \
                 /proc/sys/kernel/random/boot_id, no kern.bootsessionuuid), and \
                 {} is not on a filesystem that is cleared per boot. With \
                 neither, a ledger silently outlives a reboot and charges live \
                 processes for holders that cannot exist. Point {ENV_PATH} at \
                 /run, /var/run, $XDG_RUNTIME_DIR or /dev/shm.",
                path_buf.display()
            ));
        }

        // Naming a ledger is asking for the gate, so enforcing is the default
        // once it is named. Anything else would make the common case a two-step
        // opt-in and invite a half-configured box that records but refuses
        // nothing.
        let mode = match get(ENV_MODE) {
            Some(v) => Mode::parse(&v).map_err(|e| format!("{ENV_MODE}: {e}"))?,
            None => Mode::Enforce,
        };

        let profile = match get(ENV_PROFILE) {
            Some(v) => Some(Profile::parse(&v).map_err(|e| format!("{ENV_PROFILE}: {e}"))?),
            None => None,
        };

        // Rule 4: the container's ceiling, when it is the tighter one.
        let (detected_mem, mem_source) = match detected.cgroup_bytes {
            Some(c) if c < detected.host_bytes => (c, Provenance::Cgroup),
            _ => (detected.host_bytes, Provenance::Detected),
        };

        // Zero is never a real machine, so it means detection FAILED. That is
        // not evidence of a small box and must not be treated as one.
        let mem = resolve_axis(
            &get,
            ENV_MAX_MEM,
            ENV_MEM_FRACTION,
            profile,
            if detected_mem == 0 {
                None
            } else {
                Some(detected_mem)
            },
            mem_source,
            parse_bytes,
        )?;
        let slots = resolve_axis(
            &get,
            ENV_SLOTS,
            ENV_CPU_FRACTION,
            profile,
            if detected.cores == 0 {
                None
            } else {
                Some(detected.cores)
            },
            Provenance::Detected,
            |s| {
                s.trim()
                    .parse::<u64>()
                    .map_err(|_| format!("{s:?} is not a whole number of slots"))
            },
        )?;

        let mut caps = Amounts::new();
        caps.insert(Axis::HostBytes, mem.value);
        caps.insert(Axis::Slots, slots.value);
        Ok(Config {
            path: Some(PathBuf::from(path)),
            mode,
            caps,
            provenance: vec![(Axis::HostBytes, mem.from), (Axis::Slots, slots.from)],
            boot_identified: detected.boot_identified,
        })
    }

    /// The real environment.
    pub fn from_env() -> Result<Config, String> {
        Config::resolve(|k| std::env::var(k).ok(), detect())
    }

    /// Build the gate this configuration describes.
    pub fn into_gate(self) -> crate::HostGate {
        match self.path {
            None => crate::HostGate::off(),
            Some(p) => {
                // The BOOT identifies the ledger's lifetime. Relying on the file
                // living under a per-boot runtime directory would work, but it
                // would be an unstated requirement doing load-bearing work — and
                // an operator who points the gate at $HOME would lose it silently.
                crate::HostGate::new(p, boot_id(), self.caps, self.mode)
            }
        }
    }
}

/// Absolute, then fraction, then profile, then the whole detected amount.
#[allow(clippy::too_many_arguments)]
fn resolve_axis(
    get: &impl Fn(&str) -> Option<String>,
    abs_key: &'static str,
    frac_key: &'static str,
    profile: Option<Profile>,
    detected: Option<u64>,
    detected_from: Provenance,
    parse_abs: impl Fn(&str) -> Result<u64, String>,
) -> Result<Limit, String> {
    if let Some(v) = get(abs_key) {
        let value = parse_abs(&v).map_err(|e| format!("{abs_key}: {e}"))?;
        // Rule 2: an absolute setting is still a ceiling, so it composes by min
        // with what the machine actually has — but ONLY when we know what that
        // is. Clamping against a failed detection silently turned an operator's
        // explicit 8G into 0, and every request was then refused against "the
        // whole budget is 0" with nothing pointing at the real cause.
        let value = match detected {
            Some(d) => value.min(d),
            None => value,
        };
        return Ok(Limit {
            value,
            from: Provenance::Env(abs_key),
        });
    }
    // Everything below is a FRACTION of the machine, so without the machine
    // there is nothing to take a fraction of. Refusing here names the cause;
    // the alternative is a plausible-looking cap derived from a guess.
    let Some(detected) = detected else {
        return Err(format!(
            "could not determine this machine's size, and no absolute limit was \
             given. Set {abs_key} explicitly (a fraction or profile has nothing \
             to be a fraction OF), or fix detection."
        ));
    };
    if let Some(v) = get(frac_key) {
        let f = parse_fraction(&v).map_err(|e| format!("{frac_key}: {e}"))?;
        return Ok(Limit {
            value: scale(detected, f),
            from: Provenance::Env(frac_key),
        });
    }
    if let Some(p) = profile {
        return Ok(Limit {
            value: scale(detected, p.fraction()),
            from: Provenance::Profile,
        });
    }
    Ok(Limit {
        value: detected,
        from: detected_from,
    })
}

fn scale(total: u64, f: f64) -> u64 {
    // Floor, and never below 1 for a non-zero total: a fraction that rounds a
    // budget to zero would refuse every request with `TooLarge`, which reads as
    // a broken gate rather than a strict one.
    let v = (total as f64 * f).floor() as u64;
    if total > 0 {
        v.max(1)
    } else {
        0
    }
}

/// `48G`, `512M`, `1024`. Rule 3: anything else is an error.
pub fn parse_bytes(s: &str) -> Result<u64, String> {
    let t = s.trim();
    if t.is_empty() {
        return Err("empty".into());
    }
    let (digits, mult) = match t.chars().last().unwrap().to_ascii_uppercase() {
        'K' => (&t[..t.len() - 1], 1u64 << 10),
        'M' => (&t[..t.len() - 1], 1u64 << 20),
        'G' => (&t[..t.len() - 1], 1u64 << 30),
        'T' => (&t[..t.len() - 1], 1u64 << 40),
        _ => (t, 1),
    };
    let n: u64 = digits.trim().parse().map_err(|_| {
        format!(
            "{s:?} is not a byte size — expected a whole number with an \
             optional K, M, G or T suffix (for example 48G)"
        )
    })?;
    n.checked_mul(mult)
        .ok_or_else(|| format!("{s:?} overflows a 64-bit byte count"))
}

/// `0.25` or `25%`, and it must land in (0, 1].
pub fn parse_fraction(s: &str) -> Result<f64, String> {
    let t = s.trim();
    let f = if let Some(p) = t.strip_suffix('%') {
        p.trim()
            .parse::<f64>()
            .map_err(|_| format!("{s:?} is not a percentage"))?
            / 100.0
    } else {
        t.parse::<f64>()
            .map_err(|_| format!("{s:?} is not a fraction"))?
    };
    if !f.is_finite() || f <= 0.0 || f > 1.0 {
        return Err(format!(
            "{s:?} is out of range — a fraction must be greater than 0 and at \
             most 1 (or 1 to 100 with a % suffix)"
        ));
    }
    Ok(f)
}

/// Which boot this is.
///
/// # Why the ledger needs it
///
/// `(pid, start_time)` distinguishes a recycled PID within one boot, and on
/// Linux it does **not** distinguish across boots: field 22 of
/// `/proc/<pid>/stat` counts clock ticks since boot, so a process started five
/// seconds into one boot and another started five seconds into the next read the
/// same number, and early-boot daemons draw low, repeatable PIDs besides. A
/// ledger that survives a reboot would then match stale records against live
/// unrelated processes and never prune them.
///
/// Stamping the ledger with the boot closes it: a file from a different boot
/// describes processes that cannot exist, so its holders go wholesale.
///
/// **Never derived from a clock.** macOS's `kern.boottime` is recomputed as
/// now-minus-uptime and moves on every NTP step and every sleep/wake, which would
/// invalidate every live grant on a laptop several times a day.
/// `kern.bootsessionuuid` is the stable identifier and is what is read here.
///
/// When the boot cannot be identified the value is `"unknown-boot"`, which
/// compares equal to itself — so the protection is lost but nothing breaks, and
/// the ledger must then live on a filesystem that is itself cleared per boot
/// (`/run`, `$XDG_RUNTIME_DIR`, `/var/run`).
pub fn boot_id() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/sys/kernel/random/boot_id") {
            let t = s.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(v) = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "kern.bootsessionuuid"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
        {
            let t = v.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
    }
    UNKNOWN_BOOT.to_string()
}

/// What `boot_id` returns when the boot cannot be identified at all.
pub const UNKNOWN_BOOT: &str = "unknown-boot";

/// Filesystems that are themselves cleared per boot, so a ledger living there
/// cannot outlive one even if we cannot name the boot.
///
/// `/tmp` is deliberately NOT here. It is cleared per boot on many Linux
/// distributions and on none of the macOS ones, and a protection that depends
/// on which distribution you are running is not a protection.
fn is_per_boot_path(path: &Path, get: &impl Fn(&str) -> Option<String>) -> bool {
    let p = path.to_string_lossy();
    if let Some(rt) = get("XDG_RUNTIME_DIR") {
        let rt = rt.trim_end_matches('/');
        if !rt.is_empty() && p.starts_with(&format!("{rt}/")) {
            return true;
        }
    }
    ["/run/", "/var/run/", "/dev/shm/"]
        .iter()
        .any(|pre| p.starts_with(pre))
}

/// Ask the machine.
///
/// A failed memory detection reports **0**, which is never a real machine and is
/// therefore usable as "unknown". `Config::resolve` refuses rather than deriving
/// a cap from it. The direction matters: a cap that is too small only makes work
/// queue, while a cap that is too large is the OOM this crate exists to prevent
/// — so guessing high is the one thing detection must never do.
pub fn detect() -> Detected {
    Detected {
        host_bytes: detect_host_bytes(),
        cgroup_bytes: detect_cgroup_bytes(),
        // 0, not 1, when it cannot be determined — see `Detected::cores`.
        cores: std::thread::available_parallelism()
            .map(|n| n.get() as u64)
            .unwrap_or(0),
        boot_identified: boot_id() != UNKNOWN_BOOT,
    }
}

#[cfg(target_os = "linux")]
fn detect_host_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
        })
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

/// No `sysctl` binding in std, and this crate takes no dependency for one —
/// the same trade `omega-core::hostmem` already makes, by the same reasoning.
#[cfg(target_os = "macos")]
fn detect_host_bytes() -> u64 {
    std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn detect_host_bytes() -> u64 {
    0
}

/// cgroup v2 first, then v1. `max` means no ceiling.
///
/// # Measured in a real pod, 2026-08-21
///
/// Read from inside a running `atomqd` container on a k3s node, so the numbers
/// below are what this function and its host-side sibling actually see rather
/// than what the manifest claims:
///
/// ```text
/// /sys/fs/cgroup/memory.max      1610612736     the pod's ceiling  (1536Mi)
/// /proc/meminfo MemTotal         6063896 kB     the NODE's memory  (5.78 GiB)
/// ```
///
/// A daemon that sizes its budget from `MemTotal` therefore believes it has
/// **3.9x more memory than the kernel will let it keep**, and the kernel
/// enforces the difference by OOM-killing rather than by refusing — the exact
/// inversion of this crate's contract. Taking the smaller of the two resolves
/// 1536Mi here, which is the number the pod is actually held to.
#[cfg(target_os = "linux")]
fn detect_cgroup_bytes() -> Option<u64> {
    // Which cgroup are we actually in? With a cgroup namespace — the normal
    // container case — `/proc/self/cgroup` reads `0::/` and the container sees
    // its own cgroup as the root, so `/sys/fs/cgroup/memory.max` IS its limit.
    //
    // Without one (`--cgroupns=host`, older runtimes), that same path is the
    // HOST's root cgroup, which is almost always `max`. The read then succeeds,
    // returns a perfectly plausible answer, and is wrong in the dangerous
    // direction: no ceiling at all, on a process that has one. So the relative
    // path is consulted first when it names anything but the root.
    let rel = std::fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("0::"))
                .map(|l| l[3..].trim().to_string())
                .filter(|p| p.starts_with('/') && p != "/")
        });

    let mut candidates: Vec<String> = Vec::new();
    if let Some(rel) = rel {
        candidates.push(format!("/sys/fs/cgroup{rel}/memory.max"));
    }
    candidates.push("/sys/fs/cgroup/memory.max".to_string());
    candidates.push("/sys/fs/cgroup/memory/memory.limit_in_bytes".to_string());

    for p in candidates {
        if let Ok(s) = std::fs::read_to_string(&p) {
            let t = s.trim();
            if t == "max" {
                continue;
            }
            if let Ok(v) = t.parse::<u64>() {
                // v1 writes a sentinel near u64::MAX for "unlimited".
                if v > 0 && v < (1u64 << 62) {
                    return Some(v);
                }
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn detect_cgroup_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k: &str| m.get(k).cloned()
    }

    fn machine() -> Detected {
        Detected {
            host_bytes: 64 << 30,
            cgroup_bytes: None,
            cores: 16,
            boot_identified: true,
        }
    }

    fn blind() -> Detected {
        Detected {
            boot_identified: false,
            ..machine()
        }
    }

    #[test]
    fn unset_means_off_and_resolves_no_caps() {
        let c = Config::resolve(env(&[]), machine()).unwrap();
        assert_eq!(c.mode, Mode::Off);
        assert!(c.path.is_none());
        assert!(c.caps.is_empty());
    }

    #[test]
    fn naming_a_ledger_enforces_by_default() {
        let c = Config::resolve(env(&[(ENV_PATH, "/run/gate.json")]), machine()).unwrap();
        assert_eq!(c.mode, Mode::Enforce);
        assert_eq!(c.caps.get(&Axis::HostBytes), Some(&(64 << 30)));
        assert_eq!(c.caps.get(&Axis::Slots), Some(&16));
    }

    #[test]
    fn absolute_beats_fraction_and_is_reported_as_such() {
        let c = Config::resolve(
            env(&[
                (ENV_PATH, "/run/g"),
                (ENV_MAX_MEM, "8G"),
                (ENV_MEM_FRACTION, "0.9"),
            ]),
            machine(),
        )
        .unwrap();
        assert_eq!(c.caps.get(&Axis::HostBytes), Some(&(8 << 30)));
        let (_, from) = c
            .provenance
            .iter()
            .find(|(a, _)| *a == Axis::HostBytes)
            .unwrap();
        assert_eq!(from.label(), "env:OMEGA_HOSTGATE_MAX_MEM");
    }

    #[test]
    fn a_cap_larger_than_the_machine_is_clamped_not_honoured() {
        // Rule 2. Honouring it would admit the job that kills the box.
        let c = Config::resolve(
            env(&[(ENV_PATH, "/run/g"), (ENV_MAX_MEM, "999G")]),
            machine(),
        )
        .unwrap();
        assert_eq!(c.caps.get(&Axis::HostBytes), Some(&(64 << 30)));
    }

    #[test]
    fn a_cgroup_ceiling_wins_over_the_hosts_memory() {
        // Rule 4: the pod's limit, not the node's.
        let m = Detected {
            host_bytes: 64 << 30,
            cgroup_bytes: Some(2 << 30),
            cores: 16,
            boot_identified: true,
        };
        let c = Config::resolve(env(&[(ENV_PATH, "/run/g")]), m).unwrap();
        assert_eq!(c.caps.get(&Axis::HostBytes), Some(&(2 << 30)));
        let (_, from) = c
            .provenance
            .iter()
            .find(|(a, _)| *a == Axis::HostBytes)
            .unwrap();
        assert_eq!(from.label(), "cgroup");
    }

    #[test]
    fn a_cgroup_ceiling_above_the_host_is_ignored() {
        let m = Detected {
            host_bytes: 8 << 30,
            cgroup_bytes: Some(64 << 30),
            cores: 4,
            boot_identified: true,
        };
        let c = Config::resolve(env(&[(ENV_PATH, "/run/g")]), m).unwrap();
        assert_eq!(c.caps.get(&Axis::HostBytes), Some(&(8 << 30)));
    }

    #[test]
    fn a_profile_scales_both_axes() {
        let c = Config::resolve(
            env(&[(ENV_PATH, "/run/g"), (ENV_PROFILE, "gentle")]),
            machine(),
        )
        .unwrap();
        assert_eq!(c.caps.get(&Axis::HostBytes), Some(&(16 << 30)));
        assert_eq!(c.caps.get(&Axis::Slots), Some(&4));
    }

    #[test]
    fn a_malformed_value_is_an_error_never_a_quiet_default() {
        // The OMEGA_MAX_MEM=48G incident, kept from happening again: the
        // dangerous outcome is a gate that looks configured and is not.
        for (k, v) in [
            (ENV_MAX_MEM, "48 gigabytes"),
            (ENV_MEM_FRACTION, "2.0"),
            (ENV_MEM_FRACTION, "0"),
            (ENV_CPU_FRACTION, "-1"),
            (ENV_SLOTS, "many"),
            (ENV_PROFILE, "aggressive"),
            (ENV_MODE, "enforcing"),
        ] {
            let r = Config::resolve(env(&[(ENV_PATH, "/run/g"), (k, v)]), machine());
            let e = r.unwrap_err();
            assert!(e.contains(k), "the error must name the variable: {e}");
        }
    }

    #[test]
    fn byte_sizes_parse_the_way_an_operator_writes_them() {
        assert_eq!(parse_bytes("48G").unwrap(), 48 << 30);
        assert_eq!(parse_bytes(" 512M ").unwrap(), 512 << 20);
        assert_eq!(parse_bytes("1024").unwrap(), 1024);
        assert_eq!(parse_bytes("2T").unwrap(), 2 << 40);
        assert!(parse_bytes("48GB").is_err());
        assert!(parse_bytes("").is_err());
        assert!(parse_bytes("999999999999T").is_err(), "must not wrap");
    }

    #[test]
    fn fractions_accept_both_spellings_and_reject_nonsense() {
        assert_eq!(parse_fraction("0.25").unwrap(), 0.25);
        assert_eq!(parse_fraction("25%").unwrap(), 0.25);
        assert_eq!(parse_fraction("1").unwrap(), 1.0);
        for bad in ["0", "-0.5", "1.5", "150%", "NaN", "inf", "half"] {
            assert!(parse_fraction(bad).is_err(), "{bad} must be refused");
        }
    }

    #[test]
    fn a_fraction_never_rounds_a_budget_down_to_zero() {
        // A zero cap refuses everything as TooLarge, which reads as broken
        // rather than strict.
        let m = Detected {
            host_bytes: 3,
            cgroup_bytes: None,
            cores: 1,
            boot_identified: true,
        };
        let c =
            Config::resolve(env(&[(ENV_PATH, "/run/g"), (ENV_MEM_FRACTION, "0.01")]), m).unwrap();
        assert_eq!(c.caps.get(&Axis::HostBytes), Some(&1));
    }

    #[test]
    fn no_boot_identity_and_a_persistent_path_is_refused_not_degraded() {
        // Both reboot protections off at once. Neither errors on its own, and
        // together they are the original aliasing bug by another route.
        let e = Config::resolve(env(&[(ENV_PATH, "/home/me/gate.json")]), blind()).unwrap_err();
        assert!(e.contains("cleared per boot"), "{e}");
        assert!(e.contains("/home/me/gate.json"), "must name the path: {e}");
        assert!(e.contains("/run"), "must say how to fix it: {e}");
    }

    #[test]
    fn no_boot_identity_is_fine_on_a_per_boot_filesystem() {
        for p in ["/run/gate.json", "/var/run/g.json", "/dev/shm/g.json"] {
            let c = Config::resolve(env(&[(ENV_PATH, p)]), blind())
                .unwrap_or_else(|e| panic!("{p} should be accepted: {e}"));
            assert!(!c.boot_identified, "and it must report itself degraded");
        }
    }

    #[test]
    fn xdg_runtime_dir_counts_as_per_boot() {
        let c = Config::resolve(
            env(&[
                (ENV_PATH, "/run/user/501/gate.json"),
                ("XDG_RUNTIME_DIR", "/run/user/501"),
            ]),
            blind(),
        )
        .expect("a runtime dir is exactly the right place for it");
        assert!(!c.boot_identified);
    }

    #[test]
    fn tmp_does_not_count_because_it_is_per_boot_on_some_systems_only() {
        // A protection that depends on which distribution you run is not one.
        assert!(Config::resolve(env(&[(ENV_PATH, "/tmp/gate.json")]), blind()).is_err());
    }

    #[test]
    fn a_named_boot_places_the_ledger_wherever_the_operator_likes() {
        let c = Config::resolve(env(&[(ENV_PATH, "/home/me/gate.json")]), machine())
            .expect("with a boot identity the path does not matter");
        assert!(c.boot_identified);
    }

    #[test]
    fn an_explicit_cap_survives_a_failed_detection() {
        // The defect: an absolute cap was clamped by `min` against a detected
        // size of 0, so an operator's explicit 8G silently became 0 and every
        // request was refused against "the whole budget is 0".
        let blind_mem = Detected {
            host_bytes: 0,
            ..machine()
        };
        let c = Config::resolve(env(&[(ENV_PATH, "/run/g"), (ENV_MAX_MEM, "8G")]), blind_mem)
            .expect("an explicit cap needs no detection");
        assert_eq!(c.caps.get(&Axis::HostBytes), Some(&(8 << 30)));
    }

    #[test]
    fn a_failed_detection_with_no_explicit_cap_is_an_error_not_a_guess() {
        // A fraction of an unknown machine is a guess wearing a number. And the
        // asymmetry decides the direction: too small only queues work, too large
        // is the OOM.
        let blind_mem = Detected {
            host_bytes: 0,
            ..machine()
        };
        for extra in [
            vec![],
            vec![(ENV_MEM_FRACTION, "0.5")],
            vec![(ENV_PROFILE, "gentle")],
        ] {
            let mut pairs = vec![(ENV_PATH, "/run/g")];
            pairs.extend(extra);
            let e = Config::resolve(env(&pairs), blind_mem).unwrap_err();
            assert!(e.contains("could not determine"), "{e}");
            assert!(e.contains(ENV_MAX_MEM), "must say how to fix it: {e}");
        }
    }

    #[test]
    fn an_explicit_slot_cap_survives_a_failed_core_detection() {
        // The neighbour of `an_explicit_cap_survives_a_failed_detection`, and it
        // was left broken by the commit that fixed that one. Found by the rule
        // that a guard for a class is a reason to check the sibling fields, not
        // a reason to stop looking.
        let blind_cores = Detected {
            cores: 0,
            ..machine()
        };
        let c = Config::resolve(env(&[(ENV_PATH, "/run/g"), (ENV_SLOTS, "8")]), blind_cores)
            .expect("an explicit slot cap needs no detection");
        assert_eq!(c.caps.get(&Axis::Slots), Some(&8));
    }

    #[test]
    fn a_failed_core_detection_with_no_explicit_cap_is_an_error() {
        let blind_cores = Detected {
            cores: 0,
            ..machine()
        };
        let e = Config::resolve(env(&[(ENV_PATH, "/run/g")]), blind_cores).unwrap_err();
        assert!(e.contains("could not determine"), "{e}");
        assert!(e.contains(ENV_SLOTS), "must name the slot knob: {e}");
    }

    #[test]
    fn detection_on_this_machine_finds_memory_and_cores() {
        let d = detect();
        assert!(d.cores >= 1, "0 means detection failed on this box");
        assert!(
            d.boot_identified,
            "this platform should be able to name its boot; boot_id() gave {:?}",
            boot_id()
        );
        assert!(
            d.host_bytes > (1 << 30),
            "expected to detect more than 1 GiB on this host, saw {}",
            d.host_bytes
        );
    }
}
