// SPDX-License-Identifier: Apache-2.0
//! Operator throttles — "use only *part* of this machine".
//!
//! [`crate::worker`] answers *"will this job kill the box?"*. This module
//! answers a different question: *"please only ever use a quarter of it"* —
//! because the box may be shared with non-Aria work, or be someone's laptop.
//!
//! Three rules keep the result predictable, and each exists because the
//! opposite is easy to write by accident:
//!
//! 1. **Absolute beats fraction** when both are set, and the resolution is
//!    reported rather than silently applied.
//! 2. **Caps compose by `min`, never `max`.** Every cap is a ceiling; a
//!    generous one must never widen a strict one. Getting this backwards fails
//!    *silently*, as over-admission.
//! 3. **On unified memory, the host and device fractions address the same
//!    physical pool** and must not compound — 0.5 of RAM plus 0.5 of "VRAM"
//!    must not become 128 GB on a 128 GB DGX Spark.
//!
//! And one ergonomic rule with a sharp edge: `OMEGA_MAX_MEM=48G` currently
//! parses as garbage and falls back to a 4 GiB default, which is worse than an
//! error because it looks like it worked. Malformed values are **startup
//! errors** here.

use std::fmt;

/// Where a resolved limit came from, so `/health` can answer "why was my job
/// refused?" without anyone reading the server's source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// Set explicitly by an environment variable.
    Env(&'static str),
    /// Derived from `OMEGA_RESOURCE_PROFILE`.
    Profile,
    /// Computed from detected hardware.
    Detected,
    /// Built-in fallback.
    Default,
}

impl Provenance {
    pub fn label(&self) -> String {
        match self {
            Provenance::Env(name) => format!("env:{name}"),
            Provenance::Profile => "profile".into(),
            Provenance::Detected => "detected".into(),
            Provenance::Default => "default".into(),
        }
    }
}

/// A limit plus where it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limit<T> {
    pub value: T,
    pub from: Provenance,
}

impl<T> Limit<T> {
    pub fn new(value: T, from: Provenance) -> Self {
        Self { value, from }
    }
}

/// A malformed operator setting. Fatal at startup: a throttle that silently
/// does not apply is indistinguishable from one that does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LimitError {
    pub var: String,
    pub value: String,
    pub expected: &'static str,
}

impl fmt::Display for LimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: cannot parse {:?} — expected {}",
            self.var, self.value, self.expected
        )
    }
}

/// Parse a byte size: a plain integer, or a number with a unit suffix.
///
/// Accepts `1024`, `48G`, `48GB`, `48GiB`, `1.5T`, case-insensitively. Both
/// `GB` and `GiB` mean 2^30 — operators writing `48G` mean "48 gigs of RAM",
/// and quibbling over 7% here would only produce surprising budgets.
pub fn parse_bytes(raw: &str) -> Option<u64> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    let lower = s.to_ascii_lowercase();
    let (num_part, shift) = if let Some(p) = lower.strip_suffix("kib").or(lower.strip_suffix("kb"))
    {
        (p, 10)
    } else if let Some(p) = lower.strip_suffix("mib").or(lower.strip_suffix("mb")) {
        (p, 20)
    } else if let Some(p) = lower.strip_suffix("gib").or(lower.strip_suffix("gb")) {
        (p, 30)
    } else if let Some(p) = lower.strip_suffix("tib").or(lower.strip_suffix("tb")) {
        (p, 40)
    } else if let Some(p) = lower.strip_suffix('k') {
        (p, 10)
    } else if let Some(p) = lower.strip_suffix('m') {
        (p, 20)
    } else if let Some(p) = lower.strip_suffix('g') {
        (p, 30)
    } else if let Some(p) = lower.strip_suffix('t') {
        (p, 40)
    } else {
        (lower.as_str(), 0)
    };
    let n: f64 = num_part.trim().parse().ok()?;
    if !n.is_finite() || n < 0.0 {
        return None;
    }
    let scaled = n * (1u64 << shift) as f64;
    if scaled > u64::MAX as f64 {
        return None;
    }
    Some(scaled as u64)
}

/// Parse a fraction: `0.25`, `25%`. Must land in `(0, 1]` — a zero or negative
/// share would make the server refuse everything, which is a configuration
/// mistake rather than a policy.
pub fn parse_fraction(raw: &str) -> Option<f64> {
    let s = raw.trim();
    let v: f64 = if let Some(p) = s.strip_suffix('%') {
        p.trim().parse::<f64>().ok()? / 100.0
    } else {
        s.parse().ok()?
    };
    if v.is_finite() && v > 0.0 && v <= 1.0 {
        Some(v)
    } else {
        None
    }
}

/// One-knob presets, because most people want "don't hog my laptop" rather
/// than five environment variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceProfile {
    Gentle,
    Balanced,
    Greedy,
}

impl ResourceProfile {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "gentle" | "conservative" => Some(ResourceProfile::Gentle),
            "balanced" | "default" => Some(ResourceProfile::Balanced),
            "greedy" | "aggressive" => Some(ResourceProfile::Greedy),
            _ => None,
        }
    }

    /// Share of detected memory this profile takes.
    pub fn memory_fraction(&self) -> f64 {
        match self {
            ResourceProfile::Gentle => 0.25,
            ResourceProfile::Balanced => 0.5,
            ResourceProfile::Greedy => 0.9,
        }
    }

    /// Share of available cores this profile takes.
    pub fn cpu_fraction(&self) -> f64 {
        match self {
            ResourceProfile::Gentle => 0.25,
            ResourceProfile::Balanced => 0.75,
            ResourceProfile::Greedy => 1.0,
        }
    }
}

/// Read an env var as bytes, erroring rather than falling back.
pub fn env_bytes(var: &'static str) -> Result<Option<Limit<u64>>, LimitError> {
    match std::env::var(var) {
        Err(_) => Ok(None),
        Ok(raw) if raw.trim().is_empty() => Ok(None),
        Ok(raw) => match parse_bytes(&raw) {
            Some(v) if v > 0 => Ok(Some(Limit::new(v, Provenance::Env(var)))),
            _ => Err(LimitError {
                var: var.to_string(),
                value: raw,
                expected: "a positive byte size such as 8589934592, 8G, or 8GiB",
            }),
        },
    }
}

/// Read an env var as a fraction, erroring rather than falling back.
pub fn env_fraction(var: &'static str) -> Result<Option<Limit<f64>>, LimitError> {
    match std::env::var(var) {
        Err(_) => Ok(None),
        Ok(raw) if raw.trim().is_empty() => Ok(None),
        Ok(raw) => match parse_fraction(&raw) {
            Some(v) => Ok(Some(Limit::new(v, Provenance::Env(var)))),
            None => Err(LimitError {
                var: var.to_string(),
                value: raw,
                expected: "a fraction in (0, 1] such as 0.25 or 25%",
            }),
        },
    }
}

/// Read an env var as a positive count, erroring rather than falling back.
pub fn env_count(var: &'static str) -> Result<Option<Limit<u32>>, LimitError> {
    match std::env::var(var) {
        Err(_) => Ok(None),
        Ok(raw) if raw.trim().is_empty() => Ok(None),
        Ok(raw) => match raw.trim().parse::<u32>() {
            Ok(v) if v > 0 => Ok(Some(Limit::new(v, Provenance::Env(var)))),
            _ => Err(LimitError {
                var: var.to_string(),
                value: raw,
                expected: "a positive integer",
            }),
        },
    }
}

/// Resolve a memory budget from an absolute setting, a fraction, and detected
/// capacity — applying rules 1 and 2 from the module docs.
///
/// `detected` is the hardware ceiling. The result is never above it: a fraction
/// scales it, and an absolute is still clamped by it, because an operator
/// asking for 1 TB on a 24 GB laptop has made a mistake the server should not
/// propagate into an admission budget.
pub fn resolve_memory(
    absolute: Option<Limit<u64>>,
    fraction: Option<Limit<f64>>,
    profile: Option<ResourceProfile>,
    detected: Option<u64>,
    fallback: u64,
) -> Limit<u64> {
    let Some(detected) = detected.filter(|d| *d > 0) else {
        // Nothing detected: an absolute setting is all we have to go on.
        return match absolute {
            Some(a) => a,
            None => Limit::new(fallback, Provenance::Default),
        };
    };

    // Rule 1: absolute beats fraction. Rule 2: still clamped by what exists.
    if let Some(a) = absolute {
        return Limit::new(a.value.min(detected), a.from);
    }

    let (frac, from) = match fraction {
        Some(f) => (f.value, f.from),
        None => match profile {
            Some(p) => (p.memory_fraction(), Provenance::Profile),
            None => (
                ResourceProfile::Balanced.memory_fraction(),
                Provenance::Detected,
            ),
        },
    };
    let value = ((detected as f64) * frac) as u64;
    Limit::new(value.max(1), from)
}

/// Resolve a concurrency cap in *jobs*, from an absolute count, a CPU fraction,
/// and the detected core count.
pub fn resolve_concurrency(
    absolute: Option<Limit<u32>>,
    fraction: Option<Limit<f64>>,
    profile: Option<ResourceProfile>,
    cores: usize,
) -> Limit<u32> {
    let cores = cores.max(1) as f64;
    if let Some(a) = absolute {
        return a;
    }
    let (frac, from) = match fraction {
        Some(f) => (f.value, f.from),
        None => match profile {
            Some(p) => (p.cpu_fraction(), Provenance::Profile),
            None => (
                ResourceProfile::Balanced.cpu_fraction(),
                Provenance::Detected,
            ),
        },
    };
    Limit::new(((cores * frac) as u32).max(1), from)
}

/// Available parallelism, or 1 when the platform will not say.
pub fn detect_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    /// `OMEGA_MAX_MEM=48G` used to parse as garbage and silently fall back to
    /// a 4 GiB default — worse than an error, because it looks like it worked.
    #[test]
    fn human_byte_units_parse() {
        assert_eq!(parse_bytes("1024"), Some(1024));
        assert_eq!(parse_bytes("48G"), Some(48 * GB));
        assert_eq!(parse_bytes("48GB"), Some(48 * GB));
        assert_eq!(parse_bytes("48GiB"), Some(48 * GB));
        assert_eq!(parse_bytes("48g"), Some(48 * GB));
        assert_eq!(parse_bytes(" 8 G "), Some(8 * GB));
        assert_eq!(parse_bytes("1.5T"), Some(1536 * GB));
        assert_eq!(parse_bytes("512M"), Some(512 << 20));
        // Malformed must be rejected, never coerced.
        assert_eq!(parse_bytes("lots"), None);
        assert_eq!(parse_bytes("-4G"), None);
        assert_eq!(parse_bytes(""), None);
        assert_eq!(parse_bytes("G"), None);
    }

    #[test]
    fn fractions_parse_both_spellings() {
        assert_eq!(parse_fraction("0.25"), Some(0.25));
        assert_eq!(parse_fraction("25%"), Some(0.25));
        assert_eq!(parse_fraction(" 100% "), Some(1.0));
        // Out of range is a configuration mistake, not a policy.
        assert_eq!(parse_fraction("0"), None);
        assert_eq!(parse_fraction("-0.5"), None);
        assert_eq!(parse_fraction("1.5"), None);
        assert_eq!(parse_fraction("half"), None);
    }

    /// Rule 1: absolute beats fraction, and says so.
    #[test]
    fn absolute_beats_fraction() {
        let r = resolve_memory(
            Some(Limit::new(8 * GB, Provenance::Env("OMEGA_MAX_MEM"))),
            Some(Limit::new(0.9, Provenance::Env("OMEGA_MEM_FRACTION"))),
            None,
            Some(64 * GB),
            4 * GB,
        );
        assert_eq!(r.value, 8 * GB);
        assert_eq!(r.from, Provenance::Env("OMEGA_MAX_MEM"));
    }

    /// Rule 2: a cap never widens what the hardware has. An operator asking for
    /// 1 TB on a 24 GB laptop has made a mistake; do not turn it into a budget.
    #[test]
    fn caps_compose_by_min_never_max() {
        let r = resolve_memory(
            Some(Limit::new(1024 * GB, Provenance::Env("OMEGA_MAX_MEM"))),
            None,
            None,
            Some(24 * GB),
            4 * GB,
        );
        assert_eq!(
            r.value,
            24 * GB,
            "must clamp to detected, not widen to 1 TB"
        );
    }

    #[test]
    fn fraction_scales_detected_capacity() {
        let r = resolve_memory(
            None,
            Some(Limit::new(0.25, Provenance::Env("OMEGA_MEM_FRACTION"))),
            None,
            Some(128 * GB),
            4 * GB,
        );
        assert_eq!(r.value, 32 * GB);
    }

    #[test]
    fn profile_presets_apply_when_nothing_explicit_is_set() {
        let gentle = resolve_memory(
            None,
            None,
            Some(ResourceProfile::Gentle),
            Some(100 * GB),
            GB,
        );
        assert_eq!(gentle.value, 25 * GB);
        assert_eq!(gentle.from, Provenance::Profile);

        let greedy = resolve_memory(
            None,
            None,
            Some(ResourceProfile::Greedy),
            Some(100 * GB),
            GB,
        );
        assert_eq!(greedy.value, 90 * GB);
        // A preset is still a ceiling, not an invitation to exceed hardware.
        assert!(greedy.value <= 100 * GB);
    }

    #[test]
    fn no_detection_falls_back_without_inventing_capacity() {
        let r = resolve_memory(None, None, None, None, 4 * GB);
        assert_eq!(r.value, 4 * GB);
        assert_eq!(r.from, Provenance::Default);
        // An explicit setting is still honoured with nothing detected.
        let r = resolve_memory(
            Some(Limit::new(16 * GB, Provenance::Env("OMEGA_MAX_MEM"))),
            None,
            None,
            None,
            4 * GB,
        );
        assert_eq!(r.value, 16 * GB);
    }

    #[test]
    fn concurrency_follows_cpu_share() {
        let r = resolve_concurrency(
            None,
            Some(Limit::new(0.25, Provenance::Env("OMEGA_CPU_FRACTION"))),
            None,
            20,
        );
        assert_eq!(r.value, 5, "quarter of a 20-core box");

        // Never zero — a throttle that admits nothing is a broken server.
        let r = resolve_concurrency(
            None,
            Some(Limit::new(0.01, Provenance::Env("OMEGA_CPU_FRACTION"))),
            None,
            2,
        );
        assert_eq!(r.value, 1);

        // Absolute wins.
        let r = resolve_concurrency(
            Some(Limit::new(3, Provenance::Env("OMEGA_MAX_CONCURRENCY"))),
            Some(Limit::new(1.0, Provenance::Env("OMEGA_CPU_FRACTION"))),
            None,
            64,
        );
        assert_eq!(r.value, 3);
    }

    #[test]
    fn profiles_parse_their_aliases() {
        assert_eq!(
            ResourceProfile::parse("gentle"),
            Some(ResourceProfile::Gentle)
        );
        assert_eq!(
            ResourceProfile::parse("CONSERVATIVE"),
            Some(ResourceProfile::Gentle)
        );
        assert_eq!(
            ResourceProfile::parse(" greedy "),
            Some(ResourceProfile::Greedy)
        );
        assert_eq!(ResourceProfile::parse("turbo"), None);
    }

    #[test]
    fn limit_errors_name_the_variable_and_what_was_expected() {
        let e = LimitError {
            var: "OMEGA_MAX_MEM".into(),
            value: "lots".into(),
            expected: "a positive byte size such as 8589934592, 8G, or 8GiB",
        };
        let msg = e.to_string();
        assert!(msg.contains("OMEGA_MAX_MEM"), "{msg}");
        assert!(msg.contains("lots"), "{msg}");
        assert!(msg.contains("8G"), "{msg}");
    }
}
