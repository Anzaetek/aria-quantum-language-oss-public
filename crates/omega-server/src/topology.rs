// SPDX-License-Identifier: Apache-2.0
//! Memory topology — how many *pools* this machine really has.
//!
//! The governor ([`crate::worker`]) budgets memory. How many budgets it needs
//! depends on hardware in a way that cannot be inferred from the OS:
//!
//! | machine | topology | pools |
//! |---|---|---|
//! | DGX Spark (GB10 Grace-Blackwell) | **unified** — CPU+GPU share ~128 GB LPDDR5X | one |
//! | Mac, Apple Silicon | **unified** | one |
//! | Discrete NVIDIA (RTX / A100 / H100) | host RAM + device HBM | host + one per device |
//! | DGX A100/H100 (8 GPUs) | discrete, multi-GPU | host + 8 |
//! | CPU-only server | host only | one |
//!
//! **Getting this wrong is not symmetric.** On a unified box, treating the GPU
//! as a separate pool double-counts: host says 128 GB, device says 128 GB, the
//! governor budgets 256 GB on a 128 GB machine and cheerfully OOMs it. On a
//! discrete box, treating it as unified merely under-uses VRAM — wasteful,
//! visible in `/health`, and recoverable. So:
//!
//! > **When the topology is uncertain, assume unified.**
//!
//! That asymmetry is the whole design. It matters most on Linux+CUDA, which
//! splits into *both* discrete (A100) and unified (GB10) while producing
//! near-identical probe output.
//!
//! Detection is split in two so the policy is testable without the hardware:
//! [`probe_machine`] does the impure part (files, `sysctl`, `nvidia-smi`), and
//! [`classify`] is a pure function over that evidence. Every platform in the
//! table above is unit-tested below on whatever box happens to run CI.

/// One GPU as reported by the driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceProbe {
    pub index: u32,
    pub name: String,
    pub total_bytes: u64,
}

/// Raw evidence about the machine. Injectable, so [`classify`] can be tested
/// against hardware nobody has to own.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineProbe {
    /// Physical RAM, if the platform makes it cheap to ask.
    pub host_bytes: Option<u64>,
    /// cgroup memory limit, when running under one. Beats `host_bytes`.
    pub cgroup_bytes: Option<u64>,
    pub is_macos: bool,
    pub is_arm64: bool,
    /// GPUs reported by `nvidia-smi`, in index order.
    pub devices: Vec<DeviceProbe>,
    /// `cudaDeviceProp.integrated` when a CUDA-linked build could ask. `None`
    /// means "unknown", which is the common case for this server since it does
    /// not link CUDA.
    pub cuda_integrated: Option<bool>,
}

/// How many memory pools the governor should maintain, and how big each is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Topology {
    /// No accelerator: one budget.
    HostOnly { host_bytes: u64 },
    /// CPU and GPU share one physical pool: **one** budget, debited by both.
    Unified { total_bytes: u64 },
    /// Independent pools: host plus one per device.
    Discrete {
        host_bytes: u64,
        devices: Vec<(u32, u64)>,
    },
}

/// Why [`classify`] decided what it did — surfaced in `/health` so an operator
/// can tell "detected unified" from "defaulted to unified because unsure".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyReason {
    /// Operator said so explicitly.
    Forced,
    /// No GPU present.
    NoDevices,
    /// `cudaDeviceProp.integrated` — the canonical answer.
    CudaIntegratedFlag,
    /// macOS on Apple Silicon is unified by construction.
    AppleSilicon,
    /// Device total ≈ host total: two views of one pool, not two pools.
    DeviceMatchesHost,
    /// Host and device capacities are clearly disjoint.
    CapacitiesDisjoint,
    /// Ambiguous — fell back to the safe assumption.
    UncertainDefaultedUnified,
}

/// A classified topology plus the reason, for reporting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Classified {
    pub topology: Topology,
    pub reason: TopologyReason,
}

/// Operator override, parsed from `OMEGA_MEM_TOPOLOGY`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyOverride {
    Unified,
    Discrete,
    HostOnly,
}

impl TopologyOverride {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "unified" => Some(TopologyOverride::Unified),
            "discrete" => Some(TopologyOverride::Discrete),
            "host" | "host-only" | "hostonly" => Some(TopologyOverride::HostOnly),
            _ => None,
        }
    }
}

/// A device total within this fraction of host RAM is taken to be the *same*
/// memory seen twice rather than a second pool. Deliberately wide: a GB10
/// reports slightly less than physical RAM (firmware carve-outs), and the cost
/// of guessing "unified" wrongly is only wasted VRAM.
const SAME_POOL_TOLERANCE: f64 = 0.25;

/// A device holding less than this fraction of host RAM is clearly its own,
/// smaller pool — the discrete signature (e.g. 80 GB card, 1 TB host).
const CLEARLY_DISJOINT_RATIO: f64 = 0.5;

/// Effective host capacity: the cgroup limit wins when present, because that is
/// what will actually kill the process.
fn effective_host_bytes(probe: &MachineProbe) -> Option<u64> {
    match (probe.cgroup_bytes, probe.host_bytes) {
        (Some(c), Some(h)) => Some(c.min(h)),
        (Some(c), None) => Some(c),
        (None, h) => h,
    }
}

/// Decide the pool layout from evidence. Pure — no I/O, no environment.
pub fn classify(probe: &MachineProbe, forced: Option<TopologyOverride>) -> Classified {
    let host = effective_host_bytes(probe).unwrap_or(0);
    let device_total: u64 = probe.devices.iter().map(|d| d.total_bytes).sum();
    let devices: Vec<(u32, u64)> = probe
        .devices
        .iter()
        .map(|d| (d.index, d.total_bytes))
        .collect();

    // 1. The operator always wins. Detection is a convenience, and a container
    //    or an unusual driver can make it lie.
    if let Some(f) = forced {
        let topology = match f {
            TopologyOverride::HostOnly => Topology::HostOnly { host_bytes: host },
            TopologyOverride::Unified => Topology::Unified {
                // One pool: the shared physical memory, not host + device.
                total_bytes: host.max(device_total),
            },
            TopologyOverride::Discrete => Topology::Discrete {
                host_bytes: host,
                devices,
            },
        };
        return Classified {
            topology,
            reason: TopologyReason::Forced,
        };
    }

    // 2. No accelerator at all.
    if probe.devices.is_empty() {
        // Apple Silicon still has a GPU even with no nvidia-smi, but nothing in
        // this server dispatches to Metal, so one host pool is the truth here.
        return Classified {
            topology: Topology::HostOnly { host_bytes: host },
            reason: TopologyReason::NoDevices,
        };
    }

    // 3. The canonical answer, when a CUDA-linked build could ask for it.
    if let Some(integrated) = probe.cuda_integrated {
        return if integrated {
            Classified {
                topology: Topology::Unified {
                    total_bytes: host.max(device_total),
                },
                reason: TopologyReason::CudaIntegratedFlag,
            }
        } else {
            Classified {
                topology: Topology::Discrete {
                    host_bytes: host,
                    devices,
                },
                reason: TopologyReason::CudaIntegratedFlag,
            }
        };
    }

    // 4. Apple Silicon is unified by construction.
    if probe.is_macos && probe.is_arm64 {
        return Classified {
            topology: Topology::Unified {
                total_bytes: host.max(device_total),
            },
            reason: TopologyReason::AppleSilicon,
        };
    }

    // 5. Heuristic. One device whose capacity tracks host RAM is one pool seen
    //    twice — the GB10 / Grace-Blackwell signature. Two ~128 GB pools on a
    //    128 GB machine is not two pools.
    if host > 0 && probe.devices.len() == 1 {
        let d = probe.devices[0].total_bytes as f64;
        let h = host as f64;
        if (d - h).abs() <= h * SAME_POOL_TOLERANCE {
            return Classified {
                topology: Topology::Unified {
                    total_bytes: host.max(device_total),
                },
                reason: TopologyReason::DeviceMatchesHost,
            };
        }
    }

    // 6. Clearly-disjoint capacities: an 80 GB card in a 1 TB host is discrete.
    //    Require EVERY device to be clearly smaller, so a mixed bag stays safe.
    if host > 0
        && probe
            .devices
            .iter()
            .all(|d| (d.total_bytes as f64) < host as f64 * CLEARLY_DISJOINT_RATIO)
    {
        return Classified {
            topology: Topology::Discrete {
                host_bytes: host,
                devices,
            },
            reason: TopologyReason::CapacitiesDisjoint,
        };
    }

    // 7. Ambiguous. Assume unified: under-using VRAM is recoverable, and
    //    double-counting a shared pool is not.
    Classified {
        topology: Topology::Unified {
            total_bytes: host.max(device_total),
        },
        reason: TopologyReason::UncertainDefaultedUnified,
    }
}

// ---------------------------------------------------------------------------
// The impure half: gather evidence from this machine.
// ---------------------------------------------------------------------------

/// Inspect the running machine. Boot-time only — the subprocesses here are the
/// same precedent the governor already sets with `sysctl`.
pub fn probe_machine() -> MachineProbe {
    MachineProbe {
        host_bytes: detect_total_memory_bytes(),
        cgroup_bytes: detect_cgroup_limit_bytes(),
        is_macos: cfg!(target_os = "macos"),
        is_arm64: cfg!(target_arch = "aarch64"),
        devices: detect_nvidia_devices(),
        // This server does not link CUDA, so the canonical flag is unavailable.
        // A CUDA-enabled build can populate it and short-circuit the heuristics.
        cuda_integrated: None,
    }
}

/// GPUs via `nvidia-smi`. Absent tool ⇒ no devices, which is correct on a Mac
/// and on any CPU-only host.
fn detect_nvidia_devices() -> Vec<DeviceProbe> {
    let Ok(out) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    parse_nvidia_smi(&String::from_utf8_lossy(&out.stdout))
}

/// Parse `index, name, memory.total` (MiB) rows. Split out so the format is
/// testable without a GPU.
pub fn parse_nvidia_smi(stdout: &str) -> Vec<DeviceProbe> {
    stdout
        .lines()
        .filter_map(|line| {
            let mut parts = line.split(',').map(str::trim);
            let index = parts.next()?.parse().ok()?;
            let name = parts.next()?.to_string();
            // `--units` is suppressed by `nounits`, so this is plain MiB.
            let mib: u64 = parts.next()?.parse().ok()?;
            Some(DeviceProbe {
                index,
                name,
                total_bytes: mib.saturating_mul(1 << 20),
            })
        })
        .collect()
}

/// cgroup v2 then v1. Both report an effectively-unlimited sentinel when
/// unconstrained, which must read as "no limit" rather than a giant budget.
pub fn detect_cgroup_limit_bytes() -> Option<u64> {
    const UNLIMITED_FLOOR: u64 = 1 << 50; // 1 PiB — no real container cap is this big.
    for path in [
        "/sys/fs/cgroup/memory.max",                   // v2, unified hierarchy
        "/sys/fs/cgroup/memory/memory.limit_in_bytes", // v1
    ] {
        let Ok(raw) = std::fs::read_to_string(path) else {
            continue;
        };
        let raw = raw.trim();
        if raw == "max" {
            continue;
        }
        if let Ok(v) = raw.parse::<u64>() {
            if v > 0 && v < UNLIMITED_FLOOR {
                return Some(v);
            }
        }
    }
    None
}

/// Total physical RAM, when the platform makes it cheap to ask.
pub fn detect_total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
                return kb.checked_mul(1024);
            }
        }
        None
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

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    fn dev(index: u32, name: &str, bytes: u64) -> DeviceProbe {
        DeviceProbe {
            index,
            name: name.to_string(),
            total_bytes: bytes,
        }
    }

    /// DGX Spark: CPU and GPU on one package sharing ~128 GB. The dangerous
    /// case — naive detection sees a GPU, assumes discrete, and budgets 256 GB
    /// on a 128 GB machine.
    #[test]
    fn dgx_spark_gb10_is_unified_not_two_pools() {
        let probe = MachineProbe {
            host_bytes: Some(128 * GB),
            is_arm64: true,
            // GB10 reports slightly under physical (firmware carve-out).
            devices: vec![dev(0, "NVIDIA GB10", 120 * GB)],
            ..Default::default()
        };
        let c = classify(&probe, None);
        assert_eq!(c.reason, TopologyReason::DeviceMatchesHost);
        match c.topology {
            Topology::Unified { total_bytes } => {
                assert_eq!(total_bytes, 128 * GB, "one pool, not host + device");
                assert!(
                    total_bytes < 200 * GB,
                    "must never sum host and device on a shared pool"
                );
            }
            other => panic!("GB10 must be unified, got {other:?}"),
        }
    }

    #[test]
    fn apple_silicon_is_unified() {
        let probe = MachineProbe {
            host_bytes: Some(24 * GB),
            is_macos: true,
            is_arm64: true,
            ..Default::default()
        };
        // No nvidia-smi on a Mac, so this lands as host-only, which is the
        // truth for a server that dispatches nothing to Metal.
        let c = classify(&probe, None);
        assert_eq!(c.reason, TopologyReason::NoDevices);
        assert_eq!(
            c.topology,
            Topology::HostOnly {
                host_bytes: 24 * GB
            }
        );

        // With a device present, Apple Silicon must classify unified.
        let probe = MachineProbe {
            devices: vec![dev(0, "Apple M4 Pro", 24 * GB)],
            ..probe
        };
        let c = classify(&probe, None);
        assert!(matches!(
            c.reason,
            TopologyReason::AppleSilicon | TopologyReason::DeviceMatchesHost
        ));
        assert_eq!(
            c.topology,
            Topology::Unified {
                total_bytes: 24 * GB
            }
        );
    }

    #[test]
    fn discrete_card_in_a_big_host_is_two_pools() {
        let probe = MachineProbe {
            host_bytes: Some(1024 * GB),
            devices: vec![dev(0, "NVIDIA H100 80GB HBM3", 80 * GB)],
            ..Default::default()
        };
        let c = classify(&probe, None);
        assert_eq!(c.reason, TopologyReason::CapacitiesDisjoint);
        assert_eq!(
            c.topology,
            Topology::Discrete {
                host_bytes: 1024 * GB,
                devices: vec![(0, 80 * GB)],
            }
        );
    }

    #[test]
    fn dgx_a100_has_one_pool_per_gpu() {
        let devices: Vec<DeviceProbe> = (0..8)
            .map(|i| dev(i, "NVIDIA A100-SXM4-80GB", 80 * GB))
            .collect();
        let probe = MachineProbe {
            host_bytes: Some(2048 * GB),
            devices,
            ..Default::default()
        };
        let c = classify(&probe, None);
        match c.topology {
            Topology::Discrete { devices, .. } => {
                assert_eq!(devices.len(), 8, "device identity matters, not just total");
                assert!(devices.iter().all(|(_, b)| *b == 80 * GB));
            }
            other => panic!("8x A100 must be discrete, got {other:?}"),
        }
    }

    #[test]
    fn cpu_only_host_is_one_pool() {
        let probe = MachineProbe {
            host_bytes: Some(64 * GB),
            ..Default::default()
        };
        let c = classify(&probe, None);
        assert_eq!(c.reason, TopologyReason::NoDevices);
        assert_eq!(
            c.topology,
            Topology::HostOnly {
                host_bytes: 64 * GB
            }
        );
    }

    /// The canonical flag short-circuits every heuristic, in both directions.
    #[test]
    fn cuda_integrated_flag_wins_over_heuristics() {
        // Capacities look disjoint, but the driver says integrated.
        let probe = MachineProbe {
            host_bytes: Some(128 * GB),
            devices: vec![dev(0, "whatever", 16 * GB)],
            cuda_integrated: Some(true),
            ..Default::default()
        };
        let c = classify(&probe, None);
        assert_eq!(c.reason, TopologyReason::CudaIntegratedFlag);
        assert!(matches!(c.topology, Topology::Unified { .. }));

        // Capacities look identical, but the driver says discrete.
        let probe = MachineProbe {
            cuda_integrated: Some(false),
            ..probe
        };
        let c = classify(&probe, None);
        assert!(matches!(c.topology, Topology::Discrete { .. }));
    }

    /// The safety property this module exists for: never invent capacity that
    /// is not there. On a unified box the budget must never exceed physical.
    #[test]
    fn uncertain_topology_defaults_to_unified_and_never_sums_pools() {
        // Ambiguous: device is 60% of host — neither clearly same nor clearly
        // disjoint.
        let probe = MachineProbe {
            host_bytes: Some(100 * GB),
            devices: vec![dev(0, "mystery accelerator", 60 * GB)],
            ..Default::default()
        };
        let c = classify(&probe, None);
        assert_eq!(c.reason, TopologyReason::UncertainDefaultedUnified);
        match c.topology {
            Topology::Unified { total_bytes } => {
                assert_eq!(total_bytes, 100 * GB);
                assert!(
                    total_bytes < 160 * GB,
                    "must not sum host and device when unsure"
                );
            }
            other => panic!("uncertain must default to unified, got {other:?}"),
        }
    }

    #[test]
    fn cgroup_limit_beats_host_ram() {
        // A container with 16 GiB on a 1 TiB host must budget against 16.
        let probe = MachineProbe {
            host_bytes: Some(1024 * GB),
            cgroup_bytes: Some(16 * GB),
            ..Default::default()
        };
        let c = classify(&probe, None);
        assert_eq!(
            c.topology,
            Topology::HostOnly {
                host_bytes: 16 * GB
            }
        );
    }

    #[test]
    fn operator_override_wins() {
        let probe = MachineProbe {
            host_bytes: Some(1024 * GB),
            devices: vec![dev(0, "NVIDIA H100", 80 * GB)],
            ..Default::default()
        };
        // Detection would say discrete; the operator says otherwise.
        let c = classify(&probe, Some(TopologyOverride::Unified));
        assert_eq!(c.reason, TopologyReason::Forced);
        assert!(matches!(c.topology, Topology::Unified { .. }));

        let c = classify(&probe, Some(TopologyOverride::HostOnly));
        assert_eq!(
            c.topology,
            Topology::HostOnly {
                host_bytes: 1024 * GB
            }
        );
    }

    #[test]
    fn override_parsing_accepts_the_documented_spellings() {
        assert_eq!(
            TopologyOverride::parse("unified"),
            Some(TopologyOverride::Unified)
        );
        assert_eq!(
            TopologyOverride::parse("  DISCRETE "),
            Some(TopologyOverride::Discrete)
        );
        assert_eq!(
            TopologyOverride::parse("host-only"),
            Some(TopologyOverride::HostOnly)
        );
        assert_eq!(TopologyOverride::parse("nonsense"), None);
    }

    #[test]
    fn nvidia_smi_output_parses() {
        let out = "0, NVIDIA H100 80GB HBM3, 81559\n1, NVIDIA H100 80GB HBM3, 81559\n";
        let devs = parse_nvidia_smi(out);
        assert_eq!(devs.len(), 2);
        assert_eq!(devs[0].index, 0);
        assert_eq!(devs[0].name, "NVIDIA H100 80GB HBM3");
        assert_eq!(devs[0].total_bytes, 81559 * (1 << 20));
        // Garbage must not panic or invent devices.
        assert!(parse_nvidia_smi("").is_empty());
        assert!(parse_nvidia_smi("no GPUs found\n").is_empty());
    }
}

// ---------------------------------------------------------------------------
// Live memory pressure
// ---------------------------------------------------------------------------

/// Memory the OS says is actually available right now, in bytes.
///
/// The governor's ledger counts what *it* admitted. It cannot see a job's
/// classical side — a loaded dataset, optimizer state, an autograd tape — nor
/// anything else sharing the box. A TLA+ check of the shipped policy found the
/// consequence directly: two QML rows priced at 8 GB against a 64 GB budget,
/// reported as 56 GB of headroom, while the machine actually held 68 GB.
/// Admission was satisfied and the box was over budget.
///
/// So admission consults the machine as well as the ledger. This is the
/// measurement half.
///
/// `None` when the platform will not say, and the caller must then fall back to
/// the ledger alone rather than inventing a number.
pub fn detect_available_memory_now() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        // MemAvailable is the kernel's own estimate of what a new allocation
        // can get without swapping — better than MemFree, which ignores
        // reclaimable page cache.
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

/// `MemAvailable:` from /proc/meminfo, in bytes. Split out to be testable
/// without a Linux box — hence unused on non-Linux builds.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub fn parse_mem_available(meminfo: &str) -> Option<u64> {
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
            return kb.checked_mul(1024);
        }
    }
    None
}

/// Free + inactive pages from `vm_stat`, in bytes. Unused off macOS.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
///
/// Inactive pages count as available: macOS reclaims them under pressure. Free
/// alone reads catastrophically low on a healthy Mac and would refuse
/// everything.
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

#[cfg(test)]
mod pressure_tests {
    use super::*;

    #[test]
    fn linux_mem_available_parses() {
        let mi = "MemTotal:       131072000 kB\nMemFree:         1000000 kB\nMemAvailable:    64000000 kB\n";
        assert_eq!(parse_mem_available(mi), Some(64_000_000 * 1024));
        // Absent field must be None, never a guess.
        assert_eq!(parse_mem_available("MemTotal: 1 kB\n"), None);
    }

    #[test]
    fn macos_vm_stat_counts_inactive_as_available() {
        let out = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
Pages free:                               100000.\n\
Pages active:                             500000.\n\
Pages inactive:                           200000.\n";
        // Free alone would read 1.5 GiB on a healthy Mac and refuse everything;
        // inactive is reclaimable and must count.
        let got = parse_vm_stat(out).expect("parses");
        assert_eq!(got, (100_000 + 200_000) * 16384);
        assert!(got > 4 << 30, "free+inactive should be GB-scale, got {got}");
        assert_eq!(parse_vm_stat("garbage"), None);
    }
}
