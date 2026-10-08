// SPDX-License-Identifier: Apache-2.0
//! The device census (§4.4 addendum, written from E5's first attempt: the card
//! was at 98% with an ollama `llama-server` at 70.4 GB and a `train.py` at
//! 3.6 GB while the CPU load average read quiet; and an ollama model sharing
//! the card made our arm read 411 ms where the quiet card gives 122 ms).
//!
//! Three readings, all from `nvidia-smi`:
//! - holders and utilisation BEFORE the row, on a card no arm has touched yet;
//! - per-pid device memory DURING the row: every arm must hold at least its
//!   state on the card, or the "GPU" row is a CPU row wearing a GPU label
//!   (`cuda_svd_flat`'s 20–45% silent fallbacks are this repo's receipt);
//! - holders and utilisation AFTER the row, once every arm has exited.
//!
//! The after-utilisation is read after a settle delay as the minimum of
//! several samples. nvidia-smi's utilisation covers a recent window, and read
//! immediately it still covers our own just-exited arms: E5's harness voided
//! five healthy rows overnight on "util 70–100%, holders []" that way. Holders
//! stay the primary signal; utilisation must also clear.

use std::collections::BTreeMap;
use std::process::Command;
use std::time::Duration;

use omega_emu_compare::{GpuCensus, GpuHolder, GPU_UTIL_VOID_PCT};

/// Seconds to wait after the last arm exits before the after-reading.
pub const SETTLE_S: u64 = 5;
/// Utilisation samples taken after the settle delay; the minimum is recorded.
pub const UTIL_SAMPLES: usize = 5;

fn smi(args: &[&str]) -> Result<String, String> {
    let out = Command::new("nvidia-smi")
        .args(args)
        .output()
        .map_err(|e| format!("nvidia-smi: {e}"))?;
    if !out.status.success() {
        return Err(format!("nvidia-smi {args:?} exited {}", out.status));
    }
    String::from_utf8(out.stdout).map_err(|e| e.to_string())
}

/// `--query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits`.
pub fn parse_holders(csv: &str) -> Result<Vec<GpuHolder>, String> {
    let mut v = Vec::new();
    for line in csv.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let mut it = line.splitn(3, ',').map(str::trim);
        let (Some(pid), Some(mib), Some(name)) = (it.next(), it.next(), it.next()) else {
            return Err(format!("unreadable compute-apps line {line:?}"));
        };
        v.push(GpuHolder {
            pid: pid.parse().map_err(|e| format!("pid {pid:?}: {e}"))?,
            mib: mib.parse().map_err(|e| format!("MiB {mib:?}: {e}"))?,
            name: name.to_string(),
        });
    }
    Ok(v)
}

/// `--query-gpu=utilization.gpu --format=csv,noheader,nounits`: the busiest
/// card if there are several.
pub fn parse_util(csv: &str) -> Result<u32, String> {
    let mut best = None;
    for l in csv.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let u: u32 = l.parse().map_err(|e| format!("utilisation {l:?}: {e}"))?;
        best = Some(best.map_or(u, |b: u32| b.max(u)));
    }
    best.ok_or_else(|| "nvidia-smi reported no GPU".to_string())
}

pub fn holders() -> Result<Vec<GpuHolder>, String> {
    parse_holders(&smi(&[
        "--query-compute-apps=pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ])?)
}

pub fn util() -> Result<u32, String> {
    parse_util(&smi(&[
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ])?)
}

/// The card's name and compute capability, for the box record.
pub fn device_name() -> Result<String, String> {
    let s = smi(&[
        "--query-gpu=name,compute_cap,driver_version",
        "--format=csv,noheader",
    ])?;
    Ok(s.lines().next().unwrap_or_default().trim().to_string())
}

/// Device memory per pid, MiB, right now.
pub fn per_pid_mib() -> Result<BTreeMap<u32, u64>, String> {
    Ok(holders()?.into_iter().map(|h| (h.pid, h.mib)).collect())
}

/// The "before" half: taken on a card no arm of ours has touched.
pub struct Before {
    pub util_pct: u32,
    pub holders: Vec<GpuHolder>,
}

/// Holders now; utilisation as the minimum of three samples a second apart,
/// so the window of an arm of ours that has just exited does not read as
/// someone else's work (holders, which are instantaneous, stay primary).
pub fn before() -> Result<Before, String> {
    let holders = holders()?;
    let mut u = util()?;
    for _ in 0..2 {
        if u <= GPU_UTIL_VOID_PCT {
            break;
        }
        std::thread::sleep(Duration::from_secs(1));
        u = u.min(util()?);
    }
    Ok(Before {
        util_pct: u,
        holders,
    })
}

/// Holders other than the processes the row expects on the card right now.
/// Taken at every repeat, from the same reading as the arm's own memory: a
/// job that arrives and leaves inside a row is invisible to the before and
/// after readings, and two coexisting CUDA processes distort each other by up
/// to ~45% (measured), under the 5x spread flag.
pub fn foreign(holders: &[GpuHolder], expected: &[u32]) -> Vec<GpuHolder> {
    holders
        .iter()
        .filter(|h| !expected.contains(&h.pid))
        .cloned()
        .collect()
}

/// Whether a "before" reading admits starting a row at all.
pub fn quiet(b: &Before) -> bool {
    b.holders.is_empty() && b.util_pct <= GPU_UTIL_VOID_PCT
}

/// The "after" half and the assembled census. Call once every arm has exited.
pub fn after(b: Before) -> Result<GpuCensus, String> {
    std::thread::sleep(Duration::from_secs(SETTLE_S));
    let mut u = u32::MAX;
    for i in 0..UTIL_SAMPLES {
        if i > 0 {
            std::thread::sleep(Duration::from_secs(1));
        }
        u = u.min(util()?);
    }
    Ok(GpuCensus {
        util_before_pct: b.util_pct,
        util_after_pct: u,
        void_above_pct: GPU_UTIL_VOID_PCT,
        holders_before: b.holders,
        holders_after: holders()?,
    })
}

/// Why a census voids its row, if it does (the same rule `check` applies,
/// evaluated here so the lane can retry instead of publishing a void).
pub fn void_reason(c: &GpuCensus) -> Option<String> {
    if !c.holders_before.is_empty() || !c.holders_after.is_empty() {
        return Some(format!(
            "the GPU was not ours: {:?} before, {:?} after",
            c.holders_before, c.holders_after
        ));
    }
    let u = c.util_before_pct.max(c.util_after_pct);
    if u > c.void_above_pct {
        return Some(format!("GPU utilisation {u}% > {}%", c.void_above_pct));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holders_parse_names_and_sizes() {
        let h = parse_holders(
            "2960115, 70442, /usr/local/lib/ollama/llama-server\n3057384, 3614, .venv/bin/python\n",
        )
        .unwrap();
        assert_eq!(h.len(), 2);
        assert_eq!(
            (h[0].pid, h[0].mib, h[0].name.as_str()),
            (2960115, 70442, "/usr/local/lib/ollama/llama-server")
        );
        assert!(
            parse_holders("").unwrap().is_empty(),
            "no compute apps is an empty list, not an error"
        );
        assert!(parse_holders("garbage").is_err());
    }

    #[test]
    fn util_takes_the_busiest_card() {
        assert_eq!(parse_util("1\n").unwrap(), 1);
        assert_eq!(parse_util("3\n97\n").unwrap(), 97);
        assert!(
            parse_util("").is_err(),
            "no GPU must not read as an idle GPU"
        );
    }

    #[test]
    fn a_visitor_beside_the_running_arm_is_foreign() {
        let h = |pid, name: &str| GpuHolder {
            pid,
            mib: 1000,
            name: name.into(),
        };
        let held = vec![h(42, "emu-compare-gpu-ours"), h(7, "llama-server")];
        let f = foreign(&held, &[42]);
        assert_eq!(
            f.len(),
            1,
            "the ollama server beside our arm must be seen: {f:?}"
        );
        assert_eq!(f[0].pid, 7);
        assert!(
            foreign(&held[..1], &[42]).is_empty(),
            "our own arm is not a visitor"
        );
    }

    #[test]
    fn any_holder_or_busy_card_voids() {
        let quiet = GpuCensus {
            util_before_pct: 1,
            util_after_pct: 2,
            void_above_pct: 10,
            holders_before: vec![],
            holders_after: vec![],
        };
        assert!(void_reason(&quiet).is_none());
        let busy = GpuCensus {
            util_after_pct: 70,
            ..quiet.clone()
        };
        assert!(void_reason(&busy).unwrap().contains("70%"));
        let held = GpuCensus {
            holders_before: vec![GpuHolder {
                pid: 1,
                mib: 70442,
                name: "llama-server".into(),
            }],
            ..quiet
        };
        assert!(void_reason(&held).unwrap().contains("llama-server"));
    }
}
