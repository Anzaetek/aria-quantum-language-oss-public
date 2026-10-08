// SPDX-License-Identifier: Apache-2.0
//! The hostgate declaration a row records as `hostgate_cap`.
//!
//! `omega-hostgate run` does not tell its child what it declared, so the
//! lane's launcher is the one place the cap is written: it passes the same
//! value to `--host-bytes` and to [`HOSTGATE_CAP_ENV`]. A lane that wrote its
//! own string (every fermionic and stim row said `--host-bytes 2G`) would
//! publish andromeda's cap on a 40G akilles run.

/// The variable a lane launcher sets to the `--host-bytes` value it declared.
pub const HOSTGATE_CAP_ENV: &str = "OMEGA_EMU_HOSTGATE_CAP";

/// `--host-bytes <SIZE>` from [`HOSTGATE_CAP_ENV`]. Refuses an unset or empty
/// variable: a row that cannot name its cap is not published.
pub fn hostgate_cap() -> Result<String, String> {
    hostgate_cap_from(std::env::var(HOSTGATE_CAP_ENV).ok().as_deref())
}

/// [`hostgate_cap`] on a given value, so the refusal is testable without
/// touching the process environment.
pub fn hostgate_cap_from(value: Option<&str>) -> Result<String, String> {
    match value.map(str::trim) {
        Some(size) if !size.is_empty() && !size.starts_with('-') => {
            Ok(format!("--host-bytes {size}"))
        }
        _ => Err(format!(
            "{HOSTGATE_CAP_ENV} is unset or not a size; refusing to publish a row that cannot \
             name its cap (set it to the value passed to `omega-hostgate run --host-bytes`)"
        )),
    }
}

/// The box a row was measured on, from `hostname` with any domain suffix
/// dropped (`andromeda.local` → `andromeda`). A lane that wrote its own box
/// name published andromeda's name on akilles rows; the source scan in
/// `tests/load_threshold.rs` now refuses a literal one.
pub fn host_name() -> Result<String, String> {
    let out = std::process::Command::new("hostname")
        .output()
        .map_err(|e| format!("hostname: {e}"))?;
    let full = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let short = full.split('.').next().unwrap_or("").to_string();
    if short.is_empty() {
        return Err("hostname is empty: a row must name the box it ran on".into());
    }
    Ok(short)
}
