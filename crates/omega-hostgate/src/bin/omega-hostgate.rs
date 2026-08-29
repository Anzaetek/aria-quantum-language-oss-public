// SPDX-License-Identifier: Apache-2.0
//! `omega-hostgate` — join the machine's budget without linking anything.
//!
//! The library form is for a Rust caller. This is for everything else: a Python
//! worker in a virtualenv, a shell loop, a third-party solver, a `cargo` build.
//! Any process at all can be admitted through the same gate by being spawned
//! under `run`, which is what makes a *host* budget possible rather than merely
//! an aria one — the processes that most need throttling are usually the ones
//! that cannot import a crate.

use std::process::{Command, ExitCode};

use omega_hostgate::config::{self, Config};
use omega_hostgate::{Axis, Mode, Request};

const USAGE: &str = "\
omega-hostgate — one resource budget per machine

  omega-hostgate status
      what the budget is, what is charged against it, and where each cap
      came from

  omega-hostgate run [--host-bytes SIZE] [--slots N] [--tag TEXT] -- CMD [ARGS...]
      take the resources, run CMD, and hold them for exactly as long as it
      lives. SIZE accepts a K/M/G/T suffix (for example 8G).

  omega-hostgate --help

Configured entirely by the environment, and OFF unless OMEGA_HOSTGATE names a
ledger file:

  OMEGA_HOSTGATE            path to the shared ledger. Unset means do nothing.
  OMEGA_HOSTGATE_MODE       off | advisory | enforce   (default: enforce)
  OMEGA_HOSTGATE_PROFILE    gentle | balanced | greedy
  OMEGA_HOSTGATE_MAX_MEM    absolute, e.g. 48G         (beats the fraction)
  OMEGA_HOSTGATE_MEM_FRACTION   0.5 or 50%
  OMEGA_HOSTGATE_SLOTS      absolute core count        (beats the fraction)
  OMEGA_HOSTGATE_CPU_FRACTION   0.5 or 50%

Exit codes:
  0   granted (or nothing to do), and for `run` the command's own status
  2   usage error
  3   refused — the reason names the limit and says whether waiting can help
  4   the gate could not answer, and is configured to fail closed
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("status") => status(),
        Some("run") => run(&args[1..]),
        Some("--help") | Some("-h") | None => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("omega-hostgate: unknown command {other:?}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn load() -> Result<Config, ExitCode> {
    Config::from_env().map_err(|e| {
        // A malformed setting is an error, never a quiet fallback: a throttle
        // that turns itself off looks exactly like one that is working.
        eprintln!("omega-hostgate: {e}");
        ExitCode::from(2)
    })
}

fn status() -> ExitCode {
    let cfg = match load() {
        Ok(c) => c,
        Err(code) => return code,
    };
    if cfg.path.is_none() {
        println!("mode      off");
        println!("ledger    (unset — {} names one)", config::ENV_PATH);
        println!();
        println!("Nothing is being budgeted on this machine. Processes linking");
        println!("the gate take no lock and perform no syscall.");
        return ExitCode::SUCCESS;
    }
    let path = cfg.path.clone().unwrap();
    let provenance = cfg.provenance.clone();
    let mode = cfg.mode;
    let boot_identified = cfg.boot_identified;
    let gate = cfg.into_gate();

    println!("mode      {}", mode_label(mode));
    println!("ledger    {}", path.display());
    // Absence should render. A gate running on the weaker of its two reboot
    // protections is still correct here, but an operator should be able to SEE
    // that rather than infer it from the platform.
    if boot_identified {
        println!("boot      identified — a ledger from a previous boot is detected");
    } else {
        println!(
            "boot      NOT identified — relying on {} being cleared per boot",
            path.parent().unwrap_or(&path).display()
        );
    }
    println!();
    println!(
        "{:<14} {:>18} {:>18} {:>8}  cap from",
        "axis", "cap", "charged", "holders"
    );
    let axes = match gate.capacity() {
        Ok(a) => a,
        Err(e) => {
            // A broken gate must not print a clean empty table.
            eprintln!("omega-hostgate: {e}");
            return ExitCode::from(4);
        }
    };
    for r in axes {
        let from = provenance
            .iter()
            .find(|(a, _)| *a == r.axis)
            .map(|(_, p)| p.label())
            .unwrap_or_else(|| "-".into());
        println!(
            "{:<14} {:>18} {:>18} {:>8}  {}",
            r.axis.to_string(),
            human(&r.axis, r.cap),
            human(&r.axis, r.charged),
            r.holders,
            from
        );
    }
    ExitCode::SUCCESS
}

fn mode_label(m: Mode) -> &'static str {
    match m {
        Mode::Off => "off",
        Mode::Advisory => "advisory (records, never refuses)",
        Mode::Enforce => "enforce",
    }
}

/// Bytes read better with a suffix; slots and quotas are just numbers.
fn human(axis: &Axis, v: u64) -> String {
    match axis {
        Axis::HostBytes | Axis::DeviceBytes(_) => {
            const UNITS: [&str; 5] = ["B", "K", "M", "G", "T"];
            let mut f = v as f64;
            let mut u = 0;
            while f >= 1024.0 && u < UNITS.len() - 1 {
                f /= 1024.0;
                u += 1;
            }
            if u == 0 {
                format!("{v} B")
            } else {
                format!("{f:.1} {}", UNITS[u])
            }
        }
        _ => v.to_string(),
    }
}

fn run(args: &[String]) -> ExitCode {
    let mut req = Request::new("hostgate-run");
    let mut tag: Option<String> = None;
    let mut i = 0;
    let mut cmd: Vec<String> = Vec::new();

    while i < args.len() {
        match args[i].as_str() {
            "--" => {
                cmd = args[i + 1..].to_vec();
                break;
            }
            "--host-bytes" | "--slots" | "--tag" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("omega-hostgate: {} needs a value", args[i]);
                    return ExitCode::from(2);
                };
                match args[i].as_str() {
                    "--host-bytes" => match config::parse_bytes(v) {
                        Ok(b) => req = req.want(Axis::HostBytes, b),
                        Err(e) => {
                            eprintln!("omega-hostgate: --host-bytes: {e}");
                            return ExitCode::from(2);
                        }
                    },
                    "--slots" => match v.parse::<u64>() {
                        Ok(n) => req = req.want(Axis::Slots, n),
                        Err(_) => {
                            eprintln!("omega-hostgate: --slots: {v:?} is not a whole number");
                            return ExitCode::from(2);
                        }
                    },
                    _ => tag = Some(v.clone()),
                }
                i += 2;
            }
            other => {
                eprintln!("omega-hostgate: unexpected argument {other:?}\n\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }

    if cmd.is_empty() {
        eprintln!("omega-hostgate: nothing to run — put the command after `--`");
        return ExitCode::from(2);
    }
    if let Some(t) = tag {
        req.tag = t;
    } else {
        req.tag = cmd[0].clone();
    }

    let cfg = match load() {
        Ok(c) => c,
        Err(code) => return code,
    };
    let gate = cfg.into_gate();

    let grant = match gate.try_acquire(&req) {
        Ok(g) => g,
        Err(e) => {
            // The command is NOT run. Being refused and running anyway would
            // make the whole budget advisory by accident.
            eprintln!("omega-hostgate: refused: {e}");
            let code = if e.to_string().starts_with("the host gate could not answer") {
                4
            } else {
                3
            };
            return ExitCode::from(code);
        }
    };

    let mut child = match Command::new(&cmd[0]).args(&cmd[1..]).spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("omega-hostgate: cannot run {:?}: {e}", cmd[0]);
            return ExitCode::from(2);
        }
    };

    // Hand the charge to the process that will actually hold the memory. If it
    // stayed with this wrapper, killing the wrapper would return the tokens
    // while the orphaned child kept running — and the gate would then admit
    // fresh work on top of it.
    let pid = child.id();
    if let Err(e) = grant.transfer_to(pid) {
        // The grant is still held by THIS process, so nothing leaked and the
        // child is still accounted for while we wait. What is lost is the
        // guarantee that killing this wrapper leaves the child charged, so say
        // exactly that rather than "warning: could not charge".
        eprintln!(
            "omega-hostgate: could not re-key the grant to pid {pid} ({e}). \
             The charge stays on this wrapper, so it is released if this \
             process is killed even though the child keeps running."
        );
    }

    match child.wait() {
        Ok(st) => {
            // The child's record is reclaimed by the next prune, because its
            // identity stops matching once it exits. Nothing here has to
            // release it, which is what makes this correct even if this wrapper
            // is killed while waiting.
            ExitCode::from(st.code().unwrap_or(1) as u8)
        }
        Err(e) => {
            eprintln!("omega-hostgate: waiting for the child failed: {e}");
            ExitCode::from(1)
        }
    }
}
