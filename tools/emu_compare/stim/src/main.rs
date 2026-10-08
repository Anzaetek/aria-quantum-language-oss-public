// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let cmd = args.next();
    let result = match cmd.as_deref() {
        Some("probe") => emu_compare_stim::measure::probe(),
        Some("void") => {
            let mut out = None;
            let mut row_id = None;
            let mut reason = None;
            while let Some(a) = args.next() {
                match a.as_str() {
                    "--out" => out = args.next(),
                    "--row-id" => row_id = args.next(),
                    "--reason" => reason = args.next(),
                    other => {
                        eprintln!("unknown argument {other}");
                        return ExitCode::from(2);
                    }
                }
            }
            match (out, row_id, reason) {
                (Some(out), Some(row_id), Some(reason)) => emu_compare_stim::measure::append_void(
                    std::path::Path::new(&out),
                    &row_id,
                    &reason,
                ),
                _ => {
                    eprintln!("void needs --out PATH --row-id ID --reason TEXT");
                    return ExitCode::from(2);
                }
            }
        }
        Some("measure") => {
            let mut out = None;
            while let Some(a) = args.next() {
                if a == "--out" {
                    out = args.next();
                } else {
                    eprintln!("unknown argument {a}");
                    return ExitCode::from(2);
                }
            }
            let Some(out) = out else {
                eprintln!("measure needs --out PATH");
                return ExitCode::from(2);
            };
            emu_compare_stim::measure::measure(std::path::Path::new(&out))
        }
        _ => {
            eprintln!(
                "usage: emu-compare-stim probe | measure --out PATH | void --out PATH --row-id ID --reason TEXT"
            );
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("emu-compare-stim: {e}");
            ExitCode::from(1)
        }
    }
}
