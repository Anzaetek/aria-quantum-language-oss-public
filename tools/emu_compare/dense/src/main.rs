// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let cmd = args.next();
    let mut out = None;
    let mut row_id = None;
    let mut reason = None;
    let mut only = None;
    let mut cell = None;
    let mut append = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = args.next(),
            "--row-id" => row_id = args.next(),
            "--reason" => reason = args.next(),
            "--only" => only = args.next(),
            "--cell" => cell = args.next(),
            "--append" => append = true,
            other => {
                eprintln!("unknown argument {other}");
                return ExitCode::from(2);
            }
        }
    }
    let result = match (cmd.as_deref(), out, row_id, reason) {
        (Some("probe"), _, _, _) => {
            emu_compare_dense::measure::probe(cell.as_deref().unwrap_or("hea-14q-d12"))
        }
        (Some("measure"), Some(out), _, _) => {
            emu_compare_dense::measure::measure(std::path::Path::new(&out), only.as_deref(), append)
        }
        (Some("void"), Some(out), Some(row_id), Some(reason)) => {
            emu_compare_dense::measure::append_void(std::path::Path::new(&out), &row_id, &reason)
        }
        _ => {
            eprintln!(
                "usage: emu-compare-dense probe [--cell ID] | measure --out PATH [--only ID,ID..] [--append] | \
                 void --out PATH --row-id ID --reason TEXT"
            );
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("emu-compare-dense: {e}");
            ExitCode::from(1)
        }
    }
}
