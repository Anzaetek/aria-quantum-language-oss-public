// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let result = match args.first().map(String::as_str) {
        Some("probe") => emu_compare_pauliprop::measure::probe(),
        Some("measure") => match get("--out") {
            Some(out) => emu_compare_pauliprop::measure::measure(std::path::Path::new(&out), args.iter().any(|a| a == "--append")),
            None => Err("measure needs --out PATH".into()),
        },
        Some("void") => match (get("--out"), get("--row-id"), get("--reason")) {
            (Some(o), Some(r), Some(why)) => emu_compare_pauliprop::measure::append_void(std::path::Path::new(&o), &r, &why),
            _ => Err("void needs --out --row-id --reason".into()),
        },
        _ => Err("usage: emu-compare-pauliprop probe | measure --out PATH [--append] | void --out PATH --row-id ID --reason TEXT".into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("emu-compare-pauliprop: {e}");
            ExitCode::from(1)
        }
    }
}
