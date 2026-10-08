// SPDX-License-Identifier: Apache-2.0

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let cmd = args.next();
    let result = match cmd.as_deref() {
        Some("probe") => probe(),
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
                (Some(out), Some(row_id), Some(reason)) => {
                    emu_compare_fermionic::measure::append_void(
                        std::path::Path::new(&out),
                        &row_id,
                        &reason,
                    )
                }
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
            emu_compare_fermionic::measure::measure(std::path::Path::new(&out))
        }
        _ => {
            eprintln!(
                "usage: emu-compare-fermionic probe | measure --out PATH | void --out PATH --row-id ID --reason TEXT"
            );
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("emu-compare-fermionic: {e}");
            ExitCode::from(1)
        }
    }
}

fn probe() -> Result<(), String> {
    use emu_compare_fermionic::ffsim_client::Ffsim;
    use emu_compare_fermionic::gate::{admit, sign_flip_detected};
    use emu_compare_fermionic::workload;

    let mut ffsim = Ffsim::launch()?;
    let h2_ours = workload::h2(false)?;
    let h2_ff = ffsim.h2()?;
    let h2_theirs = h2_ff["value"].as_f64().unwrap();
    admit(h2_ours, h2_theirs, workload::h2_reference()?).map_err(|d| d.reason)?;
    let h2_flip = sign_flip_detected(h2_theirs, workload::h2(true)?)?;
    eprintln!("h2 ours {h2_ours} ffsim {h2_theirs} flip |Δ| {h2_flip:.3e}");

    let hub_ours = workload::hubbard(false)?;
    let hub_ff = ffsim.hubbard()?;
    let hub_theirs = hub_ff["value"].as_f64().unwrap();
    admit(
        hub_ours,
        hub_theirs,
        emu_compare_fermionic::gate::hubbard_analytic(),
    )
    .map_err(|d| d.reason)?;
    let hub_flip = sign_flip_detected(hub_theirs, workload::hubbard(true)?)?;
    eprintln!("hubbard ours {hub_ours} ffsim {hub_theirs} flip |Δ| {hub_flip:.3e}");

    for n in [2u32, 8] {
        let got = workload::kitaev_small(n, false)?;
        let ff = ffsim.kitaev(n)?;
        let theirs = ff["value"].as_f64().unwrap();
        admit(
            got.statevector,
            theirs,
            emu_compare_fermionic::gate::kitaev_analytic(n),
        )
        .map_err(|d| d.reason)?;
        let flipped = workload::kitaev_small(n, true)?.statevector;
        let delta = sign_flip_detected(theirs, flipped)?;
        eprintln!(
            "kitaev n{n} sv {} mp {} ffsim {theirs} flip |Δ| {delta:.3e}",
            got.statevector, got.majorana
        );
    }

    let n64 = workload::kitaev64(false)?;
    let refusal = ffsim.kitaev64()?;
    eprintln!(
        "kitaev n64 energy {} dropped {} terms {} ffsim ok {:?}",
        n64.energy,
        n64.dropped_mass,
        n64.final_terms,
        refusal.get("ok")
    );
    let lucj = workload::lucj(false)?;
    let prepared = ffsim.lucj_prepare()?;
    let sample = ffsim.lucj()?;
    eprintln!(
        "lucj sector {} sv {} ffsim {} dim {:?}",
        lucj.sector,
        lucj.statevector,
        sample["value"],
        prepared.get("dim")
    );
    let flipped = workload::lucj(true)?;
    let delta = sign_flip_detected(sample["value"].as_f64().unwrap(), flipped.sector)?;
    eprintln!("lucj flip |Δ| {delta:.3e}");
    Ok(())
}
