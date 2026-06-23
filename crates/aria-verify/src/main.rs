// SPDX-License-Identifier: Apache-2.0
//! `aria-verify` — run the Aria example applications and prove each matches a
//! pure-Rust classical oracle, numerically.
//!
//! Each example is its own crate under `crates/apps/<name>` (that's where the
//! harness lives — the `.aria` file points there too). This binary just maps
//! names to those crates' `run()`.
//!
//! Usage:
//!   aria-verify <example>          run one example (e.g. `qsvd`)
//!   aria-verify all                run every example (this is what CI asserts)
//!   aria-verify --native <…>       force the native fallback (no wasm guest)
//!   aria-verify socket --url U --token T   drive a running omega-server (feature `remote`)

use std::process::ExitCode;

use aria_verify_core::{ExampleFn, Transport};

/// Name → the example crate's `run`. The single place that lists every app.
fn registry() -> Vec<(&'static str, ExampleFn)> {
    vec![
        ("qsvd", aria_app_qsvd::run as ExampleFn),
        ("qft", aria_app_qft::run),
        ("vqe_ansatz", aria_app_vqe_ansatz::run),
        ("grover3", aria_app_grover3::run),
        ("bernstein_vazirani", aria_app_bernstein_vazirani::run),
        ("deutsch_jozsa", aria_app_deutsch_jozsa::run),
        ("swap_test", aria_app_swap_test::run),
        ("teleport", aria_app_teleport::run),
        ("qaoa_maxcut", aria_app_qaoa_maxcut::run),
        ("qml_classifier", aria_app_qml_classifier::run),
        ("qos", aria_app_qos::run),
        ("circulant", aria_app_circulant::run),
        ("cqs", aria_app_cqs::run),
        ("noise", aria_app_noise::run),
        ("bell", aria_app_bell::run),
        ("ghz", aria_app_ghz::run),
        ("superdense", aria_app_superdense::run),
        ("simon", aria_app_simon::run),
        ("qpe", aria_app_qpe::run),
        ("qsp", aria_app_qsp::run),
        // Parametrized circuits — differential forward-⟨Z⟩ vs an independent sim.
        ("iqp_born", aria_app_forward::iqp_born),
        ("quantum_kernel", aria_app_forward::quantum_kernel),
        ("qcnn", aria_app_forward::qcnn),
        ("qcbm_strongly_entangling", aria_app_forward::qcbm),
        ("qgan", aria_app_forward::qgan),
        ("qclassifier_rich", aria_app_forward::qclassifier_rich),
        ("qssl", aria_app_forward::qssl),
        ("sketch_qml", aria_app_forward::sketch_qml),
        ("strongly_entangling", aria_app_forward::strongly_entangling),
        ("qasm_gpu", aria_app_forward::qasm_gpu),
        ("hhl", aria_app_forward::hhl),
        ("qsvt_invert", aria_app_forward::qsvt_invert),
    ]
}

fn take_opt(args: &mut Vec<String>, flag: &str) -> Option<String> {
    if let Some(i) = args.iter().position(|a| a == flag) {
        if i + 1 < args.len() {
            let v = args.remove(i + 1);
            args.remove(i);
            return Some(v);
        }
    }
    None
}

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    // Socket transport: `aria-verify socket --url URL [--token TOK]`.
    if args.first().map(|s| s.as_str()) == Some("socket") {
        args.remove(0);
        let url =
            take_opt(&mut args, "--url").unwrap_or_else(|| "http://127.0.0.1:8899".to_string());
        let token = take_opt(&mut args, "--token");
        return run_socket(&url, token);
    }

    let force_native = args.iter().any(|a| a == "--native");
    args.retain(|a| a != "--native");
    let transport = if force_native {
        Transport::Native
    } else {
        Transport::WasmInProcess
    };

    let name = args.first().map(|s| s.as_str()).unwrap_or("all");
    let table = registry();
    let selected: Vec<_> = if name == "all" {
        table
    } else {
        table.into_iter().filter(|(n, _)| *n == name).collect()
    };

    if selected.is_empty() {
        eprintln!("unknown example '{name}'");
        eprintln!(
            "known: {} (or 'all', or 'socket')",
            registry()
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>()
                .join(", ")
        );
        return ExitCode::FAILURE;
    }

    let mut results: Vec<(String, bool)> = Vec::new();
    for (n, f) in &selected {
        match f(transport) {
            Ok(v) => results.push((n.to_string(), v.ok())),
            Err(e) => {
                eprintln!("\n{n}: ERROR — {e}");
                results.push((n.to_string(), false));
            }
        }
    }

    if selected.len() > 1 {
        println!("\n──────── summary ────────");
        for (n, ok) in &results {
            println!("  {:<22} {}", n, if *ok { "PASS" } else { "FAIL" });
        }
        let n_pass = results.iter().filter(|(_, ok)| *ok).count();
        println!("  {}/{} passed", n_pass, results.len());
    }

    if results.iter().all(|(_, ok)| *ok) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(feature = "remote")]
fn run_socket(url: &str, token: Option<String>) -> ExitCode {
    println!("socket transport → {url}");
    if aria_verify_core::socket::run(url, token) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(not(feature = "remote"))]
fn run_socket(_url: &str, _token: Option<String>) -> ExitCode {
    eprintln!(
        "the `socket` subcommand needs the `remote` feature:\n  \
         cargo run -p aria-verify --features remote -- socket --url URL --token TOK"
    );
    ExitCode::FAILURE
}
