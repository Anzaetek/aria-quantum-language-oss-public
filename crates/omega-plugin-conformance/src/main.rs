//! Conformance test kit for omega backend plugins.
//!
//! Usage: `omega-plugin-conformance <path-to-cdylib> [--json]`
//!
//! Loads the plugin through the same `LoadedBackend::load` path the host uses
//! (so the ABI-version handshake and vtable are exercised for free) and runs a
//! fixed corpus, comparing each circuit's sampled distribution against the
//! in-tree dense statevector oracle. Prints a pass/fail table and exits 0 iff
//! every case passes.
//!
//! It is a developer tool, not a pinned CLI surface, so it prints to stdout.

use std::process::ExitCode;

use num_complex::Complex64;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;
use omega_core::plugin::LoadedBackend;

/// Total-variation distance threshold for a sampled distribution vs the exact
/// one. At 8192 shots the sampling error is well under this for the small
/// corpus distributions.
const TVD_THRESHOLD: f64 = 0.05;
const SHOTS: u32 = 8192;

struct Case {
    name: &'static str,
    circuit: CircuitIR,
}

fn gate(kind: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

fn corpus() -> Vec<Case> {
    // Bell: (|00> + |11>)/sqrt(2).
    let mut bell = CircuitIR::new(2, CircuitType::GateBased);
    bell.add_op(gate(GateKind::H, &[0], &[]));
    bell.add_op(gate(GateKind::CX, &[0, 1], &[]));

    // GHZ-3: (|000> + |111>)/sqrt(2).
    let mut ghz = CircuitIR::new(3, CircuitType::GateBased);
    ghz.add_op(gate(GateKind::H, &[0], &[]));
    ghz.add_op(gate(GateKind::CX, &[0, 1], &[]));
    ghz.add_op(gate(GateKind::CX, &[1, 2], &[]));

    // Uniform-2: H on both qubits — spreads probability over all four states.
    let mut uniform = CircuitIR::new(2, CircuitType::GateBased);
    uniform.add_op(gate(GateKind::H, &[0], &[]));
    uniform.add_op(gate(GateKind::H, &[1], &[]));

    // Rotation: a non-uniform single-qubit distribution.
    let mut rot = CircuitIR::new(1, CircuitType::GateBased);
    rot.add_op(gate(GateKind::Ry, &[0], &[std::f64::consts::FRAC_PI_3]));

    vec![
        Case {
            name: "bell",
            circuit: bell,
        },
        Case {
            name: "ghz3",
            circuit: ghz,
        },
        Case {
            name: "uniform2",
            circuit: uniform,
        },
        Case {
            name: "ry_pi_3",
            circuit: rot,
        },
    ]
}

/// Exact probability distribution from the statevector oracle.
fn exact_probs(circuit: &CircuitIR) -> Vec<f64> {
    let cfg = ExecConfig {
        shots: None,
        seed: Some(1),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    match StatevectorBackend::new().execute(circuit, &ParameterBinding::default(), &cfg) {
        Ok(ExecResult::Statevector(sv)) => sv.iter().map(|z: &Complex64| z.norm_sqr()).collect(),
        _ => panic!("oracle did not return a statevector"),
    }
}

/// Total-variation distance between a sampled count distribution and an exact
/// probability vector indexed by basis state.
fn tvd(
    counts: &std::collections::HashMap<omega_core::outcome::Outcome, u32>,
    exact: &[f64],
    total: u32,
) -> f64 {
    // Basis index i, at the width the counts actually carry. Looking it up at a
    // different width finds nothing and silently reports TVD 1.0 as though the
    // plugin disagreed with the oracle.
    let width = counts.keys().next().map(|o| o.width()).unwrap_or(0);
    let mut diff = 0.0;
    for (i, &p) in exact.iter().enumerate() {
        let key = omega_core::outcome::Outcome::from_u64(i as u64, width);
        let c = counts.get(&key).copied().unwrap_or(0) as f64 / total as f64;
        diff += (c - p).abs();
    }
    0.5 * diff
}

fn run_case(plugin: &LoadedBackend, case: &Case) -> Result<f64, String> {
    let cfg = ExecConfig {
        shots: Some(SHOTS),
        seed: Some(12345),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let result = plugin
        .execute(&case.circuit, &ParameterBinding::default(), &cfg)
        .map_err(|e| e.to_string())?;
    let counts = match result {
        ExecResult::Counts(c) => c,
        other => return Err(format!("expected Counts, got {other:?}")),
    };
    let exact = exact_probs(&case.circuit);
    Ok(tvd(&counts, &exact, SHOTS))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let json = args.iter().any(|a| a == "--json");
    let path = match args.iter().skip(1).find(|a| !a.starts_with("--")) {
        Some(p) => p.clone(),
        None => {
            eprintln!("usage: omega-plugin-conformance <path-to-cdylib> [--json]");
            return ExitCode::from(2);
        }
    };

    let plugin = match LoadedBackend::load(std::path::Path::new(&path)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to load plugin {path}: {e}");
            return ExitCode::from(2);
        }
    };

    let mut all_pass = true;
    let mut rows: Vec<(String, f64, bool)> = Vec::new();
    for case in corpus() {
        // A gate-model corpus: skip cases the plugin declares it cannot run,
        // reporting them rather than counting them as failures.
        if let Err(e) = plugin.check_circuit_supported(&case.circuit) {
            rows.push((format!("{} (unsupported: {e})", case.name), f64::NAN, true));
            continue;
        }
        match run_case(&plugin, &case) {
            Ok(d) => {
                let pass = d <= TVD_THRESHOLD;
                all_pass &= pass;
                rows.push((case.name.to_string(), d, pass));
            }
            Err(e) => {
                all_pass = false;
                rows.push((format!("{} (error: {e})", case.name), f64::NAN, false));
            }
        }
    }

    if json {
        let entries: Vec<String> = rows
            .iter()
            .map(|(n, d, p)| format!("{{\"case\":\"{n}\",\"tvd\":{d},\"pass\":{p}}}"))
            .collect();
        println!(
            "{{\"plugin\":\"{}\",\"all_pass\":{},\"cases\":[{}]}}",
            plugin.name(),
            all_pass,
            entries.join(",")
        );
    } else {
        println!("conformance: {}", plugin.name());
        if let Some(caps) = plugin.caps() {
            println!(
                "  caps: max_qubits={} kind={} shots={} expectation={}",
                caps.max_qubits, caps.kind, caps.supports_shots, caps.supports_expectation
            );
        } else {
            println!("  caps: (none declared)");
        }
        for (name, d, pass) in &rows {
            let mark = if *pass { "PASS" } else { "FAIL" };
            if d.is_nan() {
                println!("  [{mark}] {name}");
            } else {
                println!("  [{mark}] {name:<10} TVD={d:.4}");
            }
        }
        println!("  => {}", if all_pass { "ALL PASS" } else { "FAILED" });
    }

    if all_pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
