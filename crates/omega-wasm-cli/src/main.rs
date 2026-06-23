//! `omega-wasm` — host driver for the VQE / QAOA WASM guests.
//!
//! Wires a problem (Hamiltonian + ansatz, or QUBO + QAOA circuit) into a
//! `HostState`, runs the user's `vqe.wasm` / `qaoa.wasm` guest under
//! wasmtime, and decodes the iteration / result bit streams the guest
//! sends back through `omega_report_progress` / `omega_report_result`.
//!
//! The guests pass `f64` values across the FFI boundary as `i64` bit
//! patterns; the host runtime calls `f64::from_bits` and stores them in
//! `state.progress` / `state.final_result`. This binary surfaces those
//! to the user, and (for QAOA) samples the circuit at the optimal params
//! to recover the classical bitstring distribution.

mod optimizers;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::{json, Value};
use smallvec::smallvec;

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit, SymbolId};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable, PauliOp};
use omega_core::gradient::{compute_gradient, GradMethod};
use omega_core::params::ParameterBinding;
use omega_core::qaoa::qaoa_circuit;
use omega_core::qubo::Qubo;
use omega_parser::lower_to_ir;
use omega_wasm_runtime::host::{h2_hamiltonian, HostState};
use omega_wasm_runtime::WasmRunner;

use optimizers::{OptimizationResult, OptimizerKind};

const DEFAULT_FUEL: u64 = 100_000_000_000;
const SAMPLE_SHOTS: u32 = 4096;
const SAMPLE_SEED: u64 = 0x51517E5_u64;
const TOP_K_BITSTRINGS: usize = 8;
const DEFAULT_NATIVE_ITERS: usize = 200;
const DEFAULT_GD_LR: f64 = 0.4;
const DEFAULT_ADAM_LR: f64 = 0.1;
const DEFAULT_CMAES_SIGMA: f64 = 0.5;
const NATIVE_CONV_TOL: f64 = 1e-9;

#[derive(Clone, Copy, Debug)]
enum GradMethodChoice {
    Adjoint,
    ParamShift,
    FiniteDiff,
}

impl GradMethodChoice {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "adjoint" => Some(Self::Adjoint),
            "param-shift" | "parameter-shift" => Some(Self::ParamShift),
            "finite-diff" | "fd" => Some(Self::FiniteDiff),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Adjoint => "adjoint",
            Self::ParamShift => "param-shift",
            Self::FiniteDiff => "finite-diff",
        }
    }

    fn into_method(self) -> GradMethod {
        match self {
            Self::Adjoint => GradMethod::Adjoint,
            Self::ParamShift => GradMethod::ParameterShift,
            Self::FiniteDiff => GradMethod::FiniteDifference { epsilon: 1e-7 },
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum OutputFormat {
    Text,
    Json,
}

impl OutputFormat {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "text" => Some(Self::Text),
            "json" => Some(Self::Json),
            _ => None,
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return ExitCode::FAILURE;
    }
    if matches!(args[1].as_str(), "--help" | "-h") {
        print_usage();
        return ExitCode::SUCCESS;
    }

    match args[1].as_str() {
        "vqe" => run_vqe(&args[2..]),
        "qaoa" => run_qaoa(&args[2..]),
        "qaoa-qubo" => run_qaoa_qubo(&args[2..]),
        other => {
            eprintln!("Unknown subcommand: {other}");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  omega-wasm vqe       <wasm-file> [options]");
    eprintln!("  omega-wasm qaoa      <wasm-file> [options]");
    eprintln!("  omega-wasm qaoa-qubo <wasm-file> --qubo-file FILE [options]");
    eprintln!();
    eprintln!("VQE options:");
    eprintln!("  --problem h2              Built-in 1-qubit reduction of H2 (default)");
    eprintln!("  --observable PAULI        Custom observable, e.g. '0.5*Z0+0.3*X0X1'");
    eprintln!("  --circuit FILE.qasm       Custom ansatz");
    eprintln!();
    eprintln!("QAOA options:");
    eprintln!("  --graph NAME              Built-in MaxCut graph: triangle | square | k4");
    eprintln!("  --edges 'i-j,k-l,...'     Custom MaxCut edge list");
    eprintln!("  --qubo-file FILE.json     QUBO problem ({{\"n\":N,\"Q\":[[i,j,c],...]}})");
    eprintln!("  --circuit FILE.qasm       Custom QAOA circuit (with --observable)");
    eprintln!("  --depth P                 QAOA rounds (default 1)");
    eprintln!();
    eprintln!("qaoa-qubo options (drives the qaoa_qubo.wasm guest end-to-end):");
    eprintln!("  --qubo-file FILE.json     REQUIRED — QUBO matrix shipped to the guest");
    eprintln!("  --depth P                 QAOA rounds (default 2)");
    eprintln!();
    eprintln!("Optimizer options (default --optimizer wasm-gd uses the shipped guest):");
    eprintln!("  --optimizer NAME          wasm-gd | wasm-adam | gd | adam | cmaes");
    eprintln!("                              wasm-gd   : shipped guest, in-guest GD");
    eprintln!("                              wasm-adam : shipped guest, in-guest Adam");
    eprintln!("                              gd        : native fixed-step gradient descent");
    eprintln!(
        "                              adam      : native Adam (recommended >2D, gradient-based)"
    );
    eprintln!(
        "                              cmaes     : native CMA-ES (gradient-free, noisy/black-box)"
    );
    eprintln!("  --gradient NAME           adjoint (default) | param-shift | finite-diff");
    eprintln!("  --max-iters N             Optimizer budget (default {DEFAULT_NATIVE_ITERS})");
    eprintln!("  --learning-rate F         GD/Adam step size (default {DEFAULT_GD_LR}/gd, {DEFAULT_ADAM_LR}/adam)");
    eprintln!("  --init STR                'zeros' (default) or 'v0,v1,…' (length = #free params)");
    eprintln!("  --seed N                  RNG seed (CMA-ES; default 42)");
    eprintln!();
    eprintln!("WASM input options (forwarded to the guest's omega_input_*):");
    eprintln!("  --input FILE.json         Stage a JSON config payload for the guest");
    eprintln!("  --input-inline '<json>'   Same, but inline");
    eprintln!(
        "  --no-auto-input           Don't synthesize input JSON from the convenience flags above"
    );
    eprintln!();
    eprintln!("Common options:");
    eprintln!("  --fuel N                  WASM fuel cap (default {DEFAULT_FUEL})");
    eprintln!("  --format text|json        Output format (default text)");
    eprintln!("  --verbose                 Print every iteration of the value stream");
    eprintln!();
    eprintln!("After ./build-dist.sh the guests live at dist/wasm/{{vqe,qaoa,qaoa_qubo}}.wasm.");
    eprintln!("Examples:");
    eprintln!("  # Default WASM path, NUM_PARAMS=2:");
    eprintln!("  omega-wasm vqe  dist/wasm/vqe.wasm");
    eprintln!("  omega-wasm qaoa dist/wasm/qaoa.wasm --graph triangle");
    eprintln!();
    eprintln!("  # WASM path, NUM_PARAMS-flexible (depth 2 = 4 params, in-guest Adam):");
    eprintln!("  omega-wasm qaoa dist/wasm/qaoa.wasm --graph k4 --depth 2 --optimizer wasm-adam");
    eprintln!();
    eprintln!("  # Custom-optimizer guest: SPSA inside WASM, runtime QUBO build:");
    eprintln!("  omega-wasm qaoa-qubo dist/wasm/qaoa_qubo.wasm --qubo-file my.qubo.json --depth 3");
    eprintln!();
    eprintln!("  # Native path, any number of parameters / depth:");
    eprintln!("  omega-wasm vqe  dist/wasm/vqe.wasm --optimizer adam --max-iters 500 \\");
    eprintln!("      --circuit examples/circuits/vqe_ansatz_4q.qasm \\");
    eprintln!("      --observable '0.3979*Z0-0.3979*Z1-0.0112*Z0Z1+0.1809*X0X1'");
    eprintln!("  omega-wasm qaoa dist/wasm/qaoa.wasm --graph k4 --depth 2 --optimizer cmaes");
}

fn run_vqe(args: &[String]) -> ExitCode {
    let Some(wasm_path) = args.first().map(PathBuf::from) else {
        eprintln!("vqe: missing <wasm-file>");
        return ExitCode::FAILURE;
    };

    let mut observable_str: Option<String> = None;
    let mut circuit_path: Option<PathBuf> = None;
    let mut problem = "h2".to_string();
    let mut common = CommonOpts::default();

    let rest = &args[1..];
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--problem" => {
                i += 1;
                problem = required(rest, i, "--problem");
            }
            "--observable" => {
                i += 1;
                observable_str = Some(required(rest, i, "--observable"));
            }
            "--circuit" => {
                i += 1;
                circuit_path = Some(PathBuf::from(required(rest, i, "--circuit")));
            }
            other => {
                if !common.consume(other, rest, &mut i) {
                    eprintln!("Unknown vqe option: {other}");
                    return ExitCode::FAILURE;
                }
            }
        }
        i += 1;
    }

    let observable = match observable_str.as_deref() {
        Some(s) => match parse_observable(s) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("Invalid --observable: {e}");
                return ExitCode::FAILURE;
            }
        },
        None => match problem.as_str() {
            "h2" => h2_hamiltonian(),
            other => {
                eprintln!("Unknown VQE --problem: {other} (try: h2, or pass --observable)");
                return ExitCode::FAILURE;
            }
        },
    };

    let circuit = match circuit_path {
        Some(ref p) => match load_circuit(p) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to load --circuit {}: {}", p.display(), e);
                return ExitCode::FAILURE;
            }
        },
        None => simple_2q_ansatz(),
    };

    let header = json!({
        "algo": "vqe",
        "problem": problem,
        "observable_terms": observable.terms.len(),
        "circuit_qubits": circuit.num_qubits,
        "circuit_ops": circuit.ops.len(),
        "circuit_params": circuit.symbols.len(),
        "wasm": wasm_path.display().to_string(),
        "fuel": common.fuel,
    });

    // The shipped vqe.wasm guest is now NUM_PARAMS-flexible: it queries the
    // host via `omega_circuit_num_params` and allocates accordingly. No
    // truncation warning needed.

    execute(
        &wasm_path,
        common,
        header,
        AlgoTail::Vqe {
            circuit,
            observable,
        },
    )
}

/// `omega-wasm qaoa-qubo <wasm-file> --qubo-file FILE [options]`
///
/// Runs the qaoa_qubo.wasm guest end-to-end: stages a QUBO matrix as the
/// input payload, the guest builds the QAOA circuit at runtime, runs its
/// own (SPSA-based) optimizer, and reports back. No (circuit, observable)
/// is pre-registered by the host — that is the point. The host's only
/// responsibility is to pass the QUBO bytes through.
fn run_qaoa_qubo(args: &[String]) -> ExitCode {
    let Some(wasm_path) = args.first().map(PathBuf::from) else {
        eprintln!("qaoa-qubo: missing <wasm-file>");
        return ExitCode::FAILURE;
    };

    let mut qubo_file: Option<PathBuf> = None;
    let mut depth: i32 = 2;
    let mut common = CommonOpts::default();

    let rest = &args[1..];
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--qubo-file" => {
                i += 1;
                qubo_file = Some(PathBuf::from(required(rest, i, "--qubo-file")));
            }
            "--depth" => {
                i += 1;
                depth = required(rest, i, "--depth").parse().unwrap_or_else(|e| {
                    eprintln!("invalid --depth: {e}");
                    std::process::exit(1);
                });
            }
            other => {
                if !common.consume(other, rest, &mut i) {
                    eprintln!("Unknown qaoa-qubo option: {other}");
                    return ExitCode::FAILURE;
                }
            }
        }
        i += 1;
    }

    let qubo_path = match qubo_file {
        Some(p) => p,
        None => {
            eprintln!("qaoa-qubo requires --qubo-file FILE.json");
            return ExitCode::FAILURE;
        }
    };
    let qubo_text = match fs::read_to_string(&qubo_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read {}: {e}", qubo_path.display());
            return ExitCode::FAILURE;
        }
    };

    // Validate the QUBO so we can show brute-force info in the header.
    let qubo = match Qubo::from_json(&qubo_text) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("qaoa-qubo: invalid QUBO file: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (best_assignment, best_value) = if qubo.n <= 20 {
        let (x, v) = qubo.brute_force();
        (Some(x), Some(v))
    } else {
        (None, None)
    };

    // Build the input payload. The guest reads `qubo`, `depth`, plus a few
    // SPSA hyperparameters from this JSON.
    let extras = vec![("qubo", json!(qubo_text)), ("depth", json!(depth))];
    let auto = auto_input_for_wasm(&common, &extras);
    let input_bytes = match resolve_wasm_input(&common, auto) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let bytes = match fs::read(&wasm_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read {}: {}", wasm_path.display(), e);
            eprintln!(
                "(Build the guest with `cd examples/wasm-guests/qaoa_qubo && cargo build \
                 --target wasm32-wasip1 --release` or `./build-dist.sh`.)"
            );
            return ExitCode::FAILURE;
        }
    };

    let mut host = HostState::new();
    host.set_input(input_bytes);

    let mut runner = match WasmRunner::new(host) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to init wasmtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    if matches!(common.format, OutputFormat::Json) {
        runner.redirect_guest_stdout_to_stderr(true);
    }
    // Forward the same RNG seed that drives the native optimiser to the
    // guest's WASI random source — keeps the whole run reproducible.
    runner.rng_seed(common.seed);

    let header = json!({
        "algo": "qaoa-qubo",
        "qubo_n": qubo.n,
        "qubo_entries": qubo.entries.len(),
        "depth": depth,
        "wasm": wasm_path.display().to_string(),
        "fuel": common.fuel,
        "brute_force_optimum": best_value,
        "brute_force_assignment": best_assignment.as_ref().map(|v| bool_vec_to_bitstring(v)),
        "optimizer": "in-wasm-spsa",
    });

    if matches!(common.format, OutputFormat::Text) {
        print_header_text(&header);
        println!("--- guest stdout ---");
    }

    let wasm_result = match runner.run(&bytes, common.fuel) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[host] WASM execution error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let progress = {
        let s = runner.host_state().lock().unwrap();
        s.progress.clone()
    };
    let Some(wasm_res) = wasm_result else {
        eprintln!(
            "[host] WASM completed but never called omega_report_result \
             (got {} progress entries)",
            progress.len()
        );
        return ExitCode::FAILURE;
    };

    // The guest registered exactly one circuit + observable inside the
    // host (built from the supplied QUBO); pull them back out for the
    // sample-circuit-at-optimum step in print_tail.
    let (circuit, observable) = {
        let state = runner.host_state().lock().unwrap();
        let mut cids: Vec<u32> = state.circuits.keys().copied().collect();
        let mut oids: Vec<u32> = state.observables.keys().copied().collect();
        cids.sort();
        oids.sort();
        match (cids.first(), oids.first()) {
            (Some(cid), Some(oid)) => (state.circuits[cid].clone(), state.observables[oid].clone()),
            _ => {
                eprintln!("[host] guest did not register a circuit/observable");
                return ExitCode::FAILURE;
            }
        }
    };

    let opt = OptimizationResult {
        optimal_params: wasm_res.optimal_params,
        optimal_value: wasm_res.optimal_value,
        iterations: wasm_res.iterations as usize,
        progress,
    };

    let tail = AlgoTail::Qaoa {
        circuit,
        observable,
        ising_offset: qubo.to_ising().offset,
        n: qubo.n,
    };

    print_tail(&header, &opt, &tail, common, /* native */ false);
    ExitCode::SUCCESS
}

fn run_qaoa(args: &[String]) -> ExitCode {
    let Some(wasm_path) = args.first().map(PathBuf::from) else {
        eprintln!("qaoa: missing <wasm-file>");
        return ExitCode::FAILURE;
    };

    let mut graph: Option<String> = None;
    let mut edges_str: Option<String> = None;
    let mut qubo_file: Option<PathBuf> = None;
    let mut circuit_path: Option<PathBuf> = None;
    let mut observable_str: Option<String> = None;
    let mut depth: usize = 1;
    let mut common = CommonOpts::default();

    let rest = &args[1..];
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--graph" => {
                i += 1;
                graph = Some(required(rest, i, "--graph"));
            }
            "--edges" => {
                i += 1;
                edges_str = Some(required(rest, i, "--edges"));
            }
            "--qubo-file" => {
                i += 1;
                qubo_file = Some(PathBuf::from(required(rest, i, "--qubo-file")));
            }
            "--circuit" => {
                i += 1;
                circuit_path = Some(PathBuf::from(required(rest, i, "--circuit")));
            }
            "--observable" => {
                i += 1;
                observable_str = Some(required(rest, i, "--observable"));
            }
            "--depth" => {
                i += 1;
                depth = required(rest, i, "--depth").parse().unwrap_or_else(|e| {
                    eprintln!("invalid --depth: {e}");
                    std::process::exit(1);
                });
            }
            other => {
                if !common.consume(other, rest, &mut i) {
                    eprintln!("Unknown qaoa option: {other}");
                    return ExitCode::FAILURE;
                }
            }
        }
        i += 1;
    }

    let manual_circuit = circuit_path.is_some() || observable_str.is_some();
    let auto_problem = graph.is_some() || edges_str.is_some() || qubo_file.is_some();
    if manual_circuit && auto_problem {
        eprintln!(
            "qaoa: --circuit/--observable are mutually exclusive with --graph/--edges/--qubo-file"
        );
        return ExitCode::FAILURE;
    }

    // The shipped qaoa.wasm guest is NUM_PARAMS-flexible (it queries the
    // host or reads `num_params` from the input JSON), so no warning.

    if manual_circuit {
        let Some(cp) = circuit_path.as_deref() else {
            eprintln!("qaoa --observable also needs --circuit FILE.qasm");
            return ExitCode::FAILURE;
        };
        let Some(obs_str) = observable_str.as_deref() else {
            eprintln!("qaoa --circuit also needs --observable PAULI");
            return ExitCode::FAILURE;
        };
        let circuit = match load_circuit(cp) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to load --circuit {}: {}", cp.display(), e);
                return ExitCode::FAILURE;
            }
        };
        let observable = match parse_observable(obs_str) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("Invalid --observable: {e}");
                return ExitCode::FAILURE;
            }
        };
        let header = json!({
            "algo": "qaoa",
            "mode": "manual",
            "circuit_path": cp.display().to_string(),
            "observable": obs_str,
            "depth": depth,
            "circuit_qubits": circuit.num_qubits,
            "circuit_ops": circuit.ops.len(),
            "circuit_params": circuit.symbols.len(),
            "wasm": wasm_path.display().to_string(),
            "fuel": common.fuel,
        });
        let n = circuit.num_qubits as usize;
        return execute(
            &wasm_path,
            common,
            header,
            AlgoTail::Qaoa {
                circuit,
                observable,
                ising_offset: 0.0,
                n,
            },
        );
    }

    let qubo = match build_qubo(graph.as_deref(), edges_str.as_deref(), qubo_file.as_deref()) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("QAOA: {e}");
            return ExitCode::FAILURE;
        }
    };

    let ising = qubo.to_ising();
    let circuit = qaoa_circuit(&ising, depth);
    let observable = ising.to_observable();

    let (best_assignment, best_value) = if qubo.n <= 20 {
        let (x, v) = qubo.brute_force();
        (Some(x), Some(v))
    } else {
        (None, None)
    };

    let header = json!({
        "algo": "qaoa",
        "mode": "auto-from-qubo",
        "qubo_n": qubo.n,
        "qubo_entries": qubo.entries.len(),
        "depth": depth,
        "circuit_qubits": circuit.num_qubits,
        "circuit_ops": circuit.ops.len(),
        "circuit_params": circuit.symbols.len(),
        "ising_offset": ising.offset,
        "wasm": wasm_path.display().to_string(),
        "fuel": common.fuel,
        "brute_force_optimum": best_value,
        "brute_force_assignment": best_assignment.as_ref().map(|v| bool_vec_to_bitstring(v)),
    });

    execute(
        &wasm_path,
        common,
        header,
        AlgoTail::Qaoa {
            circuit,
            observable,
            ising_offset: ising.offset,
            n: qubo.n,
        },
    )
}

#[derive(Clone)]
struct CommonOpts {
    fuel: u64,
    format: OutputFormat,
    verbose: bool,
    optimizer: OptimizerKind,
    gradient: GradMethodChoice,
    max_iters: usize,
    learning_rate: Option<f64>,
    init: Option<String>,
    seed: Option<u64>,
    /// Verbatim path to a JSON file passed to the WASM guest as input.
    input_path: Option<PathBuf>,
    /// Inline JSON passed to the WASM guest as input.
    input_inline: Option<String>,
    /// When true (default false), skip auto-building input JSON from the
    /// convenience flags. Useful if the caller wants the WASM guest to
    /// fall back to its hard-coded defaults.
    no_auto_input: bool,
}

impl Default for CommonOpts {
    fn default() -> Self {
        Self {
            fuel: DEFAULT_FUEL,
            format: OutputFormat::Text,
            verbose: false,
            optimizer: OptimizerKind::WasmGd,
            gradient: GradMethodChoice::Adjoint,
            max_iters: DEFAULT_NATIVE_ITERS,
            learning_rate: None,
            init: None,
            seed: None,
            input_path: None,
            input_inline: None,
            no_auto_input: false,
        }
    }
}

impl CommonOpts {
    fn consume(&mut self, flag: &str, rest: &[String], i: &mut usize) -> bool {
        match flag {
            "--fuel" => {
                *i += 1;
                self.fuel = required(rest, *i, "--fuel").parse().unwrap_or_else(|e| {
                    eprintln!("invalid --fuel: {e}");
                    std::process::exit(1);
                });
                true
            }
            "--format" => {
                *i += 1;
                self.format =
                    OutputFormat::parse(&required(rest, *i, "--format")).unwrap_or_else(|| {
                        eprintln!("invalid --format (expected 'text' or 'json')");
                        std::process::exit(1);
                    });
                true
            }
            "--verbose" => {
                self.verbose = true;
                true
            }
            "--optimizer" => {
                *i += 1;
                let s = required(rest, *i, "--optimizer");
                self.optimizer = OptimizerKind::parse(&s).unwrap_or_else(|| {
                    eprintln!("invalid --optimizer '{s}' (expected wasm-gd | gd | adam | cmaes)");
                    std::process::exit(1);
                });
                true
            }
            "--gradient" => {
                *i += 1;
                let s = required(rest, *i, "--gradient");
                self.gradient = GradMethodChoice::parse(&s).unwrap_or_else(|| {
                    eprintln!(
                        "invalid --gradient '{s}' (expected adjoint | param-shift | finite-diff)"
                    );
                    std::process::exit(1);
                });
                true
            }
            "--max-iters" => {
                *i += 1;
                self.max_iters = required(rest, *i, "--max-iters")
                    .parse()
                    .unwrap_or_else(|e| {
                        eprintln!("invalid --max-iters: {e}");
                        std::process::exit(1);
                    });
                true
            }
            "--learning-rate" | "--lr" => {
                *i += 1;
                self.learning_rate = Some(
                    required(rest, *i, "--learning-rate")
                        .parse()
                        .unwrap_or_else(|e| {
                            eprintln!("invalid --learning-rate: {e}");
                            std::process::exit(1);
                        }),
                );
                true
            }
            "--init" => {
                *i += 1;
                self.init = Some(required(rest, *i, "--init"));
                true
            }
            "--seed" => {
                *i += 1;
                self.seed = Some(required(rest, *i, "--seed").parse().unwrap_or_else(|e| {
                    eprintln!("invalid --seed: {e}");
                    std::process::exit(1);
                }));
                true
            }
            "--input" => {
                *i += 1;
                self.input_path = Some(PathBuf::from(required(rest, *i, "--input")));
                true
            }
            "--input-inline" => {
                *i += 1;
                self.input_inline = Some(required(rest, *i, "--input-inline"));
                true
            }
            "--no-auto-input" => {
                self.no_auto_input = true;
                true
            }
            _ => false,
        }
    }
}

/// Resolve the WASM input bytes to stage on `HostState` before running the
/// guest. Priority: `--input FILE` (verbatim) > `--input-inline JSON`
/// (verbatim) > auto-built JSON synthesised from the convenience flags
/// (suppressed by `--no-auto-input` and when no auto fields apply).
fn resolve_wasm_input(common: &CommonOpts, auto: Value) -> Result<Vec<u8>, String> {
    if let Some(p) = common.input_path.as_deref() {
        return std::fs::read(p).map_err(|e| format!("read --input {}: {e}", p.display()));
    }
    if let Some(s) = common.input_inline.as_deref() {
        return Ok(s.as_bytes().to_vec());
    }
    if common.no_auto_input {
        return Ok(Vec::new());
    }
    let obj = auto.as_object().cloned().unwrap_or_default();
    if obj.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::to_vec(&Value::Object(obj)).map_err(|e| format!("serialise auto input: {e}"))
}

/// Build the auto-generated input JSON for the WASM guest from the
/// convenience flags users may have set. Empty object if nothing applies.
fn auto_input_for_wasm(common: &CommonOpts, extra: &[(&str, Value)]) -> Value {
    let mut obj = serde_json::Map::new();
    if common.optimizer.is_wasm() {
        obj.insert(
            "optimizer".into(),
            json!(common.optimizer.guest_optimizer_name()),
        );
    }
    if common.max_iters != DEFAULT_NATIVE_ITERS {
        obj.insert("max_iters".into(), json!(common.max_iters));
    }
    if let Some(lr) = common.learning_rate {
        obj.insert("lr".into(), json!(lr));
    }
    if let Some(init) = common.init.as_deref() {
        if init != "zeros" {
            let v: Vec<f64> = init
                .split(',')
                .filter_map(|s| s.trim().parse::<f64>().ok())
                .collect();
            if !v.is_empty() {
                obj.insert("init".into(), json!(v));
            }
        }
    }
    for (k, v) in extra {
        obj.insert((*k).to_string(), v.clone());
    }
    Value::Object(obj)
}

/// Build the initial parameter vector for native optimizers.
///
/// `init`:
///   - `None`     → all zeros (length = `n_params`).
///   - "zeros"    → all zeros.
///   - "v1,v2,…"  → exact length must match `n_params`.
fn build_initial_params(init: Option<&str>, n_params: usize) -> Result<Vec<f64>, String> {
    match init {
        None | Some("zeros") => Ok(vec![0.0; n_params]),
        Some(s) => {
            let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
            if parts.len() != n_params {
                return Err(format!(
                    "--init expected {n_params} comma-separated values, got {}",
                    parts.len()
                ));
            }
            parts
                .into_iter()
                .map(|p| {
                    p.parse::<f64>()
                        .map_err(|e| format!("bad --init value '{p}': {e}"))
                })
                .collect()
        }
    }
}

/// Build a `ParameterBinding` from a flat parameter vector aligned with
/// the circuit's symbol-id sort order — same convention used by both the
/// host and the WASM guest path.
fn binding_from_flat(circuit: &CircuitIR, params: &[f64]) -> ParameterBinding {
    let mut binding = ParameterBinding::new();
    let mut sym_ids: Vec<SymbolId> = circuit.symbols.keys().copied().collect();
    sym_ids.sort();
    for (idx, &sym) in sym_ids.iter().enumerate() {
        binding.bind(sym, params.get(idx).copied().unwrap_or(0.0));
    }
    binding
}

/// Reorder gradient pairs into a flat Vec aligned with the sorted-symbol
/// param vector. Missing symbols default to 0.0 (shouldn't happen, but
/// keeps us robust against backend quirks).
fn gradient_to_flat(circuit: &CircuitIR, pairs: &[(SymbolId, f64)]) -> Vec<f64> {
    let mut sym_ids: Vec<SymbolId> = circuit.symbols.keys().copied().collect();
    sym_ids.sort();
    let mut out = vec![0.0; sym_ids.len()];
    for (sid, g) in pairs {
        if let Some(idx) = sym_ids.iter().position(|x| x == sid) {
            out[idx] = *g;
        }
    }
    out
}

enum AlgoTail {
    Vqe {
        circuit: CircuitIR,
        observable: Observable,
    },
    Qaoa {
        circuit: CircuitIR,
        observable: Observable,
        ising_offset: f64,
        n: usize,
    },
}

impl AlgoTail {
    fn circuit(&self) -> &CircuitIR {
        match self {
            Self::Vqe { circuit, .. } | Self::Qaoa { circuit, .. } => circuit,
        }
    }
    fn observable(&self) -> &Observable {
        match self {
            Self::Vqe { observable, .. } | Self::Qaoa { observable, .. } => observable,
        }
    }
}

fn execute(wasm_path: &Path, common: CommonOpts, mut header: Value, tail: AlgoTail) -> ExitCode {
    if common.optimizer.is_wasm() {
        if let Some(obj) = header.as_object_mut() {
            obj.insert("optimizer".into(), json!(common.optimizer.as_str()));
        }
        return execute_wasm(wasm_path, common, header, tail);
    }
    if let Some(obj) = header.as_object_mut() {
        obj.insert("optimizer".into(), json!(common.optimizer.as_str()));
        if common.optimizer.needs_gradient() {
            obj.insert("gradient".into(), json!(common.gradient.as_str()));
        }
        obj.insert("max_iters".into(), json!(common.max_iters));
    }
    execute_native(common, header, tail)
}

fn execute_wasm(wasm_path: &Path, common: CommonOpts, header: Value, tail: AlgoTail) -> ExitCode {
    let bytes = match fs::read(wasm_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read {}: {}", wasm_path.display(), e);
            eprintln!(
                "(Build the guests with `./build-dist.sh` or \
                 `cd examples/wasm-guests/<algo> && cargo build --target wasm32-wasip1 --release`.)"
            );
            return ExitCode::FAILURE;
        }
    };

    let mut host = HostState::new();
    let cid = host.register_circuit(tail.circuit().clone());
    let oid = host.register_observable(tail.observable().clone());
    debug_assert_eq!((cid, oid), (1, 1));

    // Stage the input payload the guest will read via `omega_input_*`. Picks
    // up `--input` / `--input-inline` verbatim, otherwise auto-builds from
    // convenience flags (max-iters, learning-rate, init, optimizer choice).
    let auto = auto_input_for_wasm(&common, &[]);
    let input_bytes = match resolve_wasm_input(&common, auto) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    host.set_input(input_bytes);

    let mut runner = match WasmRunner::new(host) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to init wasmtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    if matches!(common.format, OutputFormat::Json) {
        runner.redirect_guest_stdout_to_stderr(true);
    }
    // Forward the same RNG seed that drives the native optimiser to the
    // guest's WASI random source — keeps the whole run reproducible.
    runner.rng_seed(common.seed);

    if matches!(common.format, OutputFormat::Text) {
        print_header_text(&header);
        println!("--- guest stdout ---");
    }

    let wasm_result = match runner.run(&bytes, common.fuel) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[host] WASM execution error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let progress = {
        let s = runner.host_state().lock().unwrap();
        s.progress.clone()
    };

    let Some(wasm_res) = wasm_result else {
        eprintln!(
            "[host] WASM completed but never called omega_report_result \
             (got {} progress entries)",
            progress.len()
        );
        return ExitCode::FAILURE;
    };

    let opt = OptimizationResult {
        optimal_params: wasm_res.optimal_params,
        optimal_value: wasm_res.optimal_value,
        iterations: wasm_res.iterations as usize,
        progress,
    };

    print_tail(&header, &opt, &tail, common, /* native */ false);
    ExitCode::SUCCESS
}

fn execute_native(common: CommonOpts, header: Value, tail: AlgoTail) -> ExitCode {
    let circuit = tail.circuit().clone();
    let observable = tail.observable().clone();
    let n_params = circuit.symbols.len();

    if n_params == 0 {
        eprintln!("native optimizer needs a circuit with at least one free parameter");
        return ExitCode::FAILURE;
    }

    let initial = match build_initial_params(common.init.as_deref(), n_params) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    if matches!(common.format, OutputFormat::Text) {
        print_header_text(&header);
        println!("--- native optimizer ({}) ---", common.optimizer.as_str());
    }

    let backend = StatevectorBackend::new();
    let grad_method = common.gradient.into_method();

    let cost = |params: &[f64]| -> f64 {
        let binding = binding_from_flat(&circuit, params);
        backend
            .expectation(&circuit, &binding, &observable)
            .unwrap_or(f64::NAN)
    };
    let grad = |params: &[f64]| -> Vec<f64> {
        let binding = binding_from_flat(&circuit, params);
        match compute_gradient(&backend, &circuit, &binding, &observable, &grad_method) {
            Ok(pairs) => gradient_to_flat(&circuit, &pairs),
            Err(e) => {
                eprintln!("[host] gradient error: {e}");
                vec![0.0; params.len()]
            }
        }
    };

    let opt = match common.optimizer {
        OptimizerKind::Gd => optimizers::run_gd(
            initial,
            cost,
            grad,
            common.max_iters,
            common.learning_rate.unwrap_or(DEFAULT_GD_LR),
            NATIVE_CONV_TOL,
        ),
        OptimizerKind::Adam => optimizers::run_adam(
            initial,
            cost,
            grad,
            common.max_iters,
            common.learning_rate.unwrap_or(DEFAULT_ADAM_LR),
            NATIVE_CONV_TOL,
        ),
        OptimizerKind::CmaEs => optimizers::run_cmaes(
            initial,
            cost,
            common.max_iters,
            DEFAULT_CMAES_SIGMA,
            common.seed.or(Some(42)),
        ),
        OptimizerKind::WasmGd | OptimizerKind::WasmAdam => {
            unreachable!("WASM optimizers are handled in execute()")
        }
    };

    print_tail(&header, &opt, &tail, common, /* native */ true);
    ExitCode::SUCCESS
}

fn print_header_text(header: &Value) {
    println!("=== omega-wasm ===");
    if let Some(o) = header.as_object() {
        for (k, v) in o {
            println!("  {k}: {v}");
        }
    }
    println!();
}

fn print_tail(
    header: &Value,
    res: &OptimizationResult,
    tail: &AlgoTail,
    common: CommonOpts,
    native: bool,
) {
    match tail {
        AlgoTail::Vqe { .. } => print_vqe_tail(header, res, common, native),
        AlgoTail::Qaoa {
            circuit,
            ising_offset,
            n,
            ..
        } => {
            let samples = sample_circuit(circuit, &res.optimal_params, *n);
            print_qaoa_tail(header, res, &samples, *ising_offset, common, native);
        }
    }
}

fn print_vqe_tail(header: &Value, res: &OptimizationResult, common: CommonOpts, native: bool) {
    match common.format {
        OutputFormat::Text => {
            if !native {
                println!("--- end guest stdout ---\n");
            } else {
                println!();
            }
            if common.verbose {
                let lbl = if native { "native loop" } else { "i64 → f64" };
                println!("Per-iteration value stream ({lbl}):");
                for &(iter, v) in &res.progress {
                    println!("  iter {:>4}  E = {:.10}", iter, v);
                }
                println!();
            }
            println!("Result:");
            println!("  optimal_value (energy)  = {:.10}", res.optimal_value);
            println!("  optimal_params          = {:?}", res.optimal_params);
            println!("  reported iterations     = {}", res.iterations);
            println!("  total progress entries  = {}", res.progress.len());
        }
        OutputFormat::Json => {
            let doc = json!({
                "header": header,
                "result": {
                    "optimal_value": res.optimal_value,
                    "optimal_params": res.optimal_params,
                    "reported_iterations": res.iterations,
                },
                "progress": res.progress
                    .iter()
                    .map(|&(it, v)| json!({"iter": it, "value": v}))
                    .collect::<Vec<_>>(),
            });
            println!("{doc}");
        }
    }
}

fn print_qaoa_tail(
    header: &Value,
    res: &OptimizationResult,
    samples: &[(u64, u32)],
    ising_offset: f64,
    common: CommonOpts,
    native: bool,
) {
    let cost_with_offset = res.optimal_value + ising_offset;
    match common.format {
        OutputFormat::Text => {
            if !native {
                println!("--- end guest stdout ---\n");
            } else {
                println!();
            }
            if common.verbose {
                let lbl = if native { "native loop" } else { "i64 → f64" };
                println!("Per-iteration cost stream ({lbl}):");
                for &(iter, v) in &res.progress {
                    println!("  iter {:>4}  cost = {:.10}", iter, v);
                }
                println!();
            }
            println!("Result:");
            println!(
                "  optimal_value (Ising H, no offset)  = {:.10}",
                res.optimal_value
            );
            println!(
                "  optimal_value + ising_offset (QUBO) = {:.10}",
                cost_with_offset
            );
            println!(
                "  optimal_params                       = {:?}",
                res.optimal_params
            );
            println!(
                "  reported iterations                  = {}",
                res.iterations
            );
            println!(
                "  total progress entries               = {}",
                res.progress.len()
            );

            println!(
                "\nTop-{} measurement bitstrings at optimum (shots={}):",
                samples.len().min(TOP_K_BITSTRINGS),
                SAMPLE_SHOTS
            );
            let n = header
                .get("qubo_n")
                .or_else(|| header.get("circuit_qubits"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            for &(bits, count) in samples.iter().take(TOP_K_BITSTRINGS) {
                let pct = 100.0 * count as f64 / SAMPLE_SHOTS as f64;
                let bitstring = bits_to_bitstring(bits, n);
                println!("  |{bitstring}>  count = {count:>4}  ({pct:5.2}%)");
            }
        }
        OutputFormat::Json => {
            let n = header
                .get("qubo_n")
                .or_else(|| header.get("circuit_qubits"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let doc = json!({
                "header": header,
                "result": {
                    "optimal_value_ising": res.optimal_value,
                    "optimal_value_qubo": cost_with_offset,
                    "optimal_params": res.optimal_params,
                    "reported_iterations": res.iterations,
                },
                "progress": res.progress
                    .iter()
                    .map(|&(it, v)| json!({"iter": it, "value": v}))
                    .collect::<Vec<_>>(),
                "samples": samples
                    .iter()
                    .map(|&(bits, count)| {
                        json!({
                            "bitstring": bits_to_bitstring(bits, n),
                            "bits_le_u64": bits,
                            "count": count,
                        })
                    })
                    .collect::<Vec<_>>(),
            });
            println!("{doc}");
        }
    }
}

fn sample_circuit(circuit: &CircuitIR, params: &[f64], _n_vars: usize) -> Vec<(u64, u32)> {
    let mut binding = ParameterBinding::new();
    let mut sym_ids: Vec<SymbolId> = circuit.symbols.keys().copied().collect();
    sym_ids.sort();
    for (idx, &sym) in sym_ids.iter().enumerate() {
        binding.bind(sym, params.get(idx).copied().unwrap_or(0.0));
    }
    let cfg = ExecConfig {
        shots: Some(SAMPLE_SHOTS),
        seed: Some(SAMPLE_SEED),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let backend = StatevectorBackend::new();
    match backend.execute(circuit, &binding, &cfg) {
        Ok(ExecResult::Counts(c)) => {
            let mut v: Vec<(u64, u32)> = c.into_iter().collect();
            v.sort_by_key(|b| std::cmp::Reverse(b.1));
            v
        }
        _ => Vec::new(),
    }
}

fn bits_to_bitstring(bits: u64, n: usize) -> String {
    (0..n)
        .rev()
        .map(|q| if (bits >> q) & 1 == 1 { '1' } else { '0' })
        .collect()
}

fn bool_vec_to_bitstring(x: &[bool]) -> String {
    let mut s = String::with_capacity(x.len());
    for &b in x.iter().rev() {
        s.push(if b { '1' } else { '0' });
    }
    s
}

fn build_qubo(
    graph: Option<&str>,
    edges: Option<&str>,
    qubo_file: Option<&Path>,
) -> Result<Qubo, String> {
    if let Some(path) = qubo_file {
        return load_qubo_json(path);
    }
    let edges = match (graph, edges) {
        (Some(name), None) => match name {
            "triangle" => vec![(0, 1), (1, 2), (0, 2)],
            "square" => vec![(0, 1), (1, 2), (2, 3), (3, 0)],
            "k4" => vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            other => return Err(format!("unknown --graph {other}")),
        },
        (None, Some(s)) => parse_edges(s)?,
        (None, None) => vec![(0, 1), (1, 2), (0, 2)],
        (Some(_), Some(_)) => {
            return Err("--graph and --edges are mutually exclusive".into());
        }
    };
    Ok(maxcut_qubo(&edges))
}

fn parse_edges(s: &str) -> Result<Vec<(usize, usize)>, String> {
    s.split(',')
        .map(|tok| {
            let tok = tok.trim();
            let (a, b) = tok
                .split_once('-')
                .ok_or_else(|| format!("bad edge '{tok}' (expected 'i-j')"))?;
            let i: usize = a
                .trim()
                .parse()
                .map_err(|e| format!("bad edge '{tok}': {e}"))?;
            let j: usize = b
                .trim()
                .parse()
                .map_err(|e| format!("bad edge '{tok}': {e}"))?;
            Ok((i, j))
        })
        .collect()
}

fn maxcut_qubo(edges: &[(usize, usize)]) -> Qubo {
    let n = edges.iter().map(|&(i, j)| i.max(j)).max().unwrap_or(0) + 1;
    let mut q = Qubo::new(n);
    for &(i, j) in edges {
        q.add(i, i, -1.0);
        q.add(j, j, -1.0);
        q.add(i, j, 2.0);
    }
    q
}

fn load_qubo_json(path: &Path) -> Result<Qubo, String> {
    let s = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let v: Value = serde_json::from_str(&s).map_err(|e| format!("parse JSON: {e}"))?;
    let n = v.get("n").and_then(|x| x.as_u64()).ok_or("missing 'n'")? as usize;
    let q_arr = v
        .get("Q")
        .and_then(|x| x.as_array())
        .ok_or("missing 'Q' array")?;
    let mut q = Qubo::new(n);
    for entry in q_arr {
        let arr = entry.as_array().ok_or("Q entry not an array")?;
        if arr.len() != 3 {
            return Err("Q entry must be [i, j, c]".into());
        }
        let i = arr[0].as_u64().ok_or("Q[0] not int")? as usize;
        let j = arr[1].as_u64().ok_or("Q[1] not int")? as usize;
        let c = arr[2].as_f64().ok_or("Q[2] not float")?;
        q.add(i, j, c);
    }
    Ok(q)
}

fn load_circuit(path: &Path) -> Result<CircuitIR, String> {
    let src = fs::read_to_string(path).map_err(|e| format!("read: {e}"))?;
    lower_to_ir(&src).map_err(|e| format!("parse: {e}"))
}

fn simple_2q_ansatz() -> CircuitIR {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.symbols.insert(0, "theta0".to_string());
    c.symbols.insert(1, "theta1".to_string());
    c.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });
    c.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(1)],
        params: smallvec![ParamExpr::Symbol(1)],
        classical_bit: None,
        condition: None,
    });
    c.add_op(GateOp {
        gate: GateKind::CX,
        qubits: smallvec![Qubit(0), Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    c
}

fn parse_observable(s: &str) -> Result<Observable, String> {
    let s = s.replace(' ', "");
    let mut terms = Vec::new();
    for raw_part in s.split('+') {
        if raw_part.is_empty() {
            continue;
        }
        let (coeff, pauli_part) = if let Some(idx) = raw_part.find('*') {
            let c: f64 = raw_part[..idx]
                .parse()
                .map_err(|e| format!("coefficient '{}': {e}", &raw_part[..idx]))?;
            (c, &raw_part[idx + 1..])
        } else if raw_part.starts_with(['X', 'Y', 'Z', 'I']) {
            (1.0, raw_part)
        } else {
            let idx = raw_part
                .find(['X', 'Y', 'Z', 'I'])
                .ok_or_else(|| format!("no Pauli letter in term '{raw_part}'"))?;
            let c: f64 = raw_part[..idx]
                .parse()
                .map_err(|e| format!("coefficient '{}': {e}", &raw_part[..idx]))?;
            (c, &raw_part[idx..])
        };
        terms.push((coeff, parse_pauli_string(pauli_part)?));
    }
    if terms.is_empty() {
        return Err("empty observable".into());
    }
    Ok(Observable { terms })
}

fn parse_pauli_string(s: &str) -> Result<Vec<(u32, PauliOp)>, String> {
    let mut out = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let op = match chars[i] {
            'X' | 'x' => PauliOp::X,
            'Y' | 'y' => PauliOp::Y,
            'Z' | 'z' => PauliOp::Z,
            'I' | 'i' => PauliOp::I,
            other => return Err(format!("bad Pauli '{other}' in '{s}'")),
        };
        i += 1;
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if start == i {
            return Err(format!("missing qubit index after Pauli in '{s}'"));
        }
        let q: u32 = s[start..i].parse().map_err(|e| format!("qubit idx: {e}"))?;
        if !matches!(op, PauliOp::I) {
            out.push((q, op));
        }
    }
    Ok(out)
}

fn required(rest: &[String], i: usize, flag: &str) -> String {
    rest.get(i).cloned().unwrap_or_else(|| {
        eprintln!("{flag}: missing argument");
        std::process::exit(1);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_pauli_string_single_z() {
        let v = parse_pauli_string("Z0").unwrap();
        assert_eq!(v, vec![(0, PauliOp::Z)]);
    }

    #[test]
    fn parse_pauli_string_zz_pair() {
        let v = parse_pauli_string("Z0Z1").unwrap();
        assert_eq!(v, vec![(0, PauliOp::Z), (1, PauliOp::Z)]);
    }

    #[test]
    fn parse_pauli_string_drops_identity() {
        // I is the identity — it must not appear in the term's Pauli list.
        let v = parse_pauli_string("I0X1").unwrap();
        assert_eq!(v, vec![(1, PauliOp::X)]);
    }

    #[test]
    fn parse_pauli_string_two_digit_qubit() {
        let v = parse_pauli_string("X10Z11").unwrap();
        assert_eq!(v, vec![(10, PauliOp::X), (11, PauliOp::Z)]);
    }

    #[test]
    fn parse_pauli_string_rejects_bare_letter() {
        // Pauli letter without qubit index → error, not silent drop.
        let err = parse_pauli_string("Z").unwrap_err();
        assert!(err.contains("missing qubit index"), "{err}");
    }

    #[test]
    fn parse_observable_implicit_one_coeff() {
        let obs = parse_observable("Z0Z1").unwrap();
        assert_eq!(obs.terms.len(), 1);
        assert_eq!(obs.terms[0].0, 1.0);
        assert_eq!(obs.terms[0].1, vec![(0, PauliOp::Z), (1, PauliOp::Z)]);
    }

    #[test]
    fn parse_observable_h2_like_sum() {
        // Reproduces the H2 toy Hamiltonian shape used by host::h2_hamiltonian.
        let obs = parse_observable("0.3979*Z0+-0.3979*Z1+-0.0112*Z0Z1+0.1809*X0X1").unwrap();
        assert_eq!(obs.terms.len(), 4);
        let coeffs: Vec<f64> = obs.terms.iter().map(|t| t.0).collect();
        assert!((coeffs[0] - 0.3979).abs() < 1e-12);
        assert!((coeffs[1] + 0.3979).abs() < 1e-12);
        assert!((coeffs[2] + 0.0112).abs() < 1e-12);
        assert!((coeffs[3] - 0.1809).abs() < 1e-12);
    }

    #[test]
    fn parse_observable_rejects_empty() {
        assert!(parse_observable("").is_err());
    }

    #[test]
    fn parse_edges_basic() {
        let v = parse_edges("0-1, 1-2, 2-3").unwrap();
        assert_eq!(v, vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn parse_edges_rejects_garbage() {
        assert!(parse_edges("0-1, foo").is_err());
        assert!(parse_edges("0_1").is_err());
    }

    #[test]
    fn maxcut_qubo_triangle_has_brute_force_optimum_minus_two() {
        // Triangle MaxCut: max cut size = 2 (any 2-vs-1 partition); QUBO
        // contribution per edge is x_i + x_j - 2*x_i*x_j → minimum is -2.
        let q = maxcut_qubo(&[(0, 1), (1, 2), (0, 2)]);
        assert_eq!(q.n, 3);
        let (_x, val) = q.brute_force();
        assert!((val + 2.0).abs() < 1e-12, "got {val}");
    }

    #[test]
    fn maxcut_qubo_size_inferred_from_edges() {
        let q = maxcut_qubo(&[(0, 5)]);
        assert_eq!(q.n, 6);
    }

    #[test]
    fn bits_to_bitstring_msb_first() {
        // bit 0 is qubit 0; we render MSB first so bit 2 is leftmost.
        assert_eq!(bits_to_bitstring(0b101, 3), "101");
        assert_eq!(bits_to_bitstring(0b001, 3), "001");
        assert_eq!(bits_to_bitstring(0, 4), "0000");
    }

    #[test]
    fn bool_vec_to_bitstring_matches_bits() {
        assert_eq!(bool_vec_to_bitstring(&[true, false, true]), "101");
        assert_eq!(bool_vec_to_bitstring(&[false, false, false]), "000");
    }

    #[test]
    fn load_qubo_json_roundtrip() {
        // Same QUBO as `maxcut_qubo(triangle_edges)`. Each vertex sits on 2
        // edges, so its diagonal accumulates -1 + -1 = -2; off-diagonals are
        // +2 per edge. Brute-force minimum should be -2 (max cut size = 2).
        let tmp = std::env::temp_dir().join("omega_wasm_cli_qubo_test.json");
        std::fs::write(
            &tmp,
            r#"{"n":3,"Q":[[0,0,-2],[1,1,-2],[2,2,-2],[0,1,2],[1,2,2],[0,2,2]]}"#,
        )
        .unwrap();
        let q = load_qubo_json(&tmp).unwrap();
        assert_eq!(q.n, 3);
        let (_x, val) = q.brute_force();
        assert!((val + 2.0).abs() < 1e-12, "got {val}");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn load_qubo_json_rejects_missing_fields() {
        let tmp = std::env::temp_dir().join("omega_wasm_cli_qubo_bad.json");
        std::fs::write(&tmp, r#"{"Q":[[0,0,1]]}"#).unwrap();
        assert!(load_qubo_json(&tmp).is_err());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn output_format_parse() {
        assert!(matches!(
            OutputFormat::parse("text"),
            Some(OutputFormat::Text)
        ));
        assert!(matches!(
            OutputFormat::parse("json"),
            Some(OutputFormat::Json)
        ));
        assert!(OutputFormat::parse("yaml").is_none());
    }

    #[test]
    fn simple_2q_ansatz_has_two_symbols() {
        let c = simple_2q_ansatz();
        assert_eq!(c.num_qubits, 2);
        assert_eq!(c.symbols.len(), 2);
        assert_eq!(c.ops.len(), 3);
    }

    #[test]
    fn vqe_qasm_example_parses_with_two_params() {
        // Verify the shipped example circuit matches what vqe.wasm expects.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/circuits/vqe_ansatz_2q.qasm");
        let c = load_circuit(&path).unwrap();
        assert_eq!(c.num_qubits, 2, "VQE example must be 2-qubit");
        assert_eq!(
            c.symbols.len(),
            2,
            "VQE example must declare exactly 2 free params (vqe.wasm guest hard-codes NUM_PARAMS=2)"
        );
    }

    #[test]
    fn qaoa_triangle_qasm_example_parses_with_two_params() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/circuits/qaoa_maxcut_triangle_p1.qasm");
        let c = load_circuit(&path).unwrap();
        assert_eq!(c.num_qubits, 3);
        assert_eq!(
            c.symbols.len(),
            2,
            "depth-1 QAOA must declare exactly 2 free params (gamma, beta)"
        );
    }

    #[test]
    fn qaoa_square_qasm_example_parses_with_two_params() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/circuits/qaoa_maxcut_square_p1.qasm");
        let c = load_circuit(&path).unwrap();
        assert_eq!(c.num_qubits, 4);
        assert_eq!(c.symbols.len(), 2);
    }

    #[test]
    fn sample_circuit_at_zero_params_recovers_uniform() {
        // QAOA circuit at gamma=0, beta=0 is just H^n |0>^n = |+>^n,
        // which under Z-basis sampling is uniform over all 2^n bitstrings.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/circuits/qaoa_maxcut_triangle_p1.qasm");
        let circuit = load_circuit(&path).unwrap();
        let samples = sample_circuit(&circuit, &[0.0, 0.0], 3);
        let total: u32 = samples.iter().map(|&(_, c)| c).sum();
        assert_eq!(total, SAMPLE_SHOTS);
        // 8 outcomes for 3 qubits; with seed=0x51517E5 each should appear
        // at least once and within ~50% of uniform expectation (512 each).
        assert!(samples.len() == 8, "got {} unique outcomes", samples.len());
        let expected = SAMPLE_SHOTS as f64 / 8.0;
        for &(_, c) in &samples {
            let dev = (c as f64 - expected).abs() / expected;
            assert!(dev < 0.25, "outcome count {c} far from uniform {expected}");
        }
    }

    // ---- native optimizer end-to-end (#2 + #3 fixes) ----

    use omega_backend_statevector::StatevectorBackend;
    use omega_core::executor::Backend;
    use omega_core::gradient::compute_gradient;

    fn vqe_h2_setup() -> (CircuitIR, Observable) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/circuits/vqe_ansatz_2q.qasm");
        (load_circuit(&path).unwrap(), h2_hamiltonian())
    }

    #[allow(clippy::type_complexity)]
    fn make_cost_grad(
        circuit: CircuitIR,
        observable: Observable,
    ) -> (impl Fn(&[f64]) -> f64, impl Fn(&[f64]) -> Vec<f64>) {
        let backend = StatevectorBackend::new();
        let c1 = circuit.clone();
        let o1 = observable.clone();
        let cost = move |p: &[f64]| -> f64 {
            let b = binding_from_flat(&c1, p);
            backend.expectation(&c1, &b, &o1).unwrap_or(f64::NAN)
        };
        let backend2 = StatevectorBackend::new();
        let c2 = circuit;
        let o2 = observable;
        let grad = move |p: &[f64]| -> Vec<f64> {
            let b = binding_from_flat(&c2, p);
            let pairs = compute_gradient(&backend2, &c2, &b, &o2, &GradMethod::Adjoint).unwrap();
            gradient_to_flat(&c2, &pairs)
        };
        (cost, grad)
    }

    #[test]
    fn build_initial_params_zeros_default() {
        let v = build_initial_params(None, 4).unwrap();
        assert_eq!(v, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn build_initial_params_explicit_list() {
        let v = build_initial_params(Some("0.1, 0.2, -0.3"), 3).unwrap();
        assert!((v[0] - 0.1).abs() < 1e-12);
        assert!((v[1] - 0.2).abs() < 1e-12);
        assert!((v[2] + 0.3).abs() < 1e-12);
    }

    #[test]
    fn build_initial_params_rejects_length_mismatch() {
        let err = build_initial_params(Some("0.1,0.2"), 3).unwrap_err();
        assert!(err.contains("expected 3"));
    }

    #[test]
    fn binding_from_flat_aligns_with_sorted_symbol_ids() {
        // Symbols inserted out of order; binding must map by sorted id.
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        c.symbols.insert(7, "b".into());
        c.symbols.insert(3, "a".into());
        let b = binding_from_flat(&c, &[1.5, 2.5]);
        // Sorted: id=3 → 1.5, id=7 → 2.5
        assert!((b.get(3).unwrap() - 1.5).abs() < 1e-12);
        assert!((b.get(7).unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn gradient_to_flat_aligns_pairs() {
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        c.symbols.insert(7, "b".into());
        c.symbols.insert(3, "a".into());
        // Pairs in arbitrary order; flat must come out sorted by id.
        let pairs = vec![(7, 0.4), (3, 0.7)];
        let flat = gradient_to_flat(&c, &pairs);
        assert!((flat[0] - 0.7).abs() < 1e-12);
        assert!((flat[1] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn native_adam_recovers_h2_ground_state() {
        let (circuit, observable) = vqe_h2_setup();
        let (cost, grad) = make_cost_grad(circuit, observable);
        let r = optimizers::run_adam(vec![0.1, 0.2], cost, grad, 500, 0.1, 1e-12);
        // Same H2 toy reduction the WASM guest converges to ≈ -0.192.
        assert!(r.optimal_value < -0.18, "got {}", r.optimal_value);
    }

    #[test]
    fn native_gd_recovers_h2_ground_state() {
        let (circuit, observable) = vqe_h2_setup();
        let (cost, grad) = make_cost_grad(circuit, observable);
        let r = optimizers::run_gd(vec![0.1, 0.2], cost, grad, 500, 0.4, 1e-12);
        assert!(r.optimal_value < -0.18, "got {}", r.optimal_value);
    }

    #[test]
    fn native_cmaes_recovers_h2_ground_state() {
        let (circuit, observable) = vqe_h2_setup();
        let (cost, _grad) = make_cost_grad(circuit, observable);
        let r = optimizers::run_cmaes(vec![0.1, 0.2], cost, 60, 0.5, Some(7));
        assert!(r.optimal_value < -0.15, "got {}", r.optimal_value);
    }

    #[test]
    fn native_path_handles_four_param_vqe_ansatz() {
        // The whole point of fix #2: NUM_PARAMS=2 cap is gone in the
        // native path. Load the 4-param ansatz, run Adam, confirm the
        // optimizer was given 4 params (not silently truncated) AND
        // made meaningful progress. We don't require every param to
        // move, since the gradient at the initial point can be near-zero
        // along some directions.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/circuits/vqe_ansatz_4q.qasm");
        let circuit = load_circuit(&path).unwrap();
        assert_eq!(circuit.symbols.len(), 4);

        let observable = parse_observable("0.5*Z0+0.3*Z1Z2+0.2*X2X3").unwrap();
        let (cost, grad) = make_cost_grad(circuit.clone(), observable.clone());

        let initial = vec![0.5, 0.3, -0.2, 0.4];
        let initial_cost = cost(&initial);
        let r = optimizers::run_adam(initial, cost, grad, 500, 0.1, 1e-12);

        // 4-D parameter vector survives the round trip — proves the
        // optimizer did NOT truncate to NUM_PARAMS=2.
        assert_eq!(r.optimal_params.len(), 4);
        // Cost actually decreased.
        assert!(
            r.optimal_value < initial_cost - 0.05,
            "no meaningful progress: initial {initial_cost} → final {}",
            r.optimal_value
        );
        // Final cost is strictly below what a 2-param truncation could
        // hit: if we'd truncated to 2 params, theta2/theta3 would stay
        // at the initial values, producing a higher floor. Sanity-check
        // by running a 2-param-only Adam and comparing.
        let trunc_observable = parse_observable("0.5*Z0+0.3*Z1Z2+0.2*X2X3").unwrap();
        let trunc_circuit = {
            let mut c = circuit.clone();
            // Pretend symbols 2,3 don't exist for the optimizer (mimics
            // the WASM guest's NUM_PARAMS=2 truncation: bind them at
            // their initial values inside the cost closure).
            c.symbols.remove(&2);
            c.symbols.remove(&3);
            c
        };
        // The trunc circuit's cost still uses the full circuit at the
        // sealed initial values for theta2/theta3 — equivalent to
        // NUM_PARAMS=2 padding.
        let trunc_cost_full =
            cost_with_fixed_tail(circuit.clone(), trunc_observable.clone(), vec![-0.2, 0.4]);
        let r_trunc = optimizers::run_adam(
            vec![0.5, 0.3],
            trunc_cost_full,
            |p| {
                // No real gradient — just the projected one. The point of
                // this comparison is the *floor*, not the trajectory, so
                // a finite-diff is fine.
                let cost = cost_with_fixed_tail(
                    trunc_circuit.clone(),
                    trunc_observable.clone(),
                    vec![-0.2, 0.4],
                );
                let eps = 1e-5;
                let mut g = vec![0.0; p.len()];
                let f0 = cost(p);
                for i in 0..p.len() {
                    let mut pp = p.to_vec();
                    pp[i] += eps;
                    g[i] = (cost(&pp) - f0) / eps;
                }
                g
            },
            300,
            0.1,
            1e-12,
        );
        assert!(
            r.optimal_value < r_trunc.optimal_value + 1e-6,
            "4-param native didn't beat 2-param truncation: {} vs {}",
            r.optimal_value,
            r_trunc.optimal_value
        );
    }

    fn cost_with_fixed_tail(
        circuit: CircuitIR,
        observable: Observable,
        fixed_tail: Vec<f64>,
    ) -> impl Fn(&[f64]) -> f64 {
        let backend = StatevectorBackend::new();
        move |head: &[f64]| -> f64 {
            let mut all = head.to_vec();
            all.extend_from_slice(&fixed_tail);
            let b = binding_from_flat(&circuit, &all);
            backend
                .expectation(&circuit, &b, &observable)
                .unwrap_or(f64::NAN)
        }
    }

    #[test]
    fn native_qaoa_depth2_recovers_triangle_optimum() {
        // The other half of fix #2: the WASM path was capped at depth=1
        // (2 params). Native CMA-ES handles depth=2 (4 params) without
        // touching the WASM guest.
        let qubo = maxcut_qubo(&[(0, 1), (1, 2), (0, 2)]);
        let ising = qubo.to_ising();
        let circuit = qaoa_circuit(&ising, 2);
        let observable = ising.to_observable();
        assert_eq!(circuit.symbols.len(), 4, "depth=2 QAOA → 4 free params");

        let (cost, _grad) = make_cost_grad(circuit, observable);
        let initial = vec![0.5, 0.5, 0.3, 0.3];
        let r = optimizers::run_cmaes(initial, cost, 100, 0.5, Some(11));
        // Triangle MaxCut Ising optimum (after offset folding) is around
        // -2 — depth-2 QAOA should land below -1.8 reliably.
        assert!(
            r.optimal_value < -1.8,
            "depth-2 QAOA did not converge: {}",
            r.optimal_value
        );
    }

    #[test]
    fn grad_method_choice_parse() {
        assert!(matches!(
            GradMethodChoice::parse("adjoint"),
            Some(GradMethodChoice::Adjoint)
        ));
        assert!(matches!(
            GradMethodChoice::parse("param-shift"),
            Some(GradMethodChoice::ParamShift)
        ));
        assert!(matches!(
            GradMethodChoice::parse("parameter-shift"),
            Some(GradMethodChoice::ParamShift)
        ));
        assert!(matches!(
            GradMethodChoice::parse("finite-diff"),
            Some(GradMethodChoice::FiniteDiff)
        ));
        assert!(matches!(
            GradMethodChoice::parse("fd"),
            Some(GradMethodChoice::FiniteDiff)
        ));
        assert!(GradMethodChoice::parse("nope").is_none());
    }
}
