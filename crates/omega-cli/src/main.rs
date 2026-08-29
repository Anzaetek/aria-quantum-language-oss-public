use std::env;
use std::fs;

use omega_backend_mps::MpsBackend;
use omega_backend_pauli::PauliBackend;
use omega_backend_pauliprop::PauliPropBackend;
use omega_backend_photonics::PhotonicsBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::CircuitType;
use omega_core::executor::{Backend, ExecConfig, MidCircuitMode, Observable, PauliOp};
use omega_core::gradient::{
    compute_functional_gradient, compute_gradient, parse_functional_spec, FunctionalGradMethod,
    GradMethod,
};
use omega_core::params::ParameterBinding;

mod qubo_mode;
mod serialize;
use serialize::Format;

/// The value following a flag, or a clean refusal.
///
/// Every flag in every argument loop was written as `i += 1; args[i]`, which
/// **panics** when the flag is last on the line. Measured:
/// `omega-run c.qasm --backend mps:8 --shots 20 --noise` printed
/// `index out of bounds: the len is 9 but the index is 9` and exited 101 — a
/// Rust panic with a backtrace note, for a plain typo. A CLI that panics on
/// bad input teaches the reader to distrust every other error it prints.
fn flag_value(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    match args.get(*i) {
        Some(v) => v.clone(),
        None => {
            eprintln!("{flag} needs a value, but nothing followed it.");
            eprintln!("Use --help for usage information.");
            std::process::exit(1);
        }
    }
}

/// A flag's value parsed to `T`, or a clean refusal naming the flag, what it
/// wanted, and what it got.
///
/// The `.expect("invalid shots number")` idiom this replaces also panicked, and
/// said neither which flag nor what was actually passed.
fn flag_parse<T: std::str::FromStr>(args: &[String], i: &mut usize, flag: &str, what: &str) -> T {
    let raw = flag_value(args, i, flag);
    match raw.parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("{flag} expects {what}, got {raw:?}.");
            std::process::exit(1);
        }
    }
}

/// A comma-separated list parsed to `Vec<T>`, refusing cleanly on any element.
fn flag_parse_list<T: std::str::FromStr>(
    args: &[String],
    i: &mut usize,
    flag: &str,
    what: &str,
) -> Vec<T> {
    let raw = flag_value(args, i, flag);
    raw.split(',')
        .map(|s| match s.trim().parse() {
            Ok(v) => v,
            Err(_) => {
                eprintln!(
                    "{flag} expects a comma-separated list of {what}; {:?} is not one.",
                    s.trim()
                );
                std::process::exit(1);
            }
        })
        .collect()
}

fn print_usage() {
    eprintln!("Usage: omega-run <circuit-file> [options]");
    eprintln!();
    eprintln!("Execution modes:");
    eprintln!("  (default)              Sample with 1024 shots");
    eprintln!("  --statevector          Exact statevector output");
    eprintln!("  --dump-state-bits F    Write the analytic state as hex float bit");
    eprintln!("                         patterns to F (needs an explicit --device;");
    eprintln!("                         refuses every silent fallback). For the");
    eprintln!("                         cross-device bit-equality protocol.");
    eprintln!("  --shots N              Sample N shots");
    eprintln!("  --expectation OBS      Compute <psi|O|psi> for observable OBS");
    eprintln!("  --gradient OBS         Compute d<O>/d(params) for observable OBS");
    eprintln!("  --gradient-of-fn JSON  Compute d E[f(x)]/d(params) for a Functional");
    eprintln!("                         JSON shapes:");
    eprintln!("                           {{\"qubo\":{{\"n\":N,\"Q\":[[i,j,c],...]}}}}");
    eprintln!("                           {{\"table\":[[\"bits\",value],...],\"num_qubits\":N}}");
    eprintln!("                         Pair with --method diagonal | score-fn.");
    eprintln!("  --score-fn-shots N     Shot count for --method score-fn (default 1024)");
    eprintln!();
    eprintln!("Observable format (for --expectation and --gradient):");
    eprintln!("  Z0                     Single Pauli Z on qubit 0");
    eprintln!("  Z0Z1                   Pauli ZZ on qubits 0,1");
    eprintln!("  0.5*Z0+0.3*X1         Weighted sum of Pauli terms");
    eprintln!("  X0X1+Z0Z1             Sum with implicit coefficient 1.0");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --backend NAME         statevector (default), mps, pauli, photonics,");
    eprintln!("                         pauliprop (Pauli propagation; --expectation only).");
    eprintln!("                         `auto` picks a backend that is EXACT for the");
    eprintln!("                         circuit and says which: Clifford-only -> pauli,");
    eprintln!("                         otherwise statevector if it fits. It never");
    eprintln!("                         silently substitutes an approximate backend — a");
    eprintln!("                         circuit that needs one is refused, with options.");
    eprintln!("                         mps takes a bond dimension: mps:<chi> pins it,");
    eprintln!("                         mps:auto[:ceiling] grows it as needed (ceiling");
    eprintln!("                         defaults to 1024, which is ~33 MB per site at full");
    eprintln!("                         chi — pin a smaller chi on a memory-tight host).");
    eprintln!("                         Every mps run reports a truncation certificate and");
    eprintln!("                         refuses rather than return a silently-truncated result.");
    eprintln!(
        "  --truncate C           pauliprop: drop coefficients below C. With any of these three,"
    );
    eprintln!(
        "  --max-weight W         the run reports `dropped_mass`, a BOUND on the error in <O>"
    );
    eprintln!("  --max-freq F           (not an estimate). Without them the exact engine is used.");
    eprintln!("  --max-terms N          pauliprop: raise the Pauli-term ceiling (default 2^21 =",);
    eprintln!("                         2097152). UNLIKE the three above this does NOT change the");
    eprintln!("                         answer -- it permits a larger EXACT sum instead of");
    eprintln!("                         discarding terms, so it costs memory, not accuracy.");
    eprintln!("                         Reach for it first when a run is refused at the ceiling.");
    eprintln!("                         A name matching a loaded plugin resolves after the");
    eprintln!("                         compiled-in backends.");
    eprintln!("  --backend-dir DIR      Load backend plugins (.so/.dylib/.dll) from DIR");
    eprintln!("                         (repeatable; also OMEGA_BACKEND_DIR). Plugin loading");
    eprintln!("                         is opt-in; plugins run the default sample mode.");
    eprintln!(
        "  --qasm-dialect D       QASM2 reader for bare rxx/ryy/rzz (none are in qelib1):\n\
         \x20                        legacy  (default) accepts rxx/rzz, refuses ryy — qiskit from_qasm_str\n\
         \x20                        strict  refuses all three — qiskit qasm2.loads\n\
         \x20                        lenient accepts all three (no qiskit reader does)\n\
         \x20                        A file carrying its own `gate` definition is read in all three."
    );
    eprintln!("  --list-backends        List compiled-in backends and loaded plugins, then exit");
    eprintln!("  --bridge NAME          Route execution through an external simulator");
    eprintln!("                         (qiskit, perceval). Only the default sample mode");
    eprintln!("                         is supported; statevector / expectation / gradient");
    eprintln!("                         modes require an in-process backend. Counts are");
    eprintln!("                         displayed MSB-first to match the in-process path.");
    eprintln!("  --device NAME          cpu (default), metal, cuda, opencl — honoured for the");
    eprintln!("                         statevector backend's execute and --gradient");
    eprintln!("                         paths; falls back to cpu when the requested");
    eprintln!("                         device isn't compiled in or available");
    eprintln!("  --multi-control MODE   decompose (default) | exact — how CCX/CSwap are");
    eprintln!("                         realised on the CUDA and Metal statevector");
    eprintln!("                         backends. `decompose` is a 15-gate Nielsen-Chuang");
    eprintln!("                         chain, the same sequence on both so the two GPUs");
    eprintln!("                         agree bit-for-bit. `exact` applies the permutation");
    eprintln!("                         directly: faster and it removes 14 gates of f32");
    eprintln!("                         rounding, but it CHANGES THE NUMBERS, so it is");
    eprintln!("                         opt-in. See GATE-EXACTNESS.md. Every other path");
    eprintln!("                         (CPU included) applies the permutation directly");
    eprintln!("                         and says so when the mode is not honoured.");
    eprintln!("  --seed N               Random seed for sampling");
    eprintln!("  --params V0,V1,...     Bind free parameters (sorted by symbol ID)");
    eprintln!(
        "  --method NAME          Gradient method: adjoint (default), param-shift, finite-diff,\n\
         \x20                       diagonal (for --gradient-of-fn, default), score-fn"
    );
    eprintln!("  --input N0,N1,...      Input Fock state (photonics only)");
    eprintln!("  --format FMT           Output format: text (default), json, jsonl");
    eprintln!("  --version, -V          Print the version and the git revision this");
    eprintln!("                         binary was built from ('-dirty' if the tree");
    eprintln!("                         had uncommitted changes). Also emitted under");
    eprintln!("                         \"build\" in every --format json document.");
    eprintln!("  --noise JSON           Per-gate + readout noise model. Sampled on --backend");
    eprintln!(
        "                         statevector or mps; applied to --expectation on pauliprop."
    );
    eprintln!("                         Uniform (idealized): '{{\"depolarizing\":0.001,\"amplitude_damping\":5e-4,\"readout_flip\":0.02}}'");
    eprintln!(
        "                         Per-qubit (calibrated): '{{\"amplitude_damping\":[0.004,0.006],"
    );
    eprintln!("                           \"depolarizing\":{{\"1q\":0.001,\"2q\":0.012}},\"readout\":[{{\"p10\":0.02,\"p01\":0.03}}]}}'");
    eprintln!();
    eprintln!("QUBO solve mode (--qubo):");
    eprintln!("  --qubo FILE            QUBO problem JSON ({{\"n\":N,\"Q\":[[i,j,c],...]}})");
    eprintln!("  --qaoa-depth P         QAOA rounds (default: 2)");
    eprintln!("  --optimizer NAME       cma-es (default) or gradient");
    eprintln!("  --max-iters N          Optimizer budget (default: 50)");
    eprintln!("  --top-k K              Ranked samples emitted (default: 20)");
    eprintln!();
    eprintln!("Shor factoring demo (--shor):");
    eprintln!("  --shor                 Run Shor's algorithm");
    eprintln!("  --N N                  Composite to factor (≤ 63)");
    eprintln!("  --max-attempts M       Cap random-a trials (default: 16)");
    eprintln!("  --seed S               Deterministic RNG seed");
    eprintln!();
    eprintln!("Supported formats: QASM 2.0 (.qasm), OPTICQASM (.opticqasm),");
    eprintln!("                   Qiskit QPY (.qpy or `QISKIT` magic-byte header — requires");
    eprintln!("                   --features bridge-qiskit and a Qiskit venv on the host;");
    eprintln!("                   decoded to QASM 2.0 in-memory before the in-process pipeline)");
}

/// Fall back to the Qiskit bridge subprocess when the pure-Rust QPY
/// reader can't handle a feature in the blob. Errors here bubble up
/// as a fatal — without subprocess _and_ without the pure-Rust
/// reader we have no way to read the file.
fn qpy_via_subprocess(raw_bytes: &[u8]) -> String {
    match omega_bridges::qpy_to_qasm2(raw_bytes) {
        Ok(s) => {
            eprintln!(
                "QPY: decoded {} bytes via qiskit bridge → {} chars of QASM2",
                raw_bytes.len(),
                s.len()
            );
            s
        }
        Err(e) => {
            eprintln!("QPY decode failed: {e}");
            std::process::exit(1);
        }
    }
}

fn parse_format(args: &[String]) -> Format {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--format" && i + 1 < args.len() {
            match Format::parse(&args[i + 1]) {
                Some(f) => return f,
                None => {
                    eprintln!(
                        "Unknown format: {}. Options: text, json, jsonl",
                        args[i + 1]
                    );
                    std::process::exit(1);
                }
            }
        }
        i += 1;
    }
    Format::Text
}

/// Collect the directories to load backend plugins from: explicit
/// `--backend-dir` flags, then `OMEGA_BACKEND_DIR` (`:`-separated). Plugin
/// loading is **opt-in** — there is no implicit `~/.omega/backends` probe, so a
/// mistyped `--backend NAME` never dlopens code the user didn't ask for.
/// Missing directories are harmless — `load_dir` returns `Ok(0)` for them.
/// Build an `MpsBackend` from a `--backend mps…` selector.
///
/// The grammar lives in `omega_backend_mps::select` — one parser, shared with
/// the other front end, rather than a second copy that drifts. Before this,
/// this binary matched the literal string `"mps"` and hardcoded chi=64 at six
/// sites, so a user whose circuit needed a larger bond had no way to ask for
/// one; after the truncation gate landed they got a refusal with no route to a
/// correct answer.
fn mps_backend_from(name: &str, device: Option<&str>) -> Option<MpsBackend> {
    let backend = match omega_backend_mps::select::parse_mps(name) {
        Ok(Some(omega_backend_mps::select::MpsSelect::Fixed { chi })) => MpsBackend::new(chi),
        Ok(Some(omega_backend_mps::select::MpsSelect::Auto { max_chi })) => {
            MpsBackend::new(max_chi).with_adaptive(omega_backend_mps::select::AUTO_EPS)
        }
        // A malformed selector never reaches here: it is refused earlier, in
        // `cmd_exec`, with the parser's own message. (An earlier version of this
        // comment claimed the caller's unknown-backend path showed that message.
        // It did not — the message was discarded and each mode invented its own,
        // including two that were simply false.)
        _ => return None,
    };
    Some(apply_mps_device_hooks(backend, device))
}

/// Wire the MPS GPU hooks `aria-runtime` has wired since they landed —
/// `omega-run` never did, so `--backend mps --device metal` silently ran the
/// pure-CPU path and every MPS benchmark through this CLI measured one core
/// (found via an external 31-qubit measurement: zero thread scaling, 17x
/// behind Aer at bond 256).
///
/// Unlike `aria-runtime`, which applies hooks unconditionally when compiled
/// in, this honours the CLI's EXPLICIT device semantics: the Metal two-site
/// contraction is f32 above its bond threshold, so it changes the numbers,
/// and a `--device cpu` run must stay exact-f64 — the same reasoning that
/// makes `--dump-state-bits` refuse silent device changes. CUDA takes the
/// native-f64 `gesvdj` SVD hook instead (precision unchanged), and wins the
/// arm on a dual-vendor build for that reason.
fn apply_mps_device_hooks(backend: MpsBackend, device: Option<&str>) -> MpsBackend {
    let requested = device.and_then(|s| omega_core::device::DeviceKind::parse(s).ok());
    let resolved = omega_core::device::DeviceKind::resolve(requested);
    #[cfg(feature = "cuda")]
    let backend = if matches!(resolved, omega_core::device::DeviceKind::Cuda) {
        // stderr directly: this is a free function and `info` is a per-mode
        // closure; a device notice belongs on the human channel in all modes.
        eprintln!("Device: cuda (mps bond-compression SVD)");
        backend.with_svd_fn(omega_backend_mps_cuda::cuda_svd_flat)
    } else {
        backend
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let backend = if matches!(resolved, omega_core::device::DeviceKind::Metal) {
        eprintln!("Device: metal (mps two-site contraction, f32 above the bond threshold)");
        backend.with_contract_fn(omega_backend_mps_metal::metal_contract_2q)
    } else {
        backend
    };
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let _ = resolved;
    backend
}

/// The notice to print when `--multi-control` was passed but the run cannot
/// honour it, or `None` when there is nothing to say.
///
/// # Why this exists
///
/// `with_multi_control` is implemented by exactly two backends —
/// `CudaStatevectorBackend` (`omega-backend-statevector-cuda/src/lib.rs`) and
/// `MetalStatevectorBackend` (`omega-backend-statevector-metal/src/lib.rs`).
/// Every other path, CPU included, applies `CCX`/`CSwap` as a **direct subspace
/// permutation** (`omega-backend-statevector/src/sim.rs` `apply_ccx` /
/// `apply_cswap`). So the flag was parsed, validated, and then dropped on the
/// floor by every other run, saying nothing.
///
/// Metal was CUDA-only until the octet-permutation kernel landed
/// (`shaders/apply_octet_swap.metal`); this doc and the message text below were
/// both written for that era and are updated here in the same change, because
/// "the switch is CUDA-statevector only" becomes false the moment the Metal arm
/// exists and nothing else would catch it.
///
/// `GATE-EXACTNESS.md` refused to put `multi_control` on `QuantumExecuteReq`
/// precisely because it would be "inert, advertising a capability no code path
/// can honour". The CLI had shipped exactly that.
///
/// # Why a notice and not a refusal
///
/// The two modes are NOT symmetric here, and the asymmetry is the whole design:
///
/// * `exact` off CUDA — the direct permutation IS what `exact` asks for. The
///   caller gets the numbers they wanted by another route. Refusing would break
///   scripts that pass the flag uniformly across devices for no numeric reason.
/// * `decompose` off CUDA — the caller asked for the 15-gate Nielsen-Chuang
///   chain and got the permutation instead. This is the case that can mislead:
///   it is the DEFAULT value, so it is reached by anyone comparing a CPU run
///   against a CUDA or Metal decomposed reference, and the numbers differ.
///
/// So the message says which of those happened rather than emitting one line for
/// both. `--device` next to it takes the same shape — it falls back and says so
/// (`"cuda fallback to cpu"`), rather than refusing.
///
/// `honoured` is the caller's `use_cuda_sv || use_metal_sv` (or the gradient
/// pair), each of which is already false both when its feature is absent and
/// when the resolved device or backend is something else. That makes this a
/// RUNTIME check: `--device cpu` or `--backend mps` on a CUDA build discard the
/// mode too, and a `cfg` would catch none of it.
fn multi_control_notice(
    requested: bool,
    mode: omega_core::executor::MultiControlMode,
    honoured: bool,
) -> Option<String> {
    use omega_core::executor::MultiControlMode as M;
    if !requested || honoured {
        return None;
    }
    Some(match mode {
        M::Exact => "--multi-control exact: not applicable to this run (the switch \
                     applies to the CUDA and Metal statevector backends only). \
                     CCX/CSwap are applied as a direct permutation here, which is \
                     what 'exact' means, so the numbers are unaffected."
            .to_string(),
        M::Decompose => "--multi-control decompose: NOT honoured by this run (the \
                         switch applies to the CUDA and Metal statevector backends \
                         only). CCX/CSwap are applied as a direct permutation, not \
                         the 15-gate chain, so these numbers will NOT match a CUDA \
                         or Metal 'decompose' run."
            .to_string(),
    })
}

fn build_plugin_registry(explicit_dirs: &[String]) -> omega_core::plugin::BackendRegistry {
    let mut registry = omega_core::plugin::BackendRegistry::new();
    let mut dirs: Vec<std::path::PathBuf> =
        explicit_dirs.iter().map(std::path::PathBuf::from).collect();
    if let Ok(env_dirs) = env::var("OMEGA_BACKEND_DIR") {
        for d in env_dirs.split(':').filter(|s| !s.is_empty()) {
            dirs.push(std::path::PathBuf::from(d));
        }
    }
    for dir in dirs {
        if let Err(e) = registry.load_dir(&dir) {
            eprintln!("Warning: backend-dir {}: {}", dir.display(), e);
        }
    }
    registry
}

/// Scan raw args for `--backend-dir <dir>` pairs (used by the early
/// `--list-backends` path, which runs before the main arg loop).
fn scan_backend_dirs(args: &[String]) -> Vec<String> {
    let mut dirs = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--backend-dir" && i + 1 < args.len() {
            dirs.push(args[i + 1].clone());
            i += 2;
        } else {
            i += 1;
        }
    }
    dirs
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Before the usage check: `--version` is a valid whole invocation, and a
    // consumer scripting it should not have to also pass a circuit file.
    if args.len() >= 2 && (args[1] == "--version" || args[1] == "-V") {
        println!(
            "omega-run {} ({})",
            env!("CARGO_PKG_VERSION"),
            serialize::build_rev()
        );
        return;
    }

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        print_usage();
        if args.len() >= 2 {
            return;
        }
        std::process::exit(1);
    }

    let format = parse_format(&args);

    // --qubo / --shor have their own arg parsers with catch-all arms that would
    // silently swallow --noise. These modes don't model gate noise, so reject it
    // loudly rather than drop it (the core principle of the noise wiring).
    if args.iter().any(|a| a == "--noise")
        && (args.iter().any(|a| a == "--qubo") || args.iter().any(|a| a == "--shor"))
    {
        eprintln!("--noise is not supported in --qubo / --shor mode. Drop --noise.");
        std::process::exit(1);
    }

    // Check for --qubo mode (doesn't take a circuit file as positional arg).
    if args.iter().any(|a| a == "--qubo") {
        let opts = parse_qubo_args(&args);
        qubo_mode::run_qubo(opts, format);
        return;
    }

    // Check for --shor mode (doesn't need a circuit file).
    if args.iter().any(|a| a == "--shor") {
        run_shor(&args, format);
        return;
    }

    // --list-backends: enumerate compiled-in backends and loaded plugins, then
    // exit. Handled here (before the positional circuit file is required) so it
    // works without a circuit argument.
    if args.iter().any(|a| a == "--list-backends") {
        let registry = build_plugin_registry(&scan_backend_dirs(&args));
        for name in [
            "statevector",
            "mps",
            "mps:<chi>",
            "mps:auto",
            "mps:auto:<ceiling>",
            "pauli",
            "pauliprop",
            "photonics",
        ] {
            println!("builtin  {name}");
        }
        for name in registry.list() {
            println!("plugin   {name}");
        }
        return;
    }

    let file_path = &args[1];
    let mut shots: Option<u32> = Some(1024);
    let mut seed: Option<u64> = None;
    let mut input_state: Option<Vec<u32>> = None;
    let mut backend_name: Option<String> = None;
    // Which QASM2 reader to imitate for `rxx`/`ryy`/`rzz`, none of which is in
    // qelib1. Default matches Qiskit's legacy loader; see `Qasm2Dialect`.
    let mut qasm_dialect = omega_parser::lower::Qasm2Dialect::default();
    let mut device_name: Option<String> = None;
    let mut observable_str: Option<String> = None;
    let mut gradient_str: Option<String> = None;
    let mut gradient_fn_str: Option<String> = None;
    let mut gradient_fn_shots: Option<u32> = None;
    let mut param_values: Option<Vec<f64>> = None;
    // How CCX/CSwap are realised on backends that can choose. Default matches
    // the previous behaviour exactly; see `MultiControlMode`.
    let mut multi_control = omega_core::executor::MultiControlMode::default();
    // Whether `--multi-control` was PASSED, as distinct from what it resolved to.
    // Only an explicit request can be silently unhonoured, so only an explicit
    // request earns a notice — see `note_multi_control_ignored`.
    let mut multi_control_requested = false;
    let mut grad_method_name: Option<String> = None;
    let mut noise_json: Option<String> = None;
    let mut pp_truncate: Option<f64> = None;
    let mut pp_max_weight: Option<usize> = None;
    let mut pp_max_terms: Option<usize> = None;
    let mut pp_max_dropped: Option<f64> = None;
    let mut pp_max_freq: Option<u32> = None;
    let mut bridge_name: Option<String> = None;
    let mut backend_dirs: Vec<String> = Vec::new();
    // Write the final statevector as HEX FLOAT BIT PATTERNS for the
    // cross-device bit-equality protocol (PLAN-BITEQ-CUDA-METAL.md). A
    // measurement artifact, so its guards are hard errors, never fallbacks.
    let mut dump_state_bits: Option<String> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--shots" => {
                shots = Some(flag_parse(
                    &args,
                    &mut i,
                    "--shots",
                    "a whole number of shots",
                ));
            }
            "--statevector" | "--exact" => {
                shots = None;
            }
            "--seed" => {
                seed = Some(flag_parse(&args, &mut i, "--seed", "an integer seed"));
            }
            "--backend" => {
                backend_name = Some(flag_value(&args, &mut i, "--backend"));
            }
            "--backend-dir" => {
                backend_dirs.push(flag_value(&args, &mut i, "--backend-dir"));
            }
            "--qasm-dialect" => {
                let v = flag_value(&args, &mut i, "--qasm-dialect");
                qasm_dialect = match v.as_str() {
                    "strict" => omega_parser::lower::Qasm2Dialect::Strict,
                    "legacy" => omega_parser::lower::Qasm2Dialect::Legacy,
                    "lenient" => omega_parser::lower::Qasm2Dialect::Lenient,
                    other => {
                        eprintln!(
                            "unknown --qasm-dialect '{other}': expected `strict` \
                             (qiskit qasm2.loads), `legacy` (qiskit \
                             from_qasm_str, the default), or `lenient` (also \
                             reads a bare `ryy`, which no qiskit reader does)"
                        );
                        std::process::exit(1);
                    }
                };
            }
            "--bridge" => {
                bridge_name = Some(flag_value(&args, &mut i, "--bridge"));
            }
            "--device" => {
                device_name = Some(flag_value(&args, &mut i, "--device"));
            }
            "--dump-state-bits" => {
                dump_state_bits = Some(flag_value(&args, &mut i, "--dump-state-bits"));
            }
            "--multi-control" => {
                let v = flag_value(&args, &mut i, "--multi-control");
                multi_control = match omega_core::executor::MultiControlMode::parse(&v) {
                    Some(m) => m,
                    None => {
                        eprintln!(
                            "--multi-control: unknown mode {v:?} (expected \
                             'decompose' or 'exact')"
                        );
                        std::process::exit(1);
                    }
                };
                multi_control_requested = true;
            }
            "--input" => {
                input_state = Some(flag_parse_list(&args, &mut i, "--input", "photon numbers"));
            }
            "--expectation" => {
                observable_str = Some(flag_value(&args, &mut i, "--expectation"));
            }
            "--gradient" => {
                gradient_str = Some(flag_value(&args, &mut i, "--gradient"));
            }
            "--gradient-of-fn" => {
                gradient_fn_str = Some(flag_value(&args, &mut i, "--gradient-of-fn"));
            }
            "--score-fn-shots" => {
                gradient_fn_shots = Some(flag_parse(
                    &args,
                    &mut i,
                    "--score-fn-shots",
                    "a whole number of shots",
                ));
            }
            "--params" => {
                param_values = Some(flag_parse_list(
                    &args,
                    &mut i,
                    "--params",
                    "parameter values",
                ));
            }
            "--method" => {
                grad_method_name = Some(flag_value(&args, &mut i, "--method"));
            }
            "--format" => {
                i += 1; // already consumed by parse_format above
            }
            "--noise" => {
                noise_json = Some(flag_value(&args, &mut i, "--noise"));
            }
            // Pauli-propagation truncation, the three axes PauliPropagation.jl
            // uses. `aria` has had these; `omega-run` did not, so the only
            // engine reachable here was the exact one — and with it no
            // dropped-mass budget to report.
            "--truncate" => {
                pp_truncate = Some(flag_parse(
                    &args,
                    &mut i,
                    "--truncate",
                    "a coefficient floor, e.g. 1e-3",
                ));
            }
            "--max-weight" => {
                pp_max_weight = Some(flag_parse(
                    &args,
                    &mut i,
                    "--max-weight",
                    "a maximum Pauli weight",
                ));
            }
            "--max-freq" => {
                pp_max_freq = Some(flag_parse(
                    &args,
                    &mut i,
                    "--max-freq",
                    "a maximum split frequency",
                ));
            }
            // The ceiling has ALWAYS been configurable — `PauliPropBackend`
            // carries `max_terms: Option<usize>` and defaults it to
            // `DEFAULT_MAX_TERMS` (2^21). Only the flag was missing, so the
            // refusal read as a hardcoded wall: a bug report measured being
            // turned away at ~80 MB of terms on a 314 GB host, three orders of
            // magnitude below the machine, with no documented way up.
            //
            // Unlike the other three knobs this one does NOT change the answer.
            // Raising it lets an EXACT run complete; the others buy completion
            // by discarding terms and moving the value. That is why it is worth
            // reaching for first, and why it carries no `dropped_mass` cost.
            // The bound has always been computed; the CLI printed it to
            // stderr as prose and gated nothing on it. Now it gates, and this
            // is the override — for a caller deliberately sweeping cutoffs, who
            // wants the loose rows the gate would otherwise refuse.
            "--max-dropped-mass" => {
                pp_max_dropped = Some(flag_parse(
                    &args,
                    &mut i,
                    "--max-dropped-mass",
                    "an L1 mass ceiling, e.g. 1.0, or `inf` to disable the gate",
                ));
            }
            "--max-terms" => {
                pp_max_terms = Some(flag_parse(
                    &args,
                    &mut i,
                    "--max-terms",
                    "a maximum Pauli-term count, e.g. 16777216",
                ));
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                eprintln!("Use --help for usage information.");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // info() routes to stdout in text mode, stderr in machine modes.
    let info = |msg: String| {
        if format.is_machine() {
            eprintln!("{}", msg);
        } else {
            println!("{}", msg);
        }
    };

    // Read and parse circuit. `.qpy` (Qiskit's binary serialisation)
    // is auto-detected by extension *or* by the magic bytes. We
    // prefer the pure-Rust reader (`omega_bridges::qpy::read_qpy_circuit_ir`)
    // since it doesn't spawn a Python subprocess; on any
    // `QpyError::Unsupported{Gate,_}` we fall back to the qiskit
    // bridge subprocess that round-trips through `qpy.load + qasm2.dumps`.
    // The fallback path additionally requires `--features bridge-qiskit`
    // and a Qiskit venv on the host.
    let raw_bytes = fs::read(file_path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", file_path, e);
        std::process::exit(1);
    });
    let is_qpy_input =
        omega_bridges::is_qpy(&raw_bytes) || file_path.to_lowercase().ends_with(".qpy");

    let mut prebuilt_circuit: Option<omega_core::circuit::CircuitIR> = None;
    let source = if is_qpy_input {
        // First attempt: pure-Rust reader. Skips the Python hop
        // entirely when it succeeds. When --bridge is set the bridge
        // wants the QASM2 string itself, so we still fall through to
        // the subprocess path in that case.
        if bridge_name.is_none() {
            match omega_bridges::qpy::read_qpy_circuit_ir(&raw_bytes) {
                Ok(ir) => {
                    eprintln!(
                        "QPY: decoded {} bytes via pure-Rust reader (no Qiskit subprocess)",
                        raw_bytes.len()
                    );
                    prebuilt_circuit = Some(ir);
                    String::new()
                }
                Err(e) => {
                    eprintln!(
                        "QPY: pure-Rust reader hit {} — falling back to qiskit bridge",
                        e
                    );
                    qpy_via_subprocess(&raw_bytes)
                }
            }
        } else {
            qpy_via_subprocess(&raw_bytes)
        }
    } else {
        String::from_utf8(raw_bytes).unwrap_or_else(|e| {
            eprintln!("Error: file {} is not valid UTF-8: {e}", file_path);
            std::process::exit(1);
        })
    };

    // When --bridge is set, the QASM goes through an external simulator
    // verbatim. Local parse failures (e.g. a construct omega-parser
    // doesn't yet handle but Qiskit does) shouldn't block the run; we
    // emit a notice and proceed with placeholder metadata. The bridge
    // counts come back as bit-strings whose length tells us num_qubits.
    let circuit = if let Some(ir) = prebuilt_circuit {
        ir
    } else {
        match omega_parser::lower::lower_to_ir_with_dialect(&source, qasm_dialect) {
            Ok(c) => c,
            Err(e) => {
                if bridge_name.is_some() {
                    eprintln!(
                        "Note: omega-parser failed ({e}); forwarding QASM to the bridge anyway."
                    );
                    omega_core::circuit::CircuitIR::new(0, CircuitType::GateBased)
                } else {
                    eprintln!("Parse error: {}", e);
                    std::process::exit(1);
                }
            }
        }
    };

    // `--dump-state-bits` is defined for the PLAIN run path only. The bridge,
    // expectation and gradient modes return early, far above the dump guards —
    // so without this check the flag was accepted, the mode ran, and NO file
    // was written, leaving any stale artifact from a previous run in place for
    // diff.py to compare as if fresh (found by review; the exact
    // silent-artifact class the flag's hard errors exist to refuse).
    if dump_state_bits.is_some()
        && (bridge_name.is_some()
            || observable_str.is_some()
            || gradient_str.is_some()
            || gradient_fn_str.is_some())
    {
        eprintln!(
            "--dump-state-bits dumps the final statevector of a plain run; \
             --bridge/--expectation/--gradient modes produce no state to dump. \
             Drop the mode flag, or drop --dump-state-bits."
        );
        std::process::exit(1);
    }

    let num_modes = circuit.num_qubits;
    // Width to RENDER a counts key at. Not always `num_qubits`: in collapse mode
    // the key is packed from the CLASSICAL register, so a 1024-qubit circuit
    // measuring two qubits produces a 2-bit outcome — which was being printed
    // padded to 1024 characters. A key that is correct internally and displayed
    // at the wrong width is still wrong to the reader, and no type error catches
    // it.
    let counts_display_width = omega_core::executor::counts_outcome_width(
        &circuit,
        omega_core::executor::counts_keyed_on_creg(
            &circuit,
            omega_core::executor::needs_collapse(&circuit),
        ),
    ) as u32;
    info(format!(
        "Circuit: {} {}, {} ops, type: {:?}",
        num_modes,
        if circuit.circuit_type == CircuitType::Photonic {
            "modes"
        } else {
            "qubits"
        },
        circuit.ops.len(),
        circuit.circuit_type
    ));

    // Bind parameters
    let mut params = ParameterBinding::new();
    let mut symbol_ids: Vec<u32> = circuit.symbols.keys().copied().collect();
    symbol_ids.sort();

    if let Some(ref vals) = param_values {
        for (idx, &sym_id) in symbol_ids.iter().enumerate() {
            let val = if idx < vals.len() { vals[idx] } else { 0.0 };
            params.bind(sym_id, val);
        }
        if !symbol_ids.is_empty() {
            info(format!(
                "Parameters bound: {:?}",
                symbol_ids
                    .iter()
                    .zip(param_values.as_ref().unwrap().iter())
                    .map(|(id, v)| format!(
                        "{}={:.4}",
                        circuit.symbols.get(id).unwrap_or(&format!("sym_{}", id)),
                        v
                    ))
                    .collect::<Vec<_>>()
            ));
        }
    } else {
        for &id in &symbol_ids {
            params.bind(id, 0.0);
        }
        if !circuit.symbols.is_empty() {
            eprintln!(
                "Warning: {} unbound parameters, using 0.0 for all (use --params to set)",
                circuit.symbols.len()
            );
        }
    }

    // Select backend
    let default_backend = match circuit.circuit_type {
        CircuitType::GateBased => "statevector",
        CircuitType::Photonic => "photonics",
    };
    let chosen = backend_name.as_deref().unwrap_or(default_backend);

    // `--backend auto`: pick a backend that is EXACT for this circuit, and say
    // which. It will never silently substitute an approximate one.
    //
    // That restriction is the whole design. The obvious `auto` also falls back
    // to MPS at a fixed bond dimension for wide circuits, which is an
    // approximation chosen by a heuristic the user did not see — and MPS
    // truncation is exactly the thing this workspace spends a certificate on
    // reporting. An `auto` that can quietly hand back a truncated answer is
    // worse than no `auto`, so a circuit that is neither Clifford nor small
    // enough for a dense state is REFUSED with the options spelled out rather
    // than approximated.
    let chosen = if matches!(chosen, "auto") {
        if circuit.circuit_type == CircuitType::Photonic {
            info("auto: photonic circuit -> photonics".to_string());
            "photonics"
        } else if omega_core::circuit::is_clifford_only(&circuit) {
            // Exact and polynomial. Worth saying out loud, because the same
            // circuit on the statevector backend is exponential and a user
            // watching the clock deserves to know why it got fast.
            info(format!(
                "auto: Clifford-only circuit -> pauli (exact, polynomial in the {} qubits)",
                circuit.num_qubits
            ));
            "pauli"
        } else if omega_backend_statevector::capacity::check(circuit.num_qubits, shots.is_some())
            .is_ok()
        {
            info(format!(
                "auto: non-Clifford, {} qubits fits in memory -> statevector (exact)",
                circuit.num_qubits
            ));
            "statevector"
        } else {
            eprintln!("auto: this circuit is not Clifford, so the exact stabilizer");
            eprintln!(
                "      backend cannot run it, and a dense statevector of {} qubits",
                circuit.num_qubits
            );
            eprintln!("      does not fit on this host.");
            eprintln!();
            eprintln!("`auto` will not silently pick an approximate backend. An MPS run");
            eprintln!("at some guessed bond dimension may be accurate or may not, and");
            eprintln!("making that choice for you is what this flag refuses to do.");
            eprintln!("Pick one explicitly:");
            eprintln!();
            eprintln!("  --backend mps:<chi>   approximate; reports the discarded weight");
            eprintln!("                        it cost you, and refuses past the ceiling");
            eprintln!("  --backend mps:auto    grows the bond only as far as needed");
            eprintln!("  --backend pauliprop   exact for expectation values on shallow");
            eprintln!("                        non-Clifford circuits");
            std::process::exit(1);
        }
    } else {
        chosen
    };

    // A MALFORMED mps selector is reported with the parser's own message, once,
    // before any mode dispatches.
    //
    // Previously `mps_backend_from` mapped `Err` to `None`, so the guard arms
    // did not match and the name fell through to whatever each mode says about
    // an unknown backend. Measured, all for `--backend mps:0`:
    //
    //   --shots         -> "Unknown backend: mps:0"                    (vague)
    //   --gradient      -> "Gradient not supported for backend: mps:0" (FALSE)
    //   --expectation   -> "Expectation not supported for backend"     (FALSE)
    //   --noise         -> "supported on statevector or mps (got 'mps:0')"
    //                      — which tells the user to use the thing they typed.
    //
    // "MPS bond dimension must be ≥ 1" was never shown anywhere.
    if chosen.starts_with("mps") {
        if let Err(e) = omega_backend_mps::select::parse_mps(chosen) {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }

    // --- Mode: --bridge dispatch through an external simulator ---
    if let Some(ref bridge_str) = bridge_name {
        // Expectation over a bridge. The transport already existed —
        // `omega_bridges::expectation_qasm2`, used by the cross-check harness
        // — but it was reachable only from Rust, so the CLI turned every
        // `--bridge --expectation` away even for backends that implement it.
        // That is the whole of CR §8's first sub-ask: routing, not protocol.
        if let Some(obs_str) = observable_str.as_deref() {
            if gradient_str.is_some() || gradient_fn_str.is_some() {
                eprintln!("--bridge cannot combine --expectation with a gradient mode.");
                std::process::exit(1);
            }
            let backend = omega_bridges::Backend::parse(bridge_str).unwrap_or_else(|e| {
                eprintln!("Invalid --bridge value: {e}");
                std::process::exit(1);
            });
            // A bridge hands the circuit to an EXTERNAL runner, which has its own
            // gate realisations and no knowledge of this switch. `honoured` is
            // therefore false here for a stronger reason than on the local paths:
            // not "this build cannot", but "this process does not decide".
            if let Some(m) = multi_control_notice(multi_control_requested, multi_control, false) {
                info(m);
            }
            // The wire wants an LSB-first Pauli STRING of full width; the CLI
            // parses a sparse `Z0 X2` form. Widen here rather than teach the
            // wire a second encoding.
            let observable = parse_observable(obs_str);
            let n = circuit.num_qubits as usize;
            let wire: omega_bridges::WireObservable = observable
                .terms
                .iter()
                .map(|(coeff, paulis)| {
                    let mut s = vec![b'I'; n];
                    for (q, p) in paulis {
                        let c = match p {
                            omega_core::executor::PauliOp::I => b'I',
                            omega_core::executor::PauliOp::X => b'X',
                            omega_core::executor::PauliOp::Y => b'Y',
                            omega_core::executor::PauliOp::Z => b'Z',
                        };
                        s[*q as usize] = c;
                    }
                    (String::from_utf8(s).expect("ascii"), *coeff)
                })
                .collect();
            info(format!("Bridge: {backend:?} (expectation)"));
            let values = omega_bridges::expectation_qasm2(backend, &source, &[wire])
                .unwrap_or_else(|e| {
                    eprintln!("Bridge error: {e}");
                    std::process::exit(1);
                });
            let value = values.first().copied().unwrap_or_else(|| {
                eprintln!("Bridge returned no value for the observable");
                std::process::exit(1);
            });
            match format {
                Format::Json | Format::Jsonl => {
                    println!("{}", serialize::expectation_to_json(obs_str, value));
                }
                Format::Text => {
                    println!("\n<O> = {value:.10}");
                }
            }
            return;
        }
        if gradient_str.is_some() || gradient_fn_str.is_some() || shots.is_none() {
            eprintln!(
                "--bridge supports sampling (--shots N) and --expectation. \
                 Statevector and gradient modes require an in-process backend."
            );
            std::process::exit(1);
        }
        let n_shots = shots.unwrap();
        let backend = omega_bridges::Backend::parse(bridge_str).unwrap_or_else(|e| {
            eprintln!("Invalid --bridge value: {}", e);
            std::process::exit(1);
        });
        let noise_value: Option<serde_json::Value> = noise_json.as_deref().map(|s| {
            serde_json::from_str(s).unwrap_or_else(|e| {
                eprintln!("Invalid --noise JSON: {}", e);
                std::process::exit(1);
            })
        });
        info(format!("Bridge: {:?}", backend));
        let counts_str = omega_bridges::run_qasm2(backend, &source, n_shots, noise_value.as_ref())
            .unwrap_or_else(|e| {
                eprintln!("Bridge error: {}", e);
                std::process::exit(1);
            });

        // Bridges return LSB-first bit-strings. Convert to the u64-keyed
        // counts the rest of omega-run uses, where bit 0 of the integer
        // is qubit 0; from there `serialize::format_bits` re-emits MSB-
        // first so the CLI output matches the in-process path.
        // The bridge returns Qiskit-style keys, which are already the width of
        // the CLASSICAL register — so their own length is the authority, and
        // `num_modes` (the qubit count) is not. This preferred `num_modes`, so
        // a 20-qubit circuit measuring 2 qubits printed 20-character keys: the
        // same defect fixed at the CLI's native path and both server sites,
        // still open on this one.
        //
        // `num_modes` remains the fallback for an empty histogram, where there
        // is no key to measure.
        let inferred_qubits = counts_str.keys().map(|k| k.len() as u32).max().unwrap_or(0);
        let display_qubits = if inferred_qubits > 0 {
            inferred_qubits
        } else {
            num_modes
        };
        let mut counts_u64: std::collections::HashMap<omega_core::outcome::Outcome, u32> =
            std::collections::HashMap::new();
        for (key, n) in &counts_str {
            // The bridge's own key length IS the width — it is already the
            // classical register's. Building an `Outcome` from it directly
            // removes the u64 bottleneck this loop used to funnel through, and
            // with it the 64-qubit ceiling on bridged results.
            //
            // The bridge emits LSB-first (character i is qubit i), which is why
            // this reverses rather than calling `from_bitstring`, whose input is
            // MSB-first like everything the CLI prints.
            let bits: Vec<u8> = key
                .chars()
                .map(|c| if c == '1' { 1u8 } else { 0 })
                .collect();
            *counts_u64
                .entry(omega_core::outcome::Outcome::from_bits(&bits))
                .or_insert(0) += *n;
        }
        let result = omega_core::executor::ExecResult::Counts(counts_u64);
        match format {
            Format::Json => {
                let v = serialize::exec_result_to_json(
                    &result,
                    display_qubits,
                    Some(n_shots),
                    &CircuitType::GateBased,
                );
                println!("{}", v);
            }
            Format::Jsonl => {
                if let omega_core::executor::ExecResult::Counts(counts) = &result {
                    serialize::emit_jsonl_counts(
                        counts,
                        display_qubits,
                        &CircuitType::GateBased,
                        "bridge",
                    );
                }
            }
            Format::Text => {
                println!("\nResults:");
                println!("{}", result.format_counts(display_qubits));
            }
        }
        return;
    }

    // Gradient modes are analytic (noiseless) — never let `--noise` be silently
    // dropped on the floor there.
    if noise_json.is_some() && (gradient_str.is_some() || gradient_fn_str.is_some()) {
        eprintln!(
            "--noise is not supported with --gradient / --gradient-of-fn (gradients are computed \
             analytically on the noiseless circuit). Drop --noise, or estimate a noisy gradient \
             by sampling with --shots."
        );
        std::process::exit(1);
    }

    // --- Mode: gradient-of-functional ---
    if let Some(ref spec_str) = gradient_fn_str {
        let (functional, spec_kind) = parse_functional_spec(spec_str).unwrap_or_else(|e| {
            eprintln!("Error parsing --gradient-of-fn: {}", e);
            std::process::exit(1);
        });
        let method = match grad_method_name.as_deref() {
            Some("score-fn") | Some("score-function") | Some("reinforce") => {
                FunctionalGradMethod::ScoreFunction {
                    shots: gradient_fn_shots.unwrap_or(1024),
                }
            }
            Some("diagonal") | None => FunctionalGradMethod::DiagonalObservable,
            Some(other) => {
                eprintln!(
                    "Unknown method for --gradient-of-fn: {}. Options: diagonal, score-fn",
                    other
                );
                std::process::exit(1);
            }
        };
        info(format!("Functional: {}", spec_kind));
        info(format!("Method: {:?}", method));

        let grads = match chosen {
            "statevector" | "sv" => compute_functional_gradient(
                &StatevectorBackend::new(),
                &circuit,
                &params,
                &functional,
                &method,
            ),
            m if omega_backend_mps::select::is_mps(m) => compute_functional_gradient(
                &mps_backend_from(m, device_name.as_deref()).expect("is_mps implies parse_mps"),
                &circuit,
                &params,
                &functional,
                &method,
            ),
            other => {
                eprintln!("--gradient-of-fn not supported for backend: {}", other);
                std::process::exit(1);
            }
        };
        let grads = grads.unwrap_or_else(|e| {
            eprintln!("Functional-gradient error: {}", e);
            std::process::exit(1);
        });
        let method_name = match &method {
            FunctionalGradMethod::DiagonalObservable => "diagonal",
            FunctionalGradMethod::ScoreFunction { .. } => "score-fn",
        };
        if format.is_machine() {
            let named: Vec<(String, f64)> = grads
                .iter()
                .map(|(sym, g)| {
                    let name = circuit
                        .symbols
                        .get(sym)
                        .cloned()
                        .unwrap_or_else(|| format!("sym_{}", sym));
                    (name, *g)
                })
                .collect();
            let v = serialize::functional_gradient_to_json(spec_kind, method_name, &named);
            println!("{}", v);
        } else {
            println!("\nGradients:");
            for (sym_id, grad) in &grads {
                let name = circuit
                    .symbols
                    .get(sym_id)
                    .unwrap_or(&format!("sym_{}", sym_id))
                    .clone();
                println!("  d E[f]/d({}) = {:.10}", name, grad);
            }
        }
        return;
    }

    // --- Mode: gradient ---
    if let Some(ref obs_str) = gradient_str {
        let observable = parse_observable(obs_str);
        let method = match grad_method_name.as_deref() {
            Some("param-shift") | Some("parameter-shift") => GradMethod::ParameterShift,
            Some("finite-diff") | Some("fd") => GradMethod::FiniteDifference { epsilon: 1e-7 },
            Some("adjoint") | None => GradMethod::Adjoint,
            Some(other) => {
                eprintln!(
                    "Unknown gradient method: {}. Options: adjoint, param-shift, finite-diff",
                    other
                );
                std::process::exit(1);
            }
        };

        info(format!("Observable: {}", obs_str));
        info(format!("Method: {:?}", method));

        // Honour --device metal for the statevector gradient path.
        // Same gating as the execute mode: omega-cli `metal` feature
        // must be on; gracefully falls back to CPU on Metal failure.
        let requested_device = device_name.as_deref().map(|s| {
            omega_core::device::DeviceKind::parse(s).unwrap_or_else(|e| {
                eprintln!("Invalid --device: {e}");
                std::process::exit(1);
            })
        });
        let resolved_device = omega_core::device::DeviceKind::resolve(requested_device);
        #[cfg(feature = "metal")]
        let use_metal_grad = matches!(resolved_device, omega_core::device::DeviceKind::Metal)
            && (chosen == "statevector" || chosen == "sv");
        #[cfg(not(feature = "metal"))]
        let use_metal_grad = false;
        #[cfg(feature = "cuda")]
        let use_cuda_grad = matches!(resolved_device, omega_core::device::DeviceKind::Cuda)
            && (chosen == "statevector" || chosen == "sv");
        #[cfg(not(feature = "cuda"))]
        let use_cuda_grad = false;
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        let _ = resolved_device;
        if let Some(m) = multi_control_notice(
            multi_control_requested,
            multi_control,
            use_cuda_grad || use_metal_grad,
        ) {
            info(m);
        }

        let grads = match chosen {
            "statevector" | "sv" => {
                if use_cuda_grad {
                    #[cfg(feature = "cuda")]
                    {
                        info("Device: cuda".to_string());
                        match omega_backend_statevector_cuda::CudaStatevectorBackend::new()
                            .map(|b| b.with_multi_control(multi_control))
                        {
                            Ok(b) => {
                                match compute_gradient(&b, &circuit, &params, &observable, &method)
                                {
                                    Ok(g) => Ok(g),
                                    Err(omega_core::error::OmegaError::Unsupported(msg))
                                    | Err(omega_core::error::OmegaError::Backend(msg)) => {
                                        info(format!("cuda fallback to cpu: {msg}"));
                                        compute_gradient(
                                            &StatevectorBackend::new(),
                                            &circuit,
                                            &params,
                                            &observable,
                                            &method,
                                        )
                                    }
                                    Err(e) => Err(e),
                                }
                            }
                            Err(e) => {
                                info(format!("cuda unavailable: {e}; falling back to cpu"));
                                compute_gradient(
                                    &StatevectorBackend::new(),
                                    &circuit,
                                    &params,
                                    &observable,
                                    &method,
                                )
                            }
                        }
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        unreachable!("use_cuda_grad is false without --features cuda")
                    }
                } else if use_metal_grad {
                    #[cfg(feature = "metal")]
                    {
                        info("Device: metal".to_string());
                        match omega_backend_statevector_metal::MetalStatevectorBackend::new()
                            .map(|b| b.with_multi_control(multi_control))
                        {
                            Ok(b) => {
                                match compute_gradient(&b, &circuit, &params, &observable, &method)
                                {
                                    Ok(g) => Ok(g),
                                    Err(omega_core::error::OmegaError::Unsupported(msg))
                                    | Err(omega_core::error::OmegaError::Backend(msg)) => {
                                        info(format!("metal fallback to cpu: {msg}"));
                                        compute_gradient(
                                            &StatevectorBackend::new(),
                                            &circuit,
                                            &params,
                                            &observable,
                                            &method,
                                        )
                                    }
                                    Err(e) => Err(e),
                                }
                            }
                            Err(e) => {
                                info(format!("metal unavailable: {e}; falling back to cpu"));
                                compute_gradient(
                                    &StatevectorBackend::new(),
                                    &circuit,
                                    &params,
                                    &observable,
                                    &method,
                                )
                            }
                        }
                    }
                    #[cfg(not(feature = "metal"))]
                    {
                        unreachable!("use_metal_grad is false without --features metal")
                    }
                } else {
                    compute_gradient(
                        &StatevectorBackend::new(),
                        &circuit,
                        &params,
                        &observable,
                        &method,
                    )
                }
            }
            m if omega_backend_mps::select::is_mps(m) => compute_gradient(
                &mps_backend_from(m, device_name.as_deref()).expect("is_mps implies parse_mps"),
                &circuit,
                &params,
                &observable,
                &method,
            ),
            "photonics" => {
                let b = input_state
                    .map(PhotonicsBackend::with_input)
                    .unwrap_or_default();
                compute_gradient(&b, &circuit, &params, &observable, &method)
            }
            _ => {
                eprintln!("Gradient not supported for backend: {}", chosen);
                std::process::exit(1);
            }
        };

        let grads = grads.unwrap_or_else(|e| {
            eprintln!("Gradient error: {}", e);
            std::process::exit(1);
        });

        if format.is_machine() {
            let method_name = grad_method_name.as_deref().unwrap_or("adjoint");
            let named: Vec<(String, f64)> = grads
                .iter()
                .map(|(sym, g)| {
                    let name = circuit
                        .symbols
                        .get(sym)
                        .cloned()
                        .unwrap_or_else(|| format!("sym_{}", sym));
                    (name, *g)
                })
                .collect();
            let v = serialize::gradient_to_json(obs_str, method_name, &named);
            println!("{}", v);
        } else {
            println!("\nGradients:");
            for (sym_id, grad) in &grads {
                let name = circuit
                    .symbols
                    .get(sym_id)
                    .unwrap_or(&format!("sym_{}", sym_id))
                    .clone();
                println!("  d<O>/d({}) = {:.10}", name, grad);
            }
        }
        return;
    }

    // --- Mode: expectation ---
    if let Some(ref obs_str) = observable_str {
        let observable = parse_observable(obs_str);

        info(format!("Observable: {}", obs_str));

        // `honoured: false`, unconditionally and in EVERY build. Unlike the
        // execute and gradient paths, this one has no CUDA dispatch at all —
        // `chosen == "statevector"` goes straight to the CPU `StatevectorBackend`
        // below, with no `--device` check on the way. So `--multi-control` can
        // never apply here, not even on a `--features cuda` build with
        // `--device cuda`, which is exactly the invocation a user would expect to
        // honour it. Same shape as the `--device cuda` / MPS-SVD gap recorded in
        // CUDA_TODO.md §0: a flag that reaches one dispatch path and not its
        // neighbour. Caught by `multi_control_is_not_silently_dropped`, which
        // failed against a first fix that instrumented only the other two.
        if let Some(m) = multi_control_notice(multi_control_requested, multi_control, false) {
            info(m);
        }

        // Noise on an expectation value is exact only via the pauliprop
        // Heisenberg adjoint; statevector/mps expectations are analytic and
        // noiseless, so `--noise` there would be silently dropped — reject it.
        let exp_noise = noise_json.as_deref().map(parse_noise_model);
        if exp_noise.is_some() && !matches!(chosen, "pauliprop" | "pp") {
            eprintln!(
                "--noise with --expectation is only supported on --backend pauliprop \
                 (got '{chosen}'); statevector/mps compute an analytic noiseless expectation. \
                 Use --shots for a noisy sampled estimate instead."
            );
            std::process::exit(1);
        }

        // Set by the pauliprop arm; carried into the JSON below so a machine
        // consumer gets the error bound instead of having to scrape stderr.
        let mut pp_cert: Option<omega_backend_pauliprop::PauliPropCertificate> = None;
        let val = match chosen {
            "statevector" | "sv" => {
                StatevectorBackend::new().expectation(&circuit, &params, &observable)
            }
            m if omega_backend_mps::select::is_mps(m) => {
                mps_backend_from(m, device_name.as_deref())
                    .expect("is_mps implies parse_mps")
                    .expectation(&circuit, &params, &observable)
            }
            "pauliprop" | "pp" => {
                let truncating =
                    pp_truncate.is_some() || pp_max_weight.is_some() || pp_max_freq.is_some();
                let mut backend = if truncating {
                    PauliPropBackend::with_truncation_freq(
                        pp_truncate.unwrap_or(0.0),
                        pp_max_weight,
                        pp_max_freq,
                    )
                } else {
                    PauliPropBackend::new()
                };
                // Applied AFTER the constructor, and outside the `truncating`
                // branch, because raising the ceiling is not truncation: it
                // does not enter `truncating`, does not produce `dropped_mass`,
                // and must work on an otherwise-exact run — which is the whole
                // point of it.
                if let Some(mt) = pp_max_terms {
                    backend = backend.with_max_terms(Some(mt));
                }
                if let Some(md) = pp_max_dropped {
                    backend = backend.with_max_dropped_mass(Some(md));
                }
                if let Some(model) = &exp_noise {
                    backend = backend.with_noise(model.clone());
                }
                // The certificate is taken on EVERY pauliprop run, not only
                // truncated ones. An exact run has a certificate too — it reads
                // zero — and the report's acceptance says so explicitly: "an
                // untruncated run reports it as 0". Emitting it only when
                // truncating would leave a consumer unable to tell "exact" from
                // "this build does not report it".
                match backend.expectation_with_certificate(&circuit, &params, &observable) {
                    Ok((v, cert)) => {
                        if truncating {
                            // The dropped-mass budget is a GENUINE BOUND, not
                            // an estimate: |<P>| <= 1 for every Pauli string,
                            // so the error in <O> is at most the discarded L1
                            // mass. No gauge caveat — unlike the MPS fidelity,
                            // which is labelled `~` for exactly that reason.
                            eprintln!(
                                "pauliprop: dropped_mass={:.3e} of range {:.3e} \
                                 (a bound on |Δ⟨O⟩|, not an estimate); \
                                 terms final {} peak {}",
                                cert.dropped_mass,
                                cert.observable_range,
                                cert.final_terms,
                                cert.peak_terms,
                            );
                        }
                        pp_cert = Some(cert);
                        Ok(v)
                    }
                    Err(e) => Err(e),
                }
            }
            "photonics" => {
                let b = input_state
                    .map(PhotonicsBackend::with_input)
                    .unwrap_or_default();
                b.expectation(&circuit, &params, &observable)
            }
            _ => {
                eprintln!("Expectation not supported for backend: {}", chosen);
                std::process::exit(1);
            }
        };

        let val = val.unwrap_or_else(|e| {
            eprintln!("Expectation error: {}", e);
            std::process::exit(1);
        });

        if format.is_machine() {
            let mut doc = serialize::expectation_to_json(obs_str, val);
            if let Some(cert) = &pp_cert {
                doc = serialize::attach_pauliprop_certificate(doc, cert);
            }
            println!("{doc}");
        } else {
            println!("\n<O> = {:.10}", val);
        }
        return;
    }

    // --- Mode: execute (default) ---
    // Auto-detect mid-circuit behaviour: if the circuit has any classically
    // conditioned gate, or any measurement followed by another op, or any
    // cbit written by more than one `measure` (Qiskit semantics:
    // last-write-wins on the creg state, omega's basis-state sampling would
    // double-count), the simulator must collapse. Pure end-of-circuit
    // sampling with a 1:1 qubit→cbit mapping stays in Skip mode (cheaper,
    // preserves the analytic statevector path).
    // The predicate lives in `omega_core::executor` so the N-way counts
    // matrix drives the same decision this path ships (see its docs).
    let needs_collapse = omega_core::executor::needs_collapse(&circuit);
    let mid_circuit_mode = if needs_collapse {
        MidCircuitMode::Collapse
    } else {
        MidCircuitMode::Skip
    };
    let config = ExecConfig {
        shots,
        seed,
        mid_circuit_mode,
    };

    let noise_model = noise_json.as_deref().map(parse_noise_model);

    // Sampling noise is applied by the two trajectory samplers (statevector and
    // MPS). Any other backend would silently return the *noiseless* distribution
    // — a user who passed `--noise` would believe they measured a noisy circuit
    // when they did not. Fail loudly instead of dropping the model on the floor.
    // (pauliprop applies noise to `--expectation`, handled in that mode above.)
    // `is_mps`, not `== "mps"`: writing the literal here would spuriously
    // reject `--backend mps:512 --noise …`.
    if noise_model.is_some()
        && !(matches!(chosen, "statevector" | "sv") || omega_backend_mps::select::is_mps(chosen))
    {
        eprintln!(
            "--noise sampling is supported on --backend statevector or mps (got '{chosen}'); \
             this backend cannot apply a noise model to sampled counts. Re-run with \
             --backend statevector/mps, or drop --noise."
        );
        std::process::exit(1);
    }
    // A noise model needs shots: a single state vector can't carry a stochastic
    // channel. Reject `--statevector --noise` rather than return one random
    // trajectory dressed up as "the" state (matches the aria CLI).
    if noise_model.is_some() && shots.is_none() {
        eprintln!(
            "--noise requires sampling (--shots N); a single --statevector output cannot carry \
             a stochastic noise channel. Drop --statevector, or drop --noise."
        );
        std::process::exit(1);
    }

    // `--dump-state-bits` guards, all hard errors (PLAN-BITEQ-CUDA-METAL.md
    // H1). The artifact claims "this device, this mode, these bits", so
    // anything that could quietly change which device ran — OMEGA_DEVICE,
    // a feature-gated arm compiled out, the graceful CPU fallback in the
    // dispatch arms below — must refuse instead of falling back.
    // (`--noise` needs no guard here: `--noise` without shots is already
    // refused above, and the dump requires the analytic path.)
    if dump_state_bits.is_some() {
        if shots.is_some() {
            eprintln!(
                "--dump-state-bits needs the analytic state (--statevector); \
                 sampled counts have no amplitudes to dump. Drop --shots."
            );
            std::process::exit(1);
        }
        if !matches!(chosen, "statevector" | "sv") {
            eprintln!(
                "--dump-state-bits is defined for the statevector backends only \
                 (got '{chosen}')."
            );
            std::process::exit(1);
        }
        if device_name.is_none() {
            eprintln!(
                "--dump-state-bits requires an EXPLICIT --device (cpu, metal, \
                 cuda, opencl). The artifact names the device it came from, so \
                 the device cannot be left to OMEGA_DEVICE or a default."
            );
            std::process::exit(1);
        }
    }

    // Resolve the compute device requested via `--device`.
    // Honours OMEGA_DEVICE if --device wasn't passed; falls back to CPU
    // with a single stderr notice when the requested device isn't
    // compiled in. Metal is honoured for the statevector backend only —
    // MPS / Pauli / Photonics stay CPU on this commit (steps 5+ /
    // future phases pick those up).
    let requested_device = device_name.as_deref().map(|s| {
        omega_core::device::DeviceKind::parse(s).unwrap_or_else(|e| {
            eprintln!("Invalid --device: {e}");
            std::process::exit(1);
        })
    });
    let resolved_device = omega_core::device::DeviceKind::resolve(requested_device);
    // `use_metal_sv` only goes true when the omega-cli `metal` feature
    // is compiled in. Without it, omega-core's `is_available()` returns
    // false and `resolve()` already falls back to CPU; we keep this
    // explicitly cfg-gated so the metal crate dep itself is optional.
    #[cfg(feature = "metal")]
    let use_metal_sv = matches!(resolved_device, omega_core::device::DeviceKind::Metal)
        && (chosen == "statevector" || chosen == "sv")
        && noise_model.is_none();
    #[cfg(not(feature = "metal"))]
    let use_metal_sv = false;
    #[cfg(feature = "cuda")]
    let use_cuda_sv = matches!(resolved_device, omega_core::device::DeviceKind::Cuda)
        && (chosen == "statevector" || chosen == "sv")
        && noise_model.is_none();
    #[cfg(not(feature = "cuda"))]
    let use_cuda_sv = false;
    #[cfg(feature = "opencl")]
    let use_opencl_sv = matches!(resolved_device, omega_core::device::DeviceKind::OpenCl)
        && (chosen == "statevector" || chosen == "sv")
        && noise_model.is_none();
    #[cfg(not(feature = "opencl"))]
    let use_opencl_sv = false;
    #[cfg(not(any(feature = "metal", feature = "cuda", feature = "opencl")))]
    let _ = resolved_device;
    // Whether the statevector run will ACTUALLY dispatch to the device the
    // caller named. `DeviceKind::resolve` falls back to CPU with only a
    // stderr notice when the feature is not compiled in, so `use_*_sv` is
    // simply false and NO error ever fires for the arms' own fallback guard
    // to catch — the decision has to be read here, not inferred from errors.
    let sv_path = matches!(chosen, "statevector" | "sv");
    let dispatches_to_requested = {
        use omega_core::device::DeviceKind as D;
        match requested_device {
            Some(D::Metal) => use_metal_sv,
            Some(D::Cuda) => use_cuda_sv,
            Some(D::OpenCl) => use_opencl_sv,
            Some(D::Cpu) | None => true,
        }
    };

    // **An EXPLICIT `--device` that cannot be honoured is an error, not a
    // downgrade.** Reported from two GPU hosts: `--device cuda` above 28
    // qubits ran silently on the CPU and produced *plausible* numbers that
    // supported a confident wrong conclusion about GPU performance — a whole
    // benchmark lane was published and retracted because a silent device
    // substitution does not look like missing data, it looks like data.
    //
    // Scoped deliberately: only when the flag was PASSED (not `OMEGA_DEVICE`,
    // not the default), and only on the statevector path, which is the one
    // these arms select. Automatic selection keeps falling back with its
    // notice, because there the caller expressed no expectation to violate.
    if device_name.is_some() && sv_path && !dispatches_to_requested {
        eprintln!(
            "--device {} was requested but this binary cannot dispatch to it \
             (feature not compiled in). Refusing rather than running on the CPU \
             and reporting a device result: rebuild with --features {}, or drop \
             --device to let the run fall back deliberately.",
            requested_device.map(|d| d.name()).unwrap_or("?"),
            requested_device.map(|d| d.name()).unwrap_or("?"),
        );
        std::process::exit(1);
    }
    if dump_state_bits.is_some() && !dispatches_to_requested {
        eprintln!(
            "--dump-state-bits: --device {} was requested but this binary \
             will not dispatch to it (feature not compiled in). Refusing to \
             write an artifact that would silently describe the CPU.",
            requested_device.map(|d| d.name()).unwrap_or("?"),
        );
        std::process::exit(1);
    }
    // Which arm ACTUALLY ran — the artifact's `backend` field and hex width
    // derive from this, never from what was requested.
    #[allow(unused_mut)]
    let mut executed_arm: &str = "cpu-f64";
    if let Some(m) = multi_control_notice(
        multi_control_requested,
        multi_control,
        use_cuda_sv || use_metal_sv,
    ) {
        info(m);
    }

    // For circuits with mid-circuit measurement under Collapse mode, the
    // executor returns one trajectory per call. To get a faithful shot
    // distribution we run the circuit N times here with independent RNG
    // seeds and aggregate the counts. Done in the CLI rather than the
    // backend so the analytic Skip path stays a single execute(); only
    // conditional / mid-measure circuits pay the N× cost.
    // The MPS truncation certificate, captured from whichever dispatch arm
    // produces it so the JSON/JSONL paths can carry it as data rather than
    // leaving it only in a stderr line a machine consumer cannot use.
    let mut mps_run_stats: Option<omega_backend_mps::MpsRunStats> = None;
    let result = if let (true, "statevector" | "sv", None, Some(n_shots)) =
        (needs_collapse, chosen, &noise_model, shots)
    {
        use std::collections::HashMap;
        let mut agg: HashMap<omega_core::outcome::Outcome, u32> = HashMap::new();
        let base_seed = seed.unwrap_or(0xC0FFEE_u64);
        for s in 0..n_shots as u64 {
            let traj_seed = base_seed.wrapping_add(s).wrapping_mul(2654435761);
            let traj_cfg = ExecConfig {
                shots: Some(1),
                seed: Some(traj_seed),
                mid_circuit_mode: MidCircuitMode::Collapse,
            };
            match StatevectorBackend::new().execute(&circuit, &params, &traj_cfg) {
                Ok(omega_core::executor::ExecResult::Counts(c)) => {
                    for (k, v) in c {
                        *agg.entry(k).or_insert(0) += v;
                    }
                }
                Ok(_) => unreachable!("shots:Some(1) always yields Counts"),
                Err(e) => {
                    eprintln!("Execution error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Ok(omega_core::executor::ExecResult::Counts(agg))
    } else {
        match chosen {
            "statevector" | "sv" => {
                if let Some(model) = &noise_model {
                    info(format!("Noise model: {:?}", model));
                    omega_backend_statevector::NoisyStatevectorBackend::with_model(
                        model.clone(),
                        seed,
                    )
                    .execute(&circuit, &params, &config)
                } else if use_cuda_sv {
                    #[cfg(feature = "cuda")]
                    {
                        info("Device: cuda".to_string());
                        let cuda_result =
                            match omega_backend_statevector_cuda::CudaStatevectorBackend::new()
                                .map(|b| b.with_multi_control(multi_control))
                            {
                                Ok(b) => b.execute(&circuit, &params, &config),
                                Err(e) => Err(omega_core::error::OmegaError::Backend(format!(
                                    "cuda unavailable: {e}"
                                ))),
                            };
                        match cuda_result {
                            Ok(r) => {
                                executed_arm = "cuda-f32";
                                Ok(r)
                            }
                            // Same graceful fallback semantics as Metal.
                            Err(omega_core::error::OmegaError::Unsupported(msg))
                            | Err(omega_core::error::OmegaError::Backend(msg)) => {
                                if dump_state_bits.is_some() {
                                    eprintln!(
                                        "--dump-state-bits: cuda was requested but \
                                         execution fell back to cpu ({msg}); a \
                                         bit-dump must come from the device it names."
                                    );
                                    std::process::exit(1);
                                }
                                // Explicit --device: refuse, never substitute.
                                // See the dispatch guard above for why a silent
                                // CPU result is worse than no result.
                                if device_name.is_some() {
                                    eprintln!(
                                        "--device cuda was requested but the run \
                                         could not use it: {msg}. Refusing rather \
                                         than returning a CPU result labelled as a \
                                         device run. Drop --device to fall back \
                                         deliberately."
                                    );
                                    std::process::exit(1);
                                }
                                info(format!("cuda fallback to cpu: {msg}"));
                                StatevectorBackend::new().execute(&circuit, &params, &config)
                            }
                            Err(e) => Err(e),
                        }
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        unreachable!("use_cuda_sv is false without --features cuda")
                    }
                } else if use_opencl_sv {
                    #[cfg(feature = "opencl")]
                    {
                        info("Device: opencl".to_string());
                        let opencl_result =
                            match omega_backend_statevector_opencl::OpenClStatevectorBackend::new()
                            {
                                Ok(b) => b.execute(&circuit, &params, &config),
                                Err(e) => Err(omega_core::error::OmegaError::Backend(format!(
                                    "opencl unavailable: {e}"
                                ))),
                            };
                        match opencl_result {
                            Ok(r) => {
                                executed_arm = "opencl-f32";
                                Ok(r)
                            }
                            // Same graceful fallback as Metal / CUDA.
                            Err(omega_core::error::OmegaError::Unsupported(msg))
                            | Err(omega_core::error::OmegaError::Backend(msg)) => {
                                if dump_state_bits.is_some() {
                                    eprintln!(
                                        "--dump-state-bits: opencl was requested but \
                                         execution fell back to cpu ({msg}); a \
                                         bit-dump must come from the device it names."
                                    );
                                    std::process::exit(1);
                                }
                                // Explicit --device: refuse, never substitute.
                                // See the dispatch guard above for why a silent
                                // CPU result is worse than no result.
                                if device_name.is_some() {
                                    eprintln!(
                                        "--device opencl was requested but the run \
                                         could not use it: {msg}. Refusing rather \
                                         than returning a CPU result labelled as a \
                                         device run. Drop --device to fall back \
                                         deliberately."
                                    );
                                    std::process::exit(1);
                                }
                                info(format!("opencl fallback to cpu: {msg}"));
                                StatevectorBackend::new().execute(&circuit, &params, &config)
                            }
                            Err(e) => Err(e),
                        }
                    }
                    #[cfg(not(feature = "opencl"))]
                    {
                        unreachable!("use_opencl_sv is false without --features opencl")
                    }
                } else if use_metal_sv {
                    #[cfg(feature = "metal")]
                    {
                        info("Device: metal".to_string());
                        let metal_result =
                            match omega_backend_statevector_metal::MetalStatevectorBackend::new()
                                .map(|b| b.with_multi_control(multi_control))
                            {
                                Ok(b) => b.execute(&circuit, &params, &config),
                                Err(e) => Err(omega_core::error::OmegaError::Backend(format!(
                                    "metal unavailable: {e}"
                                ))),
                            };
                        match metal_result {
                            Ok(r) => {
                                executed_arm = "metal-f32";
                                Ok(r)
                            }
                            // Graceful fallback: when metal rejects a feature it
                            // doesn't implement yet (CCX/CSwap, Reset, mid-circuit
                            // measurement) or the device isn't usable, drop down
                            // to the CPU backend rather than fail the run. The
                            // notice is verbose-only so verify-qiskit doesn't
                            // drown in fallback noise.
                            Err(omega_core::error::OmegaError::Unsupported(msg))
                            | Err(omega_core::error::OmegaError::Backend(msg)) => {
                                if dump_state_bits.is_some() {
                                    eprintln!(
                                        "--dump-state-bits: metal was requested but \
                                         execution fell back to cpu ({msg}); a \
                                         bit-dump must come from the device it names."
                                    );
                                    std::process::exit(1);
                                }
                                // Explicit --device: refuse, never substitute.
                                // See the dispatch guard above for why a silent
                                // CPU result is worse than no result.
                                if device_name.is_some() {
                                    eprintln!(
                                        "--device metal was requested but the run \
                                         could not use it: {msg}. Refusing rather \
                                         than returning a CPU result labelled as a \
                                         device run. Drop --device to fall back \
                                         deliberately."
                                    );
                                    std::process::exit(1);
                                }
                                info(format!("metal fallback to cpu: {msg}"));
                                StatevectorBackend::new().execute(&circuit, &params, &config)
                            }
                            Err(e) => Err(e),
                        }
                    }
                    #[cfg(not(feature = "metal"))]
                    {
                        unreachable!("use_metal_sv is false without --features metal")
                    }
                } else {
                    StatevectorBackend::new().execute(&circuit, &params, &config)
                }
            }
            m if omega_backend_mps::select::is_mps(m) => {
                if let Some(model) = &noise_model {
                    info(format!("Noise model: {:?}", model));
                    omega_backend_mps::NoisyMpsBackend::with_model(
                        match omega_backend_mps::select::parse_mps(chosen) {
                            Ok(Some(omega_backend_mps::select::MpsSelect::Fixed { chi })) => chi,
                            Ok(Some(omega_backend_mps::select::MpsSelect::Auto { max_chi })) => {
                                max_chi
                            }
                            _ => omega_backend_mps::select::DEFAULT_CHI,
                        },
                        model.clone(),
                    )
                    .execute(&circuit, &params, &config)
                } else {
                    // Hold the concrete backend so its truncation certificate
                    // can be reported to stderr (stdout/exit unchanged — K14).
                    executed_arm = "mps-cpu";
                    #[cfg(feature = "cuda")]
                    if matches!(resolved_device, omega_core::device::DeviceKind::Cuda) {
                        executed_arm = "mps-cpu+cuda-svd-hook";
                    }
                    #[cfg(all(feature = "metal", not(feature = "cuda")))]
                    if matches!(resolved_device, omega_core::device::DeviceKind::Metal) {
                        executed_arm = "mps-cpu+metal-contract-hook";
                    }
                    let backend = mps_backend_from(chosen, device_name.as_deref())
                        .expect("dispatch arm is guarded by is_mps");
                    let r = backend.execute(&circuit, &params, &config);
                    let stats = backend.last_run_stats();
                    // Keep the certificate for the machine-readable path too.
                    // Reporting it only on stderr meant a JSON consumer had to
                    // scrape log text to find out whether the number it was
                    // handed had been truncated — the certificate exists
                    // precisely so that question has an answer.
                    mps_run_stats = Some(stats);
                    // Report REAL truncation only; a ~1e-28 rounding tail isn't
                    // worth a line (1e-12 floor sits above it, below real loss).
                    if stats.discarded_weight > 1e-12 {
                        // `fidelity~` — the tilde is not decoration. It is an
                        // ESTIMATE, not a proven bound: the product form is a
                        // genuine lower bound in canonical gauge and this MPS
                        // is not canonical. Measured on 108 non-Clifford
                        // circuits with non-flat Schmidt spectra: it never
                        // exceeded the true fidelity, with F_true/F_est in
                        // [1.0067, 142.7] — tight when truncation is mild, up
                        // to ~140x pessimistic when the bond is starved.
                        // Evidence, not proof, and useful for "is this run
                        // usable" rather than "how good is this bad run".
                        eprintln!(
                            "mps: fidelity~{:.4} (estimate, not a bound) \
                             discarded_weight={:.3e} max_bond_reached={}",
                            stats.fidelity_estimate, stats.discarded_weight, stats.max_bond_reached
                        );
                    }
                    r
                }
            }
            "pauli" | "stabilizer" => PauliBackend::new().execute(&circuit, &params, &config),
            "pauliprop" | "pp" => {
                eprintln!(
                    "pauliprop is an expectation-value backend; it has no sampling/\
                     statevector output. Use it with --expectation \"<Pauli string>\"."
                );
                std::process::exit(1);
            }
            "photonics" => {
                let backend = if let Some(input) = input_state {
                    PhotonicsBackend::with_input(input)
                } else {
                    let n = (num_modes as usize).div_ceil(2);
                    let mut input = vec![0u32; num_modes as usize];
                    for slot in input.iter_mut().take(n.min(num_modes as usize)) {
                        *slot = 1;
                    }
                    info(format!(
                        "Input Fock state: |{}>",
                        input
                            .iter()
                            .map(|n| n.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    ));
                    PhotonicsBackend::with_input(input)
                };
                backend.execute(&circuit, &params, &config)
            }
            other => {
                // Compiled-in names resolved above; fall back to plugins.
                // Built lazily so a normal statevector/mps run never scans a
                // plugin dir. The registry only needs to outlive this
                // `execute`, which returns an owned result.
                let registry = build_plugin_registry(&backend_dirs);
                if let Some(plugin) = registry.find_by_name(other) {
                    // Refuse a circuit the plugin doesn't declare support for,
                    // loudly — never dispatch and risk a wrong answer.
                    if let Err(e) = plugin.check_circuit_supported(&circuit) {
                        eprintln!("{e}");
                        std::process::exit(1);
                    }
                    plugin.execute(&circuit, &params, &config)
                } else {
                    let plugins = registry.list();
                    let plugin_note = if plugins.is_empty() {
                        String::new()
                    } else {
                        format!(" Plugins: {}.", plugins.join(", "))
                    };
                    eprintln!(
                        "Unknown backend: {}. Builtins: statevector, mps, mps:<chi>, \
                         mps:auto, mps:auto:<ceiling>, pauli, pauliprop, photonics.{}",
                        other, plugin_note
                    );
                    std::process::exit(1);
                }
            }
        }
    };

    let result = result.unwrap_or_else(|e| {
        eprintln!("Execution error: {}", e);
        std::process::exit(1);
    });

    if let Some(path) = &dump_state_bits {
        let omega_core::executor::ExecResult::Statevector(amps) = &result else {
            eprintln!(
                "--dump-state-bits: the run produced no statevector; nothing \
                 faithful to dump."
            );
            std::process::exit(1);
        };
        let mc = match multi_control {
            omega_core::executor::MultiControlMode::Decompose => "decompose",
            omega_core::executor::MultiControlMode::Exact => "exact",
        };
        // The exact OS build, because the verdict is scoped to a compiler
        // pair (PLAN-BITEQ H6): Metal shaders compile at runtime, so the OS
        // version IS part of the measurement's identity.
        let os_version = std::process::Command::new("uname")
            .arg("-rv")
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        let circuit_name = std::path::Path::new(file_path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| file_path.clone());
        match serialize::state_bits_to_json(&circuit_name, executed_arm, mc, &os_version, amps) {
            Ok(doc) => {
                if let Err(e) = std::fs::write(path, format!("{doc}\n")) {
                    eprintln!("--dump-state-bits: cannot write {path}: {e}");
                    std::process::exit(1);
                }
                info(format!(
                    "state bits ({executed_arm}, {mc}) written to {path}"
                ));
            }
            Err(e) => {
                eprintln!("--dump-state-bits: {e}");
                std::process::exit(1);
            }
        }
    }

    match format {
        Format::Json => {
            let mut v = serialize::exec_result_to_json(
                &result,
                counts_display_width,
                shots,
                &circuit.circuit_type,
            );
            if let Some(stats) = &mps_run_stats {
                v = serialize::attach_mps_certificate(v, stats);
            }
            v = serialize::attach_execution_identity(v, executed_arm, rayon::current_num_threads());
            println!("{}", v);
        }
        Format::Jsonl => match &result {
            omega_core::executor::ExecResult::Counts(counts) => {
                serialize::emit_jsonl_counts(
                    counts,
                    counts_display_width,
                    &circuit.circuit_type,
                    executed_arm,
                );
            }
            other => {
                // Non-counts results don't decompose into per-shot lines; emit a single JSON doc.
                let mut v =
                    serialize::exec_result_to_json(other, num_modes, shots, &circuit.circuit_type);
                if let Some(stats) = &mps_run_stats {
                    v = serialize::attach_mps_certificate(v, stats);
                }
                v = serialize::attach_execution_identity(
                    v,
                    executed_arm,
                    rayon::current_num_threads(),
                );
                println!("{}", v);
            }
        },
        Format::Text => {
            println!("\nResults:");
            match &result {
                omega_core::executor::ExecResult::Counts(counts) => {
                    let mut sorted: Vec<_> = counts.iter().collect();
                    sorted.sort_by(|a, b| b.1.cmp(a.1));
                    if circuit.circuit_type == CircuitType::Photonic {
                        for (encoded, count) in sorted.iter().take(20) {
                            let fock = omega_backend_photonics::sim::decode_fock_string(
                                // Photonic keys are packed occupancies, read as
                                // an integer; the Fock basis is far below 2^64.
                                encoded.as_u64().unwrap_or(0),
                                num_modes as usize,
                            );
                            println!("{}: {}", fock, count);
                        }
                    } else {
                        println!("{}", result.format_counts(counts_display_width));
                    }
                }
                _ => {
                    println!("{}", result.format_counts(counts_display_width));
                }
            }
        }
    }
}

fn parse_qubo_args(args: &[String]) -> qubo_mode::QuboOptions {
    let mut path: Option<String> = None;
    let mut depth: usize = 2;
    let mut shots: u32 = 2048;
    let mut seed: Option<u64> = None;
    let mut optimizer = qubo_mode::Optimizer::CmaEs;
    let mut top_k: usize = 20;
    let mut max_iters: usize = 50;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--qubo" => {
                path = Some(flag_value(args, &mut i, "--qubo"));
            }
            "--qaoa-depth" => {
                depth = flag_parse(args, &mut i, "--qaoa-depth", "a whole number of layers");
            }
            "--shots" => {
                shots = flag_parse(args, &mut i, "--shots", "a whole number of shots");
            }
            "--seed" => {
                seed = Some(flag_parse(args, &mut i, "--seed", "an integer seed"));
            }
            "--optimizer" => {
                let raw = flag_value(args, &mut i, "--optimizer");
                optimizer = match raw.as_str() {
                    "cma-es" | "cmaes" => qubo_mode::Optimizer::CmaEs,
                    "gradient" | "grad" => qubo_mode::Optimizer::Gradient,
                    other => {
                        eprintln!("Unknown optimizer: {}. Options: cma-es, gradient", other);
                        std::process::exit(1);
                    }
                };
            }
            "--top-k" => {
                top_k = flag_parse(args, &mut i, "--top-k", "a whole number");
            }
            "--max-iters" => {
                max_iters = flag_parse(args, &mut i, "--max-iters", "a whole number");
            }
            "--format" => {
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    let path = path.unwrap_or_else(|| {
        eprintln!("--qubo requires a path to the problem JSON");
        std::process::exit(1);
    });

    qubo_mode::QuboOptions {
        path,
        depth,
        shots,
        seed,
        optimizer,
        top_k,
        max_iters,
    }
}

/// Run --shor mode: factor a composite N via the Shor algorithm demo.
fn run_shor(args: &[String], format: Format) {
    let mut n: Option<u64> = None;
    let mut seed: Option<u64> = None;
    let mut max_attempts: u32 = 16;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--shor" => {}
            "--N" | "--n" => {
                n = Some(flag_parse(args, &mut i, "--N", "a composite integer"));
            }
            "--seed" => {
                seed = Some(flag_parse(args, &mut i, "--seed", "an integer seed"));
            }
            "--max-attempts" => {
                max_attempts = flag_parse(args, &mut i, "--max-attempts", "a whole number");
            }
            "--format" => {
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    let n = n.unwrap_or_else(|| {
        eprintln!("--shor requires --N <composite>");
        std::process::exit(1);
    });

    if n > 63 {
        eprintln!(
            "--shor demo is capped at N ≤ 63 (would need {}+ qubits); aborting.",
            2 * ((n as f64).log2().ceil() as usize) + ((n as f64).log2().ceil() as usize)
        );
        std::process::exit(1);
    }

    let info = |msg: String| {
        if format.is_machine() {
            eprintln!("{}", msg);
        } else {
            println!("{}", msg);
        }
    };

    info(format!("Shor: N={}, max_attempts={}", n, max_attempts));
    let result = omega_core::shor::factor(n, seed, max_attempts);

    if format.is_machine() {
        let factors = result.factors.map(|(p, q)| serde_json::json!([p, q]));
        let doc = serde_json::json!({
            "mode": "shor",
            "N": result.n,
            "factors": factors,
            "period": result.period,
            "iterations": result.iterations,
            "chosen_a": result.chosen_a,
        });
        println!("{}", doc);
    } else {
        match result.factors {
            Some((p, q)) => {
                println!("\nFactored {} = {} × {}", result.n, p, q);
                if let Some(r) = result.period {
                    println!("  period (a={}): r = {}", result.chosen_a, r);
                }
                println!("  iterations: {}", result.iterations);
            }
            None => {
                println!(
                    "\nShor demo exhausted {} attempts without a non-trivial factor of {}.",
                    result.iterations, result.n
                );
                if let Some(r) = result.period {
                    println!("  last recovered period: {}", r);
                }
            }
        }
    }
}

/// Parse an observable string like "0.5*Z0+0.3*Z1Z2" or "Z0Z1" into an Observable.
///
/// Grammar (informal):
///   observable = term ('+' term)*
///   term = [coeff '*'] pauli_string
///   pauli_string = pauli+
///   pauli = ('X'|'Y'|'Z'|'I') digit+
///
/// (`parse_functional_spec` lives in `omega_core::gradient`, shared
/// with the WASM host hook.)
fn parse_observable(s: &str) -> Observable {
    let s = s.replace(" ", "");
    let mut terms = Vec::new();

    for part in s.split('+') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        let (coeff, pauli_part) = if let Some(idx) = part.find('*') {
            let c: f64 = part[..idx].parse().unwrap_or_else(|_| {
                eprintln!("Invalid coefficient in observable: {}", &part[..idx]);
                std::process::exit(1);
            });
            (c, &part[idx + 1..])
        } else {
            // Check if it starts with a digit or minus (coefficient without explicit pauli)
            if part.starts_with(['X', 'Y', 'Z', 'I']) {
                (1.0, part)
            } else {
                // Try to parse as negative coefficient
                if let Some(idx) = part.find(['X', 'Y', 'Z', 'I']) {
                    let c: f64 = part[..idx].parse().unwrap_or(1.0);
                    (c, &part[idx..])
                } else {
                    eprintln!("Invalid observable term: {}", part);
                    std::process::exit(1);
                }
            }
        };

        let pauli_string = parse_pauli_string(pauli_part);
        terms.push((coeff, pauli_string));
    }

    if terms.is_empty() {
        eprintln!("Empty observable");
        std::process::exit(1);
    }

    Observable { terms }
}

/// Parse a `--noise '{...}'` JSON string into a `NoiseModel`.
///
/// Delegates to the shared [`omega_core::noise::NoiseModel`] parser, which
/// accepts both scalar (uniform) and per-qubit forms and rejects unknown keys.
fn parse_noise_model(s: &str) -> omega_backend_statevector::NoiseModel {
    omega_core::noise::NoiseModel::from_json(s).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    })
}

/// Parse "Z0Z1X2" into vec of (qubit, PauliOp).
fn parse_pauli_string(s: &str) -> Vec<(u32, PauliOp)> {
    let mut result = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let op = match chars[i] {
            'X' | 'x' => PauliOp::X,
            'Y' | 'y' => PauliOp::Y,
            'Z' | 'z' => PauliOp::Z,
            'I' | 'i' => PauliOp::I,
            other => {
                eprintln!("Invalid Pauli operator: {}", other);
                std::process::exit(1);
            }
        };
        i += 1;

        // Parse qubit index (one or more digits)
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if start == i {
            eprintln!("Missing qubit index after Pauli operator in: {}", s);
            std::process::exit(1);
        }
        let qubit: u32 = s[start..i].parse().unwrap();

        if !matches!(op, PauliOp::I) {
            result.push((qubit, op));
        }
    }

    result
}
