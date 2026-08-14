// SPDX-License-Identifier: Apache-2.0
//! `aria` — command-line front end for the Aria Quantum Language.
//!
//! Subcommands:
//! - `aria list   <file.aria>`                         — list circuit templates
//! - `aria parse  <file.aria> [--circuit NAME] [--int k=v]...`
//! - `aria run    <file.aria> --circuit NAME [--int k=v]... [--bind s=v]...
//!   [--shots N] [--seed S] [--backend sim|mps[:chi]|mps:auto[:ceiling]|gpu|tch|pauliprop|remote]
//!   [--statevector] [--expectation OBS] [--strict-truncation EPS]`
//! - `aria export <file.aria> --circuit NAME (--qasm|--qasm3|--json|--lean|--gate-model) [--int k=v]...`

use std::collections::HashMap;
use std::process::ExitCode;

use aria_core::ast::{parse_aria, Circuit};
use aria_runtime::{
    counts_width, expectation, run_counts, statevector, train_expectation, BackendSel, TrainConfig,
};
use omega_core::executor::ExecResult;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let rest = &args.get(1..).unwrap_or(&[]);
    // `aria run --help` used to print "--help requires a value", because the
    // parser took every `--name` outside `bool_flags` as value-taking. Asking a
    // CLI for help is the one thing that must never be an error.
    if rest.iter().any(|a| a == "-h" || a == "--help") {
        usage();
        return ExitCode::SUCCESS;
    }
    let result = match cmd {
        "list" => cmd_list(rest),
        "parse" => cmd_parse(rest),
        "run" => cmd_run(rest),
        "train" => cmd_train(rest),
        "tune" => cmd_tune(rest),
        "predict" => cmd_predict(rest),
        "export" => cmd_export(rest),
        "import" => cmd_import(rest),
        "-h" | "--help" | "help" | "" => {
            usage();
            return ExitCode::SUCCESS;
        }
        other => Err(format!("unknown subcommand '{other}' (try `aria --help`)")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    eprintln!(
        "aria — the Aria Quantum Language CLI\n\n\
         USAGE:\n  \
         aria list   <file.aria>\n  \
         aria parse  <file.aria> [--circuit NAME] [--int k=v]...\n  \
         aria run    <file.aria> --circuit NAME [--int k=v]... [--bind s=v]...\n              \
         [--shots N] [--seed S] [--backend sim|mps[:chi]|mps:auto[:ceiling]|gpu|tch|pauliprop|remote] [--statevector] [--expectation OBS]\n              \
         [--strict-truncation EPS  (fail if an MPS run's discarded weight exceeds EPS)]\n              \
         [--noise JSON  (sampled counts on --backend sim, e.g. '{{\"readout_flip\":0.02,\"amplitude_damping\":5e-4}}')]\n              \
         (pauliprop computes --expectation only; truncate deep circuits with\n              \
          [--truncate C] [--max-weight W] [--max-freq F])\n              \
         [--url URL --token TOK   (for --backend remote)]\n  \
         aria train  <file.aria> --circuit NAME --observable OBS [--int k=v]...\n              \
         [--steps N] [--lr R] [--seed S] [--init-scale S] [--opt gd|adam] [--freeze a,b] [--grad adjoint|shift|parallel]\n              \
         (supervised: --data X.csv [--labels y.csv] [--loss mse|bce] [--feature-prefix x] [--save-model m.json]  → fit a labelled dataset)\n  \
         aria tune   <file.aria> --circuit NAME --observable OBS --data X.csv [--labels y.csv]\n              \
         --space \"n=4..8:2,L=1..3,lr=log:1e-3..3e-1,opt=gd|adam\"\n              \
         [--trials N] [--steps N] [--seed S] [--sampler tpe|random|grid] [--pruner median|none] [--csv out.csv]\n  \
         aria predict <model.json> --data X.csv [--out scores.csv] [--backend B]\n  \
         aria export <file.aria> --circuit NAME (--qasm | --qasm3 | --json | --lean | --gate-model) [--int k=v]...\n  \
         aria import <file.qasm> [--name NAME]   (OpenQASM 2.0 -> .aria source on stdout)\n"
    );
}

// ---------------------------------------------------------------------------
// Argument helpers
// ---------------------------------------------------------------------------

/// A flat view over `--flag value` / `--flag` / positional arguments.
struct Args {
    positional: Vec<String>,
    /// Repeatable `--key value` options, in order.
    opts: Vec<(String, String)>,
    /// Boolean `--flag` switches.
    flags: Vec<String>,
}

/// The `--name value` options each subcommand understands, alongside its
/// boolean flags.
///
/// Declared rather than inferred so an unrecognised option is refused at PARSE
/// time, before the subcommand has read a file or reserved anything. The risk
/// of a declared list is that it drifts from what the code actually reads —
/// `unknown_flags_are_refused.rs` fails if it does, by comparing this against
/// the names the accessors were seen to ask for.
/// A refusal naming the option, and the nearest thing it might have been.
fn unknown_option(bad: &str, bool_flags: &[&str], opts: &[&str]) -> String {
    // A single substitution, or one insertion/deletion, away from a real name.
    let edit1 = |k: &&str| -> bool {
        let (a, b) = (k.as_bytes(), bad.as_bytes());
        match a.len().abs_diff(b.len()) {
            0 => a.iter().zip(b).filter(|(x, y)| x != y).count() == 1,
            1 => {
                let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
                let (mut li, mut si, mut skipped) = (0, 0, false);
                while li < long.len() && si < short.len() {
                    if long[li] == short[si] {
                        li += 1;
                        si += 1;
                    } else if skipped {
                        return false;
                    } else {
                        skipped = true;
                        li += 1;
                    }
                }
                true
            }
            _ => false,
        }
    };
    // Transposition: `--stpes` for `--steps`. Common, and not an edit-1.
    let transposed = |k: &&str| -> bool {
        let (a, b) = (k.as_bytes(), bad.as_bytes());
        a.len() == b.len()
            && (0..a.len().saturating_sub(1)).any(|i| {
                let mut t = b.to_vec();
                t.swap(i, i + 1);
                t == a
            })
    };
    let suggestion = opts
        .iter()
        .chain(bool_flags.iter())
        .find(|k| edit1(k) || transposed(k))
        .map(|k| format!(" (did you mean --{k}?)"))
        .unwrap_or_default();
    let mut known: Vec<&str> = opts.iter().chain(bool_flags.iter()).copied().collect();
    known.sort_unstable();
    format!(
        "unrecognised option --{bad}{suggestion}. It used to be accepted and \
         ignored, which for a mistyped safety flag means running without the \
         gate you asked for.\n  this subcommand understands: {}",
        known
            .iter()
            .map(|k| format!("--{k}"))
            .collect::<Vec<_>>()
            .join(" ")
    )
}

struct Vocabulary;
impl Vocabulary {
    const LIST: &'static [&'static str] = &[];
    const PARSE: &'static [&'static str] = &["circuit", "int"];
    const RUN: &'static [&'static str] = &[
        "backend", "bind", "circuit", "expectation", "int", "max-freq", "max-weight", "noise",
        "seed", "shots", "strict-truncation", "token", "truncate", "url",
    ];
    /// `train` delegates to `cmd_train_supervised` with the same `Args`, so its
    /// vocabulary is the union of both.
    const TRAIN: &'static [&'static str] = &[
        "backend", "circuit", "data", "feature-prefix", "freeze", "grad", "init-scale", "int",
        "labels", "loss", "lr", "observable", "opt", "save-model", "seed", "set", "steps",
        "strict-truncation",
    ];
    const PREDICT: &'static [&'static str] = &["backend", "data", "out", "strict-truncation"];
    const EXPORT: &'static [&'static str] = &["circuit", "int"];
    const TUNE: &'static [&'static str] = &[
        "backend", "circuit", "csv", "data", "feature-prefix", "int", "labels", "loss",
        "observable", "pruner", "sampler", "seed", "space", "steps", "strict-truncation",
        "trials",
    ];
    const IMPORT: &'static [&'static str] = &["name"];
}

fn parse_args(raw: &[String], bool_flags: &[&str], opts: &[&str]) -> Result<Args, String> {
    let mut a = Args {
        positional: Vec::new(),
        opts: Vec::new(),
        flags: Vec::new(),
    };
    let mut i = 0;
    while i < raw.len() {
        let tok = &raw[i];
        if let Some(name) = tok.strip_prefix("--") {
            if bool_flags.contains(&name) {
                a.flags.push(name.to_string());
                i += 1;
            } else {
                if !opts.contains(&name) {
                    // Previously ANY `--name` outside `bool_flags` was taken as
                    // an option, stored, and never read again — silently.
                    // Measured on examples/aria/bell.aria:
                    //   `--shot 9999`             ran 1024 shots (the default)
                    //   `--strict-trunctaion 0.5` ran with the DEFAULT gate
                    //   `--typo xyz`              ran, no diagnostic
                    // The second is the one that matters: a typo in a safety
                    // flag left the user believing they had tightened a gate
                    // they had not touched.
                    return Err(unknown_option(name, bool_flags, opts));
                }
                let val = raw
                    .get(i + 1)
                    .ok_or_else(|| format!("--{name} requires a value"))?;
                a.opts.push((name.to_string(), val.clone()));
                i += 2;
            }
        } else {
            a.positional.push(tok.clone());
            i += 1;
        }
    }
    Ok(a)
}

impl Args {
    fn first_positional(&self, what: &str) -> Result<&str, String> {
        self.positional
            .first()
            .map(String::as_str)
            .ok_or_else(|| format!("missing {what}"))
    }
    fn opt(&self, key: &str) -> Option<&str> {
        self.opts
            .iter()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
    fn has(&self, flag: &str) -> bool {
        self.flags.iter().any(|f| f == flag)
    }
    /// All values of a repeatable `--key v` option.
    fn all(&self, key: &str) -> Vec<&str> {
        self.opts
            .iter()
            .filter(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .collect()
    }
}

fn parse_kv_i64(
    items: impl Iterator<Item = impl AsRef<str>>,
) -> Result<Vec<(String, i64)>, String> {
    let mut out = Vec::new();
    for it in items {
        let s = it.as_ref();
        let (k, v) = s
            .split_once('=')
            .ok_or_else(|| format!("expected key=value, got '{s}'"))?;
        let v: i64 = v
            .trim()
            .parse()
            .map_err(|_| format!("'{v}' is not an integer (in '{s}')"))?;
        out.push((k.trim().to_string(), v));
    }
    Ok(out)
}

fn parse_kv_f64(
    items: impl Iterator<Item = impl AsRef<str>>,
) -> Result<HashMap<String, f64>, String> {
    let mut out = HashMap::new();
    for it in items {
        let s = it.as_ref();
        let (k, v) = s
            .split_once('=')
            .ok_or_else(|| format!("expected key=value, got '{s}'"))?;
        let v: f64 = v
            .trim()
            .parse()
            .map_err(|_| format!("'{v}' is not a number (in '{s}')"))?;
        out.insert(k.trim().to_string(), v);
    }
    Ok(out)
}

fn read_source(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))
}

/// Parse + instantiate a named circuit with the given integer template params.
fn instantiate(src: &str, name: &str, ints: &[(String, i64)]) -> Result<Circuit, String> {
    let prog = parse_aria(src)?;
    let params: Vec<(&str, i64)> = ints.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    prog.instantiate(name, &params)
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

fn cmd_list(raw: &[String]) -> Result<(), String> {
    let a = parse_args(raw, &[], Vocabulary::LIST)?;
    let path = a.first_positional("<file.aria>")?;
    let prog = parse_aria(&read_source(path)?)?;
    println!("{path}:");
    for c in &prog.circuits {
        let ps = c
            .params
            .iter()
            .map(|(n, _ty)| n.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        println!("  circuit {}({ps})", c.name);
    }
    for o in &prog.observables {
        println!("  observable {}", o.name);
    }
    Ok(())
}

fn cmd_parse(raw: &[String]) -> Result<(), String> {
    let a = parse_args(raw, &[], Vocabulary::PARSE)?;
    let path = a.first_positional("<file.aria>")?;
    let src = read_source(path)?;
    let prog = parse_aria(&src)?;
    let ints = parse_kv_i64(a.all("int").into_iter())?;
    let names: Vec<String> = match a.opt("circuit") {
        Some(n) => vec![n.to_string()],
        None => prog.circuits.iter().map(|c| c.name.clone()).collect(),
    };
    for name in &names {
        match instantiate(&src, name, &ints) {
            Ok(c) => println!(
                "{name}: ok — {} qubits, {} classical bits, {} instructions",
                c.n_qubits(),
                c.n_clbits(),
                c.instructions.len()
            ),
            Err(e) => println!("{name}: parametric (needs --int): {e}"),
        }
    }
    Ok(())
}

/// `n_qubits` is no longer the render width — the key carries its own.
///
/// It is kept in the signature so callers need not all change, and is checked
/// against the data rather than trusted. Padding a correct key to the caller's
/// idea of the width is how a 2-bit outcome came to be printed as 1024
/// characters, and no type error catches it.
fn print_counts(res: ExecResult, n_qubits: usize) {
    match res {
        ExecResult::Counts(counts) => {
            debug_assert!(
                counts.is_empty() || counts.keys().all(|o| o.width() as usize == n_qubits),
                "print_counts was told {n_qubits} qubits but the outcomes are \
                 {:?} wide",
                counts.keys().map(|o| o.width()).collect::<std::collections::BTreeSet<_>>()
            );
            let mut rows: Vec<_> = counts.into_iter().collect();
            rows.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
            let total: u64 = rows.iter().map(|(_, c)| *c as u64).sum::<u64>().max(1);
            for (state, count) in rows {
                let p = count as f64 / total as f64;
                println!("|{}>  {count}  ({p:.4})", state.to_bitstring());
            }
        }
        other => println!("{other:?}"),
    }
}

/// Apply `--strict-truncation` to the MPS ceiling every backend will enforce.
///
/// Called by every subcommand that takes `--backend`, not just `run`. The flag
/// briefly worked only in `run`, which meant `train`/`tune`/`predict` could not
/// opt into an approximation at all while still being gated by the default —
/// an asymmetry with no reason behind it.
fn apply_strict_truncation(a: &Args) -> Result<Option<f64>, String> {
    let eps: Option<f64> = a
        .opt("strict-truncation")
        .map(|s| {
            s.parse::<f64>()
                .map_err(|_| format!("bad --strict-truncation '{s}'"))
        })
        .transpose()?;
    if let Some(v) = eps {
        if v.is_nan() || v < 0.0 {
            return Err(format!("--strict-truncation must be ≥ 0 (got {v})"));
        }
        aria_runtime::run::set_mps_discard_ceiling(v);
    }
    Ok(eps)
}

fn cmd_run(raw: &[String]) -> Result<(), String> {
    let a = parse_args(raw, &["statevector"], Vocabulary::RUN)?;
    let path = a.first_positional("<file.aria>")?;
    let name = a.opt("circuit").ok_or("run requires --circuit NAME")?;
    let ints = parse_kv_i64(a.all("int").into_iter())?;
    let binds = parse_kv_f64(a.all("bind").into_iter())?;
    let backend_str = a.opt("backend").unwrap_or("sim");
    let shots: u32 = a
        .opt("shots")
        .map(|s| s.parse().map_err(|_| format!("bad --shots '{s}'")))
        .transpose()?
        .unwrap_or(1024);
    let seed: Option<u64> = a
        .opt("seed")
        .map(|s| s.parse().map_err(|_| format!("bad --seed '{s}'")))
        .transpose()?;
    // Opt-in fail-loud: exit non-zero when an MPS run's accumulated discarded
    // weight exceeds this. Default (absent) preserves exit-code semantics for
    // every existing backend/flag combination.
    let strict_truncation = apply_strict_truncation(&a)?;

    // Per-gate noise model (trajectory simulation). Parsed here so a bad spec
    // or a channel that can't be applied fails loudly — `--noise` must never be
    // silently discarded (that's the exact bug this path fixes).
    let noise = a
        .opt("noise")
        .map(aria_runtime::parse_noise_model)
        .transpose()?;

    let src = read_source(path)?;
    let circuit = instantiate(&src, name, &ints)?;

    // Remote omega-server backend (counts only).
    if backend_str == "remote" {
        if noise.is_some() {
            return Err(
                "--noise is not supported with --backend remote; run the noisy \
                 trajectory simulation locally with --backend sim"
                    .into(),
            );
        }
        #[cfg(feature = "remote")]
        {
            let url = a.opt("url").ok_or("--backend remote requires --url URL")?;
            let remote = aria_runtime::Remote {
                url: url.to_string(),
                token: a.opt("token").map(str::to_string),
            };
            let res = aria_runtime::run_counts_remote(&circuit, &binds, shots, seed, &remote)?;
            print_counts(res, counts_width(&circuit, &binds));
            return Ok(());
        }
        #[cfg(not(feature = "remote"))]
        return Err(
            "aria was built without remote support; rebuild with `--features remote`".into(),
        );
    }

    let sel = BackendSel::parse(backend_str)?;

    if let Some(obs) = a.opt("expectation") {
        // Noisy expectation is exact only on the Pauli-propagation backend
        // (Heisenberg adjoint); route there. sim/mps expectations are analytic
        // and noiseless, so `expectation_noisy` rejects them with guidance.
        if let Some(model) = &noise {
            let val = aria_runtime::expectation_noisy(&circuit, obs, &binds, sel, model)?;
            println!("<{obs}> = {val:.12}");
            return Ok(());
        }
        // Pauli-propagation truncation knobs (PauliPropagation.jl's three axes).
        // When any is set, use the truncated engine and also print the certified
        // dropped-mass error budget. Without them the exact engine is used and
        // the output format is unchanged.
        let truncate = a
            .opt("truncate")
            .map(|s| {
                s.parse::<f64>()
                    .map_err(|_| format!("bad --truncate '{s}'"))
            })
            .transpose()?;
        let max_weight = a
            .opt("max-weight")
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|_| format!("bad --max-weight '{s}'"))
            })
            .transpose()?;
        let max_freq = a
            .opt("max-freq")
            .map(|s| {
                s.parse::<u32>()
                    .map_err(|_| format!("bad --max-freq '{s}'"))
            })
            .transpose()?;
        if truncate.is_some() || max_weight.is_some() || max_freq.is_some() {
            if sel != BackendSel::PauliProp {
                return Err(
                    "--truncate/--max-weight/--max-freq require --backend pauliprop".into(),
                );
            }
            let trunc = aria_runtime::PauliPropTruncation {
                coeff_min: truncate.unwrap_or(0.0),
                max_weight,
                max_freq,
            };
            let (val, budget) = aria_runtime::expectation_pauliprop(&circuit, obs, &binds, trunc)?;
            println!("<{obs}> = {val:.12}  (dropped-mass budget {budget:.3e})");
            return Ok(());
        }
        let val = expectation(&circuit, obs, &binds, sel)?;
        println!("<{obs}> = {val:.12}");
        return report_mps_truncation(strict_truncation);
    }

    if a.has("statevector") {
        if noise.is_some() {
            return Err(
                "--noise applies to sampled counts, not --statevector (a single exact \
                 state vector can't carry a stochastic noise channel; drop --statevector \
                 and use --shots N)"
                    .into(),
            );
        }
        let sv = statevector(&circuit, &binds, sel)?;
        for (i, amp) in sv.iter().enumerate() {
            if amp.norm() > 1e-12 {
                println!(
                    "|{i:0width$b}>  {re:+.6}{im:+.6}i",
                    width = circuit.n_qubits().max(1),
                    re = amp.re,
                    im = amp.im
                );
            }
        }
        return report_mps_truncation(strict_truncation);
    }

    let res = match &noise {
        Some(model) => aria_runtime::run_counts_noisy(&circuit, &binds, shots, seed, sel, model)?,
        None => run_counts(&circuit, &binds, shots, seed, sel)?,
    };
    print_counts(res, counts_width(&circuit, &binds));
    report_mps_truncation(strict_truncation)
}

/// Report the MPS truncation certificate of the just-completed run to stderr
/// (only when it actually truncated), and fail loudly under
/// `--strict-truncation <eps>`. A no-op for non-MPS backends (no stats were
/// captured). stderr-only + opt-in, so stdout and default exit codes are
/// unchanged for every existing invocation.
fn report_mps_truncation(strict: Option<f64>) -> Result<(), String> {
    if let Some(eps) = strict {
        if eps.is_nan() || eps < 0.0 {
            return Err(format!("--strict-truncation must be ≥ 0 (got {eps})"));
        }
    }
    let Some(stats) = aria_runtime::take_last_mps_stats() else {
        return Ok(());
    };
    // Only report REAL truncation. A rounding-level tail (σ ≤ 1e-14 dropped by
    // an otherwise-exact split) leaves a ~1e-28 weight that isn't worth a line;
    // 1e-12 is comfortably above that floor and below any meaningful loss.
    if stats.discarded_weight > 1e-12 {
        eprintln!(
            "mps: discarded_weight={:.3e} max_bond_reached={}",
            stats.discarded_weight, stats.max_bond_reached
        );
    }
    // `--strict-truncation` OVERRIDES the default; absent, the backend's default
    // ceiling still applies. Both directions matter: a user who asks to accept a
    // large discard must be able to, and a user who asks for nothing must not
    // silently receive a state the truncation destroyed.
    // No check here. The BACKEND enforces the ceiling — see
    // `aria_runtime::run::set_mps_discard_ceiling`, called above — so a run that
    // reaches this point is already within it. Re-checking would be a second
    // policy that could disagree with the first.
    //
    // This function now only REPORTS the certificate, which is what it is for.
    let _ = strict;
    Ok(())
}

/// Read a headered-or-not numeric CSV into rows of f64. A leading line whose
/// first cell is non-numeric is treated as a header and skipped.
fn read_numeric_csv(path: &str) -> Result<Vec<Vec<f64>>, String> {
    let text = read_source(path)?;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split(',').map(|c| c.trim()).collect();
        // Skip a header row (first line, first cell not a number).
        if lineno == 0 && cells[0].parse::<f64>().is_err() {
            continue;
        }
        let row: Vec<f64> = cells
            .iter()
            .map(|c| {
                c.parse::<f64>()
                    .map_err(|_| format!("{path}:{}: '{c}' is not a number", lineno + 1))
            })
            .collect::<Result<_, _>>()?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("{path}: no numeric rows"));
    }
    Ok(rows)
}

fn cmd_train(raw: &[String]) -> Result<(), String> {
    let a = parse_args(raw, &[], Vocabulary::TRAIN)?;
    let path = a.first_positional("<file.aria>")?;
    let name = a.opt("circuit").ok_or("train requires --circuit NAME")?;
    let ints = parse_kv_i64(a.all("int").into_iter())?;
    apply_strict_truncation(&a)?;
    let sel = BackendSel::parse(a.opt("backend").unwrap_or("sim"))?;

    // Supervised dataset training: `aria train f.aria --circuit C --data X.csv
    // --labels y.csv --observable Z0 [--loss bce] [--feature-prefix x]`.
    if a.opt("data").is_some() {
        return cmd_train_supervised(&a, path, name, &ints, sel);
    }

    let obs = a
        .opt("observable")
        .ok_or("train requires --observable OBS (e.g. \"Z0\" or a Pauli sum)")?;

    let mut cfg = TrainConfig::default();
    if let Some(v) = a.opt("steps") {
        cfg.steps = v.parse().map_err(|_| format!("bad --steps '{v}'"))?;
    }
    if let Some(v) = a.opt("lr") {
        cfg.lr = v.parse().map_err(|_| format!("bad --lr '{v}'"))?;
    }
    if let Some(v) = a.opt("seed") {
        cfg.seed = v.parse().map_err(|_| format!("bad --seed '{v}'"))?;
    }
    if let Some(v) = a.opt("init-scale") {
        cfg.init_scale = v.parse().map_err(|_| format!("bad --init-scale '{v}'"))?;
    }
    // Layer-wise training support (arXiv:2606.03517):
    //   --freeze "theta_0,theta_1"     exclude symbols from updates
    //   --set "theta_0=0.42"           pin a symbol's initial value
    //   --opt adam                     Adam instead of plain GD
    //   --grad parallel                parallel commuting-block shifts
    for v in a.all("freeze") {
        cfg.frozen
            .extend(v.split(',').map(|s| s.trim().to_string()));
    }
    for v in a.all("set") {
        let (k, val) = v
            .split_once('=')
            .ok_or_else(|| format!("bad --set '{v}' (want name=value)"))?;
        let val: f64 = val
            .trim()
            .parse()
            .map_err(|_| format!("bad --set value in '{v}'"))?;
        cfg.init.insert(k.trim().to_string(), val);
    }
    if let Some(v) = a.opt("opt") {
        cfg.optimizer = match v {
            "gd" => aria_runtime::Optimizer::Gd,
            "adam" => aria_runtime::Optimizer::adam(),
            other => return Err(format!("unknown --opt '{other}' (gd | adam)")),
        };
    }
    if let Some(v) = a.opt("grad") {
        cfg.grad_method = match v {
            "shift" => omega_core::gradient::GradMethod::ParameterShift,
            "parallel" => omega_core::gradient::GradMethod::ParallelParameterShift,
            other => return Err(format!("unknown --grad '{other}' (shift | parallel)")),
        };
    }

    let src = read_source(path)?;
    let circuit = instantiate(&src, name, &ints)?;
    let result = train_expectation(&circuit, obs, &cfg, sel)?;

    let initial = result
        .history
        .first()
        .copied()
        .unwrap_or(result.final_value);
    println!("observable : {obs}");
    println!("backend    : {}", sel.name());
    println!(
        "steps      : {}  (lr {}, seed {})",
        cfg.steps, cfg.lr, cfg.seed
    );
    println!("initial <O>: {initial:.12}");
    println!("final   <O>: {:.12}", result.final_value);
    println!("improvement: {:.12}", initial - result.final_value);
    let mut params: Vec<(&String, &f64)> = result.params.iter().collect();
    params.sort_by(|x, y| x.0.cmp(y.0));
    println!("trained parameters:");
    for (k, v) in params {
        println!("  {k} = {v:.6}");
    }
    Ok(())
}

/// Supervised dataset-training path (`aria train --data`). Reads a feature
/// matrix and a labels column, fits the circuit's weights with the chosen
/// loss, and prints loss/AUC plus the trained weights.
fn cmd_train_supervised(
    a: &Args,
    path: &str,
    name: &str,
    ints: &[(String, i64)],
    sel: BackendSel,
) -> Result<(), String> {
    use aria_runtime::{train_supervised, Loss, Optimizer, SupervisedConfig};

    let data_path = a.opt("data").expect("checked by caller");
    let obs = a
        .opt("observable")
        .ok_or("supervised train requires --observable OBS (the readout, e.g. \"Z0\")")?;

    let train_x = read_numeric_csv(data_path)?;
    // Labels: a separate --labels file (column 0), else the LAST column of
    // --data. Both are common; be explicit about which was used.
    let (train_x, train_y): (Vec<Vec<f64>>, Vec<f64>) = match a.opt("labels") {
        Some(lp) => {
            let ly = read_numeric_csv(lp)?;
            if ly.len() != train_x.len() {
                return Err(format!(
                    "--data has {} rows but --labels has {}",
                    train_x.len(),
                    ly.len()
                ));
            }
            let y = ly.iter().map(|r| r[0]).collect();
            (train_x, y)
        }
        None => {
            let mut xs = train_x;
            let mut ys = Vec::with_capacity(xs.len());
            for row in &mut xs {
                ys.push(row.pop().ok_or("empty --data row")?);
            }
            eprintln!("note: no --labels file; using the last column of --data as the label");
            (xs, ys)
        }
    };

    let mut cfg = SupervisedConfig {
        loss: match a.opt("loss") {
            Some(s) => Loss::parse(s)?,
            None => Loss::Mse,
        },
        ..Default::default()
    };
    if let Some(v) = a.opt("feature-prefix") {
        cfg.feature_prefix = v.to_string();
    }
    if let Some(v) = a.opt("steps") {
        cfg.steps = v.parse().map_err(|_| format!("bad --steps '{v}'"))?;
    }
    if let Some(v) = a.opt("lr") {
        cfg.lr = v.parse().map_err(|_| format!("bad --lr '{v}'"))?;
    }
    if let Some(v) = a.opt("seed") {
        cfg.seed = v.parse().map_err(|_| format!("bad --seed '{v}'"))?;
    }
    if let Some(v) = a.opt("init-scale") {
        cfg.init_scale = v.parse().map_err(|_| format!("bad --init-scale '{v}'"))?;
    }
    for v in a.all("freeze") {
        cfg.frozen
            .extend(v.split(',').map(|s| s.trim().to_string()));
    }
    if let Some(v) = a.opt("opt") {
        cfg.optimizer = match v {
            "gd" => Optimizer::Gd,
            "adam" => Optimizer::adam(),
            other => return Err(format!("unknown --opt '{other}' (gd | adam)")),
        };
    }
    if let Some(v) = a.opt("grad") {
        cfg.grad_method = match v {
            "adjoint" => omega_core::gradient::GradMethod::Adjoint,
            "shift" => omega_core::gradient::GradMethod::ParameterShift,
            "parallel" => omega_core::gradient::GradMethod::ParallelParameterShift,
            other => {
                return Err(format!(
                    "unknown --grad '{other}' (adjoint | shift | parallel)"
                ))
            }
        };
    }

    let src = read_source(path)?;
    let circuit = instantiate(&src, name, ints)?;
    let result = train_supervised(&circuit, &train_x, &train_y, obs, &cfg, sel)?;

    let loss_name = match cfg.loss {
        Loss::Mse => "mse",
        Loss::Bce => "bce",
    };
    println!(
        "dataset    : {} rows × {} features",
        train_x.len(),
        train_x[0].len()
    );
    println!(
        "readout    : {obs}   loss: {loss_name}   backend: {}",
        sel.name()
    );
    println!(
        "steps      : {}  (lr {}, seed {}, feature-prefix '{}')",
        cfg.steps, cfg.lr, cfg.seed, cfg.feature_prefix
    );
    let initial = result
        .loss_history
        .first()
        .copied()
        .unwrap_or(result.final_loss);
    println!("initial loss: {initial:.6}");
    println!("final   loss: {:.6}", result.final_loss);
    println!("train AUC   : {:.4}", result.final_auc);
    println!(
        "affine head : (s = {:.4}, b = {:.4})",
        result.head.0, result.head.1
    );
    let mut weights: Vec<(&String, &f64)> = result.weights.iter().collect();
    weights.sort_by(|x, y| x.0.cmp(y.0));
    println!("trained weights:");
    for (k, v) in weights {
        println!("  {k} = {v:.6}");
    }

    // Optionally save a self-contained trained-model JSON.
    if let Some(model_path) = a.opt("save-model") {
        let model = aria_runtime::TrainedModel::from_result(
            src,
            name.to_string(),
            ints.to_vec(),
            cfg.feature_prefix.clone(),
            obs.to_string(),
            cfg.loss,
            &result,
            cfg.seed,
            cfg.steps,
        );
        model.save(model_path)?;
        println!("saved model : {model_path}");
    }
    Ok(())
}

/// `aria predict <model.json> --data X.csv [--out scores.csv] [--backend B]` —
/// score a feature matrix with a saved trained model.
fn cmd_predict(raw: &[String]) -> Result<(), String> {
    let a = parse_args(raw, &[], Vocabulary::PREDICT)?;
    let model_path = a.first_positional("<model.json>")?;
    let data_path = a
        .opt("data")
        .ok_or("predict requires --data X.csv (the feature matrix)")?;
    apply_strict_truncation(&a)?;
    let sel = BackendSel::parse(a.opt("backend").unwrap_or("sim"))?;

    let model = aria_runtime::TrainedModel::load(model_path)?;
    let x = read_numeric_csv(data_path)?;
    let scores = model.predict(&x, sel)?;

    match a.opt("out") {
        Some(out) => {
            let mut s = String::from("score\n");
            for v in &scores {
                s.push_str(&format!("{v:.10}\n"));
            }
            std::fs::write(out, s).map_err(|e| format!("write {out}: {e}"))?;
            println!("wrote {} scores to {out}", scores.len());
        }
        None => {
            for v in &scores {
                println!("{v:.10}");
            }
        }
    }
    Ok(())
}

fn cmd_export(raw: &[String]) -> Result<(), String> {
    let a = parse_args(raw, &["qasm", "qasm3", "json", "lean", "gate-model"], Vocabulary::EXPORT)?;
    let path = a.first_positional("<file.aria>")?;
    let name = a.opt("circuit").ok_or("export requires --circuit NAME")?;
    let ints = parse_kv_i64(a.all("int").into_iter())?;
    let src = read_source(path)?;
    let circuit = instantiate(&src, name, &ints)?;
    // Mutually-exclusive output formats — reject combos so a requested format is
    // never silently dropped (e.g. `--lean --gate-model`).
    let n_fmt = ["qasm", "qasm3", "json", "lean", "gate-model"]
        .iter()
        .filter(|f| a.has(f))
        .count();
    if n_fmt > 1 {
        return Err(
            "export takes exactly one of --qasm | --qasm3 | --json | --lean | --gate-model".into(),
        );
    }
    let out = if a.has("qasm") {
        aria_core::ast::to_qasm(&circuit)?
    } else if a.has("qasm3") {
        aria_core::ast::to_qasm3(&circuit)
    } else if a.has("json") {
        aria_core::ast::to_json(&circuit)?
    } else if a.has("lean") {
        aria_core::ast::to_lean4(&circuit)
    } else if a.has("gate-model") {
        // Sorry-free gate-model obligation closed by a proved QuantumProofs
        // theorem — only for recognized circuits (e.g. Bell). Builds against
        // proofs/lean4 (which ships BellPrep). Unrecognized ⇒ honest error.
        aria_core::ast::render_gate_model_spec(&circuit, name).ok_or_else(|| {
            format!(
                "--gate-model: circuit '{name}' is not a recognized proven circuit \
                     (today: Bell — H@0;CX0→1, or GHZ — H@0;CX0→1;CX1→2); \
                     use --lean for a sorry-stub export"
            )
        })?
    } else {
        return Err(
            "export requires one of --qasm | --qasm3 | --json | --lean | --gate-model".into(),
        );
    };
    print!("{out}");
    Ok(())
}

// ---------------------------------------------------------------------------
// `aria tune` — hyper-parameter search over a circuit's compile-time ints and
// training knobs, driven by `aria-tune`.
// ---------------------------------------------------------------------------

/// Parse a `--space` spec into an `aria_tune::Space`.
///
/// Grammar (comma-separated dimensions):
///   `n=4..8:2`            int from 4 to 8 step 2
///   `L=1..3`              int, step 1
///   `lr=log:1e-3..3e-1`   log-spaced float, default resolution
///   `lr=lin:0..1:5`       linearly spaced float, 5 grid points
///   `opt=gd|adam`         categorical
///
/// Every dimension is a grid, which is what lets one TPE cover them all —
/// see `aria_tune::space`.
fn parse_space(spec: &str) -> Result<aria_tune::Space, String> {
    const DEFAULT_RES: usize = 8;
    let mut space = aria_tune::Space::new();
    for dim in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, body) = dim
            .split_once('=')
            .ok_or_else(|| format!("bad --space dimension '{dim}' (want name=spec)"))?;
        let (name, body) = (name.trim(), body.trim());
        let num = |s: &str| -> Result<f64, String> {
            s.trim()
                .parse::<f64>()
                .map_err(|_| format!("bad number '{s}' in --space dimension '{dim}'"))
        };
        // Integer bounds come in as f64 and are cast to i64. A cast of an
        // out-of-range value *saturates* (1e20 → i64::MAX), silently building a
        // different grid than asked; reject it here instead.
        let int_bound = |s: &str| -> Result<i64, String> {
            let v = num(s)?;
            if !v.is_finite() || v.abs() >= 9.2e18 {
                return Err(format!(
                    "integer bound '{s}' in --space dimension '{dim}' is out of range"
                ));
            }
            Ok(v as i64)
        };

        if let Some(rest) = body
            .strip_prefix("log:")
            .or_else(|| body.strip_prefix("lin:"))
        {
            let is_log = body.starts_with("log:");
            let mut parts = rest.split(':');
            let range = parts
                .next()
                .ok_or_else(|| format!("bad --space dimension '{dim}'"))?;
            let res: usize = match parts.next() {
                Some(r) => r
                    .trim()
                    .parse()
                    .map_err(|_| format!("bad resolution in '{dim}'"))?,
                None => DEFAULT_RES,
            };
            let (lo, hi) = range
                .split_once("..")
                .ok_or_else(|| format!("bad range in '{dim}' (want lo..hi)"))?;
            let (lo, hi) = (num(lo)?, num(hi)?);
            space = space
                .try_add(
                    name,
                    if is_log {
                        aria_tune::Param::LogFloat { lo, hi, res }
                    } else {
                        aria_tune::Param::Float { lo, hi, res }
                    },
                )
                .map_err(|e| format!("--space dimension '{dim}': {e}"))?;
        } else if body.contains("..") {
            let (lo, rest) = body
                .split_once("..")
                .ok_or_else(|| format!("bad range in '{dim}'"))?;
            let (hi, step) = match rest.split_once(':') {
                Some((h, s)) => (h, int_bound(s)?),
                None => (rest, 1),
            };
            space = space
                .try_add(
                    name,
                    aria_tune::Param::Int {
                        lo: int_bound(lo)?,
                        hi: int_bound(hi)?,
                        step,
                    },
                )
                .map_err(|e| format!("--space dimension '{dim}': {e}"))?;
        } else {
            let choices: Vec<String> = body.split('|').map(|s| s.trim().to_string()).collect();
            space = space
                .try_add(name, aria_tune::Param::Categorical(choices))
                .map_err(|e| format!("--space dimension '{dim}': {e}"))?;
        }
    }
    if space.is_empty() {
        return Err("--space defined no dimensions".into());
    }
    Ok(space)
}

fn cmd_tune(raw: &[String]) -> Result<(), String> {
    use aria_runtime::{train_supervised, Loss, Optimizer, SupervisedConfig};
    use aria_tune::{Direction, MedianPruner, NoPruner, RandomSampler, Sampler, Study, TpeSampler};

    let a = parse_args(raw, &[], Vocabulary::TUNE)?;
    let path = a.first_positional("<file.aria>")?;
    let name = a.opt("circuit").ok_or("tune requires --circuit NAME")?;
    let spec = a
        .opt("space")
        .ok_or("tune requires --space \"n=4..8:2,L=1..3,lr=log:1e-3..3e-1,opt=gd|adam\"")?;
    let obs = a
        .opt("observable")
        .ok_or("tune requires --observable OBS (e.g. \"Z0\")")?;
    let data_path = a.opt("data").ok_or("tune requires --data X.csv")?;
    apply_strict_truncation(&a)?;
    let sel = BackendSel::parse(a.opt("backend").unwrap_or("sim"))?;
    let space = parse_space(spec)?;

    let n_trials: usize = match a.opt("trials") {
        Some(v) => v.parse().map_err(|_| format!("bad --trials '{v}'"))?,
        None => 20,
    };
    let seed: u64 = match a.opt("seed") {
        Some(v) => v.parse().map_err(|_| format!("bad --seed '{v}'"))?,
        None => 1,
    };
    let steps: usize = match a.opt("steps") {
        Some(v) => v.parse().map_err(|_| format!("bad --steps '{v}'"))?,
        None => 40,
    };
    let sampler: Box<dyn Sampler> = match a.opt("sampler").unwrap_or("tpe") {
        "tpe" => Box::new(TpeSampler::new(seed)),
        "random" => Box::new(RandomSampler::new(seed)),
        "grid" => Box::new(aria_tune::GridSampler::new()),
        other => return Err(format!("unknown --sampler '{other}' (tpe | random | grid)")),
    };
    let pruner: Box<dyn aria_tune::Pruner> = match a.opt("pruner").unwrap_or("median") {
        "median" => Box::new(MedianPruner::default()),
        "none" => Box::new(NoPruner),
        other => return Err(format!("unknown --pruner '{other}' (median | none)")),
    };

    // Dataset: same convention as `aria train --data`.
    let rows = read_numeric_csv(data_path)?;
    let (train_x, train_y): (Vec<Vec<f64>>, Vec<f64>) = match a.opt("labels") {
        Some(lp) => {
            let ly = read_numeric_csv(lp)?;
            (
                train_x_check(rows, ly.len())?,
                ly.iter().map(|r| r[0]).collect(),
            )
        }
        None => {
            let mut xs = rows;
            let mut ys = Vec::with_capacity(xs.len());
            for row in &mut xs {
                ys.push(row.pop().ok_or("empty --data row")?);
            }
            (xs, ys)
        }
    };
    if train_x.is_empty() || train_x[0].is_empty() {
        return Err("--data has no usable feature columns".into());
    }
    let src = read_source(path)?;
    let base_ints = parse_kv_i64(a.all("int").into_iter())?;

    let feat_prefix = format!("{}_", a.opt("feature-prefix").unwrap_or("x"));
    let mut narrowed = false;
    let mut study = Study::new(space, Direction::Maximize)
        .with_sampler(sampler)
        .with_pruner(pruner);

    for _ in 0..n_trials {
        let t = study.ask();
        // Any dimension whose name matches a circuit int becomes a
        // compile-time parameter; the rest are training knobs.
        let mut ints: Vec<(String, i64)> = base_ints.clone();
        for (dim, _) in t.params.iter() {
            if let Some(v) = t.int(dim) {
                if dim != "steps" {
                    ints.retain(|(k, _)| k != dim);
                    ints.push((dim.clone(), v));
                }
            }
        }
        // An infeasible shape (won't instantiate, won't lower, or needs more
        // features than the data has) is a dead trial, not a fatal error: mark
        // it and move on. A non-finite score records it as `Pruned`, which the
        // sampler's history excludes — so the point isn't actively steered
        // toward, though (unlike a completed low score) it may be re-proposed.
        let circuit = match instantiate(&src, name, &ints) {
            Ok(c) => c,
            Err(_) => {
                study.tell(t.id, f64::NEG_INFINITY);
                continue;
            }
        };
        // Tuning the wire count changes how many features the circuit
        // accepts, but --data has one fixed width. Slice each trial's data to
        // the feature symbols the instantiated circuit actually declares.
        let want = match aria_runtime::lower(&circuit) {
            Ok(low) => low
                .symbol_ids
                .keys()
                .filter(|k| {
                    k.strip_prefix(&feat_prefix)
                        .is_some_and(|r| r.parse::<usize>().is_ok())
                })
                .count(),
            Err(_) => {
                study.tell(t.id, f64::NEG_INFINITY);
                continue;
            }
        };
        if want > train_x[0].len() {
            // Same class as an un-buildable shape: this trial's wire count
            // needs more feature columns than --data carries. Skip it (dead
            // trial) instead of aborting the whole study.
            study.tell(t.id, f64::NEG_INFINITY);
            continue;
        }
        let trial_x: Vec<Vec<f64>> = if want == train_x[0].len() {
            train_x.clone()
        } else {
            if !narrowed {
                eprintln!(
                    "note: slicing --data to each trial's feature count \
                     (the circuit declares '{feat_prefix}0..' per wire)"
                );
                narrowed = true;
            }
            train_x.iter().map(|r| r[..want].to_vec()).collect()
        };

        let loss = match a.opt("loss") {
            Some(s) => Loss::parse(s)?,
            None => Loss::Bce,
        };
        let use_adam = matches!(t.cat("opt").or_else(|| t.cat("optimizer")), Some("adam"));
        let lr = t.float("lr").unwrap_or(0.1);

        // Train in rungs so the pruner has intermediate values to judge and can
        // stop a hopeless trial early. Each rung warm-starts from the previous
        // rung's weights (`init_weights`), so the split reproduces one long run
        // rather than N cold restarts. Without this the pruner is inert — a
        // single full train exposes no rung to `should_prune`.
        const RUNGS: usize = 5;
        let n_rungs = steps.clamp(1, RUNGS);
        let mut warm: Option<std::collections::HashMap<String, f64>> = None;
        let mut last_auc = f64::NEG_INFINITY;
        for rung in 0..n_rungs {
            // Even split of `steps` across the rungs (remainder in the early
            // rungs) so the total step budget is exactly `steps`.
            let rung_steps = steps / n_rungs + usize::from(rung < steps % n_rungs);
            if rung_steps == 0 {
                continue;
            }
            let cfg = SupervisedConfig {
                steps: rung_steps,
                lr,
                seed,
                loss,
                optimizer: if use_adam {
                    Optimizer::adam()
                } else {
                    Optimizer::Gd
                },
                init_weights: warm.clone(),
                ..Default::default()
            };
            let r = train_supervised(&circuit, &trial_x, &train_y, obs, &cfg, sel)?;
            warm = Some(r.weights.clone());
            last_auc = r.final_auc;
            study.report(
                t.id,
                rung,
                r.final_auc,
                &[("final_loss".to_string(), r.final_loss)],
            );
            if study.should_prune(t.id) {
                break;
            }
        }
        // A pruned trial stays pruned — `tell` only finalizes a still-running one.
        study.tell(t.id, last_auc);
    }

    let best = study.best().ok_or("no trial completed")?;
    println!("sampler    : {}", study.sampler_name());
    println!("pruner     : {}", study.pruner_name());
    println!("trials     : {n_trials}");
    println!("pruned     : {}", study.n_pruned());
    println!("best_score : {:.6}", best.state.score().unwrap_or(f64::NAN));
    println!("best_params:");
    for (k, v) in &best.params {
        println!("  {k} = {v}");
    }
    if let Some(out) = a.opt("csv") {
        std::fs::write(out, study.to_csv()).map_err(|e| format!("write {out}: {e}"))?;
        println!("csv        : {out}");
    }
    Ok(())
}

/// Row-count guard shared by the `--labels` path.
fn train_x_check(rows: Vec<Vec<f64>>, n_labels: usize) -> Result<Vec<Vec<f64>>, String> {
    if rows.len() != n_labels {
        return Err(format!(
            "--data has {} rows but --labels has {n_labels}",
            rows.len()
        ));
    }
    Ok(rows)
}

/// `aria import <file.qasm> [--name NAME]` — parse an OpenQASM 2.0 file (the
/// fail-loud importer) and print equivalent `.aria` source to stdout.
fn cmd_import(raw: &[String]) -> Result<(), String> {
    let a = parse_args(raw, &[], Vocabulary::IMPORT)?;
    let path = a.first_positional("<file.qasm>")?;
    let name = a.opt("name").unwrap_or("Imported");
    let qasm = read_source(path)?;
    let circuit = aria_core::ast::from_qasm(&qasm)?;
    let out = aria_core::ast::to_aria_source(&circuit, name);
    print!("{out}");
    Ok(())
}
