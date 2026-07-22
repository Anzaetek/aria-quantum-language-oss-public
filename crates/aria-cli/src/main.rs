// SPDX-License-Identifier: Apache-2.0
//! `aria` — command-line front end for the Aria Quantum Language.
//!
//! Subcommands:
//! - `aria list   <file.aria>`                         — list circuit templates
//! - `aria parse  <file.aria> [--circuit NAME] [--int k=v]...`
//! - `aria run    <file.aria> --circuit NAME [--int k=v]... [--bind s=v]...
//!   [--shots N] [--seed S] [--backend sim|mps|gpu|tch|pauliprop|remote] [--statevector] [--expectation OBS]`
//! - `aria export <file.aria> --circuit NAME (--qasm|--json|--lean|--gate-model) [--int k=v]...`

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
    let result = match cmd {
        "list" => cmd_list(rest),
        "parse" => cmd_parse(rest),
        "run" => cmd_run(rest),
        "train" => cmd_train(rest),
        "export" => cmd_export(rest),
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
         [--shots N] [--seed S] [--backend sim|mps|gpu|tch|pauliprop|remote] [--statevector] [--expectation OBS]\n              \
         [--noise JSON  (sampled counts on --backend sim, e.g. '{{\"readout_flip\":0.02,\"amplitude_damping\":5e-4}}')]\n              \
         (pauliprop computes --expectation only; truncate deep circuits with\n              \
          [--truncate C] [--max-weight W] [--max-freq F])\n              \
         [--url URL --token TOK   (for --backend remote)]\n  \
         aria train  <file.aria> --circuit NAME --observable OBS [--int k=v]...\n              \
         [--steps N] [--lr R] [--seed S] [--init-scale S]\n  \
         aria export <file.aria> --circuit NAME (--qasm | --json | --lean | --gate-model) [--int k=v]...\n"
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

fn parse_args(raw: &[String], bool_flags: &[&str]) -> Result<Args, String> {
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
    let a = parse_args(raw, &[])?;
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
    let a = parse_args(raw, &[])?;
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

fn print_counts(res: ExecResult, n_qubits: usize) {
    match res {
        ExecResult::Counts(counts) => {
            let mut rows: Vec<(u64, u32)> = counts.into_iter().collect();
            rows.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
            let w = n_qubits.max(1);
            let total: u64 = rows.iter().map(|(_, c)| *c as u64).sum::<u64>().max(1);
            for (state, count) in rows {
                let p = count as f64 / total as f64;
                println!("|{state:0w$b}>  {count}  ({p:.4})");
            }
        }
        other => println!("{other:?}"),
    }
}

fn cmd_run(raw: &[String]) -> Result<(), String> {
    let a = parse_args(raw, &["statevector"])?;
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
        return Ok(());
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
        return Ok(());
    }

    let res = match &noise {
        Some(model) => aria_runtime::run_counts_noisy(&circuit, &binds, shots, seed, sel, model)?,
        None => run_counts(&circuit, &binds, shots, seed, sel)?,
    };
    print_counts(res, counts_width(&circuit, &binds));
    Ok(())
}

fn cmd_train(raw: &[String]) -> Result<(), String> {
    let a = parse_args(raw, &[])?;
    let path = a.first_positional("<file.aria>")?;
    let name = a.opt("circuit").ok_or("train requires --circuit NAME")?;
    let obs = a
        .opt("observable")
        .ok_or("train requires --observable OBS (e.g. \"Z0\" or a Pauli sum)")?;
    let ints = parse_kv_i64(a.all("int").into_iter())?;
    let sel = BackendSel::parse(a.opt("backend").unwrap_or("sim"))?;

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

fn cmd_export(raw: &[String]) -> Result<(), String> {
    let a = parse_args(raw, &["qasm", "json", "lean", "gate-model"])?;
    let path = a.first_positional("<file.aria>")?;
    let name = a.opt("circuit").ok_or("export requires --circuit NAME")?;
    let ints = parse_kv_i64(a.all("int").into_iter())?;
    let src = read_source(path)?;
    let circuit = instantiate(&src, name, &ints)?;
    // Mutually-exclusive output formats — reject combos so a requested format is
    // never silently dropped (e.g. `--lean --gate-model`).
    let n_fmt = ["qasm", "json", "lean", "gate-model"]
        .iter()
        .filter(|f| a.has(f))
        .count();
    if n_fmt > 1 {
        return Err("export takes exactly one of --qasm | --json | --lean | --gate-model".into());
    }
    let out = if a.has("qasm") {
        aria_core::ast::to_qasm(&circuit)
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
        return Err("export requires one of --qasm | --json | --lean | --gate-model".into());
    };
    print!("{out}");
    Ok(())
}
