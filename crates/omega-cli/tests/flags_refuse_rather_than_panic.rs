// SPDX-License-Identifier: Apache-2.0
//! **A malformed command line must be refused, not panic.**
//!
//! Every flag in every one of `omega-run`'s three argument loops was written
//! `i += 1; args[i]`, which indexes past the end when the flag is last on the
//! line. Measured before the fix:
//!
//! ```text
//!   $ omega-run c.qasm --backend mps:8 --shots 20 --noise
//!   thread 'main' panicked at crates/omega-cli/src/main.rs:349:39:
//!   index out of bounds: the len is 9 but the index is 9
//!   note: run with `RUST_BACKTRACE=1` ...
//!   $ echo $?
//!   101
//! ```
//!
//! Value parsing had the same shape — `.expect("invalid shots number")` panics
//! on `--shots abc`, and names neither the flag nor what was passed.
//!
//! Exit status matters as much as the message: 101 is "this program has a bug",
//! 1 is "your input was wrong". A script cannot tell those apart from the text.

use std::path::PathBuf;
use std::process::{Command, Output};

fn bin() -> PathBuf {
    // The test binary lives in target/<profile>/deps/; the CLI is two up.
    let mut p = std::env::current_exe().expect("current_exe");
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("omega-run")
}

/// The shared QASM fixture, written **exactly once per process** to a
/// **process-unique** path.
///
/// Both of those matter, and neither was true before — this test was flaky at
/// roughly 1 run in 6:
///
/// ```text
///   a_well_formed_command_line_still_succeeds ... FAILED
///   Parse error: unknown circuit format: expected OPENQASM or OPTICQASM header
/// ```
///
/// The old version rewrote `temp_dir()/omega_flag_refusal_fixture.qasm` on
/// every call. All three tests call it, cargo runs them on separate threads,
/// and `fs::write` truncates before it writes — so one test could hand
/// `omega-run` a file another test had just emptied. The parse error was the
/// symptom; the fixture was momentarily zero bytes.
///
/// A fixed name in a shared temp dir is also racy ACROSS processes: two
/// concurrent `cargo test` runs (different target dirs, a second checkout, CI
/// beside a local run) collide on the same path. The pid in the name removes
/// that, and `OnceLock` removes the in-process race by making the write happen
/// once rather than per call.
fn circuit() -> PathBuf {
    static FIXTURE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let p = std::env::temp_dir()
                .join(format!("omega_flag_refusal_fixture_{}.qasm", std::process::id()));
            std::fs::write(
                &p,
                "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\n",
            )
            .expect("write fixture");
            p
        })
        .clone()
}

fn run(extra: &[&str]) -> Output {
    run_on(&circuit(), extra, &[])
}

/// `run` against an explicit circuit file, with extra environment variables.
fn run_on(circuit: &std::path::Path, extra: &[&str], env: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(bin());
    cmd.arg(circuit);
    cmd.args(extra);
    cmd.envs(env.iter().copied());
    cmd.output().expect("spawn omega-run")
}

/// A second fixture, process-unique like the first, whose body the caller
/// picks: the mid-circuit and photonic rows below need circuits the shared
/// Bell pair is not.
fn fixture(tag: &str, body: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "omega_flag_refusal_{tag}_{}.qasm",
        std::process::id()
    ));
    std::fs::write(&p, body).expect("write fixture");
    p
}

/// `h; measure; h; cx`: a measurement followed by other ops, so the
/// simulator must collapse and every run is one trajectory.
const MID_CIRCUIT_MEASURE: &str =
    "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\n\
                                   h q[0];\nmeasure q[0] -> c[0];\nh q[1];\ncx q[0], q[1];\n";

const PHOTONIC: &str = "OPTICQASM 1.0;\nphoton q[2];\nbs(0.5, 0.0) q[0], q[1];\n";

/// Every value-taking flag, given no value.
#[test]
fn a_flag_missing_its_value_is_refused_cleanly() {
    let flags = [
        "--shots",
        "--seed",
        "--backend",
        "--backend-dir",
        "--bridge",
        "--device",
        "--input",
        "--expectation",
        "--gradient",
        "--gradient-of-fn",
        "--score-fn-shots",
        "--params",
        "--method",
        "--noise",
    ];
    for f in flags {
        let out = run(&[f]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !stderr.contains("panicked"),
            "{f} with no value PANICKED:\n{stderr}"
        );
        assert_eq!(
            out.status.code(),
            Some(1),
            "{f} with no value exited {:?}; 1 means \"your input was wrong\", \
             101 means \"this program has a bug\". stderr:\n{stderr}",
            out.status.code()
        );
        assert!(
            stderr.contains(f),
            "the refusal for {f} does not name the flag, so the reader cannot \
             tell which one was wrong:\n{stderr}"
        );
    }
}

/// A value of the wrong shape names the flag AND what was passed.
#[test]
fn a_flag_with_an_unparseable_value_is_refused_cleanly() {
    for (flag, bad) in [
        ("--shots", "abc"),
        ("--seed", "not-a-seed"),
        ("--score-fn-shots", "1.5"),
        ("--params", "1,x,3"),
        ("--input", "1,two"),
        // Plugins load only when a --backend name is not compiled in, so a
        // wrong path used to pass silently on every builtin run.
        ("--backend-dir", "/nonexistent/omega-plugins"),
    ] {
        let out = run(&[flag, bad]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !stderr.contains("panicked"),
            "{flag} {bad} PANICKED:\n{stderr}"
        );
        assert_eq!(out.status.code(), Some(1), "{flag} {bad}: {stderr}");
        assert!(
            stderr.contains(flag),
            "{flag} {bad}: refusal does not name the flag:\n{stderr}"
        );
    }
}

/// **A flag that parses but no longer means what its help text says must be
/// refused, not reinterpreted** (plan §A6). `--max-weight`/`--max-freq` cut on
/// Pauli weight and split frequency — axes that do not exist for
/// `majoranaprop`, whose cut axis is Majorana monomial LENGTH. Silently
/// mapping one onto the other would produce numbers under a meaning the user
/// never chose. `main.rs` refuses with exit 2 and points at `--max-length`;
/// this pins both the status and the redirect, per flag.
#[test]
fn a_pauliprop_axis_flag_on_majoranaprop_is_refused_with_the_right_axis() {
    for (flag, val) in [("--max-weight", "4"), ("--max-freq", "2")] {
        // `--expectation Z0` keeps the command otherwise well-formed: without
        // it, the earlier "expectation-value backend" refusal (exit 1) fires
        // first and this test would pin the wrong diagnostic. The wrong-axis
        // flag must be the ONLY defect on the line.
        let out = run(&[
            "--backend",
            "majoranaprop",
            "--expectation",
            "Z0",
            flag,
            val,
        ]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !stderr.contains("panicked"),
            "majoranaprop {flag} PANICKED:\n{stderr}"
        );
        assert_eq!(
            out.status.code(),
            Some(2),
            "majoranaprop {flag} must refuse (exit 2), not reinterpret; stderr:\n{stderr}"
        );
        assert!(
            stderr.contains("--max-length"),
            "the refusal for {flag} must offer the axis that DOES exist \
             (--max-length):\n{stderr}"
        );
        assert!(
            stderr.contains("pauliprop"),
            "the refusal for {flag} must say whose flag it actually is:\n{stderr}"
        );
    }
}

/// **Guard the guard.** The fixture and binary must actually work, or every
/// assertion above passes because nothing ever ran.
#[test]
fn a_well_formed_command_line_still_succeeds() {
    let out = run(&["--backend", "mps:8", "--shots", "20", "--seed", "1"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a valid command line failed, so the refusal tests above prove nothing:\n{stderr}"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("11") || stdout.contains("00"),
        "expected Bell counts on stdout, got:\n{stdout}"
    );
}

/// **A flag outside its scope is refused, not dropped** (plan §A6, the CLI
/// arm). Every row was a well-formed line that exited 0 before the gate with
/// the named flag silently ignored — measured by running each flag against
/// each backend/mode and grepping both streams for its name. The refusal
/// must exit 2 (well-formed, meaningless — not 1, malformed) and name the
/// flag, so the reader knows which one to remove.
#[test]
fn a_flag_outside_its_scope_is_refused_and_named() {
    let rows: &[(&[&str], &str)] = &[
        // pauliprop/majoranaprop knobs on backends that never read them —
        // in sampling mode, where no gate existed at all, and in expectation.
        (
            &[
                "--backend",
                "statevector",
                "--shots",
                "10",
                "--truncate",
                "1e-3",
            ],
            "--truncate",
        ),
        (
            &[
                "--backend",
                "mps:8",
                "--shots",
                "10",
                "--max-terms",
                "100000",
            ],
            "--max-terms",
        ),
        (
            &[
                "--backend",
                "statevector",
                "--shots",
                "10",
                "--max-length",
                "4",
            ],
            "--max-length",
        ),
        (
            &["--backend", "pauli", "--shots", "10", "--max-weight", "4"],
            "--max-weight",
        ),
        (
            &["--expectation", "Z0", "--max-dropped-mass", "1.0"],
            "--max-dropped-mass",
        ),
        (
            &[
                "--backend",
                "mps:8",
                "--expectation",
                "Z0",
                "--max-freq",
                "2",
            ],
            "--max-freq",
        ),
        // photonics-only input on a gate backend.
        (&["--shots", "10", "--input", "1,0"], "--input"),
        // gradient-only flags with no gradient mode.
        (
            &["--shots", "10", "--method", "parameter-shift"],
            "--method",
        ),
        (
            &["--shots", "10", "--score-fn-shots", "10"],
            "--score-fn-shots",
        ),
        // a seed where nothing samples. The fixture has no mid-circuit
        // measurement, so its `--statevector` is analytic; with one, the seed
        // picks the trajectory and is IN scope — see
        // `a_seed_is_accepted_when_statevector_draws_a_seeded_trajectory`.
        (&["--expectation", "Z0", "--seed", "1"], "--seed"),
        (&["--statevector", "--seed", "1"], "--seed"),
        // two modes on one line: the later-checked one used to win silently.
        (&["--expectation", "Z0", "--shots", "10"], "--shots"),
        (&["--statevector", "--shots", "10"], "--statevector"),
        (&["--expectation", "Z0", "--gradient", "Z0"], "--gradient"),
        // values for parameters the circuit does not have.
        (&["--shots", "10", "--params", "0.1"], "--params"),
    ];
    for (args, flag) in rows {
        let out = run(args);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!stderr.contains("panicked"), "{args:?} PANICKED:\n{stderr}");
        assert_eq!(
            out.status.code(),
            Some(2),
            "{args:?}: {flag} is out of scope here and must refuse with exit 2 \
             (well-formed but meaningless); stderr:\n{stderr}"
        );
        assert!(
            stderr.contains(flag),
            "{args:?}: the refusal must name {flag} so the reader knows what to \
             remove:\n{stderr}"
        );
        assert!(
            String::from_utf8_lossy(&out.stdout).trim().is_empty(),
            "{args:?}: no result may be emitted alongside a scope refusal, got:\n{}",
            String::from_utf8_lossy(&out.stdout)
        );
    }
}

/// **`--statevector --seed N` is in scope when the circuit draws a sample.**
/// The shots=None path builds `ExecConfig { seed, mid_circuit_mode: Collapse }`
/// and the statevector backend seeds its rng from it for every mid-circuit
/// measurement, so with such a measurement the reported state IS one seeded
/// trajectory and the seed is the only way to reproduce it. The first version
/// of the `--seed` gate refused this line with "nothing here draws a seeded
/// sample", which was false, and removed that reproducibility.
#[test]
fn a_seed_is_accepted_when_statevector_draws_a_seeded_trajectory() {
    let c = fixture("mid_measure", MID_CIRCUIT_MEASURE);
    let a = run_on(&c, &["--statevector", "--seed", "7"], &[]);
    let stderr = String::from_utf8_lossy(&a.stderr);
    assert_eq!(
        a.status.code(),
        Some(0),
        "--statevector --seed must be accepted when a mid-circuit measurement \
         draws from the seeded rng:\n{stderr}"
    );
    let b = run_on(&c, &["--statevector", "--seed", "7"], &[]);
    assert_eq!(
        String::from_utf8_lossy(&a.stdout),
        String::from_utf8_lossy(&b.stdout),
        "the seed selects the trajectory, so two runs with the same seed must \
         report the same state"
    );
}

/// **`--input` needs the photonics backend AND an in-process run.** Under
/// `--bridge` no path forwards the Fock input (`omega_bridges::perceval::run`
/// sends `input_fock: None`), so a photonic circuit under a bridge used to
/// drop `--input` silently — the gate tested only the backend name, not the
/// `in_process` condition every other knob has. The refusal fires above the
/// bridge dispatch, so it does not depend on a bridge feature being compiled
/// in; the in-process line is the control.
#[test]
fn input_under_a_bridge_is_refused_and_in_process_still_runs() {
    let c = fixture("photonic", PHOTONIC);
    let out = run_on(
        &c,
        &["--bridge", "perceval", "--shots", "10", "--input", "1,0"],
        &[],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(2),
        "--input under --bridge is not forwarded and must refuse (exit 2):\n{stderr}"
    );
    assert!(
        stderr.contains("--input") && stderr.contains("--bridge perceval"),
        "name the flag and the bridge it is not forwarded to:\n{stderr}"
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).trim().is_empty(),
        "no result may be emitted alongside a scope refusal"
    );
    let ok = run_on(&c, &["--shots", "10", "--input", "1,0", "--seed", "1"], &[]);
    assert_eq!(
        ok.status.code(),
        Some(0),
        "--input on in-process photonics is in scope and must run:\n{}",
        String::from_utf8_lossy(&ok.stderr)
    );
}

/// **A configured plugin directory that does not exist is fatal, from either
/// source.** `--backend-dir` was validated at parse; `OMEGA_BACKEND_DIR`, which
/// `docs/PLUGINS.md` documents as THE way to load plugins, went through
/// `BackendRegistry::load_dir`, which returns `Ok(0)` for a missing path — so
/// a typo in the variable passed silently on every builtin run. Both are now
/// resolved and validated in one place, eagerly, and the refusal names the
/// source so the reader knows which setting to fix. An existing (empty)
/// directory from the variable still runs: the check is on the path, not on
/// whether any plugin was found.
#[test]
fn a_missing_plugin_directory_is_fatal_from_flag_and_environment() {
    let missing = std::env::temp_dir()
        .join(format!("omega_no_such_plugin_dir_{}", std::process::id()))
        .to_string_lossy()
        .into_owned();
    let c = circuit();
    let out = run_on(&c, &["--shots", "4"], &[("OMEGA_BACKEND_DIR", &missing)]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(1),
        "OMEGA_BACKEND_DIR={missing} names a directory that does not exist and must \
         be fatal, not loaded as zero plugins:\n{stderr}"
    );
    assert!(
        stderr.contains("OMEGA_BACKEND_DIR") && stderr.contains(&missing),
        "the refusal must name the source and the path:\n{stderr}"
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).trim().is_empty(),
        "no result may accompany the refusal"
    );
    // Same path through the flag: the message names the flag instead.
    let out = run_on(&c, &["--shots", "4", "--backend-dir", &missing], &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("--backend-dir"), "{stderr}");
    // A real, empty directory from the variable is fine.
    let present =
        std::env::temp_dir().join(format!("omega_empty_plugin_dir_{}", std::process::id()));
    std::fs::create_dir_all(&present).expect("create empty plugin dir");
    let out = run_on(
        &c,
        &["--shots", "4", "--seed", "1"],
        &[("OMEGA_BACKEND_DIR", &present.to_string_lossy())],
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "an existing empty plugin dir must not refuse:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// **The two CPU-only sub-arms of statevector sampling refuse an explicit
/// device with the true reason.** `--noise` runs the trajectory noise sampler
/// and a mid-circuit measurement runs a per-shot trajectory loop; both are
/// hard-wired to the CPU backend, so `sv_execute_dispatches` answering "yes"
/// for every device was wrong for them — on a GPU build the run would have
/// executed on the CPU in the device's name, and on this build the refusal
/// blamed a missing feature that would not have helped. Build-independent:
/// the reason is checked before feature availability.
#[test]
fn a_cpu_only_statevector_sampling_arm_refuses_an_explicit_device_with_the_reason() {
    let out = run(&[
        "--shots",
        "10",
        "--noise",
        "{\"depolarizing\":0.01}",
        "--device",
        "cuda",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("--device cuda") && stderr.contains("--noise"),
        "name the device and the reason (the noise sampler):\n{stderr}"
    );
    assert!(
        !stderr.contains("feature not compiled in"),
        "the reason is the CPU-only arm, not the build:\n{stderr}"
    );
    let c = fixture("mid_measure_device", MID_CIRCUIT_MEASURE);
    let out = run_on(&c, &["--shots", "10", "--device", "cuda"], &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("--device cuda") && stderr.contains("mid-circuit measurement"),
        "name the device and the reason (the trajectory loop):\n{stderr}"
    );
    // `--device cpu` is the documented way to say "yes, the CPU, on purpose".
    let out = run_on(
        &c,
        &["--shots", "10", "--device", "cpu", "--seed", "1"],
        &[],
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `--expectation` never parsed `--device` at all — `--device bogus` exited 0
/// on every backend — and no path checked whether the chosen backend and mode
/// have a dispatch for an explicit non-cpu device (the guard reached only
/// statevector sampling). Now: an invalid name is refused in every mode, an
/// explicit device no arm dispatches to is refused (exit 1, the status the
/// existing explicit-device refusal already pins), and cpu always runs.
/// Build-independent: statevector expectation and pauli have no GPU arm in
/// ANY build, so these rows do not depend on which features are compiled in.
#[test]
fn an_explicit_device_is_validated_and_honoured_or_refused_in_every_mode() {
    for mode in [
        &["--expectation", "Z0"][..],
        &["--shots", "10"],
        &["--statevector"],
    ] {
        let out = run(&[mode, &["--device", "bogus"]].concat());
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(1),
            "{mode:?} --device bogus must be refused as malformed:\n{stderr}"
        );
        assert!(
            stderr.contains("--device"),
            "{mode:?}: name the flag:\n{stderr}"
        );
    }
    for args in [
        &["--expectation", "Z0", "--device", "cuda"][..],
        &["--backend", "pauli", "--shots", "10", "--device", "cuda"],
    ] {
        let out = run(args);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(1),
            "{args:?}: no arm dispatches to cuda here; must refuse, not run on the \
             cpu in its name:\n{stderr}"
        );
        assert!(
            stderr.contains("--device cuda"),
            "{args:?}: the refusal must name the device:\n{stderr}"
        );
        assert!(
            String::from_utf8_lossy(&out.stdout).trim().is_empty(),
            "{args:?}: no result may accompany a device refusal"
        );
    }
    let out = run(&["--expectation", "Z0", "--device", "cpu"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "--device cpu is always honoured:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// **Guard the gate.** Every in-scope use of the same flags still runs, so the
/// refusals above are about scope, not about the flags themselves.
#[test]
fn an_in_scope_flag_still_succeeds() {
    for args in [
        &[
            "--backend",
            "pauliprop",
            "--expectation",
            "Z0",
            "--truncate",
            "1e-3",
            "--max-weight",
            "4",
            "--max-freq",
            "2",
            "--max-terms",
            "100000",
            "--max-dropped-mass",
            "1.0",
        ][..],
        &[
            "--backend",
            "majoranaprop",
            "--expectation",
            "Z0",
            "--max-length",
            "4",
            "--truncate",
            "1e-3",
        ],
        &[
            "--backend",
            "mps:8",
            "--shots",
            "10",
            "--seed",
            "1",
            "--device",
            "cpu",
        ],
        &["--backend", "auto", "--expectation", "Z0"],
    ] {
        let out = run(args);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(0),
            "{args:?} is in scope and must still run; the scope gate is over-broad:\n{stderr}"
        );
    }
}

/// A one-qubit circuit whose rotation carries an undeclared symbol, so
/// `theta` is free and the run is undetermined until `--params` says at which
/// angle it should be evaluated.
const ONE_FREE_PARAM: &str =
    "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\ncreg c[1];\nrx(theta) q[0];\n";

/// **A parameterised circuit with no `--params` is refused, not run at 0.0.**
///
/// This was the last silent-substitution site in the CLI. The `else` branch
/// bound every free symbol to 0.0 and printed
/// `Warning: 1 unbound parameters, using 0.0 for all` to **stderr** while the
/// answer went to **stdout** — so `omega-run c.qasm --format json > out.json`
/// captured a complete, well-formed result for a circuit the file does not
/// describe, with nothing in the result channel to say so. `rx(0)` is the
/// identity, so the histogram that came back was a perfectly plausible |0⟩.
///
/// Every mode here returns a number and no number is right until the caller
/// has named the angles, so the absence of `--params` is not an
/// under-specified request with an obvious default — it is an unanswerable
/// one. Exit 2 matches the wrong-length `--params` refusal: one code for both
/// parameter faults.
///
/// The refusal must also NAME the free parameters. There is no
/// `--list-params` flag on this binary, so the refusal is the only discovery
/// path: a reader who did not know the circuit had to learn what to type next
/// from this message alone.
#[test]
fn a_free_parameter_with_no_params_flag_is_refused_rather_than_zeroed() {
    let c = fixture("free_param", ONE_FREE_PARAM);

    for args in [
        &["--shots", "16"][..],
        &["--statevector"],
        &["--expectation", "Z0"],
        &["--gradient", "Z0"],
    ] {
        let out = run_on(&c, args, &[]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!stderr.contains("panicked"), "{args:?} PANICKED:\n{stderr}");
        assert_eq!(
            out.status.code(),
            Some(2),
            "{args:?}: a free parameter with no --params must refuse with exit 2, \
             not report a zero-bound run; stderr:\n{stderr}"
        );
        assert!(
            stderr.contains("--params"),
            "{args:?}: the refusal must name the flag that would fix it:\n{stderr}"
        );
        assert!(
            stderr.contains("theta"),
            "{args:?}: the refusal is the only way to discover the parameter \
             names, so it must list them:\n{stderr}"
        );
        assert!(
            String::from_utf8_lossy(&out.stdout).trim().is_empty(),
            "{args:?}: no result may be emitted alongside a refusal, got:\n{}",
            String::from_utf8_lossy(&out.stdout)
        );
    }
}

/// **The control: the same circuit with a correct `--params` still runs.**
///
/// A refusal that also turned away well-formed lines would be no better than
/// the padding it replaced. `rx(pi)|0>` is `|1>` up to phase, so every shot
/// lands on `1` — which also proves the value was really bound rather than
/// defaulted to the 0.0 that would have given `0`.
#[test]
fn a_free_parameter_with_matching_params_still_runs() {
    let c = fixture("free_param_ok", ONE_FREE_PARAM);
    let out = run_on(
        &c,
        &[
            "--shots",
            "64",
            "--seed",
            "1",
            "--params",
            "3.141592653589793",
        ],
        &[],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "one value for one free parameter must still run:\n{stderr}"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("|1>"),
        "rx(pi)|0> is |1>, so the angle was bound rather than zeroed; got:\n{stdout}"
    );
    assert!(
        !stdout.contains("|0>"),
        "a zeroed angle would have reported |0>; got:\n{stdout}"
    );

    // And the wrong count is still refused, naming the flag (this row also
    // lives in the scope table above, kept here so the three outcomes —
    // absent, wrong length, correct — read together).
    let bad = run_on(&c, &["--shots", "64", "--params", "0.1,0.2"], &[]);
    assert_eq!(bad.status.code(), Some(2));
    let bad_err = String::from_utf8_lossy(&bad.stderr);
    assert!(bad_err.contains("--params"), "{bad_err}");
}
