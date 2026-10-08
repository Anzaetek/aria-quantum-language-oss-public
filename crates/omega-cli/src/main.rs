use std::env;
use std::fs;

use omega_backend_majoranaprop::MajoranaPropBackend;
use omega_backend_mps::MpsBackend;
use omega_backend_pauli::PauliBackend;
use omega_backend_pauliprop::PauliPropBackend;
use omega_backend_photonics::PhotonicsBackend;
use omega_backend_quditsv::QuditSvBackend;
use omega_backend_sector::SectorBackend;
use omega_backend_stabrank::StabRankBackend;
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
    eprintln!("  --dump-state-npy F     Same guards as --dump-state-bits, but F is a");
    eprintln!("                         NumPy .npy file of complex128 (qubit 0 = LOW");
    eprintln!("                         bit of the index). For the emulator comparison.");
    eprintln!("  --timing-reps N        Benchmark: run the statevector evolution 1 + N");
    eprintln!("                         times in this process (the first is a warm-up)");
    eprintln!("                         and print one `omega-timing:` JSON line to");
    eprintln!("                         stderr. Times backend.execute() only: allocate");
    eprintln!("                         |0..0>, apply every gate, return the state to");
    eprintln!("                         the host. Not parse, not output; the result is");
    eprintln!("                         not printed (use --dump-state-npy). --statevector,");
    eprintln!("                         --backend statevector, --device cpu|cuda only.");
    eprintln!("  --shots N              Sample N shots");
    eprintln!("  --expectation OBS      Compute <psi|O|psi> for observable OBS");
    eprintln!("  --expectation-fermionic OBS");
    eprintln!("                         Same, for a FERMIONIC observable in the OpenFermion");
    eprintln!("                         spelling (`0.5 [1^ 3] + 0.5 [3^ 1]`; `p^` creates in");
    eprintln!("                         mode p, bare `p` annihilates, `[]` is the identity).");
    eprintln!("                         Lowered through Jordan-Wigner (mode p = qubit p, Z");
    eprintln!("                         strings included) and then measured exactly like");
    eprintln!("                         --expectation, on every backend. Non-Hermitian text");
    eprintln!("                         is refused. Exclusive with --expectation and");
    eprintln!("                         --hamiltonian.");
    eprintln!("  --hamiltonian FILE     Same, for the FCIDUMP Hamiltonian in FILE (PySCF's");
    eprintln!("                         real &FCI form). Index 1 in the file is mode 0;");
    eprintln!("                         the chemist integrals are the operator, lowered");
    eprintln!("                         through Jordan-Wigner like --expectation-fermionic.");
    eprintln!("                         Exclusive with --expectation and");
    eprintln!("                         --expectation-fermionic.");
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
    eprintln!("                         pauliprop (Pauli propagation; --expectation only),");
    eprintln!("                         majoranaprop (Majorana propagation: exact and cheap on");
    eprintln!("                         fermionic/Givens circuits; --expectation only),");
    eprintln!("                         stabrank (stabilizer-rank decomposition: a Clifford gate");
    eprintln!("                         is free and the cost is exponential in the NON-Clifford");
    eprintln!("                         count, so it is the complement of majoranaprop rather");
    eprintln!("                         than a replacement; --expectation and");
    eprintln!("                         --expectation-fermionic only),");
    eprintln!("                         sector (exact, particle-number-conserving circuits");
    eprintln!("                         only: X/Z/RZ/CZ/CP/SWAP/iSWAP/XX+YY; refuses others).");
    eprintln!("                         quditsv (exact dense mixed-radix statevector for");
    eprintln!("                         DITQASM qudit registers: h/x/z generalised, rxy, csum;");
    eprintln!("                         --statevector and --expectation; refuses past 2^24).");
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
    eprintln!("  --max-length L         majoranaprop: drop Majorana monomials longer than L. The");
    eprintln!("                         run reports `dropped_mass`, a BOUND on the error, exactly");
    eprintln!("                         as pauliprop does for --max-weight. On Givens/number-");
    eprintln!(
        "                         conserving circuits L=2 is EXACT for QUADRATIC observables"
    );
    eprintln!("                         (a single Z_i, a hopping X_iX_j+Y_iY_j); Z_iZ_j is itself");
    eprintln!("                         length 4 and needs L>=4, else the run refuses.");
    eprintln!("                         Reach for it first when a run is refused at the ceiling.");
    eprintln!("  --max-chi X            stabrank: keep only the X heaviest stabilizer branches.");
    eprintln!("                         The run reports `state_dropped_mass` (the raw discarded");
    eprintln!("                         coefficient mass m) and `expectation_error_bound`, which");
    eprintln!("                         is R*m*(2+m) and NOT m: stabrank truncates the STATE, and");
    eprintln!("                         a state enters an expectation twice. Its dropped-mass");
    eprintln!("                         field therefore does NOT compose with majoranaprop's, and");
    eprintln!("                         must never be compared to it. Two runs of one job give");
    eprintln!("                         two certified intervals, and what you may do with them is");
    eprintln!("                         INTERSECT them -- see");
    eprintln!("                         crates/omega-backend-stabrank/CROSS-LANE-INTERVALS.md.");
    eprintln!(
        "  --max-branches N       stabrank: raise the branch ceiling (default 2^16 = 65536)."
    );
    eprintln!("                         UNLIKE --max-chi this does NOT change the answer -- the");
    eprintln!("                         run is refused at the ceiling, never trimmed, so raising");
    eprintln!("                         it buys a larger EXACT decomposition and costs memory and");
    eprintln!("                         chi^2 inner products, not accuracy.");
    eprintln!("                         A name matching a loaded plugin resolves after the");
    eprintln!("                         compiled-in backends.");
    eprintln!("  --backend-dir DIR      Load backend plugins (.so/.dylib/.dll) from DIR");
    eprintln!("                         (repeatable; also OMEGA_BACKEND_DIR). Plugin loading");
    eprintln!("                         is opt-in; plugins run the default sample mode.");
    eprintln!(
        "  --qasm-dialect D       QASM2 reader for bare rxx/ryy/rzz/rbs (none are in qelib1):\n\
         \x20                        legacy  (default) accepts rxx/rzz, refuses ryy/rbs — qiskit from_qasm_str\n\
         \x20                        strict  refuses all four — qiskit qasm2.loads\n\
         \x20                        lenient accepts all four (no qiskit reader does); `rbs(θ) a,b`\n\
         \x20                        is omega's native Givens/hopping gate, kept as one op — the\n\
         \x20                        only spelling the `sector` backend admits.\n\
         \x20                        A file carrying its own `gate` definition is read in all three."
    );
    eprintln!("  --list-backends        List compiled-in backends and loaded plugins, then exit");
    eprintln!("  --bridge NAME          Route execution through an external simulator");
    eprintln!("                         (qiskit, perceval). Only the default sample mode");
    eprintln!("                         is supported; statevector / expectation / gradient");
    eprintln!("                         modes require an in-process backend. Counts are");
    eprintln!("                         displayed MSB-first to match the in-process path.");
    eprintln!("  --device NAME          cpu (default), metal, cuda, opencl — honoured by the");
    eprintln!("                         statevector backend's sampling/--statevector and");
    eprintln!("                         --gradient paths, and by mps for the CUDA SVD hook.");
    eprintln!("                         Metal does not accelerate the MPS contraction:");
    eprintln!("                         that f32 path is not installed, because");
    eprintln!("                         discarded_weight would not be a bound.");
    eprintln!("                         MPS_METAL_CONTRACT=1 opts in and warns.");
    eprintln!("                         An EXPLICIT device the chosen backend and mode");
    eprintln!("                         cannot dispatch to is refused, never run on the");
    eprintln!("                         cpu in its name; OMEGA_DEVICE still falls back.");
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
    eprintln!("  --params V0,V1,...     Bind free parameters (sorted by symbol ID).");
    eprintln!("                         REQUIRED for a circuit that has any: one value per");
    eprintln!("                         parameter, no more, no fewer. A circuit with free");
    eprintln!("                         parameters and no --params is refused, not run at");
    eprintln!("                         0.0 — the refusal names the parameters it wants.");
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
    eprintln!("                   FermionicQASM 1.0 (header `FERMIONICQASM 1.0;`,");
    eprintln!("                   conventionally .fqasm). An in-house format; it carries");
    eprintln!("                   no authority of a standard (docs/FERMIONICQASM.md).");
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

/// Wire the MPS device hooks the CLI actually stands behind.
///
/// CUDA takes the native-f64 SVD hook — cuSOLVER's exact `gesvd`, precision
/// unchanged — and wins the arm on a dual-vendor build for that reason. The Metal two-site
/// contraction does **not** install. It is f32 above its bond threshold —
/// Apple GPUs have no f64 — and on a quantity documented as a bound that is
/// disqualifying: measured on an M4, `discarded_weight` comes out smaller
/// than the CPU value on a fraction of the splits it touches (`STATUS.md`
/// §5 item 3). It is also slower than the CPU path on those shapes. `--device
/// metal` still runs, and it says so: the contraction stays on exact f64.
///
/// `MPS_METAL_CONTRACT=1` (also `true` / `yes`) opts back in. That arm warns
/// that the certificate is not a bound under it. A `--device cpu` run never
/// takes either hook.
fn apply_mps_device_hooks<B: MpsDeviceHooks>(backend: B, device: Option<&str>) -> B {
    let requested = device.and_then(|s| omega_core::device::DeviceKind::parse(s).ok());
    let resolved = omega_core::device::DeviceKind::resolve(requested);
    #[cfg(feature = "cuda")]
    let backend = if matches!(resolved, omega_core::device::DeviceKind::Cuda) {
        // stderr directly: this is a free function and `info` is a per-mode
        // closure; a device notice belongs on the human channel in all modes.
        eprintln!("Device: cuda (mps bond-compression SVD)");
        // The hook falls back to the CPU per call; count from here so the run
        // that follows can say how many SVDs actually ran on the device
        // (`report_cuda_svd_dispatch`).
        omega_backend_mps_cuda::reset_cuda_svd_dispatch();
        CUDA_SVD_HOOK_INSTALLED.with(|c| c.set(true));
        backend.svd_hook(omega_backend_mps_cuda::cuda_svd_flat)
    } else {
        backend
    };
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let backend = if matches!(resolved, omega_core::device::DeviceKind::Metal) {
        if mps_metal_contract_opt_in() {
            // Loud on purpose. The opt-in exists so the kernel can still be
            // measured; it must not look like a normal device run.
            eprintln!(
                "WARNING: MPS_METAL_CONTRACT is set, so the MPS two-site contraction \
                 is routed through Metal (f32 above the bond threshold). \
                 discarded_weight can come out SMALLER than the exact-f64 CPU value. \
                 On this run the truncation certificate is NOT a bound."
            );
            backend.contract_hook(omega_backend_mps_metal::metal_contract_2q)
        } else {
            eprintln!(
                "Device: metal requested for MPS. The two-site contraction stays on the CPU \
                 (exact f64). The Metal contraction is f32 and can under-report \
                 discarded_weight, so it is not installed: a certificate that is \
                 occasionally too small is not a bound. Measured speed is not a reason \
                 to install it either (slower than CPU per dispatched pair). \
                 Set MPS_METAL_CONTRACT=1 to opt in; under that variable discarded_weight \
                 is not a bound."
            );
            backend
        }
    } else {
        backend
    };
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let _ = resolved;
    backend
}

/// `MPS_METAL_CONTRACT=1` (or `true` / `yes`) opts into the f32 Metal
/// two-site contraction. Unset is the production default: that path is not
/// installed, because `discarded_weight` would not be a bound.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn mps_metal_contract_opt_in() -> bool {
    match std::env::var("MPS_METAL_CONTRACT") {
        Ok(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        }
        Err(_) => false,
    }
}

/// The two device hooks BOTH MPS samplers take, so `apply_mps_device_hooks`
/// wires the noisy sampler exactly as it wires the exact one. Before this the
/// `--noise` arm built `NoisyMpsBackend` bare and never read `--device`,
/// while the device gate said MPS dispatches to CUDA — so `--backend mps
/// --noise … --device cuda` ran the CPU SVD and was reported as a device run.
///
/// Each hook is guarded to MATCH ITS CALL SITE'S `cfg` EXACTLY, and the two
/// call sites are not symmetric. The GPU arms are complementary rather than
/// parallel: CUDA accelerates the SVD (`cuda_svd_flat`;
/// `omega-backend-mps-cuda` has no contraction function at all). The Metal
/// contraction (`metal_contract_2q`, SVD left on the CPU) is f32 and is not
/// installed unless `MPS_METAL_CONTRACT` is set — a bound that is occasionally
/// low is not a bound. The Metal arm is additionally `not(feature = "cuda")`,
/// because on a dual-vendor build CUDA deliberately wins the arm — `gesvdj`
/// is native f64, so precision is unchanged, whereas the Metal contraction
/// drops to f32 above its bond threshold.
///
/// So `contract_hook` is reachable only under `metal AND NOT cuda`, and only
/// as the opt-in, and `svd_hook` only under `cuda`. A trait-level guard keyed
/// on `any(cuda, metal)` claims both are live whenever either is, and breaks
/// `-D warnings` on every build that is not exactly metal-without-cuda.
///
/// This cost a CI failure to find. `ci.sh`'s CUDA-feature clippy line lints
/// precisely the cuda-only combination, and nothing had run it since the hooks
/// were split — the box that could was assumed not to exist, and this repo has
/// had an RTX PRO 6000 in it the whole time.
///
/// Worth recording that the dual-vendor precision choice above was made on
/// principle and has since been MEASURED to be right: on an M4, the f32 Metal
/// contraction perturbs `discarded_weight` by up to 2.8e-8 relative and reports
/// a SMALLER value than the exact-f64 CPU path on roughly a quarter of the
/// splits it touches (`STATUS.md` §5.3). A truncation certificate is a bound,
/// so occasionally-low is not a bound.
trait MpsDeviceHooks: Sized {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn svd_hook(self, f: omega_backend_mps::SvdFlatFn) -> Self;
    #[cfg_attr(not(all(feature = "metal", not(feature = "cuda"))), allow(dead_code))]
    fn contract_hook(self, f: omega_backend_mps::Contract2qFn) -> Self;
}

impl MpsDeviceHooks for MpsBackend {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn svd_hook(self, f: omega_backend_mps::SvdFlatFn) -> Self {
        self.with_svd_fn(f)
    }
    #[cfg_attr(not(all(feature = "metal", not(feature = "cuda"))), allow(dead_code))]
    fn contract_hook(self, f: omega_backend_mps::Contract2qFn) -> Self {
        self.with_contract_fn(f)
    }
}

impl MpsDeviceHooks for omega_backend_mps::NoisyMpsBackend {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn svd_hook(self, f: omega_backend_mps::SvdFlatFn) -> Self {
        self.with_svd_fn(f)
    }
    #[cfg_attr(not(all(feature = "metal", not(feature = "cuda"))), allow(dead_code))]
    fn contract_hook(self, f: omega_backend_mps::Contract2qFn) -> Self {
        self.with_contract_fn(f)
    }
}

/// An explicit `--device` that the arm could not use is an error, not a
/// downgrade: a CPU result labelled as a device run looks like data, and a
/// benchmark lane was published and retracted on exactly that. Shared by the
/// sampling and gradient arms so the two cannot drift — the gradient arms
/// used to fall back with only an `info` line. Every call site sits inside
/// a device arm, so a GPU-free build has none.
#[cfg_attr(
    not(any(feature = "cuda", feature = "metal", feature = "opencl")),
    allow(dead_code)
)]
fn refuse_device_fallback(device: &str, msg: &str) -> ! {
    eprintln!(
        "--device {device} was requested but the run could not use it: {msg}. Refusing \
         rather than returning a CPU result labelled as a device run. Drop --device to \
         fall back deliberately."
    );
    std::process::exit(1);
}

/// `--timing-reps N`: run one statevector evolution `1 + N` times, time each
/// with a monotonic clock, print ONE `omega-timing:` JSON line on stderr, and
/// return the LAST run's result (so what is dumped is a timed run's output).
///
/// What the clock covers is exactly the closure: for the CPU arm
/// `StatevectorBackend::new().execute(..)`, for the CUDA arm `b.execute(..)`
/// on an already-constructed backend (context creation and NVRTC kernel
/// compilation happen in `CudaStatevectorBackend::new()`, outside the clock;
/// state allocation, every gate and the device-to-host copy are inside it).
/// Parse, lowering and output are never inside it. The previous result is
/// dropped before the next run starts, so peak memory is one state, not two.
fn timed_execute<F>(
    reps: Option<usize>,
    arm: &str,
    mut run: F,
) -> omega_core::error::Result<omega_core::executor::ExecResult>
where
    F: FnMut() -> omega_core::error::Result<omega_core::executor::ExecResult>,
{
    let Some(reps) = reps else {
        return run();
    };
    let t = std::time::Instant::now();
    let mut last = run();
    let warmup_s = t.elapsed().as_secs_f64();
    let mut reps_s = Vec::with_capacity(reps);
    for _ in 0..reps {
        if last.is_err() {
            break;
        }
        drop(last);
        let t = std::time::Instant::now();
        last = run();
        reps_s.push(t.elapsed().as_secs_f64());
    }
    if last.is_ok() {
        eprintln!(
            "omega-timing: {}",
            serde_json::json!({
                "arm": arm,
                "threads": rayon::current_num_threads(),
                "warmup_s": warmup_s,
                "reps_s": reps_s,
            })
        );
    }
    last
}

/// `--dump-state-npy`: the state as a NumPy v1.0 `.npy` file, dtype `<c16`,
/// shape `(2^n,)`, amp[i] with qubit 0 the LOW bit of i. A raw binary dump,
/// because the hex JSON of `--dump-state-bits` is ~45 bytes per amplitude
/// (12 GB at n = 28) and the comparison reads states up to that width.
fn write_state_npy(path: &str, amps: &[num_complex::Complex<f64>]) -> std::io::Result<()> {
    use std::io::Write;
    let dict = format!(
        "{{'descr': '<c16', 'fortran_order': False, 'shape': ({},), }}",
        amps.len()
    );
    // magic(6) + version(2) + header-len(2) + dict + pad + '\n' = 0 (mod 64)
    let unpadded = 10 + dict.len() + 1;
    let pad = (64 - unpadded % 64) % 64;
    let header = format!("{dict}{}\n", " ".repeat(pad));
    let mut w = std::io::BufWriter::with_capacity(1 << 20, fs::File::create(path)?);
    w.write_all(b"\x93NUMPY\x01\x00")?;
    w.write_all(&(header.len() as u16).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    for a in amps {
        w.write_all(&a.re.to_le_bytes())?;
        w.write_all(&a.im.to_le_bytes())?;
    }
    w.flush()
}

thread_local! {
    /// Set by `apply_mps_device_hooks` when it installs the CUDA SVD hook, and
    /// cleared by `report_cuda_svd_dispatch`, so the report is made exactly
    /// for the runs that had the hook and by the code that installed it — not
    /// re-derived from the device flags, which could drift. Thread-local
    /// because the library's dispatch counters are.
    static CUDA_SVD_HOOK_INSTALLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The CUDA SVD dispatch of an MPS run, for the JSON document.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
struct CudaSvdCounts {
    gpu_calls: u64,
    cpu_fallbacks: u64,
    last_fallback: Option<&'static str>,
}

/// What a run's CUDA SVD dispatch says, and whether the run must be refused.
///
/// Returns the stderr line printed after EVERY MPS run that installed the CUDA
/// SVD hook, and — only when `--device cuda` was EXPLICIT and at least one SVD
/// fell back to the CPU — the reason handed to [`refuse_device_fallback`].
/// `cuda_svd_flat` falls back per call and the result looks the same either
/// way, so without this a run that did part (or all) of its SVDs on the CPU
/// was reported as a CUDA run; `STATUS.md` §5.16 measured 20-45% of calls
/// doing exactly that. Same policy as every other device arm: an explicit
/// device means no CPU substitution, and automatic selection (`OMEGA_DEVICE`)
/// keeps running with the line as its notice.
///
/// Pure, so the policy is unit-tested without having to break a device.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn cuda_svd_verdict(counts: CudaSvdCounts, explicit_device: bool) -> (String, Option<String>) {
    let CudaSvdCounts {
        gpu_calls,
        cpu_fallbacks,
        last_fallback,
    } = counts;
    let last = last_fallback
        .map(|r| format!(" (last: {r})"))
        .unwrap_or_default();
    let line = format!("cuda svd: {gpu_calls} on gpu, {cpu_fallbacks} cpu fallbacks{last}");
    let refusal = (explicit_device && cpu_fallbacks > 0).then(|| {
        format!(
            "{cpu_fallbacks} of {} MPS bond-compression SVDs fell back to the CPU (last: {})",
            gpu_calls + cpu_fallbacks,
            last_fallback.unwrap_or("reason not recorded")
        )
    });
    (line, refusal)
}

/// After a run: if it installed the CUDA SVD hook, print the dispatch line
/// and refuse an explicit `--device cuda` run that fell back (exit 1, via
/// [`refuse_device_fallback`]). Returns the counts for the JSON document;
/// `None` when the hook was not installed, including on every build without
/// the `cuda` feature.
fn report_cuda_svd_dispatch(explicit_device: bool) -> Option<CudaSvdCounts> {
    if !CUDA_SVD_HOOK_INSTALLED.with(|c| c.replace(false)) {
        return None;
    }
    #[cfg(feature = "cuda")]
    {
        let d = omega_backend_mps_cuda::cuda_svd_dispatch();
        let counts = CudaSvdCounts {
            gpu_calls: d.gpu_calls,
            cpu_fallbacks: d.cpu_fallbacks,
            last_fallback: d.last_fallback,
        };
        let (line, refusal) = cuda_svd_verdict(counts, explicit_device);
        eprintln!("{line}");
        if let Some(msg) = refusal {
            refuse_device_fallback("cuda", &msg);
        }
        Some(counts)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = explicit_device;
        None
    }
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

/// A flag that parsed, on a line that is otherwise well-formed, asking for
/// something this backend or this mode cannot mean (plan §A6). Exit 2 —
/// distinct from 1, "your input was malformed" — so a script can tell "fix
/// the value" from "this combination has no meaning" without parsing text.
fn refuse_scope(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

/// Whether the statevector sampling / `--statevector` path has an arm for
/// `dev` at all. Feature availability is the separate question asked with
/// `DeviceKind::is_available`; these three mirror which arms EXIST.
fn sv_execute_dispatches(dev: omega_core::device::DeviceKind) -> bool {
    use omega_core::device::DeviceKind as D;
    matches!(dev, D::Cpu | D::Metal | D::Cuda | D::OpenCl)
}

/// The statevector SAMPLING path has two sub-arms that stay on the CPU
/// whatever device was named: the `--noise` trajectory sampler
/// (`NoisyStatevectorBackend`; every `use_*_sv` requires `noise_model` to be
/// `None`) and the per-shot trajectory loop a mid-circuit measurement needs
/// (`needs_collapse`), which is hard-wired to `StatevectorBackend::new()`.
/// Returns why the run would stay on the CPU, so an explicit device can be
/// refused with the true reason rather than a false "feature not compiled
/// in" (the noise case) or a silent CPU run (the trajectory case).
fn sv_sampling_cpu_only_reason(noisy: bool, collapses: bool) -> Option<&'static str> {
    if noisy {
        Some("--noise: the trajectory noise sampler runs on the CPU only")
    } else if collapses {
        Some("mid-circuit measurement: each shot is a CPU trajectory")
    } else {
        None
    }
}

/// The `--gradient` path has Metal and CUDA arms and no OpenCL one.
fn sv_gradient_dispatches(dev: omega_core::device::DeviceKind) -> bool {
    use omega_core::device::DeviceKind as D;
    matches!(dev, D::Cpu | D::Metal | D::Cuda)
}

/// MPS device acceptance. CUDA installs the SVD hook. Metal is accepted so
/// `--device metal` can say, on the record, that the f32 contraction is not
/// installed and the certificate stays the CPU bound — refusing it would hide
/// that, and silently running it used to label an f32 certificate as a device
/// result. CUDA wins on a dual-vendor build (`apply_mps_device_hooks`), so
/// Metal is accepted only when CUDA is absent. There is no OpenCL hook.
fn mps_dispatches(dev: omega_core::device::DeviceKind) -> bool {
    use omega_core::device::DeviceKind as D;
    match dev {
        D::Cpu | D::Cuda => true,
        D::Metal => !cfg!(feature = "cuda"),
        D::OpenCl => false,
    }
}

/// Every directory plugins load from — explicit `--backend-dir` flags, then
/// `OMEGA_BACKEND_DIR` (`:`-separated) — resolved and VALIDATED in one place,
/// eagerly, on every run. Plugin loading is **opt-in**: there is no implicit
/// `~/.omega/backends` probe, so a mistyped `--backend NAME` never dlopens
/// code the user didn't ask for.
///
/// A configured directory that does not exist or cannot be read is fatal.
/// `BackendRegistry::load_dir` returns `Ok(0)` for a missing path, and the
/// registry is built only when a `--backend` name is not compiled in, so a
/// mistyped path would otherwise pass silently on every builtin run and
/// surface only as "Unknown backend" on the one run that needed it. The
/// flag was checked at parse time; the variable, which `docs/PLUGINS.md`
/// documents as THE way to load plugins, never was.
fn configured_backend_dirs(explicit_dirs: &[String]) -> Vec<std::path::PathBuf> {
    let mut dirs: Vec<(std::path::PathBuf, &str)> = explicit_dirs
        .iter()
        .map(|d| (std::path::PathBuf::from(d), "--backend-dir"))
        .collect();
    if let Ok(env_dirs) = env::var("OMEGA_BACKEND_DIR") {
        for d in env_dirs.split(':').filter(|s| !s.is_empty()) {
            dirs.push((std::path::PathBuf::from(d), "OMEGA_BACKEND_DIR"));
        }
    }
    dirs.into_iter()
        .map(|(dir, source)| {
            if !dir.is_dir() {
                eprintln!("{source} {}: not a directory.", dir.display());
                std::process::exit(1);
            }
            if let Err(e) = fs::read_dir(&dir) {
                eprintln!("{source} {}: cannot be read: {e}", dir.display());
                std::process::exit(1);
            }
            dir
        })
        .collect()
}

/// Load plugins from directories `configured_backend_dirs` has already
/// validated; a load failure inside one is a warning, not a refusal, because
/// the run may not need that plugin at all.
fn build_plugin_registry(dirs: &[std::path::PathBuf]) -> omega_core::plugin::BackendRegistry {
    let mut registry = omega_core::plugin::BackendRegistry::new();
    for dir in dirs {
        if let Err(e) = registry.load_dir(dir) {
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
        let registry = build_plugin_registry(&configured_backend_dirs(&scan_backend_dirs(&args)));
        for name in [
            "statevector",
            "mps",
            "mps:<chi>",
            "mps:auto",
            "mps:auto:<ceiling>",
            "pauli",
            "pauliprop",
            "majoranaprop",
            "stabrank",
            "sector",
            "quditsv",
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
    // Whether `--shots` / `--statevector` / `--qasm-dialect` were PASSED, as
    // distinct from what they resolved to: `shots` defaults to `Some(1024)`,
    // so the value alone cannot tell "the user asked to sample" from "nobody
    // said". Only an explicit request can be silently unhonoured — see the
    // flag-scope gate below `chosen`.
    let mut shots_requested = false;
    let mut statevector_requested = false;
    let mut qasm_dialect_requested = false;
    let mut input_state: Option<Vec<u32>> = None;
    let mut backend_name: Option<String> = None;
    // Which QASM2 reader to imitate for `rxx`/`ryy`/`rzz`, none of which is in
    // qelib1. Default matches Qiskit's legacy loader; see `Qasm2Dialect`.
    let mut qasm_dialect = omega_parser::lower::Qasm2Dialect::default();
    let mut device_name: Option<String> = None;
    let mut observable_str: Option<String> = None;
    let mut fermionic_str: Option<String> = None;
    let mut hamiltonian_path: Option<String> = None;
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
    let mut mp_max_length: Option<usize> = None;
    // The two stabrank knobs, kept apart for the reason the backend keeps them
    // apart: `--max-chi` truncates (it discards branches, moves the answer and
    // owes a mass and a bound) and `--max-branches` refuses (it stops a run
    // that has left the engine's regime without discarding anything). A single
    // "chi limit" flag would conflate a cut with a ceiling.
    let mut sr_max_chi: Option<usize> = None;
    let mut sr_max_branches: Option<usize> = None;
    let mut bridge_name: Option<String> = None;
    let mut backend_dirs: Vec<String> = Vec::new();
    // Write the final statevector as HEX FLOAT BIT PATTERNS for the
    // cross-device bit-equality protocol (PLAN-BITEQ-CUDA-METAL.md). A
    // measurement artifact, so its guards are hard errors, never fallbacks.
    let mut dump_state_bits: Option<String> = None;
    // `--dump-state-npy` rides on every `--dump-state-bits` guard (it sets
    // `dump_state_bits` too); only the file format at the write site differs.
    let mut dump_state_npy = false;
    let mut timing_reps: Option<usize> = None;

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
                shots_requested = true;
            }
            "--statevector" | "--exact" => {
                shots = None;
                statevector_requested = true;
            }
            "--seed" => {
                seed = Some(flag_parse(&args, &mut i, "--seed", "an integer seed"));
            }
            "--backend" => {
                backend_name = Some(flag_value(&args, &mut i, "--backend"));
            }
            "--backend-dir" => {
                // Validated with `OMEGA_BACKEND_DIR`, in `configured_backend_dirs`,
                // right after this loop.
                backend_dirs.push(flag_value(&args, &mut i, "--backend-dir"));
            }
            "--qasm-dialect" => {
                let v = flag_value(&args, &mut i, "--qasm-dialect");
                qasm_dialect_requested = true;
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
                dump_state_npy = false;
            }
            "--dump-state-npy" => {
                dump_state_bits = Some(flag_value(&args, &mut i, "--dump-state-npy"));
                dump_state_npy = true;
            }
            "--timing-reps" => {
                let n: usize = flag_parse(
                    &args,
                    &mut i,
                    "--timing-reps",
                    "a whole number of timed repetitions",
                );
                // 0 reps would report the warm-up as the median.
                if n == 0 {
                    eprintln!("--timing-reps must be at least 1 (0 would time only the warm-up)");
                    std::process::exit(2);
                }
                timing_reps = Some(n);
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
            "--expectation-fermionic" => {
                fermionic_str = Some(flag_value(&args, &mut i, "--expectation-fermionic"));
            }
            "--hamiltonian" => {
                hamiltonian_path = Some(flag_value(&args, &mut i, "--hamiltonian"));
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
            "--max-length" => {
                mp_max_length = Some(flag_parse(
                    &args,
                    &mut i,
                    "--max-length",
                    "a maximum Majorana monomial length (an even integer >= 2)",
                ));
            }
            // stabrank's truncating cut. `Some(0)` asks for a decomposition
            // with no branches in it; the backend refuses that by name rather
            // than reinterpreting it, so the CLI passes the number through
            // instead of second-guessing it here.
            "--max-chi" => {
                sr_max_chi = Some(flag_parse(
                    &args,
                    &mut i,
                    "--max-chi",
                    "a maximum stabilizer-branch count, e.g. 64",
                ));
            }
            "--max-branches" => {
                sr_max_branches = Some(flag_parse(
                    &args,
                    &mut i,
                    "--max-branches",
                    "a branch ceiling, e.g. 262144",
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

    // Plugin directories from both sources, validated now rather than on the
    // one run that turns out to need them.
    let backend_dirs = configured_backend_dirs(&backend_dirs);

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
    // `--qasm-dialect` chooses which QASM2 reader to imitate. A QPY input
    // decoded by the pure-Rust reader never becomes QASM2 text, so the
    // dialect would be accepted and silently ignored (plan §A6).
    if qasm_dialect_requested && prebuilt_circuit.is_some() {
        refuse_scope(
            "--qasm-dialect selects the QASM2 reader to imitate (rxx/ryy/rzz); this \
             input was decoded from QPY by the pure-Rust reader, which never sees QASM2 \
             text, so the dialect would be silently ignored. Drop --qasm-dialect.",
        );
    }
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
            || fermionic_str.is_some()
            || hamiltonian_path.is_some()
            || gradient_str.is_some()
            || gradient_fn_str.is_some())
    {
        eprintln!(
            "--dump-state-bits dumps the final statevector of a plain run; \
             --bridge/--expectation/--expectation-fermionic/--hamiltonian/--gradient \
             modes produce no state to dump. \
             Drop the mode flag, or drop --dump-state-bits."
        );
        std::process::exit(1);
    }

    // `--timing-reps` is refused in every mode for the same reason, and it has
    // to be refused HERE: the mode dispatches below `return` long before the
    // plain-path guard further down, so a mode check placed there was dead
    // code — `--expectation Z0 --timing-reps 2` printed the expectation, timed
    // nothing, wrote no `omega-timing:` line and exited 0 (found by review).
    if timing_reps.is_some()
        && (bridge_name.is_some()
            || observable_str.is_some()
            || fermionic_str.is_some()
            || hamiltonian_path.is_some()
            || gradient_str.is_some()
            || gradient_fn_str.is_some())
    {
        eprintln!(
            "--timing-reps times the plain statevector evolution only; \
             --bridge/--expectation/--expectation-fermionic/--hamiltonian/--gradient \
             modes are not timed by it. Drop the mode flag, or drop --timing-reps."
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
    // Whether the simulator must COLLAPSE mid-circuit: any classically
    // conditioned gate, any measurement followed by another op, or any cbit
    // written by more than one `measure` (Qiskit semantics: last-write-wins
    // on the creg; basis-state sampling would double-count). Pure
    // end-of-circuit sampling with a 1:1 qubit→cbit mapping stays in Skip
    // mode (cheaper, preserves the analytic statevector path). The predicate
    // lives in `omega_core::executor` so the N-way counts matrix drives the
    // same decision this binary ships. Computed once: the display width, the
    // `--seed` gate, the device gate and the execute path all read it.
    let needs_collapse = omega_core::executor::needs_collapse(&circuit);
    let counts_display_width = omega_core::executor::counts_outcome_width(
        &circuit,
        omega_core::executor::counts_keyed_on_creg(&circuit, needs_collapse),
    ) as u32;
    // Notices produced BEFORE the flag-scope / device gate are held here and
    // flushed only once the gate has passed. `info` writes to stdout in text
    // mode, and a reader or a pipe that sees "Circuit: 2 qubits, 2 ops"
    // beside a non-zero exit has a partial report in the result channel and
    // a refusal in the other; the A6 test `no result may be emitted alongside
    // a scope refusal` caught exactly that with the header alone.
    //
    // The invariant this buffer establishes is precisely that one: the
    // flag-scope refusals (exit 2) and the explicit-device refusals (exit 1)
    // ABOVE the flush leave stdout empty. It does NOT hold for every non-zero
    // exit. The buffer is drained right after the gate, and the mode arms
    // below still refuse — `--noise` off statevector/mps, the
    // `--dump-state-bits` guards, a device arm's fallback, a backend with no
    // arm for the mode — after `info` lines have gone to stdout. Moving those
    // checks up here would mean parsing every mode's inputs above the gate;
    // deferring the header is the cheap, honest half of that.
    let mut preamble: Vec<String> = Vec::new();
    preamble.push(format!(
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
    let symbol_ids = circuit.sorted_symbol_ids();
    // The circuit's free parameters, named, in the order `--params` binds
    // them. Both refusals below print it: it is the only discovery path this
    // CLI has — there is no `--list-params` — so a refusal that did not list
    // them would leave the reader with nothing to type next.
    let free_params = || -> String {
        if symbol_ids.is_empty() {
            "no free parameters".to_string()
        } else {
            format!(
                "{} free parameter(s), in binding order: {}",
                symbol_ids.len(),
                symbol_ids
                    .iter()
                    .map(|id| circuit.symbol_name(*id))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    };

    let params = if let Some(ref vals) = param_values {
        // One value per free parameter, exactly. Before this refusal a short
        // list bound the missing symbols to 0.0 and a long list dropped the
        // extras — both silently — and the "Parameters bound" line below zips
        // the two lists, so it hid the mismatch as well (plan §A6).
        //
        // `ParameterBinding::from_flat` is the one implementation of the rule;
        // the CLI adds only the flag name, so a script can still grep for
        // `--params`, and the parameter listing, which the typed error has no
        // room for.
        match ParameterBinding::from_flat(&circuit, vals) {
            Ok(pb) => {
                if !symbol_ids.is_empty() {
                    preamble.push(format!(
                        "Parameters bound: {:?}",
                        symbol_ids
                            .iter()
                            .zip(vals.iter())
                            .map(|(id, v)| format!("{}={:.4}", circuit.symbol_name(*id), v))
                            .collect::<Vec<_>>()
                    ));
                }
                pb
            }
            Err(e) => refuse_scope(&format!(
                "--params: {e}. This circuit has {}.",
                free_params()
            )),
        }
    } else if symbol_ids.is_empty() {
        // The only legitimate default: nothing to bind. An empty binding for a
        // circuit with no free symbols is not a choice of values, it is the
        // absence of any.
        ParameterBinding::new()
    } else {
        // NOT a default — a refusal. This branch used to bind every free
        // symbol to 0.0 and print a `Warning:` to stderr, which is the exact
        // shape this repository treats as the worst outcome: every mode below
        // (sampling, --statevector, --expectation, --gradient) then produced a
        // complete, plausible, well-formatted answer for a DIFFERENT circuit
        // than the file describes. Zero is not a neutral angle — Ry(0) is the
        // identity, so an unparameterised-looking |0...0> histogram comes back
        // — and the warning went to stderr while the number went to stdout, so
        // `omega-run c.qasm --format json > out.json` recorded the wrong answer
        // with no trace of the substitution in the result channel at all.
        //
        // There is no mode here that merely inspects a circuit (`--list-backends`
        // exits above, before the file is even read); every mode returns a
        // number, and no number is right until the caller has said at which
        // angles. So the absence of `--params` on a parameterised circuit is
        // not an under-specified request with an obvious default, it is an
        // unanswerable one.
        //
        // Exit 2, matching the wrong-length refusal above rather than the
        // exit-1 malformed-input class: a script that handles "this line does
        // not determine a run" has one code to handle for both parameter
        // faults.
        refuse_scope(&format!(
            "This circuit has {}, and no --params was given. Binding them all to \
             0.0 would report a complete, plausible result for a circuit you did \
             not ask for — so it is refused rather than defaulted. Pass --params \
             with one value per parameter, in that order (e.g. --params {}).",
            free_params(),
            vec!["0.0"; symbol_ids.len()].join(",")
        ))
    };

    // Select backend
    let default_backend = match circuit.circuit_type {
        // Fermionic lowering emits qubit gates (X, U1, CU3, Rbs, Measure).
        CircuitType::GateBased | CircuitType::Fermionic => "statevector",
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
    // The stabilizer backend samples, renders a state, and takes an
    // expectation value (`PauliBackend::expectation`, exact and polynomial);
    // it has no gradient arm. `--backend auto --expectation Z0` on a Bell
    // pair used to die with "Expectation not supported for backend: pauli",
    // which was read as auto picking a backend that cannot run the mode. It
    // was not: the CLI's expectation dispatch simply had no `pauli` arm, and
    // the first fix routed Clifford circuits away from the backend that
    // serves the mode best instead of adding the arm. The arm exists now. In
    // the gradient modes, which the stabilizer backend genuinely does not
    // serve, a Clifford circuit goes to the statevector backend when it fits
    // (exact) and is otherwise refused with the same options as any other.
    let gradient_mode = gradient_str.is_some() || gradient_fn_str.is_some();
    let clifford = omega_core::circuit::is_clifford_only(&circuit);
    let chosen = if matches!(chosen, "auto") {
        if circuit.circuit_type == CircuitType::Photonic {
            preamble.push("auto: photonic circuit -> photonics".to_string());
            "photonics"
        } else if let Some((reg, wire, d)) = circuit.first_qudit() {
            // Q1 made every qubit engine refuse a qudit register; Q2 gives
            // `auto` the one engine that runs it. Exact, so it belongs here.
            preamble.push(format!(
                "auto: qudit register '{}' (dimension {d} on wire {wire}) -> quditsv \
                 (exact, dense mixed-radix)",
                reg.name
            ));
            "quditsv"
        } else if clifford && !gradient_mode {
            // Exact and polynomial. Worth saying out loud, because the same
            // circuit on the statevector backend is exponential and a user
            // watching the clock deserves to know why it got fast.
            preamble.push(format!(
                "auto: Clifford-only circuit -> pauli (exact, polynomial in the {} qubits)",
                circuit.num_qubits
            ));
            "pauli"
        } else if omega_backend_statevector::capacity::check(circuit.num_qubits, shots.is_some())
            .is_ok()
        {
            preamble.push(format!(
                "auto: {}, {} qubits fits in memory -> statevector (exact)",
                if clifford {
                    "Clifford but a gradient mode, which the stabilizer backend does not serve"
                } else {
                    "non-Clifford"
                },
                circuit.num_qubits
            ));
            "statevector"
        } else {
            if clifford {
                eprintln!("auto: this circuit is Clifford, but the exact stabilizer backend");
                eprintln!(
                    "      has no gradient arm, and a dense statevector of {} qubits",
                    circuit.num_qubits
                );
            } else {
                eprintln!("auto: this circuit is not Clifford, so the exact stabilizer");
                eprintln!(
                    "      backend cannot run it, and a dense statevector of {} qubits",
                    circuit.num_qubits
                );
            }
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

    // ---- Flag scope (plan §A6) ----
    //
    // A flag that parses, on a line that is otherwise well-formed, but asks
    // for something this backend or this mode cannot mean, is REFUSED here —
    // never accepted and dropped. Measured before this gate, every one of
    // these exited 0 and never mentioned the flag: `--truncate 1e-3` on
    // statevector; `--max-length 4` on ANY backend in sampling mode (the only
    // gate lived inside the expectation arm); `--input 1,0` off photonics;
    // `--method` / `--score-fn-shots` with no gradient mode; `--device bogus`
    // with `--expectation` (that path never parsed it); `--seed` on a run that
    // samples nothing; `--shots` on an analytic mode; `--expectation` beside
    // `--gradient` (the later-checked mode silently won). Mechanism intact,
    // meaning gone — the A6 shape. One gate, above every mode, so a flag
    // cannot reach one dispatch path's guard and slip past its neighbour's.
    let who = match bridge_name.as_deref() {
        Some(b) => format!("--bridge {b}"),
        None => format!("--backend {chosen}"),
    };
    let in_process = bridge_name.is_none();

    // Modes are exclusive. `--shots` counts as one because it and
    // `--statevector` overwrite the same variable, so their ORDER on the
    // line used to decide the mode.
    let modes: Vec<&str> = [
        (shots_requested, "--shots"),
        (statevector_requested, "--statevector"),
        (observable_str.is_some(), "--expectation"),
        (fermionic_str.is_some(), "--expectation-fermionic"),
        (hamiltonian_path.is_some(), "--hamiltonian"),
        (gradient_str.is_some(), "--gradient"),
        (gradient_fn_str.is_some(), "--gradient-of-fn"),
    ]
    .iter()
    .filter(|(on, _)| *on)
    .map(|(_, name)| *name)
    .collect();
    if modes.len() > 1 {
        let hint = if shots_requested && gradient_fn_str.is_some() {
            " (--gradient-of-fn --method score-fn takes its shot count from \
             --score-fn-shots.)"
        } else if hamiltonian_path.is_some() && fermionic_str.is_some() {
            " (--hamiltonian reads an FCIDUMP into the ladder observable \
             --expectation-fermionic spells; one run measures one observable.)"
        } else if hamiltonian_path.is_some() && observable_str.is_some() {
            " (--hamiltonian is the fermionic observable from an FCIDUMP; --expectation \
             is a Pauli spelling; one run measures one observable.)"
        } else if observable_str.is_some() && fermionic_str.is_some() {
            " (--expectation takes a Pauli spelling, --expectation-fermionic a ladder \
             spelling; they name different algebras, and one run measures one observable.)"
        } else {
            ""
        };
        refuse_scope(&format!(
            "{} select different modes, and a run is exactly one of: sampling (--shots N, \
             the default), --statevector, --expectation OBS, --expectation-fermionic OBS, \
             --hamiltonian FILE, --gradient OBS, --gradient-of-fn JSON. Before this refusal \
             one of them silently won. Keep one.{hint}",
            modes.join(" and ")
        ));
    }
    // The single analytic mode in force, if any; `None` means the run samples.
    let analytic_mode: Option<&str> = if gradient_fn_str.is_some() {
        Some("--gradient-of-fn")
    } else if gradient_str.is_some() {
        Some("--gradient")
    } else if observable_str.is_some() {
        Some("--expectation")
    } else if fermionic_str.is_some() {
        Some("--expectation-fermionic")
    } else if hamiltonian_path.is_some() {
        Some("--hamiltonian")
    } else if shots.is_none() {
        Some("--statevector")
    } else {
        None
    };

    if seed.is_some() {
        // `--statevector` is analytic only when nothing in the circuit draws
        // a sample. With a mid-circuit measurement (or a Reset, a stochastic
        // channel) the shots=None path builds `ExecConfig { seed,
        // mid_circuit_mode: Collapse }` and the statevector backend seeds its
        // rng from it, so the seed picks WHICH trajectory the reported state
        // is. Refusing it there would remove the only way to reproduce that
        // state; the first version of this gate did exactly that.
        let seeded_trajectory = analytic_mode == Some("--statevector")
            && (needs_collapse
                || circuit
                    .ops
                    .iter()
                    .any(|op| matches!(op.gate, omega_core::circuit::GateKind::Reset)));
        if let Some(m) = analytic_mode.filter(|_| !seeded_trajectory) {
            let extra = if gradient_fn_str.is_some() {
                " (--method score-fn draws its own samples and takes no seed)"
            } else {
                ""
            };
            refuse_scope(&format!(
                "--seed seeds the shot sampler, and {m} is analytic{extra}: nothing here \
                 draws a seeded sample, so the seed would be silently ignored. Drop --seed."
            ));
        }
        if !in_process {
            refuse_scope(&format!(
                "--seed is not forwarded to {who}: the external runner draws its own \
                 samples, so the seed would be silently ignored and the run would still \
                 be nondeterministic. Drop --seed, or run in-process."
            ));
        }
    }

    // Same shape as the knobs below: the flag needs BOTH the backend that reads
    // it and an in-process run. `--bridge perceval` hands the external runner
    // its own default input (`omega_bridges::perceval::run` sends
    // `input_fock: None`), so a photonic circuit under a bridge dropped
    // `--input` silently — the gate tested only the backend name.
    if input_state.is_some() && !(chosen == "photonics" && in_process) {
        if in_process {
            refuse_scope(&format!(
                "--input is the photonics input Fock state; {who} has no Fock input and \
                 would silently ignore it. Drop --input, or run on --backend photonics."
            ));
        } else {
            refuse_scope(&format!(
                "--input is not forwarded to {who}: the external runner uses its own \
                 default input Fock state, so the flag would be silently ignored. Drop \
                 --input, or run in-process on --backend photonics."
            ));
        }
    }

    if grad_method_name.is_some() && gradient_str.is_none() && gradient_fn_str.is_none() {
        refuse_scope(
            "--method selects a gradient method and this run computes no gradient (no \
             --gradient / --gradient-of-fn), so it would be silently ignored. Drop \
             --method, or add the gradient mode it was meant for.",
        );
    }
    if gradient_fn_shots.is_some() {
        let score_fn = matches!(
            grad_method_name.as_deref(),
            Some("score-fn") | Some("score-function") | Some("reinforce")
        );
        if gradient_fn_str.is_none() {
            refuse_scope(
                "--score-fn-shots is the shot count for --gradient-of-fn --method score-fn; \
                 this run has no --gradient-of-fn, so it would be silently ignored.",
            );
        } else if !score_fn {
            refuse_scope(&format!(
                "--score-fn-shots applies to --method score-fn; this --gradient-of-fn uses \
                 {}, which draws no shots, so the count would be silently ignored. Add \
                 --method score-fn or drop --score-fn-shots.",
                grad_method_name
                    .as_deref()
                    .unwrap_or("diagonal (the default)")
            ));
        }
    }

    // The pauliprop / majoranaprop / stabrank knobs. Each changes the answer
    // (or its ceiling) on exactly those backends and means nothing anywhere
    // else.
    // (passed?, flag, owner as prose, owner to suggest, allowed on `chosen`?)
    let on_pp = matches!(chosen, "pauliprop" | "pp");
    let on_mp = matches!(chosen, "majoranaprop" | "mp");
    let on_sr = matches!(chosen, "stabrank" | "sr");
    // `--max-terms` on stabrank is the near-miss worth naming: the flag that
    // raises a ceiling without changing the answer exists on this engine too,
    // under a name counting branches rather than Pauli terms. Refused with
    // that name, in the shape majoranaprop refuses `--max-weight`, rather than
    // folded into the generic row below which would point at pauliprop.
    if on_sr && pp_max_terms.is_some() {
        refuse_scope(
            "--max-terms raises the Pauli-TERM ceiling, which is pauliprop's and \
             majoranaprop's cost axis; stabrank's is the number of stabilizer \
             BRANCHES. Use --max-branches N, which is the same kind of flag — it \
             raises a ceiling and does not change the answer.",
        );
    }
    let knobs = [
        (
            pp_truncate.is_some(),
            "--truncate",
            "pauliprop/majoranaprop/stabrank",
            "pauliprop",
            on_pp || on_mp || on_sr,
        ),
        (
            pp_max_terms.is_some(),
            "--max-terms",
            "pauliprop/majoranaprop",
            "pauliprop",
            on_pp || on_mp,
        ),
        (
            pp_max_dropped.is_some(),
            "--max-dropped-mass",
            "pauliprop/majoranaprop",
            "pauliprop",
            on_pp || on_mp,
        ),
        // majoranaprop refuses these two itself, naming the axis that DOES exist.
        (
            pp_max_weight.is_some(),
            "--max-weight",
            "pauliprop",
            "pauliprop",
            on_pp || on_mp,
        ),
        (
            pp_max_freq.is_some(),
            "--max-freq",
            "pauliprop",
            "pauliprop",
            on_pp || on_mp,
        ),
        (
            mp_max_length.is_some(),
            "--max-length",
            "majoranaprop",
            "majoranaprop",
            on_mp,
        ),
        (
            sr_max_chi.is_some(),
            "--max-chi",
            "stabrank",
            "stabrank",
            on_sr,
        ),
        (
            sr_max_branches.is_some(),
            "--max-branches",
            "stabrank",
            "stabrank",
            on_sr,
        ),
    ];
    for (passed, flag, owner, suggest, allowed) in knobs {
        if passed && !(allowed && in_process) {
            refuse_scope(&format!(
                "{flag} is a {owner} flag; {who} does not take it and would silently \
                 ignore it. Drop {flag}, or run on --backend {suggest}."
            ));
        }
    }

    // `--device`: parsed HERE for every mode (the expectation path never
    // parsed it, so `--device bogus --expectation Z0` exited 0), and an
    // explicit non-cpu device is honoured or refused — never run on the CPU
    // in its name. Same principle as the statevector execute guard further
    // down, which reached only that one path; this one covers the arms that
    // have no device dispatch at all (expectation on statevector, pauli,
    // sector, photonics, plugins, bridges) and the MPS hooks in every mode.
    let requested_device = device_name.as_deref().map(|s| {
        omega_core::device::DeviceKind::parse(s).unwrap_or_else(|e| {
            eprintln!("Invalid --device: {e}");
            std::process::exit(1);
        })
    });
    if let Some(dev) = requested_device {
        use omega_core::device::DeviceKind as D;
        if dev != D::Cpu {
            let on_sv = matches!(chosen, "statevector" | "sv");
            let on_mps = omega_backend_mps::select::is_mps(chosen);
            // Statevector SAMPLING has two sub-arms that never leave the CPU,
            // whatever device arm exists and is compiled in. Checked BEFORE
            // availability so the reason given is the true one in every
            // build: on a GPU-free binary "feature not compiled in" is a fact,
            // but rebuilding with the feature would not change the answer.
            if in_process && on_sv && analytic_mode.is_none() {
                if let Some(reason) =
                    sv_sampling_cpu_only_reason(noise_json.is_some(), needs_collapse)
                {
                    eprintln!(
                        "--device {n} was requested but this sampling run stays on the CPU \
                         regardless ({reason}), and the result would be reported as if it \
                         were a {n} run. Refusing rather than substituting a device: drop \
                         --device, or pass --device cpu.",
                        n = dev.name()
                    );
                    std::process::exit(1);
                }
            }
            if !dev.is_available() {
                eprintln!(
                    "--device {n} was requested but this binary cannot dispatch to it \
                     (feature not compiled in). Refusing rather than running on the CPU \
                     and reporting a device result: rebuild with --features {n}, or drop \
                     --device to let the run fall back deliberately.",
                    n = dev.name()
                );
                std::process::exit(1);
            }
            let arm_dispatches = if !in_process {
                false
            } else if on_sv {
                match analytic_mode {
                    None | Some("--statevector") => sv_execute_dispatches(dev),
                    Some("--gradient") => sv_gradient_dispatches(dev),
                    _ => false,
                }
            } else if on_mps {
                mps_dispatches(dev)
            } else {
                false
            };
            if !arm_dispatches {
                eprintln!(
                    "--device {n} was requested but {who} has no {n} dispatch in {mode} \
                     mode; the run would execute on the CPU and be reported as if it were \
                     a {n} result. Refusing rather than substituting a device: drop \
                     --device, pass --device cpu, or pick a backend/mode that dispatches \
                     to {n}.",
                    n = dev.name(),
                    mode = analytic_mode.unwrap_or("sampling"),
                );
                std::process::exit(1);
            }
        }
    }

    // Every refusal above has had its chance; from here the run proceeds, so
    // the notices held back for it are released in their original order.
    for m in preamble.drain(..) {
        info(m);
    }

    // --- Mode: --bridge dispatch through an external simulator ---
    if let Some(ref bridge_str) = bridge_name {
        // Expectation over a bridge. The transport already existed —
        // `omega_bridges::expectation_qasm2`, used by the cross-check harness
        // — but it was reachable only from Rust, so the CLI turned every
        // `--bridge --expectation` away even for backends that implement it.
        // That is the whole of CR §8's first sub-ask: routing, not protocol.
        if let Some(measured) = expectation_observable(
            observable_str.as_deref(),
            fermionic_str.as_deref(),
            hamiltonian_path.as_deref(),
        ) {
            let obs_str = measured.label.as_str();
            if gradient_str.is_some() || gradient_fn_str.is_some() {
                eprintln!("--bridge cannot combine --expectation with a gradient mode.");
                std::process::exit(1);
            }
            // `--noise` was accepted here and then never read. The counts path
            // below parses it and hands it to the runner; this path returns
            // before reaching that code, and the in-process guard that catches
            // the same mistake for statevector/mps lives ~400 lines further
            // down, after this `return`. So `--bridge X --expectation Z0
            // --noise {...}` printed a NOISELESS number to stdout with nothing
            // on stderr — the silent-wrong-answer shape, because a caller
            // redirecting stdout to a file captured a plausible wrong value
            // and no trace of the flag that was dropped.
            //
            // `omega_bridges::expectation_qasm2` takes no noise argument at
            // all, so this is not "unimplemented for ppvm" — no bridge can
            // express it. Refuse before `Backend::parse` so the refusal is
            // about the flag combination and fires even in a build with no
            // bridge feature compiled in.
            if noise_json.is_some() {
                eprintln!(
                    "--noise cannot be combined with --bridge --expectation: the bridge \
                     expectation protocol carries no noise model, so the value would be \
                     computed noiselessly. Use --shots for a noisy sampled estimate over \
                     the bridge, or --backend pauliprop for an exact noisy expectation \
                     in-process."
                );
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
            let observable = &measured.observable;
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
                    let mut doc = serialize::expectation_to_json(obs_str, value);
                    if let Some(mapping) = &measured.mapping {
                        doc = serialize::attach_observable_mapping(doc, mapping);
                    }
                    println!("{doc}");
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
        // Validate the noise model with the SAME parser the in-process path
        // uses before handing the JSON to a bridge. A bare `from_str` here
        // let a key the in-process path rejects loudly (a typo'd channel, a
        // rate outside [0, 1]) reach the Python side as an arbitrary dict —
        // STATUS §5 #12's last trap. The bridge still receives the JSON
        // value; it is only accepted once `NoiseModel::from_json` has.
        let noise_value: Option<serde_json::Value> = noise_json.as_deref().map(|s| {
            let _validated = parse_noise_model(s);
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
        report_cuda_svd_dispatch(device_name.is_some());
        let method_name = match &method {
            FunctionalGradMethod::DiagonalObservable => "diagonal",
            FunctionalGradMethod::ScoreFunction { .. } => "score-fn",
        };
        if format.is_machine() {
            let named: Vec<(String, f64)> = grads
                .iter()
                .map(|(sym, g)| (circuit.symbol_name(*sym), *g))
                .collect();
            let v = serialize::functional_gradient_to_json(spec_kind, method_name, &named);
            println!("{}", v);
        } else {
            println!("\nGradients:");
            for (sym_id, grad) in &grads {
                let name = circuit.symbol_name(*sym_id);
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

        // Honour --device metal / cuda for the statevector gradient path.
        // `requested_device` was parsed once, above the flag-scope gate. An
        // implicit device (OMEGA_DEVICE, the default) falls back to the CPU
        // with a notice; an EXPLICIT one the arm cannot use is refused, the
        // sampling arms' rule — these arms used to fall back with an `info`
        // line only, so `--device cuda --gradient Z0` could hand back a CPU
        // gradient labelled as a device result.
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
                                        if device_name.is_some() {
                                            refuse_device_fallback("cuda", &msg);
                                        }
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
                                if device_name.is_some() {
                                    refuse_device_fallback(
                                        "cuda",
                                        &format!("cuda unavailable: {e}"),
                                    );
                                }
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
                                        if device_name.is_some() {
                                            refuse_device_fallback("metal", &msg);
                                        }
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
                                if device_name.is_some() {
                                    refuse_device_fallback(
                                        "metal",
                                        &format!("metal unavailable: {e}"),
                                    );
                                }
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
        report_cuda_svd_dispatch(device_name.is_some());

        if format.is_machine() {
            let method_name = grad_method_name.as_deref().unwrap_or("adjoint");
            let named: Vec<(String, f64)> = grads
                .iter()
                .map(|(sym, g)| (circuit.symbol_name(*sym), *g))
                .collect();
            let v = serialize::gradient_to_json(obs_str, method_name, &named);
            println!("{}", v);
        } else {
            println!("\nGradients:");
            for (sym_id, grad) in &grads {
                let name = circuit.symbol_name(*sym_id);
                println!("  d<O>/d({}) = {:.10}", name, grad);
            }
        }
        return;
    }

    // --- Mode: expectation ---
    if let Some(measured) = expectation_observable(
        observable_str.as_deref(),
        fermionic_str.as_deref(),
        hamiltonian_path.as_deref(),
    ) {
        let obs_str = measured.label.as_str();
        let observable = measured.observable;

        info(format!("Observable: {}", obs_str));
        if let Some(mapping) = &measured.mapping {
            info(format!(
                "Fermionic input, lowered through Jordan-Wigner to {} Pauli term(s): {}",
                mapping["mapped_terms"], mapping["mapped_observable"]
            ));
        }

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
        let mut mp_cert: Option<omega_backend_majoranaprop::MajoranaPropCertificate> = None;
        let mut sr_cert: Option<omega_backend_stabrank::StabRankCertificate> = None;
        let mut mps_exp_stats: Option<omega_backend_mps::MpsRunStats> = None;
        // `--max-length` off majoranaprop is refused by the flag-scope gate
        // above `--bridge`, which covers every mode. It used to live here, so
        // it covered only this one: a sampling run took the flag silently.
        let val = match chosen {
            "statevector" | "sv" => {
                StatevectorBackend::new().expectation(&circuit, &params, &observable)
            }
            "sector" => SectorBackend::new().expectation(&circuit, &params, &observable),
            "quditsv" | "qudit" => {
                QuditSvBackend::new().expectation(&circuit, &params, &observable)
            }
            // Exact and polynomial on a Clifford circuit; the backend refuses a
            // non-Clifford one itself. This arm was missing, and its absence
            // was misread as the backend having no expectation at all.
            "pauli" | "stabilizer" => {
                PauliBackend::new().expectation(&circuit, &params, &observable)
            }
            m if omega_backend_mps::select::is_mps(m) => {
                // The backend is bound rather than used and dropped, because its
                // truncation certificate lives on it and is only reachable after
                // the run. Dropping it inline is why a truncating MPS
                // `--expectation` emitted JSON with no `mps_truncation` block
                // while the identical run in counts mode emitted one — the
                // number was printed to stderr and lost to any machine
                // consumer. Taken on EVERY run, not only truncated ones, for the
                // reason the pauliprop arm gives below: an exact run's
                // certificate reads zero, and emitting it only when truncating
                // leaves a consumer unable to tell "exact" from "this build does
                // not report it".
                let backend =
                    mps_backend_from(m, device_name.as_deref()).expect("is_mps implies parse_mps");
                let out = backend.expectation(&circuit, &params, &observable);
                if out.is_ok() {
                    mps_exp_stats = Some(backend.last_run_stats());
                }
                out
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
            "majoranaprop" | "mp" => {
                // Same shape as the pauliprop arm, and it shares `--truncate`,
                // `--max-terms` and the dropped-mass ceiling. What it does NOT
                // share is the cut axis: Majorana LENGTH, never Pauli weight
                // or split frequency, so those two flags are refused rather
                // than silently reinterpreted.
                if pp_max_weight.is_some() || pp_max_freq.is_some() {
                    eprintln!(
                        "majoranaprop truncates on Majorana monomial LENGTH, not Pauli \
                         weight or split frequency: --max-weight/--max-freq are pauliprop \
                         flags. Use --max-length L."
                    );
                    std::process::exit(2);
                }
                let truncating = pp_truncate.is_some() || mp_max_length.is_some();
                let mut backend = if truncating {
                    MajoranaPropBackend::with_truncation(pp_truncate.unwrap_or(0.0), mp_max_length)
                } else {
                    MajoranaPropBackend::new()
                };
                if let Some(mt) = pp_max_terms {
                    backend = backend.with_max_terms(Some(mt));
                }
                if let Some(md) = pp_max_dropped {
                    backend = backend.with_max_dropped_mass(Some(md));
                }
                // Fermionic input takes the direct door (PLAN-FERMIONIC
                // F3): the Majorana sum is seeded from the ladder terms, and
                // the JW Pauli image built above is never handed to the engine.
                // Both doors reach the same seeds — the engine's
                // `fermionic_seed.rs` (i) holds them to a bit-identical
                // certificate — so the `--expectation` spelling of the same
                // operator still reproduces this number. Which door was
                // taken is the certificate's `seed_basis`, set by the engine.
                let run = match &measured.fermionic {
                    Some(op) => {
                        backend.expectation_fermionic_with_certificate(&circuit, &params, op)
                    }
                    None => backend.expectation_with_certificate(&circuit, &params, &observable),
                };
                match run {
                    Ok((v, cert)) => {
                        if truncating {
                            eprintln!(
                                "majoranaprop: dropped_mass={:.3e} of range {:.3e} \
                                 (a bound on |Δ⟨O⟩|, not an estimate); \
                                 terms final {} peak {}",
                                cert.dropped_mass,
                                cert.observable_range,
                                cert.final_terms,
                                cert.peak_terms,
                            );
                        }
                        mp_cert = Some(cert);
                        Ok(v)
                    }
                    Err(e) => Err(e),
                }
            }
            "stabrank" | "sr" => {
                // **Zero engine changes, and this arm is where that shows.**
                // A fermionic observable reaches stabrank as `measured.observable`
                // — the Jordan-Wigner Pauli image the F2 dispatch already built,
                // exact, computed once for every backend — so there is no
                // `expectation_fermionic_*` door to call and none was added.
                // majoranaprop needs one because its seed IS the operator and
                // the ladder spelling seeds it directly (F3); stabrank is
                // Schrödinger, its seed is the state |0…0⟩, and the observable
                // is read out at the end whatever basis it was typed in. That
                // asymmetry is PLAN-MAJORANA-STIM.md §1.4, and the engine
                // records which door it took in `seed_basis`, which has exactly
                // one variant here for the same reason.
                let truncating = pp_truncate.is_some() || sr_max_chi.is_some();
                let mut backend = if truncating {
                    StabRankBackend::with_truncation(pp_truncate.unwrap_or(0.0), sr_max_chi)
                } else {
                    StabRankBackend::new()
                };
                // Outside the `truncating` branch, for the reason the pauliprop
                // arm gives: raising a ceiling is not truncation and has to work
                // on an otherwise-exact run.
                if let Some(mb) = sr_max_branches {
                    backend = backend.with_max_branches(mb);
                }
                match backend.expectation_with_certificate(&circuit, &params, &observable) {
                    Ok((v, cert)) => {
                        if truncating {
                            // `state_dropped_mass` is printed as what it is —
                            // the discarded mass of the STATE — and the bound
                            // beside it, never in its place. majoranaprop's
                            // line above prints one number because there the
                            // mass IS the bound; here it is an input to one,
                            // and the two fields must not be read as the same
                            // kind of thing. CROSS-LANE-INTERVALS.md is why.
                            eprintln!(
                                "stabrank: state_dropped_mass={:.3e} (the raw L1 mass of the \
                                 discarded state, NOT a bound on |Δ⟨O⟩|); \
                                 expectation_error_bound=R·m·(2+m)={:.3e} of range {:.3e}; \
                                 χ final {} peak {}",
                                cert.state_dropped_mass,
                                cert.expectation_error_bound,
                                cert.observable_range,
                                cert.final_chi,
                                cert.peak_chi,
                            );
                        }
                        sr_cert = Some(cert);
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
        let cuda_svd = report_cuda_svd_dispatch(device_name.is_some());

        if format.is_machine() {
            let mut doc = serialize::expectation_to_json(obs_str, val);
            if let Some(c) = &cuda_svd {
                doc = serialize::attach_cuda_svd_dispatch(
                    doc,
                    c.gpu_calls,
                    c.cpu_fallbacks,
                    c.last_fallback,
                );
            }
            if let Some(mapping) = &measured.mapping {
                doc = serialize::attach_observable_mapping(doc, mapping);
            }
            if let Some(cert) = &pp_cert {
                doc = serialize::attach_pauliprop_certificate(doc, cert);
            }
            if let Some(cert) = &mp_cert {
                doc = serialize::attach_majoranaprop_certificate(doc, cert);
            }
            if let Some(cert) = &sr_cert {
                doc = serialize::attach_stabrank_certificate(doc, cert);
            }
            if let Some(stats) = &mps_exp_stats {
                doc = serialize::attach_mps_certificate(doc, stats);
            }
            println!("{doc}");
        } else {
            println!("\n<O> = {:.10}", val);
        }
        return;
    }

    // --- Mode: execute (default) ---
    // `needs_collapse` (computed once, above the gates) decides the
    // mid-circuit mode; see its comment for the predicate.
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

    // `--timing-reps` is wired into exactly two dispatch arms (plain CPU
    // statevector, CUDA statevector). Anywhere else it would be accepted and
    // time nothing, so refuse every other shape by name.
    if timing_reps.is_some() {
        let modes = bridge_name.is_some()
            || observable_str.is_some()
            || fermionic_str.is_some()
            || hamiltonian_path.is_some()
            || gradient_str.is_some()
            || gradient_fn_str.is_some();
        // EXPLICIT --device only, as `--dump-state-npy` requires: left to
        // OMEGA_DEVICE, a requested-but-not-compiled-in CUDA falls back to CPU
        // with a one-line notice and the timing line would carry a CPU time
        // that a hand-run benchmark could publish under a CUDA label.
        let device_ok = matches!(
            requested_device,
            Some(omega_core::device::DeviceKind::Cpu) | Some(omega_core::device::DeviceKind::Cuda)
        );
        if modes
            || shots.is_some()
            || noise_model.is_some()
            || !matches!(chosen, "statevector" | "sv")
            || !device_ok
        {
            eprintln!(
                "--timing-reps times the plain statevector evolution only: it needs \
                 --statevector, --backend statevector, no --noise / mode flag, and an \
                 EXPLICIT --device cpu or --device cuda (it cannot be left to \
                 OMEGA_DEVICE, which falls back silently)."
            );
            std::process::exit(1);
        }
    }

    // Resolve the compute device: `requested_device` was parsed once, above
    // the flag-scope gate. Honours OMEGA_DEVICE if --device wasn't passed;
    // falls back to CPU with a single stderr notice when the requested device
    // isn't compiled in. The statevector arms and the MPS hooks read it; Pauli
    // / Photonics stay CPU.
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
    let mut qudit_cert: Option<omega_backend_quditsv::QuditSvCertificate> = None;
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
                                Ok(b) => timed_execute(timing_reps, "cuda-f32", || {
                                    b.execute(&circuit, &params, &config)
                                }),
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
                                    refuse_device_fallback("cuda", &msg);
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
                                    refuse_device_fallback("opencl", &msg);
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
                                    refuse_device_fallback("metal", &msg);
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
                    timed_execute(timing_reps, "cpu-f64", || {
                        StatevectorBackend::new().execute(&circuit, &params, &config)
                    })
                }
            }
            m if omega_backend_mps::select::is_mps(m) => {
                // Both MPS samplers take the same device hooks (the device
                // gate above says MPS dispatches to CUDA, and that must be
                // true of the `--noise` arm too, or an explicit device is
                // reported and not used).
                executed_arm = "mps-cpu";
                #[cfg(feature = "cuda")]
                if matches!(resolved_device, omega_core::device::DeviceKind::Cuda) {
                    executed_arm = "mps-cpu+cuda-svd-hook";
                }
                // The f32 contraction is not the production path. Name it in
                // the artifact only when the caller opted in and was warned
                // that discarded_weight is not a bound. The default metal
                // request stays `mps-cpu`: that is the path that ran.
                #[cfg(all(feature = "metal", not(feature = "cuda")))]
                if matches!(resolved_device, omega_core::device::DeviceKind::Metal)
                    && mps_metal_contract_opt_in()
                {
                    executed_arm = "mps-cpu+metal-contract-hook";
                }
                if let Some(model) = &noise_model {
                    info(format!("Noise model: {:?}", model));
                    let chi = match omega_backend_mps::select::parse_mps(chosen) {
                        Ok(Some(omega_backend_mps::select::MpsSelect::Fixed { chi })) => chi,
                        Ok(Some(omega_backend_mps::select::MpsSelect::Auto { max_chi })) => max_chi,
                        _ => omega_backend_mps::select::DEFAULT_CHI,
                    };
                    apply_mps_device_hooks(
                        omega_backend_mps::NoisyMpsBackend::with_model(chi, model.clone()),
                        device_name.as_deref(),
                    )
                    .execute(&circuit, &params, &config)
                } else {
                    // Hold the concrete backend so its truncation certificate
                    // can be reported to stderr (stdout/exit unchanged — K14).
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
            // `--noise` is already refused above for every backend but
            // statevector/mps, so no second check here.
            "sector" => SectorBackend::new().execute(&circuit, &params, &config),
            "quditsv" | "qudit" => {
                match QuditSvBackend::new().execute_with_certificate(&circuit, &params, &config) {
                    Ok((r, cert)) => {
                        qudit_cert = Some(cert);
                        Ok(r)
                    }
                    Err(e) => Err(e),
                }
            }
            "pauliprop" | "pp" => {
                eprintln!(
                    "pauliprop is an expectation-value backend; it has no sampling/\
                     statevector output. Use it with --expectation \"<Pauli string>\"."
                );
                std::process::exit(1);
            }
            "majoranaprop" | "mp" => {
                eprintln!(
                    "majoranaprop is an expectation-value backend; it has no sampling/\
                     statevector output. Use it with --expectation \"<Pauli string>\"."
                );
                std::process::exit(1);
            }
            "stabrank" | "sr" => {
                eprintln!(
                    "stabrank is an expectation-value backend; it has no sampling/\
                     statevector output. Use it with --expectation \"<Pauli string>\" \
                     or --expectation-fermionic \"<ladder string>\". For shots on a \
                     Clifford circuit use --backend pauli."
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
                         mps:auto, mps:auto:<ceiling>, pauli, pauliprop, sector, photonics.{}",
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
    // Before any output: an explicit `--device cuda` MPS run that fell back
    // is refused here, not after its result has been printed.
    let cuda_svd = report_cuda_svd_dispatch(device_name.is_some());

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
        if dump_state_npy {
            if let Err(e) = write_state_npy(path, amps) {
                eprintln!("--dump-state-npy: cannot write {path}: {e}");
                std::process::exit(1);
            }
            info(format!(
                "state ({executed_arm}, {mc}) written to {path} as complex128 .npy"
            ));
        } else {
            match serialize::state_bits_to_json(&circuit_name, executed_arm, mc, &os_version, amps)
            {
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
    }

    // Benchmark mode prints no result: rendering 2^n amplitudes as text is
    // not the evolution, and it would set the process's peak RSS (measured:
    // 183 MiB for a 16 MiB state at n = 20), which the comparison reports.
    // The state is available through --dump-state-npy.
    if timing_reps.is_some() {
        info("--timing-reps: result not printed (benchmark mode)".to_string());
        return;
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
            if let Some(c) = &cuda_svd {
                v = serialize::attach_cuda_svd_dispatch(
                    v,
                    c.gpu_calls,
                    c.cpu_fallbacks,
                    c.last_fallback,
                );
            }
            if let Some(cert) = &qudit_cert {
                v = serialize::attach_quditsv_certificate(v, cert);
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
                if let Some(c) = &cuda_svd {
                    v = serialize::attach_cuda_svd_dispatch(
                        v,
                        c.gpu_calls,
                        c.cpu_fallbacks,
                        c.last_fallback,
                    );
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

/// What an expectation run measures, from whichever spelling was given.
///
/// `--expectation` names a Pauli sum and is measured as written.
/// `--expectation-fermionic` names a ladder sum (`0.5 [1^ 3] + 0.5 [3^ 1]`),
/// and `--hamiltonian` names the same kind of sum read from an FCIDUMP.
/// No backend measures a ladder sum directly: it is lowered here, once,
/// through `jordan_wigner()`, and from then on the run is the SAME run as a
/// Pauli one — same dispatch, same certificates. That is the whole design
/// (PLAN-FERMIONIC.md F2): zero engine changes, because the mapping is exact
/// and every backend already does the right thing with a Pauli sum carrying
/// Z strings and an identity term.
///
/// `mapping` is `Some` only for fermionic input. It goes into the JSON so a
/// reader can re-derive the number without this binary: the basis the text
/// was in, the mapper, the term count after mapping, and the mapped Pauli
/// spelling itself in `--expectation`'s own grammar — feeding that string back
/// through `--expectation` must give the same value, and the CLI test does.
struct ExpectationObservable {
    /// The text as typed — what the `observable` JSON key echoes.
    label: String,
    observable: Observable,
    mapping: Option<serde_json::Value>,
    /// The ladder operator itself, kept for the one backend that measures
    /// it without the Pauli detour: majoranaprop seeds its Majorana sum
    /// straight from this (PLAN-FERMIONIC F3). `None` for Pauli input.
    fermionic: Option<omega_core::fermion::FermionicOp>,
}

fn expectation_from_fermionic(
    op: omega_core::fermion::FermionicOp,
    label: String,
    input_basis: &str,
    flag: &str,
) -> ExpectationObservable {
    let observable = op.jordan_wigner().unwrap_or_else(|e| {
        eprintln!("{flag}: {e}");
        std::process::exit(1);
    });
    let mapped_observable = observable
        .terms
        .iter()
        .map(|(c, paulis)| {
            let string = if paulis.is_empty() {
                "I0".to_string()
            } else {
                paulis
                    .iter()
                    .map(|(q, p)| format!("{p:?}{q}"))
                    .collect::<String>()
            };
            format!("{c}*{string}")
        })
        .collect::<Vec<_>>()
        .join("+");
    ExpectationObservable {
        label,
        mapping: Some(serde_json::json!({
            "input_basis": input_basis,
            "mapping": "jordan_wigner",
            "mapped_terms": observable.terms.len(),
            "mapped_observable": mapped_observable,
        })),
        observable,
        fermionic: Some(op),
    }
}

fn expectation_observable(
    pauli: Option<&str>,
    fermionic: Option<&str>,
    hamiltonian: Option<&str>,
) -> Option<ExpectationObservable> {
    if let Some(path) = hamiltonian {
        let text = fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("--hamiltonian: cannot read {path}: {e}");
            std::process::exit(1);
        });
        let op = omega_core::fcidump::parse(&text).unwrap_or_else(|e| {
            eprintln!("--hamiltonian: {e}");
            std::process::exit(1);
        });
        return Some(expectation_from_fermionic(
            op,
            path.to_string(),
            "fcidump",
            "--hamiltonian",
        ));
    }
    if let Some(text) = fermionic {
        // The parser and the mapper each refuse with the offending token or
        // coefficient named; forward that verbatim under the flag's name.
        let op = omega_core::fermion::FermionicOp::parse(text).unwrap_or_else(|e| {
            eprintln!("--expectation-fermionic: {e}");
            std::process::exit(1);
        });
        return Some(expectation_from_fermionic(
            op,
            text.to_string(),
            "ladder",
            "--expectation-fermionic",
        ));
    }
    pauli.map(|text| ExpectationObservable {
        label: text.to_string(),
        observable: parse_observable(text),
        mapping: None,
        fermionic: None,
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

#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod mps_metal_certificate {
    use super::apply_mps_device_hooks;
    use omega_backend_mps::MpsBackend;
    use omega_core::executor::{Backend, Observable};
    use omega_core::params::ParameterBinding;

    /// Brickwall deep enough that the middle bond saturates χ = 32 and the
    /// run truncates. Below the Metal threshold, or with nothing discarded,
    /// reinstalling the f32 hook would leave `discarded_weight` unchanged and
    /// this test could not go red.
    fn truncating_brickwall() -> omega_core::circuit::CircuitIR {
        let n = 12usize;
        let depth = 12usize;
        let mut src = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\n");
        for q in 0..n {
            src.push_str(&format!("h q[{q}];\n"));
        }
        for d in 0..depth {
            for q in 0..n {
                let ang = 0.17 + 0.01 * (q + d * n) as f64;
                src.push_str(&format!("ry({ang}) q[{q}];\n"));
                src.push_str(&format!("rz(0.23) q[{q}];\n"));
            }
            let mut q = d % 2;
            while q + 1 < n {
                src.push_str(&format!("cx q[{q}], q[{}];\n", q + 1));
                q += 2;
            }
        }
        omega_parser::lower_to_ir(&src).expect("brickwall lowers")
    }

    fn discarded_bits(
        device: Option<&str>,
        circuit: &omega_core::circuit::CircuitIR,
    ) -> (u64, f64, usize) {
        let backend = apply_mps_device_hooks(
            MpsBackend::new(32).with_max_discarded_weight(f64::INFINITY),
            device,
        );
        let obs = Observable::parse("Z0").expect("Z0");
        backend
            .expectation(circuit, &ParameterBinding::new(), &obs)
            .expect("mps expectation");
        let st = backend.last_run_stats();
        (
            st.discarded_weight.to_bits(),
            st.discarded_weight,
            st.max_bond_reached,
        )
    }

    #[test]
    fn device_metal_discarded_weight_matches_cpu_exactly() {
        // Production default. A set variable would install the f32 hook and
        // this test would be measuring the opt-in, which is allowed to differ.
        // SAFETY: test-only, and no other test in this harness reads these.
        unsafe {
            std::env::remove_var("MPS_METAL_CONTRACT");
            std::env::remove_var("MPS_METAL_MIN_BOND");
        }

        let circuit = truncating_brickwall();
        let (cpu_bits, cpu_dw, cpu_bond) = discarded_bits(Some("cpu"), &circuit);
        let (metal_bits, metal_dw, metal_bond) = discarded_bits(Some("metal"), &circuit);
        // The bare backend is the CPU path with no device arm at all. Matching
        // it, not merely matching `--device cpu`, is what goes red if the metal
        // arm installs `metal_contract_2q` again.
        let (bare_bits, bare_dw, _) = discarded_bits(None, &circuit);

        assert!(
            cpu_bond >= 32,
            "fixture max_bond_reached={cpu_bond} never reached MIN_BOND_DIM_FOR_METAL; \
             reinstalling the f32 hook would not change discarded_weight"
        );
        assert!(
            cpu_dw > 0.0,
            "fixture discarded nothing ({cpu_dw}); both paths report zero and the hook \
             would not redden this test"
        );
        assert_eq!(
            cpu_bits, bare_bits,
            "cpu device path {cpu_dw} != bare CPU {bare_dw}"
        );
        assert_eq!(
            metal_bits, cpu_bits,
            "production --device metal discarded_weight {metal_dw} != CPU {cpu_dw} \
             (bond metal {metal_bond}, cpu {cpu_bond}). The f32 contraction hook is \
             on the production path, and a smaller certificate is not a bound."
        );
    }
}

#[cfg(test)]
mod cuda_svd_verdict_tests {
    use super::{cuda_svd_verdict, CudaSvdCounts};

    const REASON: &str = "cuSOLVER gesvd reported info != 0 (non-convergence)";

    #[test]
    fn a_clean_run_prints_its_counts_and_is_not_refused() {
        let c = CudaSvdCounts {
            gpu_calls: 156,
            cpu_fallbacks: 0,
            last_fallback: None,
        };
        for explicit in [false, true] {
            let (line, refusal) = cuda_svd_verdict(c, explicit);
            assert_eq!(line, "cuda svd: 156 on gpu, 0 cpu fallbacks");
            assert_eq!(refusal, None);
        }
    }

    #[test]
    fn an_explicit_device_refuses_any_fallback_naming_counts_and_reason() {
        let c = CudaSvdCounts {
            gpu_calls: 135,
            cpu_fallbacks: 93,
            last_fallback: Some(REASON),
        };
        let (line, refusal) = cuda_svd_verdict(c, true);
        assert_eq!(
            line,
            format!("cuda svd: 135 on gpu, 93 cpu fallbacks (last: {REASON})")
        );
        assert_eq!(
            refusal.as_deref(),
            Some(
                "93 of 228 MPS bond-compression SVDs fell back to the CPU \
                 (last: cuSOLVER gesvd reported info != 0 (non-convergence))"
            )
        );
    }

    #[test]
    fn automatic_selection_reports_a_fallback_but_runs() {
        let c = CudaSvdCounts {
            gpu_calls: 0,
            cpu_fallbacks: 1,
            last_fallback: Some(REASON),
        };
        let (line, refusal) = cuda_svd_verdict(c, false);
        assert!(line.ends_with(&format!("1 cpu fallbacks (last: {REASON})")));
        assert_eq!(refusal, None);
    }
}
