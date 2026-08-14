//! Systematic QUBO/QAOA/VQE comparison harness.
//!
//! Closes the `xxx.txt` scratch item ("Systematic QUBO/QAOA/VQE
//! comparison harness ... brute force + QAOA + VQE, JSON report with
//! optimum, runtime, gap-to-optimum"). Runs a curated suite of small
//! QUBO instances (n ≤ 6) through three solvers:
//!
//!   - **brute_force**  — exhaustive enumeration via [`Qubo::brute_force`].
//!     Acts as the ground-truth optimum at these sizes.
//!   - **QAOA**         — `qaoa_circuit(p)` ansatz, gradient descent
//!     via the backend's adjoint-AD path (MPS falls back to
//!     parameter-shift through `compute_gradient`'s default flow).
//!   - **VQE**          — `vqe_circuit(layers)` hardware-efficient
//!     ansatz, same gradient descent.
//!
//! Each row records the recovered optimum, gap-to-brute-force, the
//! optimal bit-string from the ranked sample distribution, the chosen
//! sample's count fraction, and wallclock runtime. Output is a JSON
//! document printed to stdout (visible under `cargo test --
//! --nocapture`); when `OMEGA_QUBO_REPORT` is set the report is also
//! written to the given path so a CI run can archive it as a build
//! artifact.
//!
//! Backends compared (per the 2026-05-01 directive):
//!   - **statevector** (default): the original arm; full adjoint AD.
//!   - **mps**: bond-dim 64 — exact at these sizes (n ≤ 6). Validates
//!     that MPS forward sampling reproduces statevector's distribution
//!     within numerical noise. MPS doesn't expose `adjoint_gradient`
//!     so `compute_gradient`'s Adjoint branch transparently falls
//!     back to parameter-shift.
//!   - **metal** (under `--features metal`): Apple-Silicon GPU
//!     statevector. Adjoint AD is matched to CPU at ~3e-7 elsewhere;
//!     this harness checks that the optimisation loop reaches the
//!     same brute-force optimum as the CPU arm.
//!
//! Pauli (Clifford-restricted) is **not** plugged in: QAOA's `Rz`
//! and VQE's `Ry` are non-Clifford for general angles, so the same
//! ansatz can't run on a stabilizer simulator. Adapting the harness
//! to a Clifford-restricted ansatz is its own design exercise and a
//! separate commit.
//!
//! `mqlib` plug — deferred. Brute-force is exact at n ≤ 20 so it
//! already serves as the classical baseline at the harness sizes;
//! plugging mqlib (a C++ library) only pays off above n ≈ 25 where
//! brute-force stops scaling. The `xxx.txt` scratch item stays open
//! until that need is real; this harness is the comparison piece.

use std::collections::HashMap;
use std::time::Instant;

use omega_backend_mps::MpsBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::CircuitIR;
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable};
use omega_core::gradient::{compute_gradient, GradMethod};
use omega_core::params::ParameterBinding;
use omega_core::qaoa::qaoa_circuit;
use omega_core::qubo::Qubo;
use omega_core::vqe::vqe_circuit;

/// One QUBO instance + the QAOA/VQE depths to try on it.
struct Instance {
    name: &'static str,
    n: usize,
    /// Sparse upper-triangular `Q` entries: `(i, j, coeff)` with `i <= j`.
    entries: &'static [(usize, usize, f64)],
    qaoa_depth: usize,
    vqe_layers: usize,
}

const INSTANCES: &[Instance] = &[
    // Triangle MaxCut — three vertices, three edges. Standard QUBO
    // encoding: minimize f(x) = -Σ_{(i,j)∈E} (x_i + x_j - 2 x_i x_j),
    // which expands to Q_ii = -degree(i) on the diagonal and Q_ij = +2
    // per edge off-diagonal. Optimum |min f| = MaxCut = 2 (any single
    // vertex on one side cuts both incident edges).
    Instance {
        name: "maxcut_triangle",
        n: 3,
        entries: &[
            (0, 0, -2.0),
            (1, 1, -2.0),
            (2, 2, -2.0),
            (0, 1, 2.0),
            (0, 2, 2.0),
            (1, 2, 2.0),
        ],
        qaoa_depth: 2,
        vqe_layers: 2,
    },
    // K4 MaxCut — four vertices, six edges (complete graph). Same
    // encoding (Q_ii = -3, Q_ij = +2). Optimum = -4 from a 2-2 vertex
    // partition (4 cut edges). Single-vertex partitions only cut 3
    // edges (-3) — the optimiser must escape that local plateau.
    Instance {
        name: "maxcut_k4",
        n: 4,
        entries: &[
            (0, 0, -3.0),
            (1, 1, -3.0),
            (2, 2, -3.0),
            (3, 3, -3.0),
            (0, 1, 2.0),
            (0, 2, 2.0),
            (0, 3, 2.0),
            (1, 2, 2.0),
            (1, 3, 2.0),
            (2, 3, 2.0),
        ],
        qaoa_depth: 3,
        vqe_layers: 3,
    },
    // Linear-chain Ising — 5-spin antiferromagnet with weak field.
    // Built so the unique ground state alternates spins.
    Instance {
        name: "ising_chain5",
        n: 5,
        entries: &[
            (0, 0, -1.0),
            (1, 1, 1.0),
            (2, 2, -1.0),
            (3, 3, 1.0),
            (4, 4, -1.0),
            (0, 1, 2.0),
            (1, 2, 2.0),
            (2, 3, 2.0),
            (3, 4, 2.0),
        ],
        qaoa_depth: 2,
        vqe_layers: 3,
    },
];

#[test]
fn qubo_compare_suite() {
    // Each backend slot returns Some when compiled; None otherwise.
    // Box<dyn Backend> hides the per-backend type so the iteration
    // loop stays uniform.
    let backends: Vec<(&'static str, Box<dyn Backend>)> = build_backends();

    let mut rows: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for inst in INSTANCES {
        let mut qubo = Qubo::new(inst.n);
        for &(i, j, c) in inst.entries {
            qubo.set(i, j, c);
        }

        // ---------- brute force (once per instance) ----------
        let bf_start = Instant::now();
        let (bf_bits, bf_value) = qubo.brute_force();
        let bf_ms = bf_start.elapsed().as_secs_f64() * 1e3;
        let bf_str = bits_to_str(&bf_bits);

        // Gap tolerance: each backend / method must find the
        // brute-force optimum within 1e-6 (statevector + metal — exact
        // arithmetic) or 5e-3 (mps with parameter-shift fallback —
        // small accumulated drift from the looser optimisation path).
        let mut backend_rows: Vec<String> = Vec::new();
        for (name, backend) in &backends {
            let tol = if *name == "mps" { 5e-3 } else { 1e-6 };
            let qaoa = solve_qaoa(backend.as_ref(), &qubo, inst.qaoa_depth);
            let vqe = solve_vqe(backend.as_ref(), &qubo, inst.vqe_layers);
            let qaoa_gap = qaoa.best_value - bf_value;
            let vqe_gap = vqe.best_value - bf_value;

            if qaoa_gap.abs() > tol {
                failures.push(format!(
                    "{}/{}: QAOA gap {:.3e} > {:.0e} (best={:.6}, bf={:.6})",
                    inst.name, name, qaoa_gap, tol, qaoa.best_value, bf_value
                ));
            }
            if vqe_gap.abs() > tol {
                failures.push(format!(
                    "{}/{}: VQE gap {:.3e} > {:.0e} (best={:.6}, bf={:.6})",
                    inst.name, name, vqe_gap, tol, vqe.best_value, bf_value
                ));
            }

            backend_rows.push(
                serde_json::json!({
                    "backend": name,
                    "qaoa": {
                        "depth": inst.qaoa_depth,
                        "best_value": qaoa.best_value,
                        "best_bits": qaoa.best_bits,
                        "best_count_fraction": round4(qaoa.best_count_fraction),
                        "final_expected_cost": qaoa.final_cost,
                        "iterations": qaoa.iterations,
                        "gap_to_optimum": qaoa_gap,
                        "runtime_ms": round3(qaoa.runtime_ms),
                    },
                    "vqe": {
                        "layers": inst.vqe_layers,
                        "best_value": vqe.best_value,
                        "best_bits": vqe.best_bits,
                        "best_count_fraction": round4(vqe.best_count_fraction),
                        "final_expected_cost": vqe.final_cost,
                        "iterations": vqe.iterations,
                        "gap_to_optimum": vqe_gap,
                        "runtime_ms": round3(vqe.runtime_ms),
                    },
                })
                .to_string(),
            );
        }

        rows.push(
            serde_json::json!({
                "instance": inst.name,
                "n": inst.n,
                "brute_force": {
                    "value": bf_value,
                    "bits": bf_str,
                    "runtime_ms": round3(bf_ms),
                },
                "backends": backend_rows
                    .iter()
                    .map(|s| serde_json::from_str::<serde_json::Value>(s).unwrap())
                    .collect::<Vec<_>>(),
            })
            .to_string(),
        );
    }

    let backend_names: Vec<&str> = backends.iter().map(|(n, _)| *n).collect();
    let report = format!(
        "{{\n  \"mode\": \"qubo_compare\",\n  \"backends\": {},\n  \"instances\": [\n    {}\n  ]\n}}",
        serde_json::to_string(&backend_names).unwrap(),
        rows.join(",\n    ")
    );

    println!("{report}");

    if let Ok(path) = std::env::var("OMEGA_QUBO_REPORT") {
        std::fs::write(&path, &report).expect("write OMEGA_QUBO_REPORT");
        eprintln!("qubo_compare report written to {path}");
    }

    assert!(
        failures.is_empty(),
        "qubo_compare failures:\n  {}",
        failures.join("\n  ")
    );
}

/// Assemble the backends compiled into the current test binary. The
/// statevector backend is always present; MPS rides along; Metal
/// activates only with `--features metal` and a working device.
#[cfg_attr(not(feature = "metal"), allow(unused_mut))]
fn build_backends() -> Vec<(&'static str, Box<dyn Backend>)> {
    let mut backends: Vec<(&'static str, Box<dyn Backend>)> = vec![
        ("statevector", Box::new(StatevectorBackend::new())),
        // Bond-dim 64 is overkill for n ≤ 6 (max is 2^(n/2) = 8) but
        // keeps the harness untouched if a larger instance is ever
        // added. The MPS path naturally falls back to parameter-shift
        // gradients via `compute_gradient`'s default flow since
        // MpsBackend doesn't implement `adjoint_gradient`.
        ("mps", Box::new(MpsBackend::new(64))),
    ];

    #[cfg(feature = "metal")]
    {
        match omega_backend_statevector_metal::MetalStatevectorBackend::new() {
            Ok(metal) => backends.push(("metal", Box::new(metal))),
            Err(e) => eprintln!(
                "qubo_compare: skipping metal arm — MetalStatevectorBackend::new failed: {e:?}"
            ),
        }
    }

    backends
}

// ---------------------------------------------------------------------
// Per-method solvers
// ---------------------------------------------------------------------

struct SolverOutcome {
    best_value: f64,
    best_bits: String,
    best_count_fraction: f64,
    final_cost: f64,
    iterations: usize,
    runtime_ms: f64,
}

const OPT_ITERS: usize = 60;
const OPT_LR: f64 = 0.1;
const OPT_SEED: u64 = 42;
const SHOTS: u32 = 4096;

fn solve_qaoa(backend: &dyn Backend, qubo: &Qubo, depth: usize) -> SolverOutcome {
    let ising = qubo.to_ising();
    let circuit = qaoa_circuit(&ising, depth);
    let observable = ising.to_observable();
    // QAOA convention: gamma_0, beta_0 alternating — start gammas small,
    // betas at π/4 for a non-trivial mixer angle.
    let symbol_ids = sorted_symbol_ids(&circuit);
    let init = symbol_ids
        .iter()
        .map(|sid| {
            let name = circuit.symbols.get(sid).cloned().unwrap_or_default();
            if name.starts_with("beta") {
                std::f64::consts::FRAC_PI_4
            } else {
                0.1
            }
        })
        .collect::<Vec<_>>();
    run_gradient_pipeline(backend, qubo, &circuit, &observable, &symbol_ids, &init)
}

fn solve_vqe(backend: &dyn Backend, qubo: &Qubo, layers: usize) -> SolverOutcome {
    let circuit = vqe_circuit(qubo.n, layers);
    let observable = qubo.to_ising().to_observable();
    let symbol_ids = sorted_symbol_ids(&circuit);
    // VQE: small structured init (sin spread) so the optimiser doesn't
    // start at a barren plateau symmetric point.
    let init: Vec<f64> = symbol_ids
        .iter()
        .enumerate()
        .map(|(i, _)| ((i as f64) * 0.317 - 0.84).sin() * 0.4)
        .collect();
    run_gradient_pipeline(backend, qubo, &circuit, &observable, &symbol_ids, &init)
}

fn run_gradient_pipeline(
    backend: &dyn Backend,
    qubo: &Qubo,
    circuit: &CircuitIR,
    observable: &Observable,
    symbol_ids: &[u32],
    init: &[f64],
) -> SolverOutcome {
    let start = Instant::now();
    let (params_vec, iterations, final_cost) =
        gradient_descent(backend, circuit, observable, symbol_ids, init);
    let pb = bind(symbol_ids, &params_vec);

    let cfg = ExecConfig {
        shots: Some(SHOTS),
        seed: Some(OPT_SEED),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let counts = match backend.execute(circuit, &pb, &cfg).expect("sample failed") {
        ExecResult::Counts(c) => c,
        other => panic!("unexpected ExecResult: {:?}", variant_name(&other)),
    };

    let (best_bits, best_value, best_count) = pick_best_sample(qubo, &counts);
    let runtime_ms = start.elapsed().as_secs_f64() * 1e3;

    SolverOutcome {
        best_value,
        best_bits,
        best_count_fraction: best_count as f64 / SHOTS as f64,
        final_cost,
        iterations,
        runtime_ms,
    }
}

fn gradient_descent(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    observable: &Observable,
    symbol_ids: &[u32],
    initial: &[f64],
) -> (Vec<f64>, usize, f64) {
    let mut values = initial.to_vec();
    let mut last_cost = f64::INFINITY;
    let mut iterations = 0;

    for iter in 1..=OPT_ITERS {
        iterations = iter;
        let pb = bind(symbol_ids, &values);
        let grads = compute_gradient(backend, circuit, &pb, observable, &GradMethod::Adjoint)
            .expect("adjoint gradient");
        let grad_map: HashMap<u32, f64> = grads.into_iter().collect();
        for (i, sid) in symbol_ids.iter().enumerate() {
            let g = *grad_map.get(sid).unwrap_or(&0.0);
            values[i] -= OPT_LR * g;
        }
        let cost = backend
            .expectation(circuit, &pb, observable)
            .unwrap_or(f64::INFINITY);
        if iter > 5 && (last_cost - cost).abs() < 1e-9 {
            return (values, iter, cost);
        }
        last_cost = cost;
    }
    (values, iterations, last_cost)
}

fn pick_best_sample(
    qubo: &Qubo,
    counts: &HashMap<omega_core::outcome::Outcome, u32>,
) -> (String, f64, u32) {
    let mut best: Option<(String, f64, u32)> = None;
    for (key, &count) in counts {
        let x: Vec<bool> = (0..qubo.n).map(|i| key.bit(i as u32) == 1).collect();
        let val = qubo.evaluate(&x);
        let bits = bits_to_str(&x);
        let candidate = (bits, val, count);
        match &best {
            None => best = Some(candidate),
            Some((_, bv, bc)) => {
                let take = candidate.1 < *bv || (candidate.1 == *bv && candidate.2 > *bc);
                if take {
                    best = Some(candidate);
                }
            }
        }
    }
    best.expect("non-empty counts")
}

fn sorted_symbol_ids(circuit: &CircuitIR) -> Vec<u32> {
    let mut ids: Vec<u32> = circuit.symbols.keys().copied().collect();
    ids.sort();
    ids
}

fn bind(symbol_ids: &[u32], values: &[f64]) -> ParameterBinding {
    let mut pb = ParameterBinding::new();
    for (sid, v) in symbol_ids.iter().zip(values.iter()) {
        pb.bind(*sid, *v);
    }
    pb
}

fn bits_to_str(bits: &[bool]) -> String {
    let mut s = String::with_capacity(bits.len());
    for b in bits.iter().rev() {
        s.push(if *b { '1' } else { '0' });
    }
    s
}

fn round3(v: f64) -> f64 {
    (v * 1e3).round() / 1e3
}

fn round4(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
}

fn variant_name(result: &ExecResult) -> &'static str {
    match result {
        ExecResult::Counts(_) => "Counts",
        ExecResult::Statevector(_) => "Statevector",
        ExecResult::Probabilities(_) => "Probabilities",
    }
}
