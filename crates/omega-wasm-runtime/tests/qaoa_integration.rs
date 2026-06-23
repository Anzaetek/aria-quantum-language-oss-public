//! Integration test: run the QAOA WASM guest with a MaxCut problem.

use omega_core::qaoa::qaoa_circuit;
use omega_core::qubo::Qubo;
use omega_wasm_runtime::host::HostState;
use omega_wasm_runtime::WasmRunner;

/// Build a simple MaxCut QUBO on a 3-node triangle graph.
///
/// MaxCut objective: maximise edges crossing the cut.
/// For edge (i,j): contribution = x_i(1-x_j) + x_j(1-x_i) = x_i + x_j - 2·x_i·x_j
/// Minimising negative of this: -x_i - x_j + 2·x_i·x_j per edge.
fn maxcut_triangle() -> Qubo {
    let mut q = Qubo::new(3);
    // Edges: (0,1), (1,2), (0,2)
    for &(i, j) in &[(0, 1), (1, 2), (0, 2)] {
        q.add(i, i, -1.0); // -x_i
        q.add(j, j, -1.0); // -x_j
        q.add(i, j, 2.0); // +2·x_i·x_j
    }
    q
}

#[test]
fn test_qaoa_wasm_guest() {
    // 1. Build MaxCut problem → Ising → QAOA circuit + observable
    let qubo = maxcut_triangle();
    let ising = qubo.to_ising();
    let circuit = qaoa_circuit(&ising, 1); // depth-1 QAOA, 2 params (gamma, beta)
    let observable = ising.to_observable();

    // Verify brute-force optimum for sanity
    let (_best, best_val) = qubo.brute_force();
    assert!(
        best_val < 0.0,
        "MaxCut QUBO optimum should be negative, got {best_val}"
    );

    // 2. Register with host
    let mut host = HostState::new();
    let cid = host.register_circuit(circuit);
    let oid = host.register_observable(observable);
    assert_eq!(cid, 1);
    assert_eq!(oid, 1);

    // 3. Load QAOA WASM binary
    let wasm_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/wasm-guests/qaoa/target/wasm32-wasip1/release/qaoa.wasm");

    if !wasm_path.exists() {
        eprintln!(
            "Skipping QAOA integration test: WASM binary not found at {}",
            wasm_path.display()
        );
        eprintln!(
            "Build it with: cd examples/wasm-guests/qaoa && cargo build --target wasm32-wasip1 --release"
        );
        return;
    }

    let wasm_bytes = std::fs::read(&wasm_path).unwrap();

    // 4. Run with generous fuel
    let runner = WasmRunner::new(host).unwrap();
    let result = runner.run(&wasm_bytes, 100_000_000_000).unwrap();

    // 5. Verify convergence
    match result {
        Some(res) => {
            println!("QAOA converged:");
            println!("  Optimal cost:   {:.8}", res.optimal_value);
            println!("  Optimal params: {:?}", res.optimal_params);
            println!("  Iterations:     {}", res.iterations);

            // The QAOA should find a cost value — we don't require it to beat
            // brute force, but it should show meaningful optimisation.
            // For a triangle MaxCut the optimal cut is 2 (any 2-vs-1 partition).
            // The Ising energy at optimum is shifted by the QUBO offset.
            assert!(
                res.iterations > 0,
                "QAOA should have run at least one iteration"
            );
            // Triangle MaxCut brute-force optimum is -2; QAOA p=1 with the
            // best-seen iterate should land within ~25% of it. Looser than
            // theory allows so noise/seed drift won't flake, but tight
            // enough to catch a regression in the optimizer (e.g. reporting
            // the final iterate of an oscillating GD instead of the best).
            assert!(
                res.optimal_value <= -1.5,
                "QAOA optimum suspiciously poor: {} (expected ≤ -1.5)",
                res.optimal_value
            );
        }
        None => {
            // No final result — check progress instead
            let state = runner.host_state().lock().unwrap();
            if !state.progress.is_empty() {
                let last = state.progress.last().unwrap();
                println!(
                    "QAOA made progress: {} iterations, last cost = {}",
                    last.0, last.1
                );
            } else {
                panic!("QAOA produced no result and no progress");
            }
        }
    }
}

/// End-to-end smoke for the parametrized qaoa.wasm guest: the guest reads
/// `optimizer: "adam"` and `max_iters: 80` from the staged input JSON,
/// allocates 4 parameters (depth=2 QAOA on the triangle), and converges.
/// Regression for the NUM_PARAMS=2 cap that this work removed.
#[test]
fn test_qaoa_wasm_guest_depth2_adam_via_input_json() {
    let qubo = maxcut_triangle();
    let ising = qubo.to_ising();
    let circuit = qaoa_circuit(&ising, 2); // depth-2 → 4 free params
    let observable = ising.to_observable();
    assert_eq!(circuit.symbols.len(), 4);

    let mut host = HostState::new();
    host.register_circuit(circuit);
    host.register_observable(observable);
    host.set_input(
        br#"{"num_params":4,"max_iters":80,"lr":0.15,"optimizer":"adam","init":[0.5,0.5,0.3,0.3]}"#
            .to_vec(),
    );

    let wasm_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/wasm-guests/qaoa/target/wasm32-wasip1/release/qaoa.wasm");
    if !wasm_path.exists() {
        eprintln!("Skipping: qaoa.wasm not built");
        return;
    }
    let wasm_bytes = std::fs::read(&wasm_path).unwrap();
    let runner = WasmRunner::new(host).unwrap();
    let result = runner.run(&wasm_bytes, 100_000_000_000).unwrap();

    let res = result.expect("guest never reported a result");
    assert_eq!(res.optimal_params.len(), 4);
    // Triangle MaxCut Ising optimum (after offset folding) is -2; depth-2
    // QAOA + Adam should land below -1.8 reliably.
    assert!(
        res.optimal_value < -1.8,
        "depth-2 QAOA via in-WASM Adam did not converge: {}",
        res.optimal_value
    );
}

/// Smoke for the qaoa_qubo guest: pass a QUBO matrix as the input
/// payload, the guest builds the circuit on the host via
/// `omega_qaoa_from_qubo`, runs SPSA inside WASM, reports back.
#[test]
fn test_qaoa_qubo_wasm_guest_runtime_circuit_build() {
    let mut host = HostState::new();
    // Triangle QUBO. Brute-force optimum is -2 (Ising) / -3.5 (QUBO).
    let qubo_json = r#"{
        "qubo": "{\"n\":3,\"Q\":[[0,0,-2],[1,1,-2],[2,2,-2],[0,1,2],[1,2,2],[0,2,2]]}",
        "depth": 2,
        "max_iters": 150,
        "seed": 42
    }"#;
    host.set_input(qubo_json.as_bytes().to_vec());

    let wasm_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/wasm-guests/qaoa_qubo/target/wasm32-wasip1/release/qaoa_qubo.wasm");
    if !wasm_path.exists() {
        eprintln!(
            "Skipping qaoa_qubo integration: WASM binary not found at {}",
            wasm_path.display()
        );
        eprintln!(
            "Build it: cd examples/wasm-guests/qaoa_qubo && cargo build --target wasm32-wasip1 --release"
        );
        return;
    }
    let wasm_bytes = std::fs::read(&wasm_path).unwrap();
    let runner = WasmRunner::new(host).unwrap();
    let result = runner.run(&wasm_bytes, 100_000_000_000).unwrap();

    let res = result.expect("qaoa_qubo guest never reported a result");
    // The guest registered exactly one circuit + observable on the host
    // via omega_qaoa_from_qubo — verify they exist and the param count
    // matches depth=2.
    {
        let state = runner.host_state().lock().unwrap();
        assert_eq!(
            state.circuits.len(),
            1,
            "guest should have built one circuit"
        );
        assert_eq!(state.observables.len(), 1);
        let circuit = state.circuits.values().next().unwrap();
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.symbols.len(), 4); // depth=2 → 2*depth = 4 free params
    }
    assert_eq!(res.optimal_params.len(), 4);
    // SPSA on triangle MaxCut should reach Ising cost ≤ -1.5 in 150 iters.
    assert!(
        res.optimal_value < -1.5,
        "qaoa_qubo SPSA did not converge: {}",
        res.optimal_value
    );
}

/// Same `qaoa_qubo` guest as above, but on a 4-node K4 MaxCut. Proves the
/// "circuit built inside WASM via `omega_qaoa_from_qubo`" pattern scales:
/// the host pre-registers nothing, the guest reads the QUBO from input,
/// asks the host to build a depth-2 QAOA circuit + observable for it, then
/// runs SPSA entirely in WASM.
///
/// K4 brute-force optimum: any 2/2 partition cuts all 4 across-edges →
/// MaxCut = 4 → QUBO objective = -4. The observable returned by the host
/// already folds the Ising→QUBO offset in as an identity term (see
/// `IsingModel::to_observable` in omega-core/src/qubo.rs), so the guest's
/// `optimal_value` is the QUBO objective directly. SPSA at depth=2 with
/// 250 iters reliably reaches QUBO ≤ -3.85 from a near-zero start.
#[test]
fn test_qaoa_qubo_wasm_guest_k4_maxcut() {
    let mut host = HostState::new();
    // K4 MaxCut: each node in 3 edges → diag = -3; each off-diag edge = +2.
    let qubo_json = r#"{
        "qubo": "{\"n\":4,\"Q\":[[0,0,-3],[1,1,-3],[2,2,-3],[3,3,-3],[0,1,2],[0,2,2],[0,3,2],[1,2,2],[1,3,2],[2,3,2]]}",
        "depth": 2,
        "max_iters": 250,
        "seed": 1234567
    }"#;
    host.set_input(qubo_json.as_bytes().to_vec());

    let wasm_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/wasm-guests/qaoa_qubo/target/wasm32-wasip1/release/qaoa_qubo.wasm");
    if !wasm_path.exists() {
        eprintln!(
            "Skipping qaoa_qubo K4 integration: WASM binary not found at {}",
            wasm_path.display()
        );
        eprintln!(
            "Build it: cd examples/wasm-guests/qaoa_qubo && cargo build --target wasm32-wasip1 --release"
        );
        return;
    }
    let wasm_bytes = std::fs::read(&wasm_path).unwrap();
    let runner = WasmRunner::new(host).unwrap();
    let result = runner.run(&wasm_bytes, 100_000_000_000).unwrap();

    let res = result.expect("qaoa_qubo guest never reported a result");

    // Build the matching QUBO host-side and confirm the K4 brute-force
    // optimum is exactly -4 (any 2/2 partition).
    let mut qubo = Qubo::new(4);
    for i in 0..4 {
        qubo.set(i, i, -3.0);
    }
    for &(i, j) in &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        qubo.set(i, j, 2.0);
    }
    let (best_assignment, best_qubo) = qubo.brute_force();
    assert!(
        (best_qubo - (-4.0)).abs() < 1e-9,
        "K4 MaxCut brute-force optimum should be -4, got {best_qubo} at {best_assignment:?}"
    );
    // Population-count check: the optimum should be a 2/2 partition.
    let popcount = best_assignment.iter().filter(|b| **b).count();
    assert_eq!(popcount, 2, "best assignment should be a 2/2 partition");

    // Guest built a depth-2 QAOA on 4 qubits → 4 free params, single
    // (circuit, observable) registered.
    {
        let state = runner.host_state().lock().unwrap();
        assert_eq!(
            state.circuits.len(),
            1,
            "guest should have built exactly one circuit"
        );
        assert_eq!(state.observables.len(), 1);
        let circuit = state.circuits.values().next().unwrap();
        assert_eq!(circuit.num_qubits, 4);
        assert_eq!(circuit.symbols.len(), 4); // depth=2 → 2*depth params
    }
    assert_eq!(res.optimal_params.len(), 4);

    // optimal_value ≥ brute_force optimum (expectation of an observable can't
    // dip below its smallest eigenvalue) and SPSA at depth=2 on K4 should
    // close most of the gap from a near-zero random-init start.
    assert!(
        res.optimal_value >= best_qubo - 1e-9,
        "optimal_value {} is below brute-force minimum {} (eigenvalue floor)",
        res.optimal_value,
        best_qubo
    );
    let gap_to_opt = res.optimal_value - best_qubo;
    assert!(
        gap_to_opt < 0.25,
        "qaoa_qubo SPSA on K4 stalled too far from optimum: \
         qubo={:.4}, brute-force={:.4}, gap={:.4}",
        res.optimal_value,
        best_qubo,
        gap_to_opt
    );
}

#[test]
fn test_qaoa_fuel_exhaustion() {
    let qubo = maxcut_triangle();
    let ising = qubo.to_ising();
    let circuit = qaoa_circuit(&ising, 1);
    let observable = ising.to_observable();

    let mut host = HostState::new();
    host.register_circuit(circuit);
    host.register_observable(observable);

    let wasm_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/wasm-guests/qaoa/target/wasm32-wasip1/release/qaoa.wasm");

    if !wasm_path.exists() {
        eprintln!("Skipping fuel exhaustion test: WASM binary not found");
        return;
    }

    let wasm_bytes = std::fs::read(&wasm_path).unwrap();
    let runner = WasmRunner::new(host).unwrap();

    // Run with very low fuel — should return an error, not panic
    let result = runner.run(&wasm_bytes, 1000);
    assert!(result.is_err(), "should fail with insufficient fuel");
    let err_msg = format!("{}", result.unwrap_err());
    println!("Fuel exhaustion error (expected): {}", err_msg);
}

/// Pure-WASM QAOA on a QUBO: the `qaoa_qubo_self` guest assembles the
/// QAOA(depth) QASM and the Ising Pauli-sum observable inside WASM, then
/// registers them with `omega_register_qasm` / `omega_register_observable_str`
/// and runs SPSA. The host does parser + circuit evaluation only — no
/// `omega_qaoa_from_qubo`.
///
/// Same K4 MaxCut as `test_qaoa_qubo_wasm_guest_k4_maxcut` so the two
/// guests can be compared head-to-head: both must converge to the same
/// QUBO objective on the same problem, even though the circuits are
/// constructed by different code paths (host helper vs. in-WASM).
#[test]
fn test_qaoa_qubo_self_wasm_guest_builds_circuit_inside_wasm() {
    let mut host = HostState::new();
    let qubo_json = r#"{
        "qubo": "{\"n\":4,\"Q\":[[0,0,-3],[1,1,-3],[2,2,-3],[3,3,-3],[0,1,2],[0,2,2],[0,3,2],[1,2,2],[1,3,2],[2,3,2]]}",
        "depth": 2,
        "max_iters": 250,
        "seed": 1234567
    }"#;
    host.set_input(qubo_json.as_bytes().to_vec());

    let wasm_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(
            "examples/wasm-guests/qaoa_qubo_self/target/wasm32-wasip1/release/qaoa_qubo_self.wasm",
        );
    if !wasm_path.exists() {
        eprintln!(
            "Skipping qaoa_qubo_self integration: WASM binary not found at {}",
            wasm_path.display()
        );
        eprintln!(
            "Build it: cd examples/wasm-guests/qaoa_qubo_self && cargo build --target wasm32-wasip1 --release"
        );
        return;
    }
    let wasm_bytes = std::fs::read(&wasm_path).unwrap();
    let runner = WasmRunner::new(host).unwrap();
    let result = runner.run(&wasm_bytes, 100_000_000_000).unwrap();

    let res = result.expect("qaoa_qubo_self guest never reported a result");

    // The host must have ended up with exactly one circuit + one
    // observable, both registered by the GUEST (not pre-staged by the
    // test). Param count = 2*depth.
    {
        let state = runner.host_state().lock().unwrap();
        assert_eq!(
            state.circuits.len(),
            1,
            "guest must register exactly one circuit"
        );
        assert_eq!(state.observables.len(), 1);
        let circuit = state.circuits.values().next().unwrap();
        assert_eq!(circuit.num_qubits, 4);
        assert_eq!(circuit.symbols.len(), 4); // 2 * depth = 4 (gamma_0,beta_0,gamma_1,beta_1)
    }
    assert_eq!(res.optimal_params.len(), 4);

    // Brute-force check + convergence assertion. The QUBO observable
    // includes the offset, so optimal_value is the QUBO objective directly.
    let mut qubo = Qubo::new(4);
    for i in 0..4 {
        qubo.set(i, i, -3.0);
    }
    for &(i, j) in &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        qubo.set(i, j, 2.0);
    }
    let (_best, best_qubo) = qubo.brute_force();
    assert!((best_qubo - (-4.0)).abs() < 1e-9);
    assert!(
        res.optimal_value >= best_qubo - 1e-9,
        "optimal_value {} below brute-force min {}",
        res.optimal_value,
        best_qubo
    );
    assert!(
        res.optimal_value - best_qubo < 0.25,
        "qaoa_qubo_self SPSA on K4 stalled too far from optimum: \
         qubo={:.4}, brute-force={:.4}, gap={:.4}",
        res.optimal_value,
        best_qubo,
        res.optimal_value - best_qubo
    );
}
