//! Integration test: run the VQE WASM guest with a real circuit.

use omega_core::circuit::*;
use omega_wasm_runtime::host::{h2_hamiltonian, HostState};
use omega_wasm_runtime::WasmRunner;
use smallvec::smallvec;

/// Create a simple 2-qubit ansatz: Ry(θ₀)|0⟩⊗Ry(θ₁)|0⟩ followed by CNOT.
fn make_vqe_ansatz() -> CircuitIR {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.symbols.insert(0, "theta0".to_string());
    c.symbols.insert(1, "theta1".to_string());

    // Ry(θ₀) on qubit 0
    c.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });
    // Ry(θ₁) on qubit 1
    c.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(1)],
        params: smallvec![ParamExpr::Symbol(1)],
        classical_bit: None,
        condition: None,
    });
    // CNOT
    c.add_op(GateOp {
        gate: GateKind::CX,
        qubits: smallvec![Qubit(0), Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });

    c
}

#[test]
fn test_vqe_wasm_guest() {
    // Set up host state with circuit and observable
    let mut host = HostState::new();
    let circuit = make_vqe_ansatz();
    let observable = h2_hamiltonian();
    let cid = host.register_circuit(circuit);
    let oid = host.register_observable(observable);
    assert_eq!(cid, 1);
    assert_eq!(oid, 1);

    // Load VQE WASM binary
    let wasm_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/wasm-guests/vqe/target/wasm32-wasip1/release/vqe.wasm");

    if !wasm_path.exists() {
        eprintln!(
            "Skipping VQE integration test: WASM binary not found at {}",
            wasm_path.display()
        );
        eprintln!("Build it with: cd examples/wasm-guests/vqe && cargo build --target wasm32-wasip1 --release");
        return;
    }

    let wasm_bytes = std::fs::read(&wasm_path).unwrap();

    // Run with generous fuel
    let runner = WasmRunner::new(host).unwrap();
    let result = runner.run(&wasm_bytes, 100_000_000_000).unwrap();

    // Check that we got a result
    match result {
        Some(res) => {
            println!("VQE converged:");
            println!("  Optimal energy: {:.8}", res.optimal_value);
            println!("  Optimal params: {:?}", res.optimal_params);
            println!("  Iterations: {}", res.iterations);

            // The H2 Hamiltonian ground state energy should be around -0.8 to -1.0
            // (depending on the ansatz expressibility)
            assert!(
                res.optimal_value < 0.0,
                "VQE should find negative energy, got {}",
                res.optimal_value
            );
        }
        None => {
            // Check progress
            let state = runner.host_state().lock().unwrap();
            if !state.progress.is_empty() {
                let last = state.progress.last().unwrap();
                println!(
                    "VQE made progress: {} iterations, last E = {}",
                    last.0, last.1
                );
                assert!(last.1 < 1.0, "should have reduced energy");
            } else {
                panic!("VQE produced no result and no progress");
            }
        }
    }
}

/// End-to-end smoke for the parametrized vqe.wasm: 4-parameter ansatz,
/// in-WASM Adam, no NUM_PARAMS=2 truncation. Regression for the cap
/// removal — if the guest were still hard-coded to NUM_PARAMS=2 it would
/// silently optimise only theta0,theta1 and the final params vector
/// would be length 2.
#[test]
fn test_vqe_wasm_guest_4_params_via_input_json() {
    use omega_core::circuit::*;
    use omega_core::executor::Observable;
    use smallvec::smallvec;

    // 4-qubit hardware-efficient ansatz with 4 free parameters.
    let mut circuit = CircuitIR::new(4, CircuitType::GateBased);
    for i in 0..4u32 {
        circuit.symbols.insert(i, format!("theta{}", i));
        circuit.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(i)],
            params: smallvec![ParamExpr::Symbol(i)],
            classical_bit: None,
            condition: None,
        });
    }
    for (a, b) in [(0u32, 1), (1, 2), (2, 3)] {
        circuit.add_op(GateOp {
            gate: GateKind::CX,
            qubits: smallvec![Qubit(a), Qubit(b)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }
    let observable = Observable::parse("0.5*Z0+0.3*Z1Z2+0.2*X2X3").unwrap();

    let mut host = HostState::new();
    host.register_circuit(circuit);
    host.register_observable(observable);
    host.set_input(
        br#"{"num_params":4,"max_iters":300,"lr":0.1,"optimizer":"adam","init":[0.5,0.3,-0.2,0.4]}"#
            .to_vec(),
    );

    let wasm_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/wasm-guests/vqe/target/wasm32-wasip1/release/vqe.wasm");
    if !wasm_path.exists() {
        eprintln!("Skipping: vqe.wasm not built");
        return;
    }
    let wasm_bytes = std::fs::read(&wasm_path).unwrap();
    let runner = WasmRunner::new(host).unwrap();
    let result = runner.run(&wasm_bytes, 100_000_000_000).unwrap();

    let res = result.expect("guest never reported a result");
    assert_eq!(
        res.optimal_params.len(),
        4,
        "in-WASM Adam should optimize all 4 free params (no NUM_PARAMS=2 cap)"
    );
    // 4-param Adam should land below -0.5 on this fairly easy landscape.
    assert!(
        res.optimal_value < -0.5,
        "in-WASM Adam did not make meaningful progress: {}",
        res.optimal_value
    );
}
