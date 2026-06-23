//! Integration tests for mid-circuit measurement, conditional gates, and execution modes.

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::*;
use omega_core::executor::*;
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

fn collapse_config(seed: u64) -> ExecConfig {
    ExecConfig {
        shots: None,
        seed: Some(seed),
        mid_circuit_mode: MidCircuitMode::Collapse,
    }
}

fn make_measure_op(qubit: u32, cbit: u32) -> GateOp {
    GateOp {
        gate: GateKind::Measure,
        qubits: smallvec![Qubit(qubit)],
        params: smallvec![],
        classical_bit: Some(cbit),
        condition: None,
    }
}

fn make_conditional_x(qubit: u32, cbit: u32, expected: u64) -> GateOp {
    GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(qubit)],
        params: smallvec![],
        classical_bit: None,
        condition: Some((cbit, 1, expected)),
    }
}

// ---- Mid-circuit measurement ----

#[test]
fn test_measure_zero_state() {
    // |0⟩ → measure → always 0
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
    circuit.num_classical_bits = 1;
    circuit.add_op(make_measure_op(0, 0));

    // Run many times — should always give |0⟩
    for seed in 0..100 {
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &collapse_config(seed))
            .unwrap();
        let sv = result.statevector();
        assert!((sv[0].re - 1.0).abs() < 1e-10, "seed {seed}: should be |0⟩");
        assert!(sv[1].norm() < 1e-10);
    }
}

#[test]
fn test_measure_one_state() {
    // X|0⟩ = |1⟩ → measure → always 1
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
    circuit.num_classical_bits = 1;
    circuit.add_op(GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    circuit.add_op(make_measure_op(0, 0));

    for seed in 0..100 {
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &collapse_config(seed))
            .unwrap();
        let sv = result.statevector();
        assert!(
            sv[0].norm() < 1e-10,
            "seed {seed}: |0⟩ amplitude should be 0"
        );
        assert!(
            (sv[1].norm() - 1.0).abs() < 1e-10,
            "seed {seed}: should be |1⟩"
        );
    }
}

#[test]
fn test_measure_superposition() {
    // H|0⟩ → measure → ~50/50
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
    circuit.num_classical_bits = 1;
    circuit.add_op(GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    circuit.add_op(make_measure_op(0, 0));

    let mut zeros = 0;
    let trials = 1000;
    for seed in 0..trials {
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &collapse_config(seed))
            .unwrap();
        let sv = result.statevector();
        if sv[0].norm_sqr() > 0.5 {
            zeros += 1;
        }
    }

    // Should be roughly 50%
    assert!(
        zeros > 400 && zeros < 600,
        "expected ~50% zeros, got {zeros}/{trials}"
    );
}

#[test]
fn test_measure_bell_collapse() {
    // Bell state: H q0, CX q0→q1, measure q0
    // If q0 collapses to |0⟩ → q1 must also be |0⟩
    // If q0 collapses to |1⟩ → q1 must also be |1⟩
    let backend = StatevectorBackend::new();

    for seed in 0..200 {
        let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
        circuit.num_classical_bits = 1;
        circuit.add_op(GateOp {
            gate: GateKind::H,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::CX,
            qubits: smallvec![Qubit(0), Qubit(1)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(make_measure_op(0, 0));

        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &collapse_config(seed))
            .unwrap();
        let sv = result.statevector();

        // Only |00⟩ or |11⟩ should have non-zero amplitude
        let p00 = sv[0].norm_sqr();
        let p01 = sv[1].norm_sqr();
        let p10 = sv[2].norm_sqr();
        let p11 = sv[3].norm_sqr();

        assert!(p01 < 1e-10, "seed {seed}: |01⟩ should be 0");
        assert!(p10 < 1e-10, "seed {seed}: |10⟩ should be 0");
        assert!(
            (p00 > 0.99) || (p11 > 0.99),
            "seed {seed}: should be either |00⟩ or |11⟩, got p00={p00:.4} p11={p11:.4}"
        );
    }
}

// ---- Conditional gates ----

#[test]
fn test_conditional_gate() {
    // H q0 → measure q0 → c0 → if(c0==1) X q1
    // When c0=0: q1 stays |0⟩, state is |00⟩
    // When c0=1: q1 flipped, state is |11⟩
    let backend = StatevectorBackend::new();

    for seed in 0..200 {
        let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
        circuit.num_classical_bits = 1;
        circuit.add_op(GateOp {
            gate: GateKind::H,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(make_measure_op(0, 0));
        circuit.add_op(make_conditional_x(1, 0, 1));

        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &collapse_config(seed))
            .unwrap();
        let sv = result.statevector();

        let p00 = sv[0].norm_sqr();
        let p11 = sv[3].norm_sqr();

        // Should be |00⟩ or |11⟩ (never |01⟩ or |10⟩)
        assert!(sv[1].norm_sqr() < 1e-10, "seed {seed}: |01⟩ should be 0");
        assert!(sv[2].norm_sqr() < 1e-10, "seed {seed}: |10⟩ should be 0");
        assert!(
            (p00 > 0.99) || (p11 > 0.99),
            "seed {seed}: should be |00⟩ or |11⟩"
        );
    }
}

#[test]
fn test_selective_reset() {
    // X q0 → measure q0 → c0 → if(c0==1) X q0 → result should always be |0⟩
    let backend = StatevectorBackend::new();

    for seed in 0..50 {
        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.num_classical_bits = 1;
        circuit.add_op(GateOp {
            gate: GateKind::X,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(make_measure_op(0, 0));
        circuit.add_op(make_conditional_x(0, 0, 1));

        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &collapse_config(seed))
            .unwrap();
        let sv = result.statevector();
        assert!(
            (sv[0].re - 1.0).abs() < 1e-10,
            "seed {seed}: selective reset should give |0⟩"
        );
    }
}

// ---- Execution modes ----

#[test]
fn test_midcircuit_skip_mode() {
    // H q0 → measure q0 → with Skip mode, measure has no effect
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
    circuit.num_classical_bits = 1;
    circuit.add_op(GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    circuit.add_op(make_measure_op(0, 0));

    let config = ExecConfig {
        shots: None,
        seed: Some(42),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let result = backend
        .execute(&circuit, &ParameterBinding::new(), &config)
        .unwrap();
    let sv = result.statevector();

    // With Skip, the statevector should still be |+⟩ = (|0⟩+|1⟩)/√2
    let expected = 1.0 / 2.0_f64.sqrt();
    assert!(
        (sv[0].re - expected).abs() < 1e-10,
        "Skip mode: |0⟩ amplitude should be 1/√2"
    );
    assert!(
        (sv[1].re - expected).abs() < 1e-10,
        "Skip mode: |1⟩ amplitude should be 1/√2"
    );
}

#[test]
fn test_midcircuit_collapse_mode() {
    // H q0 → measure q0 → with Collapse mode, state collapses
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
    circuit.num_classical_bits = 1;
    circuit.add_op(GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    circuit.add_op(make_measure_op(0, 0));

    let result = backend
        .execute(&circuit, &ParameterBinding::new(), &collapse_config(42))
        .unwrap();
    let sv = result.statevector();

    // With Collapse, should be either |0⟩ or |1⟩ (not superposition)
    let p0 = sv[0].norm_sqr();
    let p1 = sv[1].norm_sqr();
    assert!(
        (p0 > 0.99) || (p1 > 0.99),
        "Collapse mode: should be |0⟩ or |1⟩, not superposition (p0={p0:.4} p1={p1:.4})"
    );
}

// ---- Backward compatibility ----

#[test]
fn test_backward_compat_default_config() {
    // Default ExecConfig should use Skip mode
    let config = ExecConfig::default();
    assert_eq!(config.mid_circuit_mode, MidCircuitMode::Skip);
}

// ---- Multi-bit creg conditional gate ----

#[test]
fn test_multi_bit_creg_condition_fires_on_match() {
    // `creg c[2]; if(c == 2) ...` — value 2 means c[0]=0, c[1]=1
    // (LSB-first). Manually stage classical_bits via a sequence
    // of measurement ops that produce a deterministic c value of
    // 2 (binary 10), then a conditional X that should fire.
    //
    // Setup: q0 = |0⟩, q1 = X|0⟩ = |1⟩.  Measure q0 → c[0] (=0),
    // measure q1 → c[1] (=1). c reads as 2 (binary 10). Then
    // `if(c == 2) X q2` should fire, leaving q2 in |1⟩.
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(3, CircuitType::GateBased);
    circuit.num_classical_bits = 2;
    circuit.add_op(GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    circuit.add_op(make_measure_op(0, 0));
    circuit.add_op(make_measure_op(1, 1));
    // Conditional X on q2: fire when c == 2.
    circuit.add_op(GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(2)],
        params: smallvec![],
        classical_bit: None,
        condition: Some((0, 2, 2)),
    });

    let result = backend
        .execute(&circuit, &ParameterBinding::new(), &collapse_config(42))
        .unwrap();
    let sv = result.statevector();
    // q0 = 0, q1 = 1, q2 = 1 → basis state |110⟩ in MSB-first
    // qubit-bit ordering = index 0b110 = 6.
    let p6 = sv[6].norm_sqr();
    assert!(
        p6 > 0.99,
        "multi-bit creg condition should fire: expected |110⟩ ≈ 1, got p6={p6}"
    );
}

#[test]
fn test_multi_bit_creg_condition_skips_on_mismatch() {
    // Same shape as above but condition expects c == 1, while the
    // measurement leaves c == 2. The conditional X should NOT fire.
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(3, CircuitType::GateBased);
    circuit.num_classical_bits = 2;
    circuit.add_op(GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    circuit.add_op(make_measure_op(0, 0));
    circuit.add_op(make_measure_op(1, 1));
    circuit.add_op(GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(2)],
        params: smallvec![],
        classical_bit: None,
        condition: Some((0, 2, 1)), // expects c == 1; c is actually 2
    });

    let result = backend
        .execute(&circuit, &ParameterBinding::new(), &collapse_config(42))
        .unwrap();
    let sv = result.statevector();
    // Conditional X should be skipped → q2 stays |0⟩. State is
    // |010⟩ MSB-first = index 0b010 = 2.
    let p2 = sv[2].norm_sqr();
    assert!(
        p2 > 0.99,
        "multi-bit creg condition should skip: expected |010⟩ ≈ 1, got p2={p2}"
    );
}

#[test]
fn test_pre_existing_condition_satisfied_helper() {
    // Direct unit test of the helper added in omega-core::circuit.
    // Ensures the LSB-first multi-bit assembly is correct.
    let op = GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: Some((1, 3, 5)), // bits 1..4 LSB-first == 5 (binary 101)
    };
    // classical_bits[1..4] = [1, 0, 1] → assembled as 1 + 0*2 + 1*4 = 5
    let bits = vec![0u8, 1, 0, 1, 0];
    assert!(op.condition_satisfied(&bits));

    // Same shape but bits = [1, 0, 0, 1, 0] → assembled as 1 + 0*2 + 0*4 = 1 ≠ 5
    let bits = vec![0u8, 1, 0, 0, 0];
    assert!(!op.condition_satisfied(&bits));

    // No condition → always satisfied.
    let op_none = GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    };
    assert!(op_none.condition_satisfied(&[]));
    assert!(op_none.condition_satisfied(&[0, 1, 1, 1]));
}

// ---- Shot counts keyed by creg state (not measurement history) ----

#[test]
fn test_collapse_counts_keyed_by_creg_overwrite() {
    // measure q0 -> c0; X q1; measure q1 -> c0
    // Both measurements write c0. Qiskit semantics: last-write-wins on the
    // creg, so after the run c0 == 1 (q1's outcome). Counts must be keyed
    // by the creg state, not by qubit basis index — the latter would key
    // by both q0 and q1's measurement outcomes.
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
    circuit.num_classical_bits = 1;
    // q0 starts |0⟩ → measure gives 0.
    circuit.add_op(make_measure_op(0, 0));
    // X q1 → |1⟩
    circuit.add_op(GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    // measure q1 -> c0, overwrites c0 with 1.
    circuit.add_op(make_measure_op(1, 0));

    let cfg = ExecConfig {
        shots: Some(1),
        seed: Some(7),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let result = backend
        .execute(&circuit, &ParameterBinding::new(), &cfg)
        .unwrap();
    let counts = result.counts();
    assert_eq!(counts.len(), 1, "single trajectory → single bucket");
    let (&key, &count) = counts.iter().next().unwrap();
    assert_eq!(count, 1);
    // Key must be the creg state c0 = 1, not the 2-qubit basis index 0b10 = 2.
    assert_eq!(
        key, 1,
        "counts must key by creg state (c0=1), not qubit basis (q0=0, q1=1 → 0b10)"
    );
}

#[test]
fn test_collapse_counts_creg_state_distribution() {
    // H q0 → measure q0 -> c0. Over many trajectories the creg-keyed
    // histogram converges to ~50/50 on {0, 1}, the standard
    // Hadamard-then-measure outcome.
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
    circuit.num_classical_bits = 1;
    circuit.add_op(GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    circuit.add_op(make_measure_op(0, 0));

    let mut n0 = 0u32;
    let mut n1 = 0u32;
    for seed in 0..2000u64 {
        let cfg = ExecConfig {
            shots: Some(1),
            seed: Some(seed.wrapping_mul(2654435761)),
            mid_circuit_mode: MidCircuitMode::Collapse,
        };
        let r = backend
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .unwrap();
        let c = r.counts();
        for (&k, &v) in c {
            match k {
                0 => n0 += v,
                1 => n1 += v,
                other => panic!("unexpected creg key {other}"),
            }
        }
    }
    let total = (n0 + n1) as f64;
    assert_eq!(total as u32, 2000);
    let p0 = n0 as f64 / total;
    assert!(
        (p0 - 0.5).abs() < 0.05,
        "Hadamard-then-measure creg distribution skewed: p0={p0}"
    );
}
