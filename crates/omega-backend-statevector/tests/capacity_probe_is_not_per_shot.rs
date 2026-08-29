// SPDX-License-Identifier: Apache-2.0
//! **The host-memory probe must not run once per shot.**
//!
//! `capacity::check` runs once per TRAJECTORY, and it asks the platform how
//! much memory is available. That probe is a `vm_stat` SUBPROCESS on macOS and
//! a `/proc/meminfo` read on Linux, so a 1000-shot mid-circuit-measurement run
//! forked `vm_stat` a thousand times.
//!
//! Measured before the fix: **1.05 s for 1000 shots, and the identical 1.05 s
//! for a 2-qubit/4-op circuit as for a 5-qubit/28-op one.** A cost that does not
//! move when the simulation gets seven times bigger is not simulation cost. The
//! oversubscribe escape hatch, which returns before the probe, took the same
//! run to 0.00 s — which is what identified the probe rather than the arithmetic.
//!
//! # Why this counts probes instead of measuring time
//!
//! A timing assertion here would be a benchmark: it fails on a loaded machine,
//! on a slow runner, under a debugger, and it says nothing about the cause. The
//! defect is structural — a per-run question asked per shot — so the test
//! states the structure. `hostmem::probes()` is a counter incremented only when
//! the platform is genuinely consulted, so this asserts the shape directly and
//! is immune to how fast the box is.
//!
//! This is the same reason the pauliprop light-cone claim is pinned by an insert
//! COUNT rather than a speedup: a count is reproducible on any hardware and a
//! ratio is not.
//!
//! # What would make this test vacuous
//!
//! If `available_bytes()` ever returns `None` early on some platform without
//! probing, the count stays at zero and this passes without measuring anything.
//! So the test first asserts the probe fires AT ALL, and only then that it does
//! not fire per shot. A zero-probe run fails as loudly as a per-shot one.

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::hostmem;
use omega_core::params::ParameterBinding;

fn op(gate: GateKind, qubits: &[u32], classical_bit: Option<u32>) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
        params: Default::default(),
        classical_bit,
        condition: None,
    }
}

/// `H q0; measure q0 -> c0; reset q0; measure q1 -> c1` — stochastic, so it
/// takes the per-trajectory path, and tiny, so any cost that survives is
/// overhead rather than arithmetic.
fn stochastic_circuit() -> CircuitIR {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.num_classical_bits = 2;
    c.ops.push(op(GateKind::H, &[0], None));
    c.ops.push(op(GateKind::Measure, &[0], Some(0)));
    c.ops.push(op(GateKind::Reset, &[0], None));
    c.ops.push(op(GateKind::Measure, &[1], Some(1)));
    c
}

const SHOTS: u32 = 500;

#[test]
fn the_host_memory_probe_does_not_run_once_per_shot() {
    let be = StatevectorBackend::new();
    let circuit = stochastic_circuit();
    let params = ParameterBinding::new();

    // Confirm the probe is reachable at all, so a platform that never probes
    // cannot pass this test by doing nothing.
    //
    // `available_bytes()` serves a PROCESS-GLOBAL cache with a 250 ms TTL,
    // and the sibling test in this binary runs concurrently and probes too.
    // The original guard called `available_bytes()` immediately and demanded
    // a fresh probe — which fails whenever the sibling warmed the cache
    // within the TTL, i.e. precisely on a QUIET machine where both tests are
    // fast. Measured 2026-08-20: green on every loaded run all day, red on
    // the first idle one. So: wait out the TTL first, then ask; and accept
    // ANY counter movement in the window as proof the counter is wired —
    // whether this thread's call probed or the sibling's did, the property
    // under test ("the platform is genuinely consulted and the counter
    // counts it") is established either way.
    let before_warm = hostmem::probes();
    std::thread::sleep(std::time::Duration::from_millis(300)); // > CACHE_TTL
    let _ = hostmem::available_bytes();
    let warmed = hostmem::probes();

    let cfg = ExecConfig {
        shots: Some(SHOTS),
        seed: Some(11),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let before = hostmem::probes();
    let out = be.execute(&circuit, &params, &cfg).expect("execute");
    let probes = hostmem::probes() - before;

    // The run must actually have happened.
    match out {
        ExecResult::Counts(c) => {
            let total: u32 = c.values().sum();
            assert_eq!(total, SHOTS, "expected {SHOTS} shots, got {total}");
        }
        other => panic!("expected Counts, got {other:?}"),
    }

    assert!(
        warmed > before_warm,
        "hostmem::probes() never incremented, so this test measures nothing on \
         this platform. Either `available_bytes()` is returning early without \
         probing, or the counter is no longer wired to the probe."
    );
    // 16, not 4: the counter is process-global and the sibling test probes
    // concurrently (once per TTL through its multi-second run), so a handful
    // of ITS probes can land inside this window. The defect this guards is
    // ~one probe per shot — 500 — and the bound only needs to sit far below
    // that while tolerating a neighbour's cadence.
    assert!(
        probes <= 16,
        "the host-memory probe ran {probes} times for {SHOTS} shots. It is a \
         `vm_stat` subprocess on macOS and a /proc/meminfo read on Linux, and \
         it answers a question about the HOST that cannot change during a shot \
         loop. Per-shot probing cost 1.05 s per 1000 shots and was independent \
         of circuit size. Check the cache in `omega_core::hostmem`."
    );
}

/// The fix must not become "sample once and replicate".
///
/// Replicating one trajectory is fast and, on a circuit whose outcome is
/// deterministic, indistinguishable from correct — the repetition-code circuit
/// that exposed the slowdown returns `111000000` on every shot either way. So a
/// deterministic circuit cannot guard this, and a plausible "optimisation"
/// would sail past it.
///
/// This circuit's mid-circuit measurement is a genuine coin flip. One
/// trajectory replicated 4000 times yields a single key with all the weight;
/// independent trajectories yield both keys near 50/50. The gap is total, so no
/// tolerance tuning is involved.
///
/// This is not hypothetical: the doc comment in `sim.rs` records that
/// `H q0; measure q0 -> c0; when c0 == 1 { X q1 }` once returned `|00>` on
/// 4000/4000 shots — a superposition measured with certainty — and that Aer
/// gives a 50/50 split on the same input.
#[test]
fn shots_remain_independent_trajectories_not_one_replicated() {
    let be = StatevectorBackend::new();
    let circuit = stochastic_circuit();
    let params = ParameterBinding::new();
    let shots = 4000u32;

    let cfg = ExecConfig {
        shots: Some(shots),
        seed: Some(2024),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let ExecResult::Counts(counts) = be.execute(&circuit, &params, &cfg).expect("execute") else {
        panic!("expected Counts");
    };

    let total: u32 = counts.values().sum();
    assert_eq!(total, shots);
    assert_eq!(
        counts.len(),
        2,
        "expected both outcomes of a fair coin, got {} distinct key(s): {:?}. \
         One key means every shot replayed a single trajectory.",
        counts.len(),
        counts
    );
    // 4000 fair flips: 5 sigma is ~158, so 800 is enormous slack and still
    // nowhere near the 4000/0 a replicated trajectory produces.
    for (key, n) in &counts {
        assert!(
            (*n as i64 - 2000).abs() < 800,
            "outcome {key:?} occurred {n}/{shots} times; a fair mid-circuit \
             measurement should split near 50/50"
        );
    }
}
