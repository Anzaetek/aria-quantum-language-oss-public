//! What actually happens today on the Metal backend when a shots-mode circuit
//! contains a `Reset`.
//!
//! `STATUS.md` §5.3 used to say "Metal per-shot GPU trajectories block at ~64
//! shots — in-flight command-buffer exhaustion. `execute` delegates to CPU
//! meanwhile." Measured on an M4 2026-09-25, that described code which is no
//! longer in the tree: `execute` returns `StatevectorBackend::new().execute(..)`
//! whenever `shots.is_some()` and the circuit contains a Reset, so the
//! trajectory loop is unreachable and nothing blocks at any shot count through
//! 65536. The entry was rewritten around what this probe found instead.
//!
//! The delegation is observable without parsing stderr or timing it. On the
//! calling thread, `reset_shots_cpu_fallback_count` moves when this fallback
//! runs and `metal_shots_execute_count` moves when `execute` samples on the
//! GPU. A shots-mode call moves exactly one. This probe prints that label on
//! every row and refuses a row whose counters disagree with the arm — a
//! counts-only check would stay green if the counters were deleted.
//!
//! THE SLOPE IS STILL THE COST. The delegation is O(shots); the true GPU path
//! is O(1), because it evolves once and samples every shot in one dispatch.
//! Sweep shots and look at the slope: linear is the CPU, flat is the GPU.
//! The no-Reset control arm exists so the contrast is measured rather than
//! asserted, and its flatness is CHECKED (see `PLAN-OPEN` §3f A10 — a harness
//! for an accelerated path must fail when that path did not execute, not
//! report it). The Metal counter is the same check in a form load cannot
//! fake.

use std::time::Instant;

use omega_backend_statevector_metal::{
    metal_shots_execute_count, reset_shots_cpu_fallback_count, MetalStatevectorBackend,
};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;

fn push(c: &mut CircuitIR, gate: GateKind, qs: Vec<u32>) {
    c.ops.push(GateOp {
        gate,
        qubits: qs.into_iter().map(Qubit).collect(),
        // `GateOp.params` is a SmallVec, not a Vec.
        params: vec![].into(),
        classical_bit: None,
        condition: None,
    });
}

fn bell(with_reset: bool) -> CircuitIR {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    push(&mut c, GateKind::H, vec![0]);
    push(&mut c, GateKind::CX, vec![0, 1]);
    if with_reset {
        push(&mut c, GateKind::Reset, vec![0]);
    }
    c
}

fn main() {
    let params = ParameterBinding::new();
    let backend = match MetalStatevectorBackend::new() {
        Ok(b) => b,
        Err(e) => {
            println!("Metal backend unavailable: {e:?} — nothing to probe.");
            return;
        }
    };
    println!("Metal statevector backend constructed OK\n");

    const SHOTS: [u32; 10] = [16, 32, 64, 96, 128, 256, 1024, 4096, 16384, 65536];
    let mut control: Vec<(u32, f64)> = Vec::new();

    for with_reset in [true, false] {
        let label = if with_reset {
            "WITH Reset (delegation path — expect O(shots))"
        } else {
            "NO Reset (true GPU path — expect O(1))"
        };
        println!("=== {label} ===");
        let circuit = bell(with_reset);
        for &shots in &SHOTS {
            let cfg = ExecConfig {
                shots: Some(shots),
                seed: Some(0),
                mid_circuit_mode: MidCircuitMode::Skip,
            };
            let cpu_before = reset_shots_cpu_fallback_count();
            let metal_before = metal_shots_execute_count();
            let t = Instant::now();
            let res = backend.execute(&circuit, &params, &cfg);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let cpu_delta = reset_shots_cpu_fallback_count() - cpu_before;
            let metal_delta = metal_shots_execute_count() - metal_before;
            let device = match (cpu_delta, metal_delta) {
                (1, 0) => "cpu",
                (0, 1) => "metal",
                _ => "neither",
            };
            let expect = if with_reset { "cpu" } else { "metal" };
            if !with_reset {
                control.push((shots, ms));
            }
            match res {
                Ok(ExecResult::Counts(m)) => {
                    let total: u32 = m.values().sum();
                    println!(
                        "  shots {shots:>6}: OK  {ms:>9.2} ms  device={device:<6} distinct={:<3} total={total}",
                        m.len()
                    );
                }
                Ok(other) => {
                    println!(
                        "  shots {shots:>6}: OK  {ms:>9.2} ms  device={device:<6} (non-counts: {other:?})"
                    )
                }
                Err(e) => {
                    println!("  shots {shots:>6}: ERR {ms:>9.2} ms  device={device:<6} {e:?}")
                }
            }
            assert_eq!(
                device, expect,
                "shots {shots}: counters said {device} (cpu {cpu_delta}, metal {metal_delta}), \
                 arm expected {expect}"
            );
        }
        println!();
    }

    // A10 guard. The control arm is the only evidence that the GPU path runs at
    // all; if IT has also started delegating, every conclusion drawn from the
    // contrast is void and the table still looks perfectly reasonable. The two
    // regimes are far apart — measured 2.4x across this range for the real GPU
    // path against ~65x for the delegation — so a 10x bar separates them with
    // wide margin even on a loaded desktop.
    let lo = control
        .iter()
        .find(|(s, _)| *s == 1024)
        .map(|(_, ms)| *ms)
        .unwrap_or(f64::NAN);
    let hi = control
        .iter()
        .find(|(s, _)| *s == 65536)
        .map(|(_, ms)| *ms)
        .unwrap_or(f64::NAN);
    let ratio = hi / lo;
    println!("control arm 65536/1024 time ratio: {ratio:.2}x (O(1) path stays near 1-3x)");
    assert!(
        ratio < 10.0,
        "the NO-Reset control scaled {ratio:.1}x from 1024 to 65536 shots, which is \
         O(shots), not O(1) — the control is delegating to the CPU too, so the \
         contrast this probe reports is meaningless."
    );
}
