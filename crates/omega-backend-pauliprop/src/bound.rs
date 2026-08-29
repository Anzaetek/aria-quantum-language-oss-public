// SPDX-License-Identifier: Apache-2.0
//! An UPPER BOUND on the term count a propagation can reach.
//!
//! This exists so a scheduler can price a PauliProp job by the work it asked
//! for instead of by the worst case. Before it, `omega-server` priced every
//! PauliProp job at [`DEFAULT_MAX_TERMS`] — ~285 MB for a 4-qubit Bell pair —
//! which over-refuses concurrent jobs on the one backend that has no qubit
//! ceiling at all.
//!
//! # It is a BOUND, not an estimate
//!
//! Under-pricing fails as an OOM; over-pricing merely refuses a job that would
//! have fitted. So every unknown resolves **toward the cap**: an unrecognised
//! gate, a noise model that can spawn terms, an arithmetic overflow. If you add
//! a gate to the engine and forget this file, you get over-pricing, not
//! corruption — and a test fails.
//!
//! # Why it lives here and not in the server
//!
//! [`branch_calls`] is a shadow of the `match` in `sim.rs`. In the server, a
//! new branching gate would silently under-price every job using it, from a
//! crate whose author has no reason to look. Here the two are one file apart
//! and `every_dispatched_gate_is_priced` fails instead.

use omega_core::circuit::{CircuitIR, GateKind};
use omega_core::executor::Observable;

use crate::sim::DEFAULT_MAX_TERMS;

/// The most `branch` calls any single gate makes — `CCX` and `CSwap`, which
/// expand a CCZ into the seven non-empty subsets of three qubits.
///
/// Kept for the branch-call table's own documentation. It is NO LONGER a
/// pricing factor: the engine used to check its ceiling once per gate, after
/// the whole dispatch, so a sum could enter `CCX` just under the cap and take
/// seven consecutive doublings first. That check now runs inside the expansion
/// loop, so the transient is bounded by [`term_ceiling`] instead.
pub const MAX_BRANCH_CALLS_PER_GATE: u32 = 7;

/// The most terms the engine can hold at any instant.
///
/// `DEFAULT_MAX_TERMS + 2`, and the `+ 2` is not superstition: the cap is
/// tested inside `branch`'s loop after a term's children are added, and one
/// iteration adds at most two (a cos child and a sin child). So the sum can sit
/// two past the cap for the instant before the refusal is raised.
///
/// It was `cap · 2^7` until the engine's check moved inside the loop. Pricing
/// against that is what a governor has to reserve, so the two must move
/// together — if the check ever moves back out, this must grow again.
pub fn term_ceiling() -> u64 {
    (DEFAULT_MAX_TERMS as u64).saturating_add(2)
}

/// How many times this gate calls `branch`, or `None` if it is not a gate this
/// engine dispatches.
///
/// **Branch CALLS, not gates.** Several gates branch more than once, and
/// counting gates would UNDER-count — the direction that ends in an OOM. An
/// earlier draft of this table had `CSwap` at 6 (from regex-counting the match
/// arms) when it is 7; the source comment beside it even says "the same seven
/// branchings".
///
/// The zero rows are load-bearing, not padding. With `None => cap` as the
/// fallback, omitting the Cliffords would price a Bell pair (`H; CX`) at the
/// cap and make the whole bound useless.
pub fn branch_calls(gate: &GateKind) -> Option<u32> {
    use GateKind::*;
    Some(match gate {
        // Clifford and structural: one Pauli maps to one Pauli, so the sum
        // cannot grow at all.
        H | X | Y | Z | S | Sdg | Sx | Sxdg | CX | CZ | CY | Swap | Id | Barrier | Measure => 0,
        // One rotation, one branch.
        Rz | Rx | Ry | U1 | T | Tdg => 1,
        // Controlled phase: two generators.
        CRz => 2,
        // Euler decomposition: z, y, z.
        U3 | CU3 => 3,
        // CCZ over the seven non-empty subsets of {a, b, t}. CSwap is
        // CX · CCX · CX, so it pays the same seven.
        CCX | CSwap => 7,
        // Anything else -- including gates the engine refuses outright -- is
        // priced at the cap by the caller. Refusing costs no memory, so
        // over-pricing a refusal is harmless.
        _ => return None,
    })
}

/// Upper bound on `PauliSum::terms.len()` for this circuit and observable.
///
/// Never exceeds `DEFAULT_MAX_TERMS + 2`, and is `u64`-saturating throughout.
///
/// This used to read `DEFAULT_MAX_TERMS · 2^MAX_BRANCH_CALLS_PER_GATE`, and that
/// was true while the engine tested its cap once per GATE: a sum could enter
/// `CCX`/`CSwap` near the cap and take seven consecutive doublings before any
/// check fired, so the honest bound carried a 2^7 headroom. The engine now
/// refuses INSIDE the `branch` expansion loop, on both the commuting and the
/// branching path, so the in-flight peak is `cap + 2` and the headroom is gone.
///
/// Left as an explicit note rather than a silent edit because the stale figure
/// would justify re-introducing the compensation: anyone reading `· 2^7` as the
/// current guarantee would conclude the pricing needs it. `MAX_BRANCH_CALLS_PER_GATE`
/// itself is still live and correct — ceiling 1 below is per-circuit and still
/// doubles per `branch` call.
///
/// # The three ceilings
///
/// 1. **`seed · 2^branches`** — the sum starts as the observable's terms and
///    each `branch` call at most doubles it (a cos child that always survives,
///    plus at most one sin child). Clifford gates cannot grow it.
/// 2. **`4^|cone|`** — only `4^k` distinct Paulis exist on `k` qubits, and the
///    sum is keyed by Pauli so duplicates merge. This is what makes a wide,
///    shallow circuit cheap. Requires observable indices to be in range, which
///    `Observable::validate_qubits` now enforces upstream.
/// 3. **[`term_ceiling`]** — the cap plus the two-term transient the in-loop
///    check allows. The engine refuses past `DEFAULT_MAX_TERMS`, so reserving
///    more than that (plus the transient) is never useful.
///
/// # Overflow
///
/// Release builds do not panic on overflow here (no `[profile]` overrides), and
/// every natural spelling fails *toward zero*: `1u64 << 64` masks to `1`,
/// `4u64.pow(32)` wraps to `0`. Reserving zero is the worst outcome available,
/// so the shift width is checked BEFORE shifting, not after.
pub fn term_upper_bound(circuit: &CircuitIR, observable: &Observable) -> u64 {
    let cap = term_ceiling();

    // Ceiling 1: light cone + branch count.
    //
    // Reverse pass: a gate that does not touch the current support cannot
    // affect the observable. Union BEFORE counting, so the cone can only be
    // over-approximated -- safe, since that raises the bound.
    let n = circuit.num_qubits as usize;
    let mut in_cone = vec![false; n];
    let mut cone_size = 0usize;
    for (q, _) in observable.terms.iter().flat_map(|(_, p)| p.iter()) {
        let q = *q as usize;
        // Out-of-range indices are refused upstream; be defensive rather than
        // panic if that ever regresses.
        if q < n && !in_cone[q] {
            in_cone[q] = true;
            cone_size += 1;
        }
    }

    let mut branches: u32 = 0;
    for op in circuit.ops.iter().rev() {
        let touches = op.qubits.iter().any(|q| {
            let q = q.0 as usize;
            q < n && in_cone[q]
        });
        if !touches {
            continue;
        }
        for q in &op.qubits {
            let q = q.0 as usize;
            if q < n && !in_cone[q] {
                in_cone[q] = true;
                cone_size += 1;
            }
        }
        match branch_calls(&op.gate) {
            Some(b) => branches = branches.saturating_add(b),
            // Unknown gate: cannot bound it, so do not pretend to.
            None => return cap,
        }
    }

    // A noise model that spawns terms is not modelled here. Amplitude damping
    // is `Z -> (1-y)Z + y*I`: it creates an identity companion WITHOUT going
    // through `branch`, so the branch count does not bound it.
    if noise_can_spawn_terms(circuit) {
        return cap;
    }

    let seed = observable.terms.len().max(1) as u64;

    // `2^branches`, checked before the shift.
    let by_branching = if branches >= 63 {
        cap
    } else {
        seed.saturating_mul(1u64 << branches)
    };

    // `4^cone_size` = `1 << (2 * cone_size)`, likewise checked first.
    let by_pauli_space = if cone_size >= 32 {
        cap
    } else {
        1u64 << (2 * cone_size)
    };

    by_branching.min(by_pauli_space).min(cap)
}

/// Does this circuit carry a channel that can SPAWN terms?
///
/// Only non-unital amplitude damping does: `Z -> (1-y)Z + y*I` adds an identity
/// companion. Depolarizing, phase damping and readout error only scale existing
/// coefficients, so they cannot grow the sum.
///
/// Conservative: the noise model lives on the backend rather than the circuit,
/// so this cannot see it from here and returns `false`. The BACKEND-level
/// entry point is responsible for pricing a noisy run at the cap -- recorded
/// here so the omission is deliberate and visible rather than forgotten.
fn noise_can_spawn_terms(_circuit: &CircuitIR) -> bool {
    false
}
