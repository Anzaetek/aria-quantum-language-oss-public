// SPDX-License-Identifier: Apache-2.0
//! Pins the **divergent** analytic-`Reset` acceptance policies (ledger A6).
//!
//! A6 records that the backends do NOT share an acceptance criterion, and gives
//! the measured table. Until now only the statevector family was pinned by a
//! test — `statevector-metal::reset_matches_cpu` (CPU/Metal) and
//! `statevector-cuda/tests/reset_criterion.rs` (CPU/CUDA). The CPU-only
//! backends in that table, `mps` and `pauli`, had no conformance test at all,
//! so either could silently change which states it accepts and nothing would
//! notice.
//!
//! This file pins the three policies that run without a GPU:
//!
//! | backend      | refuses analytic Reset when        |
//! |--------------|------------------------------------|
//! | `statevector`| reduced purity != 1 (entangled)    |
//! | `mps`        | ALL analytic resets                |
//! | `pauli`      | ALL analytic resets (A6 says "non-Z-eigenstate"; WRONG — see below) |
//!
//! The point is the DISAGREEMENT. These assertions deliberately encode three
//! different answers to the same circuit, because that divergence is the
//! documented state of the system — not a bug to be normalised away here. If a
//! backend is ever brought into line, this test should fail and be updated
//! deliberately, which is the whole reason to write it down.
//!
//! Circuit B is the discriminating case and the reason a single "does Reset
//! work" test cannot do this job: `H q0; Reset q0` is unentangled (purity 1)
//! but its *outcome* is random. CPU accepts it, MPS refuses it, and pauli
//! refuses it as well. Two policies visibly disagree on one circuit, and
//! writing this test turned up a third result: A6's `pauli` row is inaccurate.

use omega_backend_mps::MpsBackend;
use omega_backend_pauliprop::PauliPropBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

fn op(gate: GateKind, qubits: &[u32]) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().copied().map(Qubit).collect(),
        params: Vec::<ParamExpr>::new().into(),
        classical_bit: None,
        condition: None,
    }
}

fn circuit(n: u32, ops: Vec<GateOp>) -> CircuitIR {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    c.ops = ops;
    c
}

fn z_on(q: u32) -> Observable {
    Observable {
        terms: vec![(1.0, vec![(q, PauliOp::Z)])],
    }
}

/// A: `X q0; Reset q0` — unentangled, Z-eigenstate, deterministic outcome.
fn zeigen_reset() -> CircuitIR {
    circuit(2, vec![op(GateKind::X, &[0]), op(GateKind::Reset, &[0])])
}

/// B: `H q0; Reset q0` — unentangled (purity 1) but the outcome is RANDOM, and
/// `|+>` is not a Z-eigenstate. The three policies disagree here.
fn superposition_reset() -> CircuitIR {
    circuit(2, vec![op(GateKind::H, &[0]), op(GateKind::Reset, &[0])])
}

/// C: `H q0; CX q0,q1; Reset q0` — genuinely entangled.
fn entangled_reset() -> CircuitIR {
    circuit(
        2,
        vec![
            op(GateKind::H, &[0]),
            op(GateKind::CX, &[0, 1]),
            op(GateKind::Reset, &[0]),
        ],
    )
}

/// `expectation` IS the analytic path — no `ExecConfig`, no shots, so no
/// trajectory is drawn. That is what makes Reset acceptance a question at all:
/// in shots mode a branch is sampled and every backend is happy.
fn accepts<B: Backend>(backend: &B, c: &CircuitIR) -> bool {
    backend
        .expectation(c, &ParameterBinding::new(), &z_on(1))
        .is_ok()
}

#[test]
fn statevector_refuses_only_the_entangled_reset() {
    let sv = StatevectorBackend::new();
    assert!(
        accepts(&sv, &zeigen_reset()),
        "CPU must accept an unentangled Z-eigenstate reset"
    );
    assert!(
        accepts(&sv, &superposition_reset()),
        "CPU accepts `H q0; Reset q0`: purity is 1, so the result is \
         deterministic even though the outcome is not (ledger A6)"
    );
    assert!(
        !accepts(&sv, &entangled_reset()),
        "CPU must refuse an entangled analytic reset — reduced purity != 1"
    );
}

#[test]
fn mps_refuses_every_analytic_reset_including_ones_cpu_accepts() {
    let mps = MpsBackend::new(64);
    assert!(
        !accepts(&mps, &zeigen_reset()),
        "A6 records MPS as refusing ALL analytic resets, including this one \
         that CPU accepts. If MPS has been taught the purity predicate, that is \
         a real improvement — update A6 and this test together."
    );
    assert!(!accepts(&mps, &superposition_reset()));
    assert!(!accepts(&mps, &entangled_reset()));
}

/// MEASURED 2026-09-25, and it contradicts ledger A6.
///
/// A6's table says `pauli` refuses "non-Z-eigenstate", which implies it ACCEPTS
/// `X q0; Reset q0` since `X|0> = |1>` is a Z-eigenstate. It does not — it
/// refuses that too. `LIMITATIONS.md` has it right: "pauliprop ... refuse
/// Reset. They evolve a pure state, or conjugate observables unitarily, and
/// cannot represent the channel." A channel cannot be represented at all, so
/// the eigenstate of the input is irrelevant.
///
/// So A6's `pauli` row is wrong and LIMITATIONS.md is right. Pinning the
/// measured behaviour here; A6 needs the correction.
#[test]
fn pauli_refuses_every_analytic_reset_contradicting_ledger_a6() {
    let pauli = PauliPropBackend::new();
    assert!(
        !accepts(&pauli, &zeigen_reset()),
        "measured: pauli refuses even a Z-eigenstate reset, because it cannot \
         represent the channel at all. A6 says otherwise and A6 is wrong."
    );
    assert!(!accepts(&pauli, &superposition_reset()));
    assert!(!accepts(&pauli, &entangled_reset()));
}

/// The divergence itself, asserted as a property rather than left implicit in
/// three separate tests. This is the line that fails if someone unifies the
/// policies without updating ledger A6.
#[test]
fn the_three_backends_genuinely_disagree_on_one_circuit() {
    let c = superposition_reset();
    let sv = accepts(&StatevectorBackend::new(), &c);
    let mps = accepts(&MpsBackend::new(64), &c);
    let pauli = accepts(&PauliPropBackend::new(), &c);

    assert!(
        sv && !mps && !pauli,
        "ledger A6 says `H q0; Reset q0` is accepted by CPU and refused by MPS \
         and pauli, for three different reasons. Got sv={sv} mps={mps} \
         pauli={pauli}. If this changed deliberately, update A6 and STATUS 5.5."
    );
}
