// SPDX-License-Identifier: Apache-2.0
//! OpenFermion-FQE against OUR statevector backend, with the Jordan–Wigner
//! mapper on ONE side only.
//!
//! `ffsim_vs_statevector.rs` runs the same `CircuitIR` through ffsim over QPY
//! and reads BOTH engines out through the same `jordan_wigner()` observables.
//! That pins the evolution, but a sign error inside the mapper is applied
//! identically to both numbers and cancels. This file closes that hole:
//!
//! * FQE receives the circuit as fermionic gates (`givens`, `cphase`) and the
//!   observables as ladder products — mode `p` is FQE orbital `p`, and FQE
//!   evolves `exp(θ(a†_p a_q − a†_q a_p))` in its own FCI basis. No qubit, no
//!   Pauli string, no QPY exists on that side (`python/fqe_runner.py`).
//! * The statevector backend receives `Rbs` / `CU3` and the JW image of the
//!   same `FermionicOp`s — everything `omega_core::fermion` claims.
//!
//! Agreement at 1e-9 is therefore a statement about the mapper itself: the
//! `Rbs ≡ exp(θ(a†a − h.c.))` identity for adjacent modes, the Z strings in
//! `jordan_wigner_terms`, and the `cphase` phase convention, checked against
//! an engine that shares none of them. `the_comparison_is_not_vacuous` shows
//! the harness sees a sign flip.
//!
//! Skips out loud without `python/.venv-fqe`
//! (`make -C crates/omega-bridges/python fqe-venv`; FQE 0.3.0 needs Python ≤ 3.12).
use num_complex::Complex64;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend as _, Observable};
use omega_core::fermion::{cphase, givens, FermionicOp, Ladder};
use omega_core::params::ParameterBinding;
use serde_json::{json, Value};
use smallvec::smallvec;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn runner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")
}

fn venv_python() -> PathBuf {
    runner_dir().join(".venv-fqe").join("bin").join("python")
}

macro_rules! skip_without_venv {
    () => {
        if !venv_python().exists() {
            eprintln!(
                "fqe venv missing at {} — skipping. Build with \
                 `make -C crates/omega-bridges/python fqe-venv`.",
                venv_python().display()
            );
            return;
        }
    };
}

/// A fermionic gate as `omega_core::fermion` defines it. This is what FQE is
/// sent; `ir_of` is the only place it becomes a qubit gate.
#[derive(Clone, Copy, Debug)]
enum FGate {
    /// `exp(θ (a†_p a_q − a†_q a_p))`, adjacent modes only (`givens` refuses otherwise).
    Givens(u32, u32, f64),
    /// `exp(iφ n_p n_q)`.
    CPhase(u32, u32, f64),
}

fn x(q: u32) -> GateOp {
    GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(q)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    }
}

fn ir_of(n: u32, occupied: &[u32], gates: &[FGate], flip_givens: bool) -> CircuitIR {
    let mut ir = CircuitIR::new(n, CircuitType::GateBased);
    for &m in occupied {
        ir.add_op(x(m));
    }
    for g in gates {
        match *g {
            FGate::Givens(p, q, theta) => {
                let theta = if flip_givens { -theta } else { theta };
                ir.add_op(givens(p, q, theta).unwrap());
            }
            FGate::CPhase(p, q, phi) => ir.add_op(cphase(p, q, phi)),
        }
    }
    ir
}

fn ladder_json(l: &Ladder) -> Value {
    json!([l.mode, l.dagger])
}

fn observable_json(op: &FermionicOp) -> Value {
    Value::Array(
        op.terms
            .iter()
            .map(|(c, prod)| json!([c.re, c.im, prod.iter().map(ladder_json).collect::<Vec<_>>()]))
            .collect(),
    )
}

fn request(n: u32, occupied: &[u32], gates: &[FGate], obs: &[FermionicOp]) -> Value {
    let gates: Vec<Value> = gates
        .iter()
        .map(|g| match *g {
            FGate::Givens(p, q, theta) => json!({"kind": "givens", "p": p, "q": q, "theta": theta}),
            FGate::CPhase(p, q, phi) => json!({"kind": "cphase", "p": p, "q": q, "phi": phi}),
        })
        .collect();
    json!({
        "mode": "fermionic-expectation",
        "num_modes": n,
        "occupied": occupied,
        "gates": gates,
        "observables": obs.iter().map(observable_json).collect::<Vec<_>>(),
    })
}

fn fqe(req: &Value) -> Vec<f64> {
    let mut child = Command::new(runner_dir().join("omega-bridge-fqe-runner"))
        .env("OMEGA_BRIDGE_FQE_PYTHON", venv_python())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn omega-bridge-fqe-runner");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(req.to_string().as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or_else(|| {
            panic!(
                "fqe runner produced no output; stderr: {}",
                String::from_utf8_lossy(&out.stderr)
            )
        });
    let resp: Value = serde_json::from_str(line).expect("fqe runner emitted JSON");
    assert!(
        resp["ok"].as_bool() == Some(true),
        "fqe runner refused: {resp}\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    resp["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn statevector(ir: &CircuitIR, obs: &[FermionicOp]) -> Vec<f64> {
    let sv = StatevectorBackend::new();
    obs.iter()
        .map(|o| {
            let o: Observable = o.jordan_wigner().unwrap();
            o.validate_qubits(ir.num_qubits).unwrap();
            sv.expectation(ir, &ParameterBinding::new(), &o).unwrap()
        })
        .collect()
}

fn current(p: u32, q: u32) -> FermionicOp {
    let t = FermionicOp::term(Complex64::i(), vec![Ladder::raise(p), Ladder::lower(q)]);
    let d = t.dagger();
    t + d
}

fn observables() -> Vec<(&'static str, FermionicOp)> {
    let mut v = vec![];
    for p in 0..4 {
        v.push(("n_p", FermionicOp::number(p)));
    }
    for (p, q) in [(0, 1), (1, 2), (2, 3)] {
        v.push(("hop adjacent", FermionicOp::hopping(p, q, 1.0)));
        v.push(("current adjacent", current(p, q)));
    }
    v.push(("hop 0-2 (Z on 1)", FermionicOp::hopping(0, 2, 1.0)));
    v.push(("hop 0-3 (Z on 1,2)", FermionicOp::hopping(0, 3, 0.5)));
    v.push(("current 1-3 (Z on 2)", current(1, 3)));
    v.push(("n_0 n_1", FermionicOp::interaction(0, 1, 1.0)));
    v.push(("n_1 n_3", FermionicOp::interaction(1, 3, 1.0)));
    let h = FermionicOp::hopping(0, 1, -1.0)
        + FermionicOp::hopping(1, 2, -1.0)
        + FermionicOp::hopping(2, 3, -1.0)
        + FermionicOp::interaction(0, 1, 0.5)
        + FermionicOp::interaction(2, 3, 0.5)
        + FermionicOp::number(2).scale(Complex64::new(-0.3, 0.0));
    v.push(("Hubbard-like sum", h));
    v
}

/// Same layer as `ffsim_vs_statevector::layered`, so a disagreement here and
/// an agreement there localises the fault to the mapper.
fn layered() -> Vec<FGate> {
    vec![
        FGate::Givens(0, 1, 0.3),
        FGate::Givens(2, 3, -0.45),
        FGate::CPhase(1, 2, 0.7),
        FGate::Givens(1, 2, 0.52),
        FGate::CPhase(0, 3, -0.9),
        FGate::Givens(3, 2, 0.21), // reversed order
        FGate::CPhase(0, 1, 1.3),
        FGate::Givens(1, 0, -0.8), // reversed, negative
    ]
}

/// Worst |Δ| between the two engines over `named`; asserts per-observable
/// agreement at `tol` when `tol` is `Some`.
fn worst_delta(
    occupied: &[u32],
    gates: &[FGate],
    named: &[(&str, FermionicOp)],
    flip_givens: bool,
    tol: Option<f64>,
    what: &str,
) -> f64 {
    let obs: Vec<FermionicOp> = named.iter().map(|(_, o)| o.clone()).collect();
    let ir = ir_of(4, occupied, gates, flip_givens);
    let ours = statevector(&ir, &obs);
    let theirs = fqe(&request(4, occupied, gates, &obs));
    assert_eq!(ours.len(), theirs.len());
    let mut worst = 0.0f64;
    for (i, ((name, _), (a, b))) in named.iter().zip(ours.iter().zip(&theirs)).enumerate() {
        let d = (a - b).abs();
        worst = worst.max(d);
        if let Some(tol) = tol {
            assert!(
                d < tol,
                "{what}, observable {i} ({name}): statevector {a}, fqe {b}, |Δ| = {d:e}"
            );
        }
    }
    eprintln!("{what}: {} observables, worst |Δ| = {worst:e}", named.len());
    worst
}

#[test]
fn t1_layer_agrees_with_fqe_in_every_sector() {
    skip_without_venv!();
    let named = observables();
    for occupied in [
        &[][..],
        &[0][..],
        &[0, 2][..],
        &[1, 2][..],
        &[0, 1, 3][..],
        &[0, 1, 2, 3][..],
    ] {
        worst_delta(
            occupied,
            &layered(),
            &named,
            false,
            Some(1e-9),
            &format!("occupied {occupied:?}"),
        );
    }
}

/// The same harness with every Givens angle negated on the qubit side only.
/// If this did not move the numbers the test above would be checking nothing
/// about the rotation's sign; the hopping/current observables must see it.
#[test]
fn the_comparison_is_not_vacuous() {
    skip_without_venv!();
    let worst = worst_delta(&[0, 2], &layered(), &observables(), true, None, "flipped θ");
    assert!(
        worst > 1e-2,
        "flipping θ on one side moved nothing: worst |Δ| = {worst:e}"
    );
}

/// One Givens, one sector, closed form on both sides — the smallest case that
/// pins the direction of rotation, independently of `layered`'s mixing.
#[test]
fn a_single_givens_rotates_the_documented_way() {
    skip_without_venv!();
    let theta = 0.3;
    let named = [
        ("n_0", FermionicOp::number(0)),
        ("n_1", FermionicOp::number(1)),
        ("hop 0-1", FermionicOp::hopping(0, 1, 1.0)),
        ("current 0-1", current(0, 1)),
    ];
    let obs: Vec<FermionicOp> = named.iter().map(|(_, o)| o.clone()).collect();
    let gates = [FGate::Givens(0, 1, theta)];
    let theirs = fqe(&request(4, &[0], &gates, &obs));
    let ours = statevector(&ir_of(4, &[0], &gates, false), &obs);
    // exp(θ(a†0 a1 − a†1 a0)) |1 0⟩ = cos θ |1 0⟩ − sin θ |0 1⟩  (a†1 a0 |10⟩ = |01⟩)
    let expect = [
        theta.cos().powi(2),
        theta.sin().powi(2),
        -(2.0 * theta).sin(),
        0.0,
    ];
    for (i, ((name, _), e)) in named.iter().zip(expect).enumerate() {
        assert!(
            (theirs[i] - e).abs() < 1e-12,
            "fqe {name}: {} vs closed form {e}",
            theirs[i]
        );
        assert!(
            (ours[i] - e).abs() < 1e-12,
            "statevector {name}: {} vs closed form {e}",
            ours[i]
        );
    }
}
