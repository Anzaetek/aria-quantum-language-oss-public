//! Numeric acceptance for the Pauli-propagation backend.
//!
//! Every check compares `PauliPropBackend::expectation` against a dense
//! statevector **oracle** (built locally here, so the test has no backend
//! dependency) to a stated tolerance — or against a hand-derived exact value.

use num_complex::Complex64 as C;
use omega_backend_pauliprop::PauliPropBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

// ----- tiny circuit builder -------------------------------------------------

fn op(gate: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
        params: params.iter().map(|p| ParamExpr::Concrete(*p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

fn circuit(n: u32, ops: Vec<GateOp>) -> CircuitIR {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    c.ops = ops;
    c
}

// ----- dense statevector oracle --------------------------------------------

fn apply_1q(psi: &mut [C], q: usize, m: [[C; 2]; 2]) {
    let step = 1usize << q;
    let n = psi.len();
    let mut i = 0;
    while i < n {
        if i & step == 0 {
            let a0 = psi[i];
            let a1 = psi[i | step];
            psi[i] = m[0][0] * a0 + m[0][1] * a1;
            psi[i | step] = m[1][0] * a0 + m[1][1] * a1;
        }
        i += 1;
    }
}

fn r(x: f64) -> C {
    C::new(x, 0.0)
}

fn gate_matrix(g: &GateKind, theta: f64) -> [[C; 2]; 2] {
    let i = C::new(0.0, 1.0);
    let s2 = 1.0 / 2f64.sqrt();
    match g {
        GateKind::H => [[r(s2), r(s2)], [r(s2), r(-s2)]],
        GateKind::X => [[r(0.0), r(1.0)], [r(1.0), r(0.0)]],
        GateKind::Y => [[r(0.0), -i], [i, r(0.0)]],
        GateKind::Z => [[r(1.0), r(0.0)], [r(0.0), r(-1.0)]],
        GateKind::S => [[r(1.0), r(0.0)], [r(0.0), i]],
        GateKind::Sdg => [[r(1.0), r(0.0)], [r(0.0), -i]],
        GateKind::T => [
            [r(1.0), r(0.0)],
            [r(0.0), C::from_polar(1.0, std::f64::consts::FRAC_PI_4)],
        ],
        GateKind::Tdg => [
            [r(1.0), r(0.0)],
            [r(0.0), C::from_polar(1.0, -std::f64::consts::FRAC_PI_4)],
        ],
        GateKind::Rz => [
            [C::from_polar(1.0, -theta / 2.0), r(0.0)],
            [r(0.0), C::from_polar(1.0, theta / 2.0)],
        ],
        GateKind::Rx => {
            let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
            [[r(c), -i * s], [-i * s, r(c)]]
        }
        GateKind::Ry => {
            let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
            [[r(c), r(-s)], [r(s), r(c)]]
        }
        GateKind::U1 => [[r(1.0), r(0.0)], [r(0.0), C::from_polar(1.0, theta)]],
        _ => panic!("oracle: unhandled 1q gate {g:?}"),
    }
}

/// Run the circuit on |0…0⟩ and return the statevector.
fn simulate(c: &CircuitIR) -> Vec<C> {
    let n = c.num_qubits as usize;
    let mut psi = vec![C::new(0.0, 0.0); 1 << n];
    psi[0] = C::new(1.0, 0.0);
    for o in &c.ops {
        let q: Vec<usize> = o.qubits.iter().map(|x| x.0 as usize).collect();
        let theta = o.params.first().map(|p| match p {
            ParamExpr::Concrete(v) => *v,
            _ => panic!("oracle: non-concrete param"),
        });
        match o.gate {
            GateKind::CX => {
                let (cq, tq) = (q[0], q[1]);
                for i in 0..psi.len() {
                    if (i >> cq) & 1 == 1 && (i >> tq) & 1 == 0 {
                        psi.swap(i, i | (1 << tq));
                    }
                }
            }
            GateKind::CZ => {
                for (i, a) in psi.iter_mut().enumerate() {
                    if (i >> q[0]) & 1 == 1 && (i >> q[1]) & 1 == 1 {
                        *a = -*a;
                    }
                }
            }
            GateKind::CRz => {
                let th = theta.unwrap();
                for (i, a) in psi.iter_mut().enumerate() {
                    if (i >> q[0]) & 1 == 1 {
                        let s = if (i >> q[1]) & 1 == 1 {
                            th / 2.0
                        } else {
                            -th / 2.0
                        };
                        *a *= C::from_polar(1.0, s);
                    }
                }
            }
            GateKind::Swap => {
                let (a, b) = (q[0], q[1]);
                for i in 0..psi.len() {
                    if (i >> a) & 1 == 1 && (i >> b) & 1 == 0 {
                        psi.swap(i, (i & !(1 << a)) | (1 << b));
                    }
                }
            }
            GateKind::Barrier | GateKind::Id | GateKind::Measure | GateKind::Reset => {}
            ref gk => apply_1q(&mut psi, q[0], gate_matrix(gk, theta.unwrap_or(0.0))),
        }
    }
    psi
}

/// ⟨ψ|O|ψ⟩ for an observable (sum of Pauli strings) by direct application.
fn oracle_expectation(c: &CircuitIR, obs: &Observable) -> f64 {
    let psi = simulate(c);
    let i = C::new(0.0, 1.0);
    let mut total = 0.0;
    for (coeff, term) in &obs.terms {
        let mut phi = psi.clone();
        for (q, p) in term {
            let q = *q as usize;
            match p {
                PauliOp::I => {}
                PauliOp::X => apply_1q(&mut phi, q, [[r(0.0), r(1.0)], [r(1.0), r(0.0)]]),
                PauliOp::Y => apply_1q(&mut phi, q, [[r(0.0), -i], [i, r(0.0)]]),
                PauliOp::Z => apply_1q(&mut phi, q, [[r(1.0), r(0.0)], [r(0.0), r(-1.0)]]),
            }
        }
        let ip: C = psi.iter().zip(&phi).map(|(a, b)| a.conj() * b).sum();
        total += coeff * ip.re;
    }
    total
}

// ----- observable helpers ---------------------------------------------------

fn obs(term: Vec<(u32, PauliOp)>) -> Observable {
    Observable {
        terms: vec![(1.0, term)],
    }
}

fn pp_exact() -> PauliPropBackend {
    PauliPropBackend::new()
}

/// Deep non-Clifford circuit reused by the max_freq test: 4 qubits, 6 layers of
/// CX + Rz + Rx, observable ⟨Z0 Z2⟩.
fn deep_noncliff() -> (CircuitIR, Observable) {
    let mut ops = Vec::new();
    for q in 0..4 {
        ops.push(op(GateKind::H, &[q], &[]));
    }
    for layer in 0..6 {
        for q in 0..3 {
            ops.push(op(GateKind::CX, &[q, q + 1], &[]));
        }
        for q in 0..4 {
            ops.push(op(GateKind::Rz, &[q], &[0.3 + 0.1 * layer as f64]));
            ops.push(op(GateKind::Rx, &[q], &[0.2]));
        }
    }
    (circuit(4, ops), obs(vec![(0, PauliOp::Z), (2, PauliOp::Z)]))
}

#[test]
fn max_freq_truncation_stays_within_budget_and_converges() {
    // PauliPropagation.jl-style split-frequency truncation: capping the number
    // of non-Clifford sin-branches must (1) keep the true error within the
    // reported dropped-mass budget for every cap, and (2) improve monotonically
    // — a looser cap keeps a superset of terms, so the error is non-increasing
    // and vanishes once the cap exceeds the deepest surviving path.
    //
    // (dropped_mass itself is NOT monotone in max_freq: a looser cap lets more
    // branches spawn, so more of their over-budget children get dropped
    // downstream. The certified bound error ≤ dropped_mass still holds.)
    let (c, o) = deep_noncliff();
    let params = ParameterBinding::new();
    let exact = pp_exact().expectation(&c, &params, &o).unwrap();

    let mut prev_err = f64::INFINITY;
    let mut last_err = f64::INFINITY;
    for max_freq in [1u32, 2, 3, 5, 8, 24] {
        let engine = PauliPropBackend::new().max_freq(Some(max_freq));
        let (approx, dropped) = engine.expectation_with_budget(&c, &params, &o).unwrap();
        let err = (approx - exact).abs();
        assert!(
            err <= dropped + 1e-9,
            "max_freq={max_freq}: error {err} exceeded dropped-mass budget {dropped}"
        );
        assert!(
            err <= prev_err + 1e-9,
            "max_freq={max_freq}: error {err} worse than tighter cap's {prev_err}"
        );
        prev_err = err;
        last_err = err;
    }
    // A cap past the deepest path (24 ≥ #rotations on any path) is exact.
    assert!(
        last_err <= 1e-9,
        "loosest max_freq did not converge to exact: err {last_err}"
    );
}

#[test]
fn u1_matches_statevector_and_rz() {
    // U1(λ) propagates identically to Rz(λ) (global phase cancels under
    // conjugation). Check ⟨X0⟩ on H;U1(λ) against the dense oracle and Rz.
    let lambda = 0.9;
    let params = ParameterBinding::new();
    let o = obs(vec![(0, PauliOp::X)]);

    let c_u1 = circuit(
        1,
        vec![
            op(GateKind::H, &[0], &[]),
            op(GateKind::U1, &[0], &[lambda]),
        ],
    );
    let got = pp_exact().expectation(&c_u1, &params, &o).unwrap();
    let oracle = oracle_expectation(&c_u1, &o);
    assert!(
        (got - oracle).abs() <= 1e-9,
        "U1 expectation {got} vs oracle {oracle}"
    );

    let c_rz = circuit(
        1,
        vec![
            op(GateKind::H, &[0], &[]),
            op(GateKind::Rz, &[0], &[lambda]),
        ],
    );
    let got_rz = pp_exact().expectation(&c_rz, &params, &o).unwrap();
    assert!((got - got_rz).abs() <= 1e-12, "U1 {got} != Rz {got_rz}");
}

// ----- tests ----------------------------------------------------------------

#[test]
fn clifford_exact_matches_statevector() {
    // A fixed, non-trivial Clifford circuit on 4 qubits.
    let c = circuit(
        4,
        vec![
            op(GateKind::H, &[0], &[]),
            op(GateKind::CX, &[0, 1], &[]),
            op(GateKind::S, &[1], &[]),
            op(GateKind::H, &[2], &[]),
            op(GateKind::CZ, &[1, 2], &[]),
            op(GateKind::CX, &[2, 3], &[]),
            op(GateKind::Sdg, &[3], &[]),
            op(GateKind::Swap, &[0, 3], &[]),
            op(GateKind::Y, &[2], &[]),
        ],
    );
    let observables = vec![
        obs(vec![(0, PauliOp::Z)]),
        obs(vec![(1, PauliOp::X)]),
        obs(vec![(2, PauliOp::Y)]),
        obs(vec![(0, PauliOp::Z), (1, PauliOp::Z)]),
        obs(vec![(0, PauliOp::X), (3, PauliOp::Z)]),
        obs(vec![
            (0, PauliOp::Z),
            (1, PauliOp::Z),
            (2, PauliOp::Z),
            (3, PauliOp::Z),
        ]),
    ];
    let be = pp_exact();
    let params = ParameterBinding::new();
    for o in &observables {
        let got = be.expectation(&c, &params, o).unwrap();
        let want = oracle_expectation(&c, o);
        assert!(
            (got - want).abs() <= 1e-9,
            "clifford expectation mismatch: got {got}, want {want} (obs {:?})",
            o.terms
        );
    }
}

#[test]
fn rotations_exact_matches_statevector() {
    // Non-Clifford: Rz/Rx/Ry. With no truncation the engine is still exact.
    let c = circuit(
        3,
        vec![
            op(GateKind::H, &[0], &[]),
            op(GateKind::Rz, &[0], &[0.7]),
            op(GateKind::CX, &[0, 1], &[]),
            op(GateKind::Rx, &[1], &[1.3]),
            op(GateKind::Ry, &[2], &[-0.4]),
            op(GateKind::CX, &[1, 2], &[]),
            op(GateKind::Rz, &[2], &[2.1]),
            op(GateKind::T, &[0], &[]),
        ],
    );
    let observables = vec![
        obs(vec![(0, PauliOp::Z)]),
        obs(vec![(1, PauliOp::Z)]),
        obs(vec![(2, PauliOp::Z)]),
        obs(vec![(0, PauliOp::X)]),
        obs(vec![(0, PauliOp::Z), (2, PauliOp::Z)]),
    ];
    let be = pp_exact();
    let params = ParameterBinding::new();
    for o in &observables {
        let got = be.expectation(&c, &params, o).unwrap();
        let want = oracle_expectation(&c, o);
        assert!(
            (got - want).abs() <= 1e-9,
            "rotation expectation mismatch: got {got}, want {want} (obs {:?})",
            o.terms
        );
    }
}

#[test]
fn truncation_stays_within_dropped_mass() {
    // A deep, entangling non-Clifford circuit: many branches.
    let mut ops = Vec::new();
    for q in 0..4 {
        ops.push(op(GateKind::H, &[q], &[]));
    }
    for layer in 0..6 {
        for q in 0..3 {
            ops.push(op(GateKind::CX, &[q, q + 1], &[]));
        }
        for q in 0..4 {
            ops.push(op(GateKind::Rz, &[q], &[0.3 + 0.1 * layer as f64]));
            ops.push(op(GateKind::Rx, &[q], &[0.2]));
        }
    }
    let c = circuit(4, ops);
    let o = obs(vec![(0, PauliOp::Z), (2, PauliOp::Z)]);
    let params = ParameterBinding::new();

    let exact = pp_exact().expectation(&c, &params, &o).unwrap();
    let oracle = oracle_expectation(&c, &o);
    assert!(
        (exact - oracle).abs() <= 1e-9,
        "exact engine wrong: {exact} vs {oracle}"
    );

    // Truncated run: the error is bounded by the reported dropped L1 mass.
    let trunc = PauliPropBackend::with_truncation(1e-3, Some(4));
    let (approx, dropped) = trunc.expectation_with_budget(&c, &params, &o).unwrap();
    assert!(
        (approx - exact).abs() <= dropped + 1e-9,
        "truncation error {} exceeded dropped-mass budget {}",
        (approx - exact).abs(),
        dropped
    );
}

#[test]
fn wide_clifford_ghz_scales() {
    // 40-qubit GHZ: a dense statevector (2^40) is infeasible, but a single
    // Pauli never branches under Clifford gates, so this is instant.
    let n = 40u32;
    let mut ops = vec![op(GateKind::H, &[0], &[])];
    for q in 0..n - 1 {
        ops.push(op(GateKind::CX, &[q, q + 1], &[]));
    }
    let c = circuit(n, ops);
    let be = pp_exact();
    let params = ParameterBinding::new();

    // ⟨Z_i Z_j⟩ = +1 on a GHZ state for any i, j.
    let zz = be
        .expectation(&c, &params, &obs(vec![(0, PauliOp::Z), (39, PauliOp::Z)]))
        .unwrap();
    assert!((zz - 1.0).abs() <= 1e-12, "GHZ ⟨Z0 Z39⟩ = {zz}, expected 1");

    // ⟨Z_0⟩ = 0 on GHZ.
    let z0 = be
        .expectation(&c, &params, &obs(vec![(0, PauliOp::Z)]))
        .unwrap();
    assert!(z0.abs() <= 1e-12, "GHZ ⟨Z0⟩ = {z0}, expected 0");

    // ⟨X_0 X_1 … X_39⟩ = +1 on GHZ (the other stabilizer).
    let xall: Vec<(u32, PauliOp)> = (0..n).map(|q| (q, PauliOp::X)).collect();
    let xx = be.expectation(&c, &params, &obs(xall)).unwrap();
    assert!((xx - 1.0).abs() <= 1e-12, "GHZ ⟨X⊗40⟩ = {xx}, expected 1");
}

#[test]
fn crz_matches_statevector() {
    // CRz is decomposed internally as Rz_t(θ/2)·Rzz_{c,t}(-θ/2); check it.
    let c = circuit(
        3,
        vec![
            op(GateKind::H, &[0], &[]),
            op(GateKind::H, &[1], &[]),
            op(GateKind::CRz, &[0, 1], &[1.1]),
            op(GateKind::Ry, &[2], &[0.6]),
            op(GateKind::CRz, &[1, 2], &[-0.8]),
            op(GateKind::Rz, &[0], &[0.5]),
        ],
    );
    let be = pp_exact();
    let params = ParameterBinding::new();
    for o in [
        obs(vec![(0, PauliOp::Z)]),
        obs(vec![(1, PauliOp::Z)]),
        obs(vec![(2, PauliOp::Z)]),
        obs(vec![(0, PauliOp::X)]),
        obs(vec![(0, PauliOp::Z), (1, PauliOp::Z)]),
    ] {
        let got = be.expectation(&c, &params, &o).unwrap();
        let want = oracle_expectation(&c, &o);
        assert!(
            (got - want).abs() <= 1e-9,
            "CRz expectation mismatch: got {got}, want {want} (obs {:?})",
            o.terms
        );
    }
}

#[test]
fn trotter_ising_showcase() {
    // Trotterized 1D transverse-field Ising: per step, a ZZ rotation on each
    // bond (realized CX·Rz(2J)·CX) and an X rotation on each site (Rx(2h)).
    // This is the non-Clifford regime Pauli propagation targets.
    let n = 5u32;
    let (j, h) = (0.30, 0.20);
    let mut ops = Vec::new();
    for _ in 0..4 {
        for b in 0..n - 1 {
            ops.push(op(GateKind::CX, &[b, b + 1], &[]));
            ops.push(op(GateKind::Rz, &[b + 1], &[2.0 * j]));
            ops.push(op(GateKind::CX, &[b, b + 1], &[]));
        }
        for s in 0..n {
            ops.push(op(GateKind::Rx, &[s], &[2.0 * h]));
        }
    }
    let c = circuit(n, ops);
    let o = obs(vec![(2, PauliOp::Z)]); // central magnetization
    let params = ParameterBinding::new();

    // Exact (no truncation) reproduces the dense statevector.
    let exact = pp_exact().expectation(&c, &params, &o).unwrap();
    let oracle = oracle_expectation(&c, &o);
    assert!(
        (exact - oracle).abs() <= 1e-9,
        "trotter ⟨Z2⟩ exact {exact} vs oracle {oracle}"
    );

    // Truncated run: error bounded by the reported dropped-mass budget, and the
    // truncation actually drops something (this circuit branches).
    let trunc = PauliPropBackend::with_truncation(1e-2, Some(5));
    let (approx, dropped) = trunc.expectation_with_budget(&c, &params, &o).unwrap();
    assert!(
        dropped > 0.0,
        "expected the Ising tree to truncate, dropped {dropped}"
    );
    assert!(
        (approx - exact).abs() <= dropped + 1e-9,
        "truncation error {} exceeds budget {}",
        (approx - exact).abs(),
        dropped
    );
}

#[test]
fn truncation_error_curve_is_certified_and_converges() {
    // Sweep the coefficient threshold on a Trotterized transverse-field Ising
    // circuit and certify the controlled-approximation contract of Pauli
    // propagation: at every threshold the true error is within the reported
    // dropped-mass budget, the budget shrinks monotonically as the threshold
    // tightens, and the estimate converges to the exact value.
    let n = 6u32;
    let (j, h) = (0.30, 0.20);
    let mut ops = Vec::new();
    for _ in 0..4 {
        for b in 0..n - 1 {
            ops.push(op(GateKind::CX, &[b, b + 1], &[]));
            ops.push(op(GateKind::Rz, &[b + 1], &[2.0 * j]));
            ops.push(op(GateKind::CX, &[b, b + 1], &[]));
        }
        for s in 0..n {
            ops.push(op(GateKind::Rx, &[s], &[2.0 * h]));
        }
    }
    let c = circuit(n, ops);
    let o = obs(vec![(2, PauliOp::Z)]);
    let params = ParameterBinding::new();

    let exact = pp_exact().expectation(&c, &params, &o).unwrap();

    let thresholds = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3];
    let mut prev_dropped = f64::INFINITY;
    let mut err_loose = None;
    let mut err_tight = 0.0;
    for &thr in &thresholds {
        let be = PauliPropBackend::with_truncation(thr, None);
        let (val, dropped) = be.expectation_with_budget(&c, &params, &o).unwrap();
        let err = (val - exact).abs();
        // (1) the true error never exceeds the certified budget.
        assert!(
            err <= dropped + 1e-9,
            "at C={thr}: |err| {err} exceeds dropped_mass {dropped}"
        );
        // (2) the budget shrinks monotonically as the threshold tightens.
        assert!(
            dropped < prev_dropped,
            "dropped_mass not monotone at C={thr}: {dropped} >= {prev_dropped}"
        );
        prev_dropped = dropped;
        err_loose.get_or_insert(err);
        err_tight = err;
    }
    // (3) convergence: the tightest threshold beats the loosest and is tiny.
    let err_loose = err_loose.unwrap();
    assert!(
        err_tight < err_loose && err_tight < 1e-3,
        "no convergence: tight {err_tight} vs loose {err_loose}"
    );
}

#[test]
fn qml_z_expectation_matches_statevector() {
    // A QML "strongly-entangling" layer (the omega QML ⟨Z_q⟩ readout path):
    // per-qubit Ry,Rz rotations + a CX ring. pauliprop's expectation must match
    // the statevector's ⟨Z_q⟩ for every qubit (exact, Clifford+rotations).
    let n = 4u32;
    let th = [0.3, -0.7, 1.1, 0.5, 0.9, -0.2, 0.4, 1.3];
    let mut ops = Vec::new();
    for layer in 0..2 {
        for w in 0..n {
            ops.push(op(
                GateKind::Ry,
                &[w],
                &[th[(w as usize + 2 * layer) % th.len()]],
            ));
            ops.push(op(
                GateKind::Rz,
                &[w],
                &[th[(w as usize + 2 * layer + 1) % th.len()]],
            ));
        }
        for w in 0..n {
            ops.push(op(GateKind::CX, &[w, (w + 1) % n], &[]));
        }
    }
    let c = circuit(n, ops);
    let be = pp_exact();
    let params = ParameterBinding::new();
    for w in 0..n {
        let o = obs(vec![(w, PauliOp::Z)]);
        let got = be.expectation(&c, &params, &o).unwrap();
        let want = oracle_expectation(&c, &o);
        assert!(
            (got - want).abs() <= 1e-9,
            "QML ⟨Z{w}⟩ mismatch: got {got}, want {want}"
        );
    }
}

#[test]
fn surface_code_syndrome_via_expectation() {
    // The ECC use-case in miniature: a Z-type stabilizer check on qubits
    // {0,1,2}. With data in |0…0⟩ the check eigenvalue is +1 (syndrome bit 0);
    // an X error inside the support flips it to -1 (syndrome bit 1).
    let check = obs(vec![(0, PauliOp::Z), (1, PauliOp::Z), (2, PauliOp::Z)]);
    let params = ParameterBinding::new();
    let be = pp_exact();

    let clean = circuit(3, vec![]); // |000⟩
    let v0 = be.expectation(&clean, &params, &check).unwrap();
    assert!(
        (v0 - 1.0).abs() <= 1e-12,
        "clean check ⟨ZZZ⟩ = {v0}, expected +1"
    );
    assert_eq!(
        ((1.0 - v0) / 2.0).round() as i64,
        0,
        "syndrome bit should be 0"
    );

    let err_in = circuit(3, vec![op(GateKind::X, &[1], &[])]); // X inside support
    let v1 = be.expectation(&err_in, &params, &check).unwrap();
    assert!(
        (v1 + 1.0).abs() <= 1e-12,
        "error check ⟨ZZZ⟩ = {v1}, expected -1"
    );
    assert_eq!(
        ((1.0 - v1) / 2.0).round() as i64,
        1,
        "syndrome bit should be 1"
    );
}
