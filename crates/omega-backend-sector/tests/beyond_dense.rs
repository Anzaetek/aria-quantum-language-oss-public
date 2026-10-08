// SPDX-License-Identifier: Apache-2.0
//! Widths a dense statevector cannot hold. Two kinds of check: closed forms
//! at k = 1 and k = 2, and spectator invariance — a small circuit embedded in
//! a 50-qubit register must give exactly the numbers it gives at its own
//! width, which the dense backend can still compute.
use omega_backend_sector::{binomial, SectorBackend, MAX_DENSE_READOUT_QUBITS};
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, MidCircuitMode, Observable};
use omega_core::fermion::{cphase, givens, FermionicOp};
use omega_core::params::ParameterBinding;

fn op(gate: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

fn jw(o: FermionicOp) -> Observable {
    o.jordan_wigner().unwrap()
}

#[test]
fn one_particle_on_forty_modes() {
    // |1⟩ at mode 0, Givens chain along the register: the particle walks.
    let n = 40u32;
    let theta = 0.61;
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    c.add_op(op(GateKind::X, &[0], &[]));
    for m in 0..n - 1 {
        c.add_op(givens(m, m + 1, theta).unwrap());
    }
    // After each rotation the amplitude that moved on is sin θ of what
    // arrived: occupation of mode m is cos²θ · sin^{2m}θ for m < n−1, and the
    // last mode keeps sin^{2(n−1)}θ.
    let sb = SectorBackend::new();
    let obs: Vec<Observable> = (0..n).map(|m| jw(FermionicOp::number(m))).collect();
    let got = sb
        .expectation_multi(&c, &ParameterBinding::new(), &obs)
        .unwrap();
    let (ct, st) = (theta.cos().powi(2), theta.sin().powi(2));
    let mut total = 0.0;
    for (m, g) in got.iter().enumerate() {
        let want = if m + 1 == n as usize {
            st.powi(m as i32)
        } else {
            ct * st.powi(m as i32)
        };
        assert!((g - want).abs() < 1e-12, "mode {m}: {g} vs {want}");
        total += g;
    }
    assert!((total - 1.0).abs() < 1e-12, "one particle: Σ⟨n⟩ = {total}");
}

#[test]
fn two_far_apart_particles_on_sixty_modes_do_not_see_each_other() {
    let n = 60u32;
    assert_eq!(binomial(n, 2), Some(1770));
    let (theta, phi, lam) = (0.4, -1.2, 0.9);
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    c.add_op(op(GateKind::X, &[0], &[]));
    c.add_op(op(GateKind::X, &[59], &[]));
    c.add_op(givens(0, 1, theta).unwrap());
    c.add_op(givens(58, 59, phi).unwrap());
    // A phase between the two clusters: n_1 n_58 is 0 on every branch except
    // both-moved, where it multiplies by e^{iλ} — invisible to occupations.
    c.add_op(cphase(1, 58, lam));
    let sb = SectorBackend::new();
    let obs = vec![
        jw(FermionicOp::number(0)),
        jw(FermionicOp::number(1)),
        jw(FermionicOp::number(58)),
        jw(FermionicOp::number(59)),
        jw(FermionicOp::interaction(1, 59, 1.0)),
        jw(FermionicOp::interaction(0, 58, 1.0)),
        // Hopping across the whole register: zero, there is no coherence
        // between mode 0 and mode 59 — but the Z string over 58 modes has
        // to be evaluated to say so.
        jw(FermionicOp::hopping(0, 59, 1.0)),
    ];
    let got = sb
        .expectation_multi(&c, &ParameterBinding::new(), &obs)
        .unwrap();
    let (c2, s2) = (theta.cos().powi(2), theta.sin().powi(2));
    let (d2, t2) = (phi.cos().powi(2), phi.sin().powi(2));
    let want = [c2, s2, t2, d2, s2 * d2, c2 * t2, 0.0];
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() < 1e-12, "observable {i}: {g} vs {w}");
    }
}

#[test]
fn spectator_qubits_change_nothing() {
    // The same 4-mode circuit at width 4 (dense can do it) and embedded at
    // width 50 with 46 idle modes. Every number must match to 1e-12: the
    // Jordan–Wigner strings only run through modes the circuit uses.
    let build = |n: u32| {
        let mut c = CircuitIR::new(n, CircuitType::GateBased);
        c.add_op(op(GateKind::X, &[0], &[]));
        c.add_op(op(GateKind::X, &[2], &[]));
        c.add_op(givens(0, 1, 0.3).unwrap());
        c.add_op(givens(2, 3, -0.45).unwrap());
        c.add_op(cphase(1, 2, 0.7));
        c.add_op(givens(1, 2, 0.52).unwrap());
        c.add_op(op(GateKind::Rbs, &[0, 3], &[0.8])); // non-adjacent, raw
        c.add_op(op(GateKind::CRz, &[3, 1], &[-0.6]));
        c
    };
    let obs = vec![
        jw(FermionicOp::number(1)),
        jw(FermionicOp::hopping(0, 3, 1.0)),
        jw(FermionicOp::hopping(1, 2, 1.0)),
        jw(FermionicOp::interaction(0, 3, 1.0)),
        Observable::xx(0, 3),
        Observable::yy(1, 2),
        Observable {
            terms: vec![(
                0.5,
                vec![
                    (0, omega_core::executor::PauliOp::X),
                    (1, omega_core::executor::PauliOp::Y),
                    (2, omega_core::executor::PauliOp::Y),
                    (3, omega_core::executor::PauliOp::X),
                ],
            )],
        },
    ];
    let dense = StatevectorBackend::new()
        .expectation_multi(&build(4), &ParameterBinding::new(), &obs)
        .unwrap();
    let sb = SectorBackend::new();
    let small = sb
        .expectation_multi(&build(4), &ParameterBinding::new(), &obs)
        .unwrap();
    let wide = sb
        .expectation_multi(&build(50), &ParameterBinding::new(), &obs)
        .unwrap();
    for i in 0..obs.len() {
        assert!(
            (dense[i] - small[i]).abs() < 1e-12,
            "{i}: {} vs {}",
            dense[i],
            small[i]
        );
        assert!(
            (small[i] - wide[i]).abs() < 1e-12,
            "{i}: {} vs {}",
            small[i],
            wide[i]
        );
    }
    assert!(dense.iter().any(|v| v.abs() > 0.05), "trivial fixture");
}

#[test]
fn dense_readout_is_refused_past_the_width_it_is_for() {
    let n = MAX_DENSE_READOUT_QUBITS + 1;
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    c.add_op(op(GateKind::X, &[0], &[]));
    let r = SectorBackend::new().execute(
        &c,
        &ParameterBinding::new(),
        &ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        },
    );
    assert!(
        matches!(r, Err(omega_core::error::OmegaError::Unsupported(_))),
        "{r:?}"
    );
    // Sampling at that width is fine: keys are masks.
    let r = SectorBackend::new()
        .execute(
            &c,
            &ParameterBinding::new(),
            &ExecConfig {
                shots: Some(5),
                seed: Some(3),
                mid_circuit_mode: MidCircuitMode::Skip,
            },
        )
        .unwrap();
    assert_eq!(r.counts().values().sum::<u32>(), 5);
}
