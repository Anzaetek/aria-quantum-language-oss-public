// SPDX-License-Identifier: Apache-2.0
//! noise — modeled-noise robustness of the QLSS reproductions (QLSS Phase 4).
//!
//! Drives omega's trajectory (quantum-jump) `NoiseModel`. Each `execute` is ONE
//! trajectory, so an ensemble expectation is the average of the EXACT
//! per-trajectory functional over T trajectories (one backend seed each — the
//! trajectory average of exact expectations equals Tr(ρ·O); no shot sampling).
//!
//! ANCHORS (laws verified against the channel code, headline numeric check):
//!   (A) depolarizing — ⟨Z⟩ of RY(θ)|0⟩ = (1−4p/3)·cosθ (X,Y flip Z at p/3 each).
//!   (B) amplitude damping — ⟨Z⟩ = ⟨Z⟩₀ + 2γ·sin²(θ/2) (|1⟩ population decays).
//!
//! Laws (A)/(B) are FORMALLY PROVEN sorry-free in `proofs/lean4/QuantumProofs/Noise`:
//!   (A) ↔ `Depolarizing.depolarizing_apply` (E_p(ρ)=(1−p)ρ+(p/2)Tr ρ·1),
//!   (B) ↔ `AmplitudeDamping.amplitudeDamping_expZ` (⟨Z⟩(E_γ ρ)=⟨Z⟩(ρ)+2γ·d, here d=sin²(θ/2));
//! so these numeric anchors cross-check a machine-checked Kraus/CPTP derivation.
//!
//! REPRODUCTIONS UNDER NOISE (the actual shipped .aria circuits degrading):
//!   (C) cqs.aria Hadamard test — overlap Re⟨Z⟩ falls from 0.5.
//!   (D) circulant.aria CyclicShift — P(correct |1⟩) falls from 1.
//!   (E) qos_oracle.aria PhaseOracle — fidelity to the exact oracle falls from 1.
//!
//! CHECK: worst |anchor − law| ≤ 4σ, σ = 0.5/√T ≈ 5.6e-3 at T=8000 (≤ 0.022);
//! (C)/(D)/(E) degrade monotonically and equal their ideal at p=0.

use aria_verify_core::{banner, harness, Complex64, Transport, Verdict};
use omega_backend_statevector::{NoiseModel, NoisyStatevectorBackend, StatevectorBackend};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;

const TRAJECTORIES: u64 = 8000;
const PS: [f64; 6] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2];

fn cfg_sv() -> ExecConfig {
    ExecConfig {
        shots: None,
        seed: None, // trajectory RNG is the backend seed (Some(t)) below
        mid_circuit_mode: MidCircuitMode::Skip,
    }
}

/// Average an exact per-trajectory statevector functional over `TRAJECTORIES`
/// under the given channel (one backend seed each; ChaCha seed-avalanche
/// decorrelates them).
fn trajectory_average<F: Fn(&[Complex64]) -> f64>(
    circuit: &CircuitIR,
    params: &ParameterBinding,
    model: &NoiseModel,
    f: &F,
) -> Result<f64, String> {
    let mut sum = 0.0;
    for t in 0..TRAJECTORIES {
        let backend = NoisyStatevectorBackend::with_model(model.clone(), Some(t));
        match backend
            .execute(circuit, params, &cfg_sv())
            .map_err(|e| format!("{e:?}"))?
        {
            ExecResult::Statevector(sv) => sum += f(&sv),
            _ => return Err("expected a statevector".into()),
        }
    }
    Ok(sum / TRAJECTORIES as f64)
}

fn depolarizing(p: f64) -> NoiseModel {
    NoiseModel {
        depolarizing: omega_backend_statevector::Depolarizing::uniform(p),
        ..Default::default()
    }
}

/// Exact noiseless statevector of a circuit.
fn ideal_state(circuit: &CircuitIR, params: &ParameterBinding) -> Result<Vec<Complex64>, String> {
    match StatevectorBackend::new()
        .execute(circuit, params, &cfg_sv())
        .map_err(|e| format!("{e:?}"))?
    {
        ExecResult::Statevector(sv) => Ok(sv),
        _ => Err("expected a statevector".into()),
    }
}

fn ry_circuit(theta: f64) -> (CircuitIR, ParameterBinding) {
    let mut c = CircuitIR::new(1, CircuitType::GateBased);
    c.symbols.insert(0, "theta".to_string());
    c.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec::smallvec![Qubit(0)],
        params: smallvec::smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });
    let mut p = ParameterBinding::new();
    p.bind(0, theta);
    (c, p)
}

/// Trajectory-averaged metric across the depolarizing sweep `PS`.
fn sweep<F: Fn(&[Complex64]) -> f64>(
    circuit: &CircuitIR,
    params: &ParameterBinding,
    f: F,
) -> Result<Vec<f64>, String> {
    PS.iter()
        .map(|&p| trajectory_average(circuit, params, &depolarizing(p), &f))
        .collect()
}

fn print_curve(label: &str, curve: &[f64]) {
    let cells: Vec<String> = curve.iter().map(|v| format!("{v:+.3}")).collect();
    println!("{label}{}", cells.join("  "));
}

pub fn run(_transport: Transport) -> Result<Verdict, String> {
    banner::header(
        "noise",
        "noise robustness of the QLSS reproductions (depolarizing + amplitude damping)",
        "native (omega trajectory noise model)",
    );

    let sigma = 0.5 / (TRAJECTORIES as f64).sqrt();
    let tol = 4.0 * sigma;
    let z_of = |sv: &[Complex64]| sv[0].norm_sqr() - sv[1].norm_sqr();
    let theta = std::f64::consts::FRAC_PI_3;
    let cos = theta.cos();
    let (ryc, ryp) = ry_circuit(theta);
    let mut worst = 0.0f64;

    println!("  anchors (σ_MC={sigma:.4}, tol 4σ={tol:.4}):");

    // (A) depolarizing law + θ=π/2 falsifier.
    for &th in &[theta, std::f64::consts::FRAC_PI_2] {
        let (c, p) = ry_circuit(th);
        for &pr in &PS {
            let z = trajectory_average(&c, &p, &depolarizing(pr), &z_of)?;
            worst = worst.max((z - (1.0 - 4.0 * pr / 3.0) * th.cos()).abs());
        }
    }
    println!("    (A) depolarizing  ⟨Z⟩=(1−4p/3)cosθ      worst|Δ|={worst:.4}");

    // (B) amplitude damping law: ⟨Z⟩ = ⟨Z⟩₀ + 2γ·sin²(θ/2).
    let p1 = (theta / 2.0).sin().powi(2);
    let mut worst_ad = 0.0f64;
    for &g in &PS {
        let model = NoiseModel {
            amplitude_damping: g.into(),
            ..Default::default()
        };
        let z = trajectory_average(&ryc, &ryp, &model, &z_of)?;
        worst_ad = worst_ad.max((z - (cos + 2.0 * g * p1)).abs());
    }
    worst = worst.max(worst_ad);
    println!("    (B) amp damping   ⟨Z⟩=⟨Z⟩₀+2γsin²(θ/2)  worst|Δ|={worst_ad:.4}");
    println!(
        "      analytic ±0.05 depolarizing crossing (θ=π/3): p* = {:.4}",
        0.0375 / cos
    );

    // Reproductions under depolarizing noise.
    println!("  reproductions under depolarizing noise (p = {PS:?}):");

    // (C) CQS Hadamard test: Re⟨Z⟩ = 2·P(anc=0)−1, ancilla = qubit 0.
    let cqs = harness::load_lowered("cqs.aria", "HadamardTestZ", &[])?;
    let p_anc0 = |sv: &[Complex64]| {
        sv.iter()
            .enumerate()
            .filter(|(x, _)| x & 1 == 0)
            .map(|(_, a)| a.norm_sqr())
            .sum::<f64>()
    };
    let cqs_curve = sweep(&cqs.ir, &ParameterBinding::new(), |sv| {
        2.0 * p_anc0(sv) - 1.0
    })?;
    print_curve("    (C) cqs overlap Re⟨Z⟩  ", &cqs_curve);

    // (D) circulant CyclicShift on |000⟩: ideal output |001⟩ ⇒ P(index 1).
    let circ = harness::load_lowered("circulant.aria", "CyclicShift", &[])?;
    let circ_curve = sweep(&circ.ir, &ParameterBinding::new(), |sv| {
        sv.get(1).map(|a| a.norm_sqr()).unwrap_or(0.0)
    })?;
    print_curve("    (D) circulant P(|1⟩)   ", &circ_curve);

    // (E) QOS oracle: fidelity to the exact oracle state = avg|⟨ideal|ψ⟩|².
    let qos = harness::load_lowered("qos_oracle.aria", "PhaseOracle", &[("n", 3)])?;
    let qos_ideal = ideal_state(&qos.ir, &ParameterBinding::new())?;
    let qos_curve = sweep(&qos.ir, &ParameterBinding::new(), move |sv| {
        let ip: Complex64 = qos_ideal.iter().zip(sv).map(|(a, b)| a.conj() * b).sum();
        ip.norm_sqr()
    })?;
    print_curve("    (E) qos fidelity       ", &qos_curve);

    // Each reproduction must equal its ideal at p=0 and degrade monotonically.
    for (name, curve, ideal) in [
        ("cqs", &cqs_curve, 0.5),
        ("circulant", &circ_curve, 1.0),
        ("qos", &qos_curve, 1.0),
    ] {
        if (curve[0] - ideal).abs() > 1e-9 {
            return Err(format!("{name}: p=0 value {:.4} ≠ ideal {ideal}", curve[0]));
        }
        if curve.windows(2).any(|w| w[1] > w[0] + 1e-3) {
            return Err(format!("{name}: not monotonically degrading: {curve:?}"));
        }
    }

    Ok(banner::report_scalar(
        "noise",
        "worst |anchor − analytic law| (depolarizing + amplitude damping)",
        worst,
        "0 (laws match the channel, ≤4σ)",
        0.0,
        tol,
    ))
}
