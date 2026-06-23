// SPDX-License-Identifier: Apache-2.0
//! qos — Quantum Oracle Sketching reproduction (Aria oracle circuit + convergence).
//!
//! WHAT: QOS (Zhao–Babbush–Huang 2026, arXiv:2604.07639) instantiates a phase
//!   oracle from random classical samples. The *sketching* is the sampling
//!   (part 2/3 below, numeric). The *circuit* (part 1) is the exact oracle that
//!   sampling converges to — the endpoint — run for real through omega:
//!   (1) the EXACT phase oracle D=diag(e^{iφ·v})|+⟩^n (qos_oracle.aria, the
//!       ASYMMETRIC v_x = x so qubit wiring matters, φ=0.4) — equal to the
//!       target state up to a global phase (RZ convention), so compared with the
//!       global-phase-invariant infidelity;
//!   (2) the sketch's ~1/N² convergence, reproduced numerically; and
//!   (3) closed form == Monte-Carlo mean of the stochastic process (independent).
//!   All data synthetic.
//! QUANTUM: the omega statevector of qos_oracle.aria.
//! CLASSICAL: omega-core `qos::target_state`; the analytic convergence exponent
//!   −2; the Monte-Carlo of the stochastic sampling process.
//! CHECK: oracle infidelity ≤ 1e-6 (headline); MC mean == closed form within MC
//!   error; fitted exponent within ±0.05 of −2 for all 3 kinds.

use aria_verify_core::{banner, harness, resolve, Complex64, Transport, Verdict};
use omega_core::qos::{
    infidelity, scaling_exponent, state_sketch, state_sketch_stochastic, synthetic, target_state,
};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let (phi, p) = (0.4, 0.5);
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "qos",
        "exact phase oracle circuit (omega) + ~1/N² sketch convergence (synthetic)",
        &transport.label(guest),
    );

    // (1) Aria EXACT phase oracle through omega vs qos::target_state, with the
    // ASYMMETRIC data v_x = x so the check actually exercises qubit wiring.
    let nq: u32 = 3;
    let dim = 1usize << nq;
    let lowered = harness::load_lowered("qos_oracle.aria", "PhaseOracle", &[("n", nq as i64)])?;
    let (payload, _) = harness::execute_report(
        transport,
        lowered.ir.clone(),
        harness::AppMode::Statevector,
        &[],
    )?;
    if payload.len() != 2 * dim {
        return Err(format!(
            "expected {} payload floats, got {}",
            2 * dim,
            payload.len()
        ));
    }
    let psi: Vec<Complex64> = payload
        .chunks(2)
        .map(|c| Complex64::new(c[0], c[1]))
        .collect();
    // v_x = x; allow either qubit↔index endianness (the oracle is asymmetric, so
    // this is a genuine wiring check — we just don't hardcode the bit order).
    let v_le: Vec<f64> = (0..dim).map(|x| x as f64).collect();
    let v_be: Vec<f64> = (0..dim)
        .map(|x| {
            (0..nq)
                .filter(|j| x >> j & 1 == 1)
                .map(|j| (1u64 << (nq - 1 - j)) as f64)
                .sum()
        })
        .collect();
    let inf_le = infidelity(&psi, &target_state(&v_le, phi));
    let inf_be = infidelity(&psi, &target_state(&v_be, phi));
    let oracle_inf = inf_le.min(inf_be);
    println!(
        "  exact phase oracle (omega) vs target_state(v_x=x): infidelity = {oracle_inf:.3e} \
         ({} endian)",
        if inf_le <= inf_be { "little" } else { "big" }
    );

    // (2) Convergence reproduction (numeric): fitted exponent per oracle kind.
    let counts = [256u64, 512, 1024, 2048, 4096];
    let kinds: [(&str, Vec<f64>); 3] = [
        ("general-vector", synthetic::random_vector(256, 1)),
        ("boolean-fn", synthetic::boolean_oracle(256, 2)),
        ("matrix-element", synthetic::matrix_element_oracle(256, 3)),
    ];
    for (label, vk) in &kinds {
        let slope = scaling_exponent(vk, phi, p, &counts);
        println!("    {label:>14}: fitted exponent = {slope:+.4}");
        if (slope + 2.0).abs() > 0.05 {
            return Err(format!(
                "{label}: exponent {slope:+.4} not within ±0.05 of -2"
            ));
        }
    }

    // (3) Independent check: closed form == Monte-Carlo mean of the stochastic process.
    let vs = &kinds[0].1;
    let analytic = state_sketch(vs, phi, p, 256);
    let mut avg = vec![Complex64::new(0.0, 0.0); vs.len()];
    let mut st = 0x9E3779B97F4A7C15u64;
    for _ in 0..400 {
        for (a, x) in avg
            .iter_mut()
            .zip(&state_sketch_stochastic(vs, phi, p, 256, &mut st))
        {
            *a += x;
        }
    }
    for a in avg.iter_mut() {
        *a /= 400.0;
    }
    let mc = infidelity(&avg, &analytic);
    println!("  stochastic MC mean vs closed form: infidelity = {mc:.3e}");
    if mc > 2e-2 {
        return Err(format!("MC mean vs closed form {mc:.3e} exceeds 2e-2"));
    }

    Ok(banner::report_scalar(
        "qos",
        "infidelity(exact phase-oracle circuit via omega, qos::target_state)",
        oracle_inf,
        "0 (circuit = exact oracle up to global phase)",
        0.0,
        1e-6,
    ))
}
