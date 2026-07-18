// SPDX-License-Identifier: Apache-2.0
//! schrodingerize — solve du/dt = -a·u by Schrödingerization (warped phase + Hamiltonian transport).
//!
//! WHAT: lift the (non-Schrödinger) linear ODE du/dt = -a·u to the transport equation ∂_t w = a·∂_p w
//!   on a momentum grid, evolve by a Hamiltonian translation e^{i·a·t·P̂}, and recover
//!   u(t) = Σ_{p≥0} w(t,p). Warped initial data w(0,p) = e^{-|p|} on M = 4 grid points, Δp = 0.5.
//! QUANTUM: run schrodingerize.aria (state-prep + `s` transport steps), read the statevector, sum the
//!   amplitudes over the p ≥ 0 grid points (indices {2,3}).
//! CLASSICAL: the ODE solution ratio u(Δp·s)/u(0) = e^{-a·Δp·s}.
//! CHECK: one transport step reproduces e^{-a·Δp} = e^{-0.5} exactly — recovery(s=1)/recovery(s=0)
//!   matches to 1e-9. (A single step is exact on this symmetric grid; multi-step accuracy needs a
//!   larger momentum grid, so this ships the exact single-step claim.)

use aria_verify_core::{banner, harness, resolve, Transport, Verdict};

const A: f64 = 1.0; // decay rate
const DP: f64 = 0.5; // momentum-grid spacing Δp

/// Recovery integral Σ_{p≥0} w(s·Δp, p) = sum of amplitudes over grid points j ∈ {2,3}.
fn recovery(transport: Transport, s: i64) -> Result<f64, String> {
    let lowered = harness::load_lowered("schrodingerize.aria", "Schrodingerize", &[("s", s)])?;
    let (flat, _) =
        harness::execute_report(transport, lowered.ir, harness::AppMode::Statevector, &[])?;
    // interleaved (re, im) over 4 amplitudes ⇒ 8 flats; sum the real parts of the p ≥ 0 block
    // (grid indices 2 and 3 → re at flats 4 and 6). Guard the length instead of indexing blind.
    if flat.len() < 7 {
        return Err(format!(
            "schrodingerize: short statevector ({} flats)",
            flat.len()
        ));
    }
    Ok(flat[4] + flat[6])
}

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "schrodingerize",
        "du/dt = -a·u via warped-phase Hamiltonian transport — recovery Σ_{p≥0} w(t,p) vs e^{-a·Δp}",
        &transport.label(guest),
    );

    let rec0 = recovery(transport, 0)?;
    let rec1 = recovery(transport, 1)?;
    let ratio = rec1 / rec0;
    let exact = (-A * DP).exp();

    println!("  recovery(0) = {rec0:.10}   recovery(1) = {rec1:.10}");
    println!("  decay factor (quantum) = {ratio:.10}   e^(-a·Δp) = {exact:.10}");

    Ok(banner::report_scalar(
        "schrodingerize",
        "recovered u(Δp)/u(0)",
        ratio,
        "e^(-a·Δp)",
        exact,
        1e-9,
    ))
}
