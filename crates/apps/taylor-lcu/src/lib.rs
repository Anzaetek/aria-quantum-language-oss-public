// SPDX-License-Identifier: Apache-2.0
//! taylor_lcu — Hamiltonian simulation by a Linear Combination of Unitaries (Taylor series).
//!
//! WHAT: block-encode exp(-iXt) (t = 0.7) as a two-term LCU. The Taylor series
//!   exp(-iXt) = Σₖ (-it)ᵏXᵏ/k! resums (X²=I) to cos(t)·I + (-i·sin(t))·X, a two-term LCU with
//!   coefficients a = cos t, b = sin t. PREPARE/SELECT/PREPARE† writes U/λ (λ = a+b) into the
//!   ancilla = |0⟩ block.
//! QUANTUM: run taylor_lcu.aria through the omega runtime, read the exact statevector.
//! CLASSICAL: the closed form exp(-iXt)|0⟩ = cos(t)|0⟩ − i·sin(t)|1⟩, and success prob 1/λ².
//! CHECK: the ancilla = |0⟩ block (system amplitudes on q[0]), renormalized, equals exp(-iXt)|0⟩ to
//!   1e-9, and the post-selection success probability equals 1/λ² to 1e-9. q[0] is the low bit, so
//!   the ancilla = |0⟩ block is state indices {0, 1}.

use aria_verify_core::{banner, harness, resolve, Transport, Verdict};
use num_complex::Complex64 as C;

const T: f64 = 0.7;
const TOL: f64 = 1e-9;

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "taylor_lcu",
        "LCU block-encoding of exp(-iXt) (t=0.7) — ancilla=|0⟩ block vs closed form cos t·|0⟩ − i·sin t·|1⟩",
        &transport.label(guest),
    );

    // Run the circuit and read the exact statevector (interleaved re/im pairs).
    let lowered = harness::load_lowered("taylor_lcu.aria", "TaylorLcu", &[])?;
    let (flat, _) =
        harness::execute_report(transport, lowered.ir, harness::AppMode::Statevector, &[])?;
    let amp: Vec<C> = flat.chunks_exact(2).map(|c| C::new(c[0], c[1])).collect();
    if amp.len() != 4 {
        return Err(format!("expected 4 amplitudes, got {}", amp.len()));
    }

    // Ancilla (q[1]) = |0⟩ ⇒ high bit 0 ⇒ indices {0,1}; the system amplitude is on q[0] (low bit).
    let block = [amp[0], amp[1]];
    let prob0 = block[0].norm_sqr() + block[1].norm_sqr();
    let norm = prob0.sqrt();
    let renorm = [block[0] / norm, block[1] / norm];

    // Closed form: exp(-iXt)|0⟩ = cos t |0⟩ − i sin t |1⟩;  success prob = 1/λ².
    let (a, b) = (T.cos(), T.sin());
    let lam = a + b;
    let exact = [C::new(a, 0.0), C::new(0.0, -b)];
    let succ_exact = 1.0 / (lam * lam);

    println!(
        "  λ = a+b = {lam:.6}   success prob (quantum) = {prob0:.10}   (1/λ² = {succ_exact:.10})"
    );

    // Compare the renormalized block and the success probability as one real vector.
    let quantum = vec![
        renorm[0].re,
        renorm[0].im,
        renorm[1].re,
        renorm[1].im,
        prob0,
    ];
    let classical = vec![
        exact[0].re,
        exact[0].im,
        exact[1].re,
        exact[1].im,
        succ_exact,
    ];

    Ok(banner::report_values(
        "taylor_lcu",
        "renorm ⟨block⟩ + success",
        &quantum,
        "exp(-iXt)|0⟩ + 1/λ²",
        &classical,
        TOL,
    ))
}
