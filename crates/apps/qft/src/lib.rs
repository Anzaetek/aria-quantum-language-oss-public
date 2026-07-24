// SPDX-License-Identifier: Apache-2.0
//! qft — discrete Fourier transform of an input basis state.
//!
//! WHAT: QFT|x⟩ for an explicit input x on n qubits.
//! QUANTUM: run the shipped qft.aria (prefixed with X gates that prepare |x⟩)
//!   on the statevector backend through omega_app.wasm, returning the output
//!   amplitudes.
//! CLASSICAL: the textbook DFT matrix applied to the same input basis vector,
//!   F·e_x with F[k][j] = exp(2πi·k·j/N)/√N. The input index is read from the
//!   backend (prep-only run) so the check is independent of qubit ordering.
//! CHECK: every amplitude (re and im) agrees to ≤ 1e-6.

use aria_verify_core::{banner, harness, oracle, resolve, util, Complex64, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let n: u32 = 3;
    let x: u64 = 5; // non-trivial input so the controlled phases actually matter
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "qft",
        &format!("QFT|x⟩ on {n} qubits, x={x} (vs the classical DFT matrix)"),
        &transport.label(guest),
    );

    let lowered = harness::load_lowered("qft.aria", "QFT", &[("n", n as i64)])?;

    // Discover the backend index of |x⟩ from a prep-only run (ordering-agnostic).
    let prep = harness::basis_prep_ir(&lowered.ir, x);
    let prep_sv = harness::native_statevector(&prep)?;
    let input_index = prep_sv
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.norm().total_cmp(&b.1.norm()))
        .map(|(i, _)| i as u64)
        .unwrap_or(0);

    // qft.aria reads the register with qubit 0 as the MOST-significant bit
    // (big-endian), while the statevector is indexed little-endian. So the
    // circuit computes the DFT for the bit-reversed input index, and emits it
    // at bit-reversed output indices: quantum[k] = (1/√N)·exp(2πi·rev(x)·rev(k)/N).
    // We bit-reverse both indices so the comparison is apples-to-apples.
    let base = oracle::qft_amplitudes(n, util::bitrev(input_index as usize, n) as u64);
    let expected: Vec<Complex64> = (0..base.len()).map(|k| base[util::bitrev(k, n)]).collect();
    let expected_il = util::interleave(&expected);

    // Quantum: prepare |x⟩ then run the shipped QFT, read the statevector.
    let mut prep_qft = lowered.ir.clone();
    harness::prepend_basis_state(&mut prep_qft, x);
    let (payload, _value) =
        harness::execute_report(transport, prep_qft, harness::AppMode::Statevector, &[])?;

    Ok(banner::report_values(
        "qft",
        "QFT|x⟩ amplitudes (re,im)",
        &payload,
        "DFT matrix · e_x",
        &expected_il,
        1e-6,
    ))
}
