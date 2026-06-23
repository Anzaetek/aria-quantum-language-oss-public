// SPDX-License-Identifier: Apache-2.0
//! circulant — the circulant generator Q as a real circuit.
//!
//! WHAT: the cyclic-shift generator Q (|x⟩→|x+1 mod 8⟩) of a banded circulant
//!   C = Σⱼ cⱼ·Q^{powⱼ} (e.g. heat transfer (−2−ξ)I+Q+Q⁻¹). The QFT diagonalizes
//!   Q, which is what makes the circulant solve x = IDFT(DFT(b)/λ) work. (That
//!   solve is the Rust `omega_core::circulant` module, validated by its own
//!   tests — this harness validates only the shipped .aria circuit.)
//! QUANTUM: run circulant.aria (CyclicShift, 3 qubits) on every input basis
//!   state |x⟩ through the omega runtime and read where each maps.
//! CLASSICAL: the exact cyclic permutation σ(x) = (x+1) mod 8.
//! CHECK: every basis state maps cleanly (no superposition leakage, purity
//!   ≥0.99) to exactly σ(x). Validated in the runtime's own statevector-index
//!   convention (this does not independently pin down endianness — it confirms
//!   the circuit is an exact +1 cyclic permutation as the runtime indexes it).

use aria_verify_core::{banner, harness, resolve, Complex64, Transport, Verdict};

fn argmax_norm(sv: &[Complex64]) -> usize {
    sv.iter()
        .enumerate()
        .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).expect("finite"))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// argmax over an interleaved `[re,im,…]` payload → (index, that index's
/// probability share = |amp|²/Σ|amp|²) so we can reject superposition leakage.
fn argmax_share(p: &[f64]) -> (usize, f64) {
    let norm2: Vec<f64> = (0..p.len() / 2)
        .map(|i| p[2 * i] * p[2 * i] + p[2 * i + 1] * p[2 * i + 1])
        .collect();
    let total: f64 = norm2.iter().sum::<f64>().max(1e-300);
    let idx = (0..norm2.len())
        .max_by(|&i, &j| norm2[i].partial_cmp(&norm2[j]).expect("finite"))
        .unwrap_or(0);
    (idx, norm2[idx] / total)
}

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let dim = 8usize; // circulant.aria is fixed at 3 qubits
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "circulant",
        "circulant generator Q: |x⟩→|x+1 mod 8⟩ (the QFT diagonalizes it)",
        &transport.label(guest),
    );

    let lowered = harness::load_lowered("circulant.aria", "CyclicShift", &[])?;

    let mut perm = vec![usize::MAX; dim];
    let mut min_purity = 1.0_f64;
    for x in 0..dim as u64 {
        let prep = harness::basis_prep_ir(&lowered.ir, x);
        let in_idx = argmax_norm(&harness::native_statevector(&prep)?);

        let mut ir = lowered.ir.clone();
        harness::prepend_basis_state(&mut ir, x);
        let (payload, _) =
            harness::execute_report(transport, ir, harness::AppMode::Statevector, &[])?;
        let (out_idx, share) = argmax_share(&payload);
        min_purity = min_purity.min(share);
        perm[in_idx] = out_idx;
    }
    println!("  circuit permutation (runtime sv-index): {perm:?}  (min purity {min_purity:.4})");

    if min_purity < 0.99 {
        return Err(format!(
            "outputs are not clean basis states (min purity {min_purity:.4})"
        ));
    }

    // Exact cyclic shift σ(x) = (x+1) mod 8.
    let mismatches = (0..dim).filter(|&i| perm[i] != (i + 1) % dim).count();
    Ok(banner::report_scalar(
        "circulant",
        "basis states mis-mapped by Q (vs σ(x)=(x+1) mod 8)",
        mismatches as f64,
        "0 (exact cyclic shift)",
        0.0,
        0.0,
    ))
}
