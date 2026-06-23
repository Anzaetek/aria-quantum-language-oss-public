// SPDX-License-Identifier: Apache-2.0
//! Differential forward-⟨Z⟩ harnesses for the parametrized example circuits.
//!
//! These examples — variational ansätze, feature maps, and structural
//! algorithm templates — have no single closed-form scalar to compare against
//! (that is the point of a *trainable* circuit). Instead each is verified by a
//! genuine numeric cross-check: bind a fixed, non-trivial parameter vector, run
//! the circuit through the omega runtime to read every ⟨Z_q⟩, and compare that
//! profile against an INDEPENDENT pure-Rust statevector simulator
//! ([`aria_verify_core::sim`]). Agreement (Δ ≤ 1e-9) proves the
//! parse → lower → execute path is faithful end-to-end on the shipped circuit.
//!
//! For circuits whose *faithful* correctness is established elsewhere — HHL and
//! QSVT inversion (`proofs/lean4/QuantumProofs/{HHL,QSVT}.lean` +
//! `omega_core::{solver,chebyshev}`), the quantum kernel / IQP Born machine
//! (`omega_core::qml`) — this forward check is the example-level integration
//! gate on top of that.

use aria_verify_core::{banner, harness, resolve, sim, Observable, Transport, Verdict};

/// Bind a fixed, well-spread parameter vector, read omega's ⟨Z_q⟩ for every
/// qubit, and compare against the independent statevector oracle.
fn forward_check(
    name: &str,
    file: &str,
    circuit: &str,
    int_params: &[(&str, i64)],
    transport_override: Transport,
) -> Result<Verdict, String> {
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        name,
        "forward ⟨Z_q⟩ profile vs an INDEPENDENT statevector oracle (same lowered IR)",
        &transport.label(guest),
    );

    let lowered = harness::load_lowered(file, circuit, int_params)?;
    let n = lowered.ir.num_qubits as usize;
    let n_params = lowered.ir.symbols.len();

    // Deterministic, well-spread angles (golden-ratio low-discrepancy sequence)
    // so rotations are genuinely exercised — identical on both sides.
    let params: Vec<f64> = (0..n_params)
        .map(|i| ((i as f64 + 1.0) * 0.618_033_988_75).fract() * std::f64::consts::TAU)
        .collect();

    let obs: Vec<Observable> = (0..n).map(|q| Observable::z(q as u32)).collect();
    let (omega_z, _) = harness::execute_report(
        transport,
        lowered.ir.clone(),
        harness::AppMode::Expectations(obs),
        &params,
    )?;
    let sim_z = sim::forward_z_expectations(&lowered.ir, &params)?;

    println!("  qubits = {n}, bound params = {n_params}");
    Ok(banner::report_values(
        name,
        "omega ⟨Z_q⟩",
        &omega_z,
        "independent sim ⟨Z_q⟩",
        &sim_z,
        1e-9,
    ))
}

macro_rules! example {
    ($fn_name:ident, $name:literal, $file:literal, $circuit:literal, $params:expr) => {
        pub fn $fn_name(t: Transport) -> Result<Verdict, String> {
            forward_check($name, $file, $circuit, $params, t)
        }
    };
}

example!(
    iqp_born,
    "iqp_born",
    "iqp_born.aria",
    "IqpBorn",
    &[("n", 4)]
);
example!(
    quantum_kernel,
    "quantum_kernel",
    "quantum_kernel.aria",
    "QuantumKernelMap",
    &[("n", 2)]
);
example!(qcnn, "qcnn", "qcnn.aria", "QCNN", &[]);
example!(
    qcbm,
    "qcbm_strongly_entangling",
    "qcbm_strongly_entangling.aria",
    "QcbmStronglyEntangling",
    &[("N", 4), ("L", 2)]
);
example!(
    qgan,
    "qgan",
    "qgan.aria",
    "QGANGenerator",
    &[("n", 2), ("L", 2)]
);
example!(
    qclassifier_rich,
    "qclassifier_rich",
    "qclassifier_rich.aria",
    "QClassifierRich",
    &[("N", 4), ("L", 2)]
);
example!(
    qssl,
    "qssl",
    "qssl.aria",
    "QSSLEncoder",
    &[("N", 2), ("L", 2)]
);
example!(
    sketch_qml,
    "sketch_qml",
    "sketch_qml.aria",
    "SketchQml",
    &[("k", 4)]
);
example!(
    strongly_entangling,
    "strongly_entangling",
    "strongly_entangling.aria",
    "StronglyEntangling",
    &[("n", 3), ("L", 2)]
);
example!(qasm_gpu, "qasm_gpu", "qasm_gpu.aria", "QasmGpu", &[]);
example!(hhl, "hhl", "hhl.aria", "Hhl", &[("n", 2)]);
example!(
    qsvt_invert,
    "qsvt_invert",
    "qsvt_invert.aria",
    "QsvtInvert",
    &[("degree", 4)]
);
// NOTE: shor_ecdlp is intentionally NOT here — its `oracle ec_step` subroutine
// is declared on a separate `qreg r[7]` that the lowering does not map into the
// main circuit's qubit space (cross-register oracle inlining is unsupported), so
// it parses + instantiates but does not lower. Labelled showcase in LIMITATIONS.md.
