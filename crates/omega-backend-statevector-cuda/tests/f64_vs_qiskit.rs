// SPDX-License-Identifier: Apache-2.0
//! The f64 GPU statevector against **Qiskit**, amplitude by amplitude.
//!
//! Closed forms I write myself are not independent evidence: they check the
//! kernels against my own algebra, and a shared misunderstanding of a
//! convention passes both sides. An f64 path that is wrong is worth less than an
//! f32 path that is right, so the gate that matters is an implementation nobody
//! here wrote.
//!
//! Qiskit's `Statevector` computes in complex128, so it can actually resolve the
//! difference between f32 and f64 — comparing an f32 GPU against it bottoms out
//! at ~1e-7 no matter how good the reference is. That is the whole argument for
//! this module, and the assertion below (1e-13) is a bar f32 cannot clear.
//!
//! Conventions pinned here, because these are what silently differ:
//! * qubit ordering — Qiskit is little-endian, `Statevector` index bit `q` is
//!   qubit `q`, which matches this backend's `mask = 1 << qubit`;
//! * `RY(θ) = [[cos θ/2, −sin θ/2], [sin θ/2, cos θ/2]]`;
//! * `RZ(θ) = diag(e^{−iθ/2}, e^{+iθ/2})` — the phase convention, NOT
//!   `diag(1, e^{iθ})`, so global phase agrees rather than merely the
//!   probabilities;
//! * `CX(control=qa, target=qb)` maps row `r = bit_qb·2 + bit_qa`, per the
//!   kernel's own documented layout.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use std::sync::Arc;

use cudarc::driver::CudaContext;
use omega_backend_statevector_cuda::f64_path::{KernelsF64, StateF64};

/// One gate in the shared circuit description. Kept tiny and explicit so the
/// Rust and Python sides cannot drift.
enum G {
    Ry(u32, f64),
    Rz(u32, f64),
    Cx(u32, u32),
}

fn ry_mat(t: f64) -> [(f64, f64); 4] {
    let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
    [(c, 0.0), (-s, 0.0), (s, 0.0), (c, 0.0)]
}

fn rz_mat(t: f64) -> [(f64, f64); 4] {
    let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
    // diag(e^{-i t/2}, e^{+i t/2})
    [(c, -s), (0.0, 0.0), (0.0, 0.0), (c, s)]
}

fn cx_mat() -> [f64; 32] {
    let mut u = [0.0f64; 32];
    let mut set = |row: usize, col: usize| u[2 * (row * 4 + col)] = 1.0;
    set(0, 0); // |qb=0,qa=0>
    set(2, 2); // |qb=1,qa=0>
    set(3, 1); // |qb=0,qa=1> -> |qb=1,qa=1>
    set(1, 3); // |qb=1,qa=1> -> |qb=0,qa=1>
    u
}

/// A circuit with entanglement, non-trivial phases, and repeated layers — the
/// shape a QAS config actually has, not a single gate.
fn circuit(n: u32) -> Vec<G> {
    let mut ops = Vec::new();
    for layer in 0..3u32 {
        for q in 0..n {
            // Deterministic, irrational-ish angles: no accidental symmetry that
            // would let a wrong sign or a swapped index still agree.
            let t = 0.37 * (q as f64 + 1.0) + 0.11 * (layer as f64 + 1.0);
            ops.push(G::Ry(q, t));
            ops.push(G::Rz(q, 1.3 * t + 0.2));
        }
        for q in 0..n.saturating_sub(1) {
            ops.push(G::Cx(q, q + 1));
        }
        // Close the ring so the entanglement is not a line.
        if n > 2 {
            ops.push(G::Cx(n - 1, 0));
        }
    }
    ops
}

fn qiskit_reference(n: u32, ops: &[G]) -> Option<Vec<(f64, f64)>> {
    let mut py_ops = String::new();
    for op in ops {
        match op {
            G::Ry(q, t) => py_ops.push_str(&format!("qc.ry({t:.17e}, {q})\n")),
            G::Rz(q, t) => py_ops.push_str(&format!("qc.rz({t:.17e}, {q})\n")),
            G::Cx(a, b) => py_ops.push_str(&format!("qc.cx({a}, {b})\n")),
        }
    }
    let script = format!(
        "import json\n\
         from qiskit import QuantumCircuit\n\
         from qiskit.quantum_info import Statevector\n\
         qc = QuantumCircuit({n})\n\
         {py_ops}\
         sv = Statevector.from_instruction(qc).data\n\
         print(json.dumps([[float(z.real), float(z.imag)] for z in sv]))\n"
    );
    // The interpreter lives at the REPO root, but cargo runs a test with its
    // CWD set to the PACKAGE root, so a relative path silently fails to resolve
    // and the reference "does not run" — which this test treats as a failure,
    // correctly, but for the wrong reason. Derive it from the manifest dir.
    let py = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.venv-qiskit/bin/python");
    let out = std::process::Command::new(&py)
        .arg("-c")
        .arg(&script)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "qiskit reference failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let v: Vec<Vec<f64>> = serde_json::from_str(text.trim()).ok()?;
    Some(v.into_iter().map(|p| (p[0], p[1])).collect())
}

#[test]
fn f64_statevector_matches_qiskit_amplitude_by_amplitude() {
    let n = 6u32;
    let ops = circuit(n);

    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("no CUDA device — skipping (device test)");
        return;
    };
    let Some(reference) = qiskit_reference(n, &ops) else {
        // A missing reference must not read as a pass: this is the only
        // independent check, so say it did not run.
        panic!(
            "the Qiskit reference did not run — this test is the independent \
             gate and cannot be skipped silently. Build ./.venv-qiskit \
             (see PREREQUISITES.md)."
        );
    };

    let kernels = Arc::new(KernelsF64::load(&ctx).expect("f64 kernels compile"));
    let mut st = StateF64::zero(&ctx, kernels, n).expect("alloc");
    for op in &ops {
        match op {
            G::Ry(q, t) => st.apply_1q(*q, ry_mat(*t)).unwrap(),
            G::Rz(q, t) => st.apply_1q(*q, rz_mat(*t)).unwrap(),
            G::Cx(a, b) => st.apply_2q(*a, *b, cx_mat()).unwrap(),
        }
    }
    let host = st.to_host().unwrap();
    let dim = 1usize << n;
    assert_eq!(reference.len(), dim, "reference length");

    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    for i in 0..dim {
        let (gre, gim) = (host[2 * i], host[2 * i + 1]);
        let (rre, rim) = reference[i];
        // Complex modulus of the difference: catches a phase error that a
        // probability comparison would hide entirely.
        let d = ((gre - rre).powi(2) + (gim - rim).powi(2)).sqrt();
        if d > worst {
            worst = d;
            worst_at = i;
        }
    }
    let gates = ops.len();
    eprintln!(
        "f64 GPU vs Qiskit: n={n}, {gates} gates, worst |Δamplitude| = {worst:.3e} at index {worst_at}"
    );
    // f32 would land near 1e-7 on a circuit this size; 1e-13 is unreachable for
    // it and comfortable for f64.
    assert!(
        worst < 1e-13,
        "worst amplitude deviation {worst:.3e} at index {worst_at} — f64 path disagrees with Qiskit"
    );
}

#[test]
fn f64_expectation_matches_qiskit() {
    let n = 5u32;
    let ops = circuit(n);
    let Ok(ctx) = CudaContext::new(0) else {
        return;
    };
    let Some(reference) = qiskit_reference(n, &ops) else {
        panic!("the Qiskit reference did not run — see the other test");
    };

    let kernels = Arc::new(KernelsF64::load(&ctx).expect("kernels"));
    let mut st = StateF64::zero(&ctx, kernels, n).expect("alloc");
    for op in &ops {
        match op {
            G::Ry(q, t) => st.apply_1q(*q, ry_mat(*t)).unwrap(),
            G::Rz(q, t) => st.apply_1q(*q, rz_mat(*t)).unwrap(),
            G::Cx(a, b) => st.apply_2q(*a, *b, cx_mat()).unwrap(),
        }
    }

    let mut worst = 0.0f64;
    for q in 0..n {
        // <Z_q> from the reference amplitudes, computed here rather than asking
        // Qiskit for it, so the comparison is against the reference STATE.
        let mut want = 0.0f64;
        for (i, (re, im)) in reference.iter().enumerate() {
            let p = re * re + im * im;
            want += if i & (1usize << q) == 0 { p } else { -p };
        }
        let got = st.expectation_z(q).unwrap();
        worst = worst.max((got - want).abs());
    }
    eprintln!("f64 GPU <Z_q> vs Qiskit, worst over {n} wires = {worst:.3e}");
    assert!(worst < 1e-13, "worst <Z> deviation {worst:.3e}");
}
