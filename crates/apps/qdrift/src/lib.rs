// SPDX-License-Identifier: Apache-2.0
//! qdrift — deterministic realization of QDrift Hamiltonian simulation vs an exact exp(-iHt).
//!
//! WHAT: evolve H = a·Z₀Z₁ + b·(X₀ + X₁) (a = 0.6, b = 0.3) from |00⟩ for t = 1 with N QDrift
//!   micro-steps — each a uniform λτ rotation (λ = a+2b = 1.2, τ = t/N) about one term, terms
//!   appearing with frequency ∝ |c_j| (2:1:1 per block) — and read ⟨Z₀⟩, ⟨Z₁⟩.
//! QUANTUM: run qdrift.aria through the omega runtime at N and N/2.
//! CLASSICAL: dense U = exp(-iHt) by scaling-and-squaring (independent of the circuit); read ⟨Z_q⟩.
//! CHECK: circuit ⟨Z_q⟩ matches exact exp(-iHt) within 2e-3 at N = 1200, and the QDrift error → 0
//!   as N grows. H is symmetric under 0↔1 ⇒ ⟨Z₀⟩ = ⟨Z₁⟩, so the check is ordering-independent.

use aria_verify_core::{banner, harness, resolve, Observable, Transport, Verdict};
use num_complex::Complex64 as C;

const A: f64 = 0.6; // Z0Z1 coupling
const B: f64 = 0.3; // transverse field on each qubit
const T: f64 = 1.0; // evolution time
const N: i64 = 1200; // QDrift micro-steps for the pass/fail check (multiple of 4)
const TOL: f64 = 2e-3;

type M4 = [[C; 4]; 4];

fn zero() -> M4 {
    [[C::new(0.0, 0.0); 4]; 4]
}
fn ident() -> M4 {
    let mut m = zero();
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = C::new(1.0, 0.0);
    }
    m
}
fn matmul(a: &M4, b: &M4) -> M4 {
    let mut c = zero();
    for i in 0..4 {
        for k in 0..4 {
            let aik = a[i][k];
            if aik == C::new(0.0, 0.0) {
                continue;
            }
            for j in 0..4 {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

/// exp(M) for a 4×4 complex matrix via scaling-and-squaring + a truncated Taylor series.
fn mat_exp(m: &M4) -> M4 {
    let norm = m
        .iter()
        .map(|r| r.iter().map(|c| c.norm()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let s = (norm.log2().ceil().max(0.0)) as u32 + 4;
    let scale = 2f64.powi(s as i32);
    let mut a = zero();
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = m[i][j] / scale;
        }
    }
    let mut e = ident();
    let mut term = ident();
    for k in 1..=20 {
        term = matmul(&term, &a);
        let inv_k = C::new(1.0 / k as f64, 0.0);
        for i in 0..4 {
            for j in 0..4 {
                term[i][j] *= inv_k;
                e[i][j] += term[i][j];
            }
        }
    }
    for _ in 0..s {
        e = matmul(&e, &e);
    }
    e
}

/// Exact ⟨Z₀⟩, ⟨Z₁⟩ after |ψ(t)⟩ = exp(-iHt)|00⟩ (basis index = 2·q₁ + q₀).
fn exact_z() -> (f64, f64) {
    let zz = [1.0, -1.0, -1.0, 1.0];
    let mut h = zero();
    for i in 0..4 {
        h[i][i] += C::new(A * zz[i], 0.0);
        h[i][i ^ 1] += C::new(B, 0.0);
        h[i][i ^ 2] += C::new(B, 0.0);
    }
    let mut m = zero();
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = C::new(0.0, -T) * h[i][j];
        }
    }
    let u = mat_exp(&m);
    let psi = [u[0][0], u[1][0], u[2][0], u[3][0]];
    let z0 = [1.0, -1.0, 1.0, -1.0];
    let z1 = [1.0, 1.0, -1.0, -1.0];
    let (mut e0, mut e1) = (0.0, 0.0);
    for i in 0..4 {
        let p = psi[i].norm_sqr();
        e0 += z0[i] * p;
        e1 += z1[i] * p;
    }
    (e0, e1)
}

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "qdrift",
        "QDrift (uniform λτ micro-rotations, freq ∝ |c_j|) of H = 0.6·Z0Z1 + 0.3·(X0+X1), t=1 — ⟨Z_q⟩ vs exp(-iHt)",
        &transport.label(guest),
    );

    let (ez0, ez1) = exact_z();

    let z_at = |n: i64| -> Result<(f64, f64), String> {
        let lowered = harness::load_lowered("qdrift.aria", "QDrift", &[("N", n)])?;
        let obs = vec![Observable::z(0), Observable::z(1)];
        let (zv, _) = harness::execute_report(
            transport,
            lowered.ir,
            harness::AppMode::Expectations(obs),
            &[],
        )?;
        Ok((zv[0], zv[1]))
    };

    let (h0, h1) = z_at(N / 2)?;
    let (c0, c1) = z_at(N)?;
    let err_half = (h0 - ez0).abs().max((h1 - ez1).abs());
    let err_full = (c0 - ez0).abs().max((c1 - ez1).abs());
    // Gate the convergence claim, not just print it: doubling N must reduce the error.
    if err_full > err_half {
        return Err(format!(
            "QDrift error did not decrease with N: {err_half:.2e} (N/2) → {err_full:.2e} (N)"
        ));
    }

    println!("  exact ⟨Z0⟩ = ⟨Z1⟩ = {ez0:+.10}");
    println!(
        "  QDrift error: N={} → {err_half:.2e},  N={} → {err_full:.2e}  (→ 0 as N grows)",
        N / 2,
        N
    );

    Ok(banner::report_values(
        "qdrift",
        &format!("circuit ⟨Z_q⟩ (N={N})"),
        &[c0, c1],
        "exact exp(-iHt) ⟨Z_q⟩",
        &[ez0, ez1],
        TOL,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_z_symmetric_and_physical() {
        let (z0, z1) = exact_z();
        assert!((z0 - z1).abs() < 1e-12);
        assert!(z0 < 1.0 && z0 > 0.0);
    }
}
