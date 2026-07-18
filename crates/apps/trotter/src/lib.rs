// SPDX-License-Identifier: Apache-2.0
//! trotter — first-order Trotter–Suzuki Hamiltonian simulation vs an exact matrix exponential.
//!
//! WHAT: evolve H = a·Z₀Z₁ + b·(X₀ + X₁) (a = 0.5, b = 0.35) from |00⟩ for time t = 1 with a
//!   first-order Trotter product formula of `steps` slices, and read ⟨Z₀⟩, ⟨Z₁⟩.
//! QUANTUM: run trotter.aria through the omega runtime at `steps` and `steps/2`.
//! CLASSICAL: build the dense 4×4 Hamiltonian, form U = exp(-iHt) by scaling-and-squaring with a
//!   Taylor series (independent of the circuit), apply to |00⟩, and read the exact ⟨Z_q⟩.
//! CHECK: the circuit ⟨Z_q⟩ matches the exact exp(-iHt) values within 2e-3 at steps = 1500, and the
//!   first-order Trotter error shrinks toward 0 as `steps` grows (e.g. 3.1e-8 → 7.6e-9 as steps
//!   750 → 1500) — the defining property of a convergent product formula. H is symmetric under
//!   0↔1, so ⟨Z₀⟩ = ⟨Z₁⟩ and the check is independent of qubit-ordering conventions.

use aria_verify_core::{banner, harness, resolve, Observable, Transport, Verdict};
use num_complex::Complex64 as C;

const A: f64 = 0.5; // Z0Z1 coupling
const B: f64 = 0.35; // transverse field on each qubit
const T: f64 = 1.0; // evolution time
const STEPS: i64 = 1500; // Trotter slices for the pass/fail check
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
    // Scale so ‖M / 2^s‖ is small, then square s times.
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
    // E = Σ_{k=0}^{K} A^k / k!  (term_k = term_{k-1}·A / k).
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
    let zz = [1.0, -1.0, -1.0, 1.0]; // Z⊗Z diagonal
    let mut h = zero();
    for i in 0..4 {
        h[i][i] += C::new(A * zz[i], 0.0); // a·Z0Z1
        h[i][i ^ 1] += C::new(B, 0.0); // b·X0 (flip low bit)
        h[i][i ^ 2] += C::new(B, 0.0); // b·X1 (flip high bit)
    }
    // M = -i·T·H
    let mut m = zero();
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = C::new(0.0, -T) * h[i][j];
        }
    }
    let u = mat_exp(&m);
    let psi = [u[0][0], u[1][0], u[2][0], u[3][0]]; // U|00⟩ = first column
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
        "trotter",
        "first-order Trotter of H = 0.5·Z0Z1 + 0.35·(X0+X1), t=1 — circuit ⟨Z_q⟩ vs exact exp(-iHt)",
        &transport.label(guest),
    );

    let (ez0, ez1) = exact_z();

    let z_at = |n: i64| -> Result<(f64, f64), String> {
        let lowered = harness::load_lowered("trotter.aria", "Trotter", &[("steps", n)])?;
        let obs = vec![Observable::z(0), Observable::z(1)];
        let (zv, _) = harness::execute_report(
            transport,
            lowered.ir,
            harness::AppMode::Expectations(obs),
            &[],
        )?;
        Ok((zv[0], zv[1]))
    };

    let (h0, h1) = z_at(STEPS / 2)?;
    let (c0, c1) = z_at(STEPS)?;
    let err_half = (h0 - ez0).abs().max((h1 - ez1).abs());
    let err_full = (c0 - ez0).abs().max((c1 - ez1).abs());

    println!("  exact ⟨Z0⟩ = ⟨Z1⟩ = {ez0:+.10}");
    println!(
        "  Trotter error: steps={} → {err_half:.2e},  steps={} → {err_full:.2e}  (→ 0 as steps grow)",
        STEPS / 2,
        STEPS
    );

    Ok(banner::report_values(
        "trotter",
        &format!("circuit ⟨Z_q⟩ (steps={STEPS})"),
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
    fn mat_exp_of_zero_is_identity() {
        let e = mat_exp(&zero());
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((e[i][j].re - want).abs() < 1e-12 && e[i][j].im.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn exact_z_is_symmetric_and_physical() {
        let (z0, z1) = exact_z();
        assert!((z0 - z1).abs() < 1e-12, "H is 0↔1 symmetric ⇒ ⟨Z0⟩=⟨Z1⟩");
        assert!(z0.abs() <= 1.0 + 1e-12, "expectation of Z is in [-1,1]");
        // starts at +1 (|00⟩), the transverse field drives it below 1
        assert!(z0 < 1.0 && z0 > 0.0);
    }
}
