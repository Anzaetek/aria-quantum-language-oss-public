// SPDX-License-Identifier: Apache-2.0
//! A small **independent** dense statevector simulator — the classical oracle
//! for the parametrized example circuits (ansätze / feature maps) that have no
//! single closed-form scalar answer.
//!
//! It consumes the SAME lowered `omega_core::CircuitIR` the runtime executes,
//! but applies textbook gate matrices to a `2ⁿ` amplitude vector here, in this
//! crate, with no dependency on the omega execution backend. The forward
//! `⟨Z_q⟩` profile it produces is compared against the backend's — a genuine
//! differential cross-check that catches lowering / execution regressions.
//!
//! `⟨Z⟩` is invariant under a global phase, so the single-qubit matrices only
//! need to be correct up to global phase; the controlled gates use exact
//! (phase-faithful) target blocks, which is what makes a control-conditioned
//! phase physical.

use crate::Complex64;
use omega_core::circuit::{CircuitIR, GateKind, ParamExpr};
use omega_core::params::ParameterBinding;

type C = Complex64;

fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}

/// e^{iθ}.
fn cis(theta: f64) -> C {
    C::new(theta.cos(), theta.sin())
}

/// Apply a 2×2 matrix `m` (row-major `[m00,m01,m10,m11]`) to qubit `q`.
fn apply_1q(sv: &mut [C], q: usize, m: [C; 4]) {
    let bit = 1usize << q;
    let mut i = 0;
    while i < sv.len() {
        if i & bit == 0 {
            let a0 = sv[i];
            let a1 = sv[i | bit];
            sv[i] = m[0] * a0 + m[1] * a1;
            sv[i | bit] = m[2] * a0 + m[3] * a1;
        }
        i += 1;
    }
}

/// Apply a 2×2 matrix `m` to `tgt`, conditioned on `ctrl` being |1⟩.
fn apply_ctrl_1q(sv: &mut [C], ctrl: usize, tgt: usize, m: [C; 4]) {
    let cb = 1usize << ctrl;
    let tb = 1usize << tgt;
    let mut i = 0;
    while i < sv.len() {
        if (i & cb != 0) && (i & tb == 0) {
            let a0 = sv[i];
            let a1 = sv[i | tb];
            sv[i] = m[0] * a0 + m[1] * a1;
            sv[i | tb] = m[2] * a0 + m[3] * a1;
        }
        i += 1;
    }
}

fn swap(sv: &mut [C], a: usize, b: usize) {
    let ab = 1usize << a;
    let bb = 1usize << b;
    let mut i = 0;
    while i < sv.len() {
        let ba = (i & ab != 0) as usize;
        let bbit = (i & bb != 0) as usize;
        if ba == 1 && bbit == 0 {
            sv.swap(i, (i & !ab) | bb);
        }
        i += 1;
    }
}

const ONE: C = C { re: 1.0, im: 0.0 };
const ZERO: C = C { re: 0.0, im: 0.0 };

fn rx(t: f64) -> [C; 4] {
    let (cs, sn) = ((t / 2.0).cos(), (t / 2.0).sin());
    [c(cs, 0.0), c(0.0, -sn), c(0.0, -sn), c(cs, 0.0)]
}
fn ry(t: f64) -> [C; 4] {
    let (cs, sn) = ((t / 2.0).cos(), (t / 2.0).sin());
    [c(cs, 0.0), c(-sn, 0.0), c(sn, 0.0), c(cs, 0.0)]
}
fn rz(t: f64) -> [C; 4] {
    [cis(-t / 2.0), ZERO, ZERO, cis(t / 2.0)]
}
/// U3(θ,φ,λ) in the standard (Qiskit) convention.
fn u3(theta: f64, phi: f64, lam: f64) -> [C; 4] {
    let (cs, sn) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    [
        c(cs, 0.0),
        -cis(lam) * sn,
        cis(phi) * sn,
        cis(phi + lam) * cs,
    ]
}

fn resolve(p: &ParamExpr, b: &ParameterBinding) -> Result<f64, String> {
    b.resolve(p).map_err(|e| format!("param resolve: {e:?}"))
}

fn gate_params(op: &omega_core::circuit::GateOp, b: &ParameterBinding) -> Result<Vec<f64>, String> {
    op.params.iter().map(|p| resolve(p, b)).collect()
}

/// Simulate `ir` with `params` (one value per free symbol, in ascending
/// symbol-ID order) and return `⟨Z_q⟩` for every qubit `q`.
///
/// A thin wrapper over [`statevector`]. `⟨Z_q⟩` is a marginal, and a marginal
/// is exactly what hid the HHL defect: the shipped circuit left the counting
/// register entangled with the answer, and every single-qubit `⟨Z⟩` still
/// matched the oracle to 0.0e0 because both sides ran the same lowered IR.
/// Reach for [`statevector`] when the property under test is a relative phase,
/// a post-selected branch, or an amplitude.
pub fn forward_z_expectations(ir: &CircuitIR, params: &[f64]) -> Result<Vec<f64>, String> {
    let sv = statevector(ir, params)?;
    let n = ir.num_qubits as usize;
    let mut z = vec![0.0; n];
    for (q, zq) in z.iter_mut().enumerate() {
        let bit = 1usize << q;
        let mut e = 0.0;
        for (x, amp) in sv.iter().enumerate() {
            let p = amp.norm_sqr();
            e += if x & bit == 0 { p } else { -p };
        }
        *zq = e;
    }
    Ok(z)
}

/// Simulate `ir` with `params` and return the full `2ⁿ` amplitude vector.
///
/// Qubit `q` is bit `q` of the basis index, so `q[0]` is the least significant
/// bit. Amplitudes are phase-faithful, not just correct up to a global phase.
pub fn statevector(ir: &CircuitIR, params: &[f64]) -> Result<Vec<C>, String> {
    // This is the ORACLE. An oracle that quietly ran a d = 3 register as
    // qubits would agree with a backend making the same mistake, which is
    // the one failure a cross-check exists to catch — so it refuses too.
    ir.refuse_qudits("aria-verify oracle")
        .map_err(|e| e.to_string())?;
    // Match the runtime's param ordering exactly (host::build_binding): the
    // params slice binds the free symbols in ascending SymbolId order, and a
    // wrong length is refused rather than zero-padded. The oracle must not
    // "agree" with the transport because both silently zeroed the same angle.
    let binding = ParameterBinding::from_flat(ir, params).map_err(|e| e.to_string())?;

    let n = ir.num_qubits as usize;
    if n > 20 {
        return Err(format!("sim refuses {n} qubits (2^{n} too large)"));
    }
    let dim = 1usize << n;
    let mut sv = vec![ZERO; dim];
    sv[0] = ONE;

    let rt2 = std::f64::consts::FRAC_1_SQRT_2;
    for op in &ir.ops {
        let q: Vec<usize> = op.qubits.iter().map(|x| x.0 as usize).collect();
        match op.gate {
            GateKind::Id | GateKind::Barrier | GateKind::Measure => {}
            // Reset is NOT a no-op: it is a non-unitary channel
            // (rho -> |0><0|_q (x) Tr_q(rho)) that this pure-state reference
            // simulator cannot represent. Skipping it meant `aria verify`
            // silently verified a DIFFERENT circuit than the one submitted —
            // a soundness hole in the verification layer, not a limitation.
            GateKind::Reset => {
                return Err("verify sim: Reset is a non-unitary channel and cannot be \
                            represented by this pure-state reference simulator"
                    .to_string())
            }
            GateKind::H => apply_1q(
                &mut sv,
                q[0],
                [c(rt2, 0.0), c(rt2, 0.0), c(rt2, 0.0), c(-rt2, 0.0)],
            ),
            GateKind::X => apply_1q(&mut sv, q[0], [ZERO, ONE, ONE, ZERO]),
            GateKind::Y => apply_1q(&mut sv, q[0], [ZERO, c(0.0, -1.0), c(0.0, 1.0), ZERO]),
            GateKind::Z => apply_1q(&mut sv, q[0], [ONE, ZERO, ZERO, c(-1.0, 0.0)]),
            GateKind::S => apply_1q(&mut sv, q[0], [ONE, ZERO, ZERO, c(0.0, 1.0)]),
            GateKind::Sdg => apply_1q(&mut sv, q[0], [ONE, ZERO, ZERO, c(0.0, -1.0)]),
            GateKind::T => apply_1q(
                &mut sv,
                q[0],
                [ONE, ZERO, ZERO, cis(std::f64::consts::FRAC_PI_4)],
            ),
            GateKind::Tdg => apply_1q(
                &mut sv,
                q[0],
                [ONE, ZERO, ZERO, cis(-std::f64::consts::FRAC_PI_4)],
            ),
            GateKind::Rx => apply_1q(&mut sv, q[0], rx(gate_params(op, &binding)?[0])),
            GateKind::Ry => apply_1q(&mut sv, q[0], ry(gate_params(op, &binding)?[0])),
            GateKind::Rz => apply_1q(&mut sv, q[0], rz(gate_params(op, &binding)?[0])),
            GateKind::U1 => apply_1q(
                &mut sv,
                q[0],
                [ONE, ZERO, ZERO, cis(gate_params(op, &binding)?[0])],
            ),
            GateKind::U2 => {
                let p = gate_params(op, &binding)?;
                apply_1q(&mut sv, q[0], u3(std::f64::consts::FRAC_PI_2, p[0], p[1]));
            }
            GateKind::U3 => {
                let p = gate_params(op, &binding)?;
                apply_1q(&mut sv, q[0], u3(p[0], p[1], p[2]));
            }
            GateKind::CX => apply_ctrl_1q(&mut sv, q[0], q[1], [ZERO, ONE, ONE, ZERO]),
            GateKind::CY => {
                apply_ctrl_1q(&mut sv, q[0], q[1], [ZERO, c(0.0, -1.0), c(0.0, 1.0), ZERO])
            }
            GateKind::CZ => apply_ctrl_1q(&mut sv, q[0], q[1], [ONE, ZERO, ZERO, c(-1.0, 0.0)]),
            GateKind::CRz => apply_ctrl_1q(&mut sv, q[0], q[1], rz(gate_params(op, &binding)?[0])),
            GateKind::CU3 => {
                let p = gate_params(op, &binding)?;
                apply_ctrl_1q(&mut sv, q[0], q[1], u3(p[0], p[1], p[2]));
            }
            GateKind::Swap => swap(&mut sv, q[0], q[1]),
            GateKind::Rbs => {
                // Givens rotation on the {|01⟩, |10⟩} pair of qubits
                // (q[0], q[1]): a01' = cos·a01 − sin·a10,
                // a10' = sin·a01 + cos·a10, where a01 has q[0]=0, q[1]=1.
                // Independent textbook realisation (no shared gate table).
                let t = gate_params(op, &binding)?[0];
                let (cs, sn) = (t.cos(), t.sin());
                let (b0, b1) = (1usize << q[0], 1usize << q[1]);
                let mut i = 0;
                while i < sv.len() {
                    // i indexes the |q0=0, q1=1⟩ member of each pair.
                    if (i & b0 == 0) && (i & b1 != 0) {
                        let j = (i | b0) & !b1; // q0=1, q1=0
                        let a01 = sv[i];
                        let a10 = sv[j];
                        sv[i] = c(cs, 0.0) * a01 - c(sn, 0.0) * a10;
                        sv[j] = c(sn, 0.0) * a01 + c(cs, 0.0) * a10;
                    }
                    i += 1;
                }
            }
            GateKind::CCX => {
                // Toffoli: X on q[2] when q[0]∧q[1]. Realised directly.
                let (c0, c1, t) = (1usize << q[0], 1usize << q[1], 1usize << q[2]);
                let mut i = 0;
                while i < sv.len() {
                    if (i & c0 != 0) && (i & c1 != 0) && (i & t == 0) {
                        sv.swap(i, i | t);
                    }
                    i += 1;
                }
            }
            GateKind::CSwap => {
                let (ctrl, a, b) = (1usize << q[0], 1usize << q[1], 1usize << q[2]);
                let mut i = 0;
                while i < sv.len() {
                    if (i & ctrl != 0) && (i & a != 0) && (i & b == 0) {
                        sv.swap(i, (i & !a) | b);
                    }
                    i += 1;
                }
            }
            // Qudit gates (PLAN-QUDIT.md Q2) on d = 2 wires, realised
            // independently of `omega-backend-quditsv`'s matrix table so the
            // d = 2 embedding there is checked against a second reading:
            // rxy(0, 1, θ, φ) = exp(−iθ/2 (cos φ X + sin φ Y)); csum = CX.
            GateKind::Rxy => {
                let p = gate_params(op, &binding)?;
                if p[0] != 0.0 || p[1] != 1.0 {
                    return Err(format!(
                        "verify sim: rxy levels ({}, {}) on a qubit; only (0, 1) exists",
                        p[0], p[1]
                    ));
                }
                let (c2, s2) = ((p[2] / 2.0).cos(), (p[2] / 2.0).sin());
                let (cf, sf) = (p[3].cos(), p[3].sin());
                // −i s2 (cf X + sf Y): X = [[0,1],[1,0]], Y = [[0,−i],[i,0]]
                let off01 = c(0.0, -s2) * c(cf, 0.0) + c(0.0, -s2) * c(0.0, -sf);
                let off10 = c(0.0, -s2) * c(cf, 0.0) + c(0.0, -s2) * c(0.0, sf);
                apply_1q(&mut sv, q[0], [c(c2, 0.0), off01, off10, c(c2, 0.0)]);
            }
            GateKind::CSum => apply_ctrl_1q(&mut sv, q[0], q[1], [ZERO, ONE, ONE, ZERO]),
            ref other => return Err(format!("sim: unsupported gate {other:?}")),
        }
    }

    Ok(sv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rx_theta_circuit() -> CircuitIR {
        // RX(theta)|0>: <Z> = cos(theta). One free symbol, named.
        let mut c = CircuitIR::new(1, omega_core::circuit::CircuitType::GateBased);
        c.symbols.insert(0, "theta".into());
        c.add_op(omega_core::circuit::GateOp {
            gate: GateKind::Rx,
            qubits: smallvec::smallvec![omega_core::circuit::Qubit(0)],
            params: smallvec::smallvec![omega_core::circuit::ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });
        c
    }

    #[test]
    fn forward_z_refuses_missing_param() {
        // An empty vector used to bind theta to 0.0 and return <Z> = 1.0,
        // a plausible number. It must be a refusal naming theta.
        let err = forward_z_expectations(&rx_theta_circuit(), &[]).unwrap_err();
        assert!(err.contains("got 0 value(s)"), "{err}");
        assert!(err.contains("1 free parameter(s)"), "{err}");
        assert!(err.contains("[theta]"), "{err}");
        // Extras are refused as well, not dropped.
        assert!(forward_z_expectations(&rx_theta_circuit(), &[0.1, 0.2]).is_err());
    }

    #[test]
    fn forward_z_runs_with_complete_params() {
        let theta = 1.0_f64;
        let z = forward_z_expectations(&rx_theta_circuit(), &[theta]).unwrap();
        assert_eq!(z.len(), 1);
        assert!((z[0] - theta.cos()).abs() < 1e-10, "<Z> = {}", z[0]);
    }
}
