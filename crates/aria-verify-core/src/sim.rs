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

/// Simulate `ir` with `params` (bound in `ir.param_symbols` order) and return
/// `⟨Z_q⟩` for every qubit `q`.
pub fn forward_z_expectations(ir: &CircuitIR, params: &[f64]) -> Result<Vec<f64>, String> {
    // Match the runtime's param ordering exactly (host::build_binding): the
    // params slice binds the free symbols in ascending SymbolId order.
    let mut binding = ParameterBinding::new();
    let mut symbol_ids: Vec<omega_core::circuit::SymbolId> = ir.symbols.keys().copied().collect();
    symbol_ids.sort_unstable();
    for (i, &sid) in symbol_ids.iter().enumerate() {
        binding.bind(sid, params.get(i).copied().unwrap_or(0.0));
    }

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
            GateKind::Id | GateKind::Barrier | GateKind::Measure | GateKind::Reset => {}
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
            ref other => return Err(format!("sim: unsupported gate {other:?}")),
        }
    }

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
