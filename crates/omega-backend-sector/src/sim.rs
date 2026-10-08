// SPDX-License-Identifier: Apache-2.0
//! `SectorBackend`: the dense statevector restricted to one Hamming-weight
//! sector.
//!
//! The representation is the ordinary qubit basis `|b_{n-1} … b_0⟩`, kept
//! only for the `C(n, k)` strings with `k` ones. Under Jordan–Wigner that IS
//! the `k`-particle Fock sector, so a number-conserving circuit — `Rbs`,
//! `CU3(0, φ, λ)` = `cphase`, `CRz`, `Swap`, every diagonal gate — acts by
//! exactly the matrices the dense backend applies, and the Jordan–Wigner
//! signs stay where `FermionicOp::jordan_wigner()` already put them: in the
//! observable's Pauli strings. Nothing here knows about fermions; it knows
//! that `X` on an untouched wire sets `k`, and that every later gate must
//! commute with total number.
//!
//! **Admission is decided on the gate's matrix, not its name.** `CU3` with
//! `θ = 0` is admitted; `CU3` with `θ ≠ 0` is refused; `U3(0, φ, λ)` is a
//! phase and admitted. The check is `entries between different weights are
//! zero`, on the same `Gate1Q`/`Gate2Q` the statevector backend builds.

use num_complex::Complex64;
use omega_backend_statevector::gates::{self, Gate1Q, Gate2Q};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp};
use omega_core::defer_measure::{prepare_for_expectation, prepare_for_expectation_multi};
use omega_core::error::{OmegaError, Result};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable, PauliOp};
use omega_core::params::ParameterBinding;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::collections::HashMap;

use crate::basis::{binomial, Sector};

/// Largest sector this backend will allocate: `2^27` amplitudes = 2 GiB.
/// A reference engine, not a capacity play — the MPS backends own that.
pub const MAX_DIM: u128 = 1 << 27;

/// `execute` with `shots: None` returns the dense `2^n` statevector so the
/// result type stays what every caller expects, but only up to this width;
/// past it the dense vector is the thing this backend exists to avoid.
pub const MAX_DENSE_READOUT_QUBITS: u32 = 26;

const ZERO_TOL: f64 = 1e-12;

#[derive(Clone, Debug, Default)]
pub struct SectorBackend;

impl SectorBackend {
    pub fn new() -> Self {
        Self
    }
}

/// Everything the first pass over the circuit decides before any amplitude
/// exists: which wires start occupied, and the admitted, parameter-resolved
/// gate list.
struct Plan {
    occupation: u64,
    gates: Vec<Resolved>,
}

enum Resolved {
    OneQ { q: u32, m: Gate1Q },
    TwoQ { q0: u32, q1: u32, m: Gate2Q },
}

fn refuse(msg: String) -> OmegaError {
    OmegaError::Unsupported(format!("sector backend: {msg}"))
}

fn resolve_params(op: &GateOp, params: &ParameterBinding) -> Result<Vec<f64>> {
    op.params.iter().map(|p| params.resolve(p)).collect()
}

/// The dense backend's matrix for a one-qubit gate, or `None` if the kind is
/// not a one-qubit unitary.
fn matrix_1q(kind: &GateKind, p: &[f64]) -> Option<Gate1Q> {
    Some(match kind {
        GateKind::H => gates::h(),
        GateKind::X => gates::x(),
        GateKind::Y => gates::y(),
        GateKind::Z => gates::z(),
        GateKind::S => gates::s(),
        GateKind::Sdg => gates::sdg(),
        GateKind::T => gates::t(),
        GateKind::Tdg => gates::tdg(),
        GateKind::Sx => gates::sx(),
        GateKind::Sxdg => gates::sxdg(),
        GateKind::Id => gates::_id(),
        GateKind::Rx => gates::rx(p[0]),
        GateKind::Ry => gates::ry(p[0]),
        GateKind::Rz => gates::rz(p[0]),
        GateKind::U3 => gates::u3(p[0], p[1], p[2]),
        GateKind::U2 => gates::u2(p[0], p[1]),
        GateKind::U1 => gates::u1(p[0]),
        _ => return None,
    })
}

fn matrix_2q(kind: &GateKind, p: &[f64]) -> Option<Gate2Q> {
    Some(match kind {
        GateKind::CX => gates::cx(),
        GateKind::CY => gates::cy(),
        GateKind::CZ => gates::cz(),
        GateKind::Swap => gates::swap(),
        GateKind::CRz => gates::crz(p[0]),
        GateKind::CU3 => gates::cu3(p[0], p[1], p[2]),
        GateKind::Rbs => gates::rbs(p[0]),
        _ => return None,
    })
}

/// A one-qubit matrix commutes with number iff it is diagonal.
fn conserves_1q(m: &Gate1Q) -> bool {
    m[1].norm() < ZERO_TOL && m[2].norm() < ZERO_TOL
}

/// A two-qubit matrix commutes with number iff every entry between basis
/// states of different weight is zero: weights are `|00⟩:0, |01⟩:1, |10⟩:1,
/// |11⟩:2`, so the allowed pattern is `{0}`, `{1,2}×{1,2}`, `{3}`.
fn conserves_2q(m: &Gate2Q) -> bool {
    const W: [u32; 4] = [0, 1, 1, 2];
    (0..4).all(|r| (0..4).all(|c| W[r] == W[c] || m[r * 4 + c].norm() < ZERO_TOL))
}

fn plan(circuit: &CircuitIR, params: &ParameterBinding) -> Result<Plan> {
    if circuit.circuit_type == CircuitType::Photonic {
        return Err(refuse("photonic circuits are not qubit circuits".into()));
    }
    let n = circuit.num_qubits;
    if n > 64 {
        return Err(refuse(format!(
            "{n} qubits: the sector basis is keyed by a u64 mask, 64 is the limit"
        )));
    }
    let mut touched: u64 = 0;
    let mut occupation: u64 = 0;
    let mut out = Vec::with_capacity(circuit.ops.len());
    for (i, op) in circuit.ops.iter().enumerate() {
        if op.condition.is_some() {
            return Err(refuse(format!(
                "op {i} ({:?}) is classically conditioned; feed-forward is not supported",
                op.gate
            )));
        }
        for q in &op.qubits {
            if q.0 >= n {
                return Err(OmegaError::InvalidCircuit(format!(
                    "op {i} ({:?}) names qubit {} of {n}",
                    op.gate, q.0
                )));
            }
        }
        match &op.gate {
            GateKind::Barrier | GateKind::Id => continue,
            // Trailing measurement is what `execute` samples; a measurement
            // in the middle is what `MidCircuitMode::Skip` skips. Either way
            // the wire is now "used".
            GateKind::Measure => {
                for q in &op.qubits {
                    touched |= 1 << q.0;
                }
                continue;
            }
            GateKind::Reset => {
                return Err(refuse(format!(
                    "op {i}: Reset does not conserve particle number"
                )))
            }
            GateKind::Custom(id) => {
                return Err(refuse(format!(
                    "op {i}: custom gate {id:?} is not expanded here; inline it first"
                )))
            }
            GateKind::CCX | GateKind::CSwap => {
                return Err(refuse(format!(
                    "op {i} ({:?}): three-qubit gates are not implemented in the sector basis yet",
                    op.gate
                )))
            }
            GateKind::PhaseShifter | GateKind::BeamSplitterRx => {
                return Err(refuse(format!(
                    "op {i} ({:?}): photonic gate in a qubit circuit",
                    op.gate
                )))
            }
            // The occupation layer: X on a wire nothing has touched yet
            // creates a particle there. Anywhere else X changes the weight
            // and cannot be represented — the same rule as the ffsim bridge.
            GateKind::X => {
                let q = op.qubits[0].0;
                if touched & (1 << q) != 0 {
                    return Err(refuse(format!(
                        "op {i}: X on qubit {q} after another gate on that wire changes the \
                         particle number; X is admitted only as the occupation layer (first op \
                         on its wire)"
                    )));
                }
                occupation ^= 1 << q;
                touched |= 1 << q;
                continue;
            }
            _ => {}
        }
        let p = resolve_params(op, params)?;
        match op.qubits.len() {
            1 => {
                let m = matrix_1q(&op.gate, &p).ok_or_else(|| {
                    refuse(format!("op {i} ({:?}): unknown one-qubit gate", op.gate))
                })?;
                if !conserves_1q(&m) {
                    return Err(refuse(format!(
                        "op {i} ({:?} {p:?}) on qubit {} does not conserve particle number \
                         (its matrix has off-diagonal entries)",
                        op.gate, op.qubits[0].0
                    )));
                }
                touched |= 1 << op.qubits[0].0;
                out.push(Resolved::OneQ {
                    q: op.qubits[0].0,
                    m,
                });
            }
            2 => {
                let m = matrix_2q(&op.gate, &p).ok_or_else(|| {
                    refuse(format!("op {i} ({:?}): unknown two-qubit gate", op.gate))
                })?;
                if !conserves_2q(&m) {
                    return Err(refuse(format!(
                        "op {i} ({:?} {p:?}) on qubits ({}, {}) does not conserve particle \
                         number (its matrix connects |00⟩/|11⟩ with |01⟩/|10⟩)",
                        op.gate, op.qubits[0].0, op.qubits[1].0
                    )));
                }
                let (q0, q1) = (op.qubits[0].0, op.qubits[1].0);
                if q0 == q1 {
                    return Err(OmegaError::InvalidCircuit(format!(
                        "op {i} ({:?}) names qubit {q0} twice",
                        op.gate
                    )));
                }
                touched |= (1 << q0) | (1 << q1);
                out.push(Resolved::TwoQ { q0, q1, m });
            }
            other => return Err(refuse(format!("op {i} ({:?}) on {other} qubits", op.gate))),
        }
    }
    Ok(Plan {
        occupation,
        gates: out,
    })
}

/// The sector state after the circuit.
struct State {
    sector: Sector,
    amp: Vec<Complex64>,
}

fn run(circuit: &CircuitIR, params: &ParameterBinding) -> Result<State> {
    let plan = plan(circuit, params)?;
    let n = circuit.num_qubits;
    let k = plan.occupation.count_ones();
    let dim = binomial(n, k).unwrap_or(u128::MAX);
    if dim > MAX_DIM {
        return Err(OmegaError::Backend(format!(
            "sector backend: C({n}, {k}) = {dim} amplitudes exceeds the {MAX_DIM} cap \
             ({} GiB); this is the exact reference engine, not the large-N one",
            (MAX_DIM * 16) >> 30
        )));
    }
    let sector = Sector::new(n, k);
    let mut amp = vec![Complex64::new(0.0, 0.0); sector.dim()];
    amp[sector.rank(plan.occupation)] = Complex64::new(1.0, 0.0);
    for g in &plan.gates {
        match g {
            Resolved::OneQ { q, m } => apply_diag_1q(&sector, &mut amp, *q, m),
            Resolved::TwoQ { q0, q1, m } => apply_2q(&sector, &mut amp, *q0, *q1, m),
        }
    }
    Ok(State { sector, amp })
}

fn apply_diag_1q(sector: &Sector, amp: &mut [Complex64], q: u32, m: &Gate1Q) {
    let (d0, d1) = (m[0], m[3]);
    for (a, &s) in amp.iter_mut().zip(sector.states()) {
        *a *= if (s >> q) & 1 == 1 { d1 } else { d0 };
    }
}

/// `Gate2Q` index is `(bit q0) << 1 | (bit q1)`, q0 the first qubit of the op
/// — the dense backend's convention. Number conservation lets the matrix act
/// on |00⟩ and |11⟩ by a scalar and on {|01⟩, |10⟩} by a 2×2 block, and the
/// partner of a mixed state is in the same sector: `s ^ (1<<q0) ^ (1<<q1)`.
fn apply_2q(sector: &Sector, amp: &mut [Complex64], q0: u32, q1: u32, m: &Gate2Q) {
    let m00 = m[0];
    let m11 = m[15];
    // Block on (|01⟩, |10⟩): rows/cols 1 and 2.
    let (a11, a12, a21, a22) = (m[5], m[6], m[9], m[10]);
    let flip = (1u64 << q0) | (1u64 << q1);
    for r in 0..amp.len() {
        let s = sector.mask(r);
        let b0 = (s >> q0) & 1;
        let b1 = (s >> q1) & 1;
        match (b0, b1) {
            (0, 0) => amp[r] *= m00,
            (1, 1) => amp[r] *= m11,
            // Visit each pair once, from its |10⟩ member (index 2).
            (1, 0) => {
                let t = sector.rank(s ^ flip); // the |01⟩ member (index 1)
                let x10 = amp[r];
                let x01 = amp[t];
                amp[t] = a11 * x01 + a12 * x10;
                amp[r] = a21 * x01 + a22 * x10;
            }
            _ => {}
        }
    }
}

/// `⟨ψ|P|ψ⟩` for one Pauli string. `P|s⟩ = i^{|Y|} (−1)^{|s ∧ (Y∪Z)|} |s ⊕ (X∪Y)⟩`;
/// the target is outside the sector (amplitude 0) unless the flips balance.
fn pauli_expectation(st: &State, paulis: &[(u32, PauliOp)]) -> f64 {
    let mut xmask = 0u64;
    let mut zmask = 0u64;
    let mut ys = 0u32;
    for (q, p) in paulis {
        match p {
            PauliOp::I => {}
            PauliOp::X => xmask |= 1 << q,
            PauliOp::Y => {
                xmask |= 1 << q;
                zmask |= 1 << q;
                ys += 1;
            }
            PauliOp::Z => zmask |= 1 << q,
        }
    }
    let k = st.sector.k();
    let iy = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, -1.0),
    ][(ys % 4) as usize];
    let mut acc = Complex64::new(0.0, 0.0);
    for (r, &s) in st.sector.states().iter().enumerate() {
        let t = s ^ xmask;
        if t.count_ones() != k {
            continue;
        }
        let sign = if (s & zmask).count_ones() % 2 == 1 {
            -1.0
        } else {
            1.0
        };
        let target = if xmask == 0 { r } else { st.sector.rank(t) };
        acc += st.amp[target].conj() * st.amp[r] * iy * sign;
    }
    acc.re
}

fn observable_expectation(st: &State, o: &Observable) -> f64 {
    o.terms
        .iter()
        .map(|(c, p)| c * pauli_expectation(st, p))
        .sum()
}

impl Backend for SectorBackend {
    fn name(&self) -> &str {
        "sector"
    }

    fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> Result<ExecResult> {
        circuit.refuse_qudits(self.name())?;
        if config.mid_circuit_mode == MidCircuitMode::Collapse {
            return Err(refuse(
                "mid-circuit collapse is not supported; measurements are sampled at the end".into(),
            ));
        }
        let st = run(circuit, params)?;
        let n = circuit.num_qubits;
        match config.shots {
            None => {
                if n > MAX_DENSE_READOUT_QUBITS {
                    return Err(refuse(format!(
                        "dense statevector readout of {n} qubits (2^{n} amplitudes) is what \
                         this backend exists to avoid; use `expectation` or `shots`"
                    )));
                }
                let mut dense = vec![Complex64::new(0.0, 0.0); 1usize << n];
                for (r, &s) in st.sector.states().iter().enumerate() {
                    dense[s as usize] = st.amp[r];
                }
                Ok(ExecResult::Statevector(dense))
            }
            Some(shots) => {
                let mut rng: StdRng = match config.seed {
                    Some(seed) => StdRng::seed_from_u64(seed),
                    None => rand::make_rng::<StdRng>(),
                };
                let probs: Vec<f64> = st.amp.iter().map(|a| a.norm_sqr()).collect();
                let total: f64 = probs.iter().sum();
                let mut counts: HashMap<u64, u32> = HashMap::new();
                for _ in 0..shots {
                    let mut u = rng.random::<f64>() * total;
                    let mut pick = probs.len() - 1;
                    for (i, p) in probs.iter().enumerate() {
                        if u < *p {
                            pick = i;
                            break;
                        }
                        u -= p;
                    }
                    *counts.entry(st.sector.mask(pick)).or_insert(0) += 1;
                }
                Ok(ExecResult::counts_from_u64(counts, n))
            }
        }
    }

    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<f64> {
        circuit.refuse_qudits(self.name())?;
        observable.validate_qubits(circuit.num_qubits)?;
        let (deferred, observable) = prepare_for_expectation(circuit, observable)?;
        let st = run(&deferred, params)?;
        Ok(observable_expectation(&st, &observable))
    }

    fn expectation_multi(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observables: &[Observable],
    ) -> Result<Vec<f64>> {
        circuit.refuse_qudits(self.name())?;
        for o in observables {
            o.validate_qubits(circuit.num_qubits)?;
        }
        let (deferred, observables) = prepare_for_expectation_multi(circuit, observables)?;
        let st = run(&deferred, params)?;
        Ok(observables
            .iter()
            .map(|o| observable_expectation(&st, o))
            .collect())
    }
}
