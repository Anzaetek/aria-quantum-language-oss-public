// SPDX-License-Identifier: Apache-2.0
//! The mixed-radix state, its gate kernels, and the [`Backend`] door.

use num_complex::Complex64;
use omega_backend_statevector::gates as qg;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp};
use omega_core::defer_measure::{prepare_for_expectation, prepare_for_expectation_multi};
use omega_core::error::{OmegaError, Result};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable, PauliOp};
use omega_core::params::ParameterBinding;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::collections::HashMap;

use crate::capacity;
use crate::gates::{self, Dense};

/// The exact mixed-radix statevector backend. Expectation, statevector, and
/// sampling (qubit-only for sampling — see [`QuditSvBackend::execute`]).
#[derive(Clone, Debug, Default)]
pub struct QuditSvBackend;

impl QuditSvBackend {
    pub fn new() -> Self {
        Self
    }
}

/// What a run held and — the point of the type — that it discarded nothing.
///
/// `exact` is `true` by construction and the bound is zero; the field is
/// carried so a consumer keyed on the other engines' certificates sees the
/// same key with the same meaning, not an absence it has to interpret. It is
/// honest only because [`capacity::check`] refuses anything the engine
/// could not hold exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct QuditSvCertificate {
    pub exact: bool,
    /// The bound on `|Δ⟨O⟩|`: always `0.0` here.
    pub dropped_mass: f64,
    /// Per-wire local dimensions, in wire order.
    pub dims: Vec<u32>,
    /// `Π dᵢ` — the amplitudes held.
    pub product_dim: usize,
}

fn refuse(msg: String) -> OmegaError {
    OmegaError::Unsupported(format!("quditsv: {msg}"))
}

/// The state. `strides[w] = Π_{v<w} dims[v]`: wire 0 is the least
/// significant digit, matching qubit 0 = least significant bit.
pub struct State {
    pub dims: Vec<u32>,
    pub strides: Vec<usize>,
    pub amp: Vec<Complex64>,
}

impl State {
    fn zero(dims: Vec<u32>, product: usize) -> Self {
        let mut strides = Vec::with_capacity(dims.len());
        let mut s = 1usize;
        for &d in &dims {
            strides.push(s);
            s *= d as usize;
        }
        let mut amp = vec![Complex64::new(0.0, 0.0); product];
        if product > 0 {
            amp[0] = Complex64::new(1.0, 0.0);
        }
        Self { dims, strides, amp }
    }

    /// Digit of wire `w` in flat index `i`.
    #[inline]
    pub fn digit(&self, i: usize, w: usize) -> usize {
        (i / self.strides[w]) % self.dims[w] as usize
    }

    /// Apply a `d×d` matrix on wire `w`.
    pub fn apply_1(&mut self, w: usize, g: &Dense) {
        let d = self.dims[w] as usize;
        debug_assert_eq!(g.dim, d);
        let stride = self.strides[w];
        let mut buf = vec![Complex64::new(0.0, 0.0); d];
        for base in 0..self.amp.len() {
            if self.digit(base, w) != 0 {
                continue;
            }
            for (k, b) in buf.iter_mut().enumerate() {
                *b = self.amp[base + k * stride];
            }
            for r in 0..d {
                let mut s = Complex64::new(0.0, 0.0);
                for (c, b) in buf.iter().enumerate() {
                    s += g.at(r, c) * b;
                }
                self.amp[base + r * stride] = s;
            }
        }
    }

    /// Apply a `(d_a·d_b)`-square matrix on wires `(a, b)`, `a` the more
    /// significant index of the matrix.
    pub fn apply_2(&mut self, a: usize, b: usize, g: &Dense) {
        let (da, db) = (self.dims[a] as usize, self.dims[b] as usize);
        debug_assert_eq!(g.dim, da * db);
        debug_assert_ne!(a, b);
        let (sa, sb) = (self.strides[a], self.strides[b]);
        let mut buf = vec![Complex64::new(0.0, 0.0); da * db];
        for base in 0..self.amp.len() {
            if self.digit(base, a) != 0 || self.digit(base, b) != 0 {
                continue;
            }
            for ia in 0..da {
                for ib in 0..db {
                    buf[ia * db + ib] = self.amp[base + ia * sa + ib * sb];
                }
            }
            for ra in 0..da {
                for rb in 0..db {
                    let r = ra * db + rb;
                    let mut s = Complex64::new(0.0, 0.0);
                    for (c, v) in buf.iter().enumerate() {
                        s += g.at(r, c) * v;
                    }
                    self.amp[base + ra * sa + rb * sb] = s;
                }
            }
        }
    }

    /// Apply an `8×8` matrix on three `d = 2` wires `(q0, q1, q2)`,
    /// `q0` most significant.
    fn apply_3q(&mut self, q0: usize, q1: usize, q2: usize, g: &Dense) {
        let (s0, s1, s2) = (self.strides[q0], self.strides[q1], self.strides[q2]);
        let mut buf = [Complex64::new(0.0, 0.0); 8];
        for base in 0..self.amp.len() {
            if self.digit(base, q0) != 0 || self.digit(base, q1) != 0 || self.digit(base, q2) != 0 {
                continue;
            }
            for (k, b) in buf.iter_mut().enumerate() {
                *b = self.amp[base + (k >> 2) * s0 + ((k >> 1) & 1) * s1 + (k & 1) * s2];
            }
            for r in 0..8 {
                let mut s = Complex64::new(0.0, 0.0);
                for (c, v) in buf.iter().enumerate() {
                    s += g.at(r, c) * v;
                }
                self.amp[base + (r >> 2) * s0 + ((r >> 1) & 1) * s1 + (r & 1) * s2] = s;
            }
        }
    }

    pub fn probabilities(&self) -> Vec<f64> {
        self.amp.iter().map(|a| a.norm_sqr()).collect()
    }
}

fn resolve_params(op: &GateOp, params: &ParameterBinding) -> Result<Vec<f64>> {
    op.params.iter().map(|p| params.resolve(p)).collect()
}

/// An integer-valued parameter (a level index), refused if it is not one.
fn level(v: f64, what: &str, d: usize) -> Result<usize> {
    if v.fract() != 0.0 || v < 0.0 || v >= d as f64 {
        return Err(refuse(format!(
            "rxy: {what} = {v} is not a level of a d = {d} wire (integers 0..{d} only)"
        )));
    }
    Ok(v as usize)
}

/// Evolve `circuit` from `|0…0⟩`. Refuses, naming the construct, anything
/// this exact engine cannot represent: a photonic circuit, a `Custom` gate,
/// a classically-conditioned gate, `Reset` (a channel, not a unitary), a
/// measurement followed by more gates, and any qubit gate placed on a
/// `d > 2` wire.
pub fn run(circuit: &CircuitIR, params: &ParameterBinding) -> Result<State> {
    if circuit.circuit_type == CircuitType::Photonic {
        return Err(refuse(
            "photonic circuits are not gate-based; use `photonics`".into(),
        ));
    }
    let dims = circuit.wire_dims();
    let product = capacity::check(&dims)?;
    let mut st = State::zero(dims, product);
    let mut measured = false;
    for op in &circuit.ops {
        if op.condition.is_some() {
            return Err(refuse(
                "a classically-conditioned gate needs mid-circuit measurement, which this \
                 exact engine does not do; measurements are read out at the end"
                    .into(),
            ));
        }
        let w: Vec<usize> = op.qubits.iter().map(|q| q.0 as usize).collect();
        for &wi in &w {
            if wi >= st.dims.len() {
                return Err(refuse(format!(
                    "gate {:?} names wire {wi}, but the circuit has {} wires",
                    op.gate,
                    st.dims.len()
                )));
            }
        }
        match op.gate {
            GateKind::Measure => {
                measured = true;
                continue;
            }
            GateKind::Barrier | GateKind::Id => continue,
            _ => {}
        }
        if measured {
            return Err(refuse(format!(
                "gate {:?} after a measurement: mid-circuit measurement is not supported; \
                 measurements are read out at the end",
                op.gate
            )));
        }
        let d = |i: usize| st.dims[w[i]] as usize;
        let p = resolve_params(op, params)?;
        match op.gate {
            // ---- generalised on any d; the qubit gate at d = 2 ----------
            GateKind::H => {
                let g = if d(0) == 2 {
                    gates::embed1(&qg::h())
                } else {
                    gates::fourier(d(0))
                };
                st.apply_1(w[0], &g);
            }
            GateKind::X => {
                let g = if d(0) == 2 {
                    gates::embed1(&qg::x())
                } else {
                    gates::shift(d(0))
                };
                st.apply_1(w[0], &g);
            }
            GateKind::Z => {
                let g = if d(0) == 2 {
                    gates::embed1(&qg::z())
                } else {
                    gates::clock(d(0))
                };
                st.apply_1(w[0], &g);
            }
            GateKind::Rxy => {
                let dd = d(0);
                let (i, j) = (level(p[0], "i", dd)?, level(p[1], "j", dd)?);
                if i >= j {
                    return Err(refuse(format!(
                        "rxy: levels must satisfy i < j, got ({i}, {j}); the sign of φ \
                         depends on the order, so it is not silently swapped"
                    )));
                }
                st.apply_1(w[0], &gates::rxy(dd, i, j, p[2], p[3]));
            }
            GateKind::CSum => {
                if w[0] == w[1] {
                    return Err(refuse("csum needs two distinct wires".into()));
                }
                st.apply_2(w[0], w[1], &gates::csum(d(0), d(1)));
            }
            // ---- not unitaries at any d: refused by what they are, before
            // the "qubit gate on a qudit wire" sentence could misdescribe them
            GateKind::Reset => {
                return Err(refuse(
                    "Reset is a non-unitary channel; this exact pure-state engine \
                     does not represent it"
                        .into(),
                ))
            }
            GateKind::Custom(_) => {
                return Err(refuse(
                    "Custom gates carry no matrix this engine can read".into(),
                ))
            }
            // ---- qubit-only gates: embedded on d = 2, refused otherwise ---
            ref g => {
                if let Some(&wi) = w.iter().find(|&&wi| st.dims[wi] != 2) {
                    return Err(refuse(format!(
                        "{g:?} is a qubit gate and wire {wi} has dimension {}; on a qudit \
                         wire the generalised gates are h (Fourier), x (shift), z (clock), \
                         rxy(i, j, θ, φ) and csum — the rest have no d > 2 meaning here",
                        st.dims[wi]
                    )));
                }
                match g {
                    GateKind::Y => st.apply_1(w[0], &gates::embed1(&qg::y())),
                    GateKind::S => st.apply_1(w[0], &gates::embed1(&qg::s())),
                    GateKind::Sdg => st.apply_1(w[0], &gates::embed1(&qg::sdg())),
                    GateKind::T => st.apply_1(w[0], &gates::embed1(&qg::t())),
                    GateKind::Tdg => st.apply_1(w[0], &gates::embed1(&qg::tdg())),
                    GateKind::Sx => st.apply_1(w[0], &gates::embed1(&qg::sx())),
                    GateKind::Sxdg => st.apply_1(w[0], &gates::embed1(&qg::sxdg())),
                    GateKind::Rx => st.apply_1(w[0], &gates::embed1(&qg::rx(p[0]))),
                    GateKind::Ry => st.apply_1(w[0], &gates::embed1(&qg::ry(p[0]))),
                    GateKind::Rz => st.apply_1(w[0], &gates::embed1(&qg::rz(p[0]))),
                    GateKind::U1 => st.apply_1(w[0], &gates::embed1(&qg::u1(p[0]))),
                    GateKind::U2 => st.apply_1(w[0], &gates::embed1(&qg::u2(p[0], p[1]))),
                    GateKind::U3 => st.apply_1(w[0], &gates::embed1(&qg::u3(p[0], p[1], p[2]))),
                    GateKind::CX => st.apply_2(w[0], w[1], &gates::embed2(&qg::cx())),
                    GateKind::CY => st.apply_2(w[0], w[1], &gates::embed2(&qg::cy())),
                    GateKind::CZ => st.apply_2(w[0], w[1], &gates::embed2(&qg::cz())),
                    GateKind::Swap => st.apply_2(w[0], w[1], &gates::embed2(&qg::swap())),
                    GateKind::CRz => st.apply_2(w[0], w[1], &gates::embed2(&qg::crz(p[0]))),
                    GateKind::CU3 => {
                        st.apply_2(w[0], w[1], &gates::embed2(&qg::cu3(p[0], p[1], p[2])))
                    }
                    GateKind::Rbs => st.apply_2(w[0], w[1], &gates::embed2(&qg::rbs(p[0]))),
                    GateKind::CCX => st.apply_3q(w[0], w[1], w[2], &gates::ccx()),
                    GateKind::CSwap => st.apply_3q(w[0], w[1], w[2], &gates::cswap()),
                    other => {
                        return Err(refuse(format!("gate {other:?} is not supported")));
                    }
                }
            }
        }
    }
    Ok(st)
}

/// `⟨ψ|P|ψ⟩` for a Pauli string on `d = 2` wires of a mixed-radix state.
/// Refuses a Pauli on a `d > 2` wire: `Z` has no meaning on a qutrit.
fn pauli_expectation(st: &State, paulis: &[(u32, PauliOp)]) -> Result<f64> {
    for (q, p) in paulis {
        if !matches!(p, PauliOp::I) && st.dims[*q as usize] != 2 {
            return Err(refuse(format!(
                "observable {p:?}{q} names wire {q}, which has dimension {}; Pauli \
                 observables are defined on d = 2 wires only",
                st.dims[*q as usize]
            )));
        }
    }
    // P|ψ⟩ then ⟨ψ|·⟩: exact and simple at this engine's sizes.
    let mut v = State {
        dims: st.dims.clone(),
        strides: st.strides.clone(),
        amp: st.amp.clone(),
    };
    for (q, p) in paulis {
        let g = match p {
            PauliOp::I => continue,
            PauliOp::X => qg::x(),
            PauliOp::Y => qg::y(),
            PauliOp::Z => qg::z(),
        };
        v.apply_1(*q as usize, &gates::embed1(&g));
    }
    let s: Complex64 = st.amp.iter().zip(&v.amp).map(|(a, b)| a.conj() * b).sum();
    Ok(s.re)
}

fn observable_expectation(st: &State, o: &Observable) -> Result<f64> {
    let mut acc = 0.0;
    for (c, p) in &o.terms {
        acc += c * pauli_expectation(st, p)?;
    }
    Ok(acc)
}

impl QuditSvBackend {
    /// `execute` plus the certificate — exact, bound zero, and the
    /// dimensions it was exact over.
    pub fn execute_with_certificate(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> Result<(ExecResult, QuditSvCertificate)> {
        let result = self.execute(circuit, params, config)?;
        let dims = circuit.wire_dims();
        let product_dim = capacity::check(&dims)?;
        Ok((
            result,
            QuditSvCertificate {
                exact: true,
                dropped_mass: 0.0,
                dims,
                product_dim,
            },
        ))
    }
}

impl Backend for QuditSvBackend {
    fn name(&self) -> &str {
        "quditsv"
    }

    /// `shots: None` → the full `Π dᵢ` statevector (wire 0 least
    /// significant). `shots: Some(n)` → counts, **on all-qubit circuits
    /// only**: `ExecResult::Counts` is keyed by bit-string `Outcome`s, and a
    /// qutrit outcome is not a bit string. Rather than encode digits into
    /// bits behind the reader's back, a qudit sampling run is refused and
    /// says so — `--statevector` gives the exact distribution.
    fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> Result<ExecResult> {
        if config.mid_circuit_mode == MidCircuitMode::Collapse {
            return Err(refuse(
                "mid-circuit collapse is not supported; measurements are read out at the end"
                    .into(),
            ));
        }
        let st = run(circuit, params)?;
        match config.shots {
            None => Ok(ExecResult::Statevector(st.amp)),
            Some(shots) => {
                if let Some((reg, wire, d)) = circuit.first_qudit() {
                    return Err(refuse(format!(
                        "sampling a qudit circuit is not supported yet: register '{}' has \
                         dimension {d} on wire {wire}, and `Counts` outcomes are bit strings. \
                         Use `--statevector` for the exact distribution.",
                        reg.name
                    )));
                }
                let n = circuit.num_qubits;
                let mut rng: StdRng = match config.seed {
                    Some(seed) => StdRng::seed_from_u64(seed),
                    None => rand::make_rng::<StdRng>(),
                };
                let probs = st.probabilities();
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
                    // All wires are d = 2, so the flat index IS the bit
                    // string (wire 0 = bit 0).
                    *counts.entry(pick as u64).or_insert(0) += 1;
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
        observable.validate_qubits(circuit.num_qubits)?;
        let (deferred, observable) = prepare_for_expectation(circuit, observable)?;
        let st = run(&deferred, params)?;
        observable_expectation(&st, &observable)
    }

    fn expectation_multi(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observables: &[Observable],
    ) -> Result<Vec<f64>> {
        for o in observables {
            o.validate_qubits(circuit.num_qubits)?;
        }
        let (deferred, observables) = prepare_for_expectation_multi(circuit, observables)?;
        let st = run(&deferred, params)?;
        observables
            .iter()
            .map(|o| observable_expectation(&st, o))
            .collect()
    }
}
