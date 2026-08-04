// SPDX-License-Identifier: Apache-2.0
//! libtorch (`tch`) statevector backend for the Aria Quantum Language.
//!
//! Implements [`omega_core::executor::Backend`] on top of a complex
//! `tch::Tensor` statevector: each gate is applied by moving its target axes to
//! the end, reshaping to `[M, 2^k]`, and right-multiplying by `Uᵀ` (the proven
//! reshape+matmul scheme from the reference tensor simulator). Because it speaks
//! the same `Backend` trait as the pure-Rust backends, it slots straight into
//! `aria-runtime`'s `BackendSel` and `aria train` — the optional accelerator
//! path for batched / GPU (Metal) autograd-style workloads.

use num_complex::Complex64;
use omega_core::circuit::{CircuitIR, GateKind};
use omega_core::error::{OmegaError, Result};
use omega_core::executor::{Backend, ExecConfig, ExecResult, Observable, PauliOp};
use omega_core::params::ParameterBinding;
use std::collections::HashMap;
use tch::{Device, Kind, Tensor};

/// A libtorch statevector backend, dispatching to a `tch::Device`.
pub struct TchBackend {
    device: Device,
    rkind: Kind,
}

impl Default for TchBackend {
    fn default() -> Self {
        Self::cpu()
    }
}

impl TchBackend {
    /// CPU backend (complex128, ≤1e-15).
    pub fn cpu() -> Self {
        Self {
            device: Device::Cpu,
            rkind: Kind::Double,
        }
    }

    /// Apple Metal (MPS) backend if available, else CPU. MPS lacks float64, so
    /// it runs in complex64 (~1e-5).
    pub fn mps_or_cpu() -> Self {
        let mps_ok = std::panic::catch_unwind(|| {
            let _ = Tensor::zeros([1], (Kind::Float, Device::Mps));
        })
        .is_ok();
        if mps_ok {
            Self {
                device: Device::Mps,
                rkind: Kind::Float,
            }
        } else {
            Self::cpu()
        }
    }
}

// ---------------------------------------------------------------------------
// Statevector
// ---------------------------------------------------------------------------

struct State {
    amps: Tensor,
    n: usize,
    device: Device,
    rkind: Kind,
}

impl State {
    fn zero(n: usize, device: Device, rkind: Kind) -> Self {
        let dim = 1i64 << n;
        let re = Tensor::zeros([dim], (rkind, device));
        let im = Tensor::zeros([dim], (rkind, device));
        let _ = re.get(0).fill_(1.0);
        Self {
            amps: Tensor::complex(&re, &im),
            n,
            device,
            rkind,
        }
    }

    fn gate_tensor(&self, mat: &[Vec<Complex64>]) -> Tensor {
        let rows = mat.len() as i64;
        let cols = mat[0].len() as i64;
        let mut re = Vec::with_capacity((rows * cols) as usize);
        let mut im = Vec::with_capacity((rows * cols) as usize);
        for row in mat {
            for c in row {
                re.push(c.re);
                im.push(c.im);
            }
        }
        let shape = [rows, cols];
        let re = Tensor::from_slice(&re)
            .reshape(shape)
            .to_kind(self.rkind)
            .to_device(self.device);
        let im = Tensor::from_slice(&im)
            .reshape(shape)
            .to_kind(self.rkind)
            .to_device(self.device);
        Tensor::complex(&re, &im)
    }

    /// Apply a `2^k × 2^k` gate to `targets` (the first target is the high bit
    /// of the gate's block index). Qubit `q` maps to tensor axis `q`.
    fn apply(&mut self, mat: &[Vec<Complex64>], targets: &[usize]) {
        let k = targets.len();
        let u = self.gate_tensor(mat);
        let axes = vec![2i64; self.n];
        let view = self.amps.reshape(&axes);
        // Permutation: non-targets first, then targets in order.
        let mut perm: Vec<i64> = (0..self.n as i64)
            .filter(|a| !targets.contains(&(*a as usize)))
            .collect();
        for &t in targets {
            perm.push(t as i64);
        }
        let permuted = view.permute(&perm).contiguous();
        let block = 1i64 << k;
        let m = (1i64 << self.n) / block;
        let flat = permuted.reshape([m, block]);
        let out = flat.matmul(&u.transpose(0, 1));
        let restored = out.reshape(permuted.size());
        let inv = invert_perm(&perm);
        self.amps = restored
            .permute(&inv)
            .contiguous()
            .reshape([1i64 << self.n]);
    }

    fn prob_tensor(&self) -> Tensor {
        let re = self.amps.real();
        let im = self.amps.imag();
        &re * &re + &im * &im
    }

    fn probabilities(&self) -> Vec<f64> {
        host_vec_f64(&self.prob_tensor())
    }

    fn amps_vec(&self) -> Vec<Complex64> {
        let re = host_vec_f64(&self.amps.real());
        let im = host_vec_f64(&self.amps.imag());
        re.into_iter()
            .zip(im)
            .map(|(r, i)| Complex64::new(r, i))
            .collect()
    }

    fn duplicate(&self) -> Self {
        Self {
            amps: self.amps.copy(),
            n: self.n,
            device: self.device,
            rkind: self.rkind,
        }
    }

    /// `⟨self|other⟩ = Σ conj(self_i)·other_i`.
    fn inner_product(&self, other: &State) -> Complex64 {
        let prod = (self.amps.conj() * &other.amps).sum(self.amps.kind());
        Complex64::new(host_f64(&prod.real()), host_f64(&prod.imag()))
    }
}

fn invert_perm(perm: &[i64]) -> Vec<i64> {
    let mut inv = vec![0i64; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p as usize] = i as i64;
    }
    inv
}

fn host_f64(t: &Tensor) -> f64 {
    t.to_device(Device::Cpu)
        .to_kind(Kind::Double)
        .double_value(&[])
}

fn host_vec_f64(t: &Tensor) -> Vec<f64> {
    Vec::<f64>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Double)).expect("tensor to vec")
}

// ---------------------------------------------------------------------------
// Gate matrices
// ---------------------------------------------------------------------------

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}
const I1: Complex64 = Complex64::new(1.0, 0.0);
const I0: Complex64 = Complex64::new(0.0, 0.0);

/// Dense matrix for a gate (qubit-order: first target is the block high bit).
/// Returns `None` for non-unitary / unsupported (Measure/Barrier/Reset, CV).
fn gate_matrix(kind: &GateKind, p: &[f64]) -> Option<Vec<Vec<Complex64>>> {
    let g = |rows: &[&[Complex64]]| rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>();
    let h = std::f64::consts::FRAC_1_SQRT_2;
    Some(match kind {
        GateKind::Id => g(&[&[I1, I0], &[I0, I1]]),
        GateKind::X => g(&[&[I0, I1], &[I1, I0]]),
        GateKind::Y => g(&[&[I0, c(0.0, -1.0)], &[c(0.0, 1.0), I0]]),
        GateKind::Z => g(&[&[I1, I0], &[I0, c(-1.0, 0.0)]]),
        GateKind::H => g(&[&[c(h, 0.0), c(h, 0.0)], &[c(h, 0.0), c(-h, 0.0)]]),
        GateKind::S => g(&[&[I1, I0], &[I0, c(0.0, 1.0)]]),
        GateKind::Sdg => g(&[&[I1, I0], &[I0, c(0.0, -1.0)]]),
        GateKind::T => g(&[
            &[I1, I0],
            &[I0, Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4)],
        ]),
        GateKind::Tdg => g(&[
            &[I1, I0],
            &[I0, Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4)],
        ]),
        GateKind::Rx => {
            let (s, co) = (p[0] / 2.0).sin_cos();
            g(&[&[c(co, 0.0), c(0.0, -s)], &[c(0.0, -s), c(co, 0.0)]])
        }
        GateKind::Ry => {
            let (s, co) = (p[0] / 2.0).sin_cos();
            g(&[&[c(co, 0.0), c(-s, 0.0)], &[c(s, 0.0), c(co, 0.0)]])
        }
        GateKind::Rz => {
            let ph = p[0] / 2.0;
            g(&[
                &[Complex64::from_polar(1.0, -ph), I0],
                &[I0, Complex64::from_polar(1.0, ph)],
            ])
        }
        GateKind::U1 => g(&[&[I1, I0], &[I0, Complex64::from_polar(1.0, p[0])]]),
        GateKind::U2 => u3_matrix(std::f64::consts::FRAC_PI_2, p[0], p[1]),
        GateKind::U3 => u3_matrix(p[0], p[1], p[2]),
        GateKind::CX => controlled(&gate_matrix(&GateKind::X, &[])?),
        GateKind::CY => controlled(&gate_matrix(&GateKind::Y, &[])?),
        GateKind::CZ => controlled(&gate_matrix(&GateKind::Z, &[])?),
        GateKind::CRz => controlled(&gate_matrix(&GateKind::Rz, p)?),
        GateKind::CU3 => controlled(&u3_matrix(p[0], p[1], p[2])),
        GateKind::Swap => g(&[
            &[I1, I0, I0, I0],
            &[I0, I0, I1, I0],
            &[I0, I1, I0, I0],
            &[I0, I0, I0, I1],
        ]),
        GateKind::CCX => controlled(&controlled(&gate_matrix(&GateKind::X, &[])?)),
        GateKind::CSwap => controlled(&gate_matrix(&GateKind::Swap, &[])?),
        _ => return None,
    })
}

fn u3_matrix(theta: f64, phi: f64, lam: f64) -> Vec<Vec<Complex64>> {
    let (s, co) = (theta / 2.0).sin_cos();
    vec![
        vec![c(co, 0.0), -Complex64::from_polar(s, lam)],
        vec![
            Complex64::from_polar(s, phi),
            Complex64::from_polar(co, phi + lam),
        ],
    ]
}

/// Promote an `m×m` gate `U` to a controlled `2m×2m` block-diag(I, U) — the
/// control is the new high bit.
fn controlled(u: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let m = u.len();
    let mut out = vec![vec![I0; 2 * m]; 2 * m];
    for (i, row) in out.iter_mut().enumerate().take(m) {
        row[i] = I1;
    }
    for (r, urow) in u.iter().enumerate() {
        for (col, &val) in urow.iter().enumerate() {
            out[m + r][m + col] = val;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Backend impl
// ---------------------------------------------------------------------------

impl TchBackend {
    /// Evolve the unitary part of `circuit` (measurements/barriers/resets are
    /// skipped) and return the final statevector.
    fn evolve(&self, circuit: &CircuitIR, params: &ParameterBinding) -> Result<State> {
        let mut st = State::zero(circuit.num_qubits as usize, self.device, self.rkind);
        for op in &circuit.ops {
            match op.gate {
                GateKind::Measure | GateKind::Barrier => continue,
                // Reset is NOT skippable: it is a non-unitary channel
                // (rho -> |0><0|_q (x) Tr_q(rho)). This backend evolves a
                // single pure state through unitaries, so skipping it silently
                // answered a different circuit. Refuse and let the caller fall
                // back to the statevector backend, which samples it per shot.
                GateKind::Reset => {
                    return Err(OmegaError::Unsupported(
                        "tch: Reset is a non-unitary channel and is not supported; use the \
                         statevector or MPS backend"
                            .into(),
                    ))
                }
                _ => {}
            }
            let resolved: Vec<f64> = op
                .params
                .iter()
                .map(|e| params.resolve(e))
                .collect::<Result<_>>()?;
            if let Some(mat) = gate_matrix(&op.gate, &resolved) {
                let targets: Vec<usize> = op.qubits.iter().map(|q| q.0 as usize).collect();
                st.apply(&mat, &targets);
            } else {
                return Err(OmegaError::Unsupported(format!(
                    "tch backend has no matrix for gate {:?}",
                    op.gate
                )));
            }
        }
        Ok(st)
    }
}

impl Backend for TchBackend {
    fn name(&self) -> &str {
        "tch"
    }

    fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> Result<ExecResult> {
        let st = self.evolve(circuit, params)?;
        match config.shots {
            None => Ok(ExecResult::Statevector(st.amps_vec())),
            Some(shots) => {
                let probs = st.probabilities();
                let mut rng = SplitMix64(config.seed.unwrap_or(0x1234_5678));
                let mut counts: HashMap<u64, u32> = HashMap::new();
                for _ in 0..shots {
                    let mut u = rng.next_f64();
                    let mut idx = probs.len() - 1;
                    for (i, &p) in probs.iter().enumerate() {
                        if u < p {
                            idx = i;
                            break;
                        }
                        u -= p;
                    }
                    *counts.entry(idx as u64).or_insert(0) += 1;
                }
                Ok(ExecResult::Counts(counts))
            }
        }
    }

    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<f64> {
        let st = self.evolve(circuit, params)?;
        let mut total = 0.0;
        for (coeff, paulis) in &observable.terms {
            let mut p_psi = st.duplicate();
            for (q, op) in paulis {
                let mat = match op {
                    PauliOp::I => continue,
                    PauliOp::X => gate_matrix(&GateKind::X, &[]),
                    PauliOp::Y => gate_matrix(&GateKind::Y, &[]),
                    PauliOp::Z => gate_matrix(&GateKind::Z, &[]),
                };
                p_psi.apply(&mat.unwrap(), &[*q as usize]);
            }
            // ⟨ψ|P|ψ⟩ — Hermitian, so the imaginary part is ~0.
            total += coeff * st.inner_product(&p_psi).re;
        }
        Ok(total)
    }
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bell() -> CircuitIR {
        use omega_core::circuit::{CircuitType, GateOp, ParamExpr, Qubit};
        let mut ir = CircuitIR::new(2, CircuitType::GateBased);
        ir.ops.push(GateOp {
            gate: GateKind::H,
            qubits: vec![Qubit(0)].into(),
            params: Vec::<ParamExpr>::new().into(),
            classical_bit: None,
            condition: None,
        });
        ir.ops.push(GateOp {
            gate: GateKind::CX,
            qubits: vec![Qubit(0), Qubit(1)].into(),
            params: Vec::<ParamExpr>::new().into(),
            classical_bit: None,
            condition: None,
        });
        ir
    }

    #[test]
    fn bell_statevector_is_exact() {
        let b = TchBackend::cpu();
        let cfg = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: omega_core::executor::MidCircuitMode::Skip,
        };
        let sv = match b.execute(&bell(), &ParameterBinding::new(), &cfg).unwrap() {
            ExecResult::Statevector(v) => v,
            _ => panic!(),
        };
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sv[0] - Complex64::new(s, 0.0)).norm() < 1e-12);
        assert!((sv[3] - Complex64::new(s, 0.0)).norm() < 1e-12);
        assert!(sv[1].norm() < 1e-12 && sv[2].norm() < 1e-12);
    }

    #[test]
    fn bell_zz_expectation_is_one() {
        let b = TchBackend::cpu();
        let obs = Observable::parse("Z0 Z1").unwrap();
        let zz = b
            .expectation(&bell(), &ParameterBinding::new(), &obs)
            .unwrap();
        assert!((zz - 1.0).abs() < 1e-12, "zz={zz}");
    }
}
