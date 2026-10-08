// SPDX-License-Identifier: Apache-2.0
//! A circuit executor over [`crate::f64_path`]: the double-precision CUDA arm
//! that `omega-run --device cuda --precision f64` reaches.
//!
//! # Why it is a separate backend
//!
//! The f32 [`crate::CudaStatevectorBackend`] caps agreement with the f64 CPU
//! statevector at ~5e-7, so a GPU row against a double-precision yardstick
//! (cuStateVec at `precision="double"`) compares complex64 with complex128 and
//! can only be published labelled as such. This arm makes the matched
//! comparison possible; it does not change or repair the f32 arm, whose rows
//! remain f32-against-f32 by design.
//!
//! # Scope, refused rather than approximated
//!
//! [`crate::f64_path`] has two kernels, a 2×2 and a 4×4 unitary, so this
//! executor accepts unitary circuits of one- and two-qubit gates plus
//! measurements it can treat as terminal. Every gate matrix comes from the CPU
//! backend's `gates` module, so the two cannot disagree on a convention: a
//! phase or ordering error would have to be in the CPU reference too. Each of
//! these is refused by name, never routed to the f32 path or the CPU:
//!
//! * gates on three or more qubits (`ccx`, `cswap`) — no 8×8 f64 kernel;
//! * `reset` — a channel, not a unitary;
//! * classically conditioned gates — this arm keeps no classical register;
//! * a measurement in collapse mode — one statevector cannot carry a branch;
//! * photonic, qudit and custom gates — as on the f32 arm.
//!
//! One launch per gate: no diagonal fusion, unlike the f32 walker. A timing
//! row from this arm is therefore an unfused-f64 row, and says so in its arm
//! label (`cuda-f64`).

use std::sync::Arc;

use cudarc::driver::CudaContext;
use num_complex::Complex64;
use omega_backend_statevector::gates;
use omega_backend_statevector::sim::expectation_pauli;
use omega_core::circuit::{CircuitIR, GateKind, GateOp};
use omega_core::defer_measure::prepare_for_expectation;
use omega_core::error::OmegaError;
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable};
use omega_core::params::ParameterBinding;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::f64_path::{KernelsF64, StateF64};
use crate::CudaError;

type OmegaResult<T> = Result<T, OmegaError>;

/// The double-precision CUDA statevector. Context creation and the NVRTC
/// compile of both f64 kernels happen in [`Self::new`], so a timed `execute`
/// measures evolution and the synchronising device-to-host copy only.
pub struct CudaStatevectorF64Backend {
    ctx: Arc<CudaContext>,
    kernels: Arc<KernelsF64>,
}

impl CudaStatevectorF64Backend {
    pub fn new() -> Result<Self, CudaError> {
        let ctx = CudaContext::new(0)
            .map_err(|e| CudaError::Driver(format!("CudaContext::new(0) failed: {e}")))?;
        let kernels = Arc::new(KernelsF64::load(&ctx)?);
        Ok(Self { ctx, kernels })
    }
}

/// A one- or two-qubit gate as the matrix the f64 kernels take, or the reason
/// it cannot be.
enum Lowered {
    Skip,
    One(u32, gates::Gate1Q),
    /// `(first, second, U)` with `U` in the CPU's basis order: the FIRST qubit
    /// is the high bit of the 4×4 row index.
    Two(u32, u32, gates::Gate2Q),
}

fn lower(op: &GateOp, params: &ParameterBinding) -> OmegaResult<Lowered> {
    let refuse = |why: &str| {
        Err(OmegaError::Unsupported(format!(
            "cuda-f64: {:?} {why}",
            op.gate
        )))
    };
    if op.condition.is_some() {
        return refuse("is classically conditioned; this arm keeps no classical register");
    }
    let p: Vec<f64> = op
        .params
        .iter()
        .map(|e| params.resolve(e))
        .collect::<OmegaResult<_>>()?;
    let q = |i: usize| op.qubits[i].0;
    Ok(match &op.gate {
        GateKind::Id | GateKind::Barrier | GateKind::Measure => Lowered::Skip,
        GateKind::H => Lowered::One(q(0), gates::h()),
        GateKind::X => Lowered::One(q(0), gates::x()),
        GateKind::Y => Lowered::One(q(0), gates::y()),
        GateKind::Z => Lowered::One(q(0), gates::z()),
        GateKind::S => Lowered::One(q(0), gates::s()),
        GateKind::Sdg => Lowered::One(q(0), gates::sdg()),
        GateKind::Sx => Lowered::One(q(0), gates::sx()),
        GateKind::Sxdg => Lowered::One(q(0), gates::sxdg()),
        GateKind::T => Lowered::One(q(0), gates::t()),
        GateKind::Tdg => Lowered::One(q(0), gates::tdg()),
        GateKind::Rx => Lowered::One(q(0), gates::rx(p[0])),
        GateKind::Ry => Lowered::One(q(0), gates::ry(p[0])),
        GateKind::Rz => Lowered::One(q(0), gates::rz(p[0])),
        GateKind::U1 => Lowered::One(q(0), gates::u1(p[0])),
        GateKind::U2 => Lowered::One(q(0), gates::u2(p[0], p[1])),
        GateKind::U3 => Lowered::One(q(0), gates::u3(p[0], p[1], p[2])),
        GateKind::CX => Lowered::Two(q(0), q(1), gates::cx()),
        GateKind::CY => Lowered::Two(q(0), q(1), gates::cy()),
        GateKind::CZ => Lowered::Two(q(0), q(1), gates::cz()),
        GateKind::Swap => Lowered::Two(q(0), q(1), gates::swap()),
        GateKind::CRz => Lowered::Two(q(0), q(1), gates::crz(p[0])),
        GateKind::CU3 => Lowered::Two(q(0), q(1), gates::cu3(p[0], p[1], p[2])),
        GateKind::Rbs => Lowered::Two(q(0), q(1), gates::rbs(p[0])),
        GateKind::CCX | GateKind::CSwap => {
            return refuse("acts on three qubits; the f64 path has no 8x8 kernel");
        }
        GateKind::Reset => return refuse("is a channel, not a unitary"),
        GateKind::PhaseShifter | GateKind::BeamSplitterRx | GateKind::Custom(_) => {
            return refuse("is not supported on this backend");
        }
        GateKind::Rxy | GateKind::CSum => return refuse("is a qudit gate with no CUDA kernel"),
    })
}

/// The CPU's 4×4 (first qubit = high bit) as the kernel's 32 interleaved
/// values. The kernel indexes its row as `bit(qb)·2 + bit(qa)`, so it is
/// launched with `qa = second`, `qb = first` and the matrix is passed as is.
fn kernel_2q(u: &gates::Gate2Q) -> [f64; 32] {
    let mut flat = [0.0f64; 32];
    for (k, c) in u.iter().enumerate() {
        flat[2 * k] = c.re;
        flat[2 * k + 1] = c.im;
    }
    flat
}

/// Sample `shots` outcomes from `amps` on the host, keyed on the qubit
/// register as the f32 arm's device sampler keys them.
fn sample(
    amps: &[Complex64],
    shots: u32,
    seed: Option<u64>,
) -> std::collections::HashMap<u64, u32> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => rand::make_rng::<StdRng>(),
    };
    let mut cdf = Vec::with_capacity(amps.len());
    let mut acc = 0.0f64;
    for a in amps {
        acc += a.norm_sqr();
        cdf.push(acc);
    }
    // A unitary evolution keeps the norm at 1; a zero total would put every
    // shot on the last index.
    debug_assert!(acc > 0.0, "sampling a zero-norm state");
    let mut counts = std::collections::HashMap::new();
    for _ in 0..shots {
        let r: f64 = rng.random::<f64>() * acc;
        let i = cdf.partition_point(|&c| c <= r).min(amps.len() - 1);
        *counts.entry(i as u64).or_insert(0) += 1;
    }
    counts
}

impl CudaStatevectorF64Backend {
    /// Evolve |0…0⟩ through `circuit` on the device and copy the state back.
    fn evolve(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
    ) -> OmegaResult<Vec<Complex64>> {
        // Lower the whole circuit before touching the device, so a refusal
        // never leaves a half-evolved state or a partial timing.
        let lowered = circuit
            .ops
            .iter()
            .map(|op| lower(op, params))
            .collect::<OmegaResult<Vec<_>>>()?;

        let n = circuit.num_qubits;
        let mut state = StateF64::zero(&self.ctx, self.kernels.clone(), n)?;
        for l in &lowered {
            match l {
                Lowered::Skip => {}
                Lowered::One(q, u) => {
                    state.apply_1q(
                        *q,
                        [
                            (u[0].re, u[0].im),
                            (u[1].re, u[1].im),
                            (u[2].re, u[2].im),
                            (u[3].re, u[3].im),
                        ],
                    )?;
                }
                Lowered::Two(first, second, u) => {
                    state.apply_2q(*second, *first, kernel_2q(u))?;
                }
            }
        }
        // `to_host` synchronises the stream after the copy, so a timer around
        // `execute` includes every kernel.
        let host = state.to_host()?;
        let (pairs, rest) = host.as_chunks::<2>();
        debug_assert!(rest.is_empty(), "interleaved (re, im) pairs");
        let amps: Vec<Complex64> = pairs
            .iter()
            .map(|&[re, im]| Complex64::new(re, im))
            .collect();
        Ok(amps)
    }
}

impl Backend for CudaStatevectorF64Backend {
    fn name(&self) -> &str {
        "cuda-statevector-f64"
    }

    fn device(&self) -> omega_core::device::DeviceKind {
        omega_core::device::DeviceKind::Cuda
    }

    fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> OmegaResult<ExecResult> {
        circuit.refuse_qudits(self.name())?;
        if config.mid_circuit_mode == MidCircuitMode::Collapse
            && circuit
                .ops
                .iter()
                .any(|op| matches!(op.gate, GateKind::Measure))
        {
            return Err(OmegaError::Unsupported(
                "cuda-f64: a measurement in collapse mode branches the state; one \
                 statevector cannot carry it"
                    .into(),
            ));
        }
        let amps = self.evolve(circuit, params)?;
        match config.shots {
            None => Ok(ExecResult::Statevector(amps)),
            Some(shots) => Ok(ExecResult::counts_from_u64(
                sample(&amps, shots, config.seed),
                circuit.num_qubits,
            )),
        }
    }

    /// `⟨ψ|O|ψ⟩` for a Pauli sum. Measurements are deferred first, as on the
    /// f32 arm, so a measurement with consequences is not silently skipped.
    /// The reduction is the CPU backend's own `expectation_pauli` on the f64
    /// amplitudes: host-side and single-threaded, so it is exact to the CPU's
    /// convention (including the Y sign) and costs `O(terms · 2^n)` on the host.
    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> OmegaResult<f64> {
        circuit.refuse_qudits(self.name())?;
        let (deferred, observable) = prepare_for_expectation(circuit, observable)?;
        let amps = self.evolve(&deferred, params)?;
        Ok(observable
            .terms
            .iter()
            .map(|(coeff, pauli)| coeff * expectation_pauli(&amps, deferred.num_qubits, pauli))
            .sum())
    }
}
