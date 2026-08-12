// SPDX-License-Identifier: Apache-2.0
//! Double-precision forward statevector on the GPU.
//!
//! # Why this exists
//!
//! The main path is f32, which is measurably enough for a forward expectation
//! and is roughly 64x faster than f64 on consumer-class Blackwell. But f32 caps
//! agreement with the f64 CPU statevector at **~5e-7**, and this project gates
//! cross-checks at **1e-9 or tighter** (Qiskit differential, QEC vs Aer, the CV
//! fixture at 5.551e-17). An f32-only GPU therefore cannot enter its own
//! verification gates: every GPU number has to be qualified rather than checked.
//! This module closes that, at the cost of arithmetic throughput.
//!
//! # Scope, stated rather than implied
//!
//! **Forward and expectation only.** Gate application (1q, 2q, diagonal) and
//! Pauli expectation run in f64; the adjoint/backward path and the CUDA-graph
//! capture+replay training path stay f32. That split is deliberate — the
//! verification argument above is about forward agreement, and the graph path
//! carries its own machinery (pooled params, captured graphs) whose duplication
//! is not justified until an f64 training loop is actually wanted.
//!
//! It shares the kernels with the f32 path: same `.cu` sources, compiled a
//! second time with `-DOMEGA_REAL=double` (see `kernels/prelude.cuh`). So the
//! two precisions cannot drift apart in their math, which is the property worth
//! having.
//!
//! # The cost, measured rather than assumed
//!
//! f64 throughput is hardware-dependent to an extreme degree: GB10 (compute
//! 12.1) runs f64 at about 1/64 of f32, while an H100 (compute 9.0) runs it at
//! about 1/2. The same code is therefore a reasonable default on one machine and
//! a large regression on the other. `bytes_per_amplitude` also doubles, halving
//! the width a given device budget reaches. Any speedup claim must name the
//! machine AND the precision.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};

use crate::kernels::{try_compile_module, Precision};
use crate::CudaError;

/// f64 twin of `imp::Apply1qParams`. The struct layout must match the kernel's
/// `struct Apply1qParams` under `-DOMEGA_REAL=double`, so every `real` field is
/// an `f64` here.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Apply1qParamsF64 {
    pub qubit: u32,
    // The kernel struct starts `unsigned int` then `real`; with real = double
    // the compiler inserts 4 bytes of padding to align the doubles to 8. Rust's
    // repr(C) does the same, so the layouts agree — but it is worth naming,
    // because at f32 there is no padding and the two structs are NOT the same
    // shape with the field types swapped.
    _pad: u32,
    pub u00_re: f64,
    pub u00_im: f64,
    pub u01_re: f64,
    pub u01_im: f64,
    pub u10_re: f64,
    pub u10_im: f64,
    pub u11_re: f64,
    pub u11_im: f64,
}
unsafe impl DeviceRepr for Apply1qParamsF64 {}
unsafe impl ValidAsZeroBits for Apply1qParamsF64 {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Apply2qParamsF64 {
    pub qa: u32,
    pub qb: u32,
    pub u: [f64; 32],
}
unsafe impl DeviceRepr for Apply2qParamsF64 {}
unsafe impl ValidAsZeroBits for Apply2qParamsF64 {}

impl Default for Apply2qParamsF64 {
    fn default() -> Self {
        Self {
            qa: 0,
            qb: 1,
            u: [0.0; 32],
        }
    }
}

/// The f64 kernels, compiled once per device context.
pub struct KernelsF64 {
    pub apply_1q: CudaFunction,
    pub apply_2q: CudaFunction,
    /// Held so the modules outlive the functions taken from them.
    _modules: Vec<Arc<cudarc::driver::CudaModule>>,
}

impl KernelsF64 {
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self, CudaError> {
        let (m1, apply_1q) = try_compile_module(
            ctx,
            crate::kernels::source_apply_1q(),
            "apply_1q",
            Precision::F64,
        )?;
        let (m2, apply_2q) = try_compile_module(
            ctx,
            crate::kernels::source_apply_2q(),
            "apply_2q",
            Precision::F64,
        )?;
        Ok(Self {
            apply_1q,
            apply_2q,
            _modules: vec![m1, m2],
        })
    }
}

/// An f64 statevector living on the device.
pub struct StateF64 {
    pub num_qubits: u32,
    state: CudaSlice<f64>,
    stream: Arc<CudaStream>,
    kernels: Arc<KernelsF64>,
}

impl StateF64 {
    /// Allocate |0...0> for `num_qubits`.
    pub fn zero(
        ctx: &Arc<CudaContext>,
        kernels: Arc<KernelsF64>,
        num_qubits: u32,
    ) -> Result<Self, CudaError> {
        let dim = 1usize << num_qubits;
        let stream = ctx.default_stream();
        // Interleaved (re, im), so 2 f64 per amplitude.
        let mut host = vec![0.0f64; 2 * dim];
        host[0] = 1.0;
        let state = stream
            .clone_htod(&host)
            .map_err(|e| CudaError::Driver(format!("alloc f64 state: {e}")))?;
        Ok(Self {
            num_qubits,
            state,
            stream,
            kernels,
        })
    }

    /// Copy the state back as interleaved `(re, im)` f64 pairs.
    pub fn to_host(&self) -> Result<Vec<f64>, CudaError> {
        self.stream
            .clone_dtoh(&self.state)
            .map_err(|e| CudaError::Driver(format!("copy f64 state to host: {e}")))
    }

    fn launch_cfg(threads: u64) -> LaunchConfig {
        const BLOCK: u32 = 256;
        LaunchConfig {
            grid_dim: (threads.div_ceil(BLOCK as u64) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Apply a 2x2 unitary given row-major as `[(re, im); 4]`.
    pub fn apply_1q(&mut self, qubit: u32, u: [(f64, f64); 4]) -> Result<(), CudaError> {
        if qubit >= self.num_qubits {
            return Err(CudaError::Driver(format!(
                "qubit {qubit} out of range for {} wires",
                self.num_qubits
            )));
        }
        let params = Apply1qParamsF64 {
            qubit,
            _pad: 0,
            u00_re: u[0].0,
            u00_im: u[0].1,
            u01_re: u[1].0,
            u01_im: u[1].1,
            u10_re: u[2].0,
            u10_im: u[2].1,
            u11_re: u[3].0,
            u11_im: u[3].1,
        };
        let pairs = (1u64 << self.num_qubits) / 2;
        let cfg = Self::launch_cfg(pairs);
        let func = self.kernels.apply_1q.clone();
        let mut builder = self.stream.launch_builder(&func);
        builder.arg(&mut self.state).arg(&params).arg(&pairs);
        unsafe { builder.launch(cfg) }
            .map_err(|e| CudaError::Driver(format!("launch apply_1q (f64): {e}")))?;
        Ok(())
    }

    /// Apply a 4x4 unitary given row-major as 32 interleaved `(re, im)` values.
    pub fn apply_2q(&mut self, qa: u32, qb: u32, u: [f64; 32]) -> Result<(), CudaError> {
        if qa == qb || qa >= self.num_qubits || qb >= self.num_qubits {
            return Err(CudaError::Driver(format!(
                "bad qubit pair ({qa}, {qb}) for {} wires",
                self.num_qubits
            )));
        }
        let params = Apply2qParamsF64 { qa, qb, u };
        let quads = (1u64 << self.num_qubits) / 4;
        let cfg = Self::launch_cfg(quads);
        let func = self.kernels.apply_2q.clone();
        let mut builder = self.stream.launch_builder(&func);
        builder.arg(&mut self.state).arg(&params).arg(&quads);
        unsafe { builder.launch(cfg) }
            .map_err(|e| CudaError::Driver(format!("launch apply_2q (f64): {e}")))?;
        Ok(())
    }

    /// `⟨Z_q⟩`, computed on the host from the f64 amplitudes.
    ///
    /// Deliberately host-side: the reduction is O(2^n) adds against an O(2^n)
    /// device→host copy that has already been paid by `to_host`, and doing it
    /// here keeps the f64 surface to the two gate kernels. The point of this
    /// module is precision, not a second full backend.
    pub fn expectation_z(&self, qubit: u32) -> Result<f64, CudaError> {
        if qubit >= self.num_qubits {
            // Without this, an out-of-range mask exceeds `dim`, every `i & mask`
            // is 0, and the sum returns ~+1.0 — a plausible-looking value in
            // place of an error. Match apply_1q/apply_2q's bounds check.
            return Err(CudaError::Driver(format!(
                "qubit {qubit} out of range for {} wires",
                self.num_qubits
            )));
        }
        let host = self.to_host()?;
        let dim = 1usize << self.num_qubits;
        let mask = 1usize << qubit;
        // Kahan-compensated sum: a naive sequential sum drifts ~dim·ε, which
        // exceeds the 1e-13 bar this path is validated against once n is large.
        let mut acc = 0.0f64;
        let mut comp = 0.0f64;
        for i in 0..dim {
            let re = host[2 * i];
            let im = host[2 * i + 1];
            let p = re * re + im * im;
            let term = if i & mask == 0 { p } else { -p };
            let y = term - comp;
            let t = acc + y;
            comp = (t - acc) - y;
            acc = t;
        }
        Ok(acc)
    }
}
