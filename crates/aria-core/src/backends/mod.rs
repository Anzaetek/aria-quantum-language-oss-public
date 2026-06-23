// SPDX-License-Identifier: Apache-2.0
//! Backend-facing lowerings of the Aria `Circuit` IR.
//!
//! [`omega`] converts a `Circuit` into the omega JSON wire IR
//! (`OmegaCircuitIR`) and Pauli observables (`OmegaObservable`) consumed by the
//! omega execution backends and the omega-server HTTP bridge.
pub mod omega;
