//! Transversal / low-overhead QEC logical-circuit layer.
//!
//! Recent low-overhead architectures (neutral-atom *Transversal STAR*,
//! trapped-ion small codes, high-rate qLDPC) encode a logical qubit into a small
//! number of physical qubits and run **transversal** logical gates — parallel
//! physical operations across a code's data qubits, costing ~1 syndrome round.
//! This layer expresses algorithms (QFT, QPE, Grover) as *logical* circuits and
//! compiles them to *physical* circuits under a chosen code, run on the omega
//! MPS / Pauli-propagation / statevector backends (execution layer).
//!
//! Built on top of [`crate::ecc`] (single-patch codes + decoding), adding the
//! multi-patch geometry ([`patch`]), the transversal gadget compiler
//! ([`transversal`], [`compile`]), hardware noise models ([`noise`]),
//! magic-state distillation ([`distill`]), the code-capacity memory
//! Monte-Carlo ([`memory`]), and the effective-logical-channel extractor
//! ([`channel`]).

pub mod channel;
pub mod compile;
pub mod distill;
pub mod memory;
pub mod metrics;
pub mod noise;
pub mod patch;
pub mod transversal;

pub use channel::{
    channel_rate_rounds, extract_surface_memory_channel, surface_memory_rate_rounds,
    EffectiveLogicalChannel,
};
pub use compile::{
    compile_physical, compile_physical_opts, CompileOptions, LogicalCircuit, LogicalOp,
    PhysicalProgram, ResourceReport,
};
pub use distill::{DistillProtocol, MagicStateProtocol};
pub use memory::surface_memory_rate;
pub use metrics::{
    grover_success_prob, qpe_phase_error, readout_bitflip, sub_threshold_slope, tvd,
    LogicalErrorCurve,
};
pub use noise::{splitmix64, NoiseModel, PauliChannel};
pub use patch::{PatchLayout, PauliBasis, StabilizerCode};
pub use transversal::{
    small_angle_injection_error, GadgetCost, NonCliffordMode, SteaneTransversal, TransversalCode,
};
