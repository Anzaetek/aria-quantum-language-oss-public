//! OpenCL kernel library — embeds the `.cl` sources, builds an
//! [`ocl::Program`] at backend construction time, caches one
//! [`ocl::Kernel`] per kernel-name.
//!
//! Mirrors `omega-backend-statevector-{metal,cuda}/src/kernels.rs`.
//! The OpenCL build runs on the host CPU at construction time;
//! cached kernels are reused across every [`crate::imp::StateBuffer`]
//! through the [`crate::imp::DeviceHandle`].

use std::sync::Arc;

use ocl::{Context, Device, Program};

use crate::OpenClError;

/// OpenCL C source for the kernels we ship. Embedded at compile
/// time; no runtime filesystem access required. The `prelude` file
/// holds shared helpers (e.g. `cmul`) so each per-kernel source
/// stays self-contained and the program-wide build doesn't trip on
/// duplicate definitions.
const KERNEL_PRELUDE: &str = include_str!("kernels/prelude.cl");
const KERNEL_APPLY_1Q: &str = include_str!("kernels/apply_1q.cl");
const KERNEL_APPLY_2Q: &str = include_str!("kernels/apply_2q.cl");
const KERNEL_APPLY_DIAGONAL: &str = include_str!("kernels/apply_diagonal.cl");
const KERNEL_APPLY_DIAGONAL_2Q: &str = include_str!("kernels/apply_diagonal_2q.cl");
const KERNEL_APPLY_DIAGONAL_PRODUCT: &str = include_str!("kernels/apply_diagonal_product.cl");
const KERNEL_INNER_PRODUCT: &str = include_str!("kernels/inner_product.cl");
const KERNEL_PAULI_EXPECTATION: &str = include_str!("kernels/pauli_expectation.cl");
const KERNEL_SHOT_SAMPLE: &str = include_str!("kernels/shot_sample.cl");

/// Concatenated source so all kernels share one [`ocl::Program`] —
/// matches CUDA's NVRTC pattern and lets future kernels add a
/// `pub fn ...` import path through the same build.
fn full_source() -> String {
    [
        KERNEL_PRELUDE,
        KERNEL_APPLY_1Q,
        KERNEL_APPLY_2Q,
        KERNEL_APPLY_DIAGONAL,
        KERNEL_APPLY_DIAGONAL_2Q,
        KERNEL_APPLY_DIAGONAL_PRODUCT,
        KERNEL_INNER_PRODUCT,
        KERNEL_PAULI_EXPECTATION,
        KERNEL_SHOT_SAMPLE,
    ]
    .join("\n\n")
}

/// Bundled program + kernel handles for the OpenCL backend. One
/// [`KernelLibrary`] is built per [`crate::imp::DeviceHandle`] and
/// cloned into each `StateBuffer` via `Arc`.
pub(crate) struct KernelLibrary {
    pub program: Arc<Program>,
}

impl KernelLibrary {
    pub fn build(context: &Context, device: Device) -> Result<Self, OpenClError> {
        let src = full_source();
        let program = Program::builder()
            .src(src)
            .devices(device)
            .build(context)
            .map_err(|e| OpenClError::KernelCompile {
                // The shipped program concatenates every embedded
                // kernel; the compile failure could be in any of
                // them. We name the bundle rather than mislead the
                // operator with a single kernel name.
                kernel: "opencl-program",
                reason: format!("{e}"),
            })?;
        Ok(Self {
            program: Arc::new(program),
        })
    }
}
