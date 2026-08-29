// SPDX-License-Identifier: Apache-2.0
//! Register / local-memory footprint of the quad kernels, straight from the
//! driver.
//!
//! WHY THIS EXISTS. When `apply_quad_phase1` was split out of
//! `apply_quad_phase`, CZ recovered 18.6%. The explanation written into the
//! `.cu` was "nvcc allocates registers for the kernel's worst path, and the
//! multi-slot loop's pressure costs this memory-bound case its occupancy".
//! That was a **hypothesis stated as a fact**, in a repository whose recurring
//! defect is exactly that — and a competing explanation fits the same data:
//! `apply_quad_phase` indexes `params.slots[k]` / `params.phase_*[k]` with a
//! runtime `k`, and nvcc routinely spills a by-value parameter struct to the
//! local frame when an array member is dynamically indexed. That is a stack
//! frame, not register pressure, and it would equally explain why the
//! `if (count == 1)` fast path did not help.
//!
//! `-Xptxas -v` is not reachable through NVRTC (it emits PTX; ptxas runs at
//! module load), which is presumably why nobody looked. The driver will simply
//! answer, so ask it.
//!
//! This test asserts almost nothing on purpose — register counts are a property
//! of the toolkit and the arch, and pinning them would make an nvcc upgrade
//! look like a regression. It reports, and it fails only on the one thing that
//! would be a real defect: a kernel spilling to local memory when it has no
//! business doing so.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use omega_backend_statevector_cuda::CudaStatevectorBackend;

#[test]
fn quad_kernel_register_and_local_memory_footprint() {
    let backend = match CudaStatevectorBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: no CUDA device ({e})");
            return;
        }
    };

    let report = backend.kernel_resource_report().expect("kernel attributes");
    eprintln!(
        "{:<28} {:>6} {:>12} {:>12}",
        "kernel", "regs", "local B", "shared B"
    );
    for (name, regs, local, shared) in &report {
        eprintln!("{name:<28} {regs:>6} {local:>12} {shared:>12}");
    }

    // The claim under test: does the multi-slot kernel actually cost more than
    // the single-slot one, and by which mechanism?
    let find = |want: &str| {
        report
            .iter()
            .find(|(n, ..)| n == want)
            .unwrap_or_else(|| panic!("{want} missing from report"))
    };
    let (_, p1_regs, p1_local, _) = find("apply_quad_phase1");
    let (_, pm_regs, pm_local, _) = find("apply_quad_phase");
    eprintln!(
        "\nsingle-slot vs multi-slot: regs {p1_regs} -> {pm_regs}, local {p1_local} -> {pm_local} B"
    );

    // A statevector gate kernel has no reason to touch local memory. If one
    // does, that IS the finding — it means a parameter array is being spilled
    // to the local frame and indexed from there on every iteration.
    for (name, _, local, _) in &report {
        assert_eq!(
            *local, 0,
            "{name} spills {local} B to local memory — a gate kernel should not"
        );
    }
}
