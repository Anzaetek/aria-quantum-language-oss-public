// SPDX-License-Identifier: Apache-2.0
//! Every kernel must compile in BOTH precisions.
//!
//! This is the load-bearing test for the f64 path: the kernels are written once
//! in terms of `real`/`real2` and compiled twice by NVRTC. If a kernel ever
//! reintroduces a precision-specific intrinsic (`rsqrtf`, `__fmul_rn`, a bare
//! `float`), the f64 compile fails here rather than at runtime on the machine
//! that needed f64.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use omega_backend_statevector_cuda::kernels::{all_kernel_sources, compile_probe, Precision};

#[test]
fn every_kernel_compiles_in_f32_and_f64() {
    let Some(ctx) = compile_probe() else {
        eprintln!("no CUDA device — skipping (this test is a device test)");
        return;
    };
    let mut failures = Vec::new();
    let mut compiled = 0usize;
    for (name, src) in all_kernel_sources() {
        for prec in [Precision::F32, Precision::F64] {
            match omega_backend_statevector_cuda::kernels::try_compile(&ctx, src, name, prec) {
                Ok(()) => compiled += 1,
                Err(e) => failures.push(format!("{name} @ {}: {e}", prec.as_str())),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} compiles failed:\n  {}",
        failures.len(),
        compiled + failures.len(),
        failures.join("\n  ")
    );
    // Two precisions per source; the count guards against the list silently
    // shrinking, which would turn this test into a vacuous pass. Pinned to the
    // full kernel population (every .cu under src/kernels/), so dropping OR
    // failing to add a kernel trips this.
    let on_disk = std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernels"))
        .expect("read kernels dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "cu"))
        .count();
    let sources = all_kernel_sources().len();
    assert_eq!(
        sources, on_disk,
        "{sources} kernels listed in all_kernel_sources() but {on_disk} .cu files on disk \
         — every kernel must be in the dual-precision compile gate"
    );
    assert_eq!(
        compiled,
        2 * sources,
        "expected both precisions for every source"
    );
    eprintln!("{compiled} kernel compiles OK (f32 + f64)");
}
