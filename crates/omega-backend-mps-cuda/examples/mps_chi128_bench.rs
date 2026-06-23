//! Quick wallclock comparison for the bench shape called out in
//! `NEXT_SESSION_PLAN.md` 1a: 14q × depth-12 × χ=128 brickwall + Rx
//! ring. Runs CPU and (when `--features cuda` is set) GPU back-to-back
//! and prints the speedup.
//!
//! Acceptance gate for the CUDA arm is ≥ 3× CPU on the Linux+NVIDIA
//! host (RTX PRO 6000 Blackwell).
//!
//! Usage:
//! ```text
//! cargo run --release -p omega-backend-mps-cuda --features cuda \
//!     --example mps_chi128_bench
//! ```
//!
//! Without the feature, only the CPU number is reported.

use omega_backend_mps::gates;
use omega_backend_mps::mps::Mps;
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
use omega_backend_mps_cuda::CudaSvdContext;

const N_QUBITS: usize = 14;
const DEPTH: usize = 12;
const CHI: usize = 128;
const RZ_THETA: f64 = 0.25;
const RX_THETA: f64 = 0.15;

/// Brickwall + Rx ring template, applied directly to an `Mps`. The
/// 2-site dispatch is whatever `apply_2q` is — caller decides whether
/// that's CPU or GPU SVD by either calling `mps.apply_2q(...)`
/// directly or routing through `CudaSvdContext::apply_2q`.
fn run_circuit_cpu(mps: &mut Mps) {
    let h = gates::h();
    let cx = gates::cx();
    let rz = gates::rz(RZ_THETA);
    let rx = gates::rx(RX_THETA);

    for q in 0..N_QUBITS {
        mps.apply_1q(q, &h);
    }
    for d in 0..DEPTH {
        let offset = d & 1;
        let mut q = offset;
        while q + 1 < N_QUBITS {
            mps.apply_2q(q, &cx);
            mps.apply_1q(q + 1, &rz);
            mps.apply_2q(q, &cx);
            q += 2;
        }
        for q in 0..N_QUBITS {
            mps.apply_1q(q, &rx);
        }
    }
}

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
fn run_circuit_cuda(mps: &mut Mps, ctx: &CudaSvdContext) {
    let h = gates::h();
    let cx = gates::cx();
    let rz = gates::rz(RZ_THETA);
    let rx = gates::rx(RX_THETA);

    for q in 0..N_QUBITS {
        mps.apply_1q(q, &h);
    }
    for d in 0..DEPTH {
        let offset = d & 1;
        let mut q = offset;
        while q + 1 < N_QUBITS {
            ctx.apply_2q(mps, q, &cx);
            mps.apply_1q(q + 1, &rz);
            ctx.apply_2q(mps, q, &cx);
            q += 2;
        }
        for q in 0..N_QUBITS {
            mps.apply_1q(q, &rx);
        }
    }
}

fn time<F: FnOnce()>(label: &str, f: F) -> std::time::Duration {
    let start = std::time::Instant::now();
    f();
    let elapsed = start.elapsed();
    println!("  {label:>16}: {elapsed:?}");
    elapsed
}

fn main() {
    println!("MPS bench: n={N_QUBITS}q, depth={DEPTH}, χ={CHI}, RZ={RZ_THETA}, RX={RX_THETA}");

    // Warm-up CPU
    {
        let mut mps = Mps::zero_state(N_QUBITS, CHI);
        run_circuit_cpu(&mut mps);
    }

    let cpu = time("cpu_apply_2q", || {
        let mut mps = Mps::zero_state(N_QUBITS, CHI);
        run_circuit_cpu(&mut mps);
        std::hint::black_box(&mps);
    });
    // `cpu` is read only under cfg(cuda); silence the warning on
    // builds without the feature.
    let _ = cpu;

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    {
        match CudaSvdContext::new() {
            Some(ctx) => {
                // Warm-up: builds the shape cache + cuSOLVER handle.
                {
                    let mut mps = Mps::zero_state(N_QUBITS, CHI);
                    run_circuit_cuda(&mut mps, &ctx);
                }
                let cuda = time("cuda_apply_2q", || {
                    let mut mps = Mps::zero_state(N_QUBITS, CHI);
                    run_circuit_cuda(&mut mps, &ctx);
                    std::hint::black_box(&mps);
                });
                let speedup = cpu.as_secs_f64() / cuda.as_secs_f64();
                println!("  speedup (cpu / cuda): {speedup:.3}×");
                println!(
                    "  acceptance gate (≥ 3×): {}",
                    if speedup >= 3.0 { "PASS" } else { "MISS" }
                );
            }
            None => {
                eprintln!("(cuda feature on, but no CUDA device available)");
            }
        }
    }

    #[cfg(not(all(any(target_os = "linux", target_os = "windows"), feature = "cuda")))]
    {
        println!("(no cuda feature; rebuild with --features cuda for the GPU number)");
    }
}
