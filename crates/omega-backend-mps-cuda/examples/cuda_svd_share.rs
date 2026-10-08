//! What share of MPS evolution is the SVD once the cuSOLVER SVD replaces the
//! CPU Jacobi kernel — the measurement `STATUS.md` §5.16 names as the
//! PREREQUISITE for any Metal-specific SVD work.
//!
//! The reasoning there: the 90-97% CPU share is a property of our Jacobi
//! kernel, not of Apple, so it should hold anywhere. If swapping in a GPU SVD
//! does not move it, the kernel is not the bottleneck on any platform and the
//! whole item is mis-aimed. This harness answers that on real hardware instead
//! of assuming either way.
//!
//! The device kernel is the exact `cusolverDnZgesvd` since 2026-09-30. The
//! first table in STATUS was taken through the approximate `Zgesvda` that the
//! module then called (labelled "gesvdj" there, after the module's name).
//!
//! TWO THINGS IT GUARDS AGAINST, both learned the hard way in this file's
//! sibling `mps_stage_profile`:
//!
//! 1. **Silent CPU fallback.** `cuda_svd_flat` falls back to
//!    `omega_backend_mps::svd::truncated_svd_flat` whenever the context or an
//!    individual call fails — per call, not once. A share measured through it
//!    could be timing the CPU while reporting "CUDA". The GPU rows therefore
//!    print the LIBRARY's dispatch counters (`cuda_svd_dispatch`: GPU calls
//!    and CPU fallbacks, with the last fallback's reason); a row with cpu>0 is
//!    not a clean GPU measurement and says so.
//! 2. **Measuring reconstruction instead of evolution.** Driven through
//!    `expectation`, never `execute(shots: None)`, which ends in a dense 2^n
//!    `to_statevector` — at 20 qubits that is ~10 s of reconstruction wrapped
//!    around ~250 ms of actual evolution, and is what made the 2026-05-12
//!    deferral's "flat across chi" evidence measure the wrong quantity.
//!
//! The certificate is checked, not assumed: `discarded_weight` feeds a
//! truncation bound, and §5.16 requires that whatever replaces the kernel
//! return the same one. Both paths are fingerprinted with an order-independent
//! XOR over every returned weight (XOR because rayon makes call order
//! nondeterministic and a sum would differ from ordering alone). The GPU and
//! CPU fingerprints are NOT expected to match bit-for-bit — they are different
//! exact algorithms, agreeing to ~1e-12 relative on the certificate
//! (`tests/cuda_svd_parity.rs`), not to the bit — so the expectation values are
//! compared numerically and both fingerprints printed for the record.

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
fn main() {
    use num_complex::Complex64;
    use omega_backend_mps::svd::{truncated_svd_flat, SvdResultFlat};
    use omega_backend_mps::MpsBackend;
    use omega_backend_mps_cuda::{
        cuda_svd_dispatch, cuda_svd_flat, reset_cuda_svd_dispatch, CudaSvdContext,
    };
    use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
    use omega_core::executor::{Backend, Observable, PauliOp};
    use omega_core::params::ParameterBinding;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::time::Instant;

    static SVD_NANOS: AtomicU64 = AtomicU64::new(0);
    // Total SVD calls, whichever kernel ran them. Where they ran is the
    // LIBRARY's count (`cuda_svd_dispatch`), not this harness's: this file
    // used to shadow `cuda_svd_flat` with its own GPU/CPU counters because the
    // library had none, and a shadow can drift from what production does.
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    static MAX_COLS: AtomicUsize = AtomicUsize::new(0);
    static DW_XOR: AtomicU64 = AtomicU64::new(0);

    fn record(out: &SvdResultFlat, cols: usize, t: Instant) {
        SVD_NANOS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        MAX_COLS.fetch_max(cols, Ordering::Relaxed);
        DW_XOR.fetch_xor(out.discarded_weight.to_bits(), Ordering::Relaxed);
    }

    /// The production hook, timed. GPU first, CPU fallback per call; the
    /// library counts which.
    fn gpu_svd(
        a: &[Complex64],
        rows: usize,
        cols: usize,
        lda: usize,
        max_rank: usize,
        threshold: f64,
    ) -> SvdResultFlat {
        let t = Instant::now();
        let out = cuda_svd_flat(a, rows, cols, lda, max_rank, threshold);
        record(&out, cols, t);
        out
    }

    fn cpu_svd(
        a: &[Complex64],
        rows: usize,
        cols: usize,
        lda: usize,
        max_rank: usize,
        threshold: f64,
    ) -> SvdResultFlat {
        let t = Instant::now();
        let out = truncated_svd_flat(a, rows, cols, lda, max_rank, threshold);
        record(&out, cols, t);
        out
    }

    fn reset() {
        SVD_NANOS.store(0, Ordering::Relaxed);
        CALLS.store(0, Ordering::Relaxed);
        MAX_COLS.store(0, Ordering::Relaxed);
        DW_XOR.store(0, Ordering::Relaxed);
        reset_cuda_svd_dispatch();
    }

    /// Identical to `mps_stage_profile`'s circuit, so the tables compare.
    fn entangling_circuit(num_qubits: u32, depth: usize) -> CircuitIR {
        let mut circuit = CircuitIR::new(num_qubits, CircuitType::GateBased);
        let push = |c: &mut CircuitIR, gate: GateKind, qs: Vec<u32>, ps: Vec<f64>| {
            c.ops.push(GateOp {
                gate,
                qubits: qs.into_iter().map(Qubit).collect(),
                params: ps.into_iter().map(ParamExpr::Concrete).collect(),
                classical_bit: None,
                condition: None,
            });
        };
        for q in 0..num_qubits {
            push(&mut circuit, GateKind::H, vec![q], vec![]);
        }
        for d in 0..depth {
            let offset = d as u32 & 1;
            for q in (offset..num_qubits - 1).step_by(2) {
                push(&mut circuit, GateKind::CX, vec![q, q + 1], vec![]);
                push(&mut circuit, GateKind::Rz, vec![q + 1], vec![0.25]);
                push(&mut circuit, GateKind::CX, vec![q, q + 1], vec![]);
            }
            for q in 0..num_qubits {
                push(&mut circuit, GateKind::Rx, vec![q], vec![0.15]);
            }
        }
        circuit
    }

    // Establish availability ONCE and loudly. A "CUDA" table produced with no
    // GPU is worse than no table.
    match CudaSvdContext::try_new() {
        Ok(_) => println!("cuSOLVER gesvd context: AVAILABLE"),
        Err(e) => {
            println!("cuSOLVER gesvd context: UNAVAILABLE — {e:?}");
            println!("Every row below would be a CPU measurement mislabelled as CUDA. Stopping.");
            return;
        }
    }
    println!("rayon threads={}", rayon::current_num_threads());

    let params = ParameterBinding::new();
    println!(
        "\n{:>4} {:>6} {:>5} {:>4} {:>10} {:>10} {:>7} {:>6} {:>6} {:>6} {:>18}",
        "q", "depth", "chi", "svd", "evolve ms", "svd ms", "share", "gpu", "cpu", "maxc", "dw_xor"
    );

    for &(nq, depth) in &[(14u32, 4usize), (14, 12), (14, 24), (20, 12), (20, 16)] {
        let circuit = entangling_circuit(nq, depth);
        for &chi in &[32usize, 128, 256] {
            let obs = Observable {
                terms: vec![(1.0, vec![(0u32, PauliOp::Z)])],
            };
            for (label, f) in [
                ("cpu", cpu_svd as omega_backend_mps::SvdFlatFn),
                ("gpu", gpu_svd as omega_backend_mps::SvdFlatFn),
            ] {
                // Warm: first touch pays allocator, rayon pool and CUDA
                // context-creation costs, none of which belong in the number.
                MpsBackend::new(chi)
                    .with_svd_fn(f)
                    .expectation(&circuit, &params, &obs)
                    .unwrap();

                let iters = if nq > 14 || depth > 16 { 3 } else { 5 };
                let mut best = f64::INFINITY;
                let (mut svd_ms, mut gpu_n, mut cpu_n, mut maxc, mut xor) =
                    (0.0, 0u64, 0u64, 0, 0u64);
                let mut value = 0.0;
                let mut last_fallback: Option<&'static str> = None;
                for _ in 0..iters {
                    reset();
                    let t = Instant::now();
                    let v = MpsBackend::new(chi)
                        .with_svd_fn(f)
                        .expectation(&circuit, &params, &obs)
                        .unwrap();
                    let total = t.elapsed().as_secs_f64() * 1000.0;
                    // Keep the SVD figure from the SAME run as the total, not a
                    // min over independent runs — mixing runs can print a share
                    // above 100%.
                    if total < best {
                        best = total;
                        svd_ms = SVD_NANOS.load(Ordering::Relaxed) as f64 / 1e6;
                        let calls = CALLS.load(Ordering::Relaxed);
                        (gpu_n, cpu_n) = if label == "gpu" {
                            let d = cuda_svd_dispatch();
                            // The library must account for every call the
                            // hook saw, once each.
                            assert_eq!(
                                d.gpu_calls + d.cpu_fallbacks,
                                calls as u64,
                                "{nq}q depth={depth} chi={chi}: dispatch {d:?} vs {calls} calls"
                            );
                            last_fallback = d.last_fallback;
                            (d.gpu_calls, d.cpu_fallbacks)
                        } else {
                            (0, calls as u64)
                        };
                        maxc = MAX_COLS.load(Ordering::Relaxed);
                        xor = DW_XOR.load(Ordering::Relaxed);
                        value = v;
                    }
                }
                // A10: a "gpu" row that dispatched nothing is a CPU row with
                // a GPU label, and it looks entirely normal in the table.
                // Checking the context once at startup is not enough — the
                // fallback is per call, so a shape whose theta blocks are all
                // too small silently produces a full row of CPU timings.
                assert!(
                    label != "gpu" || gpu_n > 0,
                    "{nq}q depth={depth} chi={chi}: every SVD call fell back to the CPU, \
                     so this row would report CPU time under a GPU label."
                );
                println!(
                    "{:>4} {:>6} {:>5} {:>4} {:>10.2} {:>10.2} {:>6.1}% {:>6} {:>6} {:>6} {:>18x}   <Z>={:.12e}",
                    nq, depth, chi, label, best, svd_ms, 100.0 * svd_ms / best,
                    gpu_n, cpu_n, maxc, xor, value
                );
                if let Some(reason) = last_fallback {
                    println!("      ^ {cpu_n} cpu fallbacks on the gpu row, last: {reason}");
                }
            }
        }
    }
}

#[cfg(not(all(any(target_os = "linux", target_os = "windows"), feature = "cuda")))]
fn main() {
    println!(
        "Built without the `cuda` feature (or off Linux/Windows). Rebuild with \
         `--features cuda` on a CUDA host; measuring the CPU fallback here would \
         only reproduce `mps_stage_profile`."
    );
}
