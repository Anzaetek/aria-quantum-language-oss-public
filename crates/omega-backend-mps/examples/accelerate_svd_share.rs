//! Does Accelerate's `zgesdd` move the SVD share of MPS evolution — the CPU
//! half of `STATUS.md` §5 item 16, after that item's own criterion killed the
//! GPU half.
//!
//! The item measured the SVD at 90-97% of MPS evolution on two ISAs, then
//! measured what a GPU does to it: cuSOLVER on an RTX PRO 6000 Blackwell loses
//! on six of eight shapes, and CUDA is the BEST case for a GPU SVD here
//! (native f64, sub-microsecond dispatch). What the entry names instead is
//! "Accelerate/LAPACK on the CPU side is the cheaper first step and keeps
//! f64". This harness is that step, measured rather than assumed.
//!
//! WHAT IT MIRRORS, deliberately: the circuit, the observable and the row
//! shapes are `omega-backend-mps`'s `mps_stage_profile` and
//! `omega-backend-mps-cuda`'s `cuda_svd_share`, so all three tables compare
//! directly. Both arms run in the same process, back to back, on the same
//! circuit.
//!
//! THREE THINGS IT GUARDS AGAINST, every one of them a defect this item found
//! the hard way:
//!
//! 1. **Measuring reconstruction instead of evolution.** Driven through
//!    `expectation`, never `execute(shots: None)` — the latter ends in a dense
//!    2^n `to_statevector` that is ~97% of the wallclock and that no SVD
//!    accelerator would touch. That artefact is what made the 2026-05-12
//!    deferral's "flat across χ" evidence measure the wrong quantity.
//! 2. **A silent fallback published as an accelerated row.** `zgesdd` returns
//!    `info != 0` on non-convergence and `accelerate_svd_flat` then takes the
//!    Jacobi kernel. The CUDA arm found 20-45% of its "GPU" calls doing
//!    exactly this, unreported. So the two dispatch outcomes are counted
//!    separately and a row that fell back even once says so.
//! 3. **An accelerated path that quietly reports a looser or tighter
//!    certificate.** `discarded_weight` is a BOUND. Every call's weight is
//!    accumulated per arm, and the row prints the accumulated value together
//!    with the expectation value, so a path that moved the certificate is
//!    visible in the table and not only in the unit tests.
//!
//! Run the shapes you can afford:
//!   OMEGA_PROFILE_SHAPES=14x4,14x12 OMEGA_PROFILE_CHIS=32 cargo run --release \
//!       -p omega-backend-mps --example accelerate_svd_share

#[cfg(target_os = "macos")]
fn main() {
    use num_complex::Complex64;
    use omega_backend_mps::accelerate;
    use omega_backend_mps::svd::{truncated_svd_flat, SvdResultFlat};
    use omega_backend_mps::MpsBackend;
    use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
    use omega_core::executor::{Backend, Observable, PauliOp};
    use omega_core::params::ParameterBinding;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::time::Instant;

    static SVD_NANOS: AtomicU64 = AtomicU64::new(0);
    static LAPACK_CALLS: AtomicUsize = AtomicUsize::new(0);
    static JACOBI_CALLS: AtomicUsize = AtomicUsize::new(0);
    static MAX_COLS: AtomicUsize = AtomicUsize::new(0);
    /// Accumulated `discarded_weight` in fixed-point units of 2^-50, so the
    /// figure is comparable between arms without depending on the order rayon
    /// happened to call them in: a plain f64 sum is not associative and would
    /// differ between arms from ordering alone. 2^-50 resolves ~9e-16 per call
    /// against the ~1e-8 weights a truncating split reports, so the
    /// quantisation is far below the difference between two kernels, and the
    /// scale still holds an accumulated weight of ~16 000.
    static DW_SCALED: AtomicU64 = AtomicU64::new(0);
    const DW_SCALE: f64 = (1u64 << 50) as f64;
    /// Splits where the Accelerate bound came out BELOW the Jacobi reference's
    /// on the same input — the one outcome that would sink this. Counted only
    /// under `OMEGA_ACC_VERIFY_BOUND`, which runs both kernels per call.
    static BOUND_UNDERCUTS: AtomicUsize = AtomicUsize::new(0);
    /// Splits actually compared, so a zero-undercut report cannot come from a
    /// verification that never ran.
    static BOUND_CHECKED: AtomicUsize = AtomicUsize::new(0);

    fn record(out: &SvdResultFlat, cols: usize, t: Instant) {
        SVD_NANOS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        MAX_COLS.fetch_max(cols, Ordering::Relaxed);
        DW_SCALED.fetch_add((out.discarded_weight * DW_SCALE) as u64, Ordering::Relaxed);
    }

    /// The certificate check, on the PRODUCTION input rather than on a
    /// synthetic corpus: the unit tests compare the two kernels on matrices
    /// this file believes a θ block looks like, and that belief is the thing an
    /// item like this gets wrong. Off by default because it runs both kernels
    /// per call and so destroys the timing — the timing rows and the bound
    /// rows are deliberately separate runs.
    fn check_bound(
        out: &SvdResultFlat,
        a: &[Complex64],
        rows: usize,
        cols: usize,
        lda: usize,
        max_rank: usize,
        threshold: f64,
    ) {
        if !verify_bound() {
            return;
        }
        let reference = truncated_svd_flat(a, rows, cols, lda, max_rank, threshold);
        BOUND_CHECKED.fetch_add(1, Ordering::Relaxed);
        if out.discarded_weight < reference.discarded_weight {
            BOUND_UNDERCUTS.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The production Jacobi kernel behind the same timing shim
    /// `mps_stage_profile` uses, so the baseline column here IS that table's.
    fn jacobi_arm(
        a: &[Complex64],
        rows: usize,
        cols: usize,
        lda: usize,
        max_rank: usize,
        threshold: f64,
    ) -> SvdResultFlat {
        let t = Instant::now();
        let out = truncated_svd_flat(a, rows, cols, lda, max_rank, threshold);
        JACOBI_CALLS.fetch_add(1, Ordering::Relaxed);
        record(&out, cols, t);
        out
    }

    /// Accelerate first, Jacobi on a declined solve — the same policy
    /// `accelerate_svd_flat` ships, with the two outcomes counted instead of
    /// indistinguishable.
    fn accelerate_arm(
        a: &[Complex64],
        rows: usize,
        cols: usize,
        lda: usize,
        max_rank: usize,
        threshold: f64,
    ) -> SvdResultFlat {
        let t = Instant::now();
        if let Some(out) = accelerate::truncated_svd_flat(a, rows, cols, lda, max_rank, threshold) {
            LAPACK_CALLS.fetch_add(1, Ordering::Relaxed);
            record(&out, cols, t);
            check_bound(&out, a, rows, cols, lda, max_rank, threshold);
            return out;
        }
        let out = truncated_svd_flat(a, rows, cols, lda, max_rank, threshold);
        JACOBI_CALLS.fetch_add(1, Ordering::Relaxed);
        record(&out, cols, t);
        out
    }

    /// The SHIPPED entry point, `accelerate_svd_flat`: LAPACK above the
    /// measured size gate, Jacobi below it. The `accelerate` arm above
    /// deliberately ignores the gate so the share figure prices the kernel;
    /// this arm prices what a caller installing the hook would actually get,
    /// and the two differ exactly on the shapes the gate rejects.
    fn gated_arm(
        a: &[Complex64],
        rows: usize,
        cols: usize,
        lda: usize,
        max_rank: usize,
        threshold: f64,
    ) -> SvdResultFlat {
        let t = Instant::now();
        // Count the library's own decision rather than a copy of it.
        if accelerate::lapack_is_worth_it(rows, cols) {
            LAPACK_CALLS.fetch_add(1, Ordering::Relaxed);
        } else {
            JACOBI_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        let out = accelerate::accelerate_svd_flat(a, rows, cols, lda, max_rank, threshold);
        record(&out, cols, t);
        // The SHIPPED entry point is the one whose bound has to hold, so it is
        // verified here too and not only on the ungated kernel.
        check_bound(&out, a, rows, cols, lda, max_rank, threshold);
        out
    }

    fn verify_bound() -> bool {
        std::env::var("OMEGA_ACC_VERIFY_BOUND").is_ok_and(|v| v.trim() == "1")
    }

    fn reset() {
        SVD_NANOS.store(0, Ordering::Relaxed);
        LAPACK_CALLS.store(0, Ordering::Relaxed);
        JACOBI_CALLS.store(0, Ordering::Relaxed);
        MAX_COLS.store(0, Ordering::Relaxed);
        DW_SCALED.store(0, Ordering::Relaxed);
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

    fn shapes_from_env() -> Vec<(u32, usize)> {
        let Ok(spec) = std::env::var("OMEGA_PROFILE_SHAPES") else {
            return vec![(14, 4), (14, 12), (14, 24)];
        };
        spec.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                let (q, d) = s.split_once('x').expect("shape entry is QxDEPTH");
                (q.parse().expect("qubits"), d.parse().expect("depth"))
            })
            .collect()
    }

    fn chis_from_env() -> Vec<usize> {
        let Ok(spec) = std::env::var("OMEGA_PROFILE_CHIS") else {
            return vec![32, 128, 256];
        };
        spec.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().expect("chi"))
            .collect()
    }

    println!("rayon threads={}", rayon::current_num_threads());
    let params = ParameterBinding::new();
    println!(
        "\n{:>4} {:>6} {:>5} {:>11} {:>10} {:>10} {:>7} {:>8} {:>8} {:>6} {:>20}",
        "q",
        "depth",
        "chi",
        "svd",
        "evolve ms",
        "svd ms",
        "share",
        "lapack",
        "jacobi",
        "maxc",
        "acc discarded"
    );

    for (nq, depth) in shapes_from_env() {
        let circuit = entangling_circuit(nq, depth);
        for chi in chis_from_env() {
            let obs = Observable {
                terms: vec![(1.0, vec![(0u32, PauliOp::Z)])],
            };
            let mut baseline_ms = f64::NAN;
            for (label, f) in [
                ("jacobi", jacobi_arm as omega_backend_mps::SvdFlatFn),
                ("accelerate", accelerate_arm as omega_backend_mps::SvdFlatFn),
                ("gated", gated_arm as omega_backend_mps::SvdFlatFn),
            ] {
                // Warm: first touch pays the allocator, the rayon pool and the
                // per-thread LAPACK workspace query, none of which belong in
                // the number.
                MpsBackend::new(chi)
                    .with_svd_fn(f)
                    .expectation(&circuit, &params, &obs)
                    .unwrap();

                let iters = if nq > 14 || depth > 16 { 3 } else { 5 };
                let mut best = f64::INFINITY;
                let (mut svd_ms, mut lp, mut jb, mut maxc, mut dw) = (0.0, 0, 0, 0, 0u64);
                let mut value = 0.0;
                for _ in 0..iters {
                    reset();
                    let t = Instant::now();
                    let v = MpsBackend::new(chi)
                        .with_svd_fn(f)
                        .expectation(&circuit, &params, &obs)
                        .unwrap();
                    let total = t.elapsed().as_secs_f64() * 1000.0;
                    // Keep the SVD figure from the SAME run as the total, not
                    // a min over independent runs — mixing runs can print a
                    // share above 100%.
                    if total < best {
                        best = total;
                        svd_ms = SVD_NANOS.load(Ordering::Relaxed) as f64 / 1e6;
                        lp = LAPACK_CALLS.load(Ordering::Relaxed);
                        jb = JACOBI_CALLS.load(Ordering::Relaxed);
                        maxc = MAX_COLS.load(Ordering::Relaxed);
                        dw = DW_SCALED.load(Ordering::Relaxed);
                        value = v;
                    }
                }
                // A10: an "accelerate" row that dispatched nothing to LAPACK
                // is a Jacobi row with a LAPACK label, and it looks entirely
                // normal in the table. The fallback is per call, so checking
                // availability once would not catch it. The `gated` arm is
                // exempt: zero LAPACK dispatches there is the gate working on
                // a shape below the threshold, which is a result and not a
                // mislabelling.
                assert!(
                    label != "accelerate" || lp > 0,
                    "{nq}q depth={depth} chi={chi}: every SVD call fell back to the Jacobi \
                     kernel, so this row would report baseline time under a LAPACK label."
                );
                if label == "jacobi" {
                    baseline_ms = best;
                }
                let speedup = if label == "jacobi" {
                    String::new()
                } else {
                    format!("  {:.2}x on evolution", baseline_ms / best)
                };
                println!(
                    "{:>4} {:>6} {:>5} {:>11} {:>10.2} {:>10.2} {:>6.1}% {:>8} {:>8} {:>6} \
                     {:>20.12e}   <Z>={:.12e}{}",
                    nq,
                    depth,
                    chi,
                    label,
                    best,
                    svd_ms,
                    100.0 * svd_ms / best,
                    lp,
                    jb,
                    maxc,
                    dw as f64 / DW_SCALE,
                    value,
                    speedup
                );
            }
        }
    }

    if verify_bound() {
        let checked = BOUND_CHECKED.load(Ordering::Relaxed);
        let under = BOUND_UNDERCUTS.load(Ordering::Relaxed);
        // A10: "0 undercuts" out of 0 comparisons is what a verification that
        // never ran looks like, and it is indistinguishable from a clean
        // result in a log. Refuse to report the former as the latter.
        assert!(
            checked > 0,
            "OMEGA_ACC_VERIFY_BOUND was set but no split was compared — every SVD call \
             declined the LAPACK path, so this run proves nothing about the bound."
        );
        println!(
            "\nbound check: {under} of {checked} splits reported LESS discarded weight than \
             the Jacobi reference"
        );
        assert_eq!(
            under, 0,
            "the truncation certificate is a BOUND: {under} of {checked} splits on the real \
             production path under-reported it against the Jacobi reference."
        );
    } else {
        println!(
            "\nbound check: not run (set OMEGA_ACC_VERIFY_BOUND=1; it runs both kernels \
                  per call and the timings above would be meaningless)"
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!(
        "Accelerate is macOS-only. On another host the baseline column is already \
         `omega-backend-mps`'s mps_stage_profile."
    );
}
