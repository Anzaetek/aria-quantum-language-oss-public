//! Coarse-grained timing of the CUDA `apply_2q` hot path. Separates
//! host-side cost (Step 1 / Step 2 / Step 5 around the SVD) from
//! device-side cost (htod + cuSOLVER + dtoh + sync). Used to decide
//! how much of the speedup gap to close before / after #8.

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
use omega_backend_mps::gates;
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
use omega_backend_mps::mps::Mps;
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
use omega_backend_mps_cuda::CudaSvdContext;

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
const N_QUBITS: usize = 14;
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
const DEPTH: usize = 12;
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
const CHI: usize = 128;
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
const RZ_THETA: f64 = 0.25;
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
const RX_THETA: f64 = 0.15;

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
fn timed_apply_2q(
    ctx: &CudaSvdContext,
    mps: &mut Mps,
    q: usize,
    gate: &[num_complex::Complex64; 16],
    buckets: &mut [u128; 3],
    count: &mut usize,
) {
    use std::cell::Cell;
    use std::time::Instant;
    let t_total = Instant::now();
    let pre_svd = Cell::new(0u128);
    let svd_elapsed = Cell::new(0u128);
    mps.apply_2q_with_svd_flat(q, gate, |matrix, m, n, stride, max_rank, thr| {
        pre_svd.set(t_total.elapsed().as_nanos());
        let svd_start = Instant::now();
        let r = ctx
            .truncated_svd_flat(matrix, m, n, stride, max_rank, thr)
            .unwrap_or_else(|| {
                omega_backend_mps::svd::truncated_svd_flat(matrix, m, n, stride, max_rank, thr)
            });
        svd_elapsed.set(svd_start.elapsed().as_nanos());
        r
    });
    let total = t_total.elapsed().as_nanos();
    let pre = pre_svd.get();
    let svd = svd_elapsed.get();
    let post = total.saturating_sub(pre + svd);
    buckets[0] += pre;
    buckets[1] += svd;
    buckets[2] += post;
    *count += 1;
}

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
fn main() {
    let ctx = match CudaSvdContext::new() {
        Some(c) => c,
        None => {
            eprintln!("no CUDA device");
            return;
        }
    };

    let h = gates::h();
    let cx = gates::cx();
    let rz = gates::rz(RZ_THETA);
    let rx = gates::rx(RX_THETA);

    // Warm-up.
    {
        let mut mps = Mps::zero_state(N_QUBITS, CHI);
        let mut warmup_buckets = [0u128; 3];
        let mut warmup_count = 0;
        for q in 0..N_QUBITS {
            mps.apply_1q(q, &h);
        }
        for d in 0..DEPTH {
            let offset = d & 1;
            let mut q = offset;
            while q + 1 < N_QUBITS {
                timed_apply_2q(
                    &ctx,
                    &mut mps,
                    q,
                    &cx,
                    &mut warmup_buckets,
                    &mut warmup_count,
                );
                mps.apply_1q(q + 1, &rz);
                timed_apply_2q(
                    &ctx,
                    &mut mps,
                    q,
                    &cx,
                    &mut warmup_buckets,
                    &mut warmup_count,
                );
                q += 2;
            }
            for q in 0..N_QUBITS {
                mps.apply_1q(q, &rx);
            }
        }
    }

    // Measured run.
    let mut mps = Mps::zero_state(N_QUBITS, CHI);
    let mut bk = [0u128; 3];
    let mut cnt = 0;
    let t = std::time::Instant::now();
    for q in 0..N_QUBITS {
        mps.apply_1q(q, &h);
    }
    for d in 0..DEPTH {
        let offset = d & 1;
        let mut q = offset;
        while q + 1 < N_QUBITS {
            timed_apply_2q(&ctx, &mut mps, q, &cx, &mut bk, &mut cnt);
            mps.apply_1q(q + 1, &rz);
            timed_apply_2q(&ctx, &mut mps, q, &cx, &mut bk, &mut cnt);
            q += 2;
        }
        for q in 0..N_QUBITS {
            mps.apply_1q(q, &rx);
        }
    }
    let total = t.elapsed();
    std::hint::black_box(&mps);

    println!("MPS bench: n={N_QUBITS}q, depth={DEPTH}, χ={CHI}");
    println!("  total wallclock:    {total:?}");
    println!("  apply_2q count:     {cnt}");
    let sum = bk[0] + bk[1] + bk[2];
    let pct = |x: u128| 100.0 * x as f64 / sum as f64;
    println!(
        "  host pre-SVD (1+2): {:9.2} ms  ({:5.1}% of cuda apply_2q time)",
        bk[0] as f64 / 1e6,
        pct(bk[0])
    );
    println!(
        "  device SVD path:    {:9.2} ms  ({:5.1}%)",
        bk[1] as f64 / 1e6,
        pct(bk[1])
    );
    println!(
        "  host post-SVD (5):  {:9.2} ms  ({:5.1}%)",
        bk[2] as f64 / 1e6,
        pct(bk[2])
    );
    let per_call = (sum as f64) / cnt as f64 / 1e6;
    println!("  per apply_2q avg:   {per_call:.2} ms");
}

#[cfg(not(all(any(target_os = "linux", target_os = "windows"), feature = "cuda")))]
fn main() {
    println!("rebuild with --features cuda");
}
