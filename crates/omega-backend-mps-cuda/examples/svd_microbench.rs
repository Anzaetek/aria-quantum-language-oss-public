//! Per-call timing for the three cuSOLVER complex SVD entry points
//! at the shape that dominates our brickwall χ=128 bench:
//! `Zgesvdj` (Jacobi, current), `Zgesvd` (bidiag + QR), `Zgesvda`
//! (randomized approximate).
//!
//! Run with:
//! ```text
//! LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH \
//!     cargo run --release -p omega-backend-mps-cuda --features cuda \
//!         --example svd_microbench
//! ```

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
fn main() {
    use cudarc::cusolver::safe::DnHandle;
    use cudarc::cusolver::sys as cusolver_sys;
    use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
    use std::time::Instant;

    const M: usize = 256;
    const N: usize = 256;
    const RANK: usize = 128;
    const ITERS: usize = 50;

    let ctx = CudaContext::new(0).expect("cuda device");
    let stream = ctx.default_stream();
    let handle = DnHandle::new(stream.clone()).expect("dn handle");

    // Build a random-ish input matrix on the device. Column-major
    // (re, im) f64 pairs, m × n.
    let mut a_host: Vec<f64> = Vec::with_capacity(2 * M * N);
    for col in 0..N {
        for row in 0..M {
            let re = (1.0 + ((row * 13 + col * 7) % 19) as f64) / 19.0;
            let im = (0.5 + ((row * 11 + col * 3) % 17) as f64) / 17.0 - 0.5;
            a_host.push(re);
            a_host.push(im);
        }
    }

    // ----- Zgesvdj -----
    {
        let mut a = stream.alloc_zeros::<f64>(2 * M * N).unwrap();
        let mut s = stream.alloc_zeros::<f64>(M.min(N)).unwrap();
        let mut u = stream.alloc_zeros::<f64>(2 * M * M.min(N)).unwrap();
        let mut v = stream.alloc_zeros::<f64>(2 * N * M.min(N)).unwrap();
        let mut info = stream.alloc_zeros::<i32>(1).unwrap();

        let mut params: cusolver_sys::gesvdjInfo_t = std::ptr::null_mut();
        unsafe {
            cusolver_sys::cusolverDnCreateGesvdjInfo(&mut params)
                .result()
                .unwrap()
        };
        unsafe {
            cusolver_sys::cusolverDnXgesvdjSetTolerance(params, 1e-7)
                .result()
                .unwrap()
        };
        unsafe {
            cusolver_sys::cusolverDnXgesvdjSetMaxSweeps(params, 100)
                .result()
                .unwrap()
        };

        let jobz = cusolver_sys::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
        let econ: i32 = 1;
        let mut lwork: i32 = 0;
        {
            let (a_p, _) = a.device_ptr(&stream);
            let (s_p, _) = s.device_ptr(&stream);
            let (u_p, _) = u.device_ptr(&stream);
            let (v_p, _) = v.device_ptr(&stream);
            unsafe {
                cusolver_sys::cusolverDnZgesvdj_bufferSize(
                    handle.cu(),
                    jobz,
                    econ,
                    M as i32,
                    N as i32,
                    a_p as *const _,
                    M as i32,
                    s_p as *const _,
                    u_p as *const _,
                    M as i32,
                    v_p as *const _,
                    N as i32,
                    &mut lwork,
                    params,
                )
                .result()
                .unwrap();
            }
        }
        let mut work = stream
            .alloc_zeros::<f64>(2 * lwork.max(1) as usize)
            .unwrap();

        // Warmup.
        for _ in 0..3 {
            stream.memcpy_htod(&a_host, &mut a).unwrap();
            let (a_p, _) = a.device_ptr_mut(&stream);
            let (s_p, _) = s.device_ptr_mut(&stream);
            let (u_p, _) = u.device_ptr_mut(&stream);
            let (v_p, _) = v.device_ptr_mut(&stream);
            let (w_p, _) = work.device_ptr_mut(&stream);
            let (i_p, _) = info.device_ptr_mut(&stream);
            unsafe {
                cusolver_sys::cusolverDnZgesvdj(
                    handle.cu(),
                    jobz,
                    econ,
                    M as i32,
                    N as i32,
                    a_p as *mut _,
                    M as i32,
                    s_p as *mut _,
                    u_p as *mut _,
                    M as i32,
                    v_p as *mut _,
                    N as i32,
                    w_p as *mut _,
                    lwork,
                    i_p as *mut _,
                    params,
                )
                .result()
                .unwrap();
            }
            stream.synchronize().unwrap();
        }
        let t = Instant::now();
        for _ in 0..ITERS {
            stream.memcpy_htod(&a_host, &mut a).unwrap();
            let (a_p, _) = a.device_ptr_mut(&stream);
            let (s_p, _) = s.device_ptr_mut(&stream);
            let (u_p, _) = u.device_ptr_mut(&stream);
            let (v_p, _) = v.device_ptr_mut(&stream);
            let (w_p, _) = work.device_ptr_mut(&stream);
            let (i_p, _) = info.device_ptr_mut(&stream);
            unsafe {
                cusolver_sys::cusolverDnZgesvdj(
                    handle.cu(),
                    jobz,
                    econ,
                    M as i32,
                    N as i32,
                    a_p as *mut _,
                    M as i32,
                    s_p as *mut _,
                    u_p as *mut _,
                    M as i32,
                    v_p as *mut _,
                    N as i32,
                    w_p as *mut _,
                    lwork,
                    i_p as *mut _,
                    params,
                )
                .result()
                .unwrap();
            }
            stream.synchronize().unwrap();
        }
        let elapsed = t.elapsed().as_secs_f64() / ITERS as f64;
        println!(
            "Zgesvdj  (Jacobi, tol=1e-7): {:.2} ms/call, lwork={}",
            elapsed * 1000.0,
            lwork
        );

        unsafe {
            cusolver_sys::cusolverDnDestroyGesvdjInfo(params)
                .result()
                .unwrap()
        };
    }

    // ----- Zgesvd (bidiag + QR) -----
    {
        let mut a = stream.alloc_zeros::<f64>(2 * M * N).unwrap();
        let mut s = stream.alloc_zeros::<f64>(M.min(N)).unwrap();
        let mut u = stream.alloc_zeros::<f64>(2 * M * M.min(N)).unwrap();
        let mut vt = stream.alloc_zeros::<f64>(2 * M.min(N) * N).unwrap();
        let mut info = stream.alloc_zeros::<i32>(1).unwrap();

        let mut lwork: i32 = 0;
        unsafe {
            cusolver_sys::cusolverDnZgesvd_bufferSize(handle.cu(), M as i32, N as i32, &mut lwork)
                .result()
                .unwrap();
        }
        let mut work = stream
            .alloc_zeros::<f64>(2 * lwork.max(1) as usize)
            .unwrap();
        let mut rwork = stream.alloc_zeros::<f64>(5 * M.min(N)).unwrap();

        // jobu/jobvt: 'S' for thin SVD.
        let jobu: i8 = b'S' as i8;
        let jobvt: i8 = b'S' as i8;

        for _ in 0..3 {
            stream.memcpy_htod(&a_host, &mut a).unwrap();
            let (a_p, _) = a.device_ptr_mut(&stream);
            let (s_p, _) = s.device_ptr_mut(&stream);
            let (u_p, _) = u.device_ptr_mut(&stream);
            let (vt_p, _) = vt.device_ptr_mut(&stream);
            let (w_p, _) = work.device_ptr_mut(&stream);
            let (rw_p, _) = rwork.device_ptr_mut(&stream);
            let (i_p, _) = info.device_ptr_mut(&stream);
            unsafe {
                cusolver_sys::cusolverDnZgesvd(
                    handle.cu(),
                    jobu,
                    jobvt,
                    M as i32,
                    N as i32,
                    a_p as *mut _,
                    M as i32,
                    s_p as *mut _,
                    u_p as *mut _,
                    M as i32,
                    vt_p as *mut _,
                    M.min(N) as i32,
                    w_p as *mut _,
                    lwork,
                    rw_p as *mut _,
                    i_p as *mut _,
                )
                .result()
                .unwrap();
            }
            stream.synchronize().unwrap();
        }
        let t = Instant::now();
        for _ in 0..ITERS {
            stream.memcpy_htod(&a_host, &mut a).unwrap();
            let (a_p, _) = a.device_ptr_mut(&stream);
            let (s_p, _) = s.device_ptr_mut(&stream);
            let (u_p, _) = u.device_ptr_mut(&stream);
            let (vt_p, _) = vt.device_ptr_mut(&stream);
            let (w_p, _) = work.device_ptr_mut(&stream);
            let (rw_p, _) = rwork.device_ptr_mut(&stream);
            let (i_p, _) = info.device_ptr_mut(&stream);
            unsafe {
                cusolver_sys::cusolverDnZgesvd(
                    handle.cu(),
                    jobu,
                    jobvt,
                    M as i32,
                    N as i32,
                    a_p as *mut _,
                    M as i32,
                    s_p as *mut _,
                    u_p as *mut _,
                    M as i32,
                    vt_p as *mut _,
                    M.min(N) as i32,
                    w_p as *mut _,
                    lwork,
                    rw_p as *mut _,
                    i_p as *mut _,
                )
                .result()
                .unwrap();
            }
            stream.synchronize().unwrap();
        }
        let elapsed = t.elapsed().as_secs_f64() / ITERS as f64;
        println!(
            "Zgesvd   (bidiag + QR)     : {:.2} ms/call, lwork={}",
            elapsed * 1000.0,
            lwork
        );
    }

    // ----- Zgesvda (randomized, rank=128) -----
    {
        let mut a = stream.alloc_zeros::<f64>(2 * M * N).unwrap();
        let mut s = stream.alloc_zeros::<f64>(RANK).unwrap();
        let mut u = stream.alloc_zeros::<f64>(2 * M * RANK).unwrap();
        let mut v = stream.alloc_zeros::<f64>(2 * N * RANK).unwrap();
        let mut info = stream.alloc_zeros::<i32>(1).unwrap();

        let jobz = cusolver_sys::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
        let mut lwork: i32 = 0;
        {
            let (a_p, _) = a.device_ptr(&stream);
            let (s_p, _) = s.device_ptr(&stream);
            let (u_p, _) = u.device_ptr(&stream);
            let (v_p, _) = v.device_ptr(&stream);
            unsafe {
                cusolver_sys::cusolverDnZgesvdaStridedBatched_bufferSize(
                    handle.cu(),
                    jobz,
                    RANK as i32,
                    M as i32,
                    N as i32,
                    a_p as *const _,
                    M as i32,
                    (2 * M * N) as i64,
                    s_p as *const _,
                    RANK as i64,
                    u_p as *const _,
                    M as i32,
                    (2 * M * RANK) as i64,
                    v_p as *const _,
                    N as i32,
                    (2 * N * RANK) as i64,
                    &mut lwork,
                    1,
                )
                .result()
                .unwrap();
            }
        }
        let mut work = stream
            .alloc_zeros::<f64>(2 * lwork.max(1) as usize)
            .unwrap();
        let mut r_nrm_f = [0.0f64; 1];

        for _ in 0..3 {
            stream.memcpy_htod(&a_host, &mut a).unwrap();
            let (a_p, _) = a.device_ptr_mut(&stream);
            let (s_p, _) = s.device_ptr_mut(&stream);
            let (u_p, _) = u.device_ptr_mut(&stream);
            let (v_p, _) = v.device_ptr_mut(&stream);
            let (w_p, _) = work.device_ptr_mut(&stream);
            let (i_p, _) = info.device_ptr_mut(&stream);
            unsafe {
                cusolver_sys::cusolverDnZgesvdaStridedBatched(
                    handle.cu(),
                    jobz,
                    RANK as i32,
                    M as i32,
                    N as i32,
                    a_p as *const _,
                    M as i32,
                    (2 * M * N) as i64,
                    s_p as *mut _,
                    RANK as i64,
                    u_p as *mut _,
                    M as i32,
                    (2 * M * RANK) as i64,
                    v_p as *mut _,
                    N as i32,
                    (2 * N * RANK) as i64,
                    w_p as *mut _,
                    lwork,
                    i_p as *mut _,
                    r_nrm_f.as_mut_ptr(),
                    1,
                )
                .result()
                .unwrap();
            }
            stream.synchronize().unwrap();
        }
        let t = Instant::now();
        for _ in 0..ITERS {
            stream.memcpy_htod(&a_host, &mut a).unwrap();
            let (a_p, _) = a.device_ptr_mut(&stream);
            let (s_p, _) = s.device_ptr_mut(&stream);
            let (u_p, _) = u.device_ptr_mut(&stream);
            let (v_p, _) = v.device_ptr_mut(&stream);
            let (w_p, _) = work.device_ptr_mut(&stream);
            let (i_p, _) = info.device_ptr_mut(&stream);
            unsafe {
                cusolver_sys::cusolverDnZgesvdaStridedBatched(
                    handle.cu(),
                    jobz,
                    RANK as i32,
                    M as i32,
                    N as i32,
                    a_p as *const _,
                    M as i32,
                    (2 * M * N) as i64,
                    s_p as *mut _,
                    RANK as i64,
                    u_p as *mut _,
                    M as i32,
                    (2 * M * RANK) as i64,
                    v_p as *mut _,
                    N as i32,
                    (2 * N * RANK) as i64,
                    w_p as *mut _,
                    lwork,
                    i_p as *mut _,
                    r_nrm_f.as_mut_ptr(),
                    1,
                )
                .result()
                .unwrap();
            }
            stream.synchronize().unwrap();
        }
        let elapsed = t.elapsed().as_secs_f64() / ITERS as f64;
        println!(
            "Zgesvda  (randomized r=128): {:.2} ms/call, lwork={}",
            elapsed * 1000.0,
            lwork
        );
    }
}

#[cfg(not(all(any(target_os = "linux", target_os = "windows"), feature = "cuda")))]
fn main() {
    println!("rebuild with --features cuda");
}
