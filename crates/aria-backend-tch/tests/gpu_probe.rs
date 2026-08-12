// SPDX-License-Identifier: Apache-2.0
//! Scratch probe: is the linked libtorch CUDA-capable and does it see the GPU?
#[test]
fn report_cuda_availability() {
    let avail = tch::Cuda::is_available();
    let count = tch::Cuda::device_count();
    let cudnn = tch::Cuda::cudnn_is_available();
    eprintln!("tch::Cuda::is_available() = {avail}");
    eprintln!("tch::Cuda::device_count() = {count}");
    eprintln!("tch::Cuda::cudnn_is_available() = {cudnn}");
    // When the build was set up for CUDA (ARIA_EXPECT_CUDA=1, set by
    // tools/setup-libtorch.sh's CUDA verify gate), a false here is a HARD
    // failure — it means libtorch_cuda was dropped from the link or the driver
    // is missing. Otherwise the test just reports and, if a GPU happens to be
    // present, exercises it.
    if std::env::var("ARIA_EXPECT_CUDA").is_ok_and(|v| v == "1") {
        assert!(
            avail,
            "ARIA_EXPECT_CUDA=1 but tch::Cuda::is_available() is false — \
             libtorch_cuda dropped from the link, or no CUDA driver"
        );
    }
    // Actually run a tiny op on the GPU to prove compute works, not just probe.
    if avail {
        let x = tch::Tensor::from_slice(&[1.0f64, 2.0, 3.0]).to_device(tch::Device::Cuda(0));
        let y = (&x * 2.0).sum(tch::Kind::Double);
        let s: f64 = y.double_value(&[]);
        eprintln!("GPU compute: sum(2*[1,2,3]) = {s} (expect 12)");
        assert_eq!(s, 12.0);
        assert!(count >= 1, "CUDA available but device_count 0");
    }
}
