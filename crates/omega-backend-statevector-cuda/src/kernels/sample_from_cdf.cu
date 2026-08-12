// Per-shot binary search over the on-device CDF.
//
// Companion to `cdf_scan.cu`. One thread per shot:
//   1. Read its uniform `u ∈ [0, 1)` from `uniforms[]`.
//   2. Binary search for the smallest `i` such that `cdf[i] >= u`.
//   3. `atomicAdd(counts[i], 1)`.
//
// `counts` must be pre-zeroed on device by the caller — this kernel
// only accumulates. The output is a dense `dim`-sized `u32` counts
// array, host-side then sparsified into a `HashMap<u64, u32>`.
//
// Numerical note: the host-side path pins `cdf[dim-1] = 1.0` to
// avoid f32 rounding dropping shots on the floor. We do the same on
// the device side via a clamp in the binary search — if every cdf
// entry is `< u` we land on `dim-1`.

extern "C" {

__global__ void sample_from_cdf(
    const real* cdf,
    const real* uniforms,
    unsigned int* counts,
    unsigned long long dim,
    unsigned long long num_shots
) {
    unsigned long long shot = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (shot >= num_shots) { return; }

    real u = uniforms[shot];

    // Binary search: smallest `lo` with `cdf[lo] >= u`. Loop
    // invariants: cdf is monotonically non-decreasing; target index
    // lives in `[lo, hi]` inclusive.
    unsigned long long lo = 0;
    unsigned long long hi = dim - 1;
    while (lo < hi) {
        unsigned long long mid = lo + (hi - lo) / 2;
        if (cdf[mid] < u) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    atomicAdd(&counts[lo], 1u);
}

} // extern "C"
