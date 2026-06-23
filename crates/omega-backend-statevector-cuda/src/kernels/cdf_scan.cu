// Inclusive prefix-sum (CDF) of a per-amplitude `probs` array,
// for the on-device shot sampler (see `sample_from_cdf.cu`).
//
// Multi-level scan: this file ships two kernels and the driver in
// `imp.rs` composes them recursively so the sampler handles any
// `dim` up to the device's allocatable f32 buffer size (n_qubits ≤
// ~30 in practice).
//
// Pipeline at depth d (operating on a buffer of length `L`):
//
//   1. `cdf_block_scan_pass1`
//      Per-block inclusive Hillis-Steele scan in shared memory.
//      Writes the in-block scan back in place, and the per-block
//      sum to `block_totals[block_idx]`.
//
//   2. Recurse on `block_totals` (length `ceil(L / BS)`) until the
//      recursion's array fits in a single block, where pass1 alone
//      produces a full inclusive scan.
//
//   3. `cdf_block_scan_pass2_from_inclusive`
//      Adds the prior-block offset to each element. With
//      block_totals[] now inclusive-scanned at this level, the
//      offset for block `b` is `block_totals[b - 1]` (and 0 for
//      `b == 0`).
//
// All scans are f32 throughout, matching the existing
// `compute_probabilities` output. The host-side CDF path uses
// `f32 → f64` promote per slot which is irrelevant after the SVs
// land in the same f32 buffer here.

extern "C" {

// Phase 1: per-block inclusive scan via Hillis-Steele in shared
// memory. `blockDim.x` must equal the launch site's `BLOCK_SIZE`
// constant (1024). Threads beyond `len` read 0 so the in-block scan
// is well-defined on the partial-final-block.
__global__ void cdf_block_scan_pass1(
    float* data,
    float* block_totals,
    unsigned long long len
) {
    extern __shared__ float buf[];
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    float v = (gid < len) ? data[gid] : 0.0f;
    buf[tid] = v;
    __syncthreads();

    for (unsigned int offset = 1; offset < blockDim.x; offset *= 2) {
        float add = (tid >= offset) ? buf[tid - offset] : 0.0f;
        __syncthreads();
        buf[tid] += add;
        __syncthreads();
    }

    if (gid < len) {
        data[gid] = buf[tid];
    }
    if (tid == blockDim.x - 1) {
        block_totals[blockIdx.x] = buf[tid];
    }
}

// Phase 2: add the *prior* block's inclusive total to every element
// in this block. With `block_totals_inclusive` already containing
// the level's full inclusive scan, the offset for block `b` is
// `block_totals_inclusive[b - 1]` (or 0 for `b == 0`).
//
// (The previous single-block "exclusive scan" kernel was replaced
// by this read-time conversion so the cross-block scan can recurse
// uniformly in inclusive-scan terms.)
__global__ void cdf_block_scan_pass2_from_inclusive(
    float* data,
    const float* block_totals_inclusive,
    unsigned long long len
) {
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= len) { return; }
    float offset = (blockIdx.x == 0) ? 0.0f : block_totals_inclusive[blockIdx.x - 1];
    data[gid] += offset;
}

} // extern "C"
