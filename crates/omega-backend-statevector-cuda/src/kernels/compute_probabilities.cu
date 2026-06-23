// Per-amplitude probability kernel.
//
// Reads the interleaved float2 statevector and writes the
// per-amplitude squared magnitude to a parallel f32 array. Used by
// the shot-sampling path so the host pulls 1×dim f32s instead of
// 2×dim f32s (the full statevector). At n=20 that's 4 MB vs 8 MB
// of memcpy_dtoh.
//
// Future direction (not in this slice): parallel inclusive scan
// in-place + a sample-on-device kernel that takes (cdf, num_shots,
// seed) and writes a HashMap<u64, u32>-equivalent on device. That
// would skip the host roundtrip entirely. The pieces are:
//
//   * thread-block scan via shared memory
//   * 2-pass cross-block reduction (per-block totals → exclusive
//     scan → second pass adds the per-block offset)
//   * per-shot binary search kernel
//
// All standard but a sizable slice. The intermediate landing here
// (compute_probabilities only) gets a measurable but small win —
// host-side CDF is still O(dim), but the memcpy_dtoh transfer is
// halved.

extern "C" {

__global__ void compute_probabilities(
    const float2* state,
    float* probs,
    unsigned long long dim
) {
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= dim) { return; }
    float2 amp = state[gid];
    probs[gid] = amp.x * amp.x + amp.y * amp.y;
}

} // extern "C"
