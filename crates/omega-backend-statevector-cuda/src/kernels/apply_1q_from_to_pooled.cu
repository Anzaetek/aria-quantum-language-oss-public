// Out-of-place pooled 1q gate — reads `src` and writes `dst[i] =
// (U · src)[i]`. Used by the captured backward sweep to fuse the
// `CopyPhiToTemp` + `Deriv1q` step into one kernel: instead of
// emitting two graph nodes (a state-to-state memcpy + an in-place
// gate apply), the captured graph walks one node that reads φ and
// writes (∂U · φ) directly to temp.
//
// The amplitudes for indices NOT touched by the gate's
// flip-bit are also written through verbatim, so `dst` ends up
// being a proper full copy of (U · src) — `temp` need not be
// pre-initialised before the launch.

extern "C" {

struct Apply1qParams {
    unsigned int qubit;
    float u00_re; float u00_im;
    float u01_re; float u01_im;
    float u10_re; float u10_im;
    float u11_re; float u11_im;
};

__device__ inline float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ inline float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__global__ void apply_1q_from_to_pooled(
    const float2* src,
    float2* dst,
    const Apply1qParams* pool,
    unsigned int slot,
    unsigned long long pairs
) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= pairs) { return; }
    Apply1qParams params = pool[slot];
    unsigned long long mask = 1ULL << params.qubit;
    unsigned long long low_bits = tid & (mask - 1ULL);
    unsigned long long high = (tid >> params.qubit) << (params.qubit + 1u);
    unsigned long long i0 = low_bits | high;
    unsigned long long i1 = i0 | mask;

    float2 a = src[i0];
    float2 b = src[i1];
    float2 u00 = make_float2(params.u00_re, params.u00_im);
    float2 u01 = make_float2(params.u01_re, params.u01_im);
    float2 u10 = make_float2(params.u10_re, params.u10_im);
    float2 u11 = make_float2(params.u11_re, params.u11_im);

    dst[i0] = cadd(cmul(u00, a), cmul(u01, b));
    dst[i1] = cadd(cmul(u10, a), cmul(u11, b));
}

} // extern "C"
