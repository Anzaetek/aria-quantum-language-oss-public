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
    real u00_re; real u00_im;
    real u01_re; real u01_im;
    real u10_re; real u10_im;
    real u11_re; real u11_im;
};

__device__ inline real2 cmul(real2 a, real2 b) {
    return make_real2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ inline real2 cadd(real2 a, real2 b) {
    return make_real2(a.x + b.x, a.y + b.y);
}

__global__ void apply_1q_from_to_pooled(
    const real2* src,
    real2* dst,
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

    real2 a = src[i0];
    real2 b = src[i1];
    real2 u00 = make_real2(params.u00_re, params.u00_im);
    real2 u01 = make_real2(params.u01_re, params.u01_im);
    real2 u10 = make_real2(params.u10_re, params.u10_im);
    real2 u11 = make_real2(params.u11_re, params.u11_im);

    dst[i0] = cadd(cmul(u00, a), cmul(u01, b));
    dst[i1] = cadd(cmul(u10, a), cmul(u11, b));
}

} // extern "C"
