// Stage G4 — dual-state pooled 1q gate. Applies U from
// `pool[slot]` to BOTH `state_a` (φ) and `state_b` (ν) in one
// kernel call. Used by the captured backward sweep to fuse
// `DaggerPhi1q` + `DaggerNu1q` graph nodes for non-parameterised
// 1q ops (H, X, Y, Z, S, T, etc. on the Phase 4c HEA shape) into
// a single graph node — same pattern as `apply_2q_pooled_dual`
// but for the 1q kernel kind.

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

__global__ void apply_1q_pooled_dual(
    real2* state_a,
    real2* state_b,
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

    real2 u00 = make_real2(params.u00_re, params.u00_im);
    real2 u01 = make_real2(params.u01_re, params.u01_im);
    real2 u10 = make_real2(params.u10_re, params.u10_im);
    real2 u11 = make_real2(params.u11_re, params.u11_im);

    // State A.
    {
        real2 a = state_a[i0];
        real2 b = state_a[i1];
        state_a[i0] = cadd(cmul(u00, a), cmul(u01, b));
        state_a[i1] = cadd(cmul(u10, a), cmul(u11, b));
    }
    // State B.
    {
        real2 a = state_b[i0];
        real2 b = state_b[i1];
        state_b[i0] = cadd(cmul(u00, a), cmul(u01, b));
        state_b[i1] = cadd(cmul(u10, a), cmul(u11, b));
    }
}

} // extern "C"
