// Generic single-qubit gate kernel — pooled-param variant of
// apply_1q.cu. Reads the 9-field Apply1qParams struct from a
// device-side pool entry instead of taking it by-value. Used by
// the CUDA-graph capture+replay path so per-training-point gate
// matrices update in place via memcpy_htod between replays.

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

__global__ void apply_1q_pooled(
    float2* state,
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

    float2 a = state[i0];
    float2 b = state[i1];
    float2 u00 = make_float2(params.u00_re, params.u00_im);
    float2 u01 = make_float2(params.u01_re, params.u01_im);
    float2 u10 = make_float2(params.u10_re, params.u10_im);
    float2 u11 = make_float2(params.u11_re, params.u11_im);

    state[i0] = cadd(cmul(u00, a), cmul(u01, b));
    state[i1] = cadd(cmul(u10, a), cmul(u11, b));
}

} // extern "C"
