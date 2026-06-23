// Generic two-qubit gate kernel — pooled-param variant of
// apply_2q.cu. Reads the 34-field Apply2qParams struct from a
// device-side pool entry. Same address-arithmetic and reduction
// shape as the by-value variant; only the parameter source
// differs.

extern "C" {

struct Apply2qParams {
    unsigned int qa;
    unsigned int qb;
    float u[32];
};

__device__ inline float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ inline float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ inline float2 ucell_pooled(const Apply2qParams& p, unsigned int r, unsigned int c) {
    unsigned int k = r * 4u + c;
    return make_float2(p.u[2u * k], p.u[2u * k + 1u]);
}

__global__ void apply_2q_pooled(
    float2* state,
    const Apply2qParams* pool,
    unsigned int slot,
    unsigned long long quads
) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= quads) { return; }

    Apply2qParams params = pool[slot];
    unsigned int qa = params.qa;
    unsigned int qb = params.qb;
    unsigned int qmin = qa < qb ? qa : qb;
    unsigned int qmax = qa > qb ? qa : qb;

    unsigned long long low_mask = (1ULL << qmin) - 1ULL;
    unsigned int mid_count = qmax - qmin - 1u;
    unsigned long long mid_mask = (1ULL << mid_count) - 1ULL;

    unsigned long long low = tid & low_mask;
    unsigned long long mid = ((tid >> qmin) & mid_mask) << (qmin + 1u);
    unsigned long long high = (tid >> (qmax - 1u)) << (qmax + 1u);
    unsigned long long i00 = low | mid | high;

    unsigned long long mask_a = 1ULL << qa;
    unsigned long long mask_b = 1ULL << qb;
    unsigned long long i01 = i00 | mask_a;
    unsigned long long i10 = i00 | mask_b;
    unsigned long long i11 = i00 | mask_a | mask_b;

    float2 v[4];
    v[0] = state[i00];
    v[1] = state[i01];
    v[2] = state[i10];
    v[3] = state[i11];

    float2 o[4];
    for (unsigned int r = 0u; r < 4u; r++) {
        float2 acc = make_float2(0.0f, 0.0f);
        for (unsigned int c = 0u; c < 4u; c++) {
            acc = cadd(acc, cmul(ucell_pooled(params, r, c), v[c]));
        }
        o[r] = acc;
    }

    state[i00] = o[0];
    state[i01] = o[1];
    state[i10] = o[2];
    state[i11] = o[3];
}

} // extern "C"
