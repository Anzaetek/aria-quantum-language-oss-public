// Generic two-qubit gate kernel — CUDA port of shaders/apply_2q.metal.
//
// Applies a 4x4 unitary U to qubits (qa, qb), qa != qb. Threading is
// over the dim/4 "quads" (i00, i01, i10, i11). Convention: row index
// r = bit_qb * 2 + bit_qa — qa low, qb high. Caller arranges U
// accordingly.
//
// Address arithmetic identical to the Metal kernel; see that file for
// the derivation of the low/mid/high index split.

extern "C" {

struct Apply2qParams {
    unsigned int qa;
    unsigned int qb;
    // Row-major 4x4 complex unitary; 16 entries × (re, im) = 32 floats.
    float u[32];
};

__device__ inline float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ inline float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ inline float2 ucell(const Apply2qParams& p, unsigned int r, unsigned int c) {
    unsigned int k = r * 4u + c;
    return make_float2(p.u[2u * k], p.u[2u * k + 1u]);
}

__global__ void apply_2q(
    float2* state,
    Apply2qParams params,
    unsigned long long quads
) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= quads) { return; }

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
            acc = cadd(acc, cmul(ucell(params, r, c), v[c]));
        }
        o[r] = acc;
    }

    state[i00] = o[0];
    state[i01] = o[1];
    state[i10] = o[2];
    state[i11] = o[3];
}

} // extern "C"
