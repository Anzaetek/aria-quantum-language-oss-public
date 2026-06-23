// Stage G3 — dual-state pooled 2q gate. Applies U from
// `pool[slot]` to BOTH `state_a` (φ) and `state_b` (ν) in one
// kernel call. Used by the captured backward sweep to fuse the
// `DaggerPhi2q` + `DaggerNu2q` graph nodes for non-parameterised
// 2q ops (CNOT, CZ on the Phase 4c HEA shape) into a single graph
// node — saves one node per non-param 2q op (~13 on n=14/16-param
// HEA).
//
// Why a dual kernel rather than two launches: at 1.3 µs CPU walk
// per cuGraphLaunch node (measured), removing one node per op
// directly cuts ~30 µs/replay. Each thread reads one
// (i00, i01, i10, i11) quad from each state, applies the same 4×4
// matrix, and writes back — twice the memory traffic, but the
// kernel itself is bandwidth-bound and there's plenty of headroom
// (~975 GiB/s effective at n=20 vs ~1700 GiB/s HBM3e peak).

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

__device__ inline float2 ucell_pooled_dual(const Apply2qParams& p, unsigned int r, unsigned int c) {
    unsigned int k = r * 4u + c;
    return make_float2(p.u[2u * k], p.u[2u * k + 1u]);
}

__global__ void apply_2q_pooled_dual(
    float2* state_a,
    float2* state_b,
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

    // Pre-load the 4×4 matrix once — both states use the same U.
    float2 u00 = ucell_pooled_dual(params, 0u, 0u);
    float2 u01 = ucell_pooled_dual(params, 0u, 1u);
    float2 u02 = ucell_pooled_dual(params, 0u, 2u);
    float2 u03 = ucell_pooled_dual(params, 0u, 3u);
    float2 u10 = ucell_pooled_dual(params, 1u, 0u);
    float2 u11 = ucell_pooled_dual(params, 1u, 1u);
    float2 u12 = ucell_pooled_dual(params, 1u, 2u);
    float2 u13 = ucell_pooled_dual(params, 1u, 3u);
    float2 u20 = ucell_pooled_dual(params, 2u, 0u);
    float2 u21 = ucell_pooled_dual(params, 2u, 1u);
    float2 u22 = ucell_pooled_dual(params, 2u, 2u);
    float2 u23 = ucell_pooled_dual(params, 2u, 3u);
    float2 u30 = ucell_pooled_dual(params, 3u, 0u);
    float2 u31 = ucell_pooled_dual(params, 3u, 1u);
    float2 u32 = ucell_pooled_dual(params, 3u, 2u);
    float2 u33 = ucell_pooled_dual(params, 3u, 3u);

    // State A.
    {
        float2 v0 = state_a[i00];
        float2 v1 = state_a[i01];
        float2 v2 = state_a[i10];
        float2 v3 = state_a[i11];
        state_a[i00] = cadd(cadd(cmul(u00, v0), cmul(u01, v1)), cadd(cmul(u02, v2), cmul(u03, v3)));
        state_a[i01] = cadd(cadd(cmul(u10, v0), cmul(u11, v1)), cadd(cmul(u12, v2), cmul(u13, v3)));
        state_a[i10] = cadd(cadd(cmul(u20, v0), cmul(u21, v1)), cadd(cmul(u22, v2), cmul(u23, v3)));
        state_a[i11] = cadd(cadd(cmul(u30, v0), cmul(u31, v1)), cadd(cmul(u32, v2), cmul(u33, v3)));
    }
    // State B.
    {
        float2 v0 = state_b[i00];
        float2 v1 = state_b[i01];
        float2 v2 = state_b[i10];
        float2 v3 = state_b[i11];
        state_b[i00] = cadd(cadd(cmul(u00, v0), cmul(u01, v1)), cadd(cmul(u02, v2), cmul(u03, v3)));
        state_b[i01] = cadd(cadd(cmul(u10, v0), cmul(u11, v1)), cadd(cmul(u12, v2), cmul(u13, v3)));
        state_b[i10] = cadd(cadd(cmul(u20, v0), cmul(u21, v1)), cadd(cmul(u22, v2), cmul(u23, v3)));
        state_b[i11] = cadd(cadd(cmul(u30, v0), cmul(u31, v1)), cadd(cmul(u32, v2), cmul(u33, v3)));
    }
}

} // extern "C"
