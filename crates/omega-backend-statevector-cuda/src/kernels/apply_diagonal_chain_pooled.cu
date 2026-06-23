// Stage G5 — chain of K consecutive pooled-diagonal gates fused
// into a single kernel. Diagonal gates commute (their composition
// is also diagonal) so K diagonals applied in sequence to the
// same state can fold into ONE pass: each thread reads its
// amplitude, multiplies through K diagonal factors (one per pool
// entry in [start..start+count]), and writes back.
//
// Used by the captured forward sweep when the plan walker detects
// a run of consecutive diagonal-kind ops (e.g. the 8 RZ in the
// HEA's Layer 2). Each diagonal contributes one cmul per
// amplitude — same total work as K separate launches, but ONE
// graph node instead of K (cuGraphLaunch CPU walk dominates the
// per-replay cost on Phase 4c).

extern "C" {

struct DiagonalParams {
    unsigned int qubit;
    float d0_re;
    float d0_im;
    float d1_re;
    float d1_im;
};

__device__ inline float2 cmul_chain(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__global__ void apply_diagonal_chain_pooled(
    float2* state,
    const DiagonalParams* pool,
    unsigned int start,
    unsigned int count,
    unsigned long long dim
) {
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= dim) { return; }
    float2 amp = state[gid];
    for (unsigned int k = 0u; k < count; ++k) {
        DiagonalParams p = pool[start + k];
        unsigned int bit = (unsigned int)((gid >> p.qubit) & 1ULL);
        float2 d = (bit == 0u)
            ? make_float2(p.d0_re, p.d0_im)
            : make_float2(p.d1_re, p.d1_im);
        amp = cmul_chain(amp, d);
    }
    state[gid] = amp;
}

} // extern "C"
