// Stage G6 — dual-state chained pooled-diagonal apply. Walks
// `pool[start..start+count]` and applies all `count` diagonals to
// BOTH `state_a` and `state_b` in one pass. Used by the captured
// backward sweep to fuse the entire RZ chain's daggers (across
// both φ and ν) into a single graph node — replaces 2K separate
// dagger applies with one launch.

extern "C" {

struct DiagonalParams {
    unsigned int qubit;
    real d0_re;
    real d0_im;
    real d1_re;
    real d1_im;
};

__device__ inline real2 cmul_chain_dual(real2 a, real2 b) {
    return make_real2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__global__ void apply_diagonal_chain_dual_pooled(
    real2* state_a,
    real2* state_b,
    const DiagonalParams* pool,
    unsigned int start,
    unsigned int count,
    unsigned long long dim
) {
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= dim) { return; }
    real2 amp_a = state_a[gid];
    real2 amp_b = state_b[gid];
    for (unsigned int k = 0u; k < count; ++k) {
        DiagonalParams p = pool[start + k];
        unsigned int bit = (unsigned int)((gid >> p.qubit) & 1ULL);
        real2 d = (bit == 0u)
            ? make_real2(p.d0_re, p.d0_im)
            : make_real2(p.d1_re, p.d1_im);
        amp_a = cmul_chain_dual(amp_a, d);
        amp_b = cmul_chain_dual(amp_b, d);
    }
    state_a[gid] = amp_a;
    state_b[gid] = amp_b;
}

} // extern "C"
