// Diagonal single-qubit gate kernel — pooled-param variant of
// apply_diagonal.cu. The gate's params live in a device-side pool;
// the kernel reads pool[slot] at execution time. Same math as the
// by-value variant.
//
// Why a separate kernel: by-value kernels capture their args in the
// CUDA graph at record time. To re-use a captured graph across
// training points (different gate angles per pt), the params must
// live in device memory whose CONTENTS we update via memcpy_htod
// between graph launches. The graph replays the kernel reading from
// the same device pointer, sees the new values.

extern "C" {

struct DiagonalParams {
    unsigned int qubit;
    real d0_re;
    real d0_im;
    real d1_re;
    real d1_im;
};

__global__ void apply_diagonal_pooled(
    real2* state,
    const DiagonalParams* pool,
    unsigned int slot,
    unsigned long long dim
) {
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= dim) { return; }
    DiagonalParams params = pool[slot];
    real2 amp = state[gid];
    unsigned int bit = (unsigned int)((gid >> params.qubit) & 1ULL);
    real2 d = (bit == 0u)
        ? make_real2(params.d0_re, params.d0_im)
        : make_real2(params.d1_re, params.d1_im);
    state[gid] = make_real2(
        amp.x * d.x - amp.y * d.y,
        amp.x * d.y + amp.y * d.x
    );
}

} // extern "C"
