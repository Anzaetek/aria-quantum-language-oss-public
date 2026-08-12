// Inner-product reduction kernel — CUDA port of
// shaders/inner_product.metal.
//
// Computes ⟨a|b⟩ = Σ_i conj(a_i) · b_i for two same-length statevectors
// stored as interleaved (re, im) f32 pairs. Each block reduces
// `blockDim.x` per-amplitude products to one real2 partial; the host
// sums the (small) partial array.
//
// `partials` length = grid_dim, one real2 per block. Caller picks
// blockDim.x = power-of-two ≤ 1024 and dispatches `dim` total threads.
//
// Why not a single atomic-add reduction: keeping the same two-stage
// shape as Metal so partial sizing logic in Rust stays identical.

extern "C" {

__device__ inline real2 cmul_conj(real2 a, real2 b) {
    // (a.x - i a.y)(b.x + i b.y)
    //   = (a.x*b.x + a.y*b.y) + i(a.x*b.y - a.y*b.x)
    return make_real2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}

__global__ void inner_product(
    const real2* a,
    const real2* b,
    real2* partials,
    unsigned long long dim
) {
    extern __shared__ real2 sdata[];
    unsigned int tid = threadIdx.x;
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + tid;

    real2 v = (gid < dim) ? cmul_conj(a[gid], b[gid]) : make_real2(0.0, 0.0);
    sdata[tid] = v;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            sdata[tid].x += sdata[tid + stride].x;
            sdata[tid].y += sdata[tid + stride].y;
        }
        __syncthreads();
    }
    if (tid == 0u) {
        partials[blockIdx.x] = sdata[0];
    }
}

} // extern "C"
