// Inner-product reduction kernel — CUDA port of
// shaders/inner_product.metal.
//
// Computes ⟨a|b⟩ = Σ_i conj(a_i) · b_i for two same-length statevectors
// stored as interleaved (re, im) f32 pairs. Each block reduces
// `blockDim.x` per-amplitude products to one float2 partial; the host
// sums the (small) partial array.
//
// `partials` length = grid_dim, one float2 per block. Caller picks
// blockDim.x = power-of-two ≤ 1024 and dispatches `dim` total threads.
//
// Why not a single atomic-add reduction: keeping the same two-stage
// shape as Metal so partial sizing logic in Rust stays identical.

extern "C" {

__device__ inline float2 cmul_conj(float2 a, float2 b) {
    // (a.x - i a.y)(b.x + i b.y)
    //   = (a.x*b.x + a.y*b.y) + i(a.x*b.y - a.y*b.x)
    return make_float2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}

__global__ void inner_product(
    const float2* a,
    const float2* b,
    float2* partials,
    unsigned long long dim
) {
    extern __shared__ float2 sdata[];
    unsigned int tid = threadIdx.x;
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + tid;

    float2 v = (gid < dim) ? cmul_conj(a[gid], b[gid]) : make_float2(0.0f, 0.0f);
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
