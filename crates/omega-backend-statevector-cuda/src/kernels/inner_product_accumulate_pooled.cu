// Fused inner_product + gradient-accumulate kernel.
//
// Replaces the per-(op,sym) sequence:
//
//     ip_partials = inner_product_kernel(nu, temp)   (block reduction)
//     memcpy_dtoh(ip_partials)
//     stream.synchronize()
//     ip_re = sum_host(ip_partials).re
//     gradients[sym] += 2.0 * ip_re * chain          (host-side add)
//
// with a SINGLE kernel that:
//   1. Reduces ⟨nu | temp⟩ in shared memory per block (same as the
//      `inner_product` kernel).
//   2. Block 0..N-1 each atomic-add their per-block real-part
//      partial × `2 * chain` into `grad_dev[sym_idx]`.
//
// `chain_pool` and `sym_idx_pool` are device-side parallel arrays
// indexed by `slot` (one slot per (op, sym) call in the captured
// backward sweep). Both are populated host-side per training point
// from `params.resolve_derivative(...)` and the symbol id, then
// memcpy_htod'd before the graph replay.
//
// Numerical note: `grad_dev` is f64 so atomic-add round-off stays
// within the 1e-5 tolerance the `adjoint_cuda_matches_cpu_12q_hea`
// test pins (CPU is f64, GPU state is f32, so each ip.re is already
// only f32-precise; accumulating into f64 keeps drift bounded by
// sqrt(N_atomicAdds) × 1e-7 — fine for the 25 600-train-pt Phase 4c
// shape). atomicAdd ordering across blocks is non-deterministic; the
// existing 1e-5 tolerance accommodates this.

extern "C" {

__device__ inline real2 cmul_conj_local(real2 a, real2 b) {
    return make_real2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}

__global__ void inner_product_accumulate_pooled(
    const real2* nu,
    const real2* temp,
    double* grad_dev,
    const double* chain_pool,
    const unsigned int* sym_idx_pool,
    unsigned int slot,
    unsigned long long dim
) {
    extern __shared__ real2 sdata_acc[];
    unsigned int tid = threadIdx.x;
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + tid;

    real2 v = (gid < dim) ? cmul_conj_local(nu[gid], temp[gid])
                           : make_real2(0.0, 0.0);
    sdata_acc[tid] = v;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            sdata_acc[tid].x += sdata_acc[tid + stride].x;
            sdata_acc[tid].y += sdata_acc[tid + stride].y;
        }
        __syncthreads();
    }

    if (tid == 0u) {
        double chain = chain_pool[slot];
        unsigned int sym = sym_idx_pool[slot];
        // The 2.0 prefactor (from the adjoint identity
        //   ∂⟨ψ|O|ψ⟩/∂θ = 2 Re ⟨ν|(∂U/∂θ)|φ⟩) lives here so callers
        // populate `chain_pool` with the raw d θ_p / d sym chain
        // value, exactly the f64 `params.resolve_derivative` returns.
        double contrib = 2.0 * (double)sdata_acc[0].x * chain;
        atomicAdd(&grad_dev[sym], contrib);
    }
}

} // extern "C"
