// Per-qubit ⟨Z⟩ expectation, written into a device-side
// predictions slot.
//
// Computes ⟨ψ|Z_q|ψ⟩ = Σ_i (-1)^bit_q(i) · |ψ[i]|² for the qubit
// encoded by `sign_mask` (a single 1-bit at position q). Two-stage
// reduction: per-block in shared memory, then `atomicAdd` of the
// block's partial sum to `predictions[slot]`.
//
// Used by the captured training-step graph to compute the Q
// per-measurement-qubit predictions inside the graph itself, so a
// host_func callback can derive the residual gradient observable's
// coefficients (`2·(y_hat - y_label)`) without breaking the
// pipeline. Each invocation is one Z-only term at a fixed slot —
// the graph captures one launch per measurement qubit.
//
// `predictions` is f64 (matches the trainer's Vec<f64> output type
// + the inner_product_accumulate kernel's atomic accumulator
// precision). The host caller is responsible for zeroing
// `predictions[slot]` before the graph runs.

extern "C" {

__global__ void pauli_z_expectation_to_slot(
    const float2* psi,
    double* predictions,
    unsigned int sign_mask,
    unsigned int slot,
    unsigned long long dim
) {
    extern __shared__ float sdata_pz[];
    unsigned int tid = threadIdx.x;
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + tid;

    float v = 0.0f;
    if (gid < dim) {
        float2 amp = psi[gid];
        float prob = amp.x * amp.x + amp.y * amp.y;
        unsigned int parity = __popc((unsigned int)(gid & 0xFFFFFFFFULL) & sign_mask) & 1u;
        v = (parity == 0u) ? prob : -prob;
    }
    sdata_pz[tid] = v;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            sdata_pz[tid] += sdata_pz[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        atomicAdd(&predictions[slot], (double)sdata_pz[0]);
    }
}

} // extern "C"
