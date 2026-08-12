// Fused Pauli-string expectation reduction — CUDA port of
// shaders/pauli_expectation.metal.
//
// Computes ⟨ψ|P|ψ⟩ for a Pauli string P = ⊗_q σ^{p_q}_q where
// p_q ∈ {I, X, Y, Z}. Replaces the previous "clone state → apply each
// σ → inner_product" trio with a single dispatch.
//
// Math. For basis index i:
//   j        = i XOR x_mask
//   sign(i)  = (-1)^popcount(i & sign_mask)   sign_mask = y_mask | z_mask
//   contrib  = conj(ψ[i]) * sign(i) * ψ[j]
//   total    = (-i)^{|Y|} * Σ_i contrib(i)
//
// `y_factor = (-i)^{|Y|}` — NOT i^{|Y|} — is folded by the host before
// dispatch: `contrib` uses the matrix element P[i,j], and for a Y qubit
// that is (-i)·(-1)^bit_i (Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩), with the
// (-1)^bit part already carried by `sign_mask`. Masks are u32 because
// MAX_QUBITS = 28 (dim ≤ 2^28).

extern "C" {

__device__ inline real2 cmul_conj(real2 a, real2 b) {
    return make_real2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}

__device__ inline real2 cmul(real2 a, real2 b) {
    return make_real2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__global__ void pauli_expectation(
    const real2* psi,
    real2* partials,
    unsigned int x_mask,
    unsigned int sign_mask,
    real2 y_factor,
    unsigned long long dim
) {
    extern __shared__ real2 sdata[];
    unsigned int tid = threadIdx.x;
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + tid;

    real2 contrib = make_real2(0.0, 0.0);
    if (gid < dim) {
        unsigned int i = (unsigned int)(gid & 0xFFFFFFFFULL);
        unsigned int j = i ^ x_mask;
        contrib = cmul_conj(psi[i], psi[j]);
        unsigned int sign_bits = i & sign_mask;
        if ((__popc(sign_bits) & 1u) != 0u) {
            contrib.x = -contrib.x;
            contrib.y = -contrib.y;
        }
        contrib = cmul(contrib, y_factor);
    }
    sdata[tid] = contrib;
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
