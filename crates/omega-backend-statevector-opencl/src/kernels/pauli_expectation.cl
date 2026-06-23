// Fused Pauli-string expectation reduction — OpenCL port of
// `omega-backend-statevector-metal/src/shaders/pauli_expectation.metal`.
//
// Computes ⟨ψ|P|ψ⟩ for a Pauli string P = ⊗_q σ^{p_q}_q where
// p_q ∈ {I, X, Y, Z}. Replaces the previous host loop in
// `execute::expectation_pauli` with a single dispatch (matching the
// Metal / CUDA siblings).
//
// Math. For basis index i:
//   j        = i XOR x_mask     (X and Y flip the bit)
//   sign(i)  = (-1)^popcount(i & sign_mask)
//              where sign_mask = y_mask | z_mask
//   contrib  = conj(ψ[i]) · sign(i) · ψ[j]
//   total    = i^{|Y|} · Σ_i contrib(i)
//
// The i^{|Y|} prefactor is folded into `y_factor` host-side: 1+0i,
// 0+1i, -1+0i, 0-1i for |Y| mod 4 = 0, 1, 2, 3. Doing it in-kernel
// keeps the reduction one-pass.
//
// All masks live in `uint` because MAX_QUBITS = 28 (dim ≤ 2^28).

inline float2 ip_cmul_conj(float2 a, float2 b) {
    return (float2)(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}

inline float2 ip_cmul(float2 a, float2 b) {
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__kernel void pauli_expectation(
    __global const float2* psi,
    __global float2*       partials,
    const unsigned int     x_mask,
    const unsigned int     sign_mask,
    const float            y_factor_re,
    const float            y_factor_im,
    __local  float2*       scratch
) {
    const unsigned int gid = get_global_id(0);
    const unsigned int tid = get_local_id(0);
    const unsigned int local_size = get_local_size(0);
    const unsigned int wg_id = get_group_id(0);

    const unsigned int i = gid;
    const unsigned int j = i ^ x_mask;
    float2 contrib = ip_cmul_conj(psi[i], psi[j]);
    const unsigned int sign_bits = i & sign_mask;
    if ((popcount(sign_bits) & 1u) != 0u) {
        contrib = -contrib;
    }
    contrib = ip_cmul(contrib, (float2)(y_factor_re, y_factor_im));

    scratch[tid] = contrib;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int stride = local_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] = scratch[tid] + scratch[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0u) {
        partials[wg_id] = scratch[0];
    }
}
