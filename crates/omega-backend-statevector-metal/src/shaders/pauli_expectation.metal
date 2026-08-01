// Fused Pauli-string expectation reduction.
//
// Computes ⟨ψ|P|ψ⟩ for a Pauli string P = ⊗_q σ^{p_q}_q where
// p_q ∈ {I, X, Y, Z}. Replaces the previous "clone state → apply
// each σ → inner_product" trio with a single dispatch.
//
// Math. For basis index i:
//   j        = i XOR x_mask   (X and Y flip the bit)
//   sign(i)  = (-1)^popcount(i & sign_mask)
//              where sign_mask = y_mask | z_mask
//              (Y contributes (-1)^bit at the source qubit; Z does the same)
//   contrib  = conj(ψ[i]) * sign(i) * ψ[j]
//   total    = (-i)^{|Y|} * Σ_i contrib(i)
//
// The prefactor is (-i)^{|Y|}, NOT i^{|Y|}: `contrib` uses the matrix
// element P[i,j], and for a Y qubit that is (-i)·(-1)^bit_i (Y|0⟩ =
// i|1⟩, Y|1⟩ = -i|0⟩) — the (-1)^bit part is already in `sign_mask`,
// leaving (-i) per Y. It is folded into `y_factor` host-side: 1+0i,
// 0-1i, -1+0i, or 0+1i for |Y| mod 4 = 0, 1, 2, 3 respectively.
// Folded in here so the per-thread complex multiply costs are kept
// in-kernel rather than added as a host pass.
//
// All masks live in `uint` because MAX_QUBITS = 28 (dim ≤ 2^28) so
// 32-bit indices are sufficient and avoid the MSL 2.0 ulong-popcount
// portability question.

#include <metal_stdlib>
using namespace metal;

inline float2 cmul_conj(float2 a, float2 b) {
    return float2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

kernel void pauli_expectation(
    device const float2 *psi          [[buffer(0)]],
    device float2 *partials           [[buffer(1)]],
    constant uint &x_mask             [[buffer(2)]],
    constant uint &sign_mask          [[buffer(3)]],
    constant float2 &y_factor         [[buffer(4)]],
    threadgroup float2 *scratch       [[threadgroup(0)]],
    uint gid                          [[thread_position_in_grid]],
    uint tid                          [[thread_position_in_threadgroup]],
    uint tg_size                      [[threads_per_threadgroup]],
    uint tg_id                        [[threadgroup_position_in_grid]]
) {
    uint i = gid;
    uint j = i ^ x_mask;
    float2 contrib = cmul_conj(psi[i], psi[j]);
    // Per-index sign from the Y/Z source bits.
    uint sign_bits = i & sign_mask;
    if ((popcount(sign_bits) & 1u) != 0u) {
        contrib = -contrib;
    }
    // Apply the global (-i)^{|Y|} prefactor inside the kernel so the
    // reduction sums the final amplitude directly.
    contrib = cmul(contrib, y_factor);
    scratch[tid] = contrib;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partials[tg_id] = scratch[0];
    }
}
