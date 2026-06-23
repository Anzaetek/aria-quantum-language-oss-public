// Diagonal Pauli-sum observable kernel.
//
// Computes ν = O · ψ for a diagonal observable
//
//     O = Σ_k c_k · Z^{(s_k)}                   (s_k ⊆ qubits)
//
// where each term is `c_k` times a tensor product of Z's on the
// qubits indicated by the term's sign-mask `s_k`. Identity terms are
// represented by `s_k = 0`. No X / Y components — those need the
// general apply_1q / apply_2q dispatch path.
//
// Per-amplitude: ν[i] = ψ[i] · Σ_k c_k · (-1)^popcount(i & s_k).
//
// Memory layout: state = interleaved float2 (re, im); one float2 per
// complex amplitude. Inputs: psi (read), nu (write); both same size
// 2 · 2^n · sizeof(float). The two pointers may alias (in-place
// update) since each thread reads psi[gid] exactly once before
// writing nu[gid].
//
// Why specialise: the generic per-Pauli expectation kernel can't be
// reused as-is for `O|ψ⟩` — pauli_expectation outputs a single scalar
// reduction, while we need the full state vector ν. The
// QmlTrainer's gradient observable (a sum of per-qubit Z terms with
// real coefficients) is the canonical caller and saves the host
// roundtrip the previous adjoint init did.
//
// `num_terms` is bounded by the number of measurement qubits in the
// observable; in practice ≤ 32 for QML use cases. The per-thread
// loop is small enough that branch divergence on `popcount` doesn't
// hurt — Apple GPUs popcount in-thread cheaply.

#include <metal_stdlib>
using namespace metal;

struct DiagonalPauliSumParams {
    uint num_terms;
};

kernel void apply_diagonal_pauli_sum(
    device const float2 *psi              [[buffer(0)]],
    device float2 *nu                     [[buffer(1)]],
    constant DiagonalPauliSumParams &p    [[buffer(2)]],
    constant uint *sign_masks             [[buffer(3)]],
    constant float *coeffs                [[buffer(4)]],
    uint gid                              [[thread_position_in_grid]]
) {
    float scale = 0.0f;
    for (uint k = 0u; k < p.num_terms; ++k) {
        // (-1)^popcount(gid & sign_masks[k]) — Z eigenvalue for the
        // basis state `gid` under term `k`. popcount on Apple GPUs
        // maps to a single hardware bitcount instruction, no LUT.
        uint parity = popcount(gid & sign_masks[k]) & 1u;
        scale += (parity == 0u) ? coeffs[k] : -coeffs[k];
    }
    float2 amp = psi[gid];
    nu[gid] = float2(amp.x * scale, amp.y * scale);
}
