// Diagonal Pauli-sum observable kernel — CUDA port of
// shaders/apply_diagonal_pauli_sum.metal.
//
// Computes ν = O · ψ for a diagonal observable
//
//     O = Σ_k c_k · Z^{(s_k)}
//
// where each term is `c_k` times a tensor product of Z's on the qubits
// indicated by `s_k`. Identity terms have `s_k = 0`. No X/Y components —
// those need the general apply_1q / apply_2q dispatch path.
//
// Per-amplitude: ν[i] = ψ[i] · Σ_k c_k · (-1)^popcount(i & s_k).
//
// Eliminates the host roundtrip in the QML adjoint's ν setup for
// per-qubit-Z observables (the trainer's gradient observable).

extern "C" {

struct DiagonalPauliSumParams {
    unsigned int num_terms;
};

__global__ void apply_diagonal_pauli_sum(
    const float2* psi,
    float2* nu,
    DiagonalPauliSumParams p,
    const unsigned int* sign_masks,
    const float* coeffs,
    unsigned long long dim
) {
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= dim) { return; }
    float scale = 0.0f;
    unsigned int gid_lo = (unsigned int)(gid & 0xFFFFFFFFULL);
    for (unsigned int k = 0u; k < p.num_terms; ++k) {
        // Only the low 32 bits of `gid` matter — sign_masks are u32 and
        // MAX_QUBITS = 28 < 32. __popc maps to a single hardware
        // instruction on every CUDA arch we target.
        unsigned int parity = __popc(gid_lo & sign_masks[k]) & 1u;
        scale += (parity == 0u) ? coeffs[k] : -coeffs[k];
    }
    float2 amp = psi[gid];
    nu[gid] = make_float2(amp.x * scale, amp.y * scale);
}

} // extern "C"
