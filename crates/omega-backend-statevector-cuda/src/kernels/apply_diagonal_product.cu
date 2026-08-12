// Fused-diagonal-product kernel — CUDA port of
// shaders/apply_diagonal_product.metal.
//
// Computes  state[i] *= ∏_k diag_k[bit_qubit_k(i)]  for N independent
// diagonal 1q gates in a single dispatch. Each factor is
// `(qubit, d0, d1)`; the kernel looks up the qubit's bit in `i` and
// multiplies by `d0` (bit==0) or `d1` (bit==1), iterating the product
// in-thread over all N factors.
//
// Saves N-1 GPU dispatches per fusion run. The HEA bench's eight-Rz
// layer collapses from 8 dispatches to 1.

extern "C" {

struct DiagonalProductParams {
    unsigned int num_factors;
};

__global__ void apply_diagonal_product(
    real2* state,
    DiagonalProductParams p,
    const unsigned int* qubits,
    const real2* d0_factors,
    const real2* d1_factors,
    unsigned long long dim
) {
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= dim) { return; }
    real2 amp = state[gid];
    for (unsigned int k = 0u; k < p.num_factors; ++k) {
        unsigned int bit = (unsigned int)((gid >> qubits[k]) & 1ULL);
        real2 d = (bit == 0u) ? d0_factors[k] : d1_factors[k];
        amp = make_real2(amp.x * d.x - amp.y * d.y,
                          amp.x * d.y + amp.y * d.x);
    }
    state[gid] = amp;
}

} // extern "C"
