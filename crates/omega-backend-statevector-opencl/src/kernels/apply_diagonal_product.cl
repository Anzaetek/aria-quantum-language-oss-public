// Fused-diagonal-product kernel — OpenCL port of
// `omega-backend-statevector-metal/src/shaders/apply_diagonal_product.metal`.
//
// Computes  state[i] *= ∏_k diag_k[bit_qubit_k(i)]  for N independent
// diagonal 1q gates in a single dispatch. Each factor is
// `(qubit, d0, d1)`; the kernel looks up the qubit's bit in `i` and
// multiplies by `d0` (bit==0) or `d1` (bit==1), iterating the product
// in-thread over all N factors.
//
// Saves N-1 GPU dispatches when consecutive diagonal gates can be
// fused. Bit-for-bit equivalent (modulo f32 associativity) to N
// sequential `apply_diagonal` calls — diagonal gates commute on
// disjoint qubits and the same-qubit case also commutes since each
// factor is itself diagonal.
//
// State layout: interleaved float2 (re, im) per amplitude. The factor
// records arrive as three parallel arrays — `qubits[k]`,
// `d0_factors[k]`, `d1_factors[k]`. The Rust caller
// (`StateBuffer::apply_diagonal_product`) assembles them on every
// call (factors change shape per fusion run).

__kernel void apply_diagonal_product(
    __global float2* state,
    __global const unsigned int* qubits,
    __global const float2* d0_factors,
    __global const float2* d1_factors,
    const unsigned int num_factors,
    const unsigned long dim
) {
    const unsigned long tid = get_global_id(0);
    if (tid >= dim) { return; }
    float2 amp = state[tid];
    for (unsigned int k = 0u; k < num_factors; ++k) {
        const unsigned int bit = (unsigned int)((tid >> qubits[k]) & 1UL);
        const float2 d = (bit == 0u) ? d0_factors[k] : d1_factors[k];
        amp = (float2)(
            amp.x * d.x - amp.y * d.y,
            amp.x * d.y + amp.y * d.x
        );
    }
    state[tid] = amp;
}
