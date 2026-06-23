// Fused-diagonal-product kernel.
//
// Computes  state[i] *= ∏_k diag_k[bit_qubit_k(i)]  for N
// independent diagonal 1q gates in a single dispatch. Each factor
// is `(qubit, d0, d1)`; the shader looks up the qubit's bit in `i`
// and multiplies by `d0` (bit==0) or `d1` (bit==1), iterating the
// product in-thread over all N factors.
//
// Saves N-1 GPU dispatches when consecutive diagonal gates can be
// fused. Equivalent (mathematically and bit-for-bit, modulo f32
// associativity) to N sequential `apply_diagonal` calls — diagonal
// gates commute, so order doesn't matter.
//
// Memory layout: state = interleaved float2 (re, im) per amplitude.
// The factor records arrive as three parallel constant arrays:
// `qubits[k]`, `d0_factors[k]`, `d1_factors[k]`. The Rust caller
// (`StateBuffer::apply_diagonal_product`) assembles them.
//
// Why a separate kernel instead of looping `apply_diagonal` on host:
// each Metal dispatch carries a fixed ~5 µs overhead. The HEA
// pattern in the QML bench has 8 Rz gates per layer on disjoint
// qubits — fusing them saves 7 dispatches per layer, which adds up
// across `forward + backward + per-param derivative` sweeps over
// 32 train pts × 10 epochs.

#include <metal_stdlib>
using namespace metal;

struct DiagonalProductParams {
    uint num_factors;
};

kernel void apply_diagonal_product(
    device float2 *state                  [[buffer(0)]],
    constant DiagonalProductParams &p     [[buffer(1)]],
    constant uint *qubits                 [[buffer(2)]],
    constant float2 *d0_factors           [[buffer(3)]],
    constant float2 *d1_factors           [[buffer(4)]],
    uint gid                              [[thread_position_in_grid]]
) {
    float2 amp = state[gid];
    for (uint k = 0u; k < p.num_factors; ++k) {
        uint bit = (gid >> qubits[k]) & 1u;
        float2 d = (bit == 0u) ? d0_factors[k] : d1_factors[k];
        // Complex multiply: amp = amp * d.
        amp = float2(amp.x * d.x - amp.y * d.y,
                     amp.x * d.y + amp.y * d.x);
    }
    state[gid] = amp;
}
