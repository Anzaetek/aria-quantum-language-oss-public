// Diagonal single-qubit gate kernel.
//
// A diagonal 1q gate U = diag(d0, d1) acts on the statevector by
// multiplying each amplitude state[i] by either d0 (if qubit-bit is
// 0 in i) or d1 (if 1). Embarrassingly parallel — one thread per
// amplitude.
//
// Covers: Z (d0=1, d1=-1), S (1, i), Sdg (1, -i), T (1, e^{iπ/4}),
// Tdg (1, e^{-iπ/4}), Rz(θ) (e^{-iθ/2}, e^{iθ/2}), U1(λ) (1, e^{iλ}).
// Caller is responsible for picking d0/d1.
//
// Statevector layout: interleaved float2 (re, im) — one float2 per
// complex amplitude. `state[i] *= d` is the standard complex multiply.

#include <metal_stdlib>
using namespace metal;

struct DiagonalParams {
    uint qubit;
    float d0_re;
    float d0_im;
    float d1_re;
    float d1_im;
};

kernel void apply_diagonal(
    device float2 *state                [[buffer(0)]],
    constant DiagonalParams &params     [[buffer(1)]],
    uint gid                            [[thread_position_in_grid]]
) {
    float2 amp = state[gid];
    uint bit = (gid >> params.qubit) & 1u;
    float2 d = (bit == 0u)
        ? float2(params.d0_re, params.d0_im)
        : float2(params.d1_re, params.d1_im);
    // (ar + ai i) * (dr + di i) = (ar*dr - ai*di) + (ar*di + ai*dr) i
    state[gid] = float2(
        amp.x * d.x - amp.y * d.y,
        amp.x * d.y + amp.y * d.x
    );
}
