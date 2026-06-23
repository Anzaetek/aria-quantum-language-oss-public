// Diagonal 1q gate kernel — read from `src`, write to `dst`.
//
// Mirrors `apply_diagonal.metal` (in-place) but with separate input
// and output buffers. Sister to `apply_1q_into.metal`; used by the
// adjoint backward sweep's per-parameter dRz/dU1 derivative apply
// to skip the copy_into roundtrip when the derivative is itself
// diagonal.

#include <metal_stdlib>
using namespace metal;

struct DiagonalIntoParams {
    uint qubit;
    float d0_re;
    float d0_im;
    float d1_re;
    float d1_im;
};

kernel void apply_diagonal_into(
    device const float2 *src               [[buffer(0)]],
    device float2 *dst                     [[buffer(1)]],
    constant DiagonalIntoParams &params    [[buffer(2)]],
    uint gid                               [[thread_position_in_grid]]
) {
    float2 amp = src[gid];
    uint bit = (gid >> params.qubit) & 1u;
    float2 d = (bit == 0u)
        ? float2(params.d0_re, params.d0_im)
        : float2(params.d1_re, params.d1_im);
    dst[gid] = float2(
        amp.x * d.x - amp.y * d.y,
        amp.x * d.y + amp.y * d.x
    );
}
