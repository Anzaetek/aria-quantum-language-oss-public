// Generic 1q gate kernel — read from `src`, write to `dst`.
//
// Mirrors `apply_1q.metal` but with separate input/output buffers
// instead of in-place modification. Used by the adjoint backward
// sweep's per-parameter derivative apply: `temp = (∂U_k/∂θ_p) |φ⟩`
// without first copy_into-ing |φ⟩ into temp on host.
//
// The win: the round-16 backward sweep had |φ⟩'s daggers pending in
// phi's open batch when copy_into fired, forcing a flush so the
// host memcpy could read current phi.state. With apply_1q_into the
// derivative kernel reads phi.state directly via the
// implicit-memory-barrier-between-dispatches-in-the-same-encoder
// guarantee, and writes to dst (temp). copy_into goes away
// entirely; one less commit+wait per parameter.

#include <metal_stdlib>
using namespace metal;

struct Apply1qIntoParams {
    uint qubit;
    float u00_re;
    float u00_im;
    float u01_re;
    float u01_im;
    float u10_re;
    float u10_im;
    float u11_re;
    float u11_im;
};

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

kernel void apply_1q_into(
    device const float2 *src             [[buffer(0)]],
    device float2 *dst                   [[buffer(1)]],
    constant Apply1qIntoParams &params   [[buffer(2)]],
    uint tid                             [[thread_position_in_grid]]
) {
    uint qubit = params.qubit;
    uint mask = 1u << qubit;
    // Address arithmetic — same shape as apply_1q.
    uint low_mask = mask - 1u;
    uint low = tid & low_mask;
    uint high = (tid >> qubit) << (qubit + 1u);
    uint i0 = low | high;
    uint i1 = i0 | mask;

    float2 a0 = src[i0];
    float2 a1 = src[i1];

    float2 u00 = float2(params.u00_re, params.u00_im);
    float2 u01 = float2(params.u01_re, params.u01_im);
    float2 u10 = float2(params.u10_re, params.u10_im);
    float2 u11 = float2(params.u11_re, params.u11_im);

    dst[i0] = cmul(u00, a0) + cmul(u01, a1);
    dst[i1] = cmul(u10, a0) + cmul(u11, a1);
}
