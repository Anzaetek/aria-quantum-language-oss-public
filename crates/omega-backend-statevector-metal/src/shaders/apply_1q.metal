// Generic single-qubit gate kernel.
//
// Applies a 2x2 unitary U = [[u00, u01], [u10, u11]] to the statevector
// on `qubit`. Threading is over amplitude pairs `(i0, i1)` where i0 has
// bit `qubit` = 0 and i1 = i0 | (1<<qubit). Dispatch `dim/2` threads;
// each thread reads/writes one pair.
//
// Address arithmetic (avoids a per-thread loop):
//   mask     = 1u << qubit
//   low_bits = tid & (mask - 1)        // qubit bits below `qubit`
//   high     = (tid >> qubit) << (qubit + 1)  // qubit bits above
//   i0 = low_bits | high               // bit `qubit` = 0
//   i1 = i0 | mask                     // bit `qubit` = 1

#include <metal_stdlib>
using namespace metal;

struct Apply1qParams {
    uint qubit;
    float u00_re; float u00_im;
    float u01_re; float u01_im;
    float u10_re; float u10_im;
    float u11_re; float u11_im;
};

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

kernel void apply_1q(
    device float2 *state                [[buffer(0)]],
    constant Apply1qParams &params      [[buffer(1)]],
    uint tid                            [[thread_position_in_grid]]
) {
    uint mask = 1u << params.qubit;
    uint low_bits = tid & (mask - 1u);
    uint high = (tid >> params.qubit) << (params.qubit + 1u);
    uint i0 = low_bits | high;
    uint i1 = i0 | mask;

    float2 a = state[i0];
    float2 b = state[i1];
    float2 u00 = float2(params.u00_re, params.u00_im);
    float2 u01 = float2(params.u01_re, params.u01_im);
    float2 u10 = float2(params.u10_re, params.u10_im);
    float2 u11 = float2(params.u11_re, params.u11_im);

    state[i0] = cmul(u00, a) + cmul(u01, b);
    state[i1] = cmul(u10, a) + cmul(u11, b);
}
