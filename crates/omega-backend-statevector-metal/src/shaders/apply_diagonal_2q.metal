// Diagonal two-qubit gate kernel.
//
// A diagonal 2q gate U = diag(d00, d01, d10, d11) acts on the
// statevector by multiplying each amplitude state[i] by the d-entry
// indexed by the (bit_qb, bit_qa) pair: idx = bit_qb*2 + bit_qa.
// Embarrassingly parallel — one thread per amplitude, like the 1q
// apply_diagonal kernel, but with four entries indexed by the joint
// 2-qubit bit pattern.
//
// Covers: CRz (d=(1, e^{-iθ/2}, 1, e^{iθ/2}) under Metal's qb-high/
// qa-low convention), dCRz/dθ (d=(0, (-i/2)·e^{-iθ/2}, 0, (i/2)·
// e^{iθ/2})), CZ (d=(1, 1, 1, -1)), and any other 2q gate whose
// matrix is diagonal in the computational basis.
//
// Convention matches apply_2q: caller passes (qa, qb) and the d-entry
// at index r = bit_qb*2 + bit_qa is applied to amplitudes whose qa-
// and qb-bits encode r. Same row/col ordering as the dense 4x4 matrix
// the apply_2q kernel consumes.
//
// Statevector layout: interleaved float2 (re, im) — one float2 per
// complex amplitude. `state[i] *= d[idx]` is the standard complex
// multiply. Half the per-amplitude memory traffic vs apply_2q (one
// complex read + one complex write vs four reads + four writes per
// amplitude quad), and skips the 4x4 matvec arithmetic entirely.

#include <metal_stdlib>
using namespace metal;

struct Diagonal2qParams {
    uint qa;
    uint qb;
    // Four complex entries, (re, im) interleaved: d00, d01, d10, d11.
    float d[8];
};

kernel void apply_diagonal_2q(
    device float2 *state                  [[buffer(0)]],
    constant Diagonal2qParams &params     [[buffer(1)]],
    uint gid                              [[thread_position_in_grid]]
) {
    uint bit_a = (gid >> params.qa) & 1u;
    uint bit_b = (gid >> params.qb) & 1u;
    uint idx = bit_b * 2u + bit_a;
    float2 d = float2(params.d[2u * idx], params.d[2u * idx + 1u]);
    float2 amp = state[gid];
    // (ar + ai i) * (dr + di i) = (ar*dr - ai*di) + (ar*di + ai*dr) i
    state[gid] = float2(
        amp.x * d.x - amp.y * d.y,
        amp.x * d.y + amp.y * d.x
    );
}
