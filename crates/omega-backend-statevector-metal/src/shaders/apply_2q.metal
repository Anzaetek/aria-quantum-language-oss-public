// Generic two-qubit gate kernel.
//
// Applies a 4x4 unitary U to qubits (qa, qb), qa != qb. Threading is
// over the dim/4 "quads" of amplitudes (i00, i01, i10, i11) that
// differ only in bits qa and qb. Convention: in U, the row/column
// index r = bit_qb * 2 + bit_qa — qa is the "low" bit, qb the "high"
// bit. (Caller is responsible for ordering U accordingly; the apply_cx
// helper builds the right matrix for CX(control=qa, target=qb).)
//
// Address arithmetic — split tid into three segments around qmin/qmax:
//   qmin = min(qa, qb), qmax = max(qa, qb)
//   low  = tid & ((1 << qmin) - 1)              // bits below qmin
//   mid  = ((tid >> qmin) & mid_mask) << (qmin+1)
//          where mid_mask = (1 << (qmax-qmin-1)) - 1
//   high = (tid >> (qmax-1)) << (qmax+1)        // bits above qmax
//   i00 = low | mid | high
// then i01 = i00 | (1<<qa), i10 = i00 | (1<<qb), i11 = i00 | both.
//
// Adjacent qubits (qmax = qmin + 1) → mid_mask = 0, so the mid term
// vanishes; the formulas still hold without a special case.

#include <metal_stdlib>
using namespace metal;

struct Apply2qParams {
    uint qa;
    uint qb;
    // Row-major 4x4 complex unitary; 16 entries × (re, im) = 32 floats.
    float u[32];
};

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline float2 ucell(constant Apply2qParams &p, uint r, uint c) {
    uint k = r * 4u + c;
    return float2(p.u[2u * k], p.u[2u * k + 1u]);
}

kernel void apply_2q(
    device float2 *state                [[buffer(0)]],
    constant Apply2qParams &params      [[buffer(1)]],
    uint tid                            [[thread_position_in_grid]]
) {
    uint qa = params.qa;
    uint qb = params.qb;
    uint qmin = min(qa, qb);
    uint qmax = max(qa, qb);

    uint low_mask = (1u << qmin) - 1u;
    uint mid_count = qmax - qmin - 1u;
    uint mid_mask = (1u << mid_count) - 1u;

    uint low = tid & low_mask;
    uint mid = ((tid >> qmin) & mid_mask) << (qmin + 1u);
    uint high = (tid >> (qmax - 1u)) << (qmax + 1u);
    uint i00 = low | mid | high;

    uint mask_a = 1u << qa;
    uint mask_b = 1u << qb;
    uint i01 = i00 | mask_a;
    uint i10 = i00 | mask_b;
    uint i11 = i00 | mask_a | mask_b;

    // Input vector ordering matches row index r = bit_qb*2 + bit_qa:
    //   r=0 → bit_qb=0, bit_qa=0 → i00
    //   r=1 → bit_qb=0, bit_qa=1 → i01
    //   r=2 → bit_qb=1, bit_qa=0 → i10
    //   r=3 → bit_qb=1, bit_qa=1 → i11
    float2 v[4];
    v[0] = state[i00];
    v[1] = state[i01];
    v[2] = state[i10];
    v[3] = state[i11];

    float2 o[4];
    for (uint r = 0u; r < 4u; r++) {
        float2 acc = float2(0.0, 0.0);
        for (uint c = 0u; c < 4u; c++) {
            acc += cmul(ucell(params, r, c), v[c]);
        }
        o[r] = acc;
    }

    state[i00] = o[0];
    state[i01] = o[1];
    state[i10] = o[2];
    state[i11] = o[3];
}
