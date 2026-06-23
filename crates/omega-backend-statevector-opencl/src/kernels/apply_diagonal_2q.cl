// Diagonal two-qubit gate kernel — OpenCL port of
// `omega-backend-statevector-metal/src/shaders/apply_diagonal_2q.metal`.
//
// A diagonal 2q gate U = diag(d00, d01, d10, d11) acts on the
// statevector by multiplying each amplitude state[i] by the d-entry
// indexed by the (bit_qb, bit_qa) pair: idx = bit_qb*2 + bit_qa.
// Embarrassingly parallel — one work-item per amplitude, mirroring
// `apply_diagonal` but indexed by the joint 2-qubit bit pattern.
//
// Covers: CRz (d = (1, e^{-iθ/2}, 1, e^{iθ/2}) under the qa-low /
// qb-high convention), CZ (1, 1, 1, -1), dCRz/dθ, and any other 2q
// gate diagonal in the computational basis. Half the per-amplitude
// memory traffic vs `apply_2q` (one complex read+write per amplitude
// vs four-of-each per quad), and skips the 4x4 matvec entirely.

__kernel void apply_diagonal_2q(
    __global float2* state,
    const unsigned int qa,
    const unsigned int qb,
    const float d00_re, const float d00_im,
    const float d01_re, const float d01_im,
    const float d10_re, const float d10_im,
    const float d11_re, const float d11_im,
    const unsigned long dim
) {
    const unsigned long tid = get_global_id(0);
    if (tid >= dim) { return; }
    const unsigned int bit_a = (unsigned int)((tid >> qa) & 1UL);
    const unsigned int bit_b = (unsigned int)((tid >> qb) & 1UL);
    const unsigned int idx = bit_b * 2u + bit_a;

    float2 d;
    switch (idx) {
        case 0u: d = (float2)(d00_re, d00_im); break;
        case 1u: d = (float2)(d01_re, d01_im); break;
        case 2u: d = (float2)(d10_re, d10_im); break;
        default: d = (float2)(d11_re, d11_im); break;
    }
    const float2 amp = state[tid];
    state[tid] = (float2)(
        amp.x * d.x - amp.y * d.y,
        amp.x * d.y + amp.y * d.x
    );
}
