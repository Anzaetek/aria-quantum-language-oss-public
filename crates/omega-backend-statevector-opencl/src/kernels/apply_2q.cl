// Generic two-qubit gate kernel — OpenCL port of
// `crates/omega-backend-statevector-cuda/src/kernels/apply_2q.cu`
// (which is itself a port of
// `crates/omega-backend-statevector-metal/src/shaders/apply_2q.metal`).
//
// Applies a 4x4 unitary U to qubits (qa, qb), qa != qb. The matrix is
// passed in as a 32-element float buffer (16 complex entries, row-major,
// (re, im) interleaved). Convention: row index r = bit_qb * 2 + bit_qa
// — qa is low, qb is high. The host-side caller is responsible for
// arranging U into that ordering, identical to the Metal / CUDA paths.
//
// Threading: dispatch `dim/4` work-items, each handles one
// (i00, i01, i10, i11) quad.

inline float2 ucell(__global const float* u, unsigned int r, unsigned int c) {
    unsigned int k = r * 4u + c;
    return (float2)(u[2u * k], u[2u * k + 1u]);
}

__kernel void apply_2q(
    __global float2* state,
    __global const float* u,
    const unsigned int qa,
    const unsigned int qb,
    const unsigned long quads
) {
    const unsigned long tid = get_global_id(0);
    if (tid >= quads) { return; }

    const unsigned int qmin = qa < qb ? qa : qb;
    const unsigned int qmax = qa > qb ? qa : qb;

    const unsigned long low_mask = (1UL << qmin) - 1UL;
    const unsigned int mid_count = qmax - qmin - 1u;
    const unsigned long mid_mask = (1UL << mid_count) - 1UL;

    const unsigned long low = tid & low_mask;
    const unsigned long mid = ((tid >> qmin) & mid_mask) << (qmin + 1u);
    const unsigned long high = (tid >> (qmax - 1u)) << (qmax + 1u);
    const unsigned long i00 = low | mid | high;

    const unsigned long mask_a = 1UL << qa;
    const unsigned long mask_b = 1UL << qb;
    const unsigned long i01 = i00 | mask_a;
    const unsigned long i10 = i00 | mask_b;
    const unsigned long i11 = i00 | mask_a | mask_b;

    float2 v[4];
    v[0] = state[i00];
    v[1] = state[i01];
    v[2] = state[i10];
    v[3] = state[i11];

    float2 o[4];
    for (unsigned int r = 0u; r < 4u; r++) {
        float2 acc = (float2)(0.0f, 0.0f);
        for (unsigned int c = 0u; c < 4u; c++) {
            acc = acc + cmul(ucell(u, r, c), v[c]);
        }
        o[r] = acc;
    }

    state[i00] = o[0];
    state[i01] = o[1];
    state[i10] = o[2];
    state[i11] = o[3];
}
