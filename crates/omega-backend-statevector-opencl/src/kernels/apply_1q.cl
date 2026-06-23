// Generic single-qubit gate kernel — OpenCL port of
// `crates/omega-backend-statevector-cuda/src/kernels/apply_1q.cu`
// (which is itself a port of
// `crates/omega-backend-statevector-metal/src/shaders/apply_1q.metal`).
//
// Applies a 2x2 unitary U = [[u00, u01], [u10, u11]] to the
// statevector on `qubit`. Threading is over amplitude pairs `(i0, i1)`
// where i0 has bit `qubit` = 0 and i1 = i0 | (1 << qubit). Dispatch
// `dim/2` work-items; each work-item reads / writes one pair.
//
// State layout is interleaved float pairs: `state[2*k + 0]` = re,
// `state[2*k + 1]` = im. Matching the Metal / CUDA layout means the
// host-side gate-matrix builders the three GPU backends share keep
// emitting the same `(u00_re, u00_im, ...)` byte pattern.

__kernel void apply_1q(
    __global float2* state,
    const unsigned int qubit,
    const float u00_re, const float u00_im,
    const float u01_re, const float u01_im,
    const float u10_re, const float u10_im,
    const float u11_re, const float u11_im,
    const unsigned long pairs
) {
    const unsigned long tid = get_global_id(0);
    if (tid >= pairs) { return; }
    const unsigned long mask = 1UL << qubit;
    const unsigned long low_bits = tid & (mask - 1UL);
    const unsigned long high = (tid >> qubit) << (qubit + 1u);
    const unsigned long i0 = low_bits | high;
    const unsigned long i1 = i0 | mask;

    const float2 a = state[i0];
    const float2 b = state[i1];
    const float2 u00 = (float2)(u00_re, u00_im);
    const float2 u01 = (float2)(u01_re, u01_im);
    const float2 u10 = (float2)(u10_re, u10_im);
    const float2 u11 = (float2)(u11_re, u11_im);

    state[i0] = cmul(u00, a) + cmul(u01, b);
    state[i1] = cmul(u10, a) + cmul(u11, b);
}
