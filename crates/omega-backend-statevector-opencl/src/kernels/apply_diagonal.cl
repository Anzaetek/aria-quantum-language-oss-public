// Diagonal single-qubit gate kernel — OpenCL port of
// `omega-backend-statevector-metal/src/shaders/apply_diagonal.metal`.
//
// A diagonal 1q gate U = diag(d0, d1) acts on the statevector by
// multiplying each amplitude state[i] by either d0 (if qubit-bit is
// 0 in i) or d1 (if 1). Embarrassingly parallel — one work-item per
// amplitude.
//
// Covers: Z (d0=1, d1=-1), S (1, i), Sdg (1, -i), T (1, e^{iπ/4}),
// Tdg (1, e^{-iπ/4}), Rz(θ) (e^{-iθ/2}, e^{iθ/2}), U1(λ) (1, e^{iλ}).
// Caller picks d0 / d1.
//
// Statevector layout matches `apply_1q`: interleaved float2 (re, im).
// Half the per-amplitude memory traffic vs `apply_1q` for the diagonal
// case (one complex read+write vs two of each) and skips the gate-
// matrix matvec.

__kernel void apply_diagonal(
    __global float2* state,
    const unsigned int qubit,
    const float d0_re, const float d0_im,
    const float d1_re, const float d1_im,
    const unsigned long dim
) {
    const unsigned long tid = get_global_id(0);
    if (tid >= dim) { return; }
    const float2 amp = state[tid];
    const unsigned int bit = (unsigned int)((tid >> qubit) & 1UL);
    const float2 d = (bit == 0u)
        ? (float2)(d0_re, d0_im)
        : (float2)(d1_re, d1_im);
    // (ar + ai i)(dr + di i) = (ar*dr - ai*di) + (ar*di + ai*dr) i
    state[tid] = (float2)(
        amp.x * d.x - amp.y * d.y,
        amp.x * d.y + amp.y * d.x
    );
}
