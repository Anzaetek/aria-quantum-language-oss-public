// Diagonal single-qubit gate kernel — CUDA port of
// shaders/apply_diagonal.metal in the Metal backend.
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
// Statevector layout: interleaved real2 (re, im) — one real2 per
// complex amplitude. Identical to the Metal layout so the CPU
// gate-derivative builders carry over without permutation.

extern "C" {

struct DiagonalParams {
    unsigned int qubit;
    real d0_re;
    real d0_im;
    real d1_re;
    real d1_im;
};

__global__ void apply_diagonal(
    real2* state,
    DiagonalParams params,
    unsigned long long dim
) {
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= dim) { return; }
    real2 amp = state[gid];
    unsigned int bit = (unsigned int)((gid >> params.qubit) & 1ULL);
    real2 d = (bit == 0u)
        ? make_real2(params.d0_re, params.d0_im)
        : make_real2(params.d1_re, params.d1_im);
    // (ar + ai i) * (dr + di i) = (ar*dr - ai*di) + (ar*di + ai*dr) i
    state[gid] = make_real2(
        amp.x * d.x - amp.y * d.y,
        amp.x * d.y + amp.y * d.x
    );
}

} // extern "C"
