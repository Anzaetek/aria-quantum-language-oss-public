// Stage G7 — per-op Pauli-Y gradient fusion. Replaces the
// (DaggerPhi1q + Deriv1qInnerProductAccumulate + DaggerNu1q)
// triple for parameterised RY ops with ONE kernel. Three graph
// nodes collapse into one.
//
// Math: for an RY rotation U = exp(-i θ Y/2) with generator Y,
//
//   ∂U/∂θ · U† = -i/2 · Y_q
//
// so the adjoint identity 2 Re⟨ν_k|∂U|φ_{k-1}⟩ collapses to
// Im⟨ν_k|Y_q|φ_k⟩, evaluated on the PRE-DAGGER backward states
// (φ_k, ν_k). The kernel computes that inner product first
// (block-reduces, atomic-adds chain·Im(...) into grad_dev[sym])
// and THEN applies U_k† to both states, writing back the
// post-dagger amplitudes ready for the next backward op.
//
// Per-pair contribution to Im⟨ν|Y_q|φ⟩:
//   Y_q acts as Y on qubit q's pair (i0, i1):
//     (Y_q φ)[i0] = -i · φ[i1]
//     (Y_q φ)[i1] =  i · φ[i0]
//   Σ ν*[i] · (Y_q φ)[i] over the pair = -i · A
//     where A = ν*[i0]·φ[i1] - ν*[i1]·φ[i0].
//   Im(-i · A) = -Re(A) = Re(ν*[i1]·φ[i0]) - Re(ν*[i0]·φ[i1])
//              = (ν1.x·φ0.x + ν1.y·φ0.y) - (ν0.x·φ1.x + ν0.y·φ1.y).
//
// Daggers of phi and nu use the standard 2x2 matrix-vector form
// with the U†_pool[slot] matrix.

extern "C" {

struct Apply1qParams {
    unsigned int qubit;
    real u00_re; real u00_im;
    real u01_re; real u01_im;
    real u10_re; real u10_im;
    real u11_re; real u11_im;
};

__device__ inline real2 cmul_pyt(real2 a, real2 b) {
    return make_real2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ inline real2 cadd_pyt(real2 a, real2 b) {
    return make_real2(a.x + b.x, a.y + b.y);
}

__global__ void pauli_y_accumulate_then_dagger_both_pooled(
    real2* phi,
    real2* nu,
    const Apply1qParams* dagger_pool,
    unsigned int dagger_slot,
    double* grad_dev,
    const double* chain_pool,
    const unsigned int* sym_idx_pool,
    unsigned int accum_slot,
    unsigned long long pairs
) {
    extern __shared__ real sdata_pyt[];
    unsigned int tid = threadIdx.x;
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + tid;

    Apply1qParams params = dagger_pool[dagger_slot];

    real partial = 0.0;
    bool active = (gid < pairs);
    unsigned long long i0 = 0, i1 = 0;
    real2 phi0, phi1, nu0, nu1;
    if (active) {
        unsigned long long mask = 1ULL << params.qubit;
        unsigned long long low_bits = gid & (mask - 1ULL);
        unsigned long long high = (gid >> params.qubit) << (params.qubit + 1u);
        i0 = low_bits | high;
        i1 = i0 | mask;
        phi0 = phi[i0];
        phi1 = phi[i1];
        nu0  = nu[i0];
        nu1  = nu[i1];
        // Im⟨ν|Y_q|φ⟩ per pair:
        //   = (ν1.x·φ0.x + ν1.y·φ0.y) - (ν0.x·φ1.x + ν0.y·φ1.y)
        partial = (nu1.x * phi0.x + nu1.y * phi0.y)
                - (nu0.x * phi1.x + nu0.y * phi1.y);
    }

    // Block reduce the partial sum.
    sdata_pyt[tid] = partial;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            sdata_pyt[tid] += sdata_pyt[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        double imag = (double)sdata_pyt[0];
        double chain = chain_pool[accum_slot];
        unsigned int sym = sym_idx_pool[accum_slot];
        atomicAdd(&grad_dev[sym], chain * imag);
    }

    // Apply U†_k to phi and nu at this pair, write back. Done by
    // every active thread; the inner-product reduction above is
    // independent of these writes (each thread already cached its
    // pair amplitudes in registers).
    if (active) {
        real2 u00 = make_real2(params.u00_re, params.u00_im);
        real2 u01 = make_real2(params.u01_re, params.u01_im);
        real2 u10 = make_real2(params.u10_re, params.u10_im);
        real2 u11 = make_real2(params.u11_re, params.u11_im);
        // φ' = U† φ
        phi[i0] = cadd_pyt(cmul_pyt(u00, phi0), cmul_pyt(u01, phi1));
        phi[i1] = cadd_pyt(cmul_pyt(u10, phi0), cmul_pyt(u11, phi1));
        // ν' = U† ν
        nu[i0] = cadd_pyt(cmul_pyt(u00, nu0), cmul_pyt(u01, nu1));
        nu[i1] = cadd_pyt(cmul_pyt(u10, nu0), cmul_pyt(u11, nu1));
    }
}

} // extern "C"
