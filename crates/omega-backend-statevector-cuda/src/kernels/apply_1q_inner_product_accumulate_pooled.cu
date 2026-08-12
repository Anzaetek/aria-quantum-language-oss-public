// Stage G2 — triple fusion of (CopyPhiToTemp + Deriv1q + Accumulate)
// for the captured backward sweep's per-(op, sym) step.
//
// Replaces the three-graph-node sequence
//
//     temp ← φ                (memcpy_dtod)
//     temp ← (∂U/∂θ) · temp    (apply_1q_pooled)
//     grad_dev[sym] += 2·chain·Re⟨ν|temp⟩
//                              (inner_product_accumulate_pooled)
//
// (already collapsed to two by Stage G1's apply_1q_from_to_pooled
// when the deriv kind is 1q) with a SINGLE kernel that never
// materialises `temp`. Each pair-thread:
//
//   1. Reads φ[i0], φ[i1] and ν[i0], ν[i1].
//   2. Forms the deriv'd amplitudes
//        a' = u00·φ[i0] + u01·φ[i1]
//        b' = u10·φ[i0] + u11·φ[i1]
//      with U fetched from `deriv_1q_pool[deriv_slot]`.
//   3. Computes ⟨ν|∂U·φ⟩ contribution from this pair —
//        ν*[i0]·a' + ν*[i1]·b'   (real part only retained).
//   4. Block-reduces in shared memory.
//   5. Block-0 thread-0 atomic-adds 2·chain·partial.re into
//      grad_dev[sym], with chain + sym fetched from the
//      accumulate-keyed `chain_pool` / `sym_idx_pool` at
//      `accum_slot`.
//
// `pairs` is `1 << (num_qubits - 1)` (same as apply_1q_pooled).
// Untouched amplitudes (those NOT involved in this pair flip-bit)
// are not part of the inner product because for a 1q gate ∂U/∂θ
// only the gate's qubit matters — for the OTHER qubits the inner
// product term ν*·(∂U·φ) factors as ν*·φ (which contributes to a
// DIFFERENT param's gradient, not this one) — so we ONLY sum over
// pairs touched by the gate. This is precisely the two-amplitude-
// per-pair contraction the kernel does.
//
// Wait — that reasoning is wrong. Re-derive: ∂U/∂θ acts on the
// state space and is NOT zero on the untouched amplitudes; it is
// the IDENTITY on them (a 1q gate is U_q ⊗ I_others, and ∂U_q/∂θ
// is some matrix on qubit q tensored with I on others). So
// (∂U/∂θ · φ)[k] for k where the qubit-q bit equals 0 picks up
// contributions ONLY from amplitudes with qubit-q bit 0 — the
// kernel's "pair" structure does cover ALL amplitudes of the
// state, in one pass over pairs (each pair handles one i0 and
// its paired i1, together covering both bit-0 and bit-1 entries
// of the qubit-q index). So summing ν*[i0]·a' + ν*[i1]·b' over all
// pairs = full ⟨ν|(∂U·φ)⟩. ✓

extern "C" {

struct Apply1qParams {
    unsigned int qubit;
    real u00_re; real u00_im;
    real u01_re; real u01_im;
    real u10_re; real u10_im;
    real u11_re; real u11_im;
};

__device__ inline real2 cmul_local(real2 a, real2 b) {
    return make_real2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ inline real2 cadd_local(real2 a, real2 b) {
    return make_real2(a.x + b.x, a.y + b.y);
}

// Conjugate-of-a × b = ν*·x for inner product.
__device__ inline real2 cmul_conj_local(real2 a, real2 b) {
    return make_real2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}

__global__ void apply_1q_inner_product_accumulate_pooled(
    const real2* phi,
    const real2* nu,
    const Apply1qParams* deriv_1q_pool,
    unsigned int deriv_slot,
    double* grad_dev,
    const double* chain_pool,
    const unsigned int* sym_idx_pool,
    unsigned int accum_slot,
    unsigned long long pairs
) {
    extern __shared__ real2 sdata_g2[];
    unsigned int tid = threadIdx.x;
    unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + tid;

    real2 partial = make_real2(0.0, 0.0);
    if (gid < pairs) {
        Apply1qParams params = deriv_1q_pool[deriv_slot];
        unsigned long long mask = 1ULL << params.qubit;
        unsigned long long low_bits = gid & (mask - 1ULL);
        unsigned long long high = (gid >> params.qubit) << (params.qubit + 1u);
        unsigned long long i0 = low_bits | high;
        unsigned long long i1 = i0 | mask;

        real2 a = phi[i0];
        real2 b = phi[i1];
        real2 u00 = make_real2(params.u00_re, params.u00_im);
        real2 u01 = make_real2(params.u01_re, params.u01_im);
        real2 u10 = make_real2(params.u10_re, params.u10_im);
        real2 u11 = make_real2(params.u11_re, params.u11_im);

        real2 a_prime = cadd_local(cmul_local(u00, a), cmul_local(u01, b));
        real2 b_prime = cadd_local(cmul_local(u10, a), cmul_local(u11, b));

        real2 nv0 = nu[i0];
        real2 nv1 = nu[i1];

        real2 c0 = cmul_conj_local(nv0, a_prime);
        real2 c1 = cmul_conj_local(nv1, b_prime);
        partial = cadd_local(c0, c1);
    }

    sdata_g2[tid] = partial;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            sdata_g2[tid].x += sdata_g2[tid + stride].x;
            sdata_g2[tid].y += sdata_g2[tid + stride].y;
        }
        __syncthreads();
    }

    if (tid == 0u) {
        double chain = chain_pool[accum_slot];
        unsigned int sym = sym_idx_pool[accum_slot];
        double contrib = 2.0 * (double)sdata_g2[0].x * chain;
        atomicAdd(&grad_dev[sym], contrib);
    }
}

} // extern "C"
