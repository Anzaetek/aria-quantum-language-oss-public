// OpenCL shot-sampling kernels.
//
// 1:1 port of `omega-backend-statevector-metal/src/shaders/shot_sample.metal`.
// Three-stage pipeline replaces the host-side CDF + sampler in
// `execute.rs::sample_counts` for shot-mode CLI runs. Keeps the
// statevector device-resident — only the tiny outcomes buffer crosses
// the host boundary.
//
//   1. shot_probs       — float2 state[i] → float probs[i] = |state[i]|²
//   2. shot_scan_step   — Hillis-Steele inclusive scan step; caller
//                         ping-pongs in_buf / out_buf and runs
//                         ⌈log₂(dim)⌉ passes. Produces the inclusive CDF.
//   3. shot_sample      — one work-item per shot; Philox4×32 RNG keyed
//                         by (seed_lo, seed_hi) with counter (tid,0,0,0)
//                         draws a uniform in [0, total) and binary-
//                         searches the CDF to the outcome bin.
//
// Apple OpenCL has no f64, and we want one source for every ICD; all
// per-amplitude probabilities + the CDF stay f32. The 1e-2 TVD
// acceptance gate at 10⁵ shots in `tests/shot_sampling_tvd.rs` pins
// the precision budget.
//
// Philox4×32 reference: Salmon et al., "Parallel random numbers: as
// easy as 1, 2, 3" (SC '11). OpenCL has `mul_hi` as a built-in
// (PTX `mul.hi.u32`, AMD `v_mul_hi_u32`, Apple-OpenCL software
// implementation), so the Philox round is one line cleaner than the
// Metal version's hand-rolled `((ulong)a*(ulong)b)>>32`.

// -- Philox4x32 ---------------------------------------------------------

inline void philox_round(uint4* c, uint2 k) {
    const uint hi0 = mul_hi(0xD2511F53u, c->x);
    const uint lo0 = 0xD2511F53u * c->x;
    const uint hi1 = mul_hi(0xCD9E8D57u, c->z);
    const uint lo1 = 0xCD9E8D57u * c->z;
    uint4 n;
    n.x = hi1 ^ c->y ^ k.x;
    n.y = lo1;
    n.z = hi0 ^ c->w ^ k.y;
    n.w = lo0;
    *c = n;
}

inline uint4 philox4x32(uint4 counter, uint2 key) {
    uint4 c = counter;
    uint2 k = key;
    for (uint i = 0u; i < 10u; ++i) {
        philox_round(&c, k);
        k.x += 0x9E3779B9u;
        k.y += 0xBB67AE85u;
    }
    return c;
}

// -- Stage 1: per-amplitude probabilities -------------------------------

__kernel void shot_probs(
    __global const float2* state,
    __global float*        probs
) {
    const unsigned int tid = get_global_id(0);
    const float2 a = state[tid];
    probs[tid] = a.x * a.x + a.y * a.y;
}

// -- Stage 2: Hillis-Steele inclusive scan step -------------------------
//
//   out[i] = in[i] + (i >= stride ? in[i - stride] : 0)
//
// Caller runs ⌈log₂(dim)⌉ passes with strides 1, 2, 4, …, swapping
// in_buf / out_buf between passes. After the final pass the active
// buffer holds the inclusive CDF.

__kernel void shot_scan_step(
    __global const float* in_buf,
    __global float*       out_buf,
    const unsigned int    stride,
    const unsigned int    dim
) {
    const unsigned int tid = get_global_id(0);
    if (tid >= dim) { return; }
    float v = in_buf[tid];
    if (tid >= stride) {
        v += in_buf[tid - stride];
    }
    out_buf[tid] = v;
}

// -- Stage 3: Philox-driven CDF sampling --------------------------------
//
// One work-item = one shot. tid is the Philox counter; the host packs
// the u64 seed into key = (lo32, hi32).
//
// The uniform is built from the top 24 bits of `philox.x` over 2^24.
// 24-bit f32 precision suffices for shot counts up to ~10⁷ (TVD floor
// is dominated by the f32 amplitude error before the RNG bit-depth
// matters at the 1e-2 acceptance gate).

__kernel void shot_sample(
    __global const float* cdf,
    __global unsigned int* outcomes,
    const unsigned int    seed_lo,
    const unsigned int    seed_hi,
    const unsigned int    shots,
    const unsigned int    dim
) {
    const unsigned int tid = get_global_id(0);
    if (tid >= shots) { return; }

    const uint4 counter = (uint4)(tid, 0u, 0u, 0u);
    const uint2 key     = (uint2)(seed_lo, seed_hi);
    const uint4 rnd     = philox4x32(counter, key);

    const float total = cdf[dim - 1u];
    // 24-bit uniform in [0, 1) — top 24 bits of rnd.x over 2^24.
    const float u = (float)(rnd.x >> 8u) * (1.0f / 16777216.0f);
    const float r = u * total;

    // Binary search: smallest i such that cdf[i] >= r. cdf is monotone
    // non-decreasing by construction.
    unsigned int lo = 0u;
    unsigned int hi = dim;
    while (lo < hi) {
        const unsigned int mid = (lo + hi) >> 1u;
        if (cdf[mid] < r) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    // Guard against the (extremely unlikely) lo == dim case if r is
    // exactly cdf[dim-1] modulo f32 round-off.
    if (lo >= dim) { lo = dim - 1u; }
    outcomes[tid] = lo;
}
