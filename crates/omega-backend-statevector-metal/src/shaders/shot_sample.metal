// Metal shot-sampling kernels.
//
// Three-stage pipeline replaces the host-side CDF + sampler in
// `lib.rs::sample_counts` for shot-mode CLI runs. Keeps the statevector
// device-resident — only the tiny outcomes buffer crosses the host
// boundary.
//
//   1. shot_probs       — float2 state[i] → float probs[i] = |state[i]|²
//   2. shot_scan_step   — Hillis-Steele inclusive scan step, ping-pong
//                         buffers, run ⌈log₂(dim)⌉ times to turn the
//                         probs array into a CDF.
//   3. shot_sample      — one thread per shot; Philox4×32 RNG keyed by
//                         (seed_lo, seed_hi) with counter = (tid, 0, 0, 0)
//                         draws a uniform in [0, total) and binary
//                         searches the CDF to the outcome bin.
//
// Apple has no f64 in shaders, so all per-amplitude probabilities and
// the CDF are f32. The 1e-5 TVD acceptance gate at shots ≥ 10⁴ in
// `tests/shot_sampling_tvd.rs` pins the precision budget.
//
// Philox4×32 reference: Salmon et al., "Parallel random numbers: as easy
// as 1, 2, 3" (SC '11). Key constants `0xD2511F53`, `0xCD9E8D57` and the
// Weyl increments `0x9E3779B9`, `0xBB67AE85` are the published values.
// 10 rounds (vs the cheaper 7) keeps the statistical-quality margin —
// the per-shot cost is negligible (one Philox call per shot, not per
// amplitude).

#include <metal_stdlib>
using namespace metal;

// -- Philox4x32 ---------------------------------------------------------

inline uint mulhi32(uint a, uint b) {
    return uint(((ulong)a * (ulong)b) >> 32);
}

inline void philox_round(thread uint4 &c, thread uint2 k) {
    uint hi0 = mulhi32(0xD2511F53u, c.x);
    uint lo0 = 0xD2511F53u * c.x;
    uint hi1 = mulhi32(0xCD9E8D57u, c.z);
    uint lo1 = 0xCD9E8D57u * c.z;
    uint4 n;
    n.x = hi1 ^ c.y ^ k.x;
    n.y = lo1;
    n.z = hi0 ^ c.w ^ k.y;
    n.w = lo0;
    c = n;
}

inline uint4 philox4x32(uint4 counter, uint2 key) {
    uint4 c = counter;
    uint2 k = key;
    for (uint i = 0u; i < 10u; ++i) {
        philox_round(c, k);
        k.x += 0x9E3779B9u;
        k.y += 0xBB67AE85u;
    }
    return c;
}

// -- Stage 1: per-amplitude probabilities -------------------------------

kernel void shot_probs(
    device const float2 *state [[buffer(0)]],
    device float        *probs [[buffer(1)]],
    uint                 tid   [[thread_position_in_grid]]
) {
    float2 a = state[tid];
    probs[tid] = a.x * a.x + a.y * a.y;
}

// -- Stage 2: Hillis-Steele inclusive scan step -------------------------
//
// out[i] = in[i] + (i >= stride ? in[i - stride] : 0)
//
// Caller runs ⌈log₂(dim)⌉ passes with strides 1, 2, 4, …, swapping
// `in`/`out` between passes. After the final pass the active buffer
// holds the inclusive CDF.

struct ScanParams {
    uint stride;
    uint dim;
};

kernel void shot_scan_step(
    device const float *in_buf [[buffer(0)]],
    device float       *out_buf [[buffer(1)]],
    constant ScanParams &p      [[buffer(2)]],
    uint                 tid    [[thread_position_in_grid]]
) {
    if (tid >= p.dim) return;
    float v = in_buf[tid];
    if (tid >= p.stride) {
        v += in_buf[tid - p.stride];
    }
    out_buf[tid] = v;
}

// -- Stage 3: Philox-driven CDF sampling --------------------------------
//
// Each thread = one shot. tid is the Philox counter; the host packs the
// u64 seed into key = (lo32, hi32).
//
// The uniform is built from the top 24 bits of `philox.x` over 2^24.
// 24-bit f32 precision suffices for shot counts up to ~10⁷ (TVD floor
// is dominated by the f32 amplitude error before the RNG bit-depth
// matters at the 1e-5 acceptance gate).

struct SampleParams {
    uint seed_lo;
    uint seed_hi;
    uint shots;
    uint dim;
};

kernel void shot_sample(
    device const float *cdf       [[buffer(0)]],
    device uint        *outcomes  [[buffer(1)]],
    constant SampleParams &p      [[buffer(2)]],
    uint                  tid     [[thread_position_in_grid]]
) {
    if (tid >= p.shots) return;

    uint4 counter = uint4(tid, 0u, 0u, 0u);
    uint2 key = uint2(p.seed_lo, p.seed_hi);
    uint4 rnd = philox4x32(counter, key);

    float total = cdf[p.dim - 1u];
    // 24-bit uniform in [0, 1) — top 24 bits of rnd.x over 2^24.
    float u = float(rnd.x >> 8u) * (1.0f / 16777216.0f);
    float r = u * total;

    // Binary search: smallest i such that cdf[i] >= r. cdf is
    // monotone non-decreasing by construction.
    uint lo = 0u;
    uint hi = p.dim;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        if (cdf[mid] < r) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    // Guard against the (extremely unlikely) lo == p.dim case if
    // r is exactly cdf[dim-1] modulo f32 round-off.
    if (lo >= p.dim) lo = p.dim - 1u;
    outcomes[tid] = lo;
}
