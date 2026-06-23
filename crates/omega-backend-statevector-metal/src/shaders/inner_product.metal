// Inner-product reduction kernel.
//
// Computes ⟨a|b⟩ = Σ_i conj(a_i) · b_i for two same-length statevectors
// stored as interleaved (re, im) f32 pairs. Each threadgroup reduces
// `tg_size` per-amplitude products to one float2 partial; the host
// sums the (small) partial array.
//
// Why two-stage rather than a single atomic-add: Apple GPUs don't
// support atomic float adds without `metal::atomic<float>` plumbing,
// which carries its own overhead for small reductions like this.
// At n=20 the partial array has dim/tg_size = 4096 entries — summing
// those host-side is microseconds.
//
// `scratch` is a threadgroup memory scratch buffer of size
// `tg_size * sizeof(float2)`. Caller must
// `setThreadgroupMemoryLength` accordingly.

#include <metal_stdlib>
using namespace metal;

// conj(a) * b for f32 complex pairs.
inline float2 cmul_conj(float2 a, float2 b) {
    // (a.x - i a.y)(b.x + i b.y)
    //   = (a.x*b.x + a.y*b.y) + i(a.x*b.y - a.y*b.x)
    return float2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}

kernel void inner_product(
    device const float2 *a            [[buffer(0)]],
    device const float2 *b            [[buffer(1)]],
    device float2 *partials           [[buffer(2)]],
    threadgroup float2 *scratch       [[threadgroup(0)]],
    uint gid                          [[thread_position_in_grid]],
    uint tid                          [[thread_position_in_threadgroup]],
    uint tg_size                      [[threads_per_threadgroup]],
    uint tg_id                        [[threadgroup_position_in_grid]]
) {
    scratch[tid] = cmul_conj(a[gid], b[gid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Power-of-two reduction within the threadgroup.
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partials[tg_id] = scratch[0];
    }
}
