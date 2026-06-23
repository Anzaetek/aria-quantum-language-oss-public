// Inner-product reduction kernel — OpenCL port of
// `omega-backend-statevector-metal/src/shaders/inner_product.metal`.
//
// Computes ⟨a|b⟩ = Σ_i conj(a_i) · b_i for two same-length statevectors
// stored as interleaved (re, im) f32 pairs. Each work-group reduces
// `local_size` per-amplitude products to one float2 partial; the host
// sums the (small) partial array.
//
// Why two-stage rather than a single atomic-add: portable OpenCL has
// no f32 atomic add (only u32 / i32 in the base spec; the
// `cl_khr_fp32_atomic_add` extension isn't universal). At n=20 the
// partial array has `dim/local_size` ≈ 4096 entries — summing those
// host-side is microseconds.
//
// `scratch` is a __local memory buffer sized by the host via
// `kernel.set_arg(N, ocl::Local::new(local_size))`. Matches the
// Metal kernel's `threadgroup float2 *scratch` shape — same length
// in float2 entries.

inline float2 cmul_conj(float2 a, float2 b) {
    // (a.x - i a.y)(b.x + i b.y) = (a.x*b.x + a.y*b.y) + i(a.x*b.y - a.y*b.x)
    return (float2)(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}

__kernel void inner_product(
    __global const float2* a,
    __global const float2* b,
    __global float2*       partials,
    __local  float2*       scratch
) {
    const unsigned int gid = get_global_id(0);
    const unsigned int tid = get_local_id(0);
    const unsigned int local_size = get_local_size(0);
    const unsigned int wg_id = get_group_id(0);

    scratch[tid] = cmul_conj(a[gid], b[gid]);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Power-of-two reduction within the work-group. Caller is
    // responsible for picking a power-of-two `local_size` ≤
    // CL_DEVICE_MAX_WORK_GROUP_SIZE and ≤ dim so this loop terminates
    // cleanly with one slot left at the end.
    for (unsigned int stride = local_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] = scratch[tid] + scratch[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0u) {
        partials[wg_id] = scratch[0];
    }
}
