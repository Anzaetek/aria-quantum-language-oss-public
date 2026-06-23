// MPS θ-contraction kernels — Steps 1+2 of `apply_two_site_gate` on GPU.
//
// Layout (paired f32 — Metal has no f64, no native complex):
//   * left:  float2[bl * 2 * bm]  index l*2*bm + s0*bm + m
//   * right: float2[bm * 2 * br]  index m*2*br + s1*br + r
//   * theta: float2[bl * 4 * br]  index l*4*br + s0s1*br + r
//   * gate:  float2[16]           row-major 4×4
//
// Both kernels grid by `(l, s0s1, r)` — one thread per output amplitude.
// Threads are flattened as `tid = l*4*br + s0s1*br + r` so a single
// 1-D dispatch covers the (bl, 4, br) cube.
//
// Step 1 — `mps_contract_two_site`:
//   `theta[l,s0s1,r] = Σ_m left[l,s0,m] · right[m,s1,r]`
//   where (s0, s1) = (s0s1 >> 1, s0s1 & 1). bm is the contracted dim.
//   Each thread does `bm` complex multiply-adds. SVD stays on CPU (Apple
//   GPU f32 + dispatch overhead investigation, 2026-05-12).
//
// Step 2 — `mps_apply_two_site_gate`:
//   `theta_prime[l,s0s1',r] = Σ_{s0s1} gate[s0s1', s0s1] · theta[l,s0s1,r]`
//   Each thread fetches 4 theta amplitudes and 4 gate cells, runs a
//   4-term complex inner product. Memory-bound at this scale; the split
//   from Step 1 means the contracted theta can be reused or inspected.
//
// Caller is responsible for staging `left`/`right`/`gate` from f64
// host data to f32 device buffers and reading `theta_prime` back to
// host for the CPU SVD step.

#include <metal_stdlib>
using namespace metal;

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

struct ContractParams {
    uint bl;
    uint bm;
    uint br;
};

kernel void mps_contract_two_site(
    device const float2 *left   [[buffer(0)]],
    device const float2 *right  [[buffer(1)]],
    device float2       *theta  [[buffer(2)]],
    constant ContractParams &p  [[buffer(3)]],
    uint tid                    [[thread_position_in_grid]]
) {
    uint total = p.bl * 4u * p.br;
    if (tid >= total) return;

    // Unflatten tid as (l, s0s1, r) with row-major (bl, 4, br).
    uint r    = tid % p.br;
    uint rest = tid / p.br;
    uint s0s1 = rest % 4u;
    uint l    = rest / 4u;

    uint s0 = s0s1 >> 1u;
    uint s1 = s0s1 & 1u;

    // left[l, s0, *] base index = l*2*bm + s0*bm
    uint left_base  = (l * 2u + s0) * p.bm;
    // right[*, s1, r] base index = s1*br + r; stride p.bm direction is 2*br
    // right index for m: m*2*br + s1*br + r.
    uint right_off  = s1 * p.br + r;

    float2 acc = float2(0.0f, 0.0f);
    for (uint m = 0u; m < p.bm; ++m) {
        float2 a = left[left_base + m];
        float2 b = right[m * 2u * p.br + right_off];
        acc += cmul(a, b);
    }
    theta[tid] = acc;
}

struct GateApplyParams {
    uint bl;
    uint br;
};

kernel void mps_apply_two_site_gate(
    device const float2 *theta_in  [[buffer(0)]],
    device const float2 *gate      [[buffer(1)]],
    device float2       *theta_out [[buffer(2)]],
    constant GateApplyParams &p    [[buffer(3)]],
    uint tid                       [[thread_position_in_grid]]
) {
    uint total = p.bl * 4u * p.br;
    if (tid >= total) return;

    uint r    = tid % p.br;
    uint rest = tid / p.br;
    uint s_out = rest % 4u;
    uint l    = rest / 4u;

    // Same-(l, *, r) line of theta_in has 4 entries at strides p.br.
    uint base = l * 4u * p.br + r;

    float2 acc = float2(0.0f, 0.0f);
    // gate[s_out, s_in] for s_in ∈ {0,1,2,3}; row-major index s_out*4+s_in.
    for (uint s_in = 0u; s_in < 4u; ++s_in) {
        float2 g = gate[s_out * 4u + s_in];
        float2 t = theta_in[base + s_in * p.br];
        acc += cmul(g, t);
    }
    theta_out[tid] = acc;
}
