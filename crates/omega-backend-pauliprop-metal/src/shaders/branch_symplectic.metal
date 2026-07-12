// Symplectic half of the Pauli-propagation non-Clifford branch step.
//
// One thread per input term. For term P with symplectic string (x, z) and
// generator R = (rx, rz), the branch P → cosθ·P + i sinθ·(R·P) needs three
// integer facts, all computed here:
//
//   anti[i]  = ⟨P, R⟩ mod 2  = parity( Σ_word (x & rz) ^ (z & rx) )
//   sign[i]  = (-1)^(z_R · x_P) exponent = parity( Σ_word rz & x )
//   (ox1,oz1)= R·P raw key   = (x ^ rx, z ^ rz)   (only meaningful when anti)
//
// This mirrors the CPU `branch` (omega-backend-pauliprop/src/sim.rs) and the
// CUDA kernel bit-for-bit. The coefficient math — cosθ, i sinθ, the ± sign, the
// max_freq cap, and the term merge — is done on the CPU in f64, because Apple
// GPUs have no native double. The GPU never touches a coefficient, which is
// exactly why the result is exact.
//
// Metal buffers can't hold 64-bit integers, so each packed u64 word is uploaded
// as two little-endian `uint` lanes (lo, hi) — the u64 array's bytes reinterpret
// directly as this `uint` array. Bitwise AND/XOR are per-lane, and XOR-folding
// every lane into a single 32-bit accumulator preserves the parity we need (a
// colliding pair of set bits cancels, changing the count by two), so a single
// 32-bit `popcount` at the end is correct.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint num;   // number of input terms
    uint words; // uint lanes per symplectic string (= 2 · n.div_ceil(64))
};

kernel void branch_symplectic(
    device const uint* x    [[buffer(0)]],  // num * words
    device const uint* z    [[buffer(1)]],  // num * words
    device const uint* rx   [[buffer(2)]],  // words
    device const uint* rz   [[buffer(3)]],  // words
    device uchar* anti      [[buffer(4)]],  // num
    device uchar* sgn       [[buffer(5)]],  // num
    device uint* ox1        [[buffer(6)]],  // num * words  (child key x = x ^ rx)
    device uint* oz1        [[buffer(7)]],  // num * words  (child key z = z ^ rz)
    constant Params& p      [[buffer(8)]],
    uint i                  [[thread_position_in_grid]])
{
    if (i >= p.num) {
        return;
    }
    uint words = p.words;
    uint anti_acc = 0u;
    uint sgn_acc = 0u;
    uint base = i * words;
    for (uint k = 0; k < words; ++k) {
        uint px = x[base + k];
        uint pz = z[base + k];
        uint grx = rx[k];
        uint grz = rz[k];
        anti_acc ^= (px & grz) ^ (pz & grx);
        sgn_acc ^= (grz & px);
        ox1[base + k] = px ^ grx;
        oz1[base + k] = pz ^ grz;
    }
    anti[i] = (uchar)(popcount(anti_acc) & 1u);
    sgn[i] = (uchar)(popcount(sgn_acc) & 1u);
}
