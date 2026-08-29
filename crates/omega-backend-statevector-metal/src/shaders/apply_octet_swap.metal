// Exact CCX / CSwap as a single amplitude PERMUTATION over 3-qubit octets.
//
// Both gates are permutations of the computational basis, so the exact
// realisation moves amplitudes and computes NOTHING:
//
//   CCX(c1, c2, t)     flips t when both controls are 1  ->  swap slots 3 <-> 7
//   CSwap(c, t1, t2)   swaps t1,t2 when c is 1           ->  swap slots 3 <-> 5
//
// with the slot convention `slot = bit_qa + 2*bit_qb + 4*bit_qc`, where
// (qa, qb, qc) are the LOGICAL roles — (c1, c2, t) for CCX, (c, t1, t2) for
// CSwap — not the sorted bit positions. The caller passes the pair to swap, so
// the convention lives at the call site where it can be read against the gate
// definition rather than inferred from kernel internals.
//
// # Why this exists
//
// The default path is the 15-op Nielsen-Chuang decomposition
// (`MetalState::apply_ccx`), and CSwap is that plus two CX, so 17 ops. Every
// one of them rounds in f32, and `T`/`Tdg` carry an irrational e^{+-i*pi/4}.
// That makes the decomposition the single largest per-gate approximation on
// this backend, for a gate whose exact form is a swap of two floats.
//
// # This kernel cannot round
//
// It performs no arithmetic — no `cmul`, no accumulation, no `fma`. So unlike
// the 2q kernels there is no FMA-contraction hazard here and nothing whose
// operand order must be preserved verbatim: bit-exactness is a property of the
// operation, not of how carefully it was written. The fragile comparison is
// against the DECOMPOSED path, which legitimately differs by 14 gates' worth of
// f32 rounding, so that test needs an operand-scaled tolerance rather than an
// absolute one.
//
// # Threading
//
// One thread per octet, `dim / 8` threads. `tid` is expanded around the three
// sorted bit positions into an index with all three qubit bits cleared, then
// the two slots are addressed off it. Adjacent qubits make the corresponding
// gap masks zero, so the general formula covers them with no special case —
// same approach as `apply_2q.metal`.

#include <metal_stdlib>
using namespace metal;

struct OctetSwapParams {
    // Logical roles, NOT sorted: qa/qb/qc as the gate defines them.
    uint qa;
    uint qb;
    uint qc;
    // The two octet slots to exchange (3 and 7 for CCX, 3 and 5 for CSwap).
    uint slot_lo;
    uint slot_hi;
};

// Index of `slot` within the octet whose base index (all three qubit bits
// clear) is `base`.
inline uint slot_index(uint base, uint qa, uint qb, uint qc, uint slot) {
    uint i = base;
    if (slot & 1u) { i |= (1u << qa); }
    if (slot & 2u) { i |= (1u << qb); }
    if (slot & 4u) { i |= (1u << qc); }
    return i;
}

kernel void apply_octet_swap(
    device float2 *state                  [[buffer(0)]],
    constant OctetSwapParams &params      [[buffer(1)]],
    uint tid                              [[thread_position_in_grid]]
) {
    uint qa = params.qa;
    uint qb = params.qb;
    uint qc = params.qc;

    // Sort the three bit positions so the address expansion is monotonic.
    uint s0 = min(qa, min(qb, qc));
    uint s2 = max(qa, max(qb, qc));
    uint s1 = qa + qb + qc - s0 - s2;

    // Expand tid around s0 < s1 < s2, leaving those three bits clear.
    uint low  = tid & ((1u << s0) - 1u);
    uint g1   = (s1 - s0 - 1u);
    uint mid1 = ((tid >> s0) & ((1u << g1) - 1u)) << (s0 + 1u);
    uint g2   = (s2 - s1 - 1u);
    uint mid2 = ((tid >> (s1 - 1u)) & ((1u << g2) - 1u)) << (s1 + 1u);
    uint high = (tid >> (s2 - 2u)) << (s2 + 1u);
    uint base = low | mid1 | mid2 | high;

    uint i_lo = slot_index(base, qa, qb, qc, params.slot_lo);
    uint i_hi = slot_index(base, qa, qb, qc, params.slot_hi);

    // A swap, not a read-modify-write of a shared cell: each thread owns a
    // distinct octet, so no two threads touch the same index.
    float2 t = state[i_lo];
    state[i_lo] = state[i_hi];
    state[i_hi] = t;
}
