// Permutation / phase kernels for the 2q gates that need no arithmetic.
//
// WHY THIS EXISTS. `apply_cx`, `apply_cz` and `apply_swap` (lib.rs) each build
// a dense `[Complex64; 16]` and dispatch `apply_2q`, which per quad does
// 4 loads + 4 stores and 16 complex multiplies + 12 complex adds. But:
//
//   CX   swaps two of the four amplitudes and leaves the other two alone
//   SWAP swaps a different two, same story
//   CZ   multiplies exactly one of the four by -1
//
// So the dense path spends 160 real flops and twice the memory traffic
// computing a permutation. These kernels do the actual work instead.
//
// EXACTNESS, and it differs by gate — do not read one claim onto all three.
//
//   CX / SWAP  bit-identical. A permutation moves bits; it does not compute.
//   CZ         bit-identical. `x * -1.0` is exact in IEEE-754.
//   CRz / CP   NOT bit-identical, and cannot be. Both paths run the same
//              `cmul`, but the dense path wraps it in a 4-term accumulation,
//              and nvcc's FMA contraction (`--fmad=true` by default) then fuses
//              a different product in each. The imaginary component can differ
//              in the last bits. `tests/quad_perm_bit_identity.rs` holds these
//              to `4 * f32::EPSILON * |input|` instead, and `Bar` explains why
//              a ulp-of-RESULT bound is the wrong instrument (the two products
//              can nearly cancel, which shrinks the result without shrinking
//              the error).
//
// That last row was learned the hard way: the CRz gate asserted bit-identity
// and PASSED, until `#pragma unroll` was added below for an unrelated reason
// and the contraction moved. It had been passing by luck of one toolkit and one
// arch, and would have failed on the nvcc 12.9 / H100 box looking like a kernel
// regression.
//
// A further difference for the permutation kernels is SIGNED ZERO, and it goes
// the *good* way:
// the dense path accumulates `0*v0 + 0*v1 + 0*v2 + 1*v3`, and `0.0 + (-0.0)`
// is `+0.0`, so it silently canonicalises a negative zero to positive. These
// kernels copy the amplitude, so `-0.0` survives. The two agree under `==`
// (IEEE says `-0.0 == +0.0`) and differ only under `to_bits()`, on amplitudes
// that are exactly zero. See PLAN-SV-PERF.md on the same caveat.
//
// The slot convention matches `apply_2q.cu`: within a quad, bit 0 of the slot
// index is qa and bit 1 is qb, so slot = bit_qb*2 + bit_qa, i.e. slot 0 = i00,
// 1 = i01, 2 = i10, 3 = i11.

extern "C" {

// Address of the quad's `i00` for this thread. Identical arithmetic to
// `apply_2q.cu` — deliberately, so the two kernels cannot drift on addressing.
__device__ inline unsigned long long quad_base_i00(
    unsigned long long tid,
    unsigned int qa,
    unsigned int qb
) {
    unsigned int qmin = qa < qb ? qa : qb;
    unsigned int qmax = qa > qb ? qa : qb;

    unsigned long long low_mask = (1ULL << qmin) - 1ULL;
    unsigned int mid_count = qmax - qmin - 1u;
    unsigned long long mid_mask = (1ULL << mid_count) - 1ULL;

    unsigned long long low = tid & low_mask;
    unsigned long long mid = ((tid >> qmin) & mid_mask) << (qmin + 1u);
    unsigned long long high = (tid >> (qmax - 1u)) << (qmax + 1u);
    return low | mid | high;
}

// Complex multiply — COPIED VERBATIM from `apply_2q.cu`, operand order
// included, and that is the whole point.
//
// Writing the multiply out by hand as `v.x*phase_re - v.y*phase_im` would be
// equal under exact arithmetic (multiplication commutes) and equal under
// non-contracted f32 — but NOT equal once nvcc contracts `p*q + r*s` into
// `fma(p, q, r*s)`, which is NVRTC's default (`--fmad=true`; nothing in
// `kernels.rs` disables it). Contraction fuses the FIRST product and rounds the
// second, so swapping the operands changes WHICH product keeps its intermediate
// rounding. The real part survives that (both orders fuse the same product);
// the IMAGINARY part does not, and only when the phase is complex — i.e.
// exactly CRz and CP/CU1.
//
// Matching the operand order does NOT buy bit-identity for a complex phase —
// the dense path's surrounding 4-term accumulation still moves the contraction,
// as the header records. What it buys is that the difference stays at the
// rounding of one `cmul` rather than compounding, which is what makes the
// `4 * eps * |input|` bound in the test derivable instead of fitted.
// Do not "simplify" this back into an inline expression.
__device__ inline real2 cmul(real2 a, real2 b) {
    return make_real2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Offset of a slot within its quad.
__device__ inline unsigned long long slot_offset(
    unsigned int slot,
    unsigned int qa,
    unsigned int qb
) {
    unsigned long long off = 0ULL;
    if (slot & 1u) { off |= 1ULL << qa; }
    if (slot & 2u) { off |= 1ULL << qb; }
    return off;
}

struct QuadSwapParams {
    unsigned int qa;
    unsigned int qb;
    unsigned int slot_a;
    unsigned int slot_b;
};

// Exchange two slots of every quad. 2 loads + 2 stores + zero flops.
//
//   CX(control=qa, target=qb) : slots 1 <-> 3  (the control-is-1 half)
//   SWAP(qa, qb)              : slots 1 <-> 2
__global__ void apply_quad_swap(
    real2* state,
    QuadSwapParams params,
    unsigned long long quads
) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= quads) { return; }

    unsigned long long i00 = quad_base_i00(tid, params.qa, params.qb);
    unsigned long long ia = i00 | slot_offset(params.slot_a, params.qa, params.qb);
    unsigned long long ib = i00 | slot_offset(params.slot_b, params.qa, params.qb);

    real2 a = state[ia];
    real2 b = state[ib];
    state[ia] = b;
    state[ib] = a;
}

struct QuadSwapPhaseParams {
    unsigned int qa;
    unsigned int qb;
    unsigned int slot_a;
    unsigned int slot_b;
    // Phase applied to the value ARRIVING at each slot.
    real phase_a_re; real phase_a_im;
    real phase_b_re; real phase_b_im;
};

// Exchange two slots, multiplying each incoming value by its own phase — CY.
//
//   out[slot_a] = phase_a * in[slot_b]
//   out[slot_b] = phase_b * in[slot_a]
//
// THE PHASE ATTACHES TO THE DESTINATION. Written out because CY is Hermitian
// and an involution, so getting the two phases the wrong way round still
// satisfies CY*CY = I and would only surface against an independent
// implementation. For CY(qc, qt): out[1] = -i*in[3], out[3] = +i*in[1], which
// matches the dense matrix it replaces (u[1][3] = -i, u[3][1] = +i).
//
// EXACT. With a phase of (0, -+1), `cmul` computes `0*v.x - (-+1)*v.y` and
// `0*v.y + (-+1)*v.x` — every product is `x*0` or `x*(+-1)`, exact in IEEE-754.
// So FMA contraction has nothing to round whichever product nvcc fuses, and
// CRz's contraction problem does not apply here. Bit-identity is achievable
// and is the bar, EXCEPT on amplitudes that are exactly zero, where the dense
// path's three `cmul((0,0), v)` accumulations canonicalise -0.0 to +0.0 and a
// copy does not — the same signed-zero split as CX/SWAP/CZ.
__global__ void apply_quad_swap_phase(
    real2* state,
    QuadSwapPhaseParams params,
    unsigned long long quads
) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= quads) { return; }

    unsigned long long i00 = quad_base_i00(tid, params.qa, params.qb);
    unsigned long long ia = i00 | slot_offset(params.slot_a, params.qa, params.qb);
    unsigned long long ib = i00 | slot_offset(params.slot_b, params.qa, params.qb);

    real2 a = state[ia];
    real2 b = state[ib];
    state[ia] = cmul(make_real2(params.phase_a_re, params.phase_a_im), b);
    state[ib] = cmul(make_real2(params.phase_b_re, params.phase_b_im), a);
}

struct QuadPhase1Params {
    unsigned int qa;
    unsigned int qb;
    unsigned int slot;
    real phase_re;
    real phase_im;
};

// Single-slot phase — CZ and CP/CU1. 1 load + 1 store.
//
// A SEPARATE KERNEL, not a branch inside the multi-slot one.
//
// The history is worth keeping, because the first explanation written here was
// WRONG and the right one was only found by measuring. Routing CZ through the
// multi-slot kernel cost it 22% (37.0 -> 48.0 ms at 26 qubits). A uniform
// `if (count == 1)` fast path did NOT recover it, which ruled out the branch.
// This comment then claimed "nvcc allocates registers for the worst path" —
// asserted, never measured. `cuFuncGetAttribute` says otherwise:
//
//     apply_quad_phase1   18 regs    0 B local
//     apply_quad_phase    35 regs   64 B local   <- BEFORE the unroll fix
//
// The cost was a LOCAL MEMORY SPILL, not register pressure: the multi-slot
// loop's runtime bound made `params.slots[k]` a dynamic index into a by-value
// parameter struct, so nvcc pushed the whole 64-byte struct to the local frame.
// Fixing that (constant trip count + `#pragma unroll`, see below) brought it to
// 18 regs / 0 B local — identical to this kernel — and made CRz 16.7% faster.
//
// SO IS THIS SPLIT STILL NEEDED? MEASURED 2026-08-17: no, not on this
// toolchain. Bypassing the single-entry redirect in `imp.rs` so CZ runs through
// the multi-slot kernel gives 39.345 ms against 39.332 ms through this one at
// 26 qubits — identical within noise (p = 0.79). Once the spill was gone the
// two kernels have the same resource profile (18 regs / 0 B local each), so
// that is the expected result rather than a surprise.
//
// KEPT ANYWAY, deliberately. Register allocation and local-memory spilling are
// properties of the COMPILER VERSION and the TARGET ARCH, and that measurement
// is one box: GB10, sm_121, CUDA 13.0.88. Deleting a kernel that is measured
// harmless here, on the strength of evidence that cannot speak for nvcc 12.x on
// sm_90, is the "invisible here, fatal on the other box" move this project
// keeps writing down. The cost of keeping it is one small kernel; the cost of
// being wrong is a silent perf regression on hardware nobody here can test.
//
// Re-run the comparison on an amd64 box and this can go.
// `tests/kernel_resource_report.rs` pins local == 0 so the spill cannot return.
__global__ void apply_quad_phase1(
    real2* state,
    QuadPhase1Params params,
    unsigned long long quads
) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= quads) { return; }

    unsigned long long i =
        quad_base_i00(tid, params.qa, params.qb)
        | slot_offset(params.slot, params.qa, params.qb);

    // `cmul(phase, v)` — phase FIRST, matching `apply_2q.cu`'s `cmul(ucell, v)`.
    real2 v = state[i];
    state[i] = cmul(make_real2(params.phase_re, params.phase_im), v);
}

struct QuadPhaseParams {
    unsigned int qa;
    unsigned int qb;
    // How many of the four slots are NOT identity. Uniform across the grid, so
    // the loop below is a predictable branch, not divergence.
    unsigned int count;
    unsigned int slots[4];
    real phase_re[4];
    real phase_im[4];
};

// Multiply the non-identity slots of every quad by their phases.
//
// Every 2q DIAGONAL is this kernel, and the point is to touch only the slots
// that actually change:
//
//   CZ(qa, qb)       count 1: slot 3 x -1                    1 load + 1 store
//   CP/CU1(qa,qb,l)  count 1: slot 3 x e^{il}                1 load + 1 store
//   CRz(qc,qt,theta) count 2: slot 1 x e^{-i.th/2},
//                             slot 3 x e^{+i.th/2}           2 loads + 2 stores
//
// against `apply_2q`'s unconditional 4 + 4 and 160 real flops. Since these
// kernels are memory-bound (the dense path already achieves ~73% of this
// device's bandwidth), the win is the bytes NOT moved for the identity slots —
// which is why this takes a slot list rather than four phases with 1+0i in the
// untouched positions. Four phases would have been simpler and worth nothing.
__global__ void apply_quad_phase(
    real2* state,
    QuadPhaseParams params,
    unsigned long long quads
) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= quads) { return; }

    unsigned long long i00 = quad_base_i00(tid, params.qa, params.qb);

    // CONSTANT trip count + `#pragma unroll`, with the runtime `count` as an
    // early exit. This is not cosmetic — it is the difference between the
    // parameter arrays living in registers and living in local memory.
    //
    // Written first as `for (k = 0; k < params.count; k++)`, the bound is a
    // runtime value, so nvcc cannot unroll and `params.slots[k]` /
    // `params.phase_*[k]` become dynamic indexes into a by-value parameter
    // struct — which nvcc resolves by spilling the whole struct to the local
    // frame. MEASURED via `cuFuncGetAttribute`, not guessed:
    //
    //     apply_quad_phase1   18 regs    0 B local
    //     apply_quad_phase    35 regs   64 B local   <- the entire struct
    //
    // 64 B is exactly `slots[4] + phase_re[4] + phase_im[4]` plus the scalars,
    // padded. Every iteration then re-reads its operands through the local
    // frame. With a constant bound the indexes fold to compile-time constants
    // and the spill disappears.
    //
    // `tests/kernel_resource_report.rs` asserts local == 0 for these kernels so
    // the spill cannot come back unnoticed.
    #pragma unroll
    for (unsigned int k = 0u; k < 4u; k++) {
        if (k >= params.count) { break; }
        unsigned long long i =
            i00 | slot_offset(params.slots[k], params.qa, params.qb);
        // phase FIRST — see the note on `cmul` above.
        real2 v = state[i];
        state[i] = cmul(make_real2(params.phase_re[k], params.phase_im[k]), v);
    }
}

} // extern "C"

extern "C" {

struct OctetSwapParams {
    unsigned int q0;
    unsigned int q1;
    unsigned int q2;
    unsigned int slot_a;
    unsigned int slot_b;
};

// Exchange two slots of every OCTET — the exact CCX / CSwap path.
//
// CCX and CSwap are permutations of three qubits, but the backend applies them
// as a 15-gate Nielsen-Chuang decomposition (CSwap = CX.CCX.CX inherits it),
// each gate rounding in f32, with T/Tdg carrying an irrational e^{+-i.pi/4}.
// This does the permutation: 2 loads + 2 stores per octet, ZERO flops, exact.
//
// SLOT CONVENTION, extending `apply_2q.cu`'s quad to three qubits:
//
//     slot = bit_q0 + 2*bit_q1 + 4*bit_q2
//
//   CCX(c1, c2, t)  -> q0=c1, q1=c2, q2=t : slots 3 <-> 7
//       both controls set (bits q0,q1 = 1) and the target flips, i.e.
//       (1,1,0) = 1+2+0 = 3  <->  (1,1,1) = 1+2+4 = 7.
//
//   CSwap(c, t1, t2) -> q0=c, q1=t1, q2=t2 : slots 3 <-> 5
//       control set and the two targets differ, i.e.
//       (1,1,0) = 3  <->  (1,0,1) = 5.
//
// Every other slot is a fixed point, so the decomposition was reading and
// writing the whole state fifteen times to move a quarter of it once.
__global__ void apply_octet_swap(
    real2* state,
    OctetSwapParams params,
    unsigned long long octets
) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= octets) { return; }

    // Sort the three qubit indices so the bit-deposit below is monotone.
    unsigned int a = params.q0, b = params.q1, c = params.q2;
    unsigned int lo = a < b ? (a < c ? a : c) : (b < c ? b : c);
    unsigned int hi = a > b ? (a > c ? a : c) : (b > c ? b : c);
    unsigned int mid = a + b + c - lo - hi;

    // Deposit `tid`'s bits around the three holes at lo < mid < hi.
    unsigned long long low  = tid & ((1ULL << lo) - 1ULL);
    unsigned long long midb = ((tid >> lo) & ((1ULL << (mid - lo - 1)) - 1ULL)) << (lo + 1);
    unsigned long long high = ((tid >> (mid - 1)) & ((1ULL << (hi - mid - 1)) - 1ULL)) << (mid + 1);
    unsigned long long top  = (tid >> (hi - 2)) << (hi + 1);
    unsigned long long base = low | midb | high | top;

    unsigned long long off_a =
        (((params.slot_a >> 0) & 1u) ? (1ULL << params.q0) : 0ULL)
      | (((params.slot_a >> 1) & 1u) ? (1ULL << params.q1) : 0ULL)
      | (((params.slot_a >> 2) & 1u) ? (1ULL << params.q2) : 0ULL);
    unsigned long long off_b =
        (((params.slot_b >> 0) & 1u) ? (1ULL << params.q0) : 0ULL)
      | (((params.slot_b >> 1) & 1u) ? (1ULL << params.q1) : 0ULL)
      | (((params.slot_b >> 2) & 1u) ? (1ULL << params.q2) : 0ULL);

    unsigned long long ia = base | off_a;
    unsigned long long ib = base | off_b;

    real2 va = state[ia];
    real2 vb = state[ib];
    state[ia] = vb;
    state[ib] = va;
}

} // extern "C"
