// SPDX-License-Identifier: Apache-2.0
//! P1-a acceptance gate: the specialised CX / SWAP / CZ / CRz kernels must
//! agree with the dense `apply_2q` path they replace.
//!
//! **The bar differs by gate, and `Bar` says why for each.** CX and SWAP are
//! permutations and CZ multiplies by exactly `-1` — none of them computes
//! anything, so they are held to `to_bits()` equality, mirroring the CPU S1
//! work which kept the replaced loop verbatim and compared bits. A permutation
//! that needed a tolerance would be evidence the addressing was wrong, so a
//! loose bar there would hide the defect worth catching.
//!
//! CRz is held to one f32 ulp instead, and that is a correction rather than a
//! concession: it asserted bit-identity and passed, until an unrelated
//! `#pragma unroll` moved nvcc's FMA contraction and one amplitude shifted by
//! exactly one ulp. See `Bar::ScaledToOperand`.
//!
//! The dense matrices below are the ones `apply_cx` / `apply_cz` / `apply_swap`
//! used to build and hand to `apply_2q`, reproduced verbatim. `apply_2q` is
//! still public and still the generic path, so this compares the new kernels
//! against the real previous implementation rather than against a re-derivation
//! of it.
//!
//! SIGNED ZERO, stated rather than papered over. The dense path forms
//! `0*v0 + 0*v1 + 0*v2 + 1*v3`, and `0.0 + (-0.0) == +0.0` in IEEE-754, so it
//! canonicalises a negative zero to positive. The permutation kernel copies the
//! amplitude and preserves `-0.0`. The two therefore agree under `==` always,
//! and under `to_bits()` on every amplitude that is not exactly zero. The state
//! prepared here has **no zero amplitudes** (see `fill_dense`), which is what
//! makes the strict bit comparison both meaningful and correct; a separate case
//! covers a state that does contain zeros, under value equality.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use num_complex::Complex64;
use omega_backend_statevector_cuda::CudaStatevectorBackend;

const Z: Complex64 = Complex64::new(0.0, 0.0);
const O: Complex64 = Complex64::new(1.0, 0.0);
const M: Complex64 = Complex64::new(-1.0, 0.0);
const PI: Complex64 = Complex64::new(0.0, 1.0);
const MI: Complex64 = Complex64::new(0.0, -1.0);

#[rustfmt::skip]
const CY: [Complex64; 16] = [
    O, Z,  Z, Z,
    Z, Z,  Z, MI,
    Z, Z,  O, Z,
    Z, PI, Z, Z,
];

#[rustfmt::skip]
const CX: [Complex64; 16] = [
    O, Z, Z, Z,
    Z, Z, Z, O,
    Z, Z, O, Z,
    Z, O, Z, Z,
];

#[rustfmt::skip]
const SWAP: [Complex64; 16] = [
    O, Z, Z, Z,
    Z, Z, O, Z,
    Z, O, Z, Z,
    Z, Z, Z, O,
];

#[rustfmt::skip]
const CZ: [Complex64; 16] = [
    O, Z, Z, Z,
    Z, O, Z, Z,
    Z, Z, O, Z,
    Z, Z, Z, M,
];

/// A deterministic state with **no zero amplitudes and no repeats**, so a wrong
/// permutation cannot coincidentally land on an equal value. Values are exactly
/// representable in f32 (the device precision) so the host round-trip through
/// `Complex64` is lossless and `to_bits()` is a fair comparison.
fn fill_dense(dim: usize) -> Vec<Complex64> {
    (0..dim)
        .map(|i| {
            // Distinct dyadic rationals: exact in binary floating point.
            let re = (i as f64 + 1.0) / 1024.0;
            let im = -(i as f64 + 3.0) / 2048.0;
            Complex64::new(re, im)
        })
        .collect()
}

/// Same idea, but deliberately seeded with zeros — including a **negative**
/// zero — to exercise the one case where the two paths legitimately differ.
fn fill_with_zeros(dim: usize) -> Vec<Complex64> {
    (0..dim)
        .map(|i| match i % 4 {
            0 => Complex64::new(0.0, 0.0),
            1 => Complex64::new(-0.0, 0.0),
            2 => Complex64::new((i as f64 + 1.0) / 512.0, 0.0),
            _ => Complex64::new(0.0, -(i as f64 + 1.0) / 512.0),
        })
        .collect()
}

thread_local! {
    /// One backend per test thread, built on first use.
    ///
    /// Each `CudaStatevectorBackend::new()` NVRTC-compiles all 26 kernels, and
    /// `compare_over_pairs` is called six times — ~150 kernel compiles per run,
    /// serialised behind `RUST_TEST_THREADS=1`.
    ///
    /// **`thread_local!`, not a `OnceLock` static, and the compiler is the one
    /// insisting.** `CudaStatevectorBackend` is `!Send`/`!Sync` BY CONSTRUCTION
    /// — it holds a captured `CudaGraph` behind a raw `*mut CUgraph_st` — so
    /// `static B: OnceLock<CudaStatevectorBackend>` does not compile:
    ///
    /// ```text
    /// error[E0277]: `*mut cudarc::driver::sys::CUgraph_st` cannot be sent
    ///               between threads safely
    /// ```
    ///
    /// which is the type system refusing exactly the sharing that would have
    /// been unsound. A thread-local gives one construction per thread — one
    /// total under `RUST_TEST_THREADS=1` — and makes no cross-thread claim.
    ///
    /// It also avoids the second hazard: sharing a backend shares its
    /// `Mutex<Option<TrainStepGraph>>` cache. Nothing here touches it, but the
    /// in-crate `new_backend()` feeds graph-capture tests, which is why that
    /// one is deliberately left alone. `cx_and_swap_are_involutions_bit_for_bit`
    /// below also still builds its own, because threading a borrow through its
    /// nested loops is churn on the project's strongest gate for no measurable
    /// gain.
    static BACKEND: Option<CudaStatevectorBackend> = CudaStatevectorBackend::new()
        .map_err(|e| eprintln!("SKIP (NOT A PASS): no CUDA device ({e})"))
        .ok();
}

/// `Some(f(backend))`, or `None` when there is no device.
fn with_backend<R>(f: impl FnOnce(&CudaStatevectorBackend) -> R) -> Option<R> {
    BACKEND.with(|b| b.as_ref().map(f))
}

fn bits(v: &[Complex64]) -> Vec<(u64, u64)> {
    v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

/// How closely a specialised kernel must track the dense path.
#[derive(Clone, Copy, PartialEq)]
enum Bar {
    /// `to_bits()` equality. For gates that do not compute: CX and SWAP move
    /// amplitudes, CZ multiplies by exactly `-1`.
    Identical,
    /// Value equality, so `-0.0 == +0.0`. For states containing zeros, where
    /// the dense path's `0*v0 + ... + 1*v3` canonicalises a negative zero that
    /// a copy preserves.
    ValueEqual,
    /// Within a few f32 eps **of the input amplitude**, not of the result.
    ///
    /// SCALED BY THE OPERAND ON PURPOSE. A ulp-of-the-result bound is the wrong
    /// instrument here and was tried first: it read 1 ulp at n<=5 and 3 ulp at
    /// n=14, which looks like a bound that needs raising and is actually a
    /// bound measuring the wrong thing. `cmul`'s imaginary part is
    /// `a.x*b.y + a.y*b.x`, and for CRz's phase those two products can nearly
    /// cancel — at the failing amplitude, `0.939*b.y + 0.343*b.x` produced a
    /// result ~3x smaller than either term. Cancellation shrinks the result
    /// without shrinking the rounding error, so ulps-of-result inflates for
    /// reasons that have nothing to do with the kernel.
    ///
    /// The error of a complex multiply is bounded by a small multiple of eps
    /// times the OPERAND magnitude, and `|phase| = 1`, so `|input|` is the
    /// right scale. `4 * f32::EPSILON * |input|` is that bound with room for
    /// the two roundings inside `cmul` (one product fused into the FMA, one
    /// rounded, then the sum) differing between a standalone call and one
    /// embedded in a 4-term accumulation.
    ///
    /// REQUIRED, NOT A CONCESSION, for a gate whose phase is complex — CRz, and
    /// CP/CU1 when it lands. Both paths run the identical `cmul`, but the dense
    /// path then accumulates `0*v0 + 0*v1 + 0*v2 + phase*v3` around it, and
    /// nvcc contracts `p*q + r*s` into `fma(p, q, r*s)` by default. Which
    /// product keeps its intermediate rounding depends on the surrounding
    /// expression and on how the kernel was scheduled — so the imaginary
    /// component can differ in the last bit.
    ///
    /// This is not theoretical. The CRz arm asserted `Identical` and PASSED
    /// until `#pragma unroll` was added to the multi-slot kernel for an
    /// unrelated reason (a local-memory spill); the unroll changed the
    /// schedule, the contraction moved, and one amplitude's imaginary part
    /// shifted by exactly one f32 ulp. It had been passing by luck of codegen,
    /// on one toolkit and one arch, and would have failed on the nvcc 12.9 /
    /// H100 box reading like a kernel regression.
    ///
    /// The real part is safe under commutation and stays exact in practice, but
    /// the bound is applied to both rather than asserting a distinction that
    /// codegen does not owe us.
    ScaledToOperand,
}

/// Absolute tolerance for one complex multiply by a unit-modulus phase,
/// scaled to the OPERAND magnitude (see `Bar::ScaledToOperand`). The floor
/// keeps a zero amplitude from demanding exact equality of a rounded zero.
fn operand_tol(v: Complex64) -> f64 {
    let eps = f32::EPSILON as f64;
    4.0 * eps * v.norm().max(eps)
}

/// Run `specialised` and the dense `apply_2q` with `dense` from the same start
/// state, over every ordered qubit pair for `n` in 2..=5, and compare.
fn compare_over_pairs(
    label: &str,
    dense: &[Complex64; 16],
    specialised: impl Fn(
        &mut omega_backend_statevector_cuda::CudaState,
        u32,
        u32,
    ) -> Result<(), omega_backend_statevector_cuda::CudaError>,
    start: fn(usize) -> Vec<Complex64>,
    bar: Bar,
) {
    // The borrow stays inside the closure — see `BACKEND`.
    if with_backend(|b| compare_over_pairs_on(b, label, dense, specialised, start, bar)).is_none() {
        eprintln!("SKIP (NOT A PASS): {label} did not run — no CUDA device");
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_over_pairs_on(
    backend: &CudaStatevectorBackend,
    label: &str,
    dense: &[Complex64; 16],
    specialised: impl Fn(
        &mut omega_backend_statevector_cuda::CudaState,
        u32,
        u32,
    ) -> Result<(), omega_backend_statevector_cuda::CudaError>,
    start: fn(usize) -> Vec<Complex64>,
    bar: Bar,
) {
    // NOTE the guard at the end counts pairs. The no-device path returns BEFORE
    // it, which is correct — a machine without a GPU must skip, not fail — but
    // it means the count guard defends only against a logic error in the loops,
    // not against the commonest cause of them not running. Say so loudly here
    // so a skip is never mistaken for a pass when reading CI output.

    let mut pairs_checked = 0usize;
    for n in 2..=5u32 {
        let dim = 1usize << n;
        let init = start(dim);
        for qa in 0..n {
            for qb in 0..n {
                if qa == qb {
                    continue;
                }

                let mut fast = backend.allocate(n).expect("allocate fast");
                fast.write_state(&init).expect("write fast");
                specialised(&mut fast, qa, qb).expect("specialised kernel");
                let got = fast.read_state().expect("read fast");

                let mut slow = backend.allocate(n).expect("allocate slow");
                slow.write_state(&init).expect("write slow");
                slow.apply_2q(qa, qb, dense).expect("dense apply_2q");
                let want = slow.read_state().expect("read slow");

                match bar {
                    Bar::Identical => assert_eq!(
                        bits(&got),
                        bits(&want),
                        "{label}: n={n} qa={qa} qb={qb} — specialised kernel is not \
                         bit-identical to the dense apply_2q path it replaces"
                    ),
                    // Value equality: `-0.0 == 0.0`, the documented and only
                    // permitted divergence on states containing zeros.
                    Bar::ValueEqual => assert_eq!(
                        got, want,
                        "{label}: n={n} qa={qa} qb={qb} — specialised kernel disagrees \
                         with the dense path by more than signed zero"
                    ),
                    Bar::ScaledToOperand => {
                        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
                        for (i, ((g, w), v)) in
                            got.iter().zip(want.iter()).zip(init.iter()).enumerate()
                        {
                            let tol = operand_tol(*v);
                            let (dre, dim) = ((g.re - w.re).abs(), (g.im - w.im).abs());
                            assert!(
                                dre <= tol && dim <= tol,
                                "{label}: n={n} qa={qa} qb={qb} amp[{i}] — |dre|={dre:e} \
                                 |dim|={dim:e} vs bound {tol:e} (4 eps x |input|={:e}). \
                                 Beyond this is not FMA contraction and is a real defect.",
                                v.norm()
                            );
                        }
                    }
                }
                pairs_checked += 1;
            }
        }
    }
    // MULTI-BLOCK COVERAGE. Everything above runs at n <= 5, so `quads <= 8`
    // and `launch_dims` returns grid = 1 every single time — meaning
    // `blockIdx.x * blockDim.x` is always zero and the `tid >= quads` tail
    // guard never rejects anything. A grid-indexing bug, which is the whole
    // reason those two lines exist, would be invisible to a gate of 40 pairs
    // that all run in one block.
    //
    // n = 14 gives quads = 4096 -> grid = 16 with a 256-thread block, so both
    // the block offset and the tail guard are exercised. Kept to a few pairs
    // because the point is grid > 1, not more coverage of the addressing.
    {
        let n = 14u32;
        let init = start(1usize << n);
        for (qa, qb) in [(0u32, 1u32), (3, 9), (13, 0)] {
            let mut fast = backend.allocate(n).expect("allocate fast");
            fast.write_state(&init).expect("write fast");
            specialised(&mut fast, qa, qb).expect("specialised kernel");
            let got = fast.read_state().expect("read fast");

            let mut slow = backend.allocate(n).expect("allocate slow");
            slow.write_state(&init).expect("write slow");
            slow.apply_2q(qa, qb, dense).expect("dense apply_2q");
            let want = slow.read_state().expect("read slow");

            match bar {
                Bar::Identical => assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{label}: MULTI-BLOCK n={n} qa={qa} qb={qb} — grid>1 disagrees"
                ),
                Bar::ValueEqual => assert_eq!(
                    got, want,
                    "{label}: MULTI-BLOCK n={n} qa={qa} qb={qb} — grid>1 disagrees"
                ),
                Bar::ScaledToOperand => {
                    for (i, ((g, w), v)) in got.iter().zip(want.iter()).zip(init.iter()).enumerate()
                    {
                        let tol = operand_tol(*v);
                        let (dre, dim) = ((g.re - w.re).abs(), (g.im - w.im).abs());
                        assert!(
                            dre <= tol && dim <= tol,
                            "{label}: MULTI-BLOCK n={n} qa={qa} qb={qb} amp[{i}] — \
                             |dre|={dre:e} |dim|={dim:e} vs bound {tol:e}"
                        );
                    }
                }
            }
            pairs_checked += 1;
        }
    }

    // Guard against the loops silently not running — a vacuous pass here would
    // be indistinguishable from a green gate.
    assert_eq!(
        pairs_checked, 43,
        "{label}: expected 43 (n, qa, qb) combinations (40 single-block + 3 \
         multi-block), checked {pairs_checked}"
    );
    eprintln!("{label}: {pairs_checked} qubit pairs agree with the dense path");
}

#[test]
fn cx_permutation_is_bit_identical_to_the_dense_path() {
    compare_over_pairs(
        "cx",
        &CX,
        |s, qa, qb| s.apply_cx(qa, qb),
        fill_dense,
        Bar::Identical,
    );
}

#[test]
fn swap_permutation_is_bit_identical_to_the_dense_path() {
    compare_over_pairs(
        "swap",
        &SWAP,
        |s, qa, qb| s.apply_swap(qa, qb),
        fill_dense,
        Bar::Identical,
    );
}

/// CY — the same slot exchange as CX, with each arriving amplitude phased by
/// ∓i. Held to `Bar::Identical` because every product in `cmul` is `x·0` or
/// `x·(±1)`, exact in IEEE-754, so FMA contraction has nothing to round
/// differently — CRz's problem does not apply. The zeros case is covered
/// separately below under `ValueEqual`, like every other permutation.
#[test]
fn cy_phased_permutation_is_bit_identical_to_the_dense_path() {
    compare_over_pairs(
        "cy",
        &CY,
        |s, qa, qb| s.apply_cy(qa, qb),
        fill_dense,
        Bar::Identical,
    );
}

#[test]
fn cz_phase_is_bit_identical_to_the_dense_path() {
    compare_over_pairs(
        "cz",
        &CZ,
        |s, qa, qb| s.apply_cz(qa, qb),
        fill_dense,
        Bar::Identical,
    );
}

/// CRz exercises the multi-slot arm of the phase kernel (two slots, two
/// different phases), which CZ's single-slot case does not reach.
///
/// The angles are chosen so the phases are NOT real — a bug that dropped the
/// imaginary part, or applied `phn` and `php` to the wrong slots, would survive
/// a real-valued angle like 0 or π.
#[test]
fn crz_two_slot_phase_tracks_the_dense_path_within_f32_rounding() {
    for theta in [0.7_f64, -1.3, 2.4] {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let phn = Complex64::from_polar(1.0, -theta / 2.0);
        let php = Complex64::from_polar(1.0, theta / 2.0);
        #[rustfmt::skip]
        let dense = [
            o, z,   z, z,
            z, phn, z, z,
            z, z,   o, z,
            z, z,   z, php,
        ];
        compare_over_pairs(
            &format!("crz({theta})"),
            &dense,
            |s, qa, qb| s.apply_crz(qa, qb, theta),
            fill_dense,
            Bar::ScaledToOperand,
        );
    }
}

/// The zero-amplitude case, where the dense path's `0 + (-0.0)` canonicalisation
/// means bit-identity is NOT claimed. Value equality must still hold exactly.
#[test]
fn permutations_agree_on_states_containing_signed_zeros() {
    compare_over_pairs(
        "cx/zeros",
        &CX,
        |s, qa, qb| s.apply_cx(qa, qb),
        fill_with_zeros,
        Bar::ValueEqual,
    );
    compare_over_pairs(
        "swap/zeros",
        &SWAP,
        |s, qa, qb| s.apply_swap(qa, qb),
        fill_with_zeros,
        Bar::ValueEqual,
    );
    compare_over_pairs(
        "cz/zeros",
        &CZ,
        |s, qa, qb| s.apply_cz(qa, qb),
        fill_with_zeros,
        Bar::ValueEqual,
    );
    compare_over_pairs(
        "cy/zeros",
        &CY,
        |s, qa, qb| s.apply_cy(qa, qb),
        fill_with_zeros,
        Bar::ValueEqual,
    );
}

/// The identity that makes CX a permutation at all: applying it twice must
/// return the exact starting state, for every pair. A wrong slot choice (e.g.
/// swapping 1<->2 instead of 1<->3) would still be an involution, so this is a
/// necessary check and not a sufficient one — hence the dense comparison above.
#[test]
fn cx_and_swap_are_involutions_bit_for_bit() {
    let backend = match CudaStatevectorBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: no CUDA device ({e})");
            return;
        }
    };
    for n in 2..=5u32 {
        let init = fill_dense(1usize << n);
        for qa in 0..n {
            for qb in 0..n {
                if qa == qb {
                    continue;
                }
                for (label, twice) in [
                    ("cx", true), // apply_cx twice
                    ("swap", false),
                ] {
                    let mut st = backend.allocate(n).expect("allocate");
                    st.write_state(&init).expect("write");
                    if twice {
                        st.apply_cx(qa, qb).expect("cx");
                        st.apply_cx(qa, qb).expect("cx");
                    } else {
                        st.apply_swap(qa, qb).expect("swap");
                        st.apply_swap(qa, qb).expect("swap");
                    }
                    let got = st.read_state().expect("read");
                    assert_eq!(
                        bits(&got),
                        bits(&init),
                        "{label}² != identity at n={n} qa={qa} qb={qb}"
                    );
                }
            }
        }
    }
}
