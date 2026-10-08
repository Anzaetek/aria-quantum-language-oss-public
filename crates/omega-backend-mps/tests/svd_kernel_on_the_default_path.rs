// SPDX-License-Identifier: Apache-2.0
//! **The DEFAULT macOS bond-compression kernel is Accelerate `zgesdd`, and the
//! certificate says which kernel it was.**
//!
//! `STATUS.md` §5 item 16 took two decisions at once and this file is the
//! evidence for both.
//!
//! The first is that `zgesdd` moved from `omega-backend-mps-metal` to this
//! crate and became the macOS default. Before the move it was reachable only
//! through `omega-cli`'s `metal` feature, and the measurements that justified
//! it — 4.1-7.8x on evolution at depth ≥ 12 — were taken through a harness in
//! the crate that owned it. Nothing asserted that a DEFAULT build reaches it,
//! which is the one property the promotion consists of. This crate has no
//! `metal` feature at all, so a test living here cannot be passing because of
//! one; `cargo test -p omega-backend-mps` with no `--features` is exactly the
//! build a stock `cargo build` produces.
//!
//! The second is that the certificate names its kernel. That field is only
//! worth having if it tracks the kernel, so every assertion about it here is
//! two-sided: the same circuit is run through both kernels and the field has to
//! change with them. A field hard-coded to either string passes a one-sided
//! test, and the certificate it labels would then name a kernel that did not
//! produce it.
//!
//! # Why a label is not enough on its own, and what is checked instead
//!
//! `svd_kernel` is a `&'static str` away from being decoration, so the macOS
//! arm also checks the ARITHMETIC that only Accelerate produces. Its
//! `discarded_weight` carries a deliberate backward-error allowance of
//! `(m+n)·ε·‖A‖²_F` on top of the computed residual, which at a 64-column block
//! is ~2.8e-14 of the block norm — orders of magnitude above the ~1e-16
//! disagreement between two stable kernels' tails. So on a truncating block the
//! Accelerate bound is STRICTLY above the Jacobi one, and bounded above by
//! 1e-12 of the block norm. A default path running Jacobi while claiming
//! otherwise returns the Jacobi number exactly, and fails that.

use num_complex::Complex64;
use omega_backend_mps::svd::truncated_svd_flat as jacobi_svd_flat;
use omega_backend_mps::{default_svd_kernel, MpsBackend, NoisyMpsBackend, SvdKernel};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::noise::NoiseModel;
use omega_core::params::ParameterBinding;

/// The brickwall `accelerate_svd_share` and `mps_stage_profile` measure, so the
/// θ-block shapes here are the ones those tables priced.
fn entangling_circuit(num_qubits: u32, depth: usize) -> CircuitIR {
    let mut circuit = CircuitIR::new(num_qubits, CircuitType::GateBased);
    let push = |c: &mut CircuitIR, gate: GateKind, qs: Vec<u32>, ps: Vec<f64>| {
        c.ops.push(GateOp {
            gate,
            qubits: qs.into_iter().map(Qubit).collect(),
            params: ps.into_iter().map(ParamExpr::Concrete).collect(),
            classical_bit: None,
            condition: None,
        });
    };
    for q in 0..num_qubits {
        push(&mut circuit, GateKind::H, vec![q], vec![]);
    }
    for d in 0..depth {
        let offset = d as u32 & 1;
        for q in (offset..num_qubits - 1).step_by(2) {
            push(&mut circuit, GateKind::CX, vec![q, q + 1], vec![]);
            push(&mut circuit, GateKind::Rz, vec![q + 1], vec![0.25]);
            push(&mut circuit, GateKind::CX, vec![q, q + 1], vec![]);
        }
        for q in 0..num_qubits {
            push(&mut circuit, GateKind::Rx, vec![q], vec![0.15]);
        }
    }
    circuit
}

fn single_z() -> Observable {
    Observable {
        terms: vec![(1.0, vec![(0u32, PauliOp::Z)])],
    }
}

/// A χ the circuit SATURATES, so the θ blocks are `2χ × 2χ` = 32×32 and sit
/// above `accelerate::MIN_DIM_FOR_LAPACK` (16 on `min(m, n)`). At a χ the run
/// never reaches, every split is exact, the gate hands every block to Jacobi,
/// and the two kernels would be indistinguishable for the uninteresting reason.
const CHI: usize = 16;

/// Run the brickwall through a backend and return `(value, certificate)`.
///
/// The ceiling is lifted because this circuit is MEANT to truncate — a run
/// whose certificate is zero compares two kernels on nothing.
fn run(backend: MpsBackend) -> (f64, omega_backend_mps::MpsRunStats) {
    let circuit = entangling_circuit(10, 8);
    let params = ParameterBinding::new();
    let backend = backend.with_max_discarded_weight(f64::INFINITY);
    let value = backend
        .expectation(&circuit, &params, &single_z())
        .expect("the brickwall evolves");
    (value, backend.last_run_stats())
}

// ---------------------------------------------------------------------------
// 1. The promotion itself
// ---------------------------------------------------------------------------

/// **The acceptance criterion for the rehome.** A backend nobody configured
/// runs the kernel the target has, and on macOS that is Accelerate — in a build
/// with no `metal` feature in sight.
///
/// Reddens if the macOS arm of `default_svd_kernel` selects Jacobi, which is
/// what "the promotion silently did not happen" looks like: every other test in
/// the tree stays green, the certificate stays sound, and macOS quietly gives
/// back the 4-8x.
///
/// DOES NOT CATCH: a `record_stats` that stamps `default_svd_kernel()` instead
/// of the backend's own kernel. Everything here reads the default, and a field
/// hard-coded to the default agrees with it —
/// `the_certificate_names_the_kernel_that_was_forced_onto_the_run` is what
/// closes that.
#[test]
fn a_default_backend_runs_the_target_default_kernel() {
    let expected = if cfg!(target_os = "macos") {
        "accelerate-zgesdd"
    } else {
        "jacobi"
    };
    assert_eq!(
        default_svd_kernel().as_str(),
        expected,
        "the default SVD kernel for this target"
    );
    assert_eq!(
        MpsBackend::new(CHI).svd_kernel().as_str(),
        expected,
        "MpsBackend::new must install the target default, not something else"
    );
    assert_eq!(
        NoisyMpsBackend::with_model(CHI, NoiseModel::default())
            .svd_kernel()
            .as_str(),
        expected,
        "the --noise path must not be the one that misses the promotion"
    );
    // And the default names a kernel this target can actually build. Without
    // this, a cfg naming an unbuildable kernel is a panic on the first run
    // rather than a failure here.
    assert!(
        default_svd_kernel().is_available(),
        "the default kernel must be buildable on the target that defaults to it"
    );
}

/// **The default kernel is Accelerate by its arithmetic, not by its label.**
///
/// The allowance `(m+n)·ε·‖A‖²_F` is added to every truncating split's weight
/// by `accelerate::truncated_svd_flat` and by nothing else, so the default
/// hook's bound must sit STRICTLY above the Jacobi reference's — and still
/// within 1e-12 of the block norm, which is the "sound but still the same
/// certificate" margin `accelerate.rs`'s own bound test holds.
///
/// A label is a string; this is the kernel.
///
/// DOES NOT CATCH: the size gate. This block is 64×64, well above
/// `MIN_DIM_FOR_LAPACK`, so a gate raised to 1024 — which would hand every real
/// θ block back to Jacobi and retire the whole speedup — leaves it green.
/// `accelerate.rs`'s own crossover test pins the gate's value.
#[cfg(target_os = "macos")]
#[test]
fn the_default_macos_hook_adds_the_accelerate_backward_error_allowance() {
    let f = default_svd_kernel()
        .svd_flat_fn()
        .expect("macOS can build its default kernel");
    // A decaying spectrum on a block above the size gate, so the solve is
    // LAPACK's and the truncation discards something.
    let (m, n) = (64usize, 64usize);
    let mut a = vec![Complex64::new(0.0, 0.0); m * n];
    let mut state = 0x5EEDu64;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let z = z ^ (z >> 31);
        (z as f64) / (u64::MAX as f64) * 2.0 - 1.0
    };
    for i in 0..m.min(n) {
        a[i * n + i] = Complex64::new(0.95f64.powi(i as i32), 0.0);
    }
    for slot in a.iter_mut() {
        *slot += Complex64::new(next(), next()) * 1e-3;
    }

    let max_rank = 16;
    let reference = jacobi_svd_flat(&a, m, n, n, max_rank, 1e-14);
    let candidate = f(&a, m, n, n, max_rank, 1e-14);
    assert!(
        reference.discarded_weight > 0.0,
        "the block must truncate, or there is no bound to compare"
    );
    let frob_sq: f64 = a.iter().map(|z| z.norm_sqr()).sum();
    let excess = candidate.discarded_weight - reference.discarded_weight;
    assert!(
        excess > 0.0,
        "the default macOS hook returned a bound at or below the Jacobi \
         reference's ({} vs {}), so it is NOT carrying Accelerate's \
         backward-error allowance — the default path is the Jacobi kernel \
         whatever it is called",
        candidate.discarded_weight,
        reference.discarded_weight
    );
    assert!(
        excess <= 1e-12 * frob_sq,
        "the default hook's bound is {excess:e} above the reference's against a \
         block norm of {frob_sq:e} — sound, but no longer the same certificate"
    );
}

// ---------------------------------------------------------------------------
// 2. The field tracks the kernel, from both sides
// ---------------------------------------------------------------------------

/// **Both directions, or the field proves nothing.** Force Jacobi and the
/// certificate must say `jacobi`; take the default and on macOS it must say
/// `accelerate-zgesdd`. A constant passes exactly one of these.
///
/// The two runs are also required to DISAGREE on macOS. That is the stronger
/// half: it says the two labels correspond to two different computations
/// reaching the certificate, not to two branches of a `match` on a string.
///
/// DOES NOT CATCH: an unsound Accelerate bound. The two certificates only have
/// to DIFFER here, so a kernel reporting a weight BELOW the Jacobi reference's
/// passes — the direction is held by
/// `accelerate_discarded_weight_never_undercuts_the_jacobi_reference` and by
/// `the_default_macos_hook_adds_the_accelerate_backward_error_allowance`.
#[test]
fn the_certificate_names_the_kernel_that_was_forced_onto_the_run() {
    let (jacobi_value, jacobi_cert) = run(MpsBackend::new(CHI).with_svd_kernel(SvdKernel::Jacobi));
    assert_eq!(
        jacobi_cert.svd_kernel.as_str(),
        "jacobi",
        "a run forced onto the Jacobi kernel must say so"
    );
    assert!(
        jacobi_cert.discarded_weight > 0.0,
        "the circuit must truncate at χ={CHI}, or neither kernel is exercised"
    );

    let (default_value, default_cert) = run(MpsBackend::new(CHI));
    assert_eq!(
        default_cert.svd_kernel.as_str(),
        default_svd_kernel().as_str(),
        "an unconfigured run must report the target default"
    );

    // Both kernels answer the same question, so the observable agrees to the
    // forward tolerance whichever ran. This is what makes the certificate
    // difference below a difference in the BOUND and not in the physics.
    assert!(
        (jacobi_value - default_value).abs() < 1e-9,
        "the two kernels disagree on ⟨Z₀⟩: {jacobi_value} vs {default_value}"
    );

    #[cfg(target_os = "macos")]
    {
        assert_eq!(default_cert.svd_kernel.as_str(), "accelerate-zgesdd");
        assert!(
            default_cert.discarded_weight > jacobi_cert.discarded_weight,
            "the default macOS run reported the SAME bound as the forced-Jacobi \
             run ({} vs {}) — two labels over one computation",
            default_cert.discarded_weight,
            jacobi_cert.discarded_weight
        );
    }
    #[cfg(not(target_os = "macos"))]
    {
        assert_eq!(default_cert.svd_kernel.as_str(), "jacobi");
        assert_eq!(
            default_cert.discarded_weight.to_bits(),
            jacobi_cert.discarded_weight.to_bits(),
            "off macOS the default IS the Jacobi kernel, so the two runs must be \
             bit-identical"
        );
    }
}

/// Forcing Accelerate explicitly reports Accelerate, on the target that has it.
///
/// Distinct from the test above: that one reads the DEFAULT and checks it
/// happens to be Accelerate on macOS, so if the promotion were reverted it
/// would fail and take this evidence with it. This one pins the
/// `AccelerateZgesdd` → `"accelerate-zgesdd"` mapping independently of what the
/// default is.
///
/// DOES NOT CATCH: anything numeric. It compares two labels and never reads a
/// certificate value, so a `with_svd_kernel` that installed the wrong FUNCTION
/// while recording the right name would pass.
#[cfg(target_os = "macos")]
#[test]
fn a_run_forced_onto_accelerate_names_accelerate() {
    let (_, cert) = run(MpsBackend::new(CHI).with_svd_kernel(SvdKernel::AccelerateZgesdd));
    assert_eq!(cert.svd_kernel.as_str(), "accelerate-zgesdd");
    let (_, forced_jacobi) = run(MpsBackend::new(CHI).with_svd_kernel(SvdKernel::Jacobi));
    assert_ne!(
        cert.svd_kernel.as_str(),
        forced_jacobi.svd_kernel.as_str(),
        "the two forced kernels must not report the same name"
    );
}

/// **What the Accelerate label does NOT claim.** Below the size gate the
/// Accelerate entry point delegates, so a run small enough that every θ block
/// is under `MIN_DIM_FOR_LAPACK` carries a certificate bit-identical to the
/// Jacobi kernel's while still reporting `accelerate-zgesdd`.
///
/// Pinned rather than left implicit for two reasons. It is the first thing a
/// reader who sees two identical certificates under two different labels will
/// ask, and it is the gate's promise — "installing this never changes the
/// result below the threshold" — asserted at the certificate rather than at the
/// kernel, where `accelerate.rs`'s
/// `below_the_gate_the_result_is_the_jacobi_kernel_bit_for_bit` already holds
/// it. A gate quietly removed would redden here too, on the shallow circuits
/// the gate exists to protect.
///
/// DOES NOT CATCH: the gate moving in the other direction. Raising
/// `MIN_DIM_FOR_LAPACK` keeps these 8×8 blocks below it and this test green,
/// while retiring LAPACK from every shape that pays for it.
#[cfg(target_os = "macos")]
#[test]
fn below_the_gate_the_accelerate_label_still_carries_a_jacobi_certificate() {
    // χ=4 saturates into 8×8 θ blocks: min(m, n) = 8 < MIN_DIM_FOR_LAPACK.
    const SMALL_CHI: usize = 4;
    fn small_run(backend: MpsBackend) -> omega_backend_mps::MpsRunStats {
        let circuit = entangling_circuit(10, 8);
        let params = ParameterBinding::new();
        let backend = backend.with_max_discarded_weight(f64::INFINITY);
        backend
            .expectation(&circuit, &params, &single_z())
            .expect("the brickwall evolves");
        backend.last_run_stats()
    }
    let accelerate = small_run(MpsBackend::new(SMALL_CHI));
    let jacobi = small_run(MpsBackend::new(SMALL_CHI).with_svd_kernel(SvdKernel::Jacobi));
    assert_eq!(accelerate.svd_kernel.as_str(), "accelerate-zgesdd");
    assert_eq!(jacobi.svd_kernel.as_str(), "jacobi");
    assert!(
        accelerate.max_bond_reached >= SMALL_CHI,
        "the run must saturate χ={SMALL_CHI} for its blocks to be the gated ones"
    );
    assert!(
        jacobi.discarded_weight > 0.0,
        "the run must truncate, or the two certificates agree trivially at 0.0"
    );
    assert_eq!(
        accelerate.discarded_weight.to_bits(),
        jacobi.discarded_weight.to_bits(),
        "below the gate the Accelerate entry point must delegate, so the two \
         certificates are the same bits: {} vs {}",
        accelerate.discarded_weight,
        jacobi.discarded_weight
    );
}

/// A kernel this crate cannot name is reported as `custom`, not as the default
/// it replaced. The CUDA `gesvdj` arm installs through `with_svd_fn` and lands
/// here; so does any measurement shim.
///
/// The failure this blocks is a macOS `--device cuda` run whose certificate
/// says `accelerate-zgesdd` because the label was set at construction and never
/// revisited — a record naming a kernel that provably did not run.
///
/// DOES NOT CATCH: `with_svd_fn` failing to install `f` at all. The shim here
/// delegates to the Jacobi kernel, so a `with_svd_fn` that set only the label
/// and dropped the function pointer would pass. Nothing in this file pins that;
/// the CUDA arm's own parity tests do.
#[test]
fn an_unnameable_hook_is_reported_as_custom_rather_than_as_the_default() {
    fn shim(
        a: &[Complex64],
        m: usize,
        n: usize,
        stride: usize,
        max_rank: usize,
        threshold: f64,
    ) -> omega_backend_mps::svd::SvdResultFlat {
        jacobi_svd_flat(a, m, n, stride, max_rank, threshold)
    }
    let (_, cert) = run(MpsBackend::new(CHI).with_svd_fn(shim));
    assert_eq!(cert.svd_kernel.as_str(), "custom");
    assert_ne!(
        cert.svd_kernel.as_str(),
        default_svd_kernel().as_str(),
        "an injected hook must not inherit the default's name"
    );
}

/// The three names are three names. Two arms of `as_str` returning one string
/// would make the two-sided tests above agree for the wrong reason.
///
/// DOES NOT CATCH: the names being SWAPPED. `Jacobi => "accelerate-zgesdd"` and
/// back is still three distinct strings; the two-sided run tests are what fix
/// each name to its kernel.
#[test]
fn every_kernel_has_its_own_name() {
    let names = [
        SvdKernel::Jacobi.as_str(),
        SvdKernel::AccelerateZgesdd.as_str(),
        SvdKernel::Custom.as_str(),
    ];
    let unique: std::collections::BTreeSet<&str> = names.iter().copied().collect();
    assert_eq!(
        unique.len(),
        names.len(),
        "duplicate kernel name in {names:?}"
    );
}

/// `SvdKernel::Custom` names no function, so `with_svd_kernel` must refuse it
/// rather than install the default and report `custom` — a backend running a
/// kernel the caller did not ask for, labelled as one that does not exist.
///
/// DOES NOT CATCH: the same refusal for `AccelerateZgesdd` on a non-macOS
/// target, which is the case that actually matters for a Linux embedder. It
/// cannot be tested from here: the panic only exists off macOS, where this
/// file's macOS arms do not run.
#[test]
#[should_panic(expected = "not available on this target")]
fn asking_for_the_unnameable_kernel_is_refused() {
    let _ = MpsBackend::new(CHI).with_svd_kernel(SvdKernel::Custom);
}
