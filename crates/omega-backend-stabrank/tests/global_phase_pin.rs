// SPDX-License-Identifier: Apache-2.0
//! S0 test (ii): the phase pin.
//!
//! `⟨+|S|+⟩ = (1+i)/2`, asserted on **both** parts.
//!
//! This is the fixture that a phase-insensitive kernel cannot pass. An
//! Aaronson–Gottesman tableau quietly wrapped in a CH-form API knows the
//! stabilizer group of `|+⟩` and of `S|+⟩` and can therefore produce
//! `|⟨φ|φ′⟩| = 1/√2 = 0.7071…` — and nothing else. The real part alone
//! would already catch that (0.5 ≠ 0.7071), but the imaginary part is the
//! leg that also separates `S` from `S†`, whose overlap is the conjugate
//! `(1−i)/2` with the *same* real part and the same modulus. That is §2's
//! fixture-symmetry rule applied to this phase: the quantity that cannot
//! tell the two apart is not allowed to be the only one asserted.
//!
//! # One-wire pins cannot reach the phase table, so this file has two-wire
//! pins as well
//!
//! On one wire the Pauli accumulation inside `ChForm::amplitude` runs for
//! at most one row, so its running product is always the identity and the
//! only cell of the shared Aaronson–Gottesman table it ever reads is the
//! trivial one. **Every single-wire fixture in this file is therefore
//! green under any mutation of that table** — measured, not assumed; see
//! `mutation_phase_table.rs`. The two-wire pins at the bottom are here so
//! that fixture (ii) of PLAN-MAJORANA-STIM.md S0 reddens on that mutation
//! as the plan requires, and they are hand-computed constants like the
//! rest, not comparisons against another simulator.
//!
//! # Which mutation this fixture catches, and which it does not
//!
//! Catches: a dropped or conjugated `γ` increment on `S`/`S†`, a wrong
//! `e^{±iπ/4}` in `desuperpose`, and — through the two-wire pins — the
//! `X·Z` row of the shared table.
//!
//! Does NOT catch: an error that is *symmetric* between the bra and the
//! ket. `inner_product` applies the bra's `U_C†` to the ket, so a
//! convention error shared by both sides can cancel in the overlap pins;
//! the sweep in `inner_product_vs_dense.rs` uses two independently random
//! circuits and does not have that symmetry. It also does not reach any
//! three-or-more-wire structure, nor the CNOT-network ordering in
//! `cx_network`, which only has something to get wrong once `F` is not
//! close to a permutation — `amplitude_vs_statevector.rs` carries those.

use num_complex::Complex64;
use omega_backend_stabrank::ChForm;
use omega_core::executor::PauliOp;

/// Tight on purpose. These are exactly representable halves; the only
/// rounding in the path is `1/√2 · 1/√2`.
const TOL: f64 = 1e-15;

fn plus() -> ChForm {
    let mut state = ChForm::zero(1);
    state.h_gate(0);
    state
}

fn close(got: Complex64, want: Complex64, what: &str) {
    assert!(
        (got.re - want.re).abs() < TOL,
        "{what}: Re = {} , want {}",
        got.re,
        want.re
    );
    assert!(
        (got.im - want.im).abs() < TOL,
        "{what}: Im = {} , want {}. A kernel that returns a magnitude, or one \
         that applied S† where S was meant, has the right real part here and \
         the wrong imaginary part — which is the whole reason this leg is \
         asserted separately.",
        got.im,
        want.im
    );
}

/// The pin itself.
#[test]
fn plus_s_plus_is_one_plus_i_over_two() {
    let bra = plus();
    let mut ket = plus();
    ket.s_gate(0);
    close(bra.inner_product(&ket), Complex64::new(0.5, 0.5), "⟨+|S|+⟩");
}

/// `S†` has the same real part and the same modulus. Asserting it here is
/// what makes the imaginary-part leg above load-bearing rather than
/// decorative: delete the `Im` assertion and this pair stops distinguishing
/// the two gates.
#[test]
fn plus_sdg_plus_is_the_conjugate_and_only_the_imaginary_part_sees_it() {
    let bra = plus();
    let mut ket = plus();
    ket.sdg_gate(0);
    let v = bra.inner_product(&ket);
    close(v, Complex64::new(0.5, -0.5), "⟨+|S†|+⟩");

    let mut s_ket = plus();
    s_ket.s_gate(0);
    let s = bra.inner_product(&s_ket);
    assert!(
        (v.re - s.re).abs() < TOL,
        "the fixture-symmetry claim is false: ⟨+|S|+⟩ and ⟨+|S†|+⟩ differ in \
         Re ({} vs {}), so this pair is not actually testing the imaginary leg",
        s.re,
        v.re
    );
    assert!(
        (v.norm() - s.norm()).abs() < TOL,
        "⟨+|S|+⟩ and ⟨+|S†|+⟩ must have equal modulus for this pair to be the \
         phase-blindness trap it claims to be"
    );
    assert!(
        (v.im - s.im).abs() > 0.9,
        "the imaginary parts must actually differ, got {} and {}",
        s.im,
        v.im
    );
}

/// The same number by the other route through the kernel: sum the
/// amplitudes. `inner_product` never enumerates basis states, so this is a
/// genuine second path, and it pins the two to one convention — including
/// which side is conjugated.
#[test]
fn the_amplitude_path_and_the_inner_product_path_agree_on_the_pin() {
    let bra = plus();
    let mut ket = plus();
    ket.s_gate(0);
    let by_sum: Complex64 = bra
        .to_statevector()
        .iter()
        .zip(ket.to_statevector().iter())
        .map(|(a, b)| a.conj() * b)
        .sum();
    close(by_sum, Complex64::new(0.5, 0.5), "Σ_x conj⟨x|+⟩⟨x|S|+⟩");
    close(
        bra.inner_product(&ket),
        by_sum,
        "inner_product against the amplitude sum",
    );
}

/// `⟨φ|P|φ′⟩` carries the phase too, and `Z` between the two states turns
/// `(1+i)/2` into `(1−i)/2` — a sign change the imaginary part alone sees.
#[test]
fn a_pauli_between_two_states_keeps_the_phase() {
    let bra = plus();
    let mut ket = plus();
    ket.s_gate(0);
    close(
        bra.pauli_matrix_element(&[(0, PauliOp::Z)], &ket),
        Complex64::new(0.5, -0.5),
        "⟨+|Z S|+⟩",
    );
    // Y|0⟩ = i|1⟩ and Y|1⟩ = −i|0⟩, so Y fixes (|0⟩+i|1⟩)/√2 exactly.
    close(
        bra.pauli_matrix_element(&[(0, PauliOp::Y)], &ket),
        Complex64::new(0.5, 0.5),
        "⟨+|Y S|+⟩",
    );
    close(
        bra.pauli_matrix_element(&[(0, PauliOp::X)], &ket),
        Complex64::new(0.5, 0.5),
        "⟨+|X S|+⟩",
    );
}

/// A state against itself is exactly 1 — real part and imaginary part.
/// `inner_product` routes the bra's whole Clifford through the ket, so this
/// is a non-trivial round trip even though the answer is trivial.
#[test]
fn a_state_overlaps_itself_at_exactly_one() {
    let mut state = ChForm::zero(4);
    for q in 0..4 {
        state.h_gate(q);
        state.s_gate(q);
    }
    state.cx_gate(0, 1);
    state.cz_gate(2, 3);
    state.h_gate(2);
    state.sdg_gate(3);
    state.swap_gate(1, 3);
    state.check_invariants().unwrap();
    let v = state.inner_product(&state);
    assert!((v.re - 1.0).abs() < 1e-14, "⟨φ|φ⟩ Re = {}", v.re);
    assert!(v.im.abs() < 1e-14, "⟨φ|φ⟩ Im = {}", v.im);
}

/// `CY(0,1) (I⊗S) (H⊗H) |00⟩ = ½(|00⟩ + |10⟩ + i|01⟩ + i|11⟩)`, written
/// in wire order so bit `q` of the index is wire `q`.
///
/// By hand: `H⊗H` gives all four amplitudes `½`; `S` on wire 1 multiplies
/// the two with `q₁ = 1` by `i`; `CY` acts on the `q₀ = 1` half with
/// `Y|0⟩ = i|1⟩`, `Y|1⟩ = −i|0⟩`, which sends `(½, i½)` on that half to
/// `(−i·i½, i·½) = (½, i½)` — unchanged, as it happens, so the whole
/// state is `½(1, 1, i, i)`.
///
/// Two wires is the smallest width at which `ChForm::amplitude`
/// multiplies two Paulis together, so this is the smallest hand-computed
/// pin that reads a non-trivial cell of the shared phase table. Measured:
/// flipping the `X·Z` entry turns the last amplitude into `−i/2`.
#[test]
fn a_two_wire_amplitude_pin_reaches_the_shared_phase_table() {
    let mut state = ChForm::zero(2);
    state.h_gate(0);
    state.h_gate(1);
    state.s_gate(1);
    state.cy_gate(0, 1);
    state.check_invariants().unwrap();
    let want = [
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.0, 0.5),
        Complex64::new(0.0, 0.5),
    ];
    for (i, (got, want)) in state.to_statevector().iter().zip(want).enumerate() {
        close(*got, want, &format!("amplitude {i} of CY(I⊗S)(H⊗H)|00⟩"));
    }
}

/// `(S⊗I) CZ (H⊗H)|00⟩ = ½(|00⟩ + i|10⟩ + |01⟩ − i|11⟩)` in wire order.
///
/// By hand: `H⊗H` gives `½` four times, `CZ` negates the `q₀=q₁=1`
/// amplitude, `S` on wire 0 multiplies the two with `q₀ = 1` by `i`.
///
/// A second two-wire pin because the first one does not reach every cell
/// of the shared table: this one reads `Z·X` where that one reads `X·Z`,
/// and each is green under the other's mutation. Measured both ways in
/// `mutation_phase_table.rs`.
#[test]
fn a_second_two_wire_pin_reaches_the_other_phase_table_cell() {
    let mut state = ChForm::zero(2);
    state.h_gate(0);
    state.h_gate(1);
    state.cz_gate(0, 1);
    state.s_gate(0);
    state.check_invariants().unwrap();
    let want = [
        Complex64::new(0.5, 0.0),
        Complex64::new(0.0, 0.5),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.0, -0.5),
    ];
    for (i, (got, want)) in state.to_statevector().iter().zip(want).enumerate() {
        close(*got, want, &format!("amplitude {i} of (S⊗I)CZ(H⊗H)|00⟩"));
    }
}

/// `H₀ H₁ S₀ CX(0→1) H₀ |00⟩ = ¼√2·(1+i, 1−i, 1−i, 1+i)`, the pin for the
/// `e^{iπ/4}` factor in `desuperpose`.
///
/// By hand, in wire order: `H₀` gives `(|00⟩+|10⟩)/√2`; `CX(0→1)` makes
/// it `(|00⟩+|11⟩)/√2`; `S₀` puts `i` on `|11⟩`; `H₁` gives
/// `½(|00⟩ + i|10⟩ + |01⟩ − i|11⟩)`; `H₀` combines the two `q₁` halves
/// into `(1±i)/2√2`.
///
/// This is the shortest two-wire word that reaches the branch of
/// `desuperpose` where the Hadamard lands on a wire that already carries
/// one and the relative phase is `±i` — the only place in the kernel
/// where the global phase leaves `{±1, ±i}`. Measured: conjugating that
/// factor flips the sign of every imaginary part here, and is invisible
/// to all the other pins in this file and to the Stim oracle, which has
/// no global phase to be wrong about.
#[test]
fn the_e_to_the_i_pi_over_four_branch_has_its_own_pin() {
    let mut state = ChForm::zero(2);
    state.h_gate(0);
    state.cx_gate(0, 1);
    state.s_gate(0);
    state.h_gate(1);
    state.h_gate(0);
    state.check_invariants().unwrap();
    let a = std::f64::consts::FRAC_1_SQRT_2 / 2.0;
    let want = [
        Complex64::new(a, a),
        Complex64::new(a, -a),
        Complex64::new(a, -a),
        Complex64::new(a, a),
    ];
    for (i, (got, want)) in state.to_statevector().iter().zip(want).enumerate() {
        close(*got, want, &format!("amplitude {i} of H₀H₁S₀CX(0→1)H₀|00⟩"));
    }
}

/// The same state seen through an overlap: `⟨++|` against it is
/// `½·(½ + ½ + i½ + i½) = (1+i)/2`. Same constant as the one-wire pin,
/// reached through the two-wire `inner_product` path.
///
/// This one is *not* a phase-table probe, measured: `⟨++|` has a trivial
/// `U_C`, so the bra contributes no Clifford to push through the ket and
/// the amplitude is read at `x = 00`, where the Pauli accumulation is
/// empty. It stays green under the `X·Z` mutation. It is here for the
/// `inner_product` path, not for the table.
#[test]
fn a_two_wire_overlap_pin_is_also_one_plus_i_over_two() {
    let mut bra = ChForm::zero(2);
    bra.h_gate(0);
    bra.h_gate(1);
    let mut ket = bra.clone();
    ket.s_gate(1);
    ket.cy_gate(0, 1);
    close(
        bra.inner_product(&ket),
        Complex64::new(0.5, 0.5),
        "⟨++|CY(I⊗S)|++⟩",
    );
}

/// Orthogonality is exact, not approximate: `⟨0|1⟩ = 0` with both parts
/// zero. A kernel that lost the Kronecker-delta branch of the amplitude
/// would return something of order 1 here.
#[test]
fn orthogonal_states_overlap_at_exactly_zero() {
    let zero = ChForm::zero(1);
    let mut one = ChForm::zero(1);
    one.x_gate(0);
    let v = zero.inner_product(&one);
    assert_eq!(v.re, 0.0, "⟨0|1⟩ Re");
    assert_eq!(v.im, 0.0, "⟨0|1⟩ Im");

    let minus = {
        let mut m = ChForm::zero(1);
        m.x_gate(0);
        m.h_gate(0);
        m
    };
    let v = plus().inner_product(&minus);
    assert!(v.norm() < 1e-15, "⟨+|−⟩ = {v}");
}
