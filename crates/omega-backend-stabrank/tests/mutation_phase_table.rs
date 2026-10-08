// SPDX-License-Identifier: Apache-2.0
//! S0 test (iv): the phase-table mutation ledger, and the local pin that
//! makes a mutation fail where it was made.
//!
//! The kernel does not carry its own Pauli-product phase table. It calls
//! `omega-backend-pauli::pauli_mult_phase`, the single copy whose doc
//! comment records what two copies cost last time: both had the `X·Z` and
//! `Z·X` rows inverted, one was fixed, and measurement sampling stayed
//! broken behind correct-looking expectations. Depending on it rather than
//! duplicating it means a defect there is a defect here — which is a
//! liability unless something checks it, so this file checks it twice: by
//! re-deriving every entry from 2×2 matrix products, and by recording what
//! actually happened when entries were flipped.
//!
//! The derivation leg is the one that makes a mutation *local*. Without it
//! a flipped cell shows up as a wrong amplitude three crates away and the
//! reader has to bisect; with it, the cell itself reddens and names
//! itself.
//!
//! # The ledger
//!
//! Each mutation below was applied to the source, the whole
//! `omega-backend-stabrank` suite was run with `--no-fail-fast`, and the
//! file was restored and checked byte-for-byte with `cmp`. Baseline: 24
//! tests, all green. Line numbers are as of the commit that added this
//! file.
//!
//! **A — `omega-backend-pauli/src/stabilizer.rs:65`, the `X·Z` cell:**
//! `((true, false), (false, true)) => 3` → `=> 1`.
//! Reddened, 9 tests: both tests in this file; the convention-shim unit
//! test in `chform`; `random_clifford_amplitudes_match_the_dense_statevector`
//! and `every_gate_in_the_s0_table_is_exercised_against_the_dense_oracle`
//! (S0 leg (i), on `CY`: amplitude 3 came back `−0.5i` for `+0.5i`);
//! `a_two_wire_amplitude_pin_reaches_the_shared_phase_table` (leg (ii));
//! `pairwise_overlaps_match_the_dense_contraction` and
//! `backend_expectation_matches_the_dense_backend`;
//! `clifford_expectations_equal_stims_exactly` (leg (iii)).
//! Stayed green: every single-wire pin in `global_phase_pin.rs`, the other
//! two two-wire pins, and `clifford_expectations_land_on_exact_integers` —
//! a sign error is still an integer.
//!
//! **B — `stabilizer.rs:66`, the `Z·X` cell:**
//! `((false, true), (true, false)) => 1` → `=> 3`.
//! Reddened, 8 tests: the same set as A except that the fixture which
//! catches it in `global_phase_pin.rs` is the *other* one,
//! `a_second_two_wire_pin_reaches_the_other_phase_table_cell`, and
//! `every_gate_in_the_s0_table…` stays green because none of its
//! two-wire fixtures reaches this cell. The two two-wire pins are each
//! green under the other's mutation; that is measured, and it is why
//! there are two.
//!
//! **C — `stabilizer.rs:70`, the `(XZ)·Z` cell:**
//! `((true, true), (false, true)) => 1` → `=> 3`.
//! Reddened, 7 tests: as B, minus
//! `the_x_times_z_and_z_times_x_cells_are_the_ones_that_were_wrong_before`,
//! which asserts only the two cells it names. The exhaustive
//! matrix-product test above is what catches this one locally.
//!
//! **D — `omega-backend-stabrank/src/chform.rs:200`, `s_gate`:**
//! `γ_q += 3` → `γ_q += 1`, i.e. `S` executed as `S†`. This is K8's
//! `t → tdg` trap in its Clifford form, and §2's fixture-symmetry rule is
//! what makes it visible: `|⟨+|S|+⟩|` cannot tell the two apart and
//! `Im⟨+|S|+⟩` can.
//! Reddened, 15 tests, including `plus_s_plus_is_one_plus_i_over_two` and
//! `plus_sdg_plus_is_the_conjugate_and_only_the_imaginary_part_sees_it`
//! on their imaginary legs, all three tests in
//! `inner_product_vs_dense.rs`, and the Stim oracle.
//! Stayed green: all three phase-table tests (they pin the table, not the
//! gate); `orthogonal_states_overlap_at_exactly_zero`, which has no `S` in
//! it; `the_state_stays_normalised_…`, because `S†` is also unitary; and
//! `an_inserted_s_gate_makes_the_two_disagree`, a control that only asks
//! for *a* disagreement and gets one either way.
//!
//! **E — `chform.rs:455`, `desuperpose`:**
//! `Complex64::new(ISQRT2, ISQRT2)` → `(ISQRT2, -ISQRT2)`, i.e. the
//! `e^{iπ/4}` branch factor conjugated.
//! Reddened, exactly 3 tests:
//! `random_clifford_amplitudes_match_the_dense_statevector`,
//! `pairwise_overlaps_match_the_dense_contraction`, and
//! `the_e_to_the_i_pi_over_four_branch_has_its_own_pin`.
//! **Everything else stayed green, including the entire Stim leg and
//! `backend_expectation_matches_the_dense_backend`.** That is not a gap in
//! those fixtures, it is the point of S0: this mutation changes only the
//! global phase, Stim's tableau has none, and `⟨P⟩` on a single state is
//! invariant under one. A kernel checked against Stim alone would ship
//! this defect, and a stabilizer-rank sum built on it would have the wrong
//! cross terms.

use num_complex::Complex64;
use omega_backend_pauli::pauli_mult_phase;

type Mat = [Complex64; 4];

fn mul(a: Mat, b: Mat) -> Mat {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

fn scale(k: Complex64, a: Mat) -> Mat {
    [k * a[0], k * a[1], k * a[2], k * a[3]]
}

/// `P(x,z) = i^{xz} X^x Z^z`, built from the literal Pauli matrices.
fn p(x: bool, z: bool) -> Mat {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let eye = [one, zero, zero, one];
    let xm = [zero, one, one, zero];
    let zm = [one, zero, zero, -one];
    let body = mul(if x { xm } else { eye }, if z { zm } else { eye });
    let pref = if x && z {
        Complex64::new(0.0, 1.0)
    } else {
        one
    };
    scale(pref, body)
}

fn pow_i(k: u32) -> Complex64 {
    match k % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

/// Every one of the 16 cells of the shared table, against the matrix
/// product it is supposed to summarise. Tolerance 0: the entries are
/// powers of `i` and the matrices have entries in `{0, ±1, ±i}`.
#[test]
fn every_entry_of_the_shared_phase_table_matches_a_matrix_product() {
    for x1 in [false, true] {
        for z1 in [false, true] {
            for x2 in [false, true] {
                for z2 in [false, true] {
                    let lhs = mul(p(x1, z1), p(x2, z2));
                    let g = pauli_mult_phase(x1, z1, x2, z2);
                    assert!(
                        (0..4).contains(&g),
                        "g must be a power of i mod 4, got {g} at \
                         ({x1},{z1})·({x2},{z2})"
                    );
                    let rhs = scale(pow_i(g as u32), p(x1 ^ x2, z1 ^ z2));
                    for k in 0..4 {
                        assert_eq!(
                            lhs[k],
                            rhs[k],
                            "P({x1},{z1})·P({x2},{z2}) = i^{g} P({},{}) is false in \
                             entry {k}: {} vs {}. The shared Aaronson-Gottesman \
                             table has a wrong cell, and omega-backend-stabrank's \
                             amplitudes are wrong with it.",
                            x1 ^ x2,
                            z1 ^ z2,
                            lhs[k],
                            rhs[k]
                        );
                    }
                }
            }
        }
    }
}

/// The two cells the CH-form kernel is most exposed to, called out by
/// name so a future edit that "simplifies" them has to delete an
/// assertion rather than just change a match arm.
#[test]
fn the_x_times_z_and_z_times_x_cells_are_the_ones_that_were_wrong_before() {
    assert_eq!(pauli_mult_phase(true, false, false, true), 3, "X·Z");
    assert_eq!(pauli_mult_phase(false, true, true, false), 1, "Z·X");
    assert_ne!(
        pauli_mult_phase(true, false, false, true),
        pauli_mult_phase(false, true, true, false),
        "X·Z and Z·X must differ — they are inverses, not equals, and \
         setting them equal is exactly the defect FIXES_PLAN records"
    );
}
