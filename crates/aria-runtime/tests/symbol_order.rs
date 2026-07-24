// SPDX-License-Identifier: Apache-2.0
//! Pins the SymbolId binding-order contract documented on `lower::Lowered`:
//! symbol ids are dense (`0..N`) and assigned in first-appearance order, and
//! a flat parameter slice ordered by ascending SymbolId binds `params[id]` to
//! the symbol with that id. External integrations rely on this to move
//! weights between their own vectors and Aria's `ParameterBinding`; a change
//! here is a breaking change, so it is a tested guarantee.

use aria_core::ast::parse_aria;
use aria_runtime::lower::lower;

fn lowered(src: &str, circuit: &str) -> aria_runtime::lower::Lowered {
    let prog = parse_aria(src).expect("parse");
    let c = prog.instantiate(circuit, &[]).expect("instantiate");
    lower(&c).expect("lower")
}

#[test]
fn symbol_ids_are_dense_and_first_appearance_ordered() {
    // `a` appears before `b` in the body, so id(a) < id(b); the ids cover
    // exactly 0..N with no gaps.
    let src = "circuit Two() {\n  qreg q[2]\n  let a = symbolic[1]\n  let b = symbolic[1]\n  \
               apply RY(a[0]) on q[0]\n  apply RZ(b[0]) on q[1]\n}\n";
    let low = lowered(src, "Two");
    let a = low.symbol_ids["a_0"];
    let b = low.symbol_ids["b_0"];
    assert!(a < b, "first-appearance order: id(a)={a} !< id(b)={b}");

    let mut ids: Vec<u32> = low.symbol_ids.values().copied().collect();
    ids.sort_unstable();
    let expected: Vec<u32> = (0..ids.len() as u32).collect();
    assert_eq!(ids, expected, "ids must be dense 0..N");
}

#[test]
fn binding_by_symbolid_binds_positionally() {
    // ⟨Z0⟩ = cos(a), ⟨Z1⟩ = cos(b) for RY(a)q0, RY(b)q1. Bind a
    // `ParameterBinding` by SymbolId and confirm each qubit sees its own
    // angle — i.e. the id → value mapping consumers rely on is real. Uses
    // the public Backend surface directly (the pattern LIBRARY.md documents).
    use omega_backend_statevector::StatevectorBackend;
    use omega_core::executor::{Backend, Observable};
    use omega_core::params::ParameterBinding;

    let src = "circuit Two() {\n  qreg q[2]\n  let a = symbolic[1]\n  let b = symbolic[1]\n  \
               apply RY(a[0]) on q[0]\n  apply RY(b[0]) on q[1]\n}\n";
    let low = lowered(src, "Two");
    let (a_id, b_id) = (low.symbol_ids["a_0"], low.symbol_ids["b_0"]);

    let (theta_a, theta_b) = (0.7, 1.9);
    let mut binding = ParameterBinding::new();
    binding.bind(a_id, theta_a);
    binding.bind(b_id, theta_b);

    let backend = StatevectorBackend::new();
    let z0 = backend
        .expectation(&low.ir, &binding, &Observable::parse("Z0").unwrap())
        .unwrap();
    let z1 = backend
        .expectation(&low.ir, &binding, &Observable::parse("Z1").unwrap())
        .unwrap();
    assert!((z0 - theta_a.cos()).abs() < 1e-12, "q0 must see angle a");
    assert!((z1 - theta_b.cos()).abs() < 1e-12, "q1 must see angle b");
}
