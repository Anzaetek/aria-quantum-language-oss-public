//! `bindings/aria-py` is excluded from the workspace (a pyo3 cdylib, built with
//! maturin, not `cargo test`). Nothing in-tree executes that extension, so this
//! file pins the installer by source: re-adding the unconditional
//! `with_contract_fn(metal_contract_2q)` line reddens the assertion below.
//!
//! The aria-runtime constructors are pinned numerically next to `make_mps` /
//! `make_noisy_mps` (`mps_metal_certificate` in `src/run.rs`), which is a
//! different test binary and needs `--features metal`.

use std::path::PathBuf;

#[test]
fn aria_py_does_not_unconditionally_install_metal_contract() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bindings/aria-py/src/lib.rs");
    let src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let flat = src.split_whitespace().collect::<Vec<_>>().join(" ");
    let forbidden =
        "let backend = backend.with_contract_fn(omega_backend_mps_metal::metal_contract_2q)";
    assert!(
        !flat.contains(forbidden),
        "aria-py installs the f32 Metal contraction unconditionally; \
         discarded_weight would not be a bound"
    );
    assert!(
        src.contains("MPS_METAL_CONTRACT"),
        "aria-py dropped the shared opt-in; omega-run and aria-runtime still honour it"
    );
    let calls = src.matches("metal_contract_2q").count();
    assert_eq!(
        calls, 1,
        "aria-py should name metal_contract_2q once, inside the opt-in; found {calls}"
    );
    let opt_in = src
        .find("mps_metal_contract_opt_in()")
        .expect("opt-in predicate missing");
    let call = src.find("metal_contract_2q").expect("kernel call missing");
    assert!(
        opt_in < call,
        "metal_contract_2q is reached without the MPS_METAL_CONTRACT check"
    );
}
