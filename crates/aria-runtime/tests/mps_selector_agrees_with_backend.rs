// SPDX-License-Identifier: Apache-2.0
//! The two front ends must spell `--backend mps…` the same way.
//!
//! `aria` grew the grammar first; `omega-run` matched the literal `"mps"` and
//! hardcoded chi=64, so the same flag meant different things in the two
//! binaries and one of them could not ask for a correct answer at all.
//!
//! The grammar now lives in `omega_backend_mps::select`, which both depend on.
//! `aria-runtime` keeps its own `BackendSel` because it carries backends the
//! other binary does not have (gpu, tch, remote) — so the MPS *constants* exist
//! in two places and this pins them together.

#[test]
fn both_front_ends_use_the_same_defaults() {
    assert_eq!(
        aria_runtime::run::DEFAULT_MPS_CHI,
        omega_backend_mps::select::DEFAULT_CHI,
        "a bare `--backend mps` must mean the same bond dimension in both binaries"
    );
    assert_eq!(
        aria_runtime::run::DEFAULT_MPS_AUTO_CEILING,
        omega_backend_mps::select::DEFAULT_AUTO_CEILING,
        "`mps:auto` must mean the same ceiling in both binaries"
    );
    assert_eq!(
        aria_runtime::run::MPS_AUTO_EPS,
        omega_backend_mps::select::AUTO_EPS,
        "`mps:auto` must use the same singular-value tolerance in both binaries — \
         otherwise the same flag truncates differently depending on which one you ran"
    );
}

/// The two parsers must agree on what each spelling means, not merely on the
/// constants. Asserting the constants alone would pass if one parser mapped
/// `mps:512` to a ceiling and the other to a fixed bond.
#[test]
fn both_parsers_agree_on_every_spelling() {
    use aria_runtime::run::BackendSel;
    use omega_backend_mps::select::{parse_mps, MpsSelect};

    for spelling in ["mps", "mps:512", "mps:auto", "mps:auto:256"] {
        let theirs = parse_mps(spelling)
            .unwrap_or_else(|e| panic!("{spelling}: backend parser: {e}"))
            .unwrap_or_else(|| panic!("{spelling}: backend parser did not recognise it"));
        let ours = BackendSel::parse(spelling)
            .unwrap_or_else(|e| panic!("{spelling}: aria parser: {e}"));
        match (ours, theirs) {
            (BackendSel::Mps { chi: a }, MpsSelect::Fixed { chi: b }) => {
                assert_eq!(a, b, "{spelling}: fixed bond differs")
            }
            (BackendSel::MpsAuto { max_chi: a }, MpsSelect::Auto { max_chi: b }) => {
                assert_eq!(a, b, "{spelling}: adaptive ceiling differs")
            }
            (o, t) => panic!("{spelling}: aria says {o:?}, backend says {t:?} — different KINDS"),
        }
    }
}

/// Both must refuse the same malformed spellings. A parser that accepted
/// `mps:0` in one binary and refused it in the other is the drift this exists
/// to prevent.
#[test]
fn both_parsers_refuse_the_same_garbage() {
    use aria_runtime::run::BackendSel;
    use omega_backend_mps::select::parse_mps;
    for bad in ["mps:0", "mps:auto:0", "mps:banana", "mps:auto:banana"] {
        assert!(parse_mps(bad).is_err(), "backend parser accepted {bad}");
        assert!(BackendSel::parse(bad).is_err(), "aria parser accepted {bad}");
    }
}
