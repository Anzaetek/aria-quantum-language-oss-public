#![allow(clippy::needless_range_loop)]

pub mod gates;
pub mod mps;
mod sim;
pub mod svd;

pub use mps::{Contract2qFn, SvdFlatFn};
pub use sim::{MpsBackend, MpsRunStats, NoisyMpsBackend, DEFAULT_MAX_DISCARDED_WEIGHT};

/// How `--backend mps…` is spelled, parsed in ONE place.
///
/// The grammar lived only in `aria-runtime`, so `aria` accepted `mps:<chi>` and
/// `mps:auto` while `omega-run` accepted only the bare `mps` and hardcoded
/// χ=64 — a user whose circuit needed a larger bond had no way to say so
/// through that binary, and after the truncation gate landed they got a refusal
/// with no route to a correct answer.
///
/// The obvious fix — have `omega-run` call `aria-runtime`'s parser — inverts the
/// layering: `aria-*` is built on `omega-*`, not the other way round. So the
/// grammar lives here instead, in the crate that owns the knob, which both
/// front ends already depend on. One spelling, no second copy to drift.
pub mod select {
    /// Default fixed bond dimension for a bare `mps`.
    pub const DEFAULT_CHI: usize = 64;
    /// Default ceiling for `mps:auto`.
    pub const DEFAULT_AUTO_CEILING: usize = 1024;
    /// Relative singular-value tolerance used by `mps:auto`.
    ///
    /// Kept beside the grammar it belongs to. `aria-runtime` has its own copy
    /// (`MPS_AUTO_EPS`) predating this module; the two must agree, which is
    /// pinned by `both_front_ends_use_the_same_auto_epsilon` in aria-runtime.
    pub const AUTO_EPS: f64 = 1e-10;

    /// A parsed MPS backend selector.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum MpsSelect {
        /// Fixed bond dimension.
        Fixed { chi: usize },
        /// Adaptive bond up to a ceiling.
        Auto { max_chi: usize },
    }

    /// Parse `mps`, `mps:<chi>`, `mps:auto`, `mps:auto:<ceiling>`.
    ///
    /// Returns `Ok(None)` when `s` is not an MPS selector at all, so a caller
    /// can fall through to its own backend names without this function needing
    /// to know them.
    pub fn parse_mps(s: &str) -> Result<Option<MpsSelect>, String> {
        if s == "mps" {
            return Ok(Some(MpsSelect::Fixed { chi: DEFAULT_CHI }));
        }
        if s == "mps:auto" {
            return Ok(Some(MpsSelect::Auto {
                max_chi: DEFAULT_AUTO_CEILING,
            }));
        }
        if let Some(rest) = s.strip_prefix("mps:auto:") {
            let max_chi: usize = rest
                .parse()
                .map_err(|_| format!("bad MPS ceiling in '{s}' (want mps:auto:<int>)"))?;
            if max_chi == 0 {
                return Err("MPS bond ceiling must be ≥ 1".into());
            }
            return Ok(Some(MpsSelect::Auto { max_chi }));
        }
        if let Some(rest) = s.strip_prefix("mps:") {
            let chi: usize = rest
                .parse()
                .map_err(|_| format!("bad MPS bond dimension in '{s}' (want mps:<int>)"))?;
            if chi == 0 {
                return Err("MPS bond dimension must be ≥ 1".into());
            }
            return Ok(Some(MpsSelect::Fixed { chi }));
        }
        Ok(None)
    }

    /// Does this selector name the MPS backend at all?
    ///
    /// For call sites that gate on backend identity rather than construct one —
    /// e.g. a noise-model check written as `matches!(name, "mps")`, which would
    /// otherwise reject `mps:512` for no reason.
    pub fn is_mps(s: &str) -> bool {
        matches!(parse_mps(s), Ok(Some(_)))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_whole_grammar_parses() {
            assert_eq!(parse_mps("mps").unwrap(), Some(MpsSelect::Fixed { chi: DEFAULT_CHI }));
            assert_eq!(parse_mps("mps:512").unwrap(), Some(MpsSelect::Fixed { chi: 512 }));
            assert_eq!(
                parse_mps("mps:auto").unwrap(),
                Some(MpsSelect::Auto { max_chi: DEFAULT_AUTO_CEILING })
            );
            assert_eq!(parse_mps("mps:auto:256").unwrap(), Some(MpsSelect::Auto { max_chi: 256 }));
        }

        /// Not-an-MPS-name is `Ok(None)`, not an error — the caller owns its own
        /// backend list.
        #[test]
        fn other_backends_fall_through() {
            for s in ["statevector", "pauli", "photonics", "pauliprop", "tch"] {
                assert_eq!(parse_mps(s).unwrap(), None, "{s} must fall through");
            }
        }

        /// Zero is refused rather than silently becoming a bond of 1 — a bond of
        /// 0 has no meaning and the mistake is worth naming.
        #[test]
        fn zero_is_refused_and_garbage_is_an_error() {
            assert!(parse_mps("mps:0").is_err());
            assert!(parse_mps("mps:auto:0").is_err());
            assert!(parse_mps("mps:banana").is_err());
            assert!(parse_mps("mps:auto:banana").is_err());
        }

        /// `is_mps` must accept every spelling, not just the bare name — the
        /// point of it is call sites that wrote `matches!(name, "mps")` and
        /// would otherwise reject `mps:512`.
        #[test]
        fn is_mps_accepts_every_spelling() {
            for s in ["mps", "mps:512", "mps:auto", "mps:auto:256"] {
                assert!(is_mps(s), "{s} must be recognised as MPS");
            }
            for s in ["statevector", "pauli", "mps:0", "mps:banana"] {
                assert!(!is_mps(s), "{s} must not be recognised as MPS");
            }
        }
    }
}
