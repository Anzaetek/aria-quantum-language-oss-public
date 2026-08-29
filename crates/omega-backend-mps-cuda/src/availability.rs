// SPDX-License-Identifier: Apache-2.0
//! Why the CUDA MPS SVD hook is unavailable, as a **typed value** instead of a
//! process abort (3b.3 R1).
//!
//! # The defect
//!
//! On a CUDA 13 host, `--backend mps --device cuda` did not fail — it *died*:
//!
//! ```text
//! thread 'main' panicked at cudarc-0.19.4/src/cusolver/sys/mod.rs:23531:18:
//! Expected symbol in library: DlSym { source:
//!   "/usr/local/cuda/targets/sbsa-linux/lib/libcusolver.so:
//!    undefined symbol: cusolverDnGeqrf" }
//! ```
//!
//! CUDA 13 removed the legacy dense cuSOLVER API. `cudarc` 0.19 still declares
//! and `dlsym`s those entry points, so resolution fails on first use — inside
//! `DnHandle::new`, before any of our code runs, and **before
//! `CudaSvdContext::new`'s `Option` can express it.**
//!
//! Note what our own code calls: `cusolverDnZgesvdaStridedBatched`, which CUDA
//! 13 still has. The absent symbol is one we never use; it is absent from the
//! *binding table*, not from our call graph. So "avoid the removed function"
//! is not available as a fix — the panic happens at load, not at call.
//!
//! A missing symbol is an **environment condition, not a caller bug**, and must
//! never abort the process.
//!
//! # What is verifiable where
//!
//! This module is deliberately split. [`classify_panic`] is **pure string
//! analysis and compiles everywhere**, so it is tested here against the
//! verbatim payload from the bug report. The probe that produces such a payload
//! needs a CUDA 13 host and is gated; it is written but **unverified** — see
//! §4 of the plan.

use std::fmt;

/// Why the CUDA SVD hook is not usable on this host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaSvdUnavailable {
    /// Compiled without the `cuda` feature, or on an OS the hook does not
    /// target. Not an error — the CPU SVD is the supported path.
    NotBuilt,
    /// No CUDA device or driver could be opened.
    NoDevice,
    /// A symbol `cudarc` declares is absent from the installed library.
    /// **This is the CUDA 13 case.**
    MissingSymbol {
        /// e.g. `cusolverDnGeqrf`.
        symbol: String,
        /// Absolute path of the library that lacked it, when the payload
        /// carried one.
        library: Option<String>,
    },
    /// Initialisation failed some other way, without aborting.
    Other(String),
}

impl fmt::Display for CudaSvdUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotBuilt => write!(
                f,
                "mps-cuda: this binary has no CUDA SVD hook (built without the \
                 `cuda` feature, or for a non-Linux/Windows target). The CPU \
                 bond-compression SVD is used instead and results are unaffected."
            ),
            Self::NoDevice => write!(
                f,
                "mps-cuda: no CUDA device or driver could be opened. Run with \
                 `--device cpu`, which is unaffected."
            ),
            Self::MissingSymbol { symbol, library } => {
                write!(
                    f,
                    "mps-cuda: the CUDA library on this host does not provide \
                     `{symbol}`"
                )?;
                if let Some(lib) = library {
                    write!(f, " ({lib})")?;
                }
                write!(
                    f,
                    ". CUDA 13 removed the legacy dense cuSOLVER API \
                     (`cusolverDn<t>geqrf` and relatives) and the `cudarc` \
                     binding still resolves those entry points at load, so the \
                     GPU bond-compression SVD cannot initialise here. This is \
                     an environment condition, not a defect in the circuit or \
                     the caller. Use `--device cpu` — unaffected, and the only \
                     cost is the SVD offload — or a CUDA 12.x host, where this \
                     path is supported."
                )
            }
            Self::Other(detail) => write!(
                f,
                "mps-cuda: the CUDA SVD hook failed to initialise: {detail}. \
                 Use `--device cpu`, which is unaffected."
            ),
        }
    }
}

impl CudaSvdUnavailable {
    /// True when this is a *host configuration* problem the operator can act
    /// on, rather than "this build simply has no GPU path".
    ///
    /// The distinction decides whether the CLI should refuse or fall through
    /// silently: `NotBuilt` on a laptop is normal and must stay quiet, while a
    /// CUDA 13 host that was explicitly asked for `--device cuda` must be told.
    pub fn is_actionable(&self) -> bool {
        !matches!(self, Self::NotBuilt)
    }
}

/// Classify a panic payload from `cudarc`'s symbol resolution.
///
/// **Pure**, so it compiles and is tested on every platform, including hosts
/// with no CUDA at all. The payload shape is `cudarc` 0.19's
/// `Expected symbol in library: DlSym { source: "<lib>: undefined symbol: <sym>" }`,
/// but the parse is written against the *substrings that carry the meaning*
/// rather than the whole sentence, so a `cudarc` version that rewords the
/// wrapper still classifies correctly as long as `libloading`'s underlying
/// "undefined symbol:" text survives.
pub fn classify_panic(payload: &str) -> CudaSvdUnavailable {
    const MARKER: &str = "undefined symbol:";
    let Some(idx) = payload.find(MARKER) else {
        return CudaSvdUnavailable::Other(payload.trim().to_string());
    };

    // Symbol: everything after the marker, up to the first character that
    // cannot be part of a C identifier.
    let tail = &payload[idx + MARKER.len()..];
    let symbol: String = tail
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if symbol.is_empty() {
        return CudaSvdUnavailable::Other(payload.trim().to_string());
    }

    // Library: the last path-looking run before the marker. The payload embeds
    // it as `"<path>: undefined symbol: …"`, and the path may contain `/` and
    // `.` but not whitespace or a quote.
    let head = &payload[..idx];
    let library = head
        .trim_end()
        .trim_end_matches(':')
        .rsplit(|c: char| c.is_whitespace() || c == '"')
        .find(|s| s.contains('/') || s.contains(".so") || s.contains(".dll"))
        .map(|s| s.trim_end_matches(':').to_string());

    CudaSvdUnavailable::MissingSymbol { symbol, library }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The payload from the bug report, **verbatim**, newlines and all.
    ///
    /// Copied rather than paraphrased on purpose: a classifier tested against
    /// a tidied-up version of its input is tested against a string that never
    /// occurs.
    const REPORTED: &str = "Expected symbol in library: DlSym { source: \
        \"/usr/local/cuda/targets/sbsa-linux/lib/libcusolver.so: \
        undefined symbol: cusolverDnGeqrf\" }";

    #[test]
    fn the_reported_cuda13_panic_classifies_as_a_missing_symbol() {
        let got = classify_panic(REPORTED);
        assert_eq!(
            got,
            CudaSvdUnavailable::MissingSymbol {
                symbol: "cusolverDnGeqrf".to_string(),
                library: Some("/usr/local/cuda/targets/sbsa-linux/lib/libcusolver.so".to_string()),
            },
            "the report's own payload must classify; got {got:?}"
        );
    }

    /// The message must carry what an operator needs to act: the symbol, that
    /// CUDA 13 is the cause, and the escape hatch.
    #[test]
    fn the_message_names_the_symbol_the_cause_and_the_way_out() {
        let msg = classify_panic(REPORTED).to_string();
        for needle in [
            "cusolverDnGeqrf",
            "libcusolver.so",
            "CUDA 13",
            "--device cpu",
        ] {
            assert!(
                msg.contains(needle),
                "message must contain '{needle}': {msg}"
            );
        }
        assert!(
            msg.contains("environment condition"),
            "and must say whose problem it is, since the panic read as a crash \
             in the caller's circuit: {msg}"
        );
    }

    /// A payload that is not a symbol failure must NOT be reported as one.
    ///
    /// The failure mode this guards is specific: a classifier that reaches for
    /// `MissingSymbol` on anything it does not recognise would tell an operator
    /// to install a different CUDA over an unrelated fault.
    #[test]
    fn an_unrelated_panic_is_not_dressed_up_as_a_missing_symbol() {
        for payload in [
            "index out of bounds: the len is 2 but the index is 7",
            "called `Option::unwrap()` on a `None` value",
            "CUDA_ERROR_OUT_OF_MEMORY",
            "",
        ] {
            let got = classify_panic(payload);
            assert!(
                matches!(got, CudaSvdUnavailable::Other(_)),
                "{payload:?} must classify as Other, got {got:?}"
            );
        }
    }

    /// Windows and a bare `.dll`, with no directory component.
    #[test]
    fn a_windows_payload_still_yields_the_symbol() {
        let p = "Expected symbol in library: DlSym { source: \
                 \"cusolver64_12.dll: undefined symbol: cusolverDnGeqrf\" }";
        match classify_panic(p) {
            CudaSvdUnavailable::MissingSymbol { symbol, library } => {
                assert_eq!(symbol, "cusolverDnGeqrf");
                assert_eq!(library.as_deref(), Some("cusolver64_12.dll"));
            }
            other => panic!("expected MissingSymbol, got {other:?}"),
        }
    }

    /// A payload with the marker but no library path must still classify —
    /// the symbol is the actionable half.
    #[test]
    fn the_library_is_optional_but_the_symbol_is_not() {
        match classify_panic("undefined symbol: cusolverDnGeqrf") {
            CudaSvdUnavailable::MissingSymbol { symbol, library } => {
                assert_eq!(symbol, "cusolverDnGeqrf");
                assert_eq!(library, None);
            }
            other => panic!("expected MissingSymbol, got {other:?}"),
        }
    }

    /// `NotBuilt` on an ordinary laptop is not something to shout about;
    /// everything else is. The CLI branches on this.
    #[test]
    fn only_host_configuration_problems_are_actionable() {
        assert!(!CudaSvdUnavailable::NotBuilt.is_actionable());
        assert!(CudaSvdUnavailable::NoDevice.is_actionable());
        assert!(classify_panic(REPORTED).is_actionable());
        assert!(CudaSvdUnavailable::Other("boom".into()).is_actionable());
    }
}
