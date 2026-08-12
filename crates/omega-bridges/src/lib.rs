//! Run QASM2 circuits through external quantum simulators / hardware
//! backends.
//!
//! Reference design: `../QuetzalcoatlProto/backend/backend.py`. That
//! module exposes one dispatcher `run_qasm2(name, qasm, shots, noise)`
//! that routes to seven optional sub-backends, each loaded under a
//! `try-import + env-disable` guard so a missing toolchain never breaks
//! a build. We mirror that ergonomics in Rust:
//!
//! - Each backend lives behind a Cargo feature (`bridge-qiskit`,
//!   `bridge-perceval`, …). The default build pulls in nothing
//!   external; `cargo build --workspace` stays clean on hosts without
//!   Python / Qiskit / Perceval / Quandela installed.
//! - The dispatcher returns [`BridgeError::NotCompiled`] for any
//!   backend whose feature isn't compiled in, [`BridgeError::Unavailable`]
//!   when the toolchain is feature-compiled but missing on the host
//!   (Python interpreter not found, REST endpoint unreachable), and
//!   [`BridgeError::Backend`] for runtime errors from the backend itself.
//! - Comparison harness (separate test target) reuses the
//!   `verify-qiskit/` 369-fixture corpus and asserts per-backend L2
//!   distance vs Qiskit at thresholds matching the reference Python
//!   harness.
//!
//! Today the crate ships:
//! - `bridge-qiskit`: subprocess runner (Python venv).
//! - `bridge-perceval`: subprocess runner (Python venv).
//! - `bridge-bloqade`: subprocess runner (Python venv), gate mode.
//! - `bridge-tsim`: subprocess runner (Python venv). QuEra's
//!   ZX-calculus stabilizer-rank sampler.
//! - `bridge-ppvm`: subprocess runner (Python venv). QuEra's
//!   generalized-stabilizer-tableau sampler.
//!
//! Cirq / qadence are **deferred** — their feature flags stay reserved
//! so the dispatch surface doesn't churn, but the implementations only
//! land when a concrete user need shows up. Each must plug into
//! `tests/cross_backend.rs` at the `backend.py` thresholds before
//! flipping out of "deferred".
//!
//! qpanda2 / spinqit / IonQ-REST / Rigetti-REST / QuEra-via-Braket were
//! **dropped 2026-05-01** — feature flags, source modules, and Backend
//! variants removed in this commit. If a real need surfaces, add a
//! fresh feature for that backend rather than resurrecting the
//! scaffold. The replacement coupling for QuEra Aquila is the
//! Bloqade Python stack (gate + analog), planned under feature
//! `bridge-bloqade`.

use omega_core::error::OmegaError;
use std::collections::HashMap;
use thiserror::Error;

/// Counts dictionary: outcome bit-string (LSB-first) → frequency.
pub type Counts = HashMap<String, u32>;

/// One term of an observable: a dense Pauli string and a real coefficient.
///
/// The string is **LSB-first** — leftmost character is qubit 0 — and its
/// length must equal the circuit's qubit count. `I` pads unused qubits.
///
/// This is the same direction as [`Counts`], and the same direction as Stim's
/// `PauliString` and ppvm's `PauliSum`. It is the OPPOSITE of Qiskit's
/// `SparsePauliOp`, which is MSB-first; the Qiskit runner reverses on the way
/// in. Measured on `x q[0]`: `SparsePauliOp("IZ")` = −1 while our `"ZI"` = −1,
/// both naming qubit 0.
///
/// Choosing one direction for the whole wire means a reader never has to ask
/// which surface they are looking at. The cost is one reversal, in one place,
/// with a test that pins it on an asymmetric term.
pub type PauliTerm = (String, f64);

/// A weighted sum of Pauli strings: `Σ coeff · pauli`.
pub type WireObservable = Vec<PauliTerm>;

/// Backend identifier. Stable strings so the same name routes through
/// the CLI, REST API, and library entry points without translation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    Qiskit,
    Perceval,
    /// QuEra Aquila via the Python `bloqade` (analog AHS) and
    /// `bloqade-circuit` (gate mode) stack. Today only gate mode
    /// returns counts; analog AHS responds with `Unavailable` until
    /// its input shape lands.
    Bloqade,
    /// QuEra `ppvm` — Heisenberg Pauli-sum propagation with magnitude
    /// truncation, plus a generalized stabilizer tableau.
    ///
    /// **This is the same algorithm family as the in-tree
    /// `omega-backend-pauliprop`**, so it exists here as an INDEPENDENT
    /// NUMERIC REFERENCE that validates pauliprop — not as a new capability
    /// users pick for its own sake. Two implementations of the same idea
    /// agreeing is real evidence; one implementation agreeing with itself is
    /// not. See `fixes/TSIM-PPVM.md` and `FIXES_PLAN.md` Part E.
    Ppvm,
    /// QuEra `tsim` — ZX stabilizer-rank decomposition sampler, for noisy
    /// Clifford+T circuits at scales the MPS backend cannot reach.
    ///
    /// Reached through the QASM2 + counts bridge protocol, which means tsim's
    /// distinctive QEC feature — detector and observable sampling — is **not**
    /// expressible here. It arrives as a plain noisy sampler; a detector-aware
    /// surface is separate work.
    Tsim,
    Cirq,
    Qadence,
}

impl Backend {
    /// Parse the on-wire backend name. Mirrors the strings used in
    /// `../QuetzalcoatlProto/backend/backend.py::run_qasm2`.
    pub fn parse(name: &str) -> Result<Self, BridgeError> {
        Ok(match name.to_ascii_lowercase().as_str() {
            "qiskit" => Backend::Qiskit,
            "perceval" => Backend::Perceval,
            "bloqade" | "quera" | "aquila" => Backend::Bloqade,
            "ppvm" => Backend::Ppvm,
            "tsim" | "bloqade-tsim" => Backend::Tsim,
            "cirq" | "circ" => Backend::Cirq,
            "qadence" => Backend::Qadence,
            other => return Err(BridgeError::UnknownBackend(other.to_string())),
        })
    }

    /// Whether this backend's feature flag is compiled into the current
    /// binary. Missing toolchains at runtime are reported as
    /// [`BridgeError::Unavailable`] separately.
    pub fn is_compiled(&self) -> bool {
        match self {
            Backend::Qiskit => cfg!(feature = "bridge-qiskit"),
            Backend::Perceval => cfg!(feature = "bridge-perceval"),
            Backend::Bloqade => cfg!(feature = "bridge-bloqade"),
            Backend::Ppvm => cfg!(feature = "bridge-ppvm"),
            Backend::Tsim => cfg!(feature = "bridge-tsim"),
            Backend::Cirq => cfg!(feature = "bridge-cirq"),
            Backend::Qadence => cfg!(feature = "bridge-qadence"),
        }
    }

    /// All known backend names; useful for `omega-client backends list-bridges`.
    pub fn all_names() -> &'static [&'static str] {
        &[
            "qiskit", "perceval", "bloqade", "ppvm", "tsim", "cirq", "qadence",
        ]
    }
}

/// Optional noise-model configuration forwarded to the backend.
/// Today this is opaque JSON; each backend consumes its own subset.
/// Mirrors the dict that `backend.py` passes through unchanged.
pub type NoiseConfig = serde_json::Value;

/// Errors surfaced by the dispatcher. These map cleanly to HTTP status
/// codes when wrapped in a future REST endpoint:
///
/// - `UnknownBackend` → 404
/// - `NotCompiled` → 501 (Not Implemented)
/// - `Unavailable` → 503 (Service Unavailable)
/// - `Backend` → 500 (Internal Server Error)
#[derive(Debug, Error)]
pub enum BridgeError {
    #[error("unknown backend `{0}`")]
    UnknownBackend(String),
    #[error("backend `{0:?}` is not compiled in (rebuild with --features bridge-{1})")]
    NotCompiled(Backend, &'static str),
    #[error("backend `{0:?}` is feature-compiled but unavailable at runtime: {1}")]
    Unavailable(Backend, String),
    /// The backend ran, understood the request, and **correctly refused** it —
    /// an unsupported gate, an unsupported noise model, a circuit shape it
    /// cannot represent.
    ///
    /// Distinct from [`BridgeError::Backend`] on purpose. A cross-backend
    /// matrix has to tell three things apart that all look like "no result":
    ///
    /// 1. **cannot express** — a correct refusal. The cell is legitimately empty.
    /// 2. **not installed** ([`BridgeError::Unavailable`]) — environmental.
    /// 3. **error** ([`BridgeError::Backend`]) — a real defect.
    ///
    /// Collapsing these is how a matrix reports coverage it does not have.
    /// Before this variant existed, every structured refusal — including
    /// `tsim-unsupported-gate` and `ppvm-unsupported-gate` — landed in
    /// `Backend(..)` with the kind glued into the message string, so telling
    /// (1) from (3) meant sniffing text.
    #[error("backend `{0:?}` cannot express this circuit: {1}")]
    CannotExpress(Backend, String),
    #[error("backend `{0:?}` returned an error: {1}")]
    Backend(Backend, String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

impl From<BridgeError> for OmegaError {
    fn from(e: BridgeError) -> Self {
        OmegaError::Backend(e.to_string())
    }
}

/// Run a QASM2 source through the named backend.
///
/// Returns the bit-string counts dictionary. The dispatcher does not
/// touch `qasm` other than passing it through; backends are responsible
/// for their own parsing (Qiskit's `qasm2.loads`, Perceval's converter,
/// etc.) so this crate doesn't impose omega-parser's QASM2 subset.
pub fn run_qasm2(
    backend: Backend,
    qasm: &str,
    shots: u32,
    noise: Option<&NoiseConfig>,
) -> Result<Counts, BridgeError> {
    if shots == 0 {
        return Err(BridgeError::InvalidInput("shots must be > 0".to_string()));
    }
    match backend {
        Backend::Qiskit => qiskit::run(qasm, shots, noise),
        Backend::Perceval => perceval::run(qasm, shots, noise),
        Backend::Bloqade => bloqade::run(qasm, shots, noise),
        Backend::Ppvm => ppvm::run(qasm, shots, noise),
        Backend::Tsim => tsim::run(qasm, shots, noise),
        Backend::Cirq => cirq::run(qasm, shots, noise),
        Backend::Qadence => qadence::run(qasm, shots, noise),
    }
}

/// Exact expectation values of `observables` on the state `qasm` prepares.
///
/// Distinct from [`run_qasm2`] in kind, not just in return type: this is an
/// **analytic** query with no shots, so results are compared at ~1e-12 rather
/// than at a sampling tolerance (Part K2: "analytic vs stochastic must not
/// share a tolerance").
///
/// Only Qiskit implements it today. Every other backend returns
/// `CannotExpress` rather than `NotCompiled`, because the distinction matters:
/// the feature is compiled, the backend simply has no expectation path through
/// this protocol, and a matrix that reported that as "not installed" would be
/// claiming an environmental excuse for a capability gap.
pub fn expectation_qasm2(
    backend: Backend,
    qasm: &str,
    observables: &[WireObservable],
) -> Result<Vec<f64>, BridgeError> {
    match backend {
        Backend::Qiskit => qiskit::expectation(qasm, observables),
        // ppvm takes no truncation here; `expectation_qasm2_truncated` is the
        // entry point for that. Keeping the common call simple means a caller
        // cannot silently get a truncated answer when they asked for an exact
        // one.
        Backend::Ppvm => ppvm::expectation(qasm, observables, None),
        // Stim's tableau, reached through the tsim venv — exact integers, and
        // Clifford-only by construction. See `tsim::expectation`.
        Backend::Tsim => tsim::expectation(qasm, observables),
        other => Err(BridgeError::CannotExpress(
            other,
            format!(
                "{other:?} has no expectation mode: the bridge protocol carries \
                 QASM2-in / counts-out for every backend but Qiskit"
            ),
        )),
    }
}

/// Convenience: parse a backend name and dispatch.
pub fn run_qasm2_named(
    name: &str,
    qasm: &str,
    shots: u32,
    noise: Option<&NoiseConfig>,
) -> Result<Counts, BridgeError> {
    let backend = Backend::parse(name)?;
    run_qasm2(backend, qasm, shots, noise)
}

/// Decode a Qiskit QPY blob into the equivalent QASM2 source.
///
/// Routes through the Qiskit subprocess, which uses
/// `qiskit.qpy.load` + `qasm2.dumps` to bridge the format. Whatever
/// QPY versions the operator's installed Qiskit handles, omega
/// handles. The blob must contain exactly one circuit; multi-
/// circuit blobs surface a typed error so the caller doesn't
/// silently get only the first.
///
/// Today only [`Backend::Qiskit`] is supported. Other backends
/// either don't speak QPY (Perceval) or aren't gate-based.
pub fn qpy_to_qasm2(qpy_bytes: &[u8]) -> Result<String, BridgeError> {
    qiskit::qpy_to_qasm2(qpy_bytes)
}

/// Encode a QASM2 source as a QPY blob via the Qiskit subprocess.
/// Symmetric to [`qpy_to_qasm2`]: lets omega-cli / omega-server
/// hand `.qpy` artefacts to downstream Qiskit-only tooling without
/// the operator reaching for a Python venv themselves. The returned
/// bytes start with the `b"QISKIT"` magic header so a caller can
/// round-trip `omega-parsed QASM → QPY → CircuitIR` against the
/// pure-Rust reader to validate.
pub fn qasm2_to_qpy(qasm: &str) -> Result<Vec<u8>, BridgeError> {
    qiskit::qasm2_to_qpy(qasm)
}

// Re-export the QPY magic-byte check from the new pure-Rust qpy
// module so the existing `omega_bridges::is_qpy` callers don't move.
pub use qpy::is_qpy;

/// Run a native OPTICQASM source through a photonic backend.
///
/// Output counts are keyed by Fock-state strings (comma-separated
/// occupation numbers per mode, e.g. `"1,0,1,0"`) rather than the
/// LSB-first qubit bit-strings used by gate-based runners — the
/// natural shape for a photonic simulator. Today only [`Backend::Perceval`]
/// is supported; gate-based backends return [`BridgeError::InvalidInput`].
pub fn run_opticqasm(
    backend: Backend,
    source: &str,
    shots: u32,
    input_fock: Option<&[u32]>,
    noise: Option<&NoiseConfig>,
) -> Result<Counts, BridgeError> {
    if shots == 0 {
        return Err(BridgeError::InvalidInput("shots must be > 0".to_string()));
    }
    match backend {
        Backend::Perceval => perceval::run_opticqasm(source, shots, input_fock, noise),
        other => Err(BridgeError::InvalidInput(format!(
            "OPTICQASM is only supported on photonic backends; \
             {other:?} is gate-based or remote-only"
        ))),
    }
}

mod bloqade;
mod cirq;
// The shared cross-check corpus locator. Public and in the library (not in a
// test file) because two harnesses in two different crates must agree on
// *which* corpus ran — see the module docs.
pub mod corpus;
mod perceval;
mod ppvm;
mod qadence;
mod qiskit;
pub mod qpy;
mod tsim;
// `runner` is the shared subprocess plumbing for Python-backed
// bridges. We compile it unconditionally (rather than feature-gating)
// so its unit tests run under the default `cargo test --workspace`
// invocation in `ci.sh`. Items inside the module already carry
// `#[allow(dead_code)]` for the case where no Python bridge feature
// is enabled.
mod runner;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_known_backend_names() {
        assert_eq!(Backend::parse("qiskit").unwrap(), Backend::Qiskit);
        assert_eq!(Backend::parse("Qiskit").unwrap(), Backend::Qiskit);
        assert_eq!(Backend::parse("perceval").unwrap(), Backend::Perceval);
        assert_eq!(Backend::parse("bloqade").unwrap(), Backend::Bloqade);
        assert_eq!(Backend::parse("quera").unwrap(), Backend::Bloqade);
        assert_eq!(Backend::parse("aquila").unwrap(), Backend::Bloqade);
        assert_eq!(Backend::parse("tsim").unwrap(), Backend::Tsim);
        assert_eq!(Backend::parse("TSim").unwrap(), Backend::Tsim);
        assert_eq!(Backend::parse("bloqade-tsim").unwrap(), Backend::Tsim);
        assert_eq!(Backend::parse("ppvm").unwrap(), Backend::Ppvm);
        assert_eq!(Backend::parse("PPVM").unwrap(), Backend::Ppvm);
        assert_eq!(Backend::parse("cirq").unwrap(), Backend::Cirq);
        assert_eq!(Backend::parse("circ").unwrap(), Backend::Cirq);
        assert_eq!(Backend::parse("qadence").unwrap(), Backend::Qadence);
    }

    #[test]
    fn parse_unknown_backend_rejected() {
        match Backend::parse("oxford-quantum") {
            Err(BridgeError::UnknownBackend(s)) => assert_eq!(s, "oxford-quantum"),
            other => panic!("expected UnknownBackend, got {:?}", other),
        }
    }

    #[test]
    fn dispatch_returns_not_compiled_when_feature_off() {
        // CPU-only build: every backend should fail with NotCompiled
        // since no `bridge-*` features are enabled in the default build.
        // (If you're running this with --features bridge-foo, the test is
        // a no-op for that backend — feature-gate the assertion.)
        for name in Backend::all_names() {
            let backend = Backend::parse(name).unwrap();
            if backend.is_compiled() {
                continue;
            }
            let qasm = "OPENQASM 2.0; qreg q[1]; h q[0];";
            let res = run_qasm2(backend, qasm, 100, None);
            match res {
                Err(BridgeError::NotCompiled(b, _)) => assert_eq!(b, backend),
                other => panic!(
                    "expected NotCompiled for {} (feature off), got {:?}",
                    name, other
                ),
            }
        }
    }

    #[test]
    fn dispatch_rejects_zero_shots() {
        let res = run_qasm2(Backend::Qiskit, "OPENQASM 2.0;", 0, None);
        assert!(matches!(res, Err(BridgeError::InvalidInput(_))));
    }

    #[test]
    fn is_qpy_magic_bytes_detection() {
        // The Qiskit QPY header is `b"QISKIT"` (6 bytes) + a u8 version.
        let qpy_bytes: &[u8] = b"QISKIT\x0c\x00\x00\x00\x00";
        assert!(is_qpy(qpy_bytes));
        // Less than 6 bytes → not a QPY blob.
        assert!(!is_qpy(b"QISKI"));
        assert!(!is_qpy(b""));
        // 6 bytes but wrong magic.
        assert!(!is_qpy(b"OPENQA"));
        // Exact magic with no trailing version byte still passes the
        // header check — the version byte is consumed by the runner,
        // not by the magic-bytes test.
        assert!(is_qpy(b"QISKIT"));
    }

    #[test]
    fn run_opticqasm_rejects_zero_shots() {
        let res = run_opticqasm(Backend::Perceval, "OPTICQASM 1.0;", 0, None, None);
        match res {
            Err(BridgeError::InvalidInput(msg)) => {
                assert!(msg.contains("shots"), "msg: {msg}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn run_opticqasm_rejects_gate_based_backends() {
        // OPTICQASM is photonic-only; routing it to Qiskit must fail
        // up-front with a clear message rather than going through the
        // wire.
        let res = run_opticqasm(Backend::Qiskit, "OPTICQASM 1.0;", 100, None, None);
        match res {
            Err(BridgeError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("photonic") || msg.contains("OPTICQASM"),
                    "msg: {msg}"
                );
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn backend_all_names_covers_every_variant() {
        // Sanity: every Backend variant the dispatcher knows about
        // must be reachable via parse(). A new variant added without
        // also being added to all_names would silently never get a
        // soft-fail check in the loop.
        for name in Backend::all_names() {
            assert!(
                Backend::parse(name).is_ok(),
                "all_names() advertises {name:?} but parse() rejects it"
            );
        }
        // And the surface should be non-empty (regression guard
        // against accidental list-truncation).
        assert!(Backend::all_names().len() >= 7);
    }

    #[test]
    fn run_qasm2_named_routes_through_parse() {
        // The named variant is the wire-friendly form used by the
        // future REST surface. Verify it returns the same NotCompiled
        // error as the typed dispatch when no feature is enabled.
        let res = run_qasm2_named("qiskit", "OPENQASM 2.0; qreg q[1];", 100, None);
        if !Backend::Qiskit.is_compiled() {
            assert!(matches!(res, Err(BridgeError::NotCompiled(_, _))));
        }
    }

    #[test]
    fn run_qasm2_named_unknown_returns_unknown_backend() {
        let res = run_qasm2_named("vacuum-tube", "OPENQASM 2.0;", 100, None);
        match res {
            Err(BridgeError::UnknownBackend(s)) => assert_eq!(s, "vacuum-tube"),
            other => panic!("expected UnknownBackend, got {other:?}"),
        }
    }
}
