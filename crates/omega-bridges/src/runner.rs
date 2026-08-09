//! Shared subprocess plumbing for Python-backed bridges.
//!
//! Each bridge backend (Qiskit, Perceval, …) is invoked as an external
//! command that consumes a JSON request on stdin and emits a JSON
//! response on stdout. Keeping the integration point at the
//! subprocess boundary means the omega-functions workspace is fully
//! Python-free at the Rust level — no PyO3, no embedded interpreter,
//! no `python3-dev` headers required to build.
//!
//! Discovery order for the runner command (per backend):
//! 1. The `OMEGA_BRIDGE_<NAME>_CMD` env var (full path).
//! 2. `omega-bridge-<name>-runner` on `PATH`.
//! 3. `<workspace>/crates/omega-bridges/python/omega-bridge-<name>-runner`
//!    (only the dev-mode default; works when the binary runs from a
//!    clone of the source tree).
//!
//! The runner protocol is: `{"qasm": "...", "shots": N, "seed": ?, "noise": {...}}`
//! → `{"ok": true, "counts": {"00": 512, ...}}` or
//! `{"ok": false, "error": "...", "kind": "..."}`. The runner exits 0
//! whenever it produces a structured response; non-zero exit means a
//! transport error (missing interpreter, killed signal) and we surface
//! it as `BridgeError::Unavailable`.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::{Backend, BridgeError, Counts, NoiseConfig};

/// Inputs to a runner invocation.
#[allow(dead_code)]
pub(crate) struct RunnerRequest<'a> {
    pub qasm: &'a str,
    pub shots: u32,
    pub noise: Option<&'a NoiseConfig>,
    /// Optional input Fock state for photonic backends (Perceval).
    /// Ignored by gate-based backends like Qiskit. When `Some`, the
    /// Python runner takes the array verbatim as the source modes
    /// occupation list.
    pub input_fock: Option<&'a [u32]>,
}

/// Per-backend identity used to look up the runner command.
pub(crate) struct RunnerSpec {
    pub backend: Backend,
    /// Lowercase, hyphenless backend slug used in env-var names and
    /// PATH-discovered binary names. e.g. "qiskit", "perceval".
    pub slug: &'static str,
}

impl RunnerSpec {
    #[allow(dead_code)]
    pub const fn new(backend: Backend, slug: &'static str) -> Self {
        RunnerSpec { backend, slug }
    }

    fn env_var(&self) -> String {
        format!("OMEGA_BRIDGE_{}_CMD", self.slug.to_ascii_uppercase())
    }

    fn binary_name(&self) -> String {
        format!("omega-bridge-{}-runner", self.slug)
    }
}

/// Resolve the command for a given backend. Returns `None` if no
/// candidate is found; the caller maps that to `Unavailable` with a
/// human-readable hint.
fn locate_command(spec: &RunnerSpec) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(spec.env_var()) {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    if let Ok(p) = which(&spec.binary_name()) {
        return Some(p);
    }
    // Dev-mode default: when the binary is run from inside a checkout
    // of omega-functions, the wrapper sits at a known relative path.
    // CARGO_MANIFEST_DIR is only set during `cargo build`/`cargo test`,
    // but it's the most reliable fingerprint we have during dev.
    let manifest_dir = option_env!("CARGO_MANIFEST_DIR");
    if let Some(dir) = manifest_dir {
        let candidate = PathBuf::from(dir).join("python").join(spec.binary_name());
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Look up `name` on PATH. Trimmed-down replacement for the `which`
/// crate so omega-bridges keeps zero non-workspace deps.
fn which(name: &str) -> Result<PathBuf, ()> {
    let path = std::env::var_os("PATH").ok_or(())?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if let Ok(meta) = std::fs::metadata(&candidate) {
                    if meta.permissions().mode() & 0o111 == 0 {
                        continue;
                    }
                }
            }
            return Ok(candidate);
        }
    }
    Err(())
}

/// Run the QASM2-execute mode against a runner. Thin shim over
/// [`invoke_runner`] that builds the request payload and unwraps
/// the [`ParsedResponse::Counts`] case.
#[allow(dead_code)]
pub(crate) fn run_subprocess(
    spec: &RunnerSpec,
    req: &RunnerRequest<'_>,
) -> Result<Counts, BridgeError> {
    let mut payload = serde_json::Map::new();
    payload.insert("qasm".into(), serde_json::json!(req.qasm));
    payload.insert("shots".into(), serde_json::json!(req.shots));
    if let Some(noise) = req.noise {
        payload.insert("noise".into(), noise.clone());
    }
    if let Some(fock) = req.input_fock {
        payload.insert("input_fock".into(), serde_json::json!(fock));
    }
    let payload_bytes = serde_json::Value::Object(payload).to_string();

    match invoke_runner(spec, &payload_bytes, ParseAs::Counts)? {
        ParsedResponse::Counts(c) => Ok(c),
        other => Err(BridgeError::Backend(
            spec.backend,
            format!("internal: expected counts response, got {other:?}"),
        )),
    }
}

/// What shape of response the caller wants. The runner protocol
/// shares one transport for both modes; this enum picks the right
/// JSON field after the round-trip.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) enum ParseAs {
    Counts,
    Qasm2String,
    QpyBytes,
    /// A list of real numbers, for the `expectation` mode. Distinct from
    /// `Counts` because an expectation value is not a frequency: it is signed,
    /// unnormalised, and carries no shot count, so nothing in the counts path
    /// applies to it.
    Values,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) enum ParsedResponse {
    Counts(Counts),
    Qasm2(String),
    Qpy(Vec<u8>),
    Values(Vec<f64>),
}

#[allow(dead_code)]
pub(crate) fn invoke_runner(
    spec: &RunnerSpec,
    payload: &str,
    parse_as: ParseAs,
) -> Result<ParsedResponse, BridgeError> {
    let cmd = locate_command(spec).ok_or_else(|| {
        BridgeError::Unavailable(
            spec.backend,
            format!(
                "no runner found — set {} or place '{}' on PATH (see crates/omega-bridges/python/Makefile)",
                spec.env_var(),
                spec.binary_name()
            ),
        )
    })?;

    let mut child = Command::new(&cmd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            BridgeError::Unavailable(
                spec.backend,
                format!("failed to spawn {}: {}", cmd.display(), e),
            )
        })?;

    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| BridgeError::Backend(spec.backend, "no stdin handle".into()))?;
        stdin
            .write_all(payload.as_bytes())
            .map_err(|e| BridgeError::Backend(spec.backend, format!("write stdin: {}", e)))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| BridgeError::Backend(spec.backend, format!("wait: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(BridgeError::Unavailable(
            spec.backend,
            format!(
                "runner exited {} — {}",
                output
                    .status
                    .code()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "signal".to_string()),
                stderr.trim()
            ),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let resp: serde_json::Value = serde_json::from_str(stdout.trim()).map_err(|e| {
        BridgeError::Backend(
            spec.backend,
            format!(
                "invalid JSON from runner: {} (output: {})",
                e,
                stdout.trim()
            ),
        )
    })?;

    let ok = resp.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
    if !ok {
        let kind = resp
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("execute");
        let msg = resp
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("(no error message)");
        return Err(classify_failure(spec.backend, kind, msg));
    }

    match parse_as {
        ParseAs::Counts => {
            let counts_obj = resp
                .get("counts")
                .and_then(|v| v.as_object())
                .ok_or_else(|| {
                    BridgeError::Backend(spec.backend, "runner response missing `counts`".into())
                })?;
            let mut counts = Counts::new();
            for (k, v) in counts_obj {
                let n = v.as_u64().ok_or_else(|| {
                    BridgeError::Backend(
                        spec.backend,
                        format!("counts[{}] is not an integer: {}", k, v),
                    )
                })?;
                counts.insert(k.clone(), n as u32);
            }
            Ok(ParsedResponse::Counts(counts))
        }
        ParseAs::Qasm2String => {
            let qasm2 = resp.get("qasm2").and_then(|v| v.as_str()).ok_or_else(|| {
                BridgeError::Backend(spec.backend, "runner response missing `qasm2`".into())
            })?;
            Ok(ParsedResponse::Qasm2(qasm2.to_string()))
        }
        ParseAs::Values => {
            let arr = resp.get("values").and_then(|v| v.as_array()).ok_or_else(|| {
                BridgeError::Backend(spec.backend, "runner response missing `values`".into())
            })?;
            let mut out = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                // Reject non-finite values rather than propagating them. A NaN
                // that reaches a comparison makes every `<= tolerance` test
                // FALSE, so a defect would surface as an unexplained
                // disagreement; and a NaN reaching a `>= ` test would pass.
                let x = v.as_f64().ok_or_else(|| {
                    BridgeError::Backend(
                        spec.backend,
                        format!("values[{i}] is not a number: {v}"),
                    )
                })?;
                if !x.is_finite() {
                    return Err(BridgeError::Backend(
                        spec.backend,
                        format!("values[{i}] is not finite: {x}"),
                    ));
                }
                out.push(x);
            }
            Ok(ParsedResponse::Values(out))
        }
        ParseAs::QpyBytes => {
            use base64::Engine;
            let qpy_b64 = resp
                .get("qpy_b64")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    BridgeError::Backend(spec.backend, "runner response missing `qpy_b64`".into())
                })?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(qpy_b64.as_bytes())
                .map_err(|e| BridgeError::Backend(spec.backend, format!("decode qpy_b64: {e}")))?;
            Ok(ParsedResponse::Qpy(bytes))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests touching `OMEGA_BRIDGE_*_CMD` env vars share a process-
    // wide name table; serialise to keep parallel cargo test happy.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn spec() -> RunnerSpec {
        RunnerSpec::new(Backend::Qiskit, "qiskit")
    }

    #[test]
    fn runner_spec_env_var_uppercases_slug() {
        assert_eq!(spec().env_var(), "OMEGA_BRIDGE_QISKIT_CMD");
        let s = RunnerSpec::new(Backend::Perceval, "perceval-quandela");
        assert_eq!(s.env_var(), "OMEGA_BRIDGE_PERCEVAL-QUANDELA_CMD");
    }

    #[test]
    fn runner_spec_binary_name_uses_slug() {
        assert_eq!(spec().binary_name(), "omega-bridge-qiskit-runner");
    }

    #[cfg(unix)]
    fn write_executable(dir: &std::path::Path, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
        path
    }

    #[test]
    #[cfg(unix)]
    fn locate_command_honours_env_var_first() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let bin = write_executable(
            dir.path(),
            "omega-bridge-qiskit-runner",
            "#!/bin/sh\necho hi\n",
        );
        std::env::set_var("OMEGA_BRIDGE_QISKIT_CMD", &bin);
        let found = locate_command(&spec()).unwrap();
        assert_eq!(found, bin);
        std::env::remove_var("OMEGA_BRIDGE_QISKIT_CMD");
    }

    #[test]
    fn locate_command_returns_none_when_unset_and_unfindable() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Force a slug that won't exist anywhere.
        let bogus = RunnerSpec::new(Backend::Qiskit, "definitely-not-a-real-runner-slug-1234");
        std::env::remove_var(bogus.env_var());
        // CARGO_MANIFEST_DIR is always set during cargo test; confirm
        // that the dev-mode lookup also misses for the bogus slug.
        assert!(locate_command(&bogus).is_none());
    }

    #[test]
    #[cfg(unix)]
    fn which_skips_non_executable_files() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        // Create a file with mode 0o644 (no exec bits) under the temp
        // dir. Add the dir to PATH and verify which() skips it.
        let path = dir.path().join("ghost-runner");
        std::fs::write(&path, "not really a program").unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o644);
        std::fs::set_permissions(&path, perms).unwrap();

        let original = std::env::var_os("PATH").unwrap_or_default();
        let mut entries = std::env::split_paths(&original).collect::<Vec<_>>();
        entries.insert(0, dir.path().to_path_buf());
        let new_path = std::env::join_paths(entries).unwrap();
        std::env::set_var("PATH", &new_path);

        let result = which("ghost-runner");
        std::env::set_var("PATH", original);
        assert!(result.is_err(), "non-exec file must be skipped");
    }

    #[test]
    #[cfg(unix)]
    fn which_finds_executable_on_path() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let bin = write_executable(dir.path(), "happy-runner", "#!/bin/sh\nexit 0\n");

        let original = std::env::var_os("PATH").unwrap_or_default();
        let mut entries = std::env::split_paths(&original).collect::<Vec<_>>();
        entries.insert(0, dir.path().to_path_buf());
        let new_path = std::env::join_paths(entries).unwrap();
        std::env::set_var("PATH", &new_path);

        let result = which("happy-runner");
        std::env::set_var("PATH", original);
        assert_eq!(result.unwrap(), bin);
    }

    #[test]
    fn invoke_runner_missing_command_returns_unavailable() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let bogus = RunnerSpec::new(Backend::Qiskit, "no-such-runner-slug-please");
        std::env::remove_var(bogus.env_var());
        let err = invoke_runner(&bogus, "{}", ParseAs::Counts).expect_err("missing runner");
        match err {
            BridgeError::Unavailable(b, msg) => {
                assert_eq!(b, Backend::Qiskit);
                assert!(msg.contains("no runner found"), "unexpected message: {msg}");
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    #[cfg(unix)]
    fn invoke_runner_parses_counts_from_fake_runner() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let dir = tempfile::tempdir().unwrap();
        // Fake runner: ignore stdin, emit a fixed counts response.
        let body =
            "#!/bin/sh\ncat > /dev/null\necho '{\"ok\":true,\"counts\":{\"00\":7,\"11\":3}}'\n";
        let bin = write_executable(dir.path(), "omega-bridge-qiskit-runner", body);
        std::env::set_var("OMEGA_BRIDGE_QISKIT_CMD", &bin);

        let resp = invoke_runner(&spec(), "{}", ParseAs::Counts).unwrap();
        std::env::remove_var("OMEGA_BRIDGE_QISKIT_CMD");

        match resp {
            ParsedResponse::Counts(c) => {
                assert_eq!(c.get("00").copied(), Some(7));
                assert_eq!(c.get("11").copied(), Some(3));
            }
            other => panic!("expected Counts, got {other:?}"),
        }
    }

    #[test]
    #[cfg(unix)]
    fn invoke_runner_maps_not_installed_kind_to_unavailable() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let dir = tempfile::tempdir().unwrap();
        // Soft-fail message — runner emits ok=false with a kind ending
        // in "-not-installed" (matches the perceval-quandela case).
        let body = "#!/bin/sh\ncat > /dev/null\necho '{\"ok\":false,\"kind\":\"perceval-converter-not-installed\",\"error\":\"missing add-on\"}'\n";
        let bin = write_executable(dir.path(), "omega-bridge-qiskit-runner", body);
        std::env::set_var("OMEGA_BRIDGE_QISKIT_CMD", &bin);

        let err =
            invoke_runner(&spec(), "{}", ParseAs::Counts).expect_err("not-installed must err");
        std::env::remove_var("OMEGA_BRIDGE_QISKIT_CMD");

        match err {
            BridgeError::Unavailable(_, msg) => {
                assert!(msg.contains("missing add-on"), "msg = {msg}");
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    #[cfg(unix)]
    fn invoke_runner_maps_other_failure_kinds_to_backend_error() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let body = "#!/bin/sh\ncat > /dev/null\necho '{\"ok\":false,\"kind\":\"compile\",\"error\":\"bad gate\"}'\n";
        let bin = write_executable(dir.path(), "omega-bridge-qiskit-runner", body);
        std::env::set_var("OMEGA_BRIDGE_QISKIT_CMD", &bin);

        let err = invoke_runner(&spec(), "{}", ParseAs::Counts).expect_err("backend err");
        std::env::remove_var("OMEGA_BRIDGE_QISKIT_CMD");

        match err {
            BridgeError::Backend(_, msg) => {
                assert!(msg.contains("compile"), "msg = {msg}");
                assert!(msg.contains("bad gate"), "msg = {msg}");
            }
            other => panic!("expected Backend, got {other:?}"),
        }
    }

    #[test]
    #[cfg(unix)]
    fn invoke_runner_invalid_json_is_backend_error() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let body = "#!/bin/sh\ncat > /dev/null\necho 'not json at all'\n";
        let bin = write_executable(dir.path(), "omega-bridge-qiskit-runner", body);
        std::env::set_var("OMEGA_BRIDGE_QISKIT_CMD", &bin);

        let err = invoke_runner(&spec(), "{}", ParseAs::Counts).expect_err("bad json");
        std::env::remove_var("OMEGA_BRIDGE_QISKIT_CMD");

        match err {
            BridgeError::Backend(_, msg) => {
                assert!(
                    msg.contains("invalid JSON"),
                    "expected JSON parse error, got {msg}"
                );
            }
            other => panic!("expected Backend, got {other:?}"),
        }
    }

    #[test]
    #[cfg(unix)]
    fn invoke_runner_nonzero_exit_is_unavailable() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let body = "#!/bin/sh\ncat > /dev/null\necho 'transport problem' >&2\nexit 2\n";
        let bin = write_executable(dir.path(), "omega-bridge-qiskit-runner", body);
        std::env::set_var("OMEGA_BRIDGE_QISKIT_CMD", &bin);

        let err = invoke_runner(&spec(), "{}", ParseAs::Counts).expect_err("nonzero exit");
        std::env::remove_var("OMEGA_BRIDGE_QISKIT_CMD");

        match err {
            BridgeError::Unavailable(_, msg) => {
                assert!(msg.contains("runner exited"), "msg = {msg}");
                assert!(msg.contains("transport problem"), "msg = {msg}");
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }
}


/// Map a runner's `kind` string onto a typed [`BridgeError`].
///
/// A cross-backend matrix has to tell three things apart that all look like
/// "no result", and collapsing them is how a matrix reports coverage it does
/// not have:
///
/// 1. **cannot express** — the backend understood the request and CORRECTLY
///    refused (unsupported gate, unsupported noise, a circuit shape it cannot
///    represent). The cell is legitimately empty.
/// 2. **not installed** — environmental; says nothing about the backend.
/// 3. **error** — a real defect.
///
/// Before this existed, only `*-not-installed` was typed; every structured
/// refusal, including `tsim-unsupported-gate` and `ppvm-unsupported-gate`,
/// landed in [`BridgeError::Backend`] with the kind glued into the message, so
/// telling (1) from (3) meant sniffing text.
///
/// Suffix matching rather than an exhaustive list, so a new runner that follows
/// the naming convention is classified correctly without editing this function.
/// The cost is that a badly-named kind is misclassified — hence the convention
/// is documented in `docs/BRIDGES.md` rather than left implicit.
pub(crate) fn classify_failure(backend: Backend, kind: &str, msg: &str) -> BridgeError {
    // Checked FIRST: `-not-installed` is environmental and must not be read as
    // a refusal.
    if kind.ends_with("-not-installed") {
        return BridgeError::Unavailable(backend, msg.to_string());
    }
    if kind.ends_with("-unsupported-gate")
        || kind.ends_with("-not-supported")
        || kind.ends_with("-not-implemented")
    {
        return BridgeError::CannotExpress(backend, format!("{kind}: {msg}"));
    }
    BridgeError::Backend(backend, format!("{kind}: {msg}"))
}

#[cfg(test)]
mod classify_tests {
    use super::*;

    /// Every `kind` the Python runners actually emit, classified.
    ///
    /// The list is taken from a grep of `crates/omega-bridges/python/*.py`, not
    /// invented — a classifier tested only against kinds someone imagined is
    /// exactly the kind of check this repository keeps finding to be hollow.
    #[test]
    fn every_real_runner_kind_lands_in_the_right_bucket() {
        let cannot_express = [
            "tsim-unsupported-gate",
            "ppvm-unsupported-gate",
            "tsim-noise-not-supported",
            "ppvm-noise-not-supported",
            "bloqade-multi-creg-not-supported",
            "bloqade-ahs-not-implemented",
        ];
        let unavailable = [
            "qiskit-not-installed",
            "perceval-not-installed",
            "perceval-converter-not-installed",
            "bloqade-not-installed",
            "tsim-not-installed",
            "ppvm-not-installed",
        ];
        let real_errors = [
            "bad-request",
            "execute",
            "tsim-execute",
            "ppvm-execute",
            "tsim-lower",
            "ppvm-lower",
            "bloqade-lower",
            "bloqade-execute",
            "tsim-internal",
            "ppvm-internal",
            "qasm-parse",
            "qasm-emit",
            "qpy-parse",
            "qpy-emit",
            "qpy-multi-circuit",
            "opticqasm-parse",
            "opticqasm-build",
            "noise-model",
        ];

        for k in cannot_express {
            assert!(
                matches!(
                    classify_failure(Backend::Tsim, k, "x"),
                    BridgeError::CannotExpress(..)
                ),
                "{k} must classify as CannotExpress — a correct refusal is not a defect"
            );
        }
        for k in unavailable {
            assert!(
                matches!(
                    classify_failure(Backend::Tsim, k, "x"),
                    BridgeError::Unavailable(..)
                ),
                "{k} must classify as Unavailable — environmental, not a refusal"
            );
        }
        for k in real_errors {
            assert!(
                matches!(
                    classify_failure(Backend::Tsim, k, "x"),
                    BridgeError::Backend(..)
                ),
                "{k} must classify as Backend — a real failure must not be excused \
                 as 'cannot express', which would hide a defect as coverage"
            );
        }
    }

    /// `-not-installed` must win over the refusal suffixes.
    ///
    /// Not currently reachable (no kind ends with both), but the ordering is
    /// load-bearing and a future kind could collide.
    #[test]
    fn not_installed_is_checked_before_the_refusal_suffixes() {
        assert!(matches!(
            classify_failure(Backend::Tsim, "weird-not-supported-not-installed", "x"),
            BridgeError::Unavailable(..)
        ));
    }
}
