// SPDX-License-Identifier: Apache-2.0
//! The FQE probe. One attempt of the Makefile target. A failure to install
//! is a result, recorded with the error, not a retry loop.

use std::process::Command;

/// What the probe actually produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FqeProbe {
    /// Import succeeded. This lane did not get here on andromeda.
    Available {
        /// `fqe.__version__` or the distribution version.
        version: String,
    },
    /// The venv is not usable. `error` is the command output.
    Unavailable {
        /// Combined stdout and stderr, trimmed.
        error: String,
    },
}

/// Run `make fqe-venv` once in the bridges' Python directory.
pub fn probe() -> FqeProbe {
    let dir = crate::workload::repo_root().join("crates/omega-bridges/python");
    let output = Command::new("make")
        .arg("fqe-venv")
        .current_dir(&dir)
        .output();
    match output {
        Err(e) => FqeProbe::Unavailable {
            error: format!("make fqe-venv could not be started: {e}"),
        },
        Ok(out) if out.status.success() => {
            let python = dir.join(".venv-fqe/bin/python");
            if !python.is_file() {
                return FqeProbe::Unavailable {
                    error: format!("make fqe-venv exited 0 but {} is missing", python.display()),
                };
            }
            let imported = Command::new(&python)
                .args([
                    "-c",
                    "import fqe, importlib.metadata as m; print(m.version('fqe'))",
                ])
                .output();
            match imported {
                Ok(imp) if imp.status.success() => FqeProbe::Available {
                    version: String::from_utf8_lossy(&imp.stdout).trim().to_string(),
                },
                Ok(imp) => FqeProbe::Unavailable {
                    error: format!(
                        "fqe import failed: {}",
                        String::from_utf8_lossy(&imp.stderr).trim()
                    ),
                },
                Err(e) => FqeProbe::Unavailable {
                    error: format!("fqe import could not start: {e}"),
                },
            }
        }
        Ok(out) => {
            let mut error = String::new();
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stdout = String::from_utf8_lossy(&out.stdout);
            error.push_str(stderr.trim());
            if !stdout.trim().is_empty() {
                if !error.is_empty() {
                    error.push('\n');
                }
                error.push_str(stdout.trim());
            }
            if error.is_empty() {
                error = format!("make fqe-venv exited {}", out.status);
            }
            FqeProbe::Unavailable { error }
        }
    }
}

/// A probe that says nothing is not a result. An available probe needs a version.
pub fn require_result(probe: &FqeProbe) -> Result<String, String> {
    match probe {
        FqeProbe::Available { version } if version.is_empty() => {
            Err("fqe imported with an empty version".into())
        }
        FqeProbe::Available { version } => Ok(format!(
            "fqe {version} imports. It can second-opinion a number-conserving operator. \
             Kitaev pairing still does not conserve particle number, so no (n_alpha, n_beta) \
             sector contains that state."
        )),
        FqeProbe::Unavailable { error } if error.trim().is_empty() => {
            Err("fqe probe produced an empty error".into())
        }
        FqeProbe::Unavailable { error } => Ok(format!(
            "FQE is not installed. make fqe-venv failed: {error}. It would take uv and \
             CPython 3.11 (FQE 0.3.0 is a Cython sdist that the Makefile pins to 3.11, \
             numpy<2, installed --no-deps), then `make -C crates/omega-bridges/python fqe-venv`. \
             Hubbard, H2 and LUCJ conserve particle number, so that venv could second-opinion \
             their values. Kitaev pairing changes particle number, so a sector wavefunction \
             would not contain the state even after the install."
        )),
    }
}
