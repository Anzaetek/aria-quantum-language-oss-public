//! Compute device selection — the cross-cutting invariant for the
//! GPU plan.
//!
//! Every backend that has (or will have) a GPU implementation accepts
//! a `DeviceKind` so callers can ask for `Metal`, `Cuda`, or `OpenCL`
//! and gracefully fall back to `Cpu` when the selected device isn't
//! compiled in or isn't usable at runtime.
//!
//! Selection rules in `DeviceKind::resolve`:
//! 1. Honour an explicit `requested` argument when supplied.
//! 2. Otherwise read the `OMEGA_DEVICE` env var.
//! 3. Otherwise default to `Cpu`.
//!
//! The `available()` table is set at compile time via Cargo features.
//! When the caller asks for an unavailable device, `resolve` falls back
//! to `Cpu` *and emits a single stderr notice*, never panics.

use std::sync::atomic::{AtomicBool, Ordering};

/// Compute device used by the backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    /// Reference CPU implementation. Always available.
    #[default]
    Cpu,
    /// Apple Silicon Metal compute. Compiled in via `--features metal`.
    Metal,
    /// NVIDIA CUDA. Compiled in via `--features cuda`.
    Cuda,
    /// Cross-vendor OpenCL. Compiled in via `--features opencl`.
    OpenCl,
}

impl DeviceKind {
    /// Parse the on-wire device name. Mirrors the strings used in the
    /// CLI (`--device cpu|metal|cuda|opencl`) and the `OMEGA_DEVICE`
    /// env var.
    pub fn parse(name: &str) -> Result<Self, String> {
        Ok(match name.to_ascii_lowercase().as_str() {
            "cpu" => Self::Cpu,
            "metal" | "mps" => Self::Metal,
            "cuda" | "nvidia" => Self::Cuda,
            "opencl" | "ocl" => Self::OpenCl,
            other => return Err(format!("unknown device `{other}`")),
        })
    }

    /// Lower-case canonical name used for stderr notices and config files.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::OpenCl => "opencl",
        }
    }

    /// Whether the device's feature flag is compiled into the current
    /// binary. The CPU backend is always available.
    pub fn is_available(&self) -> bool {
        match self {
            Self::Cpu => true,
            Self::Metal => cfg!(feature = "metal"),
            Self::Cuda => cfg!(feature = "cuda"),
            Self::OpenCl => cfg!(feature = "opencl"),
        }
    }

    /// Resolve the device the backend should actually use.
    ///
    /// `requested` takes precedence; if `None`, the `OMEGA_DEVICE` env
    /// var is consulted; if neither, the result is `Cpu`. If the
    /// resolved device isn't available at compile time, this returns
    /// `Cpu` and emits a single stderr notice. Never panics.
    pub fn resolve(requested: Option<DeviceKind>) -> Self {
        let asked = requested
            .or_else(|| {
                std::env::var("OMEGA_DEVICE")
                    .ok()
                    .and_then(|s| Self::parse(&s).ok())
            })
            .unwrap_or(Self::Cpu);
        if asked.is_available() {
            return asked;
        }
        if asked != Self::Cpu {
            notice_once(asked);
        }
        Self::Cpu
    }
}

// One-shot stderr notice the first time we fall back. Avoids spamming
// the user when a hot loop calls `resolve` a million times.
static FALLBACK_NOTICE_FIRED: AtomicBool = AtomicBool::new(false);

fn notice_once(asked: DeviceKind) {
    if FALLBACK_NOTICE_FIRED.swap(true, Ordering::SeqCst) {
        return;
    }
    eprintln!(
        "omega: requested device `{}` not compiled in; falling back to cpu \
         (rebuild with --features {})",
        asked.name(),
        asked.name(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_known_names() {
        assert_eq!(DeviceKind::parse("cpu").unwrap(), DeviceKind::Cpu);
        assert_eq!(DeviceKind::parse("Metal").unwrap(), DeviceKind::Metal);
        assert_eq!(DeviceKind::parse("MPS").unwrap(), DeviceKind::Metal);
        assert_eq!(DeviceKind::parse("cuda").unwrap(), DeviceKind::Cuda);
        assert_eq!(DeviceKind::parse("OPENCL").unwrap(), DeviceKind::OpenCl);
    }

    #[test]
    fn parse_unknown_name_rejected() {
        assert!(DeviceKind::parse("vulkan").is_err());
    }

    #[test]
    fn cpu_is_always_available() {
        assert!(DeviceKind::Cpu.is_available());
    }

    #[test]
    fn resolve_falls_back_to_cpu_when_unavailable() {
        // In the default build no GPU feature is enabled, so asking for
        // anything but Cpu must come back as Cpu.
        if !DeviceKind::Metal.is_available() {
            assert_eq!(
                DeviceKind::resolve(Some(DeviceKind::Metal)),
                DeviceKind::Cpu
            );
        }
        if !DeviceKind::Cuda.is_available() {
            assert_eq!(DeviceKind::resolve(Some(DeviceKind::Cuda)), DeviceKind::Cpu);
        }
    }

    #[test]
    fn resolve_default_is_cpu() {
        assert_eq!(DeviceKind::resolve(None), DeviceKind::Cpu);
    }
}
