//! Trust store for PKI chain validation (Phase 10).
//!
//! At boot the server reads zero or more CBOR-encoded `OmegaCert`
//! files from a directory and pins them as trust roots. The
//! handshake verifier then anchors every presented chain in this
//! set via [`certificate::verify_chain`].
//!
//! Discovery rules:
//! - Path comes from `OMEGA_PKI_TRUST_STORE` (env) or an explicit
//!   `Path` passed by the caller. A missing env var is fine — the
//!   trust store is empty and chain validation will reject every
//!   chain (`unknown root`).
//! - Every regular file with a `.cbor` extension is parsed via
//!   [`OmegaCert::from_cbor`]. Other files are skipped (not an
//!   error — operators may keep notes / READMEs alongside).
//! - Each parsed cert is required to self-verify at load time.
//!   Tampered or expired roots fail the load and the operator must
//!   fix them before the server can boot. Fail-fast over fail-soft
//!   here — a corrupt trust store is a security event.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::pqc::certificate::{verify_chain, OmegaCert};

/// Set of trusted root certificates. Indexed by serial for O(1)
/// chain anchoring.
#[derive(Debug)]
pub struct TrustStore {
    /// Source directory (kept for diagnostics; `None` for an
    /// in-memory store built via [`TrustStore::from_certs`]).
    source_dir: Option<PathBuf>,
    /// Roots keyed by `body.serial`. The serial is a 16-byte UUID;
    /// using the byte vec as the key is fine — `Vec<u8>` hashes
    /// fine and the lookup cost is irrelevant next to the
    /// signature verification cost.
    roots: HashMap<Vec<u8>, OmegaCert>,
}

impl TrustStore {
    /// Empty store — every chain validation will fail. Useful as a
    /// fallback when the env var isn't set.
    pub fn empty() -> Self {
        TrustStore {
            source_dir: None,
            roots: HashMap::new(),
        }
    }

    /// Build a store directly from in-memory certs (tests / hard-
    /// coded bootstrap).
    #[allow(dead_code)]
    pub fn from_certs(certs: Vec<OmegaCert>) -> Result<Self, String> {
        for c in &certs {
            c.verify().map_err(|e| {
                format!(
                    "trust root (subject = {:?}) failed self-verification: {e}",
                    c.body.subject
                )
            })?;
        }
        let mut roots = HashMap::new();
        for c in certs {
            roots.insert(c.body.serial.clone(), c);
        }
        Ok(TrustStore {
            source_dir: None,
            roots,
        })
    }

    /// Load every `.cbor` file under `dir` as a trust root. Each
    /// loaded cert is required to self-verify; the first failure
    /// surfaces and the load aborts.
    #[allow(dead_code)]
    pub fn load_dir(dir: &Path) -> Result<Self, String> {
        let entries = fs::read_dir(dir)
            .map_err(|e| format!("read trust-store dir {}: {e}", dir.display()))?;
        let mut roots = HashMap::new();
        for entry in entries {
            let entry = entry.map_err(|e| format!("read entry under {}: {e}", dir.display()))?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("cbor") {
                continue;
            }
            let bytes = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
            let cert = OmegaCert::from_cbor(&bytes)
                .map_err(|e| format!("parse {} as OmegaCert: {e}", path.display()))?;
            cert.verify().map_err(|e| {
                format!(
                    "trust root {} (subject = {:?}) failed self-verification: {e}",
                    path.display(),
                    cert.body.subject
                )
            })?;
            if roots.insert(cert.body.serial.clone(), cert).is_some() {
                return Err(format!(
                    "duplicate serial in trust store at {}",
                    path.display()
                ));
            }
        }
        Ok(TrustStore {
            source_dir: Some(dir.to_path_buf()),
            roots,
        })
    }

    /// Load from `OMEGA_PKI_TRUST_STORE`, or return an empty store
    /// if the env var is unset / empty. Honouring the env var here
    /// keeps the boot path consistent with other `OMEGA_*` knobs.
    pub fn from_env() -> Result<Self, String> {
        match std::env::var("OMEGA_PKI_TRUST_STORE") {
            Ok(path) if !path.is_empty() => Self::load_dir(Path::new(&path)),
            _ => Ok(Self::empty()),
        }
    }

    /// Number of trust roots loaded.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// Whether any roots are loaded.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    /// Source directory the store was loaded from, if any.
    pub fn source_dir(&self) -> Option<&Path> {
        self.source_dir.as_deref()
    }

    /// Verify a presented chain against the loaded trust roots.
    /// Thin wrapper over [`verify_chain`] that materialises the
    /// HashMap into the slice the chain validator expects.
    #[allow(dead_code)]
    pub fn verify_chain<'a>(&self, chain: &'a [OmegaCert]) -> Result<&'a OmegaCert, String> {
        let roots = self.roots_vec();
        verify_chain(chain, &roots)
    }

    /// Materialise the loaded roots into a `Vec<OmegaCert>`. The WS
    /// handshake path needs the slice form for both `verify_chain`
    /// and `verify_chain_with_revocation`; this avoids each callsite
    /// reaching into the internal `HashMap`.
    pub fn roots_vec(&self) -> Vec<OmegaCert> {
        self.roots.values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_dsa::{MlDsa65, Seed as DsaSeed, SigningKey};

    fn fresh_seed() -> [u8; 32] {
        rand::random()
    }

    fn fresh_root(subject: &str) -> OmegaCert {
        let seed = fresh_seed();
        OmegaCert::self_sign(subject, 86400, &seed, None).unwrap()
    }

    #[test]
    fn empty_store_rejects_every_chain() {
        let store = TrustStore::empty();
        let chain = vec![fresh_root("anyone")];
        assert!(store.verify_chain(&chain).is_err());
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn from_env_unset_returns_empty_store() {
        // SAFETY: tests run single-threaded by default in a freshly-
        // forked process; toggling this env var is harmless.
        std::env::remove_var("OMEGA_PKI_TRUST_STORE");
        let store = TrustStore::from_env().unwrap();
        assert!(store.is_empty());
        assert!(store.source_dir().is_none());
    }

    #[test]
    fn from_certs_round_trip_validates_chain() {
        let root = fresh_root("omega-root");
        let store = TrustStore::from_certs(vec![root.clone()]).unwrap();
        assert_eq!(store.len(), 1);

        // Single-element chain anchored on the root: passes.
        let chain = vec![root];
        store.verify_chain(&chain).expect("self-rooted chain");
    }

    #[test]
    fn from_certs_rejects_tampered_root_at_load_time() {
        let mut bad = fresh_root("omega-root");
        bad.body.subject = "attacker".into();
        let result = TrustStore::from_certs(vec![bad]);
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("self-verification"))
                .unwrap_or(false),
            "expected self-verification rejection, got {result:?}"
        );
    }

    #[test]
    fn load_dir_reads_only_cbor_files() {
        // Build a temp dir with one cert + one stray text file. The
        // text file must be skipped silently; the cert must load.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = fresh_root("omega-root");
        std::fs::write(dir.path().join("root.cbor"), root.to_cbor()).unwrap();
        std::fs::write(dir.path().join("README.txt"), "not a cert").unwrap();

        let store = TrustStore::load_dir(dir.path()).expect("load");
        assert_eq!(store.len(), 1);
        assert_eq!(store.source_dir(), Some(dir.path()));
    }

    #[test]
    fn load_dir_rejects_duplicate_serials() {
        // Same root written under two filenames → duplicate serial.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = fresh_root("omega-root");
        std::fs::write(dir.path().join("a.cbor"), root.to_cbor()).unwrap();
        std::fs::write(dir.path().join("b.cbor"), root.to_cbor()).unwrap();

        let result = TrustStore::load_dir(dir.path());
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("duplicate serial"))
                .unwrap_or(false),
            "expected duplicate-serial rejection, got {result:?}"
        );
    }

    #[test]
    fn load_dir_rejects_corrupt_cbor() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("garbage.cbor"), b"not cbor").unwrap();
        let result = TrustStore::load_dir(dir.path());
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("parse") && e.contains("OmegaCert"))
                .unwrap_or(false),
            "expected parse rejection, got {result:?}"
        );
    }

    #[test]
    fn loaded_store_validates_two_level_chain() {
        // root + leaf signed by root, loaded via the directory path.
        let root_seed = fresh_seed();
        let root = OmegaCert::self_sign("omega-root", 86400, &root_seed, None).unwrap();

        let leaf_seed: [u8; 32] = rand::random();
        let leaf_sk = SigningKey::<MlDsa65>::from_seed(&DsaSeed::from(leaf_seed));
        let leaf_pk = leaf_sk.verifying_key().encode();
        let leaf_pk_bytes: &[u8] = leaf_pk.as_ref();
        let leaf = OmegaCert::sign_with_issuer(
            "alice",
            3600,
            leaf_pk_bytes,
            None,
            "omega-root",
            &root_seed,
        )
        .unwrap();

        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("root.cbor"), root.to_cbor()).unwrap();
        let store = TrustStore::load_dir(dir.path()).expect("load");

        let chain = vec![leaf, root];
        let leaf_back = store.verify_chain(&chain).expect("chain valid");
        assert_eq!(leaf_back.body.subject, "alice");
    }
}
