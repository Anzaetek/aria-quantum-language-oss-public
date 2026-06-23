//! Certificate Revocation List (Phase 10, third piece).
//!
//! Operators publish a periodically-refreshed `OmegaCrl` listing the
//! serials of revoked certificates. The handshake verifier consults
//! it after the chain check and rejects chains whose leaf or any
//! intermediate appears in the list.
//!
//! Wire format mirrors `OmegaCert`: a CBOR-encoded body + a detached
//! ML-DSA-65 signature by the issuer (typically the trust-root that
//! signed the cert in question). The body has an explicit validity
//! window — clients ignore an expired CRL and refuse to honour
//! revocations from one signed in the future.
//!
//! Today the wiring stops at "build, sign, verify, lookup-by-serial".
//! Hooking into the chain validator + persisting CRLs alongside the
//! trust store lands in follow-up commits — the structure is held
//! to the same `dead_code`-allowed convention as the rest of the
//! Phase 10 PKI work.

use std::collections::HashSet;

use ml_dsa::{EncodedVerifyingKey, MlDsa65, Seed as DsaSeed, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use signature::{Signer, Verifier};

/// Revocation reason — RFC-5280-style enum, narrowed to the cases
/// omega-server actually distinguishes today. `Unspecified` is the
/// default; operators wanting a more granular taxonomy can extend
/// this in a future commit.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RevocationReason {
    Unspecified = 0,
    KeyCompromise = 1,
    CaCompromise = 2,
    AffiliationChanged = 3,
    Superseded = 4,
    CessationOfOperation = 5,
}

/// A single revoked-cert entry.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RevokedEntry {
    /// 16-byte UUID serial of the revoked certificate.
    pub serial: Vec<u8>,
    /// Unix-second timestamp when the revocation took effect.
    pub revocation_date: i64,
    /// Why the cert was revoked.
    pub reason: RevocationReason,
}

/// Unsigned CRL body.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CrlBody {
    /// Format version (always 1).
    pub version: u8,
    /// Identifier of the issuer (the trust root, in v1).
    pub issuer: String,
    /// When this CRL was generated.
    pub this_update: i64,
    /// When clients must fetch a new CRL by; an expired CRL is
    /// ignored at validation time.
    pub next_update: i64,
    /// Revoked entries, in publication order. The verifier indexes
    /// these into a `HashSet<Vec<u8>>` for O(1) lookup.
    pub revoked: Vec<RevokedEntry>,
    /// Extension map (reserved).
    pub extensions: Vec<(String, Vec<u8>)>,
}

/// A signed CRL.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OmegaCrl {
    pub body: CrlBody,
    /// ML-DSA-65 detached signature over CBOR(body).
    pub signature: Vec<u8>,
}

impl CrlBody {
    /// CBOR-encode the body for signing or wire transport.
    pub fn to_cbor(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("CBOR encode");
        buf
    }
}

impl OmegaCrl {
    /// Build and sign a CRL with the issuer's DSA seed.
    ///
    /// `validity_seconds` controls the gap between `this_update` and
    /// `next_update`. A value of 0 produces an instantly-stale CRL
    /// (useful only for tests).
    #[allow(dead_code)]
    pub fn sign(
        issuer_subject: &str,
        validity_seconds: i64,
        revoked: Vec<RevokedEntry>,
        issuer_dsa_seed: &[u8],
    ) -> Result<Self, String> {
        let seed_arr: [u8; 32] = issuer_dsa_seed
            .try_into()
            .map_err(|_| "issuer DSA seed must be 32 bytes")?;
        let sk = SigningKey::<MlDsa65>::from_seed(&DsaSeed::from(seed_arr));

        let now = chrono::Utc::now().timestamp();
        let body = CrlBody {
            version: 1,
            issuer: issuer_subject.to_string(),
            this_update: now,
            next_update: now + validity_seconds,
            revoked,
            extensions: vec![],
        };

        let body_cbor = body.to_cbor();
        let sig = sk.sign(&body_cbor);
        let sig_enc = sig.encode();
        let sig_bytes: &[u8] = sig_enc.as_ref();

        Ok(OmegaCrl {
            body,
            signature: sig_bytes.to_vec(),
        })
    }

    /// Verify the CRL's signature against an issuer public key, plus
    /// its validity window. Returns `Ok(())` only when both check out.
    #[allow(dead_code)]
    pub fn verify(&self, issuer_pk: &[u8]) -> Result<(), String> {
        let body_cbor = self.body.to_cbor();

        let vk_enc = EncodedVerifyingKey::<MlDsa65>::try_from(issuer_pk)
            .map_err(|_| "invalid issuer DSA public key size")?;
        let vk = VerifyingKey::<MlDsa65>::decode(&vk_enc);

        let sig = ml_dsa::Signature::<MlDsa65>::try_from(self.signature.as_slice())
            .map_err(|e| format!("invalid CRL signature: {e}"))?;

        vk.verify(&body_cbor, &sig)
            .map_err(|e| format!("CRL signature verification failed: {e}"))?;

        let now = chrono::Utc::now().timestamp();
        if now < self.body.this_update {
            return Err("CRL not yet valid".into());
        }
        if now > self.body.next_update {
            return Err("CRL expired (next_update in the past)".into());
        }
        Ok(())
    }

    /// Whether the given serial appears in this CRL. Linear scan;
    /// callers that hit this in a hot path should call
    /// [`OmegaCrl::serial_set`] once and consult the resulting
    /// `HashSet`.
    #[allow(dead_code)]
    pub fn is_revoked(&self, serial: &[u8]) -> bool {
        self.body.revoked.iter().any(|r| r.serial == serial)
    }

    /// O(1)-lookup view over the revoked serials. Built once per
    /// validation pass.
    #[allow(dead_code)]
    pub fn serial_set(&self) -> HashSet<&[u8]> {
        self.body
            .revoked
            .iter()
            .map(|r| r.serial.as_slice())
            .collect()
    }

    /// CBOR-encode for wire transport.
    #[allow(dead_code)]
    pub fn to_cbor(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("CBOR encode");
        buf
    }

    /// CBOR-decode from wire bytes.
    #[allow(dead_code)]
    pub fn from_cbor(data: &[u8]) -> Result<Self, String> {
        ciborium::from_reader(data).map_err(|e| format!("CBOR decode: {e}"))
    }

    /// Read a CBOR CRL from a file path.
    #[allow(dead_code)]
    pub fn load_file(path: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read CRL {}: {e}", path.display()))?;
        Self::from_cbor(&bytes)
    }

    /// Load from `OMEGA_PKI_CRL_FILE`, or return `None` if the env
    /// var is unset / empty. Mirrors `TrustStore::from_env` so the
    /// server can run with neither / both / either configured.
    pub fn from_env() -> Result<Option<Self>, String> {
        match std::env::var("OMEGA_PKI_CRL_FILE") {
            Ok(path) if !path.is_empty() => Self::load_file(std::path::Path::new(&path)).map(Some),
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pqc::certificate::OmegaCert;

    fn fresh_seed() -> [u8; 32] {
        rand::random()
    }

    fn root_with_seed(subject: &str) -> ([u8; 32], OmegaCert) {
        let seed = fresh_seed();
        let cert = OmegaCert::self_sign(subject, 86400, &seed, None).unwrap();
        (seed, cert)
    }

    // Tests touching OMEGA_PKI_CRL_FILE serialise via this mutex —
    // Cargo runs tests in parallel by default, and a race here
    // would mask real failures with intermittent ones.
    static CRL_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_unset_returns_none() {
        let _g = CRL_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("OMEGA_PKI_CRL_FILE");
        assert!(OmegaCrl::from_env().unwrap().is_none());
    }

    #[test]
    fn from_env_loads_file_when_set() {
        use std::io::Write;

        let _g = CRL_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (seed, _) = root_with_seed("r");
        let crl = OmegaCrl::sign("r", 3600, vec![], &seed).unwrap();
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        tmp.write_all(&crl.to_cbor()).unwrap();
        let path = tmp.path().to_str().unwrap().to_string();

        std::env::set_var("OMEGA_PKI_CRL_FILE", &path);
        let loaded = OmegaCrl::from_env().expect("load").expect("Some");
        std::env::remove_var("OMEGA_PKI_CRL_FILE");
        assert_eq!(loaded.body.issuer, "r");
        assert_eq!(loaded.signature, crl.signature);
    }

    #[test]
    fn load_file_rejects_corrupt_bytes() {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"not cbor").unwrap();
        let result = OmegaCrl::load_file(tmp.path());
        assert!(result.is_err(), "expected decode error, got {result:?}");
    }

    #[test]
    fn signed_crl_roundtrips_and_verifies() {
        let (seed, root) = root_with_seed("omega-root");
        let revoked = vec![RevokedEntry {
            serial: vec![0x42; 16],
            revocation_date: chrono::Utc::now().timestamp(),
            reason: RevocationReason::KeyCompromise,
        }];
        let crl = OmegaCrl::sign("omega-root", 3600, revoked, &seed).unwrap();
        crl.verify(&root.body.ml_dsa_pk).expect("CRL valid");
    }

    #[test]
    fn cbor_roundtrip_preserves_signature() {
        let (seed, _) = root_with_seed("r");
        let crl = OmegaCrl::sign("r", 3600, vec![], &seed).unwrap();
        let bytes = crl.to_cbor();
        let crl2 = OmegaCrl::from_cbor(&bytes).unwrap();
        assert_eq!(crl.signature, crl2.signature);
        assert_eq!(crl.body.issuer, crl2.body.issuer);
    }

    #[test]
    fn verify_rejects_wrong_issuer_key() {
        let (signing_seed, _) = root_with_seed("intended");
        let (_, other_root) = root_with_seed("other");
        let crl = OmegaCrl::sign("intended", 3600, vec![], &signing_seed).unwrap();
        let result = crl.verify(&other_root.body.ml_dsa_pk);
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("verification failed"))
                .unwrap_or(false),
            "expected signature failure, got {result:?}"
        );
    }

    #[test]
    fn verify_rejects_expired_crl() {
        let (seed, root) = root_with_seed("r");
        let mut crl = OmegaCrl::sign("r", 3600, vec![], &seed).unwrap();
        // Backdate next_update so we're past the validity window.
        crl.body.next_update = crl.body.this_update - 1;
        // Re-sign over the new body so the signature still matches.
        let sk = SigningKey::<MlDsa65>::from_seed(&DsaSeed::from(seed));
        let sig = sk.sign(&crl.body.to_cbor());
        let sig_enc = sig.encode();
        let sig_bytes: &[u8] = sig_enc.as_ref();
        crl.signature = sig_bytes.to_vec();

        let result = crl.verify(&root.body.ml_dsa_pk);
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("expired"))
                .unwrap_or(false),
            "expected expiry rejection, got {result:?}"
        );
    }

    #[test]
    fn verify_rejects_tampered_revocation_list() {
        let (seed, root) = root_with_seed("r");
        let mut crl = OmegaCrl::sign("r", 3600, vec![], &seed).unwrap();
        // Add a revocation post-signing — the body no longer matches
        // the signature → verify must fail.
        crl.body.revoked.push(RevokedEntry {
            serial: vec![0xAA; 16],
            revocation_date: 0,
            reason: RevocationReason::Unspecified,
        });
        let result = crl.verify(&root.body.ml_dsa_pk);
        assert!(result.is_err(), "tampered CRL must not verify");
    }

    #[test]
    fn is_revoked_finds_listed_serials() {
        let (seed, _) = root_with_seed("r");
        let s1: Vec<u8> = vec![0x01; 16];
        let s2: Vec<u8> = vec![0x02; 16];
        let crl = OmegaCrl::sign(
            "r",
            3600,
            vec![
                RevokedEntry {
                    serial: s1.clone(),
                    revocation_date: 0,
                    reason: RevocationReason::Superseded,
                },
                RevokedEntry {
                    serial: s2.clone(),
                    revocation_date: 0,
                    reason: RevocationReason::Unspecified,
                },
            ],
            &seed,
        )
        .unwrap();

        assert!(crl.is_revoked(&s1));
        assert!(crl.is_revoked(&s2));
        assert!(!crl.is_revoked(&[0x99; 16]));
        assert_eq!(crl.serial_set().len(), 2);
    }
}
