//! OmegaCert v1 — custom PQC certificate binding ML-DSA-65 + ML-KEM-768 keys.
//!
//! Serialised as CBOR via ciborium.

use ml_dsa::{EncodedVerifyingKey, MlDsa65, Seed as DsaSeed, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use signature::{Signer, Verifier};
use uuid::Uuid;

/// Unsigned certificate body (everything that gets signed).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CertBody {
    /// Format version (always 1).
    pub version: u8,
    /// 16-byte random serial.
    pub serial: Vec<u8>,
    /// Subject identifier (principal name).
    pub subject: String,
    /// Issuer identifier (self-signed → same as subject).
    pub issuer: String,
    /// Validity window (Unix seconds).
    pub not_before: i64,
    pub not_after: i64,
    /// ML-DSA-65 verifying key (1952 bytes).
    pub ml_dsa_pk: Vec<u8>,
    /// ML-KEM-768 encapsulation key (1184 bytes, optional).
    pub ml_kem_pk: Option<Vec<u8>>,
    /// Extension map (reserved).
    pub extensions: Vec<(String, Vec<u8>)>,
}

/// A signed OmegaCert.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OmegaCert {
    pub body: CertBody,
    /// ML-DSA-65 detached signature over CBOR(body) (3309 bytes).
    pub signature: Vec<u8>,
}

impl CertBody {
    /// CBOR-encode this body.
    pub fn to_cbor(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("CBOR encode");
        buf
    }
}

impl OmegaCert {
    /// Create and self-sign a certificate.
    ///
    /// `dsa_seed` is the 32-byte ML-DSA-65 signing key seed.
    /// `kem_ek_bytes` is the optional ML-KEM-768 encapsulation key bytes.
    pub fn self_sign(
        subject: &str,
        ttl_seconds: i64,
        dsa_seed: &[u8],
        kem_ek_bytes: Option<Vec<u8>>,
    ) -> Result<Self, String> {
        let seed_arr: [u8; 32] = dsa_seed
            .try_into()
            .map_err(|_| "DSA seed must be 32 bytes")?;
        let sk = SigningKey::<MlDsa65>::from_seed(&DsaSeed::from(seed_arr));
        let vk = sk.verifying_key();
        let vk_enc = vk.encode();
        let pk_bytes: &[u8] = vk_enc.as_ref();

        let now = chrono::Utc::now().timestamp();
        let body = CertBody {
            version: 1,
            serial: Uuid::new_v4().as_bytes().to_vec(),
            subject: subject.to_string(),
            issuer: subject.to_string(),
            not_before: now,
            not_after: now + ttl_seconds,
            ml_dsa_pk: pk_bytes.to_vec(),
            ml_kem_pk: kem_ek_bytes,
            extensions: vec![],
        };

        let body_cbor = body.to_cbor();
        let sig = sk.sign(&body_cbor);
        let sig_enc = sig.encode();
        let sig_bytes: &[u8] = sig_enc.as_ref();

        Ok(OmegaCert {
            body,
            signature: sig_bytes.to_vec(),
        })
    }

    /// Sign a certificate body using an *issuer's* DSA signing key.
    ///
    /// Use this for the leaf and intermediate certs in a PKI chain;
    /// the root continues to use [`OmegaCert::self_sign`] (or this
    /// helper with `issuer_subject == subject` and the issuer seed
    /// equal to the subject seed). The subject's own DSA public key
    /// is supplied directly (the issuer doesn't need the subject's
    /// signing key — only their pub key).
    ///
    /// `dead_code` until the trust-store config + issuance endpoint
    /// land — the tests below pin the behaviour now so the issuance
    /// commit can wire it in confidently.
    #[allow(dead_code)]
    pub fn sign_with_issuer(
        subject: &str,
        ttl_seconds: i64,
        subject_dsa_pk: &[u8],
        subject_kem_pk: Option<Vec<u8>>,
        issuer_subject: &str,
        issuer_dsa_seed: &[u8],
    ) -> Result<Self, String> {
        let issuer_seed: [u8; 32] = issuer_dsa_seed
            .try_into()
            .map_err(|_| "issuer DSA seed must be 32 bytes")?;
        let issuer_sk = SigningKey::<MlDsa65>::from_seed(&DsaSeed::from(issuer_seed));

        let now = chrono::Utc::now().timestamp();
        let body = CertBody {
            version: 1,
            serial: Uuid::new_v4().as_bytes().to_vec(),
            subject: subject.to_string(),
            issuer: issuer_subject.to_string(),
            not_before: now,
            not_after: now + ttl_seconds,
            ml_dsa_pk: subject_dsa_pk.to_vec(),
            ml_kem_pk: subject_kem_pk,
            extensions: vec![],
        };

        let body_cbor = body.to_cbor();
        let sig = issuer_sk.sign(&body_cbor);
        let sig_enc = sig.encode();
        let sig_bytes: &[u8] = sig_enc.as_ref();

        Ok(OmegaCert {
            body,
            signature: sig_bytes.to_vec(),
        })
    }

    /// Verify the certificate's self-signature.
    pub fn verify(&self) -> Result<(), String> {
        // Self-signed cert ⇒ issuer pk is the cert's own pk.
        self.verify_signed_by(&self.body.ml_dsa_pk)
    }

    /// Verify this certificate's signature against an external issuer
    /// public key. Used as a building block for chain validation —
    /// the leaf is verified against the intermediate's pk, the
    /// intermediate against the next-up cert's pk, etc.
    #[allow(dead_code)]
    pub fn verify_signed_by(&self, issuer_pk: &[u8]) -> Result<(), String> {
        let body_cbor = self.body.to_cbor();

        let vk_enc = EncodedVerifyingKey::<MlDsa65>::try_from(issuer_pk)
            .map_err(|_| "invalid issuer DSA public key size")?;
        let vk = VerifyingKey::<MlDsa65>::decode(&vk_enc);

        let sig = ml_dsa::Signature::<MlDsa65>::try_from(self.signature.as_slice())
            .map_err(|e| format!("invalid signature: {e}"))?;

        vk.verify(&body_cbor, &sig)
            .map_err(|e| format!("signature verification failed: {e}"))?;

        // Check validity window — applies to every link in a chain.
        let now = chrono::Utc::now().timestamp();
        if now < self.body.not_before {
            return Err("certificate not yet valid".into());
        }
        if now > self.body.not_after {
            return Err("certificate expired".into());
        }

        Ok(())
    }

    /// Serialise to CBOR bytes.
    pub fn to_cbor(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("CBOR encode");
        buf
    }

    /// Deserialise from CBOR bytes.
    pub fn from_cbor(data: &[u8]) -> Result<Self, String> {
        ciborium::from_reader(data).map_err(|e| format!("CBOR decode: {e}"))
    }
}

/// Verify a certificate chain.
///
/// `chain` is ordered leaf-first: `chain[0]` is the leaf,
/// `chain[chain.len() - 1]` is the closest-to-root certificate
/// included by the presenting party. `trust_roots` is the operator's
/// trust store — a list of root certificates the verifier accepts
/// without further proof.
///
/// Walk:
/// 1. Each adjacent pair `(chain[i], chain[i+1])` must form a valid
///    issuer link: `chain[i]` is signed by `chain[i+1].body.ml_dsa_pk`,
///    and `chain[i].body.issuer == chain[i+1].body.subject`. Validity
///    window must hold at every step.
/// 2. The terminal cert `chain.last()` must match a trust-root by
///    serial number (16-byte UUID). Self-signature on that root is
///    also re-verified.
///
/// Returns the leaf cert on success — useful when the caller wants
/// to subsequently consult the leaf's subject / KEM key.
///
/// `dead_code` until the trust-store loader + WS-handshake
/// integration land in their own commits.
#[allow(dead_code)]
pub fn verify_chain<'a>(
    chain: &'a [OmegaCert],
    trust_roots: &[OmegaCert],
) -> Result<&'a OmegaCert, String> {
    if chain.is_empty() {
        return Err("empty chain".into());
    }

    // 1. Validate every link.
    for i in 0..chain.len() - 1 {
        let cert = &chain[i];
        let issuer = &chain[i + 1];
        if cert.body.issuer != issuer.body.subject {
            return Err(format!(
                "chain[{i}].issuer = {:?} but chain[{}].subject = {:?}",
                cert.body.issuer,
                i + 1,
                issuer.body.subject
            ));
        }
        cert.verify_signed_by(&issuer.body.ml_dsa_pk)
            .map_err(|e| format!("chain[{i}] verification failed: {e}"))?;
    }

    // 2. Anchor the terminal cert in the trust store.
    let terminal = chain.last().expect("non-empty checked above");
    let root = trust_roots
        .iter()
        .find(|r| r.body.serial == terminal.body.serial)
        .ok_or_else(|| {
            format!(
                "terminal cert (subject = {:?}, serial = {:02x?}) is not in the trust store",
                terminal.body.subject, terminal.body.serial,
            )
        })?;
    // Re-verify the trust-store root self-signature so a tampered
    // store can't smuggle a bogus root in by serial collision. Also
    // guards against the operator pinning a corrupt cert by accident.
    root.verify().map_err(|e| {
        format!(
            "root cert (serial {:02x?}) failed self-verification: {e}",
            root.body.serial
        )
    })?;

    // The terminal in the chain must be the same cert (same body) as
    // the trust root we matched on — guards against a forged terminal
    // that happens to collide on serial alone.
    if root.signature != terminal.signature {
        return Err(format!(
            "terminal cert (subject = {:?}) collides on serial with a trust root \
             but the signatures differ — likely forgery",
            terminal.body.subject
        ));
    }

    Ok(&chain[0])
}

/// Verify a certificate chain *and* check every cert in the chain
/// against a revocation list. Returns the leaf on success.
///
/// The CRL is verified against the same trust root that anchors the
/// chain — operators publish CRLs signed by their root, not by an
/// independent authority. The CRL's `issuer` field must match the
/// terminal cert's subject; mismatch fails the validation. This
/// keeps a third party from substituting a different CA's CRL to
/// suppress revocations from the actual issuing CA.
#[allow(dead_code)]
pub fn verify_chain_with_revocation<'a>(
    chain: &'a [OmegaCert],
    trust_roots: &[OmegaCert],
    crl: &crate::pqc::crl::OmegaCrl,
) -> Result<&'a OmegaCert, String> {
    let leaf = verify_chain(chain, trust_roots)?;
    let terminal = chain.last().expect("non-empty checked in verify_chain");

    // CRL must be issued by the terminal trust root.
    if crl.body.issuer != terminal.body.subject {
        return Err(format!(
            "CRL issuer = {:?} but chain anchors on subject {:?}",
            crl.body.issuer, terminal.body.subject
        ));
    }
    // CRL signature + validity window.
    crl.verify(&terminal.body.ml_dsa_pk)
        .map_err(|e| format!("CRL verification failed: {e}"))?;

    // Walk every cert in the chain and reject if any serial is listed.
    let revoked = crl.serial_set();
    for (i, cert) in chain.iter().enumerate() {
        if revoked.contains(cert.body.serial.as_slice()) {
            return Err(format!(
                "chain[{i}] (subject = {:?}, serial = {:02x?}) is revoked",
                cert.body.subject, cert.body.serial
            ));
        }
    }

    Ok(leaf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_sign_and_verify() {
        let seed: [u8; 32] = rand::random();

        let cert = OmegaCert::self_sign("omega-server", 3600, &seed, None).unwrap();
        assert_eq!(cert.body.version, 1);
        assert_eq!(cert.body.subject, "omega-server");
        assert!(cert.verify().is_ok());
    }

    #[test]
    fn test_self_sign_with_kem_pk() {
        use crate::pqc::kem::KemKeypair;

        let dsa_seed: [u8; 32] = rand::random();

        let kem_kp = KemKeypair::generate();
        let cert =
            OmegaCert::self_sign("omega-server", 3600, &dsa_seed, Some(kem_kp.ek_bytes())).unwrap();
        assert!(cert.body.ml_kem_pk.is_some());
        assert!(cert.verify().is_ok());
    }

    #[test]
    fn test_cbor_roundtrip() {
        let seed: [u8; 32] = rand::random();

        let cert = OmegaCert::self_sign("test", 3600, &seed, None).unwrap();
        let bytes = cert.to_cbor();
        let cert2 = OmegaCert::from_cbor(&bytes).unwrap();
        assert_eq!(cert.body.subject, cert2.body.subject);
        assert_eq!(cert.signature, cert2.signature);
    }

    #[test]
    fn test_tampered_cert_rejected() {
        let seed: [u8; 32] = rand::random();

        let mut cert = OmegaCert::self_sign("test", 3600, &seed, None).unwrap();
        cert.body.subject = "attacker".to_string();
        assert!(cert.verify().is_err());
    }

    // ----- Chain validation (PKI Phase 10) -----

    /// Helper: generate a fresh DSA seed + the matching public key bytes.
    fn fresh_dsa_keypair() -> ([u8; 32], Vec<u8>) {
        let seed: [u8; 32] = rand::random();
        let sk = SigningKey::<MlDsa65>::from_seed(&DsaSeed::from(seed));
        let vk = sk.verifying_key();
        let vk_enc = vk.encode();
        let pk_bytes: &[u8] = vk_enc.as_ref();
        (seed, pk_bytes.to_vec())
    }

    #[test]
    fn test_verify_chain_two_levels_root_then_leaf() {
        // Root self-signs; root issues a leaf to "alice". Chain
        // [leaf, root] must verify against trust_roots = [root].
        let (root_seed, root_pk) = fresh_dsa_keypair();
        let (_leaf_seed, leaf_pk) = fresh_dsa_keypair();

        let root = OmegaCert::self_sign("omega-root", 86400, &root_seed, None).unwrap();
        let leaf =
            OmegaCert::sign_with_issuer("alice", 3600, &leaf_pk, None, "omega-root", &root_seed)
                .unwrap();

        // Sanity: leaf carries alice's pk, signed by root_pk.
        assert_eq!(leaf.body.subject, "alice");
        assert_eq!(leaf.body.issuer, "omega-root");
        assert_eq!(leaf.body.ml_dsa_pk, leaf_pk);
        assert_eq!(root.body.ml_dsa_pk, root_pk);

        let chain = vec![leaf.clone(), root.clone()];
        let trust_roots = vec![root.clone()];
        let leaf_back = verify_chain(&chain, &trust_roots).expect("chain valid");
        assert_eq!(leaf_back.body.subject, "alice");
    }

    #[test]
    fn test_verify_chain_three_levels_leaf_intermediate_root() {
        // Root signs intermediate "ca-1"; ca-1 signs leaf "bob".
        // Chain = [bob_leaf, ca_1, root]; trust = [root]. Pass.
        let (root_seed, _root_pk) = fresh_dsa_keypair();
        let (ca_seed, ca_pk) = fresh_dsa_keypair();
        let (_leaf_seed, leaf_pk) = fresh_dsa_keypair();

        let root = OmegaCert::self_sign("omega-root", 86400, &root_seed, None).unwrap();
        let ca = OmegaCert::sign_with_issuer("ca-1", 7200, &ca_pk, None, "omega-root", &root_seed)
            .unwrap();
        let leaf =
            OmegaCert::sign_with_issuer("bob", 1800, &leaf_pk, None, "ca-1", &ca_seed).unwrap();

        let chain = vec![leaf.clone(), ca.clone(), root.clone()];
        let trust_roots = vec![root.clone()];
        verify_chain(&chain, &trust_roots).expect("3-level chain valid");
    }

    #[test]
    fn test_verify_chain_rejects_unknown_root() {
        // Build a valid chain anchored on root_a. Verify against trust
        // store containing root_b only — should reject.
        let (root_a_seed, _) = fresh_dsa_keypair();
        let (root_b_seed, _) = fresh_dsa_keypair();
        let (_leaf_seed, leaf_pk) = fresh_dsa_keypair();

        let root_a = OmegaCert::self_sign("root-a", 86400, &root_a_seed, None).unwrap();
        let root_b = OmegaCert::self_sign("root-b", 86400, &root_b_seed, None).unwrap();
        let leaf =
            OmegaCert::sign_with_issuer("leaf", 3600, &leaf_pk, None, "root-a", &root_a_seed)
                .unwrap();

        let chain = vec![leaf, root_a];
        let trust_roots = vec![root_b];
        let result = verify_chain(&chain, &trust_roots);
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("not in the trust store"))
                .unwrap_or(false),
            "expected trust-store rejection, got {result:?}"
        );
    }

    #[test]
    fn test_verify_chain_rejects_broken_link() {
        // Leaf claims to be issued by "ca-1" but is actually signed
        // by a *different* key (we use root_seed but issuer="ca-1").
        // Chain validation catches the mismatch via either the
        // signature check or the issuer-name check.
        let (root_seed, _) = fresh_dsa_keypair();
        let (_ca_seed, ca_pk) = fresh_dsa_keypair();
        let (_leaf_seed, leaf_pk) = fresh_dsa_keypair();

        let root = OmegaCert::self_sign("root", 86400, &root_seed, None).unwrap();
        let ca =
            OmegaCert::sign_with_issuer("ca-1", 7200, &ca_pk, None, "root", &root_seed).unwrap();

        // Leaf signed by ROOT but claiming "ca-1" as its issuer.
        let leaf = OmegaCert::sign_with_issuer(
            "leaf", 3600, &leaf_pk, None,
            "ca-1", // wrong: we'll sign with root_seed, but ca's pk doesn't match
            &root_seed,
        )
        .unwrap();

        let chain = vec![leaf, ca, root.clone()];
        let trust_roots = vec![root];
        let result = verify_chain(&chain, &trust_roots);
        assert!(result.is_err(), "broken link must be rejected");
    }

    #[test]
    fn test_verify_chain_rejects_expired_intermediate() {
        // Negative TTL → cert is "expired" the moment it's issued.
        // Chain validation must catch the validity-window failure.
        let (root_seed, _) = fresh_dsa_keypair();
        let (ca_seed, ca_pk) = fresh_dsa_keypair();
        let (_leaf_seed, leaf_pk) = fresh_dsa_keypair();

        let root = OmegaCert::self_sign("root", 86400, &root_seed, None).unwrap();
        let mut ca =
            OmegaCert::sign_with_issuer("ca-1", 1, &ca_pk, None, "root", &root_seed).unwrap();
        // Backdate not_after into the past so verification rejects it.
        ca.body.not_after = ca.body.not_before - 1;
        // Re-sign because we changed the body.
        let issuer_seed: [u8; 32] = root_seed;
        let issuer_sk = SigningKey::<MlDsa65>::from_seed(&DsaSeed::from(issuer_seed));
        let body_cbor = ca.body.to_cbor();
        let sig = issuer_sk.sign(&body_cbor);
        let sig_enc = sig.encode();
        let sig_bytes: &[u8] = sig_enc.as_ref();
        ca.signature = sig_bytes.to_vec();

        let leaf =
            OmegaCert::sign_with_issuer("leaf", 3600, &leaf_pk, None, "ca-1", &ca_seed).unwrap();

        let chain = vec![leaf, ca, root.clone()];
        let trust_roots = vec![root];
        let result = verify_chain(&chain, &trust_roots);
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("expired") || e.contains("not yet valid"))
                .unwrap_or(false),
            "expected expiry rejection, got {result:?}"
        );
    }

    // ----- Chain + CRL integration -----

    #[test]
    fn test_verify_chain_with_revocation_passes_unrevoked_chain() {
        use crate::pqc::crl::{OmegaCrl, RevocationReason, RevokedEntry};

        let (root_seed, _) = fresh_dsa_keypair();
        let (_leaf_seed, leaf_pk) = fresh_dsa_keypair();
        let root = OmegaCert::self_sign("omega-root", 86400, &root_seed, None).unwrap();
        let leaf =
            OmegaCert::sign_with_issuer("alice", 3600, &leaf_pk, None, "omega-root", &root_seed)
                .unwrap();

        // CRL revoking some other serial — leaf is fine.
        let unrelated = vec![RevokedEntry {
            serial: vec![0xFF; 16],
            revocation_date: 0,
            reason: RevocationReason::Unspecified,
        }];
        let crl = OmegaCrl::sign("omega-root", 3600, unrelated, &root_seed).unwrap();

        let chain = vec![leaf, root.clone()];
        let trust_roots = vec![root];
        verify_chain_with_revocation(&chain, &trust_roots, &crl).expect("unrevoked chain valid");
    }

    #[test]
    fn test_verify_chain_with_revocation_rejects_revoked_leaf() {
        use crate::pqc::crl::{OmegaCrl, RevocationReason, RevokedEntry};

        let (root_seed, _) = fresh_dsa_keypair();
        let (_leaf_seed, leaf_pk) = fresh_dsa_keypair();
        let root = OmegaCert::self_sign("omega-root", 86400, &root_seed, None).unwrap();
        let leaf =
            OmegaCert::sign_with_issuer("alice", 3600, &leaf_pk, None, "omega-root", &root_seed)
                .unwrap();

        let revoked = vec![RevokedEntry {
            serial: leaf.body.serial.clone(),
            revocation_date: chrono::Utc::now().timestamp(),
            reason: RevocationReason::KeyCompromise,
        }];
        let crl = OmegaCrl::sign("omega-root", 3600, revoked, &root_seed).unwrap();

        let chain = vec![leaf, root.clone()];
        let trust_roots = vec![root];
        let result = verify_chain_with_revocation(&chain, &trust_roots, &crl);
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("revoked"))
                .unwrap_or(false),
            "expected revoked-leaf rejection, got {result:?}"
        );
    }

    #[test]
    fn test_verify_chain_with_revocation_rejects_revoked_intermediate() {
        use crate::pqc::crl::{OmegaCrl, RevocationReason, RevokedEntry};

        let (root_seed, _) = fresh_dsa_keypair();
        let (ca_seed, ca_pk) = fresh_dsa_keypair();
        let (_leaf_seed, leaf_pk) = fresh_dsa_keypair();
        let root = OmegaCert::self_sign("root", 86400, &root_seed, None).unwrap();
        let ca =
            OmegaCert::sign_with_issuer("ca-1", 7200, &ca_pk, None, "root", &root_seed).unwrap();
        let leaf =
            OmegaCert::sign_with_issuer("bob", 1800, &leaf_pk, None, "ca-1", &ca_seed).unwrap();

        // Revoke the intermediate CA — every leaf signed by it loses
        // trust. Realistic key-compromise scenario.
        let revoked = vec![RevokedEntry {
            serial: ca.body.serial.clone(),
            revocation_date: chrono::Utc::now().timestamp(),
            reason: RevocationReason::CaCompromise,
        }];
        let crl = OmegaCrl::sign("root", 3600, revoked, &root_seed).unwrap();

        let chain = vec![leaf, ca, root.clone()];
        let trust_roots = vec![root];
        let result = verify_chain_with_revocation(&chain, &trust_roots, &crl);
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("revoked") && e.contains("ca-1"))
                .unwrap_or(false),
            "expected revoked-intermediate rejection, got {result:?}"
        );
    }

    #[test]
    fn test_verify_chain_with_revocation_rejects_wrong_issuer_crl() {
        use crate::pqc::crl::OmegaCrl;

        // CRL signed by a *different* root — even valid on its own,
        // it can't suppress revocations from the actual chain root.
        let (root_seed, _) = fresh_dsa_keypair();
        let (other_seed, _) = fresh_dsa_keypair();
        let (_leaf_seed, leaf_pk) = fresh_dsa_keypair();
        let root = OmegaCert::self_sign("root", 86400, &root_seed, None).unwrap();
        let leaf =
            OmegaCert::sign_with_issuer("alice", 3600, &leaf_pk, None, "root", &root_seed).unwrap();

        // CRL claims issuer = "rogue" — different name, different
        // signing key. Our integrator must reject this.
        let crl = OmegaCrl::sign("rogue", 3600, vec![], &other_seed).unwrap();

        let chain = vec![leaf, root.clone()];
        let trust_roots = vec![root];
        let result = verify_chain_with_revocation(&chain, &trust_roots, &crl);
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("CRL issuer"))
                .unwrap_or(false),
            "expected CRL-issuer-mismatch rejection, got {result:?}"
        );
    }

    #[test]
    fn test_verify_chain_rejects_serial_collision_with_different_signature() {
        // Forged terminal: copy a real root's serial but swap the
        // signature for a re-sign over a tampered body. The trust
        // store finds the serial but the signature mismatch triggers
        // the forgery guard.
        let (root_seed, _) = fresh_dsa_keypair();
        let (other_seed, _) = fresh_dsa_keypair();

        let real_root = OmegaCert::self_sign("root", 86400, &root_seed, None).unwrap();

        // Build a different self-signed cert and stamp it with the real
        // root's serial. It self-verifies fine — but its signature !=
        // real_root.signature, so chain validation must reject.
        let mut forged = OmegaCert::self_sign("root", 86400, &other_seed, None).unwrap();
        forged.body.serial = real_root.body.serial.clone();
        // Re-sign forged after we tampered with its body so it still
        // self-verifies on its own.
        let other_sk = SigningKey::<MlDsa65>::from_seed(&DsaSeed::from(other_seed));
        let body_cbor = forged.body.to_cbor();
        let sig = other_sk.sign(&body_cbor);
        let sig_enc = sig.encode();
        let sig_bytes: &[u8] = sig_enc.as_ref();
        forged.signature = sig_bytes.to_vec();
        // Sanity: forged self-verifies.
        forged.verify().unwrap();

        let chain = vec![forged];
        let trust_roots = vec![real_root];
        let result = verify_chain(&chain, &trust_roots);
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("collides on serial") || e.contains("likely forgery"))
                .unwrap_or(false),
            "expected forgery rejection, got {result:?}"
        );
    }
}
