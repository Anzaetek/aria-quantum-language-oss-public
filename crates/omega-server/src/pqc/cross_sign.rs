//! Cross-signed certificates for PKI key rotation (Phase 10, fifth piece).
//!
//! Two-CA rotation pattern: an operator runs a leaf cert through a
//! transition window where both the old root and the new root sign
//! the same body. Clients on either trust store accept it; once
//! every client has migrated, the operator drops the old signature
//! at the next renewal.
//!
//! Schema sits alongside [`OmegaCert`] rather than replacing it —
//! the wire format stays compatible. An [`OmegaCrossCert`] carries
//! the same [`CertBody`] plus a `Vec<CrossSignature>`, each tagged
//! with the issuer's subject name so the verifier can pick the
//! right `(issuer, sig)` pair.
//!
//! Verification semantics: a cross-signed cert is valid against a
//! set of candidate `(issuer_subject, issuer_pk)` pairs iff *at
//! least one* `(issuer, signature)` pair on the cert verifies and
//! its issuer is recognised. The validity-window check on
//! `body.not_before` / `body.not_after` runs once after the
//! signature pass — semantically the cert has one body, so one
//! window applies regardless of which signature anchored trust.

use ml_dsa::{EncodedVerifyingKey, MlDsa65, Seed as DsaSeed, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use signature::{Signer, Verifier};

use crate::pqc::certificate::{CertBody, OmegaCert};

/// One issuer's signature over a [`CertBody`]. The `issuer_subject`
/// disambiguates which `(issuer, sig)` pair the verifier should
/// match against a trust-store entry.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CrossSignature {
    /// Subject name of the issuing CA. Compared against the trust
    /// store's roots for `(issuer, pk)` matching.
    pub issuer_subject: String,
    /// ML-DSA-65 detached signature over CBOR(body).
    pub signature: Vec<u8>,
}

/// Cross-signed certificate: a single [`CertBody`] with one or more
/// signatures from distinct issuers. The wire format is independent
/// from [`OmegaCert`]; an operator publishes whichever shape fits
/// the lifecycle stage.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OmegaCrossCert {
    pub body: CertBody,
    pub signatures: Vec<CrossSignature>,
}

impl OmegaCrossCert {
    /// Wrap an existing single-signer [`OmegaCert`] as the first
    /// link in a cross-signed cert. Convenience constructor for the
    /// common case "I have an old-CA cert, now I want to add a new
    /// CA's signature alongside."
    #[allow(dead_code)]
    pub fn from_single(cert: OmegaCert) -> Self {
        OmegaCrossCert {
            signatures: vec![CrossSignature {
                issuer_subject: cert.body.issuer.clone(),
                signature: cert.signature,
            }],
            body: cert.body,
        }
    }

    /// Append a fresh signature from another issuer over the same
    /// body. Idempotent on `(issuer_subject)` — adding the same
    /// issuer twice replaces the earlier signature.
    #[allow(dead_code)]
    pub fn add_signature(
        &mut self,
        issuer_subject: &str,
        issuer_dsa_seed: &[u8],
    ) -> Result<(), String> {
        let seed: [u8; 32] = issuer_dsa_seed
            .try_into()
            .map_err(|_| "issuer DSA seed must be 32 bytes")?;
        let sk = SigningKey::<MlDsa65>::from_seed(&DsaSeed::from(seed));
        let body_cbor = self.body.to_cbor();
        let sig = sk.sign(&body_cbor);
        let sig_enc = sig.encode();
        let sig_bytes: &[u8] = sig_enc.as_ref();

        // Replace any existing signature for this issuer rather than
        // accumulating duplicates.
        if let Some(existing) = self
            .signatures
            .iter_mut()
            .find(|s| s.issuer_subject == issuer_subject)
        {
            existing.signature = sig_bytes.to_vec();
        } else {
            self.signatures.push(CrossSignature {
                issuer_subject: issuer_subject.to_string(),
                signature: sig_bytes.to_vec(),
            });
        }
        Ok(())
    }

    /// Verify the cert against a slate of candidate issuers. The
    /// cert is valid iff at least one `(issuer_subject, signature)`
    /// pair on the cert matches a `(name, pk)` in `candidates` AND
    /// the signature verifies AND the validity window holds.
    ///
    /// `candidates` is typically `[(root.body.subject,
    /// root.body.ml_dsa_pk) for root in trust_store]`. The match is
    /// by subject name, so a trust store with both `omega-root-2024`
    /// and `omega-root-2025` will accept a cert that's been signed
    /// by either (or both) during a rotation window.
    ///
    /// Returns the matched issuer's subject string on success so the
    /// caller knows which trust anchor was used.
    #[allow(dead_code)]
    pub fn verify_against_any(&self, candidates: &[(&str, &[u8])]) -> Result<String, String> {
        if self.signatures.is_empty() {
            return Err("cert has no signatures".into());
        }

        let body_cbor = self.body.to_cbor();
        let mut last_err: Option<String> = None;
        for cs in &self.signatures {
            let Some((_, pk)) = candidates
                .iter()
                .find(|(name, _)| *name == cs.issuer_subject)
            else {
                continue;
            };

            let vk_enc = match EncodedVerifyingKey::<MlDsa65>::try_from(*pk) {
                Ok(v) => v,
                Err(_) => {
                    last_err = Some(format!("invalid issuer pk for {:?}", cs.issuer_subject));
                    continue;
                }
            };
            let vk = VerifyingKey::<MlDsa65>::decode(&vk_enc);

            let sig = match ml_dsa::Signature::<MlDsa65>::try_from(cs.signature.as_slice()) {
                Ok(s) => s,
                Err(e) => {
                    last_err = Some(format!(
                        "invalid signature bytes for issuer {:?}: {e}",
                        cs.issuer_subject
                    ));
                    continue;
                }
            };

            if vk.verify(&body_cbor, &sig).is_err() {
                last_err = Some(format!(
                    "signature from issuer {:?} did not verify",
                    cs.issuer_subject
                ));
                continue;
            }

            // First match wins — apply the validity window check on
            // the shared body and return the issuer that anchored.
            let now = chrono::Utc::now().timestamp();
            if now < self.body.not_before {
                return Err("certificate not yet valid".into());
            }
            if now > self.body.not_after {
                return Err("certificate expired".into());
            }
            return Ok(cs.issuer_subject.clone());
        }

        Err(last_err.unwrap_or_else(|| {
            format!(
                "no signature on cert (issuers = {:?}) matched any of the {} candidate trust anchor(s)",
                self.signatures
                    .iter()
                    .map(|s| s.issuer_subject.as_str())
                    .collect::<Vec<_>>(),
                candidates.len()
            )
        }))
    }

    /// Number of distinct signers on this cert.
    #[allow(dead_code)]
    pub fn num_signers(&self) -> usize {
        self.signatures.len()
    }

    /// CBOR encode for wire transport.
    #[allow(dead_code)]
    pub fn to_cbor(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("CBOR encode");
        buf
    }

    /// CBOR decode from wire bytes.
    #[allow(dead_code)]
    pub fn from_cbor(data: &[u8]) -> Result<Self, String> {
        ciborium::from_reader(data).map_err(|e| format!("CBOR decode: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_seed() -> [u8; 32] {
        rand::random()
    }

    fn seeded_root(subject: &str) -> ([u8; 32], OmegaCert) {
        let seed = fresh_seed();
        let cert = OmegaCert::self_sign(subject, 86400, &seed, None).unwrap();
        (seed, cert)
    }

    fn issue_leaf(subject: &str, ttl: i64, issuer_subject: &str, issuer_seed: &[u8]) -> OmegaCert {
        let leaf_seed = fresh_seed();
        let sk = SigningKey::<MlDsa65>::from_seed(&DsaSeed::from(leaf_seed));
        let pk_enc = sk.verifying_key().encode();
        let pk_bytes: &[u8] = pk_enc.as_ref();
        OmegaCert::sign_with_issuer(subject, ttl, pk_bytes, None, issuer_subject, issuer_seed)
            .unwrap()
    }

    #[test]
    fn cross_sign_with_two_issuers_verifies_against_either() {
        let (old_seed, old_root) = seeded_root("omega-root-2024");
        let (new_seed, new_root) = seeded_root("omega-root-2025");

        // Initial leaf signed by the old root.
        let leaf = issue_leaf("alice", 3600, "omega-root-2024", &old_seed);
        let mut cross = OmegaCrossCert::from_single(leaf);
        // Add a co-signature from the new root.
        cross
            .add_signature("omega-root-2025", &new_seed)
            .expect("co-sign");

        assert_eq!(cross.num_signers(), 2);

        // Trust store with only the old root: should still verify.
        let only_old: &[(&str, &[u8])] = &[("omega-root-2024", old_root.body.ml_dsa_pk.as_slice())];
        let matched = cross
            .verify_against_any(only_old)
            .expect("old root anchors");
        assert_eq!(matched, "omega-root-2024");

        // Trust store with only the new root: also valid.
        let only_new: &[(&str, &[u8])] = &[("omega-root-2025", new_root.body.ml_dsa_pk.as_slice())];
        let matched = cross
            .verify_against_any(only_new)
            .expect("new root anchors");
        assert_eq!(matched, "omega-root-2025");

        // Trust store with both: first matching pair wins.
        let both: &[(&str, &[u8])] = &[
            ("omega-root-2024", old_root.body.ml_dsa_pk.as_slice()),
            ("omega-root-2025", new_root.body.ml_dsa_pk.as_slice()),
        ];
        cross.verify_against_any(both).expect("either anchors");
    }

    #[test]
    fn cross_sign_rejects_when_no_candidate_matches() {
        let (old_seed, _) = seeded_root("omega-root-2024");
        let leaf = issue_leaf("alice", 3600, "omega-root-2024", &old_seed);
        let cross = OmegaCrossCert::from_single(leaf);

        // Trust store has a different root.
        let (_, foreign) = seeded_root("foreign-ca");
        let candidates: &[(&str, &[u8])] = &[("foreign-ca", foreign.body.ml_dsa_pk.as_slice())];
        let result = cross.verify_against_any(candidates);
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("no signature") && e.contains("matched"))
                .unwrap_or(false),
            "expected no-match error, got {result:?}"
        );
    }

    #[test]
    fn cross_sign_rejects_signature_with_wrong_issuer_pk() {
        let (old_seed, _) = seeded_root("omega-root-2024");
        let (_, _other) = seeded_root("omega-root-2025");
        let leaf = issue_leaf("alice", 3600, "omega-root-2024", &old_seed);
        let cross = OmegaCrossCert::from_single(leaf);

        // Trust store has the right *name* but the *wrong pk* —
        // signature won't verify.
        let bogus_pk = vec![0u8; 1952];
        let candidates: &[(&str, &[u8])] = &[("omega-root-2024", bogus_pk.as_slice())];
        let result = cross.verify_against_any(candidates);
        assert!(result.is_err(), "wrong pk must not verify");
    }

    #[test]
    fn add_signature_replaces_existing_for_same_issuer() {
        let (old_seed, old_root) = seeded_root("omega-root");
        let leaf = issue_leaf("alice", 3600, "omega-root", &old_seed);
        let mut cross = OmegaCrossCert::from_single(leaf);
        let original_sig = cross.signatures[0].signature.clone();

        // Re-sign with the same issuer (e.g. after rotating to a
        // fresh seed for the same CA name). The earlier sig must be
        // replaced, not appended.
        let fresh = fresh_seed();
        cross.add_signature("omega-root", &fresh).expect("re-sign");
        assert_eq!(cross.num_signers(), 1, "must not duplicate by issuer");
        assert_ne!(
            cross.signatures[0].signature, original_sig,
            "signature must change after re-sign"
        );

        // The original key no longer verifies (it didn't sign the
        // current state).
        let only_old: &[(&str, &[u8])] = &[("omega-root", old_root.body.ml_dsa_pk.as_slice())];
        assert!(cross.verify_against_any(only_old).is_err());
    }

    #[test]
    fn cbor_round_trip_preserves_all_signatures() {
        let (a_seed, a_root) = seeded_root("ca-a");
        let (b_seed, b_root) = seeded_root("ca-b");
        let leaf = issue_leaf("subject", 3600, "ca-a", &a_seed);
        let mut cross = OmegaCrossCert::from_single(leaf);
        cross.add_signature("ca-b", &b_seed).unwrap();

        let bytes = cross.to_cbor();
        let cross2 = OmegaCrossCert::from_cbor(&bytes).unwrap();
        assert_eq!(cross2.num_signers(), 2);

        // Both candidates verify against the round-tripped form.
        for (name, root) in [("ca-a", &a_root), ("ca-b", &b_root)] {
            let cands: &[(&str, &[u8])] = &[(name, root.body.ml_dsa_pk.as_slice())];
            cross2.verify_against_any(cands).expect("post-CBOR verify");
        }
    }

    #[test]
    fn empty_signatures_rejected() {
        let (root_seed, root) = seeded_root("r");
        let leaf = issue_leaf("alice", 3600, "r", &root_seed);
        let mut cross = OmegaCrossCert::from_single(leaf);
        cross.signatures.clear();

        let cands: &[(&str, &[u8])] = &[("r", root.body.ml_dsa_pk.as_slice())];
        let result = cross.verify_against_any(cands);
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("no signatures"))
                .unwrap_or(false),
            "expected no-signatures error, got {result:?}"
        );
    }
}
