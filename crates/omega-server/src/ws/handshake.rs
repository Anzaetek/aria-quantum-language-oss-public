//! PQC WebSocket handshake state machine.
//!
//! Protocol: omega-pqc-v1
//!
//! 1. Server → Client: ServerHello (server cert + KEM encapsulation key)
//! 2. Client → Server: ClientHello (KEM ciphertext + optional client
//!    cert / chain)
//! 3. Both derive AES-256-GCM session keys via HKDF
//! 4. All subsequent frames are encrypted
//!
//! Client cert validation is policy-driven via [`ClientCertPolicy`].
//! - `Off` (default): legacy behaviour. If a client supplies a single
//!   self-signed cert via the legacy `client_cert` field it is
//!   self-verified and the subject is recorded; otherwise no cert is
//!   required. No anchoring against the trust store.
//! - `Optional`: if the client supplies a `client_chain`, it must
//!   anchor in the loaded `TrustStore` (and clear the `OmegaCrl`
//!   when one is configured). No chain is OK; legacy single-cert is
//!   rejected to keep the security bar consistent.
//! - `Required`: the client must supply a `client_chain` and the
//!   chain must validate.

use serde::{Deserialize, Serialize};

use crate::pqc::certificate::{verify_chain, verify_chain_with_revocation, OmegaCert};
use crate::pqc::crl::OmegaCrl;
use crate::pqc::kem::{self, KemKeypair};
use crate::pqc::session::SessionKeys;
use crate::pqc::trust_store::TrustStore;

/// ServerHello message: sent by server after WebSocket upgrade.
#[derive(Serialize, Deserialize, Debug)]
pub struct ServerHello {
    /// Server's OmegaCert (contains ML-DSA-65 PK + ML-KEM-768 PK).
    pub cert: Vec<u8>,
    /// Ephemeral ML-KEM-768 encapsulation key for this session.
    pub kem_ek: Vec<u8>,
}

/// ClientHello message: sent by client in response.
///
/// Carries either a legacy `client_cert` (single self-signed cert)
/// or a full `client_chain` (leaf-first list of CBOR-encoded
/// `OmegaCert` bytes). Chain takes precedence when both are present.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct ClientHello {
    /// KEM ciphertext (encapsulated shared secret).
    pub kem_ct: Vec<u8>,
    /// Legacy single client certificate. Self-verified only; never
    /// anchored. Kept so existing PQC clients continue to work under
    /// `ClientCertPolicy::Off`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_cert: Option<Vec<u8>>,
    /// Client certificate chain, leaf first. Each entry is a
    /// CBOR-encoded `OmegaCert`. The terminal entry must match a
    /// trust root by serial; revocation is checked against the
    /// configured CRL when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_chain: Option<Vec<Vec<u8>>>,
}

/// Server-side cert-presentation policy. Drives what
/// [`ServerHandshake::finish`] requires of the client side.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClientCertPolicy {
    /// No cert required; if a legacy single cert is presented it is
    /// self-verified only. Default — preserves the pre-Phase-10
    /// behaviour for existing clients.
    #[default]
    Off,
    /// If a chain is presented it must validate against the trust
    /// store (and the CRL when configured). No chain is fine, but a
    /// legacy `client_cert` alone is rejected — operators who flip
    /// this on want the validated path or nothing.
    Optional,
    /// A client chain is required. No chain → handshake aborts.
    Required,
}

impl ClientCertPolicy {
    /// Read `OMEGA_PKI_CLIENT_CERT_POLICY` (`off|optional|required`).
    /// Unset / unknown → `Off`. The `from_env` parse is permissive
    /// because operators rolling out PKI gradually want sane
    /// defaults; we surface the chosen mode in the boot log.
    pub fn from_env() -> Self {
        match std::env::var("OMEGA_PKI_CLIENT_CERT_POLICY")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("optional") => Self::Optional,
            Some("required") => Self::Required,
            _ => Self::Off,
        }
    }

    /// Short tag for boot logs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Optional => "optional",
            Self::Required => "required",
        }
    }
}

/// Outcome of a successful handshake. Carries the derived session
/// keys plus, when a client cert was validated, the leaf subject so
/// downstream handlers can attribute messages to a peer identity.
pub struct HandshakeOutcome {
    pub session: SessionKeys,
    pub client_subject: Option<String>,
}

impl std::fmt::Debug for HandshakeOutcome {
    // Manual impl: SessionKeys deliberately doesn't impl Debug (its
    // bytes are sensitive). Print only the public-facing peer identity.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandshakeOutcome")
            .field("client_subject", &self.client_subject)
            .field("session", &"<redacted>")
            .finish()
    }
}

/// Server-side handshake state machine.
pub struct ServerHandshake {
    #[allow(dead_code)]
    server_cert: OmegaCert,
    kem_keypair: KemKeypair,
    server_hello_cbor: Vec<u8>,
}

impl ServerHandshake {
    /// Begin a new handshake. Returns the ServerHello bytes (CBOR) to send.
    pub fn new(server_cert: OmegaCert) -> Self {
        let kem_keypair = KemKeypair::generate();

        let hello = ServerHello {
            cert: server_cert.to_cbor(),
            kem_ek: kem_keypair.ek_bytes(),
        };

        let mut hello_cbor = Vec::new();
        ciborium::into_writer(&hello, &mut hello_cbor).expect("CBOR encode ServerHello");

        Self {
            server_cert,
            kem_keypair,
            server_hello_cbor: hello_cbor,
        }
    }

    /// Get the ServerHello bytes to send to the client.
    pub fn server_hello_bytes(&self) -> &[u8] {
        &self.server_hello_cbor
    }

    /// Process the ClientHello, validate any presented client cert
    /// per `policy`, and derive session keys.
    pub fn finish(
        self,
        client_hello_cbor: &[u8],
        policy: ClientCertPolicy,
        trust_store: &TrustStore,
        crl: Option<&OmegaCrl>,
    ) -> Result<HandshakeOutcome, String> {
        let client_hello: ClientHello = ciborium::from_reader(client_hello_cbor)
            .map_err(|e| format!("bad ClientHello: {e}"))?;

        let client_subject = verify_client_credentials(&client_hello, policy, trust_store, crl)?;

        // Decapsulate the shared secret
        let shared_secret = kem::decapsulate(&self.kem_keypair.dk, &client_hello.kem_ct)?;

        // Transcript = ServerHello || ClientHello
        let mut transcript = self.server_hello_cbor;
        transcript.extend_from_slice(client_hello_cbor);

        let session = SessionKeys::derive(&shared_secret, &transcript)?;
        Ok(HandshakeOutcome {
            session,
            client_subject,
        })
    }
}

/// Inspect a ClientHello and apply the cert-presentation policy.
///
/// Returns `Ok(Some(subject))` when a cert was validated,
/// `Ok(None)` when no cert was required and none was sent. Any
/// policy violation (missing chain under Required, legacy cert
/// under Optional/Required, chain that fails to anchor or is
/// revoked) produces `Err`.
fn verify_client_credentials(
    hello: &ClientHello,
    policy: ClientCertPolicy,
    trust_store: &TrustStore,
    crl: Option<&OmegaCrl>,
) -> Result<Option<String>, String> {
    // Reject empty chains up-front so the policy table stays clean.
    let chain = hello
        .client_chain
        .as_ref()
        .filter(|c| !c.is_empty())
        .map(|c| c.as_slice());
    let legacy_cert = hello.client_cert.as_deref();

    match (chain, legacy_cert, policy) {
        // --- Validated chain present: always validate, regardless of policy. ---
        (Some(chain_bytes), _, _) => {
            let chain = chain_bytes
                .iter()
                .map(|b| OmegaCert::from_cbor(b))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("client chain decode: {e}"))?;
            let leaf = verify_client_chain(&chain, trust_store, crl)?;
            Ok(Some(leaf.body.subject.clone()))
        }
        // --- Legacy single cert under Off: self-verify (back-compat). ---
        (None, Some(cert_bytes), ClientCertPolicy::Off) => {
            let client_cert = OmegaCert::from_cbor(cert_bytes)
                .map_err(|e| format!("legacy client_cert decode: {e}"))?;
            client_cert
                .verify()
                .map_err(|e| format!("legacy client_cert self-verify failed: {e}"))?;
            Ok(Some(client_cert.body.subject))
        }
        // --- Legacy cert under Optional/Required: rejected. ---
        (None, Some(_), ClientCertPolicy::Optional)
        | (None, Some(_), ClientCertPolicy::Required) => {
            Err("client presented a legacy single cert but server policy \
                 requires a validated chain (use ClientHello.client_chain)"
                .into())
        }
        // --- No cert under Required: rejected. ---
        (None, None, ClientCertPolicy::Required) => {
            Err("server policy requires a client certificate chain".into())
        }
        // --- No cert under Off / Optional: fine. ---
        (None, None, ClientCertPolicy::Off) | (None, None, ClientCertPolicy::Optional) => Ok(None),
    }
}

/// Validate a parsed client chain against the trust store. When a
/// CRL is configured we use the revocation-aware path; otherwise the
/// plain `verify_chain`. Both ultimately bottom out in the same
/// chain-anchoring logic.
fn verify_client_chain<'a>(
    chain: &'a [OmegaCert],
    trust_store: &TrustStore,
    crl: Option<&OmegaCrl>,
) -> Result<&'a OmegaCert, String> {
    // The chain validators want a slice of root certs; materialise the
    // store's roots once. The trust store is small (usually a handful
    // of roots) so the clone is cheap relative to ML-DSA verification.
    let roots = trust_store.roots_vec();
    match crl {
        Some(crl) => verify_chain_with_revocation(chain, &roots, crl),
        None => verify_chain(chain, &roots),
    }
}

/// Client-side handshake state machine.
#[allow(dead_code)]
pub struct ClientHandshake;

impl ClientHandshake {
    /// Process a ServerHello and produce a ClientHello + session keys.
    ///
    /// Returns (client_hello_cbor, session_keys). `client_chain`
    /// (when supplied) takes precedence over `client_cert`; passing
    /// both is allowed but only the chain is consulted by the
    /// server.
    #[allow(dead_code)]
    pub fn respond(
        server_hello_cbor: &[u8],
        client_cert: Option<&OmegaCert>,
        client_chain: Option<&[OmegaCert]>,
    ) -> Result<(Vec<u8>, SessionKeys), String> {
        let server_hello: ServerHello = ciborium::from_reader(server_hello_cbor)
            .map_err(|e| format!("bad ServerHello: {e}"))?;

        // Verify server certificate
        let server_cert = OmegaCert::from_cbor(&server_hello.cert)?;
        server_cert.verify()?;

        // Encapsulate shared secret to server's ephemeral KEM key
        let (ct_bytes, shared_secret) = kem::encapsulate(&server_hello.kem_ek)?;

        let client_hello = ClientHello {
            kem_ct: ct_bytes,
            client_cert: client_cert.map(|c| c.to_cbor()),
            client_chain: client_chain.map(|c| c.iter().map(|cert| cert.to_cbor()).collect()),
        };

        let mut client_hello_cbor = Vec::new();
        ciborium::into_writer(&client_hello, &mut client_hello_cbor)
            .expect("CBOR encode ClientHello");

        // Transcript = ServerHello || ClientHello
        let mut transcript = server_hello_cbor.to_vec();
        transcript.extend_from_slice(&client_hello_cbor);

        let session_keys = SessionKeys::derive(&shared_secret, &transcript)?;

        Ok((client_hello_cbor, session_keys))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_dsa::{MlDsa65, Seed as DsaSeed, SigningKey};

    fn fresh_seed() -> [u8; 32] {
        rand::random()
    }

    fn test_cert(subject: &str) -> (OmegaCert, [u8; 32]) {
        let seed = fresh_seed();
        let cert = OmegaCert::self_sign(subject, 3600, &seed, None).unwrap();
        (cert, seed)
    }

    fn issued_leaf(subject: &str, issuer_subject: &str, issuer_seed: &[u8; 32]) -> OmegaCert {
        let leaf_seed = fresh_seed();
        let leaf_sk = SigningKey::<MlDsa65>::from_seed(&DsaSeed::from(leaf_seed));
        let leaf_pk = leaf_sk.verifying_key().encode();
        let leaf_pk_bytes: &[u8] = leaf_pk.as_ref();
        OmegaCert::sign_with_issuer(
            subject,
            3600,
            leaf_pk_bytes,
            None,
            issuer_subject,
            issuer_seed,
        )
        .unwrap()
    }

    #[test]
    fn test_full_handshake_no_client_cert() {
        let (server_cert, _seed) = test_cert("test-server");

        let server_hs = ServerHandshake::new(server_cert);
        let server_hello_bytes = server_hs.server_hello_bytes().to_vec();

        let (client_hello_bytes, mut client_session) =
            ClientHandshake::respond(&server_hello_bytes, None, None).unwrap();

        let outcome = server_hs
            .finish(
                &client_hello_bytes,
                ClientCertPolicy::Off,
                &TrustStore::empty(),
                None,
            )
            .unwrap();
        assert!(outcome.client_subject.is_none());
        let mut server_session = outcome.session;

        let ct = server_session.encrypt(b"hello from server").unwrap();
        let pt = client_session.decrypt(&ct).unwrap();
        assert_eq!(pt, b"hello from server");

        let ct = client_session.encrypt(b"hello from client").unwrap();
        let pt = server_session.decrypt(&ct).unwrap();
        assert_eq!(pt, b"hello from client");
    }

    #[test]
    fn test_legacy_client_cert_under_off_policy() {
        let (server_cert, _) = test_cert("test-server");
        let (client_cert, _) = test_cert("alice");

        let server_hs = ServerHandshake::new(server_cert);
        let server_hello_bytes = server_hs.server_hello_bytes().to_vec();

        let (client_hello_bytes, mut client_session) =
            ClientHandshake::respond(&server_hello_bytes, Some(&client_cert), None).unwrap();

        let outcome = server_hs
            .finish(
                &client_hello_bytes,
                ClientCertPolicy::Off,
                &TrustStore::empty(),
                None,
            )
            .unwrap();
        assert_eq!(outcome.client_subject.as_deref(), Some("alice"));
        let mut server_session = outcome.session;

        let ct = server_session.encrypt(b"authenticated").unwrap();
        assert_eq!(client_session.decrypt(&ct).unwrap(), b"authenticated");
    }

    #[test]
    fn test_legacy_cert_rejected_under_optional_policy() {
        let (server_cert, _) = test_cert("test-server");
        let (client_cert, _) = test_cert("alice");

        let server_hs = ServerHandshake::new(server_cert);
        let server_hello_bytes = server_hs.server_hello_bytes().to_vec();

        let (client_hello_bytes, _) =
            ClientHandshake::respond(&server_hello_bytes, Some(&client_cert), None).unwrap();

        let result = server_hs.finish(
            &client_hello_bytes,
            ClientCertPolicy::Optional,
            &TrustStore::empty(),
            None,
        );
        let err = result.expect_err("legacy cert under Optional must be rejected");
        assert!(
            err.contains("legacy single cert"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn test_required_policy_rejects_missing_cert() {
        let (server_cert, _) = test_cert("test-server");

        let server_hs = ServerHandshake::new(server_cert);
        let server_hello_bytes = server_hs.server_hello_bytes().to_vec();

        let (client_hello_bytes, _) =
            ClientHandshake::respond(&server_hello_bytes, None, None).unwrap();

        let err = server_hs
            .finish(
                &client_hello_bytes,
                ClientCertPolicy::Required,
                &TrustStore::empty(),
                None,
            )
            .expect_err("Required policy with no cert must reject");
        assert!(err.contains("requires a client certificate chain"));
    }

    #[test]
    fn test_optional_policy_validates_chain_against_trust_store() {
        let (server_cert, _) = test_cert("test-server");
        let (root, root_seed) = test_cert("ca-root");
        let leaf = issued_leaf("alice", "ca-root", &root_seed);

        let trust_store = TrustStore::from_certs(vec![root.clone()]).unwrap();
        let server_hs = ServerHandshake::new(server_cert);
        let server_hello_bytes = server_hs.server_hello_bytes().to_vec();

        let chain = vec![leaf.clone(), root.clone()];
        let (client_hello_bytes, mut client_session) =
            ClientHandshake::respond(&server_hello_bytes, None, Some(&chain)).unwrap();

        let outcome = server_hs
            .finish(
                &client_hello_bytes,
                ClientCertPolicy::Optional,
                &trust_store,
                None,
            )
            .unwrap();
        assert_eq!(outcome.client_subject.as_deref(), Some("alice"));
        let mut server_session = outcome.session;

        let ct = server_session.encrypt(b"validated chain").unwrap();
        assert_eq!(client_session.decrypt(&ct).unwrap(), b"validated chain");
    }

    #[test]
    fn test_required_policy_validates_chain() {
        let (server_cert, _) = test_cert("test-server");
        let (root, root_seed) = test_cert("ca-root");
        let leaf = issued_leaf("bob", "ca-root", &root_seed);

        let trust_store = TrustStore::from_certs(vec![root.clone()]).unwrap();
        let server_hs = ServerHandshake::new(server_cert);
        let server_hello_bytes = server_hs.server_hello_bytes().to_vec();

        let chain = vec![leaf, root];
        let (client_hello_bytes, _client_session) =
            ClientHandshake::respond(&server_hello_bytes, None, Some(&chain)).unwrap();

        let outcome = server_hs
            .finish(
                &client_hello_bytes,
                ClientCertPolicy::Required,
                &trust_store,
                None,
            )
            .unwrap();
        assert_eq!(outcome.client_subject.as_deref(), Some("bob"));
    }

    #[test]
    fn test_chain_rejected_when_root_not_trusted() {
        let (server_cert, _) = test_cert("test-server");
        let (root, root_seed) = test_cert("ca-root");
        let leaf = issued_leaf("eve", "ca-root", &root_seed);

        // Empty trust store — chain must fail to anchor.
        let trust_store = TrustStore::empty();
        let server_hs = ServerHandshake::new(server_cert);
        let server_hello_bytes = server_hs.server_hello_bytes().to_vec();

        let chain = vec![leaf, root];
        let (client_hello_bytes, _) =
            ClientHandshake::respond(&server_hello_bytes, None, Some(&chain)).unwrap();

        let err = server_hs
            .finish(
                &client_hello_bytes,
                ClientCertPolicy::Required,
                &trust_store,
                None,
            )
            .expect_err("chain anchored on untrusted root must reject");
        assert!(
            err.contains("not in the trust store"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_chain_with_crl_rejects_revoked_leaf() {
        use crate::pqc::crl::{OmegaCrl, RevocationReason, RevokedEntry};

        let (server_cert, _) = test_cert("test-server");
        let (root, root_seed) = test_cert("ca-root");
        let leaf = issued_leaf("compromised", "ca-root", &root_seed);

        let trust_store = TrustStore::from_certs(vec![root.clone()]).unwrap();

        // Build a CRL signed by the root that revokes the leaf serial.
        let revoked = vec![RevokedEntry {
            serial: leaf.body.serial.clone(),
            revocation_date: chrono::Utc::now().timestamp(),
            reason: RevocationReason::KeyCompromise,
        }];
        let crl = OmegaCrl::sign("ca-root", 3600, revoked, &root_seed).unwrap();

        let server_hs = ServerHandshake::new(server_cert);
        let server_hello_bytes = server_hs.server_hello_bytes().to_vec();

        let chain = vec![leaf, root];
        let (client_hello_bytes, _) =
            ClientHandshake::respond(&server_hello_bytes, None, Some(&chain)).unwrap();

        let err = server_hs
            .finish(
                &client_hello_bytes,
                ClientCertPolicy::Required,
                &trust_store,
                Some(&crl),
            )
            .expect_err("revoked leaf must reject");
        assert!(err.contains("revoked"), "unexpected error: {err}");
    }

    // Mutex to serialise access to the process-wide
    // OMEGA_PKI_CLIENT_CERT_POLICY env var while the parser test
    // mutates it. Mirrors the CRL_ENV_LOCK pattern in
    // `crate::pqc::crl` for the parallel `from_env` parser there.
    static POLICY_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_client_cert_policy_from_env_parses_known_modes() {
        let _g = POLICY_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let var = "OMEGA_PKI_CLIENT_CERT_POLICY";
        std::env::set_var(var, "off");
        assert_eq!(ClientCertPolicy::from_env(), ClientCertPolicy::Off);
        std::env::set_var(var, "Optional");
        assert_eq!(ClientCertPolicy::from_env(), ClientCertPolicy::Optional);
        std::env::set_var(var, " REQUIRED ");
        assert_eq!(ClientCertPolicy::from_env(), ClientCertPolicy::Required);
        std::env::set_var(var, "garbage");
        assert_eq!(ClientCertPolicy::from_env(), ClientCertPolicy::Off);
        std::env::remove_var(var);
        assert_eq!(ClientCertPolicy::from_env(), ClientCertPolicy::Off);
    }
}
