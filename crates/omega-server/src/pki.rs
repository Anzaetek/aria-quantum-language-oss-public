//! PKI cert issuance endpoint (Phase 10, fourth piece).
//!
//! `POST /v1/pki/issue` — admin-only handler that takes a subject's
//! ML-DSA-65 public key (and optional ML-KEM-768 encapsulation key)
//! plus a desired TTL, signs an [`OmegaCert`] using the operator's
//! issuer seed, and returns the CBOR-encoded cert as base64.
//!
//! Configuration is via two env vars:
//! - `OMEGA_PKI_ISSUER_SUBJECT` — the subject name placed in
//!   `cert.body.issuer`. Required.
//! - `OMEGA_PKI_ISSUER_SEED_FILE` — path to a 32-byte file
//!   containing the raw ML-DSA-65 signing seed. Required when
//!   issuing.
//!
//! Both unset → `IssuerConfig::from_env()` returns `Ok(None)` and
//! the handler responds 503 (Service Unavailable). This keeps the
//! endpoint a no-op for operators who haven't yet stood up a CA.
//!
//! The seed file holds the ML-DSA-65 signing seed; treat it as
//! sensitive. Compromise of this file gives an attacker the
//! ability to mint arbitrary certs the operator's clients will
//! trust. Recommended permissions: `0600`, owner-only.

use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::auth::token::TokenClaims;
use crate::auth::{middleware, rights};
use crate::pqc::certificate::OmegaCert;
use crate::AppState;

type SharedState = std::sync::Arc<tokio::sync::RwLock<AppState>>;

/// Issuer-side configuration loaded once per request from env.
/// Re-reading the seed file each call keeps key rotation simple —
/// drop the new seed in place atomically (`mv` on the same fs) and
/// next issuance picks it up.
pub struct IssuerConfig {
    /// Subject name written into `cert.body.issuer`.
    pub subject: String,
    /// 32-byte ML-DSA-65 signing seed.
    pub seed: [u8; 32],
}

// Manual Debug — print the subject but never the seed bytes (they're
// the issuing CA's signing material). Without this, an `assert!(...,
// "{result:?}")` in a downstream test could log the seed.
impl std::fmt::Debug for IssuerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IssuerConfig")
            .field("subject", &self.subject)
            .field("seed", &"<redacted 32 bytes>")
            .finish()
    }
}

impl IssuerConfig {
    /// Read the issuer config from env. Returns `Ok(None)` when
    /// neither env var is set — the handler maps that to 503.
    /// Errors here surface as 500 so the operator can fix the
    /// misconfiguration (set one but not the other, unreadable
    /// seed file, wrong-sized seed).
    pub fn from_env() -> Result<Option<Self>, String> {
        let subject = match std::env::var("OMEGA_PKI_ISSUER_SUBJECT") {
            Ok(s) if !s.is_empty() => s,
            _ => return Ok(None),
        };
        let path = std::env::var("OMEGA_PKI_ISSUER_SEED_FILE").map_err(|_| {
            "OMEGA_PKI_ISSUER_SUBJECT is set but OMEGA_PKI_ISSUER_SEED_FILE is missing".to_string()
        })?;
        let bytes = std::fs::read(&path)
            .map_err(|e| format!("read OMEGA_PKI_ISSUER_SEED_FILE ({path}): {e}"))?;
        let seed: [u8; 32] = bytes.as_slice().try_into().map_err(|_| {
            format!(
                "issuer seed must be exactly 32 bytes, got {} from {path}",
                bytes.len()
            )
        })?;
        Ok(Some(IssuerConfig { subject, seed }))
    }
}

/// Inbound JSON for `POST /v1/pki/issue`.
#[derive(Deserialize)]
pub struct IssueRequest {
    /// The subject identifier the issued cert will bind.
    pub subject: String,
    /// Cert lifetime in seconds. Must be positive.
    pub ttl_seconds: i64,
    /// Subject's ML-DSA-65 verifying key, base64-encoded.
    pub subject_dsa_pk_b64: String,
    /// Optional ML-KEM-768 encapsulation key, base64-encoded.
    /// Include this when the subject also speaks PQC WS handshakes.
    #[serde(default)]
    pub subject_kem_pk_b64: Option<String>,
}

/// Outbound JSON for `POST /v1/pki/issue`.
#[derive(Serialize, Debug)]
pub struct IssueResponse {
    /// Signed cert as CBOR, base64-encoded. Operators typically
    /// base64-decode and write to `<subject>.cbor` next to the
    /// trust store.
    pub cert_cbor_b64: String,
    /// Lower-case hex of the cert's 16-byte serial — useful for
    /// later revocation lookups.
    pub serial_hex: String,
    /// Echoed for client convenience.
    pub issuer: String,
    pub subject: String,
    pub not_before: i64,
    pub not_after: i64,
}

/// Build a signed cert from a request. Pure function; the axum
/// handler is a thin wrapper that loads the issuer config and
/// shapes the HTTP response.
pub fn issue_from_request(
    issuer: &IssuerConfig,
    req: &IssueRequest,
) -> Result<IssueResponse, String> {
    if req.subject.is_empty() {
        return Err("subject must be non-empty".to_string());
    }
    if req.ttl_seconds <= 0 {
        return Err("ttl_seconds must be > 0".to_string());
    }
    let pk = base64::engine::general_purpose::STANDARD
        .decode(req.subject_dsa_pk_b64.as_bytes())
        .map_err(|e| format!("subject_dsa_pk_b64: {e}"))?;
    let kem_pk = match req.subject_kem_pk_b64.as_deref() {
        Some(b64) => Some(
            base64::engine::general_purpose::STANDARD
                .decode(b64.as_bytes())
                .map_err(|e| format!("subject_kem_pk_b64: {e}"))?,
        ),
        None => None,
    };
    let cert = OmegaCert::sign_with_issuer(
        &req.subject,
        req.ttl_seconds,
        &pk,
        kem_pk,
        &issuer.subject,
        &issuer.seed,
    )?;
    let cbor = cert.to_cbor();
    let cbor_b64 = base64::engine::general_purpose::STANDARD.encode(&cbor);
    let serial_hex: String = cert
        .body
        .serial
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    Ok(IssueResponse {
        cert_cbor_b64: cbor_b64,
        serial_hex,
        issuer: cert.body.issuer,
        subject: cert.body.subject,
        not_before: cert.body.not_before,
        not_after: cert.body.not_after,
    })
}

/// `POST /v1/pki/issue` — admin-only cert issuance.
pub async fn issue_cert(
    Extension(claims): Extension<TokenClaims>,
    State(_state): State<SharedState>,
    Json(req): Json<IssueRequest>,
) -> impl IntoResponse {
    if let Err(resp) = middleware::check_rights(&claims, rights::ADMIN) {
        return resp;
    }
    let issuer = match IssuerConfig::from_env() {
        Ok(Some(c)) => c,
        Ok(None) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "PKI issuance not configured: set OMEGA_PKI_ISSUER_SUBJECT and OMEGA_PKI_ISSUER_SEED_FILE"
                })),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e })),
            )
                .into_response();
        }
    };
    match issue_from_request(&issuer, &req) {
        Ok(resp) => (StatusCode::CREATED, Json(resp)).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_dsa::{MlDsa65, Seed as DsaSeed, SigningKey};

    fn fresh_seed() -> [u8; 32] {
        rand::random()
    }

    #[test]
    fn issue_from_request_signs_with_issuer_seed() {
        let issuer = IssuerConfig {
            subject: "omega-root".into(),
            seed: fresh_seed(),
        };
        let subject_seed = fresh_seed();
        let subject_sk = SigningKey::<MlDsa65>::from_seed(&DsaSeed::from(subject_seed));
        let subject_vk_enc = subject_sk.verifying_key().encode();
        let subject_pk_bytes: &[u8] = subject_vk_enc.as_ref();
        let pk_b64 = base64::engine::general_purpose::STANDARD.encode(subject_pk_bytes);

        let req = IssueRequest {
            subject: "alice".into(),
            ttl_seconds: 3600,
            subject_dsa_pk_b64: pk_b64,
            subject_kem_pk_b64: None,
        };
        let resp = issue_from_request(&issuer, &req).expect("issue");
        assert_eq!(resp.subject, "alice");
        assert_eq!(resp.issuer, "omega-root");
        assert_eq!(resp.serial_hex.len(), 32); // 16 bytes -> 32 hex chars

        // Decode the returned cert and verify it against the issuer's pk.
        let cbor = base64::engine::general_purpose::STANDARD
            .decode(resp.cert_cbor_b64)
            .unwrap();
        let cert = OmegaCert::from_cbor(&cbor).unwrap();

        let issuer_sk = SigningKey::<MlDsa65>::from_seed(&DsaSeed::from(issuer.seed));
        let issuer_vk_enc = issuer_sk.verifying_key().encode();
        let issuer_pk_bytes: &[u8] = issuer_vk_enc.as_ref();
        cert.verify_signed_by(issuer_pk_bytes)
            .expect("issued cert verifies against issuer key");
    }

    #[test]
    fn issue_from_request_rejects_empty_subject() {
        let issuer = IssuerConfig {
            subject: "r".into(),
            seed: fresh_seed(),
        };
        let req = IssueRequest {
            subject: "".into(),
            ttl_seconds: 60,
            subject_dsa_pk_b64: "AAAA".into(), // shape doesn't matter, fails earlier
            subject_kem_pk_b64: None,
        };
        let result = issue_from_request(&issuer, &req);
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("subject"))
                .unwrap_or(false),
            "expected subject rejection, got {result:?}"
        );
    }

    #[test]
    fn issue_from_request_rejects_non_positive_ttl() {
        let issuer = IssuerConfig {
            subject: "r".into(),
            seed: fresh_seed(),
        };
        let req = IssueRequest {
            subject: "alice".into(),
            ttl_seconds: 0,
            subject_dsa_pk_b64: "AAAA".into(),
            subject_kem_pk_b64: None,
        };
        let result = issue_from_request(&issuer, &req);
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("ttl_seconds"))
                .unwrap_or(false),
            "expected ttl rejection, got {result:?}"
        );
    }

    #[test]
    fn issue_from_request_rejects_bad_base64() {
        let issuer = IssuerConfig {
            subject: "r".into(),
            seed: fresh_seed(),
        };
        let req = IssueRequest {
            subject: "alice".into(),
            ttl_seconds: 60,
            subject_dsa_pk_b64: "not!valid!base64".into(),
            subject_kem_pk_b64: None,
        };
        let result = issue_from_request(&issuer, &req);
        assert!(
            result
                .as_ref()
                .err()
                .map(|e| e.contains("subject_dsa_pk_b64"))
                .unwrap_or(false),
            "expected base64 rejection, got {result:?}"
        );
    }

    // Both env-precedence tests touch the same OMEGA_PKI_ISSUER_*
    // vars, so they must serialise. Cargo runs tests in parallel by
    // default, and a race here would mask real failures with
    // intermittent ones. One mutex shared between the two cases is
    // simpler than the `serial_test` dep.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_unset_returns_none() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("OMEGA_PKI_ISSUER_SUBJECT");
        std::env::remove_var("OMEGA_PKI_ISSUER_SEED_FILE");
        let cfg = IssuerConfig::from_env().unwrap();
        assert!(cfg.is_none());
    }

    #[test]
    fn from_env_subject_without_seed_file_errors() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::set_var("OMEGA_PKI_ISSUER_SUBJECT", "test-root");
        std::env::remove_var("OMEGA_PKI_ISSUER_SEED_FILE");
        let result = IssuerConfig::from_env();
        // Always restore so a subsequent test gets a clean env even
        // if this one panics partway through.
        std::env::remove_var("OMEGA_PKI_ISSUER_SUBJECT");
        match result {
            Err(e) => assert!(
                e.contains("OMEGA_PKI_ISSUER_SEED_FILE"),
                "expected SEED_FILE-missing message, got {e}"
            ),
            Ok(_) => panic!("expected partial-config Err, got Ok"),
        }
    }
}
