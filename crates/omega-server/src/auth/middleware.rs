//! Axum middleware for PQC token authentication.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use tokio::sync::RwLock;

use super::rights;
use super::store;
use super::token::{self, TokenClaims};
use crate::AppState;

type SharedState = Arc<RwLock<AppState>>;

/// Middleware that extracts and verifies the Bearer token.
/// On success, inserts `TokenClaims` into request extensions.
pub async fn require_auth(
    State(state): State<SharedState>,
    mut req: Request<Body>,
    next: Next,
) -> Result<Response, Response> {
    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            (
                StatusCode::UNAUTHORIZED,
                axum::Json(serde_json::json!({"error": "missing Authorization header"})),
            )
                .into_response()
        })?;

    let token_str = auth_header.strip_prefix("Bearer ").ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "expected Bearer token"})),
        )
            .into_response()
    })?;

    // First pass: decode payload to get kid without full verification
    let parts: Vec<&str> = token_str.splitn(2, '.').collect();
    if parts.len() != 2 {
        return Err((
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "invalid token format"})),
        )
            .into_response());
    }

    let payload_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(parts[0])
        .map_err(|_| {
            (
                StatusCode::UNAUTHORIZED,
                axum::Json(serde_json::json!({"error": "invalid token encoding"})),
            )
                .into_response()
        })?;

    let partial_claims: serde_json::Value =
        serde_json::from_slice(&payload_bytes).map_err(|_| {
            (
                StatusCode::UNAUTHORIZED,
                axum::Json(serde_json::json!({"error": "invalid token payload"})),
            )
                .into_response()
        })?;

    let kid = partial_claims["kid"].as_str().ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "missing kid in token"})),
        )
            .into_response()
    })?;

    // Look up public key
    let state_read = state.read().await;
    let pk_bytes = if kid == state_read.active_kid {
        state_read.active_pk.clone()
    } else {
        store::get_public_key(&state_read.registry.conn(), kid).map_err(|_| {
            (
                StatusCode::UNAUTHORIZED,
                axum::Json(serde_json::json!({"error": "unknown signing key"})),
            )
                .into_response()
        })?
    };

    // Verify token
    let claims = token::verify_token(token_str, &pk_bytes).map_err(|e| {
        (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": e})),
        )
            .into_response()
    })?;

    // Check revocation
    let revoked =
        store::is_token_revoked(&state_read.registry.conn(), &claims.jti).unwrap_or(false);
    if revoked {
        return Err((
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "token revoked"})),
        )
            .into_response());
    }

    drop(state_read);

    // Inject claims into request extensions
    req.extensions_mut().insert(claims);
    Ok(next.run(req).await)
}

/// Helper to check rights in a handler. Returns Err(Response) on insufficient rights.
#[allow(clippy::result_large_err)]
pub fn check_rights(claims: &TokenClaims, required: u32) -> Result<(), Response> {
    if !rights::has_right(claims.rights, required) {
        return Err((
            StatusCode::FORBIDDEN,
            axum::Json(serde_json::json!({
                "error": "insufficient rights",
                "required": rights::rights_to_names(required),
                "granted": rights::rights_to_names(claims.rights),
            })),
        )
            .into_response());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    fn claims_with_rights(r: u32) -> TokenClaims {
        TokenClaims {
            jti: "test-jti".into(),
            sub: "alice".into(),
            iat: 0,
            exp: i64::MAX,
            rights: r,
            kid: "test-kid".into(),
        }
    }

    #[test]
    fn check_rights_passes_when_required_subset_granted() {
        let claims = claims_with_rights(rights::ADMIN_ROLE);
        check_rights(&claims, rights::READ).unwrap();
        check_rights(&claims, rights::EXECUTE).unwrap();
        check_rights(&claims, rights::WRITE).unwrap();
        check_rights(&claims, rights::ADMIN).unwrap();
        // Compound requirement: still a subset.
        check_rights(&claims, rights::READ | rights::WRITE).unwrap();
    }

    #[test]
    fn check_rights_passes_when_required_is_zero() {
        // No rights required → always allow, regardless of grants.
        let claims = claims_with_rights(0);
        check_rights(&claims, 0).unwrap();
        let admin = claims_with_rights(rights::ADMIN_ROLE);
        check_rights(&admin, 0).unwrap();
    }

    #[tokio::test]
    async fn check_rights_returns_403_with_required_and_granted_in_body() {
        // VIEWER lacks WRITE — middleware must reject with 403.
        let claims = claims_with_rights(rights::VIEWER);
        let err = check_rights(&claims, rights::WRITE).expect_err("must reject");
        assert_eq!(err.status(), StatusCode::FORBIDDEN);
        // Response body carries the required + granted names so the
        // client can show a meaningful error.
        let (_parts, body) = err.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"], "insufficient rights");
        assert!(v["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n == "write"));
        assert!(v["granted"].as_array().unwrap().iter().any(|n| n == "read"));
    }

    #[test]
    fn check_rights_partial_overlap_still_rejects() {
        // OPERATOR has READ + EXECUTE but the requirement also asks
        // for WRITE — must reject (subset semantics, not any-overlap).
        let claims = claims_with_rights(rights::OPERATOR);
        let err = check_rights(&claims, rights::READ | rights::WRITE)
            .expect_err("partial overlap must reject");
        assert_eq!(err.status(), StatusCode::FORBIDDEN);
    }
}
