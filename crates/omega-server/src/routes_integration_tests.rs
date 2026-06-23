//! End-to-end axum-router integration tests.
//!
//! These tests exercise the actual `routes::create_router` output via
//! `tower::ServiceExt::oneshot`, so they cover the full middleware
//! pipeline (per-IP rate limit, auth, body limit, CORS) plus the
//! handlers themselves — none of which are reachable from the
//! per-module unit tests.
//!
//! These tests caught a pre-existing production bug where the per-IP
//! rate-limit middleware was wired via `Extension(Arc<...>)` but
//! axum's serve dispatch ran the middleware before the Extension
//! layer added the limiter to request extensions, so EVERY request
//! (including `/health`) returned 500 with "Missing request
//! extension". The fix wires the limiter via
//! `axum::middleware::from_fn_with_state` instead. Keep these tests
//! green on every change to `routes.rs::create_router` or the
//! `auth::rate_limit` module.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{header, Request, StatusCode};
use tokio::sync::RwLock;
use tower::ServiceExt;

use crate::auth::{rights, store, token};
use crate::pqc::crl::OmegaCrl;
use crate::pqc::trust_store::TrustStore;
use crate::registry::Registry;
use crate::routes::create_router;
use crate::AppState;

fn fresh_state_with_token(token_rights: u32) -> (Arc<RwLock<AppState>>, String) {
    let registry = Registry::new(":memory:").expect("registry");
    {
        let conn = registry.conn();
        store::init_auth_tables(&conn).expect("init auth tables");
    }
    let (kid, active_pk, active_sk) = {
        let conn = registry.conn();
        store::ensure_signing_key(&conn).expect("ensure key")
    };
    let (token_str, claims) =
        token::issue_token("alice", token_rights, 3600, &kid, &active_sk).expect("issue token");
    {
        let conn = registry.conn();
        store::store_token(&conn, &claims).expect("store token");
    }
    let state = Arc::new(RwLock::new(AppState {
        registry,
        active_kid: kid,
        active_pk,
        active_sk,
        trust_store: Arc::new(TrustStore::empty()),
        crl: None::<Arc<OmegaCrl>>,
    }));
    (state, token_str)
}

/// Build a router for the default (non-rate-limit-tweaked) tests.
/// Takes the rate-limit env-var mutex during build so a parallel
/// rate-limit test that's mid-setup doesn't bleed its custom limit
/// into this router's `build_subject_limiter` call. Also clears
/// both rate-limit env vars to defeat any leftover from earlier
/// tests in the same process.
fn fresh_router_default(token_rights: u32) -> (String, axum::Router) {
    let _g = RATELIMIT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("OMEGA_RATELIMIT_PER_SUBJECT_PER_MIN");
    std::env::remove_var("OMEGA_RATELIMIT_PER_IP_PER_MIN");
    let (state, token) = fresh_state_with_token(token_rights);
    let app = create_router(state, None);
    (token, app)
}

fn req_get(path: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(path)
        .body(Body::empty())
        .unwrap()
}

fn req_get_auth(path: &str, token: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(path)
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap()
}

fn req_post_auth(path: &str, token: &str, body: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap()
}

fn req_delete_auth(path: &str, token: &str) -> Request<Body> {
    Request::builder()
        .method("DELETE")
        .uri(path)
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap()
}

async fn read_body_json(body: Body) -> serde_json::Value {
    let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

// --------------------------------------------------------------------
// /health (public, no auth) — the bug detector
// --------------------------------------------------------------------

#[tokio::test]
async fn health_endpoint_returns_200_without_auth() {
    // This test directly catches the per-IP rate-limit Extension bug.
    // Pre-fix, every request returned 500 "Missing request extension".
    let (_token, app) = fresh_router_default(rights::ADMIN_ROLE);
    let resp = app.oneshot(req_get("/health")).await.unwrap();
    let status = resp.status();
    if status != StatusCode::OK {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&bytes);
        panic!("expected 200, got {status}: body = {text:?}");
    }
    let body = read_body_json(resp.into_body()).await;
    let _ = body; // /health body shape is not pinned here
}

// --------------------------------------------------------------------
// Auth gate: 401 paths
// --------------------------------------------------------------------

#[tokio::test]
async fn protected_route_returns_401_without_bearer() {
    let (_token, app) = fresh_router_default(rights::READ);
    let resp = app.oneshot(req_get("/v1/circuits")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = read_body_json(resp.into_body()).await;
    assert_eq!(body["error"], "missing Authorization header");
}

#[tokio::test]
async fn protected_route_returns_401_when_bearer_prefix_missing() {
    let (token, app) = fresh_router_default(rights::READ);
    let req = Request::builder()
        .method("GET")
        .uri("/v1/circuits")
        .header(header::AUTHORIZATION, token) // no "Bearer " prefix
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = read_body_json(resp.into_body()).await;
    assert_eq!(body["error"], "expected Bearer token");
}

#[tokio::test]
async fn protected_route_returns_401_on_garbage_token() {
    let (_token, app) = fresh_router_default(rights::READ);
    let resp = app
        .oneshot(req_get_auth("/v1/circuits", "definitely.not.a.token"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn csrf_token_smuggled_via_cookie_is_rejected() {
    // The server's CSRF stance is "Authorization header only, never
    // cookies". A browser-driven CSRF attack would land an attacker's
    // forged POST with the victim's session cookie auto-attached; this
    // test pins that even a *valid* token sent as a `Cookie:
    // omega_token=…` header is treated as unauthenticated. Combined
    // with the deny-all CORS layer, this closes the classic
    // form-CSRF + cookie-auth vector.
    let (token, app) = fresh_router_default(rights::READ);
    let req = Request::builder()
        .method("GET")
        .uri("/v1/circuits")
        .header(header::COOKIE, format!("omega_token={token}"))
        .header(header::ORIGIN, "https://attacker.example")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = read_body_json(resp.into_body()).await;
    assert_eq!(body["error"], "missing Authorization header");
}

#[tokio::test]
async fn csrf_mutating_post_without_auth_rejected_before_csrf_token_check() {
    // Belt-and-braces: a mutating endpoint with no Authorization at all
    // must 401 — never silently accept the body, never set a session
    // cookie. The /v1/circuits POST is the canonical write path.
    let (_token, app) = fresh_router_default(rights::ADMIN_ROLE);
    let body = serde_json::json!({"source": "OPENQASM 2.0;\nqreg q[1];\n"}).to_string();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/circuits")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ORIGIN, "https://attacker.example")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    // Crucially: response must NOT set a cookie (we don't issue any).
    assert!(
        resp.headers().get(header::SET_COOKIE).is_none(),
        "server must never set Set-Cookie on unauth POST"
    );
}

// --------------------------------------------------------------------
// Rights gate: 403 paths
// --------------------------------------------------------------------

#[tokio::test]
async fn create_circuit_requires_write_right() {
    let (token, app) = fresh_router_default(rights::READ);
    let body = serde_json::json!({
        "source": "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n"
    })
    .to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/circuits", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let v = read_body_json(resp.into_body()).await;
    assert_eq!(v["error"], "insufficient rights");
    assert!(v["required"]
        .as_array()
        .unwrap()
        .iter()
        .any(|n| n == "write"));
}

// --------------------------------------------------------------------
// Happy-path round-trip
// --------------------------------------------------------------------

#[tokio::test]
async fn circuit_post_get_list_round_trip() {
    let (token, app) = fresh_router_default(rights::ADMIN_ROLE);
    let body = serde_json::json!({
        "source": "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n"
    })
    .to_string();
    let resp = app
        .clone()
        .oneshot(req_post_auth("/v1/circuits", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let created = read_body_json(resp.into_body()).await;
    assert_eq!(created["num_qubits"], 2);
    assert_eq!(created["circuit_type"], "gate_based");
    let id = created["id"].as_str().unwrap().to_string();

    let resp = app
        .oneshot(req_get_auth("/v1/circuits", &token))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let listed = read_body_json(resp.into_body()).await;
    let arr = listed["circuits"].as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["id"], id);
}

#[tokio::test]
async fn backends_endpoint_lists_compiled_backends() {
    let (token, app) = fresh_router_default(rights::READ);
    let resp = app
        .oneshot(req_get_auth("/v1/backends", &token))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = read_body_json(resp.into_body()).await;
    let arr = v["backends"].as_array().unwrap();
    assert!(!arr.is_empty(), "at least one backend must be reported");
    assert!(arr.iter().any(|b| b["name"] == "statevector"));
}

// The OMEGA_RATELIMIT_* env vars are read inside `create_router` via
// `build_ip_limiter` / `build_subject_limiter`. Tests that mutate them
// share a process-wide mutex (mirrors CRL_ENV_LOCK / POLICY_ENV_LOCK
// patterns elsewhere) so parallel cargo test stays deterministic.
static RATELIMIT_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Build (token, app) with the given per-subject rate-limit env var
/// transiently set so `create_router` picks it up, then immediately
/// cleared. Holds the env-var mutex only across the synchronous
/// setup so subsequent `.await` calls don't trip clippy's
/// `await_holding_lock`.
fn router_with_subject_rate_limit(limit: &str, token_rights: u32) -> (String, axum::Router) {
    let _g = RATELIMIT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("OMEGA_RATELIMIT_PER_SUBJECT_PER_MIN", limit);
    let (state, token) = fresh_state_with_token(token_rights);
    let app = create_router(state, None);
    std::env::remove_var("OMEGA_RATELIMIT_PER_SUBJECT_PER_MIN");
    (token, app)
}

#[tokio::test]
async fn per_subject_rate_limiter_fires_through_full_router() {
    // Per-subject limit = 1 means: first authenticated request → 200,
    // second → 429 with Retry-After. Per-IP limiter is left at the
    // default and irrelevant — `oneshot` doesn't attach ConnectInfo
    // so the per-IP middleware is a no-op in tests.
    let (token, app) = router_with_subject_rate_limit("1", rights::READ);

    let resp1 = app
        .clone()
        .oneshot(req_get_auth("/v1/circuits", &token))
        .await
        .unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);

    let resp2 = app
        .clone()
        .oneshot(req_get_auth("/v1/circuits", &token))
        .await
        .unwrap();
    assert_eq!(resp2.status(), StatusCode::TOO_MANY_REQUESTS);
    let retry_after = resp2.headers().get("retry-after");
    assert!(retry_after.is_some(), "429 must carry a Retry-After header");

    let resp3 = app
        .oneshot(req_get_auth("/v1/circuits", &token))
        .await
        .unwrap();
    assert_eq!(resp3.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn per_subject_rate_limiter_zero_disables() {
    // Limit = 0 documented to disable the throttle. Hammer the
    // endpoint and verify every request lands.
    let (token, app) = router_with_subject_rate_limit("0", rights::READ);

    for _ in 0..20 {
        let resp = app
            .clone()
            .oneshot(req_get_auth("/v1/circuits", &token))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

// --------------------------------------------------------------------
// Function invoke + GET unknown circuit + auth token issue + lambda input validation
// --------------------------------------------------------------------

#[tokio::test]
async fn function_create_then_invoke_executes_bell() {
    // Full path: register a Bell circuit, define a function over it,
    // invoke with shots → counts are exactly the Bell distribution.
    let (token, app) = fresh_router_default(rights::ADMIN_ROLE);

    // 1. Create circuit.
    let body = serde_json::json!({
        "source": "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n"
    })
    .to_string();
    let resp = app
        .clone()
        .oneshot(req_post_auth("/v1/circuits", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let cid = read_body_json(resp.into_body()).await["id"]
        .as_str()
        .unwrap()
        .to_string();

    // 2. Create function that points at it.
    let body =
        serde_json::json!({"circuit_id": cid, "name": "bell", "default_shots": 256}).to_string();
    let resp = app
        .clone()
        .oneshot(req_post_auth("/v1/functions", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let fid = read_body_json(resp.into_body()).await["id"]
        .as_str()
        .unwrap()
        .to_string();

    // 3. Invoke. Assert: 256 shots, exactly two outcomes "00"+"11"
    // each within 35-65% of the total (loose bound — the seed-less
    // distribution can wander).
    let body = serde_json::json!({"params": [], "shots": 256, "seed": 42}).to_string();
    let resp = app
        .oneshot(req_post_auth(
            &format!("/v1/functions/{}/invoke", fid),
            &token,
            &body,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = read_body_json(resp.into_body()).await;
    assert_eq!(v["status"], "completed");
    let counts = v["result"]["counts"].as_object().unwrap();
    let c00 = counts.get("00").and_then(|c| c.as_u64()).unwrap_or(0);
    let c11 = counts.get("11").and_then(|c| c.as_u64()).unwrap_or(0);
    assert_eq!(c00 + c11, 256, "Bell only produces |00⟩ and |11⟩");
    assert!(
        (80..=180).contains(&c00),
        "|00⟩ count {} outside loose 35-65% band",
        c00
    );
}

#[tokio::test]
async fn get_unknown_circuit_returns_404() {
    let (token, app) = fresh_router_default(rights::READ);
    let resp = app
        .oneshot(req_get_auth("/v1/circuits/no-such-id", &token))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn issue_token_endpoint_returns_signed_token_with_rights_names() {
    // POST /v1/auth/token with admin token issues a fresh token for
    // a new subject. Response carries the encoded token, JTI,
    // expires_at, and the human-readable rights names.
    let (token, app) = fresh_router_default(rights::ADMIN_ROLE);
    let body =
        serde_json::json!({"sub": "bob", "rights": rights::READ, "ttl_seconds": 60}).to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/auth/token", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let v = read_body_json(resp.into_body()).await;
    assert_eq!(v["sub"], "bob");
    assert_eq!(v["rights"], rights::READ);
    assert!(v["jti"].as_str().is_some(), "must carry a JTI");
    assert!(v["token"].as_str().is_some(), "must carry the token");
    let names = v["rights_names"].as_array().unwrap();
    assert!(
        names.iter().any(|n| n == "read"),
        "rights_names must carry `read`"
    );
}

#[tokio::test]
async fn issue_token_endpoint_requires_admin_right() {
    // The /v1/auth/token endpoint is admin-only — a token with
    // only READ can't mint new tokens.
    let (token, app) = fresh_router_default(rights::READ);
    let body = serde_json::json!({"sub": "alice", "rights": 1, "ttl_seconds": 60}).to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/auth/token", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn revoke_token_round_trip_then_404_on_repeat() {
    // Admin issues a token, revokes it via DELETE /v1/auth/token/:jti
    // (204 NO_CONTENT), revokes again (404 — already-revoked rows
    // are filtered out by the UPDATE clause; matches the
    // store::revoke_token contract pinned by
    // `auth::store::tests::store_and_revoke_token_round_trip`).
    let (admin_token, app) = fresh_router_default(rights::ADMIN_ROLE);

    // Mint a fresh token whose jti we know.
    let body = serde_json::json!({"sub": "carol", "rights": 1, "ttl_seconds": 60}).to_string();
    let issue_resp = app
        .clone()
        .oneshot(req_post_auth("/v1/auth/token", &admin_token, &body))
        .await
        .unwrap();
    assert_eq!(issue_resp.status(), StatusCode::CREATED);
    let v = read_body_json(issue_resp.into_body()).await;
    let jti = v["jti"].as_str().unwrap().to_string();

    // First revoke: 204 NO_CONTENT.
    let resp = app
        .clone()
        .oneshot(req_delete_auth(
            &format!("/v1/auth/token/{}", jti),
            &admin_token,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // Second revoke of the same jti: 404 — the UPDATE only matches
    // rows whose `revoked_at` is NULL, so the second call returns
    // affected_rows = 0 → handler maps to NotFound.
    let resp = app
        .oneshot(req_delete_auth(
            &format!("/v1/auth/token/{}", jti),
            &admin_token,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn revoke_token_requires_admin_right() {
    // Non-admin can't revoke even a non-existent jti — auth gate
    // fires before the store lookup.
    let (token, app) = fresh_router_default(rights::READ | rights::WRITE);
    let resp = app
        .oneshot(req_delete_auth("/v1/auth/token/some-jti", &token))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

// --------------------------------------------------------------------
// /v1/functions GET list + GET by id
// --------------------------------------------------------------------

#[tokio::test]
async fn functions_list_starts_empty_then_reflects_creates() {
    let (token, app) = fresh_router_default(rights::ADMIN_ROLE);

    // Initially empty.
    let resp = app
        .clone()
        .oneshot(req_get_auth("/v1/functions", &token))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = read_body_json(resp.into_body()).await;
    assert_eq!(v["functions"].as_array().unwrap().len(), 0);

    // Create circuit + two functions.
    let body = serde_json::json!({
        "source": "OPENQASM 2.0;\nqreg q[1];\nh q[0];\n"
    })
    .to_string();
    let r = app
        .clone()
        .oneshot(req_post_auth("/v1/circuits", &token, &body))
        .await
        .unwrap();
    let cid = read_body_json(r.into_body()).await["id"]
        .as_str()
        .unwrap()
        .to_string();
    for name in ["f1", "f2"] {
        let body = serde_json::json!({
            "circuit_id": cid,
            "name": name,
            "default_shots": 10,
        })
        .to_string();
        let r = app
            .clone()
            .oneshot(req_post_auth("/v1/functions", &token, &body))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::CREATED);
    }

    let resp = app
        .oneshot(req_get_auth("/v1/functions", &token))
        .await
        .unwrap();
    let v = read_body_json(resp.into_body()).await;
    let arr = v["functions"].as_array().unwrap();
    assert_eq!(arr.len(), 2);
    let names: Vec<&str> = arr.iter().map(|e| e["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"f1") && names.contains(&"f2"));
}

#[tokio::test]
async fn function_get_by_id_then_unknown_404() {
    let (token, app) = fresh_router_default(rights::ADMIN_ROLE);
    // Create circuit + function.
    let body = serde_json::json!({"source": "OPENQASM 2.0;\nqreg q[1];\nh q[0];\n"}).to_string();
    let r = app
        .clone()
        .oneshot(req_post_auth("/v1/circuits", &token, &body))
        .await
        .unwrap();
    let cid = read_body_json(r.into_body()).await["id"]
        .as_str()
        .unwrap()
        .to_string();
    let body = serde_json::json!({
        "circuit_id": cid, "name": "fn-named", "default_shots": 50,
    })
    .to_string();
    let r = app
        .clone()
        .oneshot(req_post_auth("/v1/functions", &token, &body))
        .await
        .unwrap();
    let fid = read_body_json(r.into_body()).await["id"]
        .as_str()
        .unwrap()
        .to_string();

    // GET by id round-trips the same record.
    let resp = app
        .clone()
        .oneshot(req_get_auth(&format!("/v1/functions/{}", fid), &token))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = read_body_json(resp.into_body()).await;
    assert_eq!(v["id"], fid);
    assert_eq!(v["name"], "fn-named");
    assert_eq!(v["default_shots"], 50);

    // Unknown id surfaces from the registry.
    let resp = app
        .oneshot(req_get_auth("/v1/functions/nope", &token))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// --------------------------------------------------------------------
// /v1/lambdas GET list + delete + 404 paths
// --------------------------------------------------------------------

#[tokio::test]
async fn lambdas_list_starts_empty_and_unknown_lookup_404() {
    let (token, app) = fresh_router_default(rights::ADMIN_ROLE);

    let resp = app
        .clone()
        .oneshot(req_get_auth("/v1/lambdas", &token))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = read_body_json(resp.into_body()).await;
    assert_eq!(v["lambdas"].as_array().unwrap().len(), 0);

    let resp = app
        .clone()
        .oneshot(req_get_auth("/v1/lambdas/no-such", &token))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    // Delete unknown lambda → 404.
    let resp = app
        .oneshot(req_delete_auth("/v1/lambdas/no-such", &token))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// --------------------------------------------------------------------
// /v1/invocations/:id — GET unknown
// --------------------------------------------------------------------

#[tokio::test]
async fn invocation_get_unknown_returns_404() {
    let (token, app) = fresh_router_default(rights::READ);
    let resp = app
        .oneshot(req_get_auth("/v1/invocations/nope", &token))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// --------------------------------------------------------------------
// /v1/circuits/:id DELETE
// --------------------------------------------------------------------

#[tokio::test]
async fn delete_circuit_round_trip_then_404_on_repeat() {
    let (token, app) = fresh_router_default(rights::ADMIN_ROLE);

    let body = serde_json::json!({"source": "OPENQASM 2.0;\nqreg q[1];\nh q[0];\n"}).to_string();
    let r = app
        .clone()
        .oneshot(req_post_auth("/v1/circuits", &token, &body))
        .await
        .unwrap();
    let cid = read_body_json(r.into_body()).await["id"]
        .as_str()
        .unwrap()
        .to_string();

    // First delete: 204 NO_CONTENT.
    let resp = app
        .clone()
        .oneshot(req_delete_auth(&format!("/v1/circuits/{}", cid), &token))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // Second delete (already gone): 404.
    let resp = app
        .oneshot(req_delete_auth(&format!("/v1/circuits/{}", cid), &token))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// --------------------------------------------------------------------
// /v1/pki/issue — admin endpoint, returns 503 when CA isn't configured
// --------------------------------------------------------------------

/// Build a router for pki-tests with the issuer env vars cleared.
/// Holds the env-var mutex only across the synchronous build.
fn router_pki_unconfigured() -> (String, axum::Router) {
    static PKI_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = PKI_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("OMEGA_PKI_ISSUER_SUBJECT");
    std::env::remove_var("OMEGA_PKI_ISSUER_SEED_FILE");
    fresh_router_default(rights::ADMIN_ROLE)
}

#[tokio::test]
async fn pki_issue_returns_503_when_issuer_not_configured() {
    // Without OMEGA_PKI_ISSUER_SUBJECT + OMEGA_PKI_ISSUER_SEED_FILE
    // the endpoint returns 503 (the server has the route but no CA
    // backing it). Pin that so an operator hitting the endpoint
    // without setup gets a clear error rather than a 500.
    let (token, app) = router_pki_unconfigured();
    let body = serde_json::json!({
        "subject": "alice",
        "ttl_seconds": 60,
        "subject_dsa_pk_b64": "ignored",
    })
    .to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/pki/issue", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let v = read_body_json(resp.into_body()).await;
    assert!(
        v["error"]
            .as_str()
            .unwrap_or("")
            .contains("OMEGA_PKI_ISSUER"),
        "503 body must mention the env vars: {v}"
    );
}

#[tokio::test]
async fn pki_issue_requires_admin_right() {
    let (token, app) = fresh_router_default(rights::READ | rights::WRITE | rights::EXECUTE);
    let body = serde_json::json!({
        "subject": "alice", "ttl_seconds": 60, "subject_dsa_pk_b64": "x",
    })
    .to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/pki/issue", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

// --------------------------------------------------------------------
// /v1/quantum/execute Auto backend dispatch
// --------------------------------------------------------------------

#[tokio::test]
async fn quantum_execute_auto_picks_stabilizer_for_clifford_only() {
    // Auto-backend selection in `resolve_backend` picks Stabilizer
    // when the circuit is Clifford-only. Bell state (H + CX) qualifies.
    // Pin the dispatch decision via the `backend` field in the
    // response.
    let (token, app) = fresh_router_default(rights::EXECUTE);
    let body = serde_json::json!({
        "circuit": {
            "num_qubits": 2,
            "num_classical_bits": 0,
            "is_photonic": false,
            "mid_circuit_mode": "Skip",
            "backend": "Auto",
            "ops": [
                {"gate": "H",  "qubits": [0],    "params": [], "classical_bit": null, "condition": null},
                {"gate": "CX", "qubits": [0, 1], "params": [], "classical_bit": null, "condition": null}
            ]
        },
        "shots": 256,
        "seed": 7
    })
    .to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/quantum/execute", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = read_body_json(resp.into_body()).await;
    assert_eq!(
        v["backend"], "stabilizer",
        "Clifford-only Bell circuit should route to Stabilizer under Auto"
    );
    let counts = v["result"]["counts"].as_object().unwrap();
    let c00 = counts.get("00").and_then(|c| c.as_u64()).unwrap_or(0);
    let c11 = counts.get("11").and_then(|c| c.as_u64()).unwrap_or(0);
    assert_eq!(c00 + c11, 256, "Bell only produces |00⟩ and |11⟩");
}

// --------------------------------------------------------------------
// /v1/lambdas/:id/invoke — full HTTP → registry → wasm-runtime path
// --------------------------------------------------------------------

#[tokio::test]
async fn lambda_register_then_invoke_via_http_runs_qaoa_qubo() {
    // Mirrors the in-process `lambda_register_then_run_qaoa_qubo_end_to_end`
    // test in registry.rs, but goes through the HTTP layer:
    //   POST /v1/lambdas (base64-encoded WASM)
    //   POST /v1/lambdas/:id/invoke
    // Skipped when qaoa_qubo.wasm isn't built (matches the registry
    // test's behaviour so `cargo test` works on a fresh checkout).
    use base64::Engine;

    let wasm_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/wasm-guests/qaoa_qubo/target/wasm32-wasip1/release/qaoa_qubo.wasm");
    if !wasm_path.exists() {
        eprintln!(
            "Skipping: qaoa_qubo.wasm not built at {}",
            wasm_path.display()
        );
        return;
    }
    let wasm_bytes = std::fs::read(&wasm_path).unwrap();
    let wasm_b64 = base64::engine::general_purpose::STANDARD.encode(&wasm_bytes);

    let (token, app) = fresh_router_default(rights::ADMIN_ROLE);

    // POST /v1/lambdas — register a lambda holding the qaoa_qubo guest.
    let create_body = serde_json::json!({
        "name": "qaoa_qubo_http_smoke",
        "description": "router-integration end-to-end",
        "wasm_b64": wasm_b64,
        "default_input": "{\"depth\":2,\"max_iters\":150}",
        "fuel": 100_000_000_000i64,
    })
    .to_string();
    let resp = app
        .clone()
        .oneshot(req_post_auth("/v1/lambdas", &token, &create_body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let v = read_body_json(resp.into_body()).await;
    let lid = v["id"].as_str().unwrap().to_string();

    // POST /v1/lambdas/:id/invoke — pass a triangle MaxCut QUBO as
    // the input override; expect the SPSA inside the guest to
    // converge close to the brute-force optimum.
    let invoke_body = serde_json::json!({
        "input": "{\"qubo\":\"{\\\"n\\\":3,\\\"Q\\\":[[0,0,-2],[1,1,-2],[2,2,-2],[0,1,2],[1,2,2],[0,2,2]]}\",\"depth\":2,\"max_iters\":150,\"seed\":7}",
    })
    .to_string();
    let resp = app
        .oneshot(req_post_auth(
            &format!("/v1/lambdas/{}/invoke", lid),
            &token,
            &invoke_body,
        ))
        .await
        .unwrap();
    let status = resp.status();
    let body = read_body_json(resp.into_body()).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "lambda invoke must succeed: body = {body}"
    );
    assert_eq!(body["status"], "completed");
    let optimal_value = body["optimal_value"]
        .as_f64()
        .expect("optimal_value must be present + numeric");
    assert!(
        optimal_value < -1.5,
        "lambda did not converge: optimal_value = {optimal_value}"
    );
    let opt_params = body["optimal_params"].as_array().unwrap();
    assert_eq!(opt_params.len(), 4, "depth=2 → 4 free params");
}

// --------------------------------------------------------------------
// Body limit middleware — pin the documented 16 MiB cap
// --------------------------------------------------------------------

#[tokio::test]
async fn body_limit_middleware_rejects_oversized_payload() {
    // MAX_REQUEST_BODY_BYTES is 16 MiB. A 17 MiB payload must be
    // rejected by the RequestBodyLimitLayer before it ever reaches
    // the handler. The layer returns 413 PAYLOAD_TOO_LARGE.
    let (token, app) = fresh_router_default(rights::ADMIN_ROLE);
    let oversized = "x".repeat(17 * 1024 * 1024);
    let body = serde_json::json!({"source": oversized}).to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/circuits", &token, &body))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::PAYLOAD_TOO_LARGE,
        "17 MiB payload must be rejected as 413 (limit is 16 MiB)"
    );
}

// --------------------------------------------------------------------
// CORS middleware — pin the deny-all default
// --------------------------------------------------------------------

#[tokio::test]
async fn cors_default_does_not_echo_allow_origin() {
    // OMEGA_CORS_ALLOW_ORIGINS unset → CorsLayer with no origins
    // configured. Any request with an `Origin` header should NOT
    // get an `access-control-allow-origin` header back; browsers
    // will refuse the cross-origin XHR/fetch in that case.
    static CORS_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let app_and_token = {
        let _g = CORS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("OMEGA_CORS_ALLOW_ORIGINS");
        fresh_router_default(rights::ADMIN_ROLE)
    };
    let (_token, app) = app_and_token;
    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .header("Origin", "https://attacker.example.com")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(
        resp.headers().get("access-control-allow-origin").is_none(),
        "deny-all default must NOT echo Allow-Origin"
    );
}

#[tokio::test]
async fn cors_options_preflight_does_not_echo_allow_origin() {
    // Under the deny-all default, an OPTIONS preflight may or may
    // not be handled (depends on whether the route registers an
    // OPTIONS handler), but the CORS layer must NOT echo
    // Access-Control-Allow-Origin back to the requesting origin.
    // If it did, browsers would treat that as "this server allows
    // cross-origin requests from the attacker's site".
    static CORS_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let app = {
        let _g = CORS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("OMEGA_CORS_ALLOW_ORIGINS");
        fresh_router_default(rights::ADMIN_ROLE).1
    };
    let req = Request::builder()
        .method("OPTIONS")
        .uri("/v1/circuits")
        .header("Origin", "https://attacker.example.com")
        .header("Access-Control-Request-Method", "POST")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert!(
        resp.headers().get("access-control-allow-origin").is_none(),
        "preflight must not get Allow-Origin under deny-all default"
    );
    assert!(
        resp.headers().get("access-control-allow-methods").is_none(),
        "preflight must not get Allow-Methods under deny-all default"
    );
}

#[tokio::test]
async fn body_limit_middleware_accepts_small_payload() {
    // A modest payload (well under 16 MiB) should pass the body
    // limit and be processed by the handler. Confirms the layer
    // isn't accidentally restricting normal traffic.
    let (token, app) = fresh_router_default(rights::ADMIN_ROLE);
    let body = serde_json::json!({"source": "OPENQASM 2.0;\nqreg q[1];\n"}).to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/circuits", &token, &body))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::CREATED,
        "small payload must pass the body limit"
    );
}

#[tokio::test]
async fn quantum_execute_auto_picks_statevector_for_non_clifford() {
    // Same shape but with a non-Clifford rotation (T gate) → Auto
    // should fall through to Statevector since Stabilizer only
    // accepts Clifford gates.
    let (token, app) = fresh_router_default(rights::EXECUTE);
    let body = serde_json::json!({
        "circuit": {
            "num_qubits": 1,
            "num_classical_bits": 0,
            "is_photonic": false,
            "mid_circuit_mode": "Skip",
            "backend": "Auto",
            "ops": [
                {"gate": "H", "qubits": [0], "params": [], "classical_bit": null, "condition": null},
                {"gate": "T", "qubits": [0], "params": [], "classical_bit": null, "condition": null}
            ]
        },
        "shots": 64,
        "seed": 1
    })
    .to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/quantum/execute", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = read_body_json(resp.into_body()).await;
    assert_eq!(
        v["backend"], "statevector",
        "Non-Clifford circuit must fall through to Statevector under Auto"
    );
}

#[tokio::test]
async fn quantum_execute_pattern_bell_returns_cross_wire_golden() {
    // C1.3/C1.4: POST the compiled Bell MBQC pattern to
    // /v1/quantum/execute_pattern → photonic one-way backend → the canonical
    // output statevector. The exact pattern + golden are pinned identically on
    // the quantum-core side (its `omega_pattern_cross_wire_golden` test), so
    // matching them here proves the over-the-wire equality. MBQC prepares the
    // |+⟩ input, so the Bell circuit's output is CX·(H⊗I)|++⟩ = (|00⟩+|01⟩)/√2.
    let (token, app) = fresh_router_default(rights::EXECUTE);
    let m0 = |q: u32| serde_json::json!({"qubit": q, "angle": 0.0, "x_corr_from": [], "z_corr_from": []});
    let body = serde_json::json!({
        "pattern": {
            "vertices": [0, 1, 2, 3, 4],
            "edges": [[0, 2], [1, 3], [2, 3], [3, 4]],
            "layers": [[m0(0)], [m0(1)], [m0(3)]],
            "output": [2, 4],
            "is_photonic": true
        }
    })
    .to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/quantum/execute_pattern", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = read_body_json(resp.into_body()).await;
    assert_eq!(v["backend"], "photonic");
    assert_eq!(v["result"]["type"], "statevector");
    let amps = v["result"]["amplitudes"].as_array().unwrap();
    assert_eq!(amps.len(), 4, "2 output qubits → 4 amplitudes");
    let get = |i: usize| -> (f64, f64) {
        let p = amps[i].as_array().unwrap();
        (p[0].as_f64().unwrap(), p[1].as_f64().unwrap())
    };
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let golden = [(s, 0.0), (s, 0.0), (0.0, 0.0), (0.0, 0.0)];
    // Compare up to a global phase (golden_k real at the first nonzero index).
    let k = golden
        .iter()
        .position(|&(re, im)| re.hypot(im) > 1e-9)
        .unwrap();
    let (gr, _gi) = golden[k];
    let (or, oi) = get(k);
    let (pr, pi) = (or / gr, oi / gr); // phase = out_k / golden_k
    assert!((pr.hypot(pi) - 1.0).abs() < 1e-9, "phase not unit modulus");
    for (i, &(gr, gi)) in golden.iter().enumerate() {
        let (or, oi) = get(i);
        let (er, ei) = (pr * gr - pi * gi, pr * gi + pi * gr); // phase·golden
        assert!(
            (or - er).hypot(oi - ei) < 1e-9,
            "amp[{i}] = ({or},{oi}) vs phase·golden ({er},{ei})"
        );
    }
}

#[tokio::test]
async fn quantum_execute_pattern_requires_execute_right() {
    let (token, app) = fresh_router_default(rights::READ);
    let body = serde_json::json!({
        "pattern": {
            "vertices": [0, 1], "edges": [[0, 1]],
            "layers": [[{"qubit": 0, "angle": 0.0, "x_corr_from": [], "z_corr_from": []}]],
            "output": [1], "is_photonic": true
        }
    })
    .to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/quantum/execute_pattern", &token, &body))
        .await
        .unwrap();
    assert_ne!(
        resp.status(),
        StatusCode::OK,
        "READ-only token must be rejected"
    );
}

#[tokio::test]
async fn lambda_register_rejects_empty_wasm() {
    // Sanity: the registry rejects empty wasm bytes (it parses as a
    // module before persisting). Confirms that POST /v1/lambdas with
    // an empty wasm_b64 surfaces as a 4xx with a helpful error.
    let (token, app) = fresh_router_default(rights::ADMIN_ROLE);
    let body = serde_json::json!({"name": "empty", "wasm_b64": ""}).to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/lambdas", &token, &body))
        .await
        .unwrap();
    assert!(
        resp.status() == StatusCode::BAD_REQUEST,
        "expected 400, got {}",
        resp.status()
    );
    let v = read_body_json(resp.into_body()).await;
    assert!(
        v["error"].as_str().unwrap_or("").contains("empty"),
        "error must mention empty: {v}"
    );
}

#[tokio::test]
async fn lambda_register_rejects_invalid_base64() {
    let (token, app) = fresh_router_default(rights::ADMIN_ROLE);
    let body = serde_json::json!({"name": "garbage", "wasm_b64": "not!valid!base64"}).to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/lambdas", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = read_body_json(resp.into_body()).await;
    assert!(
        v["error"].as_str().unwrap_or("").contains("wasm_b64"),
        "error must mention wasm_b64: {v}"
    );
}

// --------------------------------------------------------------------
// /v1/quantum/execute — quantum-core wire IR (separate from circuits/functions)
// --------------------------------------------------------------------

#[tokio::test]
async fn quantum_execute_bell_via_statevector_returns_counts() {
    // POST /v1/quantum/execute with an in-flight quantum-core IR
    // (no prior /v1/circuits registration). Statevector backend +
    // shots → counts response keyed by bitstring under
    // `result.counts`.
    let (token, app) = fresh_router_default(rights::EXECUTE);
    let body = serde_json::json!({
        "circuit": {
            "num_qubits": 2,
            "num_classical_bits": 0,
            "is_photonic": false,
            "mid_circuit_mode": "Skip",
            "backend": "Statevector",
            "ops": [
                {"gate": "H",  "qubits": [0],    "params": [], "classical_bit": null, "condition": null},
                {"gate": "CX", "qubits": [0, 1], "params": [], "classical_bit": null, "condition": null}
            ]
        },
        "shots": 256,
        "seed": 42
    })
    .to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/quantum/execute", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = read_body_json(resp.into_body()).await;
    assert_eq!(v["backend"], "statevector");
    let counts = v["result"]["counts"].as_object().unwrap();
    let c00 = counts.get("00").and_then(|c| c.as_u64()).unwrap_or(0);
    let c11 = counts.get("11").and_then(|c| c.as_u64()).unwrap_or(0);
    assert_eq!(c00 + c11, 256, "Bell only produces |00⟩ and |11⟩");
}

#[tokio::test]
async fn quantum_execute_requires_execute_right() {
    let (token, app) = fresh_router_default(rights::READ);
    let body = serde_json::json!({
        "circuit": {
            "num_qubits": 1, "num_classical_bits": 0, "is_photonic": false,
            "mid_circuit_mode": "Skip", "backend": "Statevector",
            "ops": []
        },
        "shots": 1
    })
    .to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/quantum/execute", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn quantum_execute_omitted_shots_returns_statevector() {
    // No `shots` field in the body → ExecConfig.shots = None →
    // exact statevector path. Bell state → 4 amplitudes, |00⟩ and
    // |11⟩ at 1/√2, the others zero.
    let (token, app) = fresh_router_default(rights::EXECUTE);
    let body = serde_json::json!({
        "circuit": {
            "num_qubits": 2, "num_classical_bits": 0, "is_photonic": false,
            "mid_circuit_mode": "Skip", "backend": "Statevector",
            "ops": [
                {"gate": "H",  "qubits": [0],    "params": [], "classical_bit": null, "condition": null},
                {"gate": "CX", "qubits": [0, 1], "params": [], "classical_bit": null, "condition": null}
            ]
        }
    })
    .to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/quantum/execute", &token, &body))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = read_body_json(resp.into_body()).await;
    assert_eq!(v["backend"], "statevector");
    let amps = v["result"]["amplitudes"].as_array().unwrap();
    assert_eq!(amps.len(), 4);
    let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
    let a0_re = amps[0][0].as_f64().unwrap();
    let a3_re = amps[3][0].as_f64().unwrap();
    assert!((a0_re - inv_sqrt2).abs() < 1e-12);
    assert!((a3_re - inv_sqrt2).abs() < 1e-12);
    let a1_re = amps[1][0].as_f64().unwrap();
    let a2_re = amps[2][0].as_f64().unwrap();
    assert!(a1_re.abs() < 1e-12);
    assert!(a2_re.abs() < 1e-12);
}

#[tokio::test]
async fn quantum_execute_garbage_body_returns_4xx() {
    let (token, app) = fresh_router_default(rights::EXECUTE);
    let body = serde_json::json!({"circuit": "not-an-IR"}).to_string();
    let resp = app
        .oneshot(req_post_auth("/v1/quantum/execute", &token, &body))
        .await
        .unwrap();
    assert!(
        resp.status().is_client_error(),
        "garbage IR must reject, got {}",
        resp.status()
    );
}

// --------------------------------------------------------------------
// Per-IP rate limiter — the path the production bug originally lived in
// --------------------------------------------------------------------

/// Build a router for per-IP rate-limit tests, transiently setting
/// the env var so `create_router` picks it up.
fn router_with_ip_rate_limit(limit: &str) -> axum::Router {
    let _g = RATELIMIT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("OMEGA_RATELIMIT_PER_IP_PER_MIN", limit);
    std::env::remove_var("OMEGA_RATELIMIT_PER_SUBJECT_PER_MIN");
    let (state, _token) = fresh_state_with_token(rights::ADMIN_ROLE);
    let app = create_router(state, None);
    std::env::remove_var("OMEGA_RATELIMIT_PER_IP_PER_MIN");
    app
}

/// Build a `/health` request with a `ConnectInfo<SocketAddr>` extension
/// pre-populated so the per-IP rate-limit middleware can extract it.
/// In production this comes from `into_make_service_with_connect_info`;
/// for `oneshot` we have to inject it manually.
fn req_with_ip(path: &str, ip: &str, port: u16) -> Request<Body> {
    let mut req = Request::builder()
        .method("GET")
        .uri(path)
        .body(Body::empty())
        .unwrap();
    let addr: SocketAddr = format!("{ip}:{port}").parse().unwrap();
    req.extensions_mut().insert(ConnectInfo(addr));
    req
}

#[tokio::test]
async fn per_ip_rate_limiter_fires_through_full_router() {
    // Per-IP limit = 2 means: same source IP gets 2 successful
    // requests before the 3rd is rate-limited. /health is the
    // canonical "open" route so this exercises the outermost
    // middleware ring.
    let app = router_with_ip_rate_limit("2");

    for i in 1..=2 {
        let resp = app
            .clone()
            .oneshot(req_with_ip("/health", "10.0.0.1", 5555))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "per-IP request {i}/2 must be 200"
        );
    }

    let resp = app
        .clone()
        .oneshot(req_with_ip("/health", "10.0.0.1", 5556))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "3rd same-IP request must be 429"
    );
    assert!(
        resp.headers().get("retry-after").is_some(),
        "429 must carry Retry-After"
    );
}

// --------------------------------------------------------------------
// /v1/ws — verifies the ws_state=Some(...) branch of create_router
// --------------------------------------------------------------------

/// Build a router with a fresh WsState. Mirrors how `main.rs`
/// constructs the WS surface under `--auth pqc`. Holds the env-var
/// mutex only across the synchronous build so subsequent `.await`
/// calls don't trip clippy's `await_holding_lock`.
fn router_with_ws_state() -> axum::Router {
    use crate::pqc::certificate::OmegaCert;
    use crate::ws::handler::WsState;
    use crate::ws::handshake::ClientCertPolicy;

    let _g = RATELIMIT_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("OMEGA_RATELIMIT_PER_SUBJECT_PER_MIN");
    std::env::remove_var("OMEGA_RATELIMIT_PER_IP_PER_MIN");
    let (state, _token) = fresh_state_with_token(rights::ADMIN_ROLE);
    let seed: [u8; 32] = rand::random();
    let server_cert = OmegaCert::self_sign("omega-server", 3600, &seed, None).expect("self-sign");
    let ws_state = Arc::new(RwLock::new(WsState {
        server_cert,
        trust_store: Arc::new(TrustStore::empty()),
        crl: None::<Arc<OmegaCrl>>,
        client_cert_policy: ClientCertPolicy::Off,
    }));
    create_router(state, Some(ws_state))
}

#[tokio::test]
async fn ws_route_returns_4xx_when_not_a_websocket_upgrade() {
    // When create_router is called with ws_state = Some(...) the
    // /v1/ws route is registered. A plain GET (no Upgrade header)
    // should be rejected by axum's WebSocketUpgrade extractor as a
    // 4xx (typically 426 Upgrade Required or 400) rather than 404
    // — the route exists but the request isn't a valid WS upgrade.
    // Guards against the ws_state branch being silently dropped
    // from the router.
    let app = router_with_ws_state();
    let resp = app.oneshot(req_get("/v1/ws")).await.unwrap();
    assert_ne!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "/v1/ws route must be registered when ws_state is provided"
    );
    assert!(
        resp.status().is_client_error(),
        "plain GET on /v1/ws must be a 4xx (no Upgrade), got {}",
        resp.status()
    );
}

#[tokio::test]
async fn ws_route_unreachable_when_ws_state_is_none() {
    // The other branch: create_router(state, None) means /v1/ws is
    // not registered. A request to it can't reach the WS handler.
    // Empirically axum routes the unknown path through the auth
    // middleware first (because the protected sub-router's catch-
    // all fallback inherits its layers), so the response is 401
    // rather than 404 — either way the WS handler is unreachable
    // without a token, which is what `--auth bearer-only` wants.
    let (_token, app) = fresh_router_default(rights::ADMIN_ROLE);
    let resp = app.oneshot(req_get("/v1/ws")).await.unwrap();
    let status = resp.status();
    assert!(
        status == StatusCode::NOT_FOUND || status == StatusCode::UNAUTHORIZED,
        "/v1/ws must be 404 or 401 when ws_state is None, got {status}"
    );
}

#[tokio::test]
async fn per_ip_rate_limiter_independent_buckets_per_source() {
    // Different source IPs get independent buckets — limit=1 means
    // each unique IP gets exactly 1 success.
    let app = router_with_ip_rate_limit("1");

    // First IP: 1 success then 1 throttle.
    let r1 = app
        .clone()
        .oneshot(req_with_ip("/health", "10.0.0.10", 1000))
        .await
        .unwrap();
    assert_eq!(r1.status(), StatusCode::OK);
    let r2 = app
        .clone()
        .oneshot(req_with_ip("/health", "10.0.0.10", 1001))
        .await
        .unwrap();
    assert_eq!(r2.status(), StatusCode::TOO_MANY_REQUESTS);

    // Different IP: gets its own quota.
    let r3 = app
        .clone()
        .oneshot(req_with_ip("/health", "10.0.0.11", 1000))
        .await
        .unwrap();
    assert_eq!(r3.status(), StatusCode::OK, "different IP gets own bucket");
}
