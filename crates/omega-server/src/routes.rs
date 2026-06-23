//! REST API routes.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use axum::body::Body;
use axum::extract::{Extension, Path, State};
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;

use crate::auth::rate_limit::{
    build_ip_limiter, build_subject_limiter, rate_limit_ip, rate_limit_subject,
    SlidingWindowLimiter,
};
use crate::auth::token::TokenClaims;
use crate::auth::{middleware, rights, store, token};
use crate::lambda;
use crate::pki;
use crate::quantum_bridge;
use crate::ws;
use crate::AppState;

type SharedState = Arc<RwLock<AppState>>;
type WsSharedState = Arc<RwLock<ws::handler::WsState>>;

/// Maximum size of an inbound JSON body, in bytes. Anything larger is
/// rejected by the `RequestBodyLimitLayer` before the handler is reached.
/// Tunable upper bound for circuit / lambda uploads. WASM blobs are
/// base64-encoded so 16 MiB limits effective binary size to ~12 MiB.
const MAX_REQUEST_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Per-request handler timeout. The quantum-bridge / lambda paths can
/// take longer for large QUBOs; bump this if you start seeing 408s on
/// legitimate workloads.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Build the Axum router. `ws_state = Some(...)` exposes the PQC WebSocket
/// at `/v1/ws`; `ws_state = None` skips it entirely (`--auth bearer-only`
/// startup mode).
///
/// Every route is wrapped in three defensive layers:
/// - `RequestBodyLimitLayer`: rejects bodies > [`MAX_REQUEST_BODY_BYTES`].
/// - `TimeoutLayer`: bounds handler runtime at [`REQUEST_TIMEOUT`].
/// - `CorsLayer`: deny-all by default. Operators expose specific origins
///   via the `OMEGA_CORS_ALLOW_ORIGINS` env var (comma-separated origins
///   like `https://ops.example.com`). The header allowlist is conservative.
pub fn create_router(state: SharedState, ws_state: Option<WsSharedState>) -> Router {
    // Public routes (no auth required)
    let public = Router::new()
        .route("/health", get(health))
        .with_state(state.clone());

    // WebSocket route (PQC handshake handles its own auth) — opt-in.
    let ws_routes = ws_state.map(|s| {
        Router::new()
            .route("/v1/ws", get(ws::handler::ws_upgrade))
            .with_state(s)
    });

    // Phase 14a rate limiters: per-IP runs *before* auth (so anonymous
    // brute-force attempts on /v1/auth/token get throttled by source);
    // per-subject runs *after* auth (so a single token can't burn the
    // server). Both buckets must allow the request. Limits are read
    // from env at boot; setting either to 0 disables that surface.
    let ip_limiter: Arc<SlidingWindowLimiter<std::net::IpAddr>> = Arc::new(build_ip_limiter());
    let subject_limiter: Arc<SlidingWindowLimiter<String>> = Arc::new(build_subject_limiter());

    // Protected routes (auth required)
    let protected = Router::new()
        // Auth management
        .route("/v1/auth/token", post(issue_token))
        .route("/v1/auth/token/:jti", delete(revoke_token))
        // Circuits
        .route("/v1/circuits", post(create_circuit))
        .route("/v1/circuits", get(list_circuits))
        .route("/v1/circuits/:id", get(get_circuit))
        .route("/v1/circuits/:id", delete(delete_circuit))
        // Functions
        .route("/v1/functions", post(create_function))
        .route("/v1/functions", get(list_functions))
        .route("/v1/functions/:id", get(get_function))
        .route("/v1/functions/:id/invoke", post(invoke_function))
        // Lambdas — WASM-driven optimisation functions (see lambda.rs)
        .route("/v1/lambdas", post(lambda::create_lambda))
        .route("/v1/lambdas", get(lambda::list_lambdas))
        .route("/v1/lambdas/:id", get(lambda::get_lambda))
        .route("/v1/lambdas/:id", delete(lambda::delete_lambda))
        .route("/v1/lambdas/:id/invoke", post(lambda::invoke_lambda))
        // Invocations
        .route("/v1/invocations/:id", get(get_invocation))
        // Quantum-core bridge: execute an OmegaCircuitIR wire document and
        // dispatch by its `backend` field (Auto/Statevector/Mps/Stabilizer/Photonic).
        .route(
            "/v1/quantum/execute",
            post(quantum_bridge::execute_quantum_route),
        )
        // MBQC one-way measurement patterns (C1.3): execute an OmegaPatternIR
        // on the photonic graph-state backend.
        .route(
            "/v1/quantum/execute_pattern",
            post(quantum_bridge::execute_pattern_route),
        )
        // PKI cert issuance (admin-only). Returns 503 when
        // OMEGA_PKI_ISSUER_SUBJECT / OMEGA_PKI_ISSUER_SEED_FILE
        // aren't set, so a server without a CA configured stays
        // a no-op for the endpoint rather than returning surprising
        // errors.
        .route("/v1/pki/issue", post(pki::issue_cert))
        // Info
        .route("/v1/backends", get(list_backends))
        // Per-subject limiter — attached *after* require_auth so the
        // TokenClaims extension is in scope. The limiter Arc is passed
        // via `from_fn_with_state` to dodge the Extension-layer
        // ordering footgun documented on the per-IP limiter below.
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&subject_limiter),
            rate_limit_subject,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middleware::require_auth,
        ))
        .with_state(state);

    let mut app = public.merge(protected);
    if let Some(ws) = ws_routes {
        app = app.merge(ws);
    }

    // Outer security layer stack — applies to *every* route, including
    // /health and /v1/ws, so even unauth'd traffic gets per-IP capped.
    //
    // Use `from_fn_with_state` to plumb the limiter Arc into the
    // middleware function via axum's typed state machinery rather
    // than `Extension(...)`. The previous wiring used a ServiceBuilder
    // stack with `rate_limit_ip` (Extension extractor) layered alongside
    // `Extension(Arc::clone(&ip_limiter))`. Whichever ordering of those
    // two `.layer()` calls we tried, axum::serve dispatched the
    // middleware before the Extension was inserted into request
    // extensions, so every request returned 500 with "Missing request
    // extension". from_fn_with_state binds the value through a typed
    // channel that can't get the ordering wrong.
    let security = ServiceBuilder::new()
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&ip_limiter),
            rate_limit_ip,
        ))
        .layer(axum::middleware::from_fn(request_timeout))
        .layer(RequestBodyLimitLayer::new(MAX_REQUEST_BODY_BYTES))
        .layer(build_cors_layer());

    app.layer(security)
}

/// Per-request timeout middleware. tokio::time::timeout cancels the
/// downstream future on expiry; we then return a 408 response.
async fn request_timeout(req: Request<Body>, next: Next) -> Response {
    match tokio::time::timeout(REQUEST_TIMEOUT, next.run(req)).await {
        Ok(resp) => resp,
        Err(_) => (
            StatusCode::REQUEST_TIMEOUT,
            Json(serde_json::json!({"error": "request timed out"})),
        )
            .into_response(),
    }
}

fn build_cors_layer() -> CorsLayer {
    use axum::http::{header, HeaderName, HeaderValue, Method};

    // Deny-all by default. Operators opt-in by listing origins in the
    // `OMEGA_CORS_ALLOW_ORIGINS` env var (comma-separated). An empty
    // value means no cross-origin requests are accepted.
    let raw = std::env::var("OMEGA_CORS_ALLOW_ORIGINS").unwrap_or_default();
    let origins: Vec<HeaderValue> = raw
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .filter_map(|s| HeaderValue::from_str(s).ok())
        .collect();

    let allowed_methods = [
        Method::GET,
        Method::POST,
        Method::DELETE,
        Method::PUT,
        Method::OPTIONS,
    ];
    let allowed_headers = [header::AUTHORIZATION, header::CONTENT_TYPE, header::ACCEPT];
    // Custom WebSocket sub-protocol header for the PQC handshake.
    let extra_headers: Vec<HeaderName> = vec![HeaderName::from_static("sec-websocket-protocol")];

    if origins.is_empty() {
        // Strict deny-all: don't echo any Access-Control-Allow-Origin.
        // Browsers will refuse cross-origin XHR / fetch by default.
        CorsLayer::new()
    } else {
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods(allowed_methods)
            .allow_headers(
                allowed_headers
                    .into_iter()
                    .chain(extra_headers)
                    .collect::<Vec<_>>(),
            )
            .max_age(Duration::from_secs(600))
    }
}

// ---- Request/Response types ----

#[derive(Deserialize)]
struct CreateCircuitReq {
    source: String,
}

#[derive(Deserialize)]
struct CreateFunctionReq {
    circuit_id: String,
    name: String,
    #[serde(default = "default_shots")]
    default_shots: u32,
}

fn default_shots() -> u32 {
    1024
}

#[derive(Deserialize)]
struct InvokeFunctionReq {
    #[serde(default)]
    params: Vec<f64>,
    #[serde(default = "default_shots")]
    shots: u32,
    seed: Option<u64>,
}

#[derive(Deserialize)]
struct IssueTokenReq {
    sub: String,
    rights: u32,
    #[serde(default = "default_ttl")]
    ttl_seconds: i64,
}

fn default_ttl() -> i64 {
    3600
}

#[derive(Serialize)]
#[allow(dead_code)]
struct ErrorResponse {
    error: String,
}

// ---- Auth Handlers ----

async fn issue_token(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
    Json(req): Json<IssueTokenReq>,
) -> impl IntoResponse {
    if let Err(resp) = middleware::check_rights(&claims, rights::ADMIN) {
        return resp;
    }

    let state = state.read().await;
    match token::issue_token(
        &req.sub,
        req.rights,
        req.ttl_seconds,
        &state.active_kid,
        &state.active_sk,
    ) {
        Ok((token_str, new_claims)) => {
            // Store for revocation tracking
            let _ = store::store_token(&state.registry.conn(), &new_claims);

            (
                StatusCode::CREATED,
                Json(serde_json::json!({
                    "token": token_str,
                    "jti": new_claims.jti,
                    "sub": new_claims.sub,
                    "rights": new_claims.rights,
                    "rights_names": rights::rights_to_names(new_claims.rights),
                    "expires_at": new_claims.exp,
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
    }
}

async fn revoke_token(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
    Path(jti): Path<String>,
) -> impl IntoResponse {
    if let Err(resp) = middleware::check_rights(&claims, rights::ADMIN) {
        return resp;
    }

    let state = state.read().await;
    let conn = state.registry.conn();
    let result = store::revoke_token(&conn, &jti);
    drop(conn);
    drop(state);
    match result {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "token not found"})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
    }
}

// ---- Existing Handlers (now with rights checks) ----

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn list_backends(Extension(claims): Extension<TokenClaims>) -> impl IntoResponse {
    if let Err(resp) = middleware::check_rights(&claims, rights::READ) {
        return resp;
    }
    Json(serde_json::json!({
        "backends": [
            { "name": "statevector", "type": "gate_based" },
            { "name": "mps",         "type": "gate_based", "parametric": true },
            { "name": "stabilizer",  "type": "gate_based", "requires": "clifford" },
            { "name": "photonics",   "type": "photonic" },
        ]
    }))
    .into_response()
}

async fn create_circuit(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
    Json(req): Json<CreateCircuitReq>,
) -> impl IntoResponse {
    if let Err(resp) = middleware::check_rights(&claims, rights::WRITE) {
        return resp;
    }
    let state = state.read().await;
    match state.registry.register_circuit(&req.source) {
        Ok(entry) => (
            StatusCode::CREATED,
            Json(serde_json::to_value(entry).unwrap()),
        )
            .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn list_circuits(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
) -> impl IntoResponse {
    if let Err(resp) = middleware::check_rights(&claims, rights::READ) {
        return resp;
    }
    let state = state.read().await;
    match state.registry.list_circuits() {
        Ok(circuits) => Json(serde_json::json!({ "circuits": circuits })).into_response(),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })).into_response(),
    }
}

async fn get_circuit(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err(resp) = middleware::check_rights(&claims, rights::READ) {
        return resp;
    }
    let state = state.read().await;
    match state.registry.get_circuit(&id) {
        Ok(entry) => (StatusCode::OK, Json(serde_json::to_value(entry).unwrap())).into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn delete_circuit(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err(resp) = middleware::check_rights(&claims, rights::WRITE) {
        return resp;
    }
    let state = state.read().await;
    match state.registry.delete_circuit(&id) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => StatusCode::NOT_FOUND.into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn create_function(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
    Json(req): Json<CreateFunctionReq>,
) -> impl IntoResponse {
    if let Err(resp) = middleware::check_rights(&claims, rights::WRITE) {
        return resp;
    }
    let state = state.read().await;
    match state
        .registry
        .register_function(&req.circuit_id, &req.name, req.default_shots)
    {
        Ok(entry) => (
            StatusCode::CREATED,
            Json(serde_json::to_value(entry).unwrap()),
        )
            .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn list_functions(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
) -> impl IntoResponse {
    if let Err(resp) = middleware::check_rights(&claims, rights::READ) {
        return resp;
    }
    let state = state.read().await;
    match state.registry.list_functions() {
        Ok(functions) => Json(serde_json::json!({ "functions": functions })).into_response(),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })).into_response(),
    }
}

async fn get_function(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err(resp) = middleware::check_rights(&claims, rights::READ) {
        return resp;
    }
    let state = state.read().await;
    match state.registry.get_function(&id) {
        Ok(entry) => (StatusCode::OK, Json(serde_json::to_value(entry).unwrap())).into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn invoke_function(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(req): Json<InvokeFunctionReq>,
) -> impl IntoResponse {
    if let Err(resp) = middleware::check_rights(&claims, rights::EXECUTE) {
        return resp;
    }
    let state = state.read().await;

    let function = match state.registry.get_function(&id) {
        Ok(f) => f,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };

    let params_json = serde_json::to_string(&req.params).unwrap();
    let invocation = match state.registry.create_invocation(&id, &params_json) {
        Ok(inv) => inv,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response()
        }
    };

    let shots = if req.shots == 0 {
        function.default_shots
    } else {
        req.shots
    };

    match state
        .registry
        .execute_circuit(&function.circuit_id, &req.params, shots, req.seed)
    {
        Ok(result) => {
            let result_str = serde_json::to_string(&result).unwrap();
            let _ = state
                .registry
                .complete_invocation(&invocation.id, &result_str);

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "invocation_id": invocation.id,
                    "status": "completed",
                    "result": result,
                })),
            )
                .into_response()
        }
        Err(e) => {
            let _ = state
                .registry
                .fail_invocation(&invocation.id, &e.to_string());

            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "invocation_id": invocation.id,
                    "status": "failed",
                    "error": e.to_string(),
                })),
            )
                .into_response()
        }
    }
}

async fn get_invocation(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err(resp) = middleware::check_rights(&claims, rights::READ) {
        return resp;
    }
    let state = state.read().await;
    match state.registry.get_invocation(&id) {
        Ok(entry) => {
            let mut val = serde_json::to_value(&entry).unwrap();
            if let Some(ref result_str) = entry.result_json {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(result_str) {
                    val["result"] = parsed;
                }
            }
            (StatusCode::OK, Json(val)).into_response()
        }
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}
