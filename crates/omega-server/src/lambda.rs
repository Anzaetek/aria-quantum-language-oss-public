//! Lambda functions — server-side glue for WASM-driven optimisation loops.
//!
//! A **lambda** binds a WASM module to a default (QASM, observable, input)
//! triple. Callers invoke it with optional overrides, the server stages
//! everything into a fresh `HostState` and runs the WASM under wasmtime,
//! then returns the optimisation result + progress trace.
//!
//! This is the third tier of the server's function model:
//!
//! | Resource     | Purpose                                                     |
//! |--------------|-------------------------------------------------------------|
//! | `circuits`   | One QASM source                                             |
//! | `functions`  | A name binding (circuit_id, default_shots) — one-shot exec  |
//! | `lambdas`    | A WASM module that drives many circuit executions           |
//!
//! The lambda's WASM is what makes the iteration count, optimizer choice,
//! and circuit-construction logic user-extensible — no server redeploy
//! required.

use std::sync::Arc;
use tokio::sync::RwLock;

use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use base64::Engine;
use serde::{Deserialize, Serialize};

use omega_core::executor::Observable;
use omega_parser::lower_to_ir;
use omega_wasm_runtime::host::HostState;
use omega_wasm_runtime::WasmRunner;

use crate::auth::middleware::check_rights;
use crate::auth::rights;
use crate::auth::token::TokenClaims;
use crate::AppState;

type SharedState = Arc<RwLock<AppState>>;

// ---- Wire types ----

#[derive(Debug, Deserialize)]
pub struct CreateLambdaReq {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// Base64-encoded WASM module bytes. Required.
    pub wasm_b64: String,
    #[serde(default)]
    pub default_qasm: Option<String>,
    #[serde(default)]
    pub default_observable: Option<String>,
    /// Default input payload (raw JSON string). Sent verbatim to the
    /// guest's `omega_input_*`.
    #[serde(default)]
    pub default_input: Option<String>,
    #[serde(default = "default_fuel")]
    pub fuel: i64,
}

fn default_fuel() -> i64 {
    100_000_000_000
}

#[derive(Debug, Deserialize, Default)]
pub struct InvokeLambdaReq {
    /// Override QASM. If supplied (and a circuit isn't built by the guest
    /// itself via `omega_register_qasm` / `omega_qaoa_from_qubo`), this is
    /// pre-registered as circuit_id 1.
    #[serde(default)]
    pub qasm: Option<String>,
    /// Override observable Pauli-sum string, pre-registered as observable_id 1.
    #[serde(default)]
    pub observable: Option<String>,
    /// Override input payload (raw JSON string). Sent verbatim to the guest.
    #[serde(default)]
    pub input: Option<String>,
    /// Override fuel cap.
    #[serde(default)]
    pub fuel: Option<i64>,
    /// If true, include the per-iteration cost stream in the response. Off
    /// by default since traces can be large.
    #[serde(default)]
    pub include_progress: bool,
}

#[derive(Debug, Serialize)]
pub struct InvokeLambdaResp {
    pub lambda_id: String,
    pub status: String,
    pub optimal_value: f64,
    pub optimal_params: Vec<f64>,
    pub iterations: u32,
    pub progress_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<Vec<(u32, f64)>>,
}

// ---- Handlers ----

pub async fn create_lambda(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
    Json(req): Json<CreateLambdaReq>,
) -> impl IntoResponse {
    if let Err(resp) = check_rights(&claims, rights::WRITE) {
        return resp;
    }
    let wasm_bytes = match base64::engine::general_purpose::STANDARD.decode(&req.wasm_b64) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("invalid wasm_b64: {e}")
                })),
            )
                .into_response()
        }
    };
    let s = state.read().await;
    match s.registry.register_lambda(
        &req.name,
        req.description.as_deref(),
        &wasm_bytes,
        req.default_qasm.as_deref(),
        req.default_observable.as_deref(),
        req.default_input.as_deref(),
        req.fuel,
    ) {
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

pub async fn list_lambdas(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
) -> impl IntoResponse {
    if let Err(resp) = check_rights(&claims, rights::READ) {
        return resp;
    }
    let s = state.read().await;
    match s.registry.list_lambdas() {
        Ok(v) => (StatusCode::OK, Json(serde_json::json!({ "lambdas": v }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

pub async fn get_lambda(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err(resp) = check_rights(&claims, rights::READ) {
        return resp;
    }
    let s = state.read().await;
    match s.registry.get_lambda(&id) {
        Ok(entry) => (StatusCode::OK, Json(serde_json::to_value(entry).unwrap())).into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

pub async fn delete_lambda(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err(resp) = check_rights(&claims, rights::WRITE) {
        return resp;
    }
    let s = state.read().await;
    match s.registry.delete_lambda(&id) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => StatusCode::NOT_FOUND.into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

pub async fn invoke_lambda(
    Extension(claims): Extension<TokenClaims>,
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(req): Json<InvokeLambdaReq>,
) -> impl IntoResponse {
    if let Err(resp) = check_rights(&claims, rights::EXECUTE) {
        return resp;
    }
    let s = state.read().await;

    let lambda = match s.registry.get_lambda(&id) {
        Ok(l) => l,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };
    let wasm_bytes = match s.registry.get_lambda_wasm(&id) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };

    // Resolve overrides → effective qasm / observable / input.
    let qasm_src = req.qasm.as_deref().or(lambda.default_qasm.as_deref());
    let obs_src = req
        .observable
        .as_deref()
        .or(lambda.default_observable.as_deref());
    let input_src = req.input.as_deref().or(lambda.default_input.as_deref());
    let fuel = req.fuel.unwrap_or(lambda.fuel) as u64;

    // Stage HostState. Pre-register (circuit_id=1, observable_id=1) IFF the
    // caller supplied a QASM / observable; otherwise leave the host empty
    // and let the guest call `omega_register_qasm` / `omega_qaoa_from_qubo`.
    let mut host = HostState::new();
    if let Some(src) = qasm_src {
        let circuit = match lower_to_ir(src) {
            Ok(c) => c,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": format!("qasm parse: {e}") })),
                )
                    .into_response();
            }
        };
        host.register_circuit(circuit);
    }
    if let Some(spec) = obs_src {
        let observable = match Observable::parse(spec) {
            Ok(o) => o,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": format!("observable parse: {e}") })),
                )
                    .into_response();
            }
        };
        host.register_observable(observable);
    }
    if let Some(s) = input_src {
        host.set_input(s.as_bytes().to_vec());
    }

    // Run the WASM. wasmtime is sync; do the heavy work on a blocking thread.
    let result = tokio::task::spawn_blocking(move || -> std::result::Result<_, String> {
        let runner = WasmRunner::new(host).map_err(|e| e.to_string())?;
        let res = runner.run(&wasm_bytes, fuel).map_err(|e| e.to_string())?;
        let progress = {
            let st = runner.host_state().lock().unwrap();
            st.progress.clone()
        };
        Ok((res, progress))
    })
    .await;

    let (wasm_result, progress) = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("wasm: {e}") })),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("join: {e}") })),
            )
                .into_response();
        }
    };

    let Some(res) = wasm_result else {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": "WASM completed but never called omega_report_result",
                "progress_count": progress.len(),
            })),
        )
            .into_response();
    };

    let resp = InvokeLambdaResp {
        lambda_id: lambda.id,
        status: "completed".to_string(),
        optimal_value: res.optimal_value,
        optimal_params: res.optimal_params,
        iterations: res.iterations,
        progress_count: progress.len(),
        progress: if req.include_progress {
            Some(progress)
        } else {
            None
        },
    };
    (StatusCode::OK, Json(serde_json::to_value(resp).unwrap())).into_response()
}
