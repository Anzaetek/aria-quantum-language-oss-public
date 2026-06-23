//! Axum WebSocket handler with PQC-encrypted framing.
//!
//! Route: `GET /v1/ws` with `Sec-WebSocket-Protocol: omega-pqc-v1`

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use tokio::sync::RwLock;

use crate::pqc::certificate::OmegaCert;
use crate::pqc::crl::OmegaCrl;
use crate::pqc::trust_store::TrustStore;
use crate::ws::handshake::{ClientCertPolicy, ServerHandshake};

/// Shared state needed for WebSocket connections.
pub struct WsState {
    pub server_cert: OmegaCert,
    /// Trust roots used to anchor a presented client cert chain.
    /// Empty when `OMEGA_PKI_TRUST_STORE` is unset; chain-bearing
    /// clients will then be rejected by the policy check.
    pub trust_store: Arc<TrustStore>,
    /// Optional CRL — when present it is consulted by the chain
    /// validator. Loaded from `OMEGA_PKI_CRL_FILE`.
    pub crl: Option<Arc<OmegaCrl>>,
    /// What the server requires of client cert presentations. See
    /// [`ClientCertPolicy`].
    pub client_cert_policy: ClientCertPolicy,
}

/// Axum handler for WebSocket upgrade.
pub async fn ws_upgrade(
    ws: WebSocketUpgrade,
    State(ws_state): State<Arc<RwLock<WsState>>>,
) -> impl IntoResponse {
    ws.protocols(["omega-pqc-v1"])
        .on_upgrade(move |socket| handle_ws(socket, ws_state))
}

/// Handle a WebSocket connection with PQC handshake.
async fn handle_ws(mut socket: WebSocket, ws_state: Arc<RwLock<WsState>>) {
    // Step 1: Build and send ServerHello
    let (server_cert, trust_store, crl, policy) = {
        let state = ws_state.read().await;
        (
            state.server_cert.clone(),
            state.trust_store.clone(),
            state.crl.clone(),
            state.client_cert_policy,
        )
    };

    let server_hs = ServerHandshake::new(server_cert);
    let server_hello = server_hs.server_hello_bytes().to_vec();

    if socket.send(Message::Binary(server_hello)).await.is_err() {
        return;
    }

    // Step 2: Receive ClientHello
    let client_hello = match socket.recv().await {
        Some(Ok(Message::Binary(data))) => data.to_vec(),
        _ => return,
    };

    // Step 3: Derive session keys + apply client-cert policy.
    let outcome =
        match server_hs.finish(&client_hello, policy, trust_store.as_ref(), crl.as_deref()) {
            Ok(o) => o,
            Err(_e) => {
                let _ = socket
                    .send(Message::Close(Some(axum::extract::ws::CloseFrame {
                        code: 4001,
                        reason: "handshake failed".into(),
                    })))
                    .await;
                return;
            }
        };
    let mut session = outcome.session;
    // `outcome.client_subject` is the validated peer identity when a
    // client cert was presented and accepted under the configured
    // policy. Currently unused beyond closing the handshake; the
    // request-dispatch layer (next commit) will read it for per-peer
    // routing / authorisation.
    let _client_subject = outcome.client_subject;

    // Step 4: Encrypted message loop
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Binary(data) => {
                let plaintext = match session.decrypt(&data) {
                    Ok(pt) => pt,
                    Err(_) => break,
                };

                // Echo back encrypted (placeholder — real handler would dispatch to circuit execution)
                let response = match session.encrypt(&plaintext) {
                    Ok(ct) => ct,
                    Err(_) => break,
                };

                if socket.send(Message::Binary(response)).await.is_err() {
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}
