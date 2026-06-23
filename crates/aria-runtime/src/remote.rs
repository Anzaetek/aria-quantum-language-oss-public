// SPDX-License-Identifier: Apache-2.0
//! Remote execution backend: delegate an Aria circuit to a running
//! `omega-server` over HTTP (`POST /v1/quantum/execute`).
//!
//! The circuit is bound to concrete parameters, lowered to the omega JSON wire
//! IR ([`aria_core::backends::omega::to_omega_ir`]), wrapped as
//! `{ circuit, shots, seed }`, and POSTed with an optional bearer token. The
//! `{ result: { counts } }` response is decoded back into an
//! [`ExecResult::Counts`].

use std::collections::HashMap;

use aria_core::ast::Circuit;
use aria_core::backends::omega::to_omega_ir;
use omega_core::executor::ExecResult;

/// Connection to a running omega-server.
pub struct Remote {
    pub url: String,
    pub token: Option<String>,
}

/// Execute `circuit` on a remote omega-server and return measurement counts.
pub fn run_counts_remote(
    circuit: &Circuit,
    bindings: &HashMap<String, f64>,
    shots: u32,
    seed: Option<u64>,
    remote: &Remote,
) -> Result<ExecResult, String> {
    // Bind free symbols (missing → 0.0) so the wire IR is fully concrete.
    let mut full = bindings.clone();
    for s in circuit.free_symbols() {
        full.entry(s).or_insert(0.0);
    }
    let bound = circuit.bind_params(&full)?;
    let ir = to_omega_ir(&bound);

    let body = serde_json::json!({ "circuit": ir, "shots": shots, "seed": seed });
    let endpoint = format!("{}/v1/quantum/execute", remote.url.trim_end_matches('/'));

    let mut req = ureq::post(&endpoint).set("Content-Type", "application/json");
    if let Some(tok) = &remote.token {
        req = req.set("Authorization", &format!("Bearer {tok}"));
    }
    let resp = req
        .send_json(body)
        .map_err(|e| format!("omega-server request to {endpoint} failed: {e}"))?;
    let v: serde_json::Value = resp
        .into_json()
        .map_err(|e| format!("bad response JSON from omega-server: {e}"))?;

    let counts = v
        .get("result")
        .and_then(|r| r.get("counts"))
        .and_then(|c| c.as_object())
        .ok_or_else(|| format!("unexpected omega-server response: {v}"))?;

    let mut map: HashMap<u64, u32> = HashMap::new();
    for (k, val) in counts {
        let state = u64::from_str_radix(k, 2).map_err(|_| format!("bad bitstring key '{k}'"))?;
        map.insert(state, val.as_u64().unwrap_or(0) as u32);
    }
    Ok(ExecResult::Counts(map))
}
