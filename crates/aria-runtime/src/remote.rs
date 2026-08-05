// SPDX-License-Identifier: Apache-2.0
//! Remote execution backend: delegate an Aria circuit to a running
//! `omega-server` over HTTP.
//!
//! The circuit is bound to concrete parameters and lowered to the omega JSON
//! wire IR ([`aria_core::backends::omega::to_omega_ir`]), then either:
//! - [`run_counts_remote`] POSTs `{ circuit, shots, seed }` to
//!   `/v1/quantum/execute` and decodes `{ result: { counts } }` into an
//!   [`ExecResult::Counts`]; or
//! - [`expectation_remote`] POSTs `{ circuit, observable }` to
//!   `/v1/quantum/expectation` and returns the scalar `⟨O⟩` from `{ values }`,
//!   so a client gets a number back instead of the full statevector.

use std::collections::HashMap;

use aria_core::ast::Circuit;
use aria_core::backends::omega::try_to_omega_ir;
use omega_core::executor::ExecResult;

use crate::lower::lower;
use crate::run::{measure_pairs, project_counts_onto_creg};

/// Connection to a running omega-server.
pub struct Remote {
    pub url: String,
    pub token: Option<String>,
}

/// POST `body` as JSON to `remote.url + path` (with the optional bearer token)
/// and return the decoded JSON response. Shared by the counts and expectation
/// remote paths.
fn post_json(
    remote: &Remote,
    path: &str,
    body: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let endpoint = format!("{}{path}", remote.url.trim_end_matches('/'));
    let mut req = ureq::post(&endpoint).set("Content-Type", "application/json");
    if let Some(tok) = &remote.token {
        req = req.set("Authorization", &format!("Bearer {tok}"));
    }
    req.send_json(body)
        .map_err(|e| format!("omega-server request to {endpoint} failed: {e}"))?
        .into_json()
        .map_err(|e| format!("bad response JSON from omega-server: {e}"))
}

/// Extract the `values` array from a `/v1/quantum/expectation` response.
fn parse_expectation_values(v: &serde_json::Value) -> Result<Vec<f64>, String> {
    v.get("values")
        .and_then(|a| a.as_array())
        .ok_or_else(|| format!("unexpected expectation response: {v}"))?
        .iter()
        .map(|x| {
            x.as_f64()
                .ok_or_else(|| "expectation value not a number".to_string())
        })
        .collect()
}

/// Compute `⟨O⟩` of a Pauli observable on a remote omega-server
/// (`POST /v1/quantum/expectation`). The circuit is bound to concrete
/// parameters and lowered to the wire IR; the scalar comes back directly, so
/// the client never pulls the full statevector to reduce it locally.
pub fn expectation_remote(
    circuit: &Circuit,
    observable: &str,
    bindings: &HashMap<String, f64>,
    remote: &Remote,
) -> Result<f64, String> {
    let mut full = bindings.clone();
    for s in circuit.free_symbols() {
        full.entry(s).or_insert(0.0);
    }
    let bound = circuit.bind_params(&full)?;
    let ir = try_to_omega_ir(&bound)?;
    let body = serde_json::json!({ "circuit": ir, "observable": observable });
    let v = post_json(remote, "/v1/quantum/expectation", body)?;
    parse_expectation_values(&v)?
        .into_iter()
        .next()
        .ok_or_else(|| "omega-server returned no expectation value".to_string())
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
    let ir = try_to_omega_ir(&bound)?;

    let body = serde_json::json!({ "circuit": ir, "shots": shots, "seed": seed });
    let v = post_json(remote, "/v1/quantum/execute", body)?;

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
    let res = ExecResult::Counts(map);
    // Same creg semantics as the local backends, decided by the SAME
    // lowering as run_counts (not by the wire IR, whose mid-circuit
    // detection uses a different meta-gate set): project final-measurement
    // programs onto the classical register; mid-circuit programs keep
    // full-register keying. Circuits lower() can't handle (e.g. photonic
    // gates) keep the server's raw keying.
    match lower(&bound) {
        Ok(low) if !low.needs_collapse => project_counts_onto_creg(res, &measure_pairs(&low)),
        _ => Ok(res),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_core::ast::parse_aria;

    #[test]
    fn remote_counts_are_projected_onto_creg() {
        // 2 qubits, only q[0] measured into c[0]: a server response keyed
        // over the full register must collapse to creg-width keys, using
        // the same lowering-derived mapping as the local backends.
        let src = "circuit Partial() {\n  qreg q[2]\n  creg c[1]\n  apply H on q[1]\n  measure q[0] -> c[0]\n}\n";
        let circuit = parse_aria(src)
            .unwrap()
            .instantiate("Partial", &[])
            .unwrap();
        let low = lower(&circuit).unwrap();
        assert!(!low.needs_collapse);
        assert_eq!(measure_pairs(&low), vec![(0, 0)]);

        let mut map: HashMap<u64, u32> = HashMap::new();
        map.insert(0b00, 40); // q0=0
        map.insert(0b10, 41); // q0=0
        map.insert(0b01, 10); // q0=1
        map.insert(0b11, 9); //  q0=1
        let res = project_counts_onto_creg(ExecResult::Counts(map), &measure_pairs(&low)).unwrap();
        if let ExecResult::Counts(m) = res {
            assert_eq!(m.get(&0), Some(&81));
            assert_eq!(m.get(&1), Some(&19));
            assert_eq!(m.len(), 2);
        } else {
            panic!("expected counts");
        }
    }

    #[test]
    fn parse_expectation_values_reads_the_array() {
        let v = serde_json::json!({ "backend": "statevector", "values": [1.0, -0.5, 0.0] });
        assert_eq!(parse_expectation_values(&v).unwrap(), vec![1.0, -0.5, 0.0]);
    }

    #[test]
    fn parse_expectation_values_rejects_malformed() {
        assert!(parse_expectation_values(&serde_json::json!({ "error": "x" })).is_err());
        assert!(parse_expectation_values(&serde_json::json!({ "values": ["nan"] })).is_err());
    }
}
