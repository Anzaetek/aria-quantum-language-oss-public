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
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use aria_core::ast::Circuit;
use aria_core::backends::omega::try_to_omega_ir;
use omega_core::executor::ExecResult;
use omega_core::outcome::Outcome;

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

    // Serialise separately from sending so encoding cost is attributable. It is
    // not always negligible: a batch re-sends every row's full gate list, which
    // is the payload A0 is about.
    let t0 = Instant::now();
    let payload = serde_json::to_string(&body)
        .map_err(|e| format!("could not encode request for omega-server: {e}"))?;
    let bytes_up = payload.len() as u64;
    let serialize_ms = ms_since(t0);

    let t1 = Instant::now();
    let resp = req
        .set("Content-Type", "application/json")
        .send_string(&payload)
        .map_err(|e| format!("omega-server request to {endpoint} failed: {e}"))?;
    // The server reports the phases only it can see; without them "slow" cannot
    // be attributed to the wire or to the simulation.
    let server_timing = resp.header("Server-Timing").map(|h| h.to_string());
    let text = resp
        .into_string()
        .map_err(|e| format!("could not read omega-server response: {e}"))?;
    let network_ms = ms_since(t1);
    let bytes_down = text.len() as u64;

    let t2 = Instant::now();
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("bad response JSON from omega-server: {e}"))?;
    let deserialize_ms = ms_since(t2);

    stats().record(CallStats {
        bytes_up,
        bytes_down,
        serialize_ms,
        // `network_ms` covers the whole exchange, including the time the server
        // spent working. Subtracting the server's own total is what isolates
        // the wire — see `RemoteStats::summary`.
        network_ms,
        deserialize_ms,
        server_ms: server_timing.as_deref().map(parse_server_timing_total),
    });
    Ok(value)
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Sum the durations in a `Server-Timing` header value
/// (`admit;dur=0.4, exec;dur=812.5`). Unparseable parts are skipped rather than
/// failing the call: a timing header is diagnostics, never a correctness input.
fn parse_server_timing_total(header: &str) -> f64 {
    header
        .split(',')
        .filter_map(|part| part.split(";dur=").nth(1))
        .filter_map(|d| d.trim().parse::<f64>().ok())
        .sum()
}

/// One request's measurements.
#[derive(Clone, Copy, Debug, Default)]
pub struct CallStats {
    pub bytes_up: u64,
    pub bytes_down: u64,
    pub serialize_ms: f64,
    pub network_ms: f64,
    pub deserialize_ms: f64,
    /// Total the server attributed to itself, when it reported one.
    pub server_ms: Option<f64>,
}

/// Accumulated cost of talking to a remote server.
///
/// Answers the question that decides what to do next: **is remoting a drag, or
/// a detail?** Without splitting work-you-would-pay-anywhere from
/// overhead-you-pay-only-because-this-is-remote, an 80%-overhead run and an
/// 8%-overhead run look identical — both are just "slow", and the fixes are
/// completely different.
#[derive(Clone, Debug, Default)]
pub struct RemoteStats {
    pub requests: u64,
    pub bytes_up: u64,
    pub bytes_down: u64,
    pub serialize_ms: f64,
    pub network_ms: f64,
    pub deserialize_ms: f64,
    pub server_ms: f64,
}

impl RemoteStats {
    fn add(&mut self, c: CallStats) {
        self.requests += 1;
        self.bytes_up += c.bytes_up;
        self.bytes_down += c.bytes_down;
        self.serialize_ms += c.serialize_ms;
        self.network_ms += c.network_ms;
        self.deserialize_ms += c.deserialize_ms;
        self.server_ms += c.server_ms.unwrap_or(0.0);
    }

    /// Time spent inside the server doing the actual simulation — the work that
    /// would be paid on any machine, local or remote.
    pub fn compute_ms(&self) -> f64 {
        self.server_ms
    }

    /// Everything paid *only because this is remote*: encoding, the wire, and
    /// decoding. `network_ms` includes the server's own time, so remove it.
    pub fn overhead_ms(&self) -> f64 {
        let wire = (self.network_ms - self.server_ms).max(0.0);
        wire + self.serialize_ms + self.deserialize_ms
    }

    pub fn total_ms(&self) -> f64 {
        self.compute_ms() + self.overhead_ms()
    }

    /// `transfer-bound` / `compute-bound` / `balanced` — the one word that says
    /// whether to fix the client, buy a bigger box, or stop worrying.
    pub fn verdict(&self) -> &'static str {
        let total = self.total_ms();
        if total <= 0.0 {
            return "no remote calls";
        }
        let share = self.overhead_ms() / total;
        if share >= 0.6 {
            "transfer-bound"
        } else if share <= 0.25 {
            "compute-bound"
        } else {
            "balanced"
        }
    }

    /// Human-readable summary. Callers print this to **stderr** (Q8) so piping
    /// results to a file still shows the accounting in the terminal.
    pub fn summary(&self) -> String {
        if self.requests == 0 {
            return "remote: no calls".to_string();
        }
        let total = self.total_ms();
        let pct = |v: f64| if total > 0.0 { v / total * 100.0 } else { 0.0 };
        let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
        format!(
            "remote: {} request(s) | total {:.1} ms | compute {:.1} ms ({:.0}%) | \
             overhead {:.1} ms ({:.0}%) | up {:.2} MiB, down {:.2} MiB — {}",
            self.requests,
            total,
            self.compute_ms(),
            pct(self.compute_ms()),
            self.overhead_ms(),
            pct(self.overhead_ms()),
            mib(self.bytes_up),
            mib(self.bytes_down),
            self.verdict(),
        )
    }
}

/// Process-wide accumulator. A library must not print (Q8), so it records and
/// lets the caller decide when and where to report.
static STATS: OnceLock<Mutex<RemoteStats>> = OnceLock::new();

fn stats() -> &'static Mutex<RemoteStats> {
    STATS.get_or_init(|| Mutex::new(RemoteStats::default()))
}

trait Record {
    fn record(&self, c: CallStats);
}

impl Record for Mutex<RemoteStats> {
    fn record(&self, c: CallStats) {
        if let Ok(mut g) = self.lock() {
            g.add(c);
        }
    }
}

/// Snapshot the accumulated remote cost.
pub fn remote_stats() -> RemoteStats {
    stats().lock().map(|g| g.clone()).unwrap_or_default()
}

/// Reset the accumulator — for a caller that reports per epoch or per phase.
pub fn reset_remote_stats() {
    if let Ok(mut g) = stats().lock() {
        *g = RemoteStats::default();
    }
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
    let vals =
        expectation_remote_batch(circuit, observable, std::slice::from_ref(bindings), remote)?;
    vals.into_iter()
        .next()
        .ok_or_else(|| "omega-server returned no expectation value".to_string())
}

/// `⟨O⟩` for **many parameter bindings in one request**.
///
/// The server has always accepted a `circuits` batch
/// (`quantum_bridge::QuantumExpectationReq`); the client was sending one HTTP
/// request per row anyway. That costs a full round trip each: at 256 rows over
/// a tunnel with ~20 ms RTT it is ~5 s of pure latency per forward pass, before
/// any simulation happens.
///
/// Results come back **in input order**, one per binding (Q5). The server
/// preserves that order, and this function does not reorder.
///
/// What this does *not* fix: every row still carries its full gate list,
/// because the wire IR's parameters are concrete `f64` with no symbolic form
/// (`OmegaGateOp.params`). For a same-ansatz batch that payload dominates —
/// see `FIXES_PLAN.md` A0, where the template + parameter-matrix encoding is
/// the ~90x win. Use [`crate::remote::remote_stats`] to see which side you are
/// paying for rather than guessing.
pub fn expectation_remote_batch(
    circuit: &Circuit,
    observable: &str,
    bindings: &[HashMap<String, f64>],
    remote: &Remote,
) -> Result<Vec<f64>, String> {
    if bindings.is_empty() {
        return Ok(Vec::new());
    }
    let free: Vec<String> = circuit.free_symbols().into_iter().collect();
    let mut irs = Vec::with_capacity(bindings.len());
    for b in bindings {
        let mut full = b.clone();
        for s in &free {
            full.entry(s.clone()).or_insert(0.0);
        }
        let bound = circuit.bind_params(&full)?;
        irs.push(try_to_omega_ir(&bound)?);
    }

    let body = serde_json::json!({ "circuits": irs, "observable": observable });
    let v = post_json(remote, "/v1/quantum/expectation", body)?;
    let values = parse_expectation_values(&v)?;
    if values.len() != bindings.len() {
        // Row order and count are the contract (Q5): silently accepting a
        // short vector would misalign every downstream label.
        return Err(format!(
            "omega-server returned {} expectation values for {} circuits — \
             the batch contract is one value per row, in order",
            values.len(),
            bindings.len()
        ));
    }
    Ok(values)
}

/// Decode the server's `{"<bitstring>": n}` counts object.
///
/// The wire was never `u64`-shaped — only this conversion was. Reading the key
/// with `u64::from_str_radix` capped a remote run at 64 qubits: at 65 it
/// returned "bad bitstring key" without saying that width was the reason, and
/// it discarded the width even below the cliff, so `"0010"` and `"10"` decoded
/// to the same key. `PLAN-WIDE-COUNTS.md` names this class of conversion site
/// as where a width defect would survive if it survived anywhere — it did, here,
/// behind a feature the workspace test stage never builds.
///
/// `Outcome::from_bitstring` reads the same MSB-first spelling and keeps the
/// width the server sent.
fn parse_counts_map(
    counts: &serde_json::Map<String, serde_json::Value>,
) -> Result<HashMap<Outcome, u32>, String> {
    let mut map: HashMap<Outcome, u32> = HashMap::new();
    for (k, val) in counts {
        let state =
            Outcome::from_bitstring(k).map_err(|e| format!("bad bitstring key '{k}': {e}"))?;
        map.insert(state, val.as_u64().unwrap_or(0) as u32);
    }
    Ok(map)
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

    let res = ExecResult::Counts(parse_counts_map(counts)?);
    // Same creg semantics as the local backends, decided by the SAME
    // lowering as run_counts (not by the wire IR, whose mid-circuit
    // detection uses a different meta-gate set): project final-measurement
    // programs onto the classical register; mid-circuit programs keep
    // full-register keying. Circuits lower() can't handle (e.g. photonic
    // gates) keep the server's raw keying.
    match lower(&bound) {
        // Same rule as the local path: a backend that already keyed on the
        // creg must not be re-projected.
        Ok(low)
            if !low.needs_collapse
                && !omega_core::executor::counts_keyed_on_creg(&low.ir, false) =>
        {
            project_counts_onto_creg(res, &measure_pairs(&low))
        }
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

        let mut map: HashMap<Outcome, u32> = HashMap::new();
        map.insert(Outcome::from_u64(0b00, 2), 40); // q0=0
        map.insert(Outcome::from_u64(0b10, 2), 41); // q0=0
        map.insert(Outcome::from_u64(0b01, 2), 10); // q0=1
        map.insert(Outcome::from_u64(0b11, 2), 9); //  q0=1
        let res = project_counts_onto_creg(ExecResult::Counts(map), &measure_pairs(&low)).unwrap();
        if let ExecResult::Counts(m) = res {
            assert_eq!(m.get(&Outcome::from_u64(0, 1)), Some(&81));
            assert_eq!(m.get(&Outcome::from_u64(1, 1)), Some(&19));
            assert_eq!(m.len(), 2);
        } else {
            panic!("expected counts");
        }
    }

    /// The remote decoder must survive past bit 64 — the cliff the local path
    /// lost in `PLAN-WIDE-COUNTS` and this one kept until today.
    ///
    /// Chosen so it CANNOT pass under the old `u64::from_str_radix` decoder:
    /// 70 characters is `Err(PosOverflow)` there, so the function returned an
    /// error rather than a wrong key. The set bit is at index 69 — above the
    /// 64-bit word boundary — and bit 3 is set as well, so a decoder that
    /// silently truncated to the low word would return a key that compares
    /// unequal here rather than one that happens to match.
    #[test]
    fn remote_counts_decode_above_the_64_bit_cliff() {
        // MSB-first: string position j carries bit 69-j.
        let mut chars = vec!['0'; 70];
        chars[0] = '1'; // bit 69
        chars[69 - 3] = '1'; // bit 3
        let key: String = chars.into_iter().collect();

        let mut obj = serde_json::Map::new();
        obj.insert(key.clone(), serde_json::json!(7));
        let m = parse_counts_map(&obj).expect("70-bit key must decode");

        let decoded = m.keys().next().unwrap();
        assert_eq!(decoded.width(), 70, "width came from the wire, not a cast");
        assert_eq!(decoded.bit(69), 1, "the bit above word 0 survived");
        assert_eq!(decoded.bit(3), 1);
        assert_eq!(decoded.as_u64(), None, "70 bits do not fit a u64 by design");
        assert_eq!(decoded.to_bitstring(), key, "round-trips to what was sent");
        assert_eq!(m.get(decoded), Some(&7));
    }

    /// Width is part of the key: `"0010"` and `"10"` are different outcomes of
    /// different registers, and the old `u64` decoder collapsed them to `2`.
    #[test]
    fn remote_counts_keep_the_width_the_server_sent() {
        let mut obj = serde_json::Map::new();
        obj.insert("0010".to_string(), serde_json::json!(3));
        obj.insert("10".to_string(), serde_json::json!(5));
        let m = parse_counts_map(&obj).unwrap();
        assert_eq!(m.len(), 2, "two widths are two keys, not one summed key");
        assert_eq!(m.get(&Outcome::from_u64(0b0010, 4)), Some(&3));
        assert_eq!(m.get(&Outcome::from_u64(0b10, 2)), Some(&5));
    }

    #[test]
    fn remote_counts_refuse_a_non_binary_key() {
        let mut obj = serde_json::Map::new();
        obj.insert("01x1".to_string(), serde_json::json!(1));
        let e = parse_counts_map(&obj).unwrap_err();
        assert!(e.contains("01x1"), "error names the offending key: {e}");
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

#[cfg(test)]
mod timing_tests {
    use super::*;

    /// The batch contract is one value per row, in input order (Q5). A short
    /// or long vector must be an error, never silently accepted — misaligning
    /// rows against their labels is the kind of bug that produces plausible
    /// numbers forever.
    #[test]
    fn batch_length_mismatch_is_refused_not_silently_accepted() {
        let three = serde_json::json!({ "values": [0.1, 0.2, 0.3] });
        assert_eq!(parse_expectation_values(&three).unwrap().len(), 3);
        // The guard itself lives in expectation_remote_batch; this pins the
        // shape it depends on so a server change cannot quietly slip past.
        let empty = serde_json::json!({ "values": [] });
        assert_eq!(parse_expectation_values(&empty).unwrap().len(), 0);
        let junk = serde_json::json!({ "nope": 1 });
        assert!(parse_expectation_values(&junk).is_err());
        let not_numbers = serde_json::json!({ "values": ["a"] });
        assert!(parse_expectation_values(&not_numbers).is_err());
    }

    #[test]
    fn server_timing_header_total_is_the_sum_of_its_phases() {
        let h = "admit;dur=0.412, exec;dur=812.500, serialize;dur=31.204";
        let total = parse_server_timing_total(h);
        assert!((total - 844.116).abs() < 1e-6, "got {total}");
        // Diagnostics must never fail a call: junk contributes nothing.
        assert_eq!(parse_server_timing_total(""), 0.0);
        assert_eq!(parse_server_timing_total("garbage"), 0.0);
        assert_eq!(parse_server_timing_total("exec;dur=notanumber"), 0.0);
        // A partially-valid header still yields what it can.
        assert!((parse_server_timing_total("a;dur=1.5, junk, b;dur=2.5") - 4.0).abs() < 1e-9);
    }

    /// The headline split: work you would pay anywhere vs cost incurred only
    /// because the work is remote.
    #[test]
    fn overhead_excludes_server_time_from_the_wire_measurement() {
        let mut s = RemoteStats::default();
        s.add(CallStats {
            bytes_up: 1000,
            bytes_down: 100,
            serialize_ms: 2.0,
            // The round trip took 100 ms, of which the server used 80.
            network_ms: 100.0,
            deserialize_ms: 3.0,
            server_ms: Some(80.0),
        });
        assert_eq!(s.compute_ms(), 80.0);
        // wire = 100 - 80 = 20, plus 2 encode and 3 decode.
        assert_eq!(s.overhead_ms(), 25.0);
        assert_eq!(s.total_ms(), 105.0);
    }

    #[test]
    fn verdict_names_what_to_do_next() {
        // Dominated by transfer: batch the rows.
        let mut t = RemoteStats::default();
        t.add(CallStats {
            network_ms: 100.0,
            server_ms: Some(5.0),
            ..Default::default()
        });
        assert_eq!(t.verdict(), "transfer-bound");

        // Dominated by simulation: the wire is not the problem.
        let mut c = RemoteStats::default();
        c.add(CallStats {
            network_ms: 100.0,
            server_ms: Some(95.0),
            ..Default::default()
        });
        assert_eq!(c.verdict(), "compute-bound");

        assert_eq!(RemoteStats::default().verdict(), "no remote calls");
    }

    #[test]
    fn summary_leads_with_the_numbers_a_reader_acts_on() {
        let mut s = RemoteStats::default();
        s.add(CallStats {
            bytes_up: 4 * 1024 * 1024,
            bytes_down: 1024,
            serialize_ms: 1.0,
            network_ms: 100.0,
            deserialize_ms: 1.0,
            server_ms: Some(10.0),
        });
        let out = s.summary();
        assert!(out.contains("1 request"), "{out}");
        assert!(out.contains("compute"), "{out}");
        assert!(out.contains("overhead"), "{out}");
        assert!(out.contains("4.00 MiB"), "{out}");
        assert!(out.contains("transfer-bound"), "{out}");
    }

    #[test]
    fn a_server_that_reports_no_timing_does_not_corrupt_the_split() {
        // Older servers send no Server-Timing. Compute is then unknown (0) and
        // everything counts as overhead — pessimistic, but never negative and
        // never silently attributed to the wrong side.
        let mut s = RemoteStats::default();
        s.add(CallStats {
            network_ms: 50.0,
            server_ms: None,
            ..Default::default()
        });
        assert_eq!(s.compute_ms(), 0.0);
        assert_eq!(s.overhead_ms(), 50.0);
        assert!(s.overhead_ms() >= 0.0);
    }
}
