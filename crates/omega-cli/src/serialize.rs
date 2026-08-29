//! JSON serialization for omega-run output modes.
//!
//! Every mode produces a single JSON object on stdout (or, for `jsonl`, one
//! object per line). Stderr stays human-readable.

use omega_core::outcome::Outcome;
use std::collections::HashMap;

use serde_json::{json, Value};

use omega_core::circuit::CircuitType;
use omega_core::executor::ExecResult;

/// Requested output format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Text,
    Json,
    Jsonl,
}

impl Format {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "text" => Some(Format::Text),
            "json" => Some(Format::Json),
            "jsonl" => Some(Format::Jsonl),
            _ => None,
        }
    }

    pub fn is_machine(&self) -> bool {
        !matches!(self, Format::Text)
    }
}

/// Format a bitstring as "00101" (MSB = highest qubit index) for gate-based
/// circuits, or "n0,n1,..." Fock string for photonics.
fn format_bits(o: &Outcome, num_qubits: u32, circuit_type: &CircuitType) -> String {
    match circuit_type {
        CircuitType::Photonic => {
            // Photonic keys are packed occupancies (4 bits per mode), not one
            // bit per qubit, so they are read as an integer. The dense Fock
            // basis is bounded far below 2^64 entries.
            let bits = o.as_u64().unwrap_or(0);
            let modes = num_qubits as usize;
            let mut out = Vec::with_capacity(modes);
            for m in 0..modes {
                let n = (bits >> (m * 4)) & 0xF;
                out.push(n.to_string());
            }
            out.join(",")
        }
        // The WIDTH COMES FROM THE KEY, not from `num_qubits`. Those differ
        // whenever the outcome is the classical register, and padding to
        // `num_qubits` is what printed a 2-bit result as 1024 characters.
        CircuitType::GateBased => o.to_bitstring(),
    }
}

/// The git revision this binary was built from, or `unknown`.
///
/// Suffixed `-dirty` when the working tree had uncommitted changes at build
/// time — a bare `eedc80d` on a modified tree names a commit whose contents are
/// not what ran, and that is the failure this stamp exists to prevent rather
/// than a detail. See `build.rs`.
pub fn build_rev() -> &'static str {
    env!("OMEGA_BUILD_REV")
}

/// The identity every machine-readable document carries.
///
/// Every JSON/JSONL document gets this, because a result without a build
/// identity cannot be compared with another result. A downstream table recorded
/// this lane twice, five days apart, as the same string "aria omega-run"; the
/// 78x difference between those rows was two different binaries, and nothing in
/// the output could have revealed that.
fn provenance() -> Value {
    json!({
        "tool": "omega-run",
        "version": env!("CARGO_PKG_VERSION"),
        "rev": build_rev(),
    })
}

pub fn counts_to_json(
    counts: &HashMap<Outcome, u32>,
    num_qubits: u32,
    shots: u32,
    circuit_type: &CircuitType,
) -> Value {
    let mut map = serde_json::Map::new();
    let mut entries: Vec<_> = counts.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (bits, count) in entries {
        map.insert(format_bits(bits, num_qubits, circuit_type), json!(count));
    }
    json!({
        "mode": "counts",
        "shots": shots,
        "num_qubits": num_qubits,
        "counts": Value::Object(map),
        "build": provenance(),
    })
}

pub fn statevector_to_json(sv: &[num_complex::Complex<f64>], num_qubits: u32) -> Value {
    let amps: Vec<Value> = sv.iter().map(|c| json!([c.re, c.im])).collect();
    json!({
        "mode": "statevector",
        "num_qubits": num_qubits,
        "amplitudes": amps,
    })
}

/// The `--dump-state-bits` artifact (PLAN-BITEQ-CUDA-METAL.md §4): every
/// amplitude as the lowercase-hex BIT PATTERN of its float, never decimal. A
/// decimal round-trip can hide a 1-ULP difference and can also invent one,
/// and `-0.0` prints equal to `0.0` — as bits (`80000000` vs `00000000`) the
/// difference is visible, which is the point.
///
/// `backend` names the arm that ACTUALLY executed and fixes the width:
/// `*-f32` arms dump 8 hex digits of `f32::to_bits`, `cpu-f64` dumps 16 of
/// `f64::to_bits`. GPU amplitudes arrive promoted f32→f64; the promotion is
/// exact, so demoting back is lossless — and that is CHECKED per amplitude,
/// because any f64 post-processing between device and here would make the
/// demotion lossy and the artifact a lie. A failed check is an error, not a
/// rounding.
pub fn state_bits_to_json(
    circuit_name: &str,
    backend: &str,
    multi_control: &str,
    os_version: &str,
    amps: &[num_complex::Complex<f64>],
) -> Result<Value, String> {
    if !amps.len().is_power_of_two() {
        return Err(format!(
            "{} amplitudes is not a power of two; not a qubit statevector",
            amps.len()
        ));
    }
    let n_qubits = amps.len().trailing_zeros();
    let f32_bits = backend.ends_with("-f32");
    let mut hex = Vec::with_capacity(amps.len());
    for (i, a) in amps.iter().enumerate() {
        if f32_bits {
            let (re, im) = (a.re as f32, a.im as f32);
            if (f64::from(re)).to_bits() != a.re.to_bits()
                || (f64::from(im)).to_bits() != a.im.to_bits()
            {
                return Err(format!(
                    "amp[{i}] is not an exact f32 promotion ({:.17e}, {:.17e}); \
                     something reprocessed the device state in f64, and dumping \
                     it as f32 bits would misrepresent what the device computed",
                    a.re, a.im
                ));
            }
            hex.push(json!([
                format!("{:08x}", re.to_bits()),
                format!("{:08x}", im.to_bits())
            ]));
        } else {
            hex.push(json!([
                format!("{:016x}", a.re.to_bits()),
                format!("{:016x}", a.im.to_bits())
            ]));
        }
    }
    Ok(json!({
        "format": "biteq-v1",
        "build": provenance(),
        "host": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "os_version": os_version,
        },
        "backend": backend,
        "multi_control": multi_control,
        "ordering": "amp[i]: qubit 0 is the LOW bit of i",
        "circuit": circuit_name,
        "n_qubits": n_qubits,
        "precision": if f32_bits { "f32" } else { "f64" },
        "amps": hex,
    }))
}

pub fn probabilities_to_json(probs: &[f64], num_qubits: u32) -> Value {
    json!({
        "mode": "probabilities",
        "num_qubits": num_qubits,
        "probabilities": probs,
    })
}

pub fn exec_result_to_json(
    result: &ExecResult,
    num_qubits: u32,
    shots: Option<u32>,
    circuit_type: &CircuitType,
) -> Value {
    match result {
        ExecResult::Counts(c) => counts_to_json(c, num_qubits, shots.unwrap_or(0), circuit_type),
        ExecResult::Statevector(sv) => statevector_to_json(sv, num_qubits),
        ExecResult::Probabilities(p) => probabilities_to_json(p, num_qubits),
    }
}

/// Attach the MPS truncation certificate to a result document.
///
/// The certificate answers "was the number you were just handed truncated, and
/// by how much". Emitting it only to stderr meant every machine consumer had to
/// scrape log text for it, or go without — which defeats the point of computing
/// it. It is **absent**, not null-padded, for backends that produce none, so
/// "no certificate" and "a certificate reading zero" stay distinguishable.
///
/// `fidelity_estimate` is named `~` in the human output for a reason and the
/// key says so here too: it is an estimate, not a proven bound.
pub fn attach_mps_certificate(mut doc: Value, stats: &omega_backend_mps::MpsRunStats) -> Value {
    if let Some(map) = doc.as_object_mut() {
        map.insert(
            "mps_truncation".to_string(),
            json!({
                "discarded_weight": stats.discarded_weight,
                "fidelity_estimate": stats.fidelity_estimate,
                "fidelity_estimate_is_a_bound": false,
                "max_bond_reached": stats.max_bond_reached,
            }),
        );
    }
    doc
}

/// Attach the pauliprop truncation certificate to a result document.
///
/// Same contract as [`attach_mps_certificate`], and for the same reason its doc
/// gives: a bound emitted only to stderr means every machine consumer has to
/// scrape log text for it or go without. The bug report that prompted this said
/// exactly that — *"same JSON, same `"mode":"expectation"`, no flag, no
/// bound, no term count, no discarded mass"* — and its acceptance criterion is
/// that the **JSON** carries the mass.
///
/// **Absent**, not null-padded, for runs that produced no certificate, so "no
/// certificate" and "a certificate reading zero" stay distinguishable.
///
/// `dropped_mass_is_a_bound` is `true` here where MPS's
/// `fidelity_estimate_is_a_bound` is `false`, and the difference is real:
/// `|⟨P⟩| ≤ 1` for every Pauli string, so the discarded L1 mass bounds the
/// error with no gauge caveat. MPS's fidelity estimate cannot say that.
///
/// `informative` is the key a consumer should actually branch on. A bound is
/// only worth reading if it is narrower than the range the answer was already
/// known to lie in; `false` means every value the observable can take is
/// consistent with this result.
pub fn attach_pauliprop_certificate(
    mut doc: Value,
    cert: &omega_backend_pauliprop::PauliPropCertificate,
) -> Value {
    if let Some(map) = doc.as_object_mut() {
        map.insert(
            "pauliprop_truncation".to_string(),
            json!({
                "dropped_mass": cert.dropped_mass,
                "dropped_mass_is_a_bound": true,
                "observable_range": cert.observable_range,
                "vacuous_at": cert.observable_range + cert.value.abs(),
                "informative": cert.is_informative(),
                "exact": cert.is_exact(),
                "final_terms": cert.final_terms,
                "peak_terms": cert.peak_terms,
                "coeff_min": cert.coeff_min,
                "max_weight": cert.max_weight,
                "max_freq": cert.max_freq,
                "max_terms": cert.max_terms,
            }),
        );
    }
    doc
}

pub fn expectation_to_json(observable: &str, value: f64) -> Value {
    json!({
        "mode": "expectation",
        "observable": observable,
        "value": value,
        "build": provenance(),
    })
}

pub fn gradient_to_json(observable: &str, method: &str, grads: &[(String, f64)]) -> Value {
    let mut map = serde_json::Map::new();
    for (name, g) in grads {
        map.insert(name.clone(), json!(g));
    }
    json!({
        "mode": "gradient",
        "observable": observable,
        "method": method,
        "gradients": Value::Object(map),
        "build": provenance(),
    })
}

/// Output for `--gradient-of-fn` — the Phase-4 fast-AD path.
/// Mirrors `gradient_to_json` but tags the source as a functional
/// description (Qubo or Table) so downstream consumers can tell the
/// two apart without parsing the spec.
pub fn functional_gradient_to_json(
    spec_kind: &str,
    method: &str,
    grads: &[(String, f64)],
) -> Value {
    let mut map = serde_json::Map::new();
    for (name, g) in grads {
        map.insert(name.clone(), json!(g));
    }
    json!({
        "mode": "gradient_of_fn",
        "functional": spec_kind,
        "method": method,
        "gradients": Value::Object(map),
    })
}

/// Emit per-shot JSONL samples, one line per shot, flattened from counts.
pub fn emit_jsonl_counts(
    counts: &HashMap<Outcome, u32>,
    num_qubits: u32,
    circuit_type: &CircuitType,
    device_used: &str,
) {
    for line in jsonl_counts_lines(counts, num_qubits, circuit_type, device_used) {
        println!("{}", line);
    }
}

/// Stamp WHICH ARM RAN and how many threads into a machine-readable document.
///
/// Requested by an external benchmark harness that was detecting the
/// silent-CPU-fallback trap by grepping stderr prose — which breaks the first
/// time a warning is reworded. `device_used` is the arm that actually
/// executed (same source of truth as `--dump-state-bits`), so a harness can
/// gate on `device_used == "metal-f32"` instead of parsing sentences.
pub fn attach_execution_identity(mut doc: Value, device_used: &str, threads_used: usize) -> Value {
    if let Value::Object(map) = &mut doc {
        map.insert("device_used".to_string(), json!(device_used));
        map.insert("threads_used".to_string(), json!(threads_used));
    }
    doc
}

/// The JSONL stream: one `{"meta": …}` line, then one `{"bits": …}` line per
/// shot.
///
/// The meta line closes TODO §1.3's residual: every OTHER machine-readable
/// document carries the build identity, but a JSONL stream was bare per-shot
/// lines, so a consumer archiving one could not say which binary produced it —
/// the exact failure `provenance()` exists to prevent. A leading meta line
/// (keyed `meta`, so a line consumer can dispatch on the key) is the smaller
/// of the two options §1.3 recorded, and heterogeneous type-keyed lines are
/// ordinary JSONL practice.
///
/// Split from the printing wrapper so the shape is testable: stdout is not.
pub fn jsonl_counts_lines(
    counts: &HashMap<Outcome, u32>,
    num_qubits: u32,
    circuit_type: &CircuitType,
    device_used: &str,
) -> Vec<String> {
    let shots: u64 = counts.values().map(|c| u64::from(*c)).sum();
    let mut out = Vec::with_capacity(shots as usize + 1);
    out.push(
        json!({"meta": {
            "build": provenance(),
            "mode": "counts",
            "num_qubits": num_qubits,
            "shots": shots,
            "device_used": device_used,
            "threads_used": rayon::current_num_threads(),
        }})
        .to_string(),
    );
    let mut entries: Vec<_> = counts.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (bits, count) in entries {
        let s = format_bits(bits, num_qubits, circuit_type);
        for _ in 0..*count {
            out.push(json!({"bits": s}).to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_parse_known_values() {
        assert_eq!(Format::parse("text"), Some(Format::Text));
        assert_eq!(Format::parse("json"), Some(Format::Json));
        assert_eq!(Format::parse("jsonl"), Some(Format::Jsonl));
        assert_eq!(Format::parse(""), None);
        assert_eq!(Format::parse("JSON"), None, "case-sensitive");
        assert_eq!(Format::parse("yaml"), None);
    }

    #[test]
    fn format_is_machine_excludes_text_only() {
        assert!(!Format::Text.is_machine());
        assert!(Format::Json.is_machine());
        assert!(Format::Jsonl.is_machine());
    }

    /// The JSONL stream leads with an identity line, then exactly one
    /// `{"bits": …}` line per shot — and the identity carries a non-empty
    /// build rev, which is the entire point (§1.3's residual: a consumer
    /// archiving a JSONL stream could not attribute it to a binary).
    #[test]
    fn jsonl_stream_leads_with_identity_then_one_line_per_shot() {
        let mut counts = HashMap::new();
        counts.insert(Outcome::from_u64(0, 2), 3u32);
        counts.insert(Outcome::from_u64(3, 2), 2u32);
        let lines = jsonl_counts_lines(&counts, 2, &CircuitType::GateBased, "cpu-f64");

        assert_eq!(lines.len(), 1 + 5, "one meta line + one line per shot");
        let meta: Value = serde_json::from_str(&lines[0]).expect("meta is JSON");
        let m = &meta["meta"];
        assert!(!m.is_null(), "first line must be the meta line");
        assert_eq!(m["shots"], 5);
        assert_eq!(m["num_qubits"], 2);
        assert!(
            m["build"]["rev"].as_str().is_some_and(|r| !r.is_empty()),
            "meta must carry the build rev"
        );
        assert_eq!(m["device_used"], "cpu-f64", "meta names the executed arm");
        assert!(
            m["threads_used"].as_u64().is_some_and(|t| t >= 1),
            "meta carries the thread count"
        );
        for l in &lines[1..] {
            let v: Value = serde_json::from_str(l).expect("shot line is JSON");
            assert!(
                v["bits"].is_string(),
                "every non-meta line is a per-shot bits line, got {l}"
            );
            assert!(v.get("meta").is_none(), "meta appears exactly once");
        }
    }

    #[test]
    fn format_bits_gate_based_is_msb_first_zero_padded() {
        // 3 qubits: bit 0b011 (decimal 3) → "011" — MSB on the left.
        assert_eq!(
            format_bits(&Outcome::from_u64(3, 3), 3, &CircuitType::GateBased),
            "011"
        );
        // bit 0b100 (decimal 4) on 3 qubits → "100".
        assert_eq!(
            format_bits(&Outcome::from_u64(4, 3), 3, &CircuitType::GateBased),
            "100"
        );
        // 5 qubits: 1 → "00001".
        assert_eq!(
            format_bits(&Outcome::from_u64(1, 5), 5, &CircuitType::GateBased),
            "00001"
        );
        // 5 qubits: all-ones (0b11111 = 31) → "11111".
        assert_eq!(
            format_bits(&Outcome::from_u64(31, 5), 5, &CircuitType::GateBased),
            "11111"
        );
    }

    #[test]
    fn format_bits_photonic_is_comma_separated_per_mode() {
        // Photonic encoding: 4 bits per mode, LSB first.
        // 4 modes, mode 0 = 1 photon, mode 1 = 0, mode 2 = 2, mode 3 = 0
        // → bits = 0x0201 = 513. Output: "1,0,2,0".
        let bits = Outcome::from_u64(0x0201, 16);
        assert_eq!(format_bits(&bits, 4, &CircuitType::Photonic), "1,0,2,0");
        // Two modes with 1 photon each (HOM).
        assert_eq!(
            format_bits(&Outcome::from_u64(0x11, 8), 2, &CircuitType::Photonic),
            "1,1"
        );
    }

    /// **The certificate must survive as data, with real numbers in it.**
    ///
    /// End-to-end this is hard to observe: the default ceiling *refuses* a run
    /// whose discarded weight is material, so a successful run's certificate is
    /// usually ~0. That makes it easy to ship a serializer that only ever
    /// carries zeros and never notice it drops a real value. This drives the
    /// attachment directly with a starved-bond certificate.
    #[test]
    fn the_mps_certificate_carries_real_numbers() {
        let stats = omega_backend_mps::MpsRunStats {
            discarded_weight: 3.25e-3,
            fidelity_estimate: 0.87,
            max_bond_reached: 64,
        };
        let doc = attach_mps_certificate(json!({"mode": "counts"}), &stats);
        let cert = &doc["mps_truncation"];
        assert_eq!(cert["discarded_weight"], 3.25e-3);
        assert_eq!(cert["fidelity_estimate"], 0.87);
        assert_eq!(cert["max_bond_reached"], 64);
        // The estimate is not a bound, and the document must say so rather
        // than leave a consumer to assume it is one.
        assert_eq!(cert["fidelity_estimate_is_a_bound"], false);
        // The original document is preserved, not replaced.
        assert_eq!(doc["mode"], "counts");
    }

    #[test]
    fn the_pauliprop_certificate_carries_the_bound_and_its_verdict() {
        let cert = omega_backend_pauliprop::PauliPropCertificate {
            value: 0.7054827807,
            dropped_mass: 2.0611,
            final_terms: 7,
            peak_terms: 12,
            coeff_min: 1e-1,
            max_weight: None,
            max_freq: Some(3),
            max_terms: Some(1 << 21),
            observable_range: 1.0,
        };
        let doc = attach_pauliprop_certificate(json!({"mode": "expectation"}), &cert);
        let c = &doc["pauliprop_truncation"];
        assert_eq!(c["dropped_mass"], 2.0611);
        // 2.0611 >= 1.0 + |0.7055| = 1.7055, so it excludes nothing.
        assert_eq!(c["observable_range"], 1.0);
        assert_eq!(c["final_terms"], 7);
        assert_eq!(c["peak_terms"], 12);
        assert_eq!(c["max_freq"], 3);
        // Unlike the MPS fidelity estimate, this one IS a bound and says so.
        assert_eq!(c["dropped_mass_is_a_bound"], true);
        // 2.06 on a quantity confined to [-1, 1]: correct, and excluding
        // nothing. This is the key a consumer should branch on, and these are
        // the real numbers from a Trotter-Ising sweep at C=1e-1.
        assert_eq!(c["informative"], false);
        assert_eq!(c["exact"], false);
        assert_eq!(doc["mode"], "expectation");
    }

    #[test]
    fn an_exact_pauliprop_run_certifies_zero_and_informative() {
        let cert = omega_backend_pauliprop::PauliPropCertificate {
            value: 0.1118767747,
            dropped_mass: 0.0,
            final_terms: 16255,
            peak_terms: 16255,
            coeff_min: 0.0,
            max_weight: None,
            max_freq: None,
            max_terms: Some(1 << 21),
            observable_range: 1.0,
        };
        let doc = attach_pauliprop_certificate(json!({"mode": "expectation"}), &cert);
        let c = &doc["pauliprop_truncation"];
        // The bug report's own acceptance: "an untruncated run reports it as 0".
        assert_eq!(c["dropped_mass"], 0.0);
        assert_eq!(c["exact"], true);
        assert_eq!(c["informative"], true);
        assert_eq!(c["max_weight"], serde_json::Value::Null);
    }

    /// Absent, not null: "this backend produced no certificate" and "the
    /// certificate reads zero" are different facts and must stay tellable
    /// apart by a consumer.
    #[test]
    fn a_backend_without_a_certificate_emits_no_key() {
        let doc = json!({"mode": "counts"});
        assert!(doc.get("mps_truncation").is_none());
        assert!(doc.get("pauliprop_truncation").is_none());
    }

    #[test]
    fn counts_to_json_orders_by_count_then_lex() {
        // Two outcomes; the lower-count one must appear last regardless
        // of bit value.
        let mut counts: HashMap<Outcome, u32> = HashMap::new();
        counts.insert(Outcome::from_u64(0b00, 2), 700);
        counts.insert(Outcome::from_u64(0b11, 2), 300);
        let v = counts_to_json(&counts, 2, 1000, &CircuitType::GateBased);
        assert_eq!(v["mode"], "counts");
        assert_eq!(v["shots"], 1000);
        assert_eq!(v["num_qubits"], 2);
        let counts_obj = v["counts"].as_object().unwrap();
        let keys: Vec<&String> = counts_obj.keys().collect();
        // serde_json::Map preserves insertion order, and we insert
        // sorted by descending count.
        assert_eq!(keys[0], "00");
        assert_eq!(keys[1], "11");
    }

    #[test]
    fn counts_to_json_ties_break_on_lex() {
        // Two outcomes with equal counts — tie breaker is bit lex
        // (ascending). 0b01 < 0b10 in raw u64 ordering, so "01"
        // sorts before "10".
        let mut counts: HashMap<Outcome, u32> = HashMap::new();
        counts.insert(Outcome::from_u64(0b10, 2), 500);
        counts.insert(Outcome::from_u64(0b01, 2), 500);
        let v = counts_to_json(&counts, 2, 1000, &CircuitType::GateBased);
        let keys: Vec<&String> = v["counts"].as_object().unwrap().keys().collect();
        assert_eq!(keys[0], "01");
        assert_eq!(keys[1], "10");
    }

    #[test]
    fn statevector_to_json_emits_re_im_pairs() {
        use num_complex::Complex;
        let sv = vec![Complex::new(1.0, 0.0), Complex::new(0.0, -0.5)];
        let v = statevector_to_json(&sv, 1);
        assert_eq!(v["mode"], "statevector");
        assert_eq!(v["num_qubits"], 1);
        let amps = v["amplitudes"].as_array().unwrap();
        assert_eq!(amps.len(), 2);
        assert_eq!(amps[0][0], 1.0);
        assert_eq!(amps[0][1], 0.0);
        assert_eq!(amps[1][0], 0.0);
        assert_eq!(amps[1][1], -0.5);
    }

    #[test]
    fn probabilities_to_json_preserves_order() {
        let probs = vec![0.5, 0.0, 0.0, 0.5];
        let v = probabilities_to_json(&probs, 2);
        assert_eq!(v["mode"], "probabilities");
        assert_eq!(v["num_qubits"], 2);
        let arr = v["probabilities"].as_array().unwrap();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr[0], 0.5);
        assert_eq!(arr[3], 0.5);
    }

    #[test]
    fn expectation_to_json_carries_observable_and_value() {
        let v = expectation_to_json("Z0+0.5*X1", -0.42);
        assert_eq!(v["mode"], "expectation");
        assert_eq!(v["observable"], "Z0+0.5*X1");
        assert_eq!(v["value"], -0.42);
    }

    #[test]
    fn gradient_to_json_keyed_by_symbol_name() {
        let grads = vec![("theta_0".to_string(), 0.1), ("beta_1".to_string(), -0.2)];
        let v = gradient_to_json("Z0", "adjoint", &grads);
        assert_eq!(v["mode"], "gradient");
        assert_eq!(v["observable"], "Z0");
        assert_eq!(v["method"], "adjoint");
        assert_eq!(v["gradients"]["theta_0"], 0.1);
        assert_eq!(v["gradients"]["beta_1"], -0.2);
    }

    #[test]
    fn functional_gradient_to_json_tags_spec_kind() {
        let grads = vec![("th0".to_string(), 1.5)];
        let v = functional_gradient_to_json("qubo", "diagonal", &grads);
        assert_eq!(v["mode"], "gradient_of_fn");
        assert_eq!(v["functional"], "qubo");
        assert_eq!(v["method"], "diagonal");
        assert_eq!(v["gradients"]["th0"], 1.5);
    }

    #[test]
    fn exec_result_to_json_dispatches_on_variant() {
        // Counts variant with shots
        let mut c: HashMap<Outcome, u32> = HashMap::new();
        c.insert(Outcome::from_u64(0, 1), 10);
        let v = exec_result_to_json(&ExecResult::Counts(c), 1, Some(10), &CircuitType::GateBased);
        assert_eq!(v["mode"], "counts");

        // Statevector variant
        use num_complex::Complex;
        let sv = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let v = exec_result_to_json(
            &ExecResult::Statevector(sv),
            1,
            None,
            &CircuitType::GateBased,
        );
        assert_eq!(v["mode"], "statevector");

        // Probabilities variant
        let v = exec_result_to_json(
            &ExecResult::Probabilities(vec![1.0, 0.0]),
            1,
            None,
            &CircuitType::GateBased,
        );
        assert_eq!(v["mode"], "probabilities");
    }
}

#[cfg(test)]
mod build_stamp_tests {
    use super::*;

    /// The stamp must be a plausible identity, not an empty string or a shell
    /// error that got captured.
    ///
    /// `build.rs` shells out to git, and every failure mode there produces a
    /// *string* rather than a compile error — an empty capture, a "fatal: not a
    /// git repository" line, a stray newline. Any of those would be baked into
    /// every JSON document as the build identity and would look like data.
    #[test]
    fn the_build_rev_is_a_plausible_identity() {
        let rev = build_rev();
        assert!(!rev.is_empty(), "build rev is empty");
        assert!(
            !rev.contains(char::is_whitespace),
            "build rev {rev:?} contains whitespace, so it is probably captured \
             command output rather than a revision"
        );
        let core = rev.strip_suffix("-dirty").unwrap_or(rev);
        assert!(
            core == "unknown" || (core.len() >= 7 && core.chars().all(|c| c.is_ascii_hexdigit())),
            "build rev {rev:?} is neither `unknown` nor a hex object name; \
             `unknown` is the legitimate answer when git is unavailable (a source \
             tarball), and anything else means build.rs captured something it \
             should not have"
        );
    }

    /// Every machine-readable document carries the identity.
    ///
    /// Asserted per-document rather than on `provenance()` alone: the failure
    /// this guards against is a NEW emitter being added without the field, and
    /// testing the helper in isolation would pass while that happened.
    #[test]
    fn every_json_document_carries_the_build_identity() {
        use std::collections::HashMap;
        let docs = vec![
            (
                "counts",
                counts_to_json(&HashMap::new(), 2, 0, &CircuitType::GateBased),
            ),
            ("expectation", expectation_to_json("Z0", 0.5)),
            ("gradient", gradient_to_json("Z0", "adjoint", &[])),
        ];
        for (what, doc) in docs {
            let build = doc
                .get("build")
                .unwrap_or_else(|| panic!("{what} document has no `build` field"));
            assert_eq!(
                build.get("tool").and_then(|v| v.as_str()),
                Some("omega-run")
            );
            assert_eq!(
                build.get("rev").and_then(|v| v.as_str()),
                Some(build_rev()),
                "{what} document reports a different rev than build_rev()"
            );
            assert!(
                build.get("version").and_then(|v| v.as_str()).is_some(),
                "{what} document has no version"
            );
        }
    }
}
