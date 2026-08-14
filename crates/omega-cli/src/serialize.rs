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

pub fn expectation_to_json(observable: &str, value: f64) -> Value {
    json!({
        "mode": "expectation",
        "observable": observable,
        "value": value,
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
pub fn emit_jsonl_counts(counts: &HashMap<Outcome, u32>, num_qubits: u32, circuit_type: &CircuitType) {
    let mut entries: Vec<_> = counts.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (bits, count) in entries {
        let s = format_bits(bits, num_qubits, circuit_type);
        for _ in 0..*count {
            let line = json!({"bits": s});
            println!("{}", line);
        }
    }
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

    #[test]
    fn format_bits_gate_based_is_msb_first_zero_padded() {
        // 3 qubits: bit 0b011 (decimal 3) → "011" — MSB on the left.
        assert_eq!(format_bits(&Outcome::from_u64(3, 3), 3, &CircuitType::GateBased), "011");
        // bit 0b100 (decimal 4) on 3 qubits → "100".
        assert_eq!(format_bits(&Outcome::from_u64(4, 3), 3, &CircuitType::GateBased), "100");
        // 5 qubits: 1 → "00001".
        assert_eq!(format_bits(&Outcome::from_u64(1, 5), 5, &CircuitType::GateBased), "00001");
        // 5 qubits: all-ones (0b11111 = 31) → "11111".
        assert_eq!(format_bits(&Outcome::from_u64(31, 5), 5, &CircuitType::GateBased), "11111");
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
