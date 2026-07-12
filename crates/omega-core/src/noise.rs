//! Shared, backend-agnostic noise model for the Aria/omega simulators.
//!
//! One [`NoiseModel`] is consumed by every backend that can represent noise —
//! the statevector trajectory sampler, the MPS trajectory sampler, and the
//! Pauli-propagation (Heisenberg) engine. Keeping the *data* here (and the
//! *application* in each backend) means all three read exactly the same
//! `--noise` JSON and agree channel-for-channel.
//!
//! # Two regimes, one schema
//!
//! Every rate is a [`Rate`] — either a single number applied to **all** qubits,
//! or a per-qubit array:
//!
//! * **Idealized machine (adjustable):** scalar rates, e.g.
//!   `{"amplitude_damping":5e-4,"depolarizing":0.001,"readout_flip":0.02}`.
//!   Dial one knob and every qubit degrades uniformly.
//! * **Real hardware (calibrated):** per-qubit arrays, e.g.
//!   `{"amplitude_damping":[0.004,0.006,0.005],
//!     "depolarizing":{"1q":0.001,"2q":0.012},
//!     "readout":[{"p10":0.02,"p01":0.03}, …]}`.
//!   Each qubit and each gate arity gets its own error, and readout is
//!   asymmetric (`p10 = P(read 1 | true 0)`, `p01 = P(read 0 | true 1)`).
//!
//! Both forms parse into the same struct, so a backend never has to know which
//! regime it is running.

use serde_json::Value;

/// Recognised top-level keys in a `--noise` object. Anything else is rejected
/// by [`NoiseModel::from_json`] so a typo (e.g. `readout` vs `readout_flip`)
/// surfaces loudly instead of silently defaulting to zero.
const NOISE_KEYS: &[&str] = &[
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "pauli",
    "readout_flip",
    "readout",
];

/// A noise rate that is either uniform across qubits or specified per qubit.
///
/// `PerQubit` indices beyond its length read as `0.0`, so a short array leaves
/// the remaining qubits noiseless rather than erroring — convenient when only
/// a few qubits are calibrated.
#[derive(Clone, Debug, PartialEq)]
pub enum Rate {
    /// Same value on every qubit (the idealized-machine regime).
    Uniform(f64),
    /// One value per qubit index (the calibrated-hardware regime).
    PerQubit(Vec<f64>),
}

impl Default for Rate {
    fn default() -> Self {
        Rate::Uniform(0.0)
    }
}

impl From<f64> for Rate {
    fn from(v: f64) -> Self {
        Rate::Uniform(v)
    }
}

impl Rate {
    /// The rate for qubit `q` (0.0 past the end of a `PerQubit` array).
    pub fn at(&self, q: usize) -> f64 {
        match self {
            Rate::Uniform(v) => *v,
            Rate::PerQubit(v) => v.get(q).copied().unwrap_or(0.0),
        }
    }

    /// True when this rate is zero on every qubit.
    pub fn is_zero(&self) -> bool {
        match self {
            Rate::Uniform(v) => *v == 0.0,
            Rate::PerQubit(v) => v.iter().all(|x| *x == 0.0),
        }
    }

    fn from_json(v: &Value) -> Result<Rate, String> {
        if let Some(f) = v.as_f64() {
            Ok(Rate::Uniform(f))
        } else if let Some(arr) = v.as_array() {
            let mut out = Vec::with_capacity(arr.len());
            for e in arr {
                out.push(
                    e.as_f64()
                        .ok_or_else(|| "per-qubit rate array must contain numbers".to_string())?,
                );
            }
            Ok(Rate::PerQubit(out))
        } else {
            Err("rate must be a number or an array of numbers".to_string())
        }
    }
}

/// Depolarizing rate, split by gate arity so a two-qubit gate can carry a
/// larger error than a one-qubit gate (as on real hardware). A bare number or
/// array sets both arities equal.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Depolarizing {
    pub one_q: Rate,
    pub two_q: Rate,
}

impl Depolarizing {
    /// Same depolarizing rate on every qubit and both gate arities.
    pub fn uniform(p: f64) -> Self {
        Depolarizing {
            one_q: Rate::Uniform(p),
            two_q: Rate::Uniform(p),
        }
    }

    /// The depolarizing rate applied to a qubit touched by a gate acting on
    /// `arity` qubits (1 → `one_q`, otherwise `two_q`).
    pub fn at(&self, q: usize, arity: usize) -> f64 {
        if arity <= 1 {
            self.one_q.at(q)
        } else {
            self.two_q.at(q)
        }
    }

    pub fn is_zero(&self) -> bool {
        self.one_q.is_zero() && self.two_q.is_zero()
    }

    fn from_json(v: &Value) -> Result<Depolarizing, String> {
        // Object form: {"1q": rate, "2q": rate}. Scalar/array form: both equal.
        if let Some(obj) = v.as_object() {
            for k in obj.keys() {
                if !matches!(k.as_str(), "1q" | "2q") {
                    return Err(format!(
                        "unknown depolarizing key '{k}' (recognised: 1q, 2q)"
                    ));
                }
            }
            let one_q = obj.get("1q").map(Rate::from_json).transpose()?.unwrap_or_default();
            let two_q = obj.get("2q").map(Rate::from_json).transpose()?.unwrap_or_default();
            Ok(Depolarizing { one_q, two_q })
        } else {
            let r = Rate::from_json(v)?;
            Ok(Depolarizing {
                one_q: r.clone(),
                two_q: r,
            })
        }
    }
}

/// Per-qubit Pauli channel: after each gate, apply X/Y/Z with the given
/// (per-qubit) probabilities; the remainder is identity. The four rates
/// `p_x + p_y + p_z` must stay ≤ 1 per qubit.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PauliChannel {
    pub x: Rate,
    pub y: Rate,
    pub z: Rate,
}

impl PauliChannel {
    pub fn is_zero(&self) -> bool {
        self.x.is_zero() && self.y.is_zero() && self.z.is_zero()
    }

    fn from_json(v: &Value) -> Result<PauliChannel, String> {
        let obj = v
            .as_object()
            .ok_or_else(|| "\"pauli\" must be an object with X/Y/Z rates".to_string())?;
        for k in obj.keys() {
            // `I` is accepted for backward compatibility (it was the identity
            // rate) but ignored — the identity weight is implied by 1 − x − y − z.
            if !matches!(k.as_str(), "I" | "X" | "Y" | "Z") {
                return Err(format!(
                    "unknown pauli key '{k}' (recognised: I, X, Y, Z)"
                ));
            }
        }
        let get = |k: &str| obj.get(k).map(Rate::from_json).transpose();
        Ok(PauliChannel {
            x: get("X")?.unwrap_or_default(),
            y: get("Y")?.unwrap_or_default(),
            z: get("Z")?.unwrap_or_default(),
        })
    }
}

/// Per-qubit, possibly asymmetric readout (measurement) error.
///
/// `p10 = P(report 1 | qubit truly 0)`, `p01 = P(report 0 | qubit truly 1)`.
/// A symmetric `readout_flip` scalar sets `p10 = p01`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ReadoutError {
    pub p10: Rate,
    pub p01: Rate,
}

impl ReadoutError {
    /// Symmetric readout error `p` on every qubit (`p10 = p01 = p`).
    pub fn symmetric(p: f64) -> Self {
        ReadoutError {
            p10: Rate::Uniform(p),
            p01: Rate::Uniform(p),
        }
    }

    /// Probability that qubit `q`'s reported bit differs from its true value,
    /// given the true bit `true_bit` (0 or 1).
    pub fn flip_prob(&self, q: usize, true_bit: u8) -> f64 {
        if true_bit == 0 {
            self.p10.at(q)
        } else {
            self.p01.at(q)
        }
    }

    pub fn is_zero(&self) -> bool {
        self.p10.is_zero() && self.p01.is_zero()
    }

    /// Parse the `readout` key: either a scalar/array (symmetric, sets both
    /// `p10` and `p01`) or an array of `{p10, p01}` objects (per-qubit
    /// asymmetric).
    fn from_json(v: &Value) -> Result<ReadoutError, String> {
        // Array of per-qubit objects → asymmetric.
        if let Some(arr) = v.as_array() {
            if arr.iter().any(|e| e.is_object()) {
                let mut p10 = Vec::with_capacity(arr.len());
                let mut p01 = Vec::with_capacity(arr.len());
                for e in arr {
                    let o = e.as_object().ok_or_else(|| {
                        "readout array must contain {p10, p01} objects".to_string()
                    })?;
                    for k in o.keys() {
                        if !matches!(k.as_str(), "p10" | "p01") {
                            return Err(format!(
                                "unknown readout key '{k}' (recognised: p10, p01)"
                            ));
                        }
                    }
                    // A present-but-non-numeric p10/p01 is an error, not a
                    // silent 0 (a dropped channel is the bug class this guards).
                    let entry = |k: &str| -> Result<f64, String> {
                        match o.get(k) {
                            None => Ok(0.0),
                            Some(x) => x
                                .as_f64()
                                .ok_or_else(|| format!("readout \"{k}\" must be a number")),
                        }
                    };
                    p10.push(entry("p10")?);
                    p01.push(entry("p01")?);
                }
                return Ok(ReadoutError {
                    p10: Rate::PerQubit(p10),
                    p01: Rate::PerQubit(p01),
                });
            }
        }
        // Scalar or plain array → symmetric.
        let r = Rate::from_json(v)?;
        Ok(ReadoutError {
            p10: r.clone(),
            p01: r,
        })
    }
}

/// A per-gate + readout noise model shared by every noise-capable backend.
///
/// Every field defaults to zero, so `NoiseModel::default()` is the exact,
/// noise-free machine. Populate scalars for an idealized uniform device, or
/// per-qubit arrays for a calibrated one. See the module docs for the schema.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NoiseModel {
    /// Depolarizing error after each gate (arity-aware).
    pub depolarizing: Depolarizing,
    /// Amplitude-damping rate γ per qubit (T1 relaxation: |1⟩ → |0⟩).
    pub amplitude_damping: Rate,
    /// Phase-damping rate per qubit (T2 dephasing).
    pub phase_damping: Rate,
    /// Explicit Pauli channel per qubit (in addition to `depolarizing`).
    pub pauli: Option<PauliChannel>,
    /// Measurement/readout error per qubit (possibly asymmetric).
    pub readout: ReadoutError,
}

impl NoiseModel {
    /// True when no channel is configured — the backend then reduces to exact,
    /// noise-free simulation.
    pub fn noiseless(&self) -> bool {
        self.depolarizing.is_zero()
            && self.amplitude_damping.is_zero()
            && self.phase_damping.is_zero()
            && self.pauli.as_ref().map(PauliChannel::is_zero).unwrap_or(true)
            && self.readout.is_zero()
    }

    /// True when a channel acts *during* circuit evolution (anything but pure
    /// readout error). These channels are stochastic per gate, so a faithful
    /// shot distribution needs one independent trajectory per shot rather than
    /// many samples of one post-trajectory state.
    pub fn has_gate_channel(&self) -> bool {
        !self.depolarizing.is_zero()
            || !self.amplitude_damping.is_zero()
            || !self.phase_damping.is_zero()
            || self.pauli.as_ref().map(|p| !p.is_zero()).unwrap_or(false)
    }

    /// Parse a `--noise '{...}'` JSON string. Accepts both the scalar
    /// (idealized) and per-qubit (calibrated) forms; rejects unknown keys.
    pub fn from_json(s: &str) -> Result<NoiseModel, String> {
        let v: Value = serde_json::from_str(s).map_err(|e| format!("invalid --noise JSON: {e}"))?;
        let obj = v.as_object().ok_or_else(|| {
            "--noise must be a JSON object, e.g. '{\"readout_flip\":0.02}'".to_string()
        })?;
        for k in obj.keys() {
            if !NOISE_KEYS.contains(&k.as_str()) {
                return Err(format!(
                    "unknown --noise key '{k}' (recognised: {})",
                    NOISE_KEYS.join(", ")
                ));
            }
        }
        if obj.contains_key("readout") && obj.contains_key("readout_flip") {
            return Err("give either \"readout\" or \"readout_flip\", not both".to_string());
        }

        let rate = |k: &str| -> Result<Rate, String> {
            obj.get(k).map(Rate::from_json).transpose().map(Option::unwrap_or_default)
        };

        let depolarizing = obj
            .get("depolarizing")
            .map(Depolarizing::from_json)
            .transpose()?
            .unwrap_or_default();
        let pauli = obj.get("pauli").map(PauliChannel::from_json).transpose()?;
        let readout = if let Some(r) = obj.get("readout") {
            ReadoutError::from_json(r)?
        } else if let Some(r) = obj.get("readout_flip") {
            let rate = Rate::from_json(r)?;
            ReadoutError {
                p10: rate.clone(),
                p01: rate,
            }
        } else {
            ReadoutError::default()
        };

        Ok(NoiseModel {
            depolarizing,
            amplitude_damping: rate("amplitude_damping")?,
            phase_damping: rate("phase_damping")?,
            pauli,
            readout,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_form_is_uniform() {
        let m = NoiseModel::from_json(r#"{"amplitude_damping":0.1,"readout_flip":0.02}"#).unwrap();
        assert_eq!(m.amplitude_damping.at(0), 0.1);
        assert_eq!(m.amplitude_damping.at(5), 0.1);
        // symmetric readout
        assert_eq!(m.readout.flip_prob(3, 0), 0.02);
        assert_eq!(m.readout.flip_prob(3, 1), 0.02);
        assert!(m.has_gate_channel());
    }

    #[test]
    fn per_qubit_arrays() {
        let m = NoiseModel::from_json(r#"{"amplitude_damping":[0.004,0.006,0.005]}"#).unwrap();
        assert_eq!(m.amplitude_damping.at(0), 0.004);
        assert_eq!(m.amplitude_damping.at(2), 0.005);
        // past the end → 0
        assert_eq!(m.amplitude_damping.at(9), 0.0);
    }

    #[test]
    fn depolarizing_arity_split() {
        let m = NoiseModel::from_json(r#"{"depolarizing":{"1q":0.001,"2q":0.012}}"#).unwrap();
        assert_eq!(m.depolarizing.at(0, 1), 0.001);
        assert_eq!(m.depolarizing.at(0, 2), 0.012);
        // bare scalar → both arities equal
        let m2 = NoiseModel::from_json(r#"{"depolarizing":0.005}"#).unwrap();
        assert_eq!(m2.depolarizing.at(0, 1), 0.005);
        assert_eq!(m2.depolarizing.at(0, 2), 0.005);
    }

    #[test]
    fn asymmetric_readout() {
        let m =
            NoiseModel::from_json(r#"{"readout":[{"p10":0.02,"p01":0.03},{"p10":0.01,"p01":0.05}]}"#)
                .unwrap();
        assert_eq!(m.readout.flip_prob(0, 0), 0.02);
        assert_eq!(m.readout.flip_prob(0, 1), 0.03);
        assert_eq!(m.readout.flip_prob(1, 1), 0.05);
        // readout-only model has no gate channel
        assert!(!m.has_gate_channel());
        assert!(!m.noiseless());
    }

    #[test]
    fn unknown_keys_rejected() {
        assert!(NoiseModel::from_json(r#"{"readout":0.5,"readout_flip":0.5}"#).is_err());
        assert!(NoiseModel::from_json(r#"{"reado":0.5}"#).is_err());
        assert!(NoiseModel::from_json(r#"{"pauli":{"W":0.5}}"#).is_err());
        assert!(NoiseModel::from_json(r#"{"depolarizing":{"3q":0.5}}"#).is_err());
    }

    #[test]
    fn malformed_values_rejected_not_coerced() {
        // A present-but-non-numeric rate must error, never silently become 0.
        assert!(NoiseModel::from_json(r#"{"amplitude_damping":"oops"}"#).is_err());
        assert!(NoiseModel::from_json(r#"{"amplitude_damping":[0.1,"x"]}"#).is_err());
        assert!(NoiseModel::from_json(r#"{"readout":[{"p10":"nope"}]}"#).is_err());
        assert!(NoiseModel::from_json(r#"{"depolarizing":{"1q":"bad"}}"#).is_err());
    }

    #[test]
    fn default_is_noiseless() {
        assert!(NoiseModel::default().noiseless());
        assert!(!NoiseModel::default().has_gate_channel());
    }
}
