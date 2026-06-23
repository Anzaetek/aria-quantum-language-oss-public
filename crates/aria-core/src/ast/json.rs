//! JSON circuit serialization via serde.

use super::nodes::Circuit;

/// Serialize a Circuit to JSON.
pub fn to_json(circuit: &Circuit) -> Result<String, String> {
    serde_json::to_string_pretty(circuit).map_err(|e| e.to_string())
}

/// Deserialize a Circuit from JSON.
pub fn from_json(json: &str) -> Result<Circuit, String> {
    serde_json::from_str(json).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::CircuitBuilder;

    #[test]
    fn test_json_roundtrip() {
        let circ = CircuitBuilder::new("bell", 2, 2)
            .h(0)
            .cx(0, 1)
            .measure_all()
            .build();

        let json = to_json(&circ).unwrap();
        assert!(json.contains("\"name\": \"bell\""));
        assert!(json.contains("\"H\""));

        let reimported = from_json(&json).unwrap();
        assert_eq!(reimported.n_qubits(), 2);
        assert_eq!(reimported.gate_count(), 2);
        assert_eq!(reimported.instructions.len(), circ.instructions.len());
    }

    #[test]
    fn test_json_with_annotations() {
        use crate::ast::{Annotation, Property};
        let mut circ = CircuitBuilder::new("test", 1, 0).h(0).build();
        circ.annotate(Annotation::Assert(Property::Unitary));

        let json = to_json(&circ).unwrap();
        assert!(json.contains("Unitary"));

        let reimported = from_json(&json).unwrap();
        assert_eq!(reimported.annotations.len(), 1);
    }
}
