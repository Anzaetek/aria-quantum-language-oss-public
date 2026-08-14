//! SQLite-backed circuit and function registry.

use std::sync::Mutex;

use rusqlite::{params, Connection};
use uuid::Uuid;

use omega_core::error::{OmegaError, Result};
use omega_parser::lower_to_ir;

/// Map a rusqlite error from a single-row lambda lookup into an OmegaError.
/// `QueryReturnedNoRows` becomes a clean `NotFound("lambda <id>")` instead
/// of leaking the rusqlite internal phrasing.
fn lambda_lookup_err(e: rusqlite::Error, id: &str) -> OmegaError {
    if matches!(e, rusqlite::Error::QueryReturnedNoRows) {
        OmegaError::NotFound(format!("lambda {id}"))
    } else {
        OmegaError::Backend(e.to_string())
    }
}

pub struct Registry {
    conn: Mutex<Connection>,
}

impl Registry {
    /// Access the underlying connection (caller must lock).
    /// For auth store operations that need direct DB access.
    pub fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CircuitEntry {
    pub id: String,
    pub source_format: String,
    pub num_qubits: u32,
    pub num_params: u32,
    pub circuit_type: String,
    pub created_at: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FunctionEntry {
    pub id: String,
    pub circuit_id: String,
    pub name: String,
    pub default_shots: u32,
    pub created_at: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct InvocationEntry {
    pub id: String,
    pub function_id: String,
    pub status: String,
    pub result_json: Option<String>,
    pub created_at: String,
}

/// Metadata view of a registered "lambda function" — a WASM module that
/// drives its own quantum-classical optimisation loop. The `wasm_bytes`
/// blob is intentionally omitted from the serialised form; fetch it
/// separately via `get_lambda_wasm` before invoking.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LambdaEntry {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub default_qasm: Option<String>,
    pub default_observable: Option<String>,
    pub default_input: Option<String>,
    pub fuel: i64,
    pub wasm_size: i64,
    pub created_at: String,
}

impl Registry {
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path).map_err(|e| OmegaError::Backend(e.to_string()))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS circuits (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_format TEXT NOT NULL,
                num_qubits INTEGER NOT NULL,
                num_params INTEGER NOT NULL,
                circuit_type TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS functions (
                id TEXT PRIMARY KEY,
                circuit_id TEXT NOT NULL REFERENCES circuits(id),
                name TEXT NOT NULL,
                default_shots INTEGER NOT NULL DEFAULT 1024,
                wasm_module BLOB,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS invocations (
                id TEXT PRIMARY KEY,
                function_id TEXT NOT NULL REFERENCES functions(id),
                status TEXT NOT NULL DEFAULT 'pending',
                params_json TEXT,
                result_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS lambdas (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                wasm_bytes BLOB NOT NULL,
                default_qasm TEXT,
                default_observable TEXT,
                default_input TEXT,
                fuel INTEGER NOT NULL DEFAULT 100000000000,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );",
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    // ---- Circuits ----

    pub fn register_circuit(&self, source: &str) -> Result<CircuitEntry> {
        // Parse to validate and extract metadata
        let ir = lower_to_ir(source).map_err(OmegaError::Parse)?;
        let id = Uuid::new_v4().to_string();
        let source_format = if source.trim_start().starts_with("OPENQASM") {
            "qasm2"
        } else {
            "opticqasm"
        };
        let circuit_type = match ir.circuit_type {
            omega_core::circuit::CircuitType::GateBased => "gate_based",
            omega_core::circuit::CircuitType::Photonic => "photonic",
        };

        self.conn.lock().unwrap()
            .execute(
                "INSERT INTO circuits (id, source, source_format, num_qubits, num_params, circuit_type) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![id, source, source_format, ir.num_qubits, ir.num_free_symbols() as u32, circuit_type],
            )
            .map_err(|e| OmegaError::Backend(e.to_string()))?;

        Ok(CircuitEntry {
            id,
            source_format: source_format.to_string(),
            num_qubits: ir.num_qubits,
            num_params: ir.num_free_symbols() as u32,
            circuit_type: circuit_type.to_string(),
            created_at: "now".to_string(),
        })
    }

    pub fn get_circuit(&self, id: &str) -> Result<CircuitEntry> {
        self.conn.lock().unwrap()
            .query_row(
                "SELECT id, source_format, num_qubits, num_params, circuit_type, created_at FROM circuits WHERE id = ?1",
                params![id],
                |row| {
                    Ok(CircuitEntry {
                        id: row.get(0)?,
                        source_format: row.get(1)?,
                        num_qubits: row.get(2)?,
                        num_params: row.get(3)?,
                        circuit_type: row.get(4)?,
                        created_at: row.get(5)?,
                    })
                },
            )
            .map_err(|e| OmegaError::Backend(e.to_string()))
    }

    pub fn get_circuit_source(&self, id: &str) -> Result<String> {
        self.conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT source FROM circuits WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| OmegaError::Backend(e.to_string()))
    }

    pub fn list_circuits(&self) -> Result<Vec<CircuitEntry>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id, source_format, num_qubits, num_params, circuit_type, created_at FROM circuits ORDER BY created_at DESC")
            .map_err(|e| OmegaError::Backend(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                Ok(CircuitEntry {
                    id: row.get(0)?,
                    source_format: row.get(1)?,
                    num_qubits: row.get(2)?,
                    num_params: row.get(3)?,
                    circuit_type: row.get(4)?,
                    created_at: row.get(5)?,
                })
            })
            .map_err(|e| OmegaError::Backend(e.to_string()))?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| OmegaError::Backend(e.to_string()))
    }

    pub fn delete_circuit(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let affected = conn
            .execute("DELETE FROM circuits WHERE id = ?1", params![id])
            .map_err(|e| OmegaError::Backend(e.to_string()))?;
        Ok(affected > 0)
    }

    // ---- Functions ----

    pub fn register_function(
        &self,
        circuit_id: &str,
        name: &str,
        default_shots: u32,
    ) -> Result<FunctionEntry> {
        // Verify circuit exists
        self.get_circuit(circuit_id)?;

        let id = Uuid::new_v4().to_string();
        self.conn.lock().unwrap()
            .execute(
                "INSERT INTO functions (id, circuit_id, name, default_shots) VALUES (?1, ?2, ?3, ?4)",
                params![id, circuit_id, name, default_shots],
            )
            .map_err(|e| OmegaError::Backend(e.to_string()))?;

        Ok(FunctionEntry {
            id,
            circuit_id: circuit_id.to_string(),
            name: name.to_string(),
            default_shots,
            created_at: "now".to_string(),
        })
    }

    pub fn get_function(&self, id: &str) -> Result<FunctionEntry> {
        self.conn.lock().unwrap()
            .query_row(
                "SELECT id, circuit_id, name, default_shots, created_at FROM functions WHERE id = ?1",
                params![id],
                |row| {
                    Ok(FunctionEntry {
                        id: row.get(0)?,
                        circuit_id: row.get(1)?,
                        name: row.get(2)?,
                        default_shots: row.get(3)?,
                        created_at: row.get(4)?,
                    })
                },
            )
            .map_err(|e| OmegaError::Backend(e.to_string()))
    }

    pub fn list_functions(&self) -> Result<Vec<FunctionEntry>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id, circuit_id, name, default_shots, created_at FROM functions ORDER BY created_at DESC")
            .map_err(|e| OmegaError::Backend(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                Ok(FunctionEntry {
                    id: row.get(0)?,
                    circuit_id: row.get(1)?,
                    name: row.get(2)?,
                    default_shots: row.get(3)?,
                    created_at: row.get(4)?,
                })
            })
            .map_err(|e| OmegaError::Backend(e.to_string()))?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| OmegaError::Backend(e.to_string()))
    }

    // ---- Invocations ----

    pub fn create_invocation(
        &self,
        function_id: &str,
        params_json: &str,
    ) -> Result<InvocationEntry> {
        let id = Uuid::new_v4().to_string();
        self.conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO invocations (id, function_id, params_json) VALUES (?1, ?2, ?3)",
                params![id, function_id, params_json],
            )
            .map_err(|e| OmegaError::Backend(e.to_string()))?;

        Ok(InvocationEntry {
            id,
            function_id: function_id.to_string(),
            status: "pending".to_string(),
            result_json: None,
            created_at: "now".to_string(),
        })
    }

    pub fn complete_invocation(&self, id: &str, result_json: &str) -> Result<()> {
        self.conn
            .lock()
            .unwrap()
            .execute(
                "UPDATE invocations SET status = 'completed', result_json = ?2 WHERE id = ?1",
                params![id, result_json],
            )
            .map_err(|e| OmegaError::Backend(e.to_string()))?;
        Ok(())
    }

    pub fn fail_invocation(&self, id: &str, error: &str) -> Result<()> {
        self.conn
            .lock()
            .unwrap()
            .execute(
                "UPDATE invocations SET status = 'failed', result_json = ?2 WHERE id = ?1",
                params![id, error],
            )
            .map_err(|e| OmegaError::Backend(e.to_string()))?;
        Ok(())
    }

    pub fn get_invocation(&self, id: &str) -> Result<InvocationEntry> {
        self.conn.lock().unwrap()
            .query_row(
                "SELECT id, function_id, status, result_json, created_at FROM invocations WHERE id = ?1",
                params![id],
                |row| {
                    Ok(InvocationEntry {
                        id: row.get(0)?,
                        function_id: row.get(1)?,
                        status: row.get(2)?,
                        result_json: row.get(3)?,
                        created_at: row.get(4)?,
                    })
                },
            )
            .map_err(|e| OmegaError::Backend(e.to_string()))
    }

    /// Execute a circuit directly (synchronous, for simple invocations).
    pub fn execute_circuit(
        &self,
        circuit_id: &str,
        params: &[f64],
        shots: u32,
        seed: Option<u64>,
    ) -> Result<serde_json::Value> {
        let source = self.get_circuit_source(circuit_id)?;
        let ir = lower_to_ir(&source).map_err(OmegaError::Parse)?;

        let mut binding = omega_core::params::ParameterBinding::new();
        let mut sym_ids: Vec<u32> = ir.symbols.keys().copied().collect();
        sym_ids.sort();
        for (i, &sym_id) in sym_ids.iter().enumerate() {
            binding.bind(sym_id, if i < params.len() { params[i] } else { 0.0 });
        }

        let config = omega_core::executor::ExecConfig {
            shots: if shots == 0 { None } else { Some(shots) },
            seed,
            ..Default::default()
        };

        // Admission control. This path lowers caller-registered Aria source and
        // executes it directly, so without a reservation it was a second,
        // unguarded door onto the same statevector allocation the /v1/quantum
        // routes are governed on.
        let kind = match ir.circuit_type {
            omega_core::circuit::CircuitType::GateBased => {
                crate::worker::CostKind::DenseStatevector
            }
            omega_core::circuit::CircuitType::Photonic => crate::worker::CostKind::Photonic,
        };
        let base = crate::worker::JobShape::new(ir.num_qubits, kind);
        // shots == 0 means analytic here, which returns the full statevector.
        let shape = if shots == 0 {
            base.returning_statevector()
        } else {
            base.with_shots()
        };
        let _reservation = crate::worker::governor()
            .admit(&shape)
            .map_err(|rej| OmegaError::Backend(rej.message()))?;

        use omega_core::executor::Backend;
        let result = match ir.circuit_type {
            omega_core::circuit::CircuitType::GateBased => {
                omega_backend_statevector::StatevectorBackend::new()
                    .execute(&ir, &binding, &config)?
            }
            omega_core::circuit::CircuitType::Photonic => {
                omega_backend_photonics::PhotonicsBackend::new().execute(&ir, &binding, &config)?
            }
        };

        // Width to RENDER a counts key at. NOT `ir.num_qubits`: in collapse mode
        // the key is packed from the CLASSICAL register, so a 1024-qubit
        // circuit measuring two qubits produces a 2-bit outcome. This site kept
        // padding to `num_qubits` after the CLI renderers were fixed, so the
        // same run reported `"11"` from the CLI and 1024 characters over HTTP.
        let counts_width = omega_core::executor::counts_outcome_width(
            &ir,
            omega_core::executor::counts_keyed_on_creg(
                &ir,
                config.mid_circuit_mode == omega_core::executor::MidCircuitMode::Collapse,
            ),
        );
        debug_assert!(
            counts_width <= omega_core::executor::MAX_COUNTS_QUBITS,
            "counts_width {counts_width} exceeds the key width; the backend \
             should have refused this run before it reached rendering"
        );

        // Serialize result to JSON
        match result {
            omega_core::executor::ExecResult::Counts(counts) => {
                let map: std::collections::HashMap<String, u32> = counts
                    .into_iter()
                    .map(|(bs, ct)| {
                        (
                            format!("{:0>width$b}", bs, width = counts_width),
                            ct,
                        )
                    })
                    .collect();
                Ok(serde_json::json!({ "type": "counts", "counts": map }))
            }
            omega_core::executor::ExecResult::Statevector(sv) => {
                let amps: Vec<[f64; 2]> = sv.iter().map(|c| [c.re, c.im]).collect();
                Ok(serde_json::json!({ "type": "statevector", "amplitudes": amps }))
            }
            omega_core::executor::ExecResult::Probabilities(probs) => {
                Ok(serde_json::json!({ "type": "probabilities", "probabilities": probs }))
            }
        }
    }

    // ---- Lambdas (WASM-driven optimisation functions) ----

    /// Register a new lambda. `wasm_bytes` must be a valid WASM module —
    /// validation happens at invoke time so registration is cheap.
    #[allow(clippy::too_many_arguments)]
    pub fn register_lambda(
        &self,
        name: &str,
        description: Option<&str>,
        wasm_bytes: &[u8],
        default_qasm: Option<&str>,
        default_observable: Option<&str>,
        default_input: Option<&str>,
        fuel: i64,
    ) -> Result<LambdaEntry> {
        if wasm_bytes.is_empty() {
            return Err(OmegaError::Backend("wasm_bytes is empty".into()));
        }
        let id = Uuid::new_v4().to_string();
        self.conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO lambdas (id, name, description, wasm_bytes, default_qasm, \
                 default_observable, default_input, fuel) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    id,
                    name,
                    description,
                    wasm_bytes,
                    default_qasm,
                    default_observable,
                    default_input,
                    fuel,
                ],
            )
            .map_err(|e| OmegaError::Backend(e.to_string()))?;
        Ok(LambdaEntry {
            id,
            name: name.to_string(),
            description: description.map(|s| s.to_string()),
            default_qasm: default_qasm.map(|s| s.to_string()),
            default_observable: default_observable.map(|s| s.to_string()),
            default_input: default_input.map(|s| s.to_string()),
            fuel,
            wasm_size: wasm_bytes.len() as i64,
            created_at: "now".to_string(),
        })
    }

    pub fn get_lambda(&self, id: &str) -> Result<LambdaEntry> {
        self.conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT id, name, description, default_qasm, default_observable, \
                 default_input, fuel, length(wasm_bytes), created_at \
                 FROM lambdas WHERE id = ?1",
                params![id],
                |row| {
                    Ok(LambdaEntry {
                        id: row.get(0)?,
                        name: row.get(1)?,
                        description: row.get(2)?,
                        default_qasm: row.get(3)?,
                        default_observable: row.get(4)?,
                        default_input: row.get(5)?,
                        fuel: row.get(6)?,
                        wasm_size: row.get(7)?,
                        created_at: row.get(8)?,
                    })
                },
            )
            .map_err(|e| lambda_lookup_err(e, id))
    }

    pub fn get_lambda_wasm(&self, id: &str) -> Result<Vec<u8>> {
        self.conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT wasm_bytes FROM lambdas WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| lambda_lookup_err(e, id))
    }

    pub fn list_lambdas(&self) -> Result<Vec<LambdaEntry>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, name, description, default_qasm, default_observable, \
                 default_input, fuel, length(wasm_bytes), created_at \
                 FROM lambdas ORDER BY created_at DESC",
            )
            .map_err(|e| OmegaError::Backend(e.to_string()))?;
        let rows = stmt
            .query_map([], |row| {
                Ok(LambdaEntry {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    description: row.get(2)?,
                    default_qasm: row.get(3)?,
                    default_observable: row.get(4)?,
                    default_input: row.get(5)?,
                    fuel: row.get(6)?,
                    wasm_size: row.get(7)?,
                    created_at: row.get(8)?,
                })
            })
            .map_err(|e| OmegaError::Backend(e.to_string()))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| OmegaError::Backend(e.to_string()))
    }

    pub fn delete_lambda(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let affected = conn
            .execute("DELETE FROM lambdas WHERE id = ?1", params![id])
            .map_err(|e| OmegaError::Backend(e.to_string()))?;
        Ok(affected > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_registry() -> Registry {
        // SQLite ":memory:" gives us a per-test isolated DB.
        Registry::new(":memory:").expect("open in-memory registry")
    }

    #[test]
    fn lambda_register_get_list_delete_roundtrip() {
        let r = fresh_registry();
        // Use a tiny non-empty byte slice as a placeholder for the WASM.
        let wasm = b"\x00asm\x01\x00\x00\x00";
        let entry = r
            .register_lambda(
                "demo",
                Some("a description"),
                wasm,
                Some("OPENQASM 2.0; qreg q[1];"),
                Some("0.5*Z0"),
                Some("{\"max_iters\": 10}"),
                42_000,
            )
            .unwrap();
        assert_eq!(entry.name, "demo");
        assert_eq!(entry.wasm_size, wasm.len() as i64);
        assert_eq!(entry.fuel, 42_000);

        let fetched = r.get_lambda(&entry.id).unwrap();
        assert_eq!(fetched.name, "demo");
        assert_eq!(fetched.default_observable.as_deref(), Some("0.5*Z0"));
        assert_eq!(fetched.wasm_size, wasm.len() as i64);

        let blob = r.get_lambda_wasm(&entry.id).unwrap();
        assert_eq!(blob, wasm);

        let listed = r.list_lambdas().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, entry.id);

        assert!(r.delete_lambda(&entry.id).unwrap());
        assert!(r.list_lambdas().unwrap().is_empty());
        assert!(!r.delete_lambda(&entry.id).unwrap()); // already gone
    }

    #[test]
    fn lambda_rejects_empty_wasm_bytes() {
        let r = fresh_registry();
        let err = r
            .register_lambda("x", None, &[], None, None, None, 1)
            .unwrap_err();
        assert!(err.to_string().contains("empty"), "{err}");
    }

    #[test]
    fn lambda_unique_name_constraint_fires() {
        let r = fresh_registry();
        let bytes = b"\x00asm\x01\x00\x00\x00";
        r.register_lambda("dup", None, bytes, None, None, None, 1)
            .unwrap();
        let err = r
            .register_lambda("dup", None, bytes, None, None, None, 1)
            .unwrap_err();
        // SQLite UNIQUE-constraint surface in the OmegaError::Backend message.
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("unique") || msg.contains("constraint"),
            "expected UNIQUE-constraint error, got: {msg}"
        );
    }

    /// End-to-end: register a real qaoa_qubo.wasm in the registry, pull
    /// the bytes back, stage them on HostState, run the guest. Mirrors
    /// what `invoke_lambda` does HTTP-side, without booting the axum
    /// server. Skipped when the guest WASM hasn't been built.
    #[test]
    fn lambda_register_then_run_qaoa_qubo_end_to_end() {
        use omega_wasm_runtime::host::HostState;
        use omega_wasm_runtime::WasmRunner;

        let wasm_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/wasm-guests/qaoa_qubo/target/wasm32-wasip1/release/qaoa_qubo.wasm");
        if !wasm_path.exists() {
            eprintln!(
                "Skipping: qaoa_qubo.wasm not built at {}",
                wasm_path.display()
            );
            return;
        }
        let wasm_bytes = std::fs::read(&wasm_path).unwrap();

        let r = fresh_registry();
        let entry = r
            .register_lambda(
                "qaoa_qubo_spsa",
                Some("end-to-end smoke"),
                &wasm_bytes,
                None,
                None,
                Some("{\"depth\":2,\"max_iters\":150}"),
                100_000_000_000,
            )
            .unwrap();

        // Pull the WASM bytes back through the SQLite blob roundtrip.
        let stored = r.get_lambda_wasm(&entry.id).unwrap();
        assert_eq!(stored, wasm_bytes);

        // Mirrors invoke_lambda: stage input + run.
        let runtime_input = r#"{
            "qubo": "{\"n\":3,\"Q\":[[0,0,-2],[1,1,-2],[2,2,-2],[0,1,2],[1,2,2],[0,2,2]]}",
            "depth": 2,
            "max_iters": 150,
            "seed": 7
        }"#;
        let mut host = HostState::new();
        host.set_input(runtime_input.as_bytes().to_vec());

        let runner = WasmRunner::new(host).unwrap();
        let result = runner.run(&stored, 100_000_000_000).unwrap();
        let res = result.expect("guest never reported a result");
        assert_eq!(res.optimal_params.len(), 4, "depth=2 → 4 free params");
        assert!(
            res.optimal_value < -1.5,
            "lambda end-to-end did not converge: {}",
            res.optimal_value
        );

        // Cleanup.
        assert!(r.delete_lambda(&entry.id).unwrap());
    }

    // ----- Circuits -----

    const BELL_QASM: &str = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0], q[1];\n";

    #[test]
    fn circuit_register_get_list_delete_roundtrip() {
        let r = fresh_registry();
        let entry = r.register_circuit(BELL_QASM).expect("register");
        assert_eq!(entry.num_qubits, 2);
        assert_eq!(entry.num_params, 0);
        assert_eq!(entry.source_format, "qasm2");
        assert_eq!(entry.circuit_type, "gate_based");

        let fetched = r.get_circuit(&entry.id).expect("get");
        assert_eq!(fetched.id, entry.id);
        assert_eq!(fetched.num_qubits, 2);

        let src = r.get_circuit_source(&entry.id).expect("get_source");
        assert_eq!(src, BELL_QASM);

        let listed = r.list_circuits().expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, entry.id);

        assert!(r.delete_circuit(&entry.id).expect("delete"));
        assert!(!r.delete_circuit(&entry.id).expect("re-delete"));
        assert!(r.list_circuits().unwrap().is_empty());
    }

    #[test]
    fn circuit_get_unknown_id_errors() {
        let r = fresh_registry();
        let err = r.get_circuit("no-such-id").expect_err("unknown id");
        assert!(format!("{err:?}").to_lowercase().contains("rows"));
    }

    #[test]
    fn circuit_register_garbage_qasm_errors() {
        let r = fresh_registry();
        let err = r
            .register_circuit("not-qasm-at-all")
            .expect_err("garbage must reject");
        // The parser rejects, surfacing as OmegaError::Parse.
        assert!(format!("{err:?}").to_lowercase().contains("parse"));
        // No row should have been written.
        assert!(r.list_circuits().unwrap().is_empty());
    }

    // ----- Functions -----

    #[test]
    fn function_register_requires_existing_circuit() {
        let r = fresh_registry();
        let err = r
            .register_function("no-such-circuit", "f", 100)
            .expect_err("missing circuit must reject");
        assert!(format!("{err:?}").to_lowercase().contains("rows"));
    }

    #[test]
    fn function_register_get_list_roundtrip() {
        let r = fresh_registry();
        let circuit = r.register_circuit(BELL_QASM).unwrap();
        let f = r
            .register_function(&circuit.id, "bell-fn", 256)
            .expect("register fn");
        assert_eq!(f.name, "bell-fn");
        assert_eq!(f.default_shots, 256);
        assert_eq!(f.circuit_id, circuit.id);

        let fetched = r.get_function(&f.id).expect("get fn");
        assert_eq!(fetched.id, f.id);
        assert_eq!(fetched.default_shots, 256);

        let listed = r.list_functions().expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, f.id);
    }

    // ----- Invocations -----

    #[test]
    fn invocation_create_complete_round_trip() {
        let r = fresh_registry();
        let circuit = r.register_circuit(BELL_QASM).unwrap();
        let f = r.register_function(&circuit.id, "f", 100).unwrap();
        let inv = r
            .create_invocation(&f.id, "{\"shots\":42}")
            .expect("create");
        assert_eq!(inv.status, "pending");
        assert!(inv.result_json.is_none());

        r.complete_invocation(&inv.id, "{\"counts\":{\"00\":21}}")
            .expect("complete");
        let fetched = r.get_invocation(&inv.id).expect("get");
        assert_eq!(fetched.status, "completed");
        assert_eq!(
            fetched.result_json.as_deref(),
            Some("{\"counts\":{\"00\":21}}")
        );
    }

    #[test]
    fn invocation_fail_writes_error_into_result_json() {
        // The fail path stores the error message in the same column the
        // happy path uses; downstream readers distinguish by the
        // `status` field.
        let r = fresh_registry();
        let circuit = r.register_circuit(BELL_QASM).unwrap();
        let f = r.register_function(&circuit.id, "f", 100).unwrap();
        let inv = r.create_invocation(&f.id, "{}").unwrap();
        r.fail_invocation(&inv.id, "backend OOM").expect("fail");
        let fetched = r.get_invocation(&inv.id).unwrap();
        assert_eq!(fetched.status, "failed");
        assert_eq!(fetched.result_json.as_deref(), Some("backend OOM"));
    }

    #[test]
    fn invocation_get_unknown_errors() {
        let r = fresh_registry();
        let err = r.get_invocation("no-such-inv").expect_err("missing");
        assert!(format!("{err:?}").to_lowercase().contains("rows"));
    }

    // ----- execute_circuit -----

    #[test]
    fn execute_circuit_bell_returns_counts_distribution() {
        let r = fresh_registry();
        let circuit = r.register_circuit(BELL_QASM).unwrap();
        let v = r.execute_circuit(&circuit.id, &[], 4096, Some(42)).unwrap();
        assert_eq!(v["type"], "counts");
        let counts = v["counts"].as_object().unwrap();
        // Bell ⇒ exactly two outcomes "00" and "11" (the QASM has no
        // measurement, so the backend reports the basis-state
        // distribution from sampling the statevector).
        let sum: u32 = counts
            .values()
            .filter_map(|c| c.as_u64().map(|x| x as u32))
            .sum();
        assert_eq!(sum, 4096, "every shot accounted for");
        let c00 = counts.get("00").and_then(|c| c.as_u64()).unwrap_or(0);
        let c11 = counts.get("11").and_then(|c| c.as_u64()).unwrap_or(0);
        assert!(c00 + c11 == 4096, "Bell only produces |00⟩ and |11⟩");
        assert!(
            (c00 as f64 / 4096.0 - 0.5).abs() < 0.05,
            "|00⟩ fraction {} outside [0.45, 0.55]",
            c00 as f64 / 4096.0
        );
    }

    #[test]
    fn execute_circuit_shots_zero_returns_statevector() {
        // Documented contract: shots=0 → exact statevector path
        // (passed as `None` to ExecConfig).
        let r = fresh_registry();
        let circuit = r
            .register_circuit("OPENQASM 2.0;\nqreg q[1];\nh q[0];\n")
            .unwrap();
        let v = r.execute_circuit(&circuit.id, &[], 0, None).unwrap();
        assert_eq!(v["type"], "statevector");
        let amps = v["amplitudes"].as_array().unwrap();
        assert_eq!(amps.len(), 2);
        // |+⟩ = (1/√2)(|0⟩ + |1⟩); both amplitudes ≈ 1/√2 real.
        let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
        let a0_re = amps[0][0].as_f64().unwrap();
        let a1_re = amps[1][0].as_f64().unwrap();
        assert!((a0_re - inv_sqrt2).abs() < 1e-12);
        assert!((a1_re - inv_sqrt2).abs() < 1e-12);
    }

    #[test]
    fn execute_circuit_unknown_id_errors() {
        let r = fresh_registry();
        let err = r
            .execute_circuit("no-such-id", &[], 100, None)
            .expect_err("unknown id");
        assert!(format!("{err:?}").to_lowercase().contains("rows"));
    }

    #[test]
    fn execute_circuit_pads_missing_parameters_with_zero() {
        // The execute path mirrors WasmRunner: extra symbols beyond
        // the params length default to 0.0. Using Ry(theta)|0⟩ at
        // θ=0 gives ⟨Z⟩=1, so sampling returns only |0⟩.
        let r = fresh_registry();
        let src = "OPENQASM 2.0;\ngate ry(theta) q { ry(theta) q; }\nqreg q[1];\nry(0) q[0];\n";
        let circuit = r.register_circuit(src).unwrap();
        let v = r.execute_circuit(&circuit.id, &[], 1024, Some(7)).unwrap();
        // Concrete-angle gate with no free symbols — returns counts.
        assert_eq!(v["type"], "counts");
        let counts = v["counts"].as_object().unwrap();
        // Ry(0)|0⟩ = |0⟩ exactly → one outcome.
        assert_eq!(counts.len(), 1);
        let c0 = counts.get("0").and_then(|c| c.as_u64()).unwrap();
        assert_eq!(c0, 1024);
    }
}
