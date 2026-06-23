//! QAOA-on-QUBO with circuit assembly *and* optimizer running entirely in WASM.
//!
//! Sibling of `qaoa_qubo`, but does NOT call `omega_qaoa_from_qubo` — that
//! host helper builds the QAOA circuit + observable on the host side. Here
//! the guest does everything itself:
//!
//!   1. Reads the QUBO JSON from `omega_input_*`.
//!   2. Converts QUBO → Ising in WASM.
//!   3. Emits an OPENQASM 2.0 source for the QAOA(depth) circuit
//!      (initial Hadamards, p layers of cost + RX mixer, parametrised by
//!      `gamma_0..gamma_{p-1}` and `beta_0..beta_{p-1}`).
//!   4. Registers it via `omega_register_qasm`.
//!   5. Builds the Ising Pauli-sum observable string (offset·I + Σh_i Z_i +
//!      Σ J_ij Z_i Z_j) and registers via `omega_register_observable_str`.
//!   6. Runs SPSA against `(circuit_id, observable_id)` for `max_iters`
//!      iterations, two `omega_execute` calls per iteration.
//!   7. Reports back via `omega_report_result`.
//!
//! Net: the only host primitives are parser (`omega_register_qasm`),
//! observable parser (`omega_register_observable_str`), and circuit
//! evaluation (`omega_execute`). The QAOA circuit assembly, the observable
//! construction, and the optimization loop are all pure WASM. The same
//! guest serves any QUBO on up to ~20 qubits with no host redeploy.
//!
//! Input payload (same shape as `qaoa_qubo`):
//! {
//!   "qubo":      "<QUBO JSON>"  (required, e.g. {\"n\":4,\"Q\":[[i,j,c],...]}),
//!   "depth":     <int, default 2>,
//!   "max_iters": <int, default 200>,
//!   "a":         <f64, default 0.2>,
//!   "c":         <f64, default 0.1>,
//!   "alpha":     <f64, default 0.602>,
//!   "gamma":     <f64, default 0.101>,
//!   "A":         <f64, default 5.0>,
//!   "init":      [f64,...]                 (optional),
//!   "seed":      <int, default 0xC0FFEE>,
//!   "log_every": <int, default 25>
//! }

use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt::Write as _;

extern "C" {
    fn omega_execute(circuit_id: i32, params_ptr: *const f64, num_params: i32, observable_id: i32) -> f64;
    fn omega_register_qasm(src_ptr: *const u8, src_len: i32) -> i32;
    fn omega_register_observable_str(spec_ptr: *const u8, spec_len: i32) -> i32;
    fn omega_input_len() -> i32;
    fn omega_input_read(out_ptr: *mut u8, max_len: i32) -> i32;
    fn omega_report_progress(iteration: i32, value_bits: i64);
    fn omega_report_result(params_ptr: *const f64, num_params: i32, value_bits: i64, iterations: i32);
}

#[derive(Default, Debug, Deserialize)]
#[serde(default)]
struct Config {
    qubo: Option<String>,
    depth: Option<i32>,
    max_iters: Option<i32>,
    a: Option<f64>,
    c: Option<f64>,
    alpha: Option<f64>,
    gamma: Option<f64>,
    #[serde(rename = "A")]
    big_a: Option<f64>,
    init: Option<Vec<f64>>,
    seed: Option<u64>,
    log_every: Option<i32>,
}

#[derive(Deserialize)]
struct QuboPayload {
    n: usize,
    #[serde(rename = "Q")]
    q: Vec<(usize, usize, f64)>,
}

/// Compact upper-triangular QUBO in WASM memory.
struct Qubo {
    n: usize,
    /// `(i, j) -> coeff` with `i <= j`. Diagonal entries are linear.
    entries: BTreeMap<(usize, usize), f64>,
}

impl Qubo {
    fn from_json(s: &str) -> Result<Self, String> {
        let p: QuboPayload = serde_json::from_str(s).map_err(|e| format!("qubo JSON: {e}"))?;
        let mut entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for (i, j, c) in p.q {
            if i >= p.n || j >= p.n {
                return Err(format!("qubo index out of range: ({i},{j}) for n={}", p.n));
            }
            let key = if i <= j { (i, j) } else { (j, i) };
            *entries.entry(key).or_insert(0.0) += c;
        }
        Ok(Self { n: p.n, entries })
    }

    /// QUBO → Ising. Returns (couplings, fields, offset). Same convention
    /// as `omega_core::qubo::Qubo::to_ising` so the guest can be cross-
    /// checked against the host helper.
    fn to_ising(&self) -> (BTreeMap<(usize, usize), f64>, Vec<f64>, f64) {
        let mut couplings: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut fields = vec![0.0_f64; self.n];
        let mut offset = 0.0_f64;
        for (&(i, j), &q) in &self.entries {
            if i == j {
                offset += q / 2.0;
                fields[i] -= q / 2.0;
            } else {
                offset += q / 4.0;
                *couplings.entry((i, j)).or_insert(0.0) += q / 4.0;
                fields[i] -= q / 4.0;
                fields[j] -= q / 4.0;
            }
        }
        (couplings, fields, offset)
    }
}

/// Emit OPENQASM 2.0 for a depth-`p` QAOA on the given Ising. Free
/// parameters are named `gamma_0..gamma_{p-1}` then `beta_0..beta_{p-1}`,
/// in that order so the symbol-id assignment matches `[γ0,β0,γ1,β1,...]`
/// after the parser's symbol-id scheme — the SPSA loop just feeds a flat
/// `2*p`-length vector and lets the host bind by id.
fn emit_qaoa_qasm(
    n: usize,
    depth: usize,
    couplings: &BTreeMap<(usize, usize), f64>,
    fields: &[f64],
) -> String {
    let mut s = String::with_capacity(256 + depth * (couplings.len() + n) * 64);
    s.push_str("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n");
    let _ = writeln!(s, "qreg q[{n}];");
    // Initial superposition.
    for q in 0..n {
        let _ = writeln!(s, "h q[{q}];");
    }
    // p layers: cost (couplings + fields) then mixer.
    for k in 0..depth {
        // Couplings: 2γ_k J_ij Z_i Z_j → CNOT, RZ(2 J_ij γ_k), CNOT
        for (&(i, j), &jval) in couplings {
            if jval.abs() < 1e-15 {
                continue;
            }
            let _ = writeln!(s, "cx q[{i}], q[{j}];");
            let _ = writeln!(s, "rz({:.10} * gamma_{k}) q[{j}];", 2.0 * jval);
            let _ = writeln!(s, "cx q[{i}], q[{j}];");
        }
        // Local fields: 2γ_k h_i Z_i → RZ(2 h_i γ_k)
        for (i, &h) in fields.iter().enumerate() {
            if h.abs() < 1e-15 {
                continue;
            }
            let _ = writeln!(s, "rz({:.10} * gamma_{k}) q[{i}];", 2.0 * h);
        }
        // Mixer: RX(2 β_k) on every qubit
        for q in 0..n {
            let _ = writeln!(s, "rx(2.0 * beta_{k}) q[{q}];");
        }
    }
    s
}

/// Emit `c*I0+h0*Z0+...+J_ij*ZiZj` Pauli-sum string for the Ising H.
/// `IsingModel::to_observable` folds the offset in as an identity term,
/// matching the host convention (so the SPSA expectation IS the QUBO
/// objective directly, no double-counting).
fn emit_ising_observable(
    n: usize,
    couplings: &BTreeMap<(usize, usize), f64>,
    fields: &[f64],
    offset: f64,
) -> String {
    let mut terms: Vec<String> = Vec::new();
    if offset.abs() > 1e-15 {
        terms.push(format!("{offset:.10}*I0"));
    }
    for (i, &h) in fields.iter().enumerate() {
        if h.abs() > 1e-15 {
            terms.push(format!("{h:.10}*Z{i}"));
        }
    }
    for (&(i, j), &jval) in couplings {
        if jval.abs() > 1e-15 {
            terms.push(format!("{jval:.10}*Z{i}Z{j}"));
        }
    }
    if terms.is_empty() {
        // Degenerate "all-zero" Hamiltonian — keep parser happy with a 0·I.
        terms.push(format!("0.0*I0"));
    }
    // The parser splits on '+', so signed coefficients become "+-x.xxx".
    // Replace each leading "-" with "+-" except for the very first term.
    let mut out = String::new();
    for (idx, t) in terms.iter().enumerate() {
        if idx == 0 {
            out.push_str(t);
        } else if t.starts_with('-') {
            out.push('+');
            out.push_str(t);
        } else {
            out.push('+');
            out.push_str(t);
        }
        let _ = n; // silence the unused warning if the macro paths change
    }
    out
}

fn read_input() -> Vec<u8> {
    unsafe {
        let len = omega_input_len();
        if len <= 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; len as usize];
        let n = omega_input_read(buf.as_mut_ptr(), len);
        buf.truncate(n as usize);
        buf
    }
}

/// Tiny LCG so the SPSA Bernoulli draws are deterministic given `seed`.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn rademacher(&mut self) -> f64 {
        if (self.next_u64() >> 63) & 1 == 0 { 1.0 } else { -1.0 }
    }
}

fn evaluate(circuit_id: i32, observable_id: i32, params: &[f64]) -> f64 {
    unsafe {
        omega_execute(
            circuit_id,
            params.as_ptr(),
            params.len() as i32,
            observable_id,
        )
    }
}

fn main() {
    let raw = read_input();
    if raw.is_empty() {
        println!("[qaoa_qubo_self.wasm] missing input — supply a QUBO JSON via host input bytes");
        return;
    }
    let cfg: Config = match serde_json::from_slice(&raw) {
        Ok(c) => c,
        Err(e) => {
            println!("[qaoa_qubo_self.wasm] failed to parse input JSON: {e}");
            return;
        }
    };
    let qubo_json = match cfg.qubo.as_deref() {
        Some(s) => s,
        None => {
            println!("[qaoa_qubo_self.wasm] input is missing required field 'qubo'");
            return;
        }
    };
    let depth = cfg.depth.unwrap_or(2).max(1) as usize;
    let max_iters = cfg.max_iters.unwrap_or(200) as usize;
    let a = cfg.a.unwrap_or(0.2);
    let c = cfg.c.unwrap_or(0.1);
    let alpha = cfg.alpha.unwrap_or(0.602);
    let gamma = cfg.gamma.unwrap_or(0.101);
    let big_a = cfg.big_a.unwrap_or(5.0);
    let log_every = cfg.log_every.unwrap_or(25) as usize;
    let seed = cfg.seed.unwrap_or(0x00C0_FFEEu64);

    // ---- Step 1-2: parse QUBO, convert to Ising in WASM. ----
    let qubo = match Qubo::from_json(qubo_json) {
        Ok(q) => q,
        Err(e) => {
            println!("[qaoa_qubo_self.wasm] {e}");
            return;
        }
    };
    let (couplings, fields, offset) = qubo.to_ising();

    // ---- Step 3-4: emit + register QASM. ----
    let qasm = emit_qaoa_qasm(qubo.n, depth, &couplings, &fields);
    let cid = unsafe { omega_register_qasm(qasm.as_ptr(), qasm.len() as i32) };
    if cid <= 0 {
        println!("[qaoa_qubo_self.wasm] omega_register_qasm failed (cid={cid})");
        println!("--- guest-built QASM ---\n{qasm}\n--- end QASM ---");
        return;
    }

    // ---- Step 5: emit + register observable. ----
    let obs_str = emit_ising_observable(qubo.n, &couplings, &fields, offset);
    let oid = unsafe { omega_register_observable_str(obs_str.as_ptr(), obs_str.len() as i32) };
    if oid <= 0 {
        println!("[qaoa_qubo_self.wasm] omega_register_observable_str failed (oid={oid})");
        println!("[qaoa_qubo_self.wasm] observable was: {obs_str}");
        return;
    }

    let n_params = 2 * depth;
    println!(
        "[qaoa_qubo_self.wasm] depth={depth} qubits={} num_params={n_params} \
         couplings={} fields_nonzero={} offset={offset:.6} circuit={cid} observable={oid}",
        qubo.n,
        couplings.len(),
        fields.iter().filter(|h| h.abs() > 1e-15).count(),
    );

    // ---- Step 6: SPSA in WASM. ----
    let mut x: Vec<f64> = match cfg.init {
        Some(ref v) if !v.is_empty() => {
            let mut p = v.clone();
            p.resize(n_params, 0.5);
            p
        }
        _ => vec![0.5; n_params],
    };
    let mut x_plus = vec![0.0_f64; n_params];
    let mut x_minus = vec![0.0_f64; n_params];
    let mut delta = vec![0.0_f64; n_params];

    let mut rng = Lcg(seed);
    let mut best_cost = evaluate(cid, oid, &x);
    let mut best_params = x.clone();
    let mut best_iter: i32 = 0;
    unsafe { omega_report_progress(0, best_cost.to_bits() as i64); }
    println!("  iter    0: cost = {:.10}  (initial)", best_cost);

    for iter in 1..=max_iters {
        let ak = a / ((iter as f64) + big_a).powf(alpha);
        let ck = c / (iter as f64).powf(gamma);
        for i in 0..n_params {
            delta[i] = rng.rademacher();
            x_plus[i]  = x[i] + ck * delta[i];
            x_minus[i] = x[i] - ck * delta[i];
        }
        let f_plus  = evaluate(cid, oid, &x_plus);
        let f_minus = evaluate(cid, oid, &x_minus);
        let two_ck  = 2.0 * ck;
        for i in 0..n_params {
            let g_hat = (f_plus - f_minus) / (two_ck * delta[i]);
            x[i] -= ak * g_hat;
        }
        let cost = evaluate(cid, oid, &x);
        unsafe { omega_report_progress(iter as i32, cost.to_bits() as i64); }
        if cost < best_cost {
            best_cost = cost;
            best_params = x.clone();
            best_iter = iter as i32;
        }
        if log_every > 0 && iter % log_every == 0 {
            println!("  iter {:>4}: cost = {:.10}  (best so far {:.10})",
                     iter, cost, best_cost);
        }
    }

    println!(
        "[qaoa_qubo_self.wasm] best cost (= QUBO objective) = {:.10} at iter {best_iter}",
        best_cost
    );

    unsafe {
        omega_report_result(
            best_params.as_ptr(),
            n_params as i32,
            best_cost.to_bits() as i64,
            max_iters as i32,
        );
    }
}
