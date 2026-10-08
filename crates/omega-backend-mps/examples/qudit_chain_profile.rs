// SPDX-License-Identifier: Apache-2.0
// PLAN-QUDIT.md Q4 — measure the qudit MPS on a named box.
//
// The workload is qutrit chain dynamics: a brickwork of random `rxy` level
// rotations and `csum` entanglers on a chain of `n` qutrits, the same family
// `tests/qudit_backend.rs` uses for the truncation-contract sweep. Every row
// is measured THROUGH `MpsBackend::expectation_site_operators` — evolution
// plus an `O(χ²)` contraction of one site operator — and never through
// `execute(shots: None)`, which ends in `to_statevector`, a `3^n` dense
// reconstruction that is not MPS evolution and that the bond does not bound.
// STATUS.md §5.16 records the profile that made exactly that mistake.
//
// A10 (PLAN-OPEN-20260825.md:2579-2584): a measurement harness states its
// precondition and FAILS when a row would measure something other than its
// label. Here the label is "qudit MPS", so:
//   * every executed circuit must carry a wire of dimension > 2 — the harness
//     exits 2 if asked for `--d 2`, rather than publishing a qubit table under
//     a qudit heading;
//   * every published row carries its certificate (`discarded_weight`,
//     `fidelity_estimate`, `max_bond_reached`) or is published as a REFUSAL
//     with the engine's sentence — a row with neither is a bug and the
//     harness exits 3;
//   * the accelerator receipts do not apply (CPU only; no `contract_fn`, no
//     `svd_fn` override), and the harness prints the thread count and the
//     host so a number can never be quoted without its box.
//
// Certificate story (PLAN-QUDIT §Q4): no new certificate. A refused row is
// ALSO re-run with the ceiling lifted and its value published labelled
// UNCERTIFIED beside the certificate that refused it — the §3c.6 P4 "honest
// table" shape — so the reader sees what the engine would have said and why
// it declined to say it.
//
// Wall time and peak RSS together (FIXES_PLAN.md:1280): `ru_maxrss` is the
// process high-water mark, so the RSS column is cumulative over the rows in
// the order printed (small to large); the increment between consecutive rows
// is the row's own footprint.
//
// Output: a Markdown table on stdout, and with `--json PATH` a machine-readable
// record of every row PLUS each circuit as a neutral gate list, so
// `tools/qudit_chain_xcheck/tnsim_rows.py` can build the identical circuit in
// MQT Qudits' tensor-network backend (through its API — `from_qasm` drops
// `rxy` parameters, Q0 defect D2) and compare.
//
// Usage:
//   cargo run --release -p omega-backend-mps --example qudit_chain_profile -- \
//       small|large [--d 3] [--seed 20260930] [--json PATH]
//
//   small: n ∈ {6, 8, 10, 12}, 6 layers, χ ∈ {4, 8, 16, 27}  — laptop rows
//   large: n ∈ {16, 20, 24},  8 layers, χ ∈ {64, 128, 256}   — 123 GB rows

use std::time::Instant;

use num_complex::Complex64;
use omega_backend_mps::MpsBackend;
use omega_backend_quditsv::sim::run as exact_run;
use omega_core::circuit::{
    CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit, QuditRegister,
};
use omega_core::params::ParameterBinding;

type C = Complex64;

/// Deterministic LCG — the generator the qudit test estate uses, so a row
/// here is the same circuit family the contract sweep certified.
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn angle(&mut self) -> f64 {
        self.next_f64() * 2.0 * std::f64::consts::PI - std::f64::consts::PI
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next_f64() * n as f64) as u32 % n
    }
    fn pair(&mut self, n: u32) -> (u32, u32) {
        let a = self.below(n);
        let mut b = self.below(n - 1);
        if b >= a {
            b += 1;
        }
        (a, b)
    }
}

fn op(gate: GateKind, wires: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: wires.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// A neutral record of one gate, for the JSON the tnsim leg consumes.
#[derive(Clone)]
struct Gate {
    name: &'static str,
    wires: Vec<u32>,
    params: Vec<f64>,
}

/// Brickwork on `n` sites of local dimension `d`: per layer, one `rxy` on a
/// random level pair of every site, `csum` on alternating adjacent bricks,
/// and one distant `csum` so the SWAP network is exercised. Random angles
/// give a non-flat Schmidt spectrum, which is the only kind that can expose a
/// gauge problem in the certificate (`tests/fidelity_estimate.rs`).
fn brickwork(n: usize, d: u32, layers: usize, seed: u64) -> (CircuitIR, Vec<Gate>) {
    let mut rng = Lcg(seed);
    let mut ops = Vec::new();
    let mut gates = Vec::new();
    for layer in 0..layers {
        for w in 0..n as u32 {
            let i = rng.below(d - 1);
            let j = i + 1 + rng.below(d - 1 - i);
            let (theta, phi) = (rng.angle(), rng.angle());
            ops.push(op(GateKind::Rxy, &[w], &[i as f64, j as f64, theta, phi]));
            gates.push(Gate {
                name: "rxy",
                wires: vec![w],
                params: vec![i as f64, j as f64, theta, phi],
            });
        }
        let mut w = (layer % 2) as u32;
        while (w + 1) < n as u32 {
            ops.push(op(GateKind::CSum, &[w, w + 1], &[]));
            gates.push(Gate {
                name: "csum",
                wires: vec![w, w + 1],
                params: vec![],
            });
            w += 2;
        }
        let (a, b) = rng.pair(n as u32);
        ops.push(op(GateKind::CSum, &[a, b], &[]));
        gates.push(Gate {
            name: "csum",
            wires: vec![a, b],
            params: vec![],
        });
    }
    let mut c = CircuitIR::new(n as u32, CircuitType::GateBased);
    c.qudit_registers.push(QuditRegister {
        name: "q".into(),
        start: 0,
        dims: vec![d; n],
    });
    c.ops = ops;
    (c, gates)
}

/// The observable: the spin-`(d−1)/2` `S_z` on the middle site,
/// `diag((d−1)/2, (d−3)/2, …, −(d−1)/2)`, identity elsewhere. Hermitian, real,
/// traceless, and NOT a projector — so a value that agrees with the exact one
/// is a statement about the amplitudes' relative phases, not just their
/// weights. At `d = 3` it is `diag(1, 0, −1)`.
fn site_ops(n: usize, d: u32) -> Vec<Vec<C>> {
    let dd = d as usize;
    let ident = {
        let mut m = vec![C::new(0.0, 0.0); dd * dd];
        for k in 0..dd {
            m[k * dd + k] = C::new(1.0, 0.0);
        }
        m
    };
    let mut sz = vec![C::new(0.0, 0.0); dd * dd];
    for k in 0..dd {
        sz[k * dd + k] = C::new((d as f64 - 1.0) / 2.0 - k as f64, 0.0);
    }
    (0..n)
        .map(|w| {
            if w == n / 2 {
                sz.clone()
            } else {
                ident.clone()
            }
        })
        .collect()
}

/// `⟨ψ|O|ψ⟩ / ⟨ψ|ψ⟩` on the exact dense vector (wire 0 least significant),
/// the reference the small rows are measured against. `None` when quditsv
/// refuses the size — published as "no exact reference", not as 0.
fn exact_reference(circuit: &CircuitIR, ops: &[Vec<C>]) -> Option<f64> {
    let st = exact_run(circuit, &ParameterBinding::new()).ok()?;
    let mut v = st.amp.clone();
    for (w, o) in ops.iter().enumerate() {
        let d = st.dims[w] as usize;
        let s = st.strides[w];
        let mut out = vec![C::new(0.0, 0.0); v.len()];
        for (i, slot) in out.iter_mut().enumerate() {
            let r = st.digit(i, w);
            let base = i - r * s;
            let mut acc = C::new(0.0, 0.0);
            for k in 0..d {
                acc += o[r * d + k] * v[base + k * s];
            }
            *slot = acc;
        }
        v = out;
    }
    let num: C = st.amp.iter().zip(&v).map(|(a, b)| a.conj() * b).sum();
    let den: f64 = st.amp.iter().map(|a| a.norm_sqr()).sum();
    Some(num.re / den)
}

/// Process high-water RSS in bytes — what `/usr/bin/time -v` prints as
/// "Maximum resident set size". Linux reports kilobytes, macOS bytes.
fn peak_rss_bytes() -> u64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `ru` is a valid, writable `rusage`; `RUSAGE_SELF` is a constant
    // the call accepts on both platforms.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    assert_eq!(rc, 0, "getrusage failed");
    let raw = ru.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        raw
    } else {
        raw * 1024
    }
}

fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown-host".into())
}

struct Row {
    n: usize,
    layers: usize,
    chi: usize,
    ops: usize,
    wall_ms: f64,
    peak_rss: u64,
    /// `Ok(value)` when the certificate admitted the run; `Err(sentence)` when
    /// it refused. Both carry the certificate below.
    value: Result<f64, String>,
    /// The value the engine would have returned with the ceiling lifted —
    /// only for refused rows, and labelled uncertified in every output.
    uncertified: Option<f64>,
    discarded_weight: f64,
    fidelity_estimate: f64,
    max_bond: usize,
    exact: Option<f64>,
}

fn usage() -> ! {
    eprintln!(
        "usage: qudit_chain_profile small|large [--d D] [--seed S] [--json PATH]\n\
         \n  small: n in {{6, 8, 10, 12}}, 6 layers, chi in {{4, 8, 16, 27}}\n\
         \x20 large: n in {{16, 20, 24}}, 8 layers, chi in {{64, 128, 256}}"
    );
    std::process::exit(1)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut profile: Option<&str> = None;
    let mut d: u32 = 3;
    let mut seed: u64 = 20260930;
    let mut json_path: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "small" | "large" => profile = Some(if a == "small" { "small" } else { "large" }),
            "--d" => {
                d = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            "--seed" => {
                seed = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            "--json" => json_path = it.next().cloned(),
            _ => usage(),
        }
    }
    let Some(profile) = profile else { usage() };

    // A10 precondition, before any row is run: the label is "qudit MPS", so
    // a local dimension of 2 is not a smaller instance of the same thing, it
    // is a different table under the same heading. Refuse it.
    if d < 3 {
        eprintln!(
            "REFUSING: --d {d} asks for a qubit chain, and every row below would be a \
             qubit measurement published under a qudit heading (PLAN-QUDIT §Q4, A10). \
             Use --d 3 or higher."
        );
        std::process::exit(2);
    }

    let (ns, layers, chis): (Vec<usize>, usize, Vec<usize>) = match profile {
        "small" => (vec![6, 8, 10, 12], 6, vec![4, 8, 16, 27]),
        _ => (vec![16, 20, 24], 8, vec![64, 128, 256]),
    };
    let host = hostname();
    let threads = rayon::current_num_threads();
    println!(
        "# PLAN-QUDIT Q4 — qutrit chain dynamics on `{host}` ({threads} threads, \
         d = {d}, seed {seed}, profile `{profile}`)\n"
    );
    println!(
        "Measured through `MpsBackend::expectation_site_operators` (⟨S_z⟩ on the middle \
         site; evolution + O(χ²) contraction, no dense readout). A refused row shows the \
         engine's certificate and, in the last column, the UNCERTIFIED value it would have \
         returned with the ceiling lifted. RSS is the process high-water mark, cumulative \
         down the table.\n"
    );
    println!(
        "| box | n | layers | ops | χ | wall ms | peak RSS | discarded_weight | fidelity_est | max_bond | ⟨S_z⟩ (certified) | exact | \\|Δ\\| | uncertified |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");

    let mut rows: Vec<Row> = Vec::new();
    let mut circuits_json = Vec::new();
    let mut qudit_rows = 0usize;

    for &n in &ns {
        let (circuit, gates) = brickwork(n, d, layers, seed ^ (n as u64));
        // Precondition, per circuit: the executed circuit carries d > 2 on
        // every wire, read back from the IR rather than from the argument.
        let dims = circuit.wire_dims();
        assert!(
            dims.iter().all(|&x| x == d) && d > 2,
            "row n={n}: circuit dims {dims:?} are not the qudit chain the label claims"
        );
        let ops = site_ops(n, d);
        let exact = exact_reference(&circuit, &ops);
        circuits_json.push(serde_json::json!({
            "n": n, "d": d, "layers": layers, "seed": seed ^ (n as u64),
            "observable": {"site": n / 2, "kind": "Sz"},
            "gates": gates.iter().map(|g| serde_json::json!({
                "name": g.name, "wires": g.wires, "params": g.params
            })).collect::<Vec<_>>(),
            "exact": exact,
        }));
        for &chi in &chis {
            let b = MpsBackend::new(chi);
            let t = Instant::now();
            let res = b.expectation_site_operators(&circuit, &ParameterBinding::new(), &ops);
            let wall_ms = t.elapsed().as_secs_f64() * 1e3;
            // A refused row: the engine's early abort stops evolution the
            // moment the running certificate crosses the ceiling and returns
            // before `record_stats`, so the certificate of the CERTIFIED
            // attempt lives in its sentence. The published numbers for such a
            // row come from a second, ceiling-lifted run of the same circuit —
            // a complete run, whose `last_run_stats` is the certificate the
            // engine would attach to the value it declined to certify.
            let (value, uncertified, st) = match res {
                Ok(v) => (Ok(v), None, b.last_run_stats()),
                Err(e) => {
                    let open = MpsBackend::new(chi).with_max_discarded_weight(f64::INFINITY);
                    let u = open
                        .expectation_site_operators(&circuit, &ParameterBinding::new(), &ops)
                        .ok();
                    (Err(e.to_string()), u, open.last_run_stats())
                }
            };
            // A row must carry a certificate from a run that actually
            // evolved this circuit. `max_bond_reached` starts at 1 on any
            // chain that was built; 0 is the untouched default.
            if st.max_bond_reached == 0 {
                eprintln!(
                    "row n={n} chi={chi}: no certificate from any run of this circuit \
                     (max_bond_reached = 0). The door said: {}",
                    match &value {
                        Ok(v) => format!("Ok({v})"),
                        Err(e) => e.clone(),
                    }
                );
                std::process::exit(3);
            }
            let peak_rss = peak_rss_bytes();
            qudit_rows += 1;
            let row = Row {
                n,
                layers,
                chi,
                ops: circuit.ops.len(),
                wall_ms,
                peak_rss,
                value,
                uncertified,
                discarded_weight: st.discarded_weight,
                fidelity_estimate: st.fidelity_estimate,
                max_bond: st.max_bond_reached,
                exact,
            };
            print_row(&host, &row);
            rows.push(row);
        }
    }
    // The harness has measured something only if every row was a qudit row.
    assert_eq!(qudit_rows, ns.len() * chis.len(), "rows went missing");

    if let Some(path) = json_path {
        let doc = serde_json::json!({
            "plan": "PLAN-QUDIT Q4",
            "box": host,
            "threads": threads,
            "profile": profile,
            "d": d,
            "seed": seed,
            "measured_through": "MpsBackend::expectation_site_operators",
            "circuits": circuits_json,
            "rows": rows.iter().map(|r| serde_json::json!({
                "n": r.n, "layers": r.layers, "ops": r.ops, "chi": r.chi,
                "wall_ms": r.wall_ms, "peak_rss_bytes": r.peak_rss,
                "certified": r.value.as_ref().ok(),
                "refused": r.value.as_ref().err(),
                "uncertified": r.uncertified,
                "discarded_weight": r.discarded_weight,
                "fidelity_estimate": r.fidelity_estimate,
                "max_bond_reached": r.max_bond,
                "exact": r.exact,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap())
            .unwrap_or_else(|e| panic!("write {path}: {e}"));
        eprintln!("wrote {path}");
    }
}

fn print_row(host: &str, r: &Row) {
    let rss = omega_core::hostmem::human_bytes(r.peak_rss as u128);
    let (cert, unc) = match &r.value {
        Ok(v) => (format!("{v:.10}"), "—".to_string()),
        Err(msg) => {
            // The engine's sentence carries the certificate of the refused
            // attempt: "… certificate 1.219e-6 exceeded the ceiling 1.000e-6
            // partway through … (bond reached 4)". Quote those three numbers
            // rather than the whole sentence; the JSON keeps it verbatim.
            let after = |key: &str| {
                msg.split(key)
                    .nth(1)
                    .and_then(|t| t.split(|c: char| c.is_whitespace() || c == ')').next())
                    .unwrap_or("?")
                    .to_string()
            };
            let (cert, ceiling, bond) = (
                after("certificate "),
                after("ceiling "),
                after("bond reached "),
            );
            (
                format!("**REFUSED** (cert {cert} > ceiling {ceiling}, bond {bond})"),
                r.uncertified
                    .map(|u| format!("{u:.10}"))
                    .unwrap_or("—".into()),
            )
        }
    };
    let (exact, delta) = match (r.exact, &r.value) {
        (Some(e), Ok(v)) => (format!("{e:.10}"), format!("{:.2e}", (v - e).abs())),
        (Some(e), Err(_)) => (
            format!("{e:.10}"),
            r.uncertified
                .map(|u| format!("{:.2e} (uncert.)", (u - e).abs()))
                .unwrap_or("—".into()),
        ),
        (None, _) => ("no exact ref (quditsv refused the size)".into(), "—".into()),
    };
    println!(
        "| {host} | {} | {} | {} | {} | {:.1} | {rss} | {:.3e} | {:.4} | {} | {cert} | {exact} | {delta} | {unc} |",
        r.n, r.layers, r.ops, r.chi, r.wall_ms, r.discarded_weight, r.fidelity_estimate, r.max_bond
    );
}
