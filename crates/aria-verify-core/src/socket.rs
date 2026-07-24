// SPDX-License-Identifier: Apache-2.0
//! Socket transport: drive a running omega-server over HTTP instead of the
//! in-process runtime. This is the "with socket" counterpart to the default
//! all-in-one case — the SAME shipped Aria package (lowered to the omega JSON
//! wire IR) is sent over the wire, executed remotely, and the returned counts
//! are cross-checked against the SAME pure-Rust classical oracle.
//!
//! Counts-based examples use `/v1/quantum/execute`; a parametric expectation
//! example uses `/v1/quantum/expectation` (scalar ⟨O⟩ over the wire, no
//! statevector round-trip). Amplitude / variational examples keep using the
//! in-process transport, which is the canonical path.

use std::collections::HashMap;

use aria_core::ast::parse_aria;
use aria_runtime::remote::{expectation_remote, run_counts_remote, Remote};
use omega_core::executor::ExecResult;

use crate::banner::{self, Verdict};
use crate::harness;

const TRANSPORT: &str = "omega-server over HTTP (socket)";

fn counts_map(res: ExecResult) -> HashMap<u64, u32> {
    match res {
        ExecResult::Counts(m) => m,
        _ => HashMap::new(),
    }
}

fn prob_of(counts: &HashMap<u64, u32>, key: u64) -> f64 {
    let total: u64 = counts.values().map(|&v| v as u64).sum();
    if total == 0 {
        return 0.0;
    }
    *counts.get(&key).unwrap_or(&0) as f64 / total as f64
}

fn argmax(counts: &HashMap<u64, u32>) -> u64 {
    counts
        .iter()
        .max_by_key(|(_, &v)| v)
        .map(|(&k, _)| k)
        .unwrap_or(0)
}

/// Run the socket demo against `url` (optional bearer `token`). Returns true
/// iff every remote example passes its numeric check.
pub fn run(url: &str, token: Option<String>) -> bool {
    let remote = Remote {
        url: url.to_string(),
        token,
    };
    let mut ok = true;
    match grover3_remote(&remote) {
        Ok(v) => ok &= v.ok(),
        Err(e) => {
            eprintln!("grover3 (socket): ERROR — {e}");
            ok = false;
        }
    }
    match bernstein_vazirani_remote(&remote) {
        Ok(v) => ok &= v.ok(),
        Err(e) => {
            eprintln!("bernstein_vazirani (socket): ERROR — {e}");
            ok = false;
        }
    }
    match ry_expectation_remote(&remote) {
        Ok(v) => ok &= v.ok(),
        Err(e) => {
            eprintln!("ry_expectation (socket): ERROR — {e}");
            ok = false;
        }
    }
    ok
}

/// Remote expectation via `/v1/quantum/expectation`: ⟨Z0⟩ of RY(θ)|0⟩ is
/// cos(θ) exactly, so the scalar returned by the server is cross-checked
/// against the closed-form value — the expectation-endpoint counterpart of the
/// counts examples above.
fn ry_expectation_remote(remote: &Remote) -> Result<Verdict, String> {
    banner::header(
        "ry_expectation",
        "⟨Z0⟩ of RY(theta)|0⟩ over the wire == cos(theta)",
        TRANSPORT,
    );
    let src =
        "circuit RyExp() {\n  qreg q[1]\n  let t = symbolic[1]\n  apply RY(t[0]) on q[0]\n}\n";
    let circuit = parse_aria(src)?.instantiate("RyExp", &[])?;
    let theta = 0.9_f64;
    let binds = HashMap::from([("t_0".to_string(), theta)]);
    let z = expectation_remote(&circuit, "Z0", &binds, remote)?;
    Ok(banner::report_scalar(
        "ry_expectation",
        "server ⟨Z0⟩ (remote)",
        z,
        "cos(theta)",
        theta.cos(),
        1e-9,
    ))
}

fn grover3_remote(remote: &Remote) -> Result<Verdict, String> {
    let marked: u64 = 5;
    let k = 2.0_f64;
    banner::header(
        "grover3",
        &format!("Grover search on 8 items for the marked index {marked}"),
        TRANSPORT,
    );
    let circ = harness::load_circuit("grover3.aria", "Grover3", &[("marked", marked as i64)])?;
    let counts = counts_map(run_counts_remote(
        &circ,
        &HashMap::new(),
        8192,
        Some(42),
        remote,
    )?);
    let amax = argmax(&counts);
    let p_marked = prob_of(&counts, marked);
    let theta = (1.0 / 8.0_f64.sqrt()).asin();
    let p_analytic = ((2.0 * k + 1.0) * theta).sin().powi(2);
    println!("  most-likely outcome = {amax} (marked = {marked})");
    if amax != marked {
        return Err(format!("grover returned {amax}, want {marked}"));
    }
    Ok(banner::report_scalar(
        "grover3",
        "measured P(marked), 8192 shots (remote)",
        p_marked,
        "analytic sin²((2k+1)θ)",
        p_analytic,
        0.05,
    ))
}

fn bernstein_vazirani_remote(remote: &Remote) -> Result<Verdict, String> {
    let n: i64 = 4;
    let a: u64 = 5;
    banner::header(
        "bernstein_vazirani",
        &format!("recover hidden a={a} from f(x)=a·x mod 2, in one query"),
        TRANSPORT,
    );
    let circ = harness::load_circuit(
        "bernstein_vazirani.aria",
        "BernsteinVazirani",
        &[("n", n), ("a", a as i64)],
    )?;
    let counts = counts_map(run_counts_remote(
        &circ,
        &HashMap::new(),
        4096,
        Some(42),
        remote,
    )?);
    // Remote counts are keyed by the full qubit register (query qubits + the
    // answer qubit q[n], which sits in |−⟩ and reads 0/1 at random). The hidden
    // string lives in the low n bits.
    let recovered = argmax(&counts) & ((1u64 << n) - 1);
    Ok(banner::report_exact_u64(
        "bernstein_vazirani",
        "recovered string (remote)",
        recovered,
        "hidden a",
        a,
        n as usize,
    ))
}
