// SPDX-License-Identifier: Apache-2.0
//! qaoa_maxcut — approximate MaxCut on a triangle ring.
//!
//! WHAT: the MaxCut value of the 3-node ring (triangle).
//! QUANTUM: optimize QAOA(p=2) from qaoa_maxcut.aria by minimizing the cost
//!   ⟨½ ΣZᵢZⱼ⟩ (vqe.wasm); convert to an estimated cut = (|E| − 2·min)/2.
//! CLASSICAL: brute-force MaxCut over all 2³ partitions.
//! CHECK: QAOA cut is within 0.5 of the brute-force optimum (QAOA is
//!   approximate; the optimum is exact). Both values are printed.

use aria_verify_core::{banner, harness, oracle, resolve, Observable, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let n = 3usize;
    let edges = [(0usize, 1usize), (1, 2), (0, 2)];
    let guest = "vqe";
    let transport = resolve(transport_override, guest);
    banner::header(
        "qaoa_maxcut",
        "MaxCut value of the 3-node triangle (QAOA p=2 vs brute force)",
        &transport.label(guest),
    );

    let (brute, mask) = oracle::brute_force_maxcut(n, &edges);
    println!("  brute-force optimum: cut {brute} at partition {mask:03b}");

    let lowered = harness::load_lowered(
        "qaoa_maxcut.aria",
        "QAOAMaxCut",
        &[("n", n as i64), ("p", 2)],
    )?;
    let n_params = lowered.symbol_ids.len().max(1);
    // Cost observable: ½·Σ_edges Zi Zj (minimizing it maximizes the cut).
    let obs_str = edges
        .iter()
        .map(|(i, j)| format!("0.5*Z{i}Z{j}"))
        .collect::<Vec<_>>()
        .join("+");
    let obs = Observable::parse(&obs_str)?;
    let (min_cost, _params) = harness::minimize(
        transport,
        guest,
        lowered.ir,
        obs,
        n_params,
        vec![0.4; n_params],
        500,
        0.1,
    )?;
    // cut = Σ_edges (1 − ⟨ZiZj⟩)/2 = (|E| − Σ⟨ZiZj⟩)/2, and min_cost = ½·Σ⟨ZiZj⟩.
    let cut_q = (edges.len() as f64 - 2.0 * min_cost) / 2.0;

    Ok(banner::report_scalar(
        "qaoa_maxcut",
        "QAOA estimated cut",
        cut_q,
        "brute-force MaxCut",
        brute as f64,
        0.5,
    ))
}
