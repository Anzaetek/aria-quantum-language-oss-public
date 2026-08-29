use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

fn pop(g: GateKind, qs: &[u32], ps: &[f64]) -> GateOp {
    GateOp {
        gate: g,
        qubits: qs.iter().map(|&q| Qubit(q)).collect(),
        params: ps.iter().map(|&v| ParamExpr::Concrete(v)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// An angle for the rotation corpus: generic, and deliberately never a multiple
/// of pi/4.
///
/// The point of this corpus is phase conventions, and the special angles are
/// exactly where a convention error CANCELS — `Rz(pi/2)` is `S` up to phase, so
/// a wrong global-phase convention on `Rz` is invisible there and the circuit
/// degenerates back into the Clifford corpus that already passes. The offset and
/// the irrational scale keep every angle away from those points.
fn angle(seed: &mut u64) -> f64 {
    0.13 + (rnd(seed) % 100_000) as f64 * (std::f64::consts::PI / 100_003.0)
}

fn op(g: GateKind, qs: &[u32]) -> GateOp {
    GateOp {
        gate: g,
        qubits: qs.iter().map(|&q| Qubit(q)).collect(),
        params: smallvec![],
        classical_bit: None,
        condition: None,
    }
}
fn rnd(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn main() {
    let n_circ: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let cpu = omega_backend_statevector::StatevectorBackend::new();
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let metal = omega_backend_statevector_metal::MetalStatevectorBackend::new().ok();
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    let metal: Option<()> = None;
    let pb = ParameterBinding::new();
    let an = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };

    let mut seed = 0x5150u64;
    #[cfg_attr(not(all(target_os = "macos", feature = "metal")), allow(unused_mut))]
    let mut worst_metal = 0.0f64;
    let mut worst_stab = 0.0f64;
    let mut unnorm = 0usize;
    let mut n_stab = 0usize;
    let stab = omega_backend_pauli::PauliBackend::new();
    // Emit the gate list so Python/Qiskit builds the SAME circuits.
    println!("#BEGIN");
    for _ in 0..n_circ {
        let n = 2 + (rnd(&mut seed) % 3) as u32;
        let depth = 4 + (rnd(&mut seed) % 10) as usize;
        let mut c = CircuitIR::new(n, CircuitType::GateBased);
        let mut desc: Vec<String> = Vec::new();
        for _ in 0..depth {
            let q = (rnd(&mut seed) % n as u64) as u32;
            let (g, name, qs) = match rnd(&mut seed) % 8 {
                0 => (GateKind::H, "h", vec![q]),
                1 => (GateKind::S, "s", vec![q]),
                2 => (GateKind::Sdg, "sdg", vec![q]),
                3 => (GateKind::X, "x", vec![q]),
                4 => (GateKind::Z, "z", vec![q]),
                // CCX / CSwap: three-qubit permutations. Added so the exact
                // subspace kernel (MultiControlMode::Exact) and the 15-gate
                // decomposition are BOTH gated against Qiskit — comparing the
                // two against each other would only prove they share a
                // convention, which they do by construction.
                5 if n >= 3 => {
                    let b = (rnd(&mut seed) % n as u64) as u32;
                    let c = (rnd(&mut seed) % n as u64) as u32;
                    if q != b && q != c && b != c {
                        (GateKind::CCX, "ccx", vec![q, b, c])
                    } else {
                        (GateKind::H, "h", vec![q])
                    }
                }
                6 if n >= 3 => {
                    let b = (rnd(&mut seed) % n as u64) as u32;
                    let c = (rnd(&mut seed) % n as u64) as u32;
                    if q != b && q != c && b != c {
                        (GateKind::CSwap, "cswap", vec![q, b, c])
                    } else {
                        (GateKind::H, "h", vec![q])
                    }
                }
                _ => {
                    let b = (rnd(&mut seed) % n as u64) as u32;
                    if q == b {
                        (GateKind::H, "h", vec![q])
                    } else {
                        (GateKind::CX, "cx", vec![q, b])
                    }
                }
            };
            c.ops.push(op(g, &qs));
            desc.push(format!(
                "{name}:{}",
                qs.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        let sv = match cpu.execute(&c, &pb, &an).unwrap() {
            ExecResult::Statevector(v) => v,
            _ => continue,
        };
        let p_cpu: Vec<f64> = sv.iter().map(|a| a.norm_sqr()).collect();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(m) = metal.as_ref() {
            if let Ok(ExecResult::Statevector(mv)) = m.execute(&c, &pb, &an) {
                let p_metal: Vec<f64> = mv.iter().map(|a| a.norm_sqr()).collect();
                let d = p_cpu
                    .iter()
                    .zip(p_metal.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);
                if d > worst_metal {
                    worst_metal = d;
                }
            }
        }
        let _ = &metal;
        // Stabilizer backend (exact probabilities) on the SAME circuit.
        if let Ok(ExecResult::Probabilities(p_stab)) = stab.execute(&c, &pb, &an) {
            n_stab += 1;
            let ds = p_cpu
                .iter()
                .zip(p_stab.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            if ds > worst_stab {
                worst_stab = ds;
            }
            let sum: f64 = p_stab.iter().sum();
            if (sum - 1.0).abs() > 1e-9 {
                unnorm += 1;
            }
        }
        // circuit  n  gates...  |  cpu probabilities
        println!(
            "C {n} {} | {}",
            desc.join(" "),
            p_cpu
                .iter()
                .map(|p| format!("{p:.12}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    // ---- Rotation corpus: NON-CLIFFORD, generic angles ---------------------
    //
    // The corpus above is six Clifford gates plus CCX/CSwap. It cannot exercise
    // a phase-convention error, because Cliffords have no free angle to get
    // wrong — and it reported 4.441e-16 over 60 cases while three real defects
    // were live. A check whose corpus does not span the feature space reads
    // exactly like a thorough one from the summary line.
    //
    // Kept as a SEPARATE loop rather than folded into the switch above, and the
    // reason is coverage rather than tidiness: the stabilizer backend refuses
    // non-Clifford gates, so every rotation mixed into that corpus would have
    // silently removed one circuit from the stabilizer comparison. Splitting
    // them keeps the Clifford corpus exactly as it was and reports the two
    // counts separately, so neither can quietly shrink.
    //
    // `CRz` and `CP` are both here on purpose. They differ only by a relative
    // phase `e^{-i lambda/2}` on the controlled block — substituting one for the
    // other is NOT a global phase and is visible in any interference — and
    // aria-core's own `GateKind::CRz` doc calls that out as an easy confusion.
    // A corpus with one and not the other cannot catch the swap.
    let mut n_rot = 0usize;
    for _ in 0..n_circ {
        let n = 2 + (rnd(&mut seed) % 3) as u32;
        let depth = 4 + (rnd(&mut seed) % 10) as usize;
        let mut c = CircuitIR::new(n, CircuitType::GateBased);
        let mut desc: Vec<String> = Vec::new();
        // Open with H on every qubit: a rotation acting on |0..0> is largely
        // invisible in PROBABILITIES (Rz is diagonal, so it changes nothing at
        // all), and this corpus is compared on probabilities. Without an opening
        // superposition, half the gates here would be untested no-ops.
        for q in 0..n {
            c.ops.push(op(GateKind::H, &[q]));
            desc.push(format!("h:{q}"));
        }
        for _ in 0..depth {
            let q = (rnd(&mut seed) % n as u64) as u32;
            let b = (rnd(&mut seed) % n as u64) as u32;
            let (g, name, qs, ps) = match rnd(&mut seed) % 7 {
                0 => (GateKind::Rx, "rx", vec![q], vec![angle(&mut seed)]),
                1 => (GateKind::Ry, "ry", vec![q], vec![angle(&mut seed)]),
                2 => (GateKind::Rz, "rz", vec![q], vec![angle(&mut seed)]),
                3 => (
                    GateKind::U3,
                    "u3",
                    vec![q],
                    vec![angle(&mut seed), angle(&mut seed), angle(&mut seed)],
                ),
                4 if q != b => (GateKind::CRz, "crz", vec![q, b], vec![angle(&mut seed)]),
                5 if q != b => (
                    GateKind::CU3,
                    "cu3",
                    vec![q, b],
                    vec![angle(&mut seed), angle(&mut seed), angle(&mut seed)],
                ),
                _ => {
                    if q != b {
                        (GateKind::CX, "cx", vec![q, b], vec![])
                    } else {
                        (GateKind::Ry, "ry", vec![q], vec![angle(&mut seed)])
                    }
                }
            };
            c.ops.push(pop(g, &qs, &ps));
            // 17 significant digits: the two sides must run the SAME circuit,
            // and a shortened angle would make them run different ones and
            // report the difference as a numerical disagreement.
            let mut tok = format!(
                "{name}:{}",
                qs.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            if !ps.is_empty() {
                tok.push('@');
                tok.push_str(
                    &ps.iter()
                        .map(|v| format!("{v:.17e}"))
                        .collect::<Vec<_>>()
                        .join(","),
                );
            }
            desc.push(tok);
        }
        let sv = match cpu.execute(&c, &pb, &an).unwrap() {
            ExecResult::Statevector(v) => v,
            _ => continue,
        };
        let p_cpu: Vec<f64> = sv.iter().map(|a| a.norm_sqr()).collect();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(m) = metal.as_ref() {
            if let Ok(ExecResult::Statevector(mv)) = m.execute(&c, &pb, &an) {
                let d = p_cpu
                    .iter()
                    .zip(mv.iter().map(|a| a.norm_sqr()))
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);
                if d > worst_metal {
                    worst_metal = d;
                }
            }
        }
        n_rot += 1;
        println!(
            "C {n} {} | {}",
            desc.join(" "),
            p_cpu
                .iter()
                .map(|p| format!("{p:.12}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    eprintln!("  rotation corpus: {n_rot} circuits (non-Clifford, generic angles)");

    // ---- Feedforward corpus -------------------------------------------------
    //
    // The analytic corpus above is Clifford-only with `condition: None` and
    // `MidCircuitMode::Skip`, so it cannot reach mid-circuit measurement or
    // classical control — and three real defects lived exactly there while it
    // reported 4.441e-16 over 60 cases. These circuits are compared on
    // DISTRIBUTIONS over shots, because the defects were in the SAMPLING and an
    // analytic vector comparison is structurally blind to that.
    //
    // Crucially the condition is driven by `H` (sometimes false). A condition
    // that is always true hides a dropped guard: both engines then lose the
    // same thing and agree.
    const FF_SHOTS: u32 = 4000;
    for i in 0..8u32 {
        let n = 2 + (i % 2); // 2 or 3 qubits
        let mut c = CircuitIR::new(n, CircuitType::GateBased);
        c.num_classical_bits = n;
        let mut desc: Vec<String> = Vec::new();

        // Control qubit into superposition -> the condition is ~50/50.
        c.ops.push(op(GateKind::H, &[0]));
        desc.push("h:0".into());
        // Half the cases add a pre-rotation so the control is biased, which
        // exercises a condition that is neither always-true nor always-false.
        if i % 2 == 1 {
            c.ops.push(op(GateKind::S, &[0]));
            desc.push("s:0".into());
        }
        c.ops.push(GateOp {
            gate: GateKind::Measure,
            qubits: smallvec![Qubit(0)],
            params: Default::default(),
            classical_bit: Some(0),
            condition: None,
        });
        desc.push("measure:0,0".into());
        // Conditional gate on the measured bit.
        let (tg, tname) = if i % 4 < 2 {
            (GateKind::X, "x")
        } else {
            (GateKind::Z, "z")
        };
        c.ops.push(GateOp {
            gate: tg,
            qubits: smallvec![Qubit(1)],
            params: Default::default(),
            classical_bit: None,
            condition: Some((0, 1, 1)),
        });
        desc.push(format!("if:0,1,{tname}:1"));
        // Read every remaining qubit so the distribution is observable.
        for q in 1..n {
            c.ops.push(GateOp {
                gate: GateKind::Measure,
                qubits: smallvec![Qubit(q)],
                params: Default::default(),
                classical_bit: Some(q),
                condition: None,
            });
            desc.push(format!("measure:{q},{q}"));
        }

        let cfg = ExecConfig {
            shots: Some(FF_SHOTS),
            seed: Some(1234 + i as u64),
            mid_circuit_mode: MidCircuitMode::Collapse,
        };
        if let Ok(ExecResult::Counts(m)) = cpu.execute(&c, &pb, &cfg) {
            let mut keys: Vec<_> = m.into_iter().collect();
            keys.sort_by(|a, b| a.0.cmp(&b.0));
            let body = keys
                .iter()
                // The cross-check corpus is compared textually against the
                // reference runner, so this must keep emitting the DECIMAL
                // outcome it always did — `Outcome`'s Display is a bitstring.
                .map(|(k, v)| format!("{}:{v}", k.as_u64().expect("xcheck circuits are narrow")))
                .collect::<Vec<_>>()
                .join(" ");
            println!("M {n} {FF_SHOTS} {} | {}", desc.join(" "), body);
        }
    }

    // ---- Reset corpus: the stochastic CHANNEL, compared on distributions ----
    //
    // `OPTIONAL_TESTS.md` gap #3. `Reset` was covered by `tests/reset_channel.rs`
    // against Aer but never entered the random corpus — and Reset is the
    // construct that shipped WRONG IN THREE BACKENDS AT ONCE, in three different
    // bases, while every internal cross-backend agreement gate passed because
    // each pair coincided in whatever basis was being checked.
    //
    // Compared on DISTRIBUTIONS, not a statevector: Reset is a stochastic
    // channel, so one trajectory is one sample and an analytic vector
    // comparison is structurally blind to getting the sampling wrong.
    //
    // # Every circuit here must be able to SEE a dropped Reset
    //
    // `reset` on a qubit already in |0> does nothing, and a corpus full of those
    // passes whether or not Reset is implemented at all. So each circuit puts
    // the target somewhere Reset visibly changes — |1>, or a superposition —
    // and `reset_is_observable` below proves it per circuit rather than trusting
    // the construction: it runs the SAME circuit with the Reset ops removed and
    // requires the two distributions to differ. A circuit that cannot tell those
    // apart is reported and excluded rather than counted as coverage.
    const RS_SHOTS: u32 = 4000;
    let mut n_reset = 0usize;
    for i in 0..8u32 {
        let n = 2 + (i % 2);
        let mut c = CircuitIR::new(n, CircuitType::GateBased);
        c.num_classical_bits = n;
        let mut desc: Vec<String> = Vec::new();

        match i % 4 {
            // (a) |1> then reset: the sharpest signature available. With Reset
            // the qubit reads 0 on every shot; without it, 1 on every shot.
            0 => {
                c.ops.push(op(GateKind::X, &[0]));
                desc.push("x:0".into());
            }
            // (b) superposition then reset: reads 0 always with Reset, 50/50
            // without.
            1 => {
                c.ops.push(op(GateKind::H, &[0]));
                desc.push("h:0".into());
            }
            // (c) ENTANGLED, and this is the case the others cannot reach.
            // Resetting half a Bell pair must COLLAPSE the partner. An
            // implementation that forces |0> without collapsing leaves the
            // partner in superposition, which no product-state circuit above
            // would distinguish.
            2 => {
                c.ops.push(op(GateKind::H, &[0]));
                c.ops.push(op(GateKind::CX, &[0, 1]));
                desc.push("h:0".into());
                desc.push("cx:0,1".into());
            }
            // (d) reset then REUSE: the ancilla-recycling pattern real QEC
            // circuits are built from.
            _ => {
                c.ops.push(op(GateKind::H, &[0]));
                c.ops.push(op(GateKind::CX, &[0, 1]));
                desc.push("h:0".into());
                desc.push("cx:0,1".into());
            }
        }

        c.ops.push(op(GateKind::Reset, &[0]));
        desc.push("reset:0".into());

        if i % 4 == 3 {
            // Reuse the reset qubit — the ancilla-recycling pattern.
            //
            // `X`, not `H`, and that is a MEASURED correction. The first draft
            // reused it with `H`, and the non-vacuity check below rejected both
            // instances at TVD 0.0103 and 0.0085: `H; CX; reset; H` and
            // `H; CX; H` both end uniform over four outcomes, so the circuit
            // could not see a dropped Reset at all. `X` after the reset gives
            // q0 = 1 on every shot when Reset works, against a 50/50 flip of an
            // entangled qubit when it does not.
            c.ops.push(op(GateKind::X, &[0]));
            desc.push("x:0".into());
        }

        for q in 0..n {
            c.ops.push(GateOp {
                gate: GateKind::Measure,
                qubits: smallvec![Qubit(q)],
                params: Default::default(),
                classical_bit: Some(q),
                condition: None,
            });
            desc.push(format!("measure:{q},{q}"));
        }

        let cfg = ExecConfig {
            shots: Some(RS_SHOTS),
            seed: Some(0x5E7 + i as u64),
            mid_circuit_mode: MidCircuitMode::Collapse,
        };
        let Ok(ExecResult::Counts(m)) = cpu.execute(&c, &pb, &cfg) else {
            continue;
        };

        // NON-VACUITY, proved rather than assumed: the same circuit with the
        // Reset ops stripped must give a MATERIALLY different distribution.
        let mut stripped = c.clone();
        stripped.ops.retain(|o| !matches!(o.gate, GateKind::Reset));
        let Ok(ExecResult::Counts(m0)) = cpu.execute(&stripped, &pb, &cfg) else {
            continue;
        };
        let tvd = {
            let mut keys: std::collections::BTreeSet<u64> = Default::default();
            for k in m.keys().chain(m0.keys()) {
                keys.insert(k.as_u64().expect("narrow"));
            }
            let get = |src: &std::collections::HashMap<omega_core::outcome::Outcome, u32>,
                       k: u64| {
                src.iter()
                    .find(|(o, _)| o.as_u64() == Some(k))
                    .map(|(_, v)| *v)
                    .unwrap_or(0)
            };
            0.5 * keys
                .iter()
                .map(|&k| (get(&m, k) as f64 - get(&m0, k) as f64).abs())
                .sum::<f64>()
                / RS_SHOTS as f64
        };
        if tvd < 0.2 {
            eprintln!(
                "  reset corpus: circuit {i} is VACUOUS (TVD vs reset-stripped = \
                 {tvd:.4}); it cannot see a dropped Reset, so it is excluded"
            );
            continue;
        }

        let mut keys: Vec<_> = m.into_iter().collect();
        keys.sort_by(|a, b| a.0.cmp(&b.0));
        let body = keys
            .iter()
            .map(|(k, v)| format!("{}:{v}", k.as_u64().expect("xcheck circuits are narrow")))
            .collect::<Vec<_>>()
            .join(" ");
        n_reset += 1;
        println!("M {n} {RS_SHOTS} {} | {}", desc.join(" "), body);
    }
    eprintln!("  reset corpus: {n_reset} circuits (stochastic channel, distribution-compared)");

    println!("#END");
    // Report each arm's OWN coverage, not the corpus size. Metal runs on both
    // corpora and the stabilizer arm only on the Clifford one — it refuses
    // non-Clifford gates — so a single `over {n_circ} circuits` would have
    // over-reported Metal and silently over-reported the stabilizer arm by
    // twice as much. A count attached to the wrong arm is how a narrowed check
    // keeps looking wide.
    eprintln!(
        "metal-vs-cpu   worst |Δp| over {} circuits (Clifford + rotation) = {worst_metal:.3e}",
        n_circ + n_rot
    );
    eprintln!(
        "stab-vs-cpu    worst |Δp| over {n_stab} Clifford circuits = {worst_stab:.3e}, \
         unnormalised = {unnorm}  (the stabilizer backend refuses non-Clifford, \
         so the {n_rot} rotation circuits are NOT in this figure)"
    );
}
