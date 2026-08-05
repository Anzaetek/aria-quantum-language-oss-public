use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

fn op(g: GateKind, qs: &[u32]) -> GateOp {
    GateOp { gate: g, qubits: qs.iter().map(|&q| Qubit(q)).collect(), params: smallvec![],
             classical_bit: None, condition: None }
}
fn rnd(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

fn main() {
    let n_circ: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let cpu = omega_backend_statevector::StatevectorBackend::new();
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let metal = omega_backend_statevector_metal::MetalStatevectorBackend::new().ok();
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    let metal: Option<()> = None;
    let pb = ParameterBinding::new();
    let an = ExecConfig { shots: None, seed: None, mid_circuit_mode: MidCircuitMode::Skip };

    let mut seed = 0x5150u64;
    let mut worst_metal = 0.0f64;
    let mut worst_stab = 0.0f64;
    let mut unnorm = 0usize;
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
            let (g, name, qs) = match rnd(&mut seed) % 6 {
                0 => (GateKind::H, "h", vec![q]),
                1 => (GateKind::S, "s", vec![q]),
                2 => (GateKind::Sdg, "sdg", vec![q]),
                3 => (GateKind::X, "x", vec![q]),
                4 => (GateKind::Z, "z", vec![q]),
                _ => { let b = (rnd(&mut seed) % n as u64) as u32;
                       if q == b { (GateKind::H, "h", vec![q]) } else { (GateKind::CX, "cx", vec![q, b]) } }
            };
            c.ops.push(op(g, &qs));
            desc.push(format!("{name}:{}", qs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")));
        }
        let sv = match cpu.execute(&c, &pb, &an).unwrap() { ExecResult::Statevector(v) => v, _ => continue };
        let p_cpu: Vec<f64> = sv.iter().map(|a| a.norm_sqr()).collect();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(m) = metal.as_ref() {
            if let Ok(ExecResult::Statevector(mv)) = m.execute(&c, &pb, &an) {
                let p_metal: Vec<f64> = mv.iter().map(|a| a.norm_sqr()).collect();
                let d = p_cpu.iter().zip(p_metal.iter()).map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);
                if d > worst_metal { worst_metal = d; }
            }
        }
        let _ = &metal;
        // Stabilizer backend (exact probabilities) on the SAME circuit.
        if let Ok(ExecResult::Probabilities(p_stab)) = stab.execute(&c, &pb, &an) {
            let ds = p_cpu.iter().zip(p_stab.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f64, f64::max);
            if ds > worst_stab { worst_stab = ds; }
            let sum: f64 = p_stab.iter().sum();
            if (sum - 1.0).abs() > 1e-9 { unnorm += 1; }
        }
        // circuit  n  gates...  |  cpu probabilities
        println!("C {n} {} | {}", desc.join(" "),
                 p_cpu.iter().map(|p| format!("{p:.12}")).collect::<Vec<_>>().join(" "));
    }
    println!("#END");
    eprintln!("metal-vs-cpu   worst |Δp| over {n_circ} circuits = {worst_metal:.3e}");
    eprintln!("stab-vs-cpu    worst |Δp| = {worst_stab:.3e}, unnormalised = {unnorm}");
}
