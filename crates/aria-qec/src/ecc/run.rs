//! Execute ECC circuits on a selectable omega-functions simulator backend.
//!
//! ECC code construction + MWPM decoding live in pure quantum-core
//! ([`super::codes`], [`super::mwpm`]); this module is the *execution* glue
//! that actually runs a syndrome-extraction circuit and reads back the
//! measured syndrome. It is compiled only with the `omega-sim` feature, which
//! links the sibling omega-functions backends:
//!
//! * **statevector** — dense `2^n` state, exact (universal).
//! * **mps** — matrix-product-state, truncated bond dim (universal).
//! * **stabilizer** / **pauli** — Aaronson–Gottesman Clifford tableau; the
//!   natural, scalable simulator for the (pure-Clifford) surface-code
//!   syndrome cycle. `stabilizer` and `pauli` are aliases for the same engine.
//! * **pauliprop** — Pauli propagation (Heisenberg-picture Pauli-string tree).
//!   Reads syndromes from each check's *expectation value* `⟨C⟩ = ±1` rather
//!   than a projective shot, so it has no ancilla/measurement and is exempt
//!   from the 64-bit `Counts`-key cap — the only backend that reaches d ≥ 7.
//!
//! The measurement backends go through `run_counts` (`Collapse` projective
//! measurement); the expectation backend goes through `expectation_syndrome`.
//!
//! The bridge reuses [`aria_core::backends::omega::to_omega_ir`] to flatten the
//! quantum-core AST, then maps that mirror IR onto `omega_core::CircuitIR` and
//! runs it under projective ([`MidCircuitMode::Collapse`]) measurement.

use super::codes::{QECCode, SurfaceCode};
use super::mwpm::{decode_mwpm_correction, Correction};
use aria_core::ast::nodes::{Circuit, GateDef, GateKind, Instruction, Qubit};
use aria_core::ast::CircuitBuilder;
use aria_core::backends::omega::{to_omega_ir, OmegaGateKind};

use omega_core::circuit::{
    CircuitIR, CircuitType, GateKind as OGate, GateOp, ParamExpr, Qubit as OQubit,
};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable, PauliOp};
use omega_core::params::ParameterBinding;

/// Default MPS bond dimension — ample for the small (d ≤ 5) surface codes the
/// demo exercises, where the syndrome state has modest entanglement.
const MPS_BOND_DIM: usize = 64;

/// A selectable simulator backend for ECC execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimBackend {
    Statevector,
    /// Clifford tableau (stabilizer ≡ pauli).
    Stabilizer,
    Mps,
    /// Pauli propagation (Heisenberg-picture Pauli-string tree). Computes
    /// syndromes from check **expectation values**, not projective shots, so it
    /// has no ancilla/measurement and scales past the 64-bit `Counts`-key cap
    /// (d ≥ 7). Exact for the pure-Clifford syndrome cycle. See
    /// `PAULI_PROPAGATION_PLAN.md` (item 17).
    PauliProp,
}

impl SimBackend {
    /// Parse a backend name. `stabilizer` and `pauli` both select the
    /// Clifford tableau engine; `pauliprop`/`pp` selects Pauli propagation.
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "statevector" | "sv" => Some(Self::Statevector),
            "stabilizer" | "pauli" | "clifford" => Some(Self::Stabilizer),
            "mps" | "tensor" => Some(Self::Mps),
            "pauliprop" | "pp" => Some(Self::PauliProp),
            _ => None,
        }
    }

    /// Canonical display name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Statevector => "statevector",
            Self::Stabilizer => "stabilizer",
            Self::Mps => "mps",
            Self::PauliProp => "pauliprop",
        }
    }

    /// Whether this backend reads syndromes from expectation values (no
    /// projective measurement / ancilla / 64-bit `Counts` key).
    pub fn is_expectation(self) -> bool {
        matches!(self, Self::PauliProp)
    }

    /// Construct the boxed omega backend instance for this selection.
    pub fn backend(self) -> Box<dyn Backend> {
        match self {
            Self::Statevector => Box::new(omega_backend_statevector::StatevectorBackend::new()),
            Self::Stabilizer => Box::new(omega_backend_pauli::PauliBackend::new()),
            Self::Mps => Box::new(omega_backend_mps::MpsBackend::new(MPS_BOND_DIM)),
            Self::PauliProp => Box::new(omega_backend_pauliprop::PauliPropBackend::new()),
        }
    }
}

/// Map the quantum-core mirror gate kind onto the omega-core gate kind. The
/// two enums are intentionally identical in shape, so this is total for every
/// non-photonic gate the ECC circuits emit.
fn map_gate(g: &OmegaGateKind) -> OGate {
    use OmegaGateKind as Q;
    match g {
        Q::H => OGate::H,
        Q::X => OGate::X,
        Q::Y => OGate::Y,
        Q::Z => OGate::Z,
        Q::S => OGate::S,
        Q::Sdg => OGate::Sdg,
        Q::T => OGate::T,
        Q::Tdg => OGate::Tdg,
        Q::Id => OGate::Id,
        Q::Rx => OGate::Rx,
        Q::Ry => OGate::Ry,
        Q::Rz => OGate::Rz,
        Q::U3 => OGate::U3,
        Q::U2 => OGate::U2,
        Q::U1 => OGate::U1,
        Q::CX => OGate::CX,
        Q::CY => OGate::CY,
        Q::CZ => OGate::CZ,
        Q::Swap => OGate::Swap,
        Q::CRz => OGate::CRz,
        Q::CU3 => OGate::CU3,
        Q::CCX => OGate::CCX,
        Q::CSwap => OGate::CSwap,
        Q::PhaseShifter => OGate::PhaseShifter,
        Q::BeamSplitterRx => OGate::BeamSplitterRx,
        Q::Measure => OGate::Measure,
        Q::Barrier => OGate::Barrier,
        Q::Reset => OGate::Reset,
    }
}

/// Lower a quantum-core [`Circuit`] to an `omega_core::CircuitIR`.
pub fn to_omega_core_ir(circuit: &Circuit) -> CircuitIR {
    let mirror = to_omega_ir(circuit);
    let mut ir = CircuitIR::new(mirror.num_qubits, CircuitType::GateBased);
    ir.num_classical_bits = mirror.num_classical_bits;
    for op in &mirror.ops {
        ir.add_op(GateOp {
            gate: map_gate(&op.gate),
            qubits: op.qubits.iter().map(|q| OQubit(*q)).collect(),
            params: op.params.iter().map(|p| ParamExpr::Concrete(*p)).collect(),
            classical_bit: op.classical_bit,
            // quantum-core conditions are single-bit `(bit, value)`; omega uses
            // `(start_bit, num_bits, expected)`.
            condition: op.condition.map(|(bit, val)| (bit, 1, val)),
        });
    }
    ir
}

/// Run `circuit` on `backend` with projective measurement, returning the raw
/// `bitstring -> count` map (bitstring = integer over classical bits, LSB =
/// classical bit 0). `shots` outcomes are drawn with the given `seed`.
pub fn run_counts(
    circuit: &Circuit,
    backend: SimBackend,
    shots: u32,
    seed: u64,
) -> Result<std::collections::HashMap<u64, u32>, String> {
    let ir = to_omega_core_ir(circuit);
    let cfg = ExecConfig {
        shots: Some(shots),
        seed: Some(seed),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let params = ParameterBinding::new();
    match backend.backend().execute(&ir, &params, &cfg) {
        Ok(ExecResult::Counts(c)) => Ok(c),
        Ok(_) => Err("backend returned a non-counts result".into()),
        Err(e) => Err(format!("{e:?}")),
    }
}

/// Read back `n_syndrome` deterministic ancilla bits from one projective shot.
///
/// The omega backends disagree on how they key the `Counts` bitstring under
/// `MidCircuitMode::Collapse`: the **statevector** backend keys by classical
/// register (so syndrome bit `k` is key bit `k`), whereas the **stabilizer**
/// and **MPS** backends re-sample every *qubit* at the end (so syndrome bit `k`
/// — measured into ancilla qubit `n_data + k` — is key bit `n_data + k`). We
/// absorb that difference with a backend-dependent `key_base` offset so the
/// caller always gets the syndrome in check order. The state is a stabilizer
/// eigenstate, so one shot fixes the whole (deterministic) syndrome.
pub fn syndrome_bits(
    circuit: &Circuit,
    backend: SimBackend,
    n_syndrome: usize,
    n_data: usize,
    seed: u64,
) -> Result<Vec<u8>, String> {
    let counts = run_counts(circuit, backend, 1, seed)?;
    let key = counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(k, _)| k)
        .unwrap_or(0);
    let key_base = match backend {
        SimBackend::Statevector => 0,
        SimBackend::Stabilizer | SimBackend::Mps => n_data,
        // PauliProp never reaches this measurement-key path (it reads
        // expectations); keep the match exhaustive.
        SimBackend::PauliProp => n_data,
    };
    Ok((0..n_syndrome)
        .map(|i| ((key >> (key_base + i)) & 1) as u8)
        .collect())
}

// ---------------------------------------------------------------------------
// Surface-code experiment orchestration (two deterministic CSS sectors)
// ---------------------------------------------------------------------------

fn err_gate(kind: GateKind, q: usize) -> Instruction {
    Instruction {
        gate: GateDef::new(kind),
        qubits: vec![Qubit::new("q", q)],
        clbits: vec![],
        condition: None,
    }
}

/// Bit-flip sector: data |0…0⟩, inject X errors on `x_err`, measure the Z-type
/// checks (which detect X errors). Z-checks are diagonal, so the resulting
/// syndrome is deterministic and identical on every backend.
pub fn bitflip_circuit(code: &SurfaceCode, x_err: &[usize]) -> Circuit {
    let n_data = code.n_physical();
    let z = code.z_checks().to_vec();
    let mut circ = code_z_measure(code, n_data, &z);
    let mut instrs: Vec<Instruction> = x_err.iter().map(|&q| err_gate(GateKind::X, q)).collect();
    instrs.append(&mut circ.instructions);
    circ.instructions = instrs;
    circ
}

/// Phase-flip sector: data |+…+⟩ (Hadamard the whole register), inject Z errors
/// on `z_err`, measure the X-type checks (which detect Z errors). The Hadamard
/// dual of the bit-flip sector; also deterministic across backends.
pub fn phaseflip_circuit(code: &SurfaceCode, z_err: &[usize]) -> Circuit {
    let n_data = code.n_physical();
    let x = code.x_checks().to_vec();
    let n_anc = x.len();
    let mut b = CircuitBuilder::new("surface_phaseflip", n_data + n_anc, n_anc);
    for q in 0..n_data {
        b.h(q); // prepare |+…+⟩
    }
    for &q in z_err {
        b.z(q);
    }
    for (k, check) in x.iter().enumerate() {
        let anc = n_data + k;
        b.h(anc);
        for &q in check {
            b.cx(anc, q);
        }
        b.h(anc);
        b.measure(anc, k);
    }
    b.build()
}

/// Z-check measurement circuit (no error prefix) for the bit-flip sector.
fn code_z_measure(_code: &SurfaceCode, n_data: usize, z: &[Vec<usize>]) -> Circuit {
    let n_anc = z.len();
    let mut b = CircuitBuilder::new("surface_bitflip", n_data + n_anc, n_anc);
    for (k, check) in z.iter().enumerate() {
        let anc = n_data + k;
        for &q in check {
            b.cx(q, anc);
        }
        b.measure(anc, k);
    }
    b.build()
}

/// Z-type check syndrome (bit-flip sector) under injected X errors, read off
/// `backend`. One projective shot fixes the whole deterministic syndrome.
///
/// The expectation backends (PauliProp) skip the ancilla/measurement circuit
/// entirely: they prepare the data state (`|0…0⟩` + the injected `X`s) and read
/// each Z-check's expectation value directly — exact, and free of the 64-bit
/// measurement-key cap, so they extend past d=5.
pub fn bitflip_syndrome(
    code: &SurfaceCode,
    x_err: &[usize],
    backend: SimBackend,
    seed: u64,
) -> Result<Vec<u8>, String> {
    if backend.is_expectation() {
        let n_data = code.n_physical();
        let mut b = CircuitBuilder::new("bitflip_data", n_data, 0);
        for &q in x_err {
            b.x(q);
        }
        return expectation_syndrome(&b.build(), code.z_checks(), PauliOp::Z, backend);
    }
    let circ = bitflip_circuit(code, x_err);
    syndrome_bits(
        &circ,
        backend,
        code.z_checks().len(),
        code.n_physical(),
        seed,
    )
}

/// X-type check syndrome (phase-flip sector) under injected Z errors.
pub fn phaseflip_syndrome(
    code: &SurfaceCode,
    z_err: &[usize],
    backend: SimBackend,
    seed: u64,
) -> Result<Vec<u8>, String> {
    if backend.is_expectation() {
        let n_data = code.n_physical();
        let mut b = CircuitBuilder::new("phaseflip_data", n_data, 0);
        for q in 0..n_data {
            b.h(q); // |+…+⟩
        }
        for &q in z_err {
            b.z(q);
        }
        return expectation_syndrome(&b.build(), code.x_checks(), PauliOp::X, backend);
    }
    let circ = phaseflip_circuit(code, z_err);
    syndrome_bits(
        &circ,
        backend,
        code.x_checks().len(),
        code.n_physical(),
        seed,
    )
}

/// Syndrome from Pauli **expectation values** (the PauliProp path): for each
/// check operator `C` (a product of `op` over its data-qubit support), read
/// `⟨C⟩` on the prepared data state. On a stabilizer eigenstate `⟨C⟩ = ±1`, so
/// the syndrome bit is `(1 − ⟨C⟩)/2`.
fn expectation_syndrome(
    data_circuit: &Circuit,
    checks: &[Vec<usize>],
    op: PauliOp,
    backend: SimBackend,
) -> Result<Vec<u8>, String> {
    let ir = to_omega_core_ir(data_circuit);
    let be = backend.backend();
    let params = ParameterBinding::new();
    let mut bits = Vec::with_capacity(checks.len());
    for support in checks {
        let obs = Observable {
            terms: vec![(
                1.0,
                support.iter().map(|&q| (q as u32, op.clone())).collect(),
            )],
        };
        let ev = be
            .expectation(&ir, &params, &obs)
            .map_err(|e| format!("{e:?}"))?;
        bits.push(if ev < 0.0 { 1u8 } else { 0u8 });
    }
    Ok(bits)
}

/// Outcome of a single decode trial on the surface code.
#[derive(Clone, Debug)]
pub struct Trial {
    /// X-type check bits (phase-flip sector; flag Z errors).
    pub x_check_syndrome: Vec<u8>,
    /// Z-type check bits (bit-flip sector; flag X errors).
    pub z_check_syndrome: Vec<u8>,
    /// Minimum-weight correction.
    pub correction: Correction,
    /// Syndrome weight remaining after the correction (0 for a valid decode).
    pub residual_syndrome_weight: usize,
    /// `true` if the residual error is a non-trivial logical operator.
    pub logical_failure: bool,
}

impl Trial {
    /// Total number of triggered stabilizers across both sectors.
    pub fn syndrome_weight(&self) -> usize {
        self.x_check_syndrome.iter().filter(|&&b| b == 1).count()
            + self.z_check_syndrome.iter().filter(|&&b| b == 1).count()
    }
}

/// Symmetric difference of two qubit-index sets (XOR of Pauli error supports).
fn sym_diff(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut set: std::collections::BTreeSet<usize> = a.iter().copied().collect();
    for &q in b {
        if !set.remove(&q) {
            set.insert(q);
        }
    }
    set.into_iter().collect()
}

/// Parity syndrome of an error support against a set of checks.
fn parity_syndrome(error: &[usize], checks: &[Vec<usize>]) -> Vec<u8> {
    let eset: std::collections::BTreeSet<usize> = error.iter().copied().collect();
    checks
        .iter()
        .map(|c| (c.iter().filter(|q| eset.contains(q)).count() % 2) as u8)
        .collect()
}

/// A residual X error causes a logical X failure iff it anticommutes with the
/// logical-Z observable (odd overlap with its support); symmetrically for Z.
fn is_logical_failure(code: &SurfaceCode, residual_x: &[usize], residual_z: &[usize]) -> bool {
    let lz: std::collections::BTreeSet<usize> = code.logical_z().into_iter().collect();
    let lx: std::collections::BTreeSet<usize> = code.logical_x().into_iter().collect();
    let x_fail = residual_x.iter().filter(|q| lz.contains(q)).count() % 2 == 1;
    let z_fail = residual_z.iter().filter(|q| lx.contains(q)).count() % 2 == 1;
    x_fail || z_fail
}

/// Run one full decode trial: inject `x_err`/`z_err`, extract both sector
/// syndromes on `backend`, minimum-weight-decode, then verify the residual is
/// syndrome-trivial and classify a logical failure.
pub fn decode_trial(
    code: &SurfaceCode,
    x_err: &[usize],
    z_err: &[usize],
    backend: SimBackend,
    seed: u64,
) -> Result<Trial, String> {
    let z_check_syndrome = bitflip_syndrome(code, x_err, backend, seed)?;
    let x_check_syndrome = phaseflip_syndrome(code, z_err, backend, seed)?;

    // Decoder takes the full syndrome (X-check bits, then Z-check bits).
    let mut full = x_check_syndrome.clone();
    full.extend_from_slice(&z_check_syndrome);
    let correction = decode_mwpm_correction(code, &full);

    let residual_x = sym_diff(x_err, &correction.x_flips);
    let residual_z = sym_diff(z_err, &correction.z_flips);

    // Residual syndrome (algebraic — Z-checks see residual X, X-checks see
    // residual Z). Zero ⇔ the correction returned the state to the codespace.
    let rz = parity_syndrome(&residual_x, code.z_checks());
    let rx = parity_syndrome(&residual_z, code.x_checks());
    let residual_syndrome_weight =
        rz.iter().filter(|&&b| b == 1).count() + rx.iter().filter(|&&b| b == 1).count();

    let logical_failure = is_logical_failure(code, &residual_x, &residual_z);

    Ok(Trial {
        x_check_syndrome,
        z_check_syndrome,
        correction,
        residual_syndrome_weight,
        logical_failure,
    })
}

/// Deterministic SplitMix64 → uniform `f64` in `[0, 1)`, for seeded Monte-Carlo
/// error sampling that replays identically across backends and runs. The single
/// shared stream lives in [`crate::logical::noise`]; this alias keeps the local
/// call sites unchanged.
use crate::logical::noise::splitmix64 as splitmix;

/// Result of a Monte-Carlo logical-error-rate sweep.
#[derive(Clone, Debug)]
pub struct MonteCarlo {
    pub trials: u32,
    pub physical_rate: f64,
    pub logical_failures: u32,
    pub logical_rate: f64,
    /// Fraction of trials whose correction cleared the syndrome (residual 0).
    pub decode_success_rate: f64,
}

/// Estimate the logical error rate under an i.i.d. depolarizing-style channel:
/// each data qubit independently gets an X with prob `p`, a Z with prob `p`
/// (Y when both fire). Errors are seeded so every backend sees the same draws.
pub fn monte_carlo(
    code: &SurfaceCode,
    backend: SimBackend,
    p: f64,
    shots: u32,
    seed: u64,
) -> Result<MonteCarlo, String> {
    let n_data = code.n_physical();
    let mut rng = seed ^ 0xA5A5_5A5A_DEAD_BEEF;
    let mut logical_failures = 0u32;
    let mut decode_success = 0u32;
    for _ in 0..shots {
        let mut x_err = Vec::new();
        let mut z_err = Vec::new();
        for q in 0..n_data {
            if splitmix(&mut rng) < p {
                x_err.push(q);
            }
            if splitmix(&mut rng) < p {
                z_err.push(q);
            }
        }
        // Reuse a fixed per-trial sim seed: the syndrome is deterministic given
        // the (stabilizer-eigenstate) error, so the seed only needs to be valid.
        let trial = decode_trial(code, &x_err, &z_err, backend, seed)?;
        if trial.residual_syndrome_weight == 0 {
            decode_success += 1;
        }
        if trial.logical_failure {
            logical_failures += 1;
        }
    }
    Ok(MonteCarlo {
        trials: shots,
        physical_rate: p,
        logical_failures,
        logical_rate: logical_failures as f64 / shots as f64,
        decode_success_rate: decode_success as f64 / shots as f64,
    })
}

/// Differential fuzzing across the omega backends (item 21e/21f).
///
/// Closes the "backends only cross-checked on the hand-built surface-code
/// syndrome cycle" gap: on *random* circuits, the omega statevector / MPS /
/// Pauli-propagation engines must agree on every Pauli expectation (where each
/// is exact), and the statevector engine must preserve normalisation (the
/// unitarity invariant). Reuses the production [`to_omega_core_ir`] lowering and
/// [`SimBackend`] selection, so it also exercises the quantum-core → omega IR
/// bridge. Gated on `omega-sim` (links the sibling backend crates); runs under
/// the `ecc::` filter in `scripts/ci.sh`.
#[cfg(test)]
mod differential_fuzz {
    use super::*;
    use aria_core::ast::builder::CircuitBuilder;

    /// Seeded LCG → uniform `[0,1)`.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Build a random circuit on `n` qubits with `depth` gates. When
    /// `clifford_only` the gate set is {H,S,X,Y,Z,CX,CZ} (exact for *all* three
    /// backends, including Pauli propagation); otherwise rotations + T are mixed
    /// in (exact only for statevector / MPS).
    fn random_circuit(n: usize, depth: usize, clifford_only: bool, s: &mut u64) -> Circuit {
        let mut b = CircuitBuilder::new("fuzz", n, 0);
        for _ in 0..depth {
            if n >= 2 && lcg(s) < 0.4 {
                let a = (lcg(s) * n as f64) as usize;
                let mut c = (lcg(s) * n as f64) as usize;
                if c == a {
                    c = (a + 1) % n;
                }
                if lcg(s) < 0.5 {
                    b.cx(a, c);
                } else {
                    b.cz(a, c);
                }
            } else {
                let q = (lcg(s) * n as f64) as usize;
                let pick = if clifford_only { 5.0 } else { 8.0 };
                match (lcg(s) * pick) as usize {
                    0 => b.h(q),
                    1 => b.s(q),
                    2 => b.x(q),
                    3 => b.y(q),
                    4 => b.z(q),
                    5 => b.rx(q, lcg(s) * std::f64::consts::TAU),
                    6 => b.ry(q, lcg(s) * std::f64::consts::TAU),
                    _ => b.rz(q, lcg(s) * std::f64::consts::TAU),
                };
            }
        }
        b.build()
    }

    fn pauli_obs(q: usize, op: PauliOp) -> Observable {
        Observable {
            terms: vec![(1.0, vec![(q as u32, op)])],
        }
    }

    /// ⟨Z_q⟩ and ⟨X_q⟩ on a backend via its exact expectation path.
    fn expectations(ir: &CircuitIR, n: usize, be: SimBackend) -> Vec<f64> {
        let backend = be.backend();
        let params = ParameterBinding::new();
        let mut out = Vec::with_capacity(2 * n);
        for q in 0..n {
            for op in [PauliOp::Z, PauliOp::X] {
                out.push(
                    backend
                        .expectation(ir, &params, &pauli_obs(q, op.clone()))
                        .expect("expectation"),
                );
            }
        }
        out
    }

    #[test]
    fn fuzz_clifford_backends_agree() {
        // Property (21e): on random Clifford circuits, statevector / MPS /
        // pauliprop are all exact ⇒ every single-qubit Z/X expectation must
        // agree, and |⟨P⟩| ≤ 1 (the normalisation/unitarity bound).
        let mut s = 0x0DDB_A011u64;
        for _ in 0..120 {
            let n = 2 + (lcg(&mut s) * 3.0) as usize; // 2..=4
            let depth = 4 + (lcg(&mut s) * 16.0) as usize; // 4..=19
            let circ = random_circuit(n, depth, true, &mut s);
            let ir = to_omega_core_ir(&circ);
            let sv = expectations(&ir, n, SimBackend::Statevector);
            let mps = expectations(&ir, n, SimBackend::Mps);
            let pp = expectations(&ir, n, SimBackend::PauliProp);
            for k in 0..sv.len() {
                assert!(sv[k].abs() <= 1.0 + 1e-9, "‖ψ‖ violated: ⟨P⟩={}", sv[k]);
                assert!(
                    (sv[k] - mps[k]).abs() < 1e-6,
                    "sv/mps disagree {} vs {} (n={n}, depth={depth})",
                    sv[k],
                    mps[k]
                );
                assert!(
                    (sv[k] - pp[k]).abs() < 1e-6,
                    "sv/pauliprop disagree {} vs {} (n={n}, depth={depth})",
                    sv[k],
                    pp[k]
                );
            }
        }
    }

    #[test]
    fn fuzz_universal_statevector_vs_mps_agree() {
        // Property (21e): on random *universal* circuits (rotations + T), the
        // two exact dense/tensor engines must still agree. Pauliprop is excluded
        // — it is approximate on non-Clifford input.
        let mut s = 0xC0FF_EE42u64;
        for _ in 0..120 {
            let n = 2 + (lcg(&mut s) * 3.0) as usize; // 2..=4
            let depth = 4 + (lcg(&mut s) * 16.0) as usize;
            let circ = random_circuit(n, depth, false, &mut s);
            let ir = to_omega_core_ir(&circ);
            let sv = expectations(&ir, n, SimBackend::Statevector);
            let mps = expectations(&ir, n, SimBackend::Mps);
            for k in 0..sv.len() {
                assert!(
                    (sv[k] - mps[k]).abs() < 1e-6,
                    "sv/mps disagree {} vs {} (n={n}, depth={depth})",
                    sv[k],
                    mps[k]
                );
            }
        }
    }

    #[test]
    fn fuzz_statevector_preserves_norm() {
        // Property (21f): a random circuit lowered through omega and run on the
        // statevector engine yields a normalised state — Σ|aᵢ|² = 1 — the
        // numeric witness that the lowering + gate application stayed unitary.
        let mut s = 0xBADC0DEu64;
        let cfg = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Collapse,
        };
        let params = ParameterBinding::new();
        let be = SimBackend::Statevector.backend();
        for _ in 0..150 {
            let n = 1 + (lcg(&mut s) * 4.0) as usize; // 1..=4
            let depth = 4 + (lcg(&mut s) * 16.0) as usize;
            let circ = random_circuit(n, depth, false, &mut s);
            let ir = to_omega_core_ir(&circ);
            match be.execute(&ir, &params, &cfg).expect("execute") {
                ExecResult::Statevector(sv) => {
                    let norm2: f64 = sv.iter().map(|a| a.norm_sqr()).sum();
                    assert!(
                        (norm2 - 1.0).abs() < 1e-9,
                        "‖ψ‖²={norm2} (n={n}, depth={depth})"
                    );
                }
                other => panic!("expected statevector, got {other:?}"),
            }
        }
    }
}
