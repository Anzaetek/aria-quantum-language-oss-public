// SPDX-License-Identifier: Apache-2.0
//! Our arm. F2 is `FermionicOp::jordan_wigner` and then a backend
//! `expectation`. Kitaev's circuit is Hadamards, which the sector backend
//! refuses, so n=2 and n=8 go through the statevector and through
//! majoranaprop's ladder seed. LUCJ is number-conserving and goes through
//! the sector backend. n=64 is majoranaprop only: the statevector is not called.

use std::path::{Path, PathBuf};
use std::time::Instant;

use omega_backend_majoranaprop::MajoranaPropBackend;
use omega_backend_sector::SectorBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, GateKind, ParamExpr};
use omega_core::executor::{Backend, Observable};
use omega_core::fermion::FermionicOp;
use omega_core::params::ParameterBinding;
use omega_parser::lower::{lower_to_ir_with_dialect, Qasm2Dialect};
use sha2::{Digest, Sha256};

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(hex(&Sha256::digest(bytes)))
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

fn read(rel: &str) -> Result<String, String> {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))
}

fn lower(rel: &str, dialect: Qasm2Dialect) -> Result<CircuitIR, String> {
    let text = read(rel)?;
    lower_to_ir_with_dialect(&text, dialect).map_err(|e| format!("{rel}: {e}"))
}

fn parse_op(rel: &str) -> Result<FermionicOp, String> {
    let text = read(rel)?;
    FermionicOp::parse(&text).map_err(|e| format!("{rel}: {e}"))
}

fn params() -> ParameterBinding {
    ParameterBinding::new()
}

fn expect(
    backend: &impl Backend,
    circuit: &CircuitIR,
    observable: &Observable,
) -> Result<f64, String> {
    observable
        .validate_qubits(circuit.num_qubits)
        .map_err(|e| e.to_string())?;
    backend
        .expectation(circuit, &params(), observable)
        .map_err(|e| e.to_string())
}

/// Negate every term that is not a single hop (one creation and one
/// annihilation). On the Kitaev operator that is the pairing.
pub fn flip_pairing(op: &FermionicOp) -> FermionicOp {
    let mut out = op.clone();
    for (coeff, prod) in &mut out.terms {
        let raises = prod.iter().filter(|l| l.dagger).count();
        let lowers = prod.len() - raises;
        if !(raises == 1 && lowers == 1) {
            *coeff = -*coeff;
        }
    }
    out
}

fn flip_concrete(expr: &mut ParamExpr) -> Result<f64, String> {
    match expr {
        ParamExpr::Concrete(value) => {
            *value = -*value;
            Ok(*value)
        }
        other => Err(format!("parameter is not a concrete angle: {other:?}")),
    }
}

/// Flip the first gate of `kind`. Returns the flipped angle.
pub fn flip_first_angle(circuit: &mut CircuitIR, kind: GateKind) -> Result<f64, String> {
    for op in &mut circuit.ops {
        if op.gate == kind {
            let expr = op
                .params
                .first_mut()
                .ok_or_else(|| format!("{kind:?} has no parameter"))?;
            return flip_concrete(expr);
        }
    }
    Err(format!("circuit has no {kind:?}"))
}

pub struct KitaevSmall {
    /// Statevector via the Jordan–Wigner image. This is the F2 number.
    pub statevector: f64,
    /// majoranaprop, seeded from the ladder operator.
    pub majorana: f64,
    /// Certificate fields, asserted by the caller.
    pub dropped_mass: f64,
    /// Monomials at the end of the propagation.
    pub final_terms: usize,
    /// `ladder` when the fermionic door ran.
    pub seed_basis: String,
    /// True when nothing was discarded.
    pub exact: bool,
}

pub fn kitaev_small(n: u32, flip: bool) -> Result<KitaevSmall, String> {
    if n != 2 && n != 8 {
        return Err(format!("kitaev_small is n=2 or n=8, got {n}"));
    }
    let rel = format!("examples/fermionic/kitaev_n{n}.qasm");
    let op_rel = format!("examples/fermionic/kitaev_op_n{n}.txt");
    let circuit = lower(&rel, Qasm2Dialect::Legacy)?;
    let mut op = parse_op(&op_rel)?;
    if flip {
        op = flip_pairing(&op);
    }
    let observable = op.jordan_wigner().map_err(|e| e.to_string())?;
    let statevector = expect(&StatevectorBackend::new(), &circuit, &observable)?;
    let (majorana, cert) = MajoranaPropBackend::new()
        .expectation_fermionic_with_certificate(&circuit, &params(), &op)
        .map_err(|e| e.to_string())?;
    if cert.seed_basis.as_str() != "ladder" {
        return Err(format!(
            "majoranaprop seed basis is {}, want ladder",
            cert.seed_basis.as_str()
        ));
    }
    Ok(KitaevSmall {
        statevector,
        majorana,
        dropped_mass: cert.dropped_mass,
        final_terms: cert.final_terms,
        seed_basis: cert.seed_basis.as_str().to_string(),
        exact: cert.is_exact(),
    })
}

pub struct Kitaev64 {
    /// Certified energy.
    pub energy: f64,
    /// Bound on |Δ⟨O⟩|. Zero on this circuit.
    pub dropped_mass: f64,
    /// One monomial per bond.
    pub final_terms: usize,
    /// Peak monomials held.
    pub peak_terms: usize,
    /// L1 range of the observable.
    pub observable_range: f64,
    /// `ladder`.
    pub seed_basis: String,
    /// Nothing discarded.
    pub exact: bool,
    /// The bound excludes something.
    pub informative: bool,
}

pub fn kitaev64(flip: bool) -> Result<Kitaev64, String> {
    let circuit = lower("examples/fermionic/kitaev_n64.qasm", Qasm2Dialect::Legacy)?;
    if circuit.num_qubits != 64 {
        return Err(format!(
            "kitaev n64 circuit has {} qubits",
            circuit.num_qubits
        ));
    }
    let mut op = parse_op("examples/fermionic/kitaev_op_n64.txt")?;
    if flip {
        op = flip_pairing(&op);
    }
    // The statevector is not constructed. 2^64 complex amplitudes is the
    // thing this row exists to not allocate.
    let (energy, cert) = MajoranaPropBackend::new()
        .expectation_fermionic_with_certificate(&circuit, &params(), &op)
        .map_err(|e| e.to_string())?;
    Ok(Kitaev64 {
        energy,
        dropped_mass: cert.dropped_mass,
        final_terms: cert.final_terms,
        peak_terms: cert.peak_terms,
        observable_range: cert.observable_range,
        seed_basis: cert.seed_basis.as_str().to_string(),
        exact: cert.is_exact(),
        informative: cert.is_informative(),
    })
}

pub fn h2(flip_rz: bool) -> Result<f64, String> {
    let mut circuit = lower("examples/fermionic/h2_ground.qasm", Qasm2Dialect::Legacy)?;
    if flip_rz {
        let flipped = flip_first_angle(&mut circuit, GateKind::Rz)?;
        if flipped <= 0.0 {
            return Err(format!(
                "h2 rz sign flip landed at {flipped}; the committed angle is negative, \
                 so the flipped angle is positive"
            ));
        }
    }
    let op = parse_op("examples/fermionic/h2_op.txt")?;
    let observable = op.jordan_wigner().map_err(|e| e.to_string())?;
    expect(&StatevectorBackend::new(), &circuit, &observable)
}

pub fn h2_reference() -> Result<f64, String> {
    let text = read("examples/fermionic/h2_reference.txt")?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("E_FCI ") {
            return rest.trim().parse().map_err(|e| format!("E_FCI: {e}"));
        }
    }
    Err("h2_reference.txt has no E_FCI".into())
}

pub fn hubbard(flip_rbs: bool) -> Result<f64, String> {
    let mut circuit = lower("examples/fermionic/hubbard2.qasm", Qasm2Dialect::Lenient)?;
    if flip_rbs {
        let flipped = flip_first_angle(&mut circuit, GateKind::Rbs)?;
        if flipped >= 0.0 {
            return Err(format!(
                "hubbard rbs sign flip landed at {flipped}; the committed angle is +π/8, \
                 so the flipped angle is negative"
            ));
        }
    }
    let op = parse_op("examples/fermionic/hubbard2_op.txt")?;
    let observable = op.jordan_wigner().map_err(|e| e.to_string())?;
    expect(&StatevectorBackend::new(), &circuit, &observable)
}

pub struct Lucj {
    /// Sector expectation of n_0. This is the timed value.
    pub sector: f64,
    /// Dense statevector of the same lowered circuit, outside the timer.
    pub statevector: f64,
    /// Electrons the occupation layer created.
    pub electrons: u32,
    /// C(16, 8).
    pub sector_dim: u128,
}

const LUCJ_QASM: &str = "tools/emu_compare/qasm/lucj16_norb8_na4_r1_s0_j0.1.qasm";

/// Qubits the occupation layer fills, as a bit mask. An `rbs` on two filled
/// modes, or on two empty modes, is the identity on the Hartree–Fock state,
/// so flipping that angle cannot move an expectation.
fn hartree_fock_mask(circuit: &CircuitIR) -> Result<u64, String> {
    let mut touched = 0u64;
    let mut occupation = 0u64;
    for (i, op) in circuit.ops.iter().enumerate() {
        if op.gate == GateKind::X {
            let q = op.qubits.first().map(|q| q.0).ok_or("X with no qubit")?;
            if q >= 64 {
                return Err(format!("occupation qubit {q} does not fit in the mask"));
            }
            if touched & (1 << q) != 0 {
                return Err(format!("op {i}: X on qubit {q} after another gate"));
            }
            occupation |= 1 << q;
            touched |= 1 << q;
        } else if op.gate != GateKind::Barrier {
            for q in &op.qubits {
                touched |= 1 << q.0;
            }
        }
    }
    Ok(occupation)
}

/// The seeded LUCJ flip: one later `rbs` on the measured mode.
///
/// The file's first `rbs` is `rbs(θ) q[1],q[0]` while both modes are still
/// filled, and `rbs` is the identity on `|11⟩`, so that sign flip moves
/// nothing. Negating every `rbs` also leaves `⟨n_0⟩` unchanged. This angle
/// is a later Givens on the same pair, after the orbital rotation has mixed
/// the occupation, and flipping it moves `⟨n_0⟩` by about 0.59.
const LUCJ_PIN_RBS: (f64, u32, u32) = (-0.9916747497747819, 1, 0);

pub fn flip_pinned_lucj_rbs(circuit: &mut CircuitIR) -> Result<f64, String> {
    let (want, qa, qb) = LUCJ_PIN_RBS;
    let mut found = 0usize;
    let mut flipped = 0.0;
    for op in &mut circuit.ops {
        if op.gate != GateKind::Rbs || op.qubits.len() != 2 || op.params.is_empty() {
            continue;
        }
        let angle = match &op.params[0] {
            ParamExpr::Concrete(v) => *v,
            _ => continue,
        };
        if op.qubits[0].0 != qa || op.qubits[1].0 != qb || (angle - want).abs() > 1e-12 {
            continue;
        }
        flipped = flip_concrete(&mut op.params[0])?;
        found += 1;
    }
    if found != 1 {
        return Err(format!(
            "pinned rbs({want}) q[{qa}],q[{qb}] occurred {found} times; the sign flip needs exactly one"
        ));
    }
    if flipped <= 0.0 {
        return Err(format!(
            "pinned rbs flipped to {flipped}; the committed angle is negative"
        ));
    }
    Ok(flipped)
}

pub fn lucj_circuit(flip_rbs: bool) -> Result<CircuitIR, String> {
    let mut circuit = lower(LUCJ_QASM, Qasm2Dialect::Lenient)?;
    if circuit.num_qubits != 16 {
        return Err(format!("LUCJ circuit has {} qubits", circuit.num_qubits));
    }
    if flip_rbs {
        let _flipped = flip_pinned_lucj_rbs(&mut circuit)?;
    }
    Ok(circuit)
}

fn number_zero() -> Result<Observable, String> {
    FermionicOp::parse("1 [0^ 0]")
        .map_err(|e| e.to_string())?
        .jordan_wigner()
        .map_err(|e| e.to_string())
}

/// Electrons created by X on a wire nothing has touched yet. This is the
/// occupation the sector backend will use.
pub fn occupation_electrons(circuit: &CircuitIR) -> Result<u32, String> {
    Ok(hartree_fock_mask(circuit)?.count_ones())
}

pub fn binomial(n: u32, k: u32) -> u128 {
    let mut acc = 1u128;
    for i in 0..k {
        acc = acc * u128::from(n - i) / u128::from(i + 1);
    }
    acc
}

pub fn lucj(flip_first_rbs: bool) -> Result<Lucj, String> {
    let circuit = lucj_circuit(flip_first_rbs)?;
    let electrons = occupation_electrons(&circuit)?;
    if !flip_first_rbs && electrons != 8 {
        return Err(format!("LUCJ occupation is {electrons} electrons, want 8"));
    }
    let observable = number_zero()?;
    let sector = expect(&SectorBackend::new(), &circuit, &observable)?;
    let statevector = expect(&StatevectorBackend::new(), &circuit, &observable)?;
    Ok(Lucj {
        sector,
        statevector,
        electrons,
        sector_dim: binomial(16, electrons),
    })
}

/// The timed region: sector expectation only. The circuit is already lowered.
pub fn time_lucj_sector(circuit: &CircuitIR) -> Result<f64, String> {
    let observable = number_zero()?;
    let started = Instant::now();
    let _value = expect(&SectorBackend::new(), circuit, &observable)?;
    Ok(started.elapsed().as_secs_f64())
}

/// Smallest sector expectation: one occupied mode, ⟨n_0⟩ = 1.
pub fn floor_sector() -> Result<f64, String> {
    let circuit = lower_to_ir_with_dialect(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nx q[0];\n",
        Qasm2Dialect::Legacy,
    )
    .map_err(|e| e.to_string())?;
    let observable = number_zero()?;
    let started = Instant::now();
    let value = expect(&SectorBackend::new(), &circuit, &observable)?;
    let seconds = started.elapsed().as_secs_f64();
    if (value - 1.0).abs() > 1e-12 {
        return Err(format!("sector floor ⟨n_0⟩ = {value}, want 1"));
    }
    Ok(seconds)
}

pub fn lucj_lower_seconds() -> Result<(CircuitIR, f64), String> {
    let started = Instant::now();
    let circuit = lucj_circuit(false)?;
    Ok((circuit, started.elapsed().as_secs_f64()))
}

pub fn file_sha(rel: &str) -> Result<String, String> {
    sha256_file(&repo_root().join(rel))
}

pub fn threads_statevector() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}
