// SPDX-License-Identifier: Apache-2.0
//! Q3.2 (PLAN-QUDIT.md §Q3): `MpsBackend` evolves qudit circuits, measured
//! against the exact mixed-radix engine `omega-backend-quditsv`.
//!
//! The plan's four gates, and the extras the brief asked for:
//!
//! * **(i) exact regime** — random mixed-radix circuits (dims from {2,3,4},
//!   n = 4..6, depth 30; H/X/Z/Rxy/CSum on any wire, qubit gates on d = 2
//!   wires; adjacent AND distant pairs) through `MpsBackend` at a bond that
//!   never truncates, statevector vs `quditsv::sim::run` at 1e-10. Before
//!   Q3.2 every one of these runs was REFUSED at the door (Q1), so a pass
//!   requires the new path to have executed (the plan's A10 property).
//! * **(ii) truncation contract** — the qubit lane's certificate contract,
//!   transferred verbatim to a qutrit chain on a χ sweep; see
//!   [`qutrit_truncation_keeps_the_qubit_lanes_certificate_contract`].
//! * **(iii)** the existing qubit estate is untouched (the other files in
//!   `tests/` and the unit tests in `src/`).
//! * **(iv)** the mutation check is done by hand; the report records it.
//! * Sampling marginals and the `Collapse` measurement on a qutrit, against
//!   hand projections of the exact vector.
//! * The refusal line: every construct is accepted or refused by BOTH
//!   engines, with the same sentence after the engine prefix.

use num_complex::Complex64;
use omega_backend_mps::mps::Mps;
use omega_backend_mps::{MpsBackend, SvdKernel, DEFAULT_MAX_DISCARDED_WEIGHT};
use omega_backend_quditsv::gates as qg;
use omega_backend_quditsv::sim::{run as exact_run, State};
use omega_backend_quditsv::QuditSvBackend;
use omega_core::circuit::{
    CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit, QuditRegister,
};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable, PauliOp};
use omega_core::params::ParameterBinding;

type C = Complex64;

fn op(gate: GateKind, wires: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: wires.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// A circuit on wires of the given dimensions, declared as one register.
fn qudits(dims: &[u32], ops: Vec<GateOp>) -> CircuitIR {
    let mut c = CircuitIR::new(dims.len() as u32, CircuitType::GateBased);
    c.qudit_registers.push(QuditRegister {
        name: "q".into(),
        start: 0,
        dims: dims.to_vec(),
    });
    c.ops = ops;
    c
}

fn analytic() -> ExecConfig {
    ExecConfig {
        shots: None,
        seed: Some(7),
        mid_circuit_mode: MidCircuitMode::Skip,
    }
}

fn mps_statevector(b: &MpsBackend, c: &CircuitIR, cfg: &ExecConfig) -> Vec<C> {
    match b.execute(c, &ParameterBinding::new(), cfg) {
        Ok(ExecResult::Statevector(v)) => v,
        Ok(other) => panic!("expected a statevector, got {other:?}"),
        Err(e) => panic!("mps refused: {e}"),
    }
}

fn exact(c: &CircuitIR) -> State {
    exact_run(c, &ParameterBinding::new()).unwrap_or_else(|e| panic!("quditsv refused: {e}"))
}

fn max_abs_diff(a: &[C], b: &[C]) -> f64 {
    assert_eq!(a.len(), b.len(), "statevector lengths differ");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

/// `|⟨a|b⟩|² / (‖a‖²‖b‖²)` — the true fidelity, exactly as
/// `tests/fidelity_estimate.rs` computes it.
fn fidelity(a: &[C], b: &[C]) -> f64 {
    let mut ip = C::new(0.0, 0.0);
    let (mut na, mut nb) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        ip += x.conj() * y;
        na += x.norm_sqr();
        nb += y.norm_sqr();
    }
    ip.norm_sqr() / (na * nb)
}

/// Deterministic LCG, the generator `quditsv`'s differential uses.
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

/// Counts of what a generated corpus exercised, so a generator that silently
/// stopped drawing some arm cannot pass.
#[derive(Default, Debug)]
struct Coverage {
    generalised_on_qudit: usize,
    rxy_on_qudit: usize,
    csum_adjacent: usize,
    csum_distant: usize,
    csum_mixed_dims: usize,
    qubit_gates: usize,
    qubit_pairs_distant: usize,
}

/// A random mixed-radix circuit: H/X/Z and Rxy anywhere, CSum on any pair
/// (either order, any distance), and the qubit gates only where every wire
/// they name is a qubit — the line both engines draw.
fn random_mixed_circuit(
    dims: &[u32],
    depth: usize,
    rng: &mut Lcg,
    cov: &mut Coverage,
) -> CircuitIR {
    let n = dims.len() as u32;
    let qubit_wires: Vec<u32> = (0..n).filter(|&w| dims[w as usize] == 2).collect();
    let mut ops = Vec::with_capacity(depth);
    while ops.len() < depth {
        match rng.below(6) {
            0 => {
                let w = rng.below(n);
                let g = [GateKind::H, GateKind::X, GateKind::Z][rng.below(3) as usize].clone();
                if dims[w as usize] != 2 {
                    cov.generalised_on_qudit += 1;
                }
                ops.push(op(g, &[w], &[]));
            }
            1 | 2 => {
                let w = rng.below(n);
                let d = dims[w as usize];
                let i = rng.below(d - 1);
                let j = i + 1 + rng.below(d - 1 - i);
                if d != 2 {
                    cov.rxy_on_qudit += 1;
                }
                ops.push(op(
                    GateKind::Rxy,
                    &[w],
                    &[i as f64, j as f64, rng.angle(), rng.angle()],
                ));
            }
            3 | 4 => {
                let (a, b) = rng.pair(n);
                if a.abs_diff(b) == 1 {
                    cov.csum_adjacent += 1;
                } else {
                    cov.csum_distant += 1;
                }
                if dims[a as usize] != dims[b as usize] {
                    cov.csum_mixed_dims += 1;
                }
                ops.push(op(GateKind::CSum, &[a, b], &[]));
            }
            _ => {
                if qubit_wires.is_empty() {
                    continue;
                }
                cov.qubit_gates += 1;
                let pick =
                    |rng: &mut Lcg| qubit_wires[rng.below(qubit_wires.len() as u32) as usize];
                let one_q = |rng: &mut Lcg, w: u32| match rng.below(8) {
                    0 => op(GateKind::Y, &[w], &[]),
                    1 => op(GateKind::S, &[w], &[]),
                    2 => op(GateKind::T, &[w], &[]),
                    3 => op(GateKind::Sx, &[w], &[]),
                    4 => op(GateKind::Rx, &[w], &[rng.angle()]),
                    5 => op(GateKind::Ry, &[w], &[rng.angle()]),
                    6 => op(GateKind::Rz, &[w], &[rng.angle()]),
                    _ => op(GateKind::U3, &[w], &[rng.angle(), rng.angle(), rng.angle()]),
                };
                if qubit_wires.len() >= 2 && rng.below(2) == 0 {
                    let a = pick(rng);
                    let mut b = pick(rng);
                    while b == a {
                        b = pick(rng);
                    }
                    if a.abs_diff(b) > 1 {
                        cov.qubit_pairs_distant += 1;
                    }
                    let g = match rng.below(5) {
                        0 => op(GateKind::CX, &[a, b], &[]),
                        1 => op(GateKind::CZ, &[a, b], &[]),
                        2 => op(GateKind::Swap, &[a, b], &[]),
                        3 => op(GateKind::CRz, &[a, b], &[rng.angle()]),
                        _ => op(GateKind::Rbs, &[a, b], &[rng.angle()]),
                    };
                    ops.push(g);
                } else {
                    let w = pick(rng);
                    ops.push(one_q(rng, w));
                }
            }
        }
    }
    qudits(dims, ops)
}

// ---------------------------------------------------------------------------
// (i) exact regime
// ---------------------------------------------------------------------------

/// A bond no cut of these chains can reach: the widest cut of six sites of
/// d ≤ 4 has Schmidt rank at most 4³ = 64.
const EXACT_CHI: usize = 256;

thread_local! {
    /// Largest `discarded_weight` an exact-regime run in this thread reported.
    static MAX_EXACT_DW: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

/// `(max |Δamp|, max_bond_reached)` for one circuit, asserting the exact
/// certificate on the way.
fn exact_regime_case(dims: &[u32], c: &CircuitIR) -> (f64, usize) {
    let b = MpsBackend::new(EXACT_CHI);
    let got = mps_statevector(&b, c, &analytic());
    let want = exact(c).amp;
    let err = max_abs_diff(&got, &want);
    let st = b.last_run_stats();
    // Exactness is a property of the certificate too, not only of the vector.
    // The tolerance is not `== 0.0`: the SVD keeps only σ > 1e-14, so a
    // rank-deficient split (a CSum on a product pair, say) books its
    // numerically-zero tail as discarded. That is true at d = 2 as well.
    //
    // **The floor is the KERNEL's, and that is why it is read off the
    // certificate rather than written as one constant.** The Jacobi kernel
    // books the true tail, ~1e-32, so it is held to the qubit lane's 1e-15
    // (`certificate_predicates.rs::an_exact_run_reports_its_exact_peak_bond`).
    // Accelerate `zgesdd` cannot make that claim: once it has dropped a
    // singular value it adds the backward-error allowance (m+n)·ε·‖A‖²_F,
    // because a bidiagonalisation-based factorisation does not know its own
    // dropped tail to better than that — and it cannot tell "I dropped exact
    // zeros" from "I dropped something at my own error floor". So the same
    // exact run certifies ~1.5e-14 there, measured. Both are exact at any scale
    // a consumer reads the certificate at, and the number is the arithmetic's
    // floor rather than truncation the circuit caused; the amplitude comparison
    // against quditsv at 1e-10, which is what "exact" actually means here, is
    // unaffected either way. Keying on the reported kernel keeps the tight
    // bound tight where it is reachable instead of relaxing it for everyone —
    // a Linux run is still held to 1e-15.
    let exact_certificate_floor = match st.svd_kernel {
        SvdKernel::Jacobi => 1e-15,
        SvdKernel::AccelerateZgesdd | SvdKernel::Custom => 1e-12,
    };
    assert!(
        st.discarded_weight <= exact_certificate_floor,
        "dims {dims:?}: an untruncated run on the {} kernel certified {} discarded, \
         above that kernel's floor of {exact_certificate_floor:e}",
        st.svd_kernel.as_str(),
        st.discarded_weight
    );
    // Π(1 − εᵢ) differs from 1 by Σεᵢ to first order, and Σεᵢ IS
    // `discarded_weight`, so the fidelity estimate sits under the same kernel
    // floor rather than a second constant that would have to be re-derived.
    assert!(
        (st.fidelity_estimate - 1.0).abs() <= exact_certificate_floor,
        "dims {dims:?}: fidelity estimate {} on the {} kernel",
        st.fidelity_estimate,
        st.svd_kernel.as_str()
    );
    MAX_EXACT_DW.with(|m| m.set(m.get().max(st.discarded_weight)));
    let widest_cut: u128 = (0..=dims.len())
        .map(|i| {
            let l: u128 = dims[..i].iter().map(|&d| d as u128).product();
            let r: u128 = dims[i..].iter().map(|&d| d as u128).product();
            l.min(r)
        })
        .max()
        .unwrap();
    assert!(
        st.max_bond_reached >= 1 && (st.max_bond_reached as u128) <= widest_cut,
        "dims {dims:?}: max_bond_reached {} outside [1, {widest_cut}]",
        st.max_bond_reached
    );
    assert!(
        st.max_bond_reached < EXACT_CHI,
        "the ceiling was reached: not exact"
    );
    (err, st.max_bond_reached)
}

/// ≥ 50 seeds of random mixed-radix circuits, each through `MpsBackend` and
/// through quditsv, agreeing at 1e-10 on every amplitude.
#[test]
fn random_mixed_radix_circuits_match_quditsv_in_the_exact_regime() {
    let mut rng = Lcg(0x2026_0929);
    let mut cov = Coverage::default();
    let mut worst = 0.0f64;
    let mut cases = 0usize;
    let mut bonds = Vec::new();
    for seed in 0..60u64 {
        let n = 4 + (seed % 3) as usize;
        let dims: Vec<u32> = (0..n)
            .map(|_| [2u32, 3, 4][rng.below(3) as usize])
            .collect();
        let c = random_mixed_circuit(&dims, 30, &mut rng, &mut cov);
        let (err, bond) = exact_regime_case(&dims, &c);
        bonds.push(bond);
        assert!(
            err < 1e-10,
            "seed {seed} dims {dims:?}: max |Δamp| = {err:e}"
        );
        worst = worst.max(err);
        cases += 1;
    }
    eprintln!(
        "exact regime: {cases} circuits, max |Δamp| = {worst:e}, max bond reached {}, \
         max discarded_weight {:e}, coverage {cov:?}",
        bonds.iter().max().unwrap(),
        MAX_EXACT_DW.with(|m| m.get())
    );
    assert!(cases >= 50);
    assert!(cov.csum_adjacent > 50 && cov.csum_distant > 50, "{cov:?}");
    assert!(cov.csum_mixed_dims > 50, "{cov:?}");
    assert!(
        cov.generalised_on_qudit > 50 && cov.rxy_on_qudit > 50,
        "{cov:?}"
    );
    assert!(
        cov.qubit_gates > 50 && cov.qubit_pairs_distant > 5,
        "{cov:?}"
    );
    assert!(
        bonds.iter().any(|&b| b > 8),
        "the corpus must entangle past a trivial bond: {bonds:?}"
    );
}

/// A pure-qutrit chain, explicitly.
#[test]
fn a_pure_qutrit_chain_matches_quditsv_in_the_exact_regime() {
    let dims = [3u32; 5];
    let mut worst = 0.0f64;
    for seed in 0..8u64 {
        let mut rng = Lcg(0x3333 + seed);
        let c = random_mixed_circuit(&dims, 40, &mut rng, &mut Coverage::default());
        worst = worst.max(exact_regime_case(&dims, &c).0);
    }
    eprintln!("pure qutrit chain [3;5]: max |Δamp| = {worst:e}");
    assert!(worst < 1e-10, "{worst:e}");
}

/// The [3,2,4,2] chain, explicitly — every neighbouring pair has unequal
/// dimensions, so every SWAP in the network reshapes.
#[test]
fn a_3_2_4_2_chain_matches_quditsv_in_the_exact_regime() {
    let dims = [3u32, 2, 4, 2];
    let mut worst = 0.0f64;
    for seed in 0..8u64 {
        let mut rng = Lcg(0x3242 + seed);
        let c = random_mixed_circuit(&dims, 40, &mut rng, &mut Coverage::default());
        worst = worst.max(exact_regime_case(&dims, &c).0);
    }
    eprintln!("[3,2,4,2]: max |Δamp| = {worst:e}");
    assert!(worst < 1e-10, "{worst:e}");
}

/// At d = 2 the generalised matrices are the qubit ones to 1e-12 — but not
/// bit for bit, which is why the backend keeps `gates::h/x/z` on qubit wires.
/// This test pins both halves of that sentence, so the comment in
/// `apply_gate_mps` cannot go stale silently.
#[test]
fn the_generalised_gates_equal_the_qubit_gates_at_d_2_but_h_not_bitwise() {
    use omega_backend_mps::gates;
    let close = |a: &[C], b: &[C]| a.iter().zip(b).all(|(x, y)| (x - y).norm() < 1e-12);
    assert!(close(&qg::fourier(2).m, &gates::h()));
    assert!(close(&qg::shift(2).m, &gates::x()));
    assert!(close(&qg::clock(2).m, &gates::z()));
    assert!(close(&qg::csum(2, 2).m, &gates::cx()));
    let bitwise = |a: &[C], b: &[C]| {
        a.iter()
            .zip(b)
            .all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits())
    };
    assert!(
        bitwise(&qg::shift(2).m, &gates::x()),
        "shift(2) is X bit for bit"
    );
    assert!(
        !bitwise(&qg::fourier(2).m, &gates::h()),
        "fourier(2) became bit-identical to h(): the d = 2 special case in \
         apply_gate_mps is now unnecessary (harmless) — update its comment"
    );
}

// ---------------------------------------------------------------------------
// (ii) truncation contract
// ---------------------------------------------------------------------------

/// A qutrit brickwork: Rxy on random level pairs on every site, then CSum on
/// alternating bricks plus one distant CSum per layer — a non-flat Schmidt
/// spectrum (random angles), which `tests/fidelity_estimate.rs` explains is
/// the only kind that can expose a gauge problem.
fn qutrit_brickwork(n: usize, layers: usize, seed: u64) -> CircuitIR {
    let dims = vec![3u32; n];
    let mut rng = Lcg(seed);
    let mut ops = Vec::new();
    for layer in 0..layers {
        for w in 0..n as u32 {
            let i = rng.below(2);
            let j = i + 1 + rng.below(2 - i);
            ops.push(op(
                GateKind::Rxy,
                &[w],
                &[i as f64, j as f64, rng.angle(), rng.angle()],
            ));
        }
        let mut w = (layer % 2) as u32;
        while (w + 1) < n as u32 {
            ops.push(op(GateKind::CSum, &[w, w + 1], &[]));
            w += 2;
        }
        let (a, b) = rng.pair(n as u32);
        ops.push(op(GateKind::CSum, &[a, b], &[]));
    }
    // Depth counted in ops: the brief asks for ≥ 40 with entanglers.
    qudits(&dims, ops)
}

/// **The qubit lane's certificate contract, transferred to d = 3 unchanged.**
///
/// Mirrors, assertion for assertion:
///
/// * `tests/fidelity_estimate.rs::the_fidelity_estimate_never_exceeds_the_true_fidelity`
///   — the measured-vs-certified direction: `fidelity_estimate ≤ F_true +
///   1e-9` where `F_true = |⟨ψ_exact|ψ_χ⟩|²` is measured against the exact
///   state (there `to_statevector` at χ = 2^n, here quditsv), and the ratio
///   `F_true / F_est` stays in `[1 − 1e-9, 1e3)`.
/// * `tests/certificate_predicates.rs::a_starved_run_reports_a_positive_certificate_and_resets_per_execute`
///   — the relation between the two certificate fields,
///   `Π(1 − εᵢ) ≥ 1 − Σεᵢ − 1e-12`, a positive certificate and a saturated
///   bond on a starved run.
/// * `tests/certificate_predicates.rs::the_ceiling_is_straddled_by_the_certificate_itself`
///   — a ceiling equal to the run's own `discarded_weight` accepts, one a hair
///   below refuses.
///
/// No new bound is derived: in particular `discarded_weight` is NOT turned
/// into an amplitude or expectation error bound here, exactly as the qubit
/// lane does not (STATUS §1; PLAN-QUDIT §Q3 "Certificate story").
///
/// What it catches (measured by hand, PLAN-QUDIT §Q3 gate iv): halving the
/// per-split relative weight at its source (`rel_discarded = 0.5 *
/// svd.discarded_weight / total`, which feeds BOTH fields) trips this test —
/// the estimate then exceeds the measured fidelity. Halving only the
/// `discarded_weight` accumulation does NOT trip it: every run here has
/// `Σεᵢ > 2`, so `Π(1 − εᵢ) ≥ 1 − Σεᵢ` is vacuous at this depth, and the
/// closed-form test below is what catches that one.
///
/// The corpus differs from the qubit file's only in d and depth: n = 8
/// qutrits, 8 layers (8 Rxy + 3–4 adjacent CSum + 1 distant CSum per layer,
/// ≥ 40 entangling-or-rotation ops), χ ∈ {2, 3, 4, 6, 8, 12}, three seeds.
#[test]
fn qutrit_truncation_keeps_the_qubit_lanes_certificate_contract() {
    const N: usize = 8;
    const LAYERS: usize = 8;
    let chis = [2usize, 3, 4, 6, 8, 12];
    let mut violations = Vec::new();
    let mut compared = 0usize;
    let mut min_ratio = f64::INFINITY;
    let mut max_ratio: f64 = 0.0;
    eprintln!("seed  chi  discarded_weight   fidelity_estimate  F_true             1-F_true           max_bond");
    for seed in 0..3u64 {
        let c = qutrit_brickwork(N, LAYERS, 0x0DD0 + seed);
        assert!(c.ops.len() >= 40, "depth {}", c.ops.len());
        let psi = exact(&c).amp;
        for &chi in &chis {
            let b = MpsBackend::new(chi).with_max_discarded_weight(f64::INFINITY);
            let got = mps_statevector(&b, &c, &analytic());
            let st = b.last_run_stats();
            let f_true = fidelity(&psi, &got);
            let f_est = st.fidelity_estimate;
            eprintln!(
                "{seed:>4}  {chi:>3}  {:<17e}  {:<17e}  {:<17e}  {:<17e}  {}",
                st.discarded_weight,
                f_est,
                f_true,
                1.0 - f_true,
                st.max_bond_reached
            );
            compared += 1;
            if f_est > f_true + 1e-9 {
                violations.push(format!(
                    "seed={seed} chi={chi}: F_est={f_est:.6e} > F_true={f_true:.6e}"
                ));
            }
            if f_true > 1e-12 {
                let r = f_true / f_est.max(1e-300);
                min_ratio = min_ratio.min(r);
                max_ratio = max_ratio.max(r);
            }
            // certificate_predicates: the two fields cannot contradict.
            assert!(
                f_est >= 1.0 - st.discarded_weight - 1e-12,
                "seed={seed} chi={chi}: Π(1 − εᵢ) = {f_est} < 1 − Σεᵢ = {}: the two \
                 fields contradict each other",
                1.0 - st.discarded_weight
            );
            // Every χ in the sweep is below the exact bond 3^4 = 81.
            assert!(
                st.discarded_weight > 0.0 && f_est < 1.0,
                "seed={seed} chi={chi}: a starved bond must report a positive certificate"
            );
            assert_eq!(
                st.max_bond_reached, chi,
                "a starved run saturates the ceiling"
            );
        }
    }
    eprintln!(
        "qutrit certificate vs truth: {compared} cases, F_true/F_est in \
         [{min_ratio:.4}, {max_ratio:.4}]"
    );
    assert!(
        violations.is_empty(),
        "the fidelity ESTIMATE came out above the true fidelity at d = 3:\n  {}",
        violations.join("\n  ")
    );
    assert_eq!(compared, 3 * chis.len());
    assert!(min_ratio >= 1.0 - 1e-9, "{min_ratio}");
    assert!(
        max_ratio < 1e3,
        "largest F_true/F_est is {max_ratio:.3e}: the estimate has collapsed \
         toward zero and no longer discriminates"
    );
}

/// The ceiling, straddled with a qutrit run's own certificate — the qubit
/// test `the_ceiling_is_straddled_by_the_certificate_itself`, at d = 3.
#[test]
fn the_ceiling_is_straddled_by_a_qutrit_runs_own_certificate() {
    let c = qutrit_brickwork(8, 8, 0x0DD0);
    let open = MpsBackend::new(4).with_max_discarded_weight(f64::INFINITY);
    let _ = mps_statevector(&open, &c, &analytic());
    let dw = open.last_run_stats().discarded_weight;
    assert!(dw > DEFAULT_MAX_DISCARDED_WEIGHT, "{dw}");

    MpsBackend::new(4)
        .with_max_discarded_weight(dw)
        .execute(&c, &ParameterBinding::new(), &analytic())
        .expect("a ceiling EQUAL to the certificate accepts: the gate is `>`");
    let err = MpsBackend::new(4)
        .with_max_discarded_weight(dw * (1.0 - 1e-12))
        .execute(&c, &ParameterBinding::new(), &analytic())
        .expect_err("a ceiling a hair below the certificate refuses");
    assert!(
        format!("{err:?}").contains("exceeds the ceiling"),
        "{err:?}"
    );
    let err = MpsBackend::new(4)
        .execute(&c, &ParameterBinding::new(), &analytic())
        .expect_err("the default ceiling refuses a starved qutrit run");
    assert!(
        format!("{err:?}").contains("exceeded the ceiling"),
        "{err:?}"
    );
}

/// **The certificate against a known qutrit Schmidt spectrum** — the d = 3
/// transfer of `tests/certificate_predicates.rs::
/// fixed_rank_truncation_reports_the_same_closed_form` and
/// `adaptive_drop_reports_exactly_the_dropped_schmidt_weight`.
///
/// `rxy(0,1,θ₁,π/2)` then `rxy(1,2,θ₂,π/2)` on qutrit 0 gives amplitudes
/// `(c₁, s₁c₂, s₁s₂)` (cᵢ = cos θᵢ/2, sᵢ = sin θᵢ/2); `csum(0,1)` copies the
/// level, so the pair is `Σ aₖ|k,k⟩` with Schmidt values exactly
/// `{c₁, s₁c₂, s₁s₂}` — three of them, which a qubit bookkeeping cannot hold.
/// χ = 2 must drop exactly `s₁²s₂²`, χ = 1 exactly `s₁²`, adaptive ε = 0.3
/// the same as χ = 2; and the fidelity estimate is the kept weight.
///
/// This is the half of the contract the deep sweep above cannot exercise:
/// there every run's `Σεᵢ` exceeds 2, which makes the field relation
/// `Π(1 − εᵢ) ≥ 1 − Σεᵢ` vacuous. Measured with the mutation of (iv)
/// (`discarded_weight += 0.5 * rel_discarded`): the sweep stayed green and
/// this test is the one that tripped.
#[test]
fn qutrit_closed_form_spectrum_is_certified_exactly() {
    let (t1, t2) = (1.2_f64, 0.8_f64);
    let (c1, s1) = ((t1 / 2.0).cos(), (t1 / 2.0).sin());
    let (c2, s2) = ((t2 / 2.0).cos(), (t2 / 2.0).sin());
    let half_pi = std::f64::consts::FRAC_PI_2;
    let prep = |m: &mut Mps| {
        m.apply_1(0, &qg::rxy(3, 0, 1, t1, half_pi).m);
        m.apply_1(0, &qg::rxy(3, 1, 2, t2, half_pi).m);
        m.apply_2(0, &qg::csum(3, 3).m);
    };
    let close = |got: f64, want: f64, what: &str| {
        assert!(
            (got - want).abs() <= 1e-9,
            "{what}: got {got:.15}, closed form {want:.15}"
        );
    };

    let mut exact = Mps::zero_state_dims(&[3, 3], 3);
    prep(&mut exact);
    assert!(exact.discarded_weight <= 1e-15);
    assert_eq!(
        exact.max_bond_reached, 3,
        "three Schmidt values survive at χ = 3"
    );

    let mut two = Mps::zero_state_dims(&[3, 3], 2);
    prep(&mut two);
    close(two.discarded_weight, (s1 * s2).powi(2), "χ=2 drops s₁²s₂²");
    close(
        two.fidelity_estimate,
        1.0 - (s1 * s2).powi(2),
        "χ=2 keeps the rest",
    );
    assert_eq!(two.max_bond_reached, 2);

    let mut one = Mps::zero_state_dims(&[3, 3], 1);
    prep(&mut one);
    close(one.discarded_weight, s1 * s1, "χ=1 drops s₁²");
    close(one.fidelity_estimate, c1 * c1, "χ=1 keeps c₁²");
    assert_eq!(one.max_bond_reached, 1);

    // Adaptive: σ = (c₁, s₁c₂, s₁s₂) ≈ (0.825, 0.520, 0.220); ε = 0.3 puts
    // the threshold at 0.248 — drops only the smallest.
    assert!(s1 * s2 < 0.3 * c1 && s1 * c2 > 0.3 * c1);
    let mut adaptive = Mps::zero_state_dims(&[3, 3], 3);
    adaptive.set_adaptive_eps(0.3);
    prep(&mut adaptive);
    close(
        adaptive.discarded_weight,
        (s1 * s2).powi(2),
        "adaptive drops s₁²s₂²",
    );
    close(
        adaptive.fidelity_estimate,
        1.0 - (s1 * s2).powi(2),
        "adaptive keeps the rest",
    );
    assert_eq!(adaptive.max_bond_reached, 2);

    // And through the backend: the same numbers reach `last_run_stats`.
    let c = qudits(
        &[3, 3],
        vec![
            op(GateKind::Rxy, &[0], &[0.0, 1.0, t1, half_pi]),
            op(GateKind::Rxy, &[0], &[1.0, 2.0, t2, half_pi]),
            op(GateKind::CSum, &[0, 1], &[]),
        ],
    );
    let b = MpsBackend::new(2).with_max_discarded_weight(f64::INFINITY);
    let _ = mps_statevector(&b, &c, &analytic());
    let st = b.last_run_stats();
    close(
        st.discarded_weight,
        (s1 * s2).powi(2),
        "backend χ=2 certificate",
    );
    close(
        st.fidelity_estimate,
        1.0 - (s1 * s2).powi(2),
        "backend χ=2 estimate",
    );
}

// ---------------------------------------------------------------------------
// Sampling and measurement on qudit sites
// ---------------------------------------------------------------------------

/// The generalised ops of a circuit applied straight to an `Mps` (the
/// backend's `evolve_once` is private), with the same shared matrix table.
fn mps_of(dims: &[u32], ops: &[GateOp], chi: usize) -> Mps {
    let d: Vec<usize> = dims.iter().map(|&x| x as usize).collect();
    let mut m = Mps::zero_state_dims(&d, chi);
    for o in ops {
        let w: Vec<usize> = o.qubits.iter().map(|q| q.0 as usize).collect();
        let p: Vec<f64> = o
            .params
            .iter()
            .map(|x| match x {
                ParamExpr::Concrete(v) => *v,
                _ => unreachable!(),
            })
            .collect();
        match o.gate {
            GateKind::H => m.apply_1(w[0], &qg::fourier(d[w[0]]).m),
            GateKind::X => m.apply_1(w[0], &qg::shift(d[w[0]]).m),
            GateKind::Z => m.apply_1(w[0], &qg::clock(d[w[0]]).m),
            GateKind::Rxy => m.apply_1(
                w[0],
                &qg::rxy(d[w[0]], p[0] as usize, p[1] as usize, p[2], p[3]).m,
            ),
            GateKind::CSum => m.apply_2_distant(w[0], w[1], &qg::csum(d[w[0]], d[w[1]]).m),
            ref g => panic!("mps_of: {g:?} not in the generalised set"),
        }
    }
    m
}

fn generalised_only(dims: &[u32], depth: usize, seed: u64) -> Vec<GateOp> {
    let n = dims.len() as u32;
    let mut rng = Lcg(seed);
    let mut ops = Vec::new();
    for _ in 0..depth {
        match rng.below(3) {
            0 => {
                let w = rng.below(n);
                let d = dims[w as usize];
                let i = rng.below(d - 1);
                let j = i + 1 + rng.below(d - 1 - i);
                ops.push(op(
                    GateKind::Rxy,
                    &[w],
                    &[i as f64, j as f64, rng.angle(), rng.angle()],
                ));
            }
            1 => ops.push(op(GateKind::H, &[rng.below(n)], &[])),
            _ => {
                let (a, b) = rng.pair(n);
                ops.push(op(GateKind::CSum, &[a, b], &[]));
            }
        }
    }
    ops
}

/// 20000 shots from `sample_bits_with_envs_into` on a mixed chain at exact
/// χ, against quditsv's `probabilities()`: every per-site marginal and every
/// joint outcome within 0.02 (max-abs). Digits `>= 2` must actually be drawn.
#[test]
fn sampled_digits_match_quditsv_marginals_on_a_mixed_chain() {
    use rand::SeedableRng;
    let dims = [3u32, 2, 4, 2, 3];
    let ops = generalised_only(&dims, 40, 0x5A4D);
    let st = exact(&qudits(&dims, ops.clone()));
    let probs = st.probabilities();
    let m = mps_of(&dims, &ops, 256);
    assert!(m.discarded_weight <= 1e-15, "{}", m.discarded_weight);

    let envs = m.right_environments();
    let mut rng = rand::rngs::StdRng::seed_from_u64(20260929);
    let shots = 20000usize;
    let mut marg: Vec<Vec<f64>> = dims.iter().map(|&d| vec![0.0; d as usize]).collect();
    let mut joint = vec![0.0f64; probs.len()];
    let mut digits = Vec::new();
    let mut saw_high = false;
    for _ in 0..shots {
        m.sample_bits_with_envs_into(&envs, &mut rng, &mut digits);
        let mut idx = 0usize;
        let mut stride = 1usize;
        for (w, &dg) in digits.iter().enumerate() {
            assert!(
                (dg as u32) < dims[w],
                "digit {dg} on a d = {} site",
                dims[w]
            );
            saw_high |= dg >= 2;
            marg[w][dg as usize] += 1.0 / shots as f64;
            idx += dg as usize * stride;
            stride *= dims[w] as usize;
        }
        joint[idx] += 1.0 / shots as f64;
    }
    assert!(
        saw_high,
        "no digit >= 2 was ever drawn: the qudit levels were not sampled"
    );
    let mut worst_marg = 0.0f64;
    for (w, m_w) in marg.iter().enumerate() {
        for (k, f) in m_w.iter().enumerate() {
            let p: f64 = probs
                .iter()
                .enumerate()
                .filter(|(i, _)| st.digit(*i, w) == k)
                .map(|(_, p)| p)
                .sum();
            worst_marg = worst_marg.max((p - f).abs());
        }
    }
    let worst_joint = probs
        .iter()
        .zip(&joint)
        .map(|(p, f)| (p - f).abs())
        .fold(0.0, f64::max);
    eprintln!("sampling [3,2,4,2,3] x {shots}: max |Δ marginal| = {worst_marg:e}, max |Δ joint| = {worst_joint:e}");
    assert!(worst_marg <= 0.02, "{worst_marg}");
    assert!(worst_joint <= 0.02, "{worst_joint}");
}

/// Project the exact vector onto `digit(w) == k` and renormalise.
fn project(st: &State, w: usize, k: usize) -> Vec<C> {
    let mut v: Vec<C> = st
        .amp
        .iter()
        .enumerate()
        .map(|(i, a)| {
            if st.digit(i, w) == k {
                *a
            } else {
                C::new(0.0, 0.0)
            }
        })
        .collect();
    let n: f64 = v.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    for a in &mut v {
        *a /= n;
    }
    v
}

/// `MidCircuitMode::Collapse` through `MpsBackend` on a qutrit wire, with
/// gates AFTER the measurement: the post-measurement statevector equals the
/// exact state projected by hand onto the drawn level and evolved on. Over
/// the seeds every level 0, 1, 2 must be drawn at least once.
#[test]
fn collapse_mode_measurement_of_a_qutrit_matches_the_projected_exact_state() {
    let dims = [2u32, 3, 3];
    let prefix = generalised_only(&dims, 20, 0xC011);
    let suffix = generalised_only(&dims, 12, 0x5FF1);
    let exact_prefix = exact(&qudits(&dims, prefix.clone()));
    let mut seen = [false; 3];
    let mut worst = 0.0f64;
    for seed in 0..40u64 {
        let mut ops = prefix.clone();
        ops.push(op(GateKind::Measure, &[1], &[]));
        ops.extend(suffix.iter().cloned());
        let c = qudits(&dims, ops);
        let cfg = ExecConfig {
            shots: None,
            seed: Some(seed),
            mid_circuit_mode: MidCircuitMode::Collapse,
        };
        let got = mps_statevector(&MpsBackend::new(64), &c, &cfg);
        // Which level was drawn: the one whose projection matches.
        let mut matched = None;
        for k in 0..3usize {
            let mut st = State {
                dims: exact_prefix.dims.clone(),
                strides: exact_prefix.strides.clone(),
                amp: project(&exact_prefix, 1, k),
            };
            if st.amp.iter().any(|a| a.re.is_nan()) {
                continue;
            }
            for o in &suffix {
                let w: Vec<usize> = o.qubits.iter().map(|q| q.0 as usize).collect();
                let d = |i: usize| dims[w[i]] as usize;
                let p: Vec<f64> = o
                    .params
                    .iter()
                    .map(|x| match x {
                        ParamExpr::Concrete(v) => *v,
                        _ => unreachable!(),
                    })
                    .collect();
                match o.gate {
                    GateKind::H => st.apply_1(w[0], &qg::fourier(d(0))),
                    GateKind::Rxy => st.apply_1(
                        w[0],
                        &qg::rxy(d(0), p[0] as usize, p[1] as usize, p[2], p[3]),
                    ),
                    GateKind::CSum => st.apply_2(w[0], w[1], &qg::csum(d(0), d(1))),
                    ref g => panic!("{g:?}"),
                }
            }
            let err = max_abs_diff(&got, &st.amp);
            if err < 1e-10 {
                matched = Some(k);
                worst = worst.max(err);
            }
        }
        let k = matched.unwrap_or_else(|| {
            panic!("seed {seed}: the collapsed MPS state matches no projection of the exact state")
        });
        seen[k] = true;
    }
    eprintln!("collapse on a qutrit: levels seen {seen:?}, max |Δamp| = {worst:e}");
    assert!(
        seen.iter().all(|&s| s),
        "every level must be drawn: {seen:?}"
    );
}

/// A `Reset` on a qutrit wire returns it to |0⟩ in every trajectory: after
/// `x; x` (level 2) and a reset, the state is |0…0⟩ exactly.
#[test]
fn reset_returns_a_qutrit_in_level_two_to_zero() {
    let dims = [3u32, 2];
    let c = qudits(
        &dims,
        vec![
            op(GateKind::X, &[0], &[]),
            op(GateKind::X, &[0], &[]),
            op(GateKind::Reset, &[0], &[]),
        ],
    );
    let sv = mps_statevector(&MpsBackend::new(8), &c, &analytic());
    assert_eq!(sv.len(), 6);
    assert!((sv[0] - C::new(1.0, 0.0)).norm() < 1e-12, "{sv:?}");
}

// ---------------------------------------------------------------------------
// Expectation on a mixed chain
// ---------------------------------------------------------------------------

/// Pauli observables on the QUBIT wires of a mixed chain agree with quditsv
/// (identity of size d on the qudit sites); a Pauli on a qutrit wire is
/// refused by both engines with the same sentence.
#[test]
fn pauli_expectations_on_the_qubit_wires_of_a_mixed_chain_match_quditsv() {
    let dims = [3u32, 2, 4, 2];
    let mut rng = Lcg(0xE8);
    let c = random_mixed_circuit(&dims, 40, &mut rng, &mut Coverage::default());
    let obs = vec![
        Observable {
            terms: vec![(1.0, vec![(1, PauliOp::Z)])],
        },
        Observable {
            terms: vec![
                (0.5, vec![(1, PauliOp::X), (3, PauliOp::Y)]),
                (-0.25, vec![(3, PauliOp::Z)]),
            ],
        },
        Observable {
            terms: vec![(1.0, vec![(0, PauliOp::I), (1, PauliOp::Y), (3, PauliOp::X)])],
        },
    ];
    let mps = MpsBackend::new(EXACT_CHI);
    let sv = QuditSvBackend::new();
    let a = mps
        .expectation_multi(&c, &ParameterBinding::new(), &obs)
        .unwrap();
    let b = sv
        .expectation_multi(&c, &ParameterBinding::new(), &obs)
        .unwrap();
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() < 1e-10, "mps {x} vs quditsv {y}");
    }
    for o in &obs {
        let x = mps.expectation(&c, &ParameterBinding::new(), o).unwrap();
        let y = sv.expectation(&c, &ParameterBinding::new(), o).unwrap();
        assert!((x - y).abs() < 1e-10, "mps {x} vs quditsv {y}");
    }
    let on_qutrit = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };
    let em = format!(
        "{}",
        mps.expectation(&c, &ParameterBinding::new(), &on_qutrit)
            .unwrap_err()
    );
    let eq = format!(
        "{}",
        sv.expectation(&c, &ParameterBinding::new(), &on_qutrit)
            .unwrap_err()
    );
    assert_eq!(strip(&em, "mps: "), strip(&eq, "quditsv: "), "{em}\n{eq}");
}

// ---------------------------------------------------------------------------
// The refusal line: both engines draw it in the same places
// ---------------------------------------------------------------------------

/// The engine-specific part of a refusal, after `prefix`.
fn strip(msg: &str, prefix: &str) -> String {
    let i = msg
        .find(prefix)
        .unwrap_or_else(|| panic!("{prefix:?} not in {msg:?}"));
    msg[i + prefix.len()..].to_string()
}

/// Every construct below is refused by BOTH engines with the same sentence
/// (modulo the engine prefix) or accepted by both with the same statevector.
/// The known differences — constructs quditsv refuses and MPS runs — are
/// listed in `the_documented_differences_from_quditsv` so this table cannot
/// hide one.
#[test]
fn mps_and_quditsv_refuse_the_same_constructs_with_the_same_sentence() {
    let cases: Vec<(&str, CircuitIR)> = vec![
        (
            "cx on qutrits",
            qudits(&[3, 3], vec![op(GateKind::CX, &[0, 1], &[])]),
        ),
        (
            "cx qubit->qutrit",
            qudits(&[2, 3], vec![op(GateKind::CX, &[0, 1], &[])]),
        ),
        (
            "y on a ququart",
            qudits(&[4], vec![op(GateKind::Y, &[0], &[])]),
        ),
        (
            "rz on a qutrit",
            qudits(&[3, 2], vec![op(GateKind::Rz, &[0], &[0.3])]),
        ),
        (
            "swap mixed",
            qudits(&[2, 3], vec![op(GateKind::Swap, &[0, 1], &[])]),
        ),
        (
            "ccx with a qutrit",
            qudits(&[2, 2, 3], vec![op(GateKind::CCX, &[0, 1, 2], &[])]),
        ),
        (
            "cswap with a qutrit",
            qudits(&[3, 2, 2], vec![op(GateKind::CSwap, &[0, 1, 2], &[])]),
        ),
        (
            "rxy level out of range",
            qudits(&[3], vec![op(GateKind::Rxy, &[0], &[0.0, 3.0, 0.1, 0.2])]),
        ),
        (
            "rxy fractional level",
            qudits(&[3], vec![op(GateKind::Rxy, &[0], &[0.5, 2.0, 0.1, 0.2])]),
        ),
        (
            "rxy i >= j",
            qudits(&[3], vec![op(GateKind::Rxy, &[0], &[2.0, 1.0, 0.1, 0.2])]),
        ),
        (
            "csum same wire",
            qudits(&[3, 3], vec![op(GateKind::CSum, &[1, 1], &[])]),
        ),
        // Accepted by both:
        (
            "h x z on a qutrit",
            qudits(
                &[3],
                vec![
                    op(GateKind::H, &[0], &[]),
                    op(GateKind::X, &[0], &[]),
                    op(GateKind::Z, &[0], &[]),
                ],
            ),
        ),
        (
            "qubit gates on the qubit wires of a mixed chain",
            qudits(
                &[3, 2, 2],
                vec![
                    op(GateKind::H, &[1], &[]),
                    op(GateKind::CX, &[1, 2], &[]),
                    op(GateKind::Rz, &[2], &[0.4]),
                ],
            ),
        ),
        (
            "rxy on a qubit",
            qudits(&[2], vec![op(GateKind::Rxy, &[0], &[0.0, 1.0, 0.7, 0.3])]),
        ),
        (
            "csum distant reversed 4->3",
            qudits(
                &[3, 2, 4],
                vec![op(GateKind::H, &[2], &[]), op(GateKind::CSum, &[2, 0], &[])],
            ),
        ),
        (
            "measure at the end",
            qudits(
                &[3],
                vec![op(GateKind::H, &[0], &[]), op(GateKind::Measure, &[0], &[])],
            ),
        ),
    ];
    for (what, c) in &cases {
        let m = MpsBackend::new(64).execute(c, &ParameterBinding::new(), &analytic());
        let q = QuditSvBackend::new().execute(c, &ParameterBinding::new(), &analytic());
        match (m, q) {
            (Err(em), Err(eq)) => {
                let (em, eq) = (format!("{em}"), format!("{eq}"));
                assert_eq!(strip(&em, "mps: "), strip(&eq, "quditsv: "), "{what}");
            }
            (Ok(ExecResult::Statevector(a)), Ok(ExecResult::Statevector(b))) => {
                assert!(max_abs_diff(&a, &b) < 1e-10, "{what}");
            }
            (m, q) => panic!("{what}: the engines disagree on the line — mps {m:?}, quditsv {q:?}"),
        }
    }
    // Sampling a qudit circuit: refused by both, same sentence.
    let c = qudits(&[3, 2], vec![op(GateKind::H, &[0], &[])]);
    let shots = ExecConfig {
        shots: Some(16),
        seed: Some(1),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let em = format!(
        "{}",
        MpsBackend::new(8)
            .execute(&c, &ParameterBinding::new(), &shots)
            .unwrap_err()
    );
    let eq = format!(
        "{}",
        QuditSvBackend::new()
            .execute(&c, &ParameterBinding::new(), &shots)
            .unwrap_err()
    );
    assert_eq!(strip(&em, "mps: "), strip(&eq, "quditsv: "));
    assert!(
        em.contains("`Counts` outcomes are bit strings") && em.contains("--statevector"),
        "{em}"
    );
}

/// Where MPS runs what quditsv refuses. Each is a capability the MPS engine
/// already had on qubits (trajectory channels, mid-circuit measurement,
/// classical control) and that quditsv, an exact pure-state end-of-circuit
/// engine, refuses by design — not a disagreement about what a qudit gate
/// means. Pinned so the list in the report is the list in the code.
#[test]
fn the_documented_differences_from_quditsv() {
    let refused_by_quditsv_only = [
        (
            "reset on a qutrit",
            qudits(
                &[3],
                vec![op(GateKind::X, &[0], &[]), op(GateKind::Reset, &[0], &[])],
            ),
            MidCircuitMode::Skip,
        ),
        (
            "gate after a measurement",
            qudits(
                &[3],
                vec![op(GateKind::Measure, &[0], &[]), op(GateKind::H, &[0], &[])],
            ),
            MidCircuitMode::Skip,
        ),
        (
            "collapse mode",
            qudits(
                &[3],
                vec![op(GateKind::H, &[0], &[]), op(GateKind::Measure, &[0], &[])],
            ),
            MidCircuitMode::Collapse,
        ),
    ];
    let mut conditioned = op(GateKind::H, &[0], &[]);
    conditioned.condition = Some((0, 1, 0));
    let mut conditioned_c = qudits(&[3], vec![conditioned]);
    conditioned_c.num_classical_bits = 1;
    let refused_by_quditsv_only = refused_by_quditsv_only.into_iter().chain([(
        "a classically-conditioned gate",
        conditioned_c,
        MidCircuitMode::Skip,
    )]);
    for (what, c, mode) in refused_by_quditsv_only {
        let cfg = ExecConfig {
            shots: None,
            seed: Some(3),
            mid_circuit_mode: mode,
        };
        assert!(
            MpsBackend::new(8)
                .execute(&c, &ParameterBinding::new(), &cfg)
                .is_ok(),
            "{what}: mps must run it"
        );
        assert!(
            QuditSvBackend::new()
                .execute(&c, &ParameterBinding::new(), &cfg)
                .is_err(),
            "{what}: quditsv refuses it"
        );
    }
    // A classically-conditioned gate reading a digit a qutrit measurement
    // wrote: refused by MPS too (a classical bit cannot hold the digit), by
    // its own sentence.
    let mut m = op(GateKind::Measure, &[0], &[]);
    m.classical_bit = Some(0);
    let mut x = op(GateKind::X, &[1], &[]);
    x.condition = Some((0, 1, 1));
    let mut c = qudits(&[3, 2], vec![op(GateKind::H, &[0], &[]), m, x]);
    c.num_classical_bits = 1;
    let cfg = ExecConfig {
        shots: None,
        seed: Some(3),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let e = MpsBackend::new(8)
        .execute(&c, &ParameterBinding::new(), &cfg)
        .unwrap_err();
    assert!(format!("{e}").contains("which a condition reads"), "{e}");
}

// ---------------------------------------------------------------------------
// Q4 — measurement: the site-operator door.
//
// `Backend::expectation` takes a Pauli observable and refuses a `d ≠ 2` wire
// by name, so until `expectation_site_operators` the only number a qutrit
// chain could yield was `execute(shots: None)` → `to_statevector`, a `Π d_q`
// dense contraction the bond dimension does not bound. The door below is the
// first measurement that stays on the chain; these tests are its contract.
// ---------------------------------------------------------------------------

/// A random `d×d` Hermitian matrix, row-major: `R + R†` with `R` uniform.
fn random_hermitian(d: usize, rng: &mut Lcg) -> Vec<C> {
    let mut r = vec![C::new(0.0, 0.0); d * d];
    for x in r.iter_mut() {
        *x = C::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5);
    }
    let mut h = vec![C::new(0.0, 0.0); d * d];
    for i in 0..d {
        for j in 0..d {
            h[i * d + j] = r[i * d + j] + r[j * d + i].conj();
        }
    }
    h
}

/// `⟨ψ|O_0 ⊗ … ⊗ O_{n−1}|ψ⟩ / ⟨ψ|ψ⟩` on the exact vector, one site operator
/// applied at a time — the hand reference, not through any MPS code.
fn dense_site_expectation(st: &State, site_ops: &[Vec<C>]) -> f64 {
    let mut v = st.amp.clone();
    for (w, o) in site_ops.iter().enumerate() {
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
    num.re / den
}

/// ≥ 40 seeds of random mixed-radix circuits with random Hermitian site
/// operators, measured on the chain and by hand on the exact vector, agreeing
/// at 1e-10. The chain is run at the exact bond dimension so any gap is the
/// contraction's, not truncation's.
#[test]
fn site_operator_expectation_on_a_mixed_radix_chain_matches_the_dense_hand_contraction() {
    let mut rng = Lcg(0x2026_0930);
    let mut cov = Coverage::default();
    let mut worst = 0.0f64;
    let mut nontrivial = 0usize;
    for seed in 0..40u64 {
        let n = 4 + (seed % 3) as usize;
        let dims: Vec<u32> = (0..n)
            .map(|_| [2u32, 3, 4][rng.below(3) as usize])
            .collect();
        let c = random_mixed_circuit(&dims, 30, &mut rng, &mut cov);
        let ops: Vec<Vec<C>> = dims
            .iter()
            .map(|&d| random_hermitian(d as usize, &mut rng))
            .collect();
        let st = exact(&c);
        let want = dense_site_expectation(&st, &ops);
        let got = MpsBackend::new(EXACT_CHI)
            .expectation_site_operators(&c, &ParameterBinding::new(), &ops)
            .unwrap_or_else(|e| panic!("seed {seed} {dims:?}: mps refused: {e}"));
        let gap = (got - want).abs();
        worst = worst.max(gap);
        if want.abs() > 1e-3 {
            nontrivial += 1;
        }
        assert!(
            gap <= 1e-10,
            "seed {seed} {dims:?}: mps {got} vs dense {want}"
        );
    }
    assert!(cov.rxy_on_qudit > 0 && cov.csum_mixed_dims > 0, "{cov:?}");
    assert!(
        nontrivial >= 30,
        "the corpus barely measured anything: {nontrivial}/40 non-zero"
    );
    eprintln!("site-operator expectation, 40 mixed-radix seeds: max |Δ| = {worst:e}");
}

/// Identity on every site is the norm, and a diagonal projector on one wire is
/// the marginal `probabilities()` reports — two anchors that need no random
/// operator to read.
#[test]
fn site_operator_expectation_reproduces_the_norm_and_the_exact_marginals() {
    let dims = [3u32, 2, 4, 2, 3];
    let mut rng = Lcg(0xA5);
    let mut cov = Coverage::default();
    let c = random_mixed_circuit(&dims, 40, &mut rng, &mut cov);
    let st = exact(&c);
    let probs = st.probabilities();
    let b = MpsBackend::new(EXACT_CHI);
    let ident = |d: u32| -> Vec<C> {
        let d = d as usize;
        let mut m = vec![C::new(0.0, 0.0); d * d];
        for i in 0..d {
            m[i * d + i] = C::new(1.0, 0.0);
        }
        m
    };
    let all_id: Vec<Vec<C>> = dims.iter().map(|&d| ident(d)).collect();
    let one = b
        .expectation_site_operators(&c, &ParameterBinding::new(), &all_id)
        .unwrap();
    assert!((one - 1.0).abs() <= 1e-12, "⟨1⟩ = {one}");
    for (w, &d) in dims.iter().enumerate() {
        for k in 0..d as usize {
            let mut ops = all_id.clone();
            let mut p = vec![C::new(0.0, 0.0); (d * d) as usize];
            p[k * d as usize + k] = C::new(1.0, 0.0);
            ops[w] = p;
            let got = b
                .expectation_site_operators(&c, &ParameterBinding::new(), &ops)
                .unwrap();
            let want: f64 = probs
                .iter()
                .enumerate()
                .filter(|(i, _)| st.digit(*i, w) == k)
                .map(|(_, p)| *p)
                .sum();
            assert!(
                (got - want).abs() <= 1e-10,
                "P(wire {w} = {k}): mps {got} vs exact {want}"
            );
        }
    }
}

/// The two contract rows are refused before evolution, each naming what was
/// wrong: the operator count against the wire count, and an operator's size
/// against the wire's declared dimension.
#[test]
fn site_operator_expectation_refuses_a_wrong_count_or_a_wrong_sized_operator_by_name() {
    let c = qudits(
        &[3, 2, 4],
        vec![op(GateKind::H, &[0], &[]), op(GateKind::CSum, &[0, 1], &[])],
    );
    let b = MpsBackend::new(8);
    let id = |d: usize| -> Vec<C> {
        let mut m = vec![C::new(0.0, 0.0); d * d];
        for i in 0..d {
            m[i * d + i] = C::new(1.0, 0.0);
        }
        m
    };
    let e = b
        .expectation_site_operators(&c, &ParameterBinding::new(), &[id(3), id(2)])
        .unwrap_err()
        .to_string();
    assert!(e.contains("2 operators for 3 wires"), "{e}");
    // Wire 2 is a ququart; a qutrit-sized operator there is the wrong shape,
    // and the refusal says which wire, which dimension, and both sizes.
    let e = b
        .expectation_site_operators(&c, &ParameterBinding::new(), &[id(3), id(2), id(3)])
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("wire 2 has dimension 4") && e.contains("16 entries, not 9"),
        "{e}"
    );
}

/// Measurement in the site-operator door follows `defer_measure`'s contract,
/// both halves of it:
///
/// * a trailing `Measure` nothing reads is **inert** and is elided — the
///   answer is the coherent one to the last digit (Qiskit's
///   `remove_final_measurements`; dephasing here would contradict the oracle
///   on 13 of 14 crosscheck fixtures);
/// * a `Measure` a later condition reads is deferred to a controlled gate
///   and the measured wire's operator is **dephased**, `O ↦ Σ_k |k⟩⟨k|O|k⟩⟨k|`.
///   On a `d ≠ 2` wire that case is refused (a bit cannot hold a digit — the
///   test above), so the dephasing that can fire is on a qubit wire inside a
///   mixed-radix chain, and that is what is checked: the answer equals the
///   hand-deferred circuit (`CX` appended) with the hand-dephased operator, and
///   differs from the same circuit undephased, so the CX alone does not
///   account for it.
#[test]
fn measurement_in_the_site_operator_door_elides_inert_and_dephases_read_wires() {
    let dims = [3u32, 2, 3, 2];
    let mut rng = Lcg(0xD3);
    let mut cov = Coverage::default();
    let coherent = random_mixed_circuit(&dims, 24, &mut rng, &mut cov);
    let ops: Vec<Vec<C>> = dims
        .iter()
        .map(|&d| random_hermitian(d as usize, &mut rng))
        .collect();
    let b = MpsBackend::new(EXACT_CHI);
    let coherent_answer = b
        .expectation_site_operators(&coherent, &ParameterBinding::new(), &ops)
        .unwrap();

    // (a) inert: a trailing measurement of the qutrit on wire 2 that nothing
    // reads. Elided, so the coherent answer stands exactly.
    let mut inert = coherent.clone();
    let mut m = op(GateKind::Measure, &[2], &[]);
    m.classical_bit = Some(0);
    inert.ops.push(m);
    inert.num_classical_bits = 1;
    let inert_answer = b
        .expectation_site_operators(&inert, &ParameterBinding::new(), &ops)
        .unwrap();
    assert!(
        (inert_answer - coherent_answer).abs() <= 1e-12,
        "an inert measurement changed the answer: {inert_answer} vs coherent {coherent_answer}"
    );

    // (b) read: measure the qubit on wire 1 into c0 and flip the qubit on
    // wire 3 when c0 == 1. Deferred to CX(1, 3); wire 1's operator dephased.
    let mut read = coherent.clone();
    let mut m = op(GateKind::Measure, &[1], &[]);
    m.classical_bit = Some(0);
    read.ops.push(m);
    let mut x = op(GateKind::X, &[3], &[]);
    x.condition = Some((0, 1, 1));
    read.ops.push(x);
    read.num_classical_bits = 1;
    let mut hand_deferred = coherent.clone();
    hand_deferred.ops.push(op(GateKind::CX, &[1, 3], &[]));
    let mut dephased = ops.clone();
    dephased[1][1] = C::new(0.0, 0.0);
    dephased[1][2] = C::new(0.0, 0.0);
    let read_answer = b
        .expectation_site_operators(&read, &ParameterBinding::new(), &ops)
        .unwrap();
    let hand_answer = b
        .expectation_site_operators(&hand_deferred, &ParameterBinding::new(), &dephased)
        .unwrap();
    let cx_only = b
        .expectation_site_operators(&hand_deferred, &ParameterBinding::new(), &ops)
        .unwrap();
    assert!(
        (read_answer - hand_answer).abs() <= 1e-12,
        "read measurement {read_answer} vs hand-deferred+dephased {hand_answer}"
    );
    assert!(
        (read_answer - cx_only).abs() > 1e-3,
        "the dephasing changed nothing: {read_answer} vs CX-only {cx_only} — the corpus \
         has no coherence on wire 1 or the dephasing is not applied"
    );
}
