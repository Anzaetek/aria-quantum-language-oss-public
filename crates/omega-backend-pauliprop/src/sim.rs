//! Pauli-propagation backend: backward Heisenberg evolution of an observable.
//!
//! `⟨O⟩ = ⟨0…0| U† O U |0…0⟩` is computed by conjugating the observable (a
//! [`PauliSum`]) by each gate from last to first, then reading off the all-`I/Z`
//! terms. Clifford gates map one Pauli to one Pauli (no growth — exact and
//! width-unbounded); Pauli rotations branch one Pauli into two (the tree),
//! truncated by coefficient magnitude / Pauli weight.
//!
//! See `PAULI_PROPAGATION_PLAN.md` in the `quantum` repo (item 17).

use num_complex::Complex64;
use omega_core::circuit::{CircuitIR, GateKind, GateOp, ParamExpr};
use omega_core::error::{OmegaError, Result};
use omega_core::executor::{Backend, ExecConfig, ExecResult, Observable, PauliOp};
use omega_core::noise::NoiseModel;
use omega_core::params::ParameterBinding;

use crate::pauli::{mul_raw, PauliKey, PauliSum};

const I: Complex64 = Complex64::new(0.0, 1.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// Accelerator hook for the non-Clifford **branch** step — the hot per-term loop
/// `P → cosθ·P + i sinθ·(R·P)`. The generator `R` is passed in packed u64-word
/// symplectic form (`rx`, `rz`, `factor`), with the already-computed `cos`/`sin`,
/// the `max_freq` cap, and qubit count `n`. The hook replaces `sum` with the
/// branched result and returns `true` iff it handled the work; returning `false`
/// (e.g. no GPU present, or too few terms to be worth a device round-trip) lets
/// the CPU path run unchanged, so semantics are identical either way. The CUDA
/// implementation lives in `omega-backend-pauliprop-cuda`.
pub type BranchHook = fn(
    sum: &mut PauliSum,
    rx: &[u64],
    rz: &[u64],
    factor: Complex64,
    cos: f64,
    sin: f64,
    max_freq: Option<u32>,
    n: usize,
) -> bool;

/// Pauli-propagation simulator. Truncation off (`coeff_min = 0`, `max_weight =
/// None`) ⇒ exact; tighten either knob to approximate deep non-Clifford
/// circuits. Exact and cheap for Clifford circuits at any width.
#[derive(Clone, Debug)]
pub struct PauliPropBackend {
    /// Drop terms whose coefficient magnitude falls below this.
    pub coeff_min: f64,
    /// Drop terms whose Pauli weight exceeds this (`None` = no cap).
    pub max_weight: Option<usize>,
    /// Drop terms whose split frequency (number of non-Clifford sin-branches on
    /// the cheapest path to them) exceeds this (`None` = no cap). This is
    /// PauliPropagation.jl's `max_freq` truncation axis. Over-budget sin-branches
    /// are never even created, so it also bounds the tree fan-out directly.
    pub max_freq: Option<u32>,
    /// Optional GPU accelerator for the branch step (see [`BranchHook`]). `None`
    /// = pure CPU. Skipped from `Debug`/equality — it's a code pointer.
    branch_hook: Option<BranchHook>,
    /// Optional per-gate + readout noise model. `None` = exact, noise-free
    /// Heisenberg evolution. When set, each channel's *adjoint* is applied to
    /// the propagating observable — deterministically and exactly (no
    /// trajectories), which is the natural home for Pauli-Lindblad noise.
    noise: Option<NoiseModel>,
}

impl Default for PauliPropBackend {
    fn default() -> Self {
        Self {
            coeff_min: 0.0,
            max_weight: None,
            max_freq: None,
            branch_hook: None,
            noise: None,
        }
    }
}

impl PauliPropBackend {
    /// Exact engine (no truncation).
    pub fn new() -> Self {
        Self::default()
    }

    /// Engine with coefficient-magnitude and Pauli-weight truncation.
    pub fn with_truncation(coeff_min: f64, max_weight: Option<usize>) -> Self {
        Self {
            coeff_min,
            max_weight,
            max_freq: None,
            branch_hook: None,
            noise: None,
        }
    }

    /// Install a GPU accelerator for the branch step (see [`BranchHook`]).
    pub fn with_branch_hook(mut self, hook: BranchHook) -> Self {
        self.branch_hook = Some(hook);
        self
    }

    /// Attach a per-gate + readout noise model (applied via its Heisenberg
    /// adjoint during propagation). Composes with truncation and the branch
    /// accelerator.
    pub fn with_noise(mut self, model: NoiseModel) -> Self {
        self.noise = Some(model);
        self
    }

    /// Engine with the full truncation triple, including PauliPropagation.jl's
    /// split-frequency cap (`max_freq`).
    pub fn with_truncation_freq(
        coeff_min: f64,
        max_weight: Option<usize>,
        max_freq: Option<u32>,
    ) -> Self {
        Self {
            coeff_min,
            max_weight,
            max_freq,
            branch_hook: None,
            noise: None,
        }
    }

    /// Builder: set the split-frequency cap.
    pub fn max_freq(mut self, max_freq: Option<u32>) -> Self {
        self.max_freq = max_freq;
        self
    }

    /// L1 dropped-coefficient mass from the *last* `expectation` call is not
    /// retained on the backend; callers wanting the error budget should use
    /// [`PauliPropBackend::expectation_with_budget`].
    pub fn expectation_with_budget(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<(f64, f64)> {
        let sum = self.propagate(circuit, params, observable)?;
        let mut val = Complex64::new(0.0, 0.0);
        for (k, w) in &sum.terms {
            if k.is_all_iz() {
                val += w.coeff;
            }
        }
        Ok((val.re, sum.dropped_mass))
    }

    /// Backward-propagate `observable` through `circuit`, returning the
    /// resulting Pauli sum (the Heisenberg-evolved observable).
    fn propagate(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<PauliSum> {
        let n = circuit.num_qubits as usize;
        let mut sum = PauliSum::new();
        for (coeff, paulis) in &observable.terms {
            let mut key = PauliKey::identity(n);
            let mut c = Complex64::new(*coeff, 0.0);
            for (q, op) in paulis {
                let q = *q as usize;
                match op {
                    PauliOp::I => {}
                    PauliOp::X => key.x[q] = true,
                    PauliOp::Z => key.z[q] = true,
                    PauliOp::Y => {
                        key.x[q] = true;
                        key.z[q] = true;
                        c *= I; // Y = i·XZ
                    }
                }
            }
            sum.add(key, c);
        }
        // Readout error is the *last* thing to happen in forward time
        // (measurement), so in the backward (adjoint) propagation it is applied
        // *first* — to the raw observable, before any gate.
        if let Some(model) = &self.noise {
            self.apply_readout_adjoint(model, n, &mut sum)?;
        }
        for op in circuit.ops.iter().rev() {
            // A classically-conditioned gate is NOT a plain gate. Its action
            // depends on a measurement outcome, so the circuit is a classical
            // mixture over branches — not one unitary — and observable
            // conjugation cannot express it.
            //
            // This backend used to ignore `op.condition` entirely (zero
            // references to it in the whole crate), which meant a guarded gate
            // was applied **unconditionally and silently** — answering a
            // different circuit, exactly as skipping `Reset` did before the
            // refusal below was added.
            //
            // Measured on `h q0; measure q0 -> c0; if (c==1) x q1` (the shape
            // of `12_feedforward_sometimes_false.qasm`): ignoring the guard
            // gives ⟨Z₁⟩ = −1, "q1 definitely flipped", where the statevector
            // backend in collapse mode gives 0 because the X fires on about
            // half the shots. A gap of 1.0 on an observable bounded in
            // [−1, +1] — the largest error reachable when the truth is 0.
            // Pinned in `tests/conditional_refusal.rs`.
            //
            // Refuse, so the caller falls back to a backend that models it.
            // Found by the N-way matrix work (`FIXES_PLAN.md` K7): statevector,
            // MPS and Pauli all call `condition_satisfied`; this backend was
            // the only one that did not.
            if op.condition.is_some() {
                return Err(OmegaError::Unsupported(
                    "pauliprop: a classically-conditioned gate makes the circuit a mixture \
                     over measurement outcomes, which cannot be represented by observable \
                     conjugation; use the statevector or MPS backend"
                        .into(),
                ));
            }
            // The per-gate channel follows its gate in forward time, so its
            // adjoint precedes the gate's adjoint here.
            if let Some(model) = &self.noise {
                self.apply_gate_noise_adjoint(model, op, &mut sum);
            }
            self.conjugate(op, params, n, &mut sum)?;
            if self.coeff_min > 0.0 || self.max_weight.is_some() || self.max_freq.is_some() {
                sum.truncate(self.coeff_min, self.max_weight, self.max_freq);
            }
        }
        Ok(sum)
    }

    /// Apply the adjoint of the per-gate noise channel that follows `op`.
    ///
    /// Each single-qubit Pauli-Lindblad channel acts diagonally on the Pauli
    /// basis — `N†(P) = λ_P · P` — so the adjoint just rescales each term by a
    /// factor set by its local Pauli on the affected qubit. Amplitude damping is
    /// the one non-unital channel: `Z → (1−γ)Z + γ·I`, which additionally spawns
    /// a lower-weight identity term.
    ///
    /// **Order matters.** The statevector backend applies the forward channels
    /// in the order depolarizing → Pauli → amplitude-damping → phase, so the
    /// Heisenberg adjoint on the observable must apply them in *reverse*:
    /// amplitude damping first, then the diagonal channels. If a diagonal
    /// channel ran first it would rescale the parent `Z` before amplitude
    /// damping spawns its identity term, wrongly scaling that identity by the
    /// diagonal factor. The diagonal channels commute with each other, so only
    /// amplitude damping's leading position is load-bearing.
    fn apply_gate_noise_adjoint(&self, model: &NoiseModel, op: &GateOp, sum: &mut PauliSum) {
        if matches!(op.gate, GateKind::Measure | GateKind::Barrier) {
            return;
        }
        let gate_qubits: Vec<usize> = op.qubits.iter().map(|q| q.0 as usize).collect();
        for &q in &gate_qubits {
            // Amplitude damping γ (non-unital): X,Y → √(1−γ)·; Z → (1−γ)Z + γ·I.
            // Applied FIRST (see the ordering note above) so the spawned identity
            // term is not rescaled by the diagonal channels below.
            let gamma = model.amplitude_damping.at(q);
            if gamma > 0.0 {
                apply_amplitude_damping_adjoint(sum, q, gamma);
            }

            // Depolarizing: identity untouched, any non-I Pauli → (1 − 4p/3).
            // A two-qubit gate's pair selects a per-pair rate when configured.
            let p = model.depolarizing.at_gate(q, &gate_qubits);
            if p > 0.0 {
                let lam = 1.0 - 4.0 * p / 3.0;
                scale_by_local_pauli(sum, q, |x, z| if x || z { lam } else { 1.0 });
            }

            // Phase damping λ: X, Y → (1 − λ); I, Z unchanged.
            let lambda = model.phase_damping.at(q);
            if lambda > 0.0 {
                scale_by_local_pauli(sum, q, |x, _z| if x { 1.0 - lambda } else { 1.0 });
            }

            // Explicit Pauli channel (p_x, p_y, p_z): Pauli-transfer eigenvalues.
            if let Some(pauli) = &model.pauli {
                let (px, py, pz) = (pauli.x.at(q), pauli.y.at(q), pauli.z.at(q));
                if px + py + pz > 0.0 {
                    let pi = 1.0 - px - py - pz;
                    scale_by_local_pauli(sum, q, |x, z| match (x, z) {
                        (false, false) => 1.0,              // I
                        (true, false) => pi + px - py - pz, // X
                        (false, true) => pi - px - py + pz, // Z
                        (true, true) => pi - px + py - pz,  // Y
                    });
                }
            }
        }
    }

    /// Apply the adjoint of readout error to the freshly-seeded observable.
    ///
    /// Readout is modelled as an independent bit-flip on each qubit's Z-basis
    /// outcome, so its adjoint scales the measured `Z`/`Y` components by
    /// `1 − 2p` (leaving `X`), matching `readout_flip` on the statevector
    /// sampler. This backend produces expectation values (no shots), so only
    /// **symmetric** readout has an unambiguous meaning here; asymmetric readout
    /// (`p10 ≠ p01`) is rejected — sample it on `--backend sim` instead.
    fn apply_readout_adjoint(
        &self,
        model: &NoiseModel,
        n: usize,
        sum: &mut PauliSum,
    ) -> Result<()> {
        if model.readout.is_zero() {
            return Ok(());
        }
        for q in 0..n {
            let p10 = model.readout.p10.at(q);
            let p01 = model.readout.p01.at(q);
            if (p10 - p01).abs() > 1e-12 {
                return Err(OmegaError::Unsupported(format!(
                    "pauliprop expectation backend: asymmetric readout (p10={p10}, p01={p01}) \
                     on qubit {q} is not representable as an expectation-value rescale; \
                     sample it on --backend sim instead"
                )));
            }
            let p = p10;
            if p > 0.0 {
                // Bit-flip adjoint: Z,Y → (1−2p)·; X unchanged; I unchanged.
                let lam = 1.0 - 2.0 * p;
                scale_by_local_pauli(sum, q, |_x, z| if z { lam } else { 1.0 });
            }
        }
        Ok(())
    }

    /// Conjugate the whole sum by one gate `Gᵏ`: `O ← Gᵏ† O Gᵏ`.
    fn conjugate(
        &self,
        op: &GateOp,
        params: &ParameterBinding,
        n: usize,
        sum: &mut PauliSum,
    ) -> Result<()> {
        let q = |i: usize| op.qubits[i].0 as usize;
        match op.gate {
            // ----- single-qubit Cliffords (one Pauli → one Pauli) -----
            GateKind::H => self.map_single(sum, q(0), CLIFF_H),
            GateKind::X => self.map_single(sum, q(0), CLIFF_X),
            GateKind::Y => self.map_single(sum, q(0), CLIFF_Y),
            GateKind::Z => self.map_single(sum, q(0), CLIFF_Z),
            GateKind::S => self.map_single(sum, q(0), CLIFF_S),
            GateKind::Sdg => self.map_single(sum, q(0), CLIFF_SDG),

            // ----- two-qubit Cliffords -----
            GateKind::CX => self.map_two(sum, q(0), q(1), CX_IMG),
            GateKind::CZ => self.map_two(sum, q(0), q(1), CZ_IMG),
            GateKind::Swap => self.map_two(sum, q(0), q(1), SWAP_IMG),

            // ----- Pauli rotations (one Pauli → two: the branch / tree) -----
            GateKind::Rz => self.branch(
                sum,
                &gen_single(n, q(0), 'z'),
                resolve(&op.params[0], params)?,
            ),
            GateKind::Rx => self.branch(
                sum,
                &gen_single(n, q(0), 'x'),
                resolve(&op.params[0], params)?,
            ),
            GateKind::Ry => self.branch(
                sum,
                &gen_single(n, q(0), 'y'),
                resolve(&op.params[0], params)?,
            ),
            // U1(λ) = diag(1, e^{iλ}) = e^{iλ/2}·Rz(λ); the global phase cancels
            // under conjugation, so it propagates exactly as Rz(λ).
            GateKind::U1 => self.branch(
                sum,
                &gen_single(n, q(0), 'z'),
                resolve(&op.params[0], params)?,
            ),
            GateKind::T => self.branch(sum, &gen_single(n, q(0), 'z'), std::f64::consts::FRAC_PI_4),
            GateKind::Tdg => {
                self.branch(sum, &gen_single(n, q(0), 'z'), -std::f64::consts::FRAC_PI_4)
            }
            // CRz(θ) = Rz_t(θ/2) · Rzz_{c,t}(-θ/2) — two commuting Pauli rotations.
            GateKind::CRz => {
                let theta = resolve(&op.params[0], params)?;
                self.branch(sum, &gen_single(n, q(1), 'z'), theta / 2.0);
                self.branch(sum, &gen_zz(n, q(0), q(1)), -theta / 2.0);
            }

            // ----- no-ops for the unitary-conjugation picture -----
            GateKind::Id | GateKind::Barrier | GateKind::Measure => {}

            // Reset is NOT a no-op. It is a non-unitary channel
            // (rho -> |0><0|_q (x) Tr_q(rho)) and this backend evolves
            // observables by unitary conjugation, which cannot express it.
            // Silently skipping it answered a DIFFERENT circuit: after
            // Bell + reset(q0) it reported <Z_0> = 0 where the channel gives
            // +1. Refuse so the CLI falls back to a backend that models it.
            GateKind::Reset => {
                return Err(OmegaError::Unsupported(
                    "pauliprop: Reset is a non-unitary channel and cannot be represented by \
                     observable conjugation; use the statevector or MPS backend"
                        .into(),
                ));
            }

            ref other => {
                return Err(OmegaError::Unsupported(format!(
                    "pauliprop: gate {other:?} not yet supported \
                     (Clifford H/X/Y/Z/S/Sdg/CX/CZ/Swap + Rx/Ry/Rz/U1/T/Tdg/CRz)"
                )));
            }
        }
        Ok(())
    }

    /// Apply a single-qubit Clifford's local `(X→, Z→)` images to every term.
    /// Cliffords map one Pauli to one Pauli, so the split frequency is carried
    /// through unchanged.
    fn map_single(&self, sum: &mut PauliSum, q: usize, img: SingleImg) {
        let mut out = PauliSum::new();
        out.dropped_mass = sum.dropped_mass;
        for (key, w) in sum.terms.drain() {
            let (nx, nz, f) = img.apply(key.x[q], key.z[q]);
            let mut k = key;
            k.x[q] = nx;
            k.z[q] = nz;
            out.add_weighted(k, w.coeff * f, w.freq);
        }
        *sum = out;
    }

    /// Apply a two-qubit Clifford's four generator images to every term.
    fn map_two(&self, sum: &mut PauliSum, c: usize, t: usize, img: TwoImg) {
        let mut out = PauliSum::new();
        out.dropped_mass = sum.dropped_mass;
        for (key, w) in sum.terms.drain() {
            // Compose the four generator images present on (c, t).
            let mut acc = Gen2::ident();
            if key.x[c] {
                acc = acc.mul(&img.xc);
            }
            if key.z[c] {
                acc = acc.mul(&img.zc);
            }
            if key.x[t] {
                acc = acc.mul(&img.xt);
            }
            if key.z[t] {
                acc = acc.mul(&img.zt);
            }
            let mut k = key;
            k.x[c] = acc.xc;
            k.z[c] = acc.zc;
            k.x[t] = acc.xt;
            k.z[t] = acc.zt;
            out.add_weighted(k, w.coeff * acc.f, w.freq);
        }
        *sum = out;
    }

    /// Conjugate by a Pauli rotation `exp(-iθ/2 · R)` with Hermitian generator
    /// `R` ([`Gen`]). Terms that anticommute with `R` branch into two,
    /// `P → cosθ·P + i sinθ·(R·P)` — the non-Clifford tree step. Works for any
    /// single- or multi-qubit Pauli `R` (Rz/Rx/Ry → 1-qubit; the ZZ factor of
    /// CRz → 2-qubit), so the fan-out is bounded only by truncation.
    fn branch(&self, sum: &mut PauliSum, r: &Gen, theta: f64) {
        let (cos, sin) = (theta.cos(), theta.sin());
        // Offer the branch expansion to an accelerator (GPU). If it declines
        // (no device / too few terms), fall through to the CPU loop below — the
        // two are semantically identical, so results never depend on the path.
        if let Some(hook) = self.branch_hook {
            let n = r.gx.len();
            let rx = pack_bits(&r.gx);
            let rz = pack_bits(&r.gz);
            if hook(sum, &rx, &rz, r.factor, cos, sin, self.max_freq, n) {
                return;
            }
        }
        let mut out = PauliSum::new();
        out.dropped_mass = sum.dropped_mass;
        for (key, w) in sum.terms.drain() {
            // Anticommute ⇔ odd symplectic product ⟨P, R⟩ = Σ (Pₓ·R_z ⊕ P_z·Rₓ).
            let mut anti = false;
            for i in 0..key.x.len() {
                if (key.x[i] && r.gz[i]) ^ (key.z[i] && r.gx[i]) {
                    anti = !anti;
                }
            }
            if !anti {
                out.add_weighted(key, w.coeff, w.freq); // commutes → unchanged
                continue;
            }
            // cosθ·P keeps the same frequency; the sin child gains one split.
            out.add_weighted(key.clone(), w.coeff * cos, w.freq);
            let child_freq = w.freq + 1;
            let sin_coeff = w.coeff * I * sin * r.factor;
            if self.max_freq.is_some_and(|m| child_freq > m) {
                // Over the split-frequency budget: don't create the child, but
                // certify the discarded L1 mass so the error stays bounded.
                out.dropped_mass += sin_coeff.norm();
                continue;
            }
            // R · P (R on the left): raw product carries the ± sign.
            let (rk, sign) = mul_raw(&r.gx, &r.gz, &key.x, &key.z);
            out.add_weighted(rk, sin_coeff * sign, child_freq);
        }
        *sum = out;
    }
}

// --------------------------------------------------------------------------
// Single-qubit Clifford images: G† X G and G† Z G, each as (x, z, factor).
// --------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct SingleImg {
    /// G†XG = factor · raw(ax, az)
    ax: bool,
    az: bool,
    fx: Complex64,
    /// G†ZG = factor · raw(bx, bz)
    bx: bool,
    bz: bool,
    fz: Complex64,
}

impl SingleImg {
    /// Image of the local raw operator `X^{x} Z^{z}` under this Clifford.
    fn apply(&self, x: bool, z: bool) -> (bool, bool, Complex64) {
        match (x, z) {
            (false, false) => (false, false, ONE),
            (true, false) => (self.ax, self.az, self.fx),
            (false, true) => (self.bx, self.bz, self.fz),
            (true, true) => {
                // (G†XG)(G†ZG): raw-multiply the two images.
                let (k, sign) = mul_raw(&[self.ax], &[self.az], &[self.bx], &[self.bz]);
                (k.x[0], k.z[0], self.fx * self.fz * sign)
            }
        }
    }
}

const CLIFF_H: SingleImg = SingleImg {
    ax: false,
    az: true,
    fx: ONE,
    bx: true,
    bz: false,
    fz: ONE,
};
const CLIFF_X: SingleImg = SingleImg {
    ax: true,
    az: false,
    fx: ONE,
    bx: false,
    bz: true,
    fz: Complex64::new(-1.0, 0.0),
};
const CLIFF_Y: SingleImg = SingleImg {
    ax: true,
    az: false,
    fx: Complex64::new(-1.0, 0.0),
    bx: false,
    bz: true,
    fz: Complex64::new(-1.0, 0.0),
};
const CLIFF_Z: SingleImg = SingleImg {
    ax: true,
    az: false,
    fx: Complex64::new(-1.0, 0.0),
    bx: false,
    bz: true,
    fz: ONE,
};
// S† X S = -Y = (-i)·raw(1,1);  S† Z S = Z.
const CLIFF_S: SingleImg = SingleImg {
    ax: true,
    az: true,
    fx: Complex64::new(0.0, -1.0),
    bx: false,
    bz: true,
    fz: ONE,
};
// S X S† = +Y = (+i)·raw(1,1);  Z → Z.
const CLIFF_SDG: SingleImg = SingleImg {
    ax: true,
    az: true,
    fx: Complex64::new(0.0, 1.0),
    bx: false,
    bz: true,
    fz: ONE,
};

// --------------------------------------------------------------------------
// Two-qubit Clifford images: G†·{Xc,Zc,Xt,Zt}·G as 2-qubit raw operators.
// --------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Gen2 {
    xc: bool,
    zc: bool,
    xt: bool,
    zt: bool,
    f: Complex64,
}

impl Gen2 {
    const fn new(xc: bool, zc: bool, xt: bool, zt: bool) -> Self {
        Self {
            xc,
            zc,
            xt,
            zt,
            f: ONE,
        }
    }
    fn ident() -> Self {
        Self {
            xc: false,
            zc: false,
            xt: false,
            zt: false,
            f: ONE,
        }
    }
    /// Raw-multiply two 2-qubit operators (sign per qubit: Z_left past X_right).
    fn mul(&self, o: &Gen2) -> Gen2 {
        let mut neg = 0u32;
        if self.zc && o.xc {
            neg += 1;
        }
        if self.zt && o.xt {
            neg += 1;
        }
        let sign = if neg.is_multiple_of(2) {
            ONE
        } else {
            Complex64::new(-1.0, 0.0)
        };
        Gen2 {
            xc: self.xc ^ o.xc,
            zc: self.zc ^ o.zc,
            xt: self.xt ^ o.xt,
            zt: self.zt ^ o.zt,
            f: self.f * o.f * sign,
        }
    }
}

#[derive(Clone, Copy)]
struct TwoImg {
    xc: Gen2,
    zc: Gen2,
    xt: Gen2,
    zt: Gen2,
}

// CX: Xc→XcXt, Zc→Zc, Xt→Xt, Zt→ZcZt.
const CX_IMG: TwoImg = TwoImg {
    xc: Gen2::new(true, false, true, false),
    zc: Gen2::new(false, true, false, false),
    xt: Gen2::new(false, false, true, false),
    zt: Gen2::new(false, true, false, true),
};
// CZ: Xc→XcZt, Zc→Zc, Xt→ZcXt, Zt→Zt.
const CZ_IMG: TwoImg = TwoImg {
    xc: Gen2::new(true, false, false, true),
    zc: Gen2::new(false, true, false, false),
    xt: Gen2::new(false, true, true, false),
    zt: Gen2::new(false, false, false, true),
};
// Swap: Xc↔Xt, Zc↔Zt.
const SWAP_IMG: TwoImg = TwoImg {
    xc: Gen2::new(false, false, true, false),
    zc: Gen2::new(false, false, false, true),
    xt: Gen2::new(true, false, false, false),
    zt: Gen2::new(false, true, false, false),
};

// --------------------------------------------------------------------------
// Pauli rotation generators (Hermitian, over the full register).
// --------------------------------------------------------------------------

/// A Hermitian Pauli `R = factor · raw(gx, gz)` generating a rotation
/// `exp(-iθ/2 R)`. Stored full-register so single- and multi-qubit generators
/// (e.g. the `ZZ` factor of `CRz`) share one code path.
struct Gen {
    gx: Vec<bool>,
    gz: Vec<bool>,
    factor: Complex64,
}

/// Single-qubit rotation generator on qubit `q`: `'z'`→Z, `'x'`→X, `'y'`→Y.
fn gen_single(n: usize, q: usize, axis: char) -> Gen {
    let mut gx = vec![false; n];
    let mut gz = vec![false; n];
    let factor = match axis {
        'z' => {
            gz[q] = true;
            ONE
        }
        'x' => {
            gx[q] = true;
            ONE
        }
        'y' => {
            gx[q] = true;
            gz[q] = true;
            I // Y = i·XZ
        }
        _ => unreachable!("bad axis"),
    };
    Gen { gx, gz, factor }
}

/// Two-qubit `Z⊗Z` generator on `(c, t)` (the entangling factor of `CRz`).
fn gen_zz(n: usize, c: usize, t: usize) -> Gen {
    let gx = vec![false; n];
    let mut gz = vec![false; n];
    gz[c] = true;
    gz[t] = true;
    Gen {
        gx,
        gz,
        factor: ONE,
    }
}

/// Pack a per-qubit boolean vector into little-endian u64 words: qubit `i` is
/// bit `i % 64` of word `i / 64`. The GPU branch kernel consumes this SoA layout.
pub fn pack_bits(bits: &[bool]) -> Vec<u64> {
    let words = bits.len().div_ceil(64);
    let mut out = vec![0u64; words];
    for (i, &b) in bits.iter().enumerate() {
        if b {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

/// Inverse of [`pack_bits`] for `n` qubits.
pub fn unpack_bits(words: &[u64], n: usize) -> Vec<bool> {
    (0..n)
        .map(|i| (words[i / 64] >> (i % 64)) & 1 == 1)
        .collect()
}

/// Resolve a (concrete or symbolic) parameter to a number.
fn resolve(expr: &ParamExpr, params: &ParameterBinding) -> Result<f64> {
    params.resolve(expr)
}

/// Rescale every term's coefficient by a factor determined by its local Pauli
/// `(x, z)` on qubit `q`. Used for the diagonal (Pauli-Lindblad) channel
/// adjoints, which map each Pauli to a scalar multiple of itself — so the keys
/// don't change and we can mutate coefficients in place.
fn scale_by_local_pauli<F: Fn(bool, bool) -> f64>(sum: &mut PauliSum, q: usize, factor: F) {
    for (key, w) in sum.terms.iter_mut() {
        let f = factor(key.x[q], key.z[q]);
        if f != 1.0 {
            w.coeff *= f;
        }
    }
}

/// Adjoint of amplitude damping on qubit `q`: `X,Y → √(1−γ)·`, `I → I`, and the
/// non-unital `Z → Z + γ·I` (which spawns a lower-weight identity companion).
/// Rebuilds the term map so spawned/merged keys accumulate correctly.
fn apply_amplitude_damping_adjoint(sum: &mut PauliSum, q: usize, gamma: f64) {
    let s = (1.0 - gamma).sqrt();
    let old = std::mem::take(&mut sum.terms);
    for (key, w) in old {
        match (key.x[q], key.z[q]) {
            (false, false) => sum.add_weighted(key, w.coeff, w.freq), // I
            (true, false) | (true, true) => sum.add_weighted(key, w.coeff * s, w.freq), // X, Y
            (false, true) => {
                // Z → (1−γ)·Z + γ·I: the Z term is attenuated and a companion
                // identity-on-q term is added with weight γ. (Λ†(Z) =
                // diag(1, 2γ−1) = γ·I + (1−γ)·Z.)
                let mut ident = key.clone();
                ident.z[q] = false;
                sum.add_weighted(key, w.coeff * (1.0 - gamma), w.freq);
                sum.add_weighted(ident, w.coeff * gamma, w.freq);
            }
        }
    }
}

impl Backend for PauliPropBackend {
    fn name(&self) -> &str {
        "pauliprop"
    }

    fn execute(
        &self,
        _circuit: &CircuitIR,
        _params: &ParameterBinding,
        _config: &ExecConfig,
    ) -> Result<ExecResult> {
        Err(OmegaError::Unsupported(
            "pauliprop is an expectation-value backend; use `expectation()`, not execute/sampling"
                .into(),
        ))
    }

    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<f64> {
        self.expectation_with_budget(circuit, params, observable)
            .map(|(v, _)| v)
    }
}
