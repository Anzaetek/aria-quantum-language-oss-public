//! Q3.1 (PLAN-QUDIT.md §Q3): the MPS core with a per-site physical
//! dimension.
//!
//! Four things are pinned here:
//!
//! 1. **d = 2 is bit-identical through both APIs** — the fixed-size qubit
//!    wrappers (`apply_1q`, `apply_2q_distant`) and the slice kernels
//!    (`apply_1`, `apply_2_distant`) produce `==` statevectors and `==`
//!    discarded weight, exact and truncating, fixed-rank and adaptive. The
//!    regression against the PRE-generalisation kernels is the untouched
//!    qubit parity estate (`src/mps.rs` tests, `src/sim.rs` tests and the
//!    other files in `tests/`).
//! 2. **d = 3 and mixed radix are exact in the exact regime** against a dense
//!    mixed-radix reference written inline, including distant pairs in both
//!    orientations, so the SWAP network (which carries each site's dimension
//!    with its state) and the gate re-orientation are both exercised.
//! 3. **Truncation at d = 3 is norm-accounted**: the split's debug assertion
//!    `kept + discarded == ‖Θ'‖²` runs on every split (tests build in debug),
//!    and the certificate is non-zero when the bond is starved.
//! 4. **What still refuses a non-qubit site** (after Q3.2): a qubit-sized
//!    operator on a qutrit site, and a digit `>= 2` packed into a bit key.

use num_complex::Complex64;
use omega_backend_mps::mps::{Mps, MpsTensor};

type C = Complex64;

fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}

/// Deterministic xorshift — same class on every machine and run.
struct Xs(u64);
impl Xs {
    fn f(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.f() * n as f64) as usize).min(n - 1)
    }
}

// ---- 1. d = 2 bit-identity -------------------------------------------------

fn h() -> [C; 4] {
    let s = 1.0 / 2.0_f64.sqrt();
    [c(s, 0.0), c(s, 0.0), c(s, 0.0), c(-s, 0.0)]
}
fn rx(t: f64) -> [C; 4] {
    let (co, si) = ((t / 2.0).cos(), (t / 2.0).sin());
    [c(co, 0.0), c(0.0, -si), c(0.0, -si), c(co, 0.0)]
}
fn ry(t: f64) -> [C; 4] {
    let (co, si) = ((t / 2.0).cos(), (t / 2.0).sin());
    [c(co, 0.0), c(-si, 0.0), c(si, 0.0), c(co, 0.0)]
}
fn cx() -> [C; 16] {
    let (o, i) = (c(0.0, 0.0), c(1.0, 0.0));
    [i, o, o, o, o, i, o, o, o, o, o, i, o, o, i, o]
}
/// A non-symmetric entangler so the re-orientation of a reversed pair is
/// actually exercised (CX alone would still be distinguishable, but a
/// controlled rotation with a phase makes every entry matter).
fn cry_phase(t: f64) -> [C; 16] {
    let (o, i) = (c(0.0, 0.0), c(1.0, 0.0));
    let r = ry(t);
    let p = C::from_polar(1.0, 0.3 * t);
    [
        i, o, o, o, //
        o, p, o, o, //
        o, o, r[0], r[1], //
        o, o, r[2], r[3],
    ]
}

enum QOp {
    One(usize, [C; 4]),
    Two(usize, usize, [C; 16]),
}

fn qubit_circuit(n: usize, depth: usize, seed: u64) -> Vec<QOp> {
    let mut rng = Xs(seed);
    let mut ops = Vec::new();
    for _ in 0..depth {
        for q in 0..n {
            let g = match rng.below(3) {
                0 => h(),
                1 => rx(rng.f() * std::f64::consts::TAU),
                _ => ry(rng.f() * std::f64::consts::TAU),
            };
            ops.push(QOp::One(q, g));
        }
        for _ in 0..n {
            let a = rng.below(n);
            let mut b = rng.below(n);
            while b == a {
                b = rng.below(n);
            }
            let g = if rng.f() < 0.5 {
                cx()
            } else {
                cry_phase(rng.f() * std::f64::consts::TAU)
            };
            ops.push(QOp::Two(a, b, g));
        }
    }
    ops
}

fn run_fixed(ops: &[QOp], n: usize, chi: usize, eps: Option<f64>) -> Mps {
    let mut m = Mps::zero_state(n, chi);
    if let Some(e) = eps {
        m.set_adaptive_eps(e);
    }
    for op in ops {
        match op {
            QOp::One(q, g) => m.apply_1q(*q, g),
            QOp::Two(a, b, g) => m.apply_2q_distant(*a, *b, g),
        }
    }
    m
}

fn run_slices(ops: &[QOp], n: usize, chi: usize, eps: Option<f64>) -> Mps {
    let mut m = Mps::zero_state_dims(&vec![2; n], chi);
    if let Some(e) = eps {
        m.set_adaptive_eps(e);
    }
    for op in ops {
        match op {
            QOp::One(q, g) => m.apply_1(*q, &g[..]),
            QOp::Two(a, b, g) => m.apply_2_distant(*a, *b, &g[..]),
        }
    }
    m
}

#[test]
fn d2_slice_api_is_bit_identical_to_the_qubit_api() {
    const N: usize = 6;
    let variants: [(&str, usize, Option<f64>); 4] = [
        ("exact", 64, None),
        ("chi-capped", 3, None),
        ("adaptive", 64, Some(1e-2)),
        ("adaptive+capped", 4, Some(1e-3)),
    ];
    let mut truncated = 0;
    for seed in [1u64, 0x51_7e, 0xdead_beef] {
        let ops = qubit_circuit(N, 4, seed);
        for (label, chi, eps) in variants {
            let a = run_fixed(&ops, N, chi, eps);
            let b = run_slices(&ops, N, chi, eps);
            assert_eq!(a.dims, vec![2; N], "{label}: dims drifted");
            assert_eq!(b.dims, vec![2; N], "{label}: dims drifted");
            let (sa, sb) = (a.to_statevector(), b.to_statevector());
            assert_eq!(sa.len(), 1 << N);
            for (i, (x, y)) in sa.iter().zip(sb.iter()).enumerate() {
                assert!(
                    x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits(),
                    "{label} seed {seed}: amplitude {i} differs: {x} vs {y}"
                );
            }
            assert_eq!(
                a.discarded_weight.to_bits(),
                b.discarded_weight.to_bits(),
                "{label} seed {seed}"
            );
            assert_eq!(
                a.fidelity_estimate.to_bits(),
                b.fidelity_estimate.to_bits(),
                "{label} seed {seed}"
            );
            assert_eq!(
                a.max_bond_reached, b.max_bond_reached,
                "{label} seed {seed}"
            );
            if a.discarded_weight > 0.0 {
                truncated += 1;
            }
        }
    }
    assert!(
        truncated >= 3,
        "the corpus must truncate somewhere: {truncated}"
    );
}

/// The generalised contractions (`right_environments`, `norm_sqr`,
/// `state_norm_sqr_contracted`) agree with the qubit expectation sweep on a
/// qubit chain — the norm taken two ways is the same number it always was.
#[test]
fn d2_norm_paths_still_agree() {
    let ops = qubit_circuit(5, 3, 7);
    let m = run_fixed(&ops, 5, 2, None);
    let a = m.norm_sqr();
    let b = m.state_norm_sqr_contracted();
    let id = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
    let cc = m.contract_product_operator(&[id; 5]).re;
    assert_eq!(b.to_bits(), cc.to_bits());
    assert!((a - b).abs() < 1e-12 * a.max(1.0), "{a} vs {b}");
    assert!(a < 1.0, "chi = 2 must have truncated: norm {a}");
}

// ---- 2. d = 3 / mixed radix, exact regime ----------------------------------

/// Random `n×n` unitary, row-major: Gram-Schmidt on the columns of a random
/// complex matrix.
fn random_unitary(n: usize, rng: &mut Xs) -> Vec<C> {
    let mut cols: Vec<Vec<C>> = (0..n)
        .map(|_| {
            (0..n)
                .map(|_| c(2.0 * rng.f() - 1.0, 2.0 * rng.f() - 1.0))
                .collect()
        })
        .collect();
    for j in 0..n {
        for k in 0..j {
            let (lo, hi) = cols.split_at_mut(j);
            let proj: C = lo[k]
                .iter()
                .zip(hi[0].iter())
                .map(|(a, b)| a.conj() * b)
                .sum();
            for (x, e) in hi[0].iter_mut().zip(lo[k].iter()) {
                *x -= proj * e;
            }
        }
        let nrm = cols[j].iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        for x in cols[j].iter_mut() {
            *x /= nrm;
        }
    }
    let mut u = vec![c(0.0, 0.0); n * n];
    for (j, col) in cols.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            u[i * n + j] = v;
        }
    }
    // Sanity: U†U = I.
    for a in 0..n {
        for b in 0..n {
            let s: C = (0..n).map(|i| u[i * n + a].conj() * u[i * n + b]).sum();
            let want = if a == b { 1.0 } else { 0.0 };
            assert!((s - c(want, 0.0)).norm() < 1e-12);
        }
    }
    u
}

/// Dense mixed-radix statevector, site 0 the least-significant digit — the
/// convention `Mps::to_statevector` documents.
struct Dense {
    dims: Vec<usize>,
    amp: Vec<C>,
}
impl Dense {
    fn zero(dims: &[usize]) -> Self {
        let n: usize = dims.iter().product();
        let mut amp = vec![c(0.0, 0.0); n];
        amp[0] = c(1.0, 0.0);
        Dense {
            dims: dims.to_vec(),
            amp,
        }
    }
    fn digits(&self, mut i: usize) -> Vec<usize> {
        self.dims
            .iter()
            .map(|&d| {
                let x = i % d;
                i /= d;
                x
            })
            .collect()
    }
    fn index(&self, digits: &[usize]) -> usize {
        let mut i = 0;
        for (q, &d) in self.dims.iter().enumerate().rev() {
            i = i * d + digits[q];
        }
        i
    }
    fn apply_1(&mut self, q: usize, g: &[C]) {
        let d = self.dims[q];
        let mut out = vec![c(0.0, 0.0); self.amp.len()];
        for (i, &a) in self.amp.iter().enumerate() {
            let mut dg = self.digits(i);
            let s = dg[q];
            for sp in 0..d {
                dg[q] = sp;
                out[self.index(&dg)] += g[sp * d + s] * a;
            }
        }
        self.amp = out;
    }
    /// `g` acts on `(q0, q1)` with combined index `s0 * d1 + s1`.
    fn apply_2(&mut self, q0: usize, q1: usize, g: &[C]) {
        let (d0, d1) = (self.dims[q0], self.dims[q1]);
        let n = d0 * d1;
        let mut out = vec![c(0.0, 0.0); self.amp.len()];
        for (i, &a) in self.amp.iter().enumerate() {
            let mut dg = self.digits(i);
            let col = dg[q0] * d1 + dg[q1];
            for row in 0..n {
                dg[q0] = row / d1;
                dg[q1] = row % d1;
                out[self.index(&dg)] += g[row * n + col] * a;
            }
        }
        self.amp = out;
    }
}

enum DOp {
    One(usize),
    Two(usize, usize),
}

/// Run the same random unitaries through the MPS and the dense reference and
/// compare at 1e-12.
fn qudit_exact_case(dims: &[usize], program: &[DOp], seed: u64) -> Mps {
    let mut rng = Xs(seed);
    let mut mps = Mps::zero_state_dims(dims, 4096);
    let mut dense = Dense::zero(dims);
    for op in program {
        match *op {
            DOp::One(q) => {
                let u = random_unitary(dims[q], &mut rng);
                mps.apply_1(q, &u);
                dense.apply_1(q, &u);
            }
            DOp::Two(a, b) => {
                let u = random_unitary(dims[a] * dims[b], &mut rng);
                mps.apply_2_distant(a, b, &u);
                dense.apply_2(a, b, &u);
            }
        }
        assert_eq!(mps.dims, dims, "dims must be restored after every gate");
        for (q, t) in mps.tensors.iter().enumerate() {
            assert_eq!(t.phys, dims[q], "site {q} tensor phys");
        }
    }
    let sv = mps.to_statevector();
    assert_eq!(sv.len(), dense.amp.len());
    let mut worst = 0.0f64;
    for (x, y) in sv.iter().zip(dense.amp.iter()) {
        worst = worst.max((x - y).norm());
    }
    assert!(
        worst < 1e-12,
        "dims {dims:?}: max |mps - dense| = {worst:e}"
    );
    // Bond unbounded: nothing above the SVD's 1e-14 rank floor is dropped, so
    // the certificate stays at rounding level (numerically-zero σ are cut).
    assert!(
        mps.discarded_weight < 1e-20,
        "exact regime must not truncate: {:e}",
        mps.discarded_weight
    );
    assert!(1.0 - mps.fidelity_estimate < 1e-20);
    // The generalised contractions see an exactly unit-norm state.
    assert!((mps.norm_sqr() - 1.0).abs() < 1e-12, "{}", mps.norm_sqr());
    assert!((mps.state_norm_sqr_contracted() - 1.0).abs() < 1e-12);
    mps
}

#[test]
fn qutrit_chain_matches_dense_in_the_exact_regime() {
    use DOp::*;
    let program = [
        One(0),
        One(1),
        One(2),
        Two(0, 1),
        Two(1, 2),
        // distant, forward: the SWAP network with d = 3 on both ends
        Two(0, 2),
        One(1),
        // distant, reversed: SWAP network + gate re-orientation
        Two(2, 0),
        // adjacent, reversed
        Two(1, 0),
        One(2),
        Two(2, 1),
    ];
    qudit_exact_case(&[3, 3, 3], &program, 0x0003_0303);
}

#[test]
fn mixed_radix_2_3_2_matches_dense() {
    use DOp::*;
    let program = [
        One(0),
        One(1),
        One(2),
        Two(0, 1),
        Two(1, 2),
        // distant (2,2) pair across a qutrit: the SWAPs are 6x6 and reshape
        Two(0, 2),
        Two(2, 0),
        Two(2, 1),
        One(1),
        Two(1, 0),
    ];
    qudit_exact_case(&[2, 3, 2], &program, 0x0002_0302);
}

/// Distant pairs across sites of three different dimensions: every SWAP is a
/// reshaping one, and a mis-restored `dims` would fail the per-gate check.
#[test]
fn mixed_radix_2_3_4_2_distant_pairs_match_dense() {
    use DOp::*;
    let program = [
        One(0),
        One(1),
        One(2),
        One(3),
        Two(0, 3),
        Two(3, 0),
        Two(1, 3),
        Two(3, 1),
        Two(0, 2),
        Two(2, 0),
        Two(1, 2),
    ];
    qudit_exact_case(&[2, 3, 4, 2], &program, 0x0203_0402);
}

// ---- 3. truncation at d = 3 -----------------------------------------------

#[test]
fn qutrit_truncation_is_norm_accounted() {
    let dims = [3usize, 3, 3, 3];
    let mut rng = Xs(0x7e3c);
    let mut mps = Mps::zero_state_dims(&dims, 2);
    for q in 0..4 {
        let u = random_unitary(3, &mut rng);
        mps.apply_1(q, &u);
    }
    // Every split below runs the `kept + dropped == ‖Θ'‖²` debug assertion.
    for pair in [(0, 1), (1, 2), (2, 3), (0, 1), (1, 2), (0, 3), (3, 1)] {
        let u = random_unitary(9, &mut rng);
        mps.apply_2_distant(pair.0, pair.1, &u);
        assert_eq!(mps.dims, dims);
    }
    assert!(
        mps.discarded_weight > 0.0,
        "chi = 2 on qutrits must truncate: {}",
        mps.discarded_weight
    );
    assert!(mps.fidelity_estimate < 1.0);
    assert!(mps.max_bond_reached <= 2);
    // Readout still normalises, and the raw norm shows the loss.
    let sv = mps.to_statevector();
    assert_eq!(sv.len(), 81);
    let n2: f64 = sv.iter().map(|z| z.norm_sqr()).sum();
    assert!((n2 - 1.0).abs() < 1e-12);
    let raw = mps.norm_sqr();
    assert!(raw < 1.0, "raw norm {raw}");
    assert!((raw - mps.state_norm_sqr_contracted()).abs() < 1e-12);
}

/// The accelerator hook is a qubit kernel: it must never see a qutrit pair.
#[test]
fn contract_hook_is_not_offered_a_qudit_pair() {
    fn refuse(
        _: &MpsTensor,
        _: &MpsTensor,
        _: &[C; 16],
        _: usize,
        _: f64,
    ) -> Option<(MpsTensor, MpsTensor, f64)> {
        panic!("the qubit accelerator hook was offered a non-qubit pair");
    }
    let mut rng = Xs(99);
    let mut mps = Mps::zero_state_dims(&[3, 2, 3], 64);
    mps.set_contract_fn(refuse);
    mps.apply_1(0, &random_unitary(3, &mut rng));
    mps.apply_2(0, &random_unitary(6, &mut rng));
    mps.apply_2_distant(0, 2, &random_unitary(9, &mut rng));
}

// ---- 4. what still refuses a non-qubit site --------------------------------
//
// Q3.2 generalised sampling, measurement and the product-operator sweep to
// per-site d (their positive tests are in `tests/qudit_backend.rs`). What
// remains refused: a 2x2 operator handed to a qutrit site, and a digit >= 2
// packed into a bit-string key. These three tests replace Q3.1's
// "qubit-only path" refusals, which no longer exist.

#[test]
#[should_panic(expected = "operator must be 3x3")]
fn expectation_product_refuses_a_qubit_operator_on_a_qutrit_site() {
    let mps = Mps::zero_state_dims(&[3, 3], 8);
    let id = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
    let _ = mps.expectation_product(&[id, id]);
}

#[test]
#[should_panic(expected = "is not a bit")]
fn sample_refuses_to_pack_a_qutrit_digit_into_a_bit_key() {
    use rand::SeedableRng;
    // |2, 0⟩: the sampler draws digit 2 on site 0 with certainty, and the
    // u64 bit key cannot hold it.
    let mut mps = Mps::zero_state_dims(&[3, 3], 8);
    let mut shift = vec![c(0.0, 0.0); 9];
    for k in 0..3 {
        shift[((k + 1) % 3) * 3 + k] = c(1.0, 0.0);
    }
    mps.apply_1(0, &shift);
    mps.apply_1(0, &shift);
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let _ = mps.sample(&mut rng);
}

#[test]
fn measure_site_on_a_mixed_chain_returns_the_level_it_collapsed_to() {
    use rand::SeedableRng;
    // |0⟩ ⊗ |1⟩ on [2, 3]: the qutrit site is in level 1 with certainty.
    let mut mps = Mps::zero_state_dims(&[2, 3], 8);
    let mut shift = vec![c(0.0, 0.0); 9];
    for k in 0..3 {
        shift[((k + 1) % 3) * 3 + k] = c(1.0, 0.0);
    }
    mps.apply_1(1, &shift);
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    assert_eq!(mps.measure_site(0, &mut rng), 0);
    assert_eq!(mps.measure_site(1, &mut rng), 1);
}

#[test]
#[should_panic(expected = "use apply_1")]
fn apply_1q_refuses_a_qutrit_site() {
    let mut mps = Mps::zero_state_dims(&[3], 8);
    mps.apply_1q(0, &h());
}

#[test]
#[should_panic(expected = "use the slice API")]
fn apply_2q_distant_refuses_a_qutrit_pair() {
    let mut mps = Mps::zero_state_dims(&[2, 2, 3], 8);
    mps.apply_2q_distant(0, 2, &cx());
}

#[test]
#[should_panic(expected = "d >= 2")]
fn a_dimension_below_two_is_refused() {
    let _ = Mps::zero_state_dims(&[2, 1, 2], 8);
}
