// SPDX-License-Identifier: Apache-2.0
//! A **semantic** check on `hhl.aria`, as opposed to a transport check.
//!
//! `aria-verify -- hhl` compares omega's `⟨Z_q⟩` profile against an independent
//! statevector oracle **on the same lowered IR**. That catches lowering and
//! execution regressions and nothing else: it reported `Δmax = 0.000e0 PASS`
//! for as long as the inclusive-loop-bound defect existed, and it reported the
//! same `0.000e0` immediately before and immediately after every repair this
//! file guards — because both sides faithfully ran the same wrong circuit.
//! `⟨Z_q⟩` is a single-qubit marginal, and the defects lived in the
//! correlation between the counting register and the answer.
//!
//! So these assertions do not compare two engines. They state properties the
//! circuit must have to be HHL at all, and check them against closed forms
//! written out from `A`, `C` and `b` alone.
//!
//! The system is a concrete instantiation of
//! `proofs/lean4/QuantumProofs/HHL.lean`: `A = diag(1, 2)` is that file's own
//! `lam2`, while `C = 1/2` and `b = RY(0.8)|0>` are chosen here — the theorem
//! is general in both, and the Lean numeric section's `b = ![1, 1]` is too
//! symmetric to catch a swap of the two eigencomponents. `b` is **not** an
//! eigenvector of `A` — with the old `b = |1>` the `1/λ` weighting is
//! unobservable, because a single eigenvalue is populated and the direction of
//! the answer is `|1>` for every possible rotation angle.
//!
//! Each test names the Lean theorem it is the circuit-level counterpart of.

use aria_verify_core::{harness, sim, Complex64};

/// `A = diag(LAM[0], LAM[1])`, the `diagA` of `HHL.lean` at `lam = ![1, 2]`.
const LAM: [f64; 2] = [1.0, 2.0];
/// The inverse-rotation constant `C`; `HHL.controlled_inv_rotation` needs
/// `|C/λ| ≤ 1`, so `C ≤ λ_min`.
const C: f64 = 0.5;
/// `b = cos(B_ANGLE/2)|0> + sin(B_ANGLE/2)|1>`, prepared by `RY(B_ANGLE)`.
const B_ANGLE: f64 = 0.8;

/// The widths `hhl.aria` accepts. `n = 1` cannot hold `λ_max = 2`, and `n = 4`
/// would need a second work qubit for the `C^n X`; both are refused at
/// instantiation rather than run (see the width guard in `hhl.aria`).
const WIDTHS: std::ops::RangeInclusive<usize> = 2..=3;

/// `b` in the eigenbasis of `A` — the `beta` of `HHL.hhl_output`.
fn beta() -> [f64; 2] {
    [(B_ANGLE / 2.0).cos(), (B_ANGLE / 2.0).sin()]
}

/// Number of operations in Steps 1-3 of `hhl.aria` at width `n`: one `RY` to
/// prepare `b`, then `n` `H` + `n` `P` + `n` `CP` for the QPE, then
/// `n(n-1)/2` `CP` + `n` `H` + `n/2` `SWAP` for the inverse QFT.
///
/// Spelled out so the prefix cut below cannot drift: if Steps 1-3 are ever
/// restructured, this count stops matching the first ancilla touch and the
/// test says so, rather than quietly reading the register at a point that is
/// no longer the end of the QPE.
fn qpe_prefix_len(n: usize) -> usize {
    1 + 3 * n + n * (n - 1) / 2 + n + n / 2
}

/// Simulate `hhl.aria` at width `n`. With `stop_at_ancilla`, the circuit is
/// cut before the first operation that touches the ancilla `q[n+1]` — that is
/// Step 4's leading `RY`, deliberately emitted ahead of Step 4's X mask, and
/// Steps 1-3 touch no other qubit, so the prefix is exactly "prepare `b`, run
/// QPE".
fn statevector(n: usize, stop_at_ancilla: bool) -> Vec<Complex64> {
    let lowered = harness::load_lowered("hhl.aria", "Hhl", &[("n", n as i64)])
        .unwrap_or_else(|e| panic!("load hhl.aria at n={n}: {e}"));
    // `hhl.aria` binds every angle at compile time, so there are no free
    // symbols; an empty slice is the complete binding, not a zero-fill.
    assert_eq!(
        lowered.ir.symbols.len(),
        0,
        "hhl.aria gained a free parameter"
    );
    let mut ir = lowered.ir;
    if stop_at_ancilla {
        let anc = (n + 1) as u32;
        let cut = ir
            .ops
            .iter()
            .position(|op| op.qubits.iter().any(|q| q.0 == anc))
            .unwrap_or_else(|| panic!("n={n}: no operation ever touches the ancilla q[{anc}]"));
        assert_eq!(
            cut,
            qpe_prefix_len(n),
            "n={n}: the first operation on the ancilla is no longer the \
             boundary of Steps 1-3; the QPE prefix cut would read the counting \
             register at the wrong point"
        );
        ir.ops.truncate(cut);
    }
    sim::statevector(&ir, &[]).unwrap_or_else(|e| panic!("simulate hhl at n={n}: {e}"))
}

/// Counting register is `q[0..n-1]`, system is `q[n]`, ancilla `q[n+1]`,
/// work qubit `q[n+2]`.
struct Branch {
    /// `P(ancilla = 1)` over the whole state.
    p_ancilla: f64,
    /// Of that, the fraction with counting register AND work qubit back at |0>.
    restored: f64,
    /// The post-selected, un-normalized system amplitudes — `o` of
    /// `HHL.hhl_output`, read out of the clean branch.
    output: [Complex64; 2],
}

fn run(n: usize) -> Branch {
    let sv = statevector(n, false);
    let (anc, sys, work, counting) = (
        1usize << (n + 1),
        1usize << n,
        1usize << (n + 2),
        (1usize << n) - 1,
    );
    let mut p_ancilla = 0.0;
    let mut p_clean = 0.0;
    let mut output = [Complex64::new(0.0, 0.0); 2];
    for (x, amp) in sv.iter().enumerate() {
        if x & anc == 0 {
            continue;
        }
        p_ancilla += amp.norm_sqr();
        if x & counting == 0 && x & work == 0 {
            p_clean += amp.norm_sqr();
            output[usize::from(x & sys != 0)] += amp;
        }
    }
    Branch {
        p_ancilla,
        restored: p_clean / p_ancilla,
        output,
    }
}

/// The uncompute must return the counting register (and the `C^n X` work
/// qubit) to |0..0⟩.
///
/// This is the assertion the shipped circuit failed, and it failed loudly once
/// something looked: measured restored fractions were 0.500 at n=1, 0.097 at
/// n=2 and 0.011 at n=3. Step 5 undid Step 2 but never undid Step 3, so the
/// register stayed entangled with the answer and post-selecting the ancilla
/// left a mixture rather than the solution.
///
/// Circuit-level counterpart of the "uncompute the QPE register" clause of
/// step 3 in the `HHL.lean` header — the step the proof assumes rather than
/// proves, which is exactly why it needs a test.
#[test]
fn the_uncompute_disentangles_the_counting_register() {
    for n in WIDTHS {
        let b = run(n);
        assert!(
            b.restored > 1.0 - 1e-9,
            "n={n}: only {:.9} of the ancilla=|1> branch has the counting \
             register and work qubit back at |0..0>; the uncompute is incomplete",
            b.restored
        );
    }
}

/// **The capstone.** Post-selecting `ancilla = |1⟩` must leave the system
/// qubit holding `o` with `A·o = C·b` — `HHL.hhl_solves_system`, written out
/// against the same `A`, `C` and `b`.
///
/// This is the assertion the eigenvalue proxy made impossible. With the proxy
/// the measured branch was `(0.227874, 0.048551)` against a true
/// `(0.460530, 0.097355)`: both amplitudes roughly halved, yet their ratio was
/// 0.213059 against a true 0.211397, 0.8% out. Checking the un-normalized
/// amplitudes rather than the direction is what makes that visible, and it
/// pins `C` at the same time.
///
/// Only the *global* phase is divided out, so every relative phase and every
/// magnitude is still under test. A global phase is unobservable in any
/// expectation value, so asserting it would be asserting a convention.
#[test]
fn the_postselected_branch_solves_the_linear_system() {
    let beta = beta();
    for n in WIDTHS {
        let b = run(n);
        // Fix the global phase on the larger component (|β₀/λ₀| > |β₁/λ₁| here
        // by a factor 4.7, so this is never the near-zero one).
        let phase = Complex64::from_polar(1.0, -b.output[0].arg());
        for i in 0..2 {
            let a_times_o = b.output[i] * phase * LAM[i];
            let want = Complex64::new(C * beta[i], 0.0);
            assert!(
                (a_times_o - want).norm() < 1e-12,
                "n={n}, eigencomponent {i}: (A·o)_{i} = {a_times_o}, but C·b_{i} \
                 = {want}; the post-selected branch is not C·A^-1·b",
                i = i
            );
        }
    }
}

/// `P(ancilla = 1)` must equal `Σᵢ |βᵢ|²·(C/λᵢ)²` — `HHL.success_prob`, and
/// the Born-rule content of `HHL.controlled_inv_rotation`, which fixes the
/// ancilla's |1⟩-amplitude at exactly `C/λ`.
///
/// Both eigenvalues are exactly representable in `n` counting bits at every
/// accepted `n`, so widening the register must not change the answer. A
/// width-dependent result is the fingerprint of a broken inverse QFT: the
/// shipped circuit gave 0.0612 / 0.0749 / 0.0434 at n = 1 / 2 / 3, drifting
/// for no physical reason. The closed form is exact rather than golden.
#[test]
fn the_ancilla_probability_is_the_closed_form_and_width_independent() {
    let beta = beta();
    let expected: f64 = (0..2).map(|i| beta[i].powi(2) * (C / LAM[i]).powi(2)).sum();
    for n in WIDTHS {
        let b = run(n);
        assert!(
            (b.p_ancilla - expected).abs() < 1e-12,
            "n={n}: P(ancilla=1) = {:.12}, closed form sum |b_i|^2 (C/lam_i)^2 \
             = {expected:.12}",
            b.p_ancilla
        );
    }
}

/// Widths outside [`WIDTHS`] must be REFUSED, not run.
///
/// `n = 1` cannot hold `λ_max = 2`, and `n ≥ 4` needs a second work qubit for
/// the `C^n X`. In both cases every `when` arm in Step 4 misses, the
/// eigenvalue-keyed rotation silently becomes the identity, and the circuit
/// runs to completion returning `P(ancilla = 1) = 0` — a confident zero rather
/// than an error. `hhl.aria` reaches one qubit past its register instead; this
/// pins that it does.
#[test]
fn widths_outside_the_supported_range_are_refused() {
    for n in [0i64, 1, 4, 5] {
        let r = harness::load_lowered("hhl.aria", "Hhl", &[("n", n)]);
        assert!(
            r.is_err(),
            "n={n} instantiated instead of being refused; Step 4's rotation is \
             the identity at this width, so the circuit would return an \
             all-zero solution branch and call it an answer"
        );
    }
}

/// **QPE loads the eigenvalue, not a qubit index.** Right before the ancilla
/// is first touched, the state must be exactly `Σᵢ βᵢ |λᵢ⟩_count |uᵢ⟩_sys` —
/// the counting register holding the INTEGER eigenvalue. `HHL.hhl_qpe_eigenphase`
/// (i.e. `QPE.qpe_exact`) is the statement that this probability is 1.
///
/// This is the number that was 0.000000 at n = 2 and n = 3 with the H-only
/// "inverse QFT", and 0.500000 / 0.410533 with `qft.aria`'s IQFT pasted in at
/// the wrong bit-order convention. It is also what licenses Step 4 to key its
/// rotation on the register's value.
///
/// The prefix is cut at the first operation touching the ancilla `q[n+1]`,
/// which is Step 4's leading `RY`; Steps 1–3 touch no other qubit, and the
/// cut is cross-checked against [`qpe_prefix_len`].
#[test]
fn the_counting_register_holds_the_eigenvalue_after_qpe() {
    let beta = beta();
    for n in WIDTHS {
        let sv = statevector(n, true);
        let (sys, counting) = (1usize << n, (1usize << n) - 1);
        let mut p_correct = 0.0;
        for i in 0..2 {
            // |λᵢ⟩ on the counting register, |i⟩ on the system qubit.
            let x = (LAM[i] as usize) | (i * sys);
            let p = sv[x].norm_sqr();
            assert!(
                (p - beta[i].powi(2)).abs() < 1e-12,
                "n={n}: P(count={}, system={i}) = {p:.12}, expected |b_{i}|^2 = \
                 {:.12}; QPE did not load the eigenvalue",
                LAM[i],
                beta[i].powi(2)
            );
            p_correct += p;
        }
        // Nothing anywhere else: no leakage into any other (count, system) pair.
        let total: f64 = sv.iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (total - p_correct).abs() < 1e-12,
            "n={n}: P(correct eigenvalue) = {p_correct:.9} of {total:.9}; the \
             counting register is smeared over values that are not eigenvalues \
             of A (counting mask {counting:#x})"
        );
    }
}

// ---------------------------------------------------------------------------
// The three sources state one instance — checked, not maintained by hand.
// ---------------------------------------------------------------------------

/// Repository root, from this crate's manifest (`crates/apps/forward`).
fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// The right-hand side of the one line in `src` that starts with `prefix`
/// (after leading whitespace). Refuses rather than defaults: zero matches or
/// more than one is a failure naming the file, so a renamed or duplicated
/// definition cannot be silently skipped.
fn sole_definition(file: &str, src: &str, prefix: &str) -> (usize, String) {
    // Runs of whitespace are collapsed on both sides first, so a formatter pass
    // (`let  lam0 =`, `:=  ![`) cannot redden this for a non-semantic reason.
    let squash = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    let prefix = squash(prefix) + " ";
    let hits: Vec<(usize, String)> = src
        .lines()
        .enumerate()
        .filter_map(|(i, l)| {
            let l = squash(l) + " ";
            l.strip_prefix(prefix.as_str())
                .map(|rest| (i + 1, rest.trim_end().to_string()))
        })
        .collect();
    assert!(
        hits.len() == 1,
        "{file}: expected exactly one line starting with `{}`, found {} — \
         a renamed or duplicated definition would otherwise unlink this instance",
        prefix.trim_end(),
        hits.len()
    );
    hits.into_iter().next().unwrap()
}

/// A literal number, refusing anything else (an expression, a name) by file
/// and line: the point is to compare VALUES, and an expression here would be
/// compared as whatever it happened to evaluate to in someone's head.
fn number(file: &str, line: usize, text: &str) -> f64 {
    let t = text.split("--").next().unwrap_or("").trim();
    t.parse::<f64>().unwrap_or_else(|_| {
        panic!("{file}:{line}: `{t}` is not a literal number; this test compares values")
    })
}

/// `HHL.lean`'s numeric instance `lam2 := ![1, 2]`, `hhl.aria`'s
/// `lam0`/`lam1`/`cscale`/`b_angle`, and this file's `LAM`/`C`/`B_ANGLE` are
/// one instance stated three times. Before this test the correspondence was
/// kept by hand (STATUS §5 #14): changing `lam2` in the Lean file and not
/// `LAM` here would leave every check green while the circuit and the theorem
/// silently described different systems.
///
/// What is deliberately NOT linked: the Lean numeric section's `C = 1` and
/// `b = ![1, 1]`. The theorem is general in both, and this file chose
/// `C = 1/2`, `b = RY(0.8)|0>` on purpose (see the module doc) — so the link is
/// `A` across all three sources, and `C`, `b` between the circuit and this test.
#[test]
fn the_three_sources_state_one_instance() {
    let root = repo_root();
    let lean_path = "proofs/lean4/QuantumProofs/HHL.lean";
    let aria_path = "examples/aria/hhl.aria";
    let lean = std::fs::read_to_string(root.join(lean_path))
        .unwrap_or_else(|e| panic!("read {lean_path}: {e}"));
    let aria = std::fs::read_to_string(root.join(aria_path))
        .unwrap_or_else(|e| panic!("read {aria_path}: {e}"));

    // Lean: `noncomputable def lam2 : Fin 2 → ℝ := ![1, 2]`
    let (line, rhs) = sole_definition(lean_path, &lean, "noncomputable def lam2 ");
    let body = rhs
        .split_once(":= ![")
        .and_then(|(_, b)| b.split_once(']'))
        .map(|(b, _)| b)
        .unwrap_or_else(|| panic!("{lean_path}:{line}: `lam2` is not a `![…]` vector literal"));
    let lean_lam: Vec<f64> = body
        .split(',')
        .map(|e| number(lean_path, line, e))
        .collect();
    assert_eq!(
        lean_lam.len(),
        LAM.len(),
        "{lean_path}:{line}: `lam2` has {} entries, LAM has {}",
        lean_lam.len(),
        LAM.len()
    );
    assert_eq!(
        lean_lam, LAM,
        "{lean_path}:{line}: the theorem's `lam2` = {lean_lam:?} but this test's LAM = {LAM:?}"
    );

    // Aria: `let NAME = <literal>`
    let aria_value = |name: &str| {
        let (line, rhs) = sole_definition(aria_path, &aria, &format!("let {name} = "));
        number(aria_path, line, &rhs)
    };
    let pairs = [
        ("lam0", LAM[0], "LAM[0]"),
        ("lam1", LAM[1], "LAM[1]"),
        ("cscale", C, "C"),
        ("b_angle", B_ANGLE, "B_ANGLE"),
    ];
    for (name, want, here) in pairs {
        let got = aria_value(name);
        assert_eq!(
            got, want,
            "{aria_path}: `{name}` = {got} but this test's {here} = {want}; the circuit \
             and its checks describe different systems"
        );
    }
}
