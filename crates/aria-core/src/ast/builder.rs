use super::nodes::*;
use std::f64::consts::PI;

/// `2^k` as an exact `f64` — the denominator of the QFT's controlled-phase ladder.
///
/// The obvious spelling, `(1 << k) as f64`, is a trap that passed every test
/// this repo runs. The literal `1` infers `i32`, so at `k == 31` the shift
/// yields `i32::MIN` and the angle comes out NEGATIVE — one rotation in a
/// 32-qubit QFT silently sign-flipped — and at `k >= 32` it is a shift
/// overflow: a panic in debug, masked to `k & 31` in release. Measured:
///
/// ```text
/// k=30   1<<k =  1073741824   angle = +2.925836e-9
/// k=31   1<<k = -2147483648   angle = -1.462918e-9   <-- sign flipped
/// ```
///
/// Nothing caught it because a 32-qubit statevector is far past what this
/// project simulates. But the EXPORT path builds the same gate list WITHOUT
/// simulating it, so the wrong angle would be written into OPENQASM or Lean
/// and never evaluated here — which is exactly why it stayed quiet.
///
/// `powi` on a power of two is exact for every `k` this can reach.
fn two_pow(k: usize) -> f64 {
    // `k as i32` TRUNCATES, so this guard is about the cast, not the value:
    // `2f64.powi(1024)` is already +inf and `PI / +inf` is already 0.0, but
    // without the early return a `k` past i32::MAX wraps and the result is
    // finite and wrong — `two_pow(2^32 + 5)` gives 32.0, while
    // `two_pow(2^32 + 1024)` gives +inf. Non-monotonic, and silent: the same
    // defect class this helper exists to remove, one level up.
    //
    // Unreachable from `qft`/`inverse_qft`, where `j > i` holds structurally
    // and a reversed pair would panic on the `usize` subtraction first. The
    // guard is here so the helper stays correct if it is ever called from
    // somewhere that lacks those invariants.
    if k >= 1024 {
        return f64::INFINITY;
    }
    2f64.powi(k as i32)
}

/// Fluent builder for quantum circuits.
///
/// ```
/// use aria_core::ast::CircuitBuilder;
/// let circ = CircuitBuilder::new("bell", 2, 2)
///     .h(0).cx(0, 1).measure_all()
///     .build();
/// ```
pub struct CircuitBuilder {
    circuit: Circuit,
    qubits: Vec<Qubit>,
    clbits: Vec<Clbit>,
}

impl CircuitBuilder {
    pub fn new(name: &str, n_qubits: usize, n_clbits: usize) -> Self {
        let mut circuit = Circuit::new(name);
        let qubits = if n_qubits > 0 {
            circuit.qreg("q", n_qubits)
        } else {
            vec![]
        };
        let clbits = if n_clbits > 0 {
            circuit.creg("c", n_clbits)
        } else {
            vec![]
        };
        Self {
            circuit,
            qubits,
            clbits,
        }
    }

    fn q(&self, idx: usize) -> Qubit {
        self.qubits[idx].clone()
    }
    fn c(&self, idx: usize) -> Clbit {
        self.clbits[idx].clone()
    }

    // -- single-qubit gates --

    pub fn h(&mut self, q: usize) -> &mut Self {
        self.circuit.apply(h(), vec![self.q(q)]);
        self
    }
    pub fn x(&mut self, q: usize) -> &mut Self {
        self.circuit.apply(x(), vec![self.q(q)]);
        self
    }
    pub fn y(&mut self, q: usize) -> &mut Self {
        self.circuit.apply(y(), vec![self.q(q)]);
        self
    }
    pub fn z(&mut self, q: usize) -> &mut Self {
        self.circuit.apply(z(), vec![self.q(q)]);
        self
    }
    pub fn s(&mut self, q: usize) -> &mut Self {
        self.circuit.apply(s(), vec![self.q(q)]);
        self
    }
    pub fn sdg(&mut self, q: usize) -> &mut Self {
        self.circuit
            .apply(GateDef::new(GateKind::Sdg), vec![self.q(q)]);
        self
    }
    pub fn t(&mut self, q: usize) -> &mut Self {
        self.circuit.apply(t(), vec![self.q(q)]);
        self
    }
    pub fn tdg(&mut self, q: usize) -> &mut Self {
        self.circuit
            .apply(GateDef::new(GateKind::Tdg), vec![self.q(q)]);
        self
    }
    pub fn rx(&mut self, q: usize, theta: f64) -> &mut Self {
        self.circuit.apply(rx(theta), vec![self.q(q)]);
        self
    }
    pub fn ry(&mut self, q: usize, theta: f64) -> &mut Self {
        self.circuit.apply(ry(theta), vec![self.q(q)]);
        self
    }
    pub fn rz(&mut self, q: usize, theta: f64) -> &mut Self {
        self.circuit.apply(rz(theta), vec![self.q(q)]);
        self
    }

    /// Apply RX with a symbolic parameter expression.
    pub fn rx_expr(&mut self, q: usize, theta: crate::ast::expr::ParamExpr) -> &mut Self {
        self.circuit.apply(
            GateDef::with_exprs(GateKind::RX, vec![theta]),
            vec![self.q(q)],
        );
        self
    }

    /// Apply RY with a symbolic parameter expression.
    pub fn ry_expr(&mut self, q: usize, theta: crate::ast::expr::ParamExpr) -> &mut Self {
        self.circuit.apply(
            GateDef::with_exprs(GateKind::RY, vec![theta]),
            vec![self.q(q)],
        );
        self
    }

    /// Apply RZ with a symbolic parameter expression.
    pub fn rz_expr(&mut self, q: usize, theta: crate::ast::expr::ParamExpr) -> &mut Self {
        self.circuit.apply(
            GateDef::with_exprs(GateKind::RZ, vec![theta]),
            vec![self.q(q)],
        );
        self
    }
    pub fn p(&mut self, q: usize, lam: f64) -> &mut Self {
        self.circuit.apply(p(lam), vec![self.q(q)]);
        self
    }

    /// PennyLane `qml.Rot(phi, theta, omega)` = RZ(omega)·RY(theta)·RZ(phi)
    /// (ZYZ Euler decomposition). Gates are appended in application order:
    /// RZ(phi) acts first (rightmost in the matrix product).
    pub fn rot_zyz(&mut self, q: usize, phi: f64, theta: f64, omega: f64) -> &mut Self {
        self.rz(q, phi).ry(q, theta).rz(q, omega)
    }

    /// `rot_zyz` with symbolic parameter expressions.
    pub fn rot_zyz_expr(
        &mut self,
        q: usize,
        phi: crate::ast::expr::ParamExpr,
        theta: crate::ast::expr::ParamExpr,
        omega: crate::ast::expr::ParamExpr,
    ) -> &mut Self {
        self.rz_expr(q, phi).ry_expr(q, theta).rz_expr(q, omega)
    }
    pub fn u(&mut self, q: usize, theta: f64, phi: f64, lam: f64) -> &mut Self {
        self.circuit.apply(u(theta, phi, lam), vec![self.q(q)]);
        self
    }

    // -- two-qubit gates --

    pub fn cx(&mut self, ctrl: usize, tgt: usize) -> &mut Self {
        self.circuit.apply(cx(), vec![self.q(ctrl), self.q(tgt)]);
        self
    }
    pub fn cz(&mut self, q0: usize, q1: usize) -> &mut Self {
        self.circuit.apply(cz(), vec![self.q(q0), self.q(q1)]);
        self
    }
    pub fn swap(&mut self, q0: usize, q1: usize) -> &mut Self {
        self.circuit.apply(swap(), vec![self.q(q0), self.q(q1)]);
        self
    }
    pub fn cp(&mut self, ctrl: usize, tgt: usize, lam: f64) -> &mut Self {
        self.circuit.apply(cp(lam), vec![self.q(ctrl), self.q(tgt)]);
        self
    }

    // -- three-qubit gates --

    pub fn ccx(&mut self, c0: usize, c1: usize, tgt: usize) -> &mut Self {
        self.circuit
            .apply(ccx(), vec![self.q(c0), self.q(c1), self.q(tgt)]);
        self
    }

    // -- measurement / reset --

    pub fn measure(&mut self, q: usize, c: usize) -> &mut Self {
        self.circuit.measure(&self.q(q), &self.c(c));
        self
    }

    pub fn measure_all(&mut self) -> &mut Self {
        let n = self.qubits.len().min(self.clbits.len());
        for i in 0..n {
            let q = self.q(i);
            let c = self.c(i);
            self.circuit.measure(&q, &c);
        }
        self
    }

    pub fn reset(&mut self, q: usize) -> &mut Self {
        self.circuit.reset_qubit(&self.q(q));
        self
    }

    pub fn barrier_all(&mut self) -> &mut Self {
        let qs: Vec<Qubit> = self.qubits.clone();
        self.circuit.barrier(&qs);
        self
    }

    pub fn barrier(&mut self, qs: &[usize]) -> &mut Self {
        let qubits: Vec<Qubit> = qs.iter().map(|&i| self.q(i)).collect();
        self.circuit.barrier(&qubits);
        self
    }

    // -- register management --

    pub fn add_qreg(&mut self, name: &str, size: usize) -> Vec<Qubit> {
        let qubits = self.circuit.qreg(name, size);
        self.qubits.extend(qubits.clone());
        qubits
    }

    pub fn add_creg(&mut self, name: &str, size: usize) -> Vec<Clbit> {
        let clbits = self.circuit.creg(name, size);
        self.clbits.extend(clbits.clone());
        clbits
    }

    // -- QFT --

    pub fn qft(&mut self, qubits: &[usize]) -> &mut Self {
        let n = qubits.len();
        for i in 0..n {
            self.h(qubits[i]);
            for j in (i + 1)..n {
                let angle = PI / two_pow(j - i);
                self.cp(qubits[j], qubits[i], angle);
            }
        }
        for i in 0..(n / 2) {
            self.swap(qubits[i], qubits[n - 1 - i]);
        }
        self
    }

    pub fn inverse_qft(&mut self, qubits: &[usize]) -> &mut Self {
        let n = qubits.len();
        for i in 0..(n / 2) {
            self.swap(qubits[i], qubits[n - 1 - i]);
        }
        for i in (0..n).rev() {
            for j in ((i + 1)..n).rev() {
                let angle = -PI / two_pow(j - i);
                self.cp(qubits[j], qubits[i], angle);
            }
            self.h(qubits[i]);
        }
        self
    }

    // -- build --

    pub fn build(&mut self) -> Circuit {
        std::mem::replace(&mut self.circuit, Circuit::new(""))
    }

    /// Access qubits for external use.
    pub fn qubits(&self) -> &[Qubit] {
        &self.qubits
    }

    /// Access clbits for external use.
    pub fn clbits(&self) -> &[Clbit] {
        &self.clbits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `two_pow` casts `usize -> i32`, which truncates. Past i32::MAX the cast
    /// wraps and the result is finite and WRONG rather than saturating, and it
    /// is not even monotonic. Unreachable from the QFT loops, pinned so the
    /// helper stays reusable.
    #[test]
    fn two_pow_saturates_instead_of_wrapping_the_i32_cast() {
        assert_eq!(two_pow(0), 1.0);
        assert_eq!(two_pow(31), 2147483648.0);
        assert!(two_pow(1024).is_infinite(), "2^1024 overflows f64");

        // The two cases below are NOT interchangeable, and the constants are
        // not arbitrary — assert the premises so the file says why, and so
        // "tidying" them to match turns the test red instead of vacuous.
        //
        // WRAPPED band: casts back into range, so the unguarded helper returns
        // a FINITE, WRONG 2^5. This is the only line here that exercises the
        // guard at all.
        let wrapped = 2usize.pow(32) + 5;
        assert_eq!(wrapped as i32, 5, "premise: this k casts back to 5");
        assert!(
            two_pow(wrapped).is_infinite(),
            "a k past i32::MAX must saturate, not wrap to 2^5"
        );

        // DECOY: a LARGER k that happens to cast to 1024, so the unguarded
        // helper already returns +inf and this passes with or without the
        // guard. Kept because it documents the non-monotonicity — the obvious
        // "try a huge k" spot check lands here and reports a broken helper as
        // healthy, which is why the wrapped case above must exist.
        let decoy = 2usize.pow(32) + 1024;
        assert!(decoy > wrapped, "premise: the decoy is the LARGER k");
        assert_eq!(
            decoy as i32, 1024,
            "premise: the decoy casts back into range"
        );
        assert!(two_pow(decoy).is_infinite());
        // The QFT ladder's use: an unreachable depth yields a zero angle, not
        // a large one.
        assert_eq!(PI / two_pow(4096), 0.0);
    }

    /// A 33-qubit QFT is never simulated here — 2^33 amplitudes is 128 GiB — but
    /// the EXPORT path builds exactly this gate list WITHOUT simulating it, so a
    /// wrong angle reaches OPENQASM and Lean unevaluated. `j - i` runs to 32,
    /// past BOTH failure modes of the old `(1 << (j - i)) as f64`: the i32 sign
    /// flip at 31, and the shift overflow at >= 32.
    #[test]
    fn qft_angles_survive_past_the_i32_shift_cliff() {
        let n = 33;
        let qubits: Vec<usize> = (0..n).collect();
        let circ = CircuitBuilder::new("qft33", n, 0).qft(&qubits).build();

        let mut smallest = f64::INFINITY;
        for inst in &circ.instructions {
            for prm in &inst.gate.params {
                if let Some(a) = prm.try_as_f64() {
                    assert!(
                        a > 0.0,
                        "QFT emitted a non-positive controlled-phase angle {a:e}; \
                         `1 << (j - i)` infers i32 and flips sign at j - i == 31"
                    );
                    smallest = smallest.min(a);
                }
            }
        }
        // The deepest rotation is pi / 2^(n-1).
        let want = PI / two_pow(n - 1);
        assert!(
            (smallest - want).abs() <= 1e-12 * want,
            "deepest QFT angle {smallest:e}, want {want:e}"
        );
    }

    #[test]
    fn test_builder_bell_state() {
        let circ = CircuitBuilder::new("bell", 2, 2)
            .h(0)
            .cx(0, 1)
            .measure_all()
            .build();
        assert_eq!(circ.name, "bell");
        assert_eq!(circ.n_qubits(), 2);
        assert_eq!(circ.gate_count(), 2);
        assert_eq!(circ.instructions.len(), 4);
    }

    #[test]
    fn test_builder_ghz_state() {
        let n = 4;
        let mut b = CircuitBuilder::new("ghz", n, n);
        b.h(0);
        for i in 0..(n - 1) {
            b.cx(i, i + 1);
        }
        b.measure_all();
        let circ = b.build();
        assert_eq!(circ.n_qubits(), n);
        assert_eq!(circ.gate_count(), n); // 1 H + (n-1) CX
        assert_eq!(circ.depth(), n);
    }

    #[test]
    fn test_builder_qft() {
        let circ = CircuitBuilder::new("qft", 3, 0).qft(&[0, 1, 2]).build();
        assert_eq!(circ.n_qubits(), 3);
        assert!(circ.gate_count() > 0);
    }
}
