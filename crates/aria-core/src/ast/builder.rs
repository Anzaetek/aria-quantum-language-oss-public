use super::nodes::*;
use std::f64::consts::PI;

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
        self.circuit.apply(GateDef::new(GateKind::Sdg), vec![self.q(q)]);
        self
    }
    pub fn t(&mut self, q: usize) -> &mut Self {
        self.circuit.apply(t(), vec![self.q(q)]);
        self
    }
    pub fn tdg(&mut self, q: usize) -> &mut Self {
        self.circuit.apply(GateDef::new(GateKind::Tdg), vec![self.q(q)]);
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
                let angle = PI / (1 << (j - i)) as f64;
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
                let angle = -PI / (1 << (j - i)) as f64;
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
