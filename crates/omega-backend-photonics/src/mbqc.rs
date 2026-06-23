//! Measurement-Based Quantum Computing (MBQC) one-way patterns on the
//! photonic backend.
//!
//! Photonic hardware runs the one-way model natively: a graph (cluster) state
//! is prepared, then single-qubit measurements in adaptive bases
//! `{|+_φ⟩, |−_φ⟩}` drive the computation, with Pauli-byproduct corrections
//! feeding forward along the flow. This backend compiles circuits to such
//! patterns and ships them over the wire as an `OmegaPatternIR` (decision
//! C1.1/C1.2); the
//! `omega-server` quantum bridge (C1.3) dispatches them here.
//!
//! This module is the reference executor: a **frontier** cluster-state
//! simulator that materialises only the currently-unmeasured vertices, so the
//! working statevector is `2^(frontier width)` rather than `2^(total
//! vertices)`. It is the exact port of `quantum-core`'s
//! `simulate_pattern_deterministic`, so a pattern executed here and there yields
//! the same canonical output `U·|+⟩^⊗n` (the C1.4 cross-wire equality).

use num_complex::Complex64;

const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// One adaptive single-qubit measurement in basis `{|+_φ⟩, |−_φ⟩}`.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    /// Vertex being measured.
    pub qubit: usize,
    /// Base measurement angle `φ` (radians).
    pub angle: f64,
    /// Prior measurement indices whose `1`-outcome flips this angle's sign.
    pub x_corr_from: Vec<usize>,
    /// Prior measurement indices whose `1`-outcome shifts this angle by π.
    pub z_corr_from: Vec<usize>,
}

/// An MBQC measurement pattern (mirrors `quantum_core::mbqc::Pattern`).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Pattern {
    /// All graph-state vertices.
    pub vertices: Vec<usize>,
    /// Undirected graph-state edges (a CZ is applied along each).
    pub edges: Vec<(usize, usize)>,
    /// Adaptive measurement layers, in execution order.
    pub layers: Vec<Vec<Measurement>>,
    /// Unmeasured vertices holding the output state.
    pub output: Vec<usize>,
}

impl Pattern {
    /// Number of graph vertices.
    pub fn n_qubits(&self) -> usize {
        self.vertices.len()
    }

    /// Number of measurements across all layers.
    pub fn n_measurements(&self) -> usize {
        self.layers.iter().map(|l| l.len()).sum()
    }
}

/// Deterministic (`+`-branch) execution of an MBQC pattern: prepare every
/// vertex in `|+⟩`, CZ along every edge, force every measurement onto its `+`
/// outcome (with a gflow no byproduct fires, so the `+`-branch state is the
/// canonical `U·|+⟩^⊗n` up to a global phase), and read out the output vertices
/// in logical-qubit order (qubit `q` ↔ bit `n_out−1−q`, MSB-first). Normalised.
///
/// Exact port of `quantum_core::mbqc::simulate_pattern_deterministic`.
pub fn simulate_pattern_deterministic(p: &Pattern) -> Vec<Complex64> {
    let max_id = p.vertices.iter().copied().max().map_or(0, |m| m + 1);
    // adjacency: vertex → (edge index, other endpoint).
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); max_id];
    for (ei, &(u, v)) in p.edges.iter().enumerate() {
        if u < max_id && v < max_id {
            adj[u].push((ei, v));
            adj[v].push((ei, u));
        }
    }
    let mut applied = vec![false; p.edges.len()];
    let mut fr = Frontier::new(max_id);

    for layer in &p.layers {
        for m in layer {
            fr.ensure(m.qubit);
            for &(ei, w) in &adj[m.qubit] {
                if !applied[ei] {
                    fr.ensure(w);
                    fr.cz(m.qubit, w);
                    applied[ei] = true;
                }
            }
            fr.measure_plus(m.qubit, m.angle);
        }
    }
    for &v in &p.output {
        fr.ensure(v);
    }
    for (ei, &(u, v)) in p.edges.iter().enumerate() {
        if !applied[ei] && u < max_id && v < max_id {
            fr.ensure(u);
            fr.ensure(v);
            fr.cz(u, v);
            applied[ei] = true;
        }
    }

    let n_out = p.output.len();
    let mut out = vec![Complex64::new(0.0, 0.0); 1 << n_out];
    for (full_out, slot) in out.iter_mut().enumerate() {
        let mut idx = 0usize;
        for (q, &vertex) in p.output.iter().enumerate() {
            if (full_out >> (n_out - 1 - q)) & 1 == 1 {
                idx |= 1 << fr.pos[vertex].expect("output vertex must be active");
            }
        }
        *slot = fr.psi[idx];
    }
    let norm = out.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    if norm > 0.0 {
        for c in out.iter_mut() {
            *c /= Complex64::new(norm, 0.0);
        }
    }
    out
}

/// Bounded-width statevector over the currently un-measured ("active")
/// vertices. The empty product state is the scalar `1`.
struct Frontier {
    active: Vec<usize>,
    pos: Vec<Option<usize>>,
    psi: Vec<Complex64>,
}

impl Frontier {
    fn new(max_id: usize) -> Self {
        Frontier {
            active: Vec::new(),
            pos: vec![None; max_id],
            psi: vec![Complex64::new(1.0, 0.0)],
        }
    }

    /// Materialize `v` in `|+⟩` as a fresh high-order qubit (no-op if active).
    fn ensure(&mut self, v: usize) {
        if self.pos[v].is_some() {
            return;
        }
        let dim = self.psi.len();
        let mut next = vec![Complex64::new(0.0, 0.0); dim * 2];
        for s in 0..dim {
            let a = self.psi[s] * Complex64::new(FRAC_1_SQRT_2, 0.0);
            next[s] = a;
            next[s + dim] = a;
        }
        self.pos[v] = Some(self.active.len());
        self.active.push(v);
        self.psi = next;
    }

    /// Apply CZ between two active vertices.
    fn cz(&mut self, u: usize, v: usize) {
        let pu = self.pos[u].expect("cz endpoint active");
        let pv = self.pos[v].expect("cz endpoint active");
        for (s, c) in self.psi.iter_mut().enumerate() {
            if (s >> pu) & 1 == 1 && (s >> pv) & 1 == 1 {
                *c = -*c;
            }
        }
    }

    /// Project `v` onto the `+` branch of `{|+_φ⟩, |−_φ⟩}`, drop it, renormalize.
    fn measure_plus(&mut self, v: usize, angle: f64) {
        let p = self.pos[v].expect("measured vertex active");
        let dim = self.psi.len();
        let bit = 1usize << p;
        let low = bit - 1;
        let coeff = Complex64::from_polar(1.0, -angle); // `+` branch sign = +1
        let inv = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let mut next = vec![Complex64::new(0.0, 0.0); dim / 2];
        for s in 0..dim {
            if s & bit != 0 {
                continue;
            }
            let val = (self.psi[s] + coeff * self.psi[s | bit]) * inv;
            let r = (s & low) | ((s >> (p + 1)) << p);
            next[r] = val;
        }
        self.active.remove(p);
        self.pos[v] = None;
        for (i, &vid) in self.active.iter().enumerate() {
            self.pos[vid] = Some(i);
        }
        let norm = next.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm > 0.0 {
            for c in next.iter_mut() {
                *c /= Complex64::new(norm, 0.0);
            }
        }
        self.psi = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact compiled Bell pattern (quantum-core `circuit_to_pattern` of
    /// `H(0); CX(0,1)`): 5 vertices, edges `[(0,2),(1,3),(2,3),(3,4)]`, three
    /// angle-0 measurements `[0], [1], [3]`, output `[2, 4]`. MBQC prepares the
    /// `|+⟩` input, so the canonical output is `CX·(H⊗I)|++⟩ = (|00⟩+|01⟩)/√2`.
    /// These are the same numbers `quantum_core::mbqc::simulate_pattern_-
    /// deterministic` produces — the C1.4 cross-wire golden.
    fn bell_pattern() -> Pattern {
        let m0 = |q: usize| Measurement {
            qubit: q,
            angle: 0.0,
            x_corr_from: vec![],
            z_corr_from: vec![],
        };
        Pattern {
            vertices: vec![0, 1, 2, 3, 4],
            edges: vec![(0, 2), (1, 3), (2, 3), (3, 4)],
            layers: vec![vec![m0(0)], vec![m0(1)], vec![m0(3)]],
            output: vec![2, 4],
        }
    }

    #[test]
    fn bell_pattern_matches_cross_wire_golden() {
        let out = simulate_pattern_deterministic(&bell_pattern());
        assert_eq!(out.len(), 4);
        let norm: f64 = out.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-9, "‖out‖² = {norm}");
        // (|00⟩+|01⟩)/√2 up to global phase — identical to the quantum-core
        // reference (pinned in its `omega_pattern_cross_wire_golden` test).
        let s = FRAC_1_SQRT_2;
        let golden = [
            Complex64::new(s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let k = golden.iter().position(|c| c.norm() > 1e-9).unwrap();
        let phase = out[k] / golden[k];
        assert!((phase.norm() - 1.0).abs() < 1e-9);
        for (x, g) in out.iter().zip(golden.iter()) {
            assert!((x - phase * g).norm() < 1e-9, "{x} vs {g}");
        }
    }

    #[test]
    fn linear_cluster_applies_hadamard() {
        // 2-vertex line, measure vertex 0 at angle 0 → teleports H onto the
        // output: H|+⟩ = |0⟩.
        let p = Pattern {
            vertices: vec![0, 1],
            edges: vec![(0, 1)],
            layers: vec![vec![Measurement {
                qubit: 0,
                angle: 0.0,
                x_corr_from: vec![],
                z_corr_from: vec![],
            }]],
            output: vec![1],
        };
        let out = simulate_pattern_deterministic(&p);
        // H|+⟩ = |0⟩ up to global phase: amplitude on |0⟩, none on |1⟩.
        assert!(out[0].norm() > 1.0 - 1e-9);
        assert!(out[1].norm() < 1e-9);
    }
}
