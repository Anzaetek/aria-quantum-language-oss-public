#!/usr/bin/env python3
"""Emit a random LUCJ (local unitary cluster Jastrow) ansatz as QASM 2 for the
Majorana-propagation benchmark (PLAN-OPEN-20260825 §3c P4).

The circuit is ffsim's `PrepareHartreeFockJW` + `UCJOpSpinBalancedJW` with the
"linear" interaction pattern (αα nearest neighbours, αβ on-site), which is the
chemistry ansatz arXiv:2503.18939 benchmarks: Givens (length-2 Majorana
generators, never branching) plus diagonal-Coulomb phases (`cp` = number–number
= length-4, the branching gates).

Gate mapping — this repo's parser has `rbs`, not `xx_plus_yy`, and the
established convention (crates/omega-bridges/tests/qpy_rbs_vs_qiskit.rs) is
`Rbs(θ) ≡ XXPlusYY(−2θ, π/2)`. Qiskit's XXPlusYY(θ, β) is the β = π/2 gate
conjugated by a Z-rotation on the first qubit, so

    xx_plus_yy(θ, β) a, b  →  rz(δ) a; rbs(−θ/2) a, b; rz(−δ) a;   δ = β − π/2

The identity (and its sign) is CHECKED numerically below before anything is
written; a wrong sign refuses, it does not emit.

Usage: lucj_gen.py NORB NALPHA OUT.qasm [--seed S] [--reps R] [--ref OBS ...]
                   [--jastrow-scale F]
`--ref` prints exact Qiskit statevector expectations (only sane for ≤ ~14
qubits) as JSON so the CLI has a truth to hit. `--jastrow-scale` scales only
the diagonal-Coulomb (branching) angles: ffsim.random draws O(1) phases, where
MP length truncation is NOT an estimator (12q: refuses at every L ≤ 8, and
with the ceiling lifted the values wander); optimised molecular LUCJ parameters
sit at ~1e-1, where the same circuit converges to 3e-5 at L=8. Measured
2026-08-29, PLAN-OPEN-20260825 §3c P4.
"""
import argparse, json, sys
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("norb", type=int)
    ap.add_argument("nalpha", type=int)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--ref", nargs="*", default=[], help="Pauli strings like Z0Z6")
    ap.add_argument("--jastrow-scale", type=float, default=1.0,
                    help="multiply the diagonal-Coulomb (branching, length-4) angles; "
                         "ffsim.random draws O(1) phases, optimised molecular LUCJ "
                         "parameters are ~1e-1..1e-2 — this selects the regime")
    ap.add_argument("--molecule", default=None,
                    help="pyscf atom string (e.g. 'N 0 0 0; N 0 0 1.0977'): derive the "
                         "diagonal-Coulomb + Givens parameters from RHF+CCSD t2/t1 "
                         "amplitudes (ffsim from_t_amplitudes) instead of random draws; "
                         "norb/nalpha positionals must match the molecule (checked)")
    ap.add_argument("--basis", default="sto-6g", help="pyscf basis for --molecule")
    a = ap.parse_args()

    import ffsim
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate, RZGate
    from qiskit.quantum_info import Operator, Statevector, SparsePauliOp

    # --- 1. the mapping identity, checked, not assumed -------------------------
    rng = np.random.default_rng(0)
    for _ in range(5):
        th, be = rng.uniform(-3, 3), rng.uniform(-3, 3)
        d = be - np.pi / 2
        lhs = Operator(XXPlusYYGate(th, be)).data
        c = QuantumCircuit(2)
        c.rz(-d, 0)  # applied first
        c.append(XXPlusYYGate(th, np.pi / 2), [0, 1])
        c.rz(d, 0)
        rhs = Operator(c).data
        if not np.allclose(lhs, rhs, atol=1e-12):
            # try the other orientation before giving up
            c = QuantumCircuit(2)
            c.rz(d, 0)
            c.append(XXPlusYYGate(th, np.pi / 2), [0, 1])
            c.rz(-d, 0)
            if not np.allclose(lhs, Operator(c).data, atol=1e-12):
                print("REFUSE: xx_plus_yy(θ,β) ≠ rz-conjugated β=π/2 gate in either orientation", file=sys.stderr)
                return 2
            first_rz_sign = +1
        else:
            first_rz_sign = -1
    # Circuit order: emit rz(first_rz_sign*δ) a; rbs(-θ/2) a,b; rz(-first_rz_sign*δ) a

    # --- 2. the ansatz ---------------------------------------------------------
    norb, nq = a.norb, 2 * a.norb
    nelec = (a.nalpha, a.nalpha)
    pairs = ([(p, p + 1) for p in range(norb - 1)], [(p, p) for p in range(norb)])
    if a.molecule is not None:
        from pyscf import gto, scf, cc
        mol = gto.M(atom=a.molecule, basis=a.basis, symmetry=False, verbose=0)
        mf = scf.RHF(mol).run()
        if not mf.converged:
            print("REFUSE: RHF did not converge", file=sys.stderr)
            return 2
        ccsd = cc.CCSD(mf).run()
        if not ccsd.converged:
            print("REFUSE: CCSD did not converge", file=sys.stderr)
            return 2
        mol_norb, mol_nalpha = mol.nao_nr(), mol.nelectron // 2
        if mol.nelectron % 2 or (a.norb, a.nalpha) != (mol_norb, mol_nalpha):
            print(f"REFUSE: molecule gives norb={mol_norb} nalpha={mol_nalpha} "
                  f"(closed-shell required), CLI asked norb={a.norb} nalpha={a.nalpha}",
                  file=sys.stderr)
            return 2
        op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            ccsd.t2, t1=ccsd.t1, n_reps=a.reps, interaction_pairs=pairs)
        mol_info = {"molecule": a.molecule, "basis": a.basis,
                    "e_hf": float(mf.e_tot), "e_ccsd": float(ccsd.e_tot)}
    else:
        op = ffsim.random.random_ucj_op_spin_balanced(norb, n_reps=a.reps, interaction_pairs=pairs, seed=a.seed)
        mol_info = None
    if a.jastrow_scale != 1.0:
        # Only the number-number phases branch (Thm 2(1)); the Givens layers
        # are left untouched so the circuit stays a genuine LUCJ.
        op = ffsim.UCJOpSpinBalanced(
            diag_coulomb_mats=op.diag_coulomb_mats * a.jastrow_scale,
            orbital_rotations=op.orbital_rotations,
            final_orbital_rotation=op.final_orbital_rotation,
        )
    qc = QuantumCircuit(nq)
    qc.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), range(nq))
    qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(op), range(nq))

    allowed = {"x", "xx_plus_yy", "p", "cp", "rz", "rzz", "z", "cz", "barrier"}
    # Expand only what is not yet in the allowed set: a blanket `decompose()`
    # would shred `xx_plus_yy` into cx/u while another branch is still at an
    # ffsim intermediate op.
    d = qc
    for _ in range(8):
        names = {i.operation.name for i in d.data}
        if names <= allowed:
            break
        d = d.decompose(gates_to_decompose=sorted(names - allowed))
    else:
        print(f"REFUSE: could not reach the allowed gate set; got {sorted(names)}", file=sys.stderr)
        return 2

    # --- 3. emit ---------------------------------------------------------------
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{nq}];"]
    counts = {}
    for inst in d.data:
        n = inst.operation.name
        qs = [d.find_bit(q).index for q in inst.qubits]
        ps = [float(p) for p in inst.operation.params]
        counts[n] = counts.get(n, 0) + 1
        if n == "barrier":
            continue
        if n == "xx_plus_yy":
            th, be = ps
            dl = be - np.pi / 2
            a0, b0 = qs
            if abs(dl) > 1e-12:
                lines.append(f"rz({first_rz_sign * dl!r}) q[{a0}];")
            lines.append(f"rbs({-th / 2!r}) q[{a0}],q[{b0}];")
            if abs(dl) > 1e-12:
                lines.append(f"rz({-first_rz_sign * dl!r}) q[{a0}];")
        else:
            args = "(" + ",".join(repr(p) for p in ps) + ")" if ps else ""
            lines.append(f"{n}{args} " + ",".join(f"q[{q}]" for q in qs) + ";")
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")

    out = {"qubits": nq, "gates": counts, "depth": d.depth(), "seed": a.seed, "reps": a.reps}
    if mol_info is not None:
        out.update(mol_info)
    if a.ref:
        sv = Statevector(d)
        refs = {}
        for lbl in a.ref:
            paulis, idx = [], []
            i = 0
            while i < len(lbl):
                paulis.append(lbl[i]); j = i + 1
                while j < len(lbl) and lbl[j].isdigit():
                    j += 1
                idx.append(int(lbl[i + 1:j])); i = j
            psop = SparsePauliOp.from_sparse_list([("".join(paulis), idx, 1.0)], num_qubits=nq)
            refs[lbl] = float(np.real(sv.expectation_value(psop)))
        out["ref"] = refs
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
