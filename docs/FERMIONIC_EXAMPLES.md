<!-- SPDX-License-Identifier: Apache-2.0 -->
# Fermionic examples — chemistry, a lattice, and the Kitaev chain

The operator text is OpenFermion's ladder spelling (`0.5 [1^ 3] + …`),
fed to `omega-run --expectation-fermionic`. The OpenQASM 2 circuits stay
as the cross-oracle. Each FermionicQASM program below is checked against
its QASM2 twin and against the reference named for that example. A `#`
line in an operator file is a header, not a term.

**These are not `.aria` programs.** The Aria grammar has no fermionic
observable and no wire for a mode, and the application harness is
qubit-WASM-shaped. The QASM files stay the surface that an outside
program can rerun.

| example | quantity | reference |
|---|---|---|
| H₂ / STO-3G, R = 0.7414 Å | Hartree–Fock and FCI energies | OpenFermion's shipped integrals, checked against a literature print |
| 2-site Fermi–Hubbard, t = 1, U = 4 | two-electron ground energy | closed form 2 − 2√2, and OpenFermion `fermi_hubbard` in that sector |
| Kitaev chain, t = Δ = 1, μ = 0 | ground energy −(N−1) at N = 2, 8, 64 | closed form. Statevector at N = 2 and 8. At N = 64, our own certificate only |
| LiH / STO-3G, R = 1.6 Å | Hartree–Fock energy, and the gap to FCI | PySCF 2.14.0 `scf.RHF` and `fci.FCI` |

---

## 1. H₂ / STO-3G at R = 0.7414 Å

The integrals are the ones in the MolecularData file OpenFermion ships,
`H2_sto-3g_singlet_0.7414`. Nothing here calls PySCF. Regenerating that
file and diffing it is `tools/fermionic_examples/gen_h2_op.py --check`,
a registered skip in `ci.sh` when OpenFermion is not installed.

> J. T. Seeley, M. J. Richard and P. J. Love, *"The Bravyi-Kitaev
> transformation for quantum computation of electronic structure"*,
> **J. Chem. Phys. 137, 224109 (2012)**, doi:10.1063/1.4768229.
> The STO-3G integrals at this bond length.
>
> P. J. J. O'Malley et al., *"Scalable Quantum Simulation of Molecular
> Energies"*, **Phys. Rev. X 6, 031007 (2016)**,
> doi:10.1103/PhysRevX.6.031007.
> One parameter in the double excitation is exact for H₂ in a minimal basis.

Spin-orbital order, as in that file: 0 = 1σ_g ↑, 1 = 1σ_g ↓, 2 = 1σ_u ↑,
3 = 1σ_u ↓. The identity term `[]` is the nuclear repulsion,
0.713753990544915 Ha.

The file's own energies, which the CLI test matches to 1e-6:

| | Ha |
|---|---|
| Hartree–Fock | **−1.116684386906734** |
| FCI | **−1.137270174625328** |

The six-decimal print of the same quantities is **−1.116685** and
**−1.137270**. The file differs from that print by 6.1×10⁻⁷ and
1.7×10⁻⁷. The test also requires the measured value to sit within 1e-4
of the print. It does not replace the print with the file, or the file
with whatever the circuit returned.

```console
$ omega-run examples/fermionic/h2_hf.qasm \
    --expectation-fermionic "$(cat examples/fermionic/h2_op.txt)"
$ omega-run examples/fermionic/h2_ground.qasm \
    --expectation-fermionic "$(cat examples/fermionic/h2_op.txt)"
```

`h2_hf.qasm` is `x` on modes 0 and 1. `h2_ground.qasm` appends

```text
exp(−i θ X₀ X₁ X₂ Y₃),    θ = −0.11306813351297684
```

compiled as a basis change (`h` takes X to Z, `sdg` then `h` takes Y to
Z), a CX ladder, and `rz(2θ)`. On `|1100⟩` that unitary is
`cos θ |1100⟩ + sin θ |0011⟩`, and for this θ the energy is the FCI
value above. The same three commands with `--backend mps` and
`--backend majoranaprop` return the same numbers. On majoranaprop the
certificate says `seed_basis: "ladder"` and `dropped_mass: 0`: four
qubits, no length cut, so the run is untruncated. The circuit is not a
matchgate; the bound is zero because nothing was dropped, not because
every circuit of this shape is exact at larger width.

ffsim contracts the same operator on those two occupation states
(`tools/fermionic_examples/ffsim_h2.py`). That leg is a registered skip
without the ffsim venv. It agrees with the file to better than 1e-12.

**What this check cannot see.** Swapping `x q[0]` with `x q[1]` leaves
the Hartree–Fock state unchanged, so the energy does not notice.
Adding `2π` to the `rz` angle multiplies the state by a global phase
and leaves both energies where they are. A shift of the angle that is
large enough to move the energy by more than 1e-6 does redden the FCI
assertion; a smaller shift can hide inside the tolerance, because the
energy is stationary at θ.

### FermionicQASM, and the UCCSD ansatz

`h2_hf.fqasm` is `load m[0], m[1]`. That is the same occupation as
`h2_hf.qasm`. `h2_ground.fqasm` prepares the same state as
`h2_ground.qasm`,

```text
cos θ |1100⟩ + sin θ |0011⟩,    θ = −0.11306813351297684
```

with `num`, `numnum` and `givens` only. This file is a UCCSD ansatz at
a fixed parameter. The singles are absent because the orbitals are the
Hartree–Fock orbitals: Brillouin's theorem sets those amplitudes to
zero. The one double-excitation angle is the θ above, taken from
OpenFermion's MolecularData file. It is not the output of an optimiser.
There is no classical outer loop and no search. For this molecule in
this basis that one parameter is the FCI state, which is why the energy
is the FCI number. It is not a VQE run, and it is not an energy
minimisation.

A spin-symmetric orbital rotation applied to `|1100⟩` is still one
Slater determinant, so it still contains the open-shell singlet
`(|1001⟩ − |0110⟩)/√2`. `numnum(π)` on modes 0 and 1 multiplies `|1100⟩`
by −1 and leaves the other three Sz = 0 determinants alone. Between two
such rotations the open-shell amplitudes can be made to cancel. The
angles that do it, for the Givens sign this lowering actually emits
(a particle on the first mode hops to the second with amplitude −sin),
are

```text
φ = (1/2) arccos(−(cos θ + sin θ))
ψ = (1/2) arccos(−1 / (cos θ − sin θ))
```

The first rotation uses Givens angle φ and the second uses −ψ. The
same-spin pairs are modes (0, 2) and (1, 3). Those are not adjacent, and
a non-adjacent `givens` is refused. Each pair is reached by conjugating
an adjacent `givens` with a fermionic swap. The swap of neighbouring
modes p, q is `givens(π/2)` on that pair followed by `num(π)` on q:
`|10⟩` and `|01⟩` are exchanged and `|11⟩` picks up a minus, which is
the orbital permutation. π is written as a decimal because the grammar
has no `pi`.

```console
$ omega-run examples/fermionic/h2_hf.fqasm \
    --expectation-fermionic "$(cat examples/fermionic/h2_op.txt)"
$ omega-run examples/fermionic/h2_ground.fqasm \
    --expectation-fermionic "$(cat examples/fermionic/h2_op.txt)"
```

The test requires the FermionicQASM energy and the QASM2 energy to agree,
and requires that agreed value to match the OpenFermion file and the
literature print. Agreement of the two programs with each other is not
the reference.

**What this check cannot see.** Adding `2π` to any `givens` angle leaves
that gate unchanged, so neither the twin agreement nor the reference
moves. The same is true of `num(2π)` on one mode. A shift of φ small
enough that the energy stays inside 1e-6 also hides, because the energy
is stationary at θ. `tunnel` is not in this circuit: the double
excitation that reaches FCI is real, and `tunnel` is the imaginary hop.

---

## 2. Two-site Fermi–Hubbard, t = 1, U = 4

Mode order is site-major and spin-minor: 0 = site 0 ↑, 1 = site 0 ↓,
2 = site 1 ↑, 3 = site 1 ↓. The operator is OpenFermion's
`fermi_hubbard(2, 1, tunneling=1, coulomb=4, periodic=False)`, both
orderings of each hop, and no constant term. Regenerating it is the
same `tools/fermionic_examples/gen_h2_op.py --check` leg as H₂, a
registered skip when OpenFermion is not installed. Periodic boundaries
are off on purpose: the default would count the one bond twice.

The number below is the ground energy **with two electrons**. Chemical
potential zero makes the lowest eigenvalue of the whole Fock space
−t = **−1**, one electron in the bonding orbital. That is not this
example. The two-electron block of the same matrix has lowest
eigenvalue 2 − 2√2, and the script exits if a fresh diagonalization
does not.

In the singlet subspace

```text
|S⟩ = (|1001⟩ − |0110⟩) / √2
|D⟩ = (|1100⟩ + |0011⟩) / √2
```

the Hamiltonian at t = 1 is the 2×2 with diagonal (0, U) and
off-diagonal −2. Its lower eigenvalue is

```text
(U − √(U² + 16 t²)) / 2 = 2 − 2√2 ≈ −0.8284271247
```

and the eigenvector is cos(π/8) |S⟩ + sin(π/8) |D⟩, because
tan(π/8) = √2 − 1 equals −E / (2t). Left-right symmetry is why one
angle is enough: the two ionic occupations have the same amplitude, and
the two Néel occupations differ only by a sign.

```console
$ omega-run examples/fermionic/hubbard2.qasm --qasm-dialect lenient \
    --expectation-fermionic "$(cat examples/fermionic/hubbard2_op.txt)"
```

`rbs` is not in qelib1, so the dialect flag is required. The same
command with `--backend mps` and `--backend majoranaprop` returns the
same energy. On majoranaprop the certificate says `seed_basis: "ladder"`
and `dropped_mass: 0`.

The circuit has three pieces.

The Hartree–Fock product state is the Néel determinant, `x` on modes 0
and 3: one up electron on site 0 and one down electron on site 1.

The double excitation `exp(−i (π/4) X₀ X₁ X₂ X₃)`, compiled like the H₂
ladder (`h` takes X to Z, a CX chain, `rz(π/2)`, uncomputed), takes
that determinant to `(|1001⟩ − i|0110⟩) / √2`. `s` on qubit 0 multiplies
`|1001⟩` by i and leaves `|0110⟩`, which is the covalent singlet up to a
global phase. The angle π/4 is the equal-weight singlet, not a fit to
U. At exactly π/4 the `rz` is a Clifford.

`rbs(π/8)` on qubits 1 and 3, the two spin-down modes, then acts as

```text
|1001⟩ → cos(θ)|1001⟩ + sin(θ)|1100⟩
|0110⟩ → −sin(θ)|0011⟩ + cos(θ)|0110⟩
```

so on |S⟩ it is cos(θ)|S⟩ + sin(θ)|D⟩. At θ = π/8 that is the
eigenvector above. These two modes are not adjacent. `rbs` here is the
qubit beam splitter on that pair. It is not the fermionic Givens
rotation between modes 1 and 3: that rotation carries a Z on mode 2,
and the constructor that would emit it is refused. The identity used
here is the action on |S⟩ and |D⟩ just written, which the energy check
tests.

Tolerance against 2 − 2√2 is 1e-9, on statevector, mps, and
majoranaprop. OpenFermion's two-electron eigenvalue agrees with the
closed form past 1e-12; that comparison is in the generator, not in the
Rust test, because the test has to run with no venv.

**What this check cannot see.** Adding `2π` to the `rbs` angle leaves
the gate unchanged, so the energy does not move. The eigenvalue
2 − 2√2 occurs once in the 16-dimensional spectrum, so a mutation that
changes the state by more than a global phase does move the energy.
The check still cannot see a global phase.

### FermionicQASM

`hubbard2.fqasm` prepares the same ground state from the Hartree–Fock
determinant `|1100⟩` (both electrons on site 0), rather than from the
Néel determinant the QASM2 file starts in. The circuit is the same
family as the H₂ ground state: two spin-symmetric orbital rotations
with `numnum(π)` on modes 0 and 1 between them. The angles come from
π/8, the angle already fixed by `tan(π/8) = √2 − 1`:

```text
φ = (1/2) arccos(−√2 sin(π/8))
ψ = (1/2) arccos(√(√2 − 1))
```

The adjacent decomposition implements spatial rotation by −α when the
Givens angle is α, so the file enters the first rotation at −φ and the
second at +ψ. Same-spin hops are modes (0, 2) and (1, 3), reached by
the fermionic-swap conjugation described for H₂.

```console
$ omega-run examples/fermionic/hubbard2.fqasm \
    --expectation-fermionic "$(cat examples/fermionic/hubbard2_op.txt)"
```

The test requires this energy to agree with `hubbard2.qasm` and requires
that agreed value to equal 2 − 2√2. The QASM2 `rbs` on qubits 1 and 3
is a different gate from the fermionic rotation between those modes;
the two programs are checked because they share the energy, and the
closed form is what says the energy is right.

**What this check cannot see.** Adding `2π` to a `givens` angle leaves
the gate unchanged. `num(2π)` on one mode is a global phase on every
occupied component that carries that mode, and on this state it is a
global phase. The eigenvalue occurs once, so a change of the state by
more than a global phase does move the energy.

---

## 3. Kitaev chain at the sweet spot

Spinless modes `0 .. N-1`, parameters t = Δ = 1 and μ = 0:

```text
H = Σ_j [ −(a†_j a_{j+1} + h.c.) + (a_j a_{j+1} + h.c.) ]
```

Both orders of each product are written in the operator file, because a
one-sided ladder is not Hermitian. There is no chemical-potential term.

Jordan–Wigner sends this operator to −Σ_j X_j X_{j+1}. For N = 2 the
CLI prints that image as the single weight-2 Pauli `−1*X0X1`. For N = 8
it prints seven such bonds. The spectrum of i times a Majorana bilinear
on a disjoint bond is ±1, and the ground state takes the lower sign on
every bond, so

```text
E₀ = −(N − 1)
```

N = 2 gives **−1**, N = 8 gives **−7**, N = 64 gives **−63**.

`|+⟩` on every mode is a +1 eigenstate of every X_j X_{j+1}. So is
`|−⟩` on every mode. The second is the first followed by Z on every
mode, and that product of Z is the global fermion parity. The two
circuits are `kitaev_n{2,8,64}.qasm` and `kitaev_n{2,8,64}_partner.qasm`.
Tolerance against −(N−1) is 1e-9.

```console
$ omega-run examples/fermionic/kitaev_n2.qasm \
    --expectation-fermionic "$(cat examples/fermionic/kitaev_op_n2.txt)"
$ omega-run examples/fermionic/kitaev_n8.qasm \
    --backend statevector \
    --expectation-fermionic "$(cat examples/fermionic/kitaev_op_n8.txt)"
$ omega-run examples/fermionic/kitaev_n8.qasm \
    --backend majoranaprop \
    --expectation-fermionic "$(cat examples/fermionic/kitaev_op_n8.txt)"
```

N = 2 and N = 8 are run on the statevector and on majoranaprop. Both
return the analytic value, and the majoranaprop certificate says
`seed_basis: "ladder"`, `dropped_mass: 0`, `exact: true`. The Hadamard
is a π/2 rotation. A π/2 rotation does not split a Majorana monomial,
which is why nothing is dropped.

**N = 64 has no independent software oracle in this tree.** A dense
statevector of 64 qubits is not a check anyone can run here. The
reference at N = 64 is the analytic value −63 together with
majoranaprop's own certificate (`dropped_mass: 0` on this Clifford
circuit). That certificate is our own. It is not a second program.

The same construction with Z on every mode is the parity partner, and
it reports the same energy at N = 2, 8, and 64. At N = 64 that partner
is under the same limitation: analytic value, our certificate, no dense
oracle.

An S on one qubit of the N = 2 Bell state `(|00⟩+|11⟩)/√2` — itself a
ground state, prepared by H then CX — multiplies `|11⟩` by i. That
state is not an eigenstate of X⊗X, and the energy leaves −1. The
shipped prep is the product of Hadamards, not H, CX, S.

One T on mode 0 of the N = 8 circuit (`kitaev_n8_t.qasm`) takes the
state out of the ground space. The touched bond contributes −1/√2
instead of −1 and the other six bonds stay −1, so the energy is
−6 − 1/√2. The statevector returns that number. The same circuit on
majoranaprop with `--max-length 4` returns −1/√2 and `dropped_mass` 6.
The certified interval is that value ± 6, and the statevector number
sits on its lower endpoint. The interval is a bound. It is not a claim
that the truncated run found the energy by itself.

```console
$ omega-run examples/fermionic/kitaev_n8_t.qasm --backend statevector \
    --expectation-fermionic "$(cat examples/fermionic/kitaev_op_n8.txt)"
$ omega-run examples/fermionic/kitaev_n8_t.qasm \
    --backend majoranaprop --max-length 4 \
    --expectation-fermionic "$(cat examples/fermionic/kitaev_op_n8.txt)"
```

`crates/omega-backend-majoranaprop/examples/kitaev_sweet_spot.rs`
sweeps N from 2 through 128, prints the energy, the analytic value, the
absolute difference, the monomial count, and the wall time, and asserts
the difference is below 1e-9 with `dropped_mass` 0. The monomial count
equals the number of bonds.

**What this check cannot see.** Z on every mode is the parity partner
and has the same energy, so the −(N−1) assertion does not distinguish
the two ground states. Swapping two `h` lines does nothing either: the
Hadamards are on different qubits and they commute.

### FermionicQASM, N = 2 only

`kitaev_n2.fqasm` prepares `(|10⟩ + |01⟩)/√2`: `load m[0]`, then
`givens(−π/4)`. That state is an eigenstate of −X₀X₁ with eigenvalue
−1, so the energy agrees with `kitaev_n2.qasm`. The states do not
agree. `|+⟩⊗|+⟩` has components on every particle number. After `load`,
`num`, `numnum`, `givens` and `tunnel` all conserve particle number, so
that product state is not a program in this grammar.

For N > 2 the obstruction is stronger. The ground space of
−Σⱼ XⱼXⱼ₊₁ is exactly span{`|+⟩`⊗ᴺ, `|−⟩`⊗ᴺ}. Every vector in that
span has support on more than one particle number. The variational
theorem then says no fixed-particle-number state has energy −(N−1).
There is no FermionicQASM 1.0 file for N = 8 or N = 64 whose energy
matches the QASM2 twin. Those QASM2 files stay the programs that hit
the number.

```console
$ omega-run examples/fermionic/kitaev_n2.fqasm \
    --expectation-fermionic "$(cat examples/fermionic/kitaev_op_n2.txt)"
```

The test requires the energy to agree with `kitaev_n2.qasm` and to equal
−1, and it requires the statevector amplitudes `1/√2` at indices 1 and
2. The energy alone would also accept `|+⟩⊗|+⟩`. The amplitudes are
what say which ground state this file is.

**What this check cannot see.** Adding `2π` to the `givens` angle leaves
the gate unchanged, so the amplitudes and the energy stay where they
are. The energy assertion on its own also cannot tell this state from
the QASM2 product of Hadamards.

---

### Truncated stabilizer rank: T gates beyond dense reach

These rows put T gates on the sweet-spot ground state `|+⟩`⊗ᴺ and read
the energy on `--backend stabrank`, with the state cut below its exact
rank. A T gate is a local kick of the chemical potential on one mode.

**The reference.** T on a set S of sites keeps a product state. Each
site in S has ⟨X⟩ = 1/√2 and every other site has ⟨X⟩ = 1. The energy
−Σⱼ XⱼXⱼ₊₁ is then −Σⱼ cⱼcⱼ₊₁, with cⱼ = 1/√2 on S and 1 elsewhere.
That closed form holds at any N, so it is the reference where the dense
statevector cannot run. majoranaprop is a different engine in this tree,
and it returns the same number with `dropped_mass` 0 and `exact` true.
The tests pin it against the closed form too.

**The two knobs.** They do different things, and each row names the
one it used.

- **`--max-chi χ` truncates.** It keeps the χ largest branches. It
  returns a value with `state_dropped_mass` m and the bound
  `expectation_error_bound` = R·m·(2+m), where R is the sum of the
  absolute coefficients. When that bound reaches R + |value| it excludes
  nothing, and the run is refused instead.
- **`--max-branches` refuses.** It returns no value and no bound. Its
  default ceiling is 65536.

**The acceptance check.** Every truncated row asserts three things
together. First, mass was dropped, m > 0. Second, the bound is
informative, bound < R + |value|. Third, the error is real and inside
the bound, 0 < |value − reference| ≤ bound. Every row also has
|value| > 0.1. A global phase cannot move any of these numbers, since
an expectation does not see it.

| Row | Circuit | Observable, R | Knob | Reference | Value | \|Δ\| | m | Bound | Bound / \|Δ\| | Wall | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `kitaev_n8_t2`, T on 1, 5 | energy, 7 | `--max-chi 3` | dense statevector, which equals the closed form −3 − 2√2 | −5.84987 | 0.0214 | 0.1464 | 2.200 | 103 | 0.01 s | 4.3 MB |
| 2 | `kitaev_n64_t2`, T on 10, 40 | energy, 63 | `--max-chi 3` | closed form −59 − 2√2, and majoranaprop exact | −60.64886 | 1.180 | 0.1464 | 19.80 | 17 | 0.01 s | 4.5 MB |
| 3 | `kitaev_n256_t2`, T on 10, 200 | energy, 255 | `--max-chi 3` | closed form −251 − 2√2, and majoranaprop exact | −248.53111 | 5.297 | 0.1464 | 80.16 | 15 | 0.49 s | 5.8 MB |
| 4 | `kitaev_n64_t2` | bond (10, 11), 1 | `--max-chi 3` | closed form −1/√2, and majoranaprop exact | −0.72855 | 0.0214 | 0.1464 | 0.3143 | 15 | 0.01 s | 4.5 MB |
| 5 | `kitaev_n64_t2` | energy, 63 | `--max-chi 1` | refused as vacuous, m = 0.736, bound 126.9 | none | | | | | 0.01 s | 4.5 MB |
| 6 | `kitaev_n64_tall`, T on all 64 | energy, 63 | `--max-branches`, default 65536 | refused at 131072 branches. majoranaprop gives −31.5, the closed form −63/2 | none | | | | | 0.59 s | 1.33 GB |
| 7 | `kitaev_n64_t8`, T on 3, 11, …, 59 | energy, 63 | `--max-chi 200` | closed form −47 − 8√2, and majoranaprop exact | −58.27129 | 0.0424 | 0.2074 | 28.84 | 680 | 43.7 s | 12.8 MB |

Rows 1 to 6 are asserted in `crates/omega-cli/tests/modality_examples.rs`.
Row 7 is documented only, because its run takes 44 s. The untruncated
stabrank run of row 7 takes 83 s and 15 MB, and it agrees with the
closed form. The untruncated runs of rows 2 and 3 take 0.02 s and
0.85 s, and they are also asserted against the closed form. Row 6's
1.33 GB is the branch list built before the ceiling trips. It is the
largest memory cost in this file's tests. Times are wall-clock for a
single run of the release binary.

**The bound grows with R, not with the error.** Rows 2 and 4 drop the
same mass from the same state. The energy bound is 63 times the bond
bound, because R is 63 times larger. Read the bound as a worst case
over every observable with that range. It is not an estimate of the
error on this one.

**Why the full quench gets no truncated number.** k T gates carry an
L1 coefficient mass of about 1.307ᵏ. The bound is informative only when
m(2+m) < 2, which means m < √3 − 1 ≈ 0.73 whatever N is. At χ = 1,
row 5 drops 0.736 and is refused. Row 6 has 64 T gates. Untruncated, it
needs 2⁶⁴ branches and `--max-branches` refuses it. A `--max-chi 4096`
run of the same circuit did not finish within 900 s at 182 MB peak. Its
bound could not be informative unless it kept nearly all of the mass,
so that run costs more than it is worth and is not shipped. majoranaprop
returns the exact number for this circuit in 0.02 s, and that is the
number to use.

```console
$ omega-run examples/fermionic/kitaev_n64_t2.qasm --backend stabrank \
    --max-chi 3 --format json \
    --expectation-fermionic "$(cat examples/fermionic/kitaev_op_n64.txt)"
$ omega-run examples/fermionic/kitaev_n64_tall.qasm --backend majoranaprop \
    --expectation-fermionic "$(cat examples/fermionic/kitaev_op_n64.txt)"
```

**The assertions have teeth.** Each mutation below edits a fixture,
runs one test, restores the file and compares it byte for byte. The
line is where the test panics.

| Test | Fixture mutation | Assertion that fails | Line |
|---|---|---|---|
| `e3_truncated_n8_two_t_bound_holds_against_dense` | delete `t q[5]` | `exact` is true: one T has rank 2, so χ = 3 drops nothing | 566 |
| `e3_truncated_beyond_dense_reach_bound_holds_against_the_closed_form` | `t q[40]` → `s q[40]` | `exact` is true, for the same reason | 566 |
| `e3_truncated_single_bond_is_the_tight_row` | `t q[10]` → `t q[12]` | majoranaprop −1 against the closed form −1/√2 | 618 |
| `e3_truncated_to_one_branch_is_refused_as_vacuous` | `t q[40]` → `h q[40]` | the run must be refused, and it was not | 588 |
| `e3_full_quench_is_refused_by_stabrank_and_exact_on_majoranaprop` | keep only `t q[0]` to `t q[2]` | the run must be refused, and it was not | 588 |

The bond mutation passes the three-part check. With T on 12 and 40, the
truncated run returns −0.97855, which is 0.271 from −1/√2. The bound is
0.314, so a wrong fixture whose error fits inside the bound is
accepted. The majoranaprop pin is what catches it. The three-part check
tests the bound. It does not test the reference.

**What these checks cannot see.** T → T† on every site passes all five
tests. A product state with T† still has ⟨X⟩ = 1/√2 on each such site,
and the branch weights are the same, so the energy, m and the bound are
unchanged. Swapping the order of gates on different qubits is also
invisible.

## 4. LiH / STO-3G at R = 1.6 Å

Li at the origin, H on the z axis at 1.6 Å, basis STO-3G. Six spatial
orbitals, twelve spin-orbitals, four electrons. The Hamiltonian is the
FCIDUMP `examples/fermionic/lih_sto3g.fcidump`. Those bytes were written
by PySCF 2.14.0 `pyscf.tools.fcidump.from_integrals`, from the
spin-orbital chemist tensors of an RHF calculation. The script
`tools/fermionic_examples/gen_lih.py` is that call. It does not format
the integral file itself. `--check` regenerates both files and diffs;
without the ffsim venv that leg is a registered skip.

Spin-orbital order is blocked, the same order as `mode m[6] spin`:
modes 0..5 are spin-up in PySCF's MO energy order, modes 6..11 are
spin-down. The RHF determinant occupies the two lowest spatial
orbitals, both spins.

```console
$ omega-run examples/fermionic/lih_hf.fqasm \
    --hamiltonian examples/fermionic/lih_sto3g.fcidump
```

`lih_hf.fqasm` is `load` of modes 0, 1, 6 and 7. That is the
Hartree–Fock state. PySCF's numbers, which the test reads from
`lih_reference.txt`:

| | Ha |
|---|---|
| RHF, `scf.RHF` | **−7.861864769808647** |
| FCI, `fci.FCI` | **−7.882324378883495** |

The measured expectation reproduces the RHF value. The difference from
FCI is 0.020459609074848 Ha, the correlation energy of this FCI
calculation (`E_HF − E_FCI` in the reference file). The circuit does
not reproduce FCI. The test checks the RHF match and checks that the
gap equals that correlation energy. It does not move the FCI number
onto whatever the circuit returned, and it does not widen a tolerance
until the gap fits inside it.

An orbital rotation of this determinant is another single determinant.
On RHF orbitals it cannot undercut the RHF energy, so it cannot reach
FCI either. A fixed-parameter UCCSD circuit for twelve spin-orbitals
is a different ansatz; this file is the determinant the FCIDUMP was
built from.

**What this check cannot see.** The diagonal one-body integral of an
empty orbital, for example the line `h` on FCIDUMP index 4 (mode 3,
the third virtual spin-up orbital), does not appear in the
Hartree–Fock contraction. Changing it leaves both the RHF assertion
and the gap assertion where they are. `num(2π)` on an occupied mode
is a global phase and leaves them as well. Changing the core energy,
the `0 0 0 0` line, does move the measured value.
