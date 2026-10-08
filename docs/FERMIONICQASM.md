<!-- SPDX-License-Identifier: Apache-2.0 -->
# FermionicQASM 1.0

FermionicQASM is an in-house format. It carries no authority of a standard.
The name had no prior art when it was chosen, and the grammar is not an
adoption of anyone else's dialect. Signs and spin order are pinned to
ffsim 0.0.84 and to a source reading of qiskit-cold-atom; they are not
implied by the name.

The grammar is [`crates/omega-parser/src/fermionicqasm.pest`](../crates/omega-parser/src/fermionicqasm.pest).
`parse_fermionicqasm` reads it, `lower_fermionicqasm` lowers it onto the
pinned Jordan–Wigner gates, and `omega-run` takes the file by that header
(`FERMIONICQASM 1.0;`, conventionally `.fqasm`). The writer is
`to_fermionicqasm` in `crates/aria-core`, from Aria's AST, not from the
lowered circuit.

Every fenced block in this file is extracted by
[`crates/omega-parser/tests/fermionicqasm_doc_examples.rs`](../crates/omega-parser/tests/fermionicqasm_doc_examples.rs).
A `fermionicqasm` block must parse and lower. A `fermionicqasm-refuse` block
must be refused, and the error must contain the text after the tag. A
`conventions-quote` block must appear, line for line, in
`conventions.json`. A block with any other tag fails that test.

---

## 1. There is no fermionic QASM to adopt

Searched before the syntax was invented, and recorded in
[`PLAN-OPEN-20260825.md`](../PLAN-OPEN-20260825.md) §3e.2a so it does not
have to be searched again:

* **No fermionic QASM dialect exists.** Not in OpenQASM 2, not in OpenQASM 3,
  not in cQASM, not in Jaqal. The one candidate the search surfaced,
  `qiskit_cold_atom.fermions.FermiHubbard.qasm()`, is a generic inherited
  method (`FermiHubbard` → `FermionicGate` → `Gate` → `Instruction`) with no
  `qasm` override. It was deprecated in qiskit-terra 0.25.0 ("no replacement
  will be provided") and removed in Qiskit 1.0. That is not a fermionic
  dialect.
* **The name `FermionicQASM` is unused.** No prior art was found.
* **OPTICQASM is in-house as well.** It is this repository's photonic
  dialect, and it carries the same disclaimer
  ([`docs/PHOTONICS_EXAMPLES.md`](PHOTONICS_EXAMPLES.md),
  [`opticqasm.pest`](../crates/omega-parser/src/opticqasm.pest)). Writing a
  fermionic sibling follows that precedent. Neither file is a standard.

Established conventions were adopted. The surface syntax was invented.
§2 separates the two.

---

## 2. What was adopted, and what was invented

| | where it comes from | where it lives |
|---|---|---|
| A wire is one fermionic mode, empty or occupied | qiskit-cold-atom, commit `ad8893f` (source read; §6) | `mode` |
| Spin-½ in `N` spatial modes is `2N` wires, first block spin-up, second spin-down | the same source, sequential registers, not interleaved | `mode m[N] spin` |
| `num`, `numnum`, `givens`, `tunnel` | ffsim's number-conserving primitives, matrices from ffsim 0.0.84 | the gate spellings below |
| OpenFermion ladder spelling (`0.5 [1^ 3] + …`) | OpenFermion, already the observable parser | `omega-run --expectation-fermionic`. Not this file |
| A Hamiltonian travels as FCIDUMP, out of band | PySCF and ffsim both read that format | `omega-run --hamiltonian FILE`, exclusive with `--expectation-fermionic` |

Invented here: the name, the header, the statement spellings, and `load` as
a statement in this grammar. The word `load` is qiskit-cold-atom's
instruction name. The statement form is this file's.

`orbrot` was considered and left out. §8.

There is no `jw`, `bk`, or parity keyword. Jordan–Wigner is the lowering
that exists today. It is not file content.

---

## 3. Lexical structure

* **Comments** start with `//` and run to end of line.
* **Whitespace** separates tokens. Newlines are not significant.
* **Identifiers** are `[A-Za-z_][A-Za-z0-9_]*`.
* **Integers** (register sizes, mode indices, classical-bit indices) are
  `[0-9]+`. A name in that position does not parse.
* **Numbers** (gate parameters) are an optional `-`, digits, and an optional
  fraction: `-0.7`, `0.5`, `1`. There is no scientific notation and no `pi`.
* **Symbols** are `$` followed by ASCII alphanumeric or `_`: `$theta`, `$1`.
  A symbol is a parameter name. It is not an expression.
* An expression (`pi`, `theta/2`, `1+2`) is not a parameter. §8.

```fermionicqasm-refuse m[n]
FERMIONICQASM 1.0;
mode m[n];
```

```fermionicqasm-refuse 1e-3
FERMIONICQASM 1.0;
mode m[1];
num(1e-3) m[0];
```

---

## 4. The header, and the version lock

A program opens with `FERMIONICQASM 1.0;`. Comments may precede it. The
grammar admits any `D` or `D.D` version so that a later file fails in
`parse_fermionicqasm`, where the message names the version it read and the
version this reader implements (`1.0` only). `FERMIONICQASM 1.1;` and
`FERMIONICQASM 2.0;` are refused.

There is no external FermionicQASM corpus. An unknown version is a file
from a later spec, and reading it under 1.0 rules would misread whatever
that spec added. OPTICQASM stores its version and does not check it. That
gap was not copied.

```fermionicqasm-refuse version 1.1
FERMIONICQASM 1.1;
mode m[1];
```

```fermionicqasm-refuse version 2.0
FERMIONICQASM 2.0;
mode m[1];
```

---

## 5. Statements

Five forms, in this order: `mode_decl`, then `creg_decl`, then `load_stmt`,
then `measure_stmt`, then `gate_app`. The order is the parse. Pest's `|` is
ordered choice, and a gate application accepts any identifier, so the
specific forms have to come first. `creg c[4];` is a classical register. It
is not a gate named `creg`. `load` and `measure` are the same kind of
decision.

| Form | Syntax | Meaning |
|---|---|---|
| **mode** | `mode NAME '[' integer ']' ('spin')? ';'` | Fermionic register. Without `spin`, `NAME[i]` is wire `i` of that register and there are `integer` wires. With `spin`, see §6 |
| **creg** | `creg NAME '[' integer ']' ';'` | Classical register |
| **load** | `load mode_ref (',' mode_ref)* ';'` | Occupy those modes. Lowers to `X` on each wire, which is the occupation layer the sector backend already recognizes |
| **gate** | `gate_name ('(' param (',' param)* ')')? mode_ref (',' mode_ref)* ';'` | One of `num`, `numnum`, `givens`, `tunnel`. Each takes one parameter. §7 |
| **measure** | `measure mode_ref '->' cbit_ref ';'` | Measure that mode into that classical bit |

`mode_ref` is `NAME '[' integer ']'`. A gate parameter is a number or a
`$name` (§3).

`load` prepares the occupation and must precede every gate. A second `load`
of the same mode is refused: two `X`s would empty it and look like a
successful preparation. A file with no `load` parses. Every gate below
conserves particle number, so that file only phases the vacuum.

```fermionicqasm
// Every statement form. s[2] spin is wires 4..7 of the program:
// 4..5 spin-up, 6..7 spin-down. numnum on m[0], m[3] is not adjacent.
FERMIONICQASM 1.0;
mode m[4];
mode s[2] spin;
creg c[4];
load m[0], m[1];
num(0.7) m[0];
num($theta) m[3];
numnum(0.5) m[0], m[3];
givens(-0.7) s[0], s[1];
tunnel(0.4) s[2], s[3];
measure m[0] -> c[0];
```

```fermionicqasm-refuse after a gate
FERMIONICQASM 1.0;
mode m[2];
num(0.7) m[0];
load m[1];
```

```fermionicqasm-refuse already loaded
FERMIONICQASM 1.0;
mode m[2];
load m[0];
load m[0];
```

---

## 6. Spin layout

Blocked, not interleaved.

Read from qiskit-cold-atom at commit `ad8893f7b0c2d167f9295df10873a23ccfa55295`
(2024-02-14, <https://github.com/qiskit-community/qiskit-cold-atom>). The
package was not installed and not executed. The quotes are in
`crates/omega-parser/tests/fixtures/fermionicqasm/conventions.json` under
`spin_ordering`, copied from that commit:

* `qiskit_cold_atom/fermions/__init__.py` lines 43–47. A system of spin-½
  fermions in four spatial modes is eight wires, "the first four wires
  denote the spin-up and the last four wires denote the spin-down modes."
* `qiskit_cold_atom/circuit_tools.py` lines 23–32 and 276–281. Sequential
  means the first `num_sites` wires are the first species. Interleaved means
  the first `num_species` wires are the first site. The text distinguishes
  the two. FermionicQASM uses the sequential one.
* `qiskit_cold_atom/fermions/base_fermion_backend.py` lines 80–92. Species
  `i`, site `j` is wire `i * sites + j`.
* `qiskit_cold_atom/fermions/ffsim_backend.py` lines 365–380 and 631–637.
  Wire index `// norb` is the species. Species 0 is `ffsim.Spin.ALPHA`,
  species 1 is `ffsim.Spin.BETA`.

`mode m[N] spin;` allocates `2N` wires. `m[i]` for `i < N` is site `i`,
spin-up (alpha). `m[N+i]` is site `i`, spin-down (beta). The index is
already the wire. Nothing remaps it.

Two neighbouring conventions were not copied.

**The orbital reversal.** The same `ffsim_backend.py`, lines 378–380 and
392–393, reverses the orbital index before calling ffsim:

```conventions-quote
# reverse index due to qiskit convention
occupations[spin].add(norb - 1 - orb)
# reverse orbitals due to qiskit convention
alpha_orbs = [norb - 1 - orb for orb in alpha_orbs]
```

That reversal stays in the adapter. The source lines, including the two
comments "reverse index due to qiskit convention" and "reverse orbitals due
to qiskit convention", are quoted in `conventions.json` under
`spin_ordering.adapter_reverses_orbital_index_before_calling_ffsim`, with
`applied_to_these_matrices` false. The matrices pass orbital indices through
unchanged, and `ffsim.qiskit.jordan_wigner` places `number_operator(0,
Spin.ALPHA)` on qubit 0 and `number_operator(0, Spin.BETA)` on qubit `norb`.
FermionicQASM does not apply `norb - 1 - orb`. Copying it disagrees with
those matrices.

**Interleaved wires.** OPTICQASM's `pol` marker maps spatial mode `s` and
polarization `p` to wire `2s+p`. `spin` has the same optionality and a
different wire map. The lowerings are not shared. `load m[0]` on
`mode m[2] spin` occupies wire 0 (site 0, spin-up) and leaves wire 2 empty.
An interleaved map would put that particle on a different wire.

```fermionicqasm
// Site 0 up and site 0 down. numnum across the two blocks is onsite
// number-number: ffsim apply_on_site_interaction. The pair m[0], m[3]
// is number_operator product across species, also pinned.
FERMIONICQASM 1.0;
mode m[2] spin;
load m[0];
load m[2];
num(0.25) m[1];
numnum(0.7) m[0], m[2];
numnum(-0.9) m[0], m[3];
```

Under `spin`, a gate index names a wire (`0 .. 2N-1`), not a spatial mode
that expands into two wires. Fermionic gates act inside one spin species.
Per-wire addressing is the primitive.

---

## 7. Gate semantics, as F4.3 checked them

θ below is the one parameter. The operator identities are the ones the
ffsim 0.0.84 matrices pin. They are not a separate paper claim.

| Spelling | Operator | Lowering |
|---|---|---|
| `num(θ) m[p]` | `exp(i θ n_p)` | `U1(θ)` on `p`. `exp(i θ n̂) = diag(1, e^{iθ})` |
| `numnum(θ) m[p], m[q]` | `exp(i θ n_p n_q)` | `cphase` → `CU3(0, 0, θ)`, any two distinct wires. Number operators carry no Z string, so adjacency is not required, and the two wires may sit in different spin blocks (onsite, and number-operator product) |
| `givens(θ) m[p], m[q]` | `exp(θ (a†_p a_q − a†_q a_p))` | `Rbs(θ)` through `fermion::givens_expr`. Adjacent wires, same spin. §8 for the refusals |
| `tunnel(θ) m[p], m[q]` | `exp(i θ (a†_p a_q + a†_q a_p))` | In application order: `U1(−π/2)` on the second mode, `Rbs(−θ)` on the pair, `U1(+π/2)` on the second mode. Adjacent, same spin, because the `Rbs` is `givens_expr` |

`tunnel`'s pinned sign is `Rbs(−θ)`. The other conjugation,
`U1(−π/2)`, then `Rbs(+θ)`, then `U1(+π/2)`, matches the same matrices.
The lowering emits the first of those two. F4.3's sign-flip hook turns the
pinned call into `Rbs(+θ)` and requires the comparison to notice.

`givens` and `tunnel` go through `fermion::givens_expr`, so the
non-adjacent refusal has one source. `numnum` does not. A blanket adjacency
check would refuse legal number operators.

```fermionicqasm
// Adjacent givens and tunnel, and a non-adjacent numnum.
FERMIONICQASM 1.0;
mode m[4];
creg c[4];
load m[0];
load m[2];
givens(0.7) m[0], m[1];
tunnel(0.7) m[1], m[2];
numnum(0.7) m[0], m[3];
measure m[3] -> c[3];
```

### The Givens matrix ffsim executes

ffsim's `apply_givens_rotation` **docstring prints the off-diagonals the
opposite way round from the matrix it executes.** The pin is the executed
matrix. `conventions.json` records the disagreement as `not_a_pin` id
`apply_givens_rotation_docstring_matrix`. Editing the fixture to match the
docstring would put the wrong sign into the lowering. F4.3 compares the
executed matrix.

For `spinless_2/givens_0_1`, θ `0.7`, φ `0`, basis order `|00⟩, |01⟩, |10⟩,
|11⟩` as `ffsim.addresses_to_strings` and
`ffsim.qiskit.ffsim_vec_to_qiskit_vec` recorded it (orbital 0 on the right):
`+sin(0.7)` at row 1, column 2, and `−sin(0.7)` at row 2, column 1. The
docstring has those two entries swapped.

### What F4.3 ran

The test is `crates/omega-bridges/tests/ffsim_vs_fermionicqasm.rs`, at
`ca64f29`. Every circuit in it is FermionicQASM text passed to
`lower_to_ir`. ffsim never sees our matrices: the live leg ships the lowered
circuit as QPY. The fixture leg compares statevectors to `conventions.json`,
whose amplitudes are return values of ffsim 0.0.84 (`apply_num_interaction`,
`apply_num_num_interaction`, `apply_givens_rotation`,
`apply_tunneling_interaction`, `apply_on_site_interaction`,
`apply_num_op_prod_interaction`). The comparison tolerance in that file is
`1e-9`.

These are the checks that landed, not a general claim that every fermionic
circuit agrees with ffsim:

* **30 fixture gates, from text, every column.** `conventions.json` holds
  34 gate matrices (spinless 2, 3, and 4 modes, and 2 spatial orbitals with
  spin). 30 of them were lowered from `.fqasm` text and compared on every
  computational-basis column. Worst `|Δ|` `5.8e-13`. The other four are
  named and not compared: `spinless_3/givens_0_2`, `spinless_3/tunnel_0_2`,
  `spinless_4/givens_0_2`, `spinless_4/tunnel_0_2`. The fixture stores
  ffsim's matrix for a routing this lowering does not perform. The
  statements are refused (§8, Z string).
* **202 (gate, sector) pairs.** Each of those 30 gates, on one occupation
  per particle sector (and the gapped occupations the test lists). Live
  expectation against ffsim, worst `|Δ|` `1.2e-16`. The same pairs are also
  compared to the F4.0 column, phase included: on one basis state, `num`
  and `numnum` only multiply that amplitude by a phase, and an expectation
  of that one state does not see it. The count is 8×3 + 6×5 + 6×8 + 10×10
  (spinless 2, 3, 4 modes, then 2 spatial modes with spin).
* **Every particle sector of 4 spinless modes.** All 16 occupations.
  `N = 0, 1, 2, 3, 4` with 1, 4, 6, 4, 1 states. `N = 0` and `N = 4` are
  run and are not evidence: each is a global phase on one basis vector.
* **Every `(N_α, N_β)` of 2 spatial modes with spin.** The sweep's `givens`
  and `tunnel` stay inside one block. Cross-spin `numnum` is in the sweep,
  because those ffsim calls exist. Cross-spin `givens` and `tunnel` are
  not in it.
* **The sign flip is detected.** `with_tunnel_theta_sign_flipped` makes
  `tunnel` lower `Rbs(+θ)` instead of `Rbs(−θ)`. The oracle held fixed is
  the F4.0 matrix and the live ffsim expectations of the *unflipped*
  program. Sending the flipped circuit to both engines would agree on the
  mistake. The flipped state has to miss both, by more than `1e-2`, on
  `N = 1` (`load m[0]`) and on `N = 2` with a spectator on mode 2.
  Occupying both wires of the tunnelled pair makes `tunnel` the identity,
  so that occupation cannot show a sign error. The hook is compiled only
  for that test. A production build cannot flip the sign.

Same-spin agreement is not reported as covering cross-spin `givens` or
`tunnel`. There is no ffsim matrix for that pair. §8.

---

## 8. What is refused, and why

Each of these is grammatical enough to reach a named refusal, except the
expression and the non-literal size, which the grammar itself rejects.
Emitting the text and letting a later pass invent a meaning is the failure
the messages exist to prevent.

**Cross-spin `givens` and `tunnel`.** Unpinned. ffsim 0.0.84's
`apply_givens_rotation` and `apply_tunneling_interaction` each take one
`Spin` and stay inside that species. `conventions.json`
(`unpinned.cross_spin_givens_and_tunnel`) records that no matrix was
generated for a pair whose wires sit in different spin blocks, including
the adjacent pair across the block boundary (`m[N-1], m[N]`). The lowering
refuses that statement rather than inventing a matrix. Same-spin pairs on
each block still lower. `numnum` across spins is a different call
(`apply_on_site_interaction`, `apply_num_op_prod_interaction`) and is
accepted.

```fermionicqasm-refuse unpinned
FERMIONICQASM 1.0;
mode m[2] spin;
givens(0.7) m[1], m[2];
```

```fermionicqasm-refuse unpinned
FERMIONICQASM 1.0;
mode m[2] spin;
tunnel(0.7) m[1], m[2];
```

**Non-adjacent `givens` and `tunnel`.** The fermionic rotation between
modes that are not neighbours carries a Z string over every mode strictly
between them. `Rbs` does not. `fermion::givens_expr` refuses the pair
(`|p−q| ≠ 1`) with that Z-string message, for `givens` and for the `Rbs`
inside `tunnel`. Emitting `Rbs` anyway would run and return a wrong
number. The routing (a Z-string decomposition, or a swap network) is not
in this lowering. When it lands, it lands in `givens_expr`; these files
do not change. `numnum` on the same pair is accepted.

```fermionicqasm-refuse Z string
FERMIONICQASM 1.0;
mode m[4];
givens(0.7) m[0], m[2];
```

```fermionicqasm-refuse Z string
FERMIONICQASM 1.0;
mode m[4];
tunnel(0.7) m[0], m[2];
```

**`orbrot`.** Out of 1.0. A text assembly language is a poor container for
an `N×N` unitary. Writers decompose the orbital rotation to `givens`
before emission, which is what `ffsim.linalg.givens_decomposition` does,
so the decomposition has an external oracle and an `orbrot` statement
would not. The grammar admits the name as an identifier. The lowering
refuses it, and the writer will not print it.

```fermionicqasm-refuse orbrot is not in FermionicQASM 1.0
FERMIONICQASM 1.0;
mode m[2];
orbrot(0.7) m[0], m[1];
```

**Qubit gates.** `h`, `cx`, `rbs`, and any other name outside `num`,
`numnum`, `givens`, and `tunnel` are refused as an unknown fermionic gate.
`rbs` is the Jordan–Wigner spelling of adjacent `givens` after lowering.
It is not the text spelling. A qubit gate is not commented out: a comment
would re-import as a circuit missing the operation.

```fermionicqasm-refuse unknown fermionic gate
FERMIONICQASM 1.0;
mode m[1];
h m[0];
```

```fermionicqasm-refuse unknown fermionic gate
FERMIONICQASM 1.0;
mode m[2];
rbs(0.7) m[0], m[1];
```

**Expression parameters.** A parameter is a number or a `$name`. `pi`,
`theta/2`, and any other expression do not parse. The writer refuses the
same thing from the AST, rather than evaluating it: substituting a number
would emit a file that parses and computes a different circuit
(`writer_refuses_an_expression_parameter_rather_than_evaluating_it`).

```fermionicqasm-refuse pi
FERMIONICQASM 1.0;
mode m[1];
num(pi) m[0];
```

---

## 9. What a round-trip proves

F4.4 (`307027e`) is self-consistency. The test is
`write_read_write_of_every_fixture_is_byte_identical_consistency_not_validation`
in `crates/aria-core/tests/fermionicqasm_consistency_round_trip.rs`. Write,
read, write is byte-identical. That shows this emitter and this reader
agree with each other. It does not show that a gate means what the text
says it means.

There is no external oracle for the spelling, and there cannot be one we
did not write. Semantics evidence is F4.3 alone (§7). F4.3 does not consult
the writer. A green round-trip is not that result, and it must not be
quoted as that result.

The writer emits from Aria's AST. `tunnel` lowers to three operations, so
a circuit read back from `CircuitIR` cannot be folded into the spelling
`tunnel`. Re-emitting FermionicQASM from the IR is not a 1.0 goal. OPTICQASM
has the same split: `to_opticqasm` writes from its own AST.

The writer refuses the statements the lowering refuses. Emitting cross-spin
or non-adjacent `givens` / `tunnel`, or `orbrot`, would produce text the
reader accepts and the lowering then rejects.

---

## 10. The circuit file is not a Hamiltonian

The Hamiltonian is not in the file. A circuit is not a container for
integrals. FCIDUMP, the format PySCF and ffsim write, is the adopted
out-of-band convention. `omega-run circuit.fqasm --hamiltonian h.fcidump
--backend sector` reads that file into a `FermionicOp` — one FCIDUMP index,
one mode, chemist's notation — and measures it through the same Jordan–Wigner
path as `--expectation-fermionic`. The two flags are mutually exclusive, and
so is `--expectation`. The circuit file never names the Hamiltonian.

`--expectation-fermionic` remains the surface for OpenFermion's ladder
spelling. That text is not FermionicQASM, and neither is the FCIDUMP.
