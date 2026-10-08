<!-- SPDX-License-Identifier: Apache-2.0 -->
# Qudit examples — run them here, and against the analytic state

Every example is a DITQASM circuit with a number that can be derived by
hand. Two of our engines agreeing is not the reference: `quditsv` is the
dense state, `mps` is checked against that state, and the number written
below is the closed form.

**These are not `.aria` programs.** The Aria grammar has no wire
dimensions and no DITQASM import, and `crates/apps` plus `aria-verify`
are qubit-WASM-shaped. Bending `.aria` to carry a qutrit would make the
harness lie about what it ran. The surface is the `.ditqasm` file and
`omega-run`.

`h` on a qutrit is the Fourier gate `F_3`, not a qubit Hadamard.
`x` is the shift, `z` is the clock, `csum` is the digit-add
`|c⟩|t⟩ → |c⟩|t+c mod d⟩`. Wire 0 is the least significant digit, the
same convention as qubit 0 in every qubit engine here. There is no CLI
flag for a qudit observable: Pauli names are refused on a `d ≠ 2` wire,
and a site operator is a Rust call (`MpsBackend::expectation_site_operators`).

Sampling is refused. `Counts` outcomes are bit strings, and a qutrit
digit is not a bit. The readout below is `--statevector`.

| example | quantity | reference |
|---|---|---|
| qutrit GHZ | `1/√3` at indices 0, 13, 26 | analytic (`F_3\|0⟩`, then digit-copy) |
| spin-1 singlet | ⟨S₁·S₂⟩ = −2 | analytic. The singlet subspace of 1⊗1 is one-dimensional |
| qutrit Deutsch–Jozsa | P(index 0) is 1 or 0 | analytic. Constant versus balanced over Z₃ |

---

## 1. Qutrit GHZ — `qutrit_ghz.ditqasm`

```console
$ omega-run examples/qudit/qutrit_ghz.ditqasm --backend quditsv --statevector
$ omega-run examples/qudit/qutrit_ghz.ditqasm --backend mps --statevector
$ omega-run examples/qudit/qutrit_ghz.ditqasm --backend auto --statevector
```

`F_3|0⟩ = (|0⟩ + |1⟩ + |2⟩)/√3`, real and positive, because
`(F_3)_{j,0} = 1/√3`. Each `csum` from wire 0 copies that digit and adds
no phase. The state is

```text
(|000⟩ + |111⟩ + |222⟩) / √3
```

Little-endian index `q0 + 3 q1 + 9 q2`:

| state | index | amplitude |
|---|---|---|
| `\|000⟩` | 0 | `1/√3` ≈ 0.577350269190 |
| `\|111⟩` | 1+3+9 = 13 | `1/√3` |
| `\|222⟩` | 2+6+18 = 26 | `1/√3` |

Every other amplitude is 0. Tolerance against the closed form: 1e-12.
`mps` at its default bond (64) is in the exact regime on 27 amplitudes;
the two statevectors agree at the same tolerance. `--backend auto` prints

```text
auto: qudit register 'q' (dimension 3 on wire 0) -> quditsv (exact, dense mixed-radix)
```

and then the same state.

The same circuit through the API, asserting those three amplitudes:

```console
$ cargo run -p omega-backend-quditsv --example qutrit_ghz
```

**Sampling is a refusal, not a missing flag.**

```console
$ omega-run examples/qudit/qutrit_ghz.ditqasm --backend quditsv --shots 100
```

exits 1. The header on stdout still says `Circuit: 3 qubits` — that field
is the wire count — and stderr is the refusal:

```text
Execution error: unsupported operation: quditsv: sampling a qudit circuit is not supported yet: register 'q' has dimension 3 on wire 0, and `Counts` outcomes are bit strings. Use `--statevector` for the exact distribution.
```

That `Counts` sentence is the contract. A histogram of this state waits
on `Outcome` being able to hold a digit.

**What this check cannot see.** The two `csum` lines commute on this
input: swapping them leaves `|kkk⟩` unchanged, so an assertion on the
final amplitudes does not notice the swap. A relative phase between the
three peaks does change the amplitudes and is caught; a check that only
compared probabilities would miss it.

**External pin, optional.** The same gate list is one row of
`tools/quditsv_xcheck/xcheck.py` (`qutrit_ghz`). That harness compares
`quditsv` to mqt.qudits after aligning one global phase, so it does not
replace the phase-sensitive assertion above. It runs when the mqt venv
is present and is a registered skip otherwise (`ci.sh`).

---

## 2. Spin-1 singlet — `spin1_singlet.ditqasm`

Two qutrits, level 0, 1, 2 read as m = +1, 0, −1. The singlet of 1⊗1 is

```text
(|1, −1⟩ − |0, 0⟩ + |−1, 1⟩) / √3 = (|02⟩ − |11⟩ + |20⟩) / √3
```

Wire 0 is the least significant digit, so those kets are indices 6, 4
and 2. For two spins the identity

```text
S₁·S₂ = [S_tot(S_tot+1) − 2 S(S+1)] / 2
```

with S = 1 and S_tot = 0 gives **−2**. The singlet subspace appears
once, so a pure state with this value is that state up to a global phase.

There is still no CLI flag for a qudit observable. Pauli names are
refused on a wire with d ≠ 2, and `expectation_site_operators` is a Rust
call. The commands below print the state. The bond energy is contracted
from that state, and again inside
`crates/omega-backend-mps/examples/spin1_singlet.rs`, which computes it
two ways: three calls to `MpsBackend::expectation_site_operators`
(Sz⊗Sz and the two S⁺S⁻ terms) and the same matrices on quditsv's dense
amplitudes.

```console
$ omega-run examples/qudit/spin1_singlet.ditqasm --backend quditsv --statevector
$ omega-run examples/qudit/spin1_singlet.ditqasm --backend mps --statevector
$ cargo run -p omega-backend-mps --example spin1_singlet
```

The circuit is closed form, not a fit. On wire 0,

```text
rxy(0, 1, θ, −π/2),    θ = 2 arcsin(1/√3)
rxy(0, 2, π/2, π/2)
```

takes `|0⟩` to `(|0⟩ − |1⟩ + |2⟩)/√3`. Two shifts put wire 1 in `|2⟩`.
`csum` twice adds `2 q0` (mod 3), and `2 + 2 q0 ≡ −q0` (mod 3) against
that `|2⟩`, which is the correlation above. Tolerance on the amplitudes
is 1e-12. Tolerance on −2 is 1e-10.

**What this check cannot see.** The two `x` lines on wire 1 are the same
gate, and the two `csum` lines are the same gate. Swapping either pair
does not change the state. ⟨S₁·S₂⟩ is also blind to a global phase,
because the singlet subspace is one-dimensional; the signed amplitudes
are what catch a sign flip of the whole state.

---

## 3. Qutrit Deutsch–Jozsa

> D. Deutsch and R. Jozsa, *"Rapid solution of problems by quantum
> computation"*, **Proc. R. Soc. Lond. A 439, 553 (1992)**,
> doi:10.1098/rspa.1992.0167.
> The promise: the function is constant or balanced. The qutrit circuit
> below is that promise with the Fourier transform over Z₃ in place of
> the qubit Hadamard.

One work qutrit, `q[0]`, and one oracle ancilla, `q[1]`. The ancilla is
prepared in `F₃⁻¹|1⟩`, an eigenstate of the shift: adding `a` multiplies
that state by `ω^a`, `ω = exp(2πi/3)`. The constant oracle is `z` three
times on the work qutrit. `Z₃³ = I`, because `ω³ = 1`, so the phase does
not depend on the input. The balanced oracle is `csum` from the work
qutrit onto the ancilla, `f(x) = x`, which hits each output once.

The inverse Fourier is `h` three times. `F₃⁴ = I`, so `F₃³ = F₃⁻¹`.
There is no `hdg` gate. After that, `h` and two shifts return the
ancilla to `|0⟩`. Without that return, the constant case leaves the
ancilla spread over three levels and the probability of index 0 is 1/3,
not 1. With it, index 0 is the whole state.

```console
$ omega-run examples/qudit/qutrit_dj_constant.ditqasm --backend quditsv --statevector
$ omega-run examples/qudit/qutrit_dj_balanced.ditqasm --backend quditsv --statevector
```

The probability of index 0 is **1** for the constant oracle and **0** for
the balanced oracle, tolerance 1e-12. Readout is the statevector because
qudit `Counts` is refused, as in the GHZ section.

**What this check cannot see.** Three further `z` gates on the work
qutrit are another `Z₃³ = I`. The constant probability stays 1. The
three existing `z` lines also commute with each other, so reordering
them changes nothing.
