# Qudit examples

DITQASM circuits for qutrit states and algorithms. They run through
`omega-run --backend quditsv` (exact dense) or `--backend mps`. They are
not `.aria` files: the Aria grammar has no wire dimensions and no DITQASM
import, and the application harness is qubit-shaped. Do not rewrite these
as `.aria`.

The numbers, the commands, and what each check cannot see are in
[`docs/QUDIT_EXAMPLES.md`](../../docs/QUDIT_EXAMPLES.md).

| file | what it prepares | reference |
|---|---|---|
| `qutrit_ghz.ditqasm` | `(|000⟩ + |111⟩ + |222⟩)/√3` | analytic, `1/√3` at indices 0, 13, 26 |
| `spin1_singlet.ditqasm` | `(|02⟩ − |11⟩ + |20⟩)/√3` | analytic, ⟨S₁·S₂⟩ = −2 |
| `qutrit_dj_constant.ditqasm`, `qutrit_dj_balanced.ditqasm` | Deutsch–Jozsa, one work qutrit | analytic, P(index 0) is 1 or 0 |

The same circuits through the Rust API. The GHZ example asserts the three
amplitudes. The singlet example asserts ⟨S₁·S₂⟩ = −2, both from MPS site
operators and from a contraction of the dense state.

```console
$ cargo run -p omega-backend-quditsv --example qutrit_ghz
$ cargo run -p omega-backend-mps --example spin1_singlet
```
