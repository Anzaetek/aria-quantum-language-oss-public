<!-- SPDX-License-Identifier: Apache-2.0 -->
# The Aria language — reference

Aria is a small, statically-shaped DSL for **parametrized gate-model quantum
circuits**. A program is a set of *circuit templates* (and optional
*observables*); instantiating a template with integer parameters unrolls all
loops and folds all compile-time arithmetic into a concrete circuit, which the
[omega runtime](README.md) then lowers and executes.

This document is the authoritative grammar, matching the parser in
[`crates/aria-core/src/ast/aria.rs`](crates/aria-core/src/ast/aria.rs). For a
hands-on introduction see [`TUTORIAL.md`](TUTORIAL.md); for the language *in
anger* read any file under [`examples/aria/`](examples/aria).

---

## 1. Lexical structure

- **Comments** start with `--` and run to end of line. There are no block
  comments.
- **Whitespace** (including newlines) is insignificant except as a token
  separator; an annotation (`@…`) ends at end of line (see §7).
- **Identifiers** are the usual `[A-Za-z_][A-Za-z0-9_]*`. They are
  case-sensitive — gate *names* are conventionally upper-case (`H`, `CX`), but
  the gate lookup is exact (see §6).
- **Integer literals**: `0`, `42`. **Float literals**: `3.14`, `0.125`,
  `2.0`, and scientific notation `1e-05`, `2.5E+3`, `1e5` (an exponent makes
  the literal a float even without a `.`). **Booleans**: `true`, `false`
  (lower to `1` / `0`).
- **`pi`** is a built-in constant (π).
- **Keywords** (reserved): `circuit observable qreg creg let var apply on
  repeat from to step measure oracle when pi symbolic true false`.
- **Operators / punctuation**: `+ - * / % ^` `== != < <= > >=` `= @ , : ->`
  `( ) [ ] { }`.

---

## 2. Program structure

```
Program   := ( Annotation* (Circuit | Observable) )*
```

Annotations bind to the **next** `circuit` (observables do not carry
annotations). A program may contain any number of circuits and observables in
any order.

---

## 3. Circuits

```
Circuit   := 'circuit' IDENT ParamList? '{' Statement* '}'
ParamList := '(' (IDENT (':' IDENT)?  (',' …)* )? ')'
```

The parameter type annotation is optional and currently advisory (`int` is
assumed). Parameters are **compile-time integers** supplied at instantiation
(`--int n=4`); they may size registers and drive loops.

```aria
circuit QFT(n: int) {
    qreg q[n]
    ...
}
```

---

## 4. Statements

```
Statement := Qreg | Creg | Let | Symbolic | Apply | Measure
           | Repeat | When | Oracle
```

| Form | Syntax | Meaning |
|---|---|---|
| **qreg** | `qreg NAME '[' Expr ']'` | declare a quantum register of `Expr` qubits |
| **creg** | `creg NAME '[' Expr ']'` | declare a classical register |
| **let / var** | `let NAME '=' Expr` | bind a compile-time int/float constant |
| **symbolic** | `let NAME '=' symbolic '[' Expr ']'` | declare an array of `N` free (trainable) parameters `NAME[0..N-1]` |
| **apply** | `apply GATE ('(' Expr (',' …)* ')')? on Target (',' …)*` | apply a gate to one or more qubits |
| **measure** | `measure Target '->' Target` | measure qubit(s) into classical bit(s) |
| **repeat (count)** | `repeat Expr '{' Statement* '}'` | unroll the body `Expr` times |
| **repeat (range)** | `repeat IDENT from Expr to Expr ('step' Expr)? '{' … '}'` | loop `IDENT` over an inclusive range (default step 1) |
| **when** | `when Expr '{' Statement* '}'` | include the body at compile time iff `Expr ≠ 0`; if the condition is runtime (depends on a measured bit) the body lowers to classically-conditioned gates |
| **oracle** | `oracle NAME ('(' Expr,… ')')? on Target,…` | invoke a named oracle subroutine |

`measure q -> c` over whole registers is a register-wide measurement; an indexed
form (`measure q[0] -> c[0]`) measures a single qubit.

> **Note.** `repeat … from A to B` is **inclusive** of `B` (e.g. `from 0 to n-1`
> runs `n` times). Loop variables and `let` bindings are compile-time values
> usable in any later `Expr`.

---

## 5. Expressions

Expressions are integer/float arithmetic evaluated at **compile time** (for
register sizes, loop bounds, indices, `when` conditions) or lowered to a
**parameter expression** (for gate angles, which may reference symbolic
parameters). Precedence, lowest to highest:

| Level | Operators | Assoc |
|---|---|---|
| comparison | `==` `!=` `<` `<=` `>` `>=` | left |
| additive | `+` `-` | left |
| multiplicative | `*` `/` `%` | left |
| power | `^` | right |
| unary | `-` (negation) | prefix |
| atom | literal / `pi` / `IDENT` / `IDENT[Expr]` / `IDENT(Expr,…)` / `(Expr)` | — |

Atoms:

- `IDENT` — a `let`/param binding, or (in a gate angle) a free symbol.
- `IDENT[Expr]` — index into a `symbolic[]` array (in a gate angle) or a
  register (in a target).
- `IDENT(Expr,…)` — a **function call**, lowered to a runtime-resolved
  `FnCall` (e.g. `sin`, `cos`, `sqrt`).
- `pi`, integer/float literals, `true`/`false`, and parenthesised sub-expressions.

Restrictions inside **gate angles**: `^` is only foldable when both sides are
compile-time constant, `%` is not allowed, and comparisons are not values.

---

## 6. Gate set

`apply GATE(params…) on qubits…`. Names are matched exactly; aliases in
parentheses.

| Gate | Qubits | Params | Notes |
|---|---|---|---|
| `I` (`ID`) | 1 | 0 | identity |
| `X` `Y` `Z` `H` | 1 | 0 | Paulis, Hadamard |
| `S` `SDG` `T` `TDG` | 1 | 0 | phase gates and adjoints |
| `SX` | 1 | 0 | √X |
| `RX` `RY` `RZ` | 1 | 1 | axis rotations `(θ)` |
| `P` (`U1`) | 1 | 1 | phase `(λ)` = `diag(1, e^{iλ})` |
| `U` (`U3`) | 1 | 3 | general single-qubit `(θ, φ, λ)` |
| `CX` (`CNOT`) `CY` `CZ` | 2 | 0 | controlled Paulis |
| `SWAP` | 2 | 0 | swap |
| `CP` | 2 | 1 | controlled phase `(λ)` — lowers to `CU3(0,0,λ)`, exact |
| `RXX` `RYY` `RZZ` | 2 | 1 | two-qubit rotations `(θ)` |
| `RBS` | 2 | 1 | Givens rotation `exp(−iθ/2(Y⊗X−X⊗Y))` — Hamming-weight preserving (butterfly QNNs, arXiv:2606.03517) |
| `CCX` (`TOFFOLI`) | 3 | 0 | Toffoli |
| `CSWAP` (`FREDKIN`) | 3 | 0 | controlled-swap |

Conventions are standard (Qiskit): `RZ(θ)=diag(e^{-iθ/2},e^{iθ/2})`,
`U(θ,φ,λ)` the usual Euler form.

---

## 7. Annotations

```
Annotation := '@' IDENT  <rest of line / brace block>
```

Annotations decorate the following circuit. Recognized forms:

| Annotation | Effect |
|---|---|
| `@assert unitary` | record a unitarity obligation (`Property::Unitary`) |
| `@assert self_inverse` | record `U² = I` |
| `@assert hermitian` | record `U = U†` |
| `@prove "name" …` | name a correctness obligation discharged by the Lean export |
| `@bound …` (`@resource_bound`) | record a resource bound (gate count, qubits, …) as a comment |

Unrecognised `@kind …` is kept verbatim as a comment. These are *metadata*:
they drive the gate-model `--lean` / `--gate-model` export and the resource
report, and never change the executed gates.

```aria
@assert unitary
@prove "bell_correct" equiv { creates (|00> + |11>)/sqrt(2) }
@bound gate_count = 2
circuit Bell { ... }
```

---

## 8. Observables

An `observable` is a weighted sum of Pauli strings, consumed by `aria run
--expectation` and `aria train --observable`.

```
Observable := 'observable' IDENT '{' Body '}'
Body       := ( Let | Term )+                  -- the top level is a sum
Let        := 'let' IDENT '=' NumberExpr
Term       := ('+'|'-')? Factor ('*' Factor)*
Factor     := NumberExpr | 'I' | Pauli
Pauli      := ('X'|'Y'|'Z') '(' INT ')'        -- Pauli on a qubit index
NumberExpr := ['-'] (FLOAT | INT | IDENT)      -- IDENT resolved from a `let`
```

```aria
observable H {
    let g0 = -0.4804
    let g1 =  0.3435
      g0 * I
    + g1 * Z(0)
    + 0.5  * X(0) * X(1)
    - 0.25 * Y(0) * Y(1) * Z(2)
}
```

On the command line a bare Pauli string is also accepted, e.g.
`--expectation "Z0"` or `--observable "1.0*Z0 Z1"`.

---

## 9. Symbolic (trainable) parameters

`let θ = symbolic[N]` declares `N` free parameters named `θ_0 … θ_{N-1}`, used
in gate angles as `θ[k]`. They are left **unbound** by the lowering and become
the trainable degrees of freedom: bind them at run time with `--bind θ_0=0.3 …`,
or let `aria train` optimise them (pure-Rust by default; `--backend tch` for a
libtorch accelerator). The free symbols of a circuit are ordered by name for the
positional parameter vector the runtime expects.

```aria
circuit Ansatz(n: int, L: int) {
    qreg q[n]
    let w = symbolic[n * L]
    repeat l from 0 to L - 1 {
        repeat i from 0 to n - 1 { apply RY(w[l * n + i]) on q[i] }
        repeat i from 0 to n - 2 { apply CX on q[i], q[i + 1] }
    }
}
```

---

## 10. Grammar summary (EBNF)

```ebnf
Program    = { Annotation } , ( Circuit | Observable ) , { ... } ;
Annotation = "@" , IDENT , line-content ;
Circuit    = "circuit" , IDENT , [ ParamList ] , "{" , { Statement } , "}" ;
ParamList  = "(" , [ Param , { "," , Param } ] , ")" ;
Param      = IDENT , [ ":" , IDENT ] ;

Statement  = Qreg | Creg | Let | Symbolic | Apply | Measure
           | RepeatN | RepeatRange | When | Oracle ;
Qreg       = "qreg" , IDENT , "[" , Expr , "]" ;
Creg       = "creg" , IDENT , "[" , Expr , "]" ;
Let        = ("let" | "var") , IDENT , "=" , Expr ;
Symbolic   = ("let" | "var") , IDENT , "=" , "symbolic" , "[" , Expr , "]" ;
Apply      = "apply" , IDENT , [ "(" , [ Expr , { "," , Expr } ] , ")" ] ,
             "on" , TargetList ;
Measure    = "measure" , Target , "->" , Target ;
RepeatN    = "repeat" , Expr , Block ;
RepeatRange= "repeat" , IDENT , "from" , Expr , "to" , Expr ,
             [ "step" , Expr ] , Block ;
When       = "when" , Expr , Block ;
Oracle     = "oracle" , IDENT , [ "(" , [ Expr , {"," , Expr} ] , ")" ] ,
             "on" , TargetList ;
Block      = "{" , { Statement } , "}" ;

TargetList = Target , { "," , Target } ;
Target     = IDENT , [ "[" , Expr , "]" ] ;

Expr       = Cmp ;
Cmp        = Add , { ("=="|"!="|"<"|"<="|">"|">=") , Add } ;
Add        = Mul , { ("+"|"-") , Mul } ;
Mul        = Pow , { ("*"|"/"|"%") , Pow } ;
Pow        = Unary , [ "^" , Pow ] ;             (* right-associative *)
Unary      = [ "-" ] , Atom ;
Atom       = INT | FLOAT | "pi" | "true" | "false"
           | IDENT , [ "(" Args ")" | "[" Expr "]" ]
           | "(" , Expr , ")" ;
Args       = [ Expr , { "," , Expr } ] ;
```

---

## 11. Lowering semantics (what instantiation does)

1. **Parse** to templates (this grammar).
2. **Instantiate** a named circuit with integer params: bind params, evaluate
   register sizes, **unroll** every `repeat`, **evaluate** `when` conditions,
   fold all compile-time arithmetic, and resolve indices to a flat qubit space.
   The result is a concrete `Circuit` (gates over `register[index]` qubits).
3. **Lower** to `omega_core::CircuitIR`: each gate angle becomes a `ParamExpr`
   (constant, free `Symbol`, or `FnCall`); `CP(λ) → CU3(0,0,λ)`; symbols are
   collected for the parameter vector.
4. **Execute / export** via the runtime (`run`, `train`) or emitters (`--qasm`,
   `--json`, `--lean`, `--gate-model`).

Every shipped example is held to a numeric standard — see
[`VERIFICATION.md`](VERIFICATION.md). Known scope boundaries are in
[`LIMITATIONS.md`](LIMITATIONS.md).
