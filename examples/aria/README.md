# Aria quantum examples

These `.aria` files are the quantum circuits shipped with the language. Ten of
them are also **verified applications**: each runs through the omega WASM
runtime and is cross-checked against a pure-Rust classical oracle, so it is
always obvious *what* is computed and that quantum == classical numerically.

## ▶ Finding the harness for an example (the lazy path)

Open the `.aria` file — its **first two lines point straight at the harness**:

```aria
-- ▶ Application harness + classical cross-check: crates/apps/qsvd/src/lib.rs
--   run: cargo run -p aria-app-qsvd   (or: cargo run -p aria-verify -- qsvd)
```

Open that `crates/apps/<name>/src/lib.rs` and the **whole harness is right
there** (~60 lines): what is computed, the quantum run, the classical oracle,
and the numeric check — nothing else in the file. Run one standalone with
`cargo run -p aria-app-<name>`, or run them all with
`cargo run -p aria-verify -- all`. The shared toolkit (banner, oracles, the
`.aria`→wasm glue) lives once in `crates/aria-verify-core`.

| `.aria` example          | harness crate                    | computed quantity vs classical oracle      |
|--------------------------|----------------------------------|--------------------------------------------|
| `qsvd.aria`              | `crates/apps/qsvd`               | singular values of M vs Jacobi SVD         |
| `qft.aria`               | `crates/apps/qft`                | QFT\|x⟩ vs the DFT matrix                   |
| `vqe_ansatz.aria`        | `crates/apps/vqe-ansatz`         | H₂ ground energy vs exact eigenvalue       |
| `grover3.aria`           | `crates/apps/grover3`            | P(marked) vs analytic Grover probability   |
| `bernstein_vazirani.aria`| `crates/apps/bernstein-vazirani` | recovered string vs hidden `a`             |
| `deutsch_jozsa.aria`     | `crates/apps/deutsch-jozsa`      | balanced/constant vs the truth table       |
| `swap_test.aria`         | `crates/apps/swap-test`          | P(ancilla=0) vs ½+½\|⟨ψ\|φ⟩\|²             |
| `teleport.aria`          | `crates/apps/teleport`           | Bob's qubit vs Alice's input               |
| `qaoa_maxcut.aria`       | `crates/apps/qaoa-maxcut`        | QAOA cut vs brute-force MaxCut             |
| `qml_classifier.aria`    | `crates/apps/qml-classifier`     | confusion matrix vs ground-truth labels    |
| `qec_grover.aria`        | `crates/apps/qec-grover`         | encoded 2-qubit Grover ⟨Z̄⟩=±1 vs marked   |
| `qec_qft.aria`           | `crates/apps/qec-qft`            | encoded QFT uniform + QFT∘QFT⁻¹ = identity |
| `qec_qpe.aria`           | `crates/apps/qec-qpe`            | encoded QPE φ̂ vs exact φ = j/2^m           |

The three `qec_*` examples are LOGICAL twins: the harness runs each algorithm
transversally encoded on Steane [[7,1,3]] logical qubits (via the `aria-qec`
crate) and checks the encoded result equals the ideal, proving the algorithm is
invariant under the low-overhead QEC encoding. A fourth encoded demo,
`crates/apps/qec-memory` (surface-code memory: pL(d=5) < pL(d=3) under
neutral-atom noise), is native-only (no `.aria` twin — it is a Monte-Carlo, not
a fixed circuit).

See [`../../TESTING.md`](../../TESTING.md) §13–14 for the per-example numeric
goldens and the socket transport. The remaining `.aria` files (`bell`, `qpe`,
`simon`, `superdense`, `qcnn`, …) are circuit sources that parse and run but do
not have a verify harness yet.

## Syntax

Top-level block is `circuit NAME(...) { ... }`; bodies use `let` / `var` /
`apply` / `repeat` / `measure`:

```aria
circuit Bell {
    qreg q[2]
    creg c[2]

    apply H on q[0]
    apply CX on q[0], q[1]

    measure q -> c
}
```

Keywords:

- `circuit NAME(params)` — top-level declaration, optional typed params
- `qreg q[n]` / `creg c[n]` — quantum / classical register declarations
- `let x = expr` — immutable binding (constants like `pi`, `sqrt(2)`)
- `var x = expr` — mutable binding (loop accumulators)
- `let s = symbolic[k]` — `k` trainable parameters (`s[0]..s[k-1]`)
- `apply GATE(params) on q[i]`, `apply GATE on q[i], q[j]` — gate application
- `repeat N { ... }`, `repeat i from a to b { ... }` — loop forms
- `measure q -> c` — measure quantum register into classical register
- `oracle NAME(args) on q[...]` — call a sub-circuit as a named oracle
- `--` comments

Annotations are written at the top of the circuit block with `@`:

```aria
@assert unitary
@prove "grover_correct" equiv { amplifies |marked> }
@bound iterations = 2

circuit Grover3(marked: int) { ... }
```

## Parser status

Every file here is parsed by `aria_core::ast::aria::parse_aria(...)` and
instantiated via `AriaProgram::instantiate(name, &[(param, int)])`. The
`crates/aria-core/tests/aria_examples.rs` test exercises **every** `.aria` file
on disk end-to-end (parse + instantiate to a non-empty circuit).
