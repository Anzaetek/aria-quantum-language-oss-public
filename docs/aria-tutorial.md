# Aria — a tutorial for the quantum language

**Aria** is a small, readable DSL for writing quantum circuits with **built-in proof obligations**.
An Aria model is human-readable, parameterized, runs on a pure-Rust backend, and exports to QASM, a
Lean 4 theorem, and JSON — all through the `aria` CLI. This tutorial takes you from a first circuit
to parameterized, trainable, formally-specified models.

> Editor support for `.aria` files lives in `editors/`: a Tree-sitter grammar, a Neovim plugin,
> and a VS Code TextMate grammar.

## 1. Your first circuit — a Bell state

```aria
-- Bell state: (|00> + |11>) / sqrt(2)
@assert unitary
@prove "bell_correct" equiv { creates (|00> + |11>)/sqrt(2) }
@bound gate_count = 2

circuit Bell {
    qreg q[2]          -- 2 qubits
    creg c[2]          -- 2 classical bits

    apply H on q[0]            -- superpose q[0]
    apply CX on q[0], q[1]     -- entangle q[1] with q[0]

    measure q -> c            -- measure all qubits into c
}
```

- **`circuit Name { ... }`** declares a circuit.
- **`qreg q[n]` / `creg c[n]`** declare quantum / classical registers.
- **`apply GATE on q[i]`** applies a gate; two-qubit gates take two wires (`apply CX on q[0], q[1]`).
- **`measure q -> c`** measures.
- Lines starting `--` are comments.

### Annotations (the proof obligations)
- **`@assert unitary`** — the circuit must be unitary.
- **`@prove "name" equiv { ... }`** — a semantic claim carried into the Lean 4 export.
- **`@bound metric = value`** — a resource bound (e.g. `gate_count`, `depth`).

## 2. Run it with the `aria` CLI

```bash
# Run it on the pure-Rust statevector backend:
aria run bell.aria --circuit Bell --statevector       # |00> and |11> at 1/sqrt2
aria run bell.aria --circuit Bell --shots 4096        # sampled counts
aria run bell.aria --circuit Bell --expectation "Z0 Z1"   # = 1.000000000000

# Inspect / export:
aria list   bell.aria                       # list circuit templates
aria export bell.aria --circuit Bell --qasm # OPENQASM 2.0
aria export bell.aria --circuit Bell --lean # a Lean 4 theorem (proof obligation)
```

## 3. Parameters and loops — the QFT

Circuits take integer parameters and use `repeat` loops:

```aria
@assert unitary
@prove "qft_equals_dft" equiv { denote(QFT) = DFT_matrix(2^n) }
@bound depth = n * (n + 1) / 2

circuit QFT(n: int) {
    qreg q[n]

    repeat i from 0 to n - 1 {
        apply H on q[i]
        repeat j from i + 1 to n - 1 {
            apply CP(pi / (2.0 ^ (j - i))) on q[j], q[i]   -- controlled phase
        }
    }
    repeat i from 0 to (n / 2) - 1 {
        apply SWAP on q[i], q[n - 1 - i]                   -- bit-reversal
    }
}
```

Instantiate at any size: `aria run qft.aria --circuit QFT --int n=3 --statevector`.
`repeat i from a to b { ... }` is inclusive; `step -1` counts down. Angle expressions support
`pi`, `^`, `/`, `*`, `+`, `-`.

## 4. Symbolic (trainable) parameters — a QML model

Use `let x = symbolic[k]` for trainable angles (an optimizer binds them per step):

```aria
circuit QMLClassifier(L: int) {
    qreg q[1]
    let theta = symbolic[3 * L]              -- 3 trainable angles per layer

    repeat layer from 0 to L - 1 {
        apply RY(pi / 4) on q[0]             -- data-reuploading placeholder
        apply RZ(theta[3 * layer + 0]) on q[0]
        apply RY(theta[3 * layer + 1]) on q[0]
        apply RZ(theta[3 * layer + 2]) on q[0]
    }
}
```

Train those angles with pure-Rust parameter-shift gradients — no libtorch:

```bash
# Minimize an observable over the trainable angles (here a single-qubit demo):
aria train model.aria --circuit QMLClassifier --int L=3 --observable "Z0" \
    --steps 300 --lr 0.2
```

`aria train` reports the initial and final ⟨O⟩ and the trained parameters. The
shipped `vqe_ansatz.aria` recovers the exact H₂ ground-state energy this way.

## 5. Observables

Read-outs are declared separately and lowered alongside the circuit:

```aria
observable Representation {
    1.0 * Z(0)        -- <Z> on qubit 0
}
```

## 6. Gate vocabulary (common)

`H`, `X`, `Y`, `Z`, `S`, `T`, `RX(θ)`, `RY(θ)`, `RZ(θ)`, `P(λ)`/`CP(λ)`, `CX`, `CY`, `CZ`, `SWAP`,
`CCX`, `CSWAP`. Multi-qubit gates list their wires in order. (Run lowering targets this universal
gate set; `SX`/`RXX`/`RYY`/`RZZ` parse but are not yet wired for `aria run`.)

## 7. Verified applications — find the harness from the `.aria`

Ten examples are also **applications**: they run through the omega WASM runtime
and are cross-checked against a pure-Rust classical oracle. To find the harness
for any of them, just open the `.aria` file — its first two lines point at it:

```aria
-- ▶ Application harness + classical cross-check: crates/apps/qsvd/src/lib.rs
--   run: cargo run -p aria-app-qsvd   (or: cargo run -p aria-verify -- qsvd)
```

Open that `crates/apps/<name>/src/lib.rs` and the whole harness is right there
(~60 lines): what is computed, the quantum run, the classical oracle, the check.

```bash
cargo run -p aria-app-qsvd        # one example, its own crate
cargo run -p aria-verify -- all   # all ten, quantum vs classical
```

See `examples/aria/README.md` for the example→crate table and `TESTING.md`
§13–14 for the numeric goldens and the socket transport.

## 8. Where to go next

- `examples/aria/` has 27 worked models (Grover, Shor-ECDLP, QPE, teleport, QAOA, QCBM, …).
- `aria export … --lean` emits a Lean 4 theorem carrying the circuit's proof obligations.
- Pick a faster backend with `--backend mps`, or implement
  `omega_core::executor::Backend` to plug in your own.

Aria's point: a quantum model you can **read, parameterize, run, train, and prove** — from one source.
