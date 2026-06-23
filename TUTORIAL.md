<!-- SPDX-License-Identifier: Apache-2.0 -->
# Aria quantum tutorial

A hands-on tour of the **Aria Quantum Language**: write a circuit, run it,
read it numerically, parametrize and train it, and export it to QASM / Lean.
Every command below is real and its output is shown — you can paste them and
compare. For the full syntax see [`GRAMMAR.md`](GRAMMAR.md); for the verified
example catalogue see [`VERIFICATION.md`](VERIFICATION.md).

## 0. Build the CLI

```bash
cargo build -p aria-cli           # produces `aria` (debug: target/debug/aria)
alias aria="cargo run -q -p aria-cli --"
```

All examples live in [`examples/aria/`](examples/aria); we use them directly.

---

## 1. Your first circuit — a Bell state

[`examples/aria/bell.aria`](examples/aria/bell.aria):

```aria
@assert unitary
@prove "bell_correct" equiv { creates (|00> + |11>)/sqrt(2) }
@bound gate_count = 2

circuit Bell {
    qreg q[2]
    creg c[2]
    apply H on q[0]            -- superpose q[0]
    apply CX on q[0], q[1]     -- entangle q[1] with q[0]
    measure q -> c
}
```

A program is a set of **circuit templates**. `qreg`/`creg` declare quantum and
classical registers; `apply GATE on qubits` adds gates; `measure q -> c` reads
out. The `@` lines are metadata (§7 of the grammar) — they drive the proof
export and never change the gates.

List what a file defines:

```bash
$ aria list examples/aria/bell.aria
examples/aria/bell.aria:
  circuit Bell()
```

---

## 2. Run it — three numeric views

**Exact statevector** (measurements are skipped):

```bash
$ aria run examples/aria/bell.aria --circuit Bell --statevector
|00>  +0.707107+0.000000i
|11>  +0.707107+0.000000i
```

That is `(|00⟩+|11⟩)/√2` — amplitudes `1/√2 ≈ 0.707107`, exactly as the
`@prove` line claims.

**Sampled counts** (reproducible with a seed):

```bash
$ aria run examples/aria/bell.aria --circuit Bell --shots 2000 --seed 1
|00>  1002  (0.5010)
|11>  998  (0.4990)
```

Two outcomes only, ~50/50 — the hallmark of entanglement.

**Expectation value** of a Pauli observable:

```bash
$ aria run examples/aria/bell.aria --circuit Bell --expectation "Z0 Z1"
<Z0 Z1> = 1.000000000000
```

`⟨Z₀Z₁⟩ = 1` because the two qubits are perfectly correlated.

> **Try GHZ.** [`ghz.aria`](examples/aria/ghz.aria) extends the chain to three
> qubits (`H; CX 0→1; CX 1→2`). `--statevector` shows `|000⟩` and `|111⟩` at
> `0.707107` each.

---

## 3. Parameters and loops — the QFT

Templates take **compile-time integer parameters** and unroll loops. The
quantum Fourier transform, [`qft.aria`](examples/aria/qft.aria):

```aria
circuit QFT(n: int) {
    qreg q[n]
    repeat i from 0 to n - 1 {
        apply H on q[i]
        repeat j from i + 1 to n - 1 {
            apply CP(pi / (2.0 ^ (j - i))) on q[j], q[i]
        }
    }
    repeat i from 0 to (n / 2) - 1 {
        apply SWAP on q[i], q[n - 1 - i]
    }
}
```

`repeat … from A to B` is an **inclusive** compile-time loop; `pi`, `^`, `/` are
folded when instantiating. Pick `n` at run time with `--int`:

```bash
$ aria run examples/aria/qft.aria --circuit QFT --int n=3 --statevector
|000>  +0.353553+0.000000i
|001>  +0.353553+0.000000i
|010>  +0.353553+0.000000i
|011>  +0.353553+0.000000i
...
```

`QFT|000⟩` is the uniform superposition — every amplitude `1/√8 ≈ 0.353553`.

---

## 4. Compile-time choices — `when`

`when COND { … }` includes its body at instantiation iff `COND ≠ 0`. Superdense
coding, [`superdense.aria`](examples/aria/superdense.aria), uses it to encode
two classical input bits:

```aria
circuit Superdense(b0: int, b1: int) {
    qreg q[2]
    creg c[2]
    apply H on q[0]
    apply CX on q[0], q[1]          -- shared Bell pair
    when b1 == 1 { apply Z on q[0] } -- Alice's encoding
    when b0 == 1 { apply X on q[0] }
    apply CX on q[0], q[1]          -- Bob decodes
    apply H on q[0]
    measure q[0] -> c[0]
    measure q[1] -> c[1]
}
```

```bash
$ aria run examples/aria/superdense.aria --circuit Superdense --int b0=1 --int b1=0 --shots 256
|10>  256  (1.0000)
```

Deterministic: Bob recovers exactly the bits Alice sent. (When a `when`
condition depends on a *measured* bit instead of a parameter, it lowers to a
classically-conditioned gate rather than being resolved at compile time.)

---

## 5. Trainable circuits — `symbolic[]`

`let θ = symbolic[N]` declares `N` **free parameters** `θ[0..N-1]`, left unbound
by the circuit. They are the trainable degrees of freedom. The VQE ansatz,
[`vqe_ansatz.aria`](examples/aria/vqe_ansatz.aria):

```aria
circuit VQEAnsatz(n_layers: int) {
    qreg q[2]
    let theta = symbolic[4 * n_layers]
    repeat layer from 0 to n_layers - 1 {
        apply RY(theta[4*layer + 0]) on q[0]
        apply RY(theta[4*layer + 1]) on q[1]
        apply RZ(theta[4*layer + 2]) on q[0]
        apply RZ(theta[4*layer + 3]) on q[1]
        apply CX on q[0], q[1]
    }
}
```

You can **bind** the symbols by hand (`--bind theta_0=0.3 …`) for a forward run,
or let `aria train` optimise them against an observable. Training runs
**pure-Rust by default** (analytic/adjoint gradients via the omega runtime — no
libtorch):

```bash
# Minimise ⟨Z0⟩ over the ansatz parameters (toy objective).
$ aria train examples/aria/vqe_ansatz.aria --circuit VQEAnsatz \
      --observable "Z0" --int n_layers=3
...
  theta_8 = -1.138190
  theta_9 = 0.754697
```

The headline result — recovering the **H₂ ground-state energy** with the
bundled Bravyi–Kitaev `H2` observable — is checked numerically end-to-end by the
example app:

```bash
$ cargo run -q -p aria-verify -- --native vqe_ansatz
  QUANTUM   (VQE min energy):       -1.8511991070
  CLASSICAL (exact min eigenvalue): -1.8511991241
  Δmax = 1.710e-8   PASS (tol 1.0e-3)
```

`-1.851199` is the exact minimum eigenvalue of the H₂ Hamiltonian — trained in
pure Rust. (For a libtorch/GPU accelerator, add `--backend tch` or
`--backend gpu`; the result is identical to ≤ 1e-6.)

---

## 6. Observables

An `observable` is a weighted Pauli sum. [`vqe_ansatz.aria`](examples/aria/vqe_ansatz.aria)
ships the `H2` Hamiltonian as one:

```aria
observable H2 {
    let g0 = ...
      g0 * I
    + g1 * Z(0)
    + ...
}
```

On the command line you pass a Pauli string directly — `"Z0"`, `"Z0 Z1"`
(a product), or a sum like `"0.5*Z0 + 0.5*X1"` — to `--expectation` (one shot)
or `--observable` (the training objective).

---

## 7. Pluggable backends

The same circuit runs on any backend implementing the omega `Backend` contract;
select with `--backend`:

| `--backend` | engine |
|---|---|
| `sim` (default) | pure-Rust dense statevector |
| `mps` | matrix-product-state |
| `gpu` | Metal / CUDA / OpenCL (auto-fallback to `sim`) |
| `tch` | libtorch (optional feature) |
| `remote` | a running `omega-server` over HTTP (`--url`, `--token`) |

```bash
$ aria run examples/aria/qft.aria --circuit QFT --int n=3 --backend mps --statevector
# same amplitudes as `sim`, ≤ 1e-9
```

---

## 8. Export — QASM, JSON, Lean

```bash
$ aria export examples/aria/bell.aria --circuit Bell --qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
```

`--lean` emits a self-contained Lean 4 file that imports the bundled
`QuantumProofs` subtree, so the circuit's correctness can be machine-checked
(`--gate-model` emits the gate-model spec for the recognized algorithms — Bell,
GHZ, QFT, QPE, Grover — each delegating to a sorry-free theorem):

```bash
$ aria export examples/aria/bell.aria --circuit Bell --lean | head -5
-- Auto-generated by the Aria Quantum Language
-- Verify with: lake build
import QuantumProofs.CircuitSemantics
import QuantumProofs.Gates
```

Build the proofs with `ARIA_LEAN=1 ./ci.sh` (needs a warm mathlib cache).

---

## 9. Verify everything numerically

Every shipped example runs as a real application cross-checked against a
classical oracle. Run the whole suite:

```bash
# Build the wasm guests once, then:
$ cargo run -p aria-verify -- all
...
  31/31 passed
```

See [`VERIFICATION.md`](VERIFICATION.md) for the per-example oracle table, and
[`LIMITATIONS.md`](LIMITATIONS.md) for scope boundaries.

---

## Where to go next

- **Write your own.** Copy `bell.aria`, change the gates, `aria run --statevector`.
- **Read real circuits.** [`examples/aria/`](examples/aria) has 31: Grover, QPE,
  HHL, QAOA, quantum kernels, QCBM/QGAN generative models, and more.
- **Grammar.** [`GRAMMAR.md`](GRAMMAR.md) is the complete reference.
- **Backends / proofs / editors.** See the [README](README.md).
