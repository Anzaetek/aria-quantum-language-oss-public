<!-- SPDX-License-Identifier: Apache-2.0 -->
# Using Aria as a library (omega as a Rust crate)

Aria is a CLI *and* a set of Rust crates you can call directly. This page is
the entry-point reference for embedding an Aria circuit in your own Rust
program — evaluating it, differentiating it, and training its parameters — on
the pure-Rust default build (no libtorch, no GPU).

Every snippet below mirrors code that is compiled and tested in this repo:
the end-to-end flow is a **doctest** on `aria_runtime` (`cargo test -p
aria-runtime --doc`), the binding-order contract is pinned in
`aria-runtime/tests/symbol_order.rs`, and the training loop is exactly what
`crates/apps/qml-classifier` runs. If a snippet here drifts from the API, CI
fails.

## The one pattern to learn: lower once, bind many

```rust
use aria_core::ast::parse_aria;
use aria_runtime::lower::lower;
use omega_backend_statevector::StatevectorBackend;
use omega_core::executor::{Backend, Observable};
use omega_core::gradient::{compute_gradient_for, GradMethod};
use omega_core::params::ParameterBinding;

// 1. Parse + instantiate + lower ONCE.
let src = "circuit M() { qreg q[1]\n  let t = symbolic[1]\n  apply RY(t[0]) on q[0] }";
let circuit = parse_aria(src).unwrap().instantiate("M", &[]).unwrap();
let low = lower(&circuit).unwrap();          // low.ir + low.symbol_ids

// 2. Bind parameters by SymbolId (symbol_ids is the name → id map).
let t_id = low.symbol_ids["t_0"];
let mut binding = ParameterBinding::new();
binding.bind(t_id, std::f64::consts::FRAC_PI_3);

// 3. Expectation ⟨Z⟩ through the Backend trait.
let backend = StatevectorBackend::new();
let z = Observable::parse("Z0").unwrap();
let exp = backend.expectation(&low.ir, &binding, &z).unwrap();

// 4. Exact gradient ∂⟨Z⟩/∂t via adjoint AD — one call returns every symbol.
let grads = compute_gradient_for(
    &backend, &low.ir, &binding, &z, &GradMethod::Adjoint, None,
).unwrap();
```

**Lower once.** `lower()` is not free — do it a single time and reuse `low.ir`
across every evaluation, rebinding parameters. Do **not** call the
`aria_runtime::run::expectation(circuit, ..)` convenience wrapper in a hot loop:
it re-lowers on every call. The wrapper is for one-shot CLI-style use; the loop
above is for training/scoring.

## The SymbolId binding-order contract (stable)

`lower()` assigns `SymbolId`s in **first-appearance order** as it walks the
circuit, so they are dense (`0..symbol_ids.len()`). Two consequences you can
rely on:

- **Look symbols up by name** in `low.symbol_ids` — never assume an id from
  source order. `symbol_ids["theta_3"]` is the authoritative id.
- **Flat parameter slices are ordered by ascending SymbolId**: `params[id]` is
  the value for the symbol with that id. The wasm transport and the verify-core
  oracle both take parameters this way; a `ParameterBinding` built with
  `.bind(id, value)` is the typed equivalent.

This is a tested guarantee (`tests/symbol_order.rs`), not incidental behaviour.

## Gradients: prefer adjoint

`compute_gradient_for(backend, ir, binding, observable, method, only)` returns
`Vec<(SymbolId, f64)>`. `only: Some(&HashSet<SymbolId>)` restricts the result to
a subset — pass your trainable weights so data/feature symbols never receive a
spurious gradient.

| `GradMethod` | Cost | When |
|---|---|---|
| `Adjoint` | 1 forward + 1 backward sweep, **all** symbols, exact | **default choice** for measurement-free circuits |
| `ParameterShift` | 2–4 evaluations **per symbol**, exact | hardware-faithful; needed when adjoint is unavailable |
| `ParallelParameterShift` | 1 evaluation for a trailing commuting block | butterfly / RBS layers (`omega_core::parallel_shift`) |
| `FiniteDifference { epsilon }` | 2 evaluations per symbol, O(ε²) error | last resort / cross-check only |

`StatevectorBackend` also exposes `adjoint_gradient(ir, binding, obs) ->
Option<Vec<(SymbolId, f64)>>` directly (the trait method `compute_gradient_for`
calls under the hood). Reach for `compute_gradient_for` when you want method
selection and the `only` filter; call `adjoint_gradient` when you always want
adjoint.

## Training loop (MSE, exact gradients)

The full pattern — used verbatim by `crates/apps/qml-classifier`:

```rust
# use std::collections::{HashMap, HashSet};
# use omega_backend_statevector::StatevectorBackend;
# use omega_core::circuit::{CircuitIR, SymbolId};
# use omega_core::executor::{Backend, Observable};
# use omega_core::gradient::{compute_gradient_for, GradMethod};
# use omega_core::params::ParameterBinding;
# fn go(ir: &CircuitIR, x_id: SymbolId, weight_ids: &[SymbolId], train: &[(f64, f64)]) {
let backend = StatevectorBackend::new();
let z = Observable::parse("Z0").unwrap();
let trainable: HashSet<SymbolId> = weight_ids.iter().copied().collect();
let index: HashMap<SymbolId, usize> =
    weight_ids.iter().enumerate().map(|(k, &id)| (id, k)).collect();
let mut w = vec![0.1; weight_ids.len()];

for _ in 0..250 {
    let mut grad = vec![0.0; w.len()];
    for &(feature, y) in train {
        let mut b = ParameterBinding::new();
        b.bind(x_id, feature);
        for (k, &wid) in weight_ids.iter().enumerate() { b.bind(wid, w[k]); }

        let pred = backend.expectation(ir, &b, &z).unwrap();
        let residual = pred - y;                       // MSE: L = Σ(pred − y)²
        let grads = compute_gradient_for(
            &backend, ir, &b, &z, &GradMethod::Adjoint, Some(&trainable),
        ).unwrap();
        for (sym, g) in grads {                        // dL/dθ = 2·residual·∂⟨Z⟩/∂θ
            if let Some(&k) = index.get(&sym) { grad[k] += 2.0 * residual * g / train.len() as f64; }
        }
    }
    for k in 0..w.len() { w[k] -= 0.5 * grad[k]; }
}
# }
```

For a batched trainer with Adam, freeze masks, and MSE/BCE, see
`aria_runtime::train` (`train_expectation`, `TrainConfig`, `Optimizer`) and
`omega_core::qml::QmlTrainer`.

## Backends

All implement `omega_core::executor::Backend`, so the code above is
backend-agnostic — swap the constructor:

- **`StatevectorBackend::new()`** — pure-Rust dense simulator, exact, ≲24 qubits.
- **`MpsBackend::new(chi)`** — matrix-product-state with bond dimension `chi`.
  `chi = 2^(n/2)` reproduces the dense statevector exactly; smaller `chi`
  trades accuracy for scale. A clean way to *see* the truncation: run the same
  circuit at `chi = 1` (product-state, diverges on entangled circuits) and
  `chi = 2^(n/2)` (matches `StatevectorBackend` to ~1e-14). From the CLI:
  `--backend mps` (χ = 64) or `--backend mps:128`.
- GPU (`--features metal|cuda|opencl`) and libtorch (`--features tch`) backends
  exist behind feature flags; the default build needs none of them.

## Observables

`Observable::parse("0.5*X0 + Z1 Z2")` builds a weighted Pauli sum;
`Observable { terms: vec![(coeff, vec![(qubit, PauliOp::Z)]) , ..] }` builds one
by hand. Two batch reads amortise work across a whole dataset:

- `backend.expectation_multi(ir, binding, &[obs0, obs1, ..])` — several
  observables against **one** forward sweep (⟨Z_q⟩ on many qubits per point).
- `backend.expectation_batch(ir, &[&binding0, &binding1, ..], obs)` — one
  observable against **many bindings** (one per data row), run in parallel on
  the statevector backend. `adjoint_gradient_batch` is the gradient
  counterpart — the throughput lever a training loop leans on. Both are
  index-preserving, so seeded runs stay bit-identical to the sequential loop.

## Loading circuits

- In-repo examples: `aria_verify_core::harness::load_lowered("file.aria",
  "Circuit", &[("n", 4)])` resolves under `examples/aria/`.
- Your own files: `load_lowered_path(path, "Circuit", &ints)` takes an absolute
  or working-directory-relative path — no repo layout assumed.
- Or embed the source and go through `parse_aria` directly, as in the top
  snippet (`include_str!("my_circuit.aria")`).

## Exporting a trained model

`aria_core::ast::qasm::to_qasm(&circuit)` returns OPENQASM 2.0 for a circuit
with concrete parameters — bind your trained weights back into the AST and
export, and "the trained model is a circuit you can read" holds literally.

## See also

- `llms.txt` (repo root) — one-line index of every crate's public entry points.
- `TUTORIAL.md` — the language and CLI.
- `crates/apps/qml-classifier` — the smallest end-to-end trained example.
- `crates/apps/butterfly-qnn`, `crates/apps/spectra` — larger QML harnesses.
