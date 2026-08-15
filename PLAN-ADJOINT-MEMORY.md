<!-- SPDX-License-Identifier: Apache-2.0 -->
# PLAN — the CPU adjoint gradient's working set (request R8)

**Status: PLAN ONLY. Nothing implemented.** The cause below is read from the
code; the measurements quoted are the requester's, re-derived here against the
source rather than taken on trust.

Request: `fixes/ARIA-IMPROVEMENT-REQUESTS.md` §R8 — the CPU adjoint gradient
costs **9–28× its own forward pass**, where a textbook adjoint lands near 2×
and `lightning.qubit` measures 1.01×. The same algorithm on our own CUDA arm
costs 2–5×, which is the existence proof that this is implementation, not
method.

| lane | n=16, b=1 | n=16, b=64 | n=16, b=1024 |
|---|---|---|---|
| `lightning.qubit` (adjoint) | 1.01× | 1.01× | 1.04× |
| `aria:gpu` (adjoint, CUDA) | 1.07× | 1.98× | 5.04× |
| **`aria:sv`** (adjoint, CPU) | **9.35×** | **18.36×** | **16.66×** |

The requester also **refuted their own first hypothesis** with a committed
experiment (`r8_gradient_scaling.py`): holding `n` fixed while symbol count
moves 36 → 396 changes the overhead by 1.81×, not 16×, so it is *not*
per-symbol sweeps. Holding symbols roughly fixed while `n` grows is where it
lives. That narrowing is correct, and it points at exactly one line.

---

## 1. The cause, at a line number

`crates/omega-backend-statevector/src/adjoint.rs:118-128`:

```rust
let mut checkpoints: Vec<Vec<Complex64>> = Vec::with_capacity(unitary_ops.len() + 1);
checkpoints.push(state.clone());

for (_, op) in &unitary_ops {
    apply_gate_forward(&mut state, n, op, params)?;
    checkpoints.push(state.clone());        // one FULL statevector, per gate
}
```

The forward pass materialises **one complete statevector per unitary gate**.
Cost is `G · 2^n · 16 B`, and it tracks `2^n` exactly as the width test showed.

At n=16: `2^16 × 16 B = 1.05 MB` per state. The reported peak of **8.27 GB** is
≈ **7900 statevector-sized buffers** — the requester's own arithmetic, and it
lands on this line.

### Why it is flat in batch, which looked mysterious

`sim.rs:261-272`:

```rust
fn adjoint_gradient_batch(...) {
    bindings.par_iter().map(|b| self.adjoint_gradient(circuit, b, observable)).collect()
}
```

Each row builds **its own tape**, and rows run concurrently on rayon's pool. So
peak memory is `min(threads, rows) · G · 2^n` — bounded by the **thread count**,
not the batch size. That is precisely the reported signature: 8.27 GB at batch
64 and 8.29 GB at batch 1024. A per-row cost would have grown 16×; a
per-thread cost does not move.

**This is why the plan is sequenced against `PLAN-SV-PERF.md` S2.** Adding
amplitude-level MT does not change this, but any increase in *row* concurrency
multiplies an 8 GB working set. Fixing the tape first removes the coupling.

### The second allocator

`adjoint.rs:148`, inside the per-symbol loop:

```rust
let du_psi = apply_gate_derivative(psi_i, n, op, params, param_idx)?;   // -> Vec<Complex64>
```

`apply_gate_derivative` returns an owned `Vec` (signature at `adjoint.rs:271`),
so it allocates and frees a full state **per (gate, param slot, symbol)**. The
refuted per-symbol hypothesis was about *sweeps*; this is not a sweep, but it
is a per-symbol `2^n` allocation, and at 256 weight symbols it is the
allocator's problem rather than the arithmetic's.

---

## 2. The fix: the adjoint method does not need a tape

The textbook adjoint (Jones & Gacon; what `lightning.qubit` implements at
1.01×) keeps **two** state vectors, not `G`:

```
  |ψ⟩ ← the forward state, evolved ONCE to the end
  |λ⟩ ← O|ψ_n⟩

  for i = n .. 1:
      |ψ⟩ ← Uᵢ† |ψ⟩          ← recover ψ_{i-1} by UNDOING the gate
      grad += 2·Re(⟨λ| dUᵢ/dθ |ψ⟩)
      |λ⟩ ← Uᵢ† |λ⟩
```

Every gate here is unitary — the ops are already filtered by `is_unitary` at
`adjoint.rs:111-116` — so `Uᵢ†` exists and is norm-preserving, and the code
**already has the inverse application**: `apply_adjoint_gate`, used on `|λ⟩` at
line 157. The change is to apply it to `|ψ⟩` as well and delete the tape.

Working set: `2 · 2^n` instead of `G · 2^n`. At n=16 with ~660 gates that is
**8.27 GB → ~2 MB**, a factor of ~4000.

### What it costs

One extra `Uᵢ†` application per gate: the reverse sweep does two state updates
instead of one. That is a *constant* factor on arithmetic, in exchange for
removing an allocation that grows with `G`. Ratio predicted by the method:
around 2–3×, which is where every other lane in the table sits.

### What it changes numerically — and this is the part to gate

`ψ_{i-1}` recovered as `Uᵢ†ψᵢ` is **not bit-identical** to the stored
checkpoint. Unitaries are norm-preserving so the error does not amplify, but it
accumulates over the reverse sweep, and deep circuits are where it shows.

This is the same category as `PLAN-SV-PERF.md` S1b: it **changes results in the
last bits**, so it does not ride along with a bit-exact change and it needs its
own gate:

1. **Against the tape.** Keep the checkpointed implementation as the reference
   (as `PLAN-SV-PERF.md` S1 kept the scan loop) and assert agreement to a
   stated tolerance across a circuit set that includes a deep one — the depth
   test from `r8_gradient_scaling.py` is the right shape, since drift is what
   grows with depth and nothing else in the suite is long enough to show it.
2. **Against finite differences**, which is independent of both
   implementations and is the only check that cannot share their bugs.
3. **Against the CUDA arm**, which computes the same gradient by the same
   algorithm — but only on the Linux / RTX 6000 Pro box.

A tolerance chosen after seeing the failure is not a gate. Fix it up front from
the norm-preservation argument, and if the measurement exceeds it, that is a
finding rather than a reason to widen.

### Where the tape is genuinely needed

`Reset` and mid-circuit measurement are **not** invertible. They are already
excluded here (`is_unitary` filters them, `adjoint.rs:115`), so this plan's
scope is exactly the circuits the adjoint already accepts. If that filter is
ever relaxed, the tape has to come back for those segments — the standard
answer is checkpoint-and-replay at the irreversible boundaries, and it should
be written down before anyone widens `is_unitary`.

---

## 3. Steps

| step | what | gate |
|---|---|---|
| **A0** | Reproduce R8's ratio locally: `(fwd+bwd)/fwd` at n = 8, 10, 12, 14, 16, and peak RSS. The requester's numbers are from a different box; the *shape* is what must reproduce, not the constant. | numbers in this file |
| **A1** | Hoist `apply_gate_derivative`'s allocation out of the symbol loop — one scratch buffer per call site, reused. Pure allocation change, **bit-identical**, so it lands on its own and its own test says so. | bit-identity vs current |
| **A2** | Replace the tape with reverse evolution (§2). | the three checks above |
| **A3** | Re-measure A0's table; report the ratio and the peak RSS next to the request's numbers. | the table |
| **A4** | Only if a gap remains: profile the remaining `(fwd+bwd)/fwd`. Do not assume it is the same cause twice. | — |

A1 before A2 deliberately: A1 is bit-exact and small, so if A2's tolerance
discussion stalls, the allocation win is already banked.

---

## 4. What could make this pass for the wrong reason

- **Measuring `(fwd+bwd)/fwd` on a shallow circuit.** At n=8 the request
  already measures 1.57× — textbook. The defect only appears once `2^n` leaves
  cache, so a fixture that fits in L2 will report success at every stage of
  this plan.
- **Testing gradient agreement on a shallow circuit.** The reverse-evolution
  drift *accumulates with depth*: a 10-gate fixture cannot distinguish a
  correct reverse sweep from one that is quietly losing precision.
- **Finite differences with a badly chosen step.** `h` too small is
  cancellation, too large is truncation; the check must state its `h` and show
  the gradient is stable across at least two.
- **A gradient that is right because the state is symmetric.** Fixtures need
  parameters whose derivatives differ per symbol, or a permuted gradient vector
  compares equal.
- **Peak RSS measured on the batch path with one row.** The 8.27 GB signature
  needs concurrency to appear at all; measure at a batch that saturates the
  pool, and record the thread count next to the number.
- **Reading a speedup that is really the allocator warming up.** A1 and A2 both
  change allocation behaviour; report medians of repeated runs, not first runs.

---

## 5. Deliberately not done

- **Widening `is_unitary`** to cover `Reset` / mid-circuit measurement — see §2.
- **GPU.** The CUDA arm already sits at 2–5×; this is the CPU lane.
- **The `2^n` scaling of the forward pass itself.** That is
  `PLAN-SV-PERF.md`'s subject, and R8 is explicitly about the *ratio* within a
  lane, which is independent of it.
