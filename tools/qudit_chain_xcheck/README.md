<!-- SPDX-License-Identifier: Apache-2.0 -->
# Q4 — the qudit MPS measured, and MQT Qudits on the same circuits

PLAN-QUDIT.md **Q4**. Two legs, one circuit family.

**Leg 1 — Rust, `crates/omega-backend-mps/examples/qudit_chain_profile.rs`.**
Qutrit brickwork chains (random `rxy` level rotations, `csum` bricks, one
distant `csum` per layer — the family `tests/qudit_backend.rs` certified),
measured **through `MpsBackend::expectation_site_operators`**: evolution plus
an `O(χ²)` contraction of `S_z = diag(1, 0, −1)` on the middle site. Never
`execute(shots: None)`, which ends in a `3^n` dense `to_statevector` that is
not MPS evolution (STATUS §5.16 is the profile that made that mistake). Each
row: wall, process peak RSS (`getrusage`, cumulative down the table), the
certificate (`discarded_weight`, `fidelity_estimate`, `max_bond_reached`), the
certified value or a **REFUSED** cell quoting the certificate that refused it,
the exact value where `quditsv` can hold the state, and — for refused rows —
the value the engine would have returned with its ceiling lifted, labelled
uncertified. `--json` writes every row plus every circuit as a neutral gate
list.

```
cargo run --release -p omega-backend-mps --example qudit_chain_profile -- \
    small|large [--d 3] [--seed 20260930] [--json PATH]
```

A10 gates, in the harness itself: `--d 2` exits 2 (a qubit table must not be
published under a qudit heading); a row with no certificate from any run of
its circuit exits 3. The second gate fired on the first real run — the
engine's early abort returns before `record_stats`, so a refused attempt has
no stats — and the harness now takes a refused row's certificate from a
second, complete, ceiling-lifted run, which is also where the uncertified
value comes from. The refusing attempt's own certificate (at the point it
stopped) is quoted in the REFUSED cell from the engine's sentence.

**Leg 2 — Python, `tnsim_rows.py`.** Builds the identical circuits in
mqt.qudits **through its API** (`QuantumCircuit.r/csum`; `from_qasm` drops
`rxy` parameters — Q0 defect D2), runs the `tnsim` tensor-network backend in
a subprocess under an address-space limit and a timeout, and computes the
same `⟨S_z⟩` from its statevector (big-endian: site `w` has stride
`d^(n−1−w)`). It **fails** if tnsim disagrees with `quditsv` at 1e-9 or with
any certified MPS row at 1e-8; refused MPS rows are printed with tnsim's
value beside the uncertified one, reported not asserted. A tnsim run that
hits the limit or the timeout is published as `tnsim: REFUSED`.

```
<python-with-mqt.qudits> tools/qudit_chain_xcheck/tnsim_rows.py ROWS.json \
    [--as-limit-gib 32] [--timeout-s 900]
```

`ci.sh` runs `small` on every box and the tnsim leg where the venv exists
(`make -C crates/omega-bridges/python mqt-venv`), registered skip otherwise.
`large` is the 123 GB rows and is run by hand on akilles under
`omega-hostgate`; the results, with their box, are in STATUS §4j.
