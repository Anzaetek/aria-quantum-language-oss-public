<!-- SPDX-License-Identifier: Apache-2.0 -->
# OpenQASM specification examples — vendored

## Source

| | |
|---|---|
| Repository | `https://github.com/openqasm/openqasm` |
| Path | `examples/` |
| Commit | `20c1d9225ee5872909b05e898334b2130ce61b78` |
| Fetched | 2026-08-17 |
| Licence | Apache-2.0 — `Copyright 2017-2025, OpenQASM Contributors.` (`LICENSE`, verbatim copy) |

21 programs plus `stdgates.inc`. `stdgates.inc` is an **include, not a program**:
it is vendored so the programs resolve and is not a conformance row.

## Rules

**These files are byte-exact and must stay that way.** `SHA256SUMS` pins them;
verify with `shasum -a 256 -c SHA256SUMS` from this directory. Their entire value
is that we did not write them — a corpus edited to suit the reader tests the
reader against itself, which is the failure mode
`emitters_refuse_rather_than_substitute.rs` was built to avoid.

If a file needs altering to be usable, add a **separate, clearly-named derived
copy** and leave the original untouched.

Fetch with `curl`. Do not route a file through a summariser: it would no longer
be the upstream text.

## Why they are here

To answer "what fraction of the OpenQASM 3 language do we actually read?" against
the specification's own corpus rather than against files we authored. Our QASM3
export check (`tools/qiskit_xcheck/qasm3_dialect.py`) validates what we *write*;
nothing validated what we can *read*, and the read side had never met a file
written by anyone else.

The intended gate is **not** "N of 21 parse". It is that every one of the 21 is
accounted for as either **parsed** or **refused with a message naming the
unsupported construct**. A silent partial parse, a panic, or a generic parse
error is a failure. Counting only the files we can do is the same
compared-zero-cells trap this repository has found repeatedly.

## Classification, measured 2026-08-17

Each file was scanned for constructs outside the gate-model profile. **Tier 1**
is reachable; **Tier 2** needs classical compute, timing or pulse control in the
circuit and is expected to be *refused by name*.

### Tier 1 — five files, and **all five now parse**

Re-measured **2026-08-18**: `every_official_example_is_parsed_or_refused_by_name`
in `crates/omega-parser/tests/openqasm_conformance.rs` lists all five as
`Parses`, and the suite is green. The table below is kept as the 2026-08-17
record of what blocked each one, because the *reason* each was blocked is what
the grammar work had to answer.

| file | blocked on 2026-08-17 by | status 2026-08-18 |
|---|---|---|
| `rb.qasm` | the mandatory `OPENQASM` header only — everything else in it already lowers | parses |
| `qft.qasm` | header; `cphase` spelling | parses |
| `teleport.qasm` | `OPENQASM 3;` (no minor version); scalar `bit c0;`; braced `if (c) { … }` | parses |
| `inverseqft2.qasm` | header; single-qubit broadcast `h q;` | parses |
| `qpt.qasm` | header; scalar `qubit q;` / `bit c;`; single-qubit broadcast | parses |

What closed them: the header became optional (`program = { SOI ~ header? ~ …
}`) and accepts a bare major version, `qubit`/`bit` declarations became valid
without a size, the guard gained its braced form, and register broadcast landed
in `omega-parser`'s lowering.

**The superseded claim, left visible rather than deleted.** This section used to
read: *"20 of the 21 have no version statement at all … So today zero files
reach lowering, and every ledger row would be an identical `SOI` error — which
is why the grammar work has to land before the ledger is worth running."* That
was true when written and is now false in its headline number. It is quoted here
because a reader who saw the old text elsewhere needs to know it was superseded
by measurement rather than by opinion, and because "zero files reach lowering"
is exactly the kind of figure that gets carried into a summary and outlives the
condition that produced it.

**"Parses" here means the `omega-parser` lane** — the one `omega run`, the C
FFI, the WASM host ABI and the HTTP server use. There is a *second* reader,
`aria_core::ast::qasm::from_qasm`, behind `aria import`, and the two do not
accept the same set.

Register broadcast is no longer the difference: both readers now expand `h q;`,
`measure q -> c;` and `reset q;`, and `readers_agree_on_register_broadcast.rs`
pins them together on the legal forms *and* on the meaningless ones.

Measured 2026-08-18, `aria import` still refuses both of these files, but for
constructs that have nothing to do with broadcast:

| file | why `aria import` refuses it |
|---|---|
| `inverseqft2.qasm` | `unparsed statement at line 4: 'qubit[4] q;'` — OpenQASM **3** register syntax |
| `qpt.qasm` | ``gate pre` at line 3 is defined with a body this importer does not implement` |

That is a fair refusal rather than a defect: `from_qasm` is documented as an
OpenQASM **2.0** importer, and these are 3.x programs. It is recorded here only
so the earlier "blocked by single-qubit broadcast" reading is not carried
forward — that half is fixed, and what remains is a different and much narrower
gap.

### Tier 2 — sixteen files

`adder` (`for`×3, `uint[4]` decls, `bool()` cast, register slices
`measure b[0:3] -> ans[0:3]`), `gateteleport` (`const`, `extern`, `def`),
`msd` (6 `def`s, `while`, `let` alias, casts), `qec` (`def`, and a cast *inside*
the guard: `if(int[2](syn)==1)`), `inverseqft1` (`int[4](c)` cast in the guard),
`cphase` (Unicode identifiers `θ`/`π`, and it declares **no registers** — a
gate-definition snippet rather than a program), `ipe`, `rus` (`while`),
`varteleport`, `arrays`, `scqec`, `t1`, `dd`, `alignment` (`stretch` /
`durationof`), `defcal` (pulse), `vqe` (`def`, `extern`, `const`, `for`).

5 + 16 = 21. The arithmetic closing is part of the point: an earlier draft of
this classification put 10 files in Tier 1, double-counted `qec`, and did not
sum to 21 — because four entries were assigned from a *summary* of the file
rather than the file.
