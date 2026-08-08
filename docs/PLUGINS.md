<!-- SPDX-License-Identifier: Apache-2.0 -->
# Writing a backend plugin

A backend plugin is a **native shared library** (`.so` / `.dylib` / `.dll`) that
omega loads at runtime and executes circuits through. It needs no changes to
this repository, and it does not have to be written in Rust — the interface is
a plain C ABI.

There is a **working reference implementation** in
`crates/omega-backend-refplugin` (381 lines), and CI exercises it on every run:
stage 6b builds it, `dlopen`s it through `omega-run`, executes a Bell circuit
end to end, and runs it against the conformance corpus. So the path is not
theoretical — it is tested continuously.

## The contract

Export exactly two symbols:

```c
uint32_t      omega_backend_abi_version(void);   // must equal OMEGA_BACKEND_ABI_VERSION
BackendVTable omega_backend_init(void);          // returns the function table
```

The loader checks the version **first** and refuses a mismatch with a clear
error rather than calling into a library that may have a different struct
layout. An unversioned library is rejected too — silently loading one is how a
plugin ABI turns into undefined behaviour.

The vtable supplies:

| entry | purpose |
|---|---|
| `backend_name` | identifier used for `--backend <name>` |
| `supports` | which circuit types this backend accepts |
| `backend_caps` | declared capabilities |
| `execute` | run a circuit, return counts or a statevector |
| `free_result` | release whatever `execute` allocated |

**Ownership is the part to get right.** Whatever `execute` allocates,
`free_result` must release, and nothing else may free it. This is the one rule a
Rust implementation cannot enforce on a plugin's behalf, so it is stated here
rather than left implicit.

## Start from the reference

```console
$ cp -r crates/omega-backend-refplugin crates/my-backend
```

Its dependency list is **one crate** (`omega-core`), and its `Cargo.toml` sets:

```toml
[lib]
crate-type = ["cdylib", "rlib"]
```

`cdylib` produces the loadable object; `rlib` lets the in-tree tests link it
directly, which is worth keeping — it is how a plugin gets unit-tested without
going through `dlopen`.

## Loading it

Plugin loading is **opt-in**. Nothing is loaded unless an operator sets
`OMEGA_BACKEND_DIR` (`:`-separated), because a plugin is native code running in
the server's process — see `REMOTE_SETUP.md` §3.6.

```console
$ export OMEGA_BACKEND_DIR=/path/to/plugins
$ omega-run circuit.qasm --backend my-backend --shots 512
```

Copy **only** the shared library into that directory. Cargo's `.d` dep-info file
shares the basename and will fail to `dlopen`.

## Prove it works: the conformance kit

```console
$ cargo run -p omega-plugin-conformance -- /path/to/libmy_backend.dylib
```

Runs a corpus (bell / ghz3 / uniform / rotation) against the statevector oracle
and exits non-zero if any case falls outside tolerance. **Run this before
trusting a plugin with real work** — "it loaded and returned numbers" is not the
same as "the numbers are right", and this project has repeatedly found the gap
between those two to be where defects live.

## Known limits of the plugin surface

Stated so they are not discovered mid-integration:

* **Gate-only.** The vtable executes circuits and returns counts or a
  statevector. There is **no expectation-value entry point**, so a plugin cannot
  serve the `/v1/quantum/expectation` path — that dispatch refuses plugins
  explicitly rather than silently producing something.
* **Bounded arity.** The flattened gate representation caps qubits per gate, and
  `GateKind::Custom` is refused at flatten time.
* **Thread-safety is the plugin's responsibility.** The registry is shared
  across request handlers and plugin execution is not serialised.
* **Not priced by the resource governor.** A plugin's memory profile is opaque,
  so admission treats it as unpriceable and governs it by the qubit ceiling
  alone (`SCHEDULING.md`). A plugin that allocates far beyond its declared width
  can still exhaust the host.

For simulators that need an expectation path or richer semantics, the
subprocess bridge (`docs/BRIDGES.md`) carries different trade-offs — slower per
call, but no ABI coupling and no in-process crash risk.
