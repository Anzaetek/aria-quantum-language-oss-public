<!-- SPDX-License-Identifier: Apache-2.0 -->
# Running Aria across two machines — a setup guide

For the common case: **your laptop is the caller, a bigger machine is the
executor.** You write and drive circuits locally; the heavy simulation happens
on a workstation, a GPU box, or a DGX.

Written for a person standing this up for the first time. `TESTING.md` has the
verification commands and `SCHEDULING.md` covers resource limits and large
batch workloads.

---

## 1. The mental model

The client lowers a circuit to a small JSON document and posts it. The server
simulates and returns **numbers** — counts, or expectation values. The
statevector itself never crosses the wire, which is what keeps this practical
over a home connection: a 30-qubit statevector is 16 GB, its `⟨Z⟩` is 8 bytes.

```
   laptop                                   big box
┌───────────┐   circuit JSON (KBs)     ┌──────────────┐
│ aria CLI  │ ───────────────────────▶ │ omega-server │
│ / library │ ◀─────────────────────── │  simulates   │
└───────────┘   counts or ⟨O⟩ (bytes)  └──────────────┘
```

One consequence worth internalising early: **the server is a compute service,
not a sandbox.** Anyone who can reach it with a valid token can make it allocate
memory and burn CPU. Section 3 is therefore the important part of this document.

---

## 2. Quick start (SSH tunnel — the recommended setup)

### On the big box

```console
$ cargo build --release -p omega-server
$ ./target/release/omega-server --save-token-to ~/omega-token &
```

`--save-token-to` writes the bootstrap token to a file with mode `0600` instead
of printing it to stdout, so it does not end up in your shell history or scroll
buffer. Check it is alive — `/health` needs no auth:

```console
$ curl -s http://localhost:8080/health
{"status":"ok","version":"0.1.0","execution":{...}}
```

### On the laptop

```console
$ ssh -N -L 8080:localhost:8080 bigbox &          # tunnel, encrypted by SSH
$ export OMEGA_SERVER=http://localhost:8080
$ export OMEGA_TOKEN=$(ssh bigbox cat omega-token)
```

### Run something

```console
$ cargo run -p aria-cli --features remote -- run examples/aria/qft.aria \
      --circuit QFT --int n=3 \
      --backend remote --url "$OMEGA_SERVER" --token "$OMEGA_TOKEN"
```

That's it. The tunnel gives you encryption and authentication for free, and the
server never listens on a public interface.

---

## 3. Security

### 3.1 The one thing you must not skip

**The server speaks plain, unencrypted HTTP by default.** Bearer tokens cross
the wire in cleartext. This is deliberate — the server does not try to be a TLS
terminator — but it means:

> **Never expose the server's port directly to an untrusted network.**

Two supported deployments:

| deployment | how | when |
|---|---|---|
| **SSH tunnel** (§2) | `ssh -N -L 8080:localhost:8080 bigbox` | Default choice. Nothing to configure, encryption and host auth handled by SSH. |
| **TLS reverse proxy** | nginx/caddy terminating TLS in front, server started with `--auth bearer-only` | Multiple users, or a service that must stay up without your SSH session. |

`--auth bearer-only` skips the WebSocket endpoint and server-certificate
generation — the right mode when a proxy already provides transport security.
The default `pqc` mode additionally exposes a post-quantum-encrypted WebSocket
at `/v1/ws` (note: the bundled CLI client does not speak it yet).

There is also an optional built-in TLS listener behind the `tls` build feature
(`OMEGA_TLS_ADDR`), if you would rather not run a proxy.

### 3.2 Tokens: the bootstrap token is not the one you hand out

On first start the server issues a **bootstrap admin token**. It has full
rights. Treat it like a root password: it is for setting up other tokens, not
for daily use, and not for pasting into a training script.

Its lifetime defaults to 24 hours (`OMEGA_BOOTSTRAP_TTL`, seconds).

Issue a **scoped, per-client token** instead:

```console
$ curl -s -X POST http://localhost:8080/v1/auth/token \
      -H "Authorization: Bearer $ADMIN_TOKEN" \
      -H 'Content-Type: application/json' \
      -d '{"sub":"laptop-training","rights":3,"ttl_seconds":86400}'
```

`rights` is a bitmask:

| right | value | grants |
|---|---|---|
| `READ` | 1 | read-only endpoints |
| `EXECUTE` | 2 | **running circuits** — the quantum routes |
| `WRITE` | 4 | registering circuits/functions |
| `ADMIN` | 8 | issuing and revoking tokens |

Useful combinations: `1` viewer, `3` operator (read + execute — what a training
client needs), `7` developer, `15` admin.

**Give each client the least it needs.** A QML training job needs `3`. It does
not need the ability to mint tokens.

Revoke a token by its `jti`:

```console
$ curl -X DELETE http://localhost:8080/v1/auth/token/<jti> \
      -H "Authorization: Bearer $ADMIN_TOKEN"
```

### 3.3 Handling token material

- Prefer `--save-token-to FILE` (mode `0600`) over letting the token print to
  stdout.
- Pass tokens by environment variable or file, **never as a command-line
  argument** — argv is visible to every user on the box via `ps`.
- Tokens are bearer credentials: whoever holds one *is* that client. There is no
  second factor.
- Set a TTL you are willing to live with. A long-running batch that outlives its
  token will fail mid-run on a `401`, and no amount of client retry fixes that.

### 3.4 Rate limits and abuse

Per-IP and per-subject request limits are enforced, returning `429` with a
`Retry-After` header:

```
OMEGA_RATELIMIT_PER_IP_PER_MIN
OMEGA_RATELIMIT_PER_SUBJECT_PER_MIN
```

These count **requests**, not cost — one request can be enormous. Resource
limits are a separate mechanism; see `SCHEDULING.md`.

### 3.5 What `/health` reveals

`/health` is unauthenticated by design (so a load balancer can poll it) and
reports the server's execution budget and current free capacity. On a shared or
multi-tenant box that is a mild side channel: polling it reveals when neighbours
start and finish large jobs. If that matters, keep `/health` behind your reverse
proxy and expose it only to your monitoring.

### 3.6 Other knobs worth knowing

| variable | purpose |
|---|---|
| `OMEGA_CORS_ALLOW_ORIGINS` | Browser origins allowed. Do not set a wildcard on a server reachable by others. |
| `OMEGA_PKI_TRUST_STORE`, `OMEGA_PKI_CRL_FILE`, `OMEGA_PKI_CLIENT_CERT_POLICY` | Client-certificate verification for the WebSocket path. |
| `OMEGA_BACKEND_DIR` | Directories scanned for backend plugins. **Opt-in**: no plugin is loaded unless you set this. Plugins are native code — only point this at libraries you trust. |
| `OMEGA_DB_PATH` | SQLite file (default `./omega.db`) holding tokens and registered circuits. Back it up, and protect it like a credential store. |

### 3.7 Security checklist

- [ ] Server is **not** reachable on a public interface without TLS in front
- [ ] Bootstrap token stored `0600`, not pasted into scripts
- [ ] Each client has its own scoped token with the minimum rights
- [ ] Token TTLs outlive your longest expected job, but not much more
- [ ] `OMEGA_BACKEND_DIR` unset unless you deliberately use plugins
- [ ] `omega.db` treated as sensitive
- [ ] Resource limits set for your box (`SCHEDULING.md`)

---

## 4. Sizing the server

Before submitting real work, ask the server what it will accept:

```console
$ curl -s http://localhost:8080/health | python3 -m json.tool
```

`execution.capacity_bytes`, `available_bytes` and `max_qubits` tell you the
budget. Override the defaults with `OMEGA_MAX_MEM` (bytes) and
`OMEGA_MAX_QUBITS`. Defaults derive from detected memory — and from the
container's limit rather than the host's, when running in one.

A circuit that cannot fit is refused **before** anything is allocated, with the
numbers in the error. Two refusals mean different things, and clients should
treat them differently:

- **`413`** — bigger than this host's entire budget. Retrying will never help.
- **`429`** — would fit, but not right now. Retry per `Retry-After`.

See `SCHEDULING.md` for the full picture, including how batch workloads behave.

---

## 5. Confirming it actually works

Run the golden example suite against the live server before trusting it with
real work:

```console
$ cargo run -p aria-verify --features remote -- socket \
      --url "$OMEGA_SERVER" --token "$OMEGA_TOKEN"
```

Same expected numbers as the in-process run. If this passes, your transport,
auth and backend dispatch are all sound.

---

## 6. When something goes wrong

| symptom | likely cause |
|---|---|
| `connection refused` | Tunnel not up, or server not running. Check `curl $OMEGA_SERVER/health`. |
| `401` / `403` | Token missing, expired, or lacking `EXECUTE`. Not retryable — issue a new token. |
| `413` | Circuit exceeds the host budget. Reduce qubits (each one **doubles** the memory), or raise `OMEGA_MAX_MEM` if the box genuinely has room. |
| `429` | Either the rate limiter or the resource governor. Honour `Retry-After`. |
| `400` | Malformed circuit or observable. The message names the offending part. |
| Hangs on a long circuit | Execution is currently synchronous — the HTTP request stays open for the whole run, so proxy and tunnel idle timeouts apply. Keep individual requests short. |

---

## 7. Current limits worth knowing up front

Honest list, so nothing surprises you mid-project:

- **Execution is synchronous.** No job IDs, no polling; a dropped connection
  loses the work in flight.
- **No gradient endpoint.** Remote is score-only today, so training loops run
  locally.
- **A batch fails wholesale** if any row is invalid.
- **`aria train` does not target a remote server**; `aria run` and expectation
  calls do.

`SCHEDULING.md` §3 lists these with workarounds, and `FIXES_PLAN.md` tracks the
work to remove them.
