<!-- SPDX-License-Identifier: Apache-2.0 -->
# The host gate protocol, v1

One resource budget per machine, shared by processes that do not know about each
other and are not written in the same language.

This document is the **contract**, not a description of the Rust implementation.
Anything that follows it interoperates; anything that reads `gate.rs` and copies
its structure instead may not. The reference implementation is
`crates/omega-hostgate`; a second implementation is expected in Python, and the
two are held together by §9 rather than by inspection.

Formal model: `proofs/tla/HostGate.tla`, checked by `proofs/tla/check_hostgate.py`.
Where this document and the model disagree, the model is right.

---

## 1. What it is for

Three programs on one box, each with a correct internal budget, will together
overcommit the machine — because each is blind to the other two. This protocol
gives them one shared ledger so that "will this fit?" is asked of the *machine*
rather than of one process's opinion of it.

**Threat model: cooperative processes on one box.** Every participant is assumed
to be trying to co-operate. There is no defence against a hostile local user, and
none is attempted. This is accounting between programs that want to co-exist.

**It is off unless configured.** An implementation MUST do nothing at all — no
file, no lock, no syscall — when `OMEGA_HOSTGATE` is unset. A resource manager
that changes behaviour merely by being linked cannot be adopted incrementally.

---

## 2. Files

Given `OMEGA_HOSTGATE=/path/to/gate.json`:

| file | role |
|---|---|
| `/path/to/gate.json` | the ledger |
| `/path/to/gate.lock` | the lock. Created once, **never renamed, never unlinked, never written to** |

**The lock MUST be a separate file.** `flock` attaches to an open file
description, not to a path. Because §4 requires the ledger to be replaced by
`rename`, a lock taken on the ledger itself would leave the next process locking
a *different inode* — two processes each holding "the" exclusive lock, on two
different files, with no error anywhere.

For the same reason: a corrupt ledger MUST be left where it is. Moving it aside
splits the lock domain exactly the same way.

---

## 3. The ledger

```json
{
  "schema": "omega-hostgate-1",
  "instance": "<opaque, stable for this file's lifetime>",
  "caps": { "host_bytes": 68719476736, "slots": 16 },
  "holders": [
    { "pid": 4123,
      "start": 90218374,
      "tag": "qas/ring n10",
      "amounts": { "host_bytes": 29125115904 } }
  ]
}
```

* **Integers only.** No floats anywhere, including in `caps` and `amounts`. Two
  languages sharing a file must agree byte-for-byte on how a number is written,
  and float formatting is where that agreement fails; this estate has already
  paid for that once.
* **Axis names are strings**: `host_bytes`, `slots`, `device_bytes[N]`,
  `vendor[NAME]`. An implementation MUST NOT require an enum — an axis it does
  not recognise is carried through untouched, not rejected.
* **Unknown fields MUST be preserved and ignored**, at both levels. Three
  repositories deploy against one ledger on their own schedules, so version skew
  is the normal state. A reader that refused unknown fields would turn a routine
  upgrade of one consumer into an outage for the other two; one that dropped them
  would corrupt a newer writer's state.
* **`schema` MUST be checked.** A value this implementation does not speak means
  it withdraws — refuses its own participation, loudly, and leaves the file
  **untouched**. It does not rewrite a format it may not understand.

---

## 4. Writing

Every write MUST be atomic:

1. write a temp file **in the same directory**;
2. `fsync` it;
3. `rename` it over the ledger;
4. `fsync` the directory (best effort).

**The naive `seek(0); truncate; write` is forbidden.** It has a window in which
the file is zero-length, and a process dying in that window is this protocol's
premise, not an edge case. The model reaches that state in two steps and it
violates `AccountingCoversReality`: every holder vanishes from the accounting
while every holder's memory is still resident, so the next request is admitted on
top of all of it.

**An empty or unparseable ledger is an ERROR, never an empty budget.** This is
the single most important rule in the document. "I cannot read the budget" must
never resolve to "there is no budget".

---

## 5. Identity

A holder is keyed by **`(pid, start_time)`**, never by `pid` alone.

`start_time` is **opaque bytes compared only for equality** — never ordered,
never treated as a clock, never used for arithmetic. Crucially, it is compared
**between records written by different implementations sharing one ledger**, so
it is not enough for each implementation to pick something locally sufficient.
**The source is canonical per platform**, and an implementation that picks a
different one is not conformant even if its own tests pass:

| platform | `start_time` | liveness |
|---|---|---|
| Linux | field 22 of `/proc/<pid>/stat`, **as the integer it is** — no hashing. It is **index 19 after splitting on the LAST `)`**, because `comm` is unquoted and may contain spaces and parens | `/proc/<pid>` exists and its state field is not `Z` |
| macOS | FNV-1a-64 of the trimmed `ps -o lstart=` string for that PID (§5.3) | `ps -o pid=,stat=`; state not starting with `Z` |

**Why canonicalisation and not "any sufficient value".** An earlier draft of this
section offered macOS two sources — `proc_pidinfo(PROC_PIDTBSDINFO).pbi_start_tvsec`
or a hash of `lstart` — and called either acceptable. They produce different
values for the same process. Two implementations that each passed §9 and then
shared a ledger would find *every* foreign record's identity mismatched,
conclude every holder was a recycled PID, and reclaim all of them. That is the
mass-erasure failure this field exists to prevent, arrived at through the front
door by two correct-looking implementations.

### 5.1 What a probe means

Three outcomes, and **both wrong branches are fatal in opposite directions**:

| probe result | means | why |
|---|---|---|
| ran, PID present, state not `Z` | **alive** | — |
| ran, PID present, state `Z` | **dead** | the table entry survives, the memory does not |
| **ran successfully, PID absent** | **DEAD** | this is determinate evidence, and it is the entire basis of reclamation |
| **failed to run, or was refused** | **ALIVE** | `ps` that will not execute, an unreadable `/proc`, a `hidepid` mount, another user's process, an unrecognised platform |

The two failure modes are one wrong branch on the same conditional:

* Reading an **indeterminate** probe as *dead* erases live holders' grants. This
  is not hypothetical — both gates previously deployed in this estate did it on
  macOS, granting every request instantly while deleting other processes'
  records.
* Reading a **determinate absence** as *indeterminate* never reclaims anything,
  producing a gate that only ever fills until it refuses everything.

An earlier draft armed only the first of those. Say both, in the implementation
and in the tests.

### 5.2 What uniqueness actually holds

The pair `(pid, start_time)` is unique **within one boot, except when a PID is
recycled inside a single second on macOS.**

That caveat is not theoretical tidiness. `lstart` and `pbi_start_tvsec` are both
second-resolution, and measured on one developer Mac: **105 distinct `lstart`
values were shared by more than one live process, the worst by 51 of them** —
the boot-time burst. So uniqueness rests entirely on the PID half of the pair,
and the pair collides only if the kernel reissues a PID within the same second
the previous holder started. The PID space is ~100k wide, so this is rare rather
than impossible.

Linux is the stronger case: field 22 is clock ticks, so its resolution is finer
than a second by two orders of magnitude.

An implementation MUST NOT rely on `start_time` alone being unique, and MUST NOT
use it for anything but equality against a recorded value.

### 5.2b Batching the macOS probe: a silence that is not a census

A scan runs at the head of every transaction, so the probe must not cost one
fork per holder — the gate would then scale with the workers it exists to
govern. One `ps -o pid=,stat= -p <list>` for the whole ledger is therefore the
right shape, and it has one trap that no manual page mentions.

**`ps -p` validates its entire argument list before printing anything.** Give it
one PID it will not accept and it exits non-zero having printed *nothing*, with
the complaint on stderr. A process API returns success for that — the command
ran. So an empty listing has two meanings that are indistinguishable by exit
path alone: *nobody you asked about is alive*, and *I refused to answer*.

Reading the second as the first marks **every** holder dead in one scan.

Measured on macOS 15, because the behaviour is not documented:

```text
ps -p <live>,<reaped>       rc=0  "<live> Ss"     an absent PID is simply omitted
ps -p <zombie>,<reaped>     rc=0  "<zombie> Z"    zombies are still reported
ps -p <reaped>              rc=1  ""              nothing found — and that IS the answer
ps -p <live>,999999         rc=1  ""              stderr: "process id too large"
```

So a merely **absent** PID does not poison the batch; only an **unacceptable**
one does, and **stderr is what tells them apart**. An implementation MUST:

* treat a non-empty stderr as "the listing is a refusal, not a census";
* not simply return everything as indeterminate in that case either — one
  corrupt record would then blind every future scan and the budget would fill
  until it refused everything, which is §5.1's other failure mode;
* make progress by falling back to per-PID probes, where an unacceptable PID
  can only spoil its own answer. A PID `ps` will not accept cannot name a live
  process on this machine, and §5.2's boot stamp guarantees the ledger belongs
  to this machine, so it is reclaimed.

Establishing existence first by other means and asking `ps` only about the
survivors is an equally valid shape.

### 5.3 The macOS start tag, exactly

So that two implementations produce identical bytes:

1. take the trimmed output of `ps -o lstart= -p <pid>` for that PID;
2. FNV-1a, 64-bit: offset basis `0xcbf29ce484222325`, prime `0x100000001b3`,
   XOR each byte then multiply, wrapping;
3. if the result is `0`, use `1` — `0` is reserved for "no start time recorded",
   which is treated as unverifiable rather than as a match.

### 5.2 The boot stamp closes what start_time does not

`(pid, start_time)` distinguishes a recycled PID **within one boot**. On Linux it
does **not** distinguish across boots: field 22 counts clock ticks since boot, so
a process started five seconds into one boot and another started five seconds
into the next read the same number, and early-boot daemons draw low, repeatable
PIDs besides.

So the ledger's `instance` field MUST carry a boot identity, and a ledger whose
`instance` differs from the current boot's MUST have **all** its holders dropped.
That is safe precisely because the file parsed correctly and said so itself — it
is not the §4 "unreadable means empty" mistake.

Sources: `/proc/sys/kernel/random/boot_id` on Linux, `kern.bootsessionuuid` on
macOS. **Never a clock**: macOS's `kern.boottime` is recomputed as
now-minus-uptime and moves on every NTP step and sleep/wake, which would
invalidate every live grant on a laptop several times a day.

**On macOS `kern.bootsessionuuid` is the ONLY source, not the preferred one.**
None of the per-boot directories below exists on Darwin, so there is no fallback
there: failing to read it means refusing to start. Do not assume symmetry
between the platforms here.

An implementation that cannot identify the boot MUST NOT simply carry on. Two
mechanisms protect a ledger from outliving a reboot — a boot stamp, or a
filesystem that is itself cleared per boot — and with **neither** in force the
original aliasing bug returns by another route, silently.

So: if the boot cannot be identified **and** the ledger path is not under
`/run`, `/var/run`, `$XDG_RUNTIME_DIR` or `/dev/shm`, the implementation MUST
refuse to start, naming both facts and how to fix it. `/tmp` does not count — it
is cleared per boot on many Linux distributions and on no macOS one, and a
protection that depends on which system you happen to be running is not a
protection.

When the boot cannot be identified but the path IS per-boot, the gate runs
correctly on the weaker mechanism and MUST surface that — `status` says so in a
line of its own. An operator should be able to see a degraded gate rather than
infer it from the platform.

Keying by PID alone is not a small simplification. The model shows it losing
capacity **permanently**: a recycled PID makes a dead holder's record look live,
the prune never reclaims it, and that budget is gone until reboot.

### 5.1 "Cannot determine" means ALIVE

An implementation MUST treat an indeterminate probe — an unreadable `/proc`
entry, a `hidepid` mount, another user's process, a `ps` that fails to run, an
unrecognised platform — as **alive**.

Pruning a live holder does not merely lose accounting: it hands that holder's
memory to the next request, on a box already at its limit. The cost of being
wrong in the safe direction is a budget that is too small for one more scan.

**Zombies are dead.** The process-table entry survives but the memory does not,
so a zombie's tokens MUST be reclaimed.

---

## 6. The prune, which is the whole failsafe

At the head of **every** transaction, before anything else, drop every record
whose `(pid, start)` is no longer alive per §5.

There is no crash handler and there cannot be one: a `SIGKILL`ed or OOM-killed
process runs no destructor, so recovery is never something the dying process
does. It is something every *other* process does on its behalf, every time it
takes the lock. The budget repairs itself as a side effect of being used, and
nothing anywhere detects a death.

---

## 7. Acquire and release

**Acquire** is non-blocking. It fits now or it is refused now; a caller that
wants to wait retries. This protocol has **no queue** — a gate that queues on the
caller's behalf turns an admission problem into an unbounded-queue problem, and
the queue would have to live in the file, which is the worst available place for
one.

Three outcomes, never a boolean:

| outcome | meaning | HTTP analogue |
|---|---|---|
| **granted** | charged, and held until released | — |
| **`TooLarge`** | larger than the whole cap for some axis. Waiting cannot help | `413`, never retried |
| **`Busy`** | fits in an empty budget, not in the current one | `429` + `Retry-After` |
| **`Unavailable`** | the gate could not answer | fail closed in `enforce` |

Ordering is mandatory: **every axis's cap is checked before any axis's
headroom.** Otherwise a request that can never fit is told to come back later,
which is the difference between a retry loop that terminates and one that does
not.

A multi-axis request is **all-or-nothing**. A refused request MUST charge
nothing on any axis.

Refusals MUST carry the numbers — what was needed, and what the limit was. A
caller told only "refused" cannot decide whether to retry smaller or give up.

**Release** subtracts exactly what was taken, saturating at zero. An over-release
MUST NOT make a total go negative or otherwise re-inflate the budget: in a shared
file that hands capacity to everyone at once.

A process releases **the grant it took**, identified in its own memory — never
"whatever record carries my PID". A restarted process has no grant, whatever the
file says.

---

## 8. Transfer

A wrapper that acquires on behalf of a child (`omega-hostgate run -- CMD`) MUST
re-key the record to the **child's** `(pid, start)` once the child exists.

Holding it in the wrapper is the tempting simplification and it reintroduces the
failure this protocol exists to prevent: kill the wrapper, the prune reclaims its
tokens, and the orphaned child keeps running and keeps its memory — so the gate
admits fresh work on top of it.

Transferring to a process that is **already gone** MUST release rather than
transfer. There is no identity left to charge, and parking a charge on a PID that
will never be pruned is a permanent leak.

---

## 8b. Prefer exec over reimplementation

A second implementation is a liability before it is a feature. Two artifacts
that must agree, kept in agreement by discipline rather than by construction,
diverge — and they diverge quietly, in the branch nobody exercises. The existing
Rust/Python gate pair in this estate proves it: one had a `.corrupt` filename the
other did not, and both independently treated an unreadable liveness probe as
"dead", which erased live holders' grants on macOS for months.

So a non-Rust consumer SHOULD reach the budget through
`omega-hostgate run --host-bytes SIZE --slots N -- <command>` rather than by
writing a twin. Python workers across several virtualenvs — the case that most
needs throttling and least wants a dependency — get the whole protocol by exec,
and there is exactly one implementation of the liveness probe, the atomic write
and the prune to get right.

§9 remains for anyone who genuinely must implement it — a language with no
process spawn, or a hot path where a fork per acquire is too expensive. It is a
higher bar than it looks.

## 9. Conformance

An implementation is conformant when it passes all of:

1. **Round-trip.** Write a ledger, have the other implementation read it, and
   agree on every cap, holder and amount.
2. **Unknown fields survive.** Add a field this implementation does not know, at
   both levels; rewrite through it; the field is still there and unchanged.
3. **Empty file refuses.** A zero-length ledger produces an error, not an empty
   budget.
4. **Atomicity.** No code path truncates the ledger in place. (Inspect; this one
   cannot be tested from outside without killing a process mid-write.)
5. **Identity.** A record whose `start` does not match the live process with that
   PID is reclaimed.
5b. **Boot.** A ledger stamped with a different boot drops every holder.
5c. **Determinate absence reclaims.** A probe that runs and does not list the PID
   marks it dead; a probe that fails to run marks it alive. Test both branches —
   arming only one of them is how both gates in this estate went wrong, in
   opposite directions.
5d. **Identical start tags.** Given the same live process, two implementations
   produce the same `start_time` bytes (§5.3). A ledger written by one and read
   by the other must not show every holder as a recycled PID.
6. **Unknown liveness holds.** A record whose liveness cannot be determined is
   NOT reclaimed.
7. **Cap before headroom.** An over-cap request returns `TooLarge` even when the
   budget is also busy.
8. **All-or-nothing.** A two-axis request that fails on the second axis charges
   nothing on the first.
9. **The cross-language hammer.** N processes of *both* implementations against
   one ledger, acquiring and releasing at random, with the invariant
   `sum(holders) <= cap` checked continuously and never violated.

§9 is the one that matters. The others can pass in two implementations that
still disagree; the hammer is what catches the disagreement.

---

## 10. Configuration

| variable | meaning |
|---|---|
| `OMEGA_HOSTGATE` | path to the ledger. **Unset means do nothing.** |
| `OMEGA_HOSTGATE_MODE` | `off` \| `advisory` \| `enforce` (default `enforce` once a path is set) |
| `OMEGA_HOSTGATE_PROFILE` | `gentle` (0.25) \| `balanced` (0.50) \| `greedy` (0.90) |
| `OMEGA_HOSTGATE_MAX_MEM` | absolute, `48G` style. Beats the fraction |
| `OMEGA_HOSTGATE_MEM_FRACTION` | `0.5` or `50%` |
| `OMEGA_HOSTGATE_SLOTS` | absolute core count. Beats the fraction |
| `OMEGA_HOSTGATE_CPU_FRACTION` | `0.5` or `50%` |

Rules, taken from `omega-server`'s `limits.rs` so an operator learns one set:

* **absolute beats fraction**, and the resolution is reported;
* **caps compose by `min`, never `max`** — including against what the machine
  actually has, so a cap larger than the box is clamped rather than honoured;
* **a malformed value is an error, never a fallback.** A throttle that quietly
  turns itself off looks exactly like one that is working;
* **in a container the budget is the container's** — read `memory.max` (v2) then
  `memory.limit_in_bytes` (v1), and take the smaller of it and the host.
  **Read the cgroup you are actually in.** Consult `/proc/self/cgroup` first: with
  a cgroup namespace it reads `0::/` and `/sys/fs/cgroup/memory.max` is your own
  limit, but without one that same path is the HOST's root cgroup — the read
  succeeds, returns a plausible answer, and is wrong in the direction that
  removes the ceiling entirely. When the path names anything but the root, read
  the limit at that path instead.
  And read the **container's** cgroup, not the pod's: a multi-container pod has
  both, the OOM killer works on the container's, and the pod-level figure is the
  reassuring one. Measured on a two-container pod elsewhere on the same node: the
  pod peaked at 80% of its limit while one container inside it peaked at exactly
  100% of its own.

`advisory` records but never refuses, and its records MUST be excluded from the
enforcing total: a process that is not honouring the budget must not be able to
spend it on behalf of the processes that are.

---

## 11. What this protocol deliberately does not have

No queue, no fairness guarantee, no cross-machine budget, no cgroup
*enforcement* (limits are read, never created), no preemption, no priority
classes, no per-user accounting, and no persistence across reboot.

**A known limit, stated rather than discovered:** a stream of small requests can
starve a large one indefinitely. `omega-server` publishes the same limit for the
same reason, and bounded queueing is specified elsewhere in this repository.

**And the honest ceiling on the whole thing:** while any process on the box is
unconfigured or in `advisory`, the ledger is a **lower bound** on true usage.
Caps should leave headroom for processes that will never participate.
