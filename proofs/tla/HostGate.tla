---------------------------- MODULE HostGate ----------------------------
(***************************************************************************)
(* Host-wide admission across SEPARATE PROCESSES, and what happens when one *)
(* of them dies without releasing.                                          *)
(*                                                                          *)
(* Models `crates/omega-hostgate`: a shared ledger file, a lock around each  *)
(* transaction, holders keyed by an identity that survives PID reuse, and a  *)
(* prune that reclaims a dead holder's tokens.                              *)
(*                                                                          *)
(* WHY THIS EXISTS ALONGSIDE `Governor.tla`. That model has exactly one way  *)
(* to release a reservation — `Complete` — so every job that takes capacity  *)
(* politely gives it back. Real processes do not. They are OOM-killed,       *)
(* `SIGKILL`ed, and lost to a panic that skips every destructor. A gate      *)
(* whose only release path is the happy one is a gate that leaks its budget  *)
(* to zero on the first crash and then refuses everything forever.           *)
(*                                                                          *)
(* So the actions here that `Governor.tla` does not have are `Crash`,        *)
(* `Prune` and `Restart`, and they are the entire point.                     *)
(*                                                                          *)
(* THE MODEL'S CENTRAL DISTINCTION. Two quantities are tracked separately:   *)
(*                                                                          *)
(*   ledger[p]    what the FILE says p holds  — accounting                   *)
(*   resident[p]  what p ACTUALLY occupies    — the machine                  *)
(*                                                                          *)
(* A crash frees `resident` immediately (the kernel reclaims the memory) and *)
(* leaves `ledger` untouched (the destructor never ran). The gap between     *)
(* them is precisely what the prune must close, and every interesting        *)
(* property below is about that gap being closed in the right direction:     *)
(* reclaiming a DEAD holder is required, reclaiming a LIVE one is a          *)
(* catastrophe.                                                              *)
(*                                                                          *)
(* TWO TOGGLES, EACH MODELLING A DESIGN WE REJECTED, so the design can be    *)
(* shown NECESSARY rather than merely correct:                               *)
(*                                                                          *)
(*   AtomicLedger = FALSE   the ledger is rewritten in place (seek, truncate,*)
(*                          write) instead of temp-file + rename. A crash    *)
(*                          between truncate and write leaves a ZERO-LENGTH  *)
(*                          file, and an empty file reads as an EMPTY GATE.  *)
(*                          Every live holder vanishes from the accounting   *)
(*                          while its memory is still resident.              *)
(*                                                                          *)
(*   IdentityKeyed = FALSE  holders keyed by PID alone, with no start time.  *)
(*                          After a PID is recycled the dead holder's record *)
(*                          matches a LIVE process, so the prune never       *)
(*                          reclaims it — a permanent leak — and a scan that *)
(*                          trusts the match can charge the newcomer for it. *)
(*                                                                          *)
(* NOTE ON WHAT HAS ACTUALLY BEEN RUN. As of 2026-08-21 **TLC has never been *)
(* run against this module** — there was no JVM on the authoring machine and  *)
(* no tla2tools.jar on the reviewing one. Every figure quoted for it comes    *)
(* from `check_hostgate.py`, which enumerates the same state space from an    *)
(* independent implementation of the same transition relation. That is one    *)
(* implementation, not two agreeing. Said plainly because the presence of a   *)
(* .tla file reads as "this was model-checked" to everyone who did not write  *)
(* it. When TLC is available the baseline must report 13,824 distinct states; *)
(* a different count means one of the two artefacts has drifted.              *)
(*                                                                           *)
(* NOTE ON SCOPE. This verifies the PROTOCOL against the amounts it is       *)
(* handed. It cannot know that an amount is a lie about what a process       *)
(* really allocates; that gap is closed by measurement, never here. It also  *)
(* says nothing about processes that never call the gate at all — an         *)
(* unconfigured neighbour is invisible to the ledger by construction, and    *)
(* the crate's own documentation states that the accounting is therefore a   *)
(* LOWER BOUND on true usage.                                                *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Procs,          \* set of process identifiers (think: PID slots)
    Need,           \* [Procs -> Nat]  what each wants, in whole units
    Capacity,       \* Nat             the budget, same units
    AtomicLedger,   \* BOOLEAN         FALSE models in-place rewrite
    IdentityKeyed   \* BOOLEAN         FALSE models PID-only keying

ASSUME CapacityIsPositive == Capacity \in Nat /\ Capacity > 0
ASSUME NeedsArePositive ==
    /\ Need \in [Procs -> Nat]
    /\ \A p \in Procs : Need[p] > 0
ASSUME TogglesAreBoolean ==
    /\ AtomicLedger \in BOOLEAN
    /\ IdentityKeyed \in BOOLEAN

VARIABLES
    ledger,     \* [Procs -> Nat]  what the shared file records
    resident,   \* [Procs -> Nat]  what each process actually occupies
    alive,      \* SUBSET Procs    currently running
    gen,        \* [Procs -> Nat]  incarnation number: a restart bumps it
    ledgerGen   \* [Procs -> Nat]  the incarnation recorded WITH the record

vars == <<ledger, resident, alive, gen, ledgerGen>>

MaxGen == 2   \* bound the state space: one restart per slot is enough to
              \* expose PID reuse, and more only multiplies states

TypeOK ==
    /\ ledger \in [Procs -> Nat]
    /\ resident \in [Procs -> Nat]
    /\ alive \subseteq Procs
    /\ gen \in [Procs -> Nat]
    /\ ledgerGen \in [Procs -> Nat]

RECURSIVE SumOver(_, _)
SumOver(f, S) ==
    IF S = {} THEN 0
    ELSE LET pick == CHOOSE y \in S : TRUE
         IN f[pick] + SumOver(f, S \ {pick})

Charged  == SumOver(ledger, Procs)     \* what the gate believes is spent
Occupied == SumOver(resident, Procs)   \* what the machine is actually holding
Free     == IF Charged >= Capacity THEN 0 ELSE Capacity - Charged

Init ==
    /\ ledger = [p \in Procs |-> 0]
    /\ resident = [p \in Procs |-> 0]
    /\ alive = Procs
    /\ gen = [p \in Procs |-> 0]
    /\ ledgerGen = [p \in Procs |-> 0]

(* Does this record still belong to a running process?

   With identity keying we compare the incarnation too, so a recycled PID does
   not inherit the dead holder's record. Without it, "is some process using
   this number" is the only question that can be asked — and after a restart
   the answer is yes, for the wrong process. *)
StillHeld(p) ==
    IF IdentityKeyed
    THEN p \in alive /\ ledgerGen[p] = gen[p]
    ELSE p \in alive

(* Whether a process holds a grant OF ITS OWN — which is a different question
   from `StillHeld`, and the difference is the whole PID-reuse story.

   `StillHeld` is what the PRUNE asks while scanning the file, so it is exactly
   as good as the keying scheme. `Owns` is what the PROCESS knows: it has a
   guard object in its own memory, or it does not. A restarted process never
   has one, whatever the file says and whatever the keying scheme is, because
   its predecessor's guard died with its address space.

   Conflating them is what hid the leak on the first run: with PID-only keying
   `StillHeld` becomes true again after a restart, so a model that used it for
   `Release` let the newcomer release a record it could not possibly know
   about — repairing, inside the model, the exact defect the model was built to
   expose. *)
Owns(p) == p \in alive /\ ledgerGen[p] = gen[p]

(* Take capacity and then occupy it. Non-blocking: it fits now or it is
   refused now. There is no queue here, deliberately and for the same reason
   `Governor.tla` has none — see that module. *)
Acquire(p) ==
    /\ p \in alive
    /\ ledger[p] = 0
    /\ Need[p] <= Free
    /\ ledger' = [ledger EXCEPT ![p] = Need[p]]
    /\ resident' = [resident EXCEPT ![p] = Need[p]]
    /\ ledgerGen' = [ledgerGen EXCEPT ![p] = gen[p]]
    /\ UNCHANGED <<alive, gen>>

(* The orderly path: the guard drops, the record clears, the memory goes.

   `StillHeld` rather than merely `p \in alive` is load-bearing, and the first
   run of this model is what showed it. A process releases THE GRANT IT TOOK,
   not whatever record happens to carry its PID. Without that guard a restarted
   process can release its dead predecessor's record — which is wrong on its own
   terms, and worse, it silently repairs the leak that PID-only keying causes
   and so hides `IdentityKeyed = FALSE` from the liveness check entirely. A
   model that lets the newcomer tidy up cannot see the bug it exists to find. *)
Release(p) ==
    /\ Owns(p)
    /\ ledger[p] > 0
    /\ ledger' = [ledger EXCEPT ![p] = 0]
    /\ resident' = [resident EXCEPT ![p] = 0]
    /\ UNCHANGED <<alive, gen, ledgerGen>>

(* THE ACTION `Governor.tla` DOES NOT HAVE.

   The process dies. The kernel reclaims its memory, so `resident` drops
   immediately — but no destructor runs, so `ledger` keeps the record. From
   this instant the gate is charging a budget to nobody, and only the prune
   can fix it.

   With a non-atomic ledger a crash can also land mid-rewrite, which truncates
   the file to nothing. An empty file is indistinguishable from an empty gate,
   so EVERY holder's record is lost at once while their memory is still
   resident. That is modelled here rather than as a separate action because it
   is the same event: a process dying at an inconvenient moment. *)
Crash(p) ==
    /\ p \in alive
    /\ alive' = alive \ {p}
    /\ resident' = [resident EXCEPT ![p] = 0]
    /\ IF AtomicLedger
       THEN ledger' = ledger                          \* record survives intact
       ELSE ledger' = [q \in Procs |-> 0]             \* truncated to nothing
    /\ UNCHANGED <<gen, ledgerGen>>

(* The failsafe. No daemon runs it and nothing detects a death: any process
   taking the lock reclaims every record whose owner is no longer there.

   Note what this does NOT do: it never touches a record that is still held.
   Reclaiming a live holder's tokens would hand its memory to the next
   request, which is strictly worse than the leak it was trying to fix. That
   is `NoPhantomReclaim` below, and it is the property the implementation's
   liveness probe is written around — "cannot determine" is treated as alive
   precisely so this can never fire on a live process. *)
Prune ==
    /\ \E p \in Procs : ledger[p] > 0 /\ ~StillHeld(p)
    /\ ledger' = [p \in Procs |-> IF StillHeld(p) THEN ledger[p] ELSE 0]
    /\ UNCHANGED <<resident, alive, gen, ledgerGen>>

(* A dead slot comes back as a NEW process that happens to reuse the number. *)
Restart(p) ==
    /\ p \notin alive
    /\ gen[p] < MaxGen
    /\ alive' = alive \cup {p}
    /\ gen' = [gen EXCEPT ![p] = gen[p] + 1]
    /\ UNCHANGED <<ledger, resident, ledgerGen>>

Next ==
    \/ \E p \in Procs : Acquire(p)
    \/ \E p \in Procs : Release(p)
    \/ \E p \in Procs : Crash(p)
    \/ \E p \in Procs : Restart(p)
    \/ Prune

Spec == Init /\ [][Next]_vars
        /\ WF_vars(Prune)
        /\ \A p \in Procs : WF_vars(Release(p))

--------------------------------------------------------------------------
(* SAFETY *)

(* THE ONE THAT MATTERS: the machine is never oversubscribed. Not "the ledger
   adds up" — the actual resident total. A gate whose books balance while the
   box is dying has proved nothing. *)
NeverOvercommitted == Occupied <= Capacity

(* The gate never charges for less than is really there. If this fails, the
   next request is admitted on top of memory that is already in use — which
   is the previous property's cause, one step earlier. *)
AccountingCoversReality == Charged >= Occupied

(* A LIVE holder's record is never reclaimed. This is the catastrophic
   direction: the leak merely wastes budget, but pruning a live holder hands
   its memory away while it is still using it. *)
NoPhantomReclaim == \A p \in alive : resident[p] <= ledger[p]

(* NOT AN INVARIANT, and the first run of this model is why it is written down
   here rather than checked.

   The obvious property to reach for is "a live process's record always belongs
   to its own incarnation". It is FALSE, in one step, in the correct design:
   Acquire, Crash, Restart leaves a stale record next to a live process that
   does not own it. That state is exactly what a crash produces and exactly what
   the prune exists to clear — forbidding it would be forbidding the crash.
   What must be true is weaker and more useful: the newcomer can never USE that
   record (`Release` is guarded by `StillHeld`), and it is always eventually
   reclaimed (`EventuallyReclaimed`). Stating the strong version as a safety
   property produces a counterexample that looks alarming and means nothing. *)

--------------------------------------------------------------------------
(* LIVENESS — the failsafe, stated as a temporal property.

   After every death, the books come back into agreement with the machine.
   This is what says the budget is not consumed permanently by crashes: a
   fleet that loses a little capacity to every OOM kill grinds to a halt over
   a week, and no test that runs for a minute would ever show it. *)

EventuallyReclaimed == []<>(Charged = Occupied)

==========================================================================
