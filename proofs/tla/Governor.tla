---------------------------- MODULE Governor ----------------------------
(***************************************************************************)
(* Admission control for the omega-server resource governor.               *)
(*                                                                         *)
(* Models `crates/omega-server/src/worker.rs`: jobs are priced, then        *)
(* reserved against a per-pool byte budget with a NON-BLOCKING              *)
(* `try_acquire`. Refused jobs are told to retry; there is no queue.        *)
(*                                                                         *)
(* Two things are being checked, and they have very different characters:  *)
(*                                                                         *)
(*  - SAFETY (expected to hold): the admitted set never exceeds a pool's    *)
(*    capacity, and a completed job returns exactly what it took.           *)
(*                                                                         *)
(*  - LIVENESS (expected to FAIL, and that is the point): a job that fits   *)
(*    is eventually admitted. Under `try_acquire` with no queue, a steady   *)
(*    trickle of small jobs can starve a large one forever, while the       *)
(*    `Retry-After` header keeps promising a retry that never succeeds.     *)
(*    TLC's counterexample trace is the argument for a bounded queue —      *)
(*    concrete, rather than "I think this can starve".                      *)
(*                                                                         *)
(* NOTE ON SCOPE. This verifies the PROTOCOL against the weights it is      *)
(* handed. It cannot know a weight is a lie about what a backend really     *)
(* allocates — which is exactly the defect an adversarial review found in   *)
(* the first implementation (MPS priced by its tensors while the backend    *)
(* contracted to a dense statevector). That gap is closed by reading the    *)
(* backends and by differential tests, never by this model. Said out loud   *)
(* so the spec does not become the same false confidence.                   *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS
    Jobs,        \* set of job identifiers
    Weight,      \* [Jobs -> Nat]  memory each job needs, in GB
    Capacity,    \* Nat            execution budget, same units
    Governed     \* BOOLEAN        FALSE models NO resource management

ASSUME CapacityIsPositive == Capacity \in Nat /\ Capacity > 0
ASSUME GovernedIsBoolean == Governed \in BOOLEAN
ASSUME WeightsArePositive ==
    /\ Weight \in [Jobs -> Nat]
    /\ \A j \in Jobs : Weight[j] > 0

VARIABLES
    admitted,    \* jobs currently holding a reservation
    finished     \* jobs that have run to completion

vars == <<admitted, finished>>

TypeOK ==
    /\ admitted \subseteq Jobs
    /\ finished \subseteq Jobs
    /\ admitted \cap finished = {}

RECURSIVE SumWeights(_)
SumWeights(S) ==
    IF S = {} THEN 0
    ELSE LET pick == CHOOSE y \in S : TRUE
         IN Weight[pick] + SumWeights(S \ {pick})

Used == SumWeights(admitted)

Free == Capacity - Used

Init ==
    /\ admitted = {}
    /\ finished = {}

(* A job is admitted only if it fits in what is free RIGHT NOW. This is
   `try_acquire_many_owned` — no queue, no reservation, no fairness.

   With `Governed = FALSE` the capacity check disappears entirely: that is the
   server BEFORE admission control, where any submitted job simply starts. It is
   modelled so the governor can be shown NECESSARY rather than merely correct —
   `NeverExceedsCapacity` fails immediately without it. A safety property that
   holds in both the guarded and unguarded worlds would be proving nothing. *)
Admit(j) ==
    /\ j \notin admitted
    /\ j \notin finished
    /\ (Governed => Weight[j] <= Free)
    /\ admitted' = admitted \cup {j}
    /\ UNCHANGED finished

(* Completion releases exactly the weight taken — the Reservation guard's
   Drop. Releasing twice would silently inflate capacity, so the model makes
   the transition once and only from `admitted`. *)
Complete(j) ==
    /\ j \in admitted
    /\ admitted' = admitted \ {j}
    /\ finished' = finished \cup {j}

(* Small jobs are recycled: a finished small job can be resubmitted. This is
   what models a steady trickle of client traffic, and it is what makes the
   starvation scenario reachable. *)
Resubmit(j) ==
    /\ j \in finished
    /\ Weight[j] * 2 <= Capacity      \* only "small" jobs recycle
    /\ finished' = finished \ {j}
    /\ UNCHANGED admitted

Next ==
    \/ \E j \in Jobs : Admit(j)
    \/ \E j \in Jobs : Complete(j)
    \/ \E j \in Jobs : Resubmit(j)

(* Weak fairness on completion and admission: work that can proceed does. *)
Spec == Init /\ [][Next]_vars
        /\ \A c \in Jobs : WF_vars(Complete(c))
        /\ \A a \in Jobs : WF_vars(Admit(a))

--------------------------------------------------------------------------
(* SAFETY — these must hold. *)

(* The reason this module exists. *)
NeverExceedsCapacity == Used <= Capacity

(* A job cannot be both holding a reservation and done: that would mean its
   permit was released while still counted, or counted after release. *)
NoDoubleAccounting == admitted \cap finished = {}

--------------------------------------------------------------------------
(* LIVENESS — expected to FAIL under the current design.

   Read the counterexample as: the big job is repeatedly passed over while
   small jobs cycle through, so `Free` never simultaneously reaches
   Weight[big]. Fixing it needs a bounded queue with reservation, i.e. small
   jobs must stop being admitted ahead of a waiting large one.          *)

BigJobs == {j \in Jobs : Weight[j] * 2 > Capacity}

EventuallyAdmitted ==
    \A j \in BigJobs : <>(j \in admitted \/ j \in finished)

==========================================================================
