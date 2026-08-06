--------------------------- MODULE MCGovernor ---------------------------
(***************************************************************************)
(* A concrete scenario for TLC.                                            *)
(*                                                                         *)
(* The question is simple: **can a shared box be driven into memory        *)
(* exhaustion by the jobs people actually submit?**                        *)
(*                                                                         *)
(* Scenario: one 128 GB box (a DGX Spark: 128 GB coherent unified memory,  *)
(* per NVIDIA "fully available to either the CPU or the GPU without any    *)
(* static partitioning" — so nothing structurally stops CPU and GPU work   *)
(* from jointly over-committing; the governor is what stops it). Half is   *)
(* handed to simulation, so the execution budget is 64 GB.                 *)
(*                                                                         *)
(* Weights are in GB and come from the real cost model                     *)
(* (`crates/omega-server/src/worker.rs`): a dense statevector is           *)
(* 2^n * 16 bytes.                                                         *)
(*                                                                         *)
(*   qml1, qml2   28 qubits =  4 GB each — inference rows, submitted often *)
(*   sweep        32 qubits = 64 GB      — one architecture-search trial   *)
(*                                                                         *)
(* Those sizes are the point: a search trial four qubits wider than an     *)
(* inference row needs SIXTEEN TIMES the memory, so the trial needs the    *)
(* whole budget while a single inference row fits trivially.               *)
(*                                                                         *)
(* Both safety questions are about exhaustion:                             *)
(*   - can admitted jobs together exceed 64 GB?  (must be NO)              *)
(*   - does a finished job give its memory back?  (must be YES)            *)
(*                                                                         *)
(* The liveness question is about fairness, and it FAILS: one 4 GB         *)
(* inference row is enough to keep the 64 GB trial out, so a steady stream *)
(* of them starves the search indefinitely.                                *)
(***************************************************************************)
EXTENDS Governor, Platforms

MCJobs == {"qml1", "qml2", "sweep"}

\* Execution budget in GB. Swap this line to model another platform — the
\* spec and every property stay exactly as they are (see Platforms.tla).
MCCapacity == BudgetSpark

\* 2^n * 16 bytes, in GB: 28 qubits -> 4, 30 qubits -> 16.
MCWeight == [j \in MCJobs |-> IF j = "sweep" THEN MCCapacity ELSE 4]

\* Classical footprint, in GB. A QML inference row carries a feature batch and
\* optimizer state alongside its circuit; a QAS trial carries per-trial
\* bookkeeping. The governor prices NONE of it.
\*
\* MCClassicalNone models the idealisation the governor implicitly assumes:
\* that a job is nothing but its statevector.
\*
\* MCClassicalReal gives each 4 GB inference row a 30 GB classical side. That
\* is a LOADED DATASET plus optimizer state plus the autograd tape — ordinary
\* for a QML step with a torch/JAX optimizer and a training set resident in
\* memory, and it dwarfs the 4 GB circuit the governor actually prices.
MCClassicalNone == [j \in MCJobs |-> 0]
MCClassicalReal == [j \in MCJobs |-> IF j = "sweep" THEN 0 ELSE 30]

=========================================================================
