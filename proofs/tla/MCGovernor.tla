--------------------------- MODULE MCGovernor ---------------------------
(***************************************************************************)
(* Concrete instance for TLC. Kept out of Governor.tla so the spec stays    *)
(* parameterised, and because TLC's .cfg format cannot express a function   *)
(* literal — the constants are defined here and substituted with `<-`.      *)
(*                                                                         *)
(* Deliberately tiny: two small jobs (weight 1) that can recycle forever,   *)
(* and one large job (weight 3) against a budget of 4. Starvation needs an  *)
(* interleaving, not scale.                                                 *)
(***************************************************************************)
EXTENDS Governor

MCJobs == {"s1", "s2", "big"}
MCCapacity == 4
MCWeight == [j \in MCJobs |-> IF j = "big" THEN 3 ELSE 1]

=========================================================================
