--------------------------- MODULE Platforms ---------------------------
(***************************************************************************)
(* Execution budgets per platform, in GB.                                  *)
(*                                                                         *)
(* ADDING A PLATFORM IS ONE LINE HERE plus a three-line .cfg. The spec and *)
(* the properties never change — only the budget does — because the        *)
(* governor's behaviour depends on capacity and nothing else about the     *)
(* hardware.                                                               *)
(*                                                                         *)
(* Budgets are HALF of usable memory, matching the shipped default         *)
(* (`DEFAULT_MEM_FRACTION`-equivalent: half the detected pool).            *)
(*                                                                         *)
(* | platform            | memory                    | budget |            *)
(* |---------------------|---------------------------|--------|           *)
(* | DGX Spark (GB10)    | 128 GB unified            |   64   |           *)
(* | GH200 (96 GB HBM3)  | 96 GB device (discrete)   |   48   |           *)
(* | GH200 (144 GB)      | 144 GB HBM3e              |   72   |           *)
(* | H100 SXM/PCIe 80 GB | 80 GB device              |   40   |           *)
(* | H100 NVL 94 GB      | 94 GB device              |   47   |           *)
(* | GB300 / Blackwell   | 288 GB HBM3e              |  144   |           *)
(* | laptop              | 24 GB unified             |   12   |           *)
(*                                                                         *)
(* For a DISCRETE platform this is the DEVICE pool; the host pool is a     *)
(* separate budget with the same rules, which is why one number suffices   *)
(* per pool rather than per machine.                                       *)
(***************************************************************************)
EXTENDS Naturals

BudgetSpark    == 64
BudgetGH200_96 == 48
BudgetGH200144 == 72
BudgetH100_80  == 40
BudgetH100NVL  == 47
BudgetGB300    == 144
BudgetLaptop   == 12

=========================================================================
