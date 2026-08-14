-- Quantum Proofs — minimal formal-verification subtree (OSS port).
--
-- This is the dependency closure that makes `aria export <file> --circuit NAME
-- --lean` self-contained: the emitted Lean imports `QuantumProofs.Gates` /
-- `QuantumProofs.CircuitSemantics`, which build against this tree. Ported from
-- the upstream proof repository; the proven algorithm files included here are
-- the ones whose Aria examples ship in this OSS repo (circulant solver via QFT,
-- Bell state-prep, HHL + QSVT inversion via BlockEncoding).
--
-- Build (opt-in; needs a warm mathlib cache):
--   cd proofs/lean4 && lake exe cache get && lake build

import QuantumProofs.Basic
import QuantumProofs.Gates
import QuantumProofs.SqrtX
import QuantumProofs.MpsTruncation
import QuantumProofs.CircuitSemantics
import QuantumProofs.Adjoint
import QuantumProofs.QFT
import QuantumProofs.QPE
import QuantumProofs.BellPrep
import QuantumProofs.GHZPrep
import QuantumProofs.CirculantSolve
import QuantumProofs.CirculantSolveGeneral
import QuantumProofs.PauliAction
import QuantumProofs.GaussianSolve
import QuantumProofs.QFTExport
import QuantumProofs.QPEExport
import QuantumProofs.QPEFaithful
import QuantumProofs.Noise
import QuantumProofs.BlockEncoding
import QuantumProofs.HHL
import QuantumProofs.Rbs
import QuantumProofs.QSVT
import QuantumProofs.QSP
import QuantumProofs.HadamardLayer
import QuantumProofs.Grover
import QuantumProofs.GroverCircuit
