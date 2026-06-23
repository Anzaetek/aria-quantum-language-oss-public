import Lake
open Lake DSL

package quantum_proofs where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib QuantumProofs where
  srcDir := "."
  roots := #[`QuantumProofs]

-- Mathlib pin. Update this tag/SHA when bumping lean-toolchain; the two must
-- agree on the Lean version. After editing, re-resolve + fetch the cache with
--   lake update && lake exe cache get
-- Using a floating `master` makes the build non-reproducible, so always pin.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.29.0"
