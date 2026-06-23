/-
  QuantumProofs.Noise — formal backing for the `noise` emulator app's error-model laws.

  A self-contained, sorry-free library of single-qubit quantum channels in Kraus (operator-sum)
  form over `Matrix (Fin d) (Fin d) ℂ`, ported from the upstream proof repository.  These prove
  the exact closed-form laws that `crates/apps/noise` certifies numerically:

    * `Channel`          — density operators (`IsDensity`) and CPTP `KrausMap`s; the headline
                           well-definedness result `KrausMap.apply_isDensity` (Kraus maps send
                           density operators to density operators).
    * `Pauli`            — σX/σY/σZ as Hermitian involutions, the Pauli twirl, and ⟨Z⟩.
    * `Depolarizing`     — `E_p(ρ) = (1-p)ρ + (p/2)(Tr ρ)·1`, fidelity `1 - p/2`
                           (noise app law (A), `⟨Z⟩ = (1-p')cosθ`).
    * `AmplitudeDamping` — `⟨Z⟩(E_γ ρ) = ⟨Z⟩(ρ) + 2γ·d` and survival `1 - γ`
                           (noise app law (B), `⟨Z⟩ = ⟨Z⟩₀ + 2γ·sin²(θ/2)`).
    * `PhaseDamping`     — coherence decay `(E_λ ρ)₀₁ = (1-λ)·ρ₀₁` with populations preserved.
    * `DepolarizingDepth`— depth-G floor `F_G(p)=(1+(1-p)^G)/2` (per-gate global depolarizing),
                           instantiated at `G=3` for the circulant `CyclicShift` solver.
    * `TensorChannel`    — multi-qubit (width) lift: tensor of CPTP is CPTP, fidelity factorizes
                           on product states, n-qubit floor `(1-p/2)^n` (n=3 circulant register).
    * `GlobalDepolarizing`— entangled-output floor: the register-wide depolarizing channel is
                           entanglement-independent, fidelity `(1-p)^G+(1-(1-p)^G)/d` for any state
                           (d=8 circulant register).

  All channel theorems depend only on `propext / Classical.choice / Quot.sound` (no `sorryAx`).
-/
import QuantumProofs.Noise.Channel
import QuantumProofs.Noise.Pauli
import QuantumProofs.Noise.Depolarizing
import QuantumProofs.Noise.AmplitudeDamping
import QuantumProofs.Noise.PhaseDamping
import QuantumProofs.Noise.DepolarizingDepth
import QuantumProofs.Noise.TensorChannel
import QuantumProofs.Noise.GlobalDepolarizing
