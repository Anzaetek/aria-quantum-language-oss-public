# QML roadmap — Kerenidis track, Fourier-wall certification, architecture search

Status of the QML application tracks and the planned follow-ups. Everything
listed as **landed** is implemented, tested, and registered in `ci.sh`.

## Landed (July 2026)

### Kerenidis butterfly-QNN track (arXiv:2606.03517)

| Piece | Where |
|---|---|
| Native `RBS` gate (Givens rotation), all the way through lexer → AST → IR → statevector/MPS backends → adjoint AD → QASM decomposition export | `crates/omega-core/src/circuit.rs`, `crates/omega-backend-statevector/src/gates.rs`, `crates/aria-core/src/ast/*` |
| 4-term Givens parameter-shift rule (frequencies {1, 2}; the 2-term rule is provably wrong for RBS) | `crates/omega-core/src/gradient.rs::SlotShiftRule::FourTermGivens` |
| Parallelised parameter-shift: trailing commuting block → all gradients from ONE execution via commutator observables `i·[G_k, H]` | `crates/omega-core/src/parallel_shift.rs`, `GradMethod::ParallelParameterShift` |
| Layer-wise freezing + explicit inits + Adam in both trainers, CLI flags `--freeze/--set/--opt/--grad` | `crates/aria-runtime/src/train.rs`, `crates/omega-core/src/qml.rs`, `crates/aria-cli` |
| Butterfly QNN app on the open UCI Cleveland heart set (imputation under 30% MCAR), staged train-freeze-couple protocol, gradients verified parallel≡serial to 1e-9, 16× execution-count reduction demonstrated | `crates/apps/butterfly-qnn`, `examples/aria/butterfly_qnn.aria`, `examples/data/heart_cleveland.csv` |
| Lean 4 proofs: RBS unitarity, one-parameter group, generator-form derivative, G³=G | `proofs/lean4/QuantumProofs/Rbs.lean` |

### Babbush sketching track (arXiv:2604.07639)

| Piece | Where |
|---|---|
| QOS phase-oracle sketch emulation (existing) | `crates/omega-core/src/qos.rs`, `crates/apps/qos` |
| JL-sketch feature-map forward check (existing, synthetic) | `examples/aria/sketch_qml.aria`, `crates/apps/forward` |
| **New**: JL-sketch on real open data — UCI optdigits 3-vs-8 on 12 qubits, oracle-exact ⟨Z⟩ verification, honest classical-lane comparison | `crates/apps/jl-sketch-digits`, `examples/aria/jl_sketch_digits.aria`, `examples/data/optdigits_*.csv` |

Both dataset apps deliberately report classical baselines that match or beat
the quantum lane. That is the expected outcome on open tabular/image data —
see below.

## Landed: Fourier-wall certification track (arXiv:2607.15815, Mancilla–Tagliani)

*Why quantum models don't beat classical baselines on public tabular data,
and a certified recipe for where they can.* Angle-encoded QNNs are partial
Fourier series; genuine advantage needs the target to be simultaneously
**off-grid, interaction order ≥ 3, high-frequency, near-independent-feature,
and non-enumerable** — the "Fourier wall". Neither UCI set above passes,
which is exactly what our two apps observe empirically.

Implemented in `crates/apps/spectra` + `examples/aria/spectra_heisenberg.aria`
(run: `cargo run -p aria-verify -- spectra`):

- **Generators**: the §6.2 controlled sparse pocket (planted 3-way term at
  f = (3.7, 5.1, 6.8)) and the §6.3 disordered-Heisenberg substrate — the
  shipped Trotter circuit generates the labels AND doubles as the
  dynamics-matched quantum lane (DMQ) with trainable couplings.
- **Tier 1**: ρ_off off-grid ratio (> 0.3), advisory Fourier-GAM holdout
  AUC, d ≤ 14 feasibility.
- **Tier 2**: five classical lanes (LogReg, trained-frequency GAM, GA2M,
  boosted stumps, order-matched JOINT with a two-stage supervised
  periodogram scan) vs the quantum lane; certify iff paired-bootstrap
  CI_lo(Δ AUC) > 0 AND the entanglement ON−OFF ablation gate passes.
- **Verified outcomes** (the app's CHECK): UCI heart REFUSED (stumps
  0.86 ≫ QNN 0.54), sparse pocket REFUSED (JOINT recovers the planted
  frequencies and scores 0.90 while the QNN sits at 0.46 — the panel-
  completeness calibration), Heisenberg substrate CERTIFIED (DMQ learns
  the disorder couplings to ~0.05 and scores 0.99 vs 0.58 best
  classical; CI_lo(Δ) = +0.34, ablation +0.48).

Original plan (for reference):

1. **Synthetic generators first** (highest value / effort ratio):
   - *Controlled sparse pocket* (paper §6.2): 3 whitened phases,
     `z(φ) = w·Σ cos(f_jφ_j + φ_j) + cos(f₁φ₁ + f₂φ₂ + f₃φ₃ + φ)`,
     `f = (3.7, 5.1, 6.8)`, `y ~ Bernoulli(σ(g(z − τ)))` — the calibration
     instrument with a known ground-truth joint term.
   - *Dense-spectrum quantum substrate* (paper §6.3): labels from a
     disordered Heisenberg chain
     `H(φ) = Σ J_k(XX + YY + ZZ)_{k,k+1} + Σ φ_i Z_i`, `J_k ~ U[0.5, 1.5]`,
     `|+⟩⊗n` initial state, 3 Trotter steps, label = sign of
     `⟨Σ Z_iZ_{i+1}⟩ − median`. **Reuses the existing Trotter machinery**
     (`crates/apps/trotter`, `examples/aria/trotter.aria`) — generate with
     the verified simulator, ship as `examples/data/spectra_dense.csv`.
2. **Tier-1 structural screen** (new crate `crates/apps/spectra` or module
   in `aria-verify-core`): off-grid ratio ρ_off via per-feature
   periodogram; interaction gates via lane deltas (GAM vs GA2M vs
   boosted-stumps); feasibility gate `d·q ≤ 14`. Pure Rust, minutes.
3. **Tier-2 lanes** (pure Rust): LogReg, trained-frequency Fourier GAM,
   GA2M (pairwise harmonic products), boosted stumps (HGB stand-in), and
   the order-matched JOINT lane (supervised periodogram scan for k-way
   `cos(Σ f_jφ_j + b)` terms). Paired stratified bootstrap on identical
   splits; certify iff `CI_lo(AUC_QNN − max lane) > 0` **and** the
   entanglement ON−OFF ablation gate passes.
4. **Expected demo**: SPECTRA refuses heart/optdigits (like the paper's
   peak-load case) and certifies the Heisenberg substrate — giving this
   repo an honest, reproducible "where advantage lives" example with the
   butterfly QNN as the quantum lane.

## Landed: noise-robustness of the certification

*Does the certified advantage survive noise, and at what rate does it vanish?*
— the follow-up every advantage claim owes. Implemented in
`crates/apps/spectra/src/noise.rs` + `spectra_noise` (run:
`cargo run -p aria-verify -- spectra_noise`):

- The dynamics-matched quantum lane is trained ONCE on the ideal simulator
  (the deployed model), then re-scored through the **PauliProp backend**,
  which folds a per-gate depolarizing channel into the Heisenberg-adjoint
  expectation *exactly* (no trajectory sampling) and reproduces the
  statevector correlator to ~1e-7 at zero noise. The classical panel is
  scored on the same rows, so the paired bootstrap stays valid.
- **Verified sweep** (7-site substrate, 96-row balanced eval): quantum AUC
  and the certification gate `CI_lo(Δ AUC) > 0` degrade monotonically with
  the per-gate depolarizing rate —

  | depol rate | quantum AUC | CI_lo(Δ vs best classical) | verdict |
  |---|---|---|---|
  | 0.0000 | 0.993 | +0.377 | CERTIFIED |
  | 0.0050 | 0.968 | +0.352 | CERTIFIED |
  | 0.0100 | 0.885 | +0.262 | CERTIFIED |
  | 0.0200 | 0.739 | +0.093 | CERTIFIED |
  | 0.0400 | 0.645 | −0.007 | REFUSED |

  The advantage survives up to a **per-gate depolarizing rate ≈ 0.02** and is
  destroyed by 0.04 — an honest robustness margin (best classical sits at
  chance, ~0.51, throughout, since the substrate is non-enumerable). CHECK:
  PauliProp reproduces the statevector scores at zero noise (|Δ| ≤ 1e-6),
  the substrate CERTIFIES at zero noise, and a crossover to REFUSED exists
  within the sweep.

## Quantum architecture search — first increment landed

Idea (user-proposed): given a dataset, *discover* circuit patterns
independently instead of hand-picking them.

**Landed** (`aria_app_spectra::arch_search`, run:
`cargo run -p aria-verify -- arch_search`): exhaustive search over five
coupling graphs on 7 sites (chain / ring / star / stride-2 / disconnected
pairs), each trained identically (per-bond couplings, adjoint + Adam,
affine head) and scored on a holdout split with a parsimony penalty.
Given ONLY the labelled Heisenberg-substrate data, the search selects the
generator's CHAIN graph (holdout AUC 0.98 vs ≤ 0.65 for every wrong
graph), recovers the disorder couplings to ≤ 0.2, and emits the
discovered architecture as Aria source. The programmatic circuit builder
is cross-checked against the lowered `spectra_heisenberg.aria` (Δ = 0).

**Also landed** (the three follow-ups):

- `arch_evolve` — elitist evolutionary loop over layered RBS-mask
  genomes (125-genome space) on the heart-imputation task; deterministic
  with memoisation. The evolved [none | stride-4 | stride-2] BEATS the
  hand-built butterfly (held-out MSE 0.383 vs 0.392) with fewer layers;
  the no-entanglement ablation sits at 0.778.
- `arch_priors` — SPECTRA's classical joint-periodogram scan (zero
  quantum evaluations) recovers the pocket's planted frequencies and,
  used as the QNN's frequency-prior init (with trainable phase offsets
  and weight-≤3 Z-string readout), lifts holdout AUC 0.487 → 0.772
  under an identical budget.
- `spectra_scaling` — the substrate at 7…13 sites: |+⟩^⊗n invariant
  exact at every size, DMQ-vs-classical gap persists (+0.19/+0.29/+0.18
  at 7/9/11), and the measured classical-simulation cost fits
  2^(1.15·n) — the paper's 2^(1.14·n) — with an illustrative crossover
  extrapolation near n* ≈ 14 (paper: 13–19).

Original long-term sketch (for reference):
- **Search space**: layered placements of RBS / Rot / entangler gates with
  a connectivity mask per layer (butterfly = stride masks {n/2, …, 2, 1});
  Hamming-weight-preserving subspaces prunable analytically.
- **Search signal**: Tier-1 SPECTRA statistics of the dataset (off-grid
  ratio, interaction order) → prior over frequencies/depth; validation MSE
  with parameter-shift-trainability regularizer (penalize non-commuting
  final blocks — keeps the parallel-shift speedup applicable).
- **Search loop**: evolutionary or successive-halving over the mask space
  (cheap: each candidate trains layer-wise with frozen prefixes, reusing
  `TrainConfig::frozen` + `ParallelParameterShift`).
- **Success criterion**: rediscover the butterfly masks on the Heisenberg
  substrate; report what it finds on the sparse pocket (expected: a single
  3-way joint block — matching the known ground truth).

This builds only on landed pieces (freeze masks, parallel shift, SPECTRA
generators) and needs no new engine features.
