# Open datasets vendored for the QML application harnesses

Small, freely redistributable datasets used by the QML example apps so the
whole pipeline stays **offline and deterministic** (no network in CI). Both
are published by the UCI Machine Learning Repository under the
**Creative Commons Attribution 4.0 International (CC BY 4.0)** license,
which permits redistribution with attribution.

## heart_cleveland.csv

- **Source**: UCI Heart Disease dataset, Cleveland Clinic subset
  (`processed.cleveland.data`), <https://archive.ics.uci.edu/dataset/45/heart+disease>.
- **Citation**: Janosi, A., Steinbrunn, W., Pfisterer, M. & Detrano, R.
  (1989). *Heart Disease*. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C52P4X>. License: CC BY 4.0.
- **Shape**: 303 rows × 14 columns (13 clinical features + `num` diagnosis,
  0 = no disease, 1–4 = disease). Missing values appear as `?` (columns
  `ca`, `thal`) exactly as in the source file; loaders must handle them.
- **Used by**: `crates/apps/butterfly-qnn` — the open-data stand-in for the
  MIMIC-III clinical-imputation task of arXiv:2606.03517 (MIMIC-III
  requires credentialed access and cannot be vendored; Cleveland is the
  canonical open clinical tabular set).
- **Provenance**: byte-identical to the UCI file, with one header line
  prepended.

## optdigits_train.csv / optdigits_test.csv

- **Source**: UCI Optical Recognition of Handwritten Digits
  (`optdigits.tra` / `optdigits.tes`),
  <https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits>.
- **Citation**: Alpaydin, E. & Kaynak, C. (1998). *Optical Recognition of
  Handwritten Digits*. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C50P49>. License: CC BY 4.0.
- **Shape**: 64 features (8×8 downsampled pen strokes, integer 0–16) +
  `digit` class (0–9). Train: 1200 rows, test: 599 rows.
- **Subsampling**: deterministic — every 3rd row of the source file,
  capped at 1200 (train) / 600 (test); header line prepended. This keeps
  the repo small while preserving the class balance of the source.
- **Used by**: `crates/apps/jl-sketch-digits` — the "process
  high-dimensional classical data with few qubits" demo of the
  Babbush-track sketching examples (cf. arXiv:2604.07639): a
  Johnson–Lindenstrauss projection compresses 64 dims into k qubit
  angles, and a quantum feature-map kernel classifies digits.

## Why these two

Per the Fourier-wall analysis (Mancilla & Tagliani, arXiv:2607.15815),
public tabular datasets like these are **expected to be classically
matchable** — the apps therefore report the quantum model *against*
classical baselines honestly rather than claiming advantage, and the
butterfly-QNN app's value is the verified O(log n)-executions training
mechanics, not a benchmark win. See `docs/QML_ROADMAP.md` for the
planned SPECTRA-certified synthetic dataset where advantage genuinely
lives.
