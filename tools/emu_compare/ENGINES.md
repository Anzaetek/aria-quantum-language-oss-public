<!-- SPDX-License-Identifier: Apache-2.0 -->
# Comparison engines on akilles, and the bit-order check

The engines `run_engine.py` drives, as installed and verified on akilles
(RTX PRO 6000 Blackwell sm_120, driver 580.178.04, CUDA 12.9) on 2026-10-06.
Rows are produced under `docs/EMULATOR-COMPARISON.md`; this file only says
what runs here and what each engine can and cannot do.

| engine | package | precision | notes |
|---|---|---|---|
| `omega-cpu` | `omega-run --device cpu` | complex128 | |
| `omega-cuda` | `omega-run --device cuda` (`--features cuda`) | **complex64** | the default f32 arm (`cuda-f32`), diagonal-gate fusion |
| `omega-cuda-f64` | `omega-run --device cuda --precision f64` | complex128 | the double arm (`cuda-f64`): 1q/2q unitary circuits only, one launch per gate, no fusion; refuses 3-qubit gates, reset, conditions, collapse |
| `qulacs-cpu` | qulacs 0.6.14 | complex128 | its QASM converter lacks `cu1`/`rzz`; read through `run_engine.parse_qasm`, which refuses unknown gates |
| `qsim-cpu` | qsimcirq 0.22.1 (cirq 1.7.0) | **complex64 only** | no double path at all; the PyPI wheel has no GPU (`use_gpu` → "not supported"); `cpu_threads` defaults to 1 and is set explicitly |
| `aer-cpu` | qiskit-aer 0.17.2 on qiskit 2.5.2 | complex128 | |
| `aer-gpu` | qiskit-aer-gpu 0.15.1 on qiskit 1.4.6 | complex128 | Aer's own CUDA kernels |
| `aer-custatevec` | same, `cuStateVec_enable=True` | complex128 | the cuQuantum yardstick |
| `custatevec` | cuquantum-python-cu12 26.9.0 / custatevec-cu12 1.15.0, driven directly (`gpu/custatevec_arm.py`) | complex64 (complex128 for the f64 control) | the E5 competitor: native sm_120 cubin, no JIT; the loop is written from NVIDIA's own samples, and qiskit-aer-gpu at the same fusion setting is its sanity floor |
| `ref-numpy` | `qiskit.quantum_info.Statevector` | complex128 | reads QASM with qiskit's own parser, not `parse_qasm` |

**aer-gpu runs at double on Blackwell, through PTX.** The wheel carries SASS
for sm_70–sm_90 only, plus sm_90 PTX; the driver JIT-compiles the PTX for
sm_120 at load (cached in `~/.nv/ComputeCache` afterwards — the first call is
slow, so always warm up). Precision genuinely differs: on a 20-qubit random
circuit against Aer CPU double, 1−F is 6.9e-15 / 7.3e-15 (cuStateVec off/on)
at `precision="double"` and 8.1e-9 / 1.3e-7 at `"single"`. The returned array
is complex128 in all four cases, so its dtype says nothing — only a fidelity
check distinguishes them.

**Venvs.** `qiskit-aer-gpu` pins qiskit 1.x, so it cannot share a venv with
qiskit 2.x:

    python3 -m venv emu-compare-venv
    emu-compare-venv/bin/pip install qulacs qsimcirq quimb "qiskit==2.5.2" "qiskit-aer==0.17.2"
    python3 -m venv emu-compare-gpu-venv
    emu-compare-gpu-venv/bin/pip install "qiskit<2" "qiskit-aer-gpu==0.15.1"

`run_engine.py` finds them through `EMU_CMP_PY_CPU` / `EMU_CMP_PY_GPU`, and
the omega binary through `OMEGA_RUN` (default `target/release/omega-run`).

## The bit-order check — `test_bitorder.py`

qiskit/Aer, qulacs and omega put qubit 0 in the LOW bit of the state index;
cirq/qsim put `qubit_order[0]` in the HIGH bit. A comparison that does not
convert compares mirrored bitstrings, and GHZ — symmetric under reversal —
reports agreement anyway. The test runs every engine on

    x q[0]; h q[1]; cx q[1],q[n-1]; cx q[1],q[0]; u1(0.5) q[n-1];

at n = 3 and 16, against a state written down from integers, whose reversal
has disjoint support (a mirrored engine scores F = 0, not 0.99). It fails on
any FAIL or ERROR, and exits 2 if every engine skipped.

    <emu-compare-venv>/bin/python tools/emu_compare/test_bitorder.py [ENGINE ...]

Mutations shown red when it landed: qsim with `qubit_order` reversed →
`FAIL: n=3 F(little-endian)=0.000000 F(big-endian)=1.000000`, exit 1; every
selected engine skipped (`OMEGA_RUN=/nonexistent`) → exit 2.

## omega-run's timing and state flags

* `--timing-reps N` (N ≥ 1): runs the plain statevector evolution 1 + N times
  in one process and prints one stderr line
  `omega-timing: {"arm","threads","warmup_s","reps_s"}`. The clock wraps the
  CLI's own execute dispatch — CPU `StatevectorBackend::execute`, CUDA
  backend `execute` including the synchronising device→host copy; CUDA
  context creation and NVRTC compilation (in `new()`) and all parsing and
  output are outside it. Refused, by name, in every mode other than a plain
  `--statevector --backend statevector` run, and without an explicit
  `--device cpu|cuda` (left to `OMEGA_DEVICE`, a missing CUDA build would
  fall back to CPU and the line would carry a CPU time).
* `--dump-state-npy F`: the final state as complex128 `.npy`, qubit 0 = low
  bit, through every `--dump-state-bits` guard.
