#!/usr/bin/env bash
# Collect everything CUDA_TODO.md asks a Linux/NVIDIA box to report.
#
# Run on the target machine and paste the output back. The point is that the
# handoff should not depend on someone remembering which numbers matter — in
# particular the RAW PROBE (nvidia-smi memory.total + /proc/meminfo MemTotal),
# because if the topology classifier gets it wrong those two numbers are what
# shows why.
#
#   ./tools/cuda-report.sh              # environment report only
#   ./tools/cuda-report.sh --with-ci    # also runs ARIA_CUDA=1 ./ci.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

hr() { printf '\n----- %s -----\n' "$1"; }

hr "platform"
uname -srm
# aarch64 vs x86_64 selects which classifier branch runs: aarch64 is where
# GB10/GH200 live and where a misclassification is dangerous rather than wasteful.
echo "arch: $(uname -m)   (aarch64 => unified-memory heuristics apply)"
[ -r /etc/os-release ] && . /etc/os-release && echo "os: ${PRETTY_NAME:-unknown}"

hr "host memory (raw)"
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo 2>/dev/null || echo "no /proc/meminfo"

hr "cgroup limit (a container budgets against THIS, not host RAM)"
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  [ -r "$f" ] && echo "$f = $(cat "$f")"
done
echo "(absent or 'max' => not containerised)"

hr "GPUs (raw probe — this is what the classifier sees)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version \
             --format=csv,noheader 2>&1
  nvidia-smi --query-gpu=index,mig.mode.current --format=csv,noheader 2>&1 \
    | sed 's/^/mig: /'
else
  echo "nvidia-smi absent — CUDA stage cannot run here"
fi
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -2

hr "expected classification"
cat <<'NOTE'
GB10 / DGX Spark  -> Unified  (device total ~= host total; one shared pool)
GH200             -> Discrete (coherent, but distinct LPDDR5X + HBM3)
A100 / H100 / RTX  -> Discrete (host pool + one pool per GPU)
Misclassifying GB10 as Discrete lets the governor hand out ~2x the machine's
memory. Report a mismatch rather than only setting OMEGA_MEM_TOPOLOGY.
NOTE

hr "server view (start omega-server, then GET /health)"
cat <<'NOTE'
  cargo run --release -p omega-server -- --save-token-to /tmp/omega-token &
  curl -s localhost:8080/health | python3 -m json.tool
Report execution.unified, execution.pools[], max_qubits and limits_from.
NOTE

hr "OpenCL linkability (needs the ICD loader DEV symlink, not just .so.1)"
ldconfig -p 2>/dev/null | grep -E 'libOpenCL\.so' || echo "no libOpenCL in ldconfig"
for p in /usr/lib/x86_64-linux-gnu/libOpenCL.so /usr/local/cuda/targets/*/lib/libOpenCL.so; do
  [ -e "$p" ] && echo "found: $p"
done
echo "(only libOpenCL.so.1 => apt install ocl-icd-opencl-dev, or point RUSTFLAGS at the CUDA copy)"

hr "cross-check venvs (both MANDATORY)"
[ -x ./.venv-qiskit/bin/python ] \
  && ./.venv-qiskit/bin/python -c "import qiskit,qiskit_aer;print('qiskit',qiskit.__version__,'aer',qiskit_aer.__version__)" \
  || echo ".venv-qiskit MISSING — python3 -m venv .venv-qiskit && ./.venv-qiskit/bin/pip install qiskit qiskit-aer"
[ -x tools/qec_cross_check/.venv/bin/python ] \
  && tools/qec_cross_check/.venv/bin/python -c "import pymatching,stim;print('pymatching',pymatching.__version__,'stim',stim.__version__)" \
  || echo "QEC venv MISSING — it self-provisions on first ARIA_QEC_XCHECK=1 run"

hr "libtorch auto-fetch"
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64) echo "supported by tools/setup-libtorch.sh" ;;
  Linux/aarch64) echo "NO URL for Linux/aarch64 — the tch stage will SKIP here until one is added" ;;
  *) echo "unknown: $(uname -s)/$(uname -m)" ;;
esac

if [ "${1:-}" = "--with-ci" ]; then
  hr "ARIA_CUDA=1 ./ci.sh (plus the mandatory cross-checks)"
  ARIA_CUDA=1 ARIA_QISKIT_XCHECK=1 ARIA_QEC_XCHECK=1 ./ci.sh
  echo "CI_EXIT=$?"
else
  hr "next"
  echo "  ./tools/cuda-report.sh --with-ci    # runs the full CUDA + cross-check CI"
fi
