# Operator-facing setup for the omega-bridges Python runners.
#
# omega-bridges is Python-free at the Rust level. The Rust dispatcher
# spawns one of these scripts as a subprocess and exchanges JSON over
# stdin/stdout. Operators choose how to install the Python deps —
# this Makefile is the convenient default.
#
# Usage:
#   make qiskit-venv      # create .venv-qiskit and install Qiskit + qiskit-aer
#   make perceval-venv    # create .venv-perceval and install Perceval
#   make bloqade-venv     # create .venv-bloqade and install Bloqade (QuEra Aquila)
#   make all              # all three
#   make clean            # remove all venvs
#
# After install, expose the runners on PATH (or via env var):
#   export OMEGA_BRIDGE_QISKIT_CMD="$(pwd)/omega-bridge-qiskit-runner"
#   export OMEGA_BRIDGE_PERCEVAL_CMD="$(pwd)/omega-bridge-perceval-runner"
#
# The wrapper scripts (omega-bridge-*-runner) activate the matching
# venv and exec the Python runner so the calling shell doesn't need
# to know about virtualenvs.

PY ?= python3

.PHONY: all qiskit-venv perceval-venv bloqade-venv clean check-qiskit check-perceval check-bloqade

all: qiskit-venv perceval-venv bloqade-venv

qiskit-venv: .venv-qiskit/.installed

.venv-qiskit/.installed: requirements-qiskit.txt
	$(PY) -m venv .venv-qiskit
	./.venv-qiskit/bin/pip install --upgrade pip
	./.venv-qiskit/bin/pip install -r requirements-qiskit.txt
	@touch .venv-qiskit/.installed

perceval-venv: .venv-perceval/.installed

.venv-perceval/.installed: requirements-perceval.txt
	$(PY) -m venv .venv-perceval
	./.venv-perceval/bin/pip install --upgrade pip
	./.venv-perceval/bin/pip install -r requirements-perceval.txt
	@touch .venv-perceval/.installed

bloqade-venv: .venv-bloqade/.installed

.venv-bloqade/.installed: requirements-bloqade.txt
	$(PY) -m venv .venv-bloqade
	./.venv-bloqade/bin/pip install --upgrade pip
	./.venv-bloqade/bin/pip install -r requirements-bloqade.txt
	@touch .venv-bloqade/.installed

# Smoke check: round-trip a Bell-state QASM through the runner and
# inspect the counts. Skips with a clear message if the venv is missing.
check-qiskit:
	@if [ ! -x ./.venv-qiskit/bin/python ]; then \
		echo "Qiskit venv not built — run 'make qiskit-venv' first."; exit 2; \
	fi
	@printf '{"qasm":"OPENQASM 2.0; include \\"qelib1.inc\\"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;","shots":1024}' \
		| ./omega-bridge-qiskit-runner

check-perceval:
	@if [ ! -x ./.venv-perceval/bin/python ]; then \
		echo "Perceval venv not built — run 'make perceval-venv' first."; exit 2; \
	fi
	@printf '{"qasm":"OPENQASM 2.0; include \\"qelib1.inc\\"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;","shots":1024}' \
		| ./omega-bridge-perceval-runner

# Today the bloqade runner is a scaffold (see bloqade_runner.py). This
# check confirms the import path works and the runner emits the
# documented `bloqade-not-implemented` response — once the gate-mode
# call sites get wired, the same recipe doubles as the smoke test.
check-bloqade:
	@if [ ! -x ./.venv-bloqade/bin/python ]; then \
		echo "Bloqade venv not built — run 'make bloqade-venv' first."; exit 2; \
	fi
	@printf '{"qasm":"OPENQASM 2.0; include \\"qelib1.inc\\"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;","shots":1024}' \
		| ./omega-bridge-bloqade-runner

clean:
	rm -rf .venv-qiskit .venv-perceval .venv-bloqade
