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
#   make tsim-venv        # create .venv-tsim and install bloqade-tsim (QuEra)
#   make ppvm-venv        # create .venv-ppvm and install ppvm (QuEra, git)
#   make ffsim-venv       # create .venv-ffsim and install ffsim (IBM, fermionic anchor)
#   make quimb-venv       # create .venv-quimb and install quimb (MPS competitor)
#   make all              # all six
#   make clean            # remove all venvs
#
# tsim and ppvm both require Python >= 3.10, which the macOS system
# `python3` (3.9) is not. Pass a newer interpreter explicitly:
#   make tsim-venv PY=python3.12
#
# After install, expose the runners on PATH (or via env var):
#   export OMEGA_BRIDGE_QISKIT_CMD="$(pwd)/omega-bridge-qiskit-runner"
#   export OMEGA_BRIDGE_PERCEVAL_CMD="$(pwd)/omega-bridge-perceval-runner"
#
# The wrapper scripts (omega-bridge-*-runner) activate the matching
# venv and exec the Python runner so the calling shell doesn't need
# to know about virtualenvs.

PY ?= python3

# Behind a TLS-intercepting proxy (pxpipe on this host) pip fails with
# CERTIFICATE_VERIFY_FAILED unless it is handed the proxy's CA. Node is told
# through NODE_EXTRA_CA_CERTS; give pip the same file. Verification stays on —
# this is the CA, not --trusted-host.
ifdef NODE_EXTRA_CA_CERTS
export PIP_CERT ?= $(NODE_EXTRA_CA_CERTS)
endif

# Guard: every bridge venv except qiskit needs Python >= 3.10.
#
# tsim and ppvm say so in their requirements and fail loudly on 3.9.
# **Perceval does not.** On 3.9 pip silently resolves perceval-quandela back to
# 1.0.1, whose `Sampler.samples()` returns None where 1.2.x returns a dict; the
# runner then dies with `AttributeError: 'NoneType' has no attribute 'get'`, the
# cross-backend harness records "skipping", and the stage reports success having
# compared nothing on that arm. A venv that builds and then silently compares
# zero fixtures is worse than one that refuses to build, so refuse to build.
#
# macOS ships 3.9 as `python3`, which is why this is the default path and not an
# exotic one. Pass a newer interpreter:
#   make perceval-venv PY=$$(uv python find 3.12)
define REQUIRE_PY310
	@$(PY) -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' || { \
		echo "ERROR: $@ needs Python >= 3.10, but PY=$(PY) is $$($(PY) -V 2>&1)."; \
		echo "       On 3.9 pip resolves perceval-quandela to 1.0.1, whose Sampler"; \
		echo "       API differs, and the arm then skips silently instead of"; \
		echo "       comparing. Re-run with e.g.:"; \
		echo "         make $@ PY=\$$(uv python find 3.12)"; \
		exit 1; }
endef

.PHONY: all qiskit-venv perceval-venv bloqade-venv tsim-venv ppvm-venv mqt-venv quickqudits-venv quimb-venv clean \
	check-env check-qiskit check-perceval check-bloqade check-tsim check-ppvm

all: qiskit-venv perceval-venv bloqade-venv tsim-venv ppvm-venv ffsim-venv

qiskit-venv: .venv-qiskit/.installed

.venv-qiskit/.installed: requirements-qiskit.txt
	$(PY) -m venv .venv-qiskit
	./.venv-qiskit/bin/pip install --upgrade pip
	./.venv-qiskit/bin/pip install -r requirements-qiskit.txt
	# pytest: ci.sh runs crates/omega-bridges/python/tests under every
	# venv that has it. The perceval venv is the only one that can run
	# test_perceval_conventions.py, so it must have pytest or those pins
	# silently skip.
	./.venv-qiskit/bin/pip install pytest
	@touch .venv-qiskit/.installed

# ffsim (IBM) — the fermionic differential anchor, expectation-only over QPY.
# Its own venv: it pins a qiskit it is built against, and the anchor must not
# share an interpreter with the thing it anchors.
ffsim-venv: .venv-ffsim/.installed

.venv-ffsim/.installed: requirements-ffsim.txt
	$(PY) -m venv .venv-ffsim
	./.venv-ffsim/bin/pip install --upgrade pip
	./.venv-ffsim/bin/pip install -r requirements-ffsim.txt
	./.venv-ffsim/bin/pip install pytest
	@touch .venv-ffsim/.installed

# OpenFermion-FQE — the JW-free fermionic oracle behind fqe_runner.py and
# tests/fqe_vs_statevector.rs. Test-only; no product backend.
# Not $(PY): FQE 0.3.0 is a 2023 Cython sdist that builds on 3.11 (3.12 at
# most), needs numpy<2, and declares deps the runner never imports
# (cirq-google, jupyter, a pyasn1 pin). uv supplies the interpreter; the
# sdist is installed --no-deps on top of requirements-fqe.txt.
PY_FQE ?= 3.11
fqe-venv: .venv-fqe/.installed

.venv-fqe/.installed: requirements-fqe.txt
	uv venv --native-tls -p $(PY_FQE) .venv-fqe
	uv pip install --native-tls -p .venv-fqe/bin/python -r requirements-fqe.txt
	uv pip install --native-tls -p .venv-fqe/bin/python --no-deps --no-build-isolation fqe==0.3.0
	@touch .venv-fqe/.installed

# MQT Qudits — the DITQASM reference (PLAN-FERMIONIC-QUDIT Q0). Not a bridge
# runner: ci.sh uses it to re-parse crates/omega-parser/tests/fixtures/ditqasm/
# under the library that wrote them (tools/ditqasm_xcheck/). The pin lives in
# requirements-mqt.txt; the fixtures are that version's bytes.
mqt-venv: .venv-mqt/.installed

.venv-mqt/.installed: requirements-mqt.txt
	$(REQUIRE_PY310)
	$(PY) -m venv .venv-mqt
	./.venv-mqt/bin/pip install --upgrade pip
	./.venv-mqt/bin/pip install -r requirements-mqt.txt
	@touch .venv-mqt/.installed

# QuickQudits — external oracle for the qudit Clifford gates (PLAN-QUDIT.md
# Q2, tools/quditsv_xcheck/). Apache-2.0. Python >= 3.10.
quickqudits-venv: .venv-quickqudits/.installed

.venv-quickqudits/.installed: requirements-quickqudits.txt
	$(REQUIRE_PY310)
	$(PY) -m venv .venv-quickqudits
	./.venv-quickqudits/bin/pip install --upgrade pip
	./.venv-quickqudits/bin/pip install -r requirements-quickqudits.txt
	@touch .venv-quickqudits/.installed

perceval-venv: .venv-perceval/.installed

.venv-perceval/.installed: requirements-perceval.txt
	$(REQUIRE_PY310)
	$(PY) -m venv .venv-perceval
	./.venv-perceval/bin/pip install --upgrade pip
	./.venv-perceval/bin/pip install -r requirements-perceval.txt
	# pytest: ci.sh runs crates/omega-bridges/python/tests under every
	# venv that has it. The perceval venv is the only one that can run
	# test_perceval_conventions.py, so it must have pytest or those pins
	# silently skip.
	./.venv-perceval/bin/pip install pytest
	@touch .venv-perceval/.installed

bloqade-venv: .venv-bloqade/.installed

.venv-bloqade/.installed: requirements-bloqade.txt
	$(REQUIRE_PY310)
	$(PY) -m venv .venv-bloqade
	./.venv-bloqade/bin/pip install --upgrade pip
	./.venv-bloqade/bin/pip install -r requirements-bloqade.txt
	@touch .venv-bloqade/.installed

tsim-venv: .venv-tsim/.installed

.venv-tsim/.installed: requirements-tsim.txt
	$(REQUIRE_PY310)
	$(PY) -m venv .venv-tsim
	./.venv-tsim/bin/pip install --upgrade pip
	./.venv-tsim/bin/pip install -r requirements-tsim.txt
	@touch .venv-tsim/.installed

# ppvm ships as a maturin mixed Rust/Python wheel from a git
# subdirectory, so this target compiles a Rust crate. The repo already
# pins a toolchain (rust-toolchain.toml), so no extra setup — just a
# slower first build than a pure-Python venv.
ppvm-venv: .venv-ppvm/.installed

.venv-ppvm/.installed: requirements-ppvm.txt
	$(REQUIRE_PY310)
	$(PY) -m venv .venv-ppvm
	./.venv-ppvm/bin/pip install --upgrade pip
	./.venv-ppvm/bin/pip install -r requirements-ppvm.txt
	@touch .venv-ppvm/.installed

# quimb — the MPS-lane competitor (CircuitMPS). Its own venv: it pins a
# numpy/scipy/numba stack that must not share an interpreter with the
# oracle venvs. The import below is the install check. A directory with
# a `pyvenv.cfg` and no working `import quimb` is not a venv; `make`
# must not stamp `.installed` for one.
quimb-venv: .venv-quimb/.installed

.venv-quimb/.installed: requirements-quimb.txt
	@$(PY) -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' || { \
		echo "ERROR: quimb-venv needs Python >= 3.11 (quimb 1.15), but PY=$(PY) is $$($(PY) -V 2>&1)."; \
		exit 1; }
	$(PY) -m venv .venv-quimb
	./.venv-quimb/bin/pip install --upgrade pip
	./.venv-quimb/bin/pip install -r requirements-quimb.txt
	./.venv-quimb/bin/python -c 'import quimb, quimb.tensor as qtn; assert hasattr(qtn, "CircuitMPS")'
	@touch .venv-quimb/.installed

# Smoke check: round-trip a Bell-state QASM through the runner and
# inspect the counts. Skips with a clear message if the venv is missing.
# One table saying which arms this machine can actually run, and why not for
# the rest.
#
# Before this, answering "why did the bridge cross-checks skip?" meant reading
# ci.sh, then the Makefile, then each requirements file, then guessing at the
# interpreter version. Two of those guesses are traps: perceval silently
# resolves to an ancient release on Python 3.9 (see REQUIRE_PY310), and a venv
# can exist while being unusable. This probes each one by importing its actual
# package rather than testing for a directory.
check-env:
	@printf '%-10s %-9s %-8s %s\n' ARM VENV PYTHON STATUS
	@printf '%-10s %-9s %-8s %s\n' ---------- --------- -------- ------
	@for arm in qiskit perceval bloqade tsim ppvm ffsim quimb; do \
		py="./.venv-$$arm/bin/python"; \
		if [ ! -x "$$py" ]; then \
			printf '%-10s %-9s %-8s %s\n' "$$arm" "missing" "-" "run: make $$arm-venv PY=\$$(uv python find 3.12)"; \
			continue; \
		fi; \
		ver=$$("$$py" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null); \
		case "$$arm" in \
			qiskit)   mod=qiskit ;; \
			perceval) mod=perceval ;; \
			bloqade)  mod=bloqade ;; \
			tsim)     mod=tsim ;; \
			ppvm)     mod=ppvm ;; \
			ffsim)    mod=ffsim ;; \
			quimb)    mod=quimb ;; \
		esac; \
		if "$$py" -c "import $$mod" >/dev/null 2>&1; then \
			printf '%-10s %-9s %-8s %s\n' "$$arm" "built" "$$ver" "ready"; \
		else \
			printf '%-10s %-9s %-8s %s\n' "$$arm" "built" "$$ver" "BROKEN: cannot import $$mod — rebuild"; \
		fi; \
	done
	@echo
	@echo "Flags these arms gate:"
	@echo "  ARIA_BRIDGE_XCHECK=1   perceval, bloqade, tsim, ppvm (cross_backend)"
	@echo "  ARIA_QISKIT_XCHECK=1   qiskit (differential + QASM2/QASM3 dialect)"
	@echo "  ARIA_NWAY=1            qiskit (N-way counts + expectation matrix)"
	@echo "Not runnable on any machine without the hardware/toolchain:"
	@echo "  ARIA_CUDA=1            needs an NVIDIA box"
	@echo "  ARIA_LEAN=1            needs lake + the mathlib cache"

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

check-tsim:
	@if [ ! -x ./.venv-tsim/bin/python ]; then \
		echo "tsim venv not built — run 'make tsim-venv PY=python3.12' first."; exit 2; \
	fi
	@printf '{"qasm":"OPENQASM 2.0; include \\"qelib1.inc\\"; qreg q[2]; creg c[2]; h q[0]; t q[0]; cx q[0],q[1]; measure q -> c;","shots":1024,"seed":7}' \
		| ./omega-bridge-tsim-runner

check-ppvm:
	@if [ ! -x ./.venv-ppvm/bin/python ]; then \
		echo "ppvm venv not built — run 'make ppvm-venv PY=python3.12' first."; exit 2; \
	fi
	@printf '{"qasm":"OPENQASM 2.0; include \\"qelib1.inc\\"; qreg q[2]; creg c[2]; h q[0]; t q[0]; cx q[0],q[1]; measure q -> c;","shots":1024,"seed":7}' \
		| ./omega-bridge-ppvm-runner

clean:
	rm -rf .venv-qiskit .venv-perceval .venv-bloqade .venv-tsim .venv-ppvm .venv-ffsim .venv-mqt .venv-quickqudits .venv-quimb
