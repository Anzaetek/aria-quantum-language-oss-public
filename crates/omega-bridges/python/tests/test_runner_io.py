# SPDX-License-Identifier: Apache-2.0
"""The protocol descriptor must survive anything a dependency prints.

`runner_io` exists because a Perceval DeprecationWarning on **file descriptor
1** landed in front of a runner's JSON response and the Rust side reported

    invalid JSON from runner: expected value at line 1 column 2

which names the wrong culprit: the bridge looked broken, it was being talked
over. These tests plant exactly that interference and assert stdout still
carries one clean JSON line.

Run:  crates/omega-bridges/python/.venv-qiskit/bin/python -m pytest \\
        crates/omega-bridges/python/tests/test_runner_io.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
PYDIR = os.path.dirname(HERE)


def _run(body: str) -> subprocess.CompletedProcess:
    """Run a snippet that imports runner_io, prints noise, then emits.

    Built by concatenation, NOT by interpolating into an indented triple-quoted
    template + `textwrap.dedent`. A multi-line `body` has no leading whitespace
    on its second line, so dedent finds a common prefix of "" and strips
    nothing — leaving the template's own lines indented and the child dying
    with IndentationError. The first version of this helper did exactly that,
    and every test "failed" with an empty stdout that looked like the guard
    swallowing the payload. It was the harness, not the guard.
    """
    script = "\n".join(
        [
            "import sys, os",
            f"sys.path.insert(0, {PYDIR!r})",
            "from runner_io import emit, err",
            body,
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert proc.returncode == 0, (
        f"child exited {proc.returncode} — the snippet itself is broken, which "
        f"would make every assertion below vacuous.\nstderr: {proc.stderr}"
    )
    return proc


def _sole_json(out: str) -> dict:
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1, f"stdout must carry exactly one line, got {lines!r}"
    return json.loads(lines[0])


def test_python_print_does_not_reach_the_protocol():
    r = _run('print("chatty library banner")\nemit({"ok": True, "counts": {}})')
    assert _sole_json(r.stdout) == {"ok": True, "counts": {}}
    assert "chatty library banner" in r.stderr


def test_sys_stdout_write_does_not_reach_the_protocol():
    r = _run('sys.stdout.write("captured handle\\n")\nemit({"ok": True})')
    assert _sole_json(r.stdout) == {"ok": True}


def test_a_raw_fd_1_write_does_not_reach_the_protocol():
    """The case a `sys.stdout` rebind CANNOT catch.

    Perceval's logger writes to fd 1 directly, which is why the first attempt
    at this fix (`sys.stdout = sys.stderr`) failed — measured, the warning
    still appeared. Native extensions behave the same way, so this is the test
    that actually distinguishes the working fix from the plausible one.
    """
    r = _run('os.write(1, b"native library noise\\n")\nemit({"ok": True})')
    assert _sole_json(r.stdout) == {"ok": True}
    assert "native library noise" in r.stderr


def test_warnings_module_does_not_reach_the_protocol():
    r = _run(
        'import warnings\n'
        'warnings.warn("deprecated", DeprecationWarning)\n'
        'sys.stderr.write("")\n'
        'emit({"ok": True})'
    )
    assert _sole_json(r.stdout) == {"ok": True}


def test_err_carries_the_kind_contract():
    r = _run('err("nope", kind="qiskit-not-installed")')
    payload = _sole_json(r.stdout)
    assert payload["ok"] is False
    assert payload["kind"] == "qiskit-not-installed"
    assert payload["error"] == "nope"


def test_every_runner_imports_the_shared_guard():
    """A new runner must get the guard by construction, not by memory.

    The original fix went into one runner and left four exposed; this is what
    stops that recurring.
    """
    import glob

    missing = []
    for path in sorted(glob.glob(os.path.join(PYDIR, "*_runner.py"))):
        src = open(path).read()
        if "from runner_io import" not in src:
            missing.append(os.path.basename(path))
    assert not missing, f"runners not using runner_io: {missing}"
