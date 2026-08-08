#!/usr/bin/env python3
"""QASM2 → extended-Stim converter shared by the tsim and ppvm runners.

Both QuEra simulators consume the *same* dialect: vanilla Stim
instructions plus two PPVM/tsim tag extensions that carry the
non-Clifford gates Stim itself can't express.

    T                       S[T] <q>
    T_DAG                   S_DAG[T] <q>
    RX/RY/RZ(theta)         I[R_X(theta=<c>*pi)] <q>      (theta = c·pi)
    U3(theta, phi, lambda)  I[U3(theta=<a>*pi, phi=<b>*pi, lambda=<c>*pi)] <q>

`tsim.Circuit` accepts this text directly (its `shorthand_to_stim`
pre-pass leaves the canonical bracket forms alone — the `T` rewrite is
guarded by a `(?<!\\[)` lookbehind and the `R_X(<float>)` rewrite only
fires on a bare float argument). `ppvm.StimProgram.parse` lowers the
same bracket tags into its `ExtendedInstruction::{T, Rotation, U3}`
variants. Emitting the canonical form once therefore feeds both
backends from one converter.

Why hand-rolled rather than `qiskit.qasm2.loads` + a transpile: the
whole point of a bridge is that the two toolchains are independent.
Pulling Qiskit into the tsim/ppvm venvs would make "tsim agrees with
Qiskit" partly a statement about Qiskit parsing its own output. The
subset below is small, flat, and total: anything outside it is a loud
`*-unsupported-gate` refusal, never a silent approximation.

Supported QASM2 surface
-----------------------
Header (`OPENQASM 2.0;`, `include "qelib1.inc";`), `qreg`/`creg`
declarations (several of each), flat gate applications with either
indexed (`q[0]`) or whole-register (`q`) operands, `measure`, `reset`,
`barrier`, and `id`. Parameter expressions are evaluated over a
restricted arithmetic grammar (`pi`, `+ - * / **`, unary minus, and the
qelib1 classical functions).

NOT supported, by design — every one of these raises `UnsupportedGate`:
user `gate`/`opaque` definitions, classically-conditioned `if (...)`,
controlled rotations (`crz`, `cu1`, …), and any gate outside the
per-backend table. A controlled rotation in particular *cannot* be
lowered through the `u1 ≡ rz` global-phase shortcut this module uses
for the uncontrolled case, so refusing is the only correct answer.

Bit ordering
------------
Output bitstrings are indexed by **global classical-bit index, LSB
first** — `bits[0]` is the first bit of the first-declared `creg`. That
is exactly what `qiskit_runner.py` produces after its
`key.replace(" ", "")[::-1]` normalisation, so the two are directly
comparable in the cross-backend harness. Classical bits that no
`measure` writes to stay `0`.
"""

from __future__ import annotations

import ast
import math
import re

# --- gate tables -----------------------------------------------------

#: Gates both backends implement, mapped to their fixed Stim opcode.
#: (Parameterised gates and `measure`/`reset` are handled separately.)
_FIXED_1Q = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sdg": "S_DAG",
    "t": "S[T]",
    "tdg": "S_DAG[T]",
    "sx": "SQRT_X",
    "sxdg": "SQRT_X_DAG",
}

_FIXED_2Q = {
    "cx": "CX",
    "CX": "CX",
    "cy": "CY",
    "cz": "CZ",
    "swap": "SWAP",
}

#: Parameterised gates: name → (arity in qubits, number of params).
_PARAM_GATES = {
    "rx": (1, 1),
    "ry": (1, 1),
    "rz": (1, 1),
    # u1(λ) = diag(1, e^{iλ}) = e^{iλ/2}·RZ(λ). Global phase only, and
    # this module never lowers a *controlled* rotation (see module
    # docstring), so folding u1/p onto RZ is exact for measurement
    # statistics.
    "u1": (1, 1),
    "p": (1, 1),
    "u2": (1, 2),
    "u3": (1, 3),
    "u": (1, 3),
    "id": (1, 0),
    "u0": (1, 1),
}

#: Gates that expand into a Clifford+T decomposition. tsim's
#: `shorthand_to_stim` does that expansion for us; ppvm's parser has no
#: equivalent, hence the per-backend gate-set split below.
_FIXED_3Q = {
    "ccx": "CCX",
    "ccz": "CCZ",
}

#: Backend gate sets. Anything not listed is refused.
#:
#: ppvm's `stim-parser` explicitly rejects SWAP / ISWAP / SQRT_XX / the
#: XC*/YC* family (`crates/ppvm-stim/src/executor.rs` marks them
#: `unreachable!("... rejected by validate")`), and has no CCX/CCZ
#: sugar, so those live in the tsim set only.
GATE_SETS = {
    "tsim": frozenset(
        set(_FIXED_1Q) | set(_FIXED_2Q) | set(_FIXED_3Q) | set(_PARAM_GATES)
    ),
    "ppvm": frozenset(
        (set(_FIXED_1Q) | set(_FIXED_2Q) | set(_PARAM_GATES)) - {"swap"}
    ),
}

#: Structural statements every backend handles (not "gates").
_STRUCTURAL = frozenset({"measure", "reset", "barrier"})


class ConversionError(Exception):
    """QASM2 the converter cannot parse at all (bad syntax, bad index)."""


class UnsupportedGate(ConversionError):
    """A well-formed QASM2 construct outside the backend's gate set.

    Carried separately so the runners can tag it
    `<backend>-unsupported-gate`, which is a *capability* statement, not
    a parse failure.
    """


# --- parameter expressions -------------------------------------------

_ALLOWED_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log,
    "sqrt": math.sqrt,
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


def eval_param(expr: str) -> float:
    """Evaluate a QASM2 parameter expression to a float.

    Restricted arithmetic only — `pi`, numeric literals, `+ - * / **`,
    unary sign, and the qelib1 classical functions. Anything else
    (attribute access, names, subscripts, comprehensions) is rejected
    before evaluation, so this never executes attacker-chosen code even
    though it goes through `eval`.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ConversionError(f"bad parameter expression {expr!r}: {e}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ConversionError(
                f"parameter expression {expr!r} uses unsupported syntax "
                f"{type(node).__name__}"
            )
        if isinstance(node, ast.Name) and node.id not in set(_ALLOWED_FUNCS) | {"pi"}:
            raise ConversionError(
                f"parameter expression {expr!r} references unknown name {node.id!r}"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ConversionError(
                    f"parameter expression {expr!r} calls an unsupported function"
                )
    env = {"pi": math.pi, "__builtins__": {}}
    env.update(_ALLOWED_FUNCS)
    try:
        value = eval(compile(tree, "<qasm-param>", "eval"), env)  # noqa: S307
    except Exception as e:  # noqa: BLE001
        raise ConversionError(f"parameter expression {expr!r} failed: {e}") from None
    return float(value)


def _half_turns(theta: float) -> str:
    """Format an angle in radians as the `<c>*pi` literal both parsers
    require. `repr` on the coefficient round-trips exactly through
    IEEE-754 double, so no precision is lost across the text boundary.
    """
    return repr(theta / math.pi)


# --- QASM2 tokenising ------------------------------------------------

_COMMENT = re.compile(r"//.*?$", re.MULTILINE)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_DECL = re.compile(r"^(qreg|creg)\s+([A-Za-z][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]$")
_GATE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s*(?:\((.*)\))?\s+(.*)$", re.DOTALL)
_OPERAND = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s*(?:\[\s*(\d+)\s*\])?$")
_MEASURE = re.compile(r"^measure\s+(.*?)\s*->\s*(.*)$", re.DOTALL)


def _statements(qasm: str):
    """Yield semicolon-terminated statements with comments stripped."""
    src = _BLOCK_COMMENT.sub(" ", _COMMENT.sub("", qasm))
    for chunk in src.split(";"):
        stmt = " ".join(chunk.split())
        if stmt:
            yield stmt


class _Registers:
    """Qubit / classical registers with flat global index assignment."""

    def __init__(self) -> None:
        self.q: dict[str, tuple[int, int]] = {}  # name → (offset, size)
        self.c: dict[str, tuple[int, int]] = {}
        self.n_qubits = 0
        self.n_clbits = 0

    def add_q(self, name: str, size: int) -> None:
        self.q[name] = (self.n_qubits, size)
        self.n_qubits += size

    def add_c(self, name: str, size: int) -> None:
        self.c[name] = (self.n_clbits, size)
        self.n_clbits += size

    def resolve(self, table: dict, operand: str, what: str) -> list[int]:
        m = _OPERAND.match(operand.strip())
        if not m:
            raise ConversionError(f"cannot parse {what} operand {operand!r}")
        name, idx = m.group(1), m.group(2)
        if name not in table:
            raise ConversionError(f"undeclared {what} register {name!r}")
        offset, size = table[name]
        if idx is None:
            return list(range(offset, offset + size))
        i = int(idx)
        if i >= size:
            raise ConversionError(f"{what} index {name}[{i}] out of range (size {size})")
        return [offset + i]


def convert(qasm: str, gate_set) -> tuple[str, int, list[int], int]:
    """Lower a QASM2 source to extended-Stim text.

    Args:
        qasm: QASM2 source.
        gate_set: iterable of accepted lowercase QASM gate names — pass
            ``GATE_SETS["tsim"]`` or ``GATE_SETS["ppvm"]``.

    Returns:
        ``(stim_text, n_qubits, clbit_of_measurement, n_clbits)`` where
        ``clbit_of_measurement[k]`` is the global classical-bit index
        the k-th recorded measurement writes to.

    Raises:
        UnsupportedGate: a well-formed gate outside ``gate_set``.
        ConversionError: anything the converter cannot parse.
    """
    gate_set = frozenset(gate_set)
    regs = _Registers()
    lines: list[str] = []
    clbit_of_measurement: list[int] = []

    for stmt in _statements(qasm):
        head = stmt.split(None, 1)[0]

        if head in ("OPENQASM", "include"):
            continue
        if head in ("gate", "opaque"):
            raise UnsupportedGate(
                f"user-defined `{head}` declarations are not supported — "
                "flatten the circuit (e.g. qiskit's transpile to the basis "
                "gate set) before sending it to this backend"
            )
        if head == "if":
            raise UnsupportedGate(
                "classically-conditioned `if (...)` statements are not supported"
            )

        decl = _DECL.match(stmt)
        if decl:
            kind, name, size = decl.group(1), decl.group(2), int(decl.group(3))
            (regs.add_q if kind == "qreg" else regs.add_c)(name, size)
            continue

        if head == "measure":
            m = _MEASURE.match(stmt)
            if not m:
                raise ConversionError(f"cannot parse measure statement {stmt!r}")
            qubits = regs.resolve(regs.q, m.group(1), "qubit")
            clbits = regs.resolve(regs.c, m.group(2), "classical")
            if len(qubits) != len(clbits):
                raise ConversionError(
                    f"measure width mismatch in {stmt!r}: "
                    f"{len(qubits)} qubits → {len(clbits)} classical bits"
                )
            # One `M` per statement keeps the record order equal to the
            # operand order, which is what the clbit map assumes.
            lines.append("M " + " ".join(str(q) for q in qubits))
            clbit_of_measurement.extend(clbits)
            continue

        if head == "barrier":
            continue

        gm = _GATE.match(stmt)
        if not gm:
            raise ConversionError(f"cannot parse statement {stmt!r}")
        name, raw_params, raw_targets = gm.group(1), gm.group(2), gm.group(3)

        if head == "reset":
            targets = _targets(regs, raw_targets)
            for group in targets:
                lines.append("R " + " ".join(str(q) for q in group))
            continue

        if name not in gate_set:
            if name in _STRUCTURAL:
                raise ConversionError(f"cannot parse statement {stmt!r}")
            raise UnsupportedGate(
                f"gate `{name}` is outside this backend's supported gate set "
                f"({', '.join(sorted(gate_set))})"
            )

        params = (
            [eval_param(p) for p in _split_params(raw_params)] if raw_params else []
        )
        operand_groups = _targets(regs, raw_targets)
        lines.extend(_emit_gate(name, params, operand_groups, stmt))

    if regs.n_qubits == 0:
        raise ConversionError("no `qreg` declared")

    # Unitary-only fixture? Mirror qiskit_runner.py's `measure_all()`:
    # synthesise a full-width classical register and measure every
    # qubit in index order so the caller still gets counts back.
    if not clbit_of_measurement:
        regs.add_c("__omega_meas", regs.n_qubits)
        lines.append("M " + " ".join(str(q) for q in range(regs.n_qubits)))
        clbit_of_measurement = list(
            range(regs.n_clbits - regs.n_qubits, regs.n_clbits)
        )

    return "\n".join(lines) + "\n", regs.n_qubits, clbit_of_measurement, regs.n_clbits


def _split_params(raw: str) -> list[str]:
    """Split a parameter list on top-level commas (parens may nest)."""
    out, depth, cur = [], 0, []
    for ch in raw:
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return [p.strip() for p in out if p.strip()]


def _targets(regs: _Registers, raw: str) -> list[list[int]]:
    """Resolve a comma-separated operand list to per-operand qubit lists."""
    return [regs.resolve(regs.q, part, "qubit") for part in raw.split(",")]


def _emit_gate(
    name: str, params: list[float], groups: list[list[int]], stmt: str
) -> list[str]:
    """Emit the Stim line(s) for one gate application.

    QASM2 broadcasts whole-register operands elementwise; `groups` holds
    one qubit list per operand, so a broadcast is a zip over equal-length
    groups (a length-1 group is held constant, as QASM2 specifies).
    """
    arity = (
        3
        if name in _FIXED_3Q
        else 2
        if name in _FIXED_2Q
        else 1
    )
    if len(groups) != arity:
        raise ConversionError(
            f"gate `{name}` takes {arity} qubit operand(s), got {len(groups)} in {stmt!r}"
        )
    width = max(len(g) for g in groups)
    for g in groups:
        if len(g) not in (1, width):
            raise ConversionError(f"broadcast width mismatch in {stmt!r}")
    tuples = [
        tuple(g[0] if len(g) == 1 else g[i] for g in groups) for i in range(width)
    ]

    if name in _FIXED_1Q:
        return [f"{_FIXED_1Q[name]} " + " ".join(str(t[0]) for t in tuples)]
    if name in _FIXED_2Q:
        flat = " ".join(f"{a} {b}" for a, b in tuples)
        return [f"{_FIXED_2Q[name]} {flat}"]
    if name in _FIXED_3Q:
        return [f"{_FIXED_3Q[name]} {a} {b} {c}" for a, b, c in tuples]

    n_qubits, n_params = _PARAM_GATES[name]
    if len(params) != n_params:
        raise ConversionError(
            f"gate `{name}` takes {n_params} parameter(s), got {len(params)} in {stmt!r}"
        )
    qubits = [t[0] for t in tuples]
    targets = " ".join(str(q) for q in qubits)

    if name in ("id", "u0"):
        return []
    if name in ("rx", "ry", "rz"):
        axis = name[1].upper()
        return [f"I[R_{axis}(theta={_half_turns(params[0])}*pi)] {targets}"]
    if name in ("u1", "p"):
        # Global-phase-equivalent to RZ; see _PARAM_GATES comment.
        return [f"I[R_Z(theta={_half_turns(params[0])}*pi)] {targets}"]
    if name == "u2":
        return [_u3_line(math.pi / 2, params[0], params[1], targets)]
    # u3 / u
    return [_u3_line(params[0], params[1], params[2], targets)]


def _u3_line(theta: float, phi: float, lam: float, targets: str) -> str:
    return (
        f"I[U3(theta={_half_turns(theta)}*pi, "
        f"phi={_half_turns(phi)}*pi, "
        f"lambda={_half_turns(lam)}*pi)] {targets}"
    )


def bits_to_counts(
    samples, clbit_of_measurement: list[int], n_clbits: int
) -> dict[str, int]:
    """Fold per-shot measurement rows into an LSB-first counts dict.

    `samples` is any iterable of per-shot sequences of 0/1-ish values in
    measurement-record order (numpy rows from tsim, lists of
    `MeasurementResult` from ppvm — `int()` normalises both). Each
    record is placed at its classical-bit index; bits no `measure`
    wrote to stay `0`.

    Any outcome that is neither 0 nor 1 is a hard error rather than a
    coerced bit: ppvm's `MeasurementResult` has a third member (`LOST`,
    atom loss) whose numeric value would otherwise silently fold into
    the `1` bin and corrupt the distribution.
    """
    counts: dict[str, int] = {}
    for row in samples:
        bits = ["0"] * n_clbits
        for k, value in enumerate(row):
            bit = int(value)
            if bit not in (0, 1):
                raise ConversionError(
                    f"measurement record {k} came back as {value!r} "
                    "(neither 0 nor 1) — atom loss / erasure outcomes have no "
                    "bit-string representation in the counts protocol"
                )
            bits[clbit_of_measurement[k]] = "1" if bit else "0"
        key = "".join(bits)
        counts[key] = counts.get(key, 0) + 1
    return counts
