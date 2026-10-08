# SPDX-License-Identifier: Apache-2.0
"""Stim arm of the stabilizer comparison.

One process, started once so the import is outside every timed region.
Each request is one JSON object on a line; each response is one JSON
object on a line. The seconds in a response are `perf_counter` around
the call the row claims to time, and nothing else:

* expectation: `TableauSimulator.do` + `peek_observable_expectation`
  (circuit already parsed)
* sampling: `CompiledMeasurementSampler.sample` (compile_sampler already
  done, and reported separately)
* d=25: `CompiledDetectorSampler.sample` (compile reported separately)

`bit_packed=True` is Stim's denser sample format. It does not change the
shot axis, and it is the faster of the two documented layouts.
`skip_reference_sample` stays at its default False: that flag returns
flip bits rather than measurement results, which is a different
computation. The reference sample is paid inside `compile_sampler`,
which is this lane's fixed cost, not inside `sample`.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import numpy
import stim

# The d=25 capability workload. These are Stim's own generated-circuit
# knobs, each set, at the probability the docstring uses as its example.
# Zeroing a noise knob would sample a different, easier circuit.
D25_TASK = "surface_code:rotated_memory_z"
D25_DISTANCE = 25
D25_ROUNDS = 25
D25_AFTER_CLIFFORD = 0.001
D25_BEFORE_ROUND = 0.001
D25_BEFORE_MEASURE = 0.001
D25_AFTER_RESET = 0.001

PINNED_VERSION = "1.16.0"

_CIRCUITS: dict[str, stim.Circuit] = {}
_SAMPLERS: dict[str, stim.CompiledMeasurementSampler] = {}
_FLOOR_SAMPLER: stim.CompiledMeasurementSampler | None = None


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _linkage() -> str:
    # The linkage record is provenance, so it uses the platform's own tool:
    # `otool -L` exists only on macOS and `ldd` only on Linux. The lane test
    # runs on both boxes, and `otool` alone made it fail on akilles with
    # FileNotFoundError before any arm ran.
    so = next(pathlib.Path(stim.__file__).resolve().parent.rglob("*.so"))
    tool = ["otool", "-L"] if sys.platform == "darwin" else ["ldd"]
    return subprocess.check_output([*tool, str(so)], text=True)


def _smoke() -> tuple[int, int]:
    # H|0> is |+>: <X> = 1 and <Z> = 0, both exact integers. A build whose
    # peek path is the complex64 state vector fails this, which is the
    # known trap this lane exists not to repeat.
    circuit = stim.Circuit("H 0")
    sim = stim.TableauSimulator()
    sim.set_num_qubits(1)
    sim.do(circuit)
    x = int(sim.peek_observable_expectation(stim.PauliString("+X")))
    z = int(sim.peek_observable_expectation(stim.PauliString("+Z")))
    if (x, z) != (1, 0):
        raise RuntimeError(f"peek smoke failed: <X>={x} <Z>={z}")
    return x, z


def _d25_circuit() -> stim.Circuit:
    return stim.Circuit.generated(
        D25_TASK,
        distance=D25_DISTANCE,
        rounds=D25_ROUNDS,
        after_clifford_depolarization=D25_AFTER_CLIFFORD,
        before_round_data_depolarization=D25_BEFORE_ROUND,
        before_measure_flip_probability=D25_BEFORE_MEASURE,
        after_reset_flip_probability=D25_AFTER_RESET,
    )


def _exact(value: object) -> int:
    iv = int(value)  # type: ignore[arg-type]
    if iv not in (-1, 0, 1) or float(value) != float(iv):  # type: ignore[arg-type]
        raise ValueError(f"peek returned {value!r}, not an exact integer in {{-1, 0, 1}}")
    return iv


def _ready() -> dict:
    if stim.__version__ != PINNED_VERSION:
        raise RuntimeError(f"stim {stim.__version__}, pinned {PINNED_VERSION}")
    x, z = _smoke()
    return {
        "ok": True,
        "stim": stim.__version__,
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "linkage": _linkage(),
        "smoke_x": x,
        "smoke_z": z,
        "skip_reference_sample": False,
        "bit_packed": True,
    }


def _load(req: dict) -> dict:
    text = req["circuit"]
    t0 = time.perf_counter()
    circuit = stim.Circuit(text)
    parse_s = time.perf_counter() - t0
    _CIRCUITS[req["name"]] = circuit
    _SAMPLERS.pop(req["name"], None)
    return {
        "ok": True,
        "parse_s": parse_s,
        "num_qubits": circuit.num_qubits,
        "num_measurements": circuit.num_measurements,
    }


def _expectation(req: dict) -> dict:
    circuit = _CIRCUITS[req["name"]]
    pauli = stim.PauliString(req["pauli"])
    t0 = time.perf_counter()
    sim = stim.TableauSimulator()
    sim.set_num_qubits(circuit.num_qubits)
    sim.do(circuit)
    value = _exact(sim.peek_observable_expectation(pauli))
    seconds = time.perf_counter() - t0
    return {"ok": True, "value": value, "seconds": seconds}


def _heisenberg_image(req: dict) -> dict:
    """Conjugate a Pauli by the circuit's Clifford: `U P U-dagger`.

    This is how the value gate stops being degenerate. `<Z0>` on a Clifford
    brickwall is exactly 0 on both arms, so a gate pinning it alone is passed
    by a backend that computes nothing and returns 0 (A10, plan sec 4.3c).
    The image is the free fix: for |psi> = U|0>,

        <psi| U P U-dagger |psi> = <0| P |0>

    which is +1 for any Z-type P, by construction and not by measurement. So
    the gate becomes [<P>, <U P U-dagger>, <-U P U-dagger>] with a predicted
    reference of [measured, +1, -1], and no constant passes it.

    `to_tableau` refuses a circuit with measurement, reset or noise, which is
    the right refusal: the identity above holds for a unitary. Both gated
    circuits here are the measurement-free unitary (the brickwall as written,
    the syndrome circuit through `without_measures`), so it applies.
    """
    circuit = _CIRCUITS[req["name"]]
    p = stim.PauliString(req["pauli"])
    image = circuit.to_tableau()(p)
    return {"ok": True, "image": str(image), "weight": image.weight}


def _compile(req: dict) -> dict:
    circuit = _CIRCUITS[req["name"]]
    seed = int(req["seed"])
    t0 = time.perf_counter()
    # Default skip_reference_sample=False. The reference sample is the
    # fixed cost; sample() below does not pay it again.
    sampler = circuit.compile_sampler(seed=seed)
    seconds = time.perf_counter() - t0
    _SAMPLERS[req["name"]] = sampler
    return {"ok": True, "compile_s": seconds}


def _sample(req: dict) -> dict:
    sampler = _SAMPLERS[req["name"]]
    shots = int(req["shots"])
    circuit = _CIRCUITS[req["name"]]
    t0 = time.perf_counter()
    samples = sampler.sample(shots, bit_packed=True)
    seconds = time.perf_counter() - t0
    observed = int(samples.shape[0])
    if observed != shots:
        return {
            "ok": False,
            "error": f"shot shortfall: requested {shots}, observed {observed}",
            "shots_observed": observed,
        }
    return {
        "ok": True,
        "seconds": seconds,
        "shots_observed": observed,
        "num_measurements": circuit.num_measurements,
    }


def _floor_expectation() -> dict:
    circuit = stim.Circuit("I 0")
    pauli = stim.PauliString("+Z")
    t0 = time.perf_counter()
    sim = stim.TableauSimulator()
    sim.set_num_qubits(1)
    sim.do(circuit)
    value = _exact(sim.peek_observable_expectation(pauli))
    seconds = time.perf_counter() - t0
    return {"ok": True, "seconds": seconds, "value": value}


def _floor_sample() -> dict:
    global _FLOOR_SAMPLER
    if _FLOOR_SAMPLER is None:
        _FLOOR_SAMPLER = stim.Circuit("H 0\nM 0").compile_sampler(seed=0)
    t0 = time.perf_counter()
    samples = _FLOOR_SAMPLER.sample(1, bit_packed=True)
    seconds = time.perf_counter() - t0
    observed = int(samples.shape[0])
    if observed != 1:
        return {"ok": False, "error": f"floor sample returned {observed} shots"}
    return {"ok": True, "seconds": seconds, "shots_observed": observed}


def _refuse_t(req: dict) -> dict:
    try:
        stim.Circuit(req["circuit"])
    except ValueError as exc:
        return {"ok": True, "refused": True, "error": str(exc), "exc_type": type(exc).__name__}
    return {"ok": True, "refused": False, "error": "", "exc_type": ""}


def _d25_text() -> dict:
    t0 = time.perf_counter()
    circuit = _d25_circuit()
    seconds = time.perf_counter() - t0
    names = sorted({inst.name for inst in circuit})
    return {
        "ok": True,
        "text": str(circuit),
        "seconds": seconds,
        "qubits": circuit.num_qubits,
        "measurements": circuit.num_measurements,
        "detectors": circuit.num_detectors,
        "observables": circuit.num_observables,
        "instructions": names,
    }


def _d25_sample(req: dict) -> dict:
    shots = int(req["shots"])
    circuit = _d25_circuit()
    names = sorted({inst.name for inst in circuit})
    t0 = time.perf_counter()
    sampler = circuit.compile_detector_sampler(seed=0)
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    dets, obs = sampler.sample(shots, separate_observables=True, bit_packed=True)
    sample_s = time.perf_counter() - t0
    det_shots = int(dets.shape[0])
    obs_shots = int(obs.shape[0])
    if det_shots != shots or obs_shots != shots:
        return {
            "ok": False,
            "error": f"shot shortfall: requested {shots}, detectors {det_shots}, observables {obs_shots}",
            "shots_observed": det_shots,
        }
    return {
        "ok": True,
        "qubits": circuit.num_qubits,
        "measurements": circuit.num_measurements,
        "detectors": circuit.num_detectors,
        "observables": circuit.num_observables,
        "instructions": names,
        "shots_observed": det_shots,
        "obs_shots_observed": obs_shots,
        "compile_s": compile_s,
        "sample_s": sample_s,
    }


def _dispatch(req: dict) -> dict:
    cmd = req["cmd"]
    if cmd == "ready":
        return _ready()
    if cmd == "load":
        return _load(req)
    if cmd == "expectation":
        return _expectation(req)
    if cmd == "heisenberg_image":
        return _heisenberg_image(req)
    if cmd == "compile":
        return _compile(req)
    if cmd == "sample":
        return _sample(req)
    if cmd == "floor_expectation":
        return _floor_expectation()
    if cmd == "floor_sample":
        return _floor_sample()
    if cmd == "refuse_t":
        return _refuse_t(req)
    if cmd == "d25_text":
        return _d25_text()
    if cmd == "d25_sample":
        return _d25_sample(req)
    if cmd == "quit":
        _emit({"ok": True})
        raise SystemExit(0)
    return {"ok": False, "error": f"unknown cmd {cmd!r}"}


def main() -> None:
    if stim.__version__ != PINNED_VERSION:
        _emit({"ok": False, "error": f"stim {stim.__version__}, pinned {PINNED_VERSION}"})
        raise SystemExit(2)
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        try:
            req = json.loads(line)
            _emit(_dispatch(req))
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 — the protocol boundary
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--dump-d25":
        pathlib.Path(sys.argv[2]).write_text(str(_d25_circuit()), encoding="utf-8")
    else:
        main()
