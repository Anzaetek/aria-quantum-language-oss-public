// SPDX-License-Identifier: Apache-2.0
//! The pauliprop lane's preconditions, each reddened by what it refuses.
//! Venv-gated legs are a registered skip; `EMU_PAULIPROP_REQUIRE=1` makes the
//! skip a failure.

use emu_compare_pauliprop::measure::{mapped, MpKnobs, ATOLS, WEIGHTS};
use emu_compare_pauliprop::workload::{hea_cell, oracle, OursKnobs};

#[test]
fn the_grid_is_one_grid_in_both_engines_knob_names() {
    let n = 14;
    assert_eq!(
        mapped(n, None, 0.0),
        (
            OursKnobs {
                coeff_min: 0.0,
                max_weight: None
            },
            MpKnobs {
                cutoff: 14,
                lower_atol: None
            }
        )
    );
    assert_eq!(
        mapped(n, Some(6), 1e-8),
        (
            OursKnobs {
                coeff_min: 1e-8,
                max_weight: Some(6)
            },
            MpKnobs {
                cutoff: 6,
                lower_atol: Some(1e-8)
            }
        )
    );
    // A cap at or above n is no cap on our side, and n on monoprop's.
    assert_eq!(mapped(4, Some(10), 0.0).0.max_weight, None);
    assert_eq!(mapped(4, Some(10), 0.0).1.cutoff, 4);
    // The grid includes the exact point, so an exact setting is always admissible.
    assert!(WEIGHTS.contains(&None) && ATOLS.contains(&0.0));
}

fn py() -> Option<String> {
    match std::env::var("EMU_PAULIPROP_PY") {
        Ok(p) => Some(p),
        Err(_) => {
            if std::env::var("EMU_PAULIPROP_REQUIRE").as_deref() == Ok("1") {
                panic!("EMU_PAULIPROP_REQUIRE=1 but EMU_PAULIPROP_PY is unset");
            }
            eprintln!("SKIP: EMU_PAULIPROP_PY unset");
            None
        }
    }
}

fn run_py(py: &str, code: &str) -> (bool, String) {
    let out = std::process::Command::new(py)
        .arg("-c")
        .arg(code)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr),
    )
}

#[test]
fn the_transcription_refuses_what_it_cannot_map_exactly() {
    let Some(py) = py() else { return };
    let dir = std::env::temp_dir().join(format!("pp-lane-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, body) in [
        // H after a non-H op is not acting on |0>, so Ry(pi/2) would be wrong.
        ("late_h", "h q[0];\nrx(0.1) q[1];\nh q[1];\n"),
        // A bare CX with no Rz sandwich has no single-rotation form.
        ("bare_cx", "h q[0];\ncx q[0],q[1];\n"),
        // ry is not in the HEA vocabulary this transcription maps.
        ("ry", "ry(0.3) q[0];\n"),
    ] {
        let path = dir.join(format!("{name}.qasm"));
        std::fs::write(
            &path,
            format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\n{body}"),
        )
        .unwrap();
        let (ok, out) = run_py(&py, &format!("import monoprop_arm as A\ntry:\n    A.transcribe_qasm({path:?})\nexcept ValueError as e:\n    print('REFUSED', e); raise SystemExit(0)\nprint('ADMITTED'); raise SystemExit(1)\n", path = path.to_str().unwrap()));
        assert!(ok && out.contains("REFUSED"), "{name} was admitted: {out}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_transcribed_hea_is_exact_against_the_dense_oracle() {
    let Some(py) = py() else { return };
    let cell = hea_cell("hea-14q-d12").unwrap();
    let reference = oracle(&cell).unwrap();
    let code = format!(
        "import json, monoprop_arm as A\nA.load_qasm({{'name':'h','path':{p:?}}})\nprint(json.dumps(A.expectation({{'name':'h','terms':json.loads({t:?}),'cutoff':{n}}})))\n",
        p = cell.monoprop_load["path"].as_str().unwrap(),
        t = cell.terms.to_string(),
        n = cell.qubits
    );
    let (ok, out) = run_py(&py, &code);
    assert!(ok, "{out}");
    let line = out.lines().rev().find(|l| l.starts_with('{')).unwrap();
    let v: serde_json::Value = serde_json::from_str(line).unwrap();
    let got = v["value"].as_f64().unwrap();
    assert!(
        (got - reference).abs() / reference.abs().max(1.0) <= 1e-10,
        "monoprop {got} vs oracle {reference}"
    );
}

#[test]
fn monoprop_reports_no_error_quantity() {
    // §5's "bounds, not estimates" claim against monoprop rests on this: its
    // PauliPropagator exposes nothing that could be a discarded weight, an
    // error estimate or a bound. If a later version adds one, this fails and
    // the capability row must be re-read.
    let Some(py) = py() else { return };
    let (ok, out) = run_py(
        &py,
        "import monoprop as mp\nnames=[n for n in dir(mp.PauliPropagator) if not n.startswith('_')]\nbad=[n for n in names if any(k in n.lower() for k in ('drop','discard','error','bound','trunc','loss','residual'))]\nprint('NAMES',names)\nprint('BAD',bad)\nraise SystemExit(1 if bad else 0)\n",
    );
    assert!(ok, "monoprop now exposes an error-like quantity: {out}");
    assert!(
        out.contains("expectation_value"),
        "the introspection saw no methods at all: {out}"
    );
}
